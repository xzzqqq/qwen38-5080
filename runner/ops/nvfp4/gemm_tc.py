"""NVFP4 GEMV：native FP4 tensor core MMA（``mxf4nvf4``）+ swap-AB。

设计（任务 C，Phase 2）
-----------------------
M=1 decode 的 Linear 是 ``y[1, N] = a[1, K] @ w[N, K]^T``。本模块不搬动权重，
而是把 **W 当作 MMA 的 A 操作数**（swap-AB）：W 在 checkpoint 里是 ``U8 [N, K/2]``
行主序、K 连续，正好是 MMA A 操作数需要的「行 = M、K 连续」布局，所以 global load
沿 K 方向 coalesced；activation 放进 B 操作数（只有 1 列有用，broadcast 到 BN 列）。

数据流::

    w_packed U8 [N, K/2]        w_scale F8_E4M3 [N, K/16]（线性，无 swizzle）
    a_packed U8 [1, K/2]        a_scale F8_E4M3 [1, K/16]
        -> 每个 program 负责 BM 行权重，沿 K 以 BK 分块：
             tl.dot_scaled(W_tile[BM, BK/2], W_scale[BM, BK/16],
                           A_tile[BK/2, BN],  A_scale[BN, BK/16], e2m1 x e2m1)
           -> mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
              .scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3（per-16 block scale，FP32 累加）
        -> 取第 0 列 * alpha(= input_scale * weight_scale_2) -> BF16 [1, N]

为什么 swap-AB 是必要的：不 swap 时 W 是 B 操作数（逻辑 [K, N]），而 W 的 N 维在
显存里是外层 stride，K 是连续维，Triton 会退化成跨步 load（每行只取 1 个字节），
DRAM 效率崩溃。swap 后 W 侧完全 coalesced，代价只是 BN 列里只有 1 列有用（算力浪费
1/16，但本 shape 是纯带宽受限，MMA 吞吐远未饱和）。

成本口径（不能藏进计时）
-----------------------
- repack：**0**。本 kernel 直接消费 checkpoint 的线性 ``[N, K/16]`` scale 和
  ``[N, K/2]`` packed 权重；不需要 128x4 swizzle，也不需要转置。集成时反而比
  ``mm_fp4`` 路径少一次 scale swizzle（一次性，约 1.6 GB 的 1/10 量级 copy）。
- padding：**0**。所有模型 shape 的 N 都能被 BM(64/128) 整除、K 都能被 BK 整除；
  ``padding_bytes()`` 返回真实浪费的字节数。非整除 N 本轮不支持（会显式报错）。
- workspace：只有 A 的 packed bytes 和 scale（``K/2 + K/16`` 字节），无全局 scratch；
  MMA 的 shared memory 是 CTA 内资源（KB 级），不是常驻显存。

预期瓶颈：权重带宽（读 4 bit/参数 + 1/16 字节 scale）。fp4 MMA 的算术强度极低，
所以 tile 选择的目标是让每个 CTA 的 load 尽量大、并发足够多。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class Config:
    """一组粗粒度 tile 参数。"""

    block_m: int = 64
    block_k: int = 128
    block_n: int = 16
    num_warps: int = 4
    num_stages: int = 3
    cache_modifier: str = ".cg"

    @property
    def name(self) -> str:
        return (
            f"bm{self.block_m}-bk{self.block_k}-bn{self.block_n}"
            f"-w{self.num_warps}-s{self.num_stages}"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "block_m": self.block_m,
            "block_k": self.block_k,
            "block_n": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "cache_modifier": self.cache_modifier,
        }


# 3 组粗粒度参数：每 CTA 输出列数（BM）为主变量，BK/BN/warps/stages 固定。
CONFIGS: dict[str, Config] = {
    "c32x256": Config(block_m=32, block_k=256, block_n=16, num_warps=4, num_stages=3),
    "c64x256": Config(block_m=64, block_k=256, block_n=16, num_warps=4, num_stages=3),
    "c128x256": Config(block_m=128, block_k=256, block_n=16, num_warps=8, num_stages=3),
}
DEFAULT_CONFIG = "c64x256"


@triton.jit
def _gemv_kernel(
    w32_ptr,  # I32 [N, K/8]：checkpoint U8 [N, K/2] 的 4-byte 视图（K 连续）
    ws_ptr,  # F8_E4M3 [N, K/16]：线性 block scale（无 swizzle）
    a_ptr,  # U8 [K/2]
    as_ptr,  # F8_E4M3 [K/16]
    out_ptr,  # BF16 [N]
    alpha_ptr,  # FP32 [1] 设备标量（避免每步 host sync）
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CACHE: tl.constexpr,
    STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rw = tl.arange(0, BK // 8)  # 每个 u32 = 4 个 packed byte = 8 个 nibble
    rs = tl.arange(0, BK // 16)
    zero_n = tl.arange(0, BN) == 0
    alpha = tl.load(alpha_ptr)
    KP4: tl.constexpr = K // 8  # 每行 u32 个数
    BKP4: tl.constexpr = BK // 8
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, KP4, BKP4, num_stages=STAGES):
        k0 = tl.multiple_of(k0, BKP4)
        off4 = k0 + tl.max_contiguous(rw, BKP4)
        # W 行主序、K 连续 -> 每线程 4 byte、cp.async 流水；.cg 不上 L1。
        w32 = tl.load(w32_ptr + rm[:, None] * KP4 + off4[None, :], cache_modifier=CACHE)
        # u32 -> 4 个 byte lane：每个 u32 是 4 个连续 packed byte（低 byte 在前）。
        # join 的嵌套顺序决定 byte 顺序：join(join(b0,b2), join(b1,b3)) 才展开成
        # b0,b1,b2,b3（已用 Triton interpreter 在 CPU 上逐字节验证）。
        b0 = w32.to(tl.uint8)
        b1 = (w32 >> 8).to(tl.uint8)
        b2 = (w32 >> 16).to(tl.uint8)
        b3 = (w32 >> 24).to(tl.uint8)
        w = tl.reshape(tl.join(tl.join(b0, b2), tl.join(b1, b3)), (BM, BK // 2))
        ws = tl.load(ws_ptr + rm[:, None] * (K // 16) + (k0 // 2 + rs)[None, :])
        # activation 只有 1 列有效：broadcast 到 BN 列（A 很小，所有 program 共享）。
        a = tl.load(a_ptr + k0 * 4 + tl.arange(0, BK // 2))
        asc = tl.load(as_ptr + k0 // 2 + rs)
        acc = tl.dot_scaled(
            w,
            ws,
            "e2m1",
            tl.broadcast_to(a[:, None], (BK // 2, BN)),
            tl.broadcast_to(asc[None, :], (BN, BK // 16)),
            "e2m1",
            acc,
        )

    # 只有第 0 列有用（其余列是同一份数据的复制）。
    y = tl.sum(tl.where(zero_n[None, :], acc, 0.0), axis=1)
    tl.store(out_ptr + rm, (y * alpha).to(tl.bfloat16))


def padding_bytes(out_features: int, in_features: int, cfg: Config = CONFIGS[DEFAULT_CONFIG]) -> int:
    """若用 padding 把 N/K 补到 tile 的整数倍，多读的权重 + scale 字节数。

    真实模型 shape 下 N 能被 BM 整除、K 能被 BK 整除，所以恒为 0；非整除
    N/K 本轮直接报错（不做 masked 尾块），这里只用于把浪费量化上报。
    """
    n_pad = (-out_features) % cfg.block_m
    k_pad = (-in_features) % cfg.block_k
    if n_pad == 0 and k_pad == 0:
        return 0
    rows = out_features + n_pad
    return n_pad * (in_features // 2 + in_features // 16) + rows * (k_pad // 2 + k_pad // 16)


def workspace_bytes(in_features: int) -> int:
    """每次 forward 的 activation side workspace（packed + block scale）。"""
    return in_features // 2 + in_features // 16


def gemv_tc(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    alpha: torch.Tensor | float,
    *,
    cfg: Config = CONFIGS[DEFAULT_CONFIG],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """swap-AB FP4 tensor core GEMV。

    Args:
        w_packed: U8 [N, K/2]，checkpoint packed 权重（连续）。
        w_scale: F8_E4M3 [N, K/16]，**线性** block scale（不是 128x4 swizzled）。
        a_packed: U8 [1, K/2]（或 [K/2]），同一份量化后的 activation。
        a_scale: F8_E4M3 [1, K/16]（或 [K/16]）。
        alpha: FP32 标量 / [1]，``input_scale * weight_scale_2``。
        cfg: tile 参数。
        out: 可选 BF16 [N] 输出缓冲。

    Returns:
        BF16 [1, N]。
    """
    if w_packed.dim() != 2 or w_packed.dtype != torch.uint8:
        raise TypeError(f"w_packed must be uint8 [N, K/2], got {w_packed.dtype}")
    n, k_half = w_packed.shape
    k = k_half * 2
    if w_scale.shape != (n, k // 16) or w_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"w_scale must be float8_e4m3fn [N, K/16] linear, got "
            f"{w_scale.dtype} {tuple(w_scale.shape)}"
        )
    if a_packed.numel() != k // 2:
        raise ValueError(f"a_packed must hold K/2={k // 2} bytes, got {a_packed.numel()}")
    if a_scale.numel() != k // 16:
        raise ValueError(f"a_scale must hold K/16={k // 16} scales, got {a_scale.numel()}")
    if n % cfg.block_m != 0:
        raise ValueError(
            f"N={n} 不能被 block_m={cfg.block_m} 整除；本轮 swap-AB kernel 不支持 masked "
            f"尾块（模型 shape 全部整除，padding=0）"
        )
    if k % cfg.block_k != 0:
        raise ValueError(f"K={k} 不能被 block_k={cfg.block_k} 整除")

    if not w_packed.is_contiguous():
        w_packed = w_packed.contiguous()
    # 4-byte 视图：kernel 用 u32 载入再拆 byte lane（否则 Triton 只发 b8 load）。
    w32 = w_packed.view(torch.int32)
    if out is None:
        out = torch.empty(n, device=w_packed.device, dtype=torch.bfloat16)
    # alpha 作为设备标量传入（生产里是 input_scale * weight_scale_2），kernel 内 load，
    # 避免每步 .item() 的 host sync。
    if isinstance(alpha, torch.Tensor):
        alpha_arg = alpha.reshape(-1)
        if alpha_arg.numel() != 1 or alpha_arg.dtype != torch.float32:
            raise TypeError("alpha tensor 必须是 1 个 FP32 元素")
        if not alpha_arg.is_contiguous():
            alpha_arg = alpha_arg.contiguous()
    else:
        alpha_arg = torch.tensor([float(alpha)], device=w_packed.device, dtype=torch.float32)

    grid = (n // cfg.block_m,)
    _gemv_kernel[grid](
        w32,
        w_scale,
        a_packed.reshape(-1),
        a_scale.reshape(-1),
        out,
        alpha_arg,
        K=k,
        BM=cfg.block_m,
        BK=cfg.block_k,
        BN=cfg.block_n,
        CACHE=cfg.cache_modifier,
        STAGES=cfg.num_stages,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return out.view(1, n)


__all__ = ["CONFIGS", "Config", "gemv_tc", "padding_bytes", "workspace_bytes"]
