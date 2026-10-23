"""生产者侧 elementwise + NVFP4 activation quant 的融合 Triton kernel（M=1 decode）。

设计
====
第 2 轮任务 H 把「RMSNorm -> 每消费者 activation quant」融成了一个 kernel，覆盖 272 个
消费者。但还有 129 个消费者的输入不是 norm 输出，而是**上一层已经算好的 elementwise
结果**，它们仍然各自走一遍 ``nvfp4_quantize``（外加 ``1/input_scale`` 的 reciprocal
kernel 和 elementwise 本身）：

    down_proj   <- silu(gate_proj_out) * up_proj_out        # 64 层，每次 2 个 elementwise
    o_proj      <- attn_output * sigmoid(gate)              # 16 层（已在 attention.forward 里）
    lm_head     <- final norm 输出                          # 1 次（norm 本身也可以一起融）

本模块负责前两个（lm_head 的 final norm 直接复用 ``norm_quant.norm_quant_fused`` 的
单消费者形态）。一次 launch 同时完成：

    1) 按 torch 的机器算术算 producer elementwise，结果 round 回 BF16 并写出；
    2) 对写出的 BF16 值按 16-element block 算 amax；
    3) 按任务 E 冻结的规则 R2（见 ``norm_quant.py`` docstring）产 ``(a_fp4, a_sf)``。

bitwise 契约（与模型实际执行的两步路径逐字节相同）
==================================================
模型里的 producer 是两次独立的 CUDA kernel，中间结果落在 BF16：

    act_bf16 = ACT2FN["silu"](g)            # 或 torch.sigmoid(gate)
    prod     = act_bf16 * u                 # torch mul：opmath f32 乘，再 round 到 BF16
    a_fp4, a_sf = nvfp4_quantize(prod.reshape(1, K), 1 / input_scale, ...)

所以 kernel 内部不能把「silu/sigmoid 的 f32 值直接乘 u」：必须先把 activation round
回 BF16，再乘、再 round。每一步的机器算术：

- ``silu``（``aten::silu`` / ``nn.SiLU``，CUDA ``silu_kernel``）：
  ``x_acc / (1 + expf(-x_acc))``，opmath f32，结果 cast BF16；
- ``sigmoid``（``aten::sigmoid``，CUDA ``sigmoid_kernel``）：
  ``1 / (1 + expf(-x_acc))``，同上；
- ``expf`` 是 libdevice 的 ``__nv_expf``（PyTorch 编译时没有 fast-math）；
  Triton 侧用 backend 的 ``triton.language.extra.cuda.libdevice.exp``（同一个 bitcode）；
- 两个除法都必须显式 ``div.rn.f32``（``libdevice.div_rn``），不能用 fdiv 近似；
- ``* u`` 与 quant 的 contract 同 ``norm_quant.py``：f32 乘（``mul.rn``）后 round 到
  BF16，再按 R2 算 ``sf -> e4m3 -> out_scale -> e2m1``，两个倒数都用
  ``rcp.approx.ftz``。

amax / sf / code 的生成直接复用 ``norm_quant._emit_consumer``，保证与 H 的 272 个
消费者**同一条代码路径**，不出现两套量化实现。

数据流（一个 program 处理 64 个 16-element block = 1024 个元素）
==============================================================
    a[1024] BF16 (attn_output / gate_proj_out)
    b[1024] BF16 (gate / up_proj_out)
      -> f32 -> exp(-a) -> div.rn -> act f32 -> act BF16（与 torch 一致）
      -> * b（f32）-> prod BF16（写 out_bf16，供 fallback / 语义完整）
      -> amax（每 16 个）-> 每消费者 sf/e4m3/out_scale/codes -> a_fp4/a_sf
    grid = K / 1024：down_proj 17、o_proj 6（K=6144）、lm_head 5（K=5120，走 norm 路径）。

K 必须是 1024 的倍数（checkpoint 的 17408 / 6144 / 5120 都满足）；不满足时调用方回退。

预期瓶颈
========
单次调用只读写 4–12 KB，算力/带宽都不是瓶颈；收益来自 kernel 数量：现有路径每个
消费者是「silu/sigmoid + mul + quantize + reciprocal」4 个 kernel，融合后 1 个。
所以真正要压的是单 kernel 固定开销；``num_warps`` 由 ``DEFAULT_NUM_WARPS`` 控制，
op 脚本扫 2/4/8 三档（诊断用，接入固定 4）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .norm_quant import (
    BLOCK,
    C6,
    C_GROUP_STRIDE,
    C_PAIR,
    CHUNK_BLOCKS,
    DEFAULT_NUM_WARPS,
    SF_ROWS,
    _emit_consumer,
)

#: producer 的种类：silu(g)*u 与 attn*sigmoid(gate)。
KIND_SILU_MUL = 0
KIND_SIGMOID_MUL = 1
KINDS = {"silu_mul": KIND_SILU_MUL, "sigmoid_mul": KIND_SIGMOID_MUL}

#: 一个 program 处理的元素数 = CHUNK_BLOCKS(64) * BLOCK(16)；必须与 ``_emit_consumer``
#: 里的 reshape 形状一致，所以固定等于 ``CHUNK_BLOCKS``。
TILE_BLOCKS = CHUNK_BLOCKS
TILE_ELEMS = TILE_BLOCKS * BLOCK

#: ``exp`` 的 Triton 绑定：**backend 的** libdevice 才是真实现
#: （``triton.language.extra.libdevice`` 只是 stub），fp32 映射到 ``__nv_expf``，
#: 与 PyTorch 编译出的 ``expf`` 同源；``div_rn`` 映射到 ``__nv_fdiv_rn``（div.rn.f32）。
_exp = libdevice.exp
_div_rn = libdevice.div_rn

C_BLOCK = tl.constexpr(BLOCK)
C_CHUNK_BLOCKS = tl.constexpr(TILE_BLOCKS)


@triton.jit
def _producer_quant_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    c6,
    gs,
    a_fp4_ptr,
    sf_ptr,
    KIND: tl.constexpr,
):
    """一个 program：1024 个元素的 producer elementwise + 单消费者 NVFP4 quant。"""
    pid = tl.program_id(0)
    blocks = pid * C_CHUNK_BLOCKS + tl.arange(0, C_CHUNK_BLOCKS)
    elems = tl.arange(0, C_BLOCK)
    offs = blocks[:, None] * C_BLOCK + elems[None, :]

    av = tl.load(a_ptr + offs).to(tl.float32)
    bv = tl.load(b_ptr + offs).to(tl.float32)

    # torch 的机器算术：opmath f32 上的 expf + div.rn，然后一次 BF16 cast。
    denom = _exp(-av) + 1.0
    if KIND == 0:  # KIND_SILU_MUL: x / (1 + exp(-x))
        act = _div_rn(av, denom)
    else:  # KIND_SIGMOID_MUL: 1 / (1 + exp(-x))
        act = _div_rn(1.0, denom)
    act_bf = act.to(tl.bfloat16)

    # torch mul 是 opmath f32 乘再 round 到 BF16：先 cast 回来，乘完再 cast。
    prod = (act_bf.to(tl.float32) * bv).to(tl.bfloat16)
    tl.store(out_ptr + offs, prod)

    yqf = prod.to(tl.float32)
    amax = tl.max(tl.abs(yqf), axis=1)

    pair = tl.arange(0, C_PAIR)
    a_off = blocks[:, None] * C_PAIR + pair[None, :]
    sf_off = (blocks // 4) * C_GROUP_STRIDE + (blocks % 4)
    # 复用 H 的 R2 实现：两套量化语义必须完全一致。
    _emit_consumer(yqf, amax, gs, c6, a_fp4_ptr, sf_ptr, a_off, sf_off)


# ------------------------------------------------------------------ 宿主入口


def _as_u8(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.element_size() != 1:
        raise TypeError(f"expected a 1-byte dtype, got {tensor.dtype}")
    return tensor.view(torch.uint8)


def supports_k(k: int) -> bool:
    """K 必须是 1024 的倍数（checkpoint 的 5120/6144/17408 都满足）。"""
    return int(k) % TILE_ELEMS == 0


def producer_quant_fused(
    a: torch.Tensor,
    b: torch.Tensor,
    kind: str,
    gs: float,
    out_bf16: torch.Tensor,
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> None:
    """一次融合 launch：producer elementwise（BF16 输出）+ 单消费者 NVFP4 A。

    Args:
        a: 第一个输入（silu 的输入 / attention 输出），BF16 ``[1, 1, K]`` 或 ``[K]``。
        b: 第二个输入（up_proj 输出 / gate），同 shape。
        kind: ``"silu_mul"``（``silu(a) * b``）或 ``"sigmoid_mul"``
            （``a * sigmoid(b)``）。
        gs: 消费者的 global divisor ``fl32(1/input_scale)``。
        out_bf16: BF16 ``[1, 1, K]`` 输出缓冲（elementwise 结果，已 round 到 BF16）。
        a_fp4: U8 ``[1, K/2]``（或 ``[K/2]``）。
        a_sf: U8 ``[128, K/16]`` swizzled；padding 必须已清零。
        num_warps: program 的 warp 数（只影响性能）。
    """
    if kind not in KINDS:
        raise ValueError(f"unknown producer kind {kind!r}; expected {sorted(KINDS)}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"a/b must be bf16, got {a.dtype} / {b.dtype}")
    k = a.numel()
    if b.numel() != k:
        raise ValueError(f"a/b element count mismatch: {k} vs {b.numel()}")
    if not supports_k(k):
        raise ValueError(f"kernel requires K % {TILE_ELEMS} == 0, got K={k}")
    if out_bf16.numel() != k:
        raise ValueError(f"out_bf16 has {out_bf16.numel()} elements, expected {k}")
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()
    if not out_bf16.is_contiguous():
        # 输出不能悄悄 copy：调用方拿到的是原缓冲，copy 会让写入丢失。
        raise ValueError("out_bf16 must be contiguous")

    a_u8 = _as_u8(a_fp4)
    sf_u8 = _as_u8(a_sf)
    if a_u8.numel() != k // 2:
        raise ValueError(f"a_fp4 has {a_u8.numel()} bytes, expected {k // 2}")
    if sf_u8.numel() != SF_ROWS * (k // BLOCK):
        raise ValueError(
            f"a_sf has {sf_u8.numel()} bytes, expected {SF_ROWS * (k // BLOCK)}"
        )

    _producer_quant_kernel[(k // TILE_ELEMS,)](
        a,
        b,
        out_bf16,
        C6,
        float(gs),
        a_u8,
        sf_u8,
        KIND=KINDS[kind],
        num_warps=int(num_warps),
    )


def allocate_scratch(
    k: int,
    *,
    device: torch.device | str = "cuda",
    a_fp4_shape: tuple[int, ...] | None = None,
    a_sf_shape: tuple[int, ...] | None = None,
    a_fp4_dtype: torch.dtype = torch.uint8,
    a_sf_dtype: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """分配 1 份 BF16 producer 输出 + 1 份 (a_fp4, a_sf) 槽位。

    shape/dtype 由调用方按 FlashInfer ``nvfp4_quantize`` 的实际输出传进来（不猜）；
    ``a_sf`` 的 padding 必须为 0，所以用 ``zeros``。
    """
    out = torch.empty(1, 1, int(k), dtype=torch.bfloat16, device=device)
    a_fp4 = torch.empty(
        a_fp4_shape or (1, int(k) // 2), dtype=a_fp4_dtype, device=device
    )
    a_sf = torch.zeros(
        a_sf_shape or (SF_ROWS, int(k) // BLOCK), dtype=a_sf_dtype, device=device
    )
    return out, a_fp4, a_sf


__all__ = [
    "KINDS",
    "KIND_SIGMOID_MUL",
    "KIND_SILU_MUL",
    "TILE_ELEMS",
    "allocate_scratch",
    "producer_quant_fused",
    "supports_k",
]
