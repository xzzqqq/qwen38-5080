"""GDN decode（``sequence_length == 1``）recurrent step 的融合 Triton kernel。

问题：``torch_recurrent_gated_delta_rule`` 的 decode 分支是逐 token 的 PyTorch
展开，单层单步要做 ``l2norm``、``q / sqrt(128)``、decay、``kv_mem``、``delta``、
rank-1 更新和输出投影，每一步都在 ``[48,128,128]`` 的 FP32 recurrent state 上
生成新的全尺寸中间 tensor。按 state 元素数（786432，3.1 MB）估算，单层要走
约 12 次全尺寸 pass（约 44 MB），48 层约 2.1 GB/step；而 state 本身的必要流量
只有 1 读 1 写（6.3 MB/层，0.30 GB/step）。图里这些 pass 表现为
``direct_copy`` / ``BinaryFunctor`` 一类 <5 µs 的小 kernel。

数据流（单层、单 token）：
    q,k [16,128] BF16, v [48,128] BF16, g [48] FP32, beta [48] BF16
      -> 每个 (v_head, v-block) 一个 program：
         kn = l2norm(k[kh])                # kh = v_head // 3
         qn = l2norm(q[kh]) / sqrt(128)
         decay = exp(g[h]), d = beta[h]
         S = state[h, :, vblk] * decay     # [128, BV] FP32，一次读入后先衰减
         mem = kn^T S                      # kv_mem（reference 用的是衰减后的 state）
         delta = (v[h, vblk] - mem) * d
         S' = S + kn ⊗ delta               # 原地写回同一地址
         out[h, vblk] = qn^T S'
      v 维列与列之间完全独立（kv_mem / out 都是沿 k 维的 reduction），所以按 v
      切 block 不需要跨 program reduction，state 每个元素仍然只被读写一次。

布局约束（不可改）：recurrent state 是 static cache 里的
``[1, 48, 128, 128]`` contiguous FP32，必须原地更新、保持地址；conv state 是
BF16，本算子不碰。q/k 的 head 维按 stride 寻址（``kh = h // 3`` 映射），所以
既可以传 repeat_interleave 之前的 ``[1, 1, 16, 128]``，也可以传 repeat 之后的
``[:, :, ::3, :]`` view，两种都不需要额外拷贝。``decay`` 用 ``libdevice.exp``
（``__nv_expf``），与 ``torch.exp`` 在 FP32 上逐位一致。

预期瓶颈：纯 state 带宽。每层 3.1 MB 读 + 3.1 MB 写，48 层约 0.30 GB/step；
在 960 GB/s 上约 0.31 ms/step，实际受 48 个 program group 的 occupancy 和
``[128, BV]`` tile 的行宽（BV*4 B）影响。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# 粗粒度参数：v 维 block 与配套的 num_warps。三组都保持 state 元素只读写一次，
# 差别只在 tile 行宽（BV*4 B）、每线程寄存器数（K*BV/线程数）和 block 数。
BLOCK_V_CONFIGS: tuple[tuple[int, int], ...] = ((32, 4), (16, 2), (64, 8))
DEFAULT_BLOCK_V = 32
DEFAULT_NUM_WARPS = 4


@triton.jit
def _gdn_decode_kernel(
    q_ptr,  # *bf16 [num_k_heads, K]
    k_ptr,  # *bf16 [num_k_heads, K]
    v_ptr,  # *bf16 [num_v_heads, V]
    g_ptr,  # *fp32 [num_v_heads]
    b_ptr,  # *bf16|fp32 [num_v_heads]
    s_ptr,  # *fp32 [num_v_heads, K, V]，原地更新
    o_ptr,  # *bf16 [num_v_heads, V]
    ratio: tl.constexpr,  # num_v_heads // num_k_heads
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    q_stride: tl.constexpr,  # q 的 head 维步长（元素）：支持 repeat 之后的 strided view
    k_stride: tl.constexpr,
    sqrt_k: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blk: tl.constexpr = V // BV
    head = pid // n_blk
    blk = pid % n_blk
    kh = head // ratio

    kx = tl.arange(0, K)
    vx = blk * BV + tl.arange(0, BV)

    # l2norm 在 FP32 里做一次（与 reference 一致），再按 sqrt(head_dim) 缩放 q。
    q = tl.load(q_ptr + kh * q_stride + kx).to(tl.float32)
    k = tl.load(k_ptr + kh * k_stride + kx).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q) + eps) / sqrt_k
    k = k * tl.rsqrt(tl.sum(k * k) + eps)

    # libdevice.exp = __nv_expf，与 torch.exp 在 fp32 上逐位一致（tl.exp 走 ex2.approx）。
    decay = libdevice.exp(tl.load(g_ptr + head))
    beta = tl.load(b_ptr + head).to(tl.float32)
    val = tl.load(v_ptr + head * V + vx).to(tl.float32)

    offs = head * (K * V) + kx[:, None] * V + vx[None, :]
    # 注意顺序：reference 先对 state 乘 decay，再用 decay 后的 state 算 kv_mem。
    state = tl.load(s_ptr + offs) * decay
    mem = tl.sum(state * k[:, None], 0)
    delta = (val - mem) * beta
    state = state + k[:, None] * delta[None, :]
    # 原地写回：地址不变，graph replay 依赖这一点。
    tl.store(s_ptr + offs, state)
    out = tl.sum(state * q[:, None], 0)
    tl.store(o_ptr + head * V + vx, out.to(tl.bfloat16))


def gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_v: int = DEFAULT_BLOCK_V,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """单层、单 token 的 GDN recurrent step（原地更新 ``state``）。

    Args:
        q, k: BF16 ``[1, 1, num_k_heads, head_k_dim]``，最后一维 contiguous；
            允许 strided 的 head 维（接入时传 ``repeat_interleave`` 之后的
            ``[:, :, ::ratio, :]`` view，避免再拷一次）。base 必须是 storage 起点。
        v: BF16 ``[1, 1, num_v_heads, head_v_dim]``。
        g: FP32 ``[1, 1, num_v_heads]``，log decay（``decay = exp(g)``）。
        beta: BF16/FP32 ``[1, 1, num_v_heads]``。
        state: FP32 ``[1, num_v_heads, head_k_dim, head_v_dim]``，contiguous，
            原地更新；返回值就是它本身。
        out: 可选 BF16 ``[1, 1, num_v_heads, head_v_dim]`` 输出槽；不给就新分配。

    Returns:
        BF16 ``[1, 1, num_v_heads, head_v_dim]``。
    """
    if not state.is_cuda:
        raise ValueError("state 必须在 CUDA 上")
    if state.dtype != torch.float32:
        raise ValueError(f"recurrent state 必须是 float32，得到 {state.dtype}")
    if state.ndim != 4 or state.shape[0] != 1:
        raise ValueError(f"state 必须是 [1, v_heads, K, V]，得到 {tuple(state.shape)}")
    if not state.is_contiguous():
        raise ValueError("state 必须 contiguous（static cache 的预分配缓冲）")

    num_v_heads, k_dim, v_dim = state.shape[1], state.shape[2], state.shape[3]
    if v_dim % block_v != 0:
        raise ValueError(f"head_v_dim {v_dim} 不能被 block_v {block_v} 整除")
    if q.ndim < 2 or q.shape[-1] != k_dim:
        raise ValueError(f"q 最后一维应为 {k_dim}，得到 {tuple(q.shape)}")
    num_k_heads = q.shape[-2]
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"num_v_heads {num_v_heads} 不是 num_k_heads {num_k_heads} 的整数倍"
        )
    if q.numel() != num_k_heads * k_dim or k.shape != q.shape:
        raise ValueError("q/k 必须是单 token 的 [1, 1, num_k_heads, head_k_dim]")
    if v.shape[-1] != v_dim or v.numel() != num_v_heads * v_dim:
        raise ValueError("v 必须是单 token 的 [1, 1, num_v_heads, head_v_dim]")
    if g.numel() != num_v_heads or beta.numel() != num_v_heads:
        raise ValueError("g/beta 必须是 [1, 1, num_v_heads]")

    q = _flat(q, "q")
    k = _flat(k, "k")
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    if out is None:
        out = torch.empty(
            (1, 1, num_v_heads, v_dim), dtype=torch.bfloat16, device=state.device
        )
    elif out.shape[-2:] != (num_v_heads, v_dim) or out.dtype != torch.bfloat16:
        raise ValueError("out 必须是 bf16 [1, 1, num_v_heads, head_v_dim]")

    grid = (num_v_heads * (v_dim // block_v),)
    _gdn_decode_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        state,
        out,
        ratio=num_v_heads // num_k_heads,
        K=k_dim,
        V=v_dim,
        BV=block_v,
        q_stride=q.stride(-2),
        k_stride=k.stride(-2),
        sqrt_k=float(k_dim) ** 0.5,
        eps=1e-6,
        num_warps=num_warps,
    )
    return out


def _flat(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """校验 head 维可以只靠 stride 寻址，且 base 指向 storage 起点。"""
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} 的最后一维必须 stride=1")
    if tensor.storage_offset() != 0:
        raise ValueError(f"{name} 的 storage_offset 必须是 0，得到 {tensor.storage_offset()}")
    return tensor


__all__ = [
    "BLOCK_V_CONFIGS",
    "DEFAULT_BLOCK_V",
    "DEFAULT_NUM_WARPS",
    "gdn_decode_step",
]
