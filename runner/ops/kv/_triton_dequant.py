"""FP8 KV 反量化的融合 Triton kernel：读 fp8 + fp32 scale，直接写 bf16。

设计：
- 纯 torch 的 ``q.float() * scale`` 在 fp8 上会被拆成“转换 + 乘法 + 再 cast”两三个
  elementwise kernel，decode 每层要读 8.4 MB fp8、写 33.6 MB fp32 再读回来，
  比 BF16 KV 直接读还贵；这里用一次 pass 完成 ``bf16(float32(q) * scale)``。
- 行布局与 cache 一致：``q [1, H, L, D]``，scale ``[L, H]``（行是 token、列是 kv head）。
  ``STRIDE_H`` 单独传入，因为 static-exact/static-cap 的 K/V 是 capacity buffer 的
  前缀 view，head 维 stride 是 ``capacity * D`` 而不是 ``L * D``。
- 数值契约（见 ``fp8.py``）：fp32 乘法 + RNE cast bf16。op gate 用 fp64 精确乘
  逐位验证，e2e 里 Triton 路径与 torch reference 必须 bitwise 相同。

数据流（每个 program 处理 ``BLOCK_T`` 行 × ``BLOCK_D`` 个 dim）：
    q_ptr（fp8 行） + s_ptr（每行一个 fp32 scale） -> out_ptr（bf16 行）
预期瓶颈：HBM 读 fp8 + 写 bf16，2:1 的读写比，受显存带宽限制。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK_T = 8


@triton.jit
def _dequant_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    L,
    STRIDE_H,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = tl.arange(0, BLOCK_D)
    n_rows = L * H
    row_ok = rows < n_rows
    head = rows // L
    tok = rows % L
    q_off = head * STRIDE_H + tok * D
    mask = row_ok[:, None] & (cols[None, :] < D)
    x = tl.load(q_ptr + q_off[:, None] + cols[None, :], mask=mask, other=0.0)
    scale = tl.load(s_ptr + tok * H + (head % H), mask=row_ok, other=0.0)
    y = x.to(tl.float32) * scale[:, None]
    tl.store(out_ptr + rows[:, None] * D + cols[None, :], y.to(tl.bfloat16), mask=mask)


def dequant_fp8_bf16(q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """把 ``q [1, H, L, D]`` fp8 按 ``scale [L, H]`` 反量化进 ``out [1, H, L, D]`` bf16。

    调用方（``fp8.dequantize``）已经检查过布局；这里只做形状/stride 的最终断言。
    """
    if q.dim() != 4 or q.shape[0] != 1:
        raise ValueError(f"q 必须是 [1, H, L, D]，得到 {tuple(q.shape)}")
    _, h, length, d = q.shape
    if tuple(scale.shape) != (length, h):
        raise ValueError(f"scale 必须是 [L, H]={length, h}，得到 {tuple(scale.shape)}")
    grid = (triton.cdiv(h * length, BLOCK_T),)
    _dequant_kernel[grid](
        q,
        scale,
        out,
        length,
        q.stride(1),
        H=h,
        D=d,
        BLOCK_T=BLOCK_T,
        BLOCK_D=triton.next_power_of_2(d),
    )
    return out


__all__ = ["BLOCK_T", "dequant_fp8_bf16"]
