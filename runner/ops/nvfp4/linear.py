"""NVFP4Linear：用 FlashInfer ``mm_fp4`` 实现的 W4A4 Linear。

设计：
- 权重在加载时保持 checkpoint 布局：U8 [N, K/2] + F8 [N, K/16] + F32 global；
- 构造时把 weight_scale 转成 128x4 swizzled 布局，并且只保留 swizzled 这一份，
  避免全模型多占约 1.6 GB 显存；
- forward：BF16 activation -> FlashInfer ``nvfp4_quantize`` -> ``mm_fp4`` -> BF16；
- 这是 M2 的正确性实现，不追求性能；Phase 2 再替换成自写 kernel。

数据流：
    x [M, K] BF16
      -> nvfp4_quantize(x, 1/input_scale) -> a_fp4 [M, K/2], a_sf [padM, K/16]
      -> mm_fp4(a_fp4, w_fp4.T, a_sf, w_sf_swizzled.T, alpha=input_scale*w_scale_2)
      -> y [M, N] BF16

预期瓶颈：M=1 时是权重带宽受限；M2 只做 correctness，性能在 M3/Phase 2。

环境要求：FlashInfer JIT 需要 nvcc 能找到 CUDA 头文件。运行时设置
``CUDA_HOME`` 和 ``CPATH``（含 ``nvidia/cu13/include``），见
``scripts/m2_env.sh`` / ``scripts/m2_env.fish``。
"""

from __future__ import annotations

import torch
from torch import nn

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

from .reference import swizzle_scale_128x4


class NVFP4Linear(nn.Module):
    """替换 checkpoint 里的量化 Linear。

    Args:
        weight_packed: U8 [N, K/2]，checkpoint 的 ``weight``。
        weight_scale: F8_E4M3 [N, K/16]，checkpoint 的 ``weight_scale``。
        weight_scale_2: 标量 F32，checkpoint 的 ``weight_scale_2``。
        input_scale: 标量 F32，checkpoint 的 ``input_scale``。
        bias: 可选 BF16 [N]；M2 的 checkpoint 大多没有 bias。
        backend: FlashInfer backend，M2 固定 ``cutlass``，避免未记录的 ``auto``。
        out_dtype: 输出 dtype，默认 BF16。
    """

    def __init__(
        self,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        input_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        backend: str = "cutlass",
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if weight_packed.ndim != 2 or weight_packed.dtype != torch.uint8:
            raise TypeError(
                f"weight_packed must be uint8 [N, K/2], got {weight_packed.dtype} {tuple(weight_packed.shape)}"
            )
        if weight_scale.ndim != 2 or weight_scale.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"weight_scale must be float8_e4m3fn [N, K/16], got {weight_scale.dtype} {tuple(weight_scale.shape)}"
            )
        n, k_half = weight_packed.shape
        k = k_half * 2
        if weight_scale.shape != (n, k // 16):
            raise ValueError(
                f"weight_scale shape {tuple(weight_scale.shape)} != {(n, k // 16)}"
            )

        self.in_features = k
        self.out_features = n
        self.backend = backend
        self.out_dtype = out_dtype

        self.register_buffer("weight_packed", weight_packed.contiguous())
        self.register_buffer("weight_scale_2", weight_scale_2.reshape(1).float())
        self.register_buffer("input_scale", input_scale.reshape(1).float())
        # 只保留 swizzled scale；原始 linear scale 在构造后释放，避免双份显存。
        self.register_buffer(
            "weight_scale_swizzled",
            swizzle_scale_128x4(weight_scale.contiguous()),
        )
        self.register_buffer("bias", None if bias is None else bias.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"x must be [M, {self.in_features}], got {tuple(x.shape)}"
            )
        if not x.is_contiguous():
            x = x.contiguous()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # checkpoint 的 input_scale 是反量化乘数；量化用它的倒数作为 global divisor。
        global_divisor = (1.0 / self.input_scale).reshape(1)
        a_fp4, a_sf = nvfp4_quantize(
            x,
            global_divisor,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )
        alpha = (self.input_scale * self.weight_scale_2).reshape(1).float()
        out = mm_fp4(
            a_fp4,
            self.weight_packed.T,
            a_sf,
            self.weight_scale_swizzled.T,
            alpha,
            self.out_dtype,
            None,
            block_size=16,
            use_8x4_sf_layout=False,
            backend=self.backend,
            use_nvfp4=True,
        )
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"backend={self.backend!r}"
        )
