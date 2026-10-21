"""BF16 Linear 的 shape-stable 包装。

问题：同一个 `nn.Linear` 在 M=300 和 M=128 下，PyTorch 可能选择不同 GEMM
kernel/reduction 顺序，导致 BF16 输出有 1e-2 量级差异。`in_proj_a/b` 的输出
决定 GDN 的 decay / update gate，这种差异会被后续 W4A4 quantization 放大。

做法：固定按 `block_m` 行分块，尾块补零后做同一 shape 的 F.linear，再截掉
padding。这样同一行 token 无论在单段还是分段 prefill 中，都走相同 M 的 GEMM；
a/b 很小，padding 计算量可忽略。

数据流：x [..., K] -> reshape [M, K] -> 每 block_m 行 pad -> F.linear -> 截断 -> [..., N]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ShapeStableLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, block_m: int = 128):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
        if block_m <= 0:
            raise ValueError("block_m must be positive")
        self.block_m = int(block_m)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.weight = weight
        self.bias = bias

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_m: int = 128) -> "ShapeStableLinear":
        return cls(linear.weight, linear.bias, block_m=block_m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"x last dim {x.shape[-1]} != in_features {self.in_features}"
            )
        original_shape = x.shape
        flat = x.reshape(-1, self.in_features)
        rows = flat.shape[0]
        outputs = []
        for start in range(0, rows, self.block_m):
            chunk = flat[start : start + self.block_m]
            valid = chunk.shape[0]
            if valid < self.block_m:
                pad = chunk.new_zeros(self.block_m - valid, self.in_features)
                chunk = torch.cat([chunk, pad], dim=0)
            y = F.linear(chunk, self.weight, self.bias)
            outputs.append(y[:valid])
        out = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        return out.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_m={self.block_m}, bias={self.bias is not None}"
        )
