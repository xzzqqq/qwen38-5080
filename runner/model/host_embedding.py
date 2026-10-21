"""Host 侧 BF16 embedding：CPU gather 后只把需要的行传到 GPU。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class HostEmbedding(nn.Module):
    """embedding 权重常驻 host，forward 只传 gather 后的结果。

    这样 2.543 GB 的 BF16 embedding 不占 GPU 显存；代价是每个 token 一次
    H2D 传输（decode 时只有一行）。
    """

    def __init__(self, weight: torch.Tensor, output_device: torch.device):
        super().__init__()
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"embedding weight must be bf16, got {weight.dtype}")
        # 不放 state_dict，也不随 .to(device) 搬走。
        self.register_buffer("weight", weight.contiguous(), persistent=False)
        self.output_device = torch.device(output_device)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.to("cpu", non_blocking=False)
        out = F.embedding(ids, self.weight)
        return out.to(self.output_device, non_blocking=True)
