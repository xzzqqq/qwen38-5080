"""M3 验证用 reference cache：padded DynamicCache。

用途：为 ``static-cap`` 提供一份**独立**的对照实现。它沿用 M2 DynamicCache 的
追加式存储、自己的有效长度计数和 mask 尺寸，只在 decode（q_len==1）的 attention
之前把 K/V 补有限值到 capacity。

刻意不复用 ``runner/model/static_cache.py`` 的索引、位置更新和 mask 逻辑，
避免两边共享同一个错误。

数据流（decode）：
    K/V [1,4,L,256]（dynamic 追加）
      -> F.pad 到 [1,4,capacity,256]，填充值 fill（有限）
      -> 模型自己的 create_causal_mask(kv=capacity, q_offset=L)
      -> eager attention

预期瓶颈：等价于 static-cap 的扫描代价；它只用于验证，不进入性能表。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer, LinearAttentionLayer


class PaddedDynamicAttentionLayer(DynamicLayer):
    """decode 时把 K/V 补到 capacity 的 dynamic attention 层。"""

    def __init__(self, capacity: int, fill: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.capacity = int(capacity)
        self.fill = float(fill)
        self.effective_len = 0

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = DynamicLayer.update(self, key_states, value_states)
        self.effective_len = keys.shape[-2]
        if key_states.shape[-2] == 1 and self.effective_len < self.capacity:
            pad = self.capacity - self.effective_len
            keys = F.pad(keys, (0, 0, 0, pad), value=self.fill)
            values = F.pad(values, (0, 0, 0, pad), value=self.fill)
        return keys, values

    def get_seq_length(self) -> int:
        return self.effective_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if query_length == 1:
            return self.capacity, 0
        return self.effective_len + query_length, 0

    def get_max_length(self) -> int:
        return self.capacity

    def reset(self) -> None:
        DynamicLayer.reset(self)
        self.effective_len = 0


class PaddedDynamicCache(Cache):
    """按 config.layer_types 组装的 padded dynamic cache。"""

    def __init__(self, layers: list, capacity: int, fill: float):
        super().__init__(layers=layers)
        self.capacity = int(capacity)
        self.fill = float(fill)

    def __repr__(self) -> str:
        return f"PaddedDynamicCache(capacity={self.capacity}, fill={self.fill})"


def build_padded_reference(config, capacity: int, fill: float = 0.0) -> PaddedDynamicCache:
    layers: list = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "linear_attention":
            layers.append(LinearAttentionLayer())
        elif layer_type == "full_attention":
            layers.append(PaddedDynamicAttentionLayer(capacity=capacity, fill=fill))
        else:
            raise ValueError(f"unsupported layer_type {layer_type!r} at {layer_idx}")
    return PaddedDynamicCache(layers=layers, capacity=capacity, fill=fill)
