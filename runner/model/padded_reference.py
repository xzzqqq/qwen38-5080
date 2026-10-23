"""M3/Phase 2 验证用 reference cache：padded DynamicCache + FP8 dynamic reference。

用途：
- ``PaddedDynamicCache``：为 ``static-cap`` 提供一份**独立**的对照实现。它沿用
  M2 DynamicCache 的追加式存储、自己的有效长度计数和 mask 尺寸，只在 decode
  （q_len==1）的 attention 之前把 K/V 补有限值到 capacity。
- ``FP8DynamicCache``：FP8 模式下 ``eager`` 与 padded reference 的存储实现。
  它同样用追加式存储和独立的 mask 尺寸，但按 ``runner/ops/kv/fp8.py`` 的契约
  量化 K/V：**调用 reference 量化器**（``reference_quantize_kv``），不调用
  ``static_cache`` 里的候选量化器，所以 e2e 的“padded-ref -> static-cap”仍然
  是两份独立实现的对拍。

刻意不复用 ``runner/model/static_cache.py`` 的索引、位置更新和 mask 逻辑，
避免两边共享同一个错误。

数据流（decode）：
    K/V 新 token -> reference_quantize_kv -> 追加 fp8 + scale
      -> （padded 时）fp8 u8 view 补到 capacity，scale 补 0
      -> 返回 fp8 + ``[L,H]`` scale（compact_decode 反量化）
数据流（prefill，q_len>1）：
    K/V -> fp8 + scale -> 反量化成连续 bf16 前缀 -> HF eager attention

预期瓶颈：等价于 static-cap 的扫描代价 + 每步整段前缀反量化；它只用于验证，
不进入性能表。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicLayer, DynamicCache, LinearAttentionLayer

from runner.ops.kv import fp8 as fp8_kv


class PaddedDynamicAttentionLayer(DynamicLayer):
    """decode 时把 K/V 补到 capacity 的 dynamic attention 层（BF16）。"""

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

    def __init__(self, layers: list, capacity: int, fill: float, kv_format: str = "bf16"):
        super().__init__(layers=layers)
        self.capacity = int(capacity)
        self.fill = float(fill)
        self.kv_format = fp8_kv.kfmt(kv_format)

    def __repr__(self) -> str:
        return (
            f"PaddedDynamicCache(capacity={self.capacity}, fill={self.fill}, "
            f"kv_format={self.kv_format!r})"
        )


class FP8DynamicAttentionLayer(DynamicLayer):
    """FP8 的追加式 KV 存储：code + ``[L,H]`` fp32 scale，独立于 static cache。"""

    def __init__(
        self,
        *,
        capacity: int | None = None,
        fill: float = 0.0,
        padded: bool = False,
    ):
        super().__init__()
        self.capacity = int(capacity) if capacity is not None else None
        self.fill = float(fill)
        self.padded = bool(padded)
        self.keys = None
        self.values = None
        self.k_scale = None
        self.v_scale = None
        self.effective_len = 0
        self.is_initialized = False

    # ----------------------------------------------------------------- write
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qk, sk = fp8_kv.reference_quantize_kv(key_states)
        qv, sv = fp8_kv.reference_quantize_kv(value_states)
        if not self.is_initialized:
            self.dtype, self.device = qk.dtype, qk.device
            self.keys, self.values = qk, qv
            self.k_scale, self.v_scale = sk, sv
            self.is_initialized = True
        else:
            # fp8 只做搬运：拼接走 uint8 view，避免依赖 fp8 的 cat kernel。
            self.keys = torch.cat(
                [self.keys.view(torch.uint8), qk.view(torch.uint8)], dim=-2
            ).view(fp8_kv.FP8_DTYPE)
            self.values = torch.cat(
                [self.values.view(torch.uint8), qv.view(torch.uint8)], dim=-2
            ).view(fp8_kv.FP8_DTYPE)
            self.k_scale = torch.cat([self.k_scale, sk], dim=0)
            self.v_scale = torch.cat([self.v_scale, sv], dim=0)
        self.effective_len = int(self.keys.shape[-2])
        decode = key_states.shape[-2] == 1
        if decode and self.padded and self.effective_len < self.capacity:
            pad = self.capacity - self.effective_len
            # 填充值也走 fp8 code：fill=3 的隔离检查才有意义（mask 会把它遮掉，
            # 但“不可见槽位不参与输出”必须由 mask 保证，而不是靠填 0）。
            fill_code = int(
                torch.tensor([self.fill], dtype=torch.bfloat16)
                .to(fp8_kv.FP8_DTYPE)
                .view(torch.uint8)
                .item()
            )
            keys = F.pad(self.keys.view(torch.uint8), (0, 0, 0, pad), value=fill_code).view(
                fp8_kv.FP8_DTYPE
            )
            values = F.pad(
                self.values.view(torch.uint8), (0, 0, 0, pad), value=fill_code
            ).view(fp8_kv.FP8_DTYPE)
            k_scale = F.pad(self.k_scale, (0, 0, 0, pad), value=self.fill)
            v_scale = F.pad(self.v_scale, (0, 0, 0, pad), value=self.fill)
            return (
                fp8_kv.attach_scale(keys, k_scale),
                fp8_kv.attach_scale(values, v_scale),
            )
        if decode:
            return (
                fp8_kv.attach_scale(self.keys, self.k_scale),
                fp8_kv.attach_scale(self.values, self.v_scale),
            )
        length = self.effective_len
        return (
            fp8_kv.reference_dequantize(self.keys[:, :, :length], self.k_scale[:length]),
            fp8_kv.reference_dequantize(self.values[:, :, :length], self.v_scale[:length]),
        )

    # ------------------------------------------------------------------ read
    def get_seq_length(self) -> int:
        return self.effective_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if self.padded and query_length == 1:
            return self.capacity, 0
        return self.effective_len + query_length, 0

    def get_max_length(self) -> int:
        return self.capacity if self.capacity is not None else -1

    def reset(self) -> None:
        self.keys = None
        self.values = None
        self.k_scale = None
        self.v_scale = None
        self.effective_len = 0
        self.is_initialized = False


class FP8DynamicCache(Cache):
    """按 config.layer_types 组装的 FP8 dynamic cache（eager / padded reference）。"""

    def __init__(self, layers: list, *, padded: bool, capacity: int | None, fill: float):
        super().__init__(layers=layers)
        self.padded = bool(padded)
        self.capacity = int(capacity) if capacity is not None else None
        self.fill = float(fill)
        self.kv_format = "fp8"

    def __repr__(self) -> str:
        return (
            f"FP8DynamicCache(padded={self.padded}, capacity={self.capacity}, "
            f"fill={self.fill})"
        )


def _build_layers(config, factory) -> list:
    layers: list = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "linear_attention":
            layers.append(LinearAttentionLayer())
        elif layer_type == "full_attention":
            layers.append(factory(layer_idx))
        else:
            raise ValueError(f"unsupported layer_type {layer_type!r} at {layer_idx}")
    return layers


def build_padded_reference(
    config, capacity: int, fill: float = 0.0, kv_format: str | None = None
):
    """padded reference：BF16 用 PaddedDynamicCache，FP8 用 FP8DynamicCache(padded)。"""
    fmt = fp8_kv.resolve_format(kv_format)
    if fmt == "fp8":
        layers = _build_layers(
            config,
            lambda _idx: FP8DynamicAttentionLayer(
                capacity=capacity, fill=fill, padded=True
            ),
        )
        return FP8DynamicCache(layers, padded=True, capacity=capacity, fill=fill)
    layers = _build_layers(
        config, lambda _idx: PaddedDynamicAttentionLayer(capacity=capacity, fill=fill)
    )
    return PaddedDynamicCache(layers=layers, capacity=capacity, fill=fill, kv_format="bf16")


def build_dynamic_reference(config, kv_format: str | None = None):
    """eager 的 cache：BF16 用原生 DynamicCache；FP8 用同一契约的 dynamic 存储。"""
    fmt = fp8_kv.resolve_format(kv_format)
    if fmt == "fp8":
        layers = _build_layers(
            config, lambda _idx: FP8DynamicAttentionLayer(padded=False)
        )
        return FP8DynamicCache(layers, padded=False, capacity=None, fill=0.0)
    return DynamicCache(config=config)
