"""M3 static cache adapter：预分配 KV / GDN / conv state 两种 decode attention shape。

设计：
- 用 transformers 5.20.2 的 hybrid cache 层（``LinearAttentionLayer`` +
  ``LinearAttentionAndStaticFullAttentionLayer``）作为存储基础，只覆写
  ``update`` / ``get_mask_sizes``，不改 attention 数学；
- KV 在第一次 forward 之前按 capacity 预分配（``early_initialization``），
  地址固定、原地写入，供 CUDA Graph replay；
- 两种模式只差 decode 时 attention 看到的 KV 长度：
  * ``exact``：返回 ``keys[:, :, :L]`` 前缀 view，mask 长度也是 L，
    用于单独观察“存储方式变化”；
  * ``cap``：decode（q_len==1）返回整块 capacity，mask 由 device 上的有效长度
    控制可见范围；prefill（q_len>1）仍用有效前缀，避免 8K prefill 扫满 capacity；
- ``cumulative_length`` 是 device 上的 0-dim tensor，graph 内自行推进；
  ``exact`` 模式需要 ``int(...)`` 会产生同步，因此该模式不用于 capture。

数据流（decode 单步）：
    token -> embed row [1,1,5120]
      -> 48 层 GDN（conv/recurrent state 原地更新）
      -> 16 层 full attention（K/V 原地写入，读 L 或 capacity）
      -> lm_head -> logits [1,1,248320]

预期瓶颈：``exact`` 的 ``repeat_kv`` 把 4 个 KV head 展开到 24 个 query head，
KV 展开流量随 L 线性增长；``cap`` 在 capacity 远大于 L 时把这份流量放大到
capacity。两者都属于 M3 要量出的“额外流量”，不是有效模型字节。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import (
    Cache,
    LinearAttentionAndStaticFullAttentionLayer,
    LinearAttentionLayer,
    StaticLayer,
)

MODES = ("exact", "cap")


class StaticExactFullAttentionLayer(LinearAttentionAndStaticFullAttentionLayer):
    """KV 静态存储 + decode attention 只读有效前缀。"""

    # 覆写后可以在 prefill 之前就分配 KV；lazy_initialization 的 dispatch 仍由
    # LinearAttentionAndStaticFullAttentionLayer 提供（2 个位置参数 -> StaticLayer）。
    supports_early_init = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = StaticLayer.update(self, key_states, value_states)
        length = int(self.cumulative_length)
        return keys[:, :, :length], values[:, :, :length]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # mask 的 kv 长度必须和 update 返回的前缀一致：本步写入后总长度为
        # (本步之前已缓存数) + query_length。
        return int(self.cumulative_length) + query_length, 0


class StaticCapFullAttentionLayer(LinearAttentionAndStaticFullAttentionLayer):
    """KV 静态存储 + decode attention 扫满 capacity，prefill 仍读有效前缀。"""

    supports_early_init = True

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, values = StaticLayer.update(self, key_states, value_states)
        if key_states.shape[-2] == 1:
            # decode：固定 shape，mask 由 device 上的 cumulative_length 控制可见范围。
            return keys, values
        length = int(self.cumulative_length)
        return keys[:, :, :length], values[:, :, :length]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if query_length == 1:
            return self.max_cache_len, 0
        return int(self.cumulative_length) + query_length, 0


LAYER_CLASSES = {
    "exact": StaticExactFullAttentionLayer,
    "cap": StaticCapFullAttentionLayer,
}


class StaticAdapterCache(Cache):
    """按 ``config.layer_types`` 组装的 static cache。

    Args:
        layers: 预构造的 cache layer 列表（每个 decoder layer 一个）。
        mode: ``exact`` 或 ``cap``。
        capacity: KV capacity，仅用于记账/校验。
    """

    def __init__(self, layers: list, mode: str, capacity: int):
        super().__init__(layers=layers)
        if mode not in MODES:
            raise ValueError(f"unknown static cache mode {mode!r}")
        self.mode = mode
        self.capacity = int(capacity)

    def __repr__(self) -> str:
        return f"StaticAdapterCache(mode={self.mode!r}, capacity={self.capacity})"


def build_static_cache(
    config,
    capacity: int,
    mode: str = "exact",
) -> StaticAdapterCache:
    """按模型 config 的 layer_types 构造 hybrid static cache。

    只支持本项目模型的两种层：``linear_attention``（GDN，48 层）与
    ``full_attention``（KV，16 层）。
    """
    if mode not in LAYER_CLASSES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    metrics_cls = LAYER_CLASSES[mode]
    layers: list = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "linear_attention":
            layers.append(LinearAttentionLayer())
        elif layer_type == "full_attention":
            layers.append(metrics_cls(max_cache_len=capacity))
        else:
            raise ValueError(f"unsupported layer_type {layer_type!r} at {layer_idx}")
    return StaticAdapterCache(layers=layers, mode=mode, capacity=capacity)


def preallocate_kv(
    cache: StaticAdapterCache,
    config,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> int:
    """按 capacity 预分配 16 层的 KV，返回字节数。

    必须在第一次 forward 之前调用；之后的写入都是原地操作，地址不变。
    """
    cache.early_initialization(
        batch_size=batch_size,
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=torch.device(device),
    )
    kv_layers = [layer for layer in cache.layers if getattr(layer, "is_initialized", False)]
    for layer in kv_layers:
        if not isinstance(layer, StaticLayer):
            raise RuntimeError(f"unexpected initialized layer {type(layer).__name__}")
    return sum(layer.keys.numel() * layer.keys.element_size() * 2 for layer in kv_layers)


def reset_cache(cache: StaticAdapterCache) -> None:
    """原地 reset：清 KV、GDN/conv state 和累计长度，保留 buffer 地址。"""
    cache.reset()
    torch.cuda.synchronize()


def cache_length(cache: StaticAdapterCache) -> int:
    """有效 KV 长度（会同步 device tensor；只用于非 capture 路径）。"""
    return int(cache.get_seq_length())


@dataclass
class CacheStats:
    kv_bytes: int
    gdn_state_bytes: int
    conv_state_bytes: int
    kv_layers: int
    linear_layers: int
    kv_dtypes: list[str]
    gdn_dtypes: list[str]
    conv_dtypes: list[str]
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kv_bytes": self.kv_bytes,
            "gdn_state_bytes": self.gdn_state_bytes,
            "conv_state_bytes": self.conv_state_bytes,
            "kv_layers": self.kv_layers,
            "linear_layers": self.linear_layers,
            "kv_dtypes": self.kv_dtypes,
            "gdn_dtypes": self.gdn_dtypes,
            "conv_dtypes": self.conv_dtypes,
            "finite": self.finite,
        }


def cache_stats(cache: StaticAdapterCache, *, check_finite: bool = True) -> CacheStats:
    """统计 cache 各部分的实存字节、dtype 和有限性。"""
    kv_bytes = gdn_bytes = conv_bytes = 0
    kv_layers = linear_layers = 0
    kv_dtypes: set[str] = set()
    gdn_dtypes: set[str] = set()
    conv_dtypes: set[str] = set()
    finite = True
    for layer in cache.layers:
        if isinstance(layer, StaticLayer):
            kv_layers += 1
            for tensor in (layer.keys, layer.values):
                kv_bytes += tensor.numel() * tensor.element_size()
                kv_dtypes.add(str(tensor.dtype))
                if check_finite:
                    finite = finite and bool(torch.isfinite(tensor).all())
        else:
            linear_layers += 1
            for state in layer.recurrent_states.values():
                if state is None:
                    continue
                gdn_bytes += state.numel() * state.element_size()
                gdn_dtypes.add(str(state.dtype))
                if check_finite:
                    finite = finite and bool(torch.isfinite(state).all())
            for state in layer.conv_states.values():
                if state is None:
                    continue
                conv_bytes += state.numel() * state.element_size()
                conv_dtypes.add(str(state.dtype))
                if check_finite:
                    finite = finite and bool(torch.isfinite(state).all())
    return CacheStats(
        kv_bytes=kv_bytes,
        gdn_state_bytes=gdn_bytes,
        conv_state_bytes=conv_bytes,
        kv_layers=kv_layers,
        linear_layers=linear_layers,
        kv_dtypes=sorted(kv_dtypes),
        gdn_dtypes=sorted(gdn_dtypes),
        conv_dtypes=sorted(conv_dtypes),
        finite=finite,
    )


def kv_tensors(cache: StaticAdapterCache) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """返回 (layer_idx, keys, values)，只含 full attention 层。"""
    out = []
    for idx, layer in enumerate(cache.layers):
        if isinstance(layer, StaticLayer):
            out.append((idx, layer.keys, layer.values))
    return out


def linear_state_tensors(cache: StaticAdapterCache) -> list[dict[str, torch.Tensor | None]]:
    """返回每层 GDN 的 (conv_state, recurrent_state)，只含 linear attention 层。"""
    out = []
    for layer in cache.layers:
        if isinstance(layer, StaticLayer):
            continue
        out.append(
            {
                "conv": layer.conv_states.get(0),
                "recurrent": layer.recurrent_states.get(0),
            }
        )
    return out
