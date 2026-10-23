"""M3/Phase 2 static cache adapter：预分配 KV / GDN / conv state 两种 decode attention shape。

设计：
- 用 transformers 5.20.2 的 hybrid cache 层（``LinearAttentionLayer`` +
  ``LinearAttentionAndStaticFullAttentionLayer``）作为存储基础，只覆写
  ``update`` / ``get_mask_sizes`` / ``reset``，不改 attention 数学；
- KV 在第一次 forward 之前按 capacity 预分配（``early_initialization``），
  地址固定、原地写入，供 CUDA Graph replay；
- **KV 格式由配置声明**（``PHASE2_KV_FORMAT``，本分支默认 ``fp8``，
  ``bf16`` 复现旧口径）：plan §7 的验收配置是 FP8 KV。FP8 时 K/V 存
  ``float8_e4m3fn``，每层另常驻 ``[capacity, kv_heads]`` 的 fp32 scale（K/V 各一套，
  契约见 ``runner/ops/kv/fp8.py``）；
- 两种模式只差 decode 时 attention 看到的 KV 长度：
  * ``exact``：返回 ``keys[:, :, :L]`` 前缀 view，mask 长度也是 L，
    用于单独观察“存储方式变化”；
  * ``cap``：decode（q_len==1）返回整块 capacity，mask 由 device 上的有效长度
    控制可见范围；prefill（q_len>1）仍用有效前缀，避免 8K prefill 扫满 capacity；
- ``cumulative_length`` 是 device 上的 0-dim tensor，graph 内自行推进；
  ``exact`` 模式需要 ``int(...)`` 会产生同步，因此该模式不用于 capture。

两条消费路径（FP8）：
  * decode（q_len==1）返回 **fp8 前缀 + scale**（scale 挂在返回张量的
    ``kv_fp8_scale`` 上），由 ``compact_decode`` 在 bmm 前反量化；不每步把整段
    前缀物化成 bf16；
  * prefill（q_len>1）返回反量化后的连续 bf16 前缀：HF eager attention 用不上
    fp8 的 scale，而且 8K prefill 的额外流量（约 10 GB、十几毫秒）远小于权重
    读取；prefill 用纯 torch 的 reference 公式，decode 才走 Triton 热路径。

数据流（decode 单步）：
    token -> embed row [1,1,5120]
      -> 48 层 GDN（conv/recurrent state 原地更新）
      -> 16 层 full attention（fp8 K/V + scale 原地写入；decode 读 L 或 capacity）
      -> lm_head -> logits [1,1,248320]

预期瓶颈：``exact`` 的 KV 展开流量随 L 线性增长；``cap`` 在 capacity 远大于 L 时
把这份流量放大到 capacity。FP8 把 K/V 读流量减半，但 ``compact_decode`` 还要把
前缀反量化成 bf16 才能走 tensor core bmm，净收益取决于 bmm 是否命中 L2（见
``runner/ops/kv/fp8.py`` 的字节账）。
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
from transformers.utils import is_torchdynamo_compiling

from runner.ops.kv import fp8 as fp8_kv

MODES = ("exact", "cap")


def _layer_format(layer) -> str:
    return getattr(layer, "kv_format", "bf16")


class _FormatAwareStaticLayer(LinearAttentionAndStaticFullAttentionLayer):
    """KV 静态存储；按 ``kv_format`` 分配 bf16 或 fp8 + scale。"""

    # 覆写后可以在 prefill 之前就分配 KV；lazy_initialization 的 dispatch 仍由
    # LinearAttentionAndStaticFullAttentionLayer 提供（2 个位置参数 -> StaticLayer）。
    supports_early_init = True
    kv_format = "bf16"

    # ------------------------------------------------------------------ init
    def lazy_initialization(self, *args, **kwargs):
        # 2 个位置参数 = KV 初始化（Cache.early_initialization 或首次 update）；
        # 其余签名 = GDN 的 conv/recurrent state，交给 LinearAttentionLayer。
        if len(args) == 2 and len(kwargs) == 0:
            if self.kv_format == "fp8":
                self._lazy_init_fp8(*args)
            else:
                StaticLayer.lazy_initialization(self, *args)
            return
        LinearAttentionLayer.lazy_initialization(self, *args, **kwargs)

    def _lazy_init_fp8(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = fp8_kv.FP8_DTYPE
        self.device = key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]
        shape_k = (self.batch_size, self.num_heads, self.max_cache_len, self.k_head_dim)
        shape_v = (self.batch_size, self.num_heads, self.max_cache_len, self.v_head_dim)
        self.keys = torch.zeros(shape_k, dtype=fp8_kv.FP8_DTYPE, device=self.device)
        self.values = torch.zeros(shape_v, dtype=fp8_kv.FP8_DTYPE, device=self.device)
        # fp8 在部分 elementwise/index kernel 上没有实现（CPU 上 index_copy_/zero_
        # 就缺 fp8 kernel），写入/清零一律走 uint8 view；两者共享同一块 storage。
        self._keys_u8 = self.keys.view(torch.uint8)
        self._values_u8 = self.values.view(torch.uint8)
        scale_shape = (self.max_cache_len, self.num_heads)
        self.k_scale = torch.zeros(scale_shape, dtype=fp8_kv.SCALE_DTYPE, device=self.device)
        self.v_scale = torch.zeros(scale_shape, dtype=fp8_kv.SCALE_DTYPE, device=self.device)
        self.cumulative_length = self.cumulative_length.to(self.device)
        if not is_torchdynamo_compiling():
            for tensor in (
                self.keys,
                self.values,
                self.k_scale,
                self.v_scale,
                self.cumulative_length,
            ):
                torch._dynamo.mark_static_address(tensor)
        self.is_initialized = True

    # ----------------------------------------------------------------- write
    def _write_fp8(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """量化新 token 并原地写进 KV 槽（graph 内可重放，位置由 device 推进）。"""
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        kv_len = key_states.shape[-2]
        cache_position = torch.arange(kv_len, device=self.device) + self.cumulative_length
        self.cumulative_length.add_(kv_len)
        qk, sk = fp8_kv.quantize_kv(key_states)
        qv, sv = fp8_kv.quantize_kv(value_states)
        self._keys_u8.index_copy_(2, cache_position, qk.view(torch.uint8))
        self._values_u8.index_copy_(2, cache_position, qv.view(torch.uint8))
        self.k_scale.index_copy_(0, cache_position, sk)
        self.v_scale.index_copy_(0, cache_position, sv)

    # ------------------------------------------------------------------ read
    def _fp8_decode(
        self, keys: torch.Tensor, values: torch.Tensor, length: int | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """decode：返回 fp8 前缀 + ``[L,H]`` scale，不在这一步物化 bf16。"""
        if length is None:
            k_scale, v_scale = self.k_scale, self.v_scale
        else:
            k_scale, v_scale = self.k_scale[:length], self.v_scale[:length]
        return (
            fp8_kv.attach_scale(keys, k_scale),
            fp8_kv.attach_scale(values, v_scale),
        )

    def _fp8_prefill(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """prefill：返回连续 bf16 前缀（不同执行方式看到同样的 bmm 输入）。

        prefill 走纯 torch 的 reference 反量化而不是 Triton 热路径：它不是 decode
        热点（8K prefill 的额外流量约 10 GB、十几毫秒），而 reference cache 用的
        也是同一份 torch 公式，这样 numeric-512 的 eager→static-exact bitwise
        不依赖 Triton kernel 的正确性（Triton 由 op gate 单独验证）。
        """
        keys = fp8_kv.reference_dequantize(self.keys[:, :, :length], self.k_scale[:length])
        values = fp8_kv.reference_dequantize(
            self.values[:, :, :length], self.v_scale[:length]
        )
        return keys, values

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        if self.kv_format == "fp8" and self.is_initialized:
            LinearAttentionLayer.reset(self)
            self._keys_u8.zero_()
            self._values_u8.zero_()
            self.k_scale.zero_()
            self.v_scale.zero_()
            self.cumulative_length.zero_()
            return
        super().reset()


class StaticExactFullAttentionLayer(_FormatAwareStaticLayer):
    """KV 静态存储 + decode attention 只读有效前缀。"""

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kv_format == "fp8":
            self._write_fp8(key_states, value_states)
            length = int(self.cumulative_length)
            if key_states.shape[-2] == 1:
                return self._fp8_decode(
                    self.keys[:, :, :length], self.values[:, :, :length], length
                )
            return self._fp8_prefill(length)
        keys, values = StaticLayer.update(self, key_states, value_states)
        length = int(self.cumulative_length)
        return keys[:, :, :length], values[:, :, :length]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # mask 的 kv 长度必须和 update 返回的前缀一致：本步写入后总长度为
        # (本步之前已缓存数) + query_length。
        return int(self.cumulative_length) + query_length, 0


class StaticCapFullAttentionLayer(_FormatAwareStaticLayer):
    """KV 静态存储 + decode attention 扫满 capacity，prefill 仍读有效前缀。"""

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kv_format == "fp8":
            self._write_fp8(key_states, value_states)
            if key_states.shape[-2] == 1:
                # decode：固定 shape，mask 由 device 上的 cumulative_length 控制可见范围。
                return self._fp8_decode(self.keys, self.values, None)
            return self._fp8_prefill(int(self.cumulative_length))
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
        kv_format: 声明的 KV 格式（``bf16`` / ``fp8``）。
    """

    def __init__(self, layers: list, mode: str, capacity: int, kv_format: str = "bf16"):
        super().__init__(layers=layers)
        if mode not in MODES:
            raise ValueError(f"unknown static cache mode {mode!r}")
        self.mode = mode
        self.capacity = int(capacity)
        self.kv_format = fp8_kv.kfmt(kv_format)

    def __repr__(self) -> str:
        return (
            f"StaticAdapterCache(mode={self.mode!r}, capacity={self.capacity}, "
            f"kv_format={self.kv_format!r})"
        )


def build_static_cache(
    config,
    capacity: int,
    mode: str = "exact",
    kv_format: str | None = None,
) -> StaticAdapterCache:
    """按模型 config 的 layer_types 构造 hybrid static cache。

    只支持本项目模型的两种层：``linear_attention``（GDN，48 层）与
    ``full_attention``（KV，16 层）。``kv_format=None`` 时取
    ``PHASE2_KV_FORMAT``（本分支默认 fp8）。
    """
    if mode not in LAYER_CLASSES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    fmt = fp8_kv.resolve_format(kv_format)
    metrics_cls = LAYER_CLASSES[mode]
    layers: list = []
    for layer_idx, layer_type in enumerate(config.layer_types):
        if layer_type == "linear_attention":
            layers.append(LinearAttentionLayer())
        elif layer_type == "full_attention":
            layer = metrics_cls(max_cache_len=capacity)
            layer.kv_format = fmt
            layers.append(layer)
        else:
            raise ValueError(f"unsupported layer_type {layer_type!r} at {layer_idx}")
    return StaticAdapterCache(
        layers=layers, mode=mode, capacity=capacity, kv_format=fmt
    )


def preallocate_kv(
    cache: StaticAdapterCache,
    config,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> int:
    """按 capacity 预分配 16 层的 KV，返回总字节数（FP8 含 scale）。

    必须在第一次 forward 之前调用；之后的写入都是原地操作，地址不变。
    ``dtype`` 只对 BF16 模式生效；FP8 模式的存储 dtype 由契约固定。
    """
    cache.early_initialization(
        batch_size=batch_size,
        num_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
        device=torch.device(device),
    )
    kv_layers = [layer for layer in cache.layers if getattr(layer, "is_initialized", False)]
    total = 0
    for layer in kv_layers:
        if not isinstance(layer, StaticLayer):
            raise RuntimeError(f"unexpected initialized layer {type(layer).__name__}")
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
        if _layer_format(layer) == "fp8":
            total += layer.k_scale.numel() * layer.k_scale.element_size() * 2
    return total


def reset_cache(cache: StaticAdapterCache) -> None:
    """原地 reset：清 KV、GDN/conv state 和累计长度，保留 buffer 地址。"""
    cache.reset()
    torch.cuda.synchronize()


def cache_length(cache: StaticAdapterCache) -> int:
    """有效 KV 长度（会同步 device tensor；只用于非 capture 路径）。"""
    return int(cache.get_seq_length())


def _finite(tensor: torch.Tensor) -> bool:
    """fp8 上 ``torch.isfinite`` 没有实现（CPU 实测 NotImplementedError）。"""
    if tensor.dtype == fp8_kv.FP8_DTYPE:
        tensor = tensor.float()
    return bool(torch.isfinite(tensor).all())


@dataclass
class CacheStats:
    kv_bytes: int
    kv_scale_bytes: int
    gdn_state_bytes: int
    conv_state_bytes: int
    kv_layers: int
    linear_layers: int
    kv_format: str
    kv_dtypes: list[str]
    kv_scale_dtypes: list[str]
    gdn_dtypes: list[str]
    conv_dtypes: list[str]
    finite: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kv_bytes": self.kv_bytes,
            "kv_scale_bytes": self.kv_scale_bytes,
            "gdn_state_bytes": self.gdn_state_bytes,
            "conv_state_bytes": self.conv_state_bytes,
            "kv_layers": self.kv_layers,
            "linear_layers": self.linear_layers,
            "kv_format": self.kv_format,
            "kv_dtypes": self.kv_dtypes,
            "kv_scale_dtypes": self.kv_scale_dtypes,
            "gdn_dtypes": self.gdn_dtypes,
            "conv_dtypes": self.conv_dtypes,
            "finite": self.finite,
        }


def cache_stats(cache: StaticAdapterCache, *, check_finite: bool = True) -> CacheStats:
    """统计 cache 各部分的实存字节、dtype、KV 格式和有限性。

    ``kv_bytes`` 是 K/V 加 scale 的总和（plan §1：FP8 的 scale 也计入每步字节账）；
    ``kv_scale_bytes`` 单列，方便把“反量化读取的 scale”和“KV 本身”分开看。
    """
    kv_bytes = kv_scale_bytes = gdn_bytes = conv_bytes = 0
    kv_layers = linear_layers = 0
    kv_dtypes: set[str] = set()
    kv_scale_dtypes: set[str] = set()
    kv_formats: set[str] = set()
    gdn_dtypes: set[str] = set()
    conv_dtypes: set[str] = set()
    finite = True
    for layer in cache.layers:
        if isinstance(layer, StaticLayer):
            kv_layers += 1
            kv_formats.add(_layer_format(layer))
            for tensor in (layer.keys, layer.values):
                kv_bytes += tensor.numel() * tensor.element_size()
                kv_dtypes.add(str(tensor.dtype))
                if check_finite:
                    finite = finite and _finite(tensor)
            if _layer_format(layer) == "fp8":
                for tensor in (layer.k_scale, layer.v_scale):
                    count = tensor.numel() * tensor.element_size()
                    kv_bytes += count
                    kv_scale_bytes += count
                    kv_scale_dtypes.add(str(tensor.dtype))
                    if check_finite:
                        finite = finite and _finite(tensor)
        else:
            linear_layers += 1
            for state in layer.recurrent_states.values():
                if state is None:
                    continue
                gdn_bytes += state.numel() * state.element_size()
                gdn_dtypes.add(str(state.dtype))
                if check_finite:
                    finite = finite and _finite(state)
            for state in layer.conv_states.values():
                if state is None:
                    continue
                conv_bytes += state.numel() * state.element_size()
                conv_dtypes.add(str(state.dtype))
                if check_finite:
                    finite = finite and _finite(state)
    kv_format = "fp8" if "fp8" in kv_formats else "bf16"
    return CacheStats(
        kv_bytes=kv_bytes,
        kv_scale_bytes=kv_scale_bytes,
        gdn_state_bytes=gdn_bytes,
        conv_state_bytes=conv_bytes,
        kv_layers=kv_layers,
        linear_layers=linear_layers,
        kv_format=kv_format,
        kv_dtypes=sorted(kv_dtypes),
        kv_scale_dtypes=sorted(kv_scale_dtypes),
        gdn_dtypes=sorted(gdn_dtypes),
        conv_dtypes=sorted(conv_dtypes),
        finite=finite,
    )


def kv_tensors(cache: StaticAdapterCache) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """返回 (layer_idx, keys, values)，只含 full attention 层。

    FP8 格式下 ``keys`` / ``values`` 是 ``float8_e4m3fn``；对应的 ``[capacity, H]``
    scale 用 ``kv_scale_tensors`` 取。
    """
    out = []
    for idx, layer in enumerate(cache.layers):
        if isinstance(layer, StaticLayer):
            out.append((idx, layer.keys, layer.values))
    return out


def kv_scale_tensors(
    cache: StaticAdapterCache,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """返回 (layer_idx, k_scale, v_scale)，只含 full attention 层；BF16 时为空。"""
    out = []
    for idx, layer in enumerate(cache.layers):
        if isinstance(layer, StaticLayer) and _layer_format(layer) == "fp8":
            out.append((idx, layer.k_scale, layer.v_scale))
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
