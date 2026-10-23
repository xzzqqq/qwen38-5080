"""M3 对拍工具：跑固定 token 序列，抓 hidden / KV / GDN state / logits 并比较。

设计：
- 用 forward hook 记录 64 层 decoder 的逐步输出，hook 只在验证脚本里挂，
  不进入默认推理路径；
- KV 只比较有效前缀 ``[:, :, :L]``，static 与 dynamic 用同一口径；
- 快照一律拷到 CPU，避免两份 8K cache + graph pool 同时占显存；
- 比较先看 bitwise，同时记录 max_abs 和首处失败位置（step / layer）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

KV_ATTR = "keys"


@dataclass
class Snapshot:
    label: str
    logits: list[torch.Tensor]
    hidden: list[list[torch.Tensor]]
    kv_keys: list[torch.Tensor]
    kv_values: list[torch.Tensor]
    conv: list[torch.Tensor]
    recurrent: list[torch.Tensor]
    length: int
    prompt_tokens: int
    decode_steps: int
    forwards: int
    finite: bool
    kv_layer_indices: list[int] = field(default_factory=list)
    linear_layer_indices: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "length": self.length,
            "prompt_tokens": self.prompt_tokens,
            "decode_steps": self.decode_steps,
            "forwards": self.forwards,
            "kv_layers": len(self.kv_keys),
            "linear_layers": len(self.recurrent),
            "finite": self.finite,
            "logits_finite": all(bool(torch.isfinite(t).all()) for t in self.logits),
            "kv_shapes": [tuple(t.shape) for t in self.kv_keys],
        }


@dataclass
class Diff:
    name: str
    count: int = 0
    bad_count: int = 0
    max_abs: float = 0.0
    first_bad: list[int] = field(default_factory=list)
    # hidden 专用：按“最早发生的 forward”排序，避免 layer-major 扁平索引误导。
    first_bad_forward: int | None = None
    first_bad_forward_layers: list[int] = field(default_factory=list)
    shapes_equal: bool = True

    @property
    def bitwise(self) -> bool:
        return self.count > 0 and self.bad_count == 0 and self.shapes_equal

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "bad_count": self.bad_count,
            "max_abs": self.max_abs,
            "first_bad": self.first_bad,
            "first_bad_forward": self.first_bad_forward,
            "first_bad_forward_layers": self.first_bad_forward_layers,
            "shapes_equal": self.shapes_equal,
            "bitwise": self.bitwise,
        }


def _finite(tensor: torch.Tensor) -> bool:
    """fp8 上 ``torch.isfinite`` 没有实现；先 float() 再查。"""
    if tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.float()
    return bool(torch.isfinite(tensor).all())


def snapshot_state(cache):
    """把 cache 的 KV 有效前缀、conv/recurrent state 拷到 CPU。

    返回 (kv_keys, kv_values, conv, recurrent, length, finite)。
    KV 一律只取 ``[:, :, :length]``，让 static 的 capacity buffer 和 dynamic 的
    变长 buffer 用同一口径比较。FP8 KV 在 CPU 快照里是 bf16（fp8 code 直接 cast，
    不是反量化值）——对拍的是“存储里的 code 是否一致”；反量化后的数值一致性由
    ``padded-ref->static-cap`` 的 logits/hidden 和 op gate 负责。
    """
    length = int(cache.get_seq_length())
    kv_keys, kv_values, conv, recurrent = [], [], [], []
    finite = True
    for layer_idx, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        if keys is not None:
            values = getattr(layer, "values", None)
            kv_keys.append(keys[:, :, :length].detach().to("cpu", torch.bfloat16))
            kv_values.append(values[:, :, :length].detach().to("cpu", torch.bfloat16))
            finite = finite and _finite(keys)
            continue
        conv_states = getattr(layer, "conv_states", None)
        rec_states = getattr(layer, "recurrent_states", None)
        if conv_states and conv_states.get(0) is not None:
            conv.append(conv_states[0].detach().to("cpu", torch.bfloat16))
            finite = finite and _finite(conv_states[0])
        if rec_states and rec_states.get(0) is not None:
            recurrent.append(rec_states[0].detach().to("cpu", torch.float32))
            finite = finite and _finite(rec_states[0])
    return kv_keys, kv_values, conv, recurrent, length, finite


def run_sequence(
    model,
    cache,
    prompt_ids: torch.Tensor,
    trace_ids: list[int],
    *,
    label: str,
    chunk: int = 128,
    record_hidden: bool = True,
) -> Snapshot:
    """prefill prompt + teacher-forced decode trace，返回 CPU 快照。

    Args:
        prompt_ids: ``[1, S]`` CPU long。
        trace_ids: 要喂入 decode 的 token（长度 = decode step 数）。最后一个
            decode step 不再前进，因此 cache 长度为 ``S + len(trace_ids)``。
    """
    layers = list(model.model.layers)
    logits_list: list[torch.Tensor] = []
    hidden: list[list[torch.Tensor]] = [[] for _ in layers]

    def make_hook(idx):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            hidden[idx].append(tensor[:, -1:, :].detach().to("cpu", torch.bfloat16))

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)] if record_hidden else []
    finite = True
    try:
        out = None
        for start in range(0, prompt_ids.shape[1], chunk):
            out = model(
                input_ids=prompt_ids[:, start : start + chunk],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        if out is None:
            raise RuntimeError("empty prompt")
        logits = out.logits[:, -1, :]
        logits_list.append(logits.detach().to("cpu", torch.bfloat16))
        finite = finite and bool(torch.isfinite(logits).all())
        for token in trace_ids:
            nid = torch.tensor([[int(token)]], dtype=torch.long)
            out = model(
                input_ids=nid,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :]
            logits_list.append(logits.detach().to("cpu", torch.bfloat16))
            finite = finite and bool(torch.isfinite(logits).all())
    finally:
        for handle in handles:
            handle.remove()

    kv_keys, kv_values, conv, recurrent, length, state_finite = snapshot_state(cache)
    finite = finite and state_finite
    kv_layer_indices = [
        idx for idx, layer in enumerate(cache.layers) if getattr(layer, "keys", None) is not None
    ]
    linear_layer_indices = [
        idx
        for idx, layer in enumerate(cache.layers)
        if getattr(layer, "keys", None) is None
        and getattr(layer, "recurrent_states", None)
        and layer.recurrent_states.get(0) is not None
    ]

    return Snapshot(
        label=label,
        logits=logits_list,
        hidden=hidden,
        kv_keys=kv_keys,
        kv_values=kv_values,
        conv=conv,
        recurrent=recurrent,
        length=length,
        prompt_tokens=int(prompt_ids.shape[1]),
        decode_steps=len(trace_ids),
        forwards=len(logits_list),
        finite=finite,
        kv_layer_indices=kv_layer_indices,
        linear_layer_indices=linear_layer_indices,
    )


def _diff_lists(
    name: str, a: list[torch.Tensor], b: list[torch.Tensor], per_layer: int | None = None
) -> Diff:
    diff = Diff(name=name, count=0)
    n = min(len(a), len(b))
    if len(a) != len(b):
        diff.shapes_equal = False
    for i in range(n):
        x, y = a[i], b[i]
        if x.shape != y.shape:
            diff.shapes_equal = False
            diff.bad_count += 1
            diff.first_bad.append(i)
            diff.count += 1
            continue
        diff.count += 1
        if not torch.equal(x, y):
            diff.bad_count += 1
            if len(diff.first_bad) < 8:
                diff.first_bad.append(i)
            if per_layer:
                forward = i % per_layer
                if diff.first_bad_forward is None or forward < diff.first_bad_forward:
                    diff.first_bad_forward = forward
                    diff.first_bad_forward_layers = [i // per_layer]
                elif forward == diff.first_bad_forward:
                    diff.first_bad_forward_layers.append(i // per_layer)
            mx = float((x.float() - y.float()).abs().max().item())
            diff.max_abs = max(diff.max_abs, mx)
    return diff


def compare_snapshots(ref: Snapshot, got: Snapshot) -> list[Diff]:
    """比较两份快照；返回各项 Diff（bitwise 为 gate）。"""
    diffs: list[Diff] = []
    diffs.append(_diff_lists("logits", ref.logits, got.logits))
    if ref.hidden and any(ref.hidden):
        flat_ref = [t for layer in ref.hidden for t in layer]
        flat_got = [t for layer in got.hidden for t in layer]
        per_layer = len(ref.hidden[0]) if ref.hidden and ref.hidden[0] else None
        diffs.append(_diff_lists("hidden", flat_ref, flat_got, per_layer=per_layer))
    if ref.kv_keys:
        diffs.append(_diff_lists("kv.keys", ref.kv_keys, got.kv_keys))
        diffs.append(_diff_lists("kv.values", ref.kv_values, got.kv_values))
    if ref.conv:
        diffs.append(_diff_lists("conv", ref.conv, got.conv))
    if ref.recurrent:
        diffs.append(_diff_lists("recurrent", ref.recurrent, got.recurrent))
    return diffs
