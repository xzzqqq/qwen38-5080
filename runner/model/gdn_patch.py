"""GDN decode 融合 kernel 的模型接入（只 patch ``sequence_length == 1`` 的两个分支）。

两条接入缝（都是模块级函数，``forward`` 在调用点做全局名字查找）：

1. ``torch_recurrent_gated_delta_rule`` → recurrent step（第 1 轮，``runner/ops/gdn/decode.py``）。
2. ``causal_conv1d_update`` → conv step（第 2 轮，``runner/ops/gdn/conv_decode.py``）。
   它只在 ``use_precomputed_states and seq_len == 1 and not record_past`` 分支被调用，
   prefill 走 ``causal_conv1d_fn``（另一个函数），所以 conv 融合同样不影响 prefill。

设计：
- 接入点是模块级函数 ``transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule``。
  ``Qwen3_5GatedDeltaNet.forward`` 在调用点做全局名字查找，本机没有装 ``kernels``，
  ``use_kernelized_func`` 是 no-op（不会在 import 时把函数捕获成类属性），所以替换
  模块属性就能覆盖 eager / static-exact / static-cap / padded-reference / graph
  四种执行方式，不需要复制模型源码，也不会误伤 prefill（prefill 走
  ``torch_chunk_gated_delta_rule``，是另一个函数）。
- conv 的 patch 用同一套判据：单 token、``mixed_qkv`` 是 bf16 ``[1, 10240, 1]``
  （channel 维 stride=1）、``conv_state`` 是 bf16 ``[1, 10240, 4]`` contiguous、
  weight 是 contiguous bf16 ``[10240, 4]``、bias 为 None、activation 为 silu；不满足
  就回原实现并计数（``conv.counters.fallbacks``）。conv_state 由 kernel **原地**
  滚动，dtype/字节/data_ptr 不变（static cache 与 graph replay 依赖）。
- 命中条件（任一不满足就走回原实现，并把原因计数；见 ``counters.fallbacks``）：
  state 是 CUDA FP32 ``[1, 48, 128, 128]`` contiguous、单 token、qk l2norm 打开、
  无 ``cu_seqlens``、q/k 的 head 数 × ratio 等于 v head 数。
- q/k 用 ``[:, :, ::ratio, :]`` 的 strided view 传进 kernel（kernel 按 head stride
  寻址），所以**没有额外拷贝**；模型自己的 ``repeat_interleave`` 仍然执行（96 个
  tiny kernel），本轮没有为省它去复制 ``forward``。
- state 原地更新后返回 ``(out, initial_state)``；随后模型的
  ``update_recurrent_state`` 会对同一个 buffer 做 ``copy_``，patch 把这种自拷贝
  短路成 no-op（``is`` 同一对象才触发，语义恒等；不依赖 torch 是否自带 early-out）。
- out 槽按 ``state.data_ptr()`` 懒分配：48 层 × ``[1,1,48,128]`` bf16 ≈ 576 KB；
  首次 decode 发生在 warmup（capture 之前），之后地址固定，graph replay 依赖这一点。

预期瓶颈：与 ``runner/ops/gdn/decode.py`` 相同（state 带宽）；本模块只做接线，
每层每步多一次 Python 侧的 shape 检查（graph 路径在 capture 后不再执行）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from runner.ops.gdn import (
    CONV_DEFAULT_BLOCK,
    CONV_DEFAULT_NUM_WARPS,
    DEFAULT_BLOCK_V,
    DEFAULT_NUM_WARPS,
    gdn_conv_decode_step,
    gdn_decode_step,
)

LOGGER = logging.getLogger(__name__)

#: 关掉融合路径（调试/对照用）：``GDN_FUSED_DECODE=0``。
ENV_FLAG = "GDN_FUSED_DECODE"
_DISABLED = {"0", "false", "off", "no"}

_STATE: dict[str, Any] = {
    "installed": False,
    "ratio": 1,
    "layers": 0,
    "num_v_heads": 0,
    "num_k_heads": 0,
    "head_k_dim": 0,
    "head_v_dim": 0,
    "block_v": DEFAULT_BLOCK_V,
    "num_warps": DEFAULT_NUM_WARPS,
    "out": {},
    "out_bytes": 0,
    # 活计数：install() 返回的 info 里带的是同一个 dict，产物在末尾序列化时能看到最终值。
    "counters": {"fused": 0, "reference": 0, "fallbacks": {}},
    "orig_recurrent": None,
    "orig_update": None,
    "warned": set(),
}

# conv step 的 patch 状态（与 recurrent 分开装，互不依赖；同一个 env flag 关掉两者）。
_CONV: dict[str, Any] = {
    "installed": False,
    "layers": 0,
    "conv_dim": 0,
    "kernel_size": 0,
    "block": CONV_DEFAULT_BLOCK,
    "num_warps": CONV_DEFAULT_NUM_WARPS,
    "bias": False,
    "activation": None,
    # 活计数：与 _STATE["counters"] 一样，info 里带的是同一个 dict 对象。
    "counters": {"fused": 0, "reference": 0, "fallbacks": {}},
    "orig": None,
}


# ------------------------------------------------------------------------- 接入


def install(
    model: torch.nn.Module,
    *,
    block_v: int = DEFAULT_BLOCK_V,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> dict[str, Any]:
    """把模型里的 GDN decode 分支接到融合 kernel 上；重复安装是幂等的。"""
    import transformers.cache_utils as cache_utils
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    klass = getattr(modeling, "Qwen3_5GatedDeltaNet", None)
    if klass is None:
        raise RuntimeError("transformers 里找不到 Qwen3_5GatedDeltaNet")
    layers = [module for module in model.modules() if isinstance(module, klass)]
    if not layers:
        raise RuntimeError("模型里没有 Qwen3_5GatedDeltaNet 层")
    ratios = {int(layer.num_v_heads) // int(layer.num_k_heads) for layer in layers}
    if len(ratios) != 1:
        raise RuntimeError(f"GDN 层的 v/k head 比不一致: {sorted(ratios)}")

    if _STATE["installed"]:
        uninstall()

    _STATE["orig_recurrent"] = modeling.torch_recurrent_gated_delta_rule
    _STATE["orig_update"] = cache_utils.LinearAttentionLayer.update_recurrent_state
    _STATE["ratio"] = ratios.pop()
    _STATE["layers"] = len(layers)
    _STATE["num_v_heads"] = int(layers[0].num_v_heads)
    _STATE["num_k_heads"] = int(layers[0].num_k_heads)
    _STATE["head_k_dim"] = int(layers[0].head_k_dim)
    _STATE["head_v_dim"] = int(layers[0].head_v_dim)
    _STATE["block_v"] = int(block_v)
    _STATE["num_warps"] = int(num_warps)
    _STATE["out"] = {}
    _STATE["out_bytes"] = 0
    _STATE["counters"] = {"fused": 0, "reference": 0, "fallbacks": {}}
    _STATE["warned"] = set()

    modeling.torch_recurrent_gated_delta_rule = _fused_recurrent
    cache_utils.LinearAttentionLayer.update_recurrent_state = _update_recurrent_state
    _STATE["installed"] = True
    return info()


def install_conv(
    model: torch.nn.Module,
    *,
    block: int = CONV_DEFAULT_BLOCK,
    num_warps: int = CONV_DEFAULT_NUM_WARPS,
) -> dict[str, Any]:
    """把 GDN 的 conv decode step 接到融合 kernel 上；重复安装是幂等的。

    与 ``install`` 分开：两者 patch 的是不同的模块级函数，任何一条失败都不应该
    让另一条失效。``install_from_env`` 会把两条都装上。
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    klass = getattr(modeling, "Qwen3_5GatedDeltaNet", None)
    if klass is None:
        raise RuntimeError("transformers 里找不到 Qwen3_5GatedDeltaNet")
    layers = [module for module in model.modules() if isinstance(module, klass)]
    if not layers:
        raise RuntimeError("模型里没有 Qwen3_5GatedDeltaNet 层")
    dims = {(int(layer.conv_dim), int(layer.conv_kernel_size)) for layer in layers}
    if len(dims) != 1:
        raise RuntimeError(f"GDN 层的 conv shape 不一致: {sorted(dims)}")
    conv_dim, kernel_size = dims.pop()
    for layer in layers:
        weight = layer.conv1d.weight
        if tuple(weight.shape) != (conv_dim, 1, kernel_size):
            raise RuntimeError(
                f"conv1d.weight 形状意外: {tuple(weight.shape)}，"
                f"期望 {(conv_dim, 1, kernel_size)}"
            )
    if _CONV["installed"]:
        _uninstall_conv()

    _CONV["orig"] = modeling.causal_conv1d_update
    _CONV["layers"] = len(layers)
    _CONV["conv_dim"] = conv_dim
    _CONV["kernel_size"] = kernel_size
    _CONV["block"] = int(block)
    _CONV["num_warps"] = int(num_warps)
    _CONV["bias"] = any(layer.conv1d.bias is not None for layer in layers)
    _CONV["activation"] = str(getattr(layers[0], "activation", "silu"))
    _CONV["counters"] = {"fused": 0, "reference": 0, "fallbacks": {}}

    modeling.causal_conv1d_update = _fused_conv
    _CONV["installed"] = True
    return conv_info()


def install_from_env(model: torch.nn.Module) -> dict[str, Any]:
    """loader 用：默认打开，``GDN_FUSED_DECODE=0`` 关闭；失败只记录不抛。"""
    flag = os.environ.get(ENV_FLAG, "1").strip().lower()
    if flag in _DISABLED:
        return {
            "enabled": False,
            "reason": f"{ENV_FLAG}={flag}",
            "conv": {"enabled": False, "reason": f"{ENV_FLAG}={flag}"},
        }
    try:
        result = install(model)
        result["conv"] = install_conv(model)
    except Exception as exc:  # noqa: BLE001 - 接入失败时回退原路径，不让非 decode 工具崩
        LOGGER.error("GDN decode 融合接入失败（回退原路径）：%r", exc)
        try:
            uninstall()
        except Exception:  # noqa: BLE001 - 清理失败不掩盖原始错误
            pass
        return {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}
    LOGGER.warning("GDN decode 融合已接入: %s", json.dumps(result, ensure_ascii=False))
    return result


def uninstall() -> None:
    import transformers.cache_utils as cache_utils
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    if _STATE["installed"]:
        if _STATE["orig_recurrent"] is not None:
            modeling.torch_recurrent_gated_delta_rule = _STATE["orig_recurrent"]
        if _STATE["orig_update"] is not None:
            cache_utils.LinearAttentionLayer.update_recurrent_state = _STATE["orig_update"]
        _STATE["installed"] = False
        _STATE["out"] = {}
        _STATE["out_bytes"] = 0
    _uninstall_conv()


def _uninstall_conv() -> None:
    if not _CONV["installed"]:
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    if _CONV["orig"] is not None:
        modeling.causal_conv1d_update = _CONV["orig"]
    _CONV["installed"] = False


def info() -> dict[str, Any]:
    """接入信息 + 活计数（JSON 友好；``counters`` 是同一个 dict 对象）。"""
    return {
        "enabled": bool(_STATE["installed"]),
        "backend": "triton-fused-decode",
        "ratio": _STATE["ratio"],
        "layers": _STATE["layers"],
        "num_v_heads": _STATE["num_v_heads"],
        "num_k_heads": _STATE["num_k_heads"],
        "head_k_dim": _STATE["head_k_dim"],
        "head_v_dim": _STATE["head_v_dim"],
        "block_v": _STATE["block_v"],
        "num_warps": _STATE["num_warps"],
        "out_slots": len(_STATE["out"]),
        "out_bytes": _STATE["out_bytes"],
        "counters": _STATE["counters"],
    }


def counters() -> dict[str, Any]:
    return _STATE["counters"]


def conv_info() -> dict[str, Any]:
    """conv patch 的接入信息 + 活计数（``counters`` 是同一个 dict 对象）。"""
    return {
        "enabled": bool(_CONV["installed"]),
        "backend": "triton-fused-conv-decode",
        "layers": _CONV["layers"],
        "conv_dim": _CONV["conv_dim"],
        "kernel_size": _CONV["kernel_size"],
        "state_len": _CONV["kernel_size"],
        "block": _CONV["block"],
        "num_warps": _CONV["num_warps"],
        "bias": _CONV["bias"],
        "activation": _CONV["activation"],
        "kernels_per_layer": 1,
        "state_layout": "in-place bf16 [1,10240,4], data_ptr stable",
        "counters": _CONV["counters"],
    }


def conv_counters() -> dict[str, Any]:
    return _CONV["counters"]


# ------------------------------------------------------------------------- patch


def _fused_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs: Any,
):
    """``torch_recurrent_gated_delta_rule`` 的替身：decode 走融合，其余回原实现。"""
    counters = _STATE["counters"]
    reason = _fallback_reason(
        query, key, value, g, beta, initial_state, use_qk_l2norm_in_kernel, kwargs
    )
    if reason is None:
        try:
            out = _decode(query, key, value, g, beta, initial_state)
        except Exception as exc:  # noqa: BLE001 - 宁可回退也不出静默错数
            key_name = f"kernel-error:{type(exc).__name__}"
            counters["fallbacks"][key_name] = counters["fallbacks"].get(key_name, 0) + 1
            _warn_once(key_name, f"GDN 融合 kernel 抛错，回退 reference：{exc!r}")
        else:
            counters["fused"] += 1
            return (out, initial_state) if output_final_state else (out, None)
    else:
        counters["fallbacks"][reason] = counters["fallbacks"].get(reason, 0) + 1
        _warn_once(reason, f"GDN decode 融合未命中（{reason}），回退 reference")

    counters["reference"] += 1
    return _STATE["orig_recurrent"](
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        **kwargs,
    )


def _fused_conv(hidden_states, conv_state, weight, bias=None, activation=None):
    """``causal_conv1d_update`` 的替身：decode 的 conv step 走融合，其余回原实现。

    语义与 reference 完全一致（conv_state 原地滚动、窗口取最后 ``kernel`` 个、先
    舍入到 bf16 再做 silu），差别只是 kernel 数 4 → 1。
    """
    counters = _CONV["counters"]
    reason = _conv_fallback_reason(hidden_states, conv_state, weight, bias, activation)
    if reason is None:
        try:
            out = gdn_conv_decode_step(
                hidden_states,
                conv_state,
                weight,
                activation=activation,
                block=_CONV["block"],
                num_warps=_CONV["num_warps"],
            )
        except Exception as exc:  # noqa: BLE001 - 宁可回退也不出静默错数
            key_name = f"kernel-error:{type(exc).__name__}"
            counters["fallbacks"][key_name] = counters["fallbacks"].get(key_name, 0) + 1
            _warn_once(f"conv:{key_name}", f"GDN conv 融合 kernel 抛错，回退 reference：{exc!r}")
        else:
            counters["fused"] += 1
            return out
    else:
        counters["fallbacks"][reason] = counters["fallbacks"].get(reason, 0) + 1
        _warn_once(f"conv:{reason}", f"GDN conv 融合未命中（{reason}），回退 reference")

    counters["reference"] += 1
    return _CONV["orig"](hidden_states, conv_state, weight, bias, activation)


def _update_recurrent_state(self, recurrent_states, state_idx: int = 0, **kwargs: Any):
    """``LinearAttentionLayer.update_recurrent_state`` 的替身：短路自拷贝。"""
    states = self.recurrent_states
    if (
        recurrent_states is not None
        and 0 <= state_idx < len(states)
        and states[state_idx] is recurrent_states
    ):
        # 融合 kernel 已经原地写完；copy_(自己) 语义上是 no-op。
        return recurrent_states
    return _STATE["orig_update"](
        self, recurrent_states, state_idx=state_idx, **kwargs
    )


# ---------------------------------------------------------------------- helpers


def _fallback_reason(
    query, key, value, g, beta, initial_state, use_qk_l2norm_in_kernel, kwargs
) -> str | None:
    """命中融合路径的充分条件；返回 None 表示可以走融合。全部是 host 侧检查。"""
    if not use_qk_l2norm_in_kernel:
        return "qk-l2norm-off"
    if initial_state is None:
        return "no-initial-state"
    if kwargs.get("cu_seqlens") is not None:
        return "cu-seqlens"
    device_reason = _device_reason(query, key, value, g, beta, initial_state)
    if device_reason is not None:
        return device_reason
    return _shape_plan(query, key, value, initial_state, _STATE["ratio"])


def _conv_fallback_reason(
    hidden_states, conv_state, weight, bias, activation
) -> str | None:
    """conv step 命中融合路径的充分条件；返回 None 表示可以走融合。

    全部是 host 侧检查（graph capture 时执行一次，replay 不再经过 Python）：
    单 token、bf16、channel 维 stride=1、conv_state contiguous 且 state_len 等于
    conv kernel（窗口 == 新 state 的前置条件）、weight contiguous、无 bias、
    activation 为 silu/None。
    """
    if activation not in (None, "silu"):
        return f"activation-{activation}"
    if bias is not None:
        return "conv-bias"
    if not hidden_states.is_cuda:
        return "x-not-cuda"
    if hidden_states.dtype is not torch.bfloat16:
        return "x-dtype"
    if (
        hidden_states.dim() != 3
        or hidden_states.shape[0] != 1
        or hidden_states.shape[-1] != 1
    ):
        return "x-shape"
    if hidden_states.stride(-2) != 1:
        return "x-stride"
    if not conv_state.is_cuda:
        return "state-not-cuda"
    if conv_state.dtype is not torch.bfloat16:
        return "state-dtype"
    if conv_state.dim() != 3 or conv_state.shape[0] != 1:
        return "state-shape"
    if not conv_state.is_contiguous():
        return "state-not-contiguous"
    if not weight.is_cuda:
        return "weight-not-cuda"
    if weight.dtype is not torch.bfloat16:
        return "weight-dtype"
    if weight.dim() != 2 or not weight.is_contiguous():
        return "weight-layout"
    conv_dim = int(hidden_states.shape[1])
    if int(conv_state.shape[1]) != conv_dim or int(weight.shape[0]) != conv_dim:
        return "conv-dim-mismatch"
    state_len = int(conv_state.shape[-1])
    if state_len != int(weight.shape[-1]):
        return "state-len-kernel-mismatch"
    if state_len < 2 or state_len & (state_len - 1) != 0:
        return "state-len-not-pow2"
    if conv_dim < int(_CONV["block"]):
        return "conv-dim-lt-block"
    return None


def _device_reason(query, key, value, g, beta, initial_state) -> str | None:
    """设备/dtype/contiguity：这些不满足就不能进 kernel。"""
    if not initial_state.is_cuda:
        return "state-not-cuda"
    if initial_state.dtype is not torch.float32:
        return "state-dtype"
    if not initial_state.is_contiguous():
        return "state-not-contiguous"
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.dtype is not torch.bfloat16:
            return f"{name}-dtype"
        if not tensor.is_cuda:
            return f"{name}-not-cuda"
    if g.dtype is not torch.float32:
        return "g-dtype"
    if beta.dtype not in (torch.bfloat16, torch.float32):
        return "beta-dtype"
    return None


def _shape_plan(query, key, value, initial_state, ratio: int) -> str | None:
    """形状/head 映射检查（与设备无关，可在 CPU 上单测）。

    真实调用点传进来的是 ``repeat_interleave`` **之后**的 ``[1, 1, 48, 128]``
    （q/k 已经扩到 48 个 head），所以这里按 ``head == num_v_heads`` 判断，
    kernel 侧再用 ``[:, :, ::ratio, :]`` 取回 ``num_v_heads // ratio`` 个 k head。
    同时兼容 repeat 之前的 ``[1, 1, 16, 128]``（``head * ratio == num_v_heads``）。
    """
    if initial_state.dim() != 4 or initial_state.shape[0] != 1:
        return "state-shape"
    if query.dim() != 4 or query.shape[0] != 1 or query.shape[1] != 1:
        return "not-single-token"
    if key.shape != query.shape:
        return "qk-shape-mismatch"
    v_heads = int(initial_state.shape[1])
    if int(value.shape[-2]) != v_heads:
        return "value-head-count"
    if _qk_view(query, v_heads, ratio) is None:
        return "head-count"
    if _qk_view(key, v_heads, ratio) is None:
        return "head-count"
    if int(query.shape[-1]) != int(initial_state.shape[2]):
        return "head-k-dim"
    if int(value.shape[-1]) != int(initial_state.shape[3]):
        return "head-v-dim"
    return None


def _qk_view(tensor: torch.Tensor, v_heads: int, ratio: int) -> torch.Tensor | None:
    """q/k -> k-head 个 head 的 view：repeat 之后取 ``::ratio``，之前原样返回。"""
    heads = int(tensor.shape[-2])
    if heads == v_heads and ratio >= 1:
        return tensor[:, :, ::ratio, :]
    if heads * ratio == v_heads:
        return tensor
    return None


def _decode(query, key, value, g, beta, initial_state) -> torch.Tensor:
    """q/k 取 repeat_interleave 之后的 strided view，state 原地更新。"""
    ratio = _STATE["ratio"]
    v_heads = int(initial_state.shape[1])
    q = _qk_view(query, v_heads, ratio)
    k = _qk_view(key, v_heads, ratio)
    return gdn_decode_step(
        q,
        k,
        value,
        g,
        beta,
        initial_state,
        out=_out_slot(initial_state),
        block_v=_STATE["block_v"],
        num_warps=_STATE["num_warps"],
    )


def _out_slot(state: torch.Tensor) -> torch.Tensor:
    """按 state 缓冲的地址缓存 bf16 输出槽（graph 需要固定地址）。"""
    key = int(state.data_ptr())
    shape = (1, 1, int(state.shape[1]), int(state.shape[3]))
    slot = _STATE["out"].get(key)
    if slot is not None and tuple(slot.shape) == shape:
        return slot
    if slot is not None:
        _STATE["out_bytes"] -= slot.numel() * slot.element_size()
    slot = torch.empty(shape, dtype=torch.bfloat16, device=state.device)
    _STATE["out"][key] = slot
    _STATE["out_bytes"] += slot.numel() * slot.element_size()
    return slot


def _warn_once(tag: str, message: str) -> None:
    if tag in _STATE["warned"]:
        return
    _STATE["warned"].add(tag)
    LOGGER.warning(message)


__all__ = [
    "ENV_FLAG",
    "conv_counters",
    "conv_info",
    "counters",
    "info",
    "install",
    "install_conv",
    "install_from_env",
    "uninstall",
]
