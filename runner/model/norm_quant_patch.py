"""RMSNorm + NVFP4 activation quant 融合的模型接入（只走单 token decode）。

设计
====
现有 decode 路径里，每个 decoder layer 有两次 norm，每次 norm 的 BF16 输出被
若干个 ``NVFP4Linear`` 消费，每个消费者再各自调一次 FlashInfer
``quantize_with_block_size``：

    input_layernorm         -> linear_attention 层：in_proj_qkv + in_proj_z（2 个消费者）
                            -> full_attention 层：q_proj + k_proj + v_proj（3 个消费者）
    post_attention_layernorm-> gate_proj + up_proj（2 个消费者）

每个消费者有自己的 per-tensor ``input_scale``，所以融合 kernel 必须按消费者各产
一份 ``(a_fp4, a_sf)``，不能共享同一份量化结果（见任务 E 的 quant 契约）。

接入点选在**模块级** ``Qwen3_5DecoderLayer.forward``：HF 的 forward 只有
「norm -> mixer -> residual -> norm -> mlp -> residual」六步，复制这段逻辑就能在
norm 与它的消费者之间插入融合 kernel，不需要改 mixer / MLP / NVFP4Linear 的源码，
也不会影响 prefill（prefill 的 ``seq_len > 1`` 直接走原 forward）。

命中条件（任一不满足就走原 forward，并按原因计数）：

- patch 已安装（``NORM_QUANT_FUSED=0`` 可关）、install 时的 probe 通过；
- ``hidden_states`` 是 ``[1, 1, 5120]`` contiguous BF16 CUDA、且不在 autograd 里；
- 这一层的 norm / 消费者的类型与 shape 与 install 时一致（install 时已校验）。

量化结果的传递
==============
``NVFP4Linear`` 有一个单次槽位 ``_pending_quantized``：patch 在调用 mixer **之前**
把 ``(a_fp4, a_sf)`` 放到对应模块上，``forward`` 取走并立刻清空（所以同一个模块在
一个 step 里只会命中一次，多出来的调用自动回退到普通量化路径）。``finally`` 里会
清掉没被消费的槽位并计数 —— 与任务 D 的教训一致：接线是否真的命中必须能在产物里
看到（``counters.fused_norms`` / ``counters.unconsumed``），不能只看端到端时间。

为什么要复算 norm 而不是复用 PyTorch 的 BF16 输出
=================================================
norm 是这条路径上 kernel 数量的大头（``x.float()``、``pow``、``mean``、``add``、
``rsqrt``、``mul``、``1+w``、``mul``、``to(bf16)`` 约 8 个 kernel/次）。融合 kernel
按 ``runner/ops/nvfp4/norm_quant.py`` 的 bitwise 契约一次算出「BF16 归一化值 +
每消费者一份 NVFP4 A」，所以这 8 个 kernel 和每个消费者的 quantize kernel 一起被
1 个 kernel 取代。BF16 归一化值仍然写出，因为 linear_attention 层的 in_proj_a /
in_proj_b（未量化）还要读它。

scratch 缓冲在 install 时一次性分配（BF16 归一化输出 1 份 + a_fp4/a_sf 各 3 槽 +
padding 清零），地址固定：graph capture 之后 replay 依赖这一点。同一份缓冲被
128 个 norm 复用，靠 stream 顺序保证「先写、读、下一次覆盖」的依赖关系。

预期瓶颈：与 ``runner/ops/nvfp4/norm_quant.py`` 相同（单 kernel 固定开销）；本模块
只做接线，每层每步多两次 Python 侧的 shape 检查（graph 路径在 capture 后不执行）。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from runner.ops.nvfp4.linear import NVFP4Linear
from runner.ops.nvfp4.norm_quant import (
    DEFAULT_NUM_WARPS,
    HIDDEN,
    SF_ROWS,
    allocate_scratch,
    global_divisor,
    norm_quant_fused,
)

LOGGER = logging.getLogger(__name__)

#: 关掉融合路径（调试/对照用）：``NORM_QUANT_FUSED=0``。
ENV_FLAG = "NORM_QUANT_FUSED"
_DISABLED = {"0", "false", "off", "no"}

_SKIP_SHAPES = "seq_len!=1"
_SKIP_DTYPE = "dtype"
_SKIP_DEVICE = "device"
_SKIP_GRAD = "grad"
_SKIP_WIRING = "wiring"
_SKIP_MASK = "mask"

_STATE: dict[str, Any] = {
    "installed": False,
    "enabled": False,
    "orig_forward": None,
    "scratch": None,
    "plans": {},
    "num_warps": DEFAULT_NUM_WARPS,
    "counters": {
        "fused_norms": 0,
        "fused_consumers": 0,
        "unconsumed": 0,
        "fallbacks": {},
    },
    "probe": {},
    "block_types": {},
    "warned": set(),
}


# ------------------------------------------------------------------------- 工具


def _bump(counter: str) -> None:
    _STATE["counters"][counter] = _STATE["counters"].get(counter, 0) + 1


def _fallback(reason: str) -> None:
    fallbacks = _STATE["counters"]["fallbacks"]
    fallbacks[reason] = fallbacks.get(reason, 0) + 1
    warned = _STATE["warned"]
    if reason not in warned:
        warned.add(reason)
        LOGGER.warning("norm+quant 融合回退（%s），该路径继续走原 forward", reason)


class _NormQuant:
    """一个 norm 模块的融合状态：``w1 = 1 + weight.float()``、eps、消费者 scale。"""

    __slots__ = ("w1", "eps", "name")

    def __init__(self, norm: torch.nn.Module, name: str) -> None:
        weight = norm.weight
        if weight.numel() != HIDDEN:
            raise ValueError(f"{name}: expected hidden {HIDDEN}, got {weight.numel()}")
        # 与 eager 完全相同的两步：``weight.float()`` 再 ``1.0 +``（f32 add.rn）。
        self.w1 = (1.0 + weight.detach().float()).contiguous()
        self.eps = float(norm.eps)
        self.name = name

    def __call__(self, x: torch.Tensor, consumers: list[NVFP4Linear]):
        scratch = _STATE["scratch"]
        gs = [consumer._norm_quant_gs for consumer in consumers]
        norm_quant_fused(
            x,
            self.w1,
            self.eps,
            gs,
            scratch["out"],
            scratch["a_fp4"],
            scratch["a_sf"],
            num_warps=_STATE["num_warps"],
        )
        _bump("fused_norms")
        _STATE["counters"]["fused_consumers"] += len(consumers)
        quants = [
            (scratch["a_fp4"][index], scratch["a_sf"][index])
            for index in range(len(consumers))
        ]
        return scratch["out"], quants


class _LayerPlan:
    """一层 decode 的融合计划：两个 norm 的融合状态 + 各自的消费者列表。"""

    __slots__ = ("input_nq", "post_nq", "pre_consumers", "post_consumers", "is_linear")

    def __init__(
        self,
        input_nq: _NormQuant,
        post_nq: _NormQuant,
        pre_consumers: list[NVFP4Linear],
        post_consumers: list[NVFP4Linear],
        is_linear: bool,
    ) -> None:
        self.input_nq = input_nq
        self.post_nq = post_nq
        self.pre_consumers = pre_consumers
        self.post_consumers = post_consumers
        self.is_linear = is_linear


# ------------------------------------------------------------------------- 接线


def _arm(consumers: list[NVFP4Linear], quants: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for consumer, quant in zip(consumers, quants):
        consumer._pending_quantized = quant


def _disarm(consumers: list[NVFP4Linear]) -> None:
    for consumer in consumers:
        if consumer._pending_quantized is not None:
            consumer._pending_quantized = None
            _bump("unconsumed")


def _skip_reason(hidden_states: torch.Tensor) -> str | None:
    if not _STATE["enabled"]:
        return _SKIP_WIRING
    if hidden_states.dtype is not torch.bfloat16:
        return _SKIP_DTYPE
    if hidden_states.device.type != "cuda":
        return _SKIP_DEVICE
    if torch.is_grad_enabled():
        return _SKIP_GRAD
    if hidden_states.shape != (1, 1, HIDDEN) or not hidden_states.is_contiguous():
        return _SKIP_SHAPES
    return None


def _patched_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any = None,
    **kwargs: Any,
):
    """``Qwen3_5DecoderLayer.forward`` 的 decode 版：norm 与 quant 合成一个 kernel。"""
    original = _STATE["orig_forward"]
    plan = _STATE["plans"].get(id(self))
    if plan is None:
        return original(
            self, hidden_states, position_embeddings, attention_mask, position_ids,
            past_key_values, **kwargs,
        )
    reason = _skip_reason(hidden_states)
    if reason is None and plan.is_linear and attention_mask is not None:
        # GDN 会在 in_proj_qkv 之前做 ``apply_mask_to_padding_states``；一旦它真的
        # 乘了 mask，量化输入就不再是归一化值本身。当前 decode 路径
        # ``create_recurrent_attention_mask`` 对 seq_len==1 恒返回 None，这里只是
        # 把「万一」挡在门外（宁可回退，也不能静默改变语义）。
        reason = _SKIP_MASK
    if reason is not None:
        _fallback(reason)
        return original(
            self, hidden_states, position_embeddings, attention_mask, position_ids,
            past_key_values, **kwargs,
        )

    residual = hidden_states
    normed, quants = plan.input_nq(hidden_states, plan.pre_consumers)
    try:
        _arm(plan.pre_consumers, quants)
        if plan.is_linear:
            mixed = self.linear_attn(
                hidden_states=normed,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            mixed, _ = self.self_attn(
                hidden_states=normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
    finally:
        _disarm(plan.pre_consumers)

    hidden_states = residual + mixed
    residual = hidden_states
    normed, quants = plan.post_nq(hidden_states, plan.post_consumers)
    try:
        _arm(plan.post_consumers, quants)
        mixed = self.mlp(normed)
    finally:
        _disarm(plan.post_consumers)
    return residual + mixed


def _consumers_for(layer: Any, block_type: str) -> tuple[list[Any], list[Any]]:
    if block_type == "linear_attention":
        pre = [layer.linear_attn.in_proj_qkv, layer.linear_attn.in_proj_z]
    elif block_type == "full_attention":
        pre = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        ]
    else:
        raise ValueError(f"unsupported block_type {block_type!r}")
    post = [layer.mlp.gate_proj, layer.mlp.up_proj]
    return pre, post


def _torch_dtype(name: str) -> torch.dtype:
    """``"torch.uint8"`` -> ``torch.uint8``；解析不出来就用 uint8（单字节布局不变）。"""
    value = getattr(torch, str(name).rsplit(".", 1)[-1], None)
    if isinstance(value, torch.dtype):
        return value
    return torch.uint8


def _probe_quantizer(device: torch.device) -> dict[str, Any]:
    """用 FlashInfer 自己的输出确定 a_fp4/a_sf 的 dtype 与 shape（不猜）。"""
    from flashinfer import SfLayout, nvfp4_quantize

    probe = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    a_fp4, a_sf = nvfp4_quantize(
        probe,
        scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    if tuple(a_fp4.shape) != (1, HIDDEN // 2):
        raise RuntimeError(f"probe a_fp4 shape {tuple(a_fp4.shape)}")
    if tuple(a_sf.shape) != (SF_ROWS, HIDDEN // 16):
        raise RuntimeError(f"probe a_sf shape {tuple(a_sf.shape)}")
    return {
        "a_fp4_dtype": str(a_fp4.dtype),
        "a_sf_dtype": str(a_sf.dtype),
        "a_fp4_shape": list(a_fp4.shape),
        "a_sf_shape": list(a_sf.shape),
    }


def install(
    model: torch.nn.Module,
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> dict[str, Any]:
    """把模型里 128 个 decoder-layer norm 接到融合 kernel 上；重复安装是幂等的。"""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    if _STATE["installed"]:
        uninstall()

    text_model = getattr(model, "model", None)
    layers = list(getattr(text_model, "layers", []) or [])
    if not layers:
        raise RuntimeError("模型里没有 Qwen3_5TextModel.layers")

    device = layers[0].input_layernorm.weight.device
    if device.type != "cuda":
        raise RuntimeError(f"norm 权重不在 CUDA 上: {device}")
    probe = _probe_quantizer(device)
    scratch = allocate_scratch(
        device=device,
        a_fp4_shape=tuple(probe["a_fp4_shape"]),
        a_sf_shape=tuple(probe["a_sf_shape"]),
        a_fp4_dtype=_torch_dtype(probe["a_fp4_dtype"]),
        a_sf_dtype=_torch_dtype(probe["a_sf_dtype"]),
    )
    out_bf16, a_fp4, a_sf = scratch
    scratch = {
        "out": out_bf16,
        "a_fp4": a_fp4,
        "a_sf": a_sf,
        "bytes": sum(t.numel() * t.element_size() for t in [out_bf16, *a_fp4, *a_sf]),
    }

    plans: dict[int, _LayerPlan] = {}
    counts = {"linear_attention": 0, "full_attention": 0}
    for layer in layers:
        block_type = str(layer.block_type)
        pre, post = _consumers_for(layer, block_type)
        for module in [*pre, *post]:
            if not isinstance(module, NVFP4Linear):
                raise RuntimeError(
                    f"消费者不是 NVFP4Linear: {type(module).__name__}"
                )
        for consumer in pre:
            consumer._norm_quant_gs = global_divisor(
                float(consumer.input_scale.reshape(1).float().item())
            )
        for consumer in post:
            consumer._norm_quant_gs = global_divisor(
                float(consumer.input_scale.reshape(1).float().item())
            )
        index = len(plans)
        plans[id(layer)] = _LayerPlan(
            _NormQuant(layer.input_layernorm, f"layers.{index}.input_layernorm"),
            _NormQuant(layer.post_attention_layernorm, f"layers.{index}.post_attention_layernorm"),
            pre,
            post,
            block_type == "linear_attention",
        )
        counts[block_type] = counts.get(block_type, 0) + 1

    _STATE["orig_forward"] = modeling.Qwen3_5DecoderLayer.forward
    _STATE["plans"] = plans
    _STATE["scratch"] = scratch
    _STATE["num_warps"] = int(num_warps)
    _STATE["probe"] = probe
    _STATE["block_types"] = dict(counts)
    _STATE["counters"] = {
        "fused_norms": 0,
        "fused_consumers": 0,
        "unconsumed": 0,
        "fallbacks": {},
    }
    _STATE["warned"] = set()
    _STATE["enabled"] = True
    modeling.Qwen3_5DecoderLayer.forward = _patched_forward
    _STATE["installed"] = True
    return info()


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    modeling.Qwen3_5DecoderLayer.forward = _STATE["orig_forward"]
    for plan in _STATE["plans"].values():
        for consumer in [*plan.pre_consumers, *plan.post_consumers]:
            consumer._pending_quantized = None
    _STATE["installed"] = False
    _STATE["enabled"] = False
    _STATE["plans"] = {}


def info() -> dict[str, Any]:
    scratch = _STATE["scratch"] or {}
    plans = _STATE["plans"]
    consumers = sum(
        len(plan.pre_consumers) + len(plan.post_consumers) for plan in plans.values()
    )
    return {
        "enabled": _STATE["enabled"],
        "layers": len(plans),
        "norms": len(plans) * 2,
        "quantized_consumers": consumers,
        "block_types": _STATE["block_types"],
        "num_warps": _STATE["num_warps"],
        "probe": _STATE["probe"],
        "scratch_bytes": int(scratch.get("bytes", 0)),
        # info() 返回的是同一个 dict，产物在末尾序列化时能看到最终计数。
        "counters": _STATE["counters"],
    }


def install_from_env(model: torch.nn.Module) -> dict[str, Any]:
    """loader 用：默认打开，``NORM_QUANT_FUSED=0`` 关闭；失败只记录不抛。"""
    flag = os.environ.get(ENV_FLAG, "1").strip().lower()
    if flag in _DISABLED:
        return {"enabled": False, "reason": f"{ENV_FLAG}={flag}"}
    try:
        result = install(model)
    except Exception as exc:  # noqa: BLE001 - 接入失败时回退原路径，不让非 decode 工具崩
        LOGGER.error("norm+quant 融合接入失败（回退原路径）：%r", exc)
        return {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}
    LOGGER.warning("norm+quant 融合已接入: %s", json.dumps(result, ensure_ascii=False))
    return result


__all__ = ["ENV_FLAG", "info", "install", "install_from_env", "uninstall"]
