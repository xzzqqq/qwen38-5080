"""生产者侧 activation quant 融合的模型接入（只走单 token decode）。

设计
====
任务 H 把 128 个 decoder-layer norm 与它们的 272 个 NVFP4 消费者融合成了
``runner/ops/nvfp4/norm_quant.py`` 的一个 kernel。剩下 129 个消费者的输入不是 norm
输出，而是**调用点前面刚算好的 elementwise 结果**，每个仍要走
``silu/sigmoid -> mul -> quantize -> 1/input_scale`` 4 个 kernel：

    down_proj  <- silu(gate_proj_out) * up_proj_out        (64 层，MLP.forward)
    o_proj     <- attn_output * sigmoid(gate)              (16 层，Attention.forward)
    lm_head    <- final norm 输出                          (1 次，TextModel.forward 的 self.norm)

本模块把这三条边界接到 ``runner/ops/nvfp4/quant_fuse.py`` 的融合 kernel（lm_head 用
``norm_quant.norm_quant_fused`` 的单消费者形态）。三个 patch 点都是**模块级 forward**：

- ``Qwen3_5MLP.forward``：算完 gate/up 之后，用融合 kernel 一次产出
  ``silu(gate)*up`` + down_proj 的 ``(a_fp4, a_sf)``，再 arm ``down_proj``；
- ``Qwen3_5Attention.forward``：decode 时复制 HF 的实现，只把最后
  ``attn_output * sigmoid(gate) -> o_proj`` 换成融合 kernel；
- ``Qwen3_5RMSNorm.forward``：只对 ``model.model.norm``（final norm）生效，其它
  norm（层内 norm、q_norm/k_norm）原样返回原实现。final norm 的融合结果 arming
  ``lm_head``，并注册一个 forward hook 在 lm_head 调用后清理槽位。

和 H 相同的接线风格：用 ``NVFP4Linear._pending_quantized`` 单次槽位把
``(a_fp4, a_sf)`` 交给消费者，消费者统一走 ``_gemm(..., m=1)`` 的 decode backend
选型（任务 F），**不绕过** backend 表。所有 scratch 缓冲在 install 时一次性分配，
地址固定，graph capture/replay 依赖这一点。

命中条件（任一不满足就整层回退原实现，并按原因计数）：

- patch 已安装（``QUANT_FUSE=0`` 可关）+ install 时 probe 通过；
- 输入是 ``[1, 1, 5120]``（MLP / attention）或 ``[1, 1, 5120]``（final norm）的
  contiguous BF16 CUDA，且不在 autograd 里；
- producer 的两个输入都是 contiguous BF16；否则回退。
- ``lm_head`` / ``down_proj`` / ``o_proj`` 必须是 ``NVFP4Linear`` 且 in_features 与
  install 时一致（install 已校验，运行期不再重复 dtype 检查）。

失败与回退计数（``counters.fallbacks``，四种执行方式都会累计）：

- ``seq_len!=1``：prefill / chunked prefill，正常回退（数量最大的一项）；
- ``dtype`` / ``device`` / ``grad`` / ``layout``：前置条件不满足；
- ``not-planned``：模块不在 install 计划里（正常模型不会发生；计数用于暴露接线错误）；
- ``fused-error``：融合 kernel 或 producer 调用本身抛异常（整层回退 + 一条 warning）。

真的融合了多少看 ``counters.fused_mlp`` / ``fused_attention`` / ``fused_lm_head``
与 ``unconsumed``；接线是否命中必须看这些数，不能只看端到端时间（任务 D 的教训）。

为什么 final norm 用独立 out 缓冲：它的输出会作为 ``last_hidden_state`` 离开 TextModel
（即使 CausalLM 分支只把最后 token 交给 lm_head），不能和 128 个层内 norm 复用同一块
scratch，否则调用方晚读会看到被下一次 norm 覆盖的值。

预期瓶颈：与 ``runner/ops/nvfp4/quant_fuse.py`` 相同（单 kernel 固定开销）；本模块只做
接线，graph capture 之后 Python 侧不再执行。
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
    global_divisor,
    norm_quant_fused,
)
from runner.ops.nvfp4.quant_fuse import (
    allocate_scratch as allocate_producer_scratch,
    producer_quant_fused,
    supports_k,
)

LOGGER = logging.getLogger(__name__)

#: 关掉融合路径（调试/对照用）：``QUANT_FUSE=0``。
ENV_FLAG = "QUANT_FUSE"
_DISABLED = {"0", "false", "off", "no"}

_SKIP_SHAPES = "seq_len!=1"
_SKIP_DTYPE = "dtype"
_SKIP_DEVICE = "device"
_SKIP_GRAD = "grad"
_SKIP_LAYOUT = "layout"
_SKIP_NOT_PLANNED = "not-planned"
_SKIP_ERROR = "fused-error"

_STATE: dict[str, Any] = {
    "installed": False,
    "enabled": False,
    "modeling": None,
    "orig_mlp": None,
    "orig_attention": None,
    "orig_norm": None,
    "mlp_plans": {},
    "attn_plans": {},
    "norm_plans": {},
    "scratch": {},
    "hooks": [],
    "num_warps": DEFAULT_NUM_WARPS,
    "probe": {},
    "counters": {
        "fused_mlp": 0,
        "fused_attention": 0,
        "fused_lm_head": 0,
        "unconsumed": 0,
        "fallbacks": {},
    },
    "warned": set(),
}


# ------------------------------------------------------------------------- 工具


def _bump(counter: str) -> None:
    _STATE["counters"][counter] = _STATE["counters"].get(counter, 0) + 1


def _fallback(reason: str) -> None:
    fallbacks = _STATE["counters"]["fallbacks"]
    fallbacks[reason] = fallbacks.get(reason, 0) + 1
    if reason != _SKIP_SHAPES:  # prefill 回退是常态，不刷 warning
        _warn_once(reason, f"producer quant 融合回退（{reason}），该调用点继续走原实现")


def _warn_once(reason: str, message: str) -> None:
    warned = _STATE["warned"]
    if reason not in warned:
        warned.add(reason)
        LOGGER.warning("%s", message)


def _skip_reason(x: torch.Tensor, k: int) -> str | None:
    """decode 前置条件；返回 None 表示可以走融合。"""
    if not _STATE["enabled"]:
        return _SKIP_NOT_PLANNED
    if x.dtype is not torch.bfloat16:
        return _SKIP_DTYPE
    if x.device.type != "cuda":
        return _SKIP_DEVICE
    if torch.is_grad_enabled():
        return _SKIP_GRAD
    if x.shape != (1, 1, int(k)) or not x.is_contiguous():
        return _SKIP_SHAPES
    return None


def _probe_quantizer(device: torch.device, k: int) -> dict[str, Any]:
    """用 FlashInfer 自己的输出确定 (k) 的 a_fp4/a_sf dtype 与 shape（不猜）。"""
    from flashinfer import SfLayout, nvfp4_quantize

    probe = torch.zeros(1, int(k), dtype=torch.bfloat16, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    a_fp4, a_sf = nvfp4_quantize(
        probe,
        scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    if tuple(a_fp4.shape) != (1, int(k) // 2):
        raise RuntimeError(f"probe a_fp4 shape {tuple(a_fp4.shape)} for K={k}")
    if tuple(a_sf.shape) != (SF_ROWS, int(k) // 16):
        raise RuntimeError(f"probe a_sf shape {tuple(a_sf.shape)} for K={k}")
    return {
        "a_fp4_dtype": str(a_fp4.dtype),
        "a_sf_dtype": str(a_sf.dtype),
        "a_fp4_shape": list(a_fp4.shape),
        "a_sf_shape": list(a_sf.shape),
    }


def _torch_dtype(name: str) -> torch.dtype:
    value = getattr(torch, str(name).rsplit(".", 1)[-1], None)
    return value if isinstance(value, torch.dtype) else torch.uint8


def _scratch_for(device: torch.device, k: int, *, dedicated_out: bool = False) -> dict:
    """按 K 分配（或复用）融合缓冲：BF16 输出 + 1 份 (a_fp4, a_sf)。

    同一个 K 的多个调用点顺序消费同一份缓冲（与 H 的 3 槽 scratch 同理），地址固定。
    ``dedicated_out`` 给 final norm：它的输出会离开 TextModel，必须独占。
    """
    key = (int(k), bool(dedicated_out))
    cached = _STATE["scratch"].get(key)
    if cached is not None:
        return cached
    probe = _STATE["probe"][int(k)]
    out, a_fp4, a_sf = allocate_producer_scratch(
        int(k),
        device=device,
        a_fp4_shape=tuple(probe["a_fp4_shape"]),
        a_sf_shape=tuple(probe["a_sf_shape"]),
        a_fp4_dtype=_torch_dtype(probe["a_fp4_dtype"]),
        a_sf_dtype=_torch_dtype(probe["a_sf_dtype"]),
    )
    scratch = {
        "out": out,
        "a_fp4": a_fp4,
        "a_sf": a_sf,
        "bytes": (
            out.numel() * out.element_size()
            + a_fp4.numel() * a_fp4.element_size()
            + a_sf.numel() * a_sf.element_size()
        ),
    }
    _STATE["scratch"][key] = scratch
    return scratch


def _disarm(consumer: NVFP4Linear) -> None:
    if consumer._pending_quantized is not None:
        consumer._pending_quantized = None
        _bump("unconsumed")


def _finish_mlp(plan: _Plan, gate: torch.Tensor, up: torch.Tensor):
    """silu(gate)*up + quant -> arm down_proj -> down_proj 投影。失败返回 None。"""
    if (
        gate.dtype is not torch.bfloat16
        or up.dtype is not torch.bfloat16
        or not gate.is_contiguous()
        or not up.is_contiguous()
        or gate.numel() != plan.k
    ):
        _fallback(_SKIP_LAYOUT)
        return None
    scratch = plan.scratch
    down_proj = plan.down_proj
    try:
        producer_quant_fused(
            gate,
            up,
            "silu_mul",
            plan.gs,
            scratch["out"],
            scratch["a_fp4"],
            scratch["a_sf"],
            num_warps=_STATE["num_warps"],
        )
    except Exception as exc:  # noqa: BLE001 - 融合失败回退，不让 decode 崩
        _warn_once(_SKIP_ERROR, f"silu*mul 融合 kernel 失败：{exc!r}")
        _fallback(_SKIP_ERROR)
        return None
    down_proj._pending_quantized = (scratch["a_fp4"], scratch["a_sf"])
    try:
        out = down_proj(scratch["out"])
    finally:
        _disarm(down_proj)
    _bump("fused_mlp")
    return out


def _finish_attention(self, attn_output: torch.Tensor, gate: torch.Tensor):
    """attn*sigmoid(gate) + quant -> arm o_proj -> o_proj 投影。失败返回 None。"""
    plan = _STATE["attn_plans"].get(id(self))
    if plan is None:
        _fallback(_SKIP_NOT_PLANNED)
        return None
    if (
        attn_output.dtype is not torch.bfloat16
        or gate.dtype is not torch.bfloat16
        or not attn_output.is_contiguous()
        or not gate.is_contiguous()
        or attn_output.numel() != plan.k
    ):
        _fallback(_SKIP_LAYOUT)
        return None
    scratch = plan.scratch
    o_proj = plan.o_proj
    try:
        producer_quant_fused(
            attn_output,
            gate,
            "sigmoid_mul",
            plan.gs,
            scratch["out"],
            scratch["a_fp4"],
            scratch["a_sf"],
            num_warps=_STATE["num_warps"],
        )
    except Exception as exc:  # noqa: BLE001
        _warn_once(_SKIP_ERROR, f"attn*sigmoid 融合 kernel 失败：{exc!r}")
        _fallback(_SKIP_ERROR)
        return None
    o_proj._pending_quantized = (scratch["a_fp4"], scratch["a_sf"])
    try:
        out = o_proj(scratch["out"])
    finally:
        _disarm(o_proj)
    _bump("fused_attention")
    return out


# ------------------------------------------------------------------------- 接线


def _patched_mlp(self, x: torch.Tensor):
    """``Qwen3_5MLP.forward`` 的 decode 版：gate/up -> 融合 silu·mul+quant -> down_proj。"""
    original = _STATE["orig_mlp"]
    plan = _STATE["mlp_plans"].get(id(self))
    if plan is None:
        return original(self, x)
    reason = _skip_reason(x, HIDDEN)
    if reason is not None:
        _fallback(reason)
        return original(self, x)

    gate = self.gate_proj(x)
    up = self.up_proj(x)
    out = _finish_mlp(plan, gate, up)
    if out is None:
        # 语义完全等价的原实现：两份输入已经算好，只是走 torch 的 elementwise + quant。
        return self.down_proj(self.act_fn(gate) * up)
    return out


def _patched_attention(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
):
    """``Qwen3_5Attention.forward`` 的 decode 版：最后一步 o_proj 用融合 quant。

    只有 ``hidden_states`` 是单 token 时才复制 HF 的 forward；prefill 直接回原实现。
    融合失败时在同一段代码里退回 ``attn_output * sigmoid(gate) -> o_proj``。
    """
    original = _STATE["orig_attention"]
    plan = _STATE["attn_plans"].get(id(self))
    if plan is None:
        return original(
            self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs
        )
    reason = _skip_reason(hidden_states, HIDDEN)
    if reason is not None:
        _fallback(reason)
        return original(
            self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs
        )

    modeling = _STATE["modeling"]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = modeling.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    attention_interface = modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modeling.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    fused = _finish_attention(self, attn_output, gate)
    if fused is None:
        fused = self.o_proj(attn_output * torch.sigmoid(gate))
    return fused, attn_weights


def _patched_norm(self, x: torch.Tensor):
    """``Qwen3_5RMSNorm.forward``：只对 final norm 生效，其它 norm 原样返回。"""
    original = _STATE["orig_norm"]
    plan = _STATE["norm_plans"].get(id(self))
    if plan is None:
        return original(self, x)
    # 上一步的槽位理论上已被 lm_head 取走；万一没取走（例如那次 forward 没算 logits），
    # 这里先清掉并计数，避免 prefill 的 lm_head 误用陈旧量化结果。
    _disarm(plan["lm_head"])
    reason = _skip_reason(x, HIDDEN)
    if reason is not None:
        _fallback(reason)
        return original(self, x)
    try:
        norm_quant_fused(
            x,
            plan["w1"],
            plan["eps"],
            [plan["gs"]],
            plan["out"],
            [plan["a_fp4"]],
            [plan["a_sf"]],
            num_warps=_STATE["num_warps"],
        )
    except Exception as exc:  # noqa: BLE001
        _warn_once(_SKIP_ERROR, f"final norm 融合 kernel 失败：{exc!r}")
        _fallback(_SKIP_ERROR)
        return original(self, x)
    plan["lm_head"]._pending_quantized = (plan["a_fp4"], plan["a_sf"])
    _bump("fused_lm_head")
    return plan["out"]


def _lm_head_hook(module: NVFP4Linear, args: tuple, output: Any) -> None:
    """lm_head 调用后清理槽位：正常情况下 forward 已经取走；没取走说明接线漏了。"""
    _disarm(module)


class _Plan:
    """一个 producer 调用点的运行期信息（模块引用 + gs + 共享 scratch）。"""

    __slots__ = ("k", "gs", "down_proj", "o_proj", "scratch")

    def __init__(self, *, k: int, gs: float, scratch: dict, down_proj=None, o_proj=None):
        self.k = int(k)
        self.gs = float(gs)
        self.scratch = scratch
        self.down_proj = down_proj
        self.o_proj = o_proj


def _device_of(model: Any) -> torch.device:
    tensors = []
    text_model = getattr(model, "model", None)
    for layer in list(getattr(text_model, "layers", []) or []):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "down_proj"):
            tensors.append(mlp.down_proj.weight_packed)
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "o_proj"):
            tensors.append(attn.o_proj.weight_packed)
    norm = getattr(text_model, "norm", None)
    if norm is not None and getattr(norm, "weight", None) is not None:
        tensors.append(norm.weight)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight_packed"):
        tensors.append(lm_head.weight_packed)
    if not tensors:
        raise RuntimeError("找不到可用的 producer 模块")
    device = tensors[0].device
    if device.type != "cuda":
        raise RuntimeError(f"producer 模块不在 CUDA 上：{device}")
    for tensor in tensors[1:]:
        if tensor.device != device:
            raise RuntimeError(f"producer 模块不在同一 device：{tensor.device} vs {device}")
    return device


def _as_quant_linear(
    module: Any, name: str, in_features: int | None = None
) -> NVFP4Linear:
    if not isinstance(module, NVFP4Linear):
        raise RuntimeError(f"{name} 不是 NVFP4Linear: {type(module).__name__}")
    if in_features is not None and int(module.in_features) != int(in_features):
        raise RuntimeError(
            f"{name} in_features={module.in_features} != {in_features}"
        )
    return module


def install(model: torch.nn.Module, *, num_warps: int = DEFAULT_NUM_WARPS) -> dict[str, Any]:
    """把模型里的 down_proj / o_proj / lm_head 接到生产者侧融合 kernel；幂等。"""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    if _STATE["installed"]:
        uninstall()

    text_model = getattr(model, "model", None)
    layers = list(getattr(text_model, "layers", []) or [])
    if not layers:
        raise RuntimeError("模型里没有 Qwen3_5TextModel.layers")
    device = _device_of(model)

    # ---- 1) MLP：down_proj 的输入 = silu(gate_proj_out) * up_proj_out
    mlp_ks: set[int] = set()
    mlps = [layer.mlp for layer in layers if getattr(layer, "mlp", None) is not None]
    if not mlps:
        raise RuntimeError("模型里没有 Qwen3_5MLP")
    for index, mlp in enumerate(mlps):
        down = _as_quant_linear(mlp.down_proj, f"layers.{index}.mlp.down_proj")
        _as_quant_linear(mlp.gate_proj, f"layers.{index}.mlp.gate_proj", HIDDEN)
        _as_quant_linear(mlp.up_proj, f"layers.{index}.mlp.up_proj", HIDDEN)
        mlp_ks.add(int(down.in_features))
    if len(mlp_ks) != 1:
        raise RuntimeError(f"MLP 的 down_proj 输入维度不唯一：{sorted(mlp_ks)}")
    mlp_k = mlp_ks.pop()
    if not supports_k(mlp_k):
        raise RuntimeError(f"down_proj K={mlp_k} 不被 producer kernel 支持")

    # ---- 2) attention：o_proj 的输入 = attn_output * sigmoid(gate)
    attn_ks: set[int] = set()
    attns = [
        layer.self_attn for layer in layers if getattr(layer, "self_attn", None) is not None
    ]
    for index, attn in enumerate(attns):
        o_proj = _as_quant_linear(attn.o_proj, f"attn.{index}.o_proj")
        attn_ks.add(int(o_proj.in_features))
    if attns and len(attn_ks) != 1:
        raise RuntimeError(f"attention 的 o_proj 输入维度不唯一：{sorted(attn_ks)}")
    attn_k = attn_ks.pop() if attn_ks else None
    if attn_k is not None and not supports_k(attn_k):
        raise RuntimeError(f"o_proj K={attn_k} 不被 producer kernel 支持")

    # ---- 3) final norm -> lm_head
    norm = getattr(text_model, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if norm is None or not hasattr(norm, "weight"):
        raise RuntimeError("模型里没有 final norm")
    if int(norm.weight.numel()) != HIDDEN:
        raise RuntimeError(f"final norm hidden={norm.weight.numel()} != {HIDDEN}")
    lm_head = _as_quant_linear(lm_head, "lm_head", HIDDEN)

    # ---- probe + scratch（一次性分配，graph replay 依赖固定地址）
    probe: dict[int, dict] = {}
    probe_ks = [mlp_k, HIDDEN]
    if attn_k is not None:
        probe_ks.append(attn_k)
    for k in probe_ks:
        probe[int(k)] = _probe_quantizer(device, int(k))
    _STATE["probe"] = probe
    _STATE["scratch"] = {}
    producer_scratch = _scratch_for(device, mlp_k)
    attn_scratch = _scratch_for(device, attn_k) if attn_k is not None else None
    final_scratch = _scratch_for(device, HIDDEN, dedicated_out=True)

    mlp_plans: dict[int, _Plan] = {}
    for mlp in mlps:
        gs = global_divisor(float(mlp.down_proj.input_scale.reshape(1).float().item()))
        mlp_plans[id(mlp)] = _Plan(
            k=mlp_k, gs=gs, scratch=producer_scratch, down_proj=mlp.down_proj
        )
    attn_plans: dict[int, _Plan] = {}
    for attn in attns:
        gs = global_divisor(float(attn.o_proj.input_scale.reshape(1).float().item()))
        attn_plans[id(attn)] = _Plan(
            k=attn_k, gs=gs, scratch=attn_scratch, o_proj=attn.o_proj
        )
    norm_plans: dict[int, dict] = {}
    norm_plans[id(norm)] = {
        "w1": (1.0 + norm.weight.detach().float()).contiguous(),
        "eps": float(norm.eps),
        "gs": global_divisor(float(lm_head.input_scale.reshape(1).float().item())),
        "out": final_scratch["out"],
        "a_fp4": final_scratch["a_fp4"],
        "a_sf": final_scratch["a_sf"],
        "lm_head": lm_head,
    }

    _STATE["modeling"] = modeling
    _STATE["orig_mlp"] = modeling.Qwen3_5MLP.forward
    _STATE["orig_attention"] = modeling.Qwen3_5Attention.forward
    _STATE["orig_norm"] = modeling.Qwen3_5RMSNorm.forward
    _STATE["mlp_plans"] = mlp_plans
    _STATE["attn_plans"] = attn_plans
    _STATE["norm_plans"] = norm_plans
    _STATE["num_warps"] = int(num_warps)
    _STATE["counters"] = {
        "fused_mlp": 0,
        "fused_attention": 0,
        "fused_lm_head": 0,
        "unconsumed": 0,
        "fallbacks": {},
    }
    _STATE["warned"] = set()
    _STATE["hooks"] = [lm_head.register_forward_hook(_lm_head_hook)]
    _STATE["enabled"] = True
    modeling.Qwen3_5MLP.forward = _patched_mlp
    modeling.Qwen3_5Attention.forward = _patched_attention
    modeling.Qwen3_5RMSNorm.forward = _patched_norm
    _STATE["installed"] = True
    return info()


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    modeling = _STATE["modeling"]
    modeling.Qwen3_5MLP.forward = _STATE["orig_mlp"]
    modeling.Qwen3_5Attention.forward = _STATE["orig_attention"]
    modeling.Qwen3_5RMSNorm.forward = _STATE["orig_norm"]
    for handle in _STATE["hooks"]:
        handle.remove()
    for plan in list(_STATE["mlp_plans"].values()):
        if plan.down_proj is not None:
            plan.down_proj._pending_quantized = None
    for plan in list(_STATE["attn_plans"].values()):
        if plan.o_proj is not None:
            plan.o_proj._pending_quantized = None
    for plan in _STATE["norm_plans"].values():
        plan["lm_head"]._pending_quantized = None
    _STATE["installed"] = False
    _STATE["enabled"] = False
    _STATE["hooks"] = []
    _STATE["mlp_plans"] = {}
    _STATE["attn_plans"] = {}
    _STATE["norm_plans"] = {}


def info() -> dict[str, Any]:
    scratch_bytes = 0
    for scratch in _STATE["scratch"].values():
        scratch_bytes += int(scratch["bytes"])
    return {
        "enabled": _STATE["enabled"],
        "layers": len(_STATE["mlp_plans"]),
        "mlp": len(_STATE["mlp_plans"]),
        "attention": len(_STATE["attn_plans"]),
        "final_norm": len(_STATE["norm_plans"]),
        "k": {
            "down_proj": next(iter(_STATE["mlp_plans"].values())).k
            if _STATE["mlp_plans"]
            else None,
            "o_proj": next(iter(_STATE["attn_plans"].values())).k
            if _STATE["attn_plans"]
            else None,
            "lm_head": next(iter(_STATE["norm_plans"].values()))["a_fp4"].numel() * 2
            if _STATE["norm_plans"]
            else None,
        },
        "num_warps": _STATE["num_warps"],
        "probe": _STATE["probe"],
        "scratch_bytes": scratch_bytes,
        # info() 返回的是同一个 dict，产物在末尾序列化时能看到最终计数。
        "counters": _STATE["counters"],
    }


def install_from_env(model: torch.nn.Module) -> dict[str, Any]:
    """loader 用：默认打开，``QUANT_FUSE=0`` 关闭；失败只记录不抛。"""
    flag = os.environ.get(ENV_FLAG, "1").strip().lower()
    if flag in _DISABLED:
        return {"enabled": False, "reason": f"{ENV_FLAG}={flag}"}
    try:
        result = install(model)
    except Exception as exc:  # noqa: BLE001 - 接入失败时回退原路径，不让非 decode 工具崩
        LOGGER.error("producer quant 融合接入失败（回退原路径）：%r", exc)
        return {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}
    LOGGER.warning("producer quant 融合已接入: %s", json.dumps(result, ensure_ascii=False))
    return result


__all__ = ["ENV_FLAG", "info", "install", "install_from_env", "uninstall"]
