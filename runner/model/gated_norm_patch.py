"""GDN gated RMSNorm（``Qwen3_5RMSNormGated``）+ out_proj 的 activation quant 融合接入。

设计
====
48 个 linear-attention 层的 ``Qwen3_5GatedDeltaNet.forward`` 末尾是：

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)   # [48, 128]
    z = z.reshape(-1, self.head_v_dim)                           # [48, 128]
    core_attn_out = self.norm(core_attn_out, z)                  # gated RMSNorm
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
    output = self.out_proj(core_attn_out)                        # NVFP4Linear, K = 6144

``self.norm`` 是纯 PyTorch 的 12 个 tiny kernel，``self.out_proj`` 又对同一个
``[1,1,6144]`` 做一次 ``quantize_with_block_size``。本 patch 把这两步合成 1 个
Triton kernel（``runner/ops/gdn/gated_norm.py``）：kernel 写出 BF16 的
``normed * silu(gate)``（= 模块 forward 的返回值）**并且**直接产出 out_proj 需要
的 ``(a_fp4, a_sf)``。

接入点
======
patch 的是**类** ``Qwen3_5RMSNormGated.forward``。它比复制
``Qwen3_5GatedDeltaNet.forward`` 更窄：只需要知道「谁消费这个 norm 的输出」。
install 时遍历模型里的 ``Qwen3_5GatedDeltaNet`` 模块，得到
``layer.norm`` -> ``layer.out_proj`` 的对应关系（以及各自的 ``input_scale``），
所以 norm 的 patched forward 能直接把量化结果放到对应 out_proj 的
``_pending_quantized`` 单次槽位上。

- 只覆盖 ``norm(core_attn_out, z)`` 在 decode 时的**唯一**消费者 out_proj；
  prefill（``hidden_states.shape != (48, 128)``）回退原 forward。
- ``_pending_quantized`` 是 ``NVFP4Linear`` 已有的单次槽位（任务 H 引入）：
  ``forward`` 取走即清空，所以同一个模块一个 step 只会命中一次；没被消费的槽位在
  下一次 arm 之前计数并覆盖（``counters.unconsumed``）。
- **融合快路径必须走 ``NVFP4Linear._gemm(..., m=1)``**（槽位消费点在
  ``NVFP4Linear.forward`` 里），不能绕过 decode backend 选型：任务 F 把 401 个
  GEMV 从 cutlass 换到 cute-dsl，48 个 out_proj 也在表里（``(5120, 6144)``）。
  绕过它会让这 48 个 GEMV 退回 cutlass，白丢收益（任务 H §7.2 的教训）。

计数（``counters``，四种执行方式都会累计）
==========================================
- ``fused``：融合 kernel 成功执行的次数（每层每步一次，48 层）；
- ``fallbacks``：回退原因 -> 次数（``shape`` / ``dtype`` / ``device`` / ``grad`` /
  ``wiring`` / ``kernel-error``）；
- ``unconsumed``：arm 了但 out_proj 没消费的槽位数（正常应为 0）。

``fused`` / ``unconsumed`` 是「接线是否真的命中」的唯一产物级证据：m3_check 的
bitwise gate 只比较执行方式，patch 没生效时照样全绿（第 2 轮踩过）。

scratch
------------------------------------------------------------------------
install 时一次性分配：BF16 输出 ``[48,128]``（12 KB）+ ``a_fp4`` ``[1,3072]``
（3 KB）+ ``a_sf`` ``[128,384]`` 零初始化（48 KB），48 层复用同一份，靠 stream
顺序保证「先写、读、下一次覆盖」。地址在 capture 之前固定，graph replay 依赖这一点。
``a_fp4`` / ``a_sf`` 的 dtype/shape 由 FlashInfer 自己的 probe 输出决定（不猜）。

预期瓶颈：与 ``runner/ops/gdn/gated_norm.py`` 相同（单 kernel 固定开销）；本模块只做
接线，graph 路径在 capture 之后不再执行 Python。
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from runner.ops.gdn.gated_norm import (
    DEFAULT_NUM_WARPS,
    HEAD_V_DIM,
    NUM_V_HEADS,
    SF_ROWS,
    VALUE_DIM,
    allocate_scratch,
    gated_norm_fused,
)
from runner.ops.nvfp4.norm_quant import global_divisor

LOGGER = logging.getLogger(__name__)

#: 关掉融合路径（调试/对照用）：``GATED_NORM_FUSED=0``。
ENV_FLAG = "GATED_NORM_FUSED"
_DISABLED = {"0", "false", "off", "no"}

_SKIP_SHAPE = "shape"
_SKIP_DTYPE = "dtype"
_SKIP_DEVICE = "device"
_SKIP_GRAD = "grad"
_SKIP_WIRING = "wiring"
_SKIP_ERROR = "kernel-error"

_STATE: dict[str, Any] = {
    "installed": False,
    "enabled": False,
    "orig_forward": None,
    "scratch": None,
    "plans": {},
    "num_warps": DEFAULT_NUM_WARPS,
    "probe": {},
    "counters": {"fused": 0, "unconsumed": 0, "fallbacks": {}},
    "warned": set(),
}


# ------------------------------------------------------------------------- 工具


def _fallback(reason: str) -> None:
    fallbacks = _STATE["counters"]["fallbacks"]
    fallbacks[reason] = fallbacks.get(reason, 0) + 1
    _warn_once(reason, f"gated norm 融合回退（{reason}），该层继续走原 forward")


def _warn_once(reason: str, message: str) -> None:
    warned = _STATE["warned"]
    if reason not in warned:
        warned.add(reason)
        LOGGER.warning("%s", message)


class _Plan:
    """一个 GDN 层的融合状态：norm 权重 + out_proj + 它的 global divisor。"""

    __slots__ = ("out_proj", "weight", "eps", "gs", "name")

    def __init__(
        self,
        out_proj: "torch.nn.Module",
        weight: torch.Tensor,
        eps: float,
        gs: float,
        name: str,
    ) -> None:
        self.out_proj = out_proj
        self.weight = weight
        self.eps = eps
        self.gs = gs
        self.name = name


def _torch_dtype(name: str) -> torch.dtype:
    """``"torch.uint8"`` -> ``torch.uint8``；解析不出来就用 uint8（单字节布局不变）。"""
    value = getattr(torch, str(name).rsplit(".", 1)[-1], None)
    if isinstance(value, torch.dtype):
        return value
    return torch.uint8


def _probe_quantizer(device: torch.device) -> dict[str, Any]:
    """用 FlashInfer 自己的输出确定 ``a_fp4`` / ``a_sf`` 的 dtype 与 shape（不猜）。"""
    from flashinfer import SfLayout, nvfp4_quantize

    probe = torch.zeros(1, VALUE_DIM, dtype=torch.bfloat16, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    a_fp4, a_sf = nvfp4_quantize(
        probe,
        scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    if tuple(a_fp4.shape) != (1, VALUE_DIM // 2):
        raise RuntimeError(f"probe a_fp4 shape {tuple(a_fp4.shape)}")
    if tuple(a_sf.shape) != (SF_ROWS, VALUE_DIM // 16):
        raise RuntimeError(f"probe a_sf shape {tuple(a_sf.shape)}")
    return {
        "a_fp4_dtype": str(a_fp4.dtype),
        "a_sf_dtype": str(a_sf.dtype),
        "a_fp4_shape": list(a_fp4.shape),
        "a_sf_shape": list(a_sf.shape),
    }


# ------------------------------------------------------------------------- 接线


def _patched_forward(self, hidden_states: torch.Tensor, gate: torch.Tensor):
    """``Qwen3_5RMSNormGated.forward`` 的 decode 版：gated norm + out_proj quant 一次算完。"""
    original = _STATE["orig_forward"]
    plan = _STATE["plans"].get(id(self))
    reason = _skip_reason(hidden_states, gate, plan)
    if reason is not None:
        _fallback(reason)
        return original(self, hidden_states, gate)

    scratch = _STATE["scratch"]
    out = scratch["out"]
    a_fp4 = scratch["a_fp4"][0]
    a_sf = scratch["a_sf"][0]
    try:
        gated_norm_fused(
            hidden_states,
            gate,
            plan.weight,
            plan.eps,
            plan.gs,
            out,
            a_fp4,
            a_sf,
            num_warps=_STATE["num_warps"],
        )
    except Exception as exc:  # noqa: BLE001 - 融合失败回退原路径，不让 decode 崩
        _warn_once(_SKIP_ERROR, f"gated norm 融合 kernel 调用失败（{plan.name}）：{exc!r}")
        _fallback(_SKIP_ERROR)
        return original(self, hidden_states, gate)

    if plan.out_proj._pending_quantized is not None:
        # 上一层/上一步的槽位没被消费：计数并覆盖（正常路径不会发生）。
        _STATE["counters"]["unconsumed"] += 1
    plan.out_proj._pending_quantized = (a_fp4, a_sf)
    _STATE["counters"]["fused"] += 1
    return out


def _skip_reason(
    hidden_states: torch.Tensor, gate: torch.Tensor, plan: _Plan | None
) -> str | None:
    if not _STATE["enabled"] or plan is None:
        return _SKIP_WIRING
    if hidden_states.dtype is not torch.bfloat16 or gate.dtype is not torch.bfloat16:
        return _SKIP_DTYPE
    if hidden_states.device.type != "cuda":
        return _SKIP_DEVICE
    if torch.is_grad_enabled():
        return _SKIP_GRAD
    shape = (NUM_V_HEADS, HEAD_V_DIM)
    if (
        tuple(hidden_states.shape) != shape
        or tuple(gate.shape) != shape
        or not hidden_states.is_contiguous()
        or not gate.is_contiguous()
    ):
        return _SKIP_SHAPE
    return None


def install(
    model: torch.nn.Module,
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> dict[str, Any]:
    """把模型里 48 个 GDN 层的 gated norm 接到融合 kernel 上；重复安装是幂等的。"""
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    if _STATE["installed"]:
        uninstall()

    klass = getattr(modeling, "Qwen3_5GatedDeltaNet", None)
    if klass is None:
        raise RuntimeError("transformers 里找不到 Qwen3_5GatedDeltaNet")
    layers = [module for module in model.modules() if isinstance(module, klass)]
    if not layers:
        raise RuntimeError("模型里没有 Qwen3_5GatedDeltaNet 层")

    device = layers[0].norm.weight.device
    if device.type != "cuda":
        raise RuntimeError(f"gated norm 权重不在 CUDA 上: {device}")
    probe = _probe_quantizer(device)
    out_bf16, a_fp4, a_sf = allocate_scratch(
        device=device,
        a_fp4_dtype=_torch_dtype(probe["a_fp4_dtype"]),
        a_sf_dtype=_torch_dtype(probe["a_sf_dtype"]),
    )
    scratch = {
        "out": out_bf16,
        "a_fp4": [a_fp4],
        "a_sf": [a_sf],
        "bytes": sum(
            t.numel() * t.element_size() for t in (out_bf16, a_fp4, a_sf)
        ),
    }

    plans: dict[int, _Plan] = {}
    for index, layer in enumerate(layers):
        norm = layer.norm
        out_proj = layer.out_proj
        if not isinstance(norm, modeling.Qwen3_5RMSNormGated):
            raise RuntimeError(f"layers.{index}.norm 不是 Qwen3_5RMSNormGated")
        # 延迟 import：``runner.ops.nvfp4.linear`` 在模块级 import flashinfer，而 flashinfer
        # 的 import 会探测 CUDA capability。CPU-only 的接线自测必须能 import 本模块而不
        # 建 CUDA context（队列有 GPU 守卫），所以只在 install() 里 import。
        from runner.ops.nvfp4.linear import NVFP4Linear

        if not isinstance(out_proj, NVFP4Linear):
            raise RuntimeError(
                f"layers.{index}.out_proj 不是 NVFP4Linear: {type(out_proj).__name__}"
            )
        if int(out_proj.in_features) != VALUE_DIM:
            raise RuntimeError(
                f"layers.{index}.out_proj.in_features={out_proj.in_features} != {VALUE_DIM}"
            )
        if norm.weight.numel() != HEAD_V_DIM:
            raise RuntimeError(
                f"layers.{index}.norm.weight 是 {norm.weight.numel()}，期望 {HEAD_V_DIM}"
            )
        if norm.activation != "silu":
            raise RuntimeError(f"layers.{index}.norm.activation={norm.activation!r}")
        weight = norm.weight.detach().to(torch.bfloat16).contiguous()
        gs = global_divisor(float(out_proj.input_scale.reshape(1).float().item()))
        plans[id(norm)] = _Plan(out_proj, weight, float(norm.variance_epsilon), gs,
                                f"layers.{index}.linear_attn.norm")

    _STATE["orig_forward"] = modeling.Qwen3_5RMSNormGated.forward
    _STATE["plans"] = plans
    _STATE["scratch"] = scratch
    _STATE["num_warps"] = int(num_warps)
    _STATE["probe"] = probe
    _STATE["counters"] = {"fused": 0, "unconsumed": 0, "fallbacks": {}}
    _STATE["warned"] = set()
    _STATE["enabled"] = True
    modeling.Qwen3_5RMSNormGated.forward = _patched_forward
    _STATE["installed"] = True
    return info()


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    modeling.Qwen3_5RMSNormGated.forward = _STATE["orig_forward"]
    for plan in _STATE["plans"].values():
        plan.out_proj._pending_quantized = None
    _STATE["installed"] = False
    _STATE["enabled"] = False
    _STATE["plans"] = {}


def info() -> dict[str, Any]:
    scratch = _STATE["scratch"] or {}
    return {
        "enabled": _STATE["enabled"],
        "layers": len(_STATE["plans"]),
        "num_warps": _STATE["num_warps"],
        "probe": _STATE["probe"],
        "scratch_bytes": int(scratch.get("bytes", 0)),
        # info() 返回的是同一个 dict，产物在末尾序列化时能看到最终计数。
        "counters": _STATE["counters"],
    }


def install_from_env(model: torch.nn.Module) -> dict[str, Any]:
    """loader 用：默认打开，``GATED_NORM_FUSED=0`` 关闭；失败只记录不抛。"""
    flag = os.environ.get(ENV_FLAG, "1").strip().lower()
    if flag in _DISABLED:
        return {"enabled": False, "reason": f"{ENV_FLAG}={flag}"}
    try:
        result = install(model)
    except Exception as exc:  # noqa: BLE001 - 接入失败时回退原路径，不让非 decode 工具崩
        LOGGER.error("gated norm 融合接入失败（回退原路径）：%r", exc)
        return {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}
    LOGGER.warning("gated norm 融合已接入: %s", json.dumps(result, ensure_ascii=False))
    return result


__all__ = ["ENV_FLAG", "info", "install", "install_from_env", "uninstall"]
