"""紧凑 GQA decode attention（q_len==1）：不物化 repeat_kv 的 24 份 KV。

设计：
- 模型 decode 时 q_len==1、``num_key_value_groups = 24 / 4 = 6``；HF eager 的
  ``repeat_kv`` 先把 4 个 KV head 复制成 24 份（每层 K/V 各 100 MB @8192），
  再走 16 路 batched matmul。这里直接把 q 按 KV head 分组（第 j 个 KV head 对应
  q head ``[6j, 6j+6)``），在 4 个 KV head 上做 batched matmul，K/V 全程只读
  原始 buffer（可以是 static cache 的 capacity 前缀 view）。
- 数值路径与 HF eager 逐项对齐，只改变 reduction 的分组方式：
  ``bf16 matmul -> * scaling -> + additive mask -> softmax(fp32) -> bf16 -> @ V``。
  因此 local gate 用 float64 reference 定容差，而不是和旧 eager 输出比 bitwise。
- **FP8 KV**：cache 的 decode 返回 fp8 前缀，scale 挂在张量的 ``kv_fp8_scale``
  属性上（``runner/ops/kv/fp8.py`` 的 side channel）。这里在 bmm 前调用
  ``fp8.dequantize`` 写进固定 bf16 scratch，然后走完全相同的 bf16 bmm 路径；
  也就是“FP8 只改变 K/V 的存储与读取字节，不改变 attention 数学”。融合 kernel
  可以直接消费 ``(K fp8, K scale, V fp8, V scale)``，不必经过这一步 bf16 物化。

数据流（单层，batch=1）：
    q [1,24,1,256] --view--> [4,6,256]
    k [1,4,L,256]（可能是 capacity buffer 的前缀 view）
      -> [FP8 时] dequantize(k, k_scale) / dequantize(v, v_scale) -> bf16 scratch
      -> bmm(q_g, k^T) -> scores [4,6,L] bf16
      -> * scaling -> + mask[..., :L]
      -> softmax(dim=-1, dtype=float32) -> bf16
      -> bmm(probs, v) -> [4,6,256]
      -> view [1,24,1,256]（第 6j+i 行就是 q head 6j+i）
    attn@V 之后接模型的 ``* sigmoid(gate)`` 与 o_proj，不在本 op 内。

预期瓶颈：L 大时是 K/V 的读带宽（BF16 每层 ``2 * 4 * L * 256 * 2`` 字节；
FP8 读一半，但要把同样元素数的 bf16 写进 scratch，净流量取决于 scratch 是否
留在 L2）。L 小时是 kernel launch 与 softmax。
"""

from __future__ import annotations

import os
from typing import Any

import torch

from runner.ops.kv import fp8 as fp8_kv

# 候选选择：本方向先做分组 bmm；``eager`` 只作为同一实现方向的关闭开关，
# 用于在同一进程/同一协议下做对照，不是另一个候选。
BACKEND_ENV = "PHASE2_ATTN_DECODE"
BACKENDS = ("grouped_bmm", "eager")
BACKEND = os.environ.get(BACKEND_ENV, "grouped_bmm")


def _backend() -> str:
    backend = os.environ.get(BACKEND_ENV, BACKEND)
    if backend not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV}={backend!r} 必须是 {BACKENDS} 之一")
    return backend


def compact_gqa_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q_len==1 的紧凑 GQA attention。

    Args:
        query: ``[batch, num_q_heads, 1, head_dim]``，BF16/FP16/FP32。
        key: ``[batch, num_kv_heads, L, head_dim]``；允许是 capacity buffer 的
            非连续前缀 view（static cache 的真实形态）。FP8 KV 时是
            ``float8_e4m3fn``，且 ``key.kv_fp8_scale`` 是 ``[L, num_kv_heads]``
            的 fp32 scale（``value`` 同）。
        value: 同 key。
        attention_mask: 加性 mask，可广播到 ``[batch, num_kv_heads, 1, L]``；
            ``None`` 表示全部可见。bf16 的 ``finfo.min`` 掩码在 softmax(fp32)
            里 exp 后是精确 0，因此不可见 key 的取值不影响输出。
        scaling: 通常是 ``head_dim ** -0.5``。

    Returns:
        ``(attn_output [batch, num_q_heads, 1, head_dim]``,
        ``attn_weights [batch, num_q_heads, 1, L])``；weights 是 view，不复制。
    """
    # FP8 KV：先反量化到固定 bf16 scratch；之后的所有数学与 BF16 路径完全相同。
    k_scale = fp8_kv.kv_scale(key)
    v_scale = fp8_kv.kv_scale(value)
    if k_scale is not None or v_scale is not None:
        if k_scale is None or v_scale is None:
            raise ValueError("FP8 KV 必须同时提供 K 和 V 的 scale")
        key = fp8_kv.dequantize(
            key, k_scale, out=fp8_kv.dequant_scratch("k", key.shape, key.device)
        )
        value = fp8_kv.dequantize(
            value, v_scale, out=fp8_kv.dequant_scratch("v", value.shape, value.device)
        )
    batch, num_q_heads, q_len, head_dim = query.shape
    batch_k, num_kv_heads, seq_len, head_dim_k = key.shape
    if q_len != 1:
        raise ValueError(f"compact decode attention 只支持 q_len==1，得到 {q_len}")
    if batch != batch_k or key.shape != value.shape:
        raise ValueError(
            f"q/k/v shape 不匹配: q={tuple(query.shape)} k={tuple(key.shape)} "
            f"v={tuple(value.shape)}"
        )
    if head_dim != head_dim_k or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"head 配置不支持: q_heads={num_q_heads} kv_heads={num_kv_heads} "
            f"head_dim={head_dim}/{head_dim_k}"
        )
    groups = num_q_heads // num_kv_heads

    # [b, hkv, groups, 1, d] -> [b*hkv, groups, d]；head 顺序与 repeat_kv 一致。
    query_grouped = query.reshape(batch * num_kv_heads, groups, head_dim)
    key_flat = key.reshape(batch * num_kv_heads, seq_len, head_dim)
    value_flat = value.reshape(batch * num_kv_heads, seq_len, head_dim)

    scores = torch.bmm(query_grouped, key_flat.transpose(1, 2))
    # [b*hkv, groups, L] -> [b, hkv, groups, L]：mask 按 HF 的
    # [b, 1, 1, L] 形状广播，避免 3D + 4D 广播出多余的 leading dim。
    scores = scores.reshape(batch, num_kv_heads, groups, seq_len)
    scores = scores * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.bmm(probs.reshape(batch * num_kv_heads, groups, seq_len), value_flat)
    # 已经是 [b, num_q_heads, q_len, head_dim]（repeat_kv 的 head 顺序）。
    out = out.reshape(batch, num_q_heads, q_len, head_dim).contiguous()
    weights = probs.reshape(batch, num_q_heads, q_len, seq_len)
    return out, weights


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """HF eager 的 repeat_kv（baseline / reference 用，本模块 op 不调用）。"""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def eager_gqa_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """旧路径：repeat_kv 物化展开后 matmul。只用于同协议对照与 microbench。"""
    groups = query.shape[1] // key.shape[1]
    key_states = repeat_kv(key, groups)
    value_states = repeat_kv(value, groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    out = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return out, attn_weights


def reference_fp64(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """float64 reference：同样 q/k/v（转 fp64）、显式 mask、softmax 全程 fp64。"""
    groups = query.shape[1] // key.shape[1]
    q64 = query.double()
    k64 = repeat_kv(key.double(), groups)
    v64 = repeat_kv(value.double(), groups)
    weights = torch.matmul(q64, k64.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask.double()
    weights = torch.softmax(weights, dim=-1)
    return torch.matmul(weights, v64).transpose(1, 2).contiguous()


# --------------------------------------------------------------- interface patch
_ORIGINAL: Any = None
_INSTALLED: dict[str, Any] | None = None


def _attention_interface(original):
    """包一层 transformers 的 eager_attention_forward。

    只在“确实是单 token decode”时走紧凑路径；prefill（q_len>1）、
    ``num_key_value_groups==1`` 的视觉 attention、dropout>0、bool mask 或
    mask 长度不匹配时原样回退到原实现。回退条件都写在同一个函数里，避免
    模型里出现两条不可比较的路径。
    """

    def attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout: float = 0.0,
        **kwargs,
    ):
        groups = getattr(module, "num_key_value_groups", 1)
        mask_ok = attention_mask is None or (
            attention_mask.dtype != torch.bool and attention_mask.shape[-1] == key.shape[2]
        )
        # FP8 KV 时 key/value 是 float8_e4m3fn（scale 在 side channel 上），
        # dtype 检查要放行这一组；反量化由 compact_gqa_decode_attention 自己做。
        fp8_kv_ok = (
            fp8_kv.kv_scale(key) is not None
            and fp8_kv.kv_scale(value) is not None
            and key.dtype == value.dtype == fp8_kv.FP8_DTYPE
        )
        dtype_ok = (query.dtype == key.dtype == value.dtype) or fp8_kv_ok
        if (
            _backend() == "grouped_bmm"
            and not module.training
            and dropout == 0.0
            and groups > 1
            and query.shape[2] == 1
            and query.is_cuda
            and query.shape[0] == key.shape[0] == value.shape[0]
            and query.shape[1] == key.shape[1] * groups
            and key.shape[1] == value.shape[1]
            and key.shape[2] == value.shape[2]
            and dtype_ok
            and mask_ok
        ):
            return compact_gqa_decode_attention(query, key, value, attention_mask, scaling)
        return original(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling,
            dropout=dropout,
            **kwargs,
        )

    attention_forward.__name__ = "phase2_compact_decode_attention_forward"
    attention_forward.__doc__ = "q_len==1 走紧凑 GQA decode；其它情况回退 HF eager。"
    return attention_forward


def install() -> dict[str, Any]:
    """给 ``transformers.models.qwen3_5.modeling_qwen3_5`` 打 attention 补丁。

    为什么打这里：``Qwen3_5Attention.forward`` 在调用时从模块全局取
    ``eager_attention_forward`` 作为 ``get_interface`` 的 default，而本项目
    ``_attn_implementation == "eager"``，所以替换这个全局即可覆盖所有执行方式
    （eager / static-exact / static-cap / graph），且 prefill 自动回退。比复制
    整个 ``Qwen3_5Attention.forward`` 的侵入小，也不会改到 vision tower。

    幂等：重复调用返回同一份状态。
    """
    global _ORIGINAL, _INSTALLED
    if _INSTALLED is not None:
        return dict(_INSTALLED)
    from transformers.models.qwen3_5 import modeling_qwen3_5

    _ORIGINAL = modeling_qwen3_5.eager_attention_forward
    wrapper = _attention_interface(_ORIGINAL)
    modeling_qwen3_5.eager_attention_forward = wrapper
    _INSTALLED = {
        "backend": _backend(),
        "module": "transformers.models.qwen3_5.modeling_qwen3_5",
        "symbol": "eager_attention_forward",
        "q_len": "==1 (prefill q_len>1 falls back to original eager)",
        "kv_formats": ["bfloat16", "float8_e4m3fn"],
        "fp8_kv": "dequantize(k, k_scale) -> bf16 scratch before bmm",
        "grouping": "kv-head bmm, head 6j+i -> kv head j",
        "env": BACKEND_ENV,
    }
    return dict(_INSTALLED)


def uninstall() -> None:
    """恢复原 eager 实现（只用于诊断/测试）。"""
    global _ORIGINAL, _INSTALLED
    if _INSTALLED is None:
        return
    from transformers.models.qwen3_5 import modeling_qwen3_5

    modeling_qwen3_5.eager_attention_forward = _ORIGINAL
    _ORIGINAL = None
    _INSTALLED = None


def is_installed() -> bool:
    return _INSTALLED is not None


__all__ = [
    "BACKEND",
    "BACKEND_ENV",
    "BACKENDS",
    "compact_gqa_decode_attention",
    "eager_gqa_decode_attention",
    "install",
    "is_installed",
    "reference_fp64",
    "repeat_kv",
    "uninstall",
]
