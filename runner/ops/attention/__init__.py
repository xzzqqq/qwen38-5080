"""Phase 2 任务 A/K：decode（q_len==1）GQA attention 的紧凑 + 融合实现。

对外暴露：
- ``fused_gqa_decode_attention``：任务 K 的 split-K + online softmax 单 Triton
  kernel（``decode_fused.py``，支持 BF16 KV 与 per-token scale 的 FP8 KV）；
- ``compact_gqa_decode_attention``：任务 A 的分组 bmm 实现（同协议对照 / 回退）；
- ``decode_gqa_attention``：按 ``PHASE2_ATTN_DECODE=fused|grouped_bmm|eager``
  选择（默认 fused，前置条件不满足时回退 grouped_bmm）；
- ``install`` / ``uninstall``：把 ``qwen3_5`` 的 eager attention 换成上述实现
  （q_len>1 自动回退），由 loader 在模型加载时调用；
- ``eager_gqa_decode_attention`` / ``reference_fp64``：同协议对照与 local gate 的
  baseline / float64 reference。

FP8 KV：cache 的 decode 返回 fp8 前缀，``[L, kv_heads]`` 的 fp32 scale 挂在张量的
``kv_fp8_scale`` 属性上（``runner/ops/kv/fp8.py`` 的 side channel）；``fused``
后端把 scale 直接折进 kernel（零 bf16 物化），``grouped_bmm`` 回退和 prefill 才用
``dequantize`` / ``dequant_kv``。

设计、数据流和预期瓶颈见 ``decode_fused.py`` / ``compact_decode.py`` 的模块 docstring。
"""

from .compact_decode import (
    BACKEND,
    BACKEND_ENV,
    BACKENDS,
    compact_gqa_decode_attention,
    decode_backend_info,
    decode_gqa_attention,
    eager_gqa_decode_attention,
    install,
    is_installed,
    reference_fp64,
    repeat_kv,
    uninstall,
)
from .decode_fused import (
    CONFIGS as FUSED_CONFIGS,
    DEFAULT_CONFIG as FUSED_DEFAULT_CONFIG,
    can_fuse,
    fused_gqa_decode_attention,
    kernel_plan,
)

__all__ = [
    "BACKEND",
    "BACKEND_ENV",
    "BACKENDS",
    "FUSED_CONFIGS",
    "FUSED_DEFAULT_CONFIG",
    "can_fuse",
    "compact_gqa_decode_attention",
    "decode_backend_info",
    "decode_gqa_attention",
    "eager_gqa_decode_attention",
    "fused_gqa_decode_attention",
    "install",
    "is_installed",
    "kernel_plan",
    "reference_fp64",
    "repeat_kv",
    "uninstall",
]
