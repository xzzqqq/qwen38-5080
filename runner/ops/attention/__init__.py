"""Phase 2 任务 A：BF16 decode（q_len==1）的紧凑 GQA attention。

对外只暴露三件事：
- ``compact_gqa_decode_attention``：直接读 ``[1,4,L,256]`` K/V 的分组实现；
- ``install`` / ``uninstall``：把 ``qwen3_5`` 的 eager attention 换成按 KV head
  分组的实现（q_len>1 自动回退），由 loader 在模型加载时调用；
- ``eager_gqa_decode_attention`` / ``reference_fp64``：同协议对照与 local gate 的
  baseline / float64 reference。

FP8 KV：cache 的 decode 返回 fp8 前缀，scale 挂在张量的 ``kv_fp8_scale`` 上；
``compact_decode`` 在 bmm 前反量化（固定 bf16 scratch）。融合 decode kernel 的
接口是 ``runner/ops/kv/fp8.py`` 的 ``(K fp8, K scale, V fp8, V scale)`` 与
``dequantize`` / ``dequant_kv``：可以直接消费 fp8 + scale，不必物化 bf16 前缀。

设计、数据流和预期瓶颈见 ``compact_decode.py`` 模块 docstring。
"""

from .compact_decode import (
    BACKEND,
    BACKEND_ENV,
    BACKENDS,
    compact_gqa_decode_attention,
    eager_gqa_decode_attention,
    install,
    is_installed,
    reference_fp64,
    repeat_kv,
    uninstall,
)

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
