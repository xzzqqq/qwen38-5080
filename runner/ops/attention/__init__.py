"""Phase 2 任务 A：BF16 decode（q_len==1）的紧凑 GQA attention。

对外只暴露三件事：
- ``compact_gqa_decode_attention``：直接读 ``[1,4,L,256]`` K/V 的分组实现；
- ``install`` / ``uninstall``：把 ``qwen3_5`` 的 eager attention 换成按 KV head
  分组的实现（q_len>1 自动回退），由 loader 在模型加载时调用；
- ``eager_gqa_decode_attention`` / ``reference_fp64``：同协议对照与 local gate 的
  baseline / float64 reference。

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
