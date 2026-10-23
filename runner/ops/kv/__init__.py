"""Phase 2 任务 J：KV 格式（FP8）的编码契约、参考实现与反量化热路径。

对外接口（供 ``static_cache`` / ``compact_decode`` / 融合 decode kernel 共用）：
- ``quantize_kv`` / ``quantize_with_scale``：量化，返回 fp8 code 与 ``[T, H]`` scale；
- ``reference_quantize_kv``：同一契约的独立 reference（op gate 用）；
- ``dequantize`` / ``dequant_kv``：反量化到 bf16，可写进固定 scratch；
- ``dequantize_exact``：fp64 精确参考（op gate 用）；
- ``attach_scale`` / ``kv_scale``：把 scale 作为 side channel 挂在 cache 返回的
  fp8 张量上，``compact_decode`` 和融合 kernel 从这里取 ``[L, H]`` 的 fp32 scale。

设计、契约和字节账见 ``fp8.py`` 模块 docstring。
"""

from .fp8 import (
    DEFAULT_FORMAT,
    DEQUANT_ENV,
    FORMAT_ENV,
    FORMATS,
    FP8_DTYPE,
    FP8_MAX,
    SCALE_ATTR,
    SCALE_DTYPE,
    attach_scale,
    dequant_info,
    dequant_kv,
    dequant_scratch,
    dequantize,
    dequantize_exact,
    kv_format,
    kv_scale,
    quantize_kv,
    quantize_with_scale,
    reference_dequantize,
    reference_quantize_kv,
)

__all__ = [
    "DEFAULT_FORMAT",
    "DEQUANT_ENV",
    "FORMAT_ENV",
    "FORMATS",
    "FP8_DTYPE",
    "FP8_MAX",
    "SCALE_ATTR",
    "SCALE_DTYPE",
    "attach_scale",
    "dequant_info",
    "dequant_kv",
    "dequant_scratch",
    "dequantize",
    "dequantize_exact",
    "kv_format",
    "kv_scale",
    "quantize_kv",
    "quantize_with_scale",
    "reference_dequantize",
    "reference_quantize_kv",
]
