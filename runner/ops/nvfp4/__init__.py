"""NVFP4 W4A4 op：独立 reference 和 FlashInfer ``mm_fp4`` 实现。"""

from .reference import (
    dequant_nvfp4,
    error_metrics,
    pack_e2m1,
    quantize_activation_reference,
    swizzle_scale_128x4,
    unpack_e2m1,
)

__all__ = [
    "NVFP4Linear",
    "dequant_nvfp4",
    "error_metrics",
    "pack_e2m1",
    "quantize_activation_reference",
    "swizzle_scale_128x4",
    "unpack_e2m1",
]


def __getattr__(name: str):
    # 延迟 import：只用 reference 的 CPU 测试不需要拉起 FlashInfer/JIT。
    if name == "NVFP4Linear":
        from .linear import NVFP4Linear

        return NVFP4Linear
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
