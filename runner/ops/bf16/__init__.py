"""BF16 小算子：目前只有 shape-stable Linear。"""

from .stable_linear import ShapeStableLinear

__all__ = ["ShapeStableLinear"]
