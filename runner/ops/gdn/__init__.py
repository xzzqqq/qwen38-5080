"""GDN（gated delta net）decode 的融合算子。"""

from runner.ops.gdn.conv_decode import (
    BLOCK_CONFIGS as CONV_BLOCK_CONFIGS,
    DEFAULT_BLOCK as CONV_DEFAULT_BLOCK,
    DEFAULT_NUM_WARPS as CONV_DEFAULT_NUM_WARPS,
    gdn_conv_decode_step,
)
from runner.ops.gdn.decode import (
    BLOCK_V_CONFIGS,
    DEFAULT_BLOCK_V,
    DEFAULT_NUM_WARPS,
    gdn_decode_step,
)
from runner.ops.gdn.gated_norm import (
    HEAD_V_DIM as GATED_NORM_HEAD_DIM,
    NUM_V_HEADS as GATED_NORM_HEADS,
    VALUE_DIM as GATED_NORM_VALUE_DIM,
    allocate_scratch as gated_norm_allocate_scratch,
    gated_norm_fused,
)

__all__ = [
    "BLOCK_V_CONFIGS",
    "CONV_BLOCK_CONFIGS",
    "CONV_DEFAULT_BLOCK",
    "CONV_DEFAULT_NUM_WARPS",
    "DEFAULT_BLOCK_V",
    "DEFAULT_NUM_WARPS",
    "GATED_NORM_HEAD_DIM",
    "GATED_NORM_HEADS",
    "GATED_NORM_VALUE_DIM",
    "gated_norm_allocate_scratch",
    "gated_norm_fused",
    "gdn_conv_decode_step",
    "gdn_decode_step",
]
