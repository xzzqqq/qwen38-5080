"""ShapeStableLinear 的 full vs split 回归测试。"""

from __future__ import annotations

import pytest
import torch

from runner.ops.bf16.stable_linear import ShapeStableLinear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="shape-stable test needs a CUDA GPU"
)


def test_shape_stable_full_vs_split():
    torch.manual_seed(0)
    weight = torch.randn(48, 5120, device="cuda", dtype=torch.bfloat16) * 0.02
    bias = torch.randn(48, device="cuda", dtype=torch.bfloat16) * 0.01
    layer = ShapeStableLinear(weight, bias, block_m=128)
    x = torch.randn(300, 5120, device="cuda", dtype=torch.bfloat16) * 0.5

    full = layer(x)
    split = torch.cat(
        [layer(x[:128]), layer(x[128:256]), layer(x[256:])], dim=0
    )
    assert torch.equal(full, split), (
        (full.float() - split.float()).abs().max().item()
    )


def test_shape_stable_3d_matches_2d():
    torch.manual_seed(1)
    weight = torch.randn(48, 5120, device="cuda", dtype=torch.bfloat16) * 0.02
    layer = ShapeStableLinear(weight, None, block_m=128)
    x = torch.randn(2, 150, 5120, device="cuda", dtype=torch.bfloat16) * 0.5
    flat = layer(x.reshape(-1, 5120)).reshape(2, 150, 48)
    assert torch.equal(layer(x), flat)
