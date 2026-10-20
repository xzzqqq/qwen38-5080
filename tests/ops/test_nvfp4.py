"""NVFP4 op 对拍：layout、activation quant、NVFP4Linear 和字节模型。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from runner.ops.nvfp4.bytes import (
    activation_bytes,
    forward_bytes,
    output_bytes,
    weight_bytes,
)
from runner.ops.nvfp4.linear import NVFP4Linear
from runner.ops.nvfp4.reference import (
    _round_to_e2m1,
    dequant_nvfp4,
    e2m1_to_float,
    error_metrics,
    pack_e2m1,
    quantize_activation_reference,
    swizzle_scale_128x4,
    unpack_e2m1,
)

CHECKPOINT = Path("models/Qwen3.8-27B-NVFP4-RTX5090/model-00001-of-00002.safetensors")
PREFIX = "model.language_model.layers.0.linear_attn.in_proj_qkv."

# BF16 输出下的固定容差；基于 dequant reference 与 FlashInfer 的误差量级。
TOL = {"nrmse": 5.0e-3, "cosine": 0.99999, "max_rel": 1.0e-2}
GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVFP4 op tests need a CUDA GPU"
)


@pytest.fixture(scope="module")
def real_tensors():
    if not CHECKPOINT.exists():
        pytest.skip(f"checkpoint not found: {CHECKPOINT}")
    with safe_open(CHECKPOINT, framework="pt", device="cpu") as f:
        return {
            "weight": f.get_tensor(PREFIX + "weight"),
            "weight_scale": f.get_tensor(PREFIX + "weight_scale"),
            "weight_scale_2": f.get_tensor(PREFIX + "weight_scale_2"),
            "input_scale": f.get_tensor(PREFIX + "input_scale"),
        }


def test_unpack_nibble_order():
    packed = torch.tensor([[0x10, 0x32]], dtype=torch.uint8)
    codes = unpack_e2m1(packed)
    assert codes.tolist() == [[0, 1, 2, 3]]
    # 低 nibble = 偶数下标，高 nibble = 奇数下标
    assert pack_e2m1(codes).tolist() == packed.tolist()


def test_e2m1_boundary_rounding():
    values = torch.tensor(
        [
            0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            5.99,
            6.0,
            100.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5.0,
            -6.0,
            -100.0,
        ]
    )
    expected_codes = [
        0,
        0,
        2,
        2,
        4,
        4,
        6,
        6,
        7,
        7,
        7,
        8,
        10,
        10,
        12,
        12,
        14,
        14,
        15,
        15,
    ]
    codes = _round_to_e2m1(values)
    assert codes.tolist() == expected_codes
    expected_values = [0, 0, 1, 1, 2, 2, 4, 4, 6, 6, 6, -0, -1, -1, -2, -2, -4, -4, -6, -6]
    assert e2m1_to_float(codes).tolist() == expected_values


def test_dequant_known_values():
    codes = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
    packed = pack_e2m1(codes)
    scale = torch.ones(1, 1, dtype=torch.float8_e4m3fn)
    deq = dequant_nvfp4(packed, scale, 2.0)
    assert torch.allclose(deq, e2m1_to_float(codes) * 2.0)


@GPU
def test_swizzle_padding():
    from flashinfer import block_scale_interleave

    scale = torch.arange(15, dtype=torch.uint8, device="cuda").reshape(3, 5)
    ours = swizzle_scale_128x4(scale)
    ref = block_scale_interleave(scale)
    ref = ref.reshape(-1)[: ours.numel()].view(ours.shape)
    assert ours.shape == (128, 8)
    assert torch.equal(ours, ref)
    # padding 不应引入非零值；输入 arange(15) 只有 14 个非零。
    assert torch.count_nonzero(ours) == torch.count_nonzero(scale) == 14


def test_bytes_model():
    assert weight_bytes(5120, 10240) == 10240 * 5120 // 2 + 10240 * 5120 // 16 + 8
    assert activation_bytes(8, 5120) == 8 * 5120 // 2 + 8 * 5120 // 16 + 4
    assert output_bytes(8, 10240) == 8 * 10240 * 2
    assert forward_bytes(8, 5120, 10240) == (
        weight_bytes(5120, 10240) + activation_bytes(8, 5120) + output_bytes(8, 10240)
    )


@GPU
def test_swizzle_matches_flashinfer(real_tensors):
    from flashinfer import block_scale_interleave

    scale = real_tensors["weight_scale"].cuda()
    ours = swizzle_scale_128x4(scale)
    ref = block_scale_interleave(scale.view(torch.uint8))
    ref = ref.reshape(-1)[: ours.numel()].view(ours.shape)
    assert torch.equal(ours.view(torch.uint8), ref)


@GPU
def test_activation_quant_matches_flashinfer(real_tensors):
    from flashinfer import SfLayout, e2m1_and_ufp8sf_scale_to_float, nvfp4_quantize

    input_scale = real_tensors["input_scale"].cuda()
    x = torch.randn(8, 5120, device="cuda", dtype=torch.bfloat16) * 0.5
    packed, scale, deq = quantize_activation_reference(x, input_scale)

    a_fp4, a_sf = nvfp4_quantize(
        x,
        (1.0 / input_scale).reshape(1),
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    a_u8 = a_fp4.view(torch.uint8) if a_fp4.dtype != torch.uint8 else a_fp4
    ref_deq = e2m1_and_ufp8sf_scale_to_float(
        a_fp4,
        a_sf,
        input_scale.reshape(1),
        sf_vec_size=16,
        ufp8_type=1,
        is_sf_swizzled_layout=True,
    ).cuda()

    assert torch.equal(unpack_e2m1(packed), unpack_e2m1(a_u8))
    assert torch.equal(
        swizzle_scale_128x4(scale).view(torch.uint8),
        a_sf.view(torch.uint8),
    )
    assert (deq - ref_deq).abs().max().item() <= 1.0e-6


@GPU
@pytest.mark.parametrize("m", [1, 8, 128, 3, 17, 129])
def test_nvfp4_linear_matches_dequant_reference(real_tensors, m):
    w = real_tensors["weight"].cuda()
    wsf = real_tensors["weight_scale"].cuda()
    w2 = real_tensors["weight_scale_2"].cuda()
    inp = real_tensors["input_scale"].cuda()

    module = NVFP4Linear(
        real_tensors["weight"],
        real_tensors["weight_scale"],
        real_tensors["weight_scale_2"],
        real_tensors["input_scale"],
        backend="cutlass",
    ).cuda()

    x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16) * 0.5
    _, _, a_deq = quantize_activation_reference(x, inp)
    w_deq = dequant_nvfp4(w, wsf, w2)
    ref = a_deq.float() @ w_deq.T
    out = module(x)

    metrics = error_metrics(out, ref)
    assert metrics["finite"]
    assert metrics["nrmse"] <= TOL["nrmse"], metrics
    assert metrics["cosine"] >= TOL["cosine"], metrics
    assert metrics["max_rel"] <= TOL["max_rel"], metrics
