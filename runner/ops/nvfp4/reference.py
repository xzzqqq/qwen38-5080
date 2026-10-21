"""NVFP4 的独立 PyTorch reference。

设计：
- 只做 unpack / dequant / activation quant / 误差统计，不调用 FlashInfer kernel；
- E2M1 用查表，低 nibble 是偶数下标、高 nibble 是奇数下标（与 FlashInfer CPU fallback 一致）；
- checkpoint 语义：``w = e2m1 * weight_scale(F8) * weight_scale_2(F32)``；
  activation 的 ``input_scale`` 是反量化乘数，量化时用 ``1 / input_scale`` 作为 global divisor；
- 这个文件是“慢但直白”的正确性基准，性能路径在 ``linear.py``。

数据流：
    packed U8 [.., K/2] -> codes -> float [.., K]
    F8 scale [.., K/16] -> repeat 16 -> float [.., K]
    与 global scale 相乘 -> dequantized float32

预期瓶颈：只用于测试和少量 shape 对拍，不追求速度。
"""

from __future__ import annotations

import torch

# E2M1 的 16 个码字：bit3 是符号，低 3 位是 0/0.5/1/1.5/2/3/4/6。
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)
E2M1_MAG = E2M1_VALUES[:8]
E2M1_BOUNDARIES = (E2M1_MAG[:-1] + E2M1_MAG[1:]) / 2


def _lut(device: torch.device) -> torch.Tensor:
    return E2M1_VALUES.to(device)


def _boundaries(device: torch.device) -> torch.Tensor:
    return E2M1_BOUNDARIES.to(device)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """把 U8 [.., K/2] 解成 4-bit code [.., K]，低 nibble 在前。"""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must be uint8, got {packed.dtype}")
    out = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = (packed >> 4) & 0x0F
    return out


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """把 4-bit code [.., K] 打包成 U8 [.., K/2]，低 nibble 在前。"""
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"K must be even, got {codes.shape[-1]}")
    lo = codes[..., 0::2] & 0x0F
    hi = (codes[..., 1::2] & 0x0F) << 4
    return (lo | hi).to(torch.uint8)


def e2m1_to_float(codes: torch.Tensor) -> torch.Tensor:
    """4-bit code -> float32，查表。"""
    return _lut(codes.device)[codes.long()]


def dequant_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | float,
) -> torch.Tensor:
    """反量化 packed NVFP4。

    packed: U8 [.., K/2]
    scale:  F8_E4M3 [.., K/16]（线性布局）
    global_scale: 标量或 [1]
    """
    if packed.shape[-1] * 2 != scale.shape[-1] * 16:
        raise ValueError(
            f"shape mismatch: packed {tuple(packed.shape)} vs scale {tuple(scale.shape)}"
        )
    codes = unpack_e2m1(packed)
    values = e2m1_to_float(codes)
    expanded = scale.float().repeat_interleave(16, dim=-1)
    gs = global_scale.float() if isinstance(global_scale, torch.Tensor) else float(global_scale)
    return values * expanded * gs


def _round_to_e2m1(values: torch.Tensor) -> torch.Tensor:
    """按 E2M1 的 16 个码字做最近舍入，饱和到 ±6。

    对恰好落在中点的情况选择偶数 code，接近硬件 cvt.rn.satfinite.e2m1x2.f32。
    """
    mag = values.abs()
    boundaries = _boundaries(values.device)
    lo = torch.bucketize(mag, boundaries).clamp(0, 7)
    hi = (lo + 1).clamp(0, 7)
    mag_lut = E2M1_MAG.to(values.device)
    lo_val = mag_lut[lo]
    hi_val = mag_lut[hi]
    d_lo = (mag - lo_val).abs()
    d_hi = (mag - hi_val).abs()
    even = torch.where((lo % 2) == 0, lo, hi)
    code = torch.where(d_hi < d_lo, hi, torch.where(d_lo < d_hi, lo, even))
    code = torch.where(mag >= 6.0, torch.full_like(code, 7), code)
    sign = torch.where(values < 0, torch.full_like(code, 8), torch.zeros_like(code))
    return (code | sign).to(torch.uint8)


def quantize_activation_reference(
    x: torch.Tensor,
    input_scale: torch.Tensor | float,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """参考 activation NVFP4 量化。

    返回 (packed U8 [.., K/2], scale F8 [.., K/16], dequant float32 [.., K])。
    ``input_scale`` 是 checkpoint 里的反量化乘数；量化时用 ``1 / input_scale``。
    """
    if block_size != 16:
        raise ValueError("NVFP4 reference only supports block_size=16")
    if x.shape[-1] % block_size != 0:
        raise ValueError(f"K must be divisible by {block_size}, got {x.shape[-1]}")

    scale_t = (
        input_scale.float()
        if isinstance(input_scale, torch.Tensor)
        else torch.tensor(float(input_scale), device=x.device, dtype=torch.float32)
    )
    if torch.any(scale_t == 0):
        raise ValueError("input_scale must be non-zero")
    global_divisor = 1.0 / scale_t
    x_scaled = x.float() * global_divisor

    blocks = x_scaled.reshape(*x.shape[:-1], -1, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    sf = (amax / 6.0).to(torch.float8_e4m3fn).float()
    sf_safe = torch.where(sf > 0, sf, torch.ones_like(sf))
    codes = _round_to_e2m1(blocks / sf_safe)
    packed = pack_e2m1(codes).reshape(*x.shape[:-1], x.shape[-1] // 2)
    dequant = (e2m1_to_float(codes) * sf_safe * scale_t).reshape_as(x)
    return packed, sf.squeeze(-1).to(torch.float8_e4m3fn), dequant


def swizzle_scale_128x4(scale: torch.Tensor) -> torch.Tensor:
    """把线性 [rows, cols] block scale 转成 128x4 swizzled 布局。

    纯 PyTorch 版本，等价于 FlashInfer 的 block_scale_interleave；
    rows 补到 128 的倍数，cols 补到 4 的倍数，padding 为 0。
    """
    if scale.ndim != 2:
        raise ValueError(f"scale must be 2D, got {scale.shape}")
    rows, cols = scale.shape
    rows_p = ((rows + 127) // 128) * 128
    cols_p = ((cols + 3) // 4) * 4
    padded = torch.zeros(
        (1, rows_p, cols_p), dtype=scale.dtype, device=scale.device
    )
    padded[0, :rows, :cols] = scale
    swizzled = padded.reshape(1, rows_p // 128, 4, 32, cols_p // 4, 4)
    swizzled = swizzled.permute(0, 1, 4, 3, 2, 5).contiguous()
    return swizzled.reshape(rows_p, cols_p)


def unswizzle_scale_128x4(
    swizzled: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    """``swizzle_scale_128x4`` 的逆变换，裁回原始 [rows, cols]。"""
    rows_p, cols_p = swizzled.shape
    if rows_p % 128 != 0 or cols_p % 4 != 0:
        raise ValueError(f"bad swizzled shape: {tuple(swizzled.shape)}")
    if rows > rows_p or cols > cols_p:
        raise ValueError("requested shape exceeds padded swizzled shape")
    view = swizzled.reshape(1, rows_p // 128, cols_p // 4, 32, 4, 4)
    restored = view.permute(0, 1, 4, 3, 2, 5).reshape(rows_p, cols_p)
    return restored[:rows, :cols].contiguous()


def error_metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """BF16 输出与 float32 reference 的误差指标。"""
    o = out.float()
    r = ref.float()
    diff = o - r
    denom = r.norm().clamp_min(1e-12)
    ref_absmax = r.abs().max().clamp_min(1e-12)
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "nrmse": (diff.norm() / denom).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            o.flatten(), r.flatten(), dim=0
        ).item(),
        "max_rel": (diff.abs().max() / ref_absmax).item(),
        "finite": bool(torch.isfinite(o).all().item()),
    }
