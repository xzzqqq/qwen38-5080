"""NVFP4Linear 的字节模型。

口径：只算一次 forward 需要从显存读/写的大块数据；不含 kernel workspace 和
CUDA context。权重部分对应 checkpoint 的 data bytes；activation/output 按
[M, K] / [M, N] 计算。
"""

from __future__ import annotations


def weight_bytes(in_features: int, out_features: int) -> int:
    """常驻权重：U8 权重 + F8 block scale + 两个 F32 标量。"""
    return out_features * in_features // 2 + out_features * in_features // 16 + 4 + 4


def activation_bytes(m: int, in_features: int) -> int:
    """一次 forward 的 packed activation + block scale + global scale。"""
    return m * in_features // 2 + m * in_features // 16 + 4


def output_bytes(m: int, out_features: int) -> int:
    """BF16 输出。"""
    return m * out_features * 2


def forward_bytes(m: int, in_features: int, out_features: int) -> int:
    """权重 + activation + 输出的一次 forward 字节数。"""
    return (
        weight_bytes(in_features, out_features)
        + activation_bytes(m, in_features)
        + output_bytes(m, out_features)
    )
