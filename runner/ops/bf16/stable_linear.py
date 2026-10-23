"""BF16 Linear 的 shape-stable 包装。

问题：同一个 `nn.Linear` 在 M=300 和 M=128 下，PyTorch 可能选择不同 GEMM
kernel/reduction 顺序，导致 BF16 输出有 1e-2 量级差异。`in_proj_a/b` 的输出
决定 GDN 的 decay / update gate，这种差异会被后续 W4A4 quantization 放大。

做法：固定按 `block_m` 行分块，尾块补零后做同一 shape 的 F.linear，再截掉
padding。这样同一行 token 无论在单段还是分段 prefill 中，都走相同 M 的 GEMM；
a/b 很小，padding 计算量可忽略。

decode 专用分支（`M == 1`）：padding 到 M=128 的 GEMM 对单 token 是纯浪费
（8K graph profile 里 96 次/步、每次约 5 µs 的 `wmma_bf16_32x32_128x2`，外加
补零 fill + `torch.cat` 各一次）。`M == 1` 时改走不 padding 的 BF16 matvec：
每个输出行一个 program，K 方向按 `BLOCK_K` 分块、每块一个 FP32 向量累加器，
最后一次树归约。单 token 前向不参与 chunked/full prefill 的逐位对齐口径
（口径只要求同一 M 的调用逐位一致），所以这里换 reduction 顺序是允许的；
M>1 的路径完全不动。

数据流（M>1，原样）：x [..., K] -> reshape [M, K] -> 每 block_m 行 pad ->
F.linear -> 截断 -> [..., N]
数据流（M == 1）：x [..., K] -> reshape [1, K] -> matvec kernel -> [..., N]

预期瓶颈（M == 1）：权重只有 N*K*2 字节（a/b 各 480 KB），kernel 是
latency-bound；一个输出行一个 CTA → grid=48，K 方向若干次 1–2 KB 向量加载。
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn.functional as F
from torch import nn

#: 关掉 M=1 decode 分支（调试/对照用）：``SHAPE_STABLE_DECODE=0``。
ENV_FLAG = "SHAPE_STABLE_DECODE"
_DISABLED = {"0", "false", "off", "no"}

# M=1 matvec 的粗粒度参数。op 脚本扫过 3 组（block_k=512/1024/2048，warps=4）：
# 真实权重 + graph replay 下 512/4 = 1.62 µs、1024/4 = 1.86 µs、2048/4 = 1.84 µs
# 每调用（job-000031 的 op.json），所以默认取 512/4。
DECODE_BLOCK_K = 512
DECODE_NUM_WARPS = 4
# 这个分支是给 a/b 这种窄输出用的；N 很大时走原 GEMM 更合理。
DECODE_MAX_N = 1024

try:  # triton 是项目硬依赖（runner/ops/gdn 也用它），但 import 失败时安全回退。
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception as _exc:  # noqa: BLE001 - 缺 triton 时回退原路径
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAVE_TRITON = False
    _IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"
else:
    _IMPORT_ERROR = None

_STATS: dict[str, object] = {
    # Python 侧活计数：graph capture 期间的调用会记进来，replay 不会（不再执行
    # Python）。所以这些数字证明「接线是否真的命中」，不是性能计数。
    "decode_calls": 0,
    "decode_fallbacks": 0,
    "padded_calls": 0,
    "fallback_reasons": {},
    "disabled": False,
    "kernel_error": None,
    "kernel_broken": False,
}

if _HAVE_TRITON:

    @triton.jit
    def _bf16_matvec_kernel(
        x_ptr,
        w_ptr,
        b_ptr,
        y_ptr,
        k,
        HAS_BIAS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """y[n] = sum_k x[k] * w[n, k] (+ bias[n])，一个 program 算一行。

        每个 program 只读 x 的一份切片和 w 的一行；FP32 向量累加器 + 最后一次
        树归约。没有原子操作、reduction 顺序固定，所以同一输入逐位可复现，
        CUDA graph 捕获/重放安全。
        """
        n = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        w_row = w_ptr + n * k
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k0 in range(0, k, BLOCK_K):
            idx = k0 + offs
            mask = idx < k
            xv = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            wv = tl.load(w_row + idx, mask=mask, other=0.0).to(tl.float32)
            acc += xv * wv
        value = tl.sum(acc, axis=0)
        if HAS_BIAS:
            value += tl.load(b_ptr + n).to(tl.float32)
        tl.store(y_ptr + n, value.to(tl.bfloat16))


_DISABLED_SNAPSHOT: bool | None = None


def _decode_enabled() -> bool:
    """env kill switch，默认打开；进程内只解析一次。"""
    global _DISABLED_SNAPSHOT
    if _DISABLED_SNAPSHOT is None:
        flag = os.environ.get(ENV_FLAG, "1").strip().lower()
        _DISABLED_SNAPSHOT = flag in _DISABLED
    return not _DISABLED_SNAPSHOT


def decode_stats() -> dict:
    """decode 分支的活计数与回退原因（JSON 友好）。"""
    return {
        **_STATS,
        "fallback_reasons": dict(_STATS["fallback_reasons"]),  # type: ignore[arg-type]
        "kernel": f"triton bf16 matvec, block_k={DECODE_BLOCK_K}, "
        f"num_warps={DECODE_NUM_WARPS}",
        "triton": _HAVE_TRITON,
        "import_error": _IMPORT_ERROR,
    }


def reset_decode_stats() -> None:
    _STATS["decode_calls"] = 0
    _STATS["decode_fallbacks"] = 0
    _STATS["padded_calls"] = 0
    _STATS["fallback_reasons"] = {}


def _fallback(reason: str) -> None:
    _STATS["decode_fallbacks"] = int(_STATS["decode_fallbacks"]) + 1
    reasons = _STATS["fallback_reasons"]
    assert isinstance(reasons, dict)
    reasons[reason] = reasons.get(reason, 0) + 1


def _decode_matvec(
    flat: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor | None:
    """M=1 的 BF16 matvec；不满足条件时返回 ``None``（调用方走原 GEMM 路径）。"""
    if not _decode_enabled():
        _STATS["disabled"] = True
        _fallback("env-disabled")
        return None
    if not _HAVE_TRITON:
        _fallback("no-triton")
        return None
    if _STATS["kernel_broken"]:
        _fallback("kernel-error")
        return None
    if flat.device.type != "cuda":
        _fallback(f"device:{flat.device.type}")
        return None
    if weight.device != flat.device:
        _fallback("device-mismatch")
        return None
    if flat.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        _fallback(f"dtype:{flat.dtype}/{weight.dtype}")
        return None
    if flat.shape[1] != weight.shape[1] or flat.shape[0] != 1:
        _fallback("shape")
        return None
    if weight.shape[0] > DECODE_MAX_N:
        _fallback(f"n>{DECODE_MAX_N}")
        return None
    if not flat.is_contiguous() or not weight.is_contiguous():
        _fallback("non-contiguous")
        return None
    if bias is not None and (
        bias.dtype != torch.bfloat16
        or bias.shape != weight.shape[:1]
        or bias.device != weight.device
    ):
        _fallback(f"bias:{bias.dtype}")
        return None
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])
    out = torch.empty((1, out_features), device=flat.device, dtype=torch.bfloat16)
    try:
        _bf16_matvec_kernel[(out_features,)](
            flat,
            weight,
            bias if bias is not None else weight,
            out,
            in_features,
            HAS_BIAS=bias is not None,
            BLOCK_K=DECODE_BLOCK_K,
            num_warps=DECODE_NUM_WARPS,
        )
    except Exception as exc:  # noqa: BLE001 - kernel 不可用时回退原 GEMM（只警告一次）
        _STATS["kernel_error"] = f"{type(exc).__name__}: {exc}"
        _STATS["kernel_broken"] = True
        warnings.warn(
            f"stable_linear: M=1 matvec kernel 不可用，回退原 M=128 padding GEMM："
            f"{exc!r}",
            RuntimeWarning,
            stacklevel=3,
        )
        _fallback("kernel-error")
        return None
    _STATS["decode_calls"] = int(_STATS["decode_calls"]) + 1
    return out


class ShapeStableLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, block_m: int = 128):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
        if block_m <= 0:
            raise ValueError("block_m must be positive")
        self.block_m = int(block_m)
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.weight = weight
        self.bias = bias

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_m: int = 128) -> "ShapeStableLinear":
        return cls(linear.weight, linear.bias, block_m=block_m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"x last dim {x.shape[-1]} != in_features {self.in_features}"
            )
        original_shape = x.shape
        flat = x.reshape(-1, self.in_features)
        rows = flat.shape[0]
        if rows == 1:
            # decode：单 token 前向不参与 prefill 的 M=128 逐位对齐口径。
            single = _decode_matvec(flat, self.weight, self.bias)
            if single is not None:
                return single.reshape(*original_shape[:-1], self.out_features)
        _STATS["padded_calls"] = int(_STATS["padded_calls"]) + 1
        outputs = []
        for start in range(0, rows, self.block_m):
            chunk = flat[start : start + self.block_m]
            valid = chunk.shape[0]
            if valid < self.block_m:
                pad = chunk.new_zeros(self.block_m - valid, self.in_features)
                chunk = torch.cat([chunk, pad], dim=0)
            y = F.linear(chunk, self.weight, self.bias)
            outputs.append(y[:valid])
        out = torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]
        return out.reshape(*original_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_m={self.block_m}, bias={self.bias is not None}"
        )
