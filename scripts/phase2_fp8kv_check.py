#!/usr/bin/env python3
"""Phase 2 任务 J：FP8 KV（E4M3 + per-(head,token) scale）的 op gate + microbench。

候选：``runner/ops/kv/fp8.py`` 的 FP8 KV 编码 + 任务 K 的 **fused decode
attention**（``decode_fused.py``）：K/V 以 fp8 读入，per (token, kv_head) 的
scale 直接折进 kernel（K 的 scale 乘在 QK 点积后、V 的乘在 ``p`` 上），**不物化
bf16 前缀**。这是 plan §7 的验收形态：8K 每层 K+V 只读 33.55 MB fp8（BF16 是
67.1 MB），零 scratch。``grouped_bmm`` 只在回退时把 fp8 反量化成 bf16（计入
``dequant_fallback``），不在性能路径上。冻结方案（实现之前写死，见模块
``runner/ops/kv/fp8.py`` docstring）：

- 存储 ``torch.float8_e4m3fn``，K/V 都存；
- scale 是 per (layer, kv_head, token) 的 float32 标量，布局 ``[capacity, kv_heads]``；
- 量化契约 ``scale = amax/448``、``q = rne_e4m3(clamp(x/scale))``（全 fp32 算术）；
- 反量化契约 ``bf16(float32(q) * scale)``（fp32 乘一次再 cast bf16）。

## 冻结的判据（实现候选之前确定，不因结果调整）

1. **量化/反量化 bitwise vs reference**（gate）：
   - 候选 ``quantize_kv`` 与独立 reference ``reference_quantize_kv`` 的 fp8 code、
     scale 必须 **逐位** 相同（用 ``view(uint8/int32)`` 比较，不受 -0/NaN 语义影响）；
   - 候选反量化（Triton 或 torch 回退）与 ``reference_dequantize`` 逐位相同；
   - 反量化的 fp32 中间值还必须等于 ``dequantize_exact`` 的
     ``float32(fp64 精确乘积)``（q 的 4 位 × scale 的 24 位在 fp64 里精确）。
2. **FP8-KV attention vs 同一份量化后 KV 的 fp64 attention**（gate）：沿用 BF16
   attention op gate 冻结的绝对容差 ``nrmse <= 1e-2``、``cosine >= 0.9999``、
   ``max_rel <= 2e-2``；另加“fused FP8 vs 同一份 KV 的 grouped_bmm”的
   ``nrmse <= 5e-3``（只差 reduction 顺序，不是格式差）。
   - **dispatch gate**：FP8 decode 必须全部走 fused，``dequant_fallback == 0``；
     回退计数写进产物。
   - fused kernel 的确定性单独 gate（重复调用 bitwise，graph replay bitwise）。
3. **FP8 vs BF16 的差**（diagnostic，plan §5 不要求两种格式 greedy trace 相同）：
   只记录 nrmse/cosine，并用从 CPU 校准冻结的宽 sanity 界
   （``nrmse <= 0.08``、``cosine >= 0.995``）防止输出坏掉被当成“格式差异”。
4. **roundtrip 误差**（gate）：``nrmse <= 0.04``、``cosine >= 0.999``、
   ``max|err| <= 0.05*amax``。CPU 校准：随机 bf16 向量 nrmse 0.026、
   max|err|/amax 0.03；这个界只拦“scale 用错/漏乘”这类量级错误。

## coverage（每个 case 都有 ``coverage:`` gate）

- ``quant-roundtrip``：zeros / signed_zeros / subnormal / saturation（显式小 scale
  触发 clamp）/ long_tail / random / fp8 中点（RNE）/ 极值，且覆盖
  ``quantize_kv`` 与 ``quantize_with_scale`` 两条入口、per-head 独立性、scale 布局。
- ``attn-fp8-vs-ref``：L ∈ {1, 513, 1025, 8191} × {full, partial_mask,
  capacity_view, cap_mask}；L ∈ {513, 8191} 额外做 CUDA Graph capture+replay。
- ``attn-fp8-vs-bf16``：L ∈ {513, 8191}，同一份未量化 K/V 下两种存储的 attention 差。
- ``microbench``：L ∈ {513, 8191}，capacity-view 输入；计 fp8-fused、bf16-fused、
  fp8-grouped（回退物化）与 dequant-only 四条路径；有效带宽按 fp8 实际读取的
  KV 字节算。

## microbench 口径

``fp8-fused`` 与 ``bf16-fused`` 是同一个 fused kernel，唯一差别是 K/V 的存储
（fp8 + 折叠 scale vs bf16）；``fp8-grouped`` 是回退路径（fp8 反量化写 bf16
scratch 后 grouped_bmm），用来量化“物化前缀”的代价；``dequant-only`` 单列反量化
kernel 的写带宽。``speedup_fused_vs_bf16`` 才是 FP8 KV 在 e2e 里省下的 KV 读
带宽；provenance 的 git excludes 注入与 ``phase2_attn_check.py`` 相同
（执行器建的 ``.venv``/``models`` 符号链接不算源码改动）。

整份脚本在 GPU 上约 60–90 秒（含一次 Triton 编译）。

用法：
    .venv/bin/python scripts/phase2_fp8kv_check.py --out {output}/op
    .venv/bin/python scripts/phase2_fp8kv_check.py --only microbench --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from runner.experiment.op_artifact import OpReport  # noqa: E402
from runner.ops.attention import (  # noqa: E402
    compact_gqa_decode_attention,
    decode_backend_info,
    decode_gqa_attention,
    reference_fp64,
)
from runner.ops.kv import fp8 as fp8_kv  # noqa: E402

H_Q = 24
H_KV = 4
HEAD_DIM = 256
SCALING = HEAD_DIM**-0.5
CAPACITY = 8192
LENGTHS = (1, 513, 1025, 8191)
BENCH_LENGTHS = (513, 8191)
MASK_VISIBLE_LONG = 513
SAMPLES = 2
SEED = 20261023
BENCH_WARMUP = 20
BENCH_ITERS = {513: 200, 8191: 60}
GRAPH_LENGTHS = (513, 8191)

# ---- 冻结的数值容差（实现之前确定；理由见模块 docstring）--------------------
TOL_QUANT_NRMSE = 4e-2
TOL_QUANT_COSINE = 0.999
TOL_QUANT_MAX_ABS_RATIO = 5e-2
TOL_ATTN_NRMSE = 1e-2
TOL_ATTN_COSINE = 0.9999
TOL_ATTN_MAX_REL = 2e-2
TOL_DIRECT_NRMSE = 5e-3  # fused(bf16 dequant) vs grouped_bmm 同一份 KV：只差 reduction
TOL_FUSED_RATIO = 1.05   # 任务 K 已冻结：fused 到 fp64 的 nrmse <= 1.05x grouped_bmm
TOL_DIAG_NRMSE = 8e-2
TOL_DIAG_COSINE = 0.995

CANDIDATE_NAME = "fp8-kv-e4m3-per-token-per-head"
CANDIDATE_FORMAT = "fp8-kv-e4m3"

# 必须是可 AST 字面量求值的 tuple；case 名就是产物里的 case 名。
SCENARIOS = (
    "quant-roundtrip",
    "attn-fp8-vs-ref",
    "attn-fp8-vs-bf16",
    "microbench",
)

QUANT_CLASSES = (
    "zeros",
    "signed_zeros",
    "subnormal",
    "long_tail",
    "random",
    "extremes",
    "fp8_midpoints",
    "saturation",
)
# roundtrip 的误差界只对“正常量化”的类生效：saturation 故意把 scale 缩小 16 倍
# 触发 clamp、fp8_midpoints 故意落在舍入边界，它们的误差不是存储质量指标。
ROUNDTRIP_CLASSES = (
    "zeros",
    "signed_zeros",
    "subnormal",
    "long_tail",
    "random",
    "extremes",
)


# ------------------------------------------------------------------ 输入构造
def _randn(shape, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=torch.float32).to(
        torch.bfloat16
    )


def make_q(seed: int, device: torch.device) -> torch.Tensor:
    return _randn((1, H_Q, 1, HEAD_DIM), seed, device)


def make_kv(length: int, seed: int, device: torch.device):
    """返回未量化的 bf16 ``k, v [1, H_KV, L, HEAD_DIM]``。"""
    return (
        _randn((1, H_KV, length, HEAD_DIM), seed, device),
        _randn((1, H_KV, length, HEAD_DIM), seed + 1, device),
    )


def additive_mask(length: int, visible: int, device: torch.device) -> torch.Tensor:
    mask = torch.full(
        (1, 1, 1, length),
        torch.finfo(torch.bfloat16).min,
        device=device,
        dtype=torch.bfloat16,
    )
    mask[..., :visible] = 0
    return mask


def visible_keys(length: int) -> int:
    """partial_mask 的可见前缀长度：至少 1，且 L>2 时至少遮住一个 key。"""
    if length <= 2:
        return 1
    if length <= MASK_VISIBLE_LONG:
        return length - 1
    return MASK_VISIBLE_LONG


def fp8_capacity_buffers(
    k: torch.Tensor, v: torch.Tensor, capacity: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 L 个 token 量化进 capacity buffer（其余槽位 0 code + 0 scale）。

    这正是 static-cap 的真实存储形态：decode 返回整块 capacity，mask 控制可见性。
    """
    length = k.shape[2]
    k_buf = torch.zeros(
        (1, H_KV, capacity, HEAD_DIM), dtype=fp8_kv.FP8_DTYPE, device=k.device
    )
    v_buf = torch.zeros(
        (1, H_KV, capacity, HEAD_DIM), dtype=fp8_kv.FP8_DTYPE, device=v.device
    )
    k_scale = torch.zeros((capacity, H_KV), dtype=torch.float32, device=k.device)
    v_scale = torch.zeros((capacity, H_KV), dtype=torch.float32, device=v.device)
    qk, sk = fp8_kv.quantize_kv(k)
    qv, sv = fp8_kv.quantize_kv(v)
    k_buf.view(torch.uint8)[:, :, :length] = qk.view(torch.uint8)
    v_buf.view(torch.uint8)[:, :, :length] = qv.view(torch.uint8)
    k_scale[:length] = sk
    v_scale[:length] = sv
    return k_buf, k_scale, v_buf, v_scale


def attach(k: torch.Tensor, k_scale: torch.Tensor, v: torch.Tensor, v_scale: torch.Tensor):
    return (
        fp8_kv.attach_scale(k, k_scale),
        fp8_kv.attach_scale(v, v_scale),
    )


# ------------------------------------------------------------------ 数值
def numeric_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    cand = candidate.double().flatten()
    ref = reference.double().flatten()
    diff = cand - ref
    ref_norm = torch.linalg.vector_norm(ref)
    cand_norm = torch.linalg.vector_norm(cand)
    ref_absmax = float(ref.abs().max().item())
    nrmse = float((torch.linalg.vector_norm(diff) / ref_norm).item())
    cosine = float((torch.dot(cand, ref) / (cand_norm * ref_norm)).item())
    max_abs = float(diff.abs().max().item())
    return {
        "nrmse": nrmse,
        "cosine": cosine,
        "max_rel": max_abs / max(ref_absmax, 1e-30),
        "max_abs": max_abs,
        "ref_absmax": ref_absmax,
    }


def _worst(acc: dict[str, float], metrics: dict[str, float]) -> None:
    acc["nrmse"] = max(acc.get("nrmse", 0.0), metrics["nrmse"])
    acc["cosine"] = min(acc.get("cosine", 1.0), metrics["cosine"])
    acc["max_rel"] = max(acc.get("max_rel", 0.0), metrics["max_rel"])
    acc["max_abs"] = max(acc.get("max_abs", 0.0), metrics["max_abs"])


def bits_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape:
        return False
    if a.dtype == fp8_kv.FP8_DTYPE:
        return torch.equal(bits_uint8(a), bits_uint8(b))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    if a.dtype == torch.bfloat16:
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    return torch.equal(a, b)


# ------------------------------------------------------------------ quant case
def quant_inputs(device: torch.device) -> list[tuple[str, torch.Tensor, torch.Tensor | None]]:
    """返回 [(class, x [1,4,T,256] bf16, explicit_scale [T,4] 或 None)]。"""
    t = 16
    cases: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []
    cases.append(("zeros", torch.zeros(1, H_KV, t, HEAD_DIM, dtype=torch.bfloat16, device=device), None))

    signed = torch.zeros(1, H_KV, t, HEAD_DIM, dtype=torch.bfloat16, device=device)
    signed[..., ::2] = -0.0
    signed[..., 1::2] = 0.0
    cases.append(("signed_zeros", signed, None))

    tiny = _randn((1, H_KV, t, HEAD_DIM), SEED + 3, device).float() * 1e-35
    tiny = tiny.to(torch.bfloat16)
    cases.append(("subnormal", tiny, None))

    tail = _randn((1, H_KV, t, HEAD_DIM), SEED + 4, device) * 0.01
    tail[..., 0] = 200.0
    cases.append(("long_tail", tail.to(torch.bfloat16), None))

    cases.append(("random", _randn((1, H_KV, t, HEAD_DIM), SEED + 5, device), None))

    ext = _randn((1, H_KV, t, HEAD_DIM), SEED + 6, device)
    ext[..., 0] = 3.0e38
    ext[..., 1] = -3.0e38
    ext[..., 2] = 1e-40
    cases.append(("extremes", ext.to(torch.bfloat16), None))

    # fp8 中点：显式 scale=1，值落在 E4M3 相邻可表示数的正中间，专门验 RNE。
    midpoints: list[float] = []
    for exp in range(-3, 9):
        base = 2.0**exp
        step = 2.0 ** (exp - 3)
        for index in range(7):
            midpoints.append(base + (index + 0.5) * step)
    mid = torch.tensor(midpoints, dtype=torch.float32, device=device)
    mid = mid.repeat((H_KV * t * HEAD_DIM + mid.numel() - 1) // mid.numel())
    mid = mid[: H_KV * t * HEAD_DIM].reshape(1, H_KV, t, HEAD_DIM).to(torch.bfloat16)
    cases.append(("fp8_midpoints", mid, torch.ones(t, H_KV, dtype=torch.float32, device=device)))

    # saturation：显式 scale 比 amax/448 小 16 倍，必然触发 clamp 到 448。
    sat = _randn((1, H_KV, t, HEAD_DIM), SEED + 7, device)
    amax = sat.float().abs().amax(dim=-1)[0]  # [H,T]
    sat_scale = (amax.transpose(0, 1) / fp8_kv.FP8_MAX / 16.0).clamp_min(1e-12)
    cases.append(("saturation", sat, sat_scale.contiguous()))
    return cases


def ref_quantize_with_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = scale.to(torch.float32).transpose(0, 1).unsqueeze(-1)
    return (x.float() / s).clamp(-fp8_kv.FP8_MAX, fp8_kv.FP8_MAX).to(fp8_kv.FP8_DTYPE)


def run_quant_case(report: OpReport, device: torch.device) -> None:
    case = report.case(
        "quant-roundtrip",
        dtype=str(fp8_kv.FP8_DTYPE),
        scale="per (layer, kv_head, token) float32",
        scale_layout="[capacity, kv_heads]",
        dequant="bf16(float32(q) * scale)",
        classes=list(QUANT_CLASSES),
        samples=len(quant_inputs(device)),
        tolerances={
            "nrmse": TOL_QUANT_NRMSE,
            "cosine": TOL_QUANT_COSINE,
            "max_abs_ratio": TOL_QUANT_MAX_ABS_RATIO,
        },
    )
    inputs = quant_inputs(device)
    worst: dict[str, float] = {}
    roundtrip: dict[str, float] = {}
    codes_ok = True
    scale_ok = True
    dequant_ok = True
    exact_ok = True
    class_detail: list[dict] = []

    for name, x, explicit_scale in inputs:
        if explicit_scale is None:
            q, scale = fp8_kv.quantize_kv(x)
            q_ref, scale_ref = fp8_kv.reference_quantize_kv(x)
        else:
            q = fp8_kv.quantize_with_scale(x, explicit_scale)
            scale = explicit_scale.contiguous()
            q_ref = ref_quantize_with_scale(x, scale)
            scale_ref = scale
        q_bits = bits_equal(q, q_ref)
        s_bits = bits_equal(scale, scale_ref)

        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        dq = fp8_kv.dequantize(q, scale, out=out)
        dq_ref = fp8_kv.reference_dequantize(q_ref, scale_ref)
        dq_bits = bits_equal(dq, dq_ref)
        exact_bf16, _exact_f32 = fp8_kv.dequantize_exact(q, scale)
        exact_bits = bits_equal(dq, exact_bf16)

        metrics = numeric_metrics(dq, x)
        amax = float(x.float().abs().max().item())
        max_abs_ratio = metrics["max_abs"] / max(amax, 1e-30)
        _worst(worst, metrics)
        worst["max_abs_ratio"] = max(worst.get("max_abs_ratio", 0.0), max_abs_ratio)
        if name in ROUNDTRIP_CLASSES:
            _worst(roundtrip, metrics)
            roundtrip["max_abs_ratio"] = max(
                roundtrip.get("max_abs_ratio", 0.0), max_abs_ratio
            )
        codes_ok = codes_ok and q_bits
        scale_ok = scale_ok and s_bits
        dequant_ok = dequant_ok and dq_bits
        exact_ok = exact_ok and exact_bits
        class_detail.append(
            {
                "class": name,
                "codes_bitwise": q_bits,
                "scale_bitwise": s_bits,
                "dequant_bitwise": dq_bits,
                "dequant_exact_bitwise": exact_bits,
                "nrmse": metrics["nrmse"],
                "cosine": metrics["cosine"],
                "max_abs_ratio": max_abs_ratio,
                "saturated_codes": int((q.float().abs() >= fp8_kv.FP8_MAX).sum()),
            }
        )
        case.diag(
            f"detail[{name}]",
            True,
            {
                "codes_bitwise": q_bits,
                "scale_bitwise": s_bits,
                "dequant_bitwise": dq_bits,
                "dequant_exact_bitwise": exact_bits,
                "roundtrip": {
                    "nrmse": metrics["nrmse"],
                    "cosine": metrics["cosine"],
                    "max_abs_ratio": max_abs_ratio,
                },
            },
        )

    # 全零向量：scale 必须是 0，code 必须是 ±0（符号位允许，e4m3fn 保留 -0）。
    _zeros_name, zeros_x, _ = inputs[0]
    qz, sz = fp8_kv.quantize_kv(zeros_x)
    zero_scale_ok = bool((sz == 0).all())
    zero_code_ok = bool(((bits_uint8(qz) & 0x7F) == 0).all())

    # per-head 独立性：只改 (head=1, token=0) 的数值，其他 head/token 的 scale 与
    # 反量化结果必须 bitwise 不变，改动的 head/token 必须变化。
    base = _randn((1, H_KV, 8, HEAD_DIM), SEED + 11, device)
    perturbed = base.clone()
    perturbed[0, 1, 0] = perturbed[0, 1, 0] * 4.0 + 0.5
    qb, sb = fp8_kv.quantize_kv(base)
    qp, sp = fp8_kv.quantize_kv(perturbed)
    db = fp8_kv.dequantize(qb, sb)
    dp = fp8_kv.dequantize(qp, sp)
    other_heads = [0, 2, 3]
    other_ok = all(
        torch.equal(bits_uint8(qb[:, h]), bits_uint8(qp[:, h]))
        and torch.equal(
            sb[:, h].contiguous().view(torch.int32),
            sp[:, h].contiguous().view(torch.int32),
        )
        for h in other_heads
    )
    other_tokens = list(range(1, 8))
    token_ok = all(
        torch.equal(db[:, :, t], dp[:, :, t]) for t in other_tokens
    ) and not torch.equal(db[:, :, 0], dp[:, :, 0])
    head_changed = not torch.equal(bits_uint8(qb[:, 1, 0]), bits_uint8(qp[:, 1, 0]))

    layout_ok = (
        tuple(sb.shape) == (8, H_KV)
        and sb.dtype == torch.float32
        and sb.stride() == (H_KV, 1)
        and tuple(qb.shape) == tuple(base.shape)
        and qb.dtype == fp8_kv.FP8_DTYPE
    )

    case.gate(
        "coverage: 量化类 + 两条入口 + per-head 独立性 + scale 布局",
        len(class_detail) == len(QUANT_CLASSES)
        and all(item["class"] == name for item, name in zip(class_detail, QUANT_CLASSES)),
        {
            "classes": [item["class"] for item in class_detail],
            "explicit_scale_entry": "saturation",
            "derived_scale_entry": "其余类",
            "samples": len(class_detail),
        },
    )
    case.gate(
        "codes: candidate vs reference 全类 bitwise",
        codes_ok,
        {"classes": len(class_detail)},
    )
    case.gate(
        "scale: candidate vs reference 全类 bitwise",
        scale_ok,
        {"classes": len(class_detail)},
    )
    case.gate(
        "dequant: candidate vs reference 全类 bitwise",
        dequant_ok,
        {"classes": len(class_detail)},
    )
    case.gate(
        "dequant: fp32 中间值与 fp64 精确乘积 bitwise（含 bf16 cast）",
        exact_ok,
        {"note": "q 的 4 位 × scale 的 24 位在 fp64 里精确，float32 乘积是正确舍入"},
    )
    case.gate(
        "zero: 全零向量的 scale 全 0 且 code 为 ±0",
        zero_scale_ok and zero_code_ok,
        {"scale_zero": zero_scale_ok, "code_signed_zero": zero_code_ok},
    )
    case.gate(
        "layout: scale [T,H] float32 连续、code [1,H,T,D] fp8",
        layout_ok,
        {"scale_shape": list(sb.shape), "scale_stride": list(sb.stride())},
    )
    case.gate(
        "per-head 独立性：只改 (head=1, token=0) 时其他 head/token bitwise 不变",
        other_ok and token_ok and head_changed,
        {
            "other_heads_bitwise": other_ok,
            "other_tokens_bitwise": token_ok,
            "changed_head_changed": head_changed,
        },
    )
    case.gate(
        f"roundtrip: nrmse <= {TOL_QUANT_NRMSE}（正常类 {ROUNDTRIP_CLASSES}）",
        roundtrip["nrmse"] <= TOL_QUANT_NRMSE,
        {"nrmse_worst": roundtrip["nrmse"], "classes": list(ROUNDTRIP_CLASSES)},
    )
    case.gate(
        f"roundtrip: cosine >= {TOL_QUANT_COSINE}（正常类）",
        roundtrip["cosine"] >= TOL_QUANT_COSINE,
        {"cosine_min": roundtrip["cosine"]},
    )
    case.gate(
        f"roundtrip: max|err|/amax <= {TOL_QUANT_MAX_ABS_RATIO}（正常类）",
        roundtrip["max_abs_ratio"] <= TOL_QUANT_MAX_ABS_RATIO,
        {"max_abs_ratio_worst": roundtrip["max_abs_ratio"]},
    )

    case.metric("nrmse_worst", roundtrip["nrmse"])
    case.metric("cosine_min", roundtrip["cosine"])
    case.metric("max_abs_ratio_worst", roundtrip["max_abs_ratio"])
    case.metric("nrmse_worst_all_classes", worst["nrmse"])
    case.metric("classes", float(len(class_detail)))
    case.metric("saturation_codes", float(sum(item["saturated_codes"] for item in class_detail)))


# ------------------------------------------------------------------ attn case
def build_variants(length: int, seed: int, device: torch.device):
    """返回 {variant: (q, k_fp8, k_scale, v_fp8, v_scale, mask, k_bf, v_bf)}。"""
    q = make_q(seed, device)
    visible = visible_keys(length)
    cap_visible = min(length, MASK_VISIBLE_LONG)
    k, v = make_kv(length, seed + 1, device)
    k_cap_src, v_cap_src = make_kv(length, seed + 1, device)
    k_buf, k_scale, v_buf, v_scale = fp8_capacity_buffers(k_cap_src, v_cap_src, CAPACITY)
    mask_len = additive_mask(length, visible, device)
    mask_cap = additive_mask(CAPACITY, cap_visible, device)

    qk, sk = fp8_kv.quantize_kv(k)
    qv, sv = fp8_kv.quantize_kv(v)
    # capacity-view：fp8 前缀 view，scale 也取前缀（真实 static-exact 形态）。
    k_view = k_buf[:, :, :length]
    v_view = v_buf[:, :, :length]
    sk_view = k_scale[:length]
    sv_view = v_scale[:length]

    variants = {}
    variants["full"] = dict(
        q=q,
        k=qk,
        k_scale=sk,
        v=qv,
        v_scale=sv,
        mask=None,
        k_bf=fp8_kv.reference_dequantize(qk, sk),
        v_bf=fp8_kv.reference_dequantize(qv, sv),
        visible=None,
    )
    variants["partial_mask"] = dict(
        q=q,
        k=k_view,
        k_scale=sk_view,
        v=v_view,
        v_scale=sv_view,
        mask=mask_len,
        k_bf=fp8_kv.reference_dequantize(k_view, sk_view),
        v_bf=fp8_kv.reference_dequantize(v_view, sv_view),
        visible=visible,
    )
    variants["capacity_view"] = dict(
        q=q,
        k=k_view,
        k_scale=sk_view,
        v=v_view,
        v_scale=sv_view,
        mask=None,
        k_bf=fp8_kv.reference_dequantize(k_view, sk_view),
        v_bf=fp8_kv.reference_dequantize(v_view, sv_view),
        visible=None,
    )
    variants["cap_mask"] = dict(
        q=q,
        k=k_buf,
        k_scale=k_scale,
        v=v_buf,
        v_scale=v_scale,
        mask=mask_cap,
        k_bf=fp8_kv.reference_dequantize(k_buf, k_scale),
        v_bf=fp8_kv.reference_dequantize(v_buf, v_scale),
        visible=cap_visible,
    )
    return variants, visible, cap_visible


def call_fp8(variant) -> torch.Tensor:
    """候选调用：fp8 K/V + scale 直接进 fused decode attention（scale 折进 kernel）。"""
    out, _ = decode_gqa_attention(
        variant["q"],
        variant["k"],
        variant["v"],
        variant["mask"],
        SCALING,
        variant["k_scale"],
        variant["v_scale"],
    )
    return out


def call_grouped(variant) -> torch.Tensor:
    """对照：同一份 fp8 用 torch reference 反量化成 bf16 后走 grouped_bmm。

    这里刻意用 reference 反量化（不是运行后端）作对照，等价于“把同一份量化后
    KV 物化成 bf16 再算 attention”，与 fused 的差别只有 reduction 顺序和
    scale 折叠位置。
    """
    out, _ = compact_gqa_decode_attention(
        variant["q"], variant["k_bf"], variant["v_bf"], variant["mask"], SCALING
    )
    return out


def call_fused_bf16(variant) -> torch.Tensor:
    """同一份 KV 的 bf16 反量化值直接走 fused（隔离 reduction 顺序与 scale 折叠）。"""
    out, _ = decode_gqa_attention(
        variant["q"], variant["k_bf"], variant["v_bf"], variant["mask"], SCALING
    )
    return out


def run_attn_ref_case(report: OpReport, device: torch.device) -> None:
    case = report.case(
        "attn-fp8-vs-ref",
        lengths=list(LENGTHS),
        capacity=CAPACITY,
        variants=("full", "partial_mask", "capacity_view", "cap_mask"),
        samples=SAMPLES,
        q_heads=H_Q,
        kv_heads=H_KV,
        head_dim=HEAD_DIM,
        scaling=SCALING,
        graph_lengths=list(GRAPH_LENGTHS),
        backend="fused decode attention (PHASE2_ATTN_DECODE), fp8 scales folded in kernel",
        tolerances={
            "nrmse_vs_fp64": TOL_ATTN_NRMSE,
            "cosine": TOL_ATTN_COSINE,
            "max_rel": TOL_ATTN_MAX_REL,
            "fused_vs_bmm_fp64_ratio": TOL_FUSED_RATIO,
            "fused_bf16_vs_grouped_nrmse": TOL_DIRECT_NRMSE,
        },
    )
    variants_list = ("full", "partial_mask", "capacity_view", "cap_mask")
    worst: dict[str, dict[str, float]] = {name: {} for name in variants_list}
    kernel_direct: dict[str, dict[str, float]] = {name: {} for name in variants_list}
    fold_diff: dict[str, dict[str, float]] = {name: {} for name in variants_list}
    bmm_nrmse = {name: 0.0 for name in variants_list}
    fused_bf_nrmse = {name: 0.0 for name in variants_list}
    ratio_worst = {name: 0.0 for name in variants_list}
    structure = {"capacity_view_noncontiguous": True, "full_contiguous": True}
    mask_isolated = {"partial_mask": True, "cap_mask": True}
    variants_run: list[str] = []
    combos: list[str] = []
    graph_detail: list[dict] = []
    determinism = True
    fallback_before = decode_backend_info()["dequant_fallback"]
    fused_before = decode_backend_info()["fused"]

    for length in LENGTHS:
        for sample in range(SAMPLES):
            seed = SEED + 1000 * length + 17 * sample
            variants, _visible, _cap_visible = build_variants(length, seed, device)
            for name in variants_list:
                variant = variants[name]
                if name not in variants_run:
                    variants_run.append(name)
                tag = f"L{length}:{name}"
                if tag not in combos:
                    combos.append(tag)
                cand = call_fp8(variant)
                fused_bf = call_fused_bf16(variant)
                base = call_grouped(variant)
                ref = reference_fp64(
                    variant["q"], variant["k_bf"], variant["v_bf"], variant["mask"], SCALING
                )
                metrics = numeric_metrics(cand, ref)
                base_metrics = numeric_metrics(base, ref)
                fused_bf_metrics = numeric_metrics(fused_bf, ref)
                kernel_metrics = numeric_metrics(fused_bf, base)
                fold_metrics = numeric_metrics(cand, fused_bf)
                _worst(worst[name], metrics)
                _worst(kernel_direct[name], kernel_metrics)
                _worst(fold_diff[name], fold_metrics)
                bmm_nrmse[name] = max(bmm_nrmse[name], base_metrics["nrmse"])
                fused_bf_nrmse[name] = max(
                    fused_bf_nrmse[name], fused_bf_metrics["nrmse"]
                )
                # 任务 K 的冻结判据：同一份 bf16 KV 上，fused 到 fp64 的距离 <= 1.05x bmm。
                ratio_worst[name] = max(
                    ratio_worst[name],
                    fused_bf_metrics["nrmse"] / max(base_metrics["nrmse"], 1e-12),
                )
                structure["capacity_view_noncontiguous"] &= bool(
                    not variant["k"].is_contiguous() if name == "capacity_view" else True
                )
                structure["full_contiguous"] &= bool(
                    variant["k"].is_contiguous() if name == "full" else True
                )
                case.diag(
                    f"detail[{tag}][sample {sample}]",
                    True,
                    {
                        "candidate": metrics,
                        "grouped_bmm_on_dequant": base_metrics,
                        "fused_bf16_vs_grouped": kernel_metrics,
                        "fp8_fused_vs_bf16_fused": fold_metrics,
                        "shape": tuple(cand.shape),
                        "dtype": str(cand.dtype),
                    },
                )
            # 同一输入重复调用必须 bitwise 一致（fused kernel 的确定性）。
            again = call_fp8(variants["cap_mask"])
            first = call_fp8(variants["cap_mask"])
            determinism = determinism and bool(torch.equal(first, again))

            # mask 隔离：把不可见槽位的 fp8 code/scale 改成垃圾值，输出必须 bitwise 不变。
            for name in ("partial_mask", "cap_mask"):
                variant = variants[name]
                vis = variant["visible"]
                k_junk = variant["k"].clone()
                v_junk = variant["v"].clone()
                k_junk.view(torch.uint8)[:, :, vis:] = 0x7E  # e4m3 的一个有限大值
                v_junk.view(torch.uint8)[:, :, vis:] = 0x7E
                k_scale_junk = variant["k_scale"].clone()
                v_scale_junk = variant["v_scale"].clone()
                k_scale_junk[vis:] = 3.0
                v_scale_junk[vis:] = 3.0
                base_out = call_fp8(variant)
                junk_out = call_fp8(
                    {**variant, "k": k_junk, "v": v_junk,
                     "k_scale": k_scale_junk, "v_scale": v_scale_junk}
                )
                mask_isolated[name] = mask_isolated[name] and bool(
                    torch.equal(base_out, junk_out)
                )

            # CUDA Graph replay：capture 后 replay 与 eager 调用 bitwise，replay 两次也 bitwise。
            # 用 cap_mask（真实 static-cap decode 的 shape）。capture 失败写 gate failure
            # 而不是抛异常，保证 op.json 仍然落盘、其他 case 的证据不丢。
            if length in GRAPH_LENGTHS:
                variant = variants["cap_mask"]
                eager = call_fp8(variant)
                try:
                    graph, out = capture_call(lambda: call_fp8(variant))
                    graph.replay()
                    torch.cuda.synchronize()
                    replay = (out[0] if isinstance(out, tuple) else out).clone()
                    graph.replay()
                    torch.cuda.synchronize()
                    replay2 = (out[0] if isinstance(out, tuple) else out).clone()
                    ok_replay = bool(
                        torch.equal(eager, replay) and torch.equal(replay, replay2)
                    )
                    detail = {
                        "length": length,
                        "variant": "cap_mask",
                        "eager_vs_replay_bitwise": bool(torch.equal(eager, replay)),
                        "replay_stable": bool(torch.equal(replay, replay2)),
                    }
                except Exception as exc:  # noqa: BLE001 - 记录后继续，产物仍可见
                    ok_replay = False
                    detail = {
                        "length": length,
                        "variant": "cap_mask",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                graph_detail.append(detail)
                case.gate(f"graph replay[L={length}]: eager/replay bitwise", ok_replay, detail)
                case.diag(f"graph-replay[L={length}]", ok_replay, detail)

    info = decode_backend_info()
    fallback_delta = int(info["dequant_fallback"]) - int(fallback_before)
    fused_delta = int(info["fused"]) - int(fused_before)
    graph_covered = sorted({int(item["length"]) for item in graph_detail})

    case.gate(
        f"coverage: L={list(LENGTHS)} × full/partial_mask/capacity_view/cap_mask "
        f"(capacity={CAPACITY}, samples={SAMPLES}) + graph replay {list(GRAPH_LENGTHS)}",
        set(variants_run) == set(variants_list)
        and len(combos) == len(LENGTHS) * len(variants_list)
        and set(graph_covered) == set(GRAPH_LENGTHS),
        {
            "variants": variants_run,
            "combos": combos,
            "samples": SAMPLES,
            "lengths": list(LENGTHS),
            "graph_covered": graph_covered,
            "partial_mask_visible": [visible_keys(x) for x in LENGTHS],
            "cap_mask_visible": [min(x, MASK_VISIBLE_LONG) for x in LENGTHS],
        },
    )
    case.gate(
        "structure: capacity_view 是非连续 fp8 前缀 view、full 连续",
        structure["capacity_view_noncontiguous"] and structure["full_contiguous"],
        structure,
    )
    case.gate(
        "dispatch: FP8 decode 全部走 fused kernel，零 bf16 物化回退",
        fused_delta > 0 and fallback_delta == 0,
        {
            "fused_calls": fused_delta,
            "dequant_fallbacks": fallback_delta,
            "last_fallback": info["last_fallback"],
        },
    )
    case.gate(
        "determinism: 同一输入重复调用 bitwise 一致",
        determinism,
        {"bitwise": determinism},
    )
    case.gate(
        "mask: 不可见槽位的 fp8 code/scale 改写后输出 bitwise 不变",
        mask_isolated["partial_mask"] and mask_isolated["cap_mask"],
        {
            "partial_mask_visible": [visible_keys(x) for x in LENGTHS],
            "bitwise": mask_isolated,
        },
    )
    for name in variants_list:
        metrics = worst[name]
        case.gate(
            f"numeric[{name}]: nrmse <= {TOL_ATTN_NRMSE}",
            metrics["nrmse"] <= TOL_ATTN_NRMSE,
            metrics,
        )
        case.gate(
            f"numeric[{name}]: cosine >= {TOL_ATTN_COSINE}",
            metrics["cosine"] >= TOL_ATTN_COSINE,
            {"cosine": metrics["cosine"]},
        )
        case.gate(
            f"numeric[{name}]: max_rel <= {TOL_ATTN_MAX_REL}",
            metrics["max_rel"] <= TOL_ATTN_MAX_REL,
            {"max_rel": metrics["max_rel"], "max_abs": metrics["max_abs"]},
        )
        case.gate(
            f"compare[{name}]: fused(bf16 dequant) vs grouped_bmm 同一份 KV "
            f"nrmse <= {TOL_DIRECT_NRMSE}（只差 reduction 顺序）",
            kernel_direct[name]["nrmse"] <= TOL_DIRECT_NRMSE,
            kernel_direct[name],
        )
        case.gate(
            f"numeric[{name}]: FP8 fused 到 fp64 的 nrmse <= {TOL_FUSED_RATIO}x "
            f"grouped_bmm（任务 K 已冻结的相对判据）",
            ratio_worst[name] <= TOL_FUSED_RATIO,
            {
                "ratio": ratio_worst[name],
                "fused_bf16_nrmse": fused_bf_nrmse[name],
                "grouped_nrmse": bmm_nrmse[name],
            },
        )

    case.metric("nrmse_worst", max(item["nrmse"] for item in worst.values()))
    case.metric("cosine_min", min(item["cosine"] for item in worst.values()))
    case.metric("max_rel_worst", max(item["max_rel"] for item in worst.values()))
    case.metric(
        "kernel_direct_nrmse_worst",
        max(item["nrmse"] for item in kernel_direct.values()),
    )
    case.metric("fold_nrmse_worst", max(item["nrmse"] for item in fold_diff.values()))
    case.metric("ratio_worst", max(ratio_worst.values()))
    case.metric("fused_bf16_nrmse_worst", max(fused_bf_nrmse.values()))
    case.metric("bmm_nrmse_worst", max(bmm_nrmse.values()))
    case.metric("graph_replays", float(len(graph_detail)))
    case.metric("fused_calls", float(fused_delta))
    case.metric("dequant_fallbacks", float(fallback_delta))
    case.metric("samples_total", float(SAMPLES * len(LENGTHS)))


# ------------------------------------------------------------------ diag case
def run_diag_case(report: OpReport, device: torch.device) -> None:
    case = report.case(
        "attn-fp8-vs-bf16",
        lengths=list(BENCH_LENGTHS),
        purpose="diagnostic",
        backend="fused decode attention, fp8 kv vs unquantized bf16 kv（同一 kernel）",
        note="plan §5：FP8 vs BF16 的差异只作 diagnostic，不要求 greedy trace 相同",
    )
    worst: dict[str, float] = {}
    finite_ok = True
    for length in BENCH_LENGTHS:
        seed = SEED + 31 * length
        q = make_q(seed, device)
        k, v = make_kv(length, seed + 1, device)
        qk, sk = fp8_kv.quantize_kv(k)
        qv, sv = fp8_kv.quantize_kv(v)
        fp8_out, _ = decode_gqa_attention(q, qk, qv, None, SCALING, sk, sv)
        bf16_out, _ = decode_gqa_attention(q, k, v, None, SCALING)
        metrics = numeric_metrics(fp8_out, bf16_out)
        _worst(worst, metrics)
        finite_ok = finite_ok and bool(torch.isfinite(fp8_out).all())
        case.diag(
            f"fp8-vs-bf16[L={length}]",
            True,
            {"candidate": metrics, "bitwise": torch.equal(fp8_out, bf16_out)},
        )
    case.gate(
        "coverage: FP8 vs BF16 对比覆盖 L ∈ {513, 8191}",
        len(worst) > 0,
        {"lengths": list(BENCH_LENGTHS)},
    )
    case.gate(
        "sanity: FP8 输出 finite",
        finite_ok,
        {"finite": finite_ok},
    )
    case.gate(
        f"sanity: fp8-vs-bf16 nrmse <= {TOL_DIAG_NRMSE}（宽界，只防坏输出）",
        worst["nrmse"] <= TOL_DIAG_NRMSE,
        {"nrmse_worst": worst["nrmse"], "cosine_min": worst["cosine"]},
    )
    case.gate(
        f"sanity: fp8-vs-bf16 cosine >= {TOL_DIAG_COSINE}（宽界，只防坏输出）",
        worst["cosine"] >= TOL_DIAG_COSINE,
        {"cosine_min": worst["cosine"]},
    )
    case.metric("nrmse_worst", worst["nrmse"])
    case.metric("cosine_min", worst["cosine"])
    case.metric("max_rel_worst", worst["max_rel"])


# ------------------------------------------------------------------ graph/time
def capture_call(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


def time_per_call_ms(fn, *, warmup: int, iters: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for index in range(iters):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    count = len(times)
    return {
        "median_ms": times[count // 2],
        "min_ms": times[0],
        "p10_ms": times[max(0, count // 10)],
    }


def run_micro_case(report: OpReport, device: torch.device) -> None:
    info_before = decode_backend_info()
    case = report.case(
        "microbench",
        lengths=list(BENCH_LENGTHS),
        capacity=CAPACITY,
        warmup=BENCH_WARMUP,
        iters={str(k): v for k, v in BENCH_ITERS.items()},
        timing="per-call CUDA events, median",
        paths=("fp8-fused", "bf16-fused", "fp8-grouped-dequant-fallback", "dequant-only"),
        formula="fp8_kv_bytes = 2*H_kv*L*head_dim*1; bf16 = same *2",
    )
    rows: list[dict] = []
    for length in BENCH_LENGTHS:
        seed = SEED + 7 * length
        q = make_q(seed, device)
        k, v = make_kv(length, seed + 1, device)
        # capacity-view 输入：与 static-exact / static-cap 的 decode 形态一致。
        k_buf, k_scale, v_buf, v_scale = fp8_capacity_buffers(k, v, CAPACITY)
        qk = k_buf[:, :, :length]
        qv = v_buf[:, :, :length]
        sk = k_scale[:length]
        sv = v_scale[:length]
        k_bf = fp8_kv.reference_dequantize(qk, sk)
        v_bf = fp8_kv.reference_dequantize(qv, sv)

        def fp8_call():
            out, _ = decode_gqa_attention(q, qk, qv, None, SCALING, sk, sv)
            return out

        def bf16_call():
            out, _ = decode_gqa_attention(q, k, v, None, SCALING)
            return out

        def fp8_grouped_call():
            out, _ = compact_gqa_decode_attention(
                q,
                fp8_kv.attach_scale(qk, sk),
                fp8_kv.attach_scale(qv, sv),
                None,
                SCALING,
            )
            return out

        def dequant_call():
            out = fp8_kv.dequant_scratch("bench-k", qk.shape, qk.device)
            fp8_kv.dequantize(qk, sk, out=out)
            outv = fp8_kv.dequant_scratch("bench-v", qv.shape, qv.device)
            fp8_kv.dequantize(qv, sv, out=outv)

        cand, _ = fp8_call()
        ref = reference_fp64(q, k_bf, v_bf, None, SCALING)
        metrics = numeric_metrics(cand, ref)
        again, _ = fp8_call()
        determinism = bool(torch.equal(cand, again))
        fp8_time = time_per_call_ms(
            fp8_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length]
        )
        bf16_time = time_per_call_ms(
            bf16_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length]
        )
        grouped_time = time_per_call_ms(
            fp8_grouped_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length]
        )
        dequant_time = time_per_call_ms(
            dequant_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length]
        )
        fp8_bytes = 2 * H_KV * length * HEAD_DIM * 1
        bf16_bytes = 2 * H_KV * length * HEAD_DIM * 2
        dequant_bytes = 2 * H_KV * length * HEAD_DIM * 2  # 写 bf16 scratch
        row = {
            "length": length,
            "fp8_fused_ms": fp8_time["median_ms"],
            "fp8_fused_min_ms": fp8_time["min_ms"],
            "bf16_fused_ms": bf16_time["median_ms"],
            "bf16_fused_min_ms": bf16_time["min_ms"],
            "fp8_grouped_ms": grouped_time["median_ms"],
            "dequant_ms": dequant_time["median_ms"],
            "speedup_fused_vs_bf16": bf16_time["median_ms"] / fp8_time["median_ms"],
            "speedup_fused_vs_grouped_fp8": grouped_time["median_ms"]
            / fp8_time["median_ms"],
            "fp8_kv_gbps": fp8_bytes / (fp8_time["median_ms"] * 1e-3) / 1e9,
            "bf16_kv_gbps": bf16_bytes / (bf16_time["median_ms"] * 1e-3) / 1e9,
            "dequant_write_gbps": dequant_bytes
            / (dequant_time["median_ms"] * 1e-3)
            / 1e9,
            "fp8_kv_bytes_per_call": fp8_bytes,
            "deterministic": determinism,
            "nrmse_vs_fp64": metrics["nrmse"],
            "samples": BENCH_ITERS[length],
        }
        rows.append(row)
        case.diag(f"timing[L={length}]", True, row)
        case.gate(
            f"coverage: L={length} fp8-fused/bf16-fused/fp8-grouped 各 "
            f"{BENCH_ITERS[length]} 次计时 + dequant 单独计时",
            True,
            {"length": length, "warmup": BENCH_WARMUP, "iters": BENCH_ITERS[length]},
        )
        case.gate(
            f"numeric: L={length} fp8-fused vs fp64 nrmse <= {TOL_ATTN_NRMSE}",
            metrics["nrmse"] <= TOL_ATTN_NRMSE,
            metrics,
        )
        case.gate(
            f"determinism: L={length} fp8-fused 重复调用 bitwise",
            determinism,
            {"bitwise": determinism},
        )
        case.gate(
            f"timing: L={length} 计时样本有效",
            fp8_time["median_ms"] > 0
            and bf16_time["median_ms"] > 0
            and grouped_time["median_ms"] > 0
            and dequant_time["median_ms"] > 0
            and all(
                math.isfinite(item["median_ms"])
                for item in (fp8_time, bf16_time, grouped_time, dequant_time)
            ),
            {
                "fp8_fused": fp8_time,
                "bf16_fused": bf16_time,
                "fp8_grouped": grouped_time,
                "dequant": dequant_time,
            },
        )

    info = decode_backend_info()
    dequant_info = fp8_kv.dequant_info()
    case.diag("dequant backend / decode dispatch", True, {**dequant_info, **info})
    case.metric("triton_active", 1.0 if dequant_info["triton_available"] else 0.0)
    case.metric("dequant_calls", float(dequant_info["dequant_calls"]))
    case.metric("dequant_fallbacks", float(dequant_info["dequant_fallbacks"]))
    case.metric("fused_calls", float(info["fused"] - info_before["fused"]))
    # 汇总 metric 用最差（最短）长度，避免只报好看的数字。
    case.metric("fp8_fused_ms_worst", max(row["fp8_fused_ms"] for row in rows))
    case.metric("bf16_fused_ms_worst", max(row["bf16_fused_ms"] for row in rows))
    case.metric("fp8_grouped_ms_worst", max(row["fp8_grouped_ms"] for row in rows))
    case.metric("dequant_ms_worst", max(row["dequant_ms"] for row in rows))
    case.metric(
        "speedup_fused_vs_bf16_min", min(row["speedup_fused_vs_bf16"] for row in rows)
    )
    case.metric(
        "speedup_fused_vs_grouped_fp8_min",
        min(row["speedup_fused_vs_grouped_fp8"] for row in rows),
    )
    case.metric("fp8_kv_gbps_worst", min(row["fp8_kv_gbps"] for row in rows))
    case.metric("bf16_kv_gbps_worst", min(row["bf16_kv_gbps"] for row in rows))
    case.metric(
        "dequant_write_gbps_worst", min(row["dequant_write_gbps"] for row in rows)
    )
    case.metric("nrmse_vs_fp64_worst", max(row["nrmse_vs_fp64"] for row in rows))
    case.metric("lengths", float(len(rows)))


# ------------------------------------------------------------------ entry
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True, help="产物目录，写 {out}/op.json")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"只跑指定 case（默认全部）；可选 {', '.join(SCENARIOS)}",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认已冻结）")
    return parser.parse_args(argv)


def select_cases(only: list[str] | None) -> list[str]:
    if only is None:
        return list(SCENARIOS)
    names = [part for item in only for part in item.split(",") if part]
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"未知 case: {unknown}；可选 {list(SCENARIOS)}")
    return names


def _configure_git_provenance() -> Path:
    """给本进程的 git 注入临时 excludesFile，见模块 docstring 的 provenance 说明。"""
    import os
    import tempfile

    directory = Path(tempfile.mkdtemp(prefix="phase2-fp8-git-"))
    excludes = directory / "exclude"
    excludes.write_text(".venv\nmodels\n", encoding="utf-8")
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "core.excludesFile"
    os.environ["GIT_CONFIG_VALUE_0"] = str(excludes)
    return excludes


def summarize(payload: dict) -> dict:
    metrics = {entry["case"]: entry["metrics"] for entry in payload["cases"]}
    return {
        "ok": payload["summary"]["ok"],
        "cases": [
            {**entry, "metrics": metrics.get(entry["case"], {})}
            for entry in payload["summary"]["cases"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA 不可用：这个 op 脚本只能在 GPU job 里运行", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    _configure_git_provenance()
    selected = select_cases(args.only)
    report = OpReport(
        args.out,
        CANDIDATE_NAME,
        CANDIDATE_FORMAT,
        candidate_extra={
            "kv_dtype": "torch.float8_e4m3fn",
            "scale": "per (layer, kv_head, token), float32, layout [capacity, kv_heads]",
            "quantize": "scale = amax/448 (fp32); q = rne_e4m3(clamp(x/scale))",
            "dequantize": "bf16(float32(q) * scale)",
            "unused_slots": "code 0, scale 0, mask semantics unchanged",
            "attention_path": "fused split-K Triton decode attention with k_scale/v_scale folded in the kernel; zero bf16 materialization",
            "fallback": "grouped_bmm dequantizes fp8 to bf16 only when fused preconditions fail (counted)",
            "tolerances_frozen_before_candidate": {
                "quant_nrmse": TOL_QUANT_NRMSE,
                "quant_cosine": TOL_QUANT_COSINE,
                "quant_max_abs_ratio": TOL_QUANT_MAX_ABS_RATIO,
                "attn_nrmse_vs_fp64": TOL_ATTN_NRMSE,
                "attn_cosine": TOL_ATTN_COSINE,
                "attn_max_rel": TOL_ATTN_MAX_REL,
                "fused_vs_bmm_fp64_ratio": TOL_FUSED_RATIO,
                "fused_bf16_vs_grouped_nrmse": TOL_DIRECT_NRMSE,
                "diag_nrmse_vs_bf16": TOL_DIAG_NRMSE,
                "diag_cosine_vs_bf16": TOL_DIAG_COSINE,
                "reason": "fused kernel 的误差底噪是 bf16 MMA + fp32 online softmax"
                "（沿用 phase2_attn_check 的冻结容差）；candidate vs grouped_bmm 用"
                "5e-3，只差 reduction 顺序；roundtrip 界由 CPU 校准"
                "（随机 bf16 nrmse 0.026、max|err|/amax 0.03）",
            },
        },
    )
    for name in selected:
        if name == "quant-roundtrip":
            run_quant_case(report, device)
        elif name == "attn-fp8-vs-ref":
            run_attn_ref_case(report, device)
        elif name == "attn-fp8-vs-bf16":
            run_diag_case(report, device)
        elif name == "microbench":
            run_micro_case(report, device)
        else:  # pragma: no cover - select_cases 已拦截
            raise SystemExit(f"无法解析 case {name!r}")
    payload = report.finish(
        extra={
            "coverage": {
                "lengths": list(LENGTHS),
                "bench_lengths": list(BENCH_LENGTHS),
                "graph_lengths": list(GRAPH_LENGTHS),
                "capacity": CAPACITY,
                "variants": ["full", "partial_mask", "capacity_view", "cap_mask"],
            },
            "dequant_backend": fp8_kv.dequant_info(),
        }
    )
    print(json.dumps(summarize(payload), indent=2, ensure_ascii=False))
    print(f"output: {args.out / 'op.json'}")
    return 0 if payload["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
