#!/usr/bin/env python3
"""Phase 2 任务 A/K：decode attention（q_len==1）的 op gate + microbench。

两条路径，同一个脚本、同一份 fp64 reference：

- **A（既有 7 个 case）**：``runner/ops/attention`` 的紧凑 GQA decode——直接读
  ``[1,4,L,256]`` 的 K/V、按 KV head 分 6 组做 batched matmul，**不物化**
  ``repeat_kv`` 的 ``[1,24,L,256]``。candidate = grouped bmm，baseline = HF eager
  （``repeat_kv`` + 4D matmul），两者都与 float64 reference 比较。阈值保持不变。
- **K（本轮新增）**：``fused_gqa_decode_attention``——split-K + online softmax 的
  单 Triton kernel。新增 case：
  * ``fused-vs-bmm``：L ∈ {1, 513, 1025, 8191} × {full, partial_mask,
    capacity_view, cap_mask}，candidate = fused，对照 = 当前 grouped bmm，
    两者都对 fp64 reference 算距离；
  * ``fused-fp8``：随机 ``float8_e4m3fn`` KV + per (token, kv_head) fp32 scale，
    覆盖 ``[L, H_KV]`` 与 ``[H_KV, L]`` 两种布局；reference 用**同一份反量化后的
    KV** 走 fp64，量化误差本身只作 diagnostic；
  * ``micro-fused-L513`` / ``micro-fused-L8191``：三组粗粒度参数的 per-call ms +
    KV 有效带宽，并对照历史值（grouped bmm 0.1788 ms、repeat_kv 0.8565 ms @8191）。

冻结的 gate（任务 A 的 7 个 case；实现前写下，不因候选结果调整）：
- 候选 / baseline 的输入、mask、softmax 语义完全一致；
- 主判据（相对）：候选到 float64 的 nrmse 不超过 baseline（repeat_kv + matmul）
  到 float64 nrmse 的 2 倍。紧凑路径只改变 reduction 分组，误差应与 baseline
  同量级；超过 2 倍说明 head 分组、mask 或 softmax 语义有实现问题，而不是
  fp reduction 顺序。
- 副判据（绝对 sanity）：候选 vs float64 reference（同样 q/k/v、显式 mask、
  softmax 全程 fp64）``nrmse <= 1e-2``、``cosine >= 0.9999``、
  ``max_rel <= 2e-2``。
- 候选 vs baseline 直接比较：``nrmse <= 5e-3``（同一个 op 只换 reduction 分组，
  两者应当几乎相同；这一条对 head 顺序/mask 错位最敏感）。
- 容差为什么不是更紧的 5e-3 / 0.99999：BF16 的误差底噪由“BF16 输出量化 +
  BF16 softmax”决定，而不是实现正确性。M3 实测 L=513 的 nrmse 是 3.87e-3
  （``bench/micro/attn_length_effect.py``）；本脚本在实现候选之前用 CPU 复跑同一
  baseline，5 个长度里最差是 L=1025 capacity view 的 5.48e-3 / cosine
  0.9999855，**冻结的 eager baseline 自己就会失败** 5e-3 / 0.99999。因此绝对
  判据放宽到约 2× 该底噪，真正定对错的是“≤2× baseline”和“候选 vs baseline”。
- ``max_rel = max|diff| / max|ref|``（与 M2 的 ``max_rel`` 同一公式）而不是绝对
  ``max_abs``：BF16 **输出**在 |out|≈3 时的 ulp 是 1.6e-2，L=1/2 这种 softmax
  极尖的档位连 baseline 的绝对 max_abs 都会超过 1e-2；全局相对量才是在
  “rms 正常但个别元素爆掉”与“BF16 输出量化”之间可分辨的判据。

冻结的 gate（任务 K 的 fused case；同样在实现前写下）：
- 绝对 sanity：``nrmse <= 1e-2``、``cosine >= 0.9999``、``max_rel <= 2e-2``
  （与任务 A 同一组值，理由同上：BF16 输出量化底噪）；
- **主判据（相对，1.05 倍）**：``fused 到 fp64 的 nrmse <= 1.05 × grouped_bmm
  到 fp64 的 nrmse``。online softmax 的 rescale 与一次性 softmax 归约顺序不同，
  期望 fused 略好（不做 score 的 bf16 往返），1.05 只允许极小的退化；
- mask 隔离：不可见 K/V 改写后输出 bitwise 不变；
- FP8：候选 vs「同一份反量化 KV 的 fp64 attention」用上面同一组绝对容差，且
  ``<= 1.05 ×``（同一份反量化 KV 的 bf16 grouped bmm）到 fp64 的距离；
  FP8 vs BF16 的量化误差只作 diagnostic（plan §5 不要求两种格式 trace 相同）。

coverage（任务 A，每档 L ∈ {1, 2, 513, 1025, 8191}）：
- ``full``：K/V 长度 L、全可见；
- ``partial_mask``：K/V 是 8192 capacity buffer 的 ``[:, :, :L]`` view，加性 mask
  只让前缀 ``visible(L)`` 可见（L>513 时 visible=513，模拟真实长上下文）；
- ``capacity_view``：同一个 capacity buffer 的前缀 view、不加 mask（模拟 static
  cache 的非连续 batch stride）；
- ``cap_mask``：K/V 长度 = capacity 8192、只让前 ``min(L, 513)`` 个 key 可见，
  即真实 ``static-cap`` decode 的 shape（L=8191 就是 7680+511 那条路径）。
任务 K 的 ``fused-vs-bmm`` 用同一组长度与变体（不含 L=2，边界由 L=1 与任务 A 的
L=2 覆盖）。

microbench：L=513 / 8191，K/V 取 8192 capacity buffer 的前缀 view；任务 A 测
baseline（repeat_kv + matmul）与 grouped bmm；任务 K 测 fused 的三组粗粒度参数
（``decode_fused.CONFIGS``）与 KV 有效带宽（该长度 KV 字节 / 时间）。
provenance：job worktree 里的 ``.venv`` / ``models`` 是执行器建的符号链接
（指向主仓库环境），而 ``.gitignore`` 的 ``.venv/`` / ``models/`` 带斜杠只匹配
目录，符号链接不算目录，``git status --porcelain`` 会把它们当成 untracked。
``runner.experiment.op_artifact._git_state`` 用 untracked 判断 dirty，会把这种
执行器环境误判成源码脏。这里给本进程的 git 注入一个临时 ``core.excludesFile``
（``GIT_CONFIG_*``，不改仓库、不改共享的 ``.git/info/exclude``），使 dirty 的
口径与 ``runner.report.git_state`` 一致：只看源码/文档/测试的改动。

整份脚本在 GPU 上约 2–4 分钟（任务 K 新增 16 个 fused 组合 + 6 个配置的首次
Triton 编译）。

用法：
    .venv/bin/python scripts/phase2_attn_check.py --out {output}/op
    .venv/bin/python scripts/phase2_attn_check.py --only L8191 micro-L8191 --out ...
    .venv/bin/python scripts/phase2_attn_check.py --only fused-vs-bmm fused-fp8 \
        micro-fused-L8191 --out ...
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
    FUSED_CONFIGS,
    FUSED_DEFAULT_CONFIG,
    compact_gqa_decode_attention,
    eager_gqa_decode_attention,
    fused_gqa_decode_attention,
    kernel_plan,
    reference_fp64,
)

H_Q = 24
H_KV = 4
HEAD_DIM = 256
SCALING = HEAD_DIM**-0.5
CAPACITY = 8192
LENGTHS = (1, 2, 513, 1025, 8191)
BENCH_LENGTHS = (513, 8191)

# ---- 冻结的数值容差（实现候选之前确定；理由见模块 docstring）--------------------
TOL_NRMSE = 1e-2          # 绝对 sanity：约 2x 冻结 baseline 的 BF16 误差底噪
TOL_COSINE = 0.9999
TOL_MAX_REL = 2e-2
TOL_BASELINE_RATIO = 2.0  # 任务 A 主判据：候选不超过 baseline 到 fp64 距离的 2 倍
TOL_DIRECT_NRMSE = 5e-3   # 任务 A：候选 vs baseline 直接比较
# 任务 K：fused 到 fp64 的距离不超过 grouped_bmm 到 fp64 距离的 1.05 倍（主判据）。
TOL_FUSED_BASELINE_RATIO = 1.05

# 任务 K 的覆盖：长度 × 变体（与任务 A 的变体定义一致）。
FUSED_LENGTHS = (1, 513, 1025, 8191)
FUSED_VARIANTS = ("full", "partial_mask", "capacity_view", "cap_mask")
FUSED_SAMPLES = 2
FUSED_FP8_LENGTHS = (513, 8191)
FUSED_FP8_LAYOUTS = ("token_major", "kv_major")
FP8_MAX = 448.0

# microbench 的历史对照值（同一 block 里只做 diagnostic/倍率，不参与 gate）。
HISTORICAL_MS = {
    513: {"grouped_bmm": 0.0431, "repeat_kv": 0.0527},
    8191: {"grouped_bmm": 0.1788, "repeat_kv": 0.8565},
}

# 每个 case 两个独立随机样本，取最差值，避免单个幸运样本。
SAMPLES = 2
SEED = 20261023
# microbench：warmup 后逐次 CUDA event 计时，取中位数；两个长度分别给迭代数。
BENCH_WARMUP = 20
BENCH_ITERS = {513: 200, 8191: 60}
MASK_VISIBLE_LONG = 513

# 必须是可 AST 字面量求值的 tuple；case 名就是产物里的 case 名。
SCENARIOS = (
    "L1",
    "L2",
    "L513",
    "L1025",
    "L8191",
    "micro-L513",
    "micro-L8191",
    "fused-vs-bmm",
    "fused-fp8",
    "micro-fused-L513",
    "micro-fused-L8191",
)

CANDIDATE_NAME = "attn-fused-splitk-bf16"
CANDIDATE_FORMAT = "bf16-kv"


# ------------------------------------------------------------------ 输入构造
def visible_keys(length: int) -> int:
    """partial_mask 变体的可见前缀长度：至少 1，且 L>2 时至少遮住一个 key。"""
    if length <= 2:
        return 1
    if length <= MASK_VISIBLE_LONG:
        return length - 1
    return MASK_VISIBLE_LONG


def _randn(shape, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)


def make_q(length: int, seed: int, device: torch.device) -> torch.Tensor:
    return _randn((1, H_Q, 1, HEAD_DIM), seed, device)


def make_kv(length: int, seed: int, device: torch.device, *, capacity: int | None = None):
    """返回 (k, v)；``capacity`` 给定时返回 capacity buffer 的前缀 view。"""
    if capacity is None:
        return _randn((1, H_KV, length, HEAD_DIM), seed, device), _randn(
            (1, H_KV, length, HEAD_DIM), seed + 1, device
        )
    k_buf = _randn((1, H_KV, capacity, HEAD_DIM), seed, device)
    v_buf = _randn((1, H_KV, capacity, HEAD_DIM), seed + 1, device)
    return k_buf[:, :, :length], v_buf[:, :, :length]


def additive_mask(length: int, visible: int, device: torch.device) -> torch.Tensor:
    mask = torch.full(
        (1, 1, 1, length),
        torch.finfo(torch.bfloat16).min,
        device=device,
        dtype=torch.bfloat16,
    )
    mask[..., :visible] = 0
    return mask


def kv_bytes(length: int) -> int:
    """候选实际读取的 K+V 字节（每层）：2 * H_KV * L * head_dim * 2。"""
    return 2 * H_KV * length * HEAD_DIM * 2


# ------------------------------------------------------------------ 数值
def numeric_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """候选 / 参考的 nrmse、cosine、max_rel、max_abs，全部在 float64 下计算。"""
    cand = candidate.double().flatten()
    ref = reference.double().flatten()
    diff = cand - ref
    ref_norm = torch.linalg.vector_norm(ref)
    cand_norm = torch.linalg.vector_norm(cand)
    ref_absmax = float(ref.abs().max().item())
    nrmse = float((torch.linalg.vector_norm(diff) / ref_norm).item())
    cosine = float((torch.dot(cand, ref) / (cand_norm * ref_norm)).item())
    max_abs = float(diff.abs().max().item())
    max_rel = max_abs / max(ref_absmax, 1e-30)
    return {
        "nrmse": nrmse,
        "cosine": cosine,
        "max_rel": max_rel,
        "max_abs": max_abs,
        "ref_absmax": ref_absmax,
    }


def _worst(acc: dict[str, float], metrics: dict[str, float]) -> None:
    acc["nrmse"] = max(acc.get("nrmse", 0.0), metrics["nrmse"])
    acc["cosine"] = min(acc.get("cosine", 1.0), metrics["cosine"])
    acc["max_rel"] = max(acc.get("max_rel", 0.0), metrics["max_rel"])
    acc["max_abs"] = max(acc.get("max_abs", 0.0), metrics["max_abs"])


def _variant_gates(
    case,
    variant: str,
    cand_worst: dict[str, float],
    base_nrmse_worst: float,
    direct_worst: dict[str, float],
    samples: int,
) -> None:
    """一个输入变体的一组 gate：候选 vs fp64 + 候选 vs baseline + 相对 baseline。"""
    detail = {
        "candidate_vs_fp64_worst": cand_worst,
        "candidate_vs_baseline_worst": direct_worst,
        "baseline_nrmse_worst": base_nrmse_worst,
        "samples": samples,
        "tol_nrmse": TOL_NRMSE,
        "tol_cosine": TOL_COSINE,
        "tol_max_rel": TOL_MAX_REL,
        "tol_baseline_ratio": TOL_BASELINE_RATIO,
        "tol_direct_nrmse": TOL_DIRECT_NRMSE,
    }
    case.gate(f"numeric[{variant}]: nrmse <= {TOL_NRMSE}", cand_worst["nrmse"] <= TOL_NRMSE, detail)
    case.gate(
        f"numeric[{variant}]: cosine >= {TOL_COSINE}",
        cand_worst["cosine"] >= TOL_COSINE,
        {"cosine": cand_worst["cosine"]},
    )
    case.gate(
        f"numeric[{variant}]: max_rel <= {TOL_MAX_REL}",
        cand_worst["max_rel"] <= TOL_MAX_REL,
        {"max_rel": cand_worst["max_rel"], "max_abs": cand_worst["max_abs"]},
    )
    ratio = cand_worst["nrmse"] / max(base_nrmse_worst, 1e-12)
    case.gate(
        f"numeric[{variant}]: candidate <= {TOL_BASELINE_RATIO}x baseline (nrmse)",
        ratio <= TOL_BASELINE_RATIO,
        {"ratio": ratio, "candidate_nrmse": cand_worst["nrmse"], "baseline_nrmse": base_nrmse_worst},
    )
    case.gate(
        f"compare[{variant}]: candidate vs baseline nrmse <= {TOL_DIRECT_NRMSE}",
        direct_worst["nrmse"] <= TOL_DIRECT_NRMSE,
        direct_worst,
    )


def run_numeric_case(report: OpReport, length: int, device: torch.device) -> None:
    visible = visible_keys(length)
    cap_visible = min(length, MASK_VISIBLE_LONG)
    case = report.case(
        f"L{length}",
        length=length,
        q_heads=H_Q,
        kv_heads=H_KV,
        groups=H_Q // H_KV,
        head_dim=HEAD_DIM,
        dtype="bfloat16",
        scaling=SCALING,
        capacity=CAPACITY,
        partial_mask_visible=visible,
        cap_mask_visible=cap_visible,
        variants=("full", "partial_mask", "capacity_view", "cap_mask"),
        samples=SAMPLES,
    )

    worst: dict[str, dict[str, float]] = {
        name: {} for name in ("full", "partial_mask", "capacity_view", "cap_mask")
    }
    baseline_worst = {name: 0.0 for name in worst}
    direct_worst: dict[str, dict[str, float]] = {name: {} for name in worst}
    structure = {"full_contiguous": True, "capacity_view_noncontiguous": True}
    mask_isolated = {"partial_mask": True, "cap_mask": True}
    variants_run: list[str] = []

    for sample in range(SAMPLES):
        seed = SEED + 1000 * length + 17 * sample
        q = make_q(length, seed, device)
        mask_len = additive_mask(length, visible, device)
        k_full, v_full = make_kv(length, seed + 1, device)
        k_view, v_view = make_kv(length, seed + 1, device, capacity=CAPACITY)
        k_cap, v_cap = make_kv(CAPACITY, seed + 1, device)
        mask_cap = additive_mask(CAPACITY, cap_visible, device)

        variants = {
            "full": (k_full, v_full, None),
            "partial_mask": (k_view, v_view, mask_len),
            "capacity_view": (k_view, v_view, None),
            "cap_mask": (k_cap, v_cap, mask_cap),
        }
        structure["full_contiguous"] &= bool(k_full.is_contiguous() and v_full.is_contiguous())
        structure["capacity_view_noncontiguous"] &= bool(
            not k_view.is_contiguous()
            and k_view.stride(1) == CAPACITY * HEAD_DIM
            and k_view.stride(0) == H_KV * CAPACITY * HEAD_DIM
        )

        for name, (k, v, mask) in variants.items():
            if name not in variants_run:
                variants_run.append(name)
            cand, _ = compact_gqa_decode_attention(q, k, v, mask, SCALING)
            base, _ = eager_gqa_decode_attention(q, k, v, mask, SCALING)
            ref = reference_fp64(q, k, v, mask, SCALING)
            _worst(worst[name], numeric_metrics(cand, ref))
            _worst(direct_worst[name], numeric_metrics(cand, base))
            base_metrics = numeric_metrics(base, ref)
            baseline_worst[name] = max(baseline_worst[name], base_metrics["nrmse"])

            case.diag(
                f"detail[{name}][sample {sample}]",
                True,
                {
                    "candidate": numeric_metrics(cand, ref),
                    "baseline": base_metrics,
                    "shape": tuple(cand.shape),
                    "dtype": str(cand.dtype),
                },
            )

        # mask 隔离：把不可见区改成不同的有限值，候选输出必须 bitwise 不变。
        for name, k, v, mask, vis in (
            ("partial_mask", k_view, v_view, mask_len, visible),
            ("cap_mask", k_cap, v_cap, mask_cap, cap_visible),
        ):
            k_junk = k.clone()
            v_junk = v.clone()
            k_junk[:, :, vis:] = 4.0
            v_junk[:, :, vis:] = -2.5
            out_a, _ = compact_gqa_decode_attention(q, k, v, mask, SCALING)
            out_b, _ = compact_gqa_decode_attention(q, k_junk, v_junk, mask, SCALING)
            mask_isolated[name] = mask_isolated[name] and bool(torch.equal(out_a, out_b))

    # ---- gate：coverage + 结构 + 每个变体的数值容差
    all_ran = set(variants_run) == {"full", "partial_mask", "capacity_view", "cap_mask"}
    case.gate(
        f"coverage: L={length} full/partial_mask/capacity_view/cap_mask "
        f"(capacity={CAPACITY}, visible={visible}, cap_visible={cap_visible})",
        all_ran,
        {
            "variants": variants_run,
            "samples": SAMPLES,
            "lengths": list(LENGTHS),
            "bench_lengths": list(BENCH_LENGTHS),
        },
    )
    case.gate(
        "structure: capacity_view 是非连续 K/V 前缀 view",
        structure["capacity_view_noncontiguous"],
        structure,
    )
    case.gate(
        "structure: full 变体 K/V 连续",
        structure["full_contiguous"],
        {"full_contiguous": structure["full_contiguous"]},
    )
    case.gate(
        "mask: 不可见 K/V 改写后输出 bitwise 不变",
        mask_isolated["partial_mask"] and mask_isolated["cap_mask"],
        {
            "partial_mask_visible": visible,
            "cap_mask_visible": cap_visible,
            "bitwise": mask_isolated,
        },
    )

    for name in ("full", "partial_mask", "capacity_view", "cap_mask"):
        _variant_gates(case, name, worst[name], baseline_worst[name], direct_worst[name], SAMPLES)

    case.metric("nrmse_worst", max(item["nrmse"] for item in worst.values()))
    case.metric("cosine_min", min(item["cosine"] for item in worst.values()))
    case.metric("max_rel_worst", max(item["max_rel"] for item in worst.values()))
    case.metric("max_abs_worst", max(item["max_abs"] for item in worst.values()))
    case.metric("baseline_nrmse_worst", max(baseline_worst.values()))
    case.metric(
        "nrmse_ratio_worst",
        max(
            worst[name]["nrmse"] / max(baseline_worst[name], 1e-12)
            for name in worst
        ),
    )
    case.metric(
        "direct_nrmse_worst",
        max(item["nrmse"] for item in direct_worst.values()),
    )
    case.metric("samples", float(SAMPLES))


# ------------------------------------------------------------------ microbench
def time_per_call_ms(fn, *, warmup: int, iters: int, device: torch.device) -> dict[str, float]:
    """逐次 CUDA event 计时：返回中位数 / 最小值 / p10（毫秒）。"""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for index in range(iters):
            starts[index].record()
            fn()
            ends[index].record()
        torch.cuda.synchronize(device)
    times = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    count = len(times)
    return {
        "median_ms": times[count // 2],
        "min_ms": times[0],
        "p10_ms": times[max(0, count // 10)],
    }


def run_micro_case(report: OpReport, length: int, device: torch.device) -> None:
    case = report.case(
        f"micro-L{length}",
        length=length,
        q_heads=H_Q,
        kv_heads=H_KV,
        groups=H_Q // H_KV,
        head_dim=HEAD_DIM,
        dtype="bfloat16",
        capacity=CAPACITY,
        warmup=BENCH_WARMUP,
        iters=BENCH_ITERS[length],
        timing="per-call CUDA events, median",
    )
    seed = SEED + 7 * length
    q = make_q(length, seed, device)
    k, v = make_kv(length, seed + 1, device, capacity=CAPACITY)

    def candidate_call():
        return compact_gqa_decode_attention(q, k, v, None, SCALING)

    def baseline_call():
        return eager_gqa_decode_attention(q, k, v, None, SCALING)

    with torch.no_grad():
        cand, _ = candidate_call()
        base, _ = baseline_call()
        direct = numeric_metrics(cand, base)

    cand_time = time_per_call_ms(
        candidate_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length], device=device
    )
    base_time = time_per_call_ms(
        baseline_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length], device=device
    )
    read_bytes = kv_bytes(length)
    cand_gbps = read_bytes / (cand_time["median_ms"] * 1e-3) / 1e9
    base_gbps = read_bytes / (base_time["median_ms"] * 1e-3) / 1e9
    speedup = base_time["median_ms"] / cand_time["median_ms"]

    case.gate(
        f"coverage: L={length} capacity-view 输入 + baseline/candidate 各 {BENCH_ITERS[length]} 次计时",
        True,
        {
            "length": length,
            "capacity": CAPACITY,
            "warmup": BENCH_WARMUP,
            "iters": BENCH_ITERS[length],
        },
    )
    case.gate(
        f"numeric: candidate vs baseline nrmse <= {TOL_DIRECT_NRMSE}",
        direct["nrmse"] <= TOL_DIRECT_NRMSE,
        direct,
    )
    case.gate(
        "timing: 计时样本有效（中位数有限且 > 0）",
        cand_time["median_ms"] > 0.0
        and base_time["median_ms"] > 0.0
        and math.isfinite(cand_time["median_ms"])
        and math.isfinite(base_time["median_ms"]),
        {"candidate": cand_time, "baseline": base_time},
    )
    case.gate(
        "timing: 候选读取的 KV 字节全部有限",
        bool(torch.isfinite(cand).all().item()),
        {"candidate_finite": True},
    )

    case.diag(
        "倍率与带宽",
        True,
        {
            "speedup_x": speedup,
            "candidate_kv_gbps": cand_gbps,
            "baseline_kv_gbps_same_formula": base_gbps,
            "kv_bytes_read_per_call": read_bytes,
            "note": "baseline 实际流量还包含 repeat_kv 的 6 倍展开写 + 读，"
            "同一公式的 baseline 带宽不是它的真实带宽",
        },
    )

    case.metric("candidate_ms", cand_time["median_ms"])
    case.metric("candidate_min_ms", cand_time["min_ms"])
    case.metric("candidate_p10_ms", cand_time["p10_ms"])
    case.metric("baseline_ms", base_time["median_ms"])
    case.metric("baseline_min_ms", base_time["min_ms"])
    case.metric("speedup", speedup)
    case.metric("candidate_kv_gbps", cand_gbps)
    case.metric("baseline_kv_gbps_same_formula", base_gbps)
    case.metric("nrmse_vs_baseline", direct["nrmse"])
    case.metric("kv_bytes_read_per_call", float(read_bytes))


# ------------------------------------------------------- 任务 K：fused vs grouped bmm
def run_fused_vs_bmm_case(report: OpReport, device: torch.device) -> None:
    """fused kernel 与当前 grouped bmm 的逐变体对拍（主判据 1.05x，见 docstring）。"""
    plan = {length: kernel_plan(length) for length in FUSED_LENGTHS}
    case = report.case(
        "fused-vs-bmm",
        lengths=list(FUSED_LENGTHS),
        variants=list(FUSED_VARIANTS),
        samples=FUSED_SAMPLES,
        capacity=CAPACITY,
        dtype="bfloat16",
        head_dim=HEAD_DIM,
        groups=H_Q // H_KV,
        scaling=SCALING,
        split_plan={str(k): v for k, v in plan.items()},
        tolerances={
            "nrmse_vs_fp64": TOL_NRMSE,
            "cosine": TOL_COSINE,
            "max_rel": TOL_MAX_REL,
            "fused_vs_bmm_fp64_ratio": TOL_FUSED_BASELINE_RATIO,
        },
    )

    keys = {(length, name): {} for length in FUSED_LENGTHS for name in FUSED_VARIANTS}
    bmm_nrmse = {(length, name): 0.0 for length in FUSED_LENGTHS for name in FUSED_VARIANTS}
    structure = {"full_contiguous": True, "capacity_view_noncontiguous": True}
    mask_isolated = {"partial_mask": True, "cap_mask": True}
    combos: list[str] = []
    plans_seen: dict[str, dict] = {}

    for length in FUSED_LENGTHS:
        visible = visible_keys(length)
        cap_visible = min(length, MASK_VISIBLE_LONG)
        for sample in range(FUSED_SAMPLES):
            seed = SEED + 2000 * length + 31 * sample
            q = make_q(length, seed, device)
            mask_len = additive_mask(length, visible, device)
            k_full, v_full = make_kv(length, seed + 1, device)
            k_view, v_view = make_kv(length, seed + 1, device, capacity=CAPACITY)
            k_cap, v_cap = make_kv(CAPACITY, seed + 1, device)
            mask_cap = additive_mask(CAPACITY, cap_visible, device)
            variants = {
                "full": (k_full, v_full, None),
                "partial_mask": (k_view, v_view, mask_len),
                "capacity_view": (k_view, v_view, None),
                "cap_mask": (k_cap, v_cap, mask_cap),
            }
            structure["full_contiguous"] &= bool(k_full.is_contiguous() and v_full.is_contiguous())
            structure["capacity_view_noncontiguous"] &= bool(not k_view.is_contiguous())

            for name, (k, v, mask) in variants.items():
                tag = f"L{length}:{name}"
                if tag not in combos:
                    combos.append(tag)
                fused, weights = fused_gqa_decode_attention(q, k, v, mask, SCALING)
                bmm, _ = compact_gqa_decode_attention(q, k, v, mask, SCALING)
                ref = reference_fp64(q, k, v, mask, SCALING)
                _worst(keys[(length, name)], numeric_metrics(fused, ref))
                base_metrics = numeric_metrics(bmm, ref)
                bmm_nrmse[(length, name)] = max(bmm_nrmse[(length, name)], base_metrics["nrmse"])
                plans_seen[str(length)] = kernel_plan(length)
                case.diag(
                    f"detail[{tag}][sample {sample}]",
                    True,
                    {
                        "fused": numeric_metrics(fused, ref),
                        "bmm": base_metrics,
                        "fused_vs_bmm": numeric_metrics(fused, bmm),
                        "weights_is_none": weights is None,
                        "shape": tuple(fused.shape),
                        "dtype": str(fused.dtype),
                        "split_plan": kernel_plan(length),
                    },
                )

            # mask 隔离：不可见区改成不同有限值，fused 输出必须 bitwise 不变。
            for name, k, v, mask, vis in (
                ("partial_mask", k_view, v_view, mask_len, visible),
                ("cap_mask", k_cap, v_cap, mask_cap, cap_visible),
            ):
                k_junk = k.clone()
                v_junk = v.clone()
                k_junk[:, :, vis:] = 4.0
                v_junk[:, :, vis:] = -2.5
                out_a, _ = fused_gqa_decode_attention(q, k, v, mask, SCALING)
                out_b, _ = fused_gqa_decode_attention(q, k_junk, v_junk, mask, SCALING)
                mask_isolated[name] = mask_isolated[name] and bool(torch.equal(out_a, out_b))

    all_ran = len(combos) == len(FUSED_LENGTHS) * len(FUSED_VARIANTS)
    case.gate(
        f"coverage: {len(FUSED_LENGTHS)} 长度 x {len(FUSED_VARIANTS)} 变体"
        f"（capacity={CAPACITY}, 长度={list(FUSED_LENGTHS)}）",
        all_ran,
        {"combos": combos, "samples": FUSED_SAMPLES, "split_plan": plans_seen},
    )
    case.gate(
        "structure: capacity_view 是非连续 K/V 前缀 view",
        structure["capacity_view_noncontiguous"],
        structure,
    )
    case.gate(
        "structure: full 变体 K/V 连续",
        structure["full_contiguous"],
        {"full_contiguous": structure["full_contiguous"]},
    )
    case.gate(
        "mask: fused 对不可见 K/V 改写 bitwise 不变",
        mask_isolated["partial_mask"] and mask_isolated["cap_mask"],
        {"bitwise": mask_isolated},
    )

    ratio_worst = 0.0
    nrmse_worst = 0.0
    cosine_min = 1.0
    max_rel_worst = 0.0
    finite_ok = True
    for length in FUSED_LENGTHS:
        for name in FUSED_VARIANTS:
            worst = keys[(length, name)]
            tag = f"L{length}:{name}"
            ratio = worst["nrmse"] / max(bmm_nrmse[(length, name)], 1e-12)
            ratio_worst = max(ratio_worst, ratio)
            nrmse_worst = max(nrmse_worst, worst["nrmse"])
            cosine_min = min(cosine_min, worst["cosine"])
            max_rel_worst = max(max_rel_worst, worst["max_rel"])
            finite_ok = finite_ok and math.isfinite(worst["nrmse"])
            case.gate(
                f"numeric[{tag}]: fused 到 fp64 的 nrmse <= {TOL_FUSED_BASELINE_RATIO}x "
                f"grouped_bmm（主判据）",
                ratio <= TOL_FUSED_BASELINE_RATIO,
                {
                    "ratio": ratio,
                    "fused_nrmse": worst["nrmse"],
                    "bmm_nrmse": bmm_nrmse[(length, name)],
                },
            )
            case.gate(
                f"numeric[{tag}]: nrmse <= {TOL_NRMSE}",
                worst["nrmse"] <= TOL_NRMSE,
                worst,
            )
            case.gate(
                f"numeric[{tag}]: cosine >= {TOL_COSINE}",
                worst["cosine"] >= TOL_COSINE,
                {"cosine": worst["cosine"]},
            )
            case.gate(
                f"numeric[{tag}]: max_rel <= {TOL_MAX_REL}",
                worst["max_rel"] <= TOL_MAX_REL,
                {"max_rel": worst["max_rel"]},
            )
    case.gate("numeric: fused 输出全部有限", finite_ok, {"finite": finite_ok})

    case.metric("nrmse_worst", nrmse_worst)
    case.metric("cosine_min", cosine_min)
    case.metric("max_rel_worst", max_rel_worst)
    case.metric("fused_vs_bmm_fp64_ratio_worst", ratio_worst)
    case.metric("bmm_nrmse_worst", max(bmm_nrmse.values()))
    case.metric("samples", float(FUSED_SAMPLES))
    for length in FUSED_LENGTHS:
        case.metric(f"splits_L{length}", float(kernel_plan(length)["splits"]))


# --------------------------------------------------------------- 任务 K：FP8 KV 口径
def _fp8_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """per (token, kv_head) 的 e4m3 量化：``s = amax/448``，返回 (fp8, [1,H_KV,L] scale)。"""
    amax = x.float().abs().amax(dim=-1)  # [1, H_KV, L]
    scale = (amax / FP8_MAX).clamp_min(1e-8)
    quant = (x.float() / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quant, scale


def run_fused_fp8_case(report: OpReport, device: torch.device) -> None:
    """随机 FP8 KV + per-token scale 的数值 gate（接口就绪证据，不含性能）。"""
    case = report.case(
        "fused-fp8",
        lengths=list(FUSED_FP8_LENGTHS),
        layouts=list(FUSED_FP8_LAYOUTS),
        samples=1,
        dtype="float8_e4m3fn",
        scale_layout="[tokens, kv_heads] 或 [kv_heads, tokens]",
        scale_granularity="per (layer, kv_head, token)",
        head_dim=HEAD_DIM,
        groups=H_Q // H_KV,
        scaling=SCALING,
        tolerances={
            "nrmse_vs_dequant_fp64": TOL_NRMSE,
            "cosine": TOL_COSINE,
            "max_rel": TOL_MAX_REL,
            "fp8_vs_bmm_dequant_ratio": TOL_FUSED_BASELINE_RATIO,
        },
    )
    combos: list[str] = []
    worst = {}
    base_nrmse = {}
    for length in FUSED_FP8_LENGTHS:
        seed = SEED + 4000 * length
        q = make_q(length, seed, device)
        k_bf, v_bf = make_kv(length, seed + 1, device)
        k8, k_scale = _fp8_quantize(k_bf)
        v8, v_scale = _fp8_quantize(v_bf)
        # 反量化用 fp32（与 kernel 的 (k.float() * k_scale) 语义一致）；
        # reference 对同一份反量化 KV 走 fp64。
        k_deq = k8.float() * k_scale[..., None]
        v_deq = v8.float() * v_scale[..., None]
        ref = reference_fp64(q, k_deq, v_deq, None, SCALING)
        # bmm 对照：把它能吃的 bf16 反量化 KV 喂进去（同一份数字的 bf16 舍入）。
        bmm, _ = compact_gqa_decode_attention(
            q, k_deq.to(q.dtype), v_deq.to(q.dtype), None, SCALING
        )
        bmm_metrics = numeric_metrics(bmm, ref)
        for layout in FUSED_FP8_LAYOUTS:
            k_scale_arg = k_scale[0].transpose(0, 1).contiguous()
            v_scale_arg = v_scale[0].transpose(0, 1).contiguous()
            if layout == "kv_major":
                k_scale_arg = k_scale_arg.transpose(0, 1).contiguous()
                v_scale_arg = v_scale_arg.transpose(0, 1).contiguous()
            fused, _ = fused_gqa_decode_attention(
                q, k8, v8, None, SCALING, k_scale_arg, v_scale_arg
            )
            metrics = numeric_metrics(fused, ref)
            tag = f"L{length}:{layout}"
            combos.append(tag)
            worst[tag] = metrics
            base_nrmse[tag] = bmm_metrics["nrmse"]
            # FP8 vs BF16 的量化误差：只作 diagnostic，不参与判定。
            case.diag(
                f"detail[{tag}]",
                True,
                {
                    "fused_vs_dequant_fp64": metrics,
                    "bf16_bmm_dequant_vs_fp64": bmm_metrics,
                    "fused_vs_bf16_kv": numeric_metrics(fused, bmm),
                    "k_scale_min": float(k_scale.min().item()),
                    "k_scale_max": float(k_scale.max().item()),
                    "scale_layout": layout,
                },
            )

    case.gate(
        f"coverage: {len(FUSED_FP8_LENGTHS)} 长度 x {len(FUSED_FP8_LAYOUTS)} scale 布局"
        f"（fp8 e4m3, per token/head scale）",
        len(combos) == len(FUSED_FP8_LENGTHS) * len(FUSED_FP8_LAYOUTS),
        {"combos": combos, "layouts": list(FUSED_FP8_LAYOUTS)},
    )
    ratio_worst = 0.0
    for tag, metrics in worst.items():
        ratio = metrics["nrmse"] / max(base_nrmse[tag], 1e-12)
        ratio_worst = max(ratio_worst, ratio)
        case.gate(
            f"numeric[{tag}]: fp8 fused 到反量化 fp64 的 nrmse <= {TOL_NRMSE}",
            metrics["nrmse"] <= TOL_NRMSE,
            metrics,
        )
        case.gate(f"numeric[{tag}]: cosine >= {TOL_COSINE}", metrics["cosine"] >= TOL_COSINE, metrics)
        case.gate(f"numeric[{tag}]: max_rel <= {TOL_MAX_REL}", metrics["max_rel"] <= TOL_MAX_REL, metrics)
        case.gate(
            f"numeric[{tag}]: fp8 fused <= {TOL_FUSED_BASELINE_RATIO}x bf16 grouped_bmm"
            f"（同一份反量化 KV，主判据）",
            ratio <= TOL_FUSED_BASELINE_RATIO,
            {"ratio": ratio, "fused_nrmse": metrics["nrmse"], "bmm_nrmse": base_nrmse[tag]},
        )
    case.metric("nrmse_worst", max(m["nrmse"] for m in worst.values()))
    case.metric("cosine_min", min(m["cosine"] for m in worst.values()))
    case.metric("max_rel_worst", max(m["max_rel"] for m in worst.values()))
    case.metric("fp8_vs_bmm_dequant_ratio_worst", ratio_worst)


# --------------------------------------------------- 任务 K：fused microbench + 参数
def run_micro_fused_case(report: OpReport, length: int, device: torch.device) -> None:
    """fused 的三组粗粒度参数 + grouped bmm / repeat_kv 的 per-call ms 与带宽。"""
    case = report.case(
        f"micro-fused-L{length}",
        length=length,
        q_heads=H_Q,
        kv_heads=H_KV,
        groups=H_Q // H_KV,
        head_dim=HEAD_DIM,
        dtype="bfloat16",
        capacity=CAPACITY,
        configs=list(FUSED_CONFIGS),
        default_config=FUSED_DEFAULT_CONFIG,
        split_plan={name: kernel_plan(length, config=name) for name in FUSED_CONFIGS},
        warmup=BENCH_WARMUP,
        iters=BENCH_ITERS[length],
        timing="per-call CUDA events, median",
    )
    seed = SEED + 7 * length
    q = make_q(length, seed, device)
    k, v = make_kv(length, seed + 1, device, capacity=CAPACITY)
    read_bytes = kv_bytes(length)
    historical = HISTORICAL_MS[length]

    def bmm_call():
        return compact_gqa_decode_attention(q, k, v, None, SCALING)

    def eager_call():
        return eager_gqa_decode_attention(q, k, v, None, SCALING)

    with torch.no_grad():
        bmm_out, _ = bmm_call()
        eager_out, _ = eager_call()
        timings = {}
        for name in FUSED_CONFIGS:
            def fused_call(name=name):
                return fused_gqa_decode_attention(q, k, v, None, SCALING, config=name)

            out, _ = fused_call()
            timings[name] = {
                "time": time_per_call_ms(
                    fused_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length], device=device
                ),
                "nrmse_vs_bmm": numeric_metrics(out, bmm_out)["nrmse"],
                "finite": bool(torch.isfinite(out).all().item()),
                "split_plan": kernel_plan(length, config=name),
                "gpbs": None,
            }
            timings[name]["gpbs"] = read_bytes / (timings[name]["time"]["median_ms"] * 1e-3) / 1e9
        bmm_time = time_per_call_ms(
            bmm_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length], device=device
        )
        eager_time = time_per_call_ms(
            eager_call, warmup=BENCH_WARMUP, iters=BENCH_ITERS[length], device=device
        )
    bmm_gpbs = read_bytes / (bmm_time["median_ms"] * 1e-3) / 1e9
    eager_gpbs = read_bytes / (eager_time["median_ms"] * 1e-3) / 1e9

    case.gate(
        f"coverage: L={length} capacity-view 输入，{len(FUSED_CONFIGS)} 组参数 x "
        f"{BENCH_ITERS[length]} 次计时 + grouped bmm / repeat_kv 对照",
        True,
        {
            "length": length,
            "capacity": CAPACITY,
            "configs": list(FUSED_CONFIGS),
            "iters": BENCH_ITERS[length],
            "warmup": BENCH_WARMUP,
        },
    )
    timing_ok = all(
        entry["time"]["median_ms"] > 0.0 and math.isfinite(entry["time"]["median_ms"])
        for entry in timings.values()
    ) and bmm_time["median_ms"] > 0.0
    case.gate(
        "timing: 计时样本有效（中位数有限且 > 0）",
        timing_ok,
        {
            "fused": {name: entry["time"] for name, entry in timings.items()},
            "grouped_bmm": bmm_time,
            "repeat_kv": eager_time,
        },
    )
    # 数值 sanity：fused 与 grouped bmm 对同一份输入的距离，用任务 K 的绝对判据
    # （主判据 1.05x 在 fused-vs-bmm case 里对 fp64 判定）。
    for name, entry in timings.items():
        case.gate(
            f"numeric[{name}]: fused vs grouped bmm nrmse <= {TOL_NRMSE}",
            entry["nrmse_vs_bmm"] <= TOL_NRMSE and entry["finite"],
            {"nrmse": entry["nrmse_vs_bmm"], "finite": entry["finite"]},
        )

    default_name = FUSED_DEFAULT_CONFIG
    default = timings[default_name]
    case.diag(
        "per-call ms / KV 有效带宽 / 历史对照",
        True,
        {
            "configs": {
                name: {
                    "median_ms": entry["time"]["median_ms"],
                    "min_ms": entry["time"]["min_ms"],
                    "kwargs": FUSED_CONFIGS[name],
                    "split_plan": entry["split_plan"],
                    "kv_gbps": entry["gpbs"],
                    "speedup_vs_grouped_bmm": bmm_time["median_ms"] / entry["time"]["median_ms"],
                    "nrmse_vs_bmm": entry["nrmse_vs_bmm"],
                }
                for name, entry in timings.items()
            },
            "grouped_bmm_ms": bmm_time["median_ms"],
            "grouped_bmm_kv_gbps_same_formula": bmm_gpbs,
            "repeat_kv_ms": eager_time["median_ms"],
            "repeat_kv_kv_gbps_same_formula": eager_gpbs,
            "historical_ms": historical,
            "default_vs_historical_grouped_bmm": historical["grouped_bmm"]
            / default["time"]["median_ms"],
            "default_vs_historical_repeat_kv": historical["repeat_kv"]
            / default["time"]["median_ms"],
            "kv_bytes_read_per_call": read_bytes,
            "note": "历史值来自第 1 轮 job-000016（同一容量 view、同一公式）；"
            "repeat_kv 的同公式带宽不是它的真实带宽（还多 6 倍展开写 + 读）。",
        },
    )
    case.metric("default_ms", default["time"]["median_ms"])
    case.metric("default_min_ms", default["time"]["min_ms"])
    case.metric("default_kv_gbps", default["gpbs"])
    case.metric("best_ms", min(entry["time"]["median_ms"] for entry in timings.values()))
    case.metric("grouped_bmm_ms", bmm_time["median_ms"])
    case.metric("grouped_bmm_kv_gbps_same_formula", bmm_gpbs)
    case.metric("repeat_kv_ms", eager_time["median_ms"])
    case.metric("repeat_kv_kv_gbps_same_formula", eager_gpbs)
    case.metric("speedup_vs_grouped_bmm", bmm_time["median_ms"] / default["time"]["median_ms"])
    case.metric("speedup_vs_repeat_kv", eager_time["median_ms"] / default["time"]["median_ms"])
    case.metric("default_vs_historical_grouped_bmm", historical["grouped_bmm"] / default["time"]["median_ms"])
    case.metric("default_vs_historical_repeat_kv", historical["repeat_kv"] / default["time"]["median_ms"])
    case.metric("kv_bytes_read_per_call", float(read_bytes))
    for name, entry in timings.items():
        case.metric(f"ms_{name}", entry["time"]["median_ms"])
        case.metric(f"kv_gbps_{name}", entry["gpbs"])


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

    directory = Path(tempfile.mkdtemp(prefix="phase2-attn-git-"))
    excludes = directory / "exclude"
    excludes.write_text(".venv\nmodels\n", encoding="utf-8")
    os.environ["GIT_CONFIG_COUNT"] = "1"
    os.environ["GIT_CONFIG_KEY_0"] = "core.excludesFile"
    os.environ["GIT_CONFIG_VALUE_0"] = str(excludes)
    return excludes


def summarize(payload: dict) -> dict:
    """打印用摘要：``summary.cases`` 只有 gate 计数，metrics 在 ``cases`` 里。"""
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
            "backend": "fused",
            "backend_env": "PHASE2_ATTN_DECODE=fused|grouped_bmm|eager（默认 fused）",
            "fused_kernel": "runner/ops/attention/decode_fused.py",
            "fused_strategy": "split-K（每 kv head 一个 program group，SPLITS 段）+ online softmax"
            "（fp32、log2 域），二次 combine kernel 合并 partial",
            "fused_configs": {name: dict(spec) for name, spec in FUSED_CONFIGS.items()},
            "fused_default_config": FUSED_DEFAULT_CONFIG,
            "fused_split_plan": {str(L): kernel_plan(L) for L in FUSED_LENGTHS},
            "kv_dtype": "bfloat16（fused-fp8 case 另测 float8_e4m3fn + per-token scale）",
            "materializes_repeat_kv": False,
            "grouping": "q head 6j+i -> kv head j；一个 program 处理同 kv head 的 6 个 q head，K/V 只读一次",
            "patch": "transformers.models.qwen3_5.modeling_qwen3_5.eager_attention_forward",
            "q_len": "==1 (prefill falls back to eager)",
            "tolerances": {
                "nrmse_vs_fp64": TOL_NRMSE,
                "cosine": TOL_COSINE,
                "max_rel": TOL_MAX_REL,
                "baseline_ratio": TOL_BASELINE_RATIO,
                "direct_nrmse_vs_baseline": TOL_DIRECT_NRMSE,
                "fused_vs_bmm_fp64_ratio": TOL_FUSED_BASELINE_RATIO,
            },
        },
    )
    for name in selected:
        if name == "fused-vs-bmm":
            run_fused_vs_bmm_case(report, device)
        elif name == "fused-fp8":
            run_fused_fp8_case(report, device)
        elif name.startswith("micro-fused-L"):
            run_micro_fused_case(report, int(name[len("micro-fused-L") :]), device)
        elif name.startswith("micro-L"):
            run_micro_case(report, int(name[len("micro-L") :]), device)
        elif name.startswith("L"):
            run_numeric_case(report, int(name[1:]), device)
        else:  # pragma: no cover - select_cases 已拦截
            raise SystemExit(f"无法解析 case {name!r}")
    payload = report.finish(
        extra={
            "tolerances_frozen_before_candidate": {
                "nrmse_vs_fp64": TOL_NRMSE,
                "cosine": TOL_COSINE,
                "max_rel": TOL_MAX_REL,
                "baseline_ratio": TOL_BASELINE_RATIO,
                "direct_nrmse_vs_baseline": TOL_DIRECT_NRMSE,
                "fused_vs_bmm_fp64_ratio": TOL_FUSED_BASELINE_RATIO,
                "reason": "BF16 输出量化 + BF16 softmax 的误差底噪使冻结 eager baseline "
                "自身在 L=1025 达到 nrmse 5.48e-3 / cosine 0.9999855，故绝对判据取约 "
                "2x 底噪；主判据是候选 <= 2x baseline 与候选 vs baseline 直接比较。"
                "任务 K 的 fused 主判据在实现前冻结为 <= 1.05x grouped_bmm 到 fp64 的"
                "距离（online softmax 只改归约顺序，不应比一次性 softmax 差 5%）。",
            },
            "coverage": {
                "lengths": list(LENGTHS),
                "bench_lengths": list(BENCH_LENGTHS),
                "capacity": CAPACITY,
                "variants": ["full", "partial_mask", "capacity_view", "cap_mask"],
                "fused_lengths": list(FUSED_LENGTHS),
                "fused_variants": list(FUSED_VARIANTS),
                "fused_fp8_lengths": list(FUSED_FP8_LENGTHS),
                "fused_fp8_layouts": list(FUSED_FP8_LAYOUTS),
            },
            "microbench_historical_ms": {str(k): v for k, v in HISTORICAL_MS.items()},
        }
    )
    print(json.dumps(summarize(payload), indent=2, ensure_ascii=False))
    print(f"output: {args.out / 'op.json'}")
    return 0 if payload["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
