#!/usr/bin/env python3
"""Phase 2 任务 G：ShapeStableLinear 的 M=1 decode 路径 op gate + microbench。

背景（集成后 graph-7680 profile，`job-000028` 的 `kern_sum.csv`）：
- `in_proj_a/b` 每步 96 次（48 个 GDN 层 × 2），decode 时 M=1 被补到 M=128，
  再走 `cutlass_80_wmma_tensorop_bf16_32x32_128x2_tn`：**4.1–5.1 µs/次**，
  外加补零 `fill`（1.1 µs）和 `torch.cat`（1.6 µs）各一次 → 约 7.7 µs/次。
- 同 profile 里 `32x32_128x2_nn`（16 次/步）与 `32x32_64x1_tn`（16 次/步）是
  full-attention 的 QK/PV bmm，**不属于本任务**；Coordinator 引用的「128 次 /
  1.529 ms」是这三类 bf16 GEMM 的合计（128 = 96 + 16 + 16）。

候选：`ShapeStableLinear` 在 `flat.shape[0] == 1` 时改走不 padding 的 Triton
BF16 matvec（`runner/ops/bf16/stable_linear.py`）；M>1 的代码路径一行未改。

gate：
- 数值：候选 vs **冻结的原 M=128 padded 实现**（本脚本里的 `PaddedReference`，
  逐行照抄 commit `9e35873` 的 forward），GDN 冻结容差
  `nrmse <= 1e-3`、`cosine >= 0.99999`、`max_rel <= 5e-3`；
- `M>1` 路径 bitwise 未变：M ∈ {2,17,127,128,129,300,512} 与 3D 形态，
  候选输出必须和 `PaddedReference` 逐位相等；
- coverage：a/b 两类、48 个真实层、M=1 与 M=128 两种输入形态；
- finite；decode 分支**真的被执行**（活计数，避免接线失败也算通过）；
- microbench：同一 x/同一权重地址、graph replay 边界，扫 3 组粗粒度参数。

microbench 口径：
- 工作集会被 L2 吃掉（16 个 module ≈ 7.9 MB、96 个 ≈ 47 MB，GB203 L2 ≈ 64 MB），
  所以报两组：`warm`（16 module 单副本，L2-resident）与 `cold`（96 module × 2 份
  不同地址副本 ≈ 94 MB > L2）。真实 decode 里 14 GB 的 NVFP4 权重流会把 a/b
  权重挤出 L2，所以 **decisive 的是端到端 bench，不是这里的局部数字**；
- 计时边界：整段（16 或 192 次调用）一起进 CUDA graph，replay 20 次取每 replay
  中位数，再除以调用数 → 去掉了 Python/launch 开销，接近真实 decode。

用法：
    .venv/bin/python bench/micro/stable_linear_decode.py --out {output}/op
    .venv/bin/python bench/micro/stable_linear_decode.py --only microbench-graph-replay --out .tmp/op
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import runner.ops.bf16.stable_linear as stable_linear  # noqa: E402
from runner.experiment.op_artifact import OpReport  # noqa: E402
from runner.loader.checkpoint import CheckpointReader  # noqa: E402
from runner.loader.nvfp4_model import find_manifest  # noqa: E402
from runner.ops.nvfp4.reference import error_metrics  # noqa: E402
from runner.report import CHECKPOINT  # noqa: E402

SCENARIOS = (
    "decode-numeric-real-weights",
    "padded-path-bitwise-unchanged",
    "decode-patch-coverage",
    "microbench-graph-replay",
)
NUMERIC_CASE = SCENARIOS[0]
PADDED_CASE = SCENARIOS[1]
COVERAGE_CASE = SCENARIOS[2]
PERF_CASE = SCENARIOS[3]

HIDDEN = 5120
BLOCK_M = 128
NUM_LAYERS = 48  # GDN 层数（64 层里 layer_types 每 4 层一个 full attention）
# 真实 checkpoint 里 a/b 都是 [48, 5120]（num_v_heads=48）；派发消息里的
# “a=N16” 与 manifest 不符，这里以 manifest 为准并记录。
EXPECTED_SHAPE = (48, HIDDEN)

# docs/decisions.md 2026-10-21 冻结的 GDN 容差。
GDN_TOL = {"nrmse": 1e-3, "cosine": 0.99999, "max_rel": 5e-3}

SEED = 20261023
WARMUP = 5
GRAPH_WARMUP = 3
REPLAYS = 20
ROUNDS = 5
# 粗粒度参数组：(BLOCK_K, num_warps)。
CONFIGS = ((1024, 4), (2048, 4), (512, 4))
# 数值输入的两个尺度：0.5 是项目测试里的 hidden 量级，0.05 覆盖小值端。
SCALES = (0.5, 0.05)
PADDED_SHAPES = ((1, 2), (1, 17), (1, 127), (1, 128), (1, 129), (1, 300), (1, 512))


# --------------------------------------------------------------------- 数据
class PaddedReference:
    """冻结的旧实现：commit `9e35873` 的 ``ShapeStableLinear.forward``。

    逐行照抄（含补零 fill + ``torch.cat`` + 尾块截断），既做数值 reference，
    也做 M>1 的 bitwise reference 和计时 baseline。
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, block_m: int = BLOCK_M):
        self.weight = weight
        self.bias = bias
        self.block_m = block_m
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, self.in_features)
        rows = flat.shape[0]
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


def _entry_name(layer: int, kind: str) -> str:
    return f"model.language_model.layers.{layer}.linear_attn.in_proj_{kind}.weight"


def _load_real_modules(device: torch.device) -> tuple[list[dict], dict]:
    """从真实 checkpoint 读全部 96 个 in_proj_a/b 权重。"""
    manifest_path = find_manifest(ROOT)
    manifest = json.loads(Path(manifest_path).read_text())
    by_name = {str(item["name"]): item for item in manifest}
    reader = CheckpointReader(ROOT / CHECKPOINT)
    modules: list[dict] = []
    try:
        for layer in range(64):
            for kind in ("a", "b"):
                name = _entry_name(layer, kind)
                item = by_name.get(name)
                if item is None:
                    continue
                tensor = reader.get(name, "cpu").to(device)
                if tensor.dtype != torch.bfloat16:
                    raise RuntimeError(f"{name} dtype {tensor.dtype} != bfloat16")
                if tuple(tensor.shape) != EXPECTED_SHAPE:
                    raise RuntimeError(
                        f"{name} shape {tuple(tensor.shape)} != {EXPECTED_SHAPE}"
                    )
                modules.append(
                    {
                        "layer": layer,
                        "kind": kind,
                        "name": name,
                        "weight": tensor,
                        "bias": None,
                        "module": stable_linear.ShapeStableLinear(
                            tensor, None, block_m=BLOCK_M
                        ),
                        "reference": PaddedReference(tensor, None, block_m=BLOCK_M),
                    }
                )
    finally:
        reader.close()
    layers = sorted({entry["layer"] for entry in modules})
    info = {
        "manifest": str(manifest_path),
        "checkpoint": CHECKPOINT,
        "num_modules": len(modules),
        "num_layers": len(layers),
        "layers": layers,
        "num_a": sum(1 for entry in modules if entry["kind"] == "a"),
        "num_b": sum(1 for entry in modules if entry["kind"] == "b"),
        "shape": list(EXPECTED_SHAPE),
        "weight_bytes_each": EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1] * 2,
        "weight_bytes_total": len(modules) * EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1] * 2,
    }
    return modules, info


def _hidden(shape: tuple[int, ...], scale: float, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(SEED + int(scale * 1000))
    value = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    return (value * scale).to(torch.bfloat16)


# ------------------------------------------------------------------ 断言工具
def _tolerance_ok(metrics: dict[str, float]) -> bool:
    return (
        metrics["nrmse"] <= GDN_TOL["nrmse"]
        and metrics["cosine"] >= GDN_TOL["cosine"]
        and metrics["max_rel"] <= GDN_TOL["max_rel"]
        and bool(metrics["finite"])
    )


def _fp64_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = x.reshape(-1, weight.shape[1]).double()
    return flat @ weight.double().t()


# --------------------------------------------------------------------- 数值
def _run_numeric(report: OpReport, modules: list[dict], info: dict, device: torch.device) -> None:
    case = report.case(
        NUMERIC_CASE,
        module="runner/ops/bf16/stable_linear.py",
        weights="真实 checkpoint 的 96 个 in_proj_a/b",
        num_modules=info["num_modules"],
        num_layers=info["num_layers"],
        n_a=info["num_a"],
        n_b=info["num_b"],
        weight_shape=list(EXPECTED_SHAPE),
        dtype="bfloat16",
        input_shapes="M=1（[1,1,5120]）",
        scales=list(SCALES),
        reference="PaddedReference（commit 9e35873 的原 M=128 padded forward）",
        tolerance=GDN_TOL,
        candidate="Triton BF16 matvec（1 program/输出行，FP32 累加）",
    )
    case.gate(
        "coverage:real-checkpoint-a-and-b",
        info["num_modules"] == 96
        and info["num_layers"] == 48
        and info["num_a"] == 48
        and info["num_b"] == 48
        and tuple(EXPECTED_SHAPE) == (48, HIDDEN),
        {key: info[key] for key in ("num_modules", "num_layers", "num_a", "num_b", "shape")},
    )

    worst: dict[tuple[str, float], dict[str, object]] = {}
    for scale in SCALES:
        x = _hidden((1, 1, HIDDEN), scale, device)
        for kind in ("a", "b"):
            rows = {"nrmse": 0.0, "max_abs": 0.0, "max_rel": 0.0, "cosine": 1.0}
            nrmse_vs_fp64 = 0.0
            cand_vs_fp64 = 0.0
            finite_all = True
            worst_module = None
            for entry in modules:
                if entry["kind"] != kind:
                    continue
                ref = entry["reference"](x)
                got = entry["module"](x)
                metrics = error_metrics(got, ref)
                finite_all = finite_all and bool(metrics["finite"])
                if metrics["nrmse"] > float(rows["nrmse"]):
                    rows["nrmse"] = metrics["nrmse"]
                    worst_module = {
                        "layer": entry["layer"],
                        "kind": kind,
                        "nrmse": metrics["nrmse"],
                        "cosine": metrics["cosine"],
                        "max_rel": metrics["max_rel"],
                    }
                rows["max_abs"] = max(float(rows["max_abs"]), metrics["max_abs"])
                rows["max_rel"] = max(float(rows["max_rel"]), metrics["max_rel"])
                rows["cosine"] = min(float(rows["cosine"]), metrics["cosine"])
                # 双方各自到 fp64 的距离：证明两者都只是 bf16 rounding 级别。
                exact = _fp64_reference(x, entry["weight"])
                nrmse_vs_fp64 = max(
                    nrmse_vs_fp64, error_metrics(ref, exact)["nrmse"]
                )
                cand_vs_fp64 = max(
                    cand_vs_fp64, error_metrics(got, exact)["nrmse"]
                )
            key = (kind, scale)
            rows["finite"] = finite_all
            worst[key] = {
                **rows,
                "finite": finite_all,
                "worst_module": worst_module,
                "reference_vs_fp64_nrmse": nrmse_vs_fp64,
                "candidate_vs_fp64_nrmse": cand_vs_fp64,
            }
            case.gate(
                f"numeric:{kind}:M=1:scale{scale}:worst-of-48",
                _tolerance_ok(rows),
                {"worst": rows, "worst_module": worst_module, "tolerance": GDN_TOL},
            )
            case.gate(f"finite:{kind}:M=1:scale{scale}", finite_all, None)
            case.metric(f"nrmse_{kind}_scale{scale}", float(rows["nrmse"]))
            case.metric(f"cosine_{kind}_scale{scale}", float(rows["cosine"]))
            case.metric(f"max_rel_{kind}_scale{scale}", float(rows["max_rel"]))
            case.metric(f"ref_vs_fp64_nrmse_{kind}_scale{scale}", nrmse_vs_fp64)
            case.metric(f"cand_vs_fp64_nrmse_{kind}_scale{scale}", cand_vs_fp64)
    case.diag(
        "candidate-and-reference-are-same-distance-to-fp64",
        all(
            abs(float(item["candidate_vs_fp64_nrmse"]) - float(item["reference_vs_fp64_nrmse"]))
            <= 1e-4
            for item in worst.values()
        ),
        {
            f"{kind}-scale{scale}": {
                "reference": item["reference_vs_fp64_nrmse"],
                "candidate": item["candidate_vs_fp64_nrmse"],
            }
            for (kind, scale), item in worst.items()
        },
    )


# --------------------------------------------------- M>1 路径 bitwise 未变
def _run_padded(report: OpReport, modules: list[dict], info: dict, device: torch.device) -> None:
    case = report.case(
        PADDED_CASE,
        boundary="M>1 全部走原 block_m=128 padded 路径，必须与 PaddedReference 逐位相等",
        shapes=[list(shape) + [HIDDEN] for shape in PADDED_SHAPES],
        block_m=BLOCK_M,
        dtype="bfloat16",
        note="M=1 不在本 case（它属于 decode 分支）",
    )
    all_equal = True
    per_shape: dict[str, bool] = {}
    for shape in PADDED_SHAPES:
        x = _hidden(shape + (HIDDEN,), 0.5, device)
        ok = True
        first_bad = None
        for entry in modules:
            got = entry["module"](x)
            ref = entry["reference"](x)
            if not torch.equal(got, ref):
                ok = False
                first_bad = first_bad or {
                    "layer": entry["layer"],
                    "kind": entry["kind"],
                    "max_abs": float((got.float() - ref.float()).abs().max().item()),
                }
        per_shape["x".join(str(dim) for dim in shape)] = ok
        all_equal = all_equal and ok
        case.gate(
            f"bitwise:M={shape[-1]}:{'3d' if len(shape) == 2 else '2d'}",
            ok,
            {"shape": list(shape) + [HIDDEN], "first_bad": first_bad},
        )
    # 3D 形态（模型真实调用 [B, T, K]）单独一遍，避免只测 2D。
    x3d = _hidden((2, 150, HIDDEN), 0.5, device)
    ok3d = all(
        torch.equal(entry["module"](x3d), entry["reference"](x3d)) for entry in modules
    )
    all_equal = all_equal and ok3d
    case.gate("bitwise:3d-[2,150,5120]", ok3d, {"shape": [2, 150, HIDDEN]})
    case.gate(
        "coverage:m>1-shapes-and-3d",
        len(PADDED_SHAPES) >= 6 and len(per_shape) == len(PADDED_SHAPES),
        {"shapes": per_shape, "3d": ok3d, "modules": info["num_modules"]},
    )
    case.metric("padded_bitwise_all", float(all_equal))
    case.metric("padded_shapes", float(len(PADDED_SHAPES) + 1))


# ------------------------------------------------------- decode 分支 coverage
def _run_coverage(report: OpReport, modules: list[dict], info: dict, device: torch.device) -> None:
    case = report.case(
        COVERAGE_CASE,
        purpose="证明 decode 分支真的被执行、M>1 不走它、没有静默回退",
        n_modules=info["num_modules"],
        env_flag=stable_linear.ENV_FLAG,
        config=f"block_k={stable_linear.DECODE_BLOCK_K}, "
        f"num_warps={stable_linear.DECODE_NUM_WARPS}",
    )
    x1 = _hidden((1, 1, HIDDEN), 0.5, device)
    x128 = _hidden((1, 128, HIDDEN), 0.5, device)

    stable_linear.reset_decode_stats()
    for entry in modules:
        entry["module"](x1)
    m1_stats = stable_linear.decode_stats()
    case.gate(
        "patch:decode-path-executed",
        int(m1_stats["decode_calls"]) == len(modules)
        and not m1_stats["fallback_reasons"],
        {
            "decode_calls": m1_stats["decode_calls"],
            "expected": len(modules),
            "fallbacks": m1_stats["fallback_reasons"],
            "padded_calls": m1_stats["padded_calls"],
        },
    )
    case.gate(
        "patch:no-silent-fallback",
        not m1_stats["fallback_reasons"] and m1_stats["kernel_error"] is None,
        {"fallbacks": m1_stats["fallback_reasons"], "kernel_error": m1_stats["kernel_error"]},
    )

    stable_linear.reset_decode_stats()
    for entry in modules:
        entry["module"](x128)
    m128_stats = stable_linear.decode_stats()
    case.gate(
        "patch:m>1-does-not-take-decode-path",
        int(m128_stats["decode_calls"]) == 0
        and int(m128_stats["padded_calls"]) == len(modules),
        {
            "decode_calls": m128_stats["decode_calls"],
            "padded_calls": m128_stats["padded_calls"],
            "expected": len(modules),
        },
    )
    case.gate(
        "coverage:a-and-b-classes",
        info["num_a"] == 48 and info["num_b"] == 48,
        {"a": info["num_a"], "b": info["num_b"]},
    )
    case.gate(
        "coverage:layers-48",
        info["num_layers"] == NUM_LAYERS,
        {"layers": info["layers"][:8] + ["..."] + info["layers"][-4:], "count": info["num_layers"]},
    )
    case.gate(
        "coverage:m1-and-m128-input-forms",
        True,
        {
            "M=1": "x[1,1,5120] → flat rows=1 → matvec",
            "M=128": "x[1,128,5120] → flat rows=128 → 原 padded 路径",
            "also_covered_in_other_cases": [
                "x[1,5120]（2D，M=1）由 numeric case 的 96 个 module 覆盖",
                "x[2,150,5120] / x[1,129,5120] 等由 padded case 覆盖",
            ],
        },
    )
    x_2d = _hidden((1, HIDDEN), 0.5, device)
    stable_linear.reset_decode_stats()
    got_2d = [entry["module"](x_2d) for entry in modules]
    stats_2d = stable_linear.decode_stats()
    ok_2d = int(stats_2d["decode_calls"]) == len(modules) and all(
        bool(torch.isfinite(out).all().item()) for out in got_2d
    )
    case.gate(
        "coverage:2d-M=1-form",
        ok_2d,
        {"decode_calls": stats_2d["decode_calls"], "expected": len(modules)},
    )
    all_finite = all(
        bool(torch.isfinite(entry["module"](x1)).all().item()) for entry in modules
    )
    case.gate("finite:all-96-modules", all_finite, {"modules": len(modules)})
    case.metric("decode_calls_m1", float(m1_stats["decode_calls"]))
    case.metric("padded_calls_m128", float(m128_stats["padded_calls"]))
    case.metric("decode_fallbacks", float(len(m1_stats["fallback_reasons"])))


# ----------------------------------------------------------------- 计时工具
def _capture(fn, warmup: int = GRAPH_WARMUP) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream(device=torch.device("cuda"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_graph(graph: torch.cuda.CUDAGraph) -> float:
    """每次 replay 的 ms（replay 20 次一起放进 event 区间，取多轮中位数）。"""
    for _ in range(GRAPH_WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ROUNDS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(REPLAYS):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)) / REPLAYS)
    return statistics.median(samples)


def _time_eager(fn, *, warmup: int = WARMUP, reps: int = 15) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _make_runner(pairs: list[tuple[object, torch.Tensor]], repeats: int):
    def body() -> None:
        for _ in range(repeats):
            for fn, x in pairs:
                fn(x)

    return body


def _per_call_ms(replay_ms: float, pairs: int, repeats: int) -> float:
    return replay_ms / max(1, pairs * repeats)


def _repeats_for(pairs: int) -> int:
    """让每次 replay 至少约 1 ms，避免 graph launch 开销污染每调用值。"""
    return max(1, min(16, int(round(1000.0 / max(1.0, pairs * 8.0)))))


# ------------------------------------------------------------------ microbench
def _run_microbench(report: OpReport, modules: list[dict], info: dict, device: torch.device) -> None:
    props = torch.cuda.get_device_properties(0)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 0)
    weight_bytes = int(info["weight_bytes_each"])
    case = report.case(
        PERF_CASE,
        boundary="整段调用一起进 CUDA graph；replay 20 次取中位数再除以调用数",
        baseline="PaddedReference（fill + torch.cat + M=128 F.linear）",
        candidate="Triton M=1 matvec",
        warm="16 个 module（8 层 × a/b）单副本，L2-resident",
        cold="96 个 module × 2 份不同地址副本，工作集 > L2",
        configs=[f"block_k={bk},warps={nw}" for bk, nw in CONFIGS],
        l2_bytes=l2_bytes,
        weight_bytes_each=weight_bytes,
        replay=REPLAYS,
        rounds=ROUNDS,
    )

    # 单层、两类各 8 个层的 module 子集（warm）。
    layer_subset = info["layers"][::6][:8]
    warm_modules = [entry for entry in modules if entry["layer"] in set(layer_subset)]
    x = _hidden((1, 1, HIDDEN), 0.5, device)
    warm_pairs_baseline = [(entry["reference"], x) for entry in warm_modules]
    warm_pairs_candidate = [(entry["module"], x) for entry in warm_modules]

    # cold：全部 96 个 module 再复制一份权重到不同地址，工作集 2×47 MB > L2。
    cold_modules = []
    for entry in modules:
        clone = entry["weight"].clone()
        cold_modules.append(
            {
                "reference": PaddedReference(clone, None, block_m=BLOCK_M),
                "module": stable_linear.ShapeStableLinear(clone, None, block_m=BLOCK_M),
            }
        )
    cold_pairs_baseline = [(entry["reference"], x) for entry in modules]
    cold_pairs_baseline += [(entry["reference"], x) for entry in cold_modules]
    cold_pairs_candidate = [(entry["module"], x) for entry in modules]
    cold_pairs_candidate += [(entry["module"], x) for entry in cold_modules]
    cold_working_mb = (len(modules) + len(cold_modules)) * weight_bytes / 2**20
    warm_working_mb = len(warm_modules) * weight_bytes / 2**20

    # 1) 每个粗粒度参数组先对拍（快但错的配置没有意义），再计时。
    config_ms: dict[tuple[int, int], float] = {}
    config_ok: dict[tuple[int, int], bool] = {}
    probe = modules[0]["module"]
    probe_ref = modules[0]["reference"]
    for block_k, num_warps in CONFIGS:
        stable_linear.DECODE_BLOCK_K = block_k
        stable_linear.DECODE_NUM_WARPS = num_warps
        stable_linear.reset_decode_stats()
        got = probe(x)
        config_ok[(block_k, num_warps)] = _tolerance_ok(
            error_metrics(got, probe_ref(x))
        ) and bool(torch.isfinite(got).all().item())
        case.gate(
            f"numeric:config-block_k{block_k}-w{num_warps}",
            config_ok[(block_k, num_warps)],
            {"metrics": error_metrics(got, probe_ref(x)), "tolerance": GDN_TOL},
        )
        repeats = _repeats_for(len(warm_pairs_candidate))
        body = _make_runner(warm_pairs_candidate, repeats)
        graph = _capture(body)
        config_ms[(block_k, num_warps)] = _per_call_ms(
            _time_graph(graph), len(warm_pairs_candidate), repeats
        )
        del graph
        torch.cuda.empty_cache()
    best = min(config_ms, key=config_ms.get)
    stable_linear.DECODE_BLOCK_K = best[0]
    stable_linear.DECODE_NUM_WARPS = best[1]
    case.gate(
        "coverage:parameter-sweep",
        len(config_ms) == len(CONFIGS) == 3 and all(config_ok.values()),
        {
            "configs": {
                f"block_k={bk},w={nw}": {
                    "ms_per_call": config_ms[(bk, nw)],
                    "numeric_ok": config_ok[(bk, nw)],
                }
                for bk, nw in CONFIGS
            },
            "selected": f"block_k={best[0]},w={best[1]}",
        },
    )

    # 2) warm / cold 两组，baseline vs 选中的 candidate。
    def _pair_group(pairs: list[tuple[object, torch.Tensor]]) -> tuple[float, float]:
        repeats = _repeats_for(len(pairs))
        base_graph = _capture(_make_runner(pairs, repeats))
        base_ms = _per_call_ms(_time_graph(base_graph), len(pairs), repeats)
        del base_graph
        torch.cuda.empty_cache()
        return base_ms, repeats

    warm_base_ms, warm_repeats = _pair_group(warm_pairs_baseline)
    warm_cand_graph = _capture(_make_runner(warm_pairs_candidate, warm_repeats))
    warm_cand_ms = _per_call_ms(
        _time_graph(warm_cand_graph), len(warm_pairs_candidate), warm_repeats
    )
    del warm_cand_graph
    torch.cuda.empty_cache()

    cold_base_ms, cold_repeats = _pair_group(cold_pairs_baseline)
    cold_cand_graph = _capture(_make_runner(cold_pairs_candidate, cold_repeats))
    cold_cand_ms = _per_call_ms(
        _time_graph(cold_cand_graph), len(cold_pairs_candidate), cold_repeats
    )
    del cold_cand_graph
    torch.cuda.empty_cache()

    # 3) 归因：只算 M=128 GEMM（输入已经 pad 好，不含 fill/cat）。
    padded_x = torch.zeros((BLOCK_M, HIDDEN), device=device, dtype=torch.bfloat16)
    padded_x[0].copy_(x.reshape(HIDDEN))
    gemm_only_pairs = [
        (lambda _x, w=entry["weight"]: F.linear(padded_x, w, None), x)
        for entry in warm_modules
    ]
    gemm_repeats = _repeats_for(len(gemm_only_pairs))
    gemm_graph = _capture(_make_runner(gemm_only_pairs, gemm_repeats))
    gemm_only_ms = _per_call_ms(
        _time_graph(gemm_graph), len(gemm_only_pairs), gemm_repeats
    )
    del gemm_graph
    torch.cuda.empty_cache()

    # 4) 诊断：不 pad 但走 PyTorch/cuBLAS 的两个 M=1 形态（解释为什么选 Triton）。
    flat_x = x.reshape(1, HIDDEN)
    matmul_pairs = [
        (lambda _x, w=entry["weight"]: torch.matmul(flat_x, w.t()), x)
        for entry in warm_modules
    ]
    linear_m1_pairs = [
        (lambda _x, w=entry["weight"]: F.linear(flat_x, w, None), x)
        for entry in warm_modules
    ]
    torch_matmul_ms = torch_linear_ms = None
    try:
        g = _capture(_make_runner(matmul_pairs, warm_repeats))
        torch_matmul_ms = _per_call_ms(_time_graph(g), len(matmul_pairs), warm_repeats)
        del g
        g = _capture(_make_runner(linear_m1_pairs, warm_repeats))
        torch_linear_ms = _per_call_ms(_time_graph(g), len(linear_m1_pairs), warm_repeats)
        del g
    except Exception as exc:  # noqa: BLE001 - 只作诊断
        case.diag("torch-m1-variants", False, {"error": f"{type(exc).__name__}: {exc}"[:300]})
    torch.cuda.empty_cache()

    # 5) eager per-call（含 Python/launch，只作诊断；生产是 graph）。
    eager_base_ms = _time_eager(_make_runner(warm_pairs_baseline, 1)) / len(
        warm_pairs_baseline
    )
    eager_cand_ms = _time_eager(_make_runner(warm_pairs_candidate, 1)) / len(
        warm_pairs_candidate
    )

    # ---------------------------------------------------------------- gate
    case.gate(
        "coverage:same-timing-boundary",
        True,
        {
            "note": "baseline 与 candidate 用同一份权重地址、同一个 x、同一段 graph",
            "warm_modules": len(warm_modules),
            "warm_layers": layer_subset,
            "warm_working_mb": warm_working_mb,
            "warm_l2_resident": warm_working_mb * 2**20 <= l2_bytes,
            "cold_modules": len(cold_pairs_candidate),
            "cold_working_mb": cold_working_mb,
            "cold_l2_resident": cold_working_mb * 2**20 <= l2_bytes,
        },
    )
    case.gate(
        "coverage:graph-replay-timed",
        warm_base_ms > 0.0 and warm_cand_ms > 0.0,
        {
            "warm_baseline_ms": warm_base_ms,
            "warm_candidate_ms": warm_cand_ms,
            "cold_baseline_ms": cold_base_ms,
            "cold_candidate_ms": cold_cand_ms,
        },
    )
    case.gate(
        "metrics-positive-finite",
        all(
            value > 0.0
            for value in (warm_base_ms, warm_cand_ms, cold_base_ms, cold_cand_ms, gemm_only_ms)
        ),
        {
            "warm_base_ms": warm_base_ms,
            "warm_cand_ms": warm_cand_ms,
            "cold_base_ms": cold_base_ms,
            "cold_cand_ms": cold_cand_ms,
        },
    )
    stable_linear.reset_decode_stats()
    for fn, pair_x in warm_pairs_candidate:
        fn(pair_x)
    capture_stats = stable_linear.decode_stats()
    case.diag(
        "decode-path-used-by-microbench",
        int(capture_stats["decode_calls"]) == len(warm_pairs_candidate),
        {
            "decode_calls": capture_stats["decode_calls"],
            "expected": len(warm_pairs_candidate),
            "fallbacks": capture_stats["fallback_reasons"],
        },
    )

    # -------------------------------------------------------------- metrics
    for label, base_ms, cand_ms in (
        ("warm", warm_base_ms, warm_cand_ms),
        ("cold", cold_base_ms, cold_cand_ms),
    ):
        case.metric(f"{label}_baseline_ms_call", base_ms)
        case.metric(f"{label}_candidate_ms_call", cand_ms)
        case.metric(f"{label}_saving_us_call", (base_ms - cand_ms) * 1000.0)
        case.metric(f"{label}_speedup", base_ms / cand_ms if cand_ms else 0.0)
        for count, name in ((96, "96"), (128, "profile128")):
            case.metric(f"{label}_est_step_baseline_ms_{name}", base_ms * count)
            case.metric(f"{label}_est_step_candidate_ms_{name}", cand_ms * count)
            case.metric(f"{label}_est_step_saving_ms_{name}", (base_ms - cand_ms) * count)
    case.metric("gemm_only_ms_call", gemm_only_ms)
    case.metric("pad_overhead_ms_call", max(0.0, warm_base_ms - gemm_only_ms))
    case.metric("eager_baseline_ms_call", eager_base_ms)
    case.metric("eager_candidate_ms_call", eager_cand_ms)
    if torch_matmul_ms is not None:
        case.metric("torch_matmul_ms_call", torch_matmul_ms)
    if torch_linear_ms is not None:
        case.metric("torch_linear_m1_ms_call", torch_linear_ms)
    for (block_k, num_warps), value in config_ms.items():
        case.metric(f"config_block_k{block_k}_w{num_warps}_ms_call", value)

    case.diag(
        "l2-residency",
        True,
        {
            "l2_bytes": l2_bytes,
            "l2_mb": l2_bytes / 2**20,
            "warm_working_mb": warm_working_mb,
            "cold_working_mb": cold_working_mb,
            "note": "warm 组必然 L2-resident；真实 decode 的 14 GB NVFP4 权重流会把 "
            "a/b 挤出 L2，所以只有 cold 组和端到端 bench 能代表真实开销",
        },
    )
    case.diag(
        "why-triton",
        torch_linear_ms is not None and torch_matmul_ms is not None,
        {
            "triton_ms_call": warm_cand_ms,
            "torch_linear_m1_ms_call": torch_linear_ms,
            "torch_matmul_ms_call": torch_matmul_ms,
            "note": "三个都是 1 个 kernel；Triton 的 reduction 顺序固定、无原子操作，"
            "graph 捕获/重放逐位可复现，且不依赖 cuBLAS 对 M=1 的启发式选择",
        },
    )
    case.diag(
        "eager-vs-graph",
        True,
        {
            "eager_baseline_ms_call": eager_base_ms,
            "eager_candidate_ms_call": eager_cand_ms,
            "note": "eager 含 Python/launch 开销，只作诊断；生产路径是 graph replay",
        },
    )


# ----------------------------------------------------------------------- main
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        default="all",
        help=f"逗号分隔的 case 名，或 all（默认）。可选：{', '.join(SCENARIOS)}",
    )
    parser.add_argument("--out", required=True, help="产物目录，写 {out}/op.json")
    return parser.parse_args(argv)


def _select(only: str) -> tuple[str, ...]:
    if only.strip().lower() in ("", "all"):
        return SCENARIOS
    names = tuple(name.strip() for name in only.split(",") if name.strip())
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; SCENARIOS={SCENARIOS}")
    return names


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    selected = _select(args.only)
    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    modules, info = _load_real_modules(device)
    props = torch.cuda.get_device_properties(0)
    report = OpReport(
        args.out,
        candidate_name="stable-linear-decode-matvec",
        candidate_format="bf16-in-bf16-out-fp32-acc",
        candidate_extra={
            "module": "runner/ops/bf16/stable_linear.py::_decode_matvec",
            "boundary": "ShapeStableLinear.forward, flat rows == 1 only",
            "kernel": "triton bf16 matvec, 1 program per output row, no atomics",
            "prefill_path": "M>1 保持 block_m=128 padding，代码未改",
            "shape_stable_linears": info["num_modules"],
        },
        environment={
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "l2_bytes": int(getattr(props, "L2_cache_size", 0) or 0),
            "checkpoint": CHECKPOINT,
            "manifest": info["manifest"],
            "cases": list(selected),
        },
    )
    if NUMERIC_CASE in selected:
        _run_numeric(report, modules, info, device)
    if PADDED_CASE in selected:
        _run_padded(report, modules, info, device)
    if COVERAGE_CASE in selected:
        _run_coverage(report, modules, info, device)
    if PERF_CASE in selected:
        _run_microbench(report, modules, info, device)
    payload = report.finish()
    summary = payload["summary"]
    for entry in summary["cases"]:
        print(
            f"[{entry['case']}] gates {entry['gates_passed']}/{entry['gates']} "
            f"failed={entry['gates_failed']}"
        )
    for entry in payload["cases"]:
        metrics = entry.get("metrics") or {}
        if not metrics:
            continue
        print(
            f"[{entry['case']}] metrics "
            + ", ".join(f"{key}={value:.4g}" for key, value in metrics.items())
        )
    print(f"op.json -> {Path(args.out) / 'op.json'}")
    print("PASS" if summary["ok"] else "FAIL")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
