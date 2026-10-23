#!/usr/bin/env python3
"""Phase 2 任务 D：GDN decode（recurrent，``sequence_length == 1``）融合 kernel 的
op gate + microbench。

边界：只替换 ``torch_recurrent_gated_delta_rule`` 的 decode 分支。conv1d、
``in_proj_*``、out_proj、gated RMSNorm 都不在本次范围内。

gate（每个 case 都要有 ``coverage:`` 项）：
- 真实非零 history state：初值来自随机历史，不是全零；记录 |state| 的量级；
- 连续 2 步原地递推：第 2 步的 reference 从**候选更新后**的 state 出发，所以
  in-place 递推错误会被抓住；
- 48 层真实 shape：v_heads=48 / k_heads=16（repeat 3）/ head_dim=128，48 层全跑；
- dtype：recurrent state 保持 FP32、conv state 保持 BF16 且比特不变（本算子不碰 conv）；
- 原地：state 的 ``data_ptr`` 和字节数在调用前后不变（static cache / graph 依赖）；
- finite；
- 数值容差用 decisions 里 M2 冻结的 GDN 容差：
  ``nrmse <= 1e-3``、``cosine >= 0.99999``、``max_rel <= 5e-3``。

reference 是**未修改的** ``torch_recurrent_gated_delta_rule``
（``use_qk_l2norm_in_kernel=True``），即本机没有 ``flash-linear-attention`` 时的
transformers 纯 PyTorch 路径 —— 与 M3 profile 里看到的逐 token 展开一致。

microbench：同一计时边界（CUDA event 包住一次 48 层完整 step，warmup 后取 15 次
中位数）比较原 PyTorch 路径与候选 kernel，报每层 µs 与整步 ms。生产路径是
CUDA Graph，eager 数字含 Python/launch 开销，所以另外用 graph replay 计时。

用法：
    .venv/bin/python scripts/phase2_gdn_check.py --out {output}/op
    .venv/bin/python scripts/phase2_gdn_check.py --only microbench --out .tmp/op
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (  # noqa: E402
    torch_recurrent_gated_delta_rule,
)

from runner.experiment.op_artifact import OpReport  # noqa: E402
from runner.ops.gdn import (  # noqa: E402
    BLOCK_V_CONFIGS,
    DEFAULT_BLOCK_V,
    DEFAULT_NUM_WARPS,
    gdn_decode_step,
)
from runner.ops.nvfp4.reference import error_metrics  # noqa: E402

SCENARIOS = ("decode-two-steps-48-layers", "microbench-baseline-vs-candidate")
PERF_CASE = SCENARIOS[1]

# 冻结的模型 shape（64 层里 48 层是 GDN；config: linear_num_value_heads=48,
# linear_num_key_heads=16, linear_{key,value}_head_dim=128, conv_kernel_dim=4）。
NUM_LAYERS = 48
NUM_V_HEADS = 48
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
RATIO = NUM_V_HEADS // NUM_K_HEADS
CONV_DIM = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM
CONV_KERNEL = 4
STATE_ELEMS = NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM
STATE_BYTES = STATE_ELEMS * 4

# M2 冻结的 GDN 容差（docs/decisions.md 2026-10-21 M2 关闭与验收口径）。
GDN_TOL = {"nrmse": 1e-3, "cosine": 0.99999, "max_rel": 5e-3}

SEED = 20261023
STEPS = 2
WARMUP = 5
REPS = 15
GRAPH_WARMUP = 3


# --------------------------------------------------------------------------- 数据


def _layers(device: torch.device, gen: torch.Generator) -> list[dict[str, torch.Tensor]]:
    """48 层 decode 输入：q/k 是 repeat_interleave 之前的 k-head 形状。"""

    def randn(shape, dtype, scale=1.0):
        value = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
        return (value * scale).to(dtype)

    out = []
    for _ in range(NUM_LAYERS):
        out.append(
            {
                "q": randn((1, 1, NUM_K_HEADS, HEAD_K_DIM), torch.bfloat16),
                "k": randn((1, 1, NUM_K_HEADS, HEAD_K_DIM), torch.bfloat16),
                "v": randn((1, 1, NUM_V_HEADS, HEAD_V_DIM), torch.bfloat16, 0.5),
                # decay = exp(g)，g 取 -[0.005, 1.5]，覆盖接近 1 和快速衰减两端。
                "g": -torch.empty((1, 1, NUM_V_HEADS), device=device).uniform_(
                    0.005, 1.5, generator=gen
                ),
                "beta": torch.sigmoid(randn((1, 1, NUM_V_HEADS), torch.float32)).to(
                    torch.bfloat16
                ),
            }
        )
    return out


def _states(device: torch.device, gen: torch.Generator) -> list[torch.Tensor]:
    """48 个非零 history state，形状/布局与 static cache 一致。"""
    return [
        torch.randn(
            (1, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM),
            generator=gen,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
        for _ in range(NUM_LAYERS)
    ]


def _conv_states(device: torch.device, gen: torch.Generator) -> list[torch.Tensor]:
    """48 个 conv state（BF16），只用来证明本算子不碰它。"""
    return [
        torch.randn(
            (1, CONV_DIM, CONV_KERNEL),
            generator=gen,
            device=device,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        for _ in range(NUM_LAYERS)
    ]


def _reference(problem: dict[str, torch.Tensor], initial_state: torch.Tensor):
    """逐位对照对象：未修改的 torch_recurrent_gated_delta_rule（decode 分支）。"""
    query = problem["q"].repeat_interleave(RATIO, dim=2)
    key = problem["k"].repeat_interleave(RATIO, dim=2)
    return torch_recurrent_gated_delta_rule(
        query,
        key,
        problem["v"],
        g=problem["g"],
        beta=problem["beta"],
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def _call(problem, state, out, block_v, num_warps):
    return gdn_decode_step(
        problem["q"],
        problem["k"],
        problem["v"],
        problem["g"],
        problem["beta"],
        state,
        out=out,
        block_v=block_v,
        num_warps=num_warps,
    )


# ----------------------------------------------------------------------- 判定工具


def _numeric(case, name: str, got: torch.Tensor, ref: torch.Tensor) -> dict:
    metrics = error_metrics(got, ref)
    ok = (
        metrics["finite"]
        and metrics["nrmse"] <= GDN_TOL["nrmse"]
        and metrics["cosine"] >= GDN_TOL["cosine"]
        and metrics["max_rel"] <= GDN_TOL["max_rel"]
    )
    detail = {}
    for key, value in metrics.items():
        detail[key] = bool(value) if isinstance(value, bool) else float(f"{value:.9g}")
    case.gate(name, ok, detail)
    return metrics


def _track(worst: dict, metrics: dict) -> None:
    worst["nrmse"] = max(worst["nrmse"], metrics["nrmse"])
    worst["max_rel"] = max(worst["max_rel"], metrics["max_rel"])
    worst["cosine"] = min(worst["cosine"], metrics["cosine"])
    worst["finite"] = worst["finite"] and metrics["finite"]


# ------------------------------------------------------------------------- case 1


def _run_correctness(report: OpReport, device: torch.device) -> None:
    case = report.case(
        SCENARIOS[0],
        layers=NUM_LAYERS,
        num_v_heads=NUM_V_HEADS,
        num_k_heads=NUM_K_HEADS,
        head_k_dim=HEAD_K_DIM,
        head_v_dim=HEAD_V_DIM,
        io_dtype="bfloat16",
        state_dtype="float32",
        conv_state_dtype="bfloat16",
        steps=STEPS,
        kernel="gdn.ops.decode.gdn_decode_step",
        block_v=DEFAULT_BLOCK_V,
        num_warps=DEFAULT_NUM_WARPS,
        reference="transformers.torch_recurrent_gated_delta_rule(use_qk_l2norm_in_kernel=True)",
        tolerance=GDN_TOL,
    )
    gen = torch.Generator(device=device).manual_seed(SEED)
    layers = _layers(device, gen)
    states = _states(device, gen)
    conv = _conv_states(device, gen)

    history_absmax = [float(item.abs().max()) for item in states]
    conv_before = [item.clone() for item in conv]
    ptr_before = [item.data_ptr() for item in states]
    bytes_before = [item.numel() * item.element_size() for item in states]
    out = torch.empty(
        (1, 1, NUM_V_HEADS, HEAD_V_DIM), dtype=torch.bfloat16, device=device
    )

    worst = {"nrmse": 0.0, "max_rel": 0.0, "cosine": 1.0, "finite": True}
    finite = True
    reference_mutated = False
    steps_done = 0
    layers_checked = 0
    state_buffer_reused = True

    for step in range(1, STEPS + 1):
        for idx, problem in enumerate(layers):
            state = states[idx]
            # 第 2 步的 reference 从候选第 1 步写完的 state 出发 -> 验证原地递推。
            init = state.clone()
            ref_out, ref_state = _reference(problem, init)
            if not torch.equal(init, state):
                reference_mutated = True
            cand = _call(problem, state, out, DEFAULT_BLOCK_V, DEFAULT_NUM_WARPS)
            state_buffer_reused = state_buffer_reused and cand is out
            finite = finite and bool(torch.isfinite(out).all())
            finite = finite and bool(torch.isfinite(state).all())
            _track(
                worst,
                _numeric(
                    case, f"numeric:step{step}:layer{idx:02d}:out", out, ref_out
                ),
            )
            _track(
                worst,
                _numeric(
                    case, f"numeric:step{step}:layer{idx:02d}:state", state, ref_state
                ),
            )
            layers_checked += 1
        steps_done += 1

    case.gate(
        "coverage:nonzero-history-state",
        min(history_absmax) > 1e-3,
        {
            "layers": NUM_LAYERS,
            "min_absmax": min(history_absmax),
            "max_absmax": max(history_absmax),
            "note": "初值来自随机历史，不是全零 state",
        },
    )
    case.gate(
        "coverage:two-consecutive-steps",
        steps_done == STEPS == 2,
        {"steps": steps_done, "recursion": "step2 reference 从候选 step1 后的 state 出发"},
    )
    case.gate(
        "coverage:layers-48",
        len(layers) == NUM_LAYERS
        and layers_checked == NUM_LAYERS * STEPS
        and NUM_LAYERS == 48,
        {
            "layers": len(layers),
            "comparisons": layers_checked,
            "num_v_heads": NUM_V_HEADS,
            "num_k_heads": NUM_K_HEADS,
        },
    )
    case.gate(
        "coverage:state-dtype-fp32",
        all(item.dtype == torch.float32 for item in states),
        {"state_dtype": "float32"},
    )
    case.gate(
        "coverage:conv-state-dtype-unchanged",
        all(
            item.dtype == torch.bfloat16 and torch.equal(item, before)
            for item, before in zip(conv, conv_before)
        ),
        {
            "conv_dtype": "bfloat16",
            "shape": [1, CONV_DIM, CONV_KERNEL],
            "bytes_per_layer": CONV_DIM * CONV_KERNEL * 2,
            "note": "recurrent step 不应碰 conv state",
        },
    )
    case.gate(
        "coverage:state-pointer-and-bytes-stable",
        [item.data_ptr() for item in states] == ptr_before
        and [item.numel() * item.element_size() for item in states] == bytes_before
        and all(item == STATE_BYTES for item in bytes_before),
        {
            "layers": NUM_LAYERS,
            "identical_data_ptr": [item.data_ptr() for item in states] == ptr_before,
            "bytes_per_layer": STATE_BYTES,
            "total_mb": NUM_LAYERS * STATE_BYTES / 2**20,
        },
    )
    case.gate(
        "coverage:out-dtype-and-shape",
        out.dtype == torch.bfloat16 and tuple(out.shape) == (1, 1, NUM_V_HEADS, HEAD_V_DIM),
        {"out_dtype": "bfloat16", "out_shape": list(out.shape)},
    )
    case.gate("coverage:finite", finite and worst["finite"], {"finite": finite})
    case.gate(
        "coverage:out-buffer-reused",
        state_buffer_reused,
        {"note": "候选写进调用方给的 out 槽，不额外分配"},
    )
    case.gate(
        "reference-does-not-mutate-initial-state",
        not reference_mutated,
        {"note": "reference 返回新 state buffer，所以原地写回是候选独有的行为"},
    )
    case.diag(
        "in-place-vs-reference-buffer",
        True,
        {
            "reference_returns_new_buffer": True,
            "candidate_updates_same_buffer": True,
            "state_buffer_reused": state_buffer_reused,
        },
    )

    case.metric("worst_nrmse", worst["nrmse"])
    case.metric("worst_max_rel", worst["max_rel"])
    case.metric("worst_cosine", worst["cosine"])
    case.metric("layers_checked", float(layers_checked))
    case.metric("steps_checked", float(steps_done))
    case.metric("state_bytes_per_layer", float(STATE_BYTES))
    case.metric("state_total_mb", NUM_LAYERS * STATE_BYTES / 2**20)


# ------------------------------------------------------------------------- case 2


def _time_loop(fn, reset, *, warmup: int, reps: int) -> float:
    """一次 48 层 step 的墙钟中位数（ms），event 边界只包住 fn。"""
    reset()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        reset()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _capture(fn, reset) -> torch.cuda.CUDAGraph:
    reset()
    stream = torch.cuda.Stream(device=torch.device("cuda"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(GRAPH_WARMUP):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_replay(graph: torch.cuda.CUDAGraph, *, reps: int = REPS) -> float:
    for _ in range(GRAPH_WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _run_microbench(report: OpReport, device: torch.device) -> None:
    case = report.case(
        PERF_CASE,
        layers=NUM_LAYERS,
        num_v_heads=NUM_V_HEADS,
        num_k_heads=NUM_K_HEADS,
        head_k_dim=HEAD_K_DIM,
        head_v_dim=HEAD_V_DIM,
        io_dtype="bfloat16",
        state_dtype="float32",
        warmup=WARMUP,
        reps=REPS,
        graph_warmup=GRAPH_WARMUP,
        boundary="CUDA event 包住一次 48 层完整 decode step，warmup 后取 15 次中位数",
        baseline="torch_recurrent_gated_delta_rule + repeat_interleave + update_recurrent_state copy_",
        candidate="gdn_decode_step（1 kernel/layer，state 原地）",
        tolerance=GDN_TOL,
    )
    gen = torch.Generator(device=device).manual_seed(SEED + 1)
    layers = _layers(device, gen)
    states = _states(device, gen)
    snapshot = [item.clone() for item in states]
    outs = [
        torch.empty((1, 1, NUM_V_HEADS, HEAD_V_DIM), dtype=torch.bfloat16, device=device)
        for _ in range(NUM_LAYERS)
    ]

    def reset() -> None:
        for item, saved in zip(states, snapshot):
            item.copy_(saved)

    def baseline_step() -> None:
        for idx, problem in enumerate(layers):
            query = problem["q"].repeat_interleave(RATIO, dim=2)
            key = problem["k"].repeat_interleave(RATIO, dim=2)
            _, new_state = torch_recurrent_gated_delta_rule(
                query,
                key,
                problem["v"],
                g=problem["g"],
                beta=problem["beta"],
                initial_state=states[idx],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            # 等价于 cache_params.update_recurrent_state 的原地 copy_。
            states[idx].copy_(new_state)

    def candidate_step(block_v: int, num_warps: int) -> None:
        for idx, problem in enumerate(layers):
            gdn_decode_step(
                problem["q"],
                problem["k"],
                problem["v"],
                problem["g"],
                problem["beta"],
                states[idx],
                out=outs[idx],
                block_v=block_v,
                num_warps=num_warps,
            )

    # 每个粗粒度参数组都先对拍一次（快但错的配置没有意义）。
    for block_v, num_warps in BLOCK_V_CONFIGS:
        state = snapshot[0].clone()
        init = state.clone()
        ref_out, ref_state = _reference(layers[0], init)
        got = gdn_decode_step(
            layers[0]["q"],
            layers[0]["k"],
            layers[0]["v"],
            layers[0]["g"],
            layers[0]["beta"],
            state,
            out=outs[0],
            block_v=block_v,
            num_warps=num_warps,
        )
        _numeric(case, f"numeric:config-bv{block_v}-w{num_warps}:out", got, ref_out)
        _numeric(case, f"numeric:config-bv{block_v}-w{num_warps}:state", state, ref_state)

    baseline_ms = _time_loop(baseline_step, reset, warmup=WARMUP, reps=REPS)
    config_ms = {
        (block_v, num_warps): _time_loop(
            lambda bv=block_v, nw=num_warps: candidate_step(bv, nw),
            reset,
            warmup=WARMUP,
            reps=REPS,
        )
        for block_v, num_warps in BLOCK_V_CONFIGS
    }
    best = min(config_ms, key=config_ms.get)
    candidate_ms = config_ms[best]

    graph_baseline_ms = None
    graph_candidate_ms = None
    graph_error = None
    try:
        baseline_graph = _capture(baseline_step, reset)
        candidate_graph = _capture(
            lambda: candidate_step(best[0], best[1]), reset
        )
        graph_baseline_ms = _time_replay(baseline_graph)
        graph_candidate_ms = _time_replay(candidate_graph)
    except Exception as exc:  # noqa: BLE001 - graph 只作 diagnostic，不阻塞 eager 结论
        graph_error = f"{type(exc).__name__}: {exc}"[:400]

    case.gate(
        "coverage:model-shapes",
        NUM_LAYERS == 48
        and NUM_V_HEADS == 48
        and NUM_K_HEADS == 16
        and HEAD_K_DIM == 128
        and HEAD_V_DIM == 128,
        {
            "layers": NUM_LAYERS,
            "num_v_heads": NUM_V_HEADS,
            "num_k_heads": NUM_K_HEADS,
            "head_k_dim": HEAD_K_DIM,
            "head_v_dim": HEAD_V_DIM,
            "io_dtype": "bfloat16",
            "state_dtype": "float32",
            "state_mb_total": NUM_LAYERS * STATE_BYTES / 2**20,
        },
    )
    case.gate(
        "coverage:same-timing-boundary",
        True,
        {
            "boundary": "event 包住一次 48 层完整 step；reset 与 sync 在 event 之外",
            "warmup": WARMUP,
            "reps": REPS,
            "baseline_includes": [
                "repeat_interleave x2",
                "torch_recurrent_gated_delta_rule python loop",
                "update_recurrent_state copy_",
                "内部 zeros_like/中间 tensor 分配",
            ],
            "candidate_includes": ["48 次 kernel launch", "out store"],
        },
    )
    case.gate(
        "coverage:parameter-sweep",
        len(config_ms) == len(BLOCK_V_CONFIGS) == 3,
        {
            "configs": [f"bv{bv}-w{nw}" for bv, nw in BLOCK_V_CONFIGS],
            "selected": f"bv{best[0]}-w{best[1]}",
        },
    )
    case.gate(
        "coverage:state-restored-between-iterations",
        True,
        {"note": "每个计时样本前把 48 层 state 复位到同一 snapshot"},
    )
    case.gate(
        "metrics-positive-finite",
        baseline_ms > 0.0
        and candidate_ms > 0.0
        and all(value > 0.0 for value in config_ms.values()),
        {"baseline_ms": baseline_ms, "candidate_ms": candidate_ms},
    )
    if graph_baseline_ms is not None and graph_candidate_ms is not None:
        case.gate(
            "coverage:graph-replay-timed",
            graph_baseline_ms > 0.0 and graph_candidate_ms > 0.0,
            {
                "note": "生产路径是 CUDA Graph，replay 计时去掉 Python/launch 开销",
                "graph_baseline_ms": graph_baseline_ms,
                "graph_candidate_ms": graph_candidate_ms,
            },
        )
        case.metric("graph_baseline_ms_step48", graph_baseline_ms)
        case.metric("graph_candidate_ms_step48", graph_candidate_ms)
        case.metric("graph_saving_ms", graph_baseline_ms - graph_candidate_ms)
        case.metric("graph_speedup", graph_baseline_ms / graph_candidate_ms)
        case.metric("graph_baseline_us_per_layer", graph_baseline_ms * 1000.0 / NUM_LAYERS)
        case.metric("graph_candidate_us_per_layer", graph_candidate_ms * 1000.0 / NUM_LAYERS)
    else:
        case.diag("graph-capture", False, {"error": graph_error})

    case.metric("baseline_ms_step48", baseline_ms)
    case.metric("candidate_ms_step48", candidate_ms)
    case.metric("baseline_us_per_layer", baseline_ms * 1000.0 / NUM_LAYERS)
    case.metric("candidate_us_per_layer", candidate_ms * 1000.0 / NUM_LAYERS)
    case.metric("eager_speedup", baseline_ms / candidate_ms)
    for (block_v, num_warps), value in config_ms.items():
        case.metric(f"candidate_bv{block_v}_w{num_warps}_ms_step48", value)
    case.diag(
        "state-traffic-estimate",
        True,
        {
            "note": "解析估算，不是实测；state 每层 3.1 MB",
            "baseline_full_size_passes_per_layer": 12,
            "baseline_mb_per_layer": 44.0,
            "candidate_mb_per_layer": 6.3,
            "baseline_gb_per_step": 48 * 44.0 / 2**10,
            "candidate_gb_per_step": 48 * 6.3 / 2**10,
        },
    )
    case.diag(
        "reference-path",
        True,
        {
            "fla_installed": importlib.util.find_spec("fla") is not None,
            "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d")
            is not None,
            "note": "fla 未安装，reference 走 transformers 的纯 torch 展开（与 M3 profile 一致）",
        },
    )


# --------------------------------------------------------------------------- main


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
    device = torch.device("cuda")
    report = OpReport(
        args.out,
        candidate_name="gdn-decode-fused-triton",
        candidate_format="fp32-recurrent-state+bf16-io",
        candidate_extra={
            "kernel": "runner/ops/gdn/decode.py::_gdn_decode_kernel",
            "boundary": "torch_recurrent_gated_delta_rule decode branch (seq_len==1)",
            "kernels_per_layer": 1,
            "block_v_configs": [f"bv{bv}-w{nw}" for bv, nw in BLOCK_V_CONFIGS],
            "default_config": f"bv{DEFAULT_BLOCK_V}-w{DEFAULT_NUM_WARPS}",
            "state_layout": "in-place fp32 [1,48,128,128], data_ptr stable",
        },
        environment={
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "cases": list(selected),
        },
    )
    with torch.no_grad():
        if SCENARIOS[0] in selected:
            _run_correctness(report, device)
        if PERF_CASE in selected:
            _run_microbench(report, device)
    payload = report.finish()
    summary = payload["summary"]
    for entry in summary["cases"]:
        print(
            f"[{entry['case']}] gates {entry['gates_passed']}/{entry['gates']} "
            f"failed={entry['gates_failed']}"
        )
    metrics = {
        entry["case"]: entry["metrics"]
        for entry in payload["cases"]
    }
    for name, values in metrics.items():
        print(f"[{name}] metrics " + ", ".join(f"{k}={v:.4g}" for k, v in values.items()))
    print(f"op.json -> {Path(args.out) / 'op.json'}")
    print("PASS" if summary["ok"] else "FAIL")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
