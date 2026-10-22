#!/usr/bin/env python3
"""M3 P0 decode benchmark：四执行方式 × depth/capacity，两套计时。

场景（`--list` 会连估算时间一起打印）：

    eager-512 / eager-8192            M2 dynamic cache，起点
    exact-512-1024 / exact-512-8192   static cache + 有效前缀 view
    exact-8192                        static cache + 8K
    cap-512-1024 / cap-512-8192       decode 扫满 capacity
    cap-8192                          同上 + 8K
    graph-512-1024 / graph-512-8192   整步 CUDA Graph
    graph-8192                        同上 + 8K

计时协议（docs/plans/m3.md §2.2–§2.3，详见 docs/bench.md）：
prefill 产生首 token（计入 TTFT），随后 **511 次 decode forward / 511 个输出 token**
是计时区间，最终 cache 长度 = depth + 511。模型 tok/s 用逐步 CUDA events（D2D copy
在 start event 之前），系统 tok/s 是真实贪心闭环墙钟（含 host gather / H2D /
sampling / D2H）。每个场景先完整 warmup 一次，再测 `--repeats` 次，每次 reset +
重新 prefill，原始逐步数据全部保存。

四档用法（时间按 511 步、5 次估算）：

    # 1) 单场景冒烟（~15 s，开发用；--steps 小值的结果不可与正式数字比较）
    .venv/bin/python bench/e2e/decode.py --only graph-512-1024 --repeats 1 --steps 32

    # 2) 定向重跑某几个场景（每个非 graph 场景约 9 min）
    .venv/bin/python bench/e2e/decode.py --only graph-8192,cap-8192

    # 3) 全量矩阵（~89 min）
    make m3-bench

    # 4) 整个里程碑关闭（check/memory/bench/profile/sol，~135 min）
    scripts/m3_close.sh

被中断的 run 用 `--resume` 续跑。可复用的条件是**三者都一致**：源码（commit +
dirty + patch hash）、运行配置（steps、repeats、chunk_size、graph_warmup、fixtures
内容 hash）、环境指纹（torch/transformers/FlashInfer/driver/GPU）。任何一项变了就
重跑该场景；整个矩阵的配置变了会改写到一个新目录，不覆盖旧记录。

产物：`results/m3/<run-id>-m3-bench/`（`bench.json`、`summary.md`、`provenance.json`、
`source_manifest.json`、每场景 `result.json`）。跑完用 `make check-results` 校验契约。

内部参数：`--worker` 只给 driver 调用子进程用，不要手写。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runner import report as reporting  # noqa: E402
from runner.loader.nvfp4_model import load_nvfp4_text_model  # noqa: E402
from runner.model.decode_runner import DecodeRunner  # noqa: E402
from runner.model.static_cache import cache_stats  # noqa: E402

# §3.2 的 P0 矩阵。eager 没有固定 capacity，标为 dynamic。
# 估算用的每步耗时来自 2026-10-21 的实测（163844 那次），只用于 --list 选场景；
# 换模型/换 kernel 后要更新，正式数字永远以实跑为准。
EST_MS_PER_STEP = {"eager": 86.0, "static-exact": 86.0, "static-cap": 85.0}
EST_GRAPH_MS_PER_STEP = {1024: 32.0, 8192: 44.0}
EST_PREFILL_S = {512: 0.45, 7680: 6.7}
SCENARIOS: dict[str, dict] = {
    "eager-512": {"execution": "eager", "depth": 512, "capacity": None},
    "eager-8192": {"execution": "eager", "depth": 7680, "capacity": None},
    "exact-512-1024": {"execution": "static-exact", "depth": 512, "capacity": 1024},
    "exact-512-8192": {"execution": "static-exact", "depth": 512, "capacity": 8192},
    "exact-8192": {"execution": "static-exact", "depth": 7680, "capacity": 8192},
    "cap-512-1024": {"execution": "static-cap", "depth": 512, "capacity": 1024},
    "cap-512-8192": {"execution": "static-cap", "depth": 512, "capacity": 8192},
    "cap-8192": {"execution": "static-cap", "depth": 7680, "capacity": 8192},
    "graph-512-1024": {"execution": "graph", "depth": 512, "capacity": 1024},
    "graph-512-8192": {"execution": "graph", "depth": 512, "capacity": 8192},
    "graph-8192": {"execution": "graph", "depth": 7680, "capacity": 8192},
}


def load_fixtures(fixtures_dir: Path) -> dict:
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())
    files = {}
    for name, meta in manifest["files"].items():
        payload = json.loads((fixtures_dir / name).read_text())
        actual = reporting._sha256_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        if actual != meta["sha256"]:
            raise RuntimeError(f"fixture hash mismatch: {name}")
        files[name] = payload
    return {"manifest": manifest, "files": files}


def load_model(manifest: Path | None = None):
    model, cfg, info = load_nvfp4_text_model(
        ROOT / reporting.CHECKPOINT, manifest_path=manifest
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / reporting.CHECKPOINT), trust_remote_code=True
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer, info


def probe_gpu() -> dict:
    env = reporting.environment()
    return {
        "clock_sm_mhz": env.get("clock_sm_mhz"),
        "temperature_c": env.get("temperature_c"),
        "power_w": env.get("power_w"),
        "memory_used_mib": _smi("memory.used"),
    }


def _smi(field: str):
    text = reporting._run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"])
    try:
        return float(text.splitlines()[0])
    except Exception:
        return None


def measure_once(runner: DecodeRunner, prompt, trace: list[int], steps: int) -> dict:
    """一次测量 = 模型计时一遍 + 系统计时一遍，中间重建初态。"""
    # 模型计时：冻结 trace，逐步 CUDA events
    runner.prefill(prompt)
    before = probe_gpu()
    model_res = runner.model_timed_decode(trace[:steps])
    after = probe_gpu()

    # 系统计时：真实贪心闭环
    _, pre = runner.prefill(prompt)
    system_res = runner.system_decode(pre.first_token, steps, stop_on_eos=False)
    tokens = [pre.first_token] + system_res["tokens"]
    expected = trace[: steps + 1]
    first_bad = next((i for i in range(min(len(tokens), len(expected))) if tokens[i] != expected[i]), None)
    request_wall = pre.prefill_s + system_res["wall_s"]
    return {
        "model": model_res,
        "system": system_res,
        "prefill": pre.to_dict(),
        "tokens": {
            "generated": len(tokens),
            "timed_steps": system_res["steps"],
            "matches_frozen_trace": tokens[: len(expected)] == expected,
            "first_bad_step": first_bad,
        },
        "request": {
            "prefill_s": pre.prefill_s,
            "decode_wall_s": system_res["wall_s"],
            "request_wall_s": request_wall,
            "ttft_s": pre.first_token_s,
            "request_tok_s": len(tokens) / request_wall if request_wall > 0 else 0.0,
        },
        "gpu_before": before,
        "gpu_after": after,
    }


def fixtures_sha256(fixtures: dict) -> str:
    """fixtures 内容指纹：prompt / trace / manifest 变了就不能复用旧结果。"""
    return reporting._sha256_text(
        json.dumps(fixtures["manifest"], sort_keys=True, separators=(",", ":"))
    )


ENV_KEYS = (
    "python",
    "torch",
    "transformers",
    "flashinfer",
    "cuda",
    "gpu_name",
    "gpu_total_mib",
    "driver",
)


def env_fingerprint() -> dict:
    """影响数值/性能的环境字段；换 driver/torch/GPU 后不能复用旧结果。"""
    env = reporting.environment()
    return {key: env.get(key) for key in ENV_KEYS}


def effective_steps(fixtures: dict, depth: int, requested: int) -> int:
    """实际计时步数：不能超过 fixture trace 的长度（最后一个 token 不再 forward）。"""
    trace = fixtures["files"][f"fixed_{depth}.json"]["trace"]
    return int(min(requested, len(trace) - 1))


def effective_repeats(spec: dict, args) -> int:
    """某个场景实际使用的重复次数（--repeats-non-graph 只作用于非 graph）。"""
    repeats = int(args.repeats)
    if (
        args.repeats_non_graph is not None
        and args.repeats_non_graph > 0
        and spec["execution"] != "graph"
    ):
        repeats = int(args.repeats_non_graph)
    return repeats


def scenario_run_config(
    name: str, spec: dict, steps: int, repeats: int, args, fixtures_sha: str
) -> dict:
    """一个场景的完整运行配置；resume 必须逐字段一致。"""
    return {
        "schema": 2,
        "scenario": name,
        "execution": spec["execution"],
        "depth": int(spec["depth"]),
        "capacity": spec["capacity"],
        "steps": int(steps),
        "repeats": int(repeats),
        "chunk_size": int(args.chunk_size),
        "graph_warmup": int(args.graph_warmup),
        "fixtures_sha256": fixtures_sha,
    }


def resume_mismatch(prev: dict, run_config: dict, env_fp: dict) -> str:
    """返回不能复用某个场景的原因；空字符串表示可以复用。"""
    prov = prev.get("provenance") or {}
    prev_git = prov.get("git") or {}
    current_git = reporting.git_state(ROOT)
    current_manifest = reporting.source_manifest(ROOT)
    if prev_git.get("commit") != current_git.get("commit"):
        return f"commit {str(prev_git.get('commit'))[:8]} -> {str(current_git.get('commit'))[:8]}"
    if prev_git.get("dirty") != current_git.get("dirty"):
        return "dirty 状态不同"
    if (prov.get("source_manifest") or {}).get("patch_sha256") != current_manifest.get(
        "patch_sha256"
    ):
        return "源码 patch 不同"
    prev_cfg = prev.get("run_config")
    if not prev_cfg:
        return "旧结果没有 run_config，无法确认配置"
    diffs = [
        f"{key}: {prev_cfg.get(key)} -> {run_config.get(key)}"
        for key in run_config
        if prev_cfg.get(key) != run_config.get(key)
    ]
    if diffs:
        return "运行配置不同（" + "; ".join(diffs) + "）"
    prev_env = prev.get("environment_fingerprint") or {}
    if not prev_env:
        return "旧结果没有环境指纹"
    env_diffs = [
        f"{key}: {prev_env.get(key)} -> {env_fp.get(key)}"
        for key in env_fp
        if prev_env.get(key) != env_fp.get(key)
    ]
    if env_diffs:
        return "环境不同（" + "; ".join(env_diffs) + "）"
    return ""


def latest_bench_dir() -> Path | None:
    dirs = sorted((ROOT / "results/m3").glob("*-m3-bench"))
    return dirs[-1] if dirs else None


def dir_config_matches(path: Path, matrix_config: dict, env_fp: dict) -> str:
    """run dir 是否可以用当前源码+配置续写；返回不能用的原因。"""
    prov_path = path / "provenance.json"
    if not prov_path.exists():
        return "缺 provenance.json"
    try:
        prov = json.loads(prov_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return f"provenance 解析失败: {exc}"
    prev_git = prov.get("git") or {}
    current_git = reporting.git_state(ROOT)
    if prev_git.get("commit") != current_git.get("commit"):
        return f"commit {str(prev_git.get('commit'))[:8]} -> {str(current_git.get('commit'))[:8]}"
    if prev_git.get("dirty") != current_git.get("dirty"):
        return "dirty 状态不同"
    if (prov.get("source_manifest") or {}).get("patch_sha256") != reporting.source_manifest(
        ROOT
    ).get("patch_sha256"):
        return "源码 patch 不同"
    prev_cfg = prov.get("run_config") or {}
    if not prev_cfg:
        return "旧目录没有 run_config"
    diffs = [
        f"{key}: {prev_cfg.get(key)} -> {matrix_config.get(key)}"
        for key in matrix_config
        if prev_cfg.get(key) != matrix_config.get(key)
    ]
    if diffs:
        return "运行配置不同（" + "; ".join(diffs) + "）"
    prev_env = prov.get("environment_fingerprint") or {}
    env_diffs = [
        f"{key}: {prev_env.get(key)} -> {env_fp.get(key)}"
        for key in env_fp
        if prev_env.get(key) != env_fp.get(key)
    ]
    if env_diffs:
        return "环境不同（" + "; ".join(env_diffs) + "）"
    return ""


def run_scenario(name: str, spec: dict, fixtures: dict, args, repeats: int) -> dict:
    depth = int(spec["depth"])
    capacity = spec["capacity"]
    prompt_fx = fixtures["files"][f"fixed_{depth}.json"]
    prompt = torch.tensor([prompt_fx["prompt_ids"]], dtype=torch.long)
    full_trace = list(prompt_fx["trace"])

    model, tokenizer, load_info = load_model()
    runner = DecodeRunner(
        model,
        tokenizer,
        execution=spec["execution"],
        capacity=capacity,
        prefill_chunk=args.chunk_size,
        max_context=8192,
        graph_warmup=args.graph_warmup,
    )
    harness = runner.prepare()
    steps = effective_steps(fixtures, depth, args.steps)
    if spec["execution"] != "eager":
        if depth + steps > capacity:
            raise ValueError(f"{name}: depth {depth} + {steps} > capacity {capacity}")
        runner.require_capacity(0)
    trace = full_trace

    fixtures_sha = fixtures_sha256(fixtures)
    env_fp = env_fingerprint()
    run_config = scenario_run_config(name, spec, steps, repeats, args, fixtures_sha)
    result: dict = {
        "scenario": name,
        "spec": spec,
        "steps": steps,
        "run_config": run_config,
        "environment_fingerprint": env_fp,
        "load": load_info,
        "harness": harness,
        "fixture": {
            "file": f"fixed_{depth}.json",
            "prompt_sha256": prompt_fx.get("prompt_sha256"),
            "trace_sha256": prompt_fx.get("trace_sha256"),
        },
        "repeats_requested": repeats,
        "warmup_steps": steps,
    }

    if spec["execution"] == "graph":
        result["graph"] = runner.ensure_graph(prompt)
        result["cache_after_capture"] = cache_stats(runner.cache, check_finite=False).to_dict()

    # 完整 warmup 一次（和正式测量同样的两步）
    measure_once(runner, prompt, trace, steps)
    # warmup 之后 GDN/conv state 才存在，cache 字节账要在这一步之后取。
    result["harness"] = runner.harness_info()

    samples = []
    for i in range(repeats):
        print(f"  repeat {i + 1}/{repeats}", flush=True)
        sample = measure_once(runner, prompt, trace, steps)
        sample["repeat"] = i
        samples.append(sample)
        print(
            f"    model {sample['model']['model_tok_s']:.2f} tok/s "
            f"system {sample['system']['system_tok_s']:.2f} tok/s "
            f"peak {sample['model']['peak_allocated_gb']:.3f}/{sample['system']['peak_reserved_gb']:.3f} GB",
            flush=True,
        )

    # 汇总：run 间波动 vs step 分布分开
    model_tok = [s["model"]["model_tok_s"] for s in samples]
    system_tok = [s["system"]["system_tok_s"] for s in samples]
    request_tok = [s["request"]["request_tok_s"] for s in samples]
    pooled_steps = [ms for s in samples for ms in s["model"]["per_step_ms"]]
    peak_alloc = max(max(s["model"]["peak_allocated_gb"], s["system"]["peak_allocated_gb"]) for s in samples)
    peak_res = max(max(s["model"]["peak_reserved_gb"], s["system"]["peak_reserved_gb"]) for s in samples)
    free_min = min(min(s["model"]["free_gb"], s["system"]["free_gb"]) for s in samples)
    result["samples"] = samples
    result["summary"] = {
        "model_tok_s": reporting.run_stats(model_tok),
        "system_tok_s": reporting.run_stats(system_tok),
        "request_tok_s": reporting.run_stats(request_tok),
        "step_ms": reporting.step_stats(pooled_steps),
        "prefill_s": reporting.run_stats([s["prefill"]["prefill_s"] for s in samples]),
        "ttft_s": reporting.run_stats([s["request"]["ttft_s"] for s in samples]),
        "peak_allocated_gb": peak_alloc,
        "peak_reserved_gb": peak_res,
        "free_gb_min": free_min,
        "trace_match_all": all(s["tokens"]["matches_frozen_trace"] for s in samples),
        "row_copy_ms": reporting.run_stats([s["model"].get("row_copy_ms", 0.0) for s in samples]),
    }
    del runner
    torch.cuda.empty_cache()
    return result


def run_worker(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixtures = load_fixtures(Path(args.fixtures))
    spec = SCENARIOS[args.worker]
    repeats = effective_repeats(spec, args)
    try:
        result = run_scenario(args.worker, spec, fixtures, args, repeats)
        result["ok"] = True
    except Exception:
        result = {"scenario": args.worker, "ok": False, "error": traceback.format_exc()}
        print(result["error"], flush=True)
    result["provenance"] = reporting.provenance(ROOT, {"worker": args.worker})
    reporting.save_json(out / "result.json", result)
    print(f"worker {args.worker}: ok={result['ok']}", flush=True)
    return 0 if result["ok"] else 1


def format_table(results: list[dict]) -> str:
    header = (
        "| scenario | execution | depth | capacity | model tok/s (median [min,max]) | "
        "system tok/s | request tok/s | step p50/p95 ms | prefill s | peak GB | reserved GB | free GB |"
    )
    lines = [
        header,
        "|" + "---|" * 12,
    ]
    for res in results:
        spec = res.get("spec", {})
        summ = res.get("summary", {})
        if not summ:
            lines.append(f"| {res['scenario']} | FAILED | | | | | | | | | | |")
            continue
        m, s, r = summ["model_tok_s"], summ["system_tok_s"], summ["request_tok_s"]
        st = summ["step_ms"]
        lines.append(
            f"| {res['scenario']} | {spec.get('execution')} | {spec.get('depth')} | "
            f"{spec.get('capacity') or 'dynamic'} | "
            f"{m['median']:.2f} [{m['min']:.2f}, {m['max']:.2f}] | "
            f"{s['median']:.2f} [{s['min']:.2f}, {s['max']:.2f}] | "
            f"{r['median']:.2f} | {st['p50_ms']:.2f} / {st['p95_ms']:.2f} | "
            f"{summ['prefill_s']['median']:.2f} | {summ['peak_allocated_gb']:.3f} | "
            f"{summ['peak_reserved_gb']:.3f} | {summ['free_gb_min']:.3f} |"
        )
    return "\n".join(lines)


def estimate_seconds(spec: dict, steps: int, repeats: int) -> float:
    """按 2026-10-21 实测的每步耗时估算一个场景的墙钟时间（含 warmup）。"""
    depth = int(spec["depth"])
    prefill = EST_PREFILL_S.get(depth, 6.7)
    if spec["execution"] == "graph":
        ms = EST_GRAPH_MS_PER_STEP.get(spec["capacity"], 44.0)
    else:
        ms = EST_MS_PER_STEP.get(spec["execution"], 86.0)
    per_measure = 2 * prefill + 2 * steps * ms / 1000.0
    return (1 + repeats) * per_measure


def list_scenarios(
    scenarios: list[str], repeats: int, repeats_non_graph: int | None, steps: int
) -> int:
    print(f"{'scenario':18s} {'execution':12s} {'depth':>6s} {'capacity':>9s} "
          f"{'est/measure':>12s} {'est total':>10s}")
    total = 0.0
    for name in scenarios:
        spec = SCENARIOS[name]
        r = repeats
        if repeats_non_graph is not None and spec["execution"] != "graph":
            r = repeats_non_graph
        seconds = estimate_seconds(spec, steps, r)
        total += seconds
        per = seconds / (1 + r)
        print(
            f"{name:18s} {spec['execution']:12s} {spec['depth']:6d} "
            f"{str(spec['capacity'] or 'dynamic'):>9s} {per:11.1f}s {seconds / 60:9.1f}m"
        )
    print(f"\n{len(scenarios)} 个场景，合计约 {total / 60:.1f} 分钟 "
          f"（steps={steps}, repeats={repeats}"
          + (f", non-graph repeats={repeats_non_graph}" if repeats_non_graph else "")
          + "）")
    print("估算是按 2026-10-21 实测每步耗时算的，只用于选场景；正式数字以实跑为准。")
    print("单场景冒烟：--only <name> --repeats 1 --steps 32")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""常用任务：
  列出场景和预计时间        --list
  单场景冒烟               --only graph-512-1024 --repeats 1 --steps 32
  续跑被中断的 run          --resume（配合 --out）
  只重跑部分场景            --only graph-8192,cap-8192
  全量（plan 口径）         make m3-bench
  完整关闭                  scripts/m3_close.sh
文档：docs/bench.md
""",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出场景、每档估算耗时后退出（不占 GPU）",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=ROOT / "bench/workloads/m3-p0",
        help="冻结 fixtures 目录（fixed_<depth>.json + manifest.json；启动时校验 hash）",
    )
    parser.add_argument(
        "--only",
        default="all",
        help="逗号分隔的场景名，或 all（默认）。名字见 --list",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="每个场景的测量次数（默认 5，plan §2.4）；不含 1 次完整 warmup",
    )
    parser.add_argument(
        "--repeats-non-graph",
        type=int,
        default=None,
        help="非 graph 场景的重复次数（默认与 --repeats 相同）；graph 始终用 --repeats",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续跑：只复用源码/运行配置/环境指纹都一致的场景；"
        "源码或运行配置变了会重跑，矩阵配置变了会改写新目录",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=511,
        help="计时 decode 步数（默认 511 = plan §2.2 口径）。小值只用于开发冒烟，"
        "结果不可与正式数字比较",
    )
    parser.add_argument("--chunk-size", type=int, default=128, help="prefill 分块长度（默认 128）")
    parser.add_argument(
        "--graph-warmup",
        type=int,
        default=3,
        help="CUDA Graph capture 前在 side stream 上的 warmup 次数（默认 3）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="run 目录（默认 results/m3/<timestamp>-m3-bench；resume 时建议显式给出）",
    )
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tag", default="m3-bench", help="run 目录后缀（默认 m3-bench）")
    args = parser.parse_args()

    if args.only.strip() in ("", "all"):
        selected = list(SCENARIOS)
    else:
        selected = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in selected if s not in SCENARIOS]
        if unknown:
            raise SystemExit(f"unknown scenarios: {unknown}; see --list")
    if args.list:
        return list_scenarios(selected, args.repeats, args.repeats_non_graph, args.steps)

    if args.worker:
        return run_worker(args)

    scenarios = selected  # 上面已经校验过 --only
    fixtures = load_fixtures(Path(args.fixtures))
    fixtures_sha = fixtures_sha256(fixtures)
    env_fp = env_fingerprint()
    matrix_config = {
        "schema": 2,
        "scenarios": scenarios,
        "steps": int(args.steps),
        "repeats": int(args.repeats),
        "repeats_non_graph": args.repeats_non_graph,
        "chunk_size": int(args.chunk_size),
        "graph_warmup": int(args.graph_warmup),
        "fixtures_sha256": fixtures_sha,
    }

    if args.out:
        out_dir = Path(args.out)
        if args.resume:
            note = dir_config_matches(out_dir, matrix_config, env_fp)
            if note:
                print(
                    f"resume: 目录 {out_dir.name} 与本轮配置不一致（{note}），"
                    "受影响场景会重跑",
                    flush=True,
                )
    elif args.resume and latest_bench_dir() is not None:
        candidate = latest_bench_dir()
        reason = dir_config_matches(candidate, matrix_config, env_fp)
        if not reason:
            out_dir = candidate
            print(f"resume: 复用最近的 run dir {out_dir}", flush=True)
        else:
            # 源码或运行配置与当前不一致：不能往里写，否则会覆盖掉那次记录。
            out_dir = reporting.make_run_dir(ROOT, "bench", args.tag)
            print(
                f"resume: 最近目录 {candidate.name} 不可复用（{reason}），"
                f"改写到新目录 {out_dir.name}",
                flush=True,
            )
    else:
        out_dir = reporting.make_run_dir(ROOT, "bench", args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out_dir}", flush=True)
    reporting.provenance(
        ROOT,
        {
            "scenarios": scenarios,
            "repeats": args.repeats,
            "repeats_non_graph": args.repeats_non_graph,
            "steps": args.steps,
            "chunk_size": args.chunk_size,
            "graph_warmup": args.graph_warmup,
            "fixtures": str(args.fixtures),
            "run_config": matrix_config,
            "environment_fingerprint": env_fp,
        },
        out_dir=out_dir,
    )

    results = []
    for name in scenarios:
        sub = out_dir / name
        if args.resume:
            previous = sub / "result.json"
            eff_steps = effective_steps(fixtures, SCENARIOS[name]["depth"], args.steps)
            run_config = scenario_run_config(
                name,
                SCENARIOS[name],
                eff_steps,
                effective_repeats(SCENARIOS[name], args),
                args,
                fixtures_sha,
            )
            if previous.exists():
                try:
                    prev = json.loads(previous.read_text())
                except Exception as exc:  # noqa: BLE001
                    prev = {"ok": False, "error": f"解析失败: {exc}"}
                reason = "旧结果 ok=false" if not prev.get("ok") else resume_mismatch(
                    prev, run_config, env_fp
                )
                if not reason:
                    print(f"\n=== {name}: resume 跳过（源码/配置/环境一致）", flush=True)
                    results.append(prev)
                    continue
                print(f"\n=== {name}: resume 重跑（{reason}）", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            "--out",
            str(sub),
            "--fixtures",
            str(args.fixtures),
            "--repeats",
            str(args.repeats),
            "--steps",
            str(args.steps),
            "--chunk-size",
            str(args.chunk_size),
            "--graph-warmup",
            str(args.graph_warmup),
        ]
        if args.repeats_non_graph is not None:
            cmd += ["--repeats-non-graph", str(args.repeats_non_graph)]
        print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        path = sub / "result.json"
        if path.exists():
            res = json.loads(path.read_text())
        else:
            res = {"scenario": name, "ok": False, "error": f"no result (rc={rc})"}
        res["returncode"] = rc
        res["ok"] = bool(res.get("ok")) and rc == 0
        results.append(res)
        # 每个场景结束就落一次总表，方便中断后查看
        reporting.save_json(out_dir / "bench.json", {"results": results, "complete": False})

    table = format_table(results)
    payload = {
        "complete": True,
        "scenarios": scenarios,
        "repeats": args.repeats,
        "steps": args.steps,
        "results": results,
        "table": table,
    }
    reporting.save_json(out_dir / "bench.json", payload)
    lines = [
        "M3 P0 decode benchmark",
        f"run_dir: {out_dir}",
        f"repeats: {args.repeats}, steps: {args.steps}, chunk: {args.chunk_size}",
        "",
        table,
        "",
        "说明：",
        "- `model tok/s` = 511 步 CUDA events 之和；D2D copy 在 start event 之前。",
        "- `system tok/s` = 真实贪心闭环墙钟，含 host gather / pinned / H2D / sampling / D2H。",
        "- `request tok/s` = 512 个输出 token /（prefill + decode 墙钟）。",
        "- `peak/reserved/free` 是计时区间的峰值；分阶段峰值（load/cache/prefill/capture）见 `make m3-memory`。",
        "- `trace_match_all` 记录真实闭环是否与冻结 eager trace 完全一致；free-running greedy "
        "跨数值模式不保证 bitwise，差异见 m3-check 的 diagnostic。",
        "",
    ]
    for res in results:
        if not res.get("ok"):
            lines.append(f"{res['scenario']}: FAILED")
            if res.get("error"):
                lines.append(res["error"].splitlines()[-1])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
