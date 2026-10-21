#!/usr/bin/env python3
"""M3 P0 decode benchmark：四执行方式 × depth/capacity 矩阵，两套计时。

规格（docs/plans/m3.md §2–§3）：
- 每个场景在独立子进程里跑，避免 graph pool / JIT workspace / allocator 历史干扰；
- prompt = ``bench/workloads/m3-p0/fixed_<depth>.json`` 的固定 token；
- 计时协议：prefill 产生首 token（计入 TTFT），随后 **511 次 decode forward /
  511 个输出 token** 构成计时区间；最终 cache 长度 = depth + 511；
- 模型 decode tok/s：逐步 CUDA events，D2D copy 在 start event 之前；
- 系统 decode tok/s：host gather / pinned / H2D / 模型 / sampling / D2H 全计时；
- 每个场景先完整 warmup 一次，再测 5 次；每次重建逻辑初态并重新 prefill；
- 原始逐步/逐 run 数据全部保存，汇总另外给中位数/极值/波动和 step 分布。

用法：
    .venv/bin/python bench/e2e/decode.py                      # 全部场景
    .venv/bin/python bench/e2e/decode.py --only graph-512-1024 --repeats 1 --steps 32   # 冒烟
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


def same_source(result: dict) -> bool:
    """resume 时判断旧结果是否由同一份源码产生。"""
    prev = (result.get("provenance") or {}).get("git") or {}
    manifest = (result.get("provenance") or {}).get("source_manifest") or {}
    current = reporting.git_state(ROOT)
    current_manifest = reporting.source_manifest(ROOT)
    return (
        prev.get("commit") == current.get("commit")
        and prev.get("dirty") == current.get("dirty")
        and manifest.get("patch_sha256") == current_manifest.get("patch_sha256")
    )


def latest_bench_dir() -> Path | None:
    dirs = sorted((ROOT / "results/m3").glob("*-m3-bench"))
    return dirs[-1] if dirs else None


def dir_source_matches(path: Path) -> bool:
    """run dir 的 provenance 是否属于当前源码（用于决定能否 resume 进去）。"""
    prov_path = path / "provenance.json"
    if not prov_path.exists():
        return False
    try:
        prov = json.loads(prov_path.read_text())
    except Exception:
        return False
    prev = prov.get("git") or {}
    prev_manifest = prov.get("source_manifest") or {}
    current = reporting.git_state(ROOT)
    current_manifest = reporting.source_manifest(ROOT)
    return (
        prev.get("commit") == current.get("commit")
        and prev.get("dirty") == current.get("dirty")
        and prev_manifest.get("patch_sha256") == current_manifest.get("patch_sha256")
    )


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
    steps = min(args.steps, len(full_trace) - 1)
    if spec["execution"] != "eager":
        if depth + steps > capacity:
            raise ValueError(f"{name}: depth {depth} + {steps} > capacity {capacity}")
        runner.require_capacity(0)
    trace = full_trace

    result: dict = {
        "scenario": name,
        "spec": spec,
        "steps": steps,
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
    repeats = args.repeats
    if (
        args.repeats_non_graph is not None
        and args.repeats_non_graph > 0
        and spec["execution"] != "graph"
    ):
        repeats = args.repeats_non_graph
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "bench/workloads/m3-p0")
    parser.add_argument("--only", default="all")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--repeats-non-graph",
        type=int,
        default=None,
        help="非 graph 场景的重复次数（默认与 --repeats 相同）；graph 始终用 --repeats",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用 --out（或最新的 *-m3-bench）里已成功且源码一致的场景",
    )
    parser.add_argument("--steps", type=int, default=511)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--tag", default="m3-bench")
    args = parser.parse_args()

    if args.worker:
        return run_worker(args)

    if args.only.strip() in ("", "all"):
        scenarios = list(SCENARIOS)
    else:
        scenarios = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in scenarios if s not in SCENARIOS]
        if unknown:
            raise SystemExit(f"unknown scenarios: {unknown}")

    if args.out:
        out_dir = Path(args.out)
    elif args.resume and latest_bench_dir() is not None:
        candidate = latest_bench_dir()
        if dir_source_matches(candidate):
            out_dir = candidate
            print(f"resume: 复用最近的 run dir {out_dir}", flush=True)
        else:
            # 最近的 dir 来自另一份源码：不能往里写，否则会覆盖掉那次记录。
            out_dir = reporting.make_run_dir(ROOT, "bench", args.tag)
            print(
                f"resume: 最近目录 {candidate.name} 的源码与当前不一致，"
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
            "steps": args.steps,
            "chunk_size": args.chunk_size,
            "graph_warmup": args.graph_warmup,
            "fixtures": str(args.fixtures),
        },
        out_dir=out_dir,
    )

    results = []
    for name in scenarios:
        sub = out_dir / name
        if args.resume:
            previous = sub / "result.json"
            if previous.exists():
                try:
                    prev = json.loads(previous.read_text())
                except Exception:
                    prev = {}
                if prev.get("ok") and same_source(prev):
                    print(f"\n=== {name}: resume 跳过（已有 ok 结果，源码一致）", flush=True)
                    results.append(prev)
                    continue
                print(f"\n=== {name}: resume 重跑（旧结果无效或源码不同）", flush=True)
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
