#!/usr/bin/env python3
"""M3 分阶段显存：load / cache 初始化 / prefill / capture / graph 常驻后 prefill / decode / reset。

规格（docs/plans/m3.md §5.2）：
- 每个阶段开始 reset peak、结束同步，记录 peak allocated / reserved、阶段末占用和
  ``mem_get_info``；峰值不能用阶段结束值代替；
- P0 必测组合：**graph 已常驻 -> 原地 reset -> 7680 prefill -> 511 decode ->
  reset 后再次 prefill**，全程无 OOM / stale state / 持续增长；
- 记录 graph-private pool 增量、cache 各部分字节、输入/logits staging；
- 每个场景独立进程，避免多份 cache / pool 同时驻留。

用法：
    .venv/bin/python scripts/m3_memory.py
    .venv/bin/python scripts/m3_memory.py --only graph-8192 --steps 32   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from runner import report as reporting  # noqa: E402
from runner.loader.nvfp4_model import load_nvfp4_text_model  # noqa: E402
from runner.model.decode_runner import DecodeRunner  # noqa: E402
from runner.model.static_cache import cache_stats  # noqa: E402

SCENARIOS: dict[str, dict] = {
    "eager-8192": {"execution": "eager", "depth": 7680, "capacity": None},
    "exact-8192": {"execution": "static-exact", "depth": 7680, "capacity": 8192},
    "cap-8192": {"execution": "static-cap", "depth": 7680, "capacity": 8192},
    "graph-8192": {"execution": "graph", "depth": 7680, "capacity": 8192},
    "cap-512-1024": {"execution": "static-cap", "depth": 512, "capacity": 1024},
    "graph-512-1024": {"execution": "graph", "depth": 512, "capacity": 1024},
}


def smi_used_mib() -> float | None:
    text = reporting._run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    try:
        return float(text.splitlines()[0])
    except Exception:
        return None


def load_fixtures(fixtures_dir: Path) -> dict:
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())
    files = {}
    for name in manifest["files"]:
        payload = json.loads((fixtures_dir / name).read_text())
        actual = reporting._sha256_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        if actual != manifest["files"][name]["sha256"]:
            raise RuntimeError(f"fixture hash mismatch: {name}")
        files[name] = payload
    return files


class MemRecorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def stage(self, name: str, fn, **extra) -> dict:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        free_b, total_b = torch.cuda.mem_get_info()
        row = {
            "stage": name,
            "seconds": time.perf_counter() - t0,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "end_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "end_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "free_gb": free_b / 1e9,
            "total_gb": total_b / 1e9,
            "smi_used_mib": smi_used_mib(),
        }
        row.update(extra)
        if isinstance(out, dict):
            row.update({k: v for k, v in out.items() if k not in row})
        self.rows.append(row)
        print(
            f"  [{name}] peak {row['peak_allocated_gb']:.3f}/{row['peak_reserved_gb']:.3f} GB "
            f"end {row['end_allocated_gb']:.3f}/{row['end_reserved_gb']:.3f} "
            f"free {row['free_gb']:.3f} s={row['seconds']:.2f}",
            flush=True,
        )
        return row


def run_scenario(name: str, spec: dict, fixtures: dict, args) -> dict:
    depth = int(spec["depth"])
    capacity = spec["capacity"]
    fx = fixtures[f"fixed_{depth}.json"]
    prompt = torch.tensor([fx["prompt_ids"]], dtype=torch.long)
    trace = list(fx["trace"])
    steps = min(args.steps, len(trace) - 1)
    rec = MemRecorder()
    result: dict = {"scenario": name, "spec": spec, "steps": steps}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer, load_info = load_nvfp4_text_model(ROOT / reporting.CHECKPOINT)
    torch.cuda.synchronize()
    rec.rows.append(
        {
            "stage": "load",
            "seconds": load_info["load_s"],
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "end_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "end_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "free_gb": torch.cuda.mem_get_info()[0] / 1e9,
            "smi_used_mib": smi_used_mib(),
        }
    )
    print(f"  [load] end {load_info['gpu_allocated_gb']:.3f}/{load_info['gpu_reserved_gb']:.3f}", flush=True)
    model.eval()
    torch.set_grad_enabled(False)

    runner = DecodeRunner(
        model,
        tokenizer,
        execution=spec["execution"],
        capacity=capacity,
        prefill_chunk=args.chunk_size,
    )

    cache_extra: dict = {}
    if capacity is not None:
        def build():
            info = runner.prepare()
            stats = cache_stats(runner.cache, check_finite=False).to_dict()
            return {**{f"cache_{k}": v for k, v in stats.items() if isinstance(v, (int, float))}}

        row = rec.stage("cache_init", build)
        cache_extra = {k: v for k, v in row.items() if k.startswith("cache_")}
    else:
        rec.stage("cache_init", lambda: runner.prepare())

    result["cache_bytes"] = cache_extra
    result["harness"] = runner.harness_info()

    # 第一次 prefill（无 graph / capture 之前）
    holder: dict = {}

    def do_prefill_cold():
        logits, pre = runner.prefill(prompt)
        holder["logits"] = logits
        return {"prefill_s": pre.prefill_s, "cache_len": pre.cache_len}

    rec.stage("prefill_cold", do_prefill_cold)

    if spec["execution"] == "graph":
        def do_capture():
            info = runner.ensure_graph(prompt)
            return {
                "capture_s": info.get("capture_s"),
                "graph_pool_allocated_gb": (
                    info["mem_after_allocated"] - info["mem_before_allocated"]
                )
                / 1e9,
                "graph_pool_reserved_gb": (
                    info["mem_after_reserved"] - info["mem_before_reserved"]
                )
                / 1e9,
                "warmup_length_delta": info.get("warmup_length_delta"),
            }

        rec.stage("warmup_capture", do_capture)
        result["graph"] = runner.graph_info

    def do_prefill_again():
        logits, pre = runner.prefill(prompt)
        holder["logits"] = logits
        return {"prefill_s": pre.prefill_s, "cache_len": pre.cache_len}

    stage_name = "prefill_graph_resident" if spec["execution"] == "graph" else "prefill_again"
    row_gresident = rec.stage(stage_name, do_prefill_again)

    # 511 步 decode
    def do_decode():
        res = runner.model_timed_decode(trace[:steps])
        holder["decode"] = res
        return {
            "decode_peak_allocated_gb": res["peak_allocated_gb"],
            "decode_peak_reserved_gb": res["peak_reserved_gb"],
            "decode_tok_s": res["model_tok_s"],
            "cache_len_end": res["cache_len_end"],
        }

    rec.stage("decode", do_decode)

    # reset + 再次 prefill：检查无持续增长
    kv_layer = None
    if capacity is not None:
        kv_layer = next(l for l in runner.cache.layers if hasattr(l, "cumulative_length"))
    first_kv_ptr = int(kv_layer.keys.data_ptr()) if kv_layer is not None else None

    def do_reset_reprefill():
        logits, pre = runner.prefill(prompt)
        return {"prefill_s": pre.prefill_s, "cache_len": pre.cache_len}

    row_after_reset = rec.stage("reset_reprefill", do_reset_reprefill)
    second_kv_ptr = int(kv_layer.keys.data_ptr()) if kv_layer is not None else None

    # 第二个生命周期：再 decode 16 步 + reset + prefill，用来区分「一次性 workspace」
    # 和「持续增长」。
    def do_decode2():
        res = runner.model_timed_decode(trace[: min(16, steps)])
        return {
            "decode_tok_s": res["model_tok_s"],
            "cache_len_end": res["cache_len_end"],
        }

    rec.stage("decode_cycle2", do_decode2)
    row_cycle2 = rec.stage("reset_reprefill_cycle2", do_reset_reprefill)

    result["stages"] = rec.rows
    result["growth"] = {
        "allocated_delta_gb": row_after_reset["end_allocated_gb"] - row_gresident["end_allocated_gb"],
        "reserved_delta_gb": row_after_reset["end_reserved_gb"] - row_gresident["end_reserved_gb"],
        "cycle2_allocated_delta_gb": row_cycle2["end_allocated_gb"] - row_after_reset["end_allocated_gb"],
        "cycle2_reserved_delta_gb": row_cycle2["end_reserved_gb"] - row_after_reset["end_reserved_gb"],
        "kv_data_ptr_stable": (first_kv_ptr == second_kv_ptr) if first_kv_ptr is not None else True,
        "kv_data_ptr": first_kv_ptr,
        "kv_data_ptr_note": "eager 无固定 KV buffer，跳过" if first_kv_ptr is None else "",
    }
    result["peaks"] = {
        "load_peak_allocated_gb": next(r for r in rec.rows if r["stage"] == "load")["peak_allocated_gb"],
        "max_peak_allocated_gb": max(r["peak_allocated_gb"] for r in rec.rows),
        "max_peak_reserved_gb": max(r["peak_reserved_gb"] for r in rec.rows),
        "min_free_gb": min(r["free_gb"] for r in rec.rows),
        "final_end_allocated_gb": rec.rows[-1]["end_allocated_gb"],
        "final_end_reserved_gb": rec.rows[-1]["end_reserved_gb"],
    }
    result["ok"] = (
        # 第一个 delta 允许一次性 workspace（例如 FlashInfer/cudnn 计划缓存），
        # 但跨两个生命周期必须不再增长。
        abs(result["growth"]["allocated_delta_gb"]) < 0.05
        and abs(result["growth"]["cycle2_allocated_delta_gb"]) < 0.008
        and abs(result["growth"]["cycle2_reserved_delta_gb"]) < 0.008
        and result["growth"]["kv_data_ptr_stable"]
        and result["peaks"]["min_free_gb"] > 0
    )
    del runner
    torch.cuda.empty_cache()
    return result


def run_worker(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixtures = load_fixtures(Path(args.fixtures))
    try:
        result = run_scenario(args.worker, SCENARIOS[args.worker], fixtures, args)
    except Exception:
        result = {"scenario": args.worker, "ok": False, "error": traceback.format_exc()}
        print(result["error"], flush=True)
    result["provenance"] = reporting.provenance(ROOT, {"worker": args.worker})
    reporting.save_json(out / "result.json", result)
    print(f"worker {args.worker}: ok={result['ok']}", flush=True)
    return 0 if result["ok"] else 1


def format_table(results: list[dict]) -> str:
    lines = [
        "| scenario | execution | depth | capacity | load peak GB | cache init GB | "
        "prefill peak GB | capture pool GB | graph-resident prefill peak GB | decode peak GB | "
        "max peak GB | min free GB | reset->prefill delta GB | KV ptr stable |",
        "|" + "---|" * 14,
    ]
    for res in results:
        spec = res.get("spec", {})
        stages = {r["stage"]: r for r in res.get("stages", [])}
        if not stages:
            lines.append(f"| {res['scenario']} | FAILED | | | | | | | | | | | | |")
            continue

        def peak(name):
            row = stages.get(name)
            return f"{row['peak_allocated_gb']:.3f}" if row else "-"

        cap = stages.get("warmup_capture", {})
        pool = cap.get("graph_pool_allocated_gb")
        grow = res.get("growth", {})
        lines.append(
            f"| {res['scenario']} | {spec.get('execution')} | {spec.get('depth')} | "
            f"{spec.get('capacity') or 'dynamic'} | {peak('load')} | {peak('cache_init')} | "
            f"{peak('prefill_cold')} | {f'{pool:.3f}' if pool is not None else '-'} | "
            f"{peak('prefill_graph_resident') if 'prefill_graph_resident' in stages else peak('prefill_again')} | "
            f"{peak('decode')} | {res.get('peaks', {}).get('max_peak_allocated_gb', 0):.3f} | "
            f"{res.get('peaks', {}).get('min_free_gb', 0):.3f} | "
            f"{grow.get('allocated_delta_gb', 0) * 1000:.1f} MB | {grow.get('kv_data_ptr_stable')} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "bench/workloads/m3-p0")
    parser.add_argument("--only", default="all")
    parser.add_argument("--steps", type=int, default=511)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--tag", default="m3-memory")
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

    out_dir = Path(args.out) if args.out else reporting.make_run_dir(ROOT, "memory", args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out_dir}", flush=True)
    reporting.save_json(
        out_dir / "provenance.json",
        reporting.provenance(ROOT, {"scenarios": scenarios, "steps": args.steps}),
    )

    results = []
    for name in scenarios:
        sub = out_dir / name
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            "--out",
            str(sub),
            "--fixtures",
            str(args.fixtures),
            "--steps",
            str(args.steps),
            "--chunk-size",
            str(args.chunk_size),
        ]
        print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        path = sub / "result.json"
        res = json.loads(path.read_text()) if path.exists() else {"scenario": name, "ok": False}
        res["returncode"] = rc
        res["ok"] = bool(res.get("ok")) and rc == 0
        results.append(res)
        reporting.save_json(out_dir / "memory.json", {"results": results, "complete": False})

    table = format_table(results)
    reporting.save_json(
        out_dir / "memory.json", {"results": results, "complete": True, "table": table}
    )
    lines = [
        "M3 分阶段显存",
        f"run_dir: {out_dir}",
        f"steps: {args.steps}",
        "",
        table,
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
