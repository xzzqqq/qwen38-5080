#!/usr/bin/env python3
"""M3 profiling 入口：在 NVTX 标记下跑一段代表性 decode，供 Nsight Systems 采集。

每个 decode step 的结构（和正式 harness 一致）：
    [mark step]
      [range host_gather]    CPU BF16 embedding gather
      [range h2d]            pinned -> 固定 GPU 输入 buffer
      [range model]          eager/static forward 或 graph replay
      [range sample]         argmax + D2H（系统速度成分）
    [range_end]

用法（由 scripts/m3_profile.sh 调用；也可以手动跑着看时间线）：
    .venv/bin/python scripts/m3_profile.py --execution graph --depth 512 --steps 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runner import report as reporting  # noqa: E402
from runner.loader.nvfp4_model import load_nvfp4_text_model  # noqa: E402
from runner.model.decode_runner import DecodeRunner  # noqa: E402


def nvtx_push(name: str) -> None:
    torch.cuda.nvtx.range_push(name)


def nvtx_pop() -> None:
    torch.cuda.nvtx.range_pop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", default="eager")
    parser.add_argument("--depth", type=int, default=512)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "bench/workloads/m3-p0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    capacity = args.capacity
    if args.execution != "eager" and capacity is None:
        capacity = 1024 if args.depth <= 512 else 8192
    if args.execution == "eager":
        capacity = None

    fixture = json.loads((args.fixtures / f"fixed_{args.depth}.json").read_text())
    prompt = torch.tensor([fixture["prompt_ids"]], dtype=torch.long)
    trace = list(fixture["trace"])

    model, config, load_info = load_nvfp4_text_model(ROOT / reporting.CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / reporting.CHECKPOINT), trust_remote_code=True
    )
    model.eval()
    torch.set_grad_enabled(False)

    runner = DecodeRunner(
        model,
        tokenizer,
        execution=args.execution,
        capacity=capacity,
        prefill_chunk=args.chunk_size,
    )
    runner.prepare()
    if args.execution == "graph":
        runner.ensure_graph(prompt)
    runner.prefill(prompt)

    # warmup：让 FlashInfer / cuBLAS / cudnn 的 workspace 与 tactic 定下来
    for i in range(args.warmup):
        row = runner.host_gather([trace[i]])
        runner.stage_row(row, pinned=True)
        runner._step()
    torch.cuda.synchronize()

    # 只用 cudaProfilerApi 的区间进 nsys 报告：这样 report 里只有这 N 个 decode step，
    # 不会把 load / prefill / warmup 的 kernel 也平均进来（否则“每步 kernel 数”会偏大）。
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"decode:{args.execution}:{args.depth}")
    rows = []
    for i in range(args.steps):
        torch.cuda.nvtx.range_push(f"step{i}")
        t0 = time.perf_counter()
        nvtx_push("host_gather")
        row = runner.host_gather([trace[i]])
        nvtx_pop()
        nvtx_push("h2d")
        runner.stage_row(row, pinned=True)
        nvtx_pop()
        nvtx_push("model")
        logits = runner._step()
        nvtx_pop()
        nvtx_push("sample")
        token = int(logits[:, -1, :].argmax(dim=-1).item())
        nvtx_pop()
        torch.cuda.synchronize()
        rows.append({"step": i, "wall_ms": (time.perf_counter() - t0) * 1000, "token": token})
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    torch.cuda.synchronize()

    payload = {
        "execution": args.execution,
        "depth": args.depth,
        "capacity": capacity,
        "steps": args.steps,
        "load": load_info,
        "harness": runner.harness_info(),
        "graph": runner.graph_info,
        "rows": rows,
        "provenance": reporting.provenance(ROOT),
    }
    if args.out:
        reporting.save_json(Path(args.out), payload)
    print(json.dumps({k: payload[k] for k in ("execution", "depth", "capacity", "steps")}))
    print(f"wall per step ms: {[round(r['wall_ms'], 2) for r in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
