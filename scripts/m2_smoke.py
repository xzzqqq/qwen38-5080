#!/usr/bin/env python3
"""M2 smoke：math / code / chat 三条真实 prompt 的 eager 生成。

用法（fish 先 source scripts/m2_env.fish）：
    .venv/bin/python scripts/m2_smoke.py --max-new-tokens 64
    .venv/bin/python scripts/m2_smoke.py --max-new-tokens 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runner.model.eager_runner import EagerTextRunner, result_to_dict  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        default="results/payload_math.json,results/payload_code.json,results/payload_chat.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ROOT / "results/m2" / f"{run_id}-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner, info = EagerTextRunner.from_checkpoint(
        ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090",
        device=args.device,
        max_context=args.max_context,
        prefill_chunk=args.chunk_size,
        manifest_path=args.manifest,
    )
    print("load:", json.dumps(info, ensure_ascii=False), flush=True)

    records = []
    for prompt_path in [p.strip() for p in args.prompts.split(",") if p.strip()]:
        payload = json.loads((ROOT / prompt_path).read_text())
        result = runner.generate_messages(
            payload["messages"],
            max_new_tokens=args.max_new_tokens,
            greedy=float(payload.get("temperature", 0)) == 0,
        )
        record = {"prompt_file": prompt_path, **result_to_dict(result)}
        records.append(record)
        print(
            f"{prompt_path}: prompt={result.prompt_tokens} gen={result.generated_tokens} "
            f"prefill={result.prefill_s:.3f}s decode={result.decode_s:.3f}s "
            f"{result.decode_tok_s:.2f} tok/s finish={result.finish_reason} "
            f"peak={result.peak_allocated_gb:.3f}GB",
            flush=True,
        )
        print(result.text[:600].replace("\n", "\\n"), flush=True)

    (out_dir / "smoke.json").write_text(
        json.dumps({"load": info, "results": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [f"run_id: {run_id}", f"load: {info}", ""]
    for r in records:
        lines += [
            f"=== {r['prompt_file']}",
            f"prompt_tokens: {r['prompt_tokens']}",
            f"generated_tokens: {r['generated_tokens']}",
            f"prefill_s: {r['prefill_s']:.4f}",
            f"decode_s: {r['decode_s']:.4f}",
            f"decode_tok_s: {r['decode_tok_s']:.2f}",
            f"finish_reason: {r['finish_reason']}",
            f"peak_allocated_gb: {r['peak_allocated_gb']:.4f}",
            f"free_gb: {r['free_gb']:.4f}",
            "text:",
            r["text"],
            "",
        ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
