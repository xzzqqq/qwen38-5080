#!/usr/bin/env python3
"""M2 chunked prefill 一致性：只比较“边界对齐”的 full vs chunked。

为什么不是任意长度：
- full 300 的第 2 段 attention 看到 KV 长度 300（含 44 个被 causal mask 的 future）；
- chunked 128+128+44 的第 2 段只看到 KV 长度 256；
- 两者的有效 keys 相同，但 attention reduction/softmax 长度不同，FP 结果不会 bitwise 相等。
因此 reference 必须取 chunk 边界的 full 前缀（如 256 vs 128+128）；这个路径应当 exact。

用法：
    .venv/bin/python scripts/m2_chunk_consistency.py --length 256 --chunk 128
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from runner.model.eager_runner import EagerTextRunner  # noqa: E402
from scripts.m2_capacity import fixed_ids  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    if args.length % args.chunk != 0:
        raise ValueError("length must be a multiple of chunk for a boundary-aligned reference")

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ROOT / "results/m2" / f"{run_id}-chunk-consistency"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner, info = EagerTextRunner.from_checkpoint(
        ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090",
        prefill_chunk=args.chunk,
        manifest_path=args.manifest,
    )
    model = runner.model
    ids = fixed_ids(runner.tokenizer, args.length)

    with torch.no_grad():
        full = model(
            input_ids=ids,
            past_key_values=None,
            use_cache=False,
            logits_to_keep=1,
        )
        cache = DynamicCache(config=model.config)
        chunked = None
        for start in range(0, args.length, args.chunk):
            chunked = model(
                input_ids=ids[:, start : start + args.chunk],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        a = full.logits[:, -1, :].float()
        b = chunked.logits[:, -1, :].float()
        max_abs = (a - b).abs().max().item()
        cosine = torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0
        ).item()
        argmax_full = int(a.argmax(dim=-1))
        argmax_chunk = int(b.argmax(dim=-1))

    record = {
        "length": args.length,
        "chunk": args.chunk,
        "max_abs": max_abs,
        "cosine": cosine,
        "argmax_full": argmax_full,
        "argmax_chunk": argmax_chunk,
        "exact": max_abs == 0.0 and argmax_full == argmax_chunk,
        "load": info,
    }
    (out_dir / "chunk_consistency.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False), flush=True)
    print(f"output: {out_dir}")
    return 0 if record["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
