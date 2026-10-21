#!/usr/bin/env python3
"""M2 capacity：固定 token 序列 + 512 步 decode，检查 512 / 7680 depth。

不因 EOS 提前退出；目的只是验证 cache/state 增长、显存峰值和 8K 不 OOM。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from runner.model.eager_runner import EagerTextRunner  # noqa: E402


def fixed_ids(tokenizer, depth: int) -> torch.Tensor:
    text = "The quick brown fox jumps over the lazy dog. "
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    reps = math.ceil(depth / ids.numel())
    return ids.repeat(reps)[:depth].unsqueeze(0)


def cache_stats(cache: DynamicCache) -> dict:
    kv_dtypes: set[str] = set()
    rec_dtypes: set[str] = set()
    conv_dtypes: set[str] = set()
    finite = True
    kv_layers = rec_layers = conv_layers = 0
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is not None and values is not None:
            kv_layers += 1
            kv_dtypes.add(str(keys.dtype))
            finite = finite and bool(torch.isfinite(keys).all()) and bool(torch.isfinite(values).all())
        states = getattr(layer, "recurrent_states", None)
        if states:
            rec_layers += 1
            for state in states.values():
                rec_dtypes.add(str(state.dtype))
                finite = finite and bool(torch.isfinite(state).all())
        conv = getattr(layer, "conv_states", None)
        if conv:
            conv_layers += 1
            for state in conv.values():
                conv_dtypes.add(str(state.dtype))
                finite = finite and bool(torch.isfinite(state).all())
    return {
        "finite": finite,
        "kv_layers": kv_layers,
        "recurrent_layers": rec_layers,
        "conv_layers": conv_layers,
        "kv_dtypes": sorted(kv_dtypes),
        "recurrent_dtypes": sorted(rec_dtypes),
        "conv_dtypes": sorted(conv_dtypes),
    }


def run_depth(runner: EagerTextRunner, depth: int, n_decode: int, load_info: dict) -> dict:
    model = runner.model
    input_ids = fixed_ids(runner.tokenizer, depth)
    cache = DynamicCache(config=model.config)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    out = None
    for start in range(0, depth, runner.prefill_chunk):
        chunk = input_ids[:, start : start + runner.prefill_chunk]
        out = model(
            input_ids=chunk,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    prefill_peak = torch.cuda.max_memory_allocated() / 1e9
    logits = out.logits[:, -1, :]
    finite = bool(torch.isfinite(logits).all())

    t0 = time.perf_counter()
    for step in range(n_decode):
        next_id = logits.argmax(dim=-1, keepdim=True)
        if step + 1 >= n_decode:
            break
        out = model(
            input_ids=next_id.cpu(),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = out.logits[:, -1, :]
        finite = finite and bool(torch.isfinite(logits).all())
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0
    decode_peak = torch.cuda.max_memory_allocated() / 1e9
    free_b, _ = torch.cuda.mem_get_info()
    stats = cache_stats(cache)
    cache_len = int(cache.get_seq_length())
    expected = depth + n_decode - 1
    ok = (
        finite
        and stats["finite"]
        and cache_len == expected
        and stats["kv_dtypes"] == ["torch.bfloat16"]
        and stats["recurrent_dtypes"] == ["torch.float32"]
        and stats["conv_dtypes"] == ["torch.bfloat16"]
    )
    return {
        "depth": depth,
        "n_decode": n_decode,
        "prompt_tokens": depth,
        "cache_len": cache_len,
        "cache_len_expected": expected,
        "cache_len_ok": cache_len == expected,
        "finite": finite and stats["finite"],
        "cache_stats": stats,
        "ok": ok,
        "load_allocated_gb": load_info["gpu_allocated_gb"],
        "load_reserved_gb": load_info["gpu_reserved_gb"],
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tok_s": (n_decode - 1) / decode_s if decode_s else 0.0,
        "prefill_peak_gb": prefill_peak,
        "decode_peak_gb": decode_peak,
        "final_peak_gb": decode_peak,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "free_gb": free_b / 1e9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", default="512,7680")
    parser.add_argument("--n-decode", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=8192)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ROOT / "results/m2" / f"{run_id}-capacity"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner, info = EagerTextRunner.from_checkpoint(
        ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090",
        max_context=args.max_context,
        prefill_chunk=args.chunk_size,
        manifest_path=args.manifest,
    )
    print("load:", json.dumps(info, ensure_ascii=False), flush=True)

    results = []
    for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
        if depth + args.n_decode > args.max_context:
            raise ValueError(f"depth {depth} + {args.n_decode} exceeds {args.max_context}")
        print(f"running depth={depth} ...", flush=True)
        row = run_depth(runner, depth, args.n_decode, info)
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        torch.cuda.empty_cache()

    (out_dir / "capacity.json").write_text(
        json.dumps({"load": info, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [f"run_id: {run_id}", f"load: {info}", ""]
    for r in results:
        lines += [
            f"depth={r['depth']} n_decode={r['n_decode']}",
            f"  cache_len={r['cache_len']}",
            f"  prefill_s={r['prefill_s']:.3f}",
            f"  decode_s={r['decode_s']:.3f} ({r['decode_tok_s']:.2f} tok/s)",
            f"  cache_len_ok={r['cache_len_ok']} finite={r['finite']} ok={r['ok']}",
            f"  state_dtypes={r['cache_stats']}",
            f"  load_peak_gb={r['load_allocated_gb']:.3f}/{r['load_reserved_gb']:.3f}",
            f"  prefill_peak_gb={r['prefill_peak_gb']:.3f} decode_peak_gb={r['decode_peak_gb']:.3f}",
            f"  reserved_gb={r['reserved_gb']:.3f} free_gb={r['free_gb']:.3f}",
            "",
        ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output: {out_dir}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
