#!/usr/bin/env python3
"""M3 P0 fixtures：固定深度 prompt / greedy trace / 三条真实 smoke。

设计：
- 固定深度 prompt 沿用 ``scripts/m2_capacity.py:fixed_ids`` 的重复文本
  ``The quick brown fox jumps over the lazy dog. ``，截到 depth；
- 续写 trace 由 **eager reference**（M2 ``EagerTextRunner``，commit 固定）生成，
  greedy、忽略 EOS，长度 = 固定深度协议的 512 个输出 token；
- 三条 smoke 用真实 chat template + thinking，greedy，2048 上限，正常 EOS；
- 以后 bench/check 只从文件读 token，不重新生成，保证四个执行方式工作量一致；
- 结果写 ``bench/workloads/m3-p0/``，manifest 记录 tokenizer/template hash、
  生成 commit、checkpoint revision 和每个文件的 sha256。

用法：
    .venv/bin/python scripts/m3_fixtures.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import transformers  # noqa: E402

from runner.model.eager_runner import EagerTextRunner  # noqa: E402

CHECKPOINT = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"
CHECKPOINT_REVISION = "5b7a687fc8211a5d631c8ca6a593dd37eb26ce33"
FIXED_TEXT = "The quick brown fox jumps over the lazy dog. "
SMOKE_PROMPTS = (
    ("math", "results/payload_math.json"),
    ("code", "results/payload_code.json"),
    ("chat", "results/payload_chat.json"),
)
# 固定深度协议：深度 + 512 个输出 token，其中首个来自 prefill logits。
N_GEN_TRACE = 512


def fixed_ids(tokenizer, depth: int) -> torch.Tensor:
    ids = tokenizer(FIXED_TEXT, return_tensors="pt", add_special_tokens=False).input_ids[0]
    reps = math.ceil(depth / ids.numel())
    return ids.repeat(reps)[:depth]


def sha256_json(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # pragma: no cover - 无 git 时留空
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "bench/workloads/m3-p0")
    parser.add_argument("--depths", default="512,7680")
    parser.add_argument("--n-gen", type=int, default=N_GEN_TRACE)
    parser.add_argument("--smoke-max-new-tokens", type=int, default=2048)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    runner, info = EagerTextRunner.from_checkpoint(CHECKPOINT, manifest_path=args.manifest)
    tokenizer = runner.tokenizer
    print("load:", json.dumps(info, ensure_ascii=False), flush=True)

    chat_template = getattr(tokenizer, "chat_template", None)
    files: dict[str, dict] = {}

    def write_fixture(name: str, payload: dict) -> None:
        path = out_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        files[name] = {
            "sha256": sha256_json(payload),
            "bytes": path.stat().st_size,
        }
        print(f"wrote {name}", flush=True)

    for depth in [int(x) for x in args.depths.split(",") if x.strip()]:
        ids = fixed_ids(tokenizer, depth)
        if ids.numel() != depth:
            raise RuntimeError(f"fixed_ids returned {ids.numel()} != {depth}")
        prompt_ids = ids.unsqueeze(0)
        result = runner.generate_ids(
            prompt_ids,
            max_new_tokens=args.n_gen,
            greedy=True,
            stop_on_eos=False,  # 固定工作量：忽略 EOS
        )
        trace = result.token_ids
        if len(trace) != args.n_gen:
            raise RuntimeError(f"trace length {len(trace)} != {args.n_gen}")
        write_fixture(
            f"fixed_{depth}.json",
            {
                "kind": "fixed_depth",
                "depth": depth,
                "n_gen": args.n_gen,
                "text": FIXED_TEXT,
                "prompt_ids": prompt_ids[0].tolist(),
                "prompt_sha256": sha256_json(prompt_ids[0].tolist()),
                "trace": trace,
                "trace_sha256": sha256_json(trace),
                "generated_by": "EagerTextRunner.generate_ids(greedy=True, stop_on_eos=False)",
                "prefill_s": result.prefill_s,
                "decode_s": result.decode_s,
                "decode_tok_s": result.decode_tok_s,
                "cache_len": result.cache_len,
                "note": "trace[0] 来自 prefill logits；trace[:511] 是 511 次 decode forward 的输入",
            },
        )
        del result
        torch.cuda.empty_cache()

    for name, payload_path in SMOKE_PROMPTS:
        payload = json.loads((ROOT / payload_path).read_text())
        result = runner.generate_messages(
            payload["messages"],
            max_new_tokens=args.smoke_max_new_tokens,
            greedy=True,
            stop_on_eos=True,
        )
        write_fixture(
            f"smoke_{name}.json",
            {
                "kind": "smoke",
                "name": name,
                "source": payload_path,
                "messages": payload["messages"],
                "max_new_tokens": args.smoke_max_new_tokens,
                "prompt_tokens": result.prompt_tokens,
                "prompt_ids": runner.encode_messages(payload["messages"])[0].tolist(),
                "trace": result.token_ids,
                "trace_sha256": sha256_json(result.token_ids),
                "finish_reason": result.finish_reason,
                "text": result.text,
                "generated_by": "EagerTextRunner.generate_messages(greedy=True, thinking)",
                "prefill_s": result.prefill_s,
                "decode_s": result.decode_s,
            },
        )
        print(
            f"smoke {name}: prompt={result.prompt_tokens} gen={result.generated_tokens} "
            f"finish={result.finish_reason}",
            flush=True,
        )
        del result
        torch.cuda.empty_cache()

    manifest = {
        "kind": "m3-p0-fixtures",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_revision": CHECKPOINT_REVISION,
        "reference_runner": "runner/model/eager_runner.py (M2 eager, frozen at ca1bd2e)",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": sha256_text(chat_template) if chat_template else None,
        "fixed_text": FIXED_TEXT,
        "n_gen": args.n_gen,
        "smoke_max_new_tokens": args.smoke_max_new_tokens,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
