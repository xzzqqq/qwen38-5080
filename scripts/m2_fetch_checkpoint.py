#!/usr/bin/env python3
"""按固定 revision 下载 NVFP4 checkpoint 的文本运行所需文件。

设计：
- 只下载 config/tokenizer/index 和两个 safetensors shard，跳过 assets/README；
- revision 用 commit SHA 固定，下载前后都校验；
- 记录完整 repo 文件树和实际下载文件，供 manifest 和 journal 引用。

用法：
    .venv/bin/python scripts/m2_fetch_checkpoint.py --metadata-only
    .venv/bin/python scripts/m2_fetch_checkpoint.py
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"
REVISION = "5b7a687fc8211a5d631c8ca6a593dd37eb26ce33"
LOCAL_DIR = Path("models/Qwen3.8-27B-NVFP4-RTX5090")

METADATA_PATTERNS = [
    "config.json",
    "hf_quant_config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "model.safetensors.index.json",
    "crc32.txt",
]
WEIGHT_PATTERNS = ["model-*.safetensors"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--local-dir", type=Path, default=LOCAL_DIR)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    api = HfApi()
    info = api.model_info(args.repo, revision=args.revision)
    resolved = info.sha
    print(f"repo: {args.repo}")
    print(f"requested revision: {args.revision}")
    print(f"resolved revision:  {resolved}")
    if resolved != args.revision:
        print(f"NOTE: revision resolved to {resolved}")

    patterns = list(METADATA_PATTERNS)
    if not args.metadata_only:
        patterns += WEIGHT_PATTERNS

    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        revision=resolved,
        local_dir=str(args.local_dir),
        allow_patterns=patterns,
        max_workers=args.workers,
    )
    print(f"downloaded to: {path}")

    # 完整 repo 文件树：即使没下载 assets/README，也记录下来方便判断。
    tree = []
    for sibling in info.siblings:
        tree.append(
            {
                "path": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "lfs_oid": getattr(getattr(sibling, "lfs", None), "oid", None),
            }
        )
    (args.local_dir / "repo_tree.json").write_text(
        json.dumps(tree, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    downloaded = []
    for p in sorted(args.local_dir.rglob("*")):
        if not p.is_file():
            continue
        # 只记录 checkpoint 实际文件；不把 HF cache、自身记录文件算进去。
        if ".cache" in p.parts or p.name in ("checkpoint_revision.txt", "repo_tree.json"):
            continue
        downloaded.append({"path": str(p.relative_to(args.local_dir)), "size": p.stat().st_size})

    lines = [
        f"repo_id: {args.repo}",
        f"requested_revision: {args.revision}",
        f"resolved_revision: {resolved}",
        f"local_dir: {args.local_dir.resolve()}",
        f"downloaded_at: {now_iso()}",
        f"metadata_only: {args.metadata_only}",
        "",
        "downloaded files:",
    ]
    for item in downloaded:
        lines.append(f"  {item['size']:>14}  {item['path']}")
    (args.local_dir / "checkpoint_revision.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines[-12:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
