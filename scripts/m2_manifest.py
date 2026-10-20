#!/usr/bin/env python3
"""从 safetensors header 生成 M2 的 tensor manifest 和初版显存账。

设计：
- 只读每个 shard 的文件头（8 字节长度 + JSON header），不加载 tensor 数据；
- 用 index.json 校验 tensor 名和 shard 归属；
- 按文本/vision、NVFP4/高精度、GPU/host 分类，输出 JSON + CSV + 汇总；
- 用 config 计算 8K KV 和 GDN state 容量，和 manifest 的权重字节合并成显存账。

用法：
    .venv/bin/python scripts/m2_manifest.py
    .venv/bin/python scripts/m2_manifest.py --model-dir models/... --output-dir results/m2/run-001
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIB_TO_GB = 1.048576 / 1000  # 1 MiB = 1.048576 MB，再换成 GB


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], int, int]:
    """返回 (header, header_bytes, file_bytes)，不读取 tensor 数据。"""
    file_bytes = path.stat().st_size
    with path.open("rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"{path}: truncated safetensors prefix")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len <= 0 or 8 + header_len > file_bytes:
            raise RuntimeError(
                f"{path}: invalid header_len={header_len}, file_bytes={file_bytes}"
            )
        raw = f.read(header_len)
    header = json.loads(raw)
    return header, header_len, file_bytes


def product(shape: list[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def classify(name: str, dtype: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if "embed_tokens" in name:
        return "host_embedding"
    if name.startswith("lm_head."):
        if name.endswith(".weight"):
            return "lm_head_nvfp4_weight"
        if name.endswith(".weight_scale"):
            return "lm_head_weight_scale"
        if name.endswith(".weight_scale_2"):
            return "lm_head_global_scale"
        if name.endswith(".input_scale"):
            return "lm_head_input_scale"
        return "lm_head_other"
    if name.endswith(".weight") and dtype == "U8":
        return "nvfp4_weight"
    if name.endswith(".weight_scale"):
        return "nvfp4_weight_scale"
    if name.endswith(".weight_scale_2"):
        return "nvfp4_global_scale"
    if name.endswith(".input_scale"):
        return "nvfp4_input_scale"
    if name.endswith(".weight") and dtype in ("BF16", "F16", "F32"):
        return "norm" if "norm" in name else "high_precision_weight"
    if name.endswith(".bias"):
        return "bias"
    if name.endswith("A_log") or name.endswith("dt_bias"):
        return "state_param"
    return "other"


def placement_for(name: str) -> str:
    if name.startswith("model.visual."):
        return "skip"
    if "embed_tokens" in name:
        return "host"
    return "gpu"


def logical_shape_for(name: str, dtype: str, shape: list[int]) -> list[int]:
    # NVFP4 权重在文件里是两个 4-bit 值打包成一个 U8，逻辑最后一维要乘 2。
    if name.endswith(".weight") and dtype == "U8" and len(shape) == 2:
        return [int(shape[0]), int(shape[1]) * 2]
    return [int(x) for x in shape]


def is_excluded(name: str, excludes: list[str]) -> bool:
    return any(name == item or name.startswith(item + ".") for item in excludes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/Qwen3.8-27B-NVFP4-RTX5090")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--workspace-gb", type=float, default=0.5)
    parser.add_argument("--cuda-context-gb", type=float, default=269.1 * MIB_TO_GB)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = args.output_dir or Path("results/m2") / f"{run_id}-manifest"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = args.model_dir
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    index_metadata = index.get("metadata", {})
    config = json.loads((model_dir / "config.json").read_text())
    quant = json.loads((model_dir / "hf_quant_config.json").read_text())
    text_config = config["text_config"]
    quantization = quant.get("quantization", {})
    excludes = quantization.get("exclude_modules", [])

    shards = sorted(set(weight_map.values()))
    headers: dict[str, dict[str, Any]] = {}
    shard_meta: dict[str, dict[str, Any]] = {}
    for shard in shards:
        path = model_dir / shard
        header, header_len, file_bytes = read_safetensors_header(path)
        headers[shard] = header
        shard_meta[shard] = {
            "header_bytes": header_len,
            "file_bytes": file_bytes,
            "tensors": len(header) - (1 if "__metadata__" in header else 0),
        }

    # 校验 index 和 header 双向一致。
    files: dict[str, dict[str, Any]] = {}
    for shard, header in headers.items():
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if name in files:
                raise RuntimeError(f"duplicate tensor {name}")
            files[name] = {"shard": shard, **meta}
    missing_in_files = sorted(set(weight_map) - set(files))
    missing_in_index = sorted(set(files) - set(weight_map))
    wrong_shard = sorted(
        name for name, shard in weight_map.items() if files.get(name, {}).get("shard") != shard
    )
    if missing_in_files or missing_in_index or wrong_shard:
        raise RuntimeError(
            "index/header mismatch: "
            f"missing_in_files={missing_in_files[:5]}, "
            f"missing_in_index={missing_in_index[:5]}, "
            f"wrong_shard={wrong_shard[:5]}"
        )

    rows: list[dict[str, Any]] = []
    for name in sorted(files):
        meta = files[name]
        dtype = str(meta["dtype"])
        shape = [int(x) for x in meta["shape"]]
        start, end = (int(x) for x in meta["data_offsets"])
        data_bytes = end - start
        role = classify(name, dtype)
        placement = placement_for(name)
        logical_shape = logical_shape_for(name, dtype, shape)
        logical_params = product(logical_shape) if shape else 1
        is_weight = name.endswith(".weight")
        rows.append(
            {
                "name": name,
                "shard": meta["shard"],
                "dtype": dtype,
                "shape": shape,
                "logical_shape": logical_shape,
                "data_offsets": [start, end],
                "data_bytes": data_bytes,
                "role": role,
                "placement": placement,
                "logical_params": logical_params if is_weight else None,
                "excluded_by_config": is_excluded(name, excludes),
                "is_nvfp4_weight": role in ("nvfp4_weight", "lm_head_nvfp4_weight"),
                "is_scale": role.endswith("scale") or role.endswith("input_scale"),
            }
        )

    by_placement: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "bytes": 0})
    by_role: dict[str, dict[str, int]] = defaultdict(lambda: {"tensors": 0, "bytes": 0})
    for row in rows:
        by_placement[row["placement"]]["tensors"] += 1
        by_placement[row["placement"]]["bytes"] += row["data_bytes"]
        by_role[row["role"]]["tensors"] += 1
        by_role[row["role"]]["bytes"] += row["data_bytes"]

    gpu_bytes = by_placement["gpu"]["bytes"]
    host_bytes = by_placement["host"]["bytes"]
    skip_bytes = by_placement["skip"]["bytes"]
    total_bytes = sum(r["data_bytes"] for r in rows)

    nvfp4_rows = [r for r in rows if r["is_nvfp4_weight"]]
    nvfp4_logical_params = sum(r["logical_params"] or 0 for r in nvfp4_rows)
    nvfp4_storage_bytes = sum(r["data_bytes"] for r in nvfp4_rows)

    # 8K KV / GDN state 容量（字节）。
    layer_types = text_config["layer_types"]
    full_layers = sum(1 for x in layer_types if x == "full_attention")
    linear_layers = sum(1 for x in layer_types if x == "linear_attention")
    kv_bf16 = (
        full_layers
        * text_config["num_key_value_heads"]
        * text_config["head_dim"]
        * 2  # K 和 V
        * 2  # BF16
        * args.context_tokens
    )
    kv_fp8 = kv_bf16 // 2
    gdn_state = (
        linear_layers
        * text_config["linear_num_value_heads"]
        * text_config["linear_key_head_dim"]
        * text_config["linear_value_head_dim"]
        * 4  # FP32
    )
    conv_dim = (
        2 * text_config["linear_num_key_heads"] * text_config["linear_key_head_dim"]
        + text_config["linear_num_value_heads"] * text_config["linear_value_head_dim"]
    )
    conv_state = linear_layers * conv_dim * text_config["linear_conv_kernel_dim"] * 4

    usable_gb = 15841 * MIB_TO_GB
    base_gpu = gpu_bytes / 1e9
    context_gb = args.cuda_context_gb
    workspace_gb = args.workspace_gb
    total_bf16 = (
        base_gpu + kv_bf16 / 1e9 + (gdn_state + conv_state) / 1e9 + workspace_gb + context_gb
    )
    total_fp8 = (
        base_gpu + kv_fp8 / 1e9 + (gdn_state + conv_state) / 1e9 + workspace_gb + context_gb
    )

    report = {
        "run_id": run_id,
        "timestamp": now_iso(),
        "model_dir": str(model_dir.resolve()),
        "repo": "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090",
        "checkpoint_revision": "5b7a687fc8211a5d631c8ca6a593dd37eb26ce33",
        "quantization": {
            "quant_algo": quantization.get("quant_algo"),
            "kv_cache_quant_algo": quantization.get("kv_cache_quant_algo"),
            "group_size": quantization.get("group_size"),
            "exclude_modules_count": len(excludes),
        },
        "text_config": {
            "num_hidden_layers": text_config["num_hidden_layers"],
            "full_attention_layers": full_layers,
            "linear_attention_layers": linear_layers,
            "hidden_size": text_config["hidden_size"],
            "vocab_size": text_config["vocab_size"],
        },
        "shards": shard_meta,
        "index_metadata": index_metadata,
        "tensor_count": len(rows),
        "weight_param_sum": sum(
            r["logical_params"] or 0 for r in rows if r["name"].endswith(".weight")
        ),
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1e9,
        "by_placement": dict(by_placement),
        "by_role": dict(by_role),
        "gpu_text_gb": gpu_bytes / 1e9,
        "host_embedding_gb": host_bytes / 1e9,
        "vision_skip_gb": skip_bytes / 1e9,
        "nvfp4": {
            "weight_tensors": len(nvfp4_rows),
            "logical_params": nvfp4_logical_params,
            "storage_bytes": nvfp4_storage_bytes,
            "storage_gb": nvfp4_storage_bytes / 1e9,
            "effective_bits_per_param": (
                nvfp4_storage_bytes * 8 / nvfp4_logical_params
                if nvfp4_logical_params
                else None
            ),
        },
        "vram_account": {
            "usable_gb": usable_gb,
            "gpu_text_weights_gb": base_gpu,
            "kv_bf16_8k_gb": kv_bf16 / 1e9,
            "kv_fp8_8k_gb": kv_fp8 / 1e9,
            "gdn_state_gb": gdn_state / 1e9,
            "conv_state_gb": conv_state / 1e9,
            "workspace_gb": workspace_gb,
            "workspace_note": "占位值；M2 §4 实测",
            "cuda_context_gb": context_gb,
            "cuda_context_note": "preflight 中 torch 初始化后 free 减少 269.1 MiB",
            "estimated_total_bf16_kv_gb": total_bf16,
            "estimated_margin_bf16_kv_gb": usable_gb - total_bf16,
            "estimated_total_fp8_kv_gb": total_fp8,
            "estimated_margin_fp8_kv_gb": usable_gb - total_fp8,
            "assumptions": [
                f"context={args.context_tokens}",
                "KV BF16/FP8 按全量 16 个 full-attention 层计算",
                "GDN recurrent state + conv state 按 FP32 计算",
                "workspace 和 CUDA context 为占位/预检值，需 M2 §4 实测",
            ],
        },
    }

    (out_dir / "tensor_manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "vram_account.json").write_text(
        json.dumps(report["vram_account"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "tensor_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "shard",
                "dtype",
                "shape",
                "logical_shape",
                "data_bytes",
                "role",
                "placement",
                "logical_params",
                "excluded_by_config",
                "is_nvfp4_weight",
                "is_scale",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "shape": json.dumps(row["shape"]), "logical_shape": json.dumps(row["logical_shape"])})

    lines = [
        f"run_id: {run_id}",
        f"model_dir: {model_dir}",
        f"revision: {report['checkpoint_revision']}",
        f"shards: {shard_meta}",
        f"tensors: {len(rows)}",
        f"total: {total_bytes/1e9:.3f} GB",
        f"GPU text: {gpu_bytes/1e9:.3f} GB",
        f"host embedding: {host_bytes/1e9:.3f} GB",
        f"vision skipped: {skip_bytes/1e9:.3f} GB",
        f"NVFP4 weights: {len(nvfp4_rows)} tensors, "
        f"{nvfp4_logical_params/1e9:.3f} B params, "
        f"{nvfp4_storage_bytes/1e9:.3f} GB storage, "
        f"{report['nvfp4']['effective_bits_per_param']:.2f} bits/param",
        "",
        "by placement:",
    ]
    for key, value in sorted(by_placement.items()):
        lines.append(f"  {key:5s} {value['tensors']:5d} {value['bytes']/1e9:9.3f} GB")
    lines += ["", "by role:"]
    for key, value in sorted(by_role.items(), key=lambda kv: -kv[1]["bytes"]):
        lines.append(f"  {key:28s} {value['tensors']:5d} {value['bytes']/1e9:9.3f} GB")
    lines += [
        "",
        "VRAM account (GB, decimal):",
        f"  usable: {usable_gb:.3f}",
        f"  text weights: {base_gpu:.3f}",
        f"  KV BF16 8K: {kv_bf16/1e9:.3f}",
        f"  KV FP8 8K: {kv_fp8/1e9:.3f}",
        f"  GDN state: {gdn_state/1e9:.3f}",
        f"  conv state: {conv_state/1e9:.3f}",
        f"  workspace: {workspace_gb:.3f} (placeholder)",
        f"  CUDA context: {context_gb:.3f}",
        f"  total BF16 KV: {total_bf16:.3f} -> margin {usable_gb-total_bf16:+.3f}",
        f"  total FP8 KV:  {total_fp8:.3f} -> margin {usable_gb-total_fp8:+.3f}",
        "",
        f"output: {out_dir}",
    ]
    summary = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
