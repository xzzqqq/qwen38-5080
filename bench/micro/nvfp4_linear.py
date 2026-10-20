#!/usr/bin/env python3
"""NVFP4Linear microbench：activation quant / mm_fp4 GEMM / 完整 forward。

设计：
- 用真实 checkpoint 权重，固定 backend，warmup 后用 CUDA event 取中位数；
- 分别测 activation quant、GEMM、完整 forward，避免把 Python 开销混进 GEMM；
- 只作 M2 诊断，不设 tok/s 门槛；
- 结果写 results/m2/<run-id>/nvfp4_microbench.{json,txt}。

用法（先 source scripts/m2_env.fish 或 .sh）：
    .venv/bin/python bench/micro/nvfp4_linear.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 与 scripts/m2_env.* 保持一致：让 FlashInfer JIT 找到 CUDA/curand 头文件。
CUDA_HOME = Path(os.environ.get("CUDA_HOME") or Path.home() / "micromamba/envs/cuda13")
PY_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
NVIDIA_INCLUDE = Path(sys.prefix) / "lib" / PY_TAG / "site-packages/nvidia/cu13/include"
NVIDIA_LIB = Path(sys.prefix) / "lib" / PY_TAG / "site-packages/nvidia/cu13/lib"
os.environ.setdefault("CUDA_HOME", str(CUDA_HOME))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT / ".tmp/flashinfer"))
os.environ["PATH"] = f"{CUDA_HOME / 'bin'}:{os.environ.get('PATH', '')}"
for var, paths in {
    "CPATH": [NVIDIA_INCLUDE, CUDA_HOME / "targets/x86_64-linux/include"],
    "LIBRARY_PATH": [
        NVIDIA_LIB,
        CUDA_HOME / "targets/x86_64-linux/lib",
        CUDA_HOME / "lib",
    ],
}.items():
    merged = []
    for item in [*map(str, paths), *os.environ.get(var, "").split(os.pathsep)]:
        if item and item not in merged:
            merged.append(item)
    os.environ[var] = os.pathsep.join(merged)

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

sys.path.insert(0, str(ROOT))

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize  # noqa: E402

from runner.ops.nvfp4.bytes import (  # noqa: E402
    activation_bytes,
    forward_bytes,
    output_bytes,
    weight_bytes,
)
from runner.ops.nvfp4.linear import NVFP4Linear  # noqa: E402

MODEL_DIR = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"
PRESETS = {
    "in_proj_qkv": ("model-00001-of-00002.safetensors", "model.language_model.layers.0.linear_attn.in_proj_qkv."),
    "gate_proj": ("model-00001-of-00002.safetensors", "model.language_model.layers.0.mlp.gate_proj."),
    "lm_head": ("model-00002-of-00002.safetensors", "lm_head."),
}


def load_preset(name: str) -> dict[str, torch.Tensor]:
    shard, prefix = PRESETS[name]
    with safe_open(MODEL_DIR / shard, framework="pt", device="cpu") as f:
        return {
            "weight": f.get_tensor(prefix + "weight"),
            "weight_scale": f.get_tensor(prefix + "weight_scale"),
            "weight_scale_2": f.get_tensor(prefix + "weight_scale_2"),
            "input_scale": f.get_tensor(prefix + "input_scale"),
        }


def timed(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default="in_proj_qkv,gate_proj,lm_head")
    parser.add_argument("--m", default="1,8,128")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--backend", default="cutlass")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run in full access")
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ROOT / "results/m2" / f"{run_id}-nvfp4-microbench"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for preset in [x.strip() for x in args.presets.split(",") if x.strip()]:
        tensors = load_preset(preset)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        module = NVFP4Linear(
            tensors["weight"],
            tensors["weight_scale"],
            tensors["weight_scale_2"],
            tensors["input_scale"],
            backend=args.backend,
        ).cuda()
        base_mib = torch.cuda.memory_allocated() / 1024**2
        k = module.in_features
        n = module.out_features
        for m in [int(x) for x in args.m.split(",") if x.strip()]:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5

            # 预计算一份 activation，用于单独测 GEMM。
            a_fp4, a_sf = nvfp4_quantize(
                x,
                (1.0 / module.input_scale).reshape(1),
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
                backend="cuda",
            )
            alpha = (module.input_scale * module.weight_scale_2).reshape(1).float()

            def full():
                return module(x)

            def quant():
                return nvfp4_quantize(
                    x,
                    (1.0 / module.input_scale).reshape(1),
                    sfLayout=SfLayout.layout_128x4,
                    do_shuffle=False,
                    backend="cuda",
                )

            def gemm():
                return mm_fp4(
                    a_fp4,
                    module.weight_packed.T,
                    a_sf,
                    module.weight_scale_swizzled.T,
                    alpha,
                    module.out_dtype,
                    None,
                    block_size=16,
                    use_8x4_sf_layout=False,
                    backend=args.backend,
                    use_nvfp4=True,
                )

            full_ms = timed(full, args.warmup, args.iters)
            quant_ms = timed(quant, args.warmup, args.iters)
            gemm_ms = timed(gemm, args.warmup, args.iters)
            torch.cuda.synchronize()
            peak_mib = torch.cuda.max_memory_allocated() / 1024**2
            act_bytes = activation_bytes(m, k)
            out_bytes = output_bytes(m, n)
            # upper bound：含 activation/output 和 allocator 开销，不是纯 workspace。
            workspace_upper_mib = max(
                0.0, peak_mib - base_mib - (act_bytes + out_bytes) / 1024**2
            )
            total_bytes = forward_bytes(m, k, n)
            row = {
                "preset": preset,
                "m": m,
                "n": n,
                "k": k,
                "backend": args.backend,
                "full_ms": full_ms,
                "quant_ms": quant_ms,
                "gemm_ms": gemm_ms,
                "other_ms": full_ms - quant_ms - gemm_ms,
                "logical_gbs": total_bytes / (full_ms * 1e-3) / 1e9,
                "weight_gb": weight_bytes(k, n) / 1e9,
                "total_bytes": total_bytes,
                "base_allocated_mib": base_mib,
                "peak_allocated_mib": peak_mib,
                "workspace_upper_mib": workspace_upper_mib,
            }
            results.append(row)
            print(
                f"{preset} M={m}: full={full_ms:.3f} ms quant={quant_ms:.3f} "
                f"gemm={gemm_ms:.3f} logical={row['logical_gbs']:.1f} GB/s "
                f"peak={peak_mib:.0f} MiB ws≤{workspace_upper_mib:.0f} MiB",
                flush=True,
            )
            del x, a_fp4, a_sf
        del module
        torch.cuda.empty_cache()

    (out_dir / "nvfp4_microbench.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"run_id: {run_id}",
        f"backend: {args.backend}",
        "note: M2 diagnostic only; not a tok/s gate",
        "",
        "preset, m, n, k, full_ms, quant_ms, gemm_ms, other_ms, logical_gbs, base_mib, peak_mib, workspace_upper_mib",
    ]
    for r in results:
        lines.append(
            f"{r['preset']},{r['m']},{r['n']},{r['k']},"
            f"{r['full_ms']:.4f},{r['quant_ms']:.4f},{r['gemm_ms']:.4f},"
            f"{r['other_ms']:.4f},{r['logical_gbs']:.2f},"
            f"{r['base_allocated_mib']:.0f},{r['peak_allocated_mib']:.0f},"
            f"{r['workspace_upper_mib']:.0f}"
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\noutput: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
