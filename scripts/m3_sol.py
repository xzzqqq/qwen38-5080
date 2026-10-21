#!/usr/bin/env python3
"""M3 SOL：按实际 tensor 字节重算每步理想流量、SOL 和逻辑带宽，生成 docs/sol.md。

口径（docs/plans/m3.md §6）：
- 权重从**运行时唯一 storage** 统计（按 storage 去重，排除 vision 和 host embedding）；
- KV / GDN / conv 用实存 shape/dtype，两档 capacity 分别给；
- 对第 t 个计时 decode step，L_t 是加入当前 token 后的有效 KV 长度；
- ``B_ideal(t) = B_weights + 2*N_full*H_kv*D*kv_bytes*L_t + 2*N_full*H_kv*D*kv_bytes
  + 2*B_gdn + B_conv_rw``；
- ``B_mean = mean(B_ideal)``，``SOL = 960e9 / B_mean``，``%SOL = measured / SOL``；
- BF16 KV 是 M3 实测；FP8 只给理论值和“未实测”标注。

用法：
    .venv/bin/python scripts/m3_sol.py
    .venv/bin/python scripts/m3_sol.py --bench results/m3/20261021T...-bench/bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from runner import report as reporting  # noqa: E402
from runner.loader.nvfp4_model import load_nvfp4_text_model  # noqa: E402
from runner.ops.nvfp4.linear import NVFP4Linear  # noqa: E402

N_FULL_ATTN = 16
H_KV = 4
D_HEAD = 256
KV_BYTES_BF16 = 2
KV_BYTES_FP8 = 1
BANDWIDTH = reporting.SOL_BANDWIDTH_BYTES


def weight_accounting(model) -> dict:
    """按 storage 去重统计 GPU 上权重的实际字节。"""
    seen: dict[int, tuple[str, int]] = {}
    per_module: dict[str, int] = {}

    def add(tensor: torch.Tensor | None, category: str) -> None:
        if tensor is None or tensor.device.type != "cuda":
            return
        storage = tensor.untyped_storage()
        ptr = int(storage.data_ptr())
        if ptr in seen:
            return
        nbytes = int(storage.nbytes())
        seen[ptr] = (category, nbytes)
        per_module[category] = per_module.get(category, 0) + nbytes

    nvfp4_count = 0
    for name, module in model.named_modules():
        if isinstance(module, NVFP4Linear):
            nvfp4_count += 1
            add(module.weight_packed, "nvfp4_weight_packed")
            add(module.weight_scale_swizzled, "nvfp4_weight_scale_swizzled")
            add(module.weight_scale_2, "nvfp4_scalars")
            add(module.input_scale, "nvfp4_scalars")
            add(module.bias, "nvfp4_bias")
    for name, param in model.named_parameters():
        add(param, "bf16_params")
    for name, buf in model.named_buffers():
        add(buf, "bf16_buffers")

    host_embedding = model.model.embed_tokens.weight
    return {
        "categories": per_module,
        "total_cuda_bytes": sum(per_module.values()),
        "nvfp4_linears": nvfp4_count,
        "host_embedding_bytes": host_embedding.numel() * host_embedding.element_size()
        if host_embedding.device.type == "cpu"
        else 0,
    }


def cache_accounting(config, capacity: int) -> dict:
    """两档 capacity 的 KV / GDN / conv 字节（按实存 shape/dtype 解析计算）。"""
    kv_bytes = (
        2
        * N_FULL_ATTN
        * 1
        * config.num_key_value_heads
        * capacity
        * config.head_dim
        * KV_BYTES_BF16
    )
    conv_dim = (
        config.linear_key_head_dim * config.linear_num_key_heads * 2
        + config.linear_value_head_dim * config.linear_num_value_heads
    )
    linear_layers = sum(1 for t in config.layer_types if t == "linear_attention")
    gdn_bytes = (
        linear_layers
        * config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * 4
    )
    conv_bytes = linear_layers * conv_dim * config.linear_conv_kernel_dim * 2
    return {
        "capacity": capacity,
        "kv_bytes": kv_bytes,
        "gdn_state_bytes": gdn_bytes,
        "conv_state_bytes": conv_bytes,
        "total_bytes": kv_bytes + gdn_bytes + conv_bytes,
    }


def ideal_bytes(
    weights: int,
    gdn: int,
    conv: int,
    depth: int,
    steps: int,
    kv_bytes_per_element: int,
) -> dict:
    """按 depth + 1..depth+steps 的 L_t 求每步理想字节。"""
    per_step = []
    for t in range(1, steps + 1):
        length = depth + t
        read_kv = 2 * N_FULL_ATTN * H_KV * D_HEAD * kv_bytes_per_element * length
        write_kv = 2 * N_FULL_ATTN * H_KV * D_HEAD * kv_bytes_per_element
        total = weights + read_kv + write_kv + 2 * gdn + 2 * conv
        per_step.append(
            {
                "t": t,
                "L_t": length,
                "read_kv": read_kv,
                "write_kv": write_kv,
                "total": total,
            }
        )
    total_bytes = [r["total"] for r in per_step]
    return {
        "steps": steps,
        "depth": depth,
        "first": per_step[0],
        "last": per_step[-1],
        "mean_bytes": sum(total_bytes) / len(total_bytes),
        "min_bytes": min(total_bytes),
        "max_bytes": max(total_bytes),
        "kv_bytes_total": sum(r["read_kv"] + r["write_kv"] for r in per_step) / len(per_step),
    }


def gqa_expansion(config, scan_length: int, kv_bytes_per_element: int = KV_BYTES_BF16) -> dict:
    """§6 的 eager attention 额外流量模型（repeat_kv 物化后再 matmul）。"""
    h_q = config.num_attention_heads
    compact = H_KV * scan_length * D_HEAD * kv_bytes_per_element
    expanded = h_q * scan_length * D_HEAD * kv_bytes_per_element
    total = 2 * N_FULL_ATTN * (compact + 2 * expanded)
    return {
        "scan_length": scan_length,
        "compact_kv": compact,
        "compact_full_model": 2 * N_FULL_ATTN * compact,
        "expanded_kv": expanded,
        "per_step_total": total,
        "extra_vs_compact": total - 2 * N_FULL_ATTN * compact,
    }


def find_bench() -> Path | None:
    candidates = sorted((ROOT / "results/m3").glob("*-bench/bench.json"))
    return candidates[-1] if candidates else None


def join_bench(bench_path: Path | None, weights: int, gdn: int, conv: int, capacities: dict) -> list[dict]:
    """把 bench 的实测 tok/s 和按有效 L_t 的 B_ideal 对齐。

    plan §6 规定所有执行模式的 ``B_ideal`` 都用有效 ``L_t``；static-cap / graph
    扫满 capacity 以及 GQA 展开属于**额外流量**，单列，不改变理想分母。
    """
    if bench_path is None or not bench_path.exists():
        return []
    data = json.loads(bench_path.read_text())
    rows = []
    for res in data.get("results", []):
        if not res.get("ok"):
            continue
        spec = res.get("spec", {})
        depth = int(spec["depth"])
        steps = int(res.get("steps", 0))
        execution = spec.get("execution")
        capacity = spec.get("capacity")
        ideal = ideal_bytes(weights, gdn, conv, depth, steps, KV_BYTES_BF16)
        mean_bytes = ideal["mean_bytes"]
        mean_length = depth + (steps + 1) / 2.0
        write_kv = 2 * N_FULL_ATTN * H_KV * D_HEAD * KV_BYTES_BF16
        if execution in ("static-cap", "graph"):
            scan_bytes = 2 * N_FULL_ATTN * H_KV * D_HEAD * KV_BYTES_BF16 * capacity + write_kv
            extra_capacity = scan_bytes - ideal["kv_bytes_total"]
            scan = f"scan={capacity}（额外 {extra_capacity / 1e9:.3f} GB/step）"
        else:
            scan_bytes = ideal["kv_bytes_total"]
            extra_capacity = 0.0
            scan = f"scan=L_t~{depth + steps // 2}"
        sol = BANDWIDTH / mean_bytes
        summary = res.get("summary", {})
        model_tok = summary.get("model_tok_s", {}).get("median")
        system_tok = summary.get("system_tok_s", {}).get("median")
        rows.append(
            {
                "scenario": res["scenario"],
                "execution": execution,
                "depth": depth,
                "capacity": capacity,
                "scan": scan,
                "mean_length": mean_length,
                "steps": steps,
                "mean_bytes": mean_bytes,
                "kv_bytes": ideal["kv_bytes_total"],
                "extra_capacity_bytes": extra_capacity,
                "scan_bytes": scan_bytes,
                "sol_tok_s": sol,
                "model_tok_s": model_tok,
                "system_tok_s": system_tok,
                "model_pct_sol": model_tok / sol if model_tok else None,
                "system_pct_sol": system_tok / sol if system_tok else None,
                "model_logical_gb_s": mean_bytes * model_tok / 1e9 if model_tok else None,
                "system_logical_gb_s": mean_bytes * system_tok / 1e9 if system_tok else None,
                "peak_allocated_gb": summary.get("peak_allocated_gb"),
                "free_gb_min": summary.get("free_gb_min"),
            }
        )
    return rows


def fmt_gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.3f}"


def fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "-"


def write_markdown(path: Path, ctx: dict) -> None:
    w = ctx["weights"]
    cats = w["categories"]
    rows = ctx["bench_rows"]
    lines = [
        "# SOL 模型（M3 实际字节）",
        "",
        f"- 生成时间：{ctx['created_at']}",
        f"- commit：`{ctx['git_commit']}`（dirty={ctx['dirty']}）",
        f"- checkpoint：`{reporting.CHECKPOINT}` revision `{reporting.CHECKPOINT_REVISION}`",
        f"- 环境：torch {ctx['env']['torch']}、transformers {ctx['env']['transformers']}、"
        f"FlashInfer {ctx['env']['flashinfer']}、driver {ctx['env']['driver']}",
        f"- allocator：`{ctx['alloc_config']}`（见 §5 口径说明）",
        f"- 生成命令：`.venv/bin/python scripts/m3_sol.py`"
        + (f" --bench {ctx['bench_path']}" if ctx["bench_path"] else ""),
        "",
        "## 1. 权重字节（运行时唯一 storage）",
        "",
        "| 类别 | 字节 | GB | 说明 |",
        "|---|---|---|---|",
    ]
    notes = {
        "nvfp4_weight_packed": "NVFP4 E2M1 packed（U8，[N, K/2]）",
        "nvfp4_weight_scale_swizzled": "每 16 元素一个 UE4M3 scale，128x4 swizzled",
        "nvfp4_scalars": "weight_scale_2 / input_scale（FP32 标量）",
        "nvfp4_bias": "checkpoint 里的量化 Linear bias（多数为 None）",
        "bf16_params": "norm / conv1d / in_proj_a,b / A_log / dt_bias 等 BF16 参数",
        "bf16_buffers": "非持久 buffer（rotary inv_freq 等）",
    }
    for key, value in sorted(cats.items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{key}` | {value:,} | {fmt_gb(value)} | {notes.get(key, '')} |")
    lines += [
        f"| **合计（GPU）** | **{w['total_cuda_bytes']:,}** | **{fmt_gb(w['total_cuda_bytes'])}** | "
        f"{w['nvfp4_linears']} 个 NVFP4Linear |",
        f"| host embedding（不计入 GPU） | {w['host_embedding_bytes']:,} | "
        f"{fmt_gb(w['host_embedding_bytes'])} | BF16，常驻 host |",
        "",
        "vision tower 未加载；embedding 在 host，均不计入每步 GPU 流量。",
        "",
        "## 2. KV / GDN / conv state 字节",
        "",
        "| capacity | KV（16 层 BF16 K+V） | GDN recurrent（48 层 FP32） | conv（48 层 BF16） | 合计 |",
        "|---|---|---|---|---|",
    ]
    for cap, acct in ctx["cache"].items():
        lines.append(
            f"| {cap} | {acct['kv_bytes']:,} ({fmt_gb(acct['kv_bytes'])} GB) | "
            f"{acct['gdn_state_bytes']:,} | {acct['conv_state_bytes']:,} | "
            f"{fmt_gb(acct['total_bytes'])} GB |"
        )
    lines += [
        "",
        "## 3. 每步理想字节与 SOL",
        "",
        "```text",
        "B_ideal(t) = B_weights",
        "           + 2 * 16 * 4 * 256 * kv_bytes * L_t   # K/V 读",
        "           + 2 * 16 * 4 * 256 * kv_bytes         # 新 K/V 写",
        "           + 2 * B_gdn_state                     # recurrent 读写",
        "           + 2 * B_conv_state                    # conv 读写",
        "B_mean = mean(B_ideal(t)),  t = 1..511,  L_t = depth + t",
        "SOL_tok_s = 960e9 / B_mean",
        "```",
        "",
    ]
    ideal = ctx["ideal"]
    for key, label in (("bf16_512", "depth 512 / BF16 KV"), ("bf16_7680", "depth 7680 / BF16 KV")):
        stats = ideal[key]
        lines.append(
            f"- **{label}**：B_mean {fmt_gb(stats['mean_bytes'])} GB/step "
            f"（首步 {fmt_gb(stats['first']['total'])}、末步 {fmt_gb(stats['last']['total'])}），"
            f"其中 KV 平均 {fmt_gb(stats['kv_bytes_total'])} GB；"
            f"**SOL {BANDWIDTH / stats['mean_bytes']:.1f} tok/s**"
        )
    lines += [
        "",
        "> eager / static-exact 的 decode attention 只读有效前缀 `L_t`；static-cap / graph "
        "在 decode 扫满 capacity，按 capacity 代入（见下表 scan 列）。两者都按各自实际扫描长度"
        "计算 SOL。",
        "",
        "## 4. 实测 vs SOL（BF16 KV，M3 P0）",
        "",
    ]
    if rows:
        lines += [
            "| scenario | execution | depth | capacity | decode 实际扫描 | B_mean GB（L_t） | "
            "capacity 额外流量 GB/step | SOL tok/s | 模型 tok/s | 模型 %SOL | 模型逻辑 GB/s | "
            "系统 tok/s | 系统 %SOL | 系统逻辑 GB/s | peak GB | min free GB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['scenario']} | {row['execution']} | {row['depth']} | "
                f"{row['capacity'] or 'dynamic'} | {row['scan']} | {fmt_gb(row['mean_bytes'])} | "
                f"{row['extra_capacity_bytes'] / 1e9:.3f} | "
                f"{row['sol_tok_s']:.1f} | "
                f"{row['model_tok_s']:.2f} | {fmt_pct(row['model_pct_sol'])} | "
                f"{row['model_logical_gb_s']:.1f} | "
                f"{row['system_tok_s']:.2f} | {fmt_pct(row['system_pct_sol'])} | "
                f"{row['system_logical_gb_s']:.1f} | "
                f"{(row['peak_allocated_gb'] or 0):.3f} | {(row['free_gb_min'] or 0):.3f} |"
            )
        lines += [
            "",
            "模型 tok/s 与系统 tok/s 的定义见 `docs/plans/m3.md` §2.3；模型 %SOL 和系统 %SOL "
            "使用同一 B_mean，因此系统百分比包含 sampling / host 传输 / 同步的损失。",
        ]
    else:
        lines.append("（没有可用的 bench 结果；先跑 `make m3-bench`。）")
    lines += [
        "",
        "## 5. 口径说明",
        "",
        "1. **逻辑带宽不是 DRAM 实测**：这里用“每步字节模型 × tok/s”估算；要拿物理字节需要 "
        "`ncu` 的 `dram__bytes`。Nsight Systems 的时间线只解释 kernel 与空隙，不提供这份字节。",
        "2. **B_ideal 只含必需访问**：activation quant、临时 tensor、GQA 展开（`repeat_kv`）、"
        "static-cap 的 masked 槽访问、ShapeStableLinear 的 padding 都单列，不计入 B_ideal。"
        "额外流量不等于有效模型字节变大。",
        "3. **BF16 vs FP8**：M3 只实测 BF16 KV。FP8 的理论值见 §6，标注“未实测”，"
        "不为了填表提前引入 FP8 attention kernel。",
        "4. **allocator**：M3 全流程用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，"
        "它不改变数值，但会影响 peak/reserved 的观测值。",
        "5. **单位**：本文 GB = 10^9 字节。",
        "",
        "## 6. FP8 KV 理论值（未实测）",
        "",
        "| 场景 | B_mean GB（BF16→FP8） | SOL tok/s（BF16→FP8） |",
        "|---|---|---|",
    ]
    for key, label in (("fp8_512", "depth 512"), ("fp8_7680", "depth 7680")):
        stats = ideal[key]
        lines.append(
            f"| {label} | {fmt_gb(ideal['bf16_' + key.split('_')[1]]['mean_bytes'])} → "
            f"{fmt_gb(stats['mean_bytes'])} | "
            f"{BANDWIDTH / ideal['bf16_' + key.split('_')[1]]['mean_bytes']:.1f} → "
            f"{BANDWIDTH / stats['mean_bytes']:.1f} |"
        )
    lines += [
        "",
        "## 7. eager attention 的额外流量（`repeat_kv` 展开）",
        "",
        "```text",
        "O(C) = H_kv * C * D * kv_bytes        # 单层、单个 K/V 的紧凑字节",
        "E(C) = H_q  * C * D * kv_bytes        # H_q=24，展开后的字节",
        "B_expand_model(C) = 2 * 16 * (O(C) + 2 * E(C))",
        "```",
        "",
        "| 扫描长度 C | 全模型紧凑 KV 读（16 层 K+V） | 展开路径总量估算 | 相比紧凑 KV 的额外量 |",
        "|---|---|---|---|",
    ]
    for scan_length in ctx["expansion"].get("scan_lengths", []):
        exp = gqa_expansion(ctx["config"], scan_length)
        lines.append(
            f"| {scan_length} | {fmt_gb(exp['compact_full_model'])} GB | "
            f"{fmt_gb(exp['per_step_total'])} GB/step | "
            f"{fmt_gb(exp['extra_vs_compact'])} GB/step |"
        )
    lines += [
        "",
        "`O(C)`/`E(C)` 是**单层单个 K 或 V** 的字节；表里的“紧凑 KV 读”是乘过 "
        "`2 * 16`（K+V、16 个 full attention 层）的全模型值。展开路径总量按 "
        "`2 * 16 * (O(C) + 2 * E(C))` 算。",
        "",
        "这些是数组访问量估算，不是实测 DRAM 流量或毫秒数；cache 命中、kernel 实现和调度都会"
        "影响实际成本。static-cap 的 decode 用 capacity 代入，eager/static-exact 用 `L_t` 代入。",
        "",
        "## 8. 复现",
        "",
        "```bash",
        "make m3-fixtures",
        "make m3-check",
        "make m3-bench",
        "make m3-memory",
        "make m3-sol",
        "```",
        "",
        "结果目录：`results/m3/<run-id>-{check,bench,memory}/`；本文件由 `scripts/m3_sol.py` 生成。",
        "",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=ROOT / "docs/sol.md")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or reporting.make_run_dir(ROOT, "sol")
    out_dir.mkdir(parents=True, exist_ok=True)
    model, config, load_info = load_nvfp4_text_model(
        ROOT / reporting.CHECKPOINT, manifest_path=args.manifest
    )
    weights = weight_accounting(model)
    del model
    torch.cuda.empty_cache()

    capacities = {}
    for cap in (1024, 8192):
        capacities[cap] = cache_accounting(config, cap)

    ideal = {}
    for kv_bytes, tag in ((KV_BYTES_BF16, "bf16"), (KV_BYTES_FP8, "fp8")):
        for depth in (512, 7680):
            ideal[f"{tag}_{depth}"] = ideal_bytes(
                weights["total_cuda_bytes"],
                capacities[8192]["gdn_state_bytes"],
                capacities[8192]["conv_state_bytes"],
                depth,
                511,
                kv_bytes,
            )

    bench_path = args.bench or find_bench()
    rows = join_bench(bench_path, weights["total_cuda_bytes"],
                      capacities[8192]["gdn_state_bytes"],
                      capacities[8192]["conv_state_bytes"], capacities)

    prov = reporting.provenance(ROOT, out_dir=out_dir)
    ctx = {
        "created_at": prov["created_at"],
        "git_commit": prov["git"]["commit"],
        "dirty": prov["git"]["dirty"],
        "env": prov["environment"],
        "alloc_config": __import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "bench_path": str(bench_path) if bench_path else None,
        "weights": weights,
        "cache": {str(k): v for k, v in capacities.items()},
        "ideal": ideal,
        "expansion": {"scan_lengths": [1024, 8192]},
        "config": config,
        "bench_rows": rows,
        "load": load_info,
    }
    payload = {
        "weights": weights,
        "cache": ctx["cache"],
        "ideal": ideal,
        "bench_rows": rows,
        "provenance": prov,
        "expansion": {str(s): gqa_expansion(config, s) for s in (1024, 8192)},
    }
    reporting.save_json(out_dir / "sol.json", payload)
    write_markdown(Path(args.out_md), ctx)
    print(json.dumps({k: v for k, v in payload.items() if k != "provenance"}, indent=2, ensure_ascii=False)[:3000])
    print(f"wrote {args.out_md}")
    print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
