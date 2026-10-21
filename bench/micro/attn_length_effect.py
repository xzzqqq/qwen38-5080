#!/usr/bin/env python3
"""M3 诊断：padded / unpadded attention 的 FP reduction 差异。

背景：M3 的 ``static-cap`` 在 decode 时扫满 capacity，``static-exact``/eager 只读
有效前缀 L+1。两者 mask 后的有效 key 相同，但 matmul 的 K 维长度不同，
float32 累加的分组顺序可能不同 -> BF16 输出可能差 1 ulp，并在 64 层 bf16
残差 + 401 个 W4A4 GEMM 中被放大。

这个 microbench 只隔离**单个 attention op**：
    Q [1,24,1,256] BF16, K/V [1,4,513,256] BF16
    unpadded:  K/V 原始长度
    padded:    K/V 补到 1024，mask 只允许前 513 个
    fp64 ref:  用同样 513 个 K/V 在 float64 下算

预期瓶颈：只测数值差异，不测速度。

用法：
    .venv/bin/python bench/micro/attn_length_effect.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from runner import report as reporting  # noqa: E402

H_Q = 24
H_KV = 4
HEAD_DIM = 256
SCALING = HEAD_DIM**-0.5


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    mask_value: float | None = None,
) -> torch.Tensor:
    """复刻 transformers 的 eager_attention_forward（不含 dropout）。"""
    key_states = repeat_kv(k, H_Q // H_KV)
    value_states = repeat_kv(v, H_Q // H_KV)
    attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * SCALING
    if mask is not None:
        attn_weights = attn_weights + mask
    elif mask_value is not None:
        raise ValueError("mask_value 需要配合 mask 形状")
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn_weights, value_states)
    return out.transpose(1, 2).contiguous()


def fp64_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q64 = q.double()
    k64 = repeat_kv(k.double(), H_Q // H_KV)
    v64 = repeat_kv(v.double(), H_Q // H_KV)
    weights = torch.matmul(q64, k64.transpose(2, 3)) * SCALING
    weights = torch.softmax(weights, dim=-1)
    return torch.matmul(weights, v64).transpose(1, 2).contiguous()


def bf16_ulp(x: torch.Tensor) -> float:
    """当前数值附近的 BF16 最小间隔（用最小正数的近似：2^-8 相对）。"""
    return float(x.abs().clamp_min(1e-6).max().item() * 2**-8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--length", type=int, default=513)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = args.out or reporting.make_run_dir(ROOT, "attn-length-effect")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in range(args.samples):
        gen = torch.Generator(device=device).manual_seed(args.seed + sample)
        q = torch.randn(1, H_Q, 1, HEAD_DIM, generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
        k = torch.randn(1, H_KV, args.length, HEAD_DIM, generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
        v = torch.randn(1, H_KV, args.length, HEAD_DIM, generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)

        unpadded = eager_attention(q, k, v)

        pad_len = args.capacity - args.length
        k_pad = torch.cat([k, torch.zeros(1, H_KV, pad_len, HEAD_DIM, device=device, dtype=k.dtype)], dim=-2)
        v_pad = torch.cat([v, torch.zeros(1, H_KV, pad_len, HEAD_DIM, device=device, dtype=v.dtype)], dim=-2)
        mask = torch.full((1, 1, 1, args.capacity), torch.finfo(torch.bfloat16).min, device=device, dtype=torch.bfloat16)
        mask[..., : args.length] = 0
        padded = eager_attention(q, k_pad, v_pad, mask=mask)

        ref = fp64_reference(q, k, v)
        d_pad_unpad = float((padded.float() - unpadded.float()).abs().max().item())
        d_unpad_ref = float((unpadded.double() - ref).abs().max().item())
        d_pad_ref = float((padded.double() - ref).abs().max().item())
        n_diff = int((padded != unpadded).sum().item())
        rows.append(
            {
                "sample": sample,
                "unpadded_vs_padded_max_abs": d_pad_unpad,
                "unpadded_vs_fp64_max_abs": d_unpad_ref,
                "padded_vs_fp64_max_abs": d_pad_ref,
                "changed_elements": n_diff,
                "elements": int(padded.numel()),
                "bf16_ulp_scale": bf16_ulp(unpadded),
                "unpadded_vs_padded_ulps": d_pad_unpad / bf16_ulp(unpadded),
            }
        )

    summary = {
        "length": args.length,
        "capacity": args.capacity,
        "samples": args.samples,
        "max_unpadded_vs_padded": max(r["unpadded_vs_padded_max_abs"] for r in rows),
        "max_unpadded_vs_fp64": max(r["unpadded_vs_fp64_max_abs"] for r in rows),
        "max_padded_vs_fp64": max(r["padded_vs_fp64_max_abs"] for r in rows),
        "changed_element_fraction": sum(r["changed_elements"] for r in rows)
        / sum(r["elements"] for r in rows),
        "median_ulps": sorted(r["unpadded_vs_padded_ulps"] for r in rows)[len(rows) // 2],
    }
    payload = {"provenance": reporting.provenance(ROOT), "summary": summary, "rows": rows}
    reporting.save_json(out_dir / "attn_length_effect.json", payload)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
