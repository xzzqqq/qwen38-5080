#!/usr/bin/env python3
"""M2 runner 数值验收。

固定三条真实 prompt + 16 步 greedy token，逐步 teacher forcing，检查：

1. 每个 NVFP4Linear 在 prefill 和 16 步 decode 的实际 activation 上，与同 A/W
   量化的 dequant reference 对比；
2. 每个 GDN chunk/recurrent 调用与 FP32 逐 token recurrence 对比（输出和 final state）；
3. 逐层 hidden states、KV cache、GDN/conv state 的 dtype/finite/length；
4. 每一步 decode logits 与 lm_head dequant reference 对比。

不做跨 chunk/recurrent kernel 的 bitwise hidden 对拍；hidden 的数值正确性由
逐 op 的 NVFP4/GDN reference 覆盖，hidden/cache 本身做结构检查。

reference 不常驻整份 BF16 模型：NVFP4 逐 module dequant，GDN 逐调用跑 FP32
recurrence，lm_head 按 vocab block dequant。

用法：
    .venv/bin/python scripts/m2_numeric.py
    .venv/bin/python scripts/m2_numeric.py --max-new-tokens 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as qmodel  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from runner.env import configure_flashinfer_env  # noqa: E402

configure_flashinfer_env()

from flashinfer import SfLayout, nvfp4_quantize  # noqa: E402

from runner.model.eager_runner import EagerTextRunner  # noqa: E402
from runner.ops.nvfp4.linear import NVFP4Linear  # noqa: E402
from runner.ops.nvfp4.reference import (  # noqa: E402
    dequant_nvfp4,
    error_metrics,
    quantize_activation_reference,
    unswizzle_scale_128x4,
)

NVFP4_TOL = {"nrmse": 5.0e-3, "cosine": 0.99999, "max_rel": 1.0e-2}
ACT_TOL = {"nrmse": 3.0e-2, "cosine": 0.9995, "max_rel": 2.0e-1}
GDN_TOL = {"nrmse": 5.0e-3, "cosine": 0.9999, "max_rel": 1.0e-2}


def _activation_flash(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """用 FlashInfer 的 A 量化结果做 dequant；只用于验证 W layout/GEMM。"""
    k = x.shape[-1]
    flat = x.reshape(-1, k).contiguous()
    a_fp4, a_sf = nvfp4_quantize(
        flat,
        (1.0 / input_scale).reshape(1),
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    if a_sf.dtype == torch.uint8:
        a_sf = a_sf.view(torch.float8_e4m3fn)
    scale = unswizzle_scale_128x4(a_sf, flat.shape[0], k // 16)
    return dequant_nvfp4(a_fp4, scale, input_scale)


def _metrics(out: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    return error_metrics(out, ref)


def _worse(best: dict[str, Any] | None, cur: dict[str, Any], name: str) -> dict[str, Any]:
    """独立维护各指标极值，避免更大的 nrmse 覆盖更差的 max_abs/max_rel/cosine。"""
    if best is None:
        return {
            "name": name,
            "nrmse_name": name,
            "max_abs_name": name,
            "max_rel_name": name,
            "cosine_name": name,
            "nrmse": cur["nrmse"],
            "max_abs": cur["max_abs"],
            "max_rel": cur["max_rel"],
            "cosine": cur["cosine"],
            "mean_abs": cur["mean_abs"],
            "finite": cur["finite"],
        }
    if cur["nrmse"] > best["nrmse"]:
        best["nrmse"] = cur["nrmse"]
        best["nrmse_name"] = name
        best["name"] = name
    if cur["max_abs"] > best["max_abs"]:
        best["max_abs"] = cur["max_abs"]
        best["max_abs_name"] = name
    if cur["max_rel"] > best["max_rel"]:
        best["max_rel"] = cur["max_rel"]
        best["max_rel_name"] = name
    if cur["cosine"] < best["cosine"]:
        best["cosine"] = cur["cosine"]
        best["cosine_name"] = name
    best["mean_abs"] = max(best["mean_abs"], cur["mean_abs"])
    best["finite"] = best["finite"] and cur["finite"]
    return best


def _record(store: dict[str, Any], name: str, metrics: dict[str, float], tol: dict[str, float]) -> None:
    store["count"] += 1
    store["worst"] = _worse(store.get("worst"), metrics, name)
    ok = (
        metrics["finite"]
        and metrics["nrmse"] <= tol["nrmse"]
        and metrics["cosine"] >= tol["cosine"]
        and metrics["max_rel"] <= tol["max_rel"]
    )
    if not ok:
        store["failures"].append({"name": name, **metrics})


def _cache_stats(cache: DynamicCache) -> dict[str, Any]:
    kv_dtypes: set[str] = set()
    recurrent_dtypes: set[str] = set()
    conv_dtypes: set[str] = set()
    finite = True
    kv_layers = 0
    recurrent_layers = 0
    conv_layers = 0
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is not None and values is not None:
            kv_layers += 1
            kv_dtypes.add(str(keys.dtype))
            finite = finite and bool(torch.isfinite(keys).all()) and bool(torch.isfinite(values).all())
        states = getattr(layer, "recurrent_states", None)
        if states:
            recurrent_layers += 1
            for state in states.values():
                recurrent_dtypes.add(str(state.dtype))
                finite = finite and bool(torch.isfinite(state).all())
        conv = getattr(layer, "conv_states", None)
        if conv:
            conv_layers += 1
            for state in conv.values():
                conv_dtypes.add(str(state.dtype))
                finite = finite and bool(torch.isfinite(state).all())
    return {
        "cache_len": int(cache.get_seq_length()),
        "finite": finite,
        "kv_layers": kv_layers,
        "recurrent_layers": recurrent_layers,
        "conv_layers": conv_layers,
        "kv_dtypes": sorted(kv_dtypes),
        "recurrent_dtypes": sorted(recurrent_dtypes),
        "conv_dtypes": sorted(conv_dtypes),
    }


def _hidden_stats(outputs) -> dict[str, Any]:
    hidden = outputs.hidden_states
    finite = all(bool(torch.isfinite(h).all()) for h in hidden)
    dtypes = sorted({str(h.dtype) for h in hidden})
    return {"layers": len(hidden), "finite": finite, "dtypes": dtypes}


def _lm_head_reference(module: NVFP4Linear, hidden: torch.Tensor, block: int = 2048) -> torch.Tensor:
    """按 vocab block dequant lm_head，避免整份 FP32 lm_head 驻留 GPU。"""
    k = module.in_features
    n = module.out_features
    scale = unswizzle_scale_128x4(module.weight_scale_swizzled, n, k // 16)
    a_deq = _activation_flash(hidden, module.input_scale)
    a = a_deq.reshape(-1, k).float()
    pieces = []
    for start in range(0, n, block):
        end = min(start + block, n)
        w = dequant_nvfp4(
            module.weight_packed[start:end],
            scale[start:end],
            module.weight_scale_2,
        )
        pieces.append(a @ w.float().T)
    ref = torch.cat(pieces, dim=1)
    if module.bias is not None:
        ref = ref + module.bias.float()
    return ref


def _gdn_recurrent_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """本地 FP32 逐 token gated delta rule；用于验证 decode 的 state 更新。"""
    dtype = query.dtype
    batch, seq, _, k_dim = key.shape
    v_heads, v_dim = value.shape[-2:]
    q, k, v, b, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    q = q / (k_dim**0.5)
    if initial_state is None:
        state = torch.zeros(
            (batch, v_heads, k_dim, v_dim), dtype=torch.float32, device=v.device
        )
    else:
        state = initial_state.to(torch.float32)
    out = torch.zeros_like(v)
    for i in range(seq):
        state = state * decay[:, :, i].exp()[..., None, None]
        kv_mem = (state * k[:, :, i].unsqueeze(-1)).sum(dim=-2)
        delta = (v[:, :, i] - kv_mem) * b[:, :, i].unsqueeze(-1)
        state = state + k[:, :, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q[:, :, i].unsqueeze(-1)).sum(dim=-2)
    out = out.transpose(1, 2).contiguous().to(dtype)
    return out, state if output_final_state else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        default="results/payload_math.json,results/payload_code.json,results/payload_chat.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = ROOT / "results/m2" / f"{run_id}-numeric"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner, load_info = EagerTextRunner.from_checkpoint(
        ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090",
        manifest_path=args.manifest,
    )
    model = runner.model
    tokenizer = runner.tokenizer

    # --- hooks: NVFP4 dequant reference (prefill only) ---
    nvfp4_store: dict[str, Any] = {"count": 0, "failures": [], "worst": None}
    act_store: dict[str, Any] = {"count": 0, "failures": [], "worst": None}
    nvfp4_capture = False
    hooks = []

    def make_hook(module: NVFP4Linear):
        def hook(_module, inputs, output):
            if not nvfp4_capture:
                return
            x = inputs[0]
            k = module.in_features
            n = module.out_features
            scale = unswizzle_scale_128x4(
                module.weight_scale_swizzled, n, k // 16
            )
            a_deq = _activation_flash(x, module.input_scale)
            _, _, a_ref = quantize_activation_reference(x, module.input_scale)
            _record(
                act_store,
                f"{module.out_features}x{module.in_features}:act",
                _metrics(a_deq, a_ref),
                ACT_TOL,
            )
            a = a_deq.reshape(-1, k).float()
            ref = torch.empty(a.shape[0], n, device=a.device, dtype=torch.float32)
            # 逐 output row block 反量化，避免 up_proj 的 FP32 权重一次占 300+ MB。
            row_block = 1024
            for start in range(0, n, row_block):
                end = min(start + row_block, n)
                w = dequant_nvfp4(
                    module.weight_packed[start:end],
                    scale[start:end],
                    module.weight_scale_2,
                )
                ref[:, start:end] = a @ w.float().T
            if module.bias is not None:
                ref = ref + module.bias.float()
            _record(
                nvfp4_store,
                f"{module.out_features}x{module.in_features}",
                _metrics(output.reshape(-1, n), ref),
                NVFP4_TOL,
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, NVFP4Linear) and module is not model.lm_head:
            hooks.append(module.register_forward_hook(make_hook(module)))

    # --- GDN FP32 recurrence reference ---
    gdn_store: dict[str, Any] = {"count": 0, "failures": [], "worst": None}
    gdn_state_store: dict[str, Any] = {"count": 0, "failures": [], "worst": None}
    gdn_capture = False
    orig_chunk = qmodel.torch_chunk_gated_delta_rule
    orig_recurrent = qmodel.torch_recurrent_gated_delta_rule
    ref_chunk = orig_recurrent

    def wrap_gdn(orig, label: str, ref_fn):
        def wrapped(query, key, value, g, beta, **kwargs):
            out, state = orig(query, key, value, g=g, beta=beta, **kwargs)
            if gdn_capture:
                ref_out, ref_state = ref_fn(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=kwargs.get("initial_state"),
                    output_final_state=kwargs.get("output_final_state", False),
                    use_qk_l2norm_in_kernel=kwargs.get(
                        "use_qk_l2norm_in_kernel", False
                    ),
                )
                seq = query.shape[1]
                _record(gdn_store, f"{label}:seq{seq}", _metrics(out, ref_out), GDN_TOL)
                if state is not None and ref_state is not None:
                    _record(
                        gdn_state_store,
                        f"{label}:seq{seq}:state",
                        _metrics(state, ref_state),
                        GDN_TOL,
                    )
            return out, state

        return wrapped

    qmodel.torch_chunk_gated_delta_rule = wrap_gdn(orig_chunk, "chunk", ref_chunk)
    qmodel.torch_recurrent_gated_delta_rule = wrap_gdn(
        orig_recurrent, "recurrent", _gdn_recurrent_reference
    )

    records = []
    try:
        for prompt_path in [p.strip() for p in args.prompts.split(",") if p.strip()]:
            payload = json.loads((ROOT / prompt_path).read_text())
            prompt_ids = runner.encode_messages(payload["messages"])
            # 固定 greedy 16 步 token
            gen = runner.generate_ids(
                prompt_ids, max_new_tokens=args.max_new_tokens, greedy=True
            )
            fixed = list(gen.token_ids)
            prompt_len = int(prompt_ids.shape[1])

            cache = DynamicCache(config=model.config)
            hidden_checks = []
            step_logits = []
            t0 = time.perf_counter()
            nvfp4_store = {"count": 0, "failures": [], "worst": None}
            act_store = {"count": 0, "failures": [], "worst": None}
            gdn_store = {"count": 0, "failures": [], "worst": None}
            gdn_state_store = {"count": 0, "failures": [], "worst": None}
            lm_store = {"count": 0, "failures": [], "worst": None}
            gdn_capture = True
            nvfp4_capture = True

            # prefill chunks；NVFP4 reference 在 prefill 和 decode 都开
            out = None
            for start in range(0, prompt_len, runner.prefill_chunk):
                out = model(
                    input_ids=prompt_ids[:, start : start + runner.prefill_chunk],
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    output_hidden_states=True,
                )
                hidden_checks.append(_hidden_stats(out))
            prefill_cache = _cache_stats(cache)

            # teacher forcing 固定 token
            last_hidden = out.hidden_states[-1][:, -1:, :]
            lm_s = 0.0
            lm_metrics = None
            for step, token in enumerate(fixed):
                out = model(
                    input_ids=torch.tensor([[token]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    output_hidden_states=True,
                )
                hidden_checks.append(_hidden_stats(out))
                last_hidden = out.hidden_states[-1][:, -1:, :]
                step_logits.append(out.logits[:, -1, :].float())

                # 每一步用 lm_head dequant reference 检查最终 logits；hidden/cache
                # 只做 finite/dtype/length 结构检查，不做跨 chunk/recurrent kernel 的
                # bitwise hidden 对拍。
                t_lm = time.perf_counter()
                lm_ref = _lm_head_reference(model.lm_head, last_hidden)
                lm_metrics = _metrics(out.logits[:, -1, :], lm_ref)
                lm_s += time.perf_counter() - t_lm
                _record(lm_store, f"step{step}", lm_metrics, NVFP4_TOL)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            final_cache = _cache_stats(cache)
            prompt_chunks = (prompt_len + runner.prefill_chunk - 1) // runner.prefill_chunk
            expected_nvfp4 = 400 * (prompt_chunks + len(fixed))
            expected_gdn = 48 * (prompt_chunks + len(fixed))

            def cache_ok(stats, expected_len):
                return (
                    stats["finite"]
                    and stats["cache_len"] == expected_len
                    and stats["kv_layers"] == 16
                    and stats["recurrent_layers"] == 48
                    and stats["conv_layers"] == 48
                    and stats["kv_dtypes"] == ["torch.bfloat16"]
                    and stats["recurrent_dtypes"] == ["torch.float32"]
                    and stats["conv_dtypes"] == ["torch.bfloat16"]
                )

            hidden_ok = (
                len(hidden_checks) == prompt_chunks + len(fixed)
                and all(
                    h["finite"]
                    and h["layers"] == 65
                    and h["dtypes"] == ["torch.bfloat16"]
                    for h in hidden_checks
                )
            )
            counts_ok = (
                nvfp4_store["count"] == expected_nvfp4
                and act_store["count"] == expected_nvfp4
                and gdn_store["count"] == expected_gdn
                and gdn_state_store["count"] == expected_gdn
                and lm_store["count"] == len(fixed)
            )
            caches_ok = cache_ok(prefill_cache, prompt_len) and cache_ok(
                final_cache, prompt_len + len(fixed)
            )
            lm_ok = (
                lm_store["count"] == len(fixed)
                and len(lm_store["failures"]) == 0
            )
            record_ok = (
                counts_ok
                and hidden_ok
                and caches_ok
                and lm_ok
                and nvfp4_store["count"] > 0
                and gdn_store["count"] > 0
                and len(nvfp4_store["failures"]) == 0
                and len(gdn_store["failures"]) == 0
                and len(gdn_state_store["failures"]) == 0
                and len(lm_store["failures"]) == 0
            )

            record = {
                "prompt_file": prompt_path,
                "prompt_tokens": prompt_len,
                "fixed_tokens": fixed,
                "cache_len_expected": prompt_len + len(fixed),
                "expected": {
                    "prompt_chunks": prompt_chunks,
                    "nvfp4": expected_nvfp4,
                    "gdn": expected_gdn,
                },
                "checks": {
                    "counts_ok": counts_ok,
                    "hidden_ok": hidden_ok,
                    "caches_ok": caches_ok,
                    "lm_ok": lm_ok,
                    "ok": record_ok,
                },
                "prefill_cache": prefill_cache,
                "final_cache": final_cache,
                "hidden_checks": hidden_checks,
                "nvfp4": {
                    "count": nvfp4_store["count"],
                    "failures": len(nvfp4_store["failures"]),
                    "worst": nvfp4_store["worst"],
                },
                "activation_quant": {
                    "count": act_store["count"],
                    "failures": len(act_store["failures"]),
                    "diagnostic": True,
                    "worst": act_store["worst"],
                },
                "gdn": {
                    "count": gdn_store["count"],
                    "failures": len(gdn_store["failures"]),
                    "worst": gdn_store["worst"],
                },
                "gdn_state": {
                    "count": gdn_state_store["count"],
                    "failures": len(gdn_state_store["failures"]),
                    "worst": gdn_state_store["worst"],
                },
                "lm_head": {
                    "count": lm_store["count"],
                    "failures": len(lm_store["failures"]),
                    "worst": lm_store["worst"],
                },
                "lm_head_s": lm_s,
                "elapsed_s": elapsed,
                "steps": len(step_logits),
                "ok": record_ok,
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "prompt": prompt_path,
                        "prompt_tokens": prompt_len,
                        "fixed": len(fixed),
                        "nvfp4": record["nvfp4"],
                        "activation_quant": record["activation_quant"],
                        "gdn": record["gdn"],
                        "gdn_state": record["gdn_state"],
                        "lm_head": record["lm_head"],
                        "cache": final_cache,
                        "checks": record["checks"],
                        "ok": record_ok,
                        "elapsed_s": elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        for h in hooks:
            h.remove()
        qmodel.torch_chunk_gated_delta_rule = orig_chunk
        qmodel.torch_recurrent_gated_delta_rule = orig_recurrent

    ok = bool(records) and all(r["ok"] for r in records)
    result = {"ok": ok, "load": load_info, "tolerances": {"nvfp4": NVFP4_TOL, "activation_quant": ACT_TOL, "gdn": GDN_TOL}, "results": records}
    (out_dir / "numeric.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [f"run_id: {run_id}", f"ok: {ok}", f"load: {load_info}", ""]
    for r in records:
        lines += [
            f"=== {r['prompt_file']}",
            f"prompt_tokens={r['prompt_tokens']} fixed={len(r['fixed_tokens'])}",
            f"nvfp4={r['nvfp4']}",
            f"activation_quant={r['activation_quant']}",
            f"gdn={r['gdn']}",
            f"gdn_state={r['gdn_state']}",
            f"lm_head={r['lm_head']}",
            f"prefill_cache={r['prefill_cache']}",
            f"final_cache={r['final_cache']}",
            f"elapsed_s={r['elapsed_s']:.3f} lm_head_s={r['lm_head_s']:.3f}",
            "",
        ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ok={ok}")
    print(f"output: {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
