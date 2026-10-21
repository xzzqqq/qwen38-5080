#!/usr/bin/env python3
"""M3 P0 正确性验收：四模式、独立 padded reference、state/reset/边界/真实生成。

设计（对应 docs/plans/m3.md §5.1）：
- 每个场景在**独立子进程**里跑，避免两份 8K cache + graph pool 同时驻留；
- 对拍先看 bitwise：hidden / 有效 KV / GDN+conv state / logits / cache 长度；
- `static-cap` 的对照是独立的 padded DynamicCache reference（自己的位置计数和
  mask 尺寸），不复用 static adapter 的索引逻辑；
- 缓冲填充值隔离检查：fill=0 与 fill=3 必须给出相同输出；
- reset 覆盖 A→B→A、短→长→短，并与**独立新建 cache** 对比；
- 边界长度 1/127/128/129/300、两档 capacity 末端与越界拒绝；
- 三条真实 smoke 用 graph 模式跑 2048 上限：gate 是 EOS + 内容合理，
  与冻结 eager trace 的差异只作 diagnostic（free-running greedy 跨数值模式不保证 bitwise）。

用法：
    .venv/bin/python scripts/m3_check.py                 # 全部场景
    .venv/bin/python scripts/m3_check.py --only numeric-512,smoke-graph
    .venv/bin/python scripts/m3_check.py --worker numeric-512 --out /tmp/x
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoTokenizer, DynamicCache  # noqa: E402

from runner import report as reporting  # noqa: E402
from runner.loader.nvfp4_model import load_nvfp4_text_model  # noqa: E402
from runner.model.decode_runner import DecodeRunner  # noqa: E402
from runner.model.padded_reference import build_padded_reference  # noqa: E402
from runner.model.snapshots import (  # noqa: E402
    Snapshot,
    compare_snapshots,
    run_sequence,
    snapshot_state,
)
from runner.model.static_cache import (  # noqa: E402
    build_static_cache,
    cache_stats,
    preallocate_kv,
    reset_cache,
)

SCENARIOS = ("numeric-512", "numeric-8192", "graph-1024", "graph-8192", "smoke-graph")
BOUNDARY_LENGTHS = (1, 127, 128, 129, 300)


# --------------------------------------------------------------------- report
class CheckReport:
    def __init__(self, scenario: str, required: list[str] | None = None):
        self.scenario = scenario
        self.items: list[dict] = []
        # 必测项前缀：缺失即算 gate failure，避免“reference 没跑也算通过”。
        self.required = list(required or [])

    def _add(self, level: str, name: str, ok: bool, detail=None) -> bool:
        item = {"level": level, "name": name, "ok": bool(ok)}
        if detail is not None:
            item["detail"] = detail
        self.items.append(item)
        tag = "PASS" if ok else ("DIFF" if level == "diagnostic" else "FAIL")
        suffix = ""
        if detail is not None:
            try:
                suffix = " " + json.dumps(detail, ensure_ascii=False)[:600]
            except Exception:
                suffix = f" {detail}"
        print(f"[{tag}][{level}] {name}{suffix}", flush=True)
        return bool(ok)

    def gate(self, name: str, ok: bool, detail=None) -> bool:
        return self._add("gate", name, ok, detail)

    def diag(self, name: str, ok: bool, detail=None) -> bool:
        return self._add("diagnostic", name, ok, detail)

    def result(self) -> dict:
        if self.required:
            missing = [
                pattern
                for pattern in self.required
                if not any(
                    item["level"] == "gate" and item["name"].startswith(pattern)
                    for item in self.items
                )
            ]
            self.gate(
                "coverage: 必测项齐全",
                not missing,
                {"missing": missing, "required": len(self.required)},
            )
        gates = [i for i in self.items if i["level"] == "gate"]
        return {
            "scenario": self.scenario,
            "ok": bool(gates) and all(i["ok"] for i in gates),
            "gate_count": len(gates),
            "gate_failures": [i for i in gates if not i["ok"]],
            "items": self.items,
        }


# ------------------------------------------------------------------- helpers
def load_model(manifest: Path | None = None):
    model, cfg, info = load_nvfp4_text_model(
        ROOT / reporting.CHECKPOINT, manifest_path=manifest
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / reporting.CHECKPOINT), trust_remote_code=True
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer, info


def load_fixtures(fixtures_dir: Path) -> dict:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing fixtures manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    fixtures = {}
    for name, meta in manifest["files"].items():
        path = fixtures_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        actual = reporting._sha256_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        if meta.get("sha256") and meta["sha256"] != actual:
            raise RuntimeError(f"fixture hash mismatch: {name}")
        fixtures[name] = payload
    return {"manifest": manifest, "files": fixtures}


def fixed_prompt(fixtures: dict, depth: int) -> tuple[torch.Tensor, list[int]]:
    fx = fixtures["files"][f"fixed_{depth}.json"]
    return torch.tensor([fx["prompt_ids"]], dtype=torch.long), list(fx["trace"])


def check_snapshot(
    rep: CheckReport, name: str, snap: Snapshot, expected_len: int, *, gate: bool = True
) -> None:
    emit = rep.gate if gate else rep.diag
    emit(f"{name}: finite", snap.finite)
    emit(
        f"{name}: cache length",
        snap.length == expected_len,
        {"got": snap.length, "expected": expected_len, "forwards": snap.forwards},
    )
    emit(
        f"{name}: layer counts",
        len(snap.kv_keys) == 16 and len(snap.recurrent) == 48,
        {"kv": len(snap.kv_keys), "recurrent": len(snap.recurrent), "conv": len(snap.conv)},
    )
    emit(
        f"{name}: forward 数",
        snap.forwards == snap.decode_steps + 1,
        {"forwards": snap.forwards, "decode_steps": snap.decode_steps},
    )


def compare_and_report(
    rep: CheckReport,
    ref: Snapshot,
    got: Snapshot,
    prefix: str,
    *,
    gate: bool = True,
    skip_hidden: bool = False,
) -> bool:
    if not got.hidden or not got.hidden[0]:
        skip_hidden = True
    diffs = compare_snapshots(ref, got)
    ok_all = True
    skipped = []
    for diff in diffs:
        if skip_hidden and diff.name == "hidden":
            skipped.append(diff.name)
            continue
        emit = rep.gate if gate else rep.diag
        detail = diff.to_dict()
        if diff.name == "hidden" and ref.hidden and ref.hidden[0]:
            # 按最早发生的 forward 报告（flat 索引是 layer-major，容易误读）。
            detail["first_bad_forward"] = diff.first_bad_forward
            detail["first_bad_forward_layers"] = diff.first_bad_forward_layers[:8]
            per_layer = len(ref.hidden[0])
            detail["first_bad_pairs"] = [
                {"layer": i // per_layer, "forward": i % per_layer}
                for i in diff.first_bad[:4]
            ]
        if diff.name.startswith("kv.") and ref.kv_layer_indices:
            detail["first_bad_layers"] = [
                ref.kv_layer_indices[i] for i in diff.first_bad[:4] if i < len(ref.kv_layer_indices)
            ]
        if diff.name in ("conv", "recurrent") and ref.linear_layer_indices:
            detail["first_bad_layers"] = [
                ref.linear_layer_indices[i]
                for i in diff.first_bad[:4]
                if i < len(ref.linear_layer_indices)
            ]
        ok = diff.bitwise
        ok_all = ok_all and ok
        emit(f"{prefix}: {diff.name}", ok, detail)
    if skipped:
        rep.diag(
            f"{prefix}: 未记录项",
            True,
            {"skipped": skipped, "reason": "graph replay 内没有 Python hook"},
        )
    return ok_all


def build_static(model, capacity: int, mode: str):
    cache = build_static_cache(model.config, capacity, mode)
    kv_bytes = preallocate_kv(cache, model.config)
    return cache, kv_bytes


def cache_gate(rep: CheckReport, name: str, cache, capacity: int, kv_bytes: int) -> None:
    stats = cache_stats(cache, check_finite=False)
    ok = (
        stats.kv_layers == 16
        and stats.linear_layers == 48
        and stats.kv_dtypes == ["torch.bfloat16"]
        and stats.gdn_dtypes == ["torch.float32"]
        and stats.conv_dtypes == ["torch.bfloat16"]
        and stats.kv_bytes == kv_bytes
    )
    rep.gate(
        f"{name}: cache 布局",
        ok,
        {
            "kv_layers": stats.kv_layers,
            "linear_layers": stats.linear_layers,
            "kv_dtypes": stats.kv_dtypes,
            "gdn_dtypes": stats.gdn_dtypes,
            "conv_dtypes": stats.conv_dtypes,
            "kv_bytes": stats.kv_bytes,
            "capacity": capacity,
        },
    )


def logits_diff(ref: Snapshot, got: Snapshot) -> dict:
    """只比 logits 的紧凑摘要（diagnostic 用）。"""
    bad = [i for i, (a, b) in enumerate(zip(ref.logits, got.logits)) if not torch.equal(a, b)]
    mx = 0.0
    for a, b in zip(ref.logits, got.logits):
        if a.shape == b.shape and not torch.equal(a, b):
            mx = max(mx, float((a.float() - b.float()).abs().max().item()))
    return {"bitwise": not bad, "first_bad": bad[:8], "max_abs": mx, "forwards": len(ref.logits)}


# ------------------------------------------------------------------ numeric
def scenario_numeric(
    rep: CheckReport, depth: int, capacity: int, steps: int, fixtures: dict, *, boundaries: bool
) -> None:
    model, tokenizer, load_info = load_model()
    prompt, trace_list = fixed_prompt(fixtures, depth)
    trace = trace_list[:steps]
    expected_len = depth + steps
    rep.diag("load", True, load_info)

    # 1) eager reference
    dyn = DynamicCache(config=model.config)
    ref = run_sequence(model, dyn, prompt, trace, label="eager")
    del dyn
    torch.cuda.empty_cache()
    check_snapshot(rep, "eager", ref, expected_len)

    # 2) static-exact
    cache, kv_bytes = build_static(model, capacity, "exact")
    exact = run_sequence(model, cache, prompt, trace, label="static-exact")
    cache_gate(rep, "static-exact", cache, capacity, kv_bytes)
    compare_and_report(rep, ref, exact, "eager->static-exact")
    del cache
    torch.cuda.empty_cache()

    # 3) static-cap
    cache, kv_bytes = build_static(model, capacity, "cap")
    cap = run_sequence(model, cache, prompt, trace, label="static-cap")
    cache_gate(rep, "static-cap", cache, capacity, kv_bytes)
    compare_and_report(rep, ref, cap, "eager->static-cap", gate=False)
    compare_and_report(rep, exact, cap, "static-exact->static-cap", gate=False)
    del cache
    torch.cuda.empty_cache()

    # 4) padded dynamic reference vs static-cap
    padded = None
    try:
        pad_cache = build_padded_reference(model.config, capacity, fill=0.0)
        padded = run_sequence(model, pad_cache, prompt, trace, label="padded-dynamic")
        del pad_cache
        torch.cuda.empty_cache()
        compare_and_report(rep, padded, cap, "padded-ref->static-cap")
        compare_and_report(rep, ref, padded, "eager->padded-ref", gate=False)
    except Exception as exc:  # 必测 reference 没跑成 = gate failure，不能降级成 diagnostic
        rep.gate(
            "padded-ref->static-cap: 执行",
            False,
            {"error": f"{type(exc).__name__}: {exc}"[:600]},
        )

    # 5) 填充值隔离：fill=3 与 fill=0 必须一致
    if padded is not None:
        try:
            pad3_cache = build_padded_reference(model.config, capacity, fill=3.0)
            padded3 = run_sequence(model, pad3_cache, prompt, trace, label="padded-fill3")
            del pad3_cache
            torch.cuda.empty_cache()
            compare_and_report(rep, padded, padded3, "padded-fill3 vs fill0")
        except Exception as exc:
            rep.gate(
                "padded-fill3 vs fill0: 执行",
                False,
                {"error": f"{type(exc).__name__}: {exc}"[:600]},
            )

    # 6) reset / 新请求
    reset_checks(rep, model, capacity, fixtures, depth, steps)

    # 7) 边界长度 + capacity 越界拒绝
    if boundaries:
        boundary_checks(rep, model, tokenizer, capacity, fixtures)
    else:
        fx = fixtures["files"][f"fixed_{depth}.json"]
        capacity_end_check(rep, model, tokenizer, capacity, list(fx["prompt_ids"]))

    del ref, exact, cap
    torch.cuda.empty_cache()


def reset_checks(rep: CheckReport, model, capacity: int, fixtures: dict, depth: int, steps: int) -> None:
    """A→B→A、短→长→短、与独立新建 cache 对比。"""
    prompt_a, trace_a = fixed_prompt(fixtures, depth)
    trace_a = trace_a[:steps]
    fx = fixtures["files"][f"fixed_{depth}.json"]
    prompt_b = torch.tensor([fx["prompt_ids"][::-1]], dtype=torch.long)
    short = prompt_a[:, :1]
    long_prompt = prompt_a[:, : min(300, prompt_a.shape[1])]

    cache, _ = build_static(model, capacity, "exact")
    snap_a1 = run_sequence(model, cache, prompt_a, trace_a, label="A1")
    reset_cache(cache)
    snap_a2 = run_sequence(model, cache, prompt_a, trace_a, label="A2-reset")
    compare_and_report(rep, snap_a1, snap_a2, "reset: 原地 reset 重跑 A")

    reset_cache(cache)
    run_sequence(model, cache, prompt_b, trace_a, label="B")
    reset_cache(cache)
    snap_a3 = run_sequence(model, cache, prompt_a, trace_a, label="A3")
    compare_and_report(rep, snap_a1, snap_a3, "reset: A->B->A")

    reset_cache(cache)
    snap_short = run_sequence(model, cache, short, trace_a[:2], label="short")
    reset_cache(cache)
    run_sequence(model, cache, long_prompt, trace_a[:2], label="long")
    reset_cache(cache)
    snap_short2 = run_sequence(model, cache, short, trace_a[:2], label="short2")
    compare_and_report(rep, snap_short, snap_short2, "reset: 短->长->短")
    del cache
    torch.cuda.empty_cache()

    fresh, _ = build_static(model, capacity, "exact")
    fresh_a = run_sequence(model, fresh, prompt_a, trace_a, label="A-fresh")
    del fresh
    torch.cuda.empty_cache()
    compare_and_report(rep, snap_a1, fresh_a, "reset: 与独立新建 cache 一致")


def capacity_end_check(rep: CheckReport, model, tokenizer, capacity: int, base_ids: list[int]) -> None:
    """从**公开入口**验证 capacity 边界和越界拒绝。

    - prefill 超过 capacity 必须拒绝；
    - 填满 capacity 后 system_decode / model_timed_decode 必须在提交前拒绝；
    - graph 的 capture warmup 必须为 warmup 步数预留 headroom；
    - reset 后能继续写。
    """
    # 用 max_context 比 capacity 大，确保触发的是 capacity guard 而不是 max_context。
    runner = DecodeRunner(
        model,
        tokenizer,
        execution="static-cap",
        capacity=capacity,
        max_context=max(8192, capacity + 2),
    )
    runner.prepare()

    reps = (capacity + 1) // len(base_ids) + 1
    too_long = torch.tensor([(base_ids * reps)[: capacity + 1]], dtype=torch.long)
    raised = False
    try:
        runner.prefill(too_long)
    except ValueError as exc:
        raised = True
        rep.diag("capacity: prefill 拒绝信息", True, {"error": str(exc)})
    rep.gate("capacity: prefill 拒绝超长 prompt", raised)

    fill = torch.tensor([(base_ids * (capacity // len(base_ids) + 1))[:capacity]], dtype=torch.long)
    _, stats = runner.prefill(fill)
    rep.gate(
        "capacity: 填满 capacity 后长度正确",
        stats.cache_len == capacity,
        {"cache_len": stats.cache_len, "capacity": capacity},
    )

    for label, fn in (
        ("system_decode", lambda: runner.system_decode(0, 1)),
        ("model_timed_decode", lambda: runner.model_timed_decode([0])),
    ):
        raised = False
        try:
            fn()
        except ValueError as exc:
            raised = True
            rep.diag(f"capacity: {label} 拒绝信息", True, {"error": str(exc)})
        rep.gate(f"capacity: {label} 拒绝越界", raised)

    del runner
    torch.cuda.empty_cache()

    if capacity == 1024:
        # graph warmup 也需要 headroom：填到 capacity-1 后 warmup 3 步必然越界。
        graph_runner = DecodeRunner(
            model,
            tokenizer,
            execution="graph",
            capacity=capacity,
            max_context=max(8192, capacity + 2),
        )
        graph_runner.prepare()
        near_full = torch.tensor([(base_ids * (capacity // len(base_ids) + 1))[: capacity - 1]], dtype=torch.long)
        graph_runner.prefill(near_full)
        raised = False
        try:
            graph_runner.ensure_graph(near_full)
        except ValueError as exc:
            raised = True
            rep.diag("capacity: graph warmup 拒绝信息", True, {"error": str(exc)})
        rep.gate("capacity: graph warmup 拒绝越界", raised)
        del graph_runner
        torch.cuda.empty_cache()

    # reset 后可继续写（重新 prefill 短 prompt）
    cache_runner = DecodeRunner(
        model,
        tokenizer,
        execution="static-cap",
        capacity=capacity,
        max_context=max(8192, capacity + 2),
    )
    cache_runner.prepare()
    cache_runner.prefill(fill)
    cache_runner.reset()
    _, stats = cache_runner.prefill(torch.tensor([base_ids[:8]], dtype=torch.long))
    rep.gate(
        "capacity: reset 后可继续写",
        stats.cache_len == 8,
        {"cache_len": stats.cache_len},
    )
    del cache_runner
    torch.cuda.empty_cache()


def boundary_checks(rep: CheckReport, model, tokenizer, capacity: int, fixtures: dict) -> None:
    """prompt 长度 1/127/128/129/300。

    gate：eager vs static-exact（同 shape）、padded reference vs static-cap
    （同 decode shape）。eager vs static-cap 只报 attention 长度效应。
    """
    fx = fixtures["files"]["fixed_512.json"]
    trace = fx["trace"][:4]
    base_ids = list(fx["prompt_ids"])
    for length in BOUNDARY_LENGTHS:
        ids = base_ids[:length]
        prompt = torch.tensor([ids], dtype=torch.long)
        dyn = DynamicCache(config=model.config)
        eager = run_sequence(model, dyn, prompt, trace, label=f"eager-{length}")
        del dyn
        torch.cuda.empty_cache()

        cache, _ = build_static(model, capacity, "exact")
        exact = run_sequence(model, cache, prompt, trace, label=f"exact-{length}")
        compare_and_report(rep, eager, exact, f"boundary {length} eager->static-exact")
        del cache
        torch.cuda.empty_cache()

        cache, _ = build_static(model, capacity, "cap")
        cap = run_sequence(model, cache, prompt, trace, label=f"cap-{length}")
        del cache
        torch.cuda.empty_cache()

        pad_cache = build_padded_reference(model.config, capacity, fill=0.0)
        padded = run_sequence(model, pad_cache, prompt, trace, label=f"padded-{length}")
        del pad_cache
        torch.cuda.empty_cache()
        compare_and_report(rep, padded, cap, f"boundary {length} padded-ref->static-cap")

        rep.diag(
            f"boundary {length} eager->static-cap（attention 长度效应）",
            False,
            logits_diff(eager, cap),
        )
        del eager, exact, cap, padded
        torch.cuda.empty_cache()

    capacity_end_check(rep, model, tokenizer, capacity, base_ids)


# -------------------------------------------------------------------- graph
def scenario_graph(rep: CheckReport, depth: int, capacity: int, steps: int, fixtures: dict) -> None:
    model, tokenizer, load_info = load_model()
    prompt, trace_list = fixed_prompt(fixtures, depth)
    trace = trace_list[:steps]
    expected_len = depth + steps
    rep.diag("load", True, load_info)

    cache, kv_bytes = build_static(model, capacity, "cap")
    base = run_sequence(model, cache, prompt, trace, label="static-cap")
    cache_gate(rep, "static-cap", cache, capacity, kv_bytes)
    del cache
    torch.cuda.empty_cache()
    check_snapshot(rep, "static-cap", base, expected_len)

    runner = DecodeRunner(model, tokenizer, execution="graph", capacity=capacity)
    runner.prepare()
    info = runner.ensure_graph(prompt)
    rep.gate(
        "graph: capture 不推进位置",
        info["length_before_capture"] == info["length_after_capture"],
        {
            "length_before_capture": info["length_before_capture"],
            "length_after_capture": info["length_after_capture"],
        },
    )
    rep.gate(
        "graph: entry→capture 的增量等于 warmup 步数",
        info["length_before_capture"] - info["length_at_entry"] == runner.graph_warmup,
        {
            "length_at_entry": info["length_at_entry"],
            "length_before_capture": info["length_before_capture"],
            "warmup": runner.graph_warmup,
        },
    )
    rep.gate(
        "graph: warmup 推进了 static cache",
        info["warmup_length_delta"] == runner.graph_warmup,
        {"warmup_length_delta": info["warmup_length_delta"], "expected": runner.graph_warmup},
    )
    rep.gate(
        "graph: logits 形状",
        tuple(info["logits_shape"]) == (1, 1, int(model.config.vocab_size)),
        {"logits_shape": info["logits_shape"]},
    )
    ptr_capture = info["logits_data_ptr"]
    pool_alloc = info["mem_after_allocated"] - info["mem_before_allocated"]
    pool_reserved = info["mem_after_reserved"] - info["mem_before_reserved"]
    rep.diag(
        "graph: capture pool 增量",
        True,
        {
            "allocated_gb": pool_alloc / 1e9,
            "reserved_gb": pool_reserved / 1e9,
            "capture_s": info["capture_s"],
        },
    )

    logits0, pre = runner.prefill(prompt)
    rep.gate("graph: 正式初态长度", pre.cache_len == depth, {"cache_len": pre.cache_len, "depth": depth})
    rep.gate(
        "graph: prefill logits == static-cap",
        torch.equal(logits0.to("cpu", torch.bfloat16), base.logits[0]),
    )
    res = runner.trace_decode(trace)
    rep.gate(
        "graph: decode 后长度",
        res["cache_len"] == expected_len,
        {"cache_len": res["cache_len"], "expected": expected_len},
    )
    kv_layers = [layer for layer in runner.cache.layers if hasattr(layer, "cumulative_length")]
    written = [bool(torch.any(layer.keys[:, :, depth : depth + 1] != 0)) for layer in kv_layers]
    rep.gate(
        "graph: replay 写入 static KV 槽",
        all(written) and len(written) == 16,
        {"layers": len(written), "written": sum(written)},
    )
    bad = [i for i, (a, b) in enumerate(zip(base.logits[1:], res["logits"])) if not torch.equal(a, b)]
    rep.gate("graph: 逐步 logits bitwise", not bad, {"first_bad_steps": bad[:8], "steps": len(trace)})

    argmax_match = []
    margins = []
    for i, logits in enumerate(res["logits"][:-1]):
        row = logits.float().flatten()
        top2 = torch.topk(row, 2)
        argmax_match.append(int(top2.indices[0].item()) == trace[i + 1])
        margins.append(float(top2.values[0].item() - top2.values[1].item()))
    bad_steps = [i for i, ok in enumerate(argmax_match) if not ok]
    rep.diag(
        "graph: 逐步 argmax 与冻结 trace 一致",
        not bad_steps,
        {
            "bad_steps": bad_steps[:8],
            "checked": len(argmax_match),
            "bad_step_top2_margin": [round(margins[i], 4) for i in bad_steps[:8]],
            "min_margin": round(min(margins), 4) if margins else None,
        },
    )

    kv, val, conv, rec, length, finite = snapshot_state(runner.cache)
    graph_snap = Snapshot(
        label="graph",
        logits=[base.logits[0]] + res["logits"],
        hidden=[],
        kv_keys=kv,
        kv_values=val,
        conv=conv,
        recurrent=rec,
        length=length,
        prompt_tokens=depth,
        decode_steps=len(trace),
        forwards=len(res["logits"]),
        finite=finite,
    )
    compare_and_report(rep, base, graph_snap, "static-cap->graph")
    rep.gate(
        "graph: logits 地址稳定",
        runner.graph_step.logits_ptr == ptr_capture,
        {"ptr": runner.graph_step.logits_ptr, "capture_ptr": ptr_capture},
    )

    logits0b, _ = runner.prefill(prompt)
    res2 = runner.trace_decode(trace)
    rep.gate(
        "graph: reset 后 prefill logits 一致",
        torch.equal(logits0.to("cpu", torch.bfloat16), logits0b.to("cpu", torch.bfloat16)),
    )
    rep.gate(
        "graph: reset 后 replay 结果一致",
        all(torch.equal(a, b) for a, b in zip(res["logits"], res2["logits"])),
    )

    runner.prefill(prompt)
    rows = runner.host_gather([trace[0], trace[1]])
    runner.stage_row(rows[0:1], pinned=True)
    la = runner._step().detach().to("cpu", torch.bfloat16)
    runner.stage_row(rows[1:2], pinned=True)
    lb = runner._step().detach().to("cpu", torch.bfloat16)
    rep.gate(
        "graph: 输入 buffer 影响输出",
        not torch.equal(la, lb),
        {"max_abs": float((la.float() - lb.float()).abs().max().item())},
    )

    del runner
    torch.cuda.empty_cache()


# -------------------------------------------------------------------- smoke
SMOKE_EXPECT = {
    "math": {"all": ["72"], "any": [], "min_any": 0, "text": "答案 72 clips"},
    "code": {
        "all": ["is_palindrome", "def "],
        "any": ["assert", "print(", "case", "test"],
        "min_any": 1,
        "text": "函数 + 测试",
    },
    "chat": {"all": ["memory", "bandwidth"], "any": [], "min_any": 0, "text": "解释 memory-bandwidth"},
}


def scenario_smoke(rep: CheckReport, fixtures: dict, *, max_new_tokens: int) -> None:
    """真实 smoke：graph 模式跑 thinking + 2048 上限。

    gate 是「停止规则 + 输出合理」；与冻结 eager trace 的差异只作 diagnostic，
    并给出首处分叉的 top-2 margin。逐 step 的 bitwise 对照由 graph-1024/8192 负责。
    """
    model, tokenizer, load_info = load_model()
    rep.diag("load", True, load_info)

    runner = DecodeRunner(model, tokenizer, execution="graph", capacity=8192)
    runner.prepare()
    for short in ("math", "code", "chat"):
        fx = fixtures["files"][f"smoke_{short}.json"]
        prompt = torch.tensor([fx["prompt_ids"]], dtype=torch.long)
        budget = min(max_new_tokens, runner.max_context - prompt.shape[1])
        info = runner.ensure_graph(prompt)
        _logits, pre = runner.prefill(prompt)
        res = runner.system_decode(
            pre.first_token, budget, stop_on_eos=True, record_margins=True
        )
        tokens = [pre.first_token] + res["tokens"]
        expected = list(fx["trace"])[:budget]
        n = min(len(tokens), len(expected))
        first_bad = next((i for i in range(n) if tokens[i] != expected[i]), None)
        margins = res["margins"]

        rep.gate(
            f"smoke {short}: EOS 停止",
            res["finish_reason"] == "eos" and fx["finish_reason"] == "eos",
            {
                "finish_reason": res["finish_reason"],
                "expected_finish": fx["finish_reason"],
                "generated": len(tokens),
                "budget": budget,
            },
        )
        rep.gate(
            f"smoke {short}: 输出长度合理",
            2 < len(tokens) <= budget,
            {"generated": len(tokens), "budget": budget},
        )
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        spec = SMOKE_EXPECT[short]
        missing = [kw for kw in spec["all"] if kw not in text]
        any_hits = [kw for kw in spec["any"] if kw in text]
        rep.gate(
            f"smoke {short}: 内容合理（{spec['text']}）",
            not missing and len(any_hits) >= spec["min_any"],
            {"missing": missing, "any_hits": any_hits},
        )
        near_ties = [i for i, m in enumerate(margins) if m < 1e-2]
        rep.diag(
            f"smoke {short}: 与冻结 eager trace 的差异",
            first_bad is None,
            {
                "generated": len(tokens),
                "expected": len(expected),
                "first_bad_step": first_bad,
                "margin_at_first_bad": (
                    round(margins[first_bad], 4)
                    if first_bad is not None and first_bad < len(margins)
                    else None
                ),
                "prefix_match_tokens": n if first_bad is None else first_bad,
                "steps_with_margin_below_1e-2": len(near_ties),
            },
        )
        rep.diag(
            f"smoke {short}: prefill/显存/速度",
            True,
            {
                "prompt_tokens": pre.prompt_tokens,
                "prefill_s": round(pre.prefill_s, 3),
                "system_tok_s": round(res["system_tok_s"], 3),
                "peak_allocated_gb": round(res["peak_allocated_gb"], 3),
                "peak_reserved_gb": round(res["peak_reserved_gb"], 3),
                "free_gb": round(res["free_gb"], 3),
                "graph_capture": bool(info),
            },
        )
        rep.diag(f"smoke {short}: 文本", True, {"text": text[:4000]})
    del runner
    torch.cuda.empty_cache()


# ------------------------------------------------------------- required items
_NUMERIC_COMMON = [
    "eager: finite",
    "eager: cache length",
    "eager: forward 数",
    "eager->static-exact: logits",
    "eager->static-exact: hidden",
    "eager->static-exact: kv.keys",
    "eager->static-exact: kv.values",
    "eager->static-exact: conv",
    "eager->static-exact: recurrent",
    "padded-ref->static-cap: logits",
    "padded-ref->static-cap: hidden",
    "padded-ref->static-cap: kv.keys",
    "padded-ref->static-cap: recurrent",
    "padded-fill3 vs fill0: logits",
    "reset: 原地 reset 重跑 A: logits",
    "reset: A->B->A: logits",
    "reset: 短->长->短: logits",
    "reset: 与独立新建 cache 一致: logits",
    "capacity: prefill 拒绝超长 prompt",
    "capacity: 填满 capacity 后长度正确",
    "capacity: system_decode 拒绝越界",
    "capacity: model_timed_decode 拒绝越界",
    "capacity: reset 后可继续写",
]
_GRAPH_REQUIRED = [
    "static-cap: cache length",
    "static-cap->graph: logits",
    "static-cap->graph: kv.keys",
    "static-cap->graph: kv.values",
    "static-cap->graph: conv",
    "static-cap->graph: recurrent",
    "graph: 逐步 logits bitwise",
    "graph: replay 写入 static KV 槽",
    "graph: warmup 推进了 static cache",
    "graph: entry→capture 的增量等于 warmup 步数",
    "graph: reset 后 replay 结果一致",
    "graph: 输入 buffer 影响输出",
]
_SMOKE_REQUIRED = [
    f"smoke {name}: {suffix}"
    for name in ("math", "code", "chat")
    for suffix in ("EOS 停止", "输出长度合理", "内容合理")
]


def required_for(worker: str) -> list[str]:
    if worker == "numeric-512":
        boundary = [
            f"boundary {length} {prefix}: logits"
            for length in BOUNDARY_LENGTHS
            for prefix in ("eager->static-exact", "padded-ref->static-cap")
        ]
        return _NUMERIC_COMMON + boundary + ["capacity: graph warmup 拒绝越界"]
    if worker == "numeric-8192":
        return list(_NUMERIC_COMMON)
    if worker in ("graph-1024", "graph-8192"):
        return list(_GRAPH_REQUIRED)
    if worker == "smoke-graph":
        return list(_SMOKE_REQUIRED)
    raise ValueError(f"unknown worker {worker!r}")


# ------------------------------------------------------------------- worker
def run_worker(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixtures = load_fixtures(Path(args.fixtures))
    steps = args.steps
    rep = CheckReport(args.worker, required=required_for(args.worker))
    try:
        if args.worker == "numeric-512":
            scenario_numeric(rep, 512, 1024, steps, fixtures, boundaries=True)
        elif args.worker == "numeric-8192":
            scenario_numeric(rep, 7680, 8192, steps, fixtures, boundaries=False)
        elif args.worker == "graph-1024":
            scenario_graph(rep, 512, 1024, steps, fixtures)
        elif args.worker == "graph-8192":
            scenario_graph(rep, 7680, 8192, steps, fixtures)
        elif args.worker == "smoke-graph":
            scenario_smoke(rep, fixtures, max_new_tokens=args.smoke_max_new_tokens)
        else:
            raise ValueError(f"unknown worker {args.worker!r}")
        result = rep.result()
    except Exception:
        error = traceback.format_exc()
        result = rep.result()
        result["ok"] = False
        result["error"] = error
        print(error, flush=True)
    result["provenance"] = reporting.provenance(ROOT, {"worker": args.worker, "steps": steps})
    result["fixtures"] = {
        "dir": str(Path(args.fixtures)),
        "manifest_sha256": reporting._sha256_text(
            json.dumps(fixtures["manifest"], sort_keys=True, separators=(",", ":"))
        ),
    }
    reporting.save_json(out / "report.json", result)
    print(f"worker {args.worker}: ok={result['ok']}", flush=True)
    return 0 if result["ok"] else 1


# ------------------------------------------------------------------- driver
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "bench/workloads/m3-p0")
    parser.add_argument("--only", default="all")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--smoke-max-new-tokens", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--worker", default=None)
    parser.add_argument("--tag", default="m3-check")
    args = parser.parse_args()

    if args.worker:
        return run_worker(args)

    if args.only.strip() in ("", "all"):
        scenarios = list(SCENARIOS)
    else:
        scenarios = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in scenarios if s not in SCENARIOS]
        if unknown:
            raise SystemExit(f"unknown scenarios: {unknown}; choose from {SCENARIOS}")

    out_dir = Path(args.out) if args.out else reporting.make_run_dir(ROOT, "check", args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out_dir}", flush=True)
    reporting.provenance(
        ROOT,
        {
            "scenarios": scenarios,
            "fixtures": str(args.fixtures),
            "steps": args.steps,
            "smoke_max_new_tokens": args.smoke_max_new_tokens,
        },
        out_dir=out_dir,
    )

    results = []
    for scenario in scenarios:
        sub = out_dir / scenario
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            scenario,
            "--out",
            str(sub),
            "--fixtures",
            str(args.fixtures),
            "--steps",
            str(args.steps),
            "--smoke-max-new-tokens",
            str(args.smoke_max_new_tokens),
        ]
        print(f"\n=== {scenario}: {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=ROOT)
        path = sub / "report.json"
        if path.exists():
            res = json.loads(path.read_text())
        else:
            res = {"scenario": scenario, "ok": False, "items": [], "error": f"no report (rc={rc})"}
        res["returncode"] = rc
        res["ok"] = bool(res.get("ok")) and rc == 0
        results.append(res)

    summary = {
        "run_dir": str(out_dir),
        "scenarios": [{k: v for k, v in r.items() if k not in ("items", "provenance")} for r in results],
        "ok": all(r["ok"] for r in results),
    }
    reporting.save_json(out_dir / "check.json", {"summary": summary, "results": results})

    lines = ["M3 P0 correctness check", f"run_dir: {out_dir}", ""]
    for res in results:
        gates = [i for i in res.get("items", []) if i["level"] == "gate"]
        failed = [i for i in gates if not i["ok"]]
        lines.append(
            f"=== {res['scenario']}: {'OK' if res['ok'] else 'FAIL'} "
            f"(gates {len(gates) - len(failed)}/{len(gates)})"
        )
        if res.get("error"):
            lines.append(f"  error: {res['error'].splitlines()[-1]}")
        for item in failed:
            lines.append(
                f"  FAIL {item['name']} "
                f"{json.dumps(item.get('detail', {}), ensure_ascii=False)[:300]}"
            )
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
