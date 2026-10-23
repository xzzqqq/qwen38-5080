#!/usr/bin/env python3
"""Phase 2 任务 I：GDN conv decode 片段融合 kernel 的 op gate + microbench。

边界：只替换 ``causal_conv1d_update``（``sequence_length == 1`` 的 decode 分支）。
把 reference 的 4 个 kernel（``cat([conv_state, x])``、``conv_state.copy_``、
depthwise ``F.conv1d``、``silu``）合成 1 个 Triton kernel；conv_state 原地滚动、
dtype/字节/data_ptr 不变。recurrent step、``in_proj_*``、out_proj、gated RMSNorm
都不在本次范围内（recurrent 已由第 1 轮融合）。

reference 是**未修改的** ``transformers...modeling_qwen3_5.causal_conv1d_update``
（cat + copy_ + F.conv1d + ACT2FN），在本机没有 ``causal_conv1d`` 时就是 M3 profile
里看到的 ``CatArrayBatchedCopy`` / ``direct_copy`` / ``conv_depthwise2d`` /
``silu_kernel`` 四个 kernel。

gate（每个 case 都要有 ``coverage:`` 项）：
- 真实非零 history conv_state（随机历史，min absmax > 1e-3），不是全零；
- 连续 2 步原地递推：第 2 步的 reference 从**候选第 1 步写完后**的 state 出发（两边
  各自维护自己的 state），所以「不更新 state / 更新错列」会被抓住；
- 48 层真实 shape：conv_dim=10240、conv kernel=4、state_len=4（见下）、bf16；
- dtype/字节/data_ptr：conv_state 在调用前后不变（static cache / graph replay 依赖）；
- finite；
- 数值容差用 decisions 里 M2 冻结的 GDN 容差：
  ``nrmse <= 1e-3``、``cosine >= 0.99999``、``max_rel <= 5e-3``。

**state_len 是 4，不是 3**。``update_conv_state`` 在 prefill 时把 conv_state 初始化
成 ``[1, C, conv_kernel_size]``，decode 的 ``cat([state(4), x(1)])`` 因此长度 5、
conv 输出 2 个、reference 取最后一个：``out = w0*s1 + w1*s2 + w2*s3 + w3*x``，
新 state = ``[s1, s2, s3, x]``。任务描述里写的 ``[1,10240,3]`` 与源码不符，这里以
源码/真实 shape 为准。

独立证据 + 负对照：``conv-negative-controls`` 用三种错实现（全零 state ≈「只看当前
token」、state 错位一格、weight 反向）跑**同一个正确 kernel**，必须显著超出容差；
否则说明 gate 没有分辨力。

接线门：``patch-conv-wiring-real-shapes`` 用真实 config 的 **meta** GDN 模块 + 真实
CUDA 张量走 ``causal_conv1d_update`` 的模块级入口，断言
``conv.counters.fused == 96、reference == 0、fallbacks == {}``。m3_check 的 bitwise
gate 只比较执行方式，patch 没生效时照样全绿（第 1 轮踩过），所以「融合是否真的被
执行」必须成为产物级 gate。

microbench：同一计时边界（CUDA event 包住一次 48 层完整 conv step，warmup 后取 15 次
中位数）比较 reference 与候选；生产路径是 CUDA Graph，所以另外用 graph replay 计时。

用法：
    .venv/bin/python scripts/phase2_gdn_conv_check.py --out {output}/op
    .venv/bin/python scripts/phase2_gdn_conv_check.py --only microbench-conv-baseline-vs-candidate --out .tmp/op
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import transformers.models.qwen3_5.modeling_qwen3_5 as gdn_modeling  # noqa: E402

import runner.model.gdn_patch as gdn_patch  # noqa: E402
from runner.experiment.op_artifact import OpReport  # noqa: E402
from runner.ops.gdn import (  # noqa: E402
    CONV_BLOCK_CONFIGS,
    CONV_DEFAULT_BLOCK,
    CONV_DEFAULT_NUM_WARPS,
    gdn_conv_decode_step,
)
from runner.ops.nvfp4.reference import error_metrics  # noqa: E402

#: 未修改的 reference（import 时抓取；后面 case 才会 patch 模块属性）。
ORIG_CONV = gdn_modeling.causal_conv1d_update

SCENARIOS = (
    "conv-two-steps-48-layers",
    "conv-negative-controls",
    "patch-conv-wiring-real-shapes",
    "microbench-conv-baseline-vs-candidate",
)
CORRECT_CASE = SCENARIOS[0]
NEGATIVE_CASE = SCENARIOS[1]
PATCH_CASE = SCENARIOS[2]
PERF_CASE = SCENARIOS[3]
HIDDEN_SIZE = 5120

# 冻结的模型 shape：config.linear_conv_kernel_dim=4, linear_num_value_heads=48,
# linear_num_key_heads=16, head_dim=128 -> conv_dim = 2*16*128 + 48*128 = 10240。
NUM_LAYERS = 48
NUM_V_HEADS = 48
NUM_K_HEADS = 16
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CONV_DIM = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM
CONV_KERNEL = 4
STATE_LEN = CONV_KERNEL  # 见模块 docstring：decode 的 conv_state 长度 == kernel
STATE_BYTES = CONV_DIM * STATE_LEN * 2
X_BYTES = CONV_DIM * 2
WEIGHT_BYTES = CONV_DIM * CONV_KERNEL * 2

# M2 冻结的 GDN 容差（docs/decisions.md 2026-10-21 M2 关闭与验收口径）。
GDN_TOL = {"nrmse": 1e-3, "cosine": 0.99999, "max_rel": 5e-3}

SEED = 20261023
STEPS = 2
WARMUP = 5
REPS = 15
GRAPH_WARMUP = 3
# 负对照至少要超出容差这么多倍，才说明 gate 有分辨力。
NEGATIVE_MARGIN = 10.0


# --------------------------------------------------------------------------- 数据


def _conv_inputs(device: torch.device, gen: torch.Generator) -> list[dict]:
    """48 层 conv decode 输入：x = in_proj_qkv 输出（[1,C,1]），state/weight 真实形状。

    取值口径贴近真实路径：``mixed_qkv`` 是 NVFP4 GEMM 的输出、量级 ~0.1；conv_state
    里存的就是历史 ``mixed_qkv``；depthwise conv 的 weight 初始化是 U(-1/sqrt(4), 1/sqrt(4))
    （kaiming uniform a=sqrt(5)，fan_in=kernel=4）。
    """
    layers = []
    for _ in range(NUM_LAYERS):
        x = (
            torch.randn((1, CONV_DIM, 1), generator=gen, device=device, dtype=torch.float32)
            * 0.1
        ).to(torch.bfloat16)
        state = (
            torch.randn(
                (1, CONV_DIM, STATE_LEN), generator=gen, device=device, dtype=torch.float32
            )
            * 0.1
        ).to(torch.bfloat16)
        weight = (
            torch.rand((CONV_DIM, CONV_KERNEL), generator=gen, device=device)
            - 0.5
        ).to(torch.bfloat16)
        layers.append({"x": x, "state": state, "weight": weight})
    return layers


def _reference(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, activation="silu"):
    """未修改的 torch fallback：cat + copy_ + F.conv1d + ACT2FN（原地写 state）。"""
    return ORIG_CONV(x, state, weight, None, activation)


def _states_snapshot(layers: list[dict]) -> list[torch.Tensor]:
    return [layer["state"].clone() for layer in layers]


def _restore(layers: list[dict], snapshot: list[torch.Tensor]) -> None:
    for layer, saved in zip(layers, snapshot):
        layer["state"].copy_(saved)


# ----------------------------------------------------------------------- 判定工具


def _numeric(case, name: str, got: torch.Tensor, ref: torch.Tensor) -> dict:
    metrics = error_metrics(got, ref)
    ok = (
        metrics["finite"]
        and metrics["nrmse"] <= GDN_TOL["nrmse"]
        and metrics["cosine"] >= GDN_TOL["cosine"]
        and metrics["max_rel"] <= GDN_TOL["max_rel"]
    )
    detail = {}
    for key, value in metrics.items():
        detail[key] = bool(value) if isinstance(value, bool) else float(f"{value:.9g}")
    case.gate(name, ok, detail)
    return metrics


def _track(worst: dict, metrics: dict) -> None:
    worst["nrmse"] = max(worst["nrmse"], metrics["nrmse"])
    worst["max_rel"] = max(worst["max_rel"], metrics["max_rel"])
    worst["cosine"] = min(worst["cosine"], metrics["cosine"])
    worst["finite"] = worst["finite"] and metrics["finite"]


# ------------------------------------------------------------------------- case 1


def _run_correctness(report: OpReport, device: torch.device) -> None:
    case = report.case(
        CORRECT_CASE,
        layers=NUM_LAYERS,
        conv_dim=CONV_DIM,
        conv_kernel=CONV_KERNEL,
        state_len=STATE_LEN,
        conv_state_shape=[1, CONV_DIM, STATE_LEN],
        io_dtype="bfloat16",
        steps=STEPS,
        kernel="gdn.ops.conv_decode.gdn_conv_decode_step",
        block=CONV_DEFAULT_BLOCK,
        num_warps=CONV_DEFAULT_NUM_WARPS,
        reference="transformers.causal_conv1d_update(cat+copy_+F.conv1d+silu)",
        tolerance=GDN_TOL,
    )
    gen = torch.Generator(device=device).manual_seed(SEED)
    layers = _conv_inputs(device, gen)
    x_snapshot = [layer["x"].clone() for layer in layers]
    history_absmax = [float(layer["state"].abs().max()) for layer in layers]
    ptr_before = [layer["state"].data_ptr() for layer in layers]
    bytes_before = [layer["state"].numel() * layer["state"].element_size() for layer in layers]
    dtype_before = [layer["state"].dtype for layer in layers]
    outs = [
        torch.empty((1, CONV_DIM, 1), dtype=torch.bfloat16, device=device)
        for _ in range(NUM_LAYERS)
    ]

    worst = {"nrmse": 0.0, "max_rel": 0.0, "cosine": 1.0, "finite": True}
    finite = True
    state_exact = True
    state_rolled = True
    layers_checked = 0
    steps_done = 0

    for step in range(1, STEPS + 1):
        for idx, layer in enumerate(layers):
            state = layer["state"]
            x = layer["x"]
            before = state.clone()
            # 第 2 步的 reference 从候选第 1 步写完后**自己的** state 出发。
            ref_state = state.clone()
            ref_out = _reference(x, ref_state, layer["weight"])
            cand = gdn_conv_decode_step(
                x,
                state,
                layer["weight"],
                out=outs[idx],
                block=CONV_DEFAULT_BLOCK,
                num_warps=CONV_DEFAULT_NUM_WARPS,
            )
            if cand is not outs[idx]:
                raise RuntimeError("候选没有写进调用方给的 out 槽")
            state_exact = state_exact and torch.equal(state, ref_state)
            state_rolled = state_rolled and not torch.equal(state, before)
            finite = finite and bool(torch.isfinite(cand).all())
            finite = finite and bool(torch.isfinite(state).all())
            _track(worst, _numeric(case, f"numeric:step{step}:layer{idx:02d}:out", cand, ref_out))
            _track(
                worst,
                _numeric(case, f"numeric:step{step}:layer{idx:02d}:conv_state", state, ref_state),
            )
            layers_checked += 1
        steps_done += 1

    # ``out=None``（未接线时的默认 API）必须新分配输出、不碰 x，且与显式 out 槽
    # 路径逐位一致。两条路径都用**同一个初值**，所以结果可以直接逐位比。
    slot_x = layers[0]["x"].clone()
    slot_state = layers[0]["state"].clone()
    slot_out = torch.empty((1, CONV_DIM, 1), dtype=torch.bfloat16, device=device)
    gdn_conv_decode_step(
        slot_x, slot_state, layers[0]["weight"], out=slot_out, activation="silu"
    )
    scratch_x = layers[0]["x"].clone()
    scratch_state = layers[0]["state"].clone()
    # reference 必须从**调用前**的 state 快照出发（job-000030 的第一类失败：
    # 拿候选写完之后的 state 当 reference 的初值，两边差一步 → nrmse 0.7）。
    ref_state = scratch_state.clone()
    ref_out = _reference(scratch_x.clone(), ref_state, layers[0]["weight"])
    fresh_out = gdn_conv_decode_step(
        scratch_x, scratch_state, layers[0]["weight"], activation="silu"
    )
    _track(worst, _numeric(case, "numeric:out-none-path:out", fresh_out, ref_out))
    _track(
        worst,
        _numeric(case, "numeric:out-none-path:conv_state", scratch_state, ref_state),
    )
    out_none_ok = (
        fresh_out is not scratch_x
        and fresh_out.dtype is torch.bfloat16
        and tuple(fresh_out.shape) == (1, CONV_DIM, 1)
        and torch.equal(fresh_out, slot_out)
        and torch.equal(scratch_state, slot_state)
        and torch.equal(scratch_x, layers[0]["x"])
    )

    # x 在整条路径上都必须逐位不变（in_proj_qkv 的输出只读）。
    input_preserved = all(
        torch.equal(layer["x"], saved) for layer, saved in zip(layers, x_snapshot)
    )

    # 激活真的被应用：同一初值下，候选 out 必须与「不做 activation」的 reference
    # 显著不同（同一个 kernel、同一条输入，只差 activation 参数）。
    act_slot = torch.empty((1, CONV_DIM, 1), dtype=torch.bfloat16, device=device)
    act_state = layers[0]["state"].clone()
    act_out = gdn_conv_decode_step(
        layers[0]["x"], act_state, layers[0]["weight"], out=act_slot, activation="silu"
    )
    no_act_state = layers[0]["state"].clone()
    no_act_out = ORIG_CONV(
        layers[0]["x"].clone(), no_act_state, layers[0]["weight"], None, None
    )
    silu_metrics = error_metrics(act_out, no_act_out)

    case.gate(
        "coverage:conv-shape-real",
        all(
            tuple(layer["x"].shape) == (1, CONV_DIM, 1)
            and tuple(layer["state"].shape) == (1, CONV_DIM, STATE_LEN)
            and tuple(layer["weight"].shape) == (CONV_DIM, CONV_KERNEL)
            and layer["x"].dtype is torch.bfloat16
            and layer["state"].dtype is torch.bfloat16
            and layer["weight"].dtype is torch.bfloat16
            for layer in layers
        )
        and CONV_DIM == 10240
        and CONV_KERNEL == 4,
        {
            "x_shape": [1, CONV_DIM, 1],
            "conv_state_shape": [1, CONV_DIM, STATE_LEN],
            "weight_shape": [CONV_DIM, CONV_KERNEL],
            "conv_dim": CONV_DIM,
            "conv_kernel": CONV_KERNEL,
            "state_len": STATE_LEN,
            "dtype": "bfloat16",
            "note": "state_len == conv kernel（reference 的 cat 长度是 5、取最后一个输出）",
        },
    )
    case.gate(
        "coverage:nonzero-history-state",
        min(history_absmax) > 1e-3,
        {
            "layers": NUM_LAYERS,
            "min_absmax": min(history_absmax),
            "max_absmax": max(history_absmax),
            "note": "初值是随机历史 mixed_qkv，不是全零 state",
        },
    )
    case.gate(
        "coverage:two-consecutive-steps",
        steps_done == STEPS == 2 and state_rolled,
        {
            "steps": steps_done,
            "state_rolled_every_step": state_rolled,
            "recursion": "step2 reference 从候选 step1 后的 state 出发",
        },
    )
    case.gate(
        "coverage:layers-48",
        len(layers) == NUM_LAYERS == 48 and layers_checked == NUM_LAYERS * STEPS,
        {"layers": len(layers), "comparisons": layers_checked},
    )
    case.gate(
        "coverage:state-pointer-bytes-dtype-stable",
        [layer["state"].data_ptr() for layer in layers] == ptr_before
        and [layer["state"].numel() * layer["state"].element_size() for layer in layers]
        == bytes_before
        and [layer["state"].dtype for layer in layers] == dtype_before
        and all(value == STATE_BYTES for value in bytes_before),
        {
            "layers": NUM_LAYERS,
            "identical_data_ptr": [layer["state"].data_ptr() for layer in layers] == ptr_before,
            "bytes_per_layer": STATE_BYTES,
            "total_mb": NUM_LAYERS * STATE_BYTES / 2**20,
            "note": "static cache / graph replay 依赖原地更新",
        },
    )
    case.gate(
        "coverage:state-bitwise-equals-reference",
        state_exact,
        {
            "note": "conv_state 只做 bf16 值搬运，候选必须与 reference 逐位相同",
        },
    )
    case.gate(
        "coverage:out-none-allocates-and-keeps-input",
        out_none_ok,
        {
            "returns_fresh_buffer": fresh_out is not scratch_x,
            "bitwise_equals_out_slot_path": torch.equal(fresh_out, slot_out),
            "state_bitwise_equals_out_slot_path": torch.equal(scratch_state, slot_state),
            "x_bitwise_unchanged": torch.equal(scratch_x, layers[0]["x"]),
            "out_shape": list(fresh_out.shape),
            "note": "out=None 新分配 [1,C,1]；x 只读（第一版的就地覆盖已删除）",
        },
    )
    case.gate(
        "coverage:input-read-only",
        input_preserved,
        {
            "layers": NUM_LAYERS,
            "note": "48 层的 x 在 2 步调用后逐位不变（in_proj_qkv 的输出只读）",
        },
    )
    case.gate(
        "coverage:silu-applied",
        silu_metrics["nrmse"] > GDN_TOL["nrmse"],
        {
            "nrmse_vs_no_activation": silu_metrics["nrmse"],
            "threshold": GDN_TOL["nrmse"],
            "note": "activation 没生效时 out 会与不做 silu 的 reference 相同",
        },
    )
    case.gate("coverage:finite", finite and worst["finite"], {"finite": finite})

    case.metric("worst_nrmse", worst["nrmse"])
    case.metric("worst_max_rel", worst["max_rel"])
    case.metric("worst_cosine", worst["cosine"])
    case.metric("layers_checked", float(layers_checked))
    case.metric("steps_checked", float(steps_done))
    case.metric("conv_state_bytes_per_layer", float(STATE_BYTES))
    case.metric("conv_state_total_mb", NUM_LAYERS * STATE_BYTES / 2**20)
    case.metric("x_bytes_per_layer", float(X_BYTES))
    case.metric("weight_bytes_per_layer", float(WEIGHT_BYTES))


# ------------------------------------------------------------------------- case 2


def _run_negative_controls(report: OpReport, device: torch.device) -> None:
    """三种错实现（用**同一个正确 kernel**、喂错输入）必须被容差挡住。

    1. 全零 conv_state ≈ 「kernel 只看当前 token」（out = silu(w3*x)）；
    2. conv_state 错位一格（``[s1,s2,s3,s0]``）：滚动/窗口取错；
    3. weight 反向（``w[:, ::-1]``）：conv 核方向接反。

    三者都是真实 bug 类型，且都不需要改 kernel 就能复现（喂错输入即可）。判据是
    **与正确输入的 reference 对比**：错输入算出的 out 必须显著超出容差，否则说明
    这个 case 没有分辨力。第一版把「reference 喂同样的错输入」当参照，两边一致地
    错、nrmse 恒为 0 —— 这正是负对照要防的失败模式（见 2026-10-23 job-000030）。
    """
    case = report.case(
        NEGATIVE_CASE,
        layers=NUM_LAYERS,
        conv_dim=CONV_DIM,
        state_len=STATE_LEN,
        steps=STEPS,
        controls=[
            "zero-conv-state(current-token-only)",
            "misaligned-conv-state(rolled-by-one)",
            "flipped-kernel-weight",
        ],
        tolerance=GDN_TOL,
        negative_margin=NEGATIVE_MARGIN,
    )
    gen = torch.Generator(device=device).manual_seed(SEED + 1)
    layers = _conv_inputs(device, gen)
    ptr_before = [layer["state"].data_ptr() for layer in layers]
    outs = [
        torch.empty((1, CONV_DIM, 1), dtype=torch.bfloat16, device=device)
        for _ in range(NUM_LAYERS)
    ]

    positive = {"nrmse": 0.0, "max_rel": 0.0, "cosine": 1.0, "finite": True}
    negatives = {
        "zero-conv-state": {"nrmse": 0.0, "max_rel": 0.0, "cosine": 1.0, "finite": True},
        "misaligned-conv-state": {"nrmse": 0.0, "max_rel": 0.0, "cosine": 1.0, "finite": True},
        "flipped-kernel-weight": {"nrmse": 0.0, "max_rel": 0.0, "cosine": 1.0, "finite": True},
    }
    steps_done = 0
    controls_checked = 0

    for step in range(1, STEPS + 1):
        for idx, layer in enumerate(layers):
            # 本步的输入 state：正对照和三个负对照都从**同一个** snapshot 出发，
            # 这样负对照的差异只来自被破坏的那一项（state 或 weight）。
            state_before = layer["state"].clone()
            ref_state = state_before.clone()
            ref_out = _reference(layer["x"], ref_state, layer["weight"])
            gdn_conv_decode_step(
                layer["x"],
                layer["state"],
                layer["weight"],
                out=outs[idx],
                block=CONV_DEFAULT_BLOCK,
                num_warps=CONV_DEFAULT_NUM_WARPS,
            )
            _track(positive, _numeric(case, f"positive:step{step}:layer{idx:02d}", outs[idx], ref_out))

            variants = {
                "zero-conv-state": (
                    torch.zeros_like(state_before),
                    layer["weight"],
                ),
                "misaligned-conv-state": (
                    torch.roll(state_before.clone(), shifts=-1, dims=-1),
                    layer["weight"],
                ),
                "flipped-kernel-weight": (
                    state_before.clone(),
                    torch.flip(layer["weight"], dims=[-1]).contiguous(),
                ),
            }
            for name, (wrong_state, wrong_weight) in variants.items():
                got = gdn_conv_decode_step(
                    layer["x"],
                    wrong_state,
                    wrong_weight,
                    out=outs[idx],
                    num_warps=CONV_DEFAULT_NUM_WARPS,
                )
                # 关键：负对照必须与**正确输入**的 reference 比。第一版拿
                # 「reference 喂同样的错输入」当参照，两边一致地错，nrmse 恒为 0，
                # 这个 case 就完全没有分辨力了。
                metrics = error_metrics(got, ref_out)
                current = negatives[name]
                current["nrmse"] = max(current["nrmse"], metrics["nrmse"])
                current["max_rel"] = max(current["max_rel"], metrics["max_rel"])
                current["cosine"] = min(current["cosine"], metrics["cosine"])
                current["finite"] = current["finite"] and metrics["finite"]
                controls_checked += 1
        steps_done += 1

    case.gate(
        "coverage:negative-controls",
        controls_checked == len(negatives) * NUM_LAYERS * STEPS,
        {
            "controls": sorted(negatives),
            "comparisons": controls_checked,
            "note": "每个负对照都跑 48 层 × 2 步，取最坏值",
        },
    )
    case.gate(
        "coverage:positive-control-in-same-case",
        positive["nrmse"] < GDN_TOL["nrmse"]
        and positive["cosine"] >= GDN_TOL["cosine"]
        and positive["max_rel"] <= GDN_TOL["max_rel"],
        {
            "nrmse": positive["nrmse"],
            "max_rel": positive["max_rel"],
            "cosine": positive["cosine"],
            "note": "同一份数据、同一个 kernel 的正确输入必须通过",
        },
    )
    for name in sorted(negatives):
        metrics = negatives[name]
        case.gate(
            f"coverage:negative-detected:{name}",
            metrics["nrmse"] > NEGATIVE_MARGIN * GDN_TOL["nrmse"]
            and metrics["cosine"] < GDN_TOL["cosine"]
            and metrics["finite"],
            {
                "nrmse": metrics["nrmse"],
                "max_rel": metrics["max_rel"],
                "cosine": metrics["cosine"],
                "threshold_nrmse": NEGATIVE_MARGIN * GDN_TOL["nrmse"],
                "positive_nrmse": positive["nrmse"],
                "nrmse_ratio_vs_positive": (
                    metrics["nrmse"] / positive["nrmse"] if positive["nrmse"] > 0 else None
                ),
                "note": "错实现必须被容差挡住，否则这个 case 没有分辨力",
            },
        )
    case.gate(
        "coverage:state-buffer-stable-across-controls",
        [layer["state"].data_ptr() for layer in layers] == ptr_before
        and all(torch.isfinite(layer["state"]).all() for layer in layers),
        {
            "note": "负对照在 state 的副本上跑；真实 state 只被正对照原地更新，data_ptr 不变",
            "layers": len(layers),
        },
    )

    case.metric("positive_nrmse", positive["nrmse"])
    case.metric("positive_max_rel", positive["max_rel"])
    case.metric("positive_cosine", positive["cosine"])
    for name in sorted(negatives):
        case.metric(f"negative_nrmse_{name.replace('-', '_')}", negatives[name]["nrmse"])
        case.metric(f"negative_cosine_{name.replace('-', '_')}", negatives[name]["cosine"])
    case.metric("steps_checked", float(steps_done))
    case.metric("controls_checked", float(controls_checked))
    case.metric("negative_margin", NEGATIVE_MARGIN)


# ------------------------------------------------------------------------- case 3


def _run_patch_wiring(report: OpReport, device: torch.device) -> None:
    """走 ``runner/model/gdn_patch.py`` 的真实接线（不加载 27B 模型）。

    用真实 config 的 meta GDN 模块（只取属性，不占显存）+ 真实调用点形状与真实 CUDA
    张量，把「conv 融合到底有没有被执行」变成可判定项：
    ``conv.counters.fused == 48 * 2``、``reference == 0``、``fallbacks == {}``。
    """
    case = report.case(
        PATCH_CASE,
        patch_module="runner/model/gdn_patch.py",
        entry="transformers...modeling_qwen3_5.causal_conv1d_update",
        layers=NUM_LAYERS,
        conv_dim=CONV_DIM,
        conv_kernel=CONV_KERNEL,
        state_len=STATE_LEN,
        call_x_shape=[1, CONV_DIM, 1],
        call_x_layout="in_proj_qkv(...).transpose(1,2) -> strides (C,1,C), channel stride 1",
        call_weight_shape=[CONV_DIM, CONV_KERNEL],
        steps=STEPS,
        tolerance=GDN_TOL,
    )
    gen = torch.Generator(device=device).manual_seed(SEED + 2)
    layers = _conv_inputs(device, gen)
    # 真实调用点布局：``mixed_qkv = in_proj_qkv(hidden).transpose(1, 2)``，即
    # [1, 1, C] contiguous 的 transpose —— shape [1, C, 1]、strides (C, 1, C)。
    # torch 仍把它判成 contiguous（size-1 维不参与判定），kernel 只依赖 channel
    # 维 stride=1，所以能直接收。
    for layer in layers:
        layer["x"] = layer["x"].reshape(1, 1, CONV_DIM).transpose(1, 2)
    x_snapshot = [layer["x"].clone() for layer in layers]
    x_strides = [tuple(layer["x"].stride()) for layer in layers]
    history_absmax = [float(layer["state"].abs().max()) for layer in layers]
    ptr_before = [layer["state"].data_ptr() for layer in layers]
    bytes_before = [layer["state"].numel() * layer["state"].element_size() for layer in layers]
    dtype_before = [layer["state"].dtype for layer in layers]

    config = SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        linear_num_value_heads=NUM_V_HEADS,
        linear_num_key_heads=NUM_K_HEADS,
        linear_key_head_dim=HEAD_K_DIM,
        linear_value_head_dim=HEAD_V_DIM,
        linear_conv_kernel_dim=CONV_KERNEL,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    with torch.device("meta"):
        gdn_module = gdn_modeling.Qwen3_5GatedDeltaNet(config, 0)

    # --- env flag：GDN_FUSED_DECODE=0 必须整体关掉（不 patch conv）
    gdn_patch.uninstall()
    patched_before = gdn_modeling.causal_conv1d_update is ORIG_CONV
    os.environ[gdn_patch.ENV_FLAG] = "0"
    try:
        disabled = gdn_patch.install_from_env(gdn_module)
    finally:
        del os.environ[gdn_patch.ENV_FLAG]
    env_disabled = (
        patched_before
        and disabled.get("enabled") is False
        and disabled.get("conv", {}).get("enabled") is False
        and gdn_modeling.causal_conv1d_update is ORIG_CONV
    )

    installed = gdn_patch.install_from_env(gdn_module)
    conv_info = installed.get("conv", {})
    worst = {"nrmse": 0.0, "max_rel": 0.0, "cosine": 1.0, "finite": True}
    finite = True
    state_exact = True
    steps_done = 0
    layers_checked = 0
    counters: dict = {}
    conv_counters: dict = {}
    fallback_output_matches = False
    fallback_reason_seen = None
    restored_after_uninstall = False
    final_info: dict = {}
    slot_ptrs: list[int] = []
    try:
        for step in range(1, STEPS + 1):
            for idx, layer in enumerate(layers):
                ref_state = layer["state"].clone()
                ref_out = _reference(layer["x"], ref_state, layer["weight"])
                # 真实调用点：模型把 repeat 前后的 mixed_qkv 与 conv_state 交给
                # 模块级函数（这里是 patch 之后的版本）。
                cand = gdn_modeling.causal_conv1d_update(
                    layer["x"], layer["state"], layer["weight"], None, "silu"
                )
                if cand.data_ptr() == layer["x"].data_ptr():
                    raise RuntimeError("conv patch 不应该改写 x（应写进独立 out 槽）")
                state_exact = state_exact and torch.equal(layer["state"], ref_state)
                finite = finite and bool(torch.isfinite(cand).all())
                finite = finite and bool(torch.isfinite(layer["state"]).all())
                _track(
                    worst,
                    _numeric(
                        case, f"numeric:step{step}:layer{idx:02d}:out", cand, ref_out
                    ),
                )
                _track(
                    worst,
                    _numeric(
                        case,
                        f"numeric:step{step}:layer{idx:02d}:conv_state",
                        layer["state"],
                        ref_state,
                    ),
                )
                layers_checked += 1
            if step == 1:
                slot_ptrs = [
                    gdn_patch._conv_out_slot(layer["state"]).data_ptr()
                    for layer in layers
                ]
            steps_done += 1

        slot_ptrs_after = [
            gdn_patch._conv_out_slot(layer["state"]).data_ptr() for layer in layers
        ]
        counters = dict(gdn_patch.counters())
        conv_counters = dict(gdn_patch.conv_counters())
        conv_counters["fallbacks"] = dict(conv_counters.get("fallbacks", {}))

        # 回退门：不支持的 dtype 必须回原实现（逐位一致）并计数，不能静默走 kernel。
        fused_before = int(conv_counters.get("fused", 0))
        wrong_x = layers[0]["x"].float()
        wrong_state = layers[0]["state"].clone()
        got = gdn_modeling.causal_conv1d_update(
            wrong_x, wrong_state, layers[0]["weight"], None, "silu"
        )
        expect_state = layers[0]["state"].clone()
        expected = ORIG_CONV(wrong_x.clone(), expect_state, layers[0]["weight"], None, "silu")
        fallback_output_matches = torch.equal(got, expected) and torch.equal(
            wrong_state, expect_state
        )
        after = dict(gdn_patch.conv_counters())
        after["fallbacks"] = dict(after.get("fallbacks", {}))
        fallback_reason_seen = sorted(
            set(after["fallbacks"]) - set(conv_counters["fallbacks"])
        )
        fallback_counted = (
            int(after.get("fused", 0)) == fused_before
            and int(after.get("reference", 0)) == int(conv_counters.get("reference", 0)) + 1
        )
        final_info = gdn_patch.conv_info()
    finally:
        gdn_patch.uninstall()
        restored_after_uninstall = gdn_modeling.causal_conv1d_update is ORIG_CONV

    fused = int(conv_counters.get("fused", 0))
    reference = int(conv_counters.get("reference", 0))
    fallbacks = conv_counters.get("fallbacks", {})
    expected_calls = NUM_LAYERS * STEPS
    case.gate(
        "patch:installed",
        bool(installed.get("conv", {}).get("enabled"))
        and conv_info.get("conv_dim") == CONV_DIM
        and conv_info.get("kernel_size") == CONV_KERNEL
        and conv_info.get("state_len") == STATE_LEN
        and conv_info.get("layers") == 1
        and conv_info.get("bias") is False,
        {
            key: conv_info.get(key)
            for key in ("enabled", "layers", "conv_dim", "kernel_size", "state_len", "bias", "activation")
        },
    )
    # 关键回归门：融合必须真的被执行，且没有任何 fallback。
    case.gate(
        "patch:fused-path-executed",
        fused == expected_calls and reference == 0 and not fallbacks,
        {
            "fused_calls": fused,
            "expected": expected_calls,
            "reference_calls": reference,
            "fallbacks": fallbacks,
            "note": "counters 是活 dict，check.json / bench.json 的 load.gdn_decode.conv 里也能看到",
        },
    )
    case.gate(
        "patch:falls-back-on-unsupported-dtype",
        fallback_counted
        and fallback_output_matches
        and fallback_reason_seen == ["x-dtype"],
        {
            "reason": fallback_reason_seen,
            "output_matches_reference": fallback_output_matches,
            "fused_before": fused,
            "note": "fp32 x 必须回原实现、逐位一致，并计入 conv.counters.fallbacks",
        },
    )
    case.gate(
        "patch:env-flag-disables",
        env_disabled,
        {
            "env": f"{gdn_patch.ENV_FLAG}=0",
            "install_result": {
                "enabled": disabled.get("enabled"),
                "conv_enabled": disabled.get("conv", {}).get("enabled"),
                "reason": disabled.get("reason"),
            },
            "module_attribute_restored": gdn_modeling.causal_conv1d_update is ORIG_CONV,
        },
    )
    case.gate(
        "patch:restored-after-uninstall",
        restored_after_uninstall,
        {"note": "uninstall() 必须同时恢复 recurrent 与 conv 两个模块级函数"},
    )
    case.gate(
        "coverage:real-call-shape",
        conv_info.get("conv_dim") == 10240
        and conv_info.get("kernel_size") == 4
        and all(
            tuple(layer["x"].shape) == (1, CONV_DIM, 1)
            and tuple(layer["state"].shape) == (1, CONV_DIM, STATE_LEN)
            for layer in layers
        ),
        {
            "x_shape": [1, CONV_DIM, 1],
            "conv_state_shape": [1, CONV_DIM, STATE_LEN],
            "weight_shape": [CONV_DIM, CONV_KERNEL],
            "note": "调用点传的是 in_proj_qkv 输出 transpose 后的 [1, 10240, 1]",
        },
    )
    case.gate(
        "coverage:nonzero-history-state",
        min(history_absmax) > 1e-3,
        {"min_absmax": min(history_absmax), "max_absmax": max(history_absmax)},
    )
    case.gate(
        "coverage:two-consecutive-steps",
        steps_done == STEPS == 2 and state_exact,
        {
            "steps": steps_done,
            "conv_state_bitwise_matches_reference": state_exact,
            "recursion": "step2 从 patch 写完的 state 出发",
        },
    )
    case.gate(
        "coverage:layers-48",
        len(layers) == NUM_LAYERS == 48 and layers_checked == NUM_LAYERS * STEPS,
        {"layers": len(layers), "comparisons": layers_checked},
    )
    case.gate(
        "coverage:state-pointer-bytes-dtype-stable",
        [layer["state"].data_ptr() for layer in layers] == ptr_before
        and [layer["state"].numel() * layer["state"].element_size() for layer in layers]
        == bytes_before
        and [layer["state"].dtype for layer in layers] == dtype_before,
        {"bytes_per_layer": STATE_BYTES, "layers": NUM_LAYERS},
    )
    case.gate(
        "coverage:out-slot-cache",
        int(final_info.get("out_slots", 0)) == NUM_LAYERS
        and int(final_info.get("out_bytes", 0)) == NUM_LAYERS * CONV_DIM * 2
        and slot_ptrs == slot_ptrs_after
        and len(set(slot_ptrs)) == NUM_LAYERS,
        {
            "out_slots": final_info.get("out_slots"),
            "out_bytes": final_info.get("out_bytes"),
            "expected_bytes": NUM_LAYERS * CONV_DIM * 2,
            "data_ptr_stable_across_steps": slot_ptrs == slot_ptrs_after,
            "distinct_slots": len(set(slot_ptrs)),
            "note": "按 conv_state.data_ptr() 缓存 48 个固定地址槽；capture 之后 replay 依赖",
        },
    )
    case.gate(
        "coverage:input-read-only",
        all(torch.equal(layer["x"], saved) for layer, saved in zip(layers, x_snapshot)),
        {
            "x_strides": x_strides[0] if x_strides else None,
            "note": "x 是 in_proj_qkv(transpose) 的 view，patch 只读，结果写独立 out 槽",
        },
    )
    case.gate(
        "coverage:non-contiguous-input-falls-back",
        gdn_patch._conv_fallback_reason(
            torch.empty(1, 2 * CONV_DIM, 1, dtype=torch.bfloat16, device=device)[:, ::2, :],
            layers[0]["state"],
            layers[0]["weight"],
            None,
            "silu",
        )
        == "x-stride",
        {
            "reason": "x-stride",
            "note": "channel 维不是 stride=1 的布局必须回退 reference，不能猜",
        },
    )
    case.gate("coverage:finite", finite and worst["finite"], {"finite": finite})

    case.metric("worst_nrmse", worst["nrmse"])
    case.metric("worst_max_rel", worst["max_rel"])
    case.metric("worst_cosine", worst["cosine"])
    case.metric("fused_calls", float(fused))
    case.metric("reference_calls", float(reference))
    case.metric("fallback_kinds", float(len(fallbacks)))
    case.metric("layers_checked", float(layers_checked))
    case.metric("steps_checked", float(steps_done))
    case.metric("conv_state_bytes_per_layer", float(STATE_BYTES))
    case.metric("out_bytes", float(final_info.get("out_bytes", 0)))


# ------------------------------------------------------------------------- case 4


def _time_loop(fn, reset, *, warmup: int, reps: int) -> float:
    """一次 48 层 conv step 的墙钟中位数（ms），event 边界只包住 fn。"""
    reset()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        reset()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _capture(fn, reset) -> torch.cuda.CUDAGraph:
    reset()
    stream = torch.cuda.Stream(device=torch.device("cuda"))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(GRAPH_WARMUP):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def _time_replay(graph: torch.cuda.CUDAGraph, *, reps: int = REPS) -> float:
    for _ in range(GRAPH_WARMUP):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _run_microbench(report: OpReport, device: torch.device) -> None:
    case = report.case(
        PERF_CASE,
        layers=NUM_LAYERS,
        conv_dim=CONV_DIM,
        state_len=STATE_LEN,
        io_dtype="bfloat16",
        warmup=WARMUP,
        reps=REPS,
        graph_warmup=GRAPH_WARMUP,
        boundary="CUDA event 包住一次 48 层完整 conv step，warmup 后取 15 次中位数",
        baseline="causal_conv1d_update（cat + copy_ + depthwise conv1d + silu，4 kernel/层）",
        candidate="gdn_conv_decode_step（1 kernel/层，conv_state 与 x 原地）",
        tolerance=GDN_TOL,
    )
    gen = torch.Generator(device=device).manual_seed(SEED + 3)
    layers = _conv_inputs(device, gen)
    snapshot = _states_snapshot(layers)
    outs = [
        torch.empty((1, CONV_DIM, 1), dtype=torch.bfloat16, device=device)
        for _ in range(NUM_LAYERS)
    ]

    def reset() -> None:
        _restore(layers, snapshot)

    def baseline_step() -> None:
        for layer in layers:
            _reference(layer["x"], layer["state"], layer["weight"], "silu")

    def candidate_step(block: int, warps: int) -> None:
        for idx, layer in enumerate(layers):
            gdn_conv_decode_step(
                layer["x"],
                layer["state"],
                layer["weight"],
                out=outs[idx],
                block=block,
                num_warps=warps,
            )

    # 每个粗粒度参数组都先对拍一次（快但错的配置没有意义）。
    for block, warps in CONV_BLOCK_CONFIGS:
        reset()
        ref_state = layers[0]["state"].clone()
        ref_out = _reference(layers[0]["x"], ref_state, layers[0]["weight"], "silu")
        got = gdn_conv_decode_step(
            layers[0]["x"],
            layers[0]["state"],
            layers[0]["weight"],
            out=outs[0],
            block=block,
            num_warps=warps,
        )
        _numeric(case, f"numeric:config-block{block}-w{warps}:out", got, ref_out)
        _numeric(
            case, f"numeric:config-block{block}-w{warps}:conv_state",
            layers[0]["state"], ref_state,
        )

    baseline_ms = _time_loop(baseline_step, reset, warmup=WARMUP, reps=REPS)
    config_ms = {
        (block, warps): _time_loop(
            lambda b=block, w=warps: candidate_step(b, w), reset, warmup=WARMUP, reps=REPS
        )
        for block, warps in CONV_BLOCK_CONFIGS
    }
    best = min(config_ms, key=config_ms.get)
    candidate_ms = config_ms[best]

    graph_baseline_ms = None
    graph_candidate_ms = None
    graph_error = None
    try:
        baseline_graph = _capture(baseline_step, reset)
        candidate_graph = _capture(lambda: candidate_step(best[0], best[1]), reset)
        graph_baseline_ms = _time_replay(baseline_graph)
        graph_candidate_ms = _time_replay(candidate_graph)
    except Exception as exc:  # noqa: BLE001 - graph 只作 diagnostic，不阻塞 eager 结论
        graph_error = f"{type(exc).__name__}: {exc}"[:400]

    case.gate(
        "coverage:model-shapes",
        NUM_LAYERS == 48 and CONV_DIM == 10240 and STATE_LEN == CONV_KERNEL == 4,
        {
            "layers": NUM_LAYERS,
            "conv_dim": CONV_DIM,
            "state_len": STATE_LEN,
            "conv_kernel": CONV_KERNEL,
            "io_dtype": "bfloat16",
            "conv_state_total_mb": NUM_LAYERS * STATE_BYTES / 2**20,
        },
    )
    case.gate(
        "coverage:same-timing-boundary",
        True,
        {
            "boundary": "event 包住一次 48 层完整 conv step；reset 与 sync 在 event 之外",
            "warmup": WARMUP,
            "reps": REPS,
            "baseline_includes": [
                "torch.cat([conv_state, x])",
                "conv_state.copy_ 回滚",
                "F.conv1d(depthwise, kernel=4)",
                "F.silu",
                "中间 tensor 分配",
            ],
            "candidate_includes": ["48 次 kernel launch", "conv_state / x 原地写回"],
        },
    )
    case.gate(
        "coverage:parameter-sweep",
        len(config_ms) == len(CONV_BLOCK_CONFIGS) == 3,
        {
            "configs": [f"block{b}-w{w}" for b, w in CONV_BLOCK_CONFIGS],
            "selected": f"block{best[0]}-w{best[1]}",
        },
    )
    case.gate(
        "coverage:state-restored-between-iterations",
        True,
        {"note": "每个计时样本前把 48 层 conv_state 复位到同一 snapshot"},
    )
    case.gate(
        "metrics-positive-finite",
        baseline_ms > 0.0
        and candidate_ms > 0.0
        and all(value > 0.0 for value in config_ms.values()),
        {"baseline_ms": baseline_ms, "candidate_ms": candidate_ms},
    )
    if graph_baseline_ms is not None and graph_candidate_ms is not None:
        case.gate(
            "coverage:graph-replay-timed",
            graph_baseline_ms > 0.0 and graph_candidate_ms > 0.0,
            {
                "note": "生产路径是 CUDA Graph，replay 计时去掉 Python/launch 开销",
                "graph_baseline_ms": graph_baseline_ms,
                "graph_candidate_ms": graph_candidate_ms,
            },
        )
        case.metric("graph_baseline_ms_step48", graph_baseline_ms)
        case.metric("graph_candidate_ms_step48", graph_candidate_ms)
        case.metric("graph_saving_ms", graph_baseline_ms - graph_candidate_ms)
        case.metric("graph_speedup", graph_baseline_ms / graph_candidate_ms)
        case.metric("graph_baseline_us_per_layer", graph_baseline_ms * 1000.0 / NUM_LAYERS)
        case.metric("graph_candidate_us_per_layer", graph_candidate_ms * 1000.0 / NUM_LAYERS)
    else:
        case.diag("graph-capture", False, {"error": graph_error})

    case.metric("baseline_ms_step48", baseline_ms)
    case.metric("candidate_ms_step48", candidate_ms)
    case.metric("baseline_us_per_layer", baseline_ms * 1000.0 / NUM_LAYERS)
    case.metric("candidate_us_per_layer", candidate_ms * 1000.0 / NUM_LAYERS)
    case.metric("eager_speedup", baseline_ms / candidate_ms)
    for (block, warps), value in config_ms.items():
        case.metric(f"candidate_block{block}_w{warps}_ms_step48", value)
    case.diag(
        "traffic-estimate",
        True,
        {
            "note": "解析估算，不是实测",
            "conv_state_rw_mb_per_layer": 2 * CONV_DIM * STATE_LEN * 2 / 2**20,
            "x_read_mb_per_layer": X_BYTES / 2**20,
            "out_write_mb_per_layer": X_BYTES / 2**20,
            "weight_read_mb_per_layer": WEIGHT_BYTES / 2**20,
            "total_mb_per_step": (
                NUM_LAYERS
                * (2 * STATE_BYTES + 2 * X_BYTES + WEIGHT_BYTES)
                / 2**20
            ),
            "baseline_kernels_per_layer": 4,
            "candidate_kernels_per_layer": 1,
            "baseline_nodes_per_step": 4 * NUM_LAYERS,
            "candidate_nodes_per_step": NUM_LAYERS,
        },
    )
    case.diag(
        "reference-path",
        True,
        {
            "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d")
            is not None,
            "note": "causal_conv1d 未安装，reference 走 transformers 的纯 torch fallback",
        },
    )


# --------------------------------------------------------------------------- main


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        default="all",
        help=f"逗号分隔的 case 名，或 all（默认）。可选：{', '.join(SCENARIOS)}",
    )
    parser.add_argument("--out", required=True, help="产物目录，写 {out}/op.json")
    return parser.parse_args(argv)


def _select(only: str) -> tuple[str, ...]:
    if only.strip().lower() in ("", "all"):
        return SCENARIOS
    names = tuple(name.strip() for name in only.split(",") if name.strip())
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; SCENARIOS={SCENARIOS}")
    return names


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    selected = _select(args.only)
    device = torch.device("cuda")
    report = OpReport(
        args.out,
        candidate_name="gdn-conv-decode-fused-triton",
        candidate_format="bf16-conv-state+bf16-io",
        candidate_extra={
            "kernel": "runner/ops/gdn/conv_decode.py::_gdn_conv_decode_kernel",
            "boundary": "causal_conv1d_update decode branch (seq_len==1)",
            "kernels_per_layer": "4 -> 1",
            "block_configs": [f"block{b}-w{w}" for b, w in CONV_BLOCK_CONFIGS],
            "default_config": f"block{CONV_DEFAULT_BLOCK}-w{CONV_DEFAULT_NUM_WARPS}",
            "state_layout": "in-place bf16 [1,10240,4], data_ptr stable",
            "out_layout": "per-state bf16 [1,10240,1] slot (48 x 20 KB), address stable",
            "patch": "runner/model/gdn_patch.py::install_conv",
        },
        environment={
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(str(v) for v in torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "cases": list(selected),
        },
    )
    with torch.no_grad():
        if CORRECT_CASE in selected:
            _run_correctness(report, device)
        if NEGATIVE_CASE in selected:
            _run_negative_controls(report, device)
        if PERF_CASE in selected:
            _run_microbench(report, device)
        # 会临时改动全局 patch，放在最后。
        if PATCH_CASE in selected:
            _run_patch_wiring(report, device)
    payload = report.finish()
    summary = payload["summary"]
    for entry in summary["cases"]:
        print(
            f"[{entry['case']}] gates {entry['gates_passed']}/{entry['gates']} "
            f"failed={entry['gates_failed']}"
        )
    for entry in payload["cases"]:
        print(
            f"[{entry['case']}] metrics "
            + ", ".join(f"{k}={v:.4g}" for k, v in entry["metrics"].items())
        )
    print(f"op.json -> {Path(args.out) / 'op.json'}")
    print("PASS" if summary["ok"] else "FAIL")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
