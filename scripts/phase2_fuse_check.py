#!/usr/bin/env python3
"""Phase 2 第 3 轮任务 L：GDN gated RMSNorm + out_proj activation quant 融合的 op gate。

候选：``runner/ops/gdn/gated_norm.py`` 的单个 Triton kernel —— 一次 launch 取代
``Qwen3_5RMSNormGated.forward``（decode 时约 12 个 tiny kernel）**加** out_proj 的
``nvfp4_quantize``，输出 BF16 的 ``normed * silu(gate)`` 和 out_proj 需要的
``(a_fp4, a_sf)``。

决定性参考 = **模型实际执行的那条路径**，判定逐字节相同（不是容差）：

    y = Qwen3_5RMSNormGated.forward(core_attn_out, z)      # bf16 权重，weight 直接乘
    a_fp4, a_sf = nvfp4_quantize(y.reshape(1, 6144), 1 / out_proj.input_scale,
                                 layout_128x4, do_shuffle=False)

冻结的 gate（实现候选之前写下，不因结果调整）
============================================
1. ``bitwise[...]``：``out_bf16``（[48,128]）、``a_fp4`` 字节、``a_sf`` 字节（含
   127 行 padding 必须为 0）三者都与模块路径 + ``nvfp4_quantize`` 逐字节相同。
2. ``mean-tree``：``x.pow(2).mean(-1)`` 在 **[48,128]** 上的 reduction 顺序用 torch op
   逐级复刻后与 torch 自身逐位相同。把「对 launch 配置的读法」和「Triton 实现」分开
   定位 —— 这是本任务 bitwise 的最大风险点（reduction 维是 128 而不是 5120，
   ``Reduce.cuh::setReduceConfig`` 给出的 block/vec 配置完全不同）。
3. ``negative-controls``：三种错实现（``1 + w`` 的 sibling convention、丢掉 gate、
   reduction 维用 6144）必须**显著**超差；本 case 里正确候选仍必须逐字节相同
   （正对照），否则说明 gate 没有分辨力。

coverage（由 ``coverage:`` gate 证明清单齐全）
=============================================
- **48 个真实 GDN 层**（config 里 48 个 linear_attention 层）× 每个层**自己的**
  ``linear_attn.norm.weight`` 与 ``linear_attn.out_proj.input_scale``；
  5 种输入形态：``gauss-0.5`` / ``gauss-2`` / ``gauss-8``（带 20x 离群通道，
  残差流/conv 输出的典型形态）、``saturation``（让 ``sf_pre`` 冲 e4m3 的 satfinite）、
  ``zeros``（``amax == 0`` 分支）。gate 输入另用独立的 ``gauss-2`` / ``gauss-4``。
- 48 个真实 ``gs`` 一起覆盖 scale 范围（产物里记 min/max）。
- ``real`` 形态用 checkpoint 的 embedding 行拼成 ``[48,128]``（真实量级、非零），
  避免「全随机」把 gate 变成只测一种分布。
- ``wiring``：真实 ``Qwen3_5GatedDeltaNet``（meta）+ 真实 CUDA 权重走
  ``layer.norm(...)`` -> ``layer.out_proj(...)``，断言融合真的被执行
  （``fused=1`` / ``unconsumed=0`` / 无 fallback）、且 out_proj 仍走
  ``_gemm(m=1)`` 的 decode backend 选型（``selected["5120x6144"] == "cute-dsl"``）。
  m3_check 的 bitwise gate 只比较执行方式，patch 没生效时照样全绿（第 2 轮踩过），
  所以「融合是否被执行」必须是产物级 gate。

microbench：同一计时边界（CUDA event / graph replay）比较「模块 norm + 一次
``nvfp4_quantize``」与融合 kernel 的 per-call ms，再按 48 层折算整步节省。

整份脚本在 GPU 上约 60–120 秒（含 Triton / FlashInfer 首次编译）。

用法：
    .venv/bin/python scripts/phase2_fuse_check.py --out {output}/op
    .venv/bin/python scripts/phase2_fuse_check.py --only mean-tree micro --out .tmp/op
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from runner.env import configure_flashinfer_env  # noqa: E402

configure_flashinfer_env()

from runner.experiment.op_artifact import OpReport  # noqa: E402
from runner.loader.checkpoint import CheckpointReader  # noqa: E402
from runner.model import gated_norm_patch  # noqa: E402
from runner.ops.gdn.gated_norm import (  # noqa: E402
    DEFAULT_NUM_WARPS,
    HEAD_V_DIM,
    MEAN_FACTOR,
    N_BLOCKS,
    NUM_V_HEADS,
    NUM_WARPS_CONFIGS,
    SF_COLS,
    SF_ROWS,
    VALUE_DIM,
    allocate_scratch,
    gated_norm_fused,
    mean_factor,
)
from runner.ops.nvfp4.linear import NVFP4Linear  # noqa: E402
from runner.ops.nvfp4.norm_quant import C6, global_divisor  # noqa: E402
from runner.ops.nvfp4.reference import error_metrics  # noqa: E402

MODEL_DIR = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"
CKPT_PREFIX = "model.language_model."

SCENARIOS = (
    "mean-tree",
    "real-layers",
    "negative-controls",
    "wiring",
    "micro",
)

CANDIDATE_NAME = "gdn-gated-norm-quant-fused-r2"
CANDIDATE_FORMAT = "nvfp4-w4a4"

# ---- 冻结的输入与采样参数（实现候选之前确定）------------------------------------
SEED = 20261023
RMS_LEVELS = (0.5, 2.0, 8.0)
GATE_LEVELS = (2.0, 4.0)
OUTLIER_CHANNELS = (3, 37, 91)
OUTLIER_GAIN = 20.0
SATURATION_INDEX = (7, 61)
SATURATION_VALUE = 3.0e4
#: embedding 行拼成 [48,128]：48*128 = 6144 = 1.2 个 5120 行的长度，取两行拼接。
EMBED_ROWS = (11, 12, 13, 14)
INPUT_KINDS = ("gauss-0.5", "gauss-2", "gauss-8", "saturation", "zeros", "real")

BENCH_WARMUP = 20
BENCH_ITERS = 200
GRAPH_WARMUP = 5
GRAPH_ITERS = 8
GRAPH_REPEATS = 7

#: M2 冻结的 GDN 容差（负对照只需要「显著超差」，用同一把尺子）。
GDN_TOL = {"nrmse": 1e-3, "cosine": 0.99999, "max_rel": 5e-3}
NEGATIVE_MARGIN = 10.0


# ------------------------------------------------------------------ checkpoint


@dataclass
class Checkpoint:
    gdn_layers: list[int]  # 48 个 linear_attention 层的下标
    norm_weights: dict[int, torch.Tensor]  # layer -> bf16 [128] CPU
    out_proj_scales: dict[int, float]  # layer -> float
    embedding: torch.Tensor  # bf16 [rows, 5120] CPU

    def gs(self, layer: int) -> float:
        return global_divisor(self.out_proj_scales[layer])

    def gs_range(self) -> tuple[float, float]:
        values = [self.gs(layer) for layer in self.gdn_layers]
        return min(values), max(values)


def load_checkpoint() -> Checkpoint:
    config = json.loads((MODEL_DIR / "config.json").read_text())["text_config"]
    layer_types = list(config["layer_types"])
    gdn_layers = [i for i, kind in enumerate(layer_types) if kind == "linear_attention"]
    if len(gdn_layers) != NUM_V_HEADS:
        raise SystemExit(f"GDN 层数 {len(gdn_layers)} != kernel 的 {NUM_V_HEADS}")
    norm_weights: dict[int, torch.Tensor] = {}
    out_proj_scales: dict[int, float] = {}
    with CheckpointReader(MODEL_DIR) as reader:
        for layer in gdn_layers:
            base = CKPT_PREFIX + f"layers.{layer}.linear_attn"
            norm_weights[layer] = reader.get(base + ".norm.weight", "cpu").to(torch.bfloat16)
            out_proj_scales[layer] = float(reader.get(base + ".out_proj.input_scale", "cpu").item())
        embed_name = CKPT_PREFIX + "embed_tokens.weight"
        handle = reader._handle(reader.weight_map[embed_name], "cpu")  # noqa: SLF001
        embedding = handle.get_slice(embed_name)[EMBED_ROWS[0] : EMBED_ROWS[-1] + 1].float()
    return Checkpoint(gdn_layers, norm_weights, out_proj_scales, embedding)


# ------------------------------------------------------------------ 参考路径


class NormGatedReference:
    """模型实际执行的 ``Qwen3_5RMSNormGated.forward``（bf16 权重，weight 直接乘）。"""

    def __init__(self, weight_bf16: torch.Tensor, eps: float, device: torch.device) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated

        module = Qwen3_5RMSNormGated(HEAD_V_DIM, eps=float(eps))
        module.weight = torch.nn.Parameter(
            weight_bf16.detach().to(device).to(torch.bfloat16), requires_grad=False
        )
        self.module = module.to(device).eval()
        self.weight = module.weight.detach()

    def bf16(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module(x, gate)


def nvfp4_quantize_bytewise(y_bf16: torch.Tensor, gs: float, device: torch.device):
    """模型里的 ``nvfp4_quantize``（per-tensor global divisor，128x4 swizzled，不 shuffle）。"""
    from flashinfer import SfLayout, nvfp4_quantize

    gs_t = torch.tensor([gs], dtype=torch.float32, device=device)
    return nvfp4_quantize(
        y_bf16.reshape(1, VALUE_DIM),
        gs_t,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )


def reference_two_step(
    x: torch.Tensor, gate: torch.Tensor, norm_ref: NormGatedReference, gs: float,
    device: torch.device,
):
    """现有两步路径：模块 gated norm -> out_proj 的 quantize。"""
    y = norm_ref.bf16(x, gate)
    a_fp4, a_sf = nvfp4_quantize_bytewise(y, gs, device)
    return y, a_fp4, a_sf


def run_fused(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gs: float,
    scratch,
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
):
    out, a_fp4, a_sf = scratch
    gated_norm_fused(x, gate, weight, eps, gs, out, a_fp4, a_sf, num_warps=num_warps)
    return out, a_fp4, a_sf


# ------------------------------------------------------------------ 比较工具


def byte_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """两个 tensor 的字节差异数（按扁平字节比，rank 差异不影响）；长度不同返回 -1。"""
    left = a.contiguous().view(torch.uint8).reshape(-1)
    right = b.contiguous().view(torch.uint8).reshape(-1)
    if left.numel() != right.numel():
        return -1
    return int((left != right).sum().item())


# ------------------------------------------------------------------ 输入构造


def _fill_embedding(checkpoint: Checkpoint, device: torch.device) -> torch.Tensor:
    """用真实 embedding 行拼出 ``[48,128]``（6144 个真实激活量级的元素）。"""
    flat = checkpoint.embedding.reshape(-1)
    if flat.numel() < VALUE_DIM:
        raise SystemExit("embedding 行数不足，拼不出 [48,128]")
    return flat[:VALUE_DIM].to(torch.bfloat16).reshape(NUM_V_HEADS, HEAD_V_DIM)


def make_inputs(
    kind: str, layer: int, checkpoint: Checkpoint, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """返回 ``(core_attn_out [48,128], z [48,128], 描述)``，同一个 (kind, layer) 恒给同一份。"""
    seed = SEED + 7919 * layer + 13 * sum(ord(c) for c in kind)
    gen = torch.Generator().manual_seed(seed)
    if kind == "zeros":
        x = torch.zeros(NUM_V_HEADS, HEAD_V_DIM, dtype=torch.float32)
    elif kind == "real":
        x = _fill_embedding(checkpoint, device).float()
    else:
        rms = {"gauss-0.5": RMS_LEVELS[0], "gauss-2": RMS_LEVELS[1],
               "gauss-8": RMS_LEVELS[2], "saturation": RMS_LEVELS[0]}[kind]
        x = torch.randn(NUM_V_HEADS, HEAD_V_DIM, generator=gen, dtype=torch.float32) * rms
        for channel in OUTLIER_CHANNELS:
            x[:, channel] *= OUTLIER_GAIN
        if kind == "saturation":
            row, col = SATURATION_INDEX
            x[row, col] = SATURATION_VALUE
    gate = torch.randn(
        NUM_V_HEADS, HEAD_V_DIM, generator=gen, dtype=torch.float32
    ) * GATE_LEVELS[layer % len(GATE_LEVELS)]
    return (
        x.to(torch.bfloat16).to(device).contiguous(),
        gate.to(torch.bfloat16).to(device).contiguous(),
        kind,
    )


# ------------------------------------------------------------------ case


def run_mean_tree(report: OpReport, checkpoint: Checkpoint, device: torch.device) -> None:
    """``[48,128]`` contiguous FP32 的 ``pow(2).mean(-1)`` reduction 顺序（bitwise）。"""
    case = report.case(
        "mean-tree",
        shape=[NUM_V_HEADS, HEAD_V_DIM],
        dtype="float32",
        reduction="torch.mean(-1) 的 gpu_reduce_kernel 顺序",
        launch="block=(32,16) grid=(3,1) vec4 lanes=32 values_per_thread=4",
        tree="ThreadReduce float4 -> 线程内 4 累加器顺序合并 -> offset 16,8,4,2,1",
        factor=mean_factor(),
    )
    factor = torch.tensor(mean_factor(), dtype=torch.float32, device=device)
    variants = {"halving-16-1": 0, "tree-merge": 0}
    samples = 0
    seq_vs_tree = 0
    for kind in INPUT_KINDS:
        for layer in (0, 7, 16, 33, 47):
            x, _, _ = make_inputs(kind, layer, checkpoint, device)
            squares = x.float().pow(2)
            lane = torch.arange(HEAD_V_DIM // 4, device=device)

            def lane_values(order: str) -> torch.Tensor:
                cols = [4 * lane + j for j in range(4)]
                q = [squares[:, c] for c in cols]
                if order == "tree-merge":
                    return (q[0] + q[1]) + (q[2] + q[3])
                return ((q[0] + q[1]) + q[2]) + q[3]

            # 线程内合并顺序：顺序 vs 树。只作 diagnostic（两者一般不同）。
            seq_vs_tree += byte_diff(lane_values("sequential"), lane_values("tree-merge"))
            for name in variants:
                total = lane_values(name)
                for width in (16, 8, 4, 2, 1):
                    total = total.reshape(NUM_V_HEADS, 2, width).sum(dim=1)
                variance = total.reshape(-1) * factor
                reference = squares.mean(-1)
                diff = byte_diff(variance, reference)
                variants[name] += 0 if diff == 0 else 1
                samples += 1
    failed = variants["halving-16-1"]
    case.gate(
        "bitwise: 五级 halving（offset 16,8,4,2,1）+ 线程内顺序合并 == torch.mean",
        failed == 0,
        {"mismatched_samples": failed, "samples": samples},
    )
    case.diag(
        "线程内 4 个累加器按 0,1,2,3 顺序合并 vs 树合并（(a+b)+(c+d)）的字节差异",
        True,
        {"sequential_vs_tree_diff_bytes": seq_vs_tree, "samples": samples},
    )
    case.diag(
        "变体对照：线程内改成树合并（(a+b)+(c+d)）必须对不上",
        variants["tree-merge"] > 0,
        {"mismatched_samples": variants["tree-merge"], "samples": samples},
    )
    case.gate(
        "coverage: 5 层 × 6 种输入（含 zeros/saturation/real/3 个 RMS 量级）",
        samples == len(INPUT_KINDS) * 5 * len(variants),
        {"inputs": list(INPUT_KINDS), "layers": [0, 7, 16, 33, 47]},
    )
    case.metric("mean_tree_samples", float(samples))
    case.metric("mean_tree_mismatches", float(failed))
    case.metric("mean_tree_tree_merge_mismatches", float(variants["tree-merge"]))


def run_real_layers(
    report: OpReport,
    checkpoint: Checkpoint,
    device: torch.device,
    scratch,
    norm_refs: dict[int, NormGatedReference],
    eps: float,
) -> None:
    """48 个真实层 × 6 种输入：``out_bf16`` / ``a_fp4`` / ``a_sf`` 全部逐字节相同。"""
    sample_bytes = {
        "out_bf16": NUM_V_HEADS * HEAD_V_DIM * 2,
        "a_fp4": N_BLOCKS * 8,
        "a_sf": SF_ROWS * SF_COLS,
    }
    case = report.case(
        "real-layers",
        layers=len(checkpoint.gdn_layers),
        shapes={"core_attn_out": [NUM_V_HEADS, HEAD_V_DIM], "out_proj": [5120, VALUE_DIM]},
        inputs=list(INPUT_KINDS),
        consumers=1,
        quant_rule="R2（任务 E 冻结）：PTX 算术序列 + rcp.approx.ftz",
        rmsnorm_gated_rule="weight 直接乘；先 cast 回 bf16 再乘；reduction 在 head_v_dim=128",
        decisive_reference="Qwen3_5RMSNormGated.forward（模块）+ nvfp4_quantize",
        gate="bitwise（逐字节），不用容差",
        gs_range=list(checkpoint.gs_range()),
        sample_bytes=sample_bytes,
    )
    worst = {"out_bf16": 0, "a_fp4": 0, "a_sf": 0}
    codes = 0
    samples = 0
    nonzero_amax = 0
    finite = True
    failed_layers: list[int] = []
    for layer in checkpoint.gdn_layers:
        norm_ref = norm_refs[layer]
        gs = checkpoint.gs(layer)
        details: list[dict] = []
        for kind in INPUT_KINDS:
            x, gate, _ = make_inputs(kind, layer, checkpoint, device)
            y_ref, a_ref, s_ref = reference_two_step(x, gate, norm_ref, gs, device)
            y, a_fp4, a_sf = run_fused(x, gate, norm_ref.weight, eps, gs, scratch)
            torch.cuda.synchronize()
            diffs = {
                "out_bf16": byte_diff(y, y_ref),
                "a_fp4": byte_diff(a_fp4, a_ref),
                "a_sf": byte_diff(a_sf, s_ref),
            }
            for name, value in diffs.items():
                worst[name] = max(worst[name], value)
            finite = finite and all(
                bool(torch.isfinite(t.float()).all().item()) for t in (y, a_fp4, a_sf)
            )
            codes += a_fp4.numel() * 2
            samples += 1
            nonzero_amax += 0 if kind == "zeros" else 1
            if any(value != 0 for value in diffs.values()):
                # 附上定位信息：差异幅度与第一个不同的字节下标，方便一次修复。
                diff = (y.float() - y_ref.float()).abs().flatten()
                details.append(
                    {
                        "kind": kind,
                        "diff_bytes": diffs,
                        "out_max_abs": float(diff.max().item()),
                        "first_out_index": int(diff.argmax().item()),
                        "out_amax": float(y.float().abs().max().item()),
                    }
                )
        if details:
            failed_layers.append(layer)
        case.gate(
            f"bitwise[layers.{layer}.linear_attn.norm]",
            not details,
            details or {"kinds": list(INPUT_KINDS), "gs": gs},
        )
    case.gate(
        "coverage: 48 个真实 GDN 层（各自 norm.weight + out_proj.input_scale）× 6 种输入"
        " × {out_bf16, a_fp4, a_sf} 三个张量",
        samples == len(checkpoint.gdn_layers) * len(INPUT_KINDS),
        {
            "layers": len(checkpoint.gdn_layers),
            "inputs": list(INPUT_KINDS),
            "tensors": ["out_bf16", "a_fp4", "a_sf"],
            "samples": samples,
        },
    )
    case.gate(
        "finite: 所有 out_bf16 / a_fp4 / a_sf 都是有限值",
        finite,
        {"samples": samples},
    )
    case.metric("worst_diff_out_bf16_bytes", float(worst["out_bf16"]))
    case.metric("worst_diff_a_fp4_bytes", float(worst["a_fp4"]))
    case.metric("worst_diff_a_sf_bytes", float(worst["a_sf"]))
    case.metric("samples", float(samples))
    case.metric("e2m1_codes_compared", float(codes))
    case.metric("layers_with_mismatch", float(len(failed_layers)))
    case.metric("nonzero_amax_samples", float(nonzero_amax))


def run_negative_controls(
    report: OpReport,
    checkpoint: Checkpoint,
    device: torch.device,
    scratch,
    norm_refs: dict[int, NormGatedReference],
    eps: float,
) -> None:
    """三种错实现必须显著超差；正确候选在本 case 里仍逐字节相同（正对照）。"""
    import triton
    import triton.language as tl

    layer = checkpoint.gdn_layers[5]
    norm_ref = norm_refs[layer]
    gs = checkpoint.gs(layer)
    weight = norm_ref.weight
    case = report.case(
        "negative-controls",
        layer=layer,
        variants={
            "weight-plus-one": "用 sibling 的 1 + w（Qwen3_5RMSNorm 的 convention）",
            "no-gate": "不做 gate，直接输出 normed",
            "reduce-6144": "variance 在整条 6144 上求平均，而不是每行 128",
        },
        negative_margin=NEGATIVE_MARGIN,
        tolerance=GDN_TOL,
    )

    @triton.jit
    def _wrong_kernel(x_ptr, g_ptr, w_ptr, out_ptr, eps, factor, factor_all,
                      MODE: tl.constexpr):
        head = tl.arange(0, 128)
        w = tl.load(w_ptr + head).to(tl.float32)
        all_sum = tl.zeros((1,), tl.float32)
        for chunk in tl.static_range(6):
            rows = chunk * 8 + tl.arange(0, 8)
            offs = rows[:, None] * 128 + head[None, :]
            xv = tl.load(x_ptr + offs).to(tl.float32)
            v4 = tl.reshape(xv, (8, 32, 2, 2))
            even, odd = tl.split(v4)
            x0, x2 = tl.split(even)
            x1, x3 = tl.split(odd)
            q0 = x0 * x0
            q1 = x1 * x1
            q2 = x2 * x2
            q3 = x3 * x3
            total = ((q0 + q1) + q2) + q3
            total = tl.sum(tl.reshape(total, (8, 2, 16)), axis=1)
            total = tl.sum(tl.reshape(total, (8, 2, 8)), axis=1)
            total = tl.sum(tl.reshape(total, (8, 2, 4)), axis=1)
            total = tl.sum(tl.reshape(total, (8, 2, 2)), axis=1)
            total = tl.sum(tl.reshape(total, (8, 2, 1)), axis=1)  # (8,1)
            if MODE == 2:
                all_sum += tl.sum(tl.reshape(total, (8,)), axis=0)
                variance = all_sum * factor_all + eps
            else:
                variance = total * factor + eps
            rstd = tl.rsqrt(variance)
            xn = xv * rstd
            xb = xn.to(tl.bfloat16)
            if MODE == 0:
                wv = 1.0 + w
            else:
                wv = w
            nb = (xb.to(tl.float32) * wv[None, :]).to(tl.bfloat16)
            if MODE == 1:
                yf = nb.to(tl.float32)
            else:
                gv = tl.load(g_ptr + offs).to(tl.float32)
                yf = nb.to(tl.float32) * (gv / (1.0 + tl.exp(-gv)))
            tl.store(out_ptr + offs, yf.to(tl.bfloat16))

    x, gate, _ = make_inputs("gauss-2", layer, checkpoint, device)
    y_ref, a_ref, s_ref = reference_two_step(x, gate, norm_ref, gs, device)
    weight_plus_one = (1.0 + weight.float()).to(torch.bfloat16).contiguous()
    out = scratch[0]
    details: dict[str, dict] = {}
    for mode, name in enumerate(("weight-plus-one", "no-gate", "reduce-6144")):
        wv = weight_plus_one if mode == 0 else weight
        _wrong_kernel[(1,)](
            x, gate, wv, out, float(eps), MEAN_FACTOR, 1.0 / VALUE_DIM, MODE=mode,
        )
        torch.cuda.synchronize()
        y_wrong = out.clone()
        metrics = error_metrics(y_wrong, y_ref)
        diff_bytes = byte_diff(y_wrong, y_ref)
        exceeds = metrics["nrmse"] > NEGATIVE_MARGIN * GDN_TOL["nrmse"]
        details[name] = {**metrics, "diff_bytes": diff_bytes, "exceeds": exceeds}
        case.gate(
            f"negative[{name}] 显著超差（nrmse > {NEGATIVE_MARGIN} × {GDN_TOL['nrmse']}）",
            exceeds and diff_bytes > 0,
            details[name],
        )
    # 正对照：同一个输入、正确候选逐字节相同（负对照的同一把尺子下 nrmse = 0）。
    y, a_fp4, a_sf = run_fused(x, gate, weight, eps, gs, scratch)
    torch.cuda.synchronize()
    positive = {
        "out_bf16": byte_diff(y, y_ref),
        "a_fp4": byte_diff(a_fp4, a_ref),
        "a_sf": byte_diff(a_sf, s_ref),
    }
    case.gate(
        "positive: 正确候选在负对照的同一份输入上逐字节相同",
        all(value == 0 for value in positive.values()),
        positive,
    )
    case.gate(
        "coverage: 3 种错实现 × 同一份输入 + 正对照（含 a_fp4/a_sf）",
        len(details) == 3 and all(item["diff_bytes"] > 0 for item in details.values()),
        {"variants": list(details), "positive": positive},
    )
    case.metric("positive_nrmse", 0.0)
    for name, item in details.items():
        case.metric(f"negative_nrmse_{name}", float(item["nrmse"]))
        case.metric(f"negative_diff_bytes_{name}", float(item["diff_bytes"]))


def run_wiring(
    report: OpReport,
    checkpoint: Checkpoint,
    device: torch.device,
    eps: float,
) -> None:
    """真实 GDN 层（meta）+ 真实 CUDA 权重：融合真的被执行，且 out_proj 仍走 decode 选型。"""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5GatedDeltaNet,
        Qwen3_5RMSNormGated,
    )
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config_json = json.loads((MODEL_DIR / "config.json").read_text())["text_config"]
    text_config = Qwen3_5TextConfig(**config_json)
    layer_index = checkpoint.gdn_layers[0]
    base = CKPT_PREFIX + f"layers.{layer_index}.linear_attn"

    with torch.device("meta"):
        layer = Qwen3_5GatedDeltaNet(text_config, layer_index)
    layer.norm = Qwen3_5RMSNormGated(HEAD_V_DIM, eps=float(eps))
    layer.norm.weight = torch.nn.Parameter(
        checkpoint.norm_weights[layer_index].to(device).to(torch.bfloat16),
        requires_grad=False,
    )
    with CheckpointReader(MODEL_DIR) as reader:
        weight = reader.get(base + ".out_proj.weight", "cpu").to(device)
        scale = reader.get(base + ".out_proj.weight_scale", "cpu").to(device)
        scale2 = reader.get(base + ".out_proj.weight_scale_2", "cpu").to(device)
        input_scale = reader.get(base + ".out_proj.input_scale", "cpu").to(device)
    layer.out_proj = NVFP4Linear(weight, scale, scale2, input_scale, backend="cutlass")

    holder = torch.nn.Module()
    holder.add_module("layer0", layer)

    x, gate, _ = make_inputs("gauss-2", layer_index, checkpoint, device)
    case = report.case(
        "wiring",
        layer=layer_index,
        path="layer.norm(core_attn_out, z) -> layer.out_proj(...)（真实模块，meta harness）",
        backend_expectation="selected['5120x6144'] == 'cute-dsl'（任务 F 的 decode 选型）",
        gate="out_bf16 / out_proj 输出逐字节相同 + 计数器证据",
    )
    gated_norm_patch.uninstall()
    info: dict = {}
    counters: dict = {}
    backend: dict = {}
    y_ref = layer.norm(x, gate)
    out_ref = layer.out_proj(y_ref.reshape(1, 1, VALUE_DIM))
    torch.cuda.synchronize()
    try:
        info = gated_norm_patch.install(holder)
        y_fused = layer.norm(x, gate)
        out_fused = layer.out_proj(y_fused.reshape(1, 1, VALUE_DIM))
        torch.cuda.synchronize()
        counters = dict(gated_norm_patch.info()["counters"])
        counters["fallbacks"] = dict(counters["fallbacks"])
        from runner.ops.nvfp4.linear import decode_backend_stats

        backend = decode_backend_stats()
        backend["selected"] = dict(backend["selected"])
    finally:
        gated_norm_patch.uninstall()

    out_diff = byte_diff(out_fused, out_ref)
    y_diff = byte_diff(y_fused, y_ref)
    case.gate(
        "bitwise: norm 输出与 out_proj 输出都与未打 patch 的模块路径相同",
        out_diff == 0 and y_diff == 0,
        {"out_bf16_diff_bytes": y_diff, "out_proj_diff_bytes": out_diff},
    )
    case.gate(
        "wiring: fused == 1、unconsumed == 0、无 fallback（融合真的被执行）",
        counters.get("fused") == 1
        and counters.get("unconsumed") == 0
        and not counters.get("fallbacks"),
        counters,
    )
    case.gate(
        "backend: out_proj 的 M=1 调用仍走 _gemm 的 decode 选型（cute-dsl）",
        backend.get("selected", {}).get("5120x6144") == "cute-dsl"
        and backend.get("fallbacks", 0) == 0,
        backend,
    )
    case.gate(
        "coverage: 真实 Qwen3_5GatedDeltaNet + 真实 norm/out_proj 权重（layer "
        f"{layer_index}）",
        isinstance(info, dict) and info.get("layers") == 1 and info.get("enabled") is True,
        {"install_info": {k: v for k, v in info.items() if k != "counters"}},
    )
    case.metric("out_bf16_diff_bytes", float(y_diff))
    case.metric("out_proj_diff_bytes", float(out_diff))
    case.metric("fused_calls", float(counters.get("fused", 0)))
    case.metric("unconsumed", float(counters.get("unconsumed", 0)))
    case.metric("fallback_kinds", float(len(counters.get("fallbacks", {}))))


# ------------------------------------------------------------------ microbench


def time_per_call(fn, *, warmup: int, iters: int) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for index in range(iters):
            starts[index].record()
            fn()
            ends[index].record()
        torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    return {"median_ms": times[len(times) // 2], "min_ms": times[0]}


def graph_per_call(fn, *, warmup: int, iters: int, repeats: int) -> dict[str, float] | None:
    """把 ``iters`` 次调用录进一张 CUDA graph，replay 多次取 per-call 中位数。"""
    try:
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iters):
                fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / iters)
        samples.sort()
        return {"median_ms": samples[len(samples) // 2], "min_ms": samples[0]}
    except Exception:  # noqa: BLE001 - capture 不可用时退回 eager 口径
        return None


def run_micro(
    report: OpReport,
    checkpoint: Checkpoint,
    device: torch.device,
    scratch,
    norm_refs: dict[int, NormGatedReference],
    eps: float,
) -> None:
    """同一计时边界比较「模块 norm + quantize」与融合 kernel 的 per-call ms。"""
    layer = checkpoint.gdn_layers[3]
    norm_ref = norm_refs[layer]
    gs = checkpoint.gs(layer)
    x, gate, _ = make_inputs("gauss-2", layer, checkpoint, device)
    out, a_fp4, a_sf = scratch
    gs_t = torch.tensor([gs], dtype=torch.float32, device=device)

    from flashinfer import SfLayout, nvfp4_quantize

    def two_step():
        y = norm_ref.bf16(x, gate)
        nvfp4_quantize(
            y.reshape(1, VALUE_DIM),
            gs_t,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )

    def fused():
        gated_norm_fused(x, gate, norm_ref.weight, eps, gs, out, a_fp4, a_sf)

    sweep = {
        warps: time_per_call(
            lambda warps=warps: gated_norm_fused(
                x, gate, norm_ref.weight, eps, gs, out, a_fp4, a_sf, num_warps=warps
            ),
            warmup=BENCH_WARMUP,
            iters=BENCH_ITERS,
        )
        for warps in NUM_WARPS_CONFIGS
    }
    eager = {
        "two_step": time_per_call(two_step, warmup=BENCH_WARMUP, iters=BENCH_ITERS),
        "fused": time_per_call(fused, warmup=BENCH_WARMUP, iters=BENCH_ITERS),
    }
    graph_two = graph_per_call(
        two_step, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, repeats=GRAPH_REPEATS
    )
    graph_fused = graph_per_call(
        fused, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, repeats=GRAPH_REPEATS
    )

    case = report.case(
        "micro",
        layer=layer,
        warmup=BENCH_WARMUP,
        iters=BENCH_ITERS,
        graph_iters=GRAPH_ITERS,
        graph_repeats=GRAPH_REPEATS,
        num_warps_sweep=list(NUM_WARPS_CONFIGS),
        timing="per-call CUDA events median（eager）；graph = 8 次调用录一张图后 replay / 8",
        step_model="48 个 GDN 层，每层 1 次",
        reference="Qwen3_5RMSNormGated.forward（模块）+ 一次 nvfp4_quantize",
    )
    steps = 48
    estimate = {
        "eager_saved_per_call_ms": eager["two_step"]["median_ms"] - eager["fused"]["median_ms"],
        "eager_saved_step_ms": steps
        * (eager["two_step"]["median_ms"] - eager["fused"]["median_ms"]),
    }
    if graph_two is not None and graph_fused is not None:
        estimate["graph_saved_per_call_ms"] = graph_two["median_ms"] - graph_fused["median_ms"]
        estimate["graph_saved_step_ms"] = steps * (
            graph_two["median_ms"] - graph_fused["median_ms"]
        )
    case.gate(
        "coverage: eager 口径 × num_warps 2/4/8 三档（graph 口径另作 diagnostic）",
        len(sweep) == len(NUM_WARPS_CONFIGS) and len(eager) == 2,
        {
            "num_warps_sweep": list(NUM_WARPS_CONFIGS),
            "graph_captured": graph_two is not None and graph_fused is not None,
        },
    )
    case.gate(
        "timing: 所有中位数都是有限正数",
        all(
            entry["median_ms"] > 0.0 and math.isfinite(entry["median_ms"])
            for entry in [*eager.values(), *sweep.values()]
            if entry is not None
        ),
        {
            "two_step_ms": eager["two_step"]["median_ms"],
            "fused_ms": eager["fused"]["median_ms"],
        },
    )
    for name, entry in eager.items():
        case.metric(f"eager_{name}_ms", entry["median_ms"])
    for warps, entry in sweep.items():
        case.metric(f"fused_warps{warps}_ms", entry["median_ms"])
    for name, value in estimate.items():
        case.metric(name, value)
    if graph_two is not None and graph_fused is not None:
        case.metric("graph_two_step_ms", graph_two["median_ms"])
        case.metric("graph_fused_ms", graph_fused["median_ms"])
        case.diag(
            "graph 口径（模型实际执行方式）",
            True,
            {"two_step": graph_two, "fused": graph_fused},
        )


# ------------------------------------------------------------------ entry


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True, help="产物目录，写 {out}/op.json")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=f"只跑指定 case（默认全部）；可选 {', '.join(SCENARIOS)}",
    )
    return parser.parse_args(argv)


def select_cases(only: list[str] | None) -> list[str]:
    if only is None:
        return list(SCENARIOS)
    names = [part for item in only for part in item.split(",") if part]
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"未知 case: {unknown}；可选 {list(SCENARIOS)}")
    return names


def summarize(payload: dict) -> dict:
    metrics = {entry["case"]: entry.get("metrics", {}) for entry in payload["cases"]}
    return {
        "ok": payload["summary"]["ok"],
        "cases": [
            {**entry, "metrics": metrics.get(entry["case"], {})}
            for entry in payload["summary"]["cases"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA 不可用：这个 op 脚本只能在 GPU job 里运行", file=sys.stderr)
        return 2
    device = torch.device("cuda")
    # 与模型一致：loader 在载入后 `torch.set_grad_enabled(False)`；patch 的命中条件里
    # 有 ``not torch.is_grad_enabled()``，op 脚本必须同样关掉，否则接线 gate 只会看到
    # 回退计数（那是**正确行为**，不是 bug）。
    torch.set_grad_enabled(False)
    torch.manual_seed(SEED)
    selected = select_cases(args.only)
    checkpoint = load_checkpoint()
    eps = 1e-6
    norm_refs = {
        layer: NormGatedReference(checkpoint.norm_weights[layer], eps, device)
        for layer in checkpoint.gdn_layers
    }
    scratch = allocate_scratch(device=device)
    report = OpReport(
        args.out,
        CANDIDATE_NAME,
        CANDIDATE_FORMAT,
        candidate_extra={
            "kernel": "runner/ops/gdn/gated_norm.py::_gated_norm_quant_kernel",
            "patch": "runner/model/gated_norm_patch.py (Qwen3_5RMSNormGated.forward, decode only)",
            "consumers": "1（每层只有 out_proj）",
            "hidden": VALUE_DIM,
            "head_v_dim": HEAD_V_DIM,
            "num_v_heads": NUM_V_HEADS,
            "num_warps": DEFAULT_NUM_WARPS,
            "mean_factor": MEAN_FACTOR,
            "c6_bits": hex(0x3E2AAAAB),
            "c6": C6,
            "quant_rule": "R2（任务 E 冻结）：PTX 算术序列 + rcp.approx.ftz",
            "gated_norm_rule": "weight 直接乘（不是 1+w）；先 cast bf16 再乘；"
            "reduction 在 head_v_dim=128 上按 gpu_reduce_kernel 的顺序",
            "silu": "PyTorch silu_kernel：x / (1 + exp(-x))，div.rn + libdevice __nv_expf",
            "decisive_reference": "Qwen3_5RMSNormGated.forward（模块）+ nvfp4_quantize",
            "gate": "bitwise（逐字节），不用容差",
            "negative_controls": ["weight-plus-one", "no-gate", "reduce-6144"],
        },
    )
    handler = {
        "mean-tree": lambda rep: run_mean_tree(rep, checkpoint, device),
        "real-layers": lambda rep: run_real_layers(
            rep, checkpoint, device, scratch, norm_refs, eps
        ),
        "negative-controls": lambda rep: run_negative_controls(
            rep, checkpoint, device, scratch, norm_refs, eps
        ),
        "wiring": lambda rep: run_wiring(rep, checkpoint, device, eps),
        "micro": lambda rep: run_micro(rep, checkpoint, device, scratch, norm_refs, eps),
    }
    for name in selected:
        handler[name](report)
    payload = report.finish(
        extra={
            "coverage": {
                "gdn_layers": len(checkpoint.gdn_layers),
                "gs_range": list(checkpoint.gs_range()),
                "input_kinds": list(INPUT_KINDS),
                "tensors": ["out_bf16", "a_fp4", "a_sf"],
            },
            "model_dir": str(MODEL_DIR),
        }
    )
    print(json.dumps(summarize(payload), indent=2, ensure_ascii=False))
    print(f"output: {args.out / 'op.json'}")
    return 0 if payload["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
