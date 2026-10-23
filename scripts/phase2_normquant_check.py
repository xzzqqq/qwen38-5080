#!/usr/bin/env python3
"""Phase 2 任务 H：RMSNorm + NVFP4 activation quant 融合的 op gate + microbench。

候选：``runner/ops/nvfp4/norm_quant.py`` 的单个 Triton kernel —— 一次 launch 读
pre-norm BF16 hidden，算 zero-centered RMSNorm（结果 round 回 BF16），并为该 norm 的
每个 NVFP4 消费者各产一份 ``(a_fp4, a_sf)``（128x4 swizzled）。

参考 = 模型现有的两步路径，**判定逐字节相同**（不是容差）：

    y = F.rms_norm(x.float(), (K,), 1 + weight.float(), eps).to(bf16)
    a_fp4, a_sf = nvfp4_quantize(y, 1 / input_scale, layout_128x4, do_shuffle=False)

``F.rms_norm`` 在 torch 2.14 是 composite（``input.mul(rsqrt(pow(input,2).mean(-1)+eps))
.mul(weight)``），与 ``Qwen3_5RMSNorm.forward`` 的 eager 序列同序；本脚本用
``reference-agreement`` gate 证明这两条参考途径逐位相同（不靠文档假设）。

冻结的 gate（实现候选之前写下，不因结果调整）
================================================
1. ``bitwise[...]``：融合 kernel 的 ``out_bf16``、每个消费者的 ``a_fp4`` 字节、``a_sf``
   字节都与两步路径逐字节相同。三条都要过：
   - ``out_bf16`` 也必须逐位相同，因为 linear-attention 层的 ``in_proj_a`` /
     ``in_proj_b``（未量化）直接消费这个 BF16 张量；
   - ``a_fp4`` 覆盖 E2M1 code 与 nibble 顺序；``a_sf`` 覆盖 e4m3 scale 与 128x4
     swizzle 落点（含 padding 必须为 0）。
2. ``mean-tree``：``x.pow(2).mean(-1)`` 的 reduction 顺序用 torch op 逐级复刻后与
   torch 自身逐位相同。把「对 launch 配置的读法」和「Triton 实现」分开定位：
   这条失败说明我复刻的顺序错了，而不是 kernel 写错了。
3. ``rsqrt-approx``：inline PTX ``rsqrt.approx.f32`` 与 ``torch.rsqrt`` 在真实
   variance 值上逐位相同。rstd 差 1 ulp 就会改变 BF16 rounding => code 变。
4. ``reference-agreement``：``F.rms_norm`` 与 ``Qwen3_5RMSNorm`` 模块逐位相同。

coverage（由 ``coverage:`` gate 证明清单齐全）
=============================================
- **128 个真实 decoder-layer norm**（64 层 × input/post）× checkpoint 里**它们自己的**
  消费者 ``input_scale``：48 个 linear_attention 层的 input norm 是 2 消费者
  （in_proj_qkv + in_proj_z）、16 个 full_attention 层是 3 消费者（q/k/v）、
  64 个 post norm 是 2 消费者（gate/up）。所以 2/3 消费者两种形态都被真实覆盖。
- ``input_scale`` 的**全局最小**（``layers.0.mlp.gate_proj`` = 0.0012091，gs = 827.14）
  与**全局最大**（``layers.63.mlp.down_proj`` = 0.18155，gs = 5.5080）。后者不在融合
  消费者里，所以单独一档强制覆盖两端。
- ``amax == 0``（全零输入 = padding 行）、离群值、sf_pre 冲 ``satfinite``（amax 大到
  e4m3 饱和）、真实 embedding 行、以及多种真实量级的输入（RMS 0.5 / 2 / 8 + 少量
  20x 离群通道）。

microbench：分别测「norm + 2 消费者」「norm + 3 消费者」的两步路径与融合路径
per-call ms（CUDA event 中位数；eager 提交区间 + 能 capture 时的 graph replay 区间），
再按真实的 48/16/64 层分布折算整步节省。graph 数字更接近模型实际执行（模型跑在
CUDA graph 里，单 kernel 成本是 GPU 侧 dispatch），eager 数字包含 CPU 提交开销，
只作诊断。

整份脚本在 GPU 上约 30–60 秒（checkpoint 读 128 个 norm 权重 + 401 个 scale 是毫秒级）。

用法：
    .venv/bin/python scripts/phase2_normquant_check.py --out {output}/op
    .venv/bin/python scripts/phase2_normquant_check.py --only mean-tree micro --out ...
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
from runner.ops.nvfp4.norm_quant import (  # noqa: E402
    C6,
    DEFAULT_NUM_WARPS,
    HIDDEN,
    MEAN_FACTOR,
    SF_ROWS,
    global_divisor,
    norm_quant_fused,
)

MODEL_DIR = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"
CKPT_PREFIX = "model.language_model."

# ---- 冻结的输入与采样参数（实现候选之前确定）------------------------------------
SEED = 20261023
RMS_LEVELS = (0.5, 2.0, 8.0)
OUTLIER_CHANNELS = (17, 1024, 4095)
OUTLIER_GAIN = 20.0
SATURATION_CHANNEL = 7
SATURATION_VALUE = 3.0e4
EMBED_ROWS = (0, 1, 2, 3, 4, 5)  # embedding 的前几行作为真实数据
BENCH_WARMUP = 20
BENCH_ITERS = 200
GRAPH_WARMUP = 5
GRAPH_ITERS = 8
GRAPH_REPEATS = 7
NUM_WARPS_SWEEP = (2, DEFAULT_NUM_WARPS, 8)

SCENARIOS = (
    "mean-tree",
    "rsqrt-approx",
    "reference-agreement",
    "real-layers",
    "scales-extremes",
    "distribution-edge",
    "micro",
)

CANDIDATE_NAME = "norm-quant-fused-r2"
CANDIDATE_FORMAT = "nvfp4-w4a4"


# ------------------------------------------------------------------ checkpoint


@dataclass
class Checkpoint:
    layer_types: list[str]
    norm_weights: dict[str, torch.Tensor]  # short name -> bf16 [K] CPU
    input_scales: dict[str, float]  # short module name -> float
    embedding_rows: torch.Tensor  # bf16 [n, K] CPU

    def consumers(self, layer_index: int) -> tuple[list[str], list[str]]:
        """该层 input norm / post norm 的消费者（短模块名）。"""
        if self.layer_types[layer_index] == "linear_attention":
            pre = [
                f"layers.{layer_index}.linear_attn.in_proj_qkv",
                f"layers.{layer_index}.linear_attn.in_proj_z",
            ]
        else:
            pre = [
                f"layers.{layer_index}.self_attn.q_proj",
                f"layers.{layer_index}.self_attn.k_proj",
                f"layers.{layer_index}.self_attn.v_proj",
            ]
        post = [
            f"layers.{layer_index}.mlp.gate_proj",
            f"layers.{layer_index}.mlp.up_proj",
        ]
        return pre, post

    def gs(self, module: str) -> float:
        return global_divisor(self.input_scales[module])

    def norm_names(self) -> list[str]:
        return sorted(self.norm_weights)

    def scale_extremes(self) -> tuple[str, str]:
        ordered = sorted(self.input_scales.items(), key=lambda item: item[1])
        return ordered[0][0], ordered[-1][0]


def load_checkpoint() -> Checkpoint:
    import json as _json

    config = _json.loads((MODEL_DIR / "config.json").read_text())["text_config"]
    layer_types = list(config["layer_types"])
    hidden = int(config["hidden_size"])
    if hidden != HIDDEN:
        raise SystemExit(f"checkpoint hidden {hidden} != kernel {HIDDEN}")

    norm_weights: dict[str, torch.Tensor] = {}
    input_scales: dict[str, float] = {}
    with CheckpointReader(MODEL_DIR) as reader:
        for layer_index in range(len(layer_types)):
            for suffix in ("input_layernorm", "post_attention_layernorm"):
                short = f"layers.{layer_index}.{suffix}"
                norm_weights[short] = reader.get(
                    CKPT_PREFIX + short + ".weight", "cpu"
                ).float()
        for name in reader.weight_map:
            if not name.endswith(".input_scale") or not name.startswith(CKPT_PREFIX):
                continue
            short = name[len(CKPT_PREFIX) : -len(".input_scale")]
            input_scales[short] = float(reader.get(name, "cpu").float().item())
        embed_name = CKPT_PREFIX + "embed_tokens.weight"
        shard = reader.weight_map[embed_name]
        handle = reader._handle(shard, "cpu")  # noqa: SLF001 - 只读切片，避免整表 2.5 GB
        start = EMBED_ROWS[0]
        stop = EMBED_ROWS[-1] + 1
        embedding_rows = handle.get_slice(embed_name)[start:stop].float()
    return Checkpoint(layer_types, norm_weights, input_scales, embedding_rows)


# ------------------------------------------------------------------ 输入构造


def gauss_input(seed: int, rms: float, *, outliers: bool = True) -> torch.Tensor:
    """真实量级的 pre-norm hidden：高斯底噪 + 少量放大的离群通道（残差流的典型形态）。"""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(HIDDEN, generator=generator, dtype=torch.float32) * rms
    if outliers:
        for channel in OUTLIER_CHANNELS:
            x[channel] *= OUTLIER_GAIN
    return x.to(torch.bfloat16).reshape(1, 1, HIDDEN)


def saturation_input(seed: int) -> torch.Tensor:
    """让 amax/6*gs 超过 e4m3 的 448：覆盖 cvt.rn.satfinite 的饱和分支。"""
    x = gauss_input(seed, 0.5, outliers=False).reshape(HIDDEN).float()
    x[SATURATION_CHANNEL] = SATURATION_VALUE
    return x.to(torch.bfloat16).reshape(1, 1, HIDDEN)


def zero_input() -> torch.Tensor:
    return torch.zeros(1, 1, HIDDEN, dtype=torch.bfloat16)


def embed_input(checkpoint: Checkpoint, index: int) -> torch.Tensor:
    return checkpoint.embedding_rows[index].to(torch.bfloat16).reshape(1, 1, HIDDEN)


def input_for(
    kind: str, layer_index: int, checkpoint: Checkpoint, device: torch.device
) -> tuple[torch.Tensor, str]:
    """返回 (x [1,1,K] bf16 cuda, 描述)。同一个 (kind, layer) 恒给同一份输入。"""
    seed = SEED + 7919 * layer_index + 13 * sum(ord(c) for c in kind)
    if kind == "gauss-0.5":
        x = gauss_input(seed, RMS_LEVELS[0])
    elif kind == "gauss-2":
        x = gauss_input(seed, RMS_LEVELS[1])
    elif kind == "gauss-8":
        x = gauss_input(seed, RMS_LEVELS[2])
    elif kind == "zeros":
        x = zero_input()
    elif kind == "saturation":
        x = saturation_input(seed)
    elif kind == "embed":
        x = embed_input(checkpoint, layer_index % checkpoint.embedding_rows.shape[0])
    else:
        raise SystemExit(f"unknown input kind {kind!r}")
    return x.to(device, non_blocking=False).contiguous(), kind


# ------------------------------------------------------------------ 参考路径


def _flashinfer():
    from flashinfer import SfLayout, nvfp4_quantize

    return SfLayout, nvfp4_quantize


def reference_two_step(
    x: torch.Tensor,
    w1: torch.Tensor,
    eps: float,
    gs_list: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """现有两步路径：``F.rms_norm(f32)`` -> round 到 BF16 -> 每消费者一次 quantize。"""
    SfLayout, nvfp4_quantize = _flashinfer()
    xf = x.reshape(1, HIDDEN).float()
    y = torch.nn.functional.rms_norm(xf, (HIDDEN,), w1, float(eps)).to(torch.bfloat16)
    outputs = []
    for gs in gs_list:
        gs_t = torch.tensor([gs], dtype=torch.float32, device=device)
        a_fp4, a_sf = nvfp4_quantize(
            y,
            gs_t,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )
        outputs.append((a_fp4, a_sf))
    return y, outputs


def model_path_reference(
    x: torch.Tensor, norm_weight_bf16: torch.Tensor, eps: float
) -> torch.Tensor:
    """模型实际执行的 ``Qwen3_5RMSNorm.forward``（zero-centered，bf16 权重）。"""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    module = Qwen3_5RMSNorm(HIDDEN, eps=float(eps))
    module.weight.data.copy_(norm_weight_bf16.to(module.weight.device))
    module = module.to(x.device)
    module.eval()
    with torch.no_grad():
        return module(x).clone()


def emulate_mean_tree(x: torch.Tensor) -> torch.Tensor:
    """用 torch op 逐级复刻 ``x.pow(2).mean(-1)`` 的 reduction 顺序（bitwise gate）。"""
    squares = x.reshape(HIDDEN).float().pow(2)
    # 每个 PyTorch 线程 4 个 float4 累加器，步长 512 个 vec 组，第三次迭代只有前 256 组
    lane = torch.arange(512, device=x.device)
    acc = None
    for j in range(4):
        a = squares[4 * lane + j]
        b = squares[4 * (lane + 512) + j]
        c = torch.where(
            lane < 256,
            squares[(4 * (lane + 1024) + j).clamp_max(HIDDEN - 1)],
            torch.zeros((), dtype=torch.float32, device=x.device),
        )
        term = (a + b) + c
        acc = term if acc is None else acc + term
    total = acc.reshape(512)
    for _ in range(9):
        total = total.reshape(2, -1).sum(dim=0)
    factor = torch.tensor(MEAN_FACTOR, dtype=torch.float32, device=x.device)
    return total * factor


def rsqrt_probe(values: torch.Tensor) -> torch.Tensor:
    """inline PTX ``rsqrt.approx.f32`` 探针（与 kernel 内用的是同一条指令）。"""
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        y = tl.inline_asm_elementwise(
            "rsqrt.approx.f32 $0, $1;",
            "=r,r",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(y_ptr + offs, y, mask=mask)

    x = values.float().contiguous()
    y = torch.empty_like(x)
    _kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), BLOCK=256)
    return y


# ------------------------------------------------------------------ 比较工具


def byte_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """两个 tensor 的字节差异数（按扁平字节比，rank 差异不影响）；长度不同返回 -1。

    a_sf 的 rank 可能在 FlashInfer 版本间变成 ``[1, 128, 320]``，但 128x4 swizzle 的
    语义是扁平字节布局，所以这里按字节比；shape 单独记在 gate detail 里。
    """
    left = a.contiguous().view(torch.uint8).reshape(-1)
    right = b.contiguous().view(torch.uint8).reshape(-1)
    if left.numel() != right.numel():
        return -1
    return int((left != right).sum().item())


def _scratch(device: torch.device, n: int):
    out = torch.empty(1, 1, HIDDEN, dtype=torch.bfloat16, device=device)
    a_fp4 = [torch.empty(1, HIDDEN // 2, dtype=torch.uint8, device=device) for _ in range(n)]
    a_sf = [
        torch.zeros(SF_ROWS, HIDDEN // 16, dtype=torch.uint8, device=device)
        for _ in range(n)
    ]
    return out, a_fp4, a_sf


def run_case(
    case,
    x: torch.Tensor,
    w1: torch.Tensor,
    eps: float,
    gs_list: list[float],
    device: torch.device,
    *,
    label: str,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> dict:
    """一次融合 vs 两步的逐字节比较 + 每个 gate。"""
    out, a_fp4, a_sf = _scratch(device, len(gs_list))
    with torch.no_grad():
        norm_quant_fused(
            x, w1, eps, gs_list, out, a_fp4, a_sf, num_warps=num_warps
        )
        y_ref, refs = reference_two_step(x, w1, eps, gs_list, device)
    torch.cuda.synchronize()

    bf16_diff = byte_diff(out, y_ref)
    codes = 0
    details = []
    ok = bf16_diff == 0
    for index, (a_ref, sf_ref) in enumerate(refs):
        a_diff = byte_diff(a_fp4[index], a_ref)
        sf_diff = byte_diff(a_sf[index], sf_ref)
        codes += a_ref.numel() * 2
        ok = ok and a_diff == 0 and sf_diff == 0
        details.append(
            {
                "consumer": index,
                "gs": gs_list[index],
                "a_fp4_diff_bytes": a_diff,
                "a_sf_diff_bytes": sf_diff,
                "a_fp4_shape": list(a_ref.shape),
                "a_sf_shape": list(sf_ref.shape),
            }
        )
    case.gate(
        f"bitwise[{label}]: out_bf16 + 每消费者 (a_fp4, a_sf) 与两步路径逐字节相同",
        ok,
        {
            "out_bf16_diff_bytes": bf16_diff,
            "consumers": details,
            "codes": codes,
            "num_warps": num_warps,
        },
    )
    if not ok:
        case.diag(
            f"mismatch[{label}]",
            False,
            {
                "out_bf16_diff_bytes": bf16_diff,
                "consumers": details,
                "hint": "out_bf16 不同 => RMSNorm 算术（reduction 顺序 / rsqrt / fma）；"
                "只有 a_sf 不同 => swizzle 落点或 e4m3 舍入；只有 a_fp4 不同 => "
                "E2M1 cvt 或 nibble 顺序",
            },
        )
    return {"ok": ok, "bf16_diff": bf16_diff, "details": details, "codes": codes}


# ------------------------------------------------------------------ cases


def run_mean_tree(report: OpReport, checkpoint: Checkpoint, device: torch.device) -> None:
    case = report.case(
        "mean-tree",
        reference="x.pow(2).mean(-1) 的 launch 顺序（torch op 逐级复刻）",
        hidden=HIDDEN,
        lanes=512,
        vec=4,
        factor=MEAN_FACTOR,
        sample_layers=(0, 1, 16, 32, 63),
        input_kinds=("gauss-2", "saturation", "zeros", "embed"),
    )
    worst = 0
    samples = 0
    per_kind: list[dict] = []
    for layer_index in (0, 1, 16, 32, 63):
        for kind in ("gauss-2", "saturation", "zeros", "embed"):
            x, _ = input_for(kind, layer_index, checkpoint, device)
            with torch.no_grad():
                reference = x.reshape(HIDDEN).float().pow(2).mean(-1, keepdim=True)
                emulated = emulate_mean_tree(x)
            torch.cuda.synchronize()
            diff = byte_diff(reference.reshape(-1), emulated.reshape(-1))
            worst = max(worst, diff)
            samples += 1
            per_kind.append(
                {"layer": layer_index, "kind": kind, "diff_bytes": diff,
                 "reference": float(reference.item()), "emulated": float(emulated.item())}
            )
    case.gate(
        "mean-tree: 复刻顺序与 torch.mean 逐位相同（5 层 × 4 分布）",
        worst == 0,
        {"samples": samples, "worst_diff_bytes": worst},
    )
    case.gate(
        "coverage: mean-tree 覆盖 5 层 × {gauss-2, saturation, zeros, embed} 共 20 个样本",
        samples == 20,
        {"samples": samples, "kinds": ["gauss-2", "saturation", "zeros", "embed"]},
    )
    case.metric("mean_tree_diff_bytes", float(worst))
    case.metric("mean_tree_samples", float(samples))
    case.diag("逐样本", worst == 0, per_kind)


def run_rsqrt(report: OpReport, checkpoint: Checkpoint, device: torch.device) -> None:
    case = report.case(
        "rsqrt-approx",
        instruction="rsqrt.approx.f32",
        compare="torch.rsqrt",
        sample_layers=(0, 3, 16, 32, 63),
        note="rstd 差 1 ulp 会改变 BF16 rounding，从而改变 code",
    )
    values = []
    with torch.no_grad():
        for layer_index in (0, 3, 16, 32, 63):
            for kind in ("gauss-0.5", "gauss-2", "gauss-8", "embed"):
                x, _ = input_for(kind, layer_index, checkpoint, device)
                var = x.reshape(HIDDEN).float().pow(2).mean(-1, keepdim=True)
                values.append(var + 1e-6)
        # 再补上真实量级附近的稠密网格（覆盖 rsqrt 的近似分段）
        grid = torch.linspace(1e-3, 30.0, 4096, device=device, dtype=torch.float32)
    dense = torch.cat([item.reshape(1) for item in values] + [grid])
    probe = rsqrt_probe(dense)
    reference = torch.rsqrt(dense)
    diff = byte_diff(probe, reference)
    case.gate(
        "rsqrt-approx: inline rsqrt.approx.f32 与 torch.rsqrt 逐位相同",
        diff == 0,
        {"samples": int(dense.numel()), "diff_bytes": diff},
    )
    case.gate(
        "coverage: rsqrt 探针覆盖 5 层 × 4 分布的真实 variance + [1e-3, 30] 的 4096 点网格",
        dense.numel() >= 4096 + 20,
        {"samples": int(dense.numel()), "grid_points": 4096, "real_variance": len(values)},
    )
    case.metric("rsqrt_diff_bytes", float(diff))
    case.metric("rsqrt_samples", float(dense.numel()))
    if diff:
        case.diag(
            "mismatch",
            False,
            {
                "max_ulp_distance": float(
                    (
                        (probe.view(torch.int32).long() - reference.view(torch.int32).long())
                        .abs()
                        .max()
                    ).item()
                )
            },
        )


def run_reference_agreement(
    report: OpReport, checkpoint: Checkpoint, device: torch.device
) -> None:
    case = report.case(
        "reference-agreement",
        left="F.rms_norm(f32 input, f32 weight=1+w) -> .to(bf16)",
        right="Qwen3_5RMSNorm(x bf16)（模型实际执行）",
        layers=(0, 7, 32, 63),
        eps=1e-6,
    )
    worst = 0
    samples = 0
    with torch.no_grad():
        for layer_index in (0, 7, 32, 63):
            for kind in ("gauss-2", "embed", "saturation"):
                x, _ = input_for(kind, layer_index, checkpoint, device)
                for suffix in ("input_layernorm", "post_attention_layernorm"):
                    weight = checkpoint.norm_weights[f"layers.{layer_index}.{suffix}"].to(
                        device
                    )
                    w1 = (1.0 + weight.float()).contiguous()
                    y_function = torch.nn.functional.rms_norm(
                        x.reshape(1, HIDDEN).float(), (HIDDEN,), w1, 1e-6
                    ).to(torch.bfloat16)
                    y_module = model_path_reference(x, weight, 1e-6)
                    diff = byte_diff(y_function, y_module)
                    worst = max(worst, diff)
                    samples += 1
    case.gate(
        "reference-agreement: F.rms_norm 与 Qwen3_5RMSNorm 逐位相同",
        worst == 0,
        {"samples": samples, "worst_diff_bytes": worst},
    )
    case.gate(
        "coverage: 4 层 × 3 种输入 × 2 个 norm = 24 个参考比对",
        samples == 24,
        {"samples": samples, "layers": [0, 7, 32, 63]},
    )
    case.metric("reference_diff_bytes", float(worst))
    case.metric("reference_samples", float(samples))


def run_real_layers(
    report: OpReport, checkpoint: Checkpoint, device: torch.device
) -> None:
    names = checkpoint.norm_names()
    pre_count = sum(1 for name in names if name.endswith("input_layernorm"))
    case = report.case(
        "real-layers",
        norms=len(names),
        sources="checkpoint 的真实 norm 权重 + 各自的真实消费者 input_scale",
        input_kind="gauss-2 + 20x 离群通道（残差流形态）",
        eps=1e-6,
        consumers_per_norm="input norm: 2 (GDN) / 3 (full attention)，post norm: 2",
    )
    norms_ok = 0
    norms_bad: list[str] = []
    total_codes = 0
    consumer_counts = {2: 0, 3: 0}
    with torch.no_grad():
        for name in names:
            layer_index = int(name.split(".")[1])
            pre, post = checkpoint.consumers(layer_index)
            consumers = pre if name.endswith("input_layernorm") else post
            weight = checkpoint.norm_weights[name].to(device)
            w1 = (1.0 + weight.float()).contiguous()
            x, _ = input_for("gauss-2", layer_index, checkpoint, device)
            gs_list = [checkpoint.gs(module) for module in consumers]
            consumer_counts[len(gs_list)] = consumer_counts.get(len(gs_list), 0) + 1
            result = run_case(
                case,
                x,
                w1,
                1e-6,
                gs_list,
                device,
                label=f"{name}[{len(gs_list)} consumers]",
            )
            total_codes += result["codes"]
            if result["ok"]:
                norms_ok += 1
            else:
                norms_bad.append(name)
    case.gate(
        f"coverage: 128 个真实 norm（input {pre_count} / post {len(names) - pre_count}）"
        f"× 各自消费者 ({consumer_counts.get(2, 0)} 个 2 消费者, "
        f"{consumer_counts.get(3, 0)} 个 3 消费者)",
        len(names) == 128 and consumer_counts.get(2) == 112 and consumer_counts.get(3) == 16,
        {
            "norms": len(names),
            "input_norms": pre_count,
            "post_norms": len(names) - pre_count,
            "consumer_counts": {str(k): v for k, v in consumer_counts.items()},
            "expected": {
                "norms": 128,
                "two_consumers": "48 GDN input + 64 post = 112",
                "three_consumers": "16 full-attention input = 16",
                "quantized_consumers": 112 * 2 + 16 * 3,
            },
        },
    )
    case.metric("real_norms", float(len(names)))
    case.metric("real_norms_passed", float(norms_ok))
    case.metric("real_codes", float(total_codes))
    if norms_bad:
        case.diag("failed norms", False, {"names": norms_bad[:20], "count": len(norms_bad)})


def _scale_extreme_records(checkpoint: Checkpoint) -> dict:
    min_module, max_module = checkpoint.scale_extremes()
    return {
        "min": {
            "module": min_module,
            "input_scale": checkpoint.input_scales[min_module],
            "gs": global_divisor(checkpoint.input_scales[min_module]),
        },
        "max": {
            "module": max_module,
            "input_scale": checkpoint.input_scales[max_module],
            "gs": global_divisor(checkpoint.input_scales[max_module]),
        },
        "note": "两端都不在融合消费者集合里（都是非 norm 消费者的 proj），单独一档强制覆盖",
    }


def run_scales_extremes(
    report: OpReport, checkpoint: Checkpoint, device: torch.device
) -> None:
    min_module, max_module = checkpoint.scale_extremes()
    min_scale = checkpoint.input_scales[min_module]
    max_scale = checkpoint.input_scales[max_module]
    case = report.case(
        "scales-extremes",
        min_module=min_module,
        min_input_scale=min_scale,
        min_gs=global_divisor(min_scale),
        max_module=max_module,
        max_input_scale=max_scale,
        max_gs=global_divisor(max_scale),
        note="两端都不在融合消费者里（层内只有 attention/mlp 的 proj），所以单独强制覆盖",
        layers=(0, 32, 63),
        input_kinds=("gauss-2", "embed"),
    )
    combos = [
        ("both-min", (min_scale, min_scale)),
        ("both-max", (max_scale, max_scale)),
        ("min+max", (min_scale, max_scale)),
        ("max+min", (max_scale, min_scale)),
    ]
    ok_all = True
    ran = 0
    for layer_index in (0, 32, 63):
        for kind in ("gauss-2", "embed"):
            x, _ = input_for(kind, layer_index, checkpoint, device)
            for suffix in ("input_layernorm", "post_attention_layernorm"):
                name = f"layers.{layer_index}.{suffix}"
                w1 = (1.0 + checkpoint.norm_weights[name].to(device).float()).contiguous()
                for label, scales in combos:
                    gs_list = [global_divisor(scale) for scale in scales]
                    result = run_case(
                        case,
                        x,
                        w1,
                        1e-6,
                        gs_list,
                        device,
                        label=f"{name}/{kind}/{label}",
                    )
                    ok_all = ok_all and result["ok"]
                    ran += 1
    case.gate(
        "coverage: 全局最小/最大 input_scale × 4 种组合 × 3 层 × 2 个 norm × 2 种输入 = 48 次",
        ran == 48,
        {
            "runs": ran,
            "combos": [list(item) for item in combos],
            "layers": [0, 32, 63],
            "norms_per_layer": 2,
            "inputs": ["gauss-2", "embed"],
        },
    )
    case.metric("scales_extremes_passed", 1.0 if ok_all else 0.0)
    case.metric("min_gs", global_divisor(min_scale))
    case.metric("max_gs", global_divisor(max_scale))


def run_distribution_edge(
    report: OpReport, checkpoint: Checkpoint, device: torch.device
) -> None:
    case = report.case(
        "distribution-edge",
        kinds=("zeros", "saturation", "embed", "gauss-0.5", "gauss-2", "gauss-8"),
        layers=(0, 3, 32, 63),
        eps=1e-6,
        note="zeros = 全零 padding 行（amax==0 分支）；saturation = sf_pre 冲到 e4m3 448",
    )
    ok_all = True
    ran = 0
    with torch.no_grad():
        for layer_index in (0, 3, 32, 63):
            pre, post = checkpoint.consumers(layer_index)
            for kind in ("zeros", "saturation", "embed", "gauss-0.5", "gauss-2", "gauss-8"):
                x, _ = input_for(kind, layer_index, checkpoint, device)
                for suffix, consumers in (
                    ("input_layernorm", pre),
                    ("post_attention_layernorm", post),
                ):
                    name = f"layers.{layer_index}.{suffix}"
                    w1 = (
                        1.0 + checkpoint.norm_weights[name].to(device).float()
                    ).contiguous()
                    gs_list = [checkpoint.gs(module) for module in consumers]
                    result = run_case(
                        case, x, w1, 1e-6, gs_list, device, label=f"{name}/{kind}"
                    )
                    ok_all = ok_all and result["ok"]
                    ran += 1
    case.gate(
        "coverage: 4 层 × 2 个 norm × 6 种输入（zeros/saturation/embed/3 个 RMS 量级）",
        ran == 48,
        {"samples": ran, "kinds": ["zeros", "saturation", "embed", "gauss-0.5", "gauss-2", "gauss-8"]},
    )
    case.gate(
        "amax==0: 全零输入确实走到 sf=0 / code=0 分支",
        _zeros_case_is_zero(checkpoint, device),
        {"zeros_input": "x = 0 -> variance = 0 -> rstd = rsqrt(eps) -> y = 0 -> amax = 0"},
    )
    case.metric("distribution_edge_passed", 1.0 if ok_all else 0.0)
    case.metric("distribution_edge_samples", float(ran))


def _zeros_case_is_zero(checkpoint: Checkpoint, device: torch.device) -> bool:
    name = "layers.0.input_layernorm"
    w1 = (1.0 + checkpoint.norm_weights[name].to(device).float()).contiguous()
    x = zero_input().to(device)
    out, a_fp4, a_sf = _scratch(device, 2)
    gs_list = [checkpoint.gs("layers.0.linear_attn.in_proj_qkv"),
               checkpoint.gs("layers.0.linear_attn.in_proj_z")]
    with torch.no_grad():
        norm_quant_fused(x, w1, 1e-6, gs_list, out, a_fp4, a_sf)
        torch.cuda.synchronize()
    return (
        int(torch.count_nonzero(out).item()) == 0
        and int(torch.count_nonzero(a_fp4[0]).item()) == 0
        and int(torch.count_nonzero(a_sf[0]).item()) == 0
    )


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
    count = len(times)
    return {"median_ms": times[count // 2], "min_ms": times[0]}


def graph_per_call(fn, *, warmup: int, iters: int, repeats: int) -> dict[str, float] | None:
    """把 ``iters`` 次调用录进一张 CUDA graph，replay 多次取 per-call 中位数。

    模型实际跑在 graph 里，单 kernel 成本是 GPU 侧 dispatch；这张图是更接近的口径。
    capture 失败（某些算子在 capture 下不可用）返回 None，由调用方记 diagnostic。
    """
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


def run_micro(report: OpReport, checkpoint: Checkpoint, device: torch.device) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    # 用第 3 层（full_attention）：input norm 有 3 个消费者（q/k/v），post norm 有 2 个
    # （gate/up），一次覆盖两种形态。
    layer_index = 3
    pre, post = checkpoint.consumers(layer_index)
    if len(pre) != 3:
        raise SystemExit(f"layer {layer_index} 不是 3 消费者的 full attention 层")
    post_name = f"layers.{layer_index}.post_attention_layernorm"
    weight2 = checkpoint.norm_weights[post_name].to(device)
    w1_2 = (1.0 + weight2.float()).contiguous()
    norm2 = Qwen3_5RMSNorm(HIDDEN, eps=1e-6)
    norm2.weight.data.copy_(weight2)
    norm2 = norm2.to(device)
    norm2.eval()
    pre_name = f"layers.{layer_index}.input_layernorm"
    w1_3 = (
        1.0 + checkpoint.norm_weights[pre_name].to(device).float()
    ).contiguous()
    x, _ = input_for("gauss-2", layer_index, checkpoint, device)
    gs2 = [checkpoint.gs(post[0]), checkpoint.gs(post[1])]
    gs3 = [
        checkpoint.gs(pre[0]),
        checkpoint.gs(pre[1]),
        checkpoint.gs(pre[2]),
    ]
    out2, a2, s2 = _scratch(device, 2)
    out3, a3, s3 = _scratch(device, 3)
    SfLayout, nvfp4_quantize = _flashinfer()
    gs2_t = [torch.tensor([g], dtype=torch.float32, device=device) for g in gs2]
    gs3_t = [torch.tensor([g], dtype=torch.float32, device=device) for g in gs3]

    def two_step_2():
        y = norm2(x)
        for gs_t in gs2_t:
            nvfp4_quantize(
                y.reshape(1, HIDDEN),
                gs_t,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
                backend="cuda",
            )

    def two_step_3():
        y = norm2(x)
        for gs_t in gs3_t:
            nvfp4_quantize(
                y.reshape(1, HIDDEN),
                gs_t,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
                backend="cuda",
            )

    fused_times = {}
    for warps in NUM_WARPS_SWEEP:
        fused_times[warps] = time_per_call(
            lambda warps=warps: norm_quant_fused(
                x, w1_2, 1e-6, gs2, out2, a2, s2, num_warps=warps
            ),
            warmup=BENCH_WARMUP,
            iters=BENCH_ITERS,
        )

    def fused_2():
        norm_quant_fused(x, w1_2, 1e-6, gs2, out2, a2, s2)

    def fused_3():
        norm_quant_fused(x, w1_3, 1e-6, gs3, out3, a3, s3)

    eager = {
        "two_step_2": time_per_call(two_step_2, warmup=BENCH_WARMUP, iters=BENCH_ITERS),
        "fused_2": time_per_call(fused_2, warmup=BENCH_WARMUP, iters=BENCH_ITERS),
        "two_step_3": time_per_call(two_step_3, warmup=BENCH_WARMUP, iters=BENCH_ITERS),
        "fused_3": time_per_call(fused_3, warmup=BENCH_WARMUP, iters=BENCH_ITERS),
    }
    graph_two_2 = graph_per_call(
        two_step_2, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, repeats=GRAPH_REPEATS
    )
    graph_two_3 = graph_per_call(
        two_step_3, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, repeats=GRAPH_REPEATS
    )
    graph_fused_2 = graph_per_call(
        fused_2, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, repeats=GRAPH_REPEATS
    )
    graph_fused_3 = graph_per_call(
        fused_3, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, repeats=GRAPH_REPEATS
    )

    # 真实分布：48 个 GDN 层（input 2 消费者 + post 2 消费者）+ 16 个 full attention
    # 层（input 3 消费者 + post 2 消费者）。
    def step_estimate(two2, fused2, two3, fused3, mode: str) -> dict[str, float]:
        saved_2 = two2 - fused2
        saved_3 = two3 - fused3
        step = 112 * saved_2 + 16 * saved_3
        return {
            f"{mode}_saved_per_2consumer_ms": saved_2,
            f"{mode}_saved_per_3consumer_ms": saved_3,
            f"{mode}_est_step_saved_ms": step,
            f"{mode}_step_ratio": (112 * fused2 + 16 * fused3) / (112 * two2 + 16 * two3),
        }

    estimates = step_estimate(
        eager["two_step_2"]["median_ms"], eager["fused_2"]["median_ms"],
        eager["two_step_3"]["median_ms"], eager["fused_3"]["median_ms"], "eager",
    )
    if all(item is not None for item in (graph_two_2, graph_fused_2, graph_two_3, graph_fused_3)):
        estimates.update(
            step_estimate(
                graph_two_2["median_ms"], graph_fused_2["median_ms"],
                graph_two_3["median_ms"], graph_fused_3["median_ms"], "graph",
            )
        )
    case = report.case(
        "micro",
        layer=layer_index,
        consumers_2=post,
        consumers_3=pre,
        warmup=BENCH_WARMUP,
        iters=BENCH_ITERS,
        graph_iters=GRAPH_ITERS,
        graph_repeats=GRAPH_REPEATS,
        num_warps_sweep=list(NUM_WARPS_SWEEP),
        timing="per-call CUDA events median（eager）；graph = 8 次调用录一张图后 replay / 8",
        step_model="112 次 2 消费者 + 16 次 3 消费者（48 GDN + 16 full attention）",
    )
    numeric = run_case(
        case, x, w1_2, 1e-6, gs2, device, label="micro-2consumers"
    )
    numeric3 = run_case(
        case, x, w1_3, 1e-6, gs3, device, label="micro-3consumers"
    )
    case.gate(
        "coverage: 2 消费者与 3 消费者两种形态都测，且 num_warps 扫 2/4/8 三档",
        len(NUM_WARPS_SWEEP) == 3 and numeric["ok"] and numeric3["ok"],
        {
            "num_warps_sweep": list(NUM_WARPS_SWEEP),
            "two_consumers_bitwise": numeric["ok"],
            "three_consumers_bitwise": numeric3["ok"],
        },
    )
    case.gate(
        "timing: eager 与 (可 capture 时) graph 的中位数都是有限正数",
        all(
            item["median_ms"] > 0.0 and math.isfinite(item["median_ms"])
            for item in eager.values()
        ),
        {"eager": {key: value["median_ms"] for key, value in eager.items()}},
    )
    case.diag(
        "graph 口径",
        all(item is not None for item in (graph_two_2, graph_fused_2, graph_two_3, graph_fused_3)),
        {
            "two_step_2": graph_two_2,
            "fused_2": graph_fused_2,
            "two_step_3": graph_two_3,
            "fused_3": graph_fused_3,
        },
    )
    case.diag(
        "num_warps 扫描（2 消费者，eager）",
        True,
        {str(warps): value for warps, value in fused_times.items()},
    )

    for key, value in eager.items():
        case.metric(f"eager_{key}_ms", value["median_ms"])
    for warps, value in fused_times.items():
        case.metric(f"fused_warps{warps}_ms", value["median_ms"])
    for key, value in estimates.items():
        case.metric(key, value)
    if graph_fused_2 is not None:
        case.metric("graph_two_step_2_ms", graph_two_2["median_ms"])
        case.metric("graph_fused_2_ms", graph_fused_2["median_ms"])
    if graph_fused_3 is not None:
        case.metric("graph_two_step_3_ms", graph_two_3["median_ms"])
        case.metric("graph_fused_3_ms", graph_fused_3["median_ms"])
    case.metric("speedup_2consumers_eager", eager["two_step_2"]["median_ms"] / eager["fused_2"]["median_ms"])


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
    metrics = {entry["case"]: entry["metrics"] for entry in payload["cases"]}
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
    torch.manual_seed(SEED)
    selected = select_cases(args.only)
    checkpoint = load_checkpoint()
    report = OpReport(
        args.out,
        CANDIDATE_NAME,
        CANDIDATE_FORMAT,
        candidate_extra={
            "kernel": "runner/ops/nvfp4/norm_quant.py::_norm_quant_kernel",
            "patch": "runner/model/norm_quant_patch.py (Qwen3_5DecoderLayer.forward, decode only)",
            "quant_rule": "R2（任务 E 冻结）：PTX 算术序列 + rcp.approx.ftz",
            "rmsnorm_rule": "PyTorch mean(-1) 的 launch 顺序 + rsqrt.approx.f32 + BF16 rounding",
            "consumers": "2 (GDN in_proj_qkv+z / gate+up) 或 3 (full attention q/k/v)",
            "hidden": HIDDEN,
            "num_warps": DEFAULT_NUM_WARPS,
            "c6_bits": hex(0x3E2AAAAB),
            "c6": C6,
            "mean_factor": MEAN_FACTOR,
            "gate": "bitwise（逐字节），不用容差",
        },
    )
    handler = {
        "mean-tree": run_mean_tree,
        "rsqrt-approx": run_rsqrt,
        "reference-agreement": run_reference_agreement,
        "real-layers": run_real_layers,
        "scales-extremes": run_scales_extremes,
        "distribution-edge": run_distribution_edge,
        "micro": run_micro,
    }
    for name in selected:
        handler[name](report, checkpoint, device)
    payload = report.finish(
        extra={
            "coverage": {
                "real_norms": 128,
                "real_consumers": 272,
                "input_scale_extremes": _scale_extreme_records(checkpoint),
                "edge_cases": ["zeros(amax==0)", "saturation", "embed", "outliers"],
            },
            "model_dir": str(MODEL_DIR),
        }
    )
    print(json.dumps(summarize(payload), indent=2, ensure_ascii=False))
    print(f"output: {args.out / 'op.json'}")
    return 0 if payload["summary"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
