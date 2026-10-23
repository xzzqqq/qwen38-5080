#!/usr/bin/env python3
"""Phase 2 任务 H：RMSNorm + NVFP4 activation quant 融合的 op gate + microbench。

候选：``runner/ops/nvfp4/norm_quant.py`` 的单个 Triton kernel —— 一次 launch 读
pre-norm BF16 hidden，算 zero-centered RMSNorm（结果 round 回 BF16），并为该 norm 的
每个 NVFP4 消费者各产一份 ``(a_fp4, a_sf)``（128x4 swizzled）。

决定性参考 = **模型实际执行的那条路径**，判定逐字节相同（不是容差）：

    y = Qwen3_5RMSNorm.forward(x)          # bf16 权重，zero-centered (1 + w)，fp32 里算完再 cast
    a_fp4, a_sf = nvfp4_quantize(y, 1 / input_scale, layout_128x4, do_shuffle=False)

为什么不用 ``F.rms_norm`` 当 gate（上一轮 job-000036 的教训）：``aten::rms_norm`` 是
composite，但 composite 里的幂是 ``at::pow(input, 2.0)``（double 字面量，走 CUDA 的
``powf``），而 ``Qwen3_5RMSNorm._norm`` 写的是 ``x.pow(2)``（int 字面量）。两者在极端
量级（例如 3e4 的离群通道）上会差 1 ulp，CPU 上测不出来。job-000036 的 24 个样本里
``F.rms_norm`` 与模块差 1 个 bf16 字节，于是把「两个参考实现不同」误判成了候选失败。
所以 ``F.rms_norm`` 现在只作 diagnostic，并在 ``square-semantics`` 里把
「int pow / double pow / x*x」三者的逐位差异直接测出来存档。

冻结的 gate（实现候选之前写下，不因结果调整）
================================================
1. ``bitwise[...]``：融合 kernel 的 ``out_bf16``、每个消费者的 ``a_fp4`` 字节、``a_sf``
   字节都与模块路径 + ``nvfp4_quantize`` 逐字节相同。三条都要过：
   - ``out_bf16`` 也必须逐位相同，因为 linear-attention 层的 ``in_proj_a`` /
     ``in_proj_b``（未量化）直接消费这个 BF16 张量；
   - ``a_fp4`` 覆盖 E2M1 code 与 nibble 顺序；``a_sf`` 覆盖 e4m3 scale 与 128x4
     swizzle 落点（含 padding 必须为 0）。
2. ``mean-tree``：``x.pow(2).mean(-1)`` 的 reduction 顺序用 torch op 逐级复刻后与
   torch 自身逐位相同。把「对 launch 配置的读法」和「Triton 实现」分开定位。
3. ``rsqrt-approx``：inline PTX ``rsqrt.approx.f32`` 与 ``torch.rsqrt`` 在所有真实
   variance（含 saturation 的 ~1.8e5）与 logspace(1e-6, 1e9) 的 8192 点网格上逐位相同。

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

microbench：分别测「norm + 2 消费者」「norm + 3 消费者」的两步路径（模块 + 每消费者
一次 ``nvfp4_quantize``）与融合路径的 per-call ms（CUDA event 中位数；eager 提交区间
+ 能 capture 时的 graph replay 区间），再按真实的 48/16/64 层分布折算整步节省。graph
数字更接近模型实际执行（模型跑在 CUDA graph 里），eager 数字包含 CPU 提交开销，只作诊断。

任务 M（生产者侧 elementwise + quant，脚本名不变）
=================================================
在 H 的 6 个 case（``mean-tree`` / ``rsqrt-approx`` / ``real-layers`` /
``scales-extremes`` / ``distribution-edge`` / ``micro``，**保持不动**）之后追加：

- ``downproj-silu-mul``：``silu(gate_proj_out) * up_proj_out`` + quant，与
  「``ACT2FN["silu"]`` -> ``*`` -> ``nvfp4_quantize``」逐字节相同；
- ``oproj-gate``：``attn_output * sigmoid(gate)`` + quant，同上；
- ``lmhead-final-norm``：final norm（1 消费者形态的 ``norm_quant_fused``）+ quant；
- ``fuse-micro``：三档 per-call 计时与 ``64 down_proj + 16 o_proj + 1 lm_head`` 的
  整步估计（以及把 48 个 out_proj 按同 per-consumer 收益外推的 129 估计）。
  **命名说明**：既有的 ``micro`` 是任务 H 的 norm 口径，不能改；任务 M 的 micro
  tier 落在 ``fuse-micro``；
- ``producer-wiring``：走 ``runner/model/quant_fuse_patch.py`` 的真实接线（假模型 +
  真实第 3 层权重 + 真实 final norm），判定 fused 执行 / bitwise / prefill 回退 /
  ``QUANT_FUSE=0``。

三个数值 case 都用真实 checkpoint 权重造输入（第 3 层 gate/up、q/v、final norm），
gs 覆盖该消费者类型的**全部真实 input_scale** 加全局 min/max；每个 case 带一个负对照
（silu→relu、sigmoid→1、final norm 用 w 而不是 1+w），错实现必须显著超差。

整份脚本在 GPU 上约 3–5 分钟（含 Triton / FlashInfer 首次编译与真实权重加载）。

用法：
    .venv/bin/python scripts/phase2_normquant_check.py --out {output}/op
    .venv/bin/python scripts/phase2_normquant_check.py --only mean-tree micro --out ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from runner.ops.nvfp4.quant_fuse import (  # noqa: E402
    KINDS,
    allocate_scratch as allocate_producer_scratch,
    producer_quant_fused,
)

MODEL_DIR = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"
CKPT_PREFIX = "model.language_model."
EPS = 1e-6

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
    "real-layers",
    "scales-extremes",
    "distribution-edge",
    "micro",
    # 任务 M：生产者侧（silu·mul / attn·sigmoid / final norm）的 elementwise + quant
    "downproj-silu-mul",
    "oproj-gate",
    "lmhead-final-norm",
    "fuse-micro",
    "producer-wiring",
)

CANDIDATE_NAME = "norm-quant+producer-fuse"
CANDIDATE_FORMAT = "nvfp4-w4a4"

INPUT_KINDS = ("gauss-0.5", "gauss-2", "gauss-8", "embed", "saturation", "zeros")


# ------------------------------------------------------------------ checkpoint


@dataclass
class Checkpoint:
    layer_types: list[str]
    norm_weights: dict[str, torch.Tensor]  # short name -> bf16 [K] CPU
    input_scales: dict[str, float]  # short module name -> float
    embedding_rows: torch.Tensor  # bf16 [n, K] CPU
    # 任务 M 用：final norm 权重与 text config（K/head 数等）
    final_norm_weight: torch.Tensor | None = None
    text_config: dict | None = None

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

    def scales_of(self, suffix: str) -> dict[str, float]:
        """按模块名后缀取真实 input_scale（如 ``.mlp.down_proj``），返回有序 dict。"""
        return {
            name: scale
            for name, scale in sorted(self.input_scales.items())
            if name.endswith(suffix)
        }

    def gs_with_extremes(self, suffix: str) -> list[float]:
        """该消费者类型的全部真实 gs + 全局最小/最大 gs（去重，保持稳定顺序）。"""
        min_module, max_module = self.scale_extremes()
        values = [self.gs(name) for name in self.scales_of(suffix)]
        for module in (min_module, max_module):
            gs = self.gs(module)
            if gs not in values:
                values.append(gs)
        return values


def load_checkpoint() -> Checkpoint:
    import json as _json

    config = _json.loads((MODEL_DIR / "config.json").read_text())["text_config"]
    layer_types = list(config["layer_types"])
    hidden = int(config["hidden_size"])
    if hidden != HIDDEN:
        raise SystemExit(f"checkpoint hidden {hidden} != kernel {HIDDEN}")

    norm_weights: dict[str, torch.Tensor] = {}
    input_scales: dict[str, float] = {}
    final_norm_weight = None
    with CheckpointReader(MODEL_DIR) as reader:
        for layer_index in range(len(layer_types)):
            for suffix in ("input_layernorm", "post_attention_layernorm"):
                short = f"layers.{layer_index}.{suffix}"
                norm_weights[short] = reader.get(
                    CKPT_PREFIX + short + ".weight", "cpu"
                ).to(torch.bfloat16)
        for name in reader.weight_map:
            if not name.endswith(".input_scale") or not name.startswith(CKPT_PREFIX):
                continue
            short = name[len(CKPT_PREFIX) : -len(".input_scale")]
            input_scales[short] = float(reader.get(name, "cpu").float().item())
        final_norm_weight = reader.get(
            CKPT_PREFIX + "norm.weight", "cpu"
        ).to(torch.bfloat16)
        # lm_head 不在 model.language_model. 前缀下，单独读；它的 input_scale（0.0185）
        # 落在 H 的全局 min/max 之内，不会改变既有的 scales-extremes case。
        input_scales["lm_head"] = float(
            reader.get("lm_head.input_scale", "cpu").float().item()
        )
        embed_name = CKPT_PREFIX + "embed_tokens.weight"
        shard = reader.weight_map[embed_name]
        handle = reader._handle(shard, "cpu")  # noqa: SLF001 - 只读切片，避免整表 2.5 GB
        start = EMBED_ROWS[0]
        stop = EMBED_ROWS[-1] + 1
        embedding_rows = handle.get_slice(embed_name)[start:stop].float()
    return Checkpoint(
        layer_types,
        norm_weights,
        input_scales,
        embedding_rows,
        final_norm_weight=final_norm_weight,
        text_config=config,
    )


# ------------------------------------------------------------------ 参考路径


class NormReference:
    """模型实际执行的 ``Qwen3_5RMSNorm``（bf16 权重 + zero-centered ``1 + w``）。

    权重保持 BF16（与 checkpoint / 模型一致），所以模块里的 ``1.0 + self.weight.float()``
    与模型跑的是同一条指令序列；``w1`` 只是把同一个乘积预先算好，值逐位相同。
    """

    def __init__(self, weight_bf16: torch.Tensor, eps: float, device: torch.device) -> None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

        module = Qwen3_5RMSNorm(HIDDEN, eps=float(eps))
        module.weight = torch.nn.Parameter(
            weight_bf16.detach().to(device).to(torch.bfloat16), requires_grad=False
        )
        self.module = module.to(device).eval()
        self.eps = float(eps)
        self.w1 = (1.0 + weight_bf16.detach().to(device).float()).contiguous()

    def bf16(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module(x)

    def f32(self, x: torch.Tensor) -> torch.Tensor:
        """forward 里 cast 之前的 FP32 值（同一串 op，只少了最后一次 ``type_as``）。"""
        with torch.no_grad():
            out = self.module._norm(x.float())
            return out * (1.0 + self.module.weight.float())


class NormReferences:
    """按 norm 名缓存 ``NormReference``（128 个，每个 20 KB，代价可忽略）。"""

    def __init__(self, checkpoint: Checkpoint, device: torch.device) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self._cache: dict[str, NormReference] = {}

    def get(self, name: str) -> NormReference:
        if name not in self._cache:
            self._cache[name] = NormReference(
                self.checkpoint.norm_weights[name], EPS, self.device
            )
        return self._cache[name]

    def for_layer(self, layer_index: int, which: str) -> NormReference:
        return self.get(norm_ref_name(layer_index, which))


def norm_ref_name(layer_index: int, which: str) -> str:
    return f"layers.{layer_index}.{which}_layernorm"


def _flashinfer():
    from flashinfer import SfLayout, nvfp4_quantize

    return SfLayout, nvfp4_quantize


def nvfp4_quantize_bytewise(
    y_bf16: torch.Tensor,
    gs_list: list[float],
    device: torch.device,
    *,
    k: int = HIDDEN,
):
    """模型里的 ``nvfp4_quantize``（per-tensor global divisor，128x4 swizzled，不 shuffle）。"""
    SfLayout, nvfp4_quantize = _flashinfer()
    outputs = []
    for gs in gs_list:
        gs_t = torch.tensor([gs], dtype=torch.float32, device=device)
        a_fp4, a_sf = nvfp4_quantize(
            y_bf16.reshape(1, int(k)),
            gs_t,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )
        outputs.append((a_fp4, a_sf))
    return outputs


def reference_two_step(
    x: torch.Tensor, norm_ref: NormReference, gs_list: list[float], device: torch.device
):
    """现有两步路径：``Qwen3_5RMSNorm.forward``（模块，bf16 权重）-> 每消费者一次 quantize。"""
    y = norm_ref.bf16(x)
    return y, nvfp4_quantize_bytewise(y, gs_list, device)


def reference_rms_norm_function(
    x: torch.Tensor, norm_ref: NormReference, *, flat: bool = False
) -> torch.Tensor:
    """``F.rms_norm``（**只作 diagnostic**）。

    ``aten::rms_norm`` 的 composite 与 ``Qwen3_5RMSNorm._norm`` 是同一串 aten op
    （dispatch trace 里唯一差别是 composite 用 in-place ``add_``、且 ``mean`` 的
    reduce dim 写成 ``[1]`` 而不是 ``[-1]``），所以差异只可能来自**输入 shape**：
    ``flat=True`` 传 ``[1, K]``、``flat=False`` 传模型真实的 ``[1, 1, K]``。
    """
    source = x.reshape(1, HIDDEN).float() if flat else x.float()
    return torch.nn.functional.rms_norm(
        source, (HIDDEN,), norm_ref.w1, norm_ref.eps
    ).to(torch.bfloat16)


def mean_shape_probe(x: torch.Tensor) -> dict:
    """同一个 ``mean(-1)`` 在 ``[1,1,K]`` / ``[1,K]`` / ``[K]`` 三种 shape 下是否逐位相同。"""
    xf = x.float()
    m3 = xf.pow(2).mean(-1, keepdim=True)
    m2 = xf.reshape(1, HIDDEN).pow(2).mean(-1, keepdim=True)
    m1 = xf.reshape(HIDDEN).pow(2).mean(-1, keepdim=True)
    return {
        "mean_3d_vs_2d_diff_bytes": byte_diff(m3.reshape(-1), m2.reshape(-1)),
        "mean_3d_vs_1d_diff_bytes": byte_diff(m3.reshape(-1), m1.reshape(-1)),
        "mean_3d": float(m3.item()),
    }


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


# ------------------------------------------------------------------ 比较工具


def byte_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """两个 tensor 的字节差异数（按扁平字节比，rank 差异不影响）；长度不同返回 -1。"""
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


def _bf16_bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int16).reshape(-1).long() & 0xFFFF


def _f32_ulp(value: float) -> float:
    """value 处的 f32 ulp（用 nextafter 量）。"""
    t = torch.tensor([value], dtype=torch.float32)
    return float((torch.nextafter(t, torch.tensor([math.inf])) - t).item())


def _bf16_midpoint_distance(value: float) -> float:
    """value 到最近的 bf16 舍入边界（中点）的距离，单位：value 处的 f32 ulp。

    < 1 说明差异只能来自「上游 rstd 差 1 ulp 把值推过 boundary」这种边界情形。
    """
    v = torch.tensor([value], dtype=torch.float32)
    b = v.to(torch.bfloat16)
    bf = b.float()
    step = float(
        (
            torch.nextafter(b, torch.tensor([math.inf], dtype=torch.bfloat16)).float() - bf
        ).item()
    )
    if step <= 0.0:
        return float("inf")
    midpoint = float(bf.item()) + (step / 2.0 if value >= float(bf.item()) else -step / 2.0)
    ulp = _f32_ulp(value)
    if ulp <= 0.0:
        return float("inf")
    return abs(value - midpoint) / ulp


def _mismatch_detail(
    x: torch.Tensor,
    out: torch.Tensor,
    y_ref: torch.Tensor,
    norm_ref: NormReference,
    a_fp4: list[torch.Tensor],
    a_sf: list[torch.Tensor],
    refs,
    device: torch.device,
    limit: int = 8,
) -> dict:
    """失败时的定位信息：哪个元素、两边 bit、到 bf16 边界的距离、两个参考各自差多少。"""
    detail: dict = {}
    try:
        fused_bits = _bf16_bits(out)
        ref_bits = _bf16_bits(y_ref)
        diff_bits = (fused_bits != ref_bits).nonzero().flatten()
        detail["out_bf16_mismatch_count"] = int(diff_bits.numel())
        y_f32 = norm_ref.f32(x).reshape(-1)
        y_function = reference_rms_norm_function(x, norm_ref)
        variance = (
            x.reshape(HIDDEN).float().pow(2).mean(-1, keepdim=True) + norm_ref.eps
        )
        probe = rsqrt_probe(variance.reshape(1))
        torch_rsqrt = torch.rsqrt(variance.reshape(1))
        detail["variance"] = float(variance.item())
        detail["rsqrt_probe_bits"] = hex(int(probe.view(torch.int32)[0].item()))
        detail["rsqrt_torch_bits"] = hex(int(torch_rsqrt.view(torch.int32)[0].item()))
        detail["rsqrt_match"] = int(probe.view(torch.int32)[0].item()) == int(
            torch_rsqrt.view(torch.int32)[0].item()
        )
        detail["function_vs_module_diff_bytes"] = byte_diff(y_function, y_ref)
        xf = x.reshape(HIDDEN).float()
        detail["pow_int_vs_mul"] = int((xf.pow(2) != xf * xf).sum().item())
        detail["pow_double_vs_mul"] = int((torch.pow(xf, 2.0) != xf * xf).sum().item())
        records = []
        for index in diff_bits.tolist()[:limit]:
            value = float(y_f32[index].item())
            records.append(
                {
                    "index": int(index),
                    "fused_bits": hex(int(fused_bits[index])),
                    "ref_bits": hex(int(ref_bits[index])),
                    "ref_f32": value,
                    "ref_f32_hex": hex(
                        int(y_f32[index].view(torch.int32).item()) & 0xFFFFFFFF
                    ),
                    "f32_ulp": _f32_ulp(value),
                    "distance_to_bf16_midpoint_ulps": _bf16_midpoint_distance(value),
                    "function_bits": hex(int(_bf16_bits(y_function)[index])),
                }
            )
        detail["out_bf16_samples"] = records
        for slot, (a_ref, sf_ref) in enumerate(refs):
            a_diff = (
                a_fp4[slot].contiguous().view(torch.uint8).reshape(-1)
                != a_ref.contiguous().view(torch.uint8).reshape(-1)
            ).nonzero().flatten()
            sf_diff = (
                a_sf[slot].contiguous().view(torch.uint8).reshape(-1)
                != sf_ref.contiguous().view(torch.uint8).reshape(-1)
            ).nonzero().flatten()
            detail[f"consumer{slot}_a_fp4_indices"] = a_diff.tolist()[:limit]
            detail[f"consumer{slot}_a_sf_indices"] = sf_diff.tolist()[:limit]
    except Exception as exc:  # noqa: BLE001 - 诊断不能把 job 弄崩
        detail["diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    return detail


def run_case(
    case,
    x: torch.Tensor,
    norm_ref: NormReference,
    gs_list: list[float],
    device: torch.device,
    *,
    label: str,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> dict:
    """一次融合 vs 模块两步路径的逐字节比较 + gate。"""
    out, a_fp4, a_sf = _scratch(device, len(gs_list))
    with torch.no_grad():
        norm_quant_fused(
            x, norm_ref.w1, norm_ref.eps, gs_list, out, a_fp4, a_sf, num_warps=num_warps
        )
        y_ref, refs = reference_two_step(x, norm_ref, gs_list, device)
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
        f"bitwise[{label}]: out_bf16 + 每消费者 (a_fp4, a_sf) 与模块两步路径逐字节相同",
        ok,
        {
            "reference": "Qwen3_5RMSNorm.forward -> nvfp4_quantize",
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
            _mismatch_detail(x, out, y_ref, norm_ref, a_fp4, a_sf, refs, device),
        )
    return {"ok": ok, "bf16_diff": bf16_diff, "details": details, "codes": codes}


# ------------------------------------------------------------------ cases


def run_mean_tree(report: OpReport, checkpoint: Checkpoint, device: torch.device) -> None:
    case = report.case(
        "mean-tree",
        reference="x.pow(2).mean(-1) 的 launch 顺序（torch op 逐级复刻，输入是模型真实 [1,1,K]）",
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
                # 用模型真实的 [1,1,K] shape（不是 reshape 成 1-D），这样这条 gate
                # 直接覆盖模型实际走的 reduce launch 配置
                reference = x.float().pow(2).mean(-1, keepdim=True)
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
        sample_layers=(0, 3, 7, 16, 32, 63),
        input_kinds=INPUT_KINDS,
        grid="logspace(1e-6, 1e9, 8192)",
        note="rstd 差 1 ulp 会改变 BF16 rounding，从而改变 code",
    )
    values = []
    with torch.no_grad():
        for layer_index in (0, 3, 7, 16, 32, 63):
            for kind in INPUT_KINDS:
                x, _ = input_for(kind, layer_index, checkpoint, device)
                var = x.reshape(HIDDEN).float().pow(2).mean(-1, keepdim=True)
                values.append(var + EPS)
        # 对数网格覆盖 1e-6 ~ 1e9（saturation 的真实 variance ≈ 1.8e5 必须在内）
        grid = torch.logspace(-6.0, 9.0, 8192, device=device, dtype=torch.float32)
    dense = torch.cat([item.reshape(1) for item in values] + [grid])
    probe = rsqrt_probe(dense)
    reference = torch.rsqrt(dense)
    diff = byte_diff(probe, reference)
    case.gate(
        "rsqrt-approx: inline rsqrt.approx.f32 与 torch.rsqrt 逐位相同",
        diff == 0,
        {
            "samples": int(dense.numel()),
            "diff_bytes": diff,
            "min": float(dense.min().item()),
            "max": float(dense.max().item()),
        },
    )
    case.gate(
        "coverage: rsqrt 探针覆盖 6 层 × 6 种输入的真实 variance + logspace(1e-6,1e9) 8192 点",
        dense.numel() == 8192 + len(values) and float(dense.max().item()) > 1e5,
        {
            "samples": int(dense.numel()),
            "grid_points": 8192,
            "real_variance": len(values),
            "max_variance": float(dense.max().item()),
        },
    )
    case.metric("rsqrt_diff_bytes", float(diff))
    case.metric("rsqrt_samples", float(dense.numel()))
    case.metric("rsqrt_max_variance", float(dense.max().item()))
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


def run_real_layers(
    report: OpReport,
    checkpoint: Checkpoint,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    names = checkpoint.norm_names()
    pre_count = sum(1 for name in names if name.endswith("input_layernorm"))
    case = report.case(
        "real-layers",
        norms=len(names),
        reference="Qwen3_5RMSNorm.forward（模型实际执行，bf16 权重）-> nvfp4_quantize",
        sources="checkpoint 的真实 norm 权重 + 各自的真实消费者 input_scale",
        input_kind="gauss-2 + 20x 离群通道（残差流形态）",
        eps=EPS,
        consumers_per_norm="input norm: 2 (GDN) / 3 (full attention)，post norm: 2",
    )
    norms_ok = 0
    norms_bad: list[str] = []
    total_codes = 0
    consumer_counts = {2: 0, 3: 0}
    function_diff_worst = 0
    square_stats = {"pow_int_vs_mul": 0, "pow_double_vs_mul": 0, "samples": 0}
    with torch.no_grad():
        for name in names:
            layer_index = int(name.split(".")[1])
            pre, post = checkpoint.consumers(layer_index)
            consumers = pre if name.endswith("input_layernorm") else post
            norm_ref = norm_refs.get(name)
            x, _ = input_for("gauss-2", layer_index, checkpoint, device)
            gs_list = [checkpoint.gs(module) for module in consumers]
            consumer_counts[len(gs_list)] = consumer_counts.get(len(gs_list), 0) + 1
            result = run_case(
                case,
                x,
                norm_ref,
                gs_list,
                device,
                label=f"{name}[{len(gs_list)} consumers]",
            )
            total_codes += result["codes"]
            if result["ok"]:
                norms_ok += 1
            else:
                norms_bad.append(name)
        # F.rms_norm 只作 diagnostic：逐样本记录它与模块路径的差异
        function_records = []
        shape_records = []
        for layer_index in (0, 3, 7, 32, 63):
            for kind in ("gauss-2", "embed", "saturation"):
                x, _ = input_for(kind, layer_index, checkpoint, device)
                xf = x.reshape(HIDDEN).float()
                square_stats["pow_int_vs_mul"] += int((xf.pow(2) != xf * xf).sum().item())
                square_stats["pow_double_vs_mul"] += int(
                    (torch.pow(xf, 2.0) != xf * xf).sum().item()
                )
                square_stats["samples"] += 1
                if kind == "saturation":
                    probe = mean_shape_probe(x)
                    shape_records.append({"layer": layer_index, "kind": kind, **probe})
                for which in ("input", "post_attention"):
                    norm_ref = norm_refs.for_layer(layer_index, which)
                    module = norm_ref.bf16(x)
                    diff_3d = byte_diff(
                        reference_rms_norm_function(x, norm_ref, flat=False), module
                    )
                    diff_2d = byte_diff(
                        reference_rms_norm_function(x, norm_ref, flat=True), module
                    )
                    function_diff_worst = max(function_diff_worst, diff_3d, diff_2d)
                    function_records.append(
                        {
                            "layer": layer_index,
                            "kind": kind,
                            "norm": which,
                            "f_rms_norm_3d_diff_bytes": diff_3d,
                            "f_rms_norm_2d_diff_bytes": diff_2d,
                        }
                    )
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
    case.metric("f_rms_norm_diff_bytes_worst", float(function_diff_worst))
    if norms_bad:
        case.diag("failed norms", False, {"names": norms_bad[:20], "count": len(norms_bad)})
    case.diag(
        "square-semantics（诊断）：int pow / double pow / x*x 的逐位差异计数",
        True,
        {
            **square_stats,
            "note": "int pow 与 x*x 逐位相同；double pow（F.rms_norm composite 里的写法）"
            "在极端量级上会差 1 ulp —— 这是上一轮把 F.rms_norm 当 gate 时的 1 字节差异来源",
        },
    )
    case.diag(
        "F.rms_norm vs Qwen3_5RMSNorm（诊断，不参与判定）",
        function_diff_worst == 0,
        {
            "worst_diff_bytes": function_diff_worst,
            "note": "3d = 模型真实 shape [1,1,K]；2d = [1,K]。两者与模块的差若不同，"
            "说明差异来自 mean 的 reduce 配置而不是算术序列",
            "records": function_records,
        },
    )
    case.diag(
        "mean(-1) 的 shape 敏感性（诊断）",
        all(
            item["mean_3d_vs_2d_diff_bytes"] == 0 and item["mean_3d_vs_1d_diff_bytes"] == 0
            for item in shape_records
        ),
        {
            "note": "同一个 mean(-1) 在 [1,1,K] / [1,K] / [K] 上是否逐位相同；不同即"
            "说明 TensorIterator 的 coalesce 改变了 reduce launch 配置",
            "records": shape_records,
        },
    )


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
    report: OpReport,
    checkpoint: Checkpoint,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    min_module, max_module = checkpoint.scale_extremes()
    min_scale = checkpoint.input_scales[min_module]
    max_scale = checkpoint.input_scales[max_module]
    case = report.case(
        "scales-extremes",
        reference="Qwen3_5RMSNorm.forward -> nvfp4_quantize",
        min_module=min_module,
        min_input_scale=min_scale,
        min_gs=global_divisor(min_scale),
        max_module=max_module,
        max_input_scale=max_scale,
        max_gs=global_divisor(max_scale),
        note="两端都不在融合消费者里（层内只有 attention/mlp 的 proj），所以单独强制覆盖",
        layers=(0, 32, 63),
        input_kinds=("gauss-2", "saturation", "embed"),
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
        for kind in ("gauss-2", "saturation", "embed"):
            x, _ = input_for(kind, layer_index, checkpoint, device)
            for which in ("input", "post_attention"):
                norm_ref = norm_refs.for_layer(layer_index, which)
                for label, scales in combos:
                    gs_list = [global_divisor(scale) for scale in scales]
                    result = run_case(
                        case,
                        x,
                        norm_ref,
                        gs_list,
                        device,
                        label=f"{norm_ref_name(layer_index, which)}/{kind}/{label}",
                    )
                    ok_all = ok_all and result["ok"]
                    ran += 1
    case.gate(
        "coverage: 全局最小/最大 input_scale × 4 种组合 × 3 层 × 2 个 norm × 3 种输入 = 72 次",
        ran == 72,
        {
            "runs": ran,
            "combos": [list(item) for item in combos],
            "layers": [0, 32, 63],
            "norms_per_layer": 2,
            "inputs": ["gauss-2", "saturation", "embed"],
        },
    )
    case.metric("scales_extremes_passed", 1.0 if ok_all else 0.0)
    case.metric("scales_extremes_runs", float(ran))
    case.metric("min_gs", global_divisor(min_scale))
    case.metric("max_gs", global_divisor(max_scale))


def run_distribution_edge(
    report: OpReport,
    checkpoint: Checkpoint,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    case = report.case(
        "distribution-edge",
        reference="Qwen3_5RMSNorm.forward -> nvfp4_quantize",
        kinds=INPUT_KINDS,
        layers=(0, 3, 32, 63),
        eps=EPS,
        note="zeros = 全零 padding 行（amax==0 分支）；saturation = sf_pre 冲到 e4m3 448",
    )
    ok_all = True
    ran = 0
    with torch.no_grad():
        for layer_index in (0, 3, 32, 63):
            for kind in INPUT_KINDS:
                x, _ = input_for(kind, layer_index, checkpoint, device)
                for which in ("input", "post_attention"):
                    norm_ref = norm_refs.for_layer(layer_index, which)
                    pre, post = checkpoint.consumers(layer_index)
                    consumers = pre if which == "input" else post
                    gs_list = [checkpoint.gs(module) for module in consumers]
                    result = run_case(
                        case,
                        x,
                        norm_ref,
                        gs_list,
                        device,
                        label=f"{norm_ref_name(layer_index, which)}/{kind}",
                    )
                    ok_all = ok_all and result["ok"]
                    ran += 1
    case.gate(
        "coverage: 4 层 × 2 个 norm × 6 种输入（zeros/saturation/embed/3 个 RMS 量级）",
        ran == 48,
        {"samples": ran, "kinds": list(INPUT_KINDS)},
    )
    case.gate(
        "amax==0: 全零输入确实走到 sf=0 / code=0 分支",
        _zeros_case_is_zero(checkpoint, norm_refs, device),
        {"zeros_input": "x = 0 -> variance = 0 -> rstd = rsqrt(eps) -> y = 0 -> amax = 0"},
    )
    case.metric("distribution_edge_passed", 1.0 if ok_all else 0.0)
    case.metric("distribution_edge_samples", float(ran))


def _zeros_case_is_zero(
    checkpoint: Checkpoint, norm_refs: NormReferences, device: torch.device
) -> bool:
    norm_ref = norm_refs.get("layers.0.input_layernorm")
    x = zero_input().to(device)
    out, a_fp4, a_sf = _scratch(device, 2)
    gs_list = [checkpoint.gs("layers.0.linear_attn.in_proj_qkv"),
               checkpoint.gs("layers.0.linear_attn.in_proj_z")]
    with torch.no_grad():
        norm_quant_fused(x, norm_ref.w1, norm_ref.eps, gs_list, out, a_fp4, a_sf)
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


def run_micro(
    report: OpReport,
    checkpoint: Checkpoint,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    # 用第 3 层（full_attention）：input norm 有 3 个消费者（q/k/v），post norm 有 2 个
    # （gate/up），一次覆盖两种形态。
    layer_index = 3
    pre, post = checkpoint.consumers(layer_index)
    if len(pre) != 3:
        raise SystemExit(f"layer {layer_index} 不是 3 消费者的 full attention 层")
    norm_ref_2 = norm_refs.for_layer(layer_index, "post_attention")
    norm_ref_3 = norm_refs.for_layer(layer_index, "input")
    x, _ = input_for("gauss-2", layer_index, checkpoint, device)
    gs2 = [checkpoint.gs(post[0]), checkpoint.gs(post[1])]
    gs3 = [checkpoint.gs(pre[0]), checkpoint.gs(pre[1]), checkpoint.gs(pre[2])]
    out2, a2, s2 = _scratch(device, 2)
    out3, a3, s3 = _scratch(device, 3)
    SfLayout, nvfp4_quantize = _flashinfer()
    gs2_t = [torch.tensor([g], dtype=torch.float32, device=device) for g in gs2]
    gs3_t = [torch.tensor([g], dtype=torch.float32, device=device) for g in gs3]

    def two_step_2():
        y = norm_ref_2.bf16(x)
        for gs_t in gs2_t:
            nvfp4_quantize(
                y.reshape(1, HIDDEN),
                gs_t,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
                backend="cuda",
            )

    def two_step_3():
        y = norm_ref_3.bf16(x)
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
                x, norm_ref_2.w1, norm_ref_2.eps, gs2, out2, a2, s2, num_warps=warps
            ),
            warmup=BENCH_WARMUP,
            iters=BENCH_ITERS,
        )

    def fused_2():
        norm_quant_fused(x, norm_ref_2.w1, norm_ref_2.eps, gs2, out2, a2, s2)

    def fused_3():
        norm_quant_fused(x, norm_ref_3.w1, norm_ref_3.eps, gs3, out3, a3, s3)

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
        reference="Qwen3_5RMSNorm.forward（模块）+ 每消费者一次 nvfp4_quantize",
    )
    numeric = run_case(case, x, norm_ref_2, gs2, device, label="micro-2consumers")
    numeric3 = run_case(case, x, norm_ref_3, gs3, device, label="micro-3consumers")
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
    case.metric(
        "speedup_2consumers_eager",
        eager["two_step_2"]["median_ms"] / eager["fused_2"]["median_ms"],
    )


# ------------------------------------------------------------------ 任务 M：生产者侧融合

#: 用第 3 层（full_attention）：同时有 mlp 与 self_attn，一次覆盖 down_proj / o_proj
#: 两个 producer；lm_head 用 final norm。
PRODUCER_LAYER = 3
#: 负对照的最小 code 差异比例（错实现必须显著超差，否则这个 case 没有分辨力）。
PRODUCER_NEG_MARGIN = 0.05

_SILU = None


def _silu():
    """模型里 ``ACT2FN["silu"]``（``nn.functional.silu`` -> ``aten::silu``）。"""
    global _SILU
    if _SILU is None:
        from transformers.activations import ACT2FN

        _SILU = ACT2FN["silu"]
    return _SILU


class ProducerWeights:
    """真实 checkpoint 权重的 ``NVFP4Linear``（按短名缓存），给 producer 端造真实量级输入。

    只加载第 3 层的 gate/up/down 与 q/k/v/o（约 0.2 GB），不碰整模型；``lm_head`` 的
    wiring 用一个 N=1024 的随机权重模块（K 与真实一致），因为两条路径用同一个模块，
    比较的是量化/接线是否 bitwise，与权重数值无关。
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._reader = CheckpointReader(MODEL_DIR)
        self._cache: dict = {}

    def close(self) -> None:
        self._reader.close()

    def linear(self, short: str):
        if short not in self._cache:
            from runner.ops.nvfp4.linear import NVFP4Linear

            prefix = CKPT_PREFIX + short
            self._cache[short] = NVFP4Linear(
                self._reader.get(prefix + ".weight", "cuda"),
                self._reader.get(prefix + ".weight_scale", "cuda"),
                self._reader.get(prefix + ".weight_scale_2", "cuda"),
                self._reader.get(prefix + ".input_scale", "cuda"),
                backend="cutlass",
            )
        return self._cache[short]

    def small_lm_head(self, n: int, input_scale: float):
        key = ("lm_head-probe", int(n))
        if key not in self._cache:
            from runner.ops.nvfp4.linear import NVFP4Linear

            generator = torch.Generator(device=self.device).manual_seed(SEED + 991)
            packed = torch.randint(
                0,
                256,
                (int(n), HIDDEN // 2),
                dtype=torch.uint8,
                generator=generator,
                device=self.device,
            )
            scale = (
                torch.rand(
                    (int(n), HIDDEN // 16),
                    generator=generator,
                    device=self.device,
                    dtype=torch.float32,
                )
                * 0.02
                + 0.01
            ).to(torch.float8_e4m3fn)
            self._cache[key] = NVFP4Linear(
                packed,
                scale,
                torch.ones(1, dtype=torch.float32, device=self.device),
                torch.tensor([float(input_scale)], dtype=torch.float32, device=self.device),
                backend="cutlass",
            )
        return self._cache[key]


def _producer_scratch(k: int, device: torch.device):
    """按 FlashInfer 自己返回的 shape/dtype 分配 producer 的 out/a_fp4/a_sf 槽位。"""
    SfLayout, nvfp4_quantize = _flashinfer()
    probe = torch.zeros(1, int(k), dtype=torch.bfloat16, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    a_fp4, a_sf = nvfp4_quantize(
        probe,
        scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    return allocate_producer_scratch(
        int(k),
        device=device,
        a_fp4_shape=tuple(a_fp4.shape),
        a_sf_shape=tuple(a_sf.shape),
        a_fp4_dtype=a_fp4.dtype,
        a_sf_dtype=a_sf.dtype,
    )


def _nrmse(value: torch.Tensor, reference: torch.Tensor) -> float:
    diff = value.float().reshape(-1) - reference.float().reshape(-1)
    denom = reference.float().reshape(-1).pow(2).mean().sqrt()
    return float((diff.pow(2).mean().sqrt() / (denom + 1e-30)).item())


def _byte_diff_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    left = a.contiguous().view(torch.uint8).reshape(-1)
    right = b.contiguous().view(torch.uint8).reshape(-1)
    if left.numel() != right.numel():
        return 1.0
    return float((left != right).float().mean().item())


def _dequant_activation(a_fp4: torch.Tensor, a_sf: torch.Tensor, k: int, gs: float):
    from runner.ops.nvfp4.reference import dequant_nvfp4

    scale = a_sf.reshape(SF_ROWS, int(k) // 16)[0, : int(k) // 16]
    return dequant_nvfp4(
        a_fp4.reshape(1, int(k) // 2),
        scale.reshape(1, int(k) // 16),
        1.0 / float(gs),
    ).reshape(-1)


def _producer_compare(
    out: torch.Tensor,
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    ref_out: torch.Tensor,
    ref_quant,
    k: int,
    gs: float,
) -> dict:
    a_ref, sf_ref = ref_quant
    result = {
        "out_bf16_diff_bytes": byte_diff(out, ref_out),
        "a_fp4_diff_bytes": byte_diff(a_fp4, a_ref),
        "a_sf_diff_bytes": byte_diff(a_sf, sf_ref),
        "gs": float(gs),
    }
    fused_deq = _dequant_activation(a_fp4, a_sf, k, gs)
    reference_deq = _dequant_activation(a_ref, sf_ref, k, gs)
    result["dequant_nrmse"] = _nrmse(fused_deq, reference_deq)
    result["reference_dequant_nrmse_vs_bf16"] = _nrmse(
        reference_deq, ref_out.reshape(-1).float()
    )
    result["ok"] = (
        result["out_bf16_diff_bytes"] == 0
        and result["a_fp4_diff_bytes"] == 0
        and result["a_sf_diff_bytes"] == 0
    )
    return result


def _downproj_inputs(
    checkpoint: Checkpoint,
    weights: ProducerWeights,
    norm_refs: NormReferences,
    device: torch.device,
    kind: str,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """真实 gate/up 投影造 ``silu`` 的两个输入；saturation 手工放大一个通道。"""
    layer = PRODUCER_LAYER
    x, _ = input_for(kind, layer, checkpoint, device)
    normed = norm_refs.for_layer(layer, "post_attention").bf16(x)
    gate = weights.linear(f"layers.{layer}.mlp.gate_proj")(normed)
    up = weights.linear(f"layers.{layer}.mlp.up_proj")(normed)
    note = kind
    if kind == "saturation":
        gate = gate.clone()
        up = up.clone()
        gate[0, 0, SATURATION_CHANNEL] = SATURATION_VALUE
        up[0, 0, SATURATION_CHANNEL] = 1.0
        note = "saturation（真实 g/u 上把一个通道抬到 3e4，强制 sf 饱和）"
    return gate, up, note


def _oproj_inputs(
    checkpoint: Checkpoint,
    weights: ProducerWeights,
    norm_refs: NormReferences,
    device: torch.device,
    kind: str,
):
    """真实 q_proj 的 gate 半 + 真实 v_proj（按 kv-head 重复 6 次）当 attn 代理。

    真实 attn_output 是每个 head 的 softmax(QK)V（V 的凸组合），量级不超过 V；这里用
    真实 V 的单头重复来对齐 head 布局（query head 6j+i 读 kv head j），是量级/布局
    都正确的代理。gate 一半是模型真实的 ``q_proj(...).chunk(2)[1]``。
    """
    config = checkpoint.text_config or {}
    head_dim = int(config["head_dim"])
    num_heads = int(config["num_attention_heads"])
    num_kv = int(config["num_key_value_heads"])
    layer = PRODUCER_LAYER
    x, _ = input_for(kind, layer, checkpoint, device)
    normed = norm_refs.for_layer(layer, "input").bf16(x)
    q_out = weights.linear(f"layers.{layer}.self_attn.q_proj")(normed)
    q_view = q_out.view(1, 1, num_heads, 2 * head_dim)
    _query, gate = torch.chunk(q_view, 2, dim=-1)
    gate = gate.reshape(1, 1, num_heads * head_dim)
    value = weights.linear(f"layers.{layer}.self_attn.v_proj")(normed)
    value = value.view(1, 1, num_kv, head_dim)
    attn = value.repeat_interleave(num_heads // num_kv, dim=2).reshape(
        1, 1, num_heads * head_dim
    )
    note = kind
    if kind == "saturation":
        attn = attn.clone()
        attn[0, 0, SATURATION_CHANNEL] = SATURATION_VALUE
        gate = gate.clone()
        gate[0, 0, SATURATION_CHANNEL] = 0.0
        note = "saturation（真实 attn 代理上把一个通道抬到 3e4，强制 sf 饱和）"
    return attn.contiguous(), gate.contiguous(), note



def _activation_formula_probe(gate: torch.Tensor, sigmoid: torch.Tensor) -> dict:
    """用纯 torch 找出 ``aten::silu`` / ``aten::sigmoid`` 实际用的机器算术。

    diagnostic 用：如果主 gate 失败，这里能立刻区分「公式选错」和「exp/div 的指令
    不匹配」。三种 silu 候选都是 CUDA kernel 里可能出现的写法。
    """
    import torch.nn.functional as F

    ref_silu = F.silu(gate)
    xf = gate.float()
    e = torch.exp(-xf)
    silu_div = (xf / (1.0 + e)).to(torch.bfloat16)
    silu_mul = (xf * (1.0 / (1.0 + e))).to(torch.bfloat16)
    silu_sig = (xf * torch.sigmoid(xf)).to(torch.bfloat16)
    ref_sig = torch.sigmoid(sigmoid)
    sf = sigmoid.float()
    es = torch.exp(-sf)
    sig_div = ((1.0 / (1.0 + es)).to(torch.bfloat16))
    sig_mul = (torch.exp(sf) / (1.0 + torch.exp(sf))).to(torch.bfloat16)
    return {
        "samples": int(ref_silu.numel()),
        "silu_div_diff_bytes": byte_diff(silu_div, ref_silu),
        "silu_mul_diff_bytes": byte_diff(silu_mul, ref_silu),
        "silu_sigmoid_diff_bytes": byte_diff(silu_sig, ref_silu),
        "sigmoid_div_diff_bytes": byte_diff(sig_div, ref_sig),
        "sigmoid_mul_diff_bytes": byte_diff(sig_mul, ref_sig),
        "note": "diff=0 的写法就是 torch 实际执行的机器算术；给主 gate 失败时定位用",
    }

def run_downproj_silu_mul(
    report: OpReport,
    checkpoint: Checkpoint,
    weights: ProducerWeights,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    k = int((checkpoint.text_config or {})["intermediate_size"])
    gs_list = checkpoint.gs_with_extremes(".mlp.down_proj")
    kinds = ("gauss-0.5", "gauss-2", "gauss-8", "embed", "saturation", "zeros")
    case = report.case(
        "downproj-silu-mul",
        producer="down_proj <- silu(gate_proj_out) * up_proj_out",
        k=k,
        layer=PRODUCER_LAYER,
        weights="checkpoint 的真实 gate_proj/up_proj（第 3 层）",
        reference="ACT2FN['silu'](g) * u（torch，两次 kernel）-> nvfp4_quantize",
        gs="全部 64 个真实 down_proj input_scale + 全局 min/max",
        input_kinds=list(kinds),
        rule="R2（任务 E 冻结）+ div.rn + libdevice __nv_expf",
    )
    out, a_fp4, a_sf = _producer_scratch(k, device)
    act = _silu()
    worst = {
        "out_bf16_diff_bytes": 0,
        "a_fp4_diff_bytes": 0,
        "a_sf_diff_bytes": 0,
        "dequant_nrmse": 0.0,
        "reference_dequant_nrmse_vs_bf16": 0.0,
    }
    runs = 0
    codes = 0
    records: list[dict] = []
    negative_fraction = 0.0
    negative_out_fraction = 0.0
    probe_gate = None
    with torch.no_grad():
        for kind in kinds:
            gate, up, note = _downproj_inputs(checkpoint, weights, norm_refs, device, kind)
            if kind == "gauss-2":
                probe_gate = gate
            reference = (act(gate) * up).to(torch.bfloat16)
            refs = nvfp4_quantize_bytewise(reference, gs_list, device, k=k)
            for gs, ref_quant in zip(gs_list, refs):
                producer_quant_fused(
                    gate, up, "silu_mul", gs, out, a_fp4, a_sf
                )
                result = _producer_compare(
                    out, a_fp4, a_sf, reference, ref_quant, k, gs
                )
                codes += ref_quant[0].numel() * 2
                runs += 1
                for key in worst:
                    if key.endswith("_bytes"):
                        worst[key] = max(worst[key], result[key])
                    else:
                        worst[key] = max(worst[key], result[key])
                if not result["ok"] and len(records) < 8:
                    records.append({"kind": note, **result})
            # 负对照：silu -> relu（错实现必须显著改变 code；同一输入上比较）
            gs = gs_list[0]
            wrong_reference = (torch.relu(gate) * up).to(torch.bfloat16)
            wrong_q = nvfp4_quantize_bytewise(wrong_reference, [gs], device, k=k)[0]
            producer_quant_fused(gate, up, "silu_mul", gs, out, a_fp4, a_sf)
            negative_fraction = max(
                negative_fraction,
                _byte_diff_fraction(a_fp4, wrong_q[0]),
            )
            negative_out_fraction = max(
                negative_out_fraction,
                _byte_diff_fraction(reference, wrong_reference),
            )
    torch.cuda.synchronize()
    case.diag(
        "silu 公式探针（诊断：torch 实际用哪种写法，样本是 gauss-2 的真实 gate 输出）",
        True,
        _activation_formula_probe(probe_gate, probe_gate),
    )
    case.gate(
        "bitwise[downproj-silu-mul]: out_bf16 + (a_fp4, a_sf) 与 torch 两步路径逐字节相同",
        worst["out_bf16_diff_bytes"] == 0
        and worst["a_fp4_diff_bytes"] == 0
        and worst["a_sf_diff_bytes"] == 0,
        {
            "worst": worst,
            "runs": runs,
            "failing_samples": records,
            "num_warps": DEFAULT_NUM_WARPS,
        },
    )
    case.gate(
        "coverage: 真实第 3 层权重 × 6 种输入 × 全部 64 个 down_proj gs（含全局 min/max）",
        len(checkpoint.scales_of(".mlp.down_proj")) == 64
        and len(gs_list) >= 64
        and runs == len(kinds) * len(gs_list)
        and checkpoint.gs(checkpoint.scale_extremes()[0]) in gs_list
        and checkpoint.gs(checkpoint.scale_extremes()[1]) in gs_list,
        {
            "k": k,
            "real_down_proj_scales": len(checkpoint.scales_of(".mlp.down_proj")),
            "gs_count": len(gs_list),
            "runs": runs,
            "input_kinds": list(kinds),
            "kinds": sorted(KINDS),
            "includes_global_extremes": True,
            "saturation": "gate[7] = 3e4",
            "zeros": "amax == 0 分支",
        },
    )
    case.gate(
        "negative-control[downproj-silu-mul]: silu 换成 relu 必须显著超差",
        negative_fraction > PRODUCER_NEG_MARGIN
        and negative_out_fraction > PRODUCER_NEG_MARGIN,
        {
            "code_diff_fraction": negative_fraction,
            "out_bf16_diff_fraction": negative_out_fraction,
            "threshold": PRODUCER_NEG_MARGIN,
        },
    )
    case.metric("downproj_out_bf16_diff_bytes", float(worst["out_bf16_diff_bytes"]))
    case.metric("downproj_a_fp4_diff_bytes", float(worst["a_fp4_diff_bytes"]))
    case.metric("downproj_a_sf_diff_bytes", float(worst["a_sf_diff_bytes"]))
    case.metric("downproj_dequant_nrmse", float(worst["dequant_nrmse"]))
    case.metric(
        "downproj_reference_dequant_nrmse_vs_bf16",
        float(worst["reference_dequant_nrmse_vs_bf16"]),
    )
    case.metric("downproj_runs", float(runs))
    case.metric("downproj_codes", float(codes))
    case.metric("downproj_negative_code_diff_fraction", float(negative_fraction))


def run_oproj_gate(
    report: OpReport,
    checkpoint: Checkpoint,
    weights: ProducerWeights,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    config = checkpoint.text_config or {}
    k = int(config["num_attention_heads"]) * int(config["head_dim"])
    gs_list = checkpoint.gs_with_extremes(".self_attn.o_proj")
    kinds = ("gauss-0.5", "gauss-2", "gauss-8", "embed", "saturation", "zeros")
    case = report.case(
        "oproj-gate",
        producer="o_proj <- attn_output * sigmoid(gate)",
        k=k,
        layer=PRODUCER_LAYER,
        weights="checkpoint 的真实 q_proj / v_proj（第 3 层）",
        reference="attn * torch.sigmoid(gate)（torch）-> nvfp4_quantize",
        gs="全部 16 个真实 o_proj input_scale + 全局 min/max",
        input_kinds=list(kinds),
        attn_proxy="v_proj 输出按 kv-head 重复 num_key_value_groups=6 次",
    )
    out, a_fp4, a_sf = _producer_scratch(k, device)
    worst = {
        "out_bf16_diff_bytes": 0,
        "a_fp4_diff_bytes": 0,
        "a_sf_diff_bytes": 0,
        "dequant_nrmse": 0.0,
        "reference_dequant_nrmse_vs_bf16": 0.0,
    }
    runs = 0
    codes = 0
    records: list[dict] = []
    negative_fraction = 0.0
    negative_out_fraction = 0.0
    probe_gate = None
    with torch.no_grad():
        for kind in kinds:
            attn, gate, note = _oproj_inputs(
                checkpoint, weights, norm_refs, device, kind
            )
            if kind == "gauss-2":
                probe_gate = gate
            reference = (attn * torch.sigmoid(gate)).to(torch.bfloat16)
            refs = nvfp4_quantize_bytewise(reference, gs_list, device, k=k)
            for gs, ref_quant in zip(gs_list, refs):
                producer_quant_fused(
                    attn, gate, "sigmoid_mul", gs, out, a_fp4, a_sf
                )
                result = _producer_compare(
                    out, a_fp4, a_sf, reference, ref_quant, k, gs
                )
                codes += ref_quant[0].numel() * 2
                runs += 1
                for key in worst:
                    worst[key] = max(worst[key], result[key])
                if not result["ok"] and len(records) < 8:
                    records.append({"kind": note, **result})
            gs = gs_list[0]
            # 负对照：sigmoid 换成常数 1（即 attn 本身）
            wrong_reference = attn.clone()
            wrong_q = nvfp4_quantize_bytewise(wrong_reference, [gs], device, k=k)[0]
            producer_quant_fused(attn, gate, "sigmoid_mul", gs, out, a_fp4, a_sf)
            negative_fraction = max(
                negative_fraction, _byte_diff_fraction(a_fp4, wrong_q[0])
            )
            negative_out_fraction = max(
                negative_out_fraction,
                _byte_diff_fraction(reference, wrong_reference),
            )
    torch.cuda.synchronize()
    case.diag(
        "sigmoid 公式探针（诊断：torch 实际用哪种写法，样本是 gauss-2 的真实 gate）",
        True,
        _activation_formula_probe(probe_gate, probe_gate),
    )
    case.gate(
        "bitwise[oproj-gate]: out_bf16 + (a_fp4, a_sf) 与 torch 两步路径逐字节相同",
        worst["out_bf16_diff_bytes"] == 0
        and worst["a_fp4_diff_bytes"] == 0
        and worst["a_sf_diff_bytes"] == 0,
        {"worst": worst, "runs": runs, "failing_samples": records},
    )
    case.gate(
        "coverage: 真实第 3 层 q/v 权重 × 6 种输入 × 全部 16 个 o_proj gs（含全局 min/max）",
        len(checkpoint.scales_of(".self_attn.o_proj")) == 16
        and len(gs_list) >= 16
        and runs == len(kinds) * len(gs_list)
        and checkpoint.gs(checkpoint.scale_extremes()[0]) in gs_list
        and checkpoint.gs(checkpoint.scale_extremes()[1]) in gs_list,
        {
            "k": k,
            "real_o_proj_scales": len(checkpoint.scales_of(".self_attn.o_proj")),
            "gs_count": len(gs_list),
            "runs": runs,
            "input_kinds": list(kinds),
            "attn_proxy": "v_proj(真实) 按 kv-head 重复 6 次",
        },
    )
    case.gate(
        "negative-control[oproj-gate]: sigmoid 换成常数 1 必须显著超差",
        negative_fraction > PRODUCER_NEG_MARGIN
        and negative_out_fraction > PRODUCER_NEG_MARGIN,
        {
            "code_diff_fraction": negative_fraction,
            "out_bf16_diff_fraction": negative_out_fraction,
            "threshold": PRODUCER_NEG_MARGIN,
        },
    )
    case.metric("oproj_out_bf16_diff_bytes", float(worst["out_bf16_diff_bytes"]))
    case.metric("oproj_a_fp4_diff_bytes", float(worst["a_fp4_diff_bytes"]))
    case.metric("oproj_a_sf_diff_bytes", float(worst["a_sf_diff_bytes"]))
    case.metric("oproj_dequant_nrmse", float(worst["dequant_nrmse"]))
    case.metric(
        "oproj_reference_dequant_nrmse_vs_bf16",
        float(worst["reference_dequant_nrmse_vs_bf16"]),
    )
    case.metric("oproj_runs", float(runs))
    case.metric("oproj_codes", float(codes))
    case.metric("oproj_negative_code_diff_fraction", float(negative_fraction))


def run_lmhead_final_norm(
    report: OpReport,
    checkpoint: Checkpoint,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    if checkpoint.final_norm_weight is None:
        raise SystemExit("checkpoint 没有 final norm 权重")
    norm_ref = NormReference(checkpoint.final_norm_weight, EPS, device)
    gs_list = checkpoint.gs_with_extremes("lm_head")
    kinds = ("gauss-0.5", "gauss-2", "gauss-8", "embed", "saturation", "zeros")
    case = report.case(
        "lmhead-final-norm",
        producer="lm_head <- Qwen3_5RMSNorm(final norm) 输出",
        k=HIDDEN,
        consumers=1,
        weights="checkpoint 的真实 model.norm.weight（zero-centered）",
        reference="Qwen3_5RMSNorm.forward -> nvfp4_quantize",
        gs="真实的 lm_head input_scale + 全局 min/max",
        input_kinds=list(kinds),
        path="norm_quant.norm_quant_fused（1 消费者形态）",
    )
    out, a_fp4, a_sf = _producer_scratch(HIDDEN, device)
    worst = {
        "out_bf16_diff_bytes": 0,
        "a_fp4_diff_bytes": 0,
        "a_sf_diff_bytes": 0,
        "dequant_nrmse": 0.0,
        "reference_dequant_nrmse_vs_bf16": 0.0,
    }
    runs = 0
    codes = 0
    records: list[dict] = []
    negative_fraction = 0.0
    wrong_w1 = norm_ref.module.weight.detach().float().contiguous()
    with torch.no_grad():
        for kind in kinds:
            x, _ = input_for(kind, PRODUCER_LAYER, checkpoint, device)
            reference = norm_ref.bf16(x)
            refs = nvfp4_quantize_bytewise(reference, gs_list, device)
            for gs, ref_quant in zip(gs_list, refs):
                norm_quant_fused(
                    x, norm_ref.w1, norm_ref.eps, [gs], out, [a_fp4], [a_sf]
                )
                result = _producer_compare(
                    out, a_fp4, a_sf, reference, ref_quant, HIDDEN, gs
                )
                codes += ref_quant[0].numel() * 2
                runs += 1
                for key in worst:
                    worst[key] = max(worst[key], result[key])
                if not result["ok"] and len(records) < 8:
                    records.append({"kind": kind, **result})
            # 负对照：用 weight 而不是 1 + weight。比较「错实现的量化结果」与
            # 「正确参考的量化结果」——同一份 kernel 自比会恒等，没有分辨力。
            gs = gs_list[0]
            wrong_ref = (
                x.float()
                * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + norm_ref.eps)
                * wrong_w1
            ).to(torch.bfloat16)
            wrong_a, _ = nvfp4_quantize_bytewise(wrong_ref, [gs], device)[0]
            negative_fraction = max(
                negative_fraction, _byte_diff_fraction(wrong_a, refs[0][0])
            )
    torch.cuda.synchronize()
    case.gate(
        "bitwise[lmhead-final-norm]: out_bf16 + (a_fp4, a_sf) 与模块两步路径逐字节相同",
        worst["out_bf16_diff_bytes"] == 0
        and worst["a_fp4_diff_bytes"] == 0
        and worst["a_sf_diff_bytes"] == 0,
        {"worst": worst, "runs": runs, "failing_samples": records},
    )
    case.gate(
        "coverage: 真实 final norm × 6 种输入 × lm_head 的 1 个真实 gs + 全局 min/max（1 消费者）",
        len(checkpoint.scales_of("lm_head")) == 1
        and len(gs_list) >= 1
        and runs == len(kinds) * len(gs_list)
        and checkpoint.gs(checkpoint.scale_extremes()[0]) in gs_list
        and checkpoint.gs(checkpoint.scale_extremes()[1]) in gs_list,
        {
            "k": HIDDEN,
            "real_lm_head_scales": len(checkpoint.scales_of("lm_head")),
            "gs_count": len(gs_list),
            "runs": runs,
            "consumers": 1,
            "input_kinds": list(kinds),
        },
    )
    case.gate(
        "negative-control[lmhead-final-norm]: 用 weight 而不是 1+weight 必须显著超差",
        negative_fraction > PRODUCER_NEG_MARGIN,
        {"code_diff_fraction": negative_fraction, "threshold": PRODUCER_NEG_MARGIN},
    )
    case.metric("lmhead_out_bf16_diff_bytes", float(worst["out_bf16_diff_bytes"]))
    case.metric("lmhead_a_fp4_diff_bytes", float(worst["a_fp4_diff_bytes"]))
    case.metric("lmhead_a_sf_diff_bytes", float(worst["a_sf_diff_bytes"]))
    case.metric("lmhead_dequant_nrmse", float(worst["dequant_nrmse"]))
    case.metric(
        "lmhead_reference_dequant_nrmse_vs_bf16",
        float(worst["reference_dequant_nrmse_vs_bf16"]),
    )
    case.metric("lmhead_runs", float(runs))
    case.metric("lmhead_codes", float(codes))
    case.metric("lmhead_negative_code_diff_fraction", float(negative_fraction))


def run_fuse_micro(
    report: OpReport,
    checkpoint: Checkpoint,
    weights: ProducerWeights,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    """生产者侧融合的 per-call 计时 + 按真实 64/16/1 分布折算整步。"""
    SfLayout, nvfp4_quantize = _flashinfer()
    act = _silu()
    config = checkpoint.text_config or {}
    o_k = int(config["num_attention_heads"]) * int(config["head_dim"])
    down_k = int((config)["intermediate_size"])
    gs_down = checkpoint.gs(f"layers.{PRODUCER_LAYER}.mlp.down_proj")
    gs_o = checkpoint.gs(f"layers.{PRODUCER_LAYER}.self_attn.o_proj")
    gs_lm = checkpoint.gs("lm_head")

    gate, up, _ = _downproj_inputs(
        checkpoint, weights, norm_refs, device, "gauss-2"
    )
    attn, attn_gate, _ = _oproj_inputs(
        checkpoint, weights, norm_refs, device, "gauss-2"
    )
    x, _ = input_for("gauss-2", PRODUCER_LAYER, checkpoint, device)
    final_norm = NormReference(checkpoint.final_norm_weight, EPS, device)
    down_out, down_a, down_sf = _producer_scratch(down_k, device)
    o_out, o_a, o_sf = _producer_scratch(o_k, device)
    lm_out, lm_a, lm_sf = _producer_scratch(HIDDEN, device)
    gs_down_t = torch.tensor([gs_down], dtype=torch.float32, device=device)
    gs_o_t = torch.tensor([gs_o], dtype=torch.float32, device=device)
    gs_lm_t = torch.tensor([gs_lm], dtype=torch.float32, device=device)

    def two_down():
        product = act(gate) * up
        nvfp4_quantize(
            product.reshape(1, down_k),
            gs_down_t,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )

    def fused_down():
        producer_quant_fused(
            gate, up, "silu_mul", gs_down, down_out, down_a, down_sf
        )

    def two_o():
        product = attn * torch.sigmoid(attn_gate)
        nvfp4_quantize(
            product.reshape(1, o_k),
            gs_o_t,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )

    def fused_o():
        producer_quant_fused(
            attn, attn_gate, "sigmoid_mul", gs_o, o_out, o_a, o_sf
        )

    def two_lm():
        y = final_norm.bf16(x)
        nvfp4_quantize(
            y.reshape(1, HIDDEN),
            gs_lm_t,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )

    def fused_lm():
        norm_quant_fused(
            x, final_norm.w1, final_norm.eps, [gs_lm], lm_out, [lm_a], [lm_sf]
        )

    pairs = {
        "down_proj": (two_down, fused_down),
        "o_proj": (two_o, fused_o),
        "lm_head": (two_lm, fused_lm),
    }
    eager = {
        name: {
            "two_step_ms": time_per_call(two, warmup=BENCH_WARMUP, iters=BENCH_ITERS)["median_ms"],
            "fused_ms": time_per_call(fused, warmup=BENCH_WARMUP, iters=BENCH_ITERS)["median_ms"],
        }
        for name, (two, fused) in pairs.items()
    }
    graph = {}
    for name, (two, fused) in pairs.items():
        g_two = graph_per_call(
            two, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, repeats=GRAPH_REPEATS
        )
        g_fused = graph_per_call(
            fused, warmup=GRAPH_WARMUP, iters=GRAPH_ITERS, repeats=GRAPH_REPEATS
        )
        if g_two is not None and g_fused is not None:
            graph[name] = {
                "two_step_ms": g_two["median_ms"],
                "fused_ms": g_fused["median_ms"],
            }

    def estimate(timings: dict) -> dict:
        down_saved = timings["down_proj"]["two_step_ms"] - timings["down_proj"]["fused_ms"]
        o_saved = timings["o_proj"]["two_step_ms"] - timings["o_proj"]["fused_ms"]
        lm_saved = timings["lm_head"]["two_step_ms"] - timings["lm_head"]["fused_ms"]
        ours = 64 * down_saved + 16 * o_saved + 1 * lm_saved
        per_consumer = (64 * down_saved + 16 * o_saved) / 80.0
        all129 = ours + 48 * per_consumer
        return {
            "down_proj_saved_ms": down_saved,
            "o_proj_saved_ms": o_saved,
            "lm_head_saved_ms": lm_saved,
            "est_step_saved_ms": ours,
            "est_step_saved_129_ms": all129,
            "per_consumer_saved_ms": per_consumer,
        }

    eager_est = estimate(eager)
    graph_est = estimate(graph) if len(graph) == len(pairs) else None
    case = report.case(
        "fuse-micro",
        note="任务 M 的 micro tier；脚本里既有的 ``micro`` case 是任务 H 的 norm 口径，保持不变",
        timing="per-call CUDA events median（eager）；graph = 8 次调用录一张图后 replay / 8",
        step_model="64 down_proj + 16 o_proj + 1 lm_head；129 = 再加 48 个 out_proj（不是本任务）",
        reference="silu/sigmoid->mul + nvfp4_quantize（torch + FlashInfer）",
        warmup=BENCH_WARMUP,
        iters=BENCH_ITERS,
        graph_iters=GRAPH_ITERS,
        graph_repeats=GRAPH_REPEATS,
    )
    case.gate(
        "timing: eager 与（可 capture 时）graph 的三档 per-call 中位数都是有限正数",
        all(
            item["two_step_ms"] > 0.0 and math.isfinite(item["two_step_ms"])
            and item["fused_ms"] > 0.0 and math.isfinite(item["fused_ms"])
            for item in eager.values()
        ),
        {"eager": eager, "graph": graph},
    )
    case.gate(
        "coverage: down_proj / o_proj / lm_head 三档都测，且 K 与真实一致",
        set(pairs) == {"down_proj", "o_proj", "lm_head"}
        and down_k == 17408
        and o_k == 6144
        and HIDDEN == 5120,
        {
            "k": {"down_proj": down_k, "o_proj": o_k, "lm_head": HIDDEN},
            "step_model": {"down_proj": 64, "o_proj": 16, "lm_head": 1, "out_proj": 48},
        },
    )
    for name, item in eager.items():
        case.metric(f"eager_{name}_two_step_ms", item["two_step_ms"])
        case.metric(f"eager_{name}_fused_ms", item["fused_ms"])
    for name, item in graph.items():
        case.metric(f"graph_{name}_two_step_ms", item["two_step_ms"])
        case.metric(f"graph_{name}_fused_ms", item["fused_ms"])
    for key, value in eager_est.items():
        case.metric(f"eager_{key}", value)
    if graph_est is not None:
        for key, value in graph_est.items():
            case.metric(f"graph_{key}", value)
    case.diag("graph 口径（模型实际执行）", graph_est is not None, graph)


def run_producer_wiring(
    report: OpReport,
    checkpoint: Checkpoint,
    weights: ProducerWeights,
    norm_refs: NormReferences,
    device: torch.device,
) -> None:
    """走 ``runner/model/quant_fuse_patch.py`` 的真实接线（不加载 27B 模型）。

    用真实第 3 层的 gate/up/down 与 o_proj 模块、真实 final norm、K 相同的探测
    lm_head，把「融合路径到底有没有执行 / 有没有绕过 decode backend / 是否 bitwise」
    变成可判定项：
    - 三个调用点各执行一次，``fallbacks == {}``、``unconsumed == 0``；
    - 融合输出与「torch 两步路径」逐字节相同（量化 bitwise + 同一 GEMV backend）；
    - prefill shape 整层回退且结果与原实现逐字节相同；
    - ``QUANT_FUSE=0`` 整体关闭并恢复类方法。
    """
    from types import SimpleNamespace

    import runner.model.quant_fuse_patch as patch

    layer = PRODUCER_LAYER
    act = _silu()
    mlp = SimpleNamespace(
        gate_proj=weights.linear(f"layers.{layer}.mlp.gate_proj"),
        up_proj=weights.linear(f"layers.{layer}.mlp.up_proj"),
        down_proj=weights.linear(f"layers.{layer}.mlp.down_proj"),
        act_fn=act,
    )
    attn = SimpleNamespace(
        o_proj=weights.linear(f"layers.{layer}.self_attn.o_proj")
    )
    final_ref = NormReference(checkpoint.final_norm_weight, EPS, device)
    lm_head = weights.small_lm_head(1024, checkpoint.input_scales["lm_head"])
    fake_layer = SimpleNamespace(mlp=mlp, self_attn=attn)
    fake = SimpleNamespace(
        model=SimpleNamespace(layers=[fake_layer], norm=final_ref.module),
        lm_head=lm_head,
    )

    x, _ = input_for("gauss-2", layer, checkpoint, device)
    normed = norm_refs.for_layer(layer, "post_attention").bf16(x)
    attn_in, attn_gate, _ = _oproj_inputs(
        checkpoint, weights, norm_refs, device, "gauss-2"
    )
    attn_in = attn_in.clone()
    attn_gate = attn_gate.clone()

    case = report.case(
        "producer-wiring",
        patch_module="runner/model/quant_fuse_patch.py",
        entry=(
            "Qwen3_5MLP.forward / Qwen3_5Attention.forward / "
            "Qwen3_5RMSNorm.forward (final norm only)"
        ),
        call_sites=3,
        k={"down_proj": 17408, "o_proj": 6144, "lm_head": 5120},
        weights="checkpoint 真实第 3 层 gate/up/down/o_proj + 真实 final norm",
        lm_head="N=1024 探测模块，K=5120、input_scale 真实（两条路径用同一模块）",
        fallback="prefill shape [1,4,5120] 必须整层回退且逐字节相同",
    )

    patch.uninstall()
    info = patch.install(fake)
    fused_ok = {
        "mlp": False,
        "attention": False,
        "lm_head": False,
        "norm_bf16": False,
        "logits": False,
    }
    fallback_ok = {"mlp": False, "norm": False}
    counters: dict = {}
    after_fallback: dict = {}
    env_disabled = False
    restored_after_uninstall = False
    outputs: dict = {}
    try:
        fused_mlp = patch._patched_mlp(mlp, normed)
        ref_mlp = mlp.down_proj(act(mlp.gate_proj(normed)) * mlp.up_proj(normed))
        fused_ok["mlp"] = (
            fused_mlp is not None and byte_diff(fused_mlp, ref_mlp) == 0
        )

        fused_o = patch._finish_attention(attn, attn_in, attn_gate)
        ref_o = attn.o_proj(attn_in * torch.sigmoid(attn_gate))
        fused_ok["attention"] = (
            fused_o is not None and byte_diff(fused_o, ref_o) == 0
        )

        fused_norm = patch._patched_norm(final_ref.module, x)
        logits_fused = lm_head(fused_norm)
        ref_norm = patch._STATE["orig_norm"](final_ref.module, x)
        logits_ref = lm_head(ref_norm)
        fused_ok["norm_bf16"] = byte_diff(fused_norm, ref_norm) == 0
        fused_ok["lm_head"] = byte_diff(logits_fused, logits_ref) == 0
        fused_ok["logits"] = fused_ok["lm_head"] and fused_ok["norm_bf16"]
        outputs = {
            "fused_mlp_first": (
                float(fused_mlp.reshape(-1)[0].float().item())
                if fused_mlp is not None
                else None
            ),
            "ref_mlp_first": float(ref_mlp.reshape(-1)[0].float().item()),
            "fused_o_first": (
                float(fused_o.reshape(-1)[0].float().item())
                if fused_o is not None
                else None
            ),
            "ref_o_first": float(ref_o.reshape(-1)[0].float().item()),
            "fused_logits_first": float(logits_fused.reshape(-1)[0].float().item()),
            "ref_logits_first": float(logits_ref.reshape(-1)[0].float().item()),
        }
        counters = {
            key: value for key, value in patch.info()["counters"].items()
        }
        counters["fallbacks"] = dict(counters.get("fallbacks", {}))

        # 回退门：prefill shape 必须回原实现，结果逐字节相同，且计数。
        prefill = torch.randn(1, 4, HIDDEN, dtype=torch.bfloat16, device=device)
        fb_mlp = patch._patched_mlp(mlp, prefill)
        ref_fb_mlp = mlp.down_proj(
            act(mlp.gate_proj(prefill)) * mlp.up_proj(prefill)
        )
        fb_norm = patch._patched_norm(final_ref.module, prefill)
        ref_fb_norm = patch._STATE["orig_norm"](final_ref.module, prefill)
        fallback_ok["mlp"] = byte_diff(fb_mlp, ref_fb_mlp) == 0
        fallback_ok["norm"] = byte_diff(fb_norm, ref_fb_norm) == 0
        after_fallback = {
            key: value for key, value in patch.info()["counters"].items()
        }
        after_fallback["fallbacks"] = dict(after_fallback.get("fallbacks", {}))
    finally:
        patch.uninstall()
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

        restored_after_uninstall = (
            modeling.Qwen3_5MLP.forward.__name__ == "forward"
            and modeling.Qwen3_5Attention.forward.__name__ == "forward"
            and modeling.Qwen3_5RMSNorm.forward.__name__ == "forward"
        )

    # env flag：QUANT_FUSE=0 必须整体关掉（不 patch 任何类）
    patch.uninstall()
    os.environ[patch.ENV_FLAG] = "0"
    try:
        disabled = patch.install_from_env(fake)
    finally:
        del os.environ[patch.ENV_FLAG]
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

    env_disabled = (
        disabled.get("enabled") is False
        and modeling.Qwen3_5MLP.forward.__name__ == "forward"
        and modeling.Qwen3_5RMSNorm.forward.__name__ == "forward"
        and restored_after_uninstall
    )

    fused_total = int(counters.get("fused_mlp", 0)) + int(
        counters.get("fused_attention", 0)
    ) + int(counters.get("fused_lm_head", 0))
    case.gate(
        "patch:installed",
        bool(info.get("enabled"))
        and info.get("mlp") == 1
        and info.get("attention") == 1
        and info.get("final_norm") == 1
        and info.get("k")
        == {"down_proj": 17408, "o_proj": 6144, "lm_head": 5120},
        {
            "enabled": info.get("enabled"),
            "mlp": info.get("mlp"),
            "attention": info.get("attention"),
            "final_norm": info.get("final_norm"),
            "k": info.get("k"),
            "scratch_bytes": info.get("scratch_bytes"),
        },
    )
    case.gate(
        "patch:fused-path-executed",
        counters.get("fused_mlp") == 1
        and counters.get("fused_attention") == 1
        and counters.get("fused_lm_head") == 1
        and not counters.get("fallbacks")
        and counters.get("unconsumed") == 0,
        {
            "counters": counters,
            "note": "counters 是活 dict，check/bench 的 load.quant_fuse.counters 里也能看到",
        },
    )
    case.gate(
        "patch:bitwise-vs-two-step",
        all(fused_ok.values()),
        {"checks": fused_ok, "outputs": outputs},
    )
    case.gate(
        "patch:prefill-falls-back-bitwise",
        fallback_ok["mlp"]
        and fallback_ok["norm"]
        and after_fallback.get("fallbacks", {}).get("seq_len!=1", 0) == 2,
        {
            "checks": fallback_ok,
            "fallbacks": after_fallback.get("fallbacks", {}),
            "note": "prefill 两次（MLP + final norm）都回原实现且逐字节相同",
        },
    )
    case.gate(
        "patch:env-flag-disables",
        env_disabled,
        {"env": f"{patch.ENV_FLAG}=0", "enabled": disabled.get("enabled")},
    )
    case.gate(
        "coverage: down_proj(17408) / o_proj(6144) / lm_head(5120) 三个 K 都接到融合路径",
        fused_total == 3
        and set(info.get("k", {}).keys()) == {"down_proj", "o_proj", "lm_head"}
        and info.get("mlp") == 1
        and info.get("attention") == 1
        and info.get("final_norm") == 1,
        {
            "fused_calls": fused_total,
            "k": info.get("k"),
            "call_sites": {"mlp": info.get("mlp"), "attention": info.get("attention"), "final_norm": info.get("final_norm")},
        },
    )
    case.metric("wiring_fused_calls", float(fused_total))
    case.metric("wiring_fallbacks", float(len(counters.get("fallbacks", {}))))
    case.metric("wiring_mlp_bitwise", 1.0 if fused_ok["mlp"] else 0.0)
    case.metric("wiring_attention_bitwise", 1.0 if fused_ok["attention"] else 0.0)
    case.metric("wiring_lm_head_bitwise", 1.0 if fused_ok["lm_head"] else 0.0)


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
    # 整份脚本都是推理；模型 loader 也是 set_grad_enabled(False)。producer patch 把
    # grad 作为回退条件，所以这里必须先关掉，否则 producer-wiring 会整层回退。
    torch.set_grad_enabled(False)
    selected = select_cases(args.only)
    checkpoint = load_checkpoint()
    norm_refs = NormReferences(checkpoint, device)
    report = OpReport(
        args.out,
        CANDIDATE_NAME,
        CANDIDATE_FORMAT,
        candidate_extra={
            "kernel": "runner/ops/nvfp4/norm_quant.py::_norm_quant_kernel",
            "patch": "runner/model/norm_quant_patch.py (Qwen3_5DecoderLayer.forward, decode only)",
            "quant_rule": "R2（任务 E 冻结）：PTX 算术序列 + rcp.approx.ftz",
            "rmsnorm_rule": "Qwen3_5RMSNorm：x.pow(2).mean(-1) 的 launch 顺序 + rsqrt.approx + 单次 BF16 cast",
            "decisive_reference": "Qwen3_5RMSNorm.forward（模块，bf16 权重）+ nvfp4_quantize",
            "f_rms_norm_note": "F.rms_norm 仅 diagnostic：composite 里的 pow 是 double，"
            "与模块的 int pow 在极端量级上差 1 ulp（job-000036 的 1 字节差异来源）",
            "consumers": "2 (GDN in_proj_qkv+z / gate+up) 或 3 (full attention q/k/v)",
            "hidden": HIDDEN,
            "num_warps": DEFAULT_NUM_WARPS,
            "c6_bits": hex(0x3E2AAAAB),
            "c6": C6,
            "mean_factor": MEAN_FACTOR,
            "gate": "bitwise（逐字节），不用容差",
            # 任务 M：生产者侧 elementwise + quant
            "producer_kernel": "runner/ops/nvfp4/quant_fuse.py::_producer_quant_kernel",
            "producer_patch": "runner/model/quant_fuse_patch.py (MLP/Attention/final norm, decode only)",
            "producer_callsites": "down_proj x64, o_proj x16, lm_head x1（out_proj 不在本任务）",
            "producer_elementwise": "silu: x/(1+__nv_expf(-x)); sigmoid: 1/(1+__nv_expf(-x)); "
            "两者 div.rn.f32 -> BF16，再 mul（opmath f32）-> BF16",
        },
    )
    producer = ProducerWeights(device)
    handler = {
        "mean-tree": lambda rep, ck, dev: run_mean_tree(rep, ck, dev),
        "rsqrt-approx": lambda rep, ck, dev: run_rsqrt(rep, ck, dev),
        "real-layers": lambda rep, ck, dev: run_real_layers(rep, ck, norm_refs, dev),
        "scales-extremes": lambda rep, ck, dev: run_scales_extremes(rep, ck, norm_refs, dev),
        "distribution-edge": lambda rep, ck, dev: run_distribution_edge(rep, ck, norm_refs, dev),
        "micro": lambda rep, ck, dev: run_micro(rep, ck, norm_refs, dev),
        "downproj-silu-mul": lambda rep, ck, dev: run_downproj_silu_mul(
            rep, ck, producer, norm_refs, dev
        ),
        "oproj-gate": lambda rep, ck, dev: run_oproj_gate(
            rep, ck, producer, norm_refs, dev
        ),
        "lmhead-final-norm": lambda rep, ck, dev: run_lmhead_final_norm(
            rep, ck, norm_refs, dev
        ),
        "fuse-micro": lambda rep, ck, dev: run_fuse_micro(
            rep, ck, producer, norm_refs, dev
        ),
        "producer-wiring": lambda rep, ck, dev: run_producer_wiring(
            rep, ck, producer, norm_refs, dev
        ),
    }
    try:
        for name in selected:
            handler[name](report, checkpoint, device)
    finally:
        producer.close()
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
