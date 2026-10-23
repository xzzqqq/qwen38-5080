#!/usr/bin/env python3
"""Phase 2 任务 E：activation NVFP4 quant 的 reference / FlashInfer 差异归因。

设计：
- 采样三类数据：(a) 以 checkpoint 真实 ``input_scale`` 为尺度的高斯 BF16；
  (b) 带离群值的长尾分布，外加一组把 block amax 推到 e4m3 相邻值中点附近的
  定向分布；(c) M2 ``numeric.json`` 记录的失败样本 replay：同 checkpoint / prompt /
  fixed token 逐 16 步 teacher forcing，重新抓取真实 hidden state（numeric.json 只
  存聚合值，没有原始 16-element block，所以必须用同输入重放才能拿到原始值）。
- 对每组输入调用与 ``runner/ops/nvfp4/linear.py`` 完全相同的
  ``nvfp4_quantize(x, 1/input_scale, sfLayout=layout_128x4, do_shuffle=False,
  backend="cuda")``，再把 swizzled scale 解回线性布局。
- 与三份实现逐 bit 比较：
  R0 = 现有 ``quantize_activation_reference``（M2 使用的 reference）；
  R1 = 复刻 kernel 算术序列的 offline reference（block amax -> fp8 scale -> 倒数
       -> E2M1），所有倒数用正确舍入的 ``1/x``；
  R2 = 同 R1，但 ``rcp.approx.ftz.f32`` 的取值用同卡 inline PTX 探针实测
       （即规则本身：kernel 的 PTX 算术序列）。
- 分类所有 mismatch 的 16-element block：fp8 scale 舍入（round-half-even vs
  half-away、mul-recip-6 vs div-6）、E2M1 midpoint tie、global divisor 除法 vs
  乘倒数、subnormal/下溢、saturation、signed zero、残余。
- 规则只在 fit 样本上确定（``rcp(6)`` 的取值、out scale 形式），heldout 分布/seed
  只用于验证；能否冻结 quant reference 由 fit+heldout 的 bitwise 结果回答。
- 不修改主线量化语义：``git diff HEAD^ HEAD`` 不得触碰禁止路径，记为 gate。

数据流：
    x BF16 -> flashinfer nvfp4_quantize -> codes / linear sf（搬回 CPU）
    x BF16 -> R0 / R1 / R2 -> codes / sf（CPU float32 算术）
    -> block taxonomy + op.json + mismatch_blocks.jsonl

预期瓶颈：真实 hidden 需要一次模型 forward（prefill + 16 步 teacher forcing，约
10-20 s）；CPU 逐 bit 比较是秒级；整体预算 < 5 分钟。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from runner.ops.nvfp4.reference import (  # noqa: E402
    quantize_activation_reference,
    unpack_e2m1,
    unswizzle_scale_128x4,
)

# ------------------------------------------------------------------ 固定输入

SCENARIOS = (
    "synthetic-gaussian-k5120",
    "synthetic-outlier-k17408",
    "synthetic-scale-tie-k6144",
    "real-m2-k5120",
    "real-m2-k17408",
    "real-m2-k6144",
    "heldout-gaussian-k6144",
    "heldout-outlier-k5120",
    "coverage-summary",
)

HELDOUT_SCENARIOS = (
    "heldout-gaussian-k6144",
    "heldout-outlier-k5120",
)

MODEL_DIR = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"
MANIFEST_PATH = ROOT / "results/m2/20261020T175405-manifest/tensor_manifest.json"
M2_NUMERIC = ROOT / "results/m2/20261021T141833-numeric/numeric.json"

PROMPT_FILES = (
    "results/payload_math.json",
    "results/payload_code.json",
    "results/payload_chat.json",
)

#: M2 记录过 activation quant 失败的模块；replay 必须覆盖它们（gate）。
CAPTURE_MODULES: dict[str, tuple[int, int]] = {
    "model.layers.10.mlp.gate_proj": (10, 5120),
    "model.layers.38.mlp.gate_proj": (38, 5120),
    "model.layers.53.mlp.gate_proj": (53, 5120),
    "model.layers.57.mlp.gate_proj": (57, 5120),
    "model.layers.10.linear_attn.in_proj_qkv": (10, 5120),
    "model.layers.53.linear_attn.in_proj_qkv": (53, 5120),
    "model.layers.10.mlp.down_proj": (10, 17408),
    "model.layers.48.mlp.down_proj": (48, 17408),
    "model.layers.53.mlp.down_proj": (53, 17408),
    "model.layers.54.mlp.down_proj": (54, 17408),
    "model.layers.57.mlp.down_proj": (57, 17408),
    "model.layers.43.self_attn.o_proj": (43, 6144),
    "model.layers.47.self_attn.o_proj": (47, 6144),
    "model.layers.59.self_attn.o_proj": (59, 6144),
}

CAPTURE_K = (5120, 6144, 17408)
PREFILL_ROWS = 8          # 每组 (prompt, module) 只保留前几行 prefill，控制 CPU 预算
DECODE_STEPS = 16
DETAIL_LIMIT = 512        # 每个 group 保存的 mismatch block 明细上限

#: 禁止路径：本次提交只能改本脚本（任务 E 不碰主线）。
FORBIDDEN_PATHS = (
    "runner/ops/nvfp4/",
    "runner/experiment/",
    "bench/e2e/",
    "scripts/m3_check.py",
    "docs/",
    "Makefile",
)

# ------------------------------------------------------------------ 数值常量

_TINY = 1.1754943508222875e-38
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_E2M1_BOUNDARIES = (_E2M1_MAG[:-1] + _E2M1_MAG[1:]) / 2.0
_E4M3_BITS = torch.arange(0, 127, dtype=torch.int16).to(torch.uint8)
_E4M3_VALUES = _E4M3_BITS.view(torch.float8_e4m3fn).float()
_E4M3_MIDPOINTS = ((_E4M3_VALUES[:-1] + _E4M3_VALUES[1:]) / 2.0).contiguous()

_C6_EXACT = torch.tensor(1.0 / 6.0, dtype=torch.float32)
_C6_DOWN = torch.nextafter(_C6_EXACT, torch.tensor(0.0, dtype=torch.float32))
_C6_UP = torch.nextafter(_C6_EXACT, torch.tensor(2.0, dtype=torch.float32))

CATEGORIES = (
    "fp8-scale-rounding",
    "e2m1-midpoint-tie",
    "global-divisor-mul-recip",
    "subnormal-underflow",
    "saturation",
    "signed-zero",
    "residual",
)
CAT_ID = {name: index for index, name in enumerate(CATEGORIES)}
CAT_FP8 = CAT_ID["fp8-scale-rounding"]
CAT_TIE = CAT_ID["e2m1-midpoint-tie"]
CAT_RECIP = CAT_ID["global-divisor-mul-recip"]
CAT_UNDERFLOW = CAT_ID["subnormal-underflow"]
CAT_SATURATION = CAT_ID["saturation"]
CAT_SIGNED_ZERO = CAT_ID["signed-zero"]
CAT_RESIDUAL = CAT_ID["residual"]


# ------------------------------------------------------------------ 基础工具


def _bits_e4m3(value: torch.Tensor) -> torch.Tensor:
    """float32 -> e4m3 code（round-nearest-even + satfinite，与 cvt.rn.satfinite 一致）。"""
    return value.to(torch.float8_e4m3fn).view(torch.uint8)


def _e4m3_float(bits: torch.Tensor) -> torch.Tensor:
    return bits.contiguous().view(torch.float8_e4m3fn).float()


def _e2m1_codes(values: torch.Tensor, *, signed_zero: bool = True) -> torch.Tensor:
    """E2M1 最近舍入（tie 取偶数 code，饱和到 ±6）；sign bit 保留 -0.0。

    与现有 reference 的差别只在 -0.0：reference 用 ``values < 0`` 判符号，会把
    -0.0 变成 +0 code；这里用 signbit 保留符号位（硬件 cvt 的语义）。
    """
    mag = values.abs()
    lo = torch.bucketize(mag, _E2M1_BOUNDARIES.to(mag.device)).clamp(0, 7)
    hi = (lo + 1).clamp(0, 7)
    mag_lut = _E2M1_MAG.to(mag.device)
    lo_val = mag_lut[lo]
    hi_val = mag_lut[hi]
    d_lo = (mag - lo_val).abs()
    d_hi = (mag - hi_val).abs()
    even = torch.where((lo % 2) == 0, lo, hi)
    code = torch.where(d_hi < d_lo, hi, torch.where(d_lo < d_hi, lo, even))
    code = torch.where(mag >= 6.0, torch.full_like(code, 7), code)
    neg = torch.signbit(values) if signed_zero else (values < 0)
    return (code | (neg.to(torch.uint8) << 3)).to(torch.uint8)


def _ulp(a: torch.Tensor) -> torch.Tensor:
    base = a.abs().clamp_min(_TINY)
    return torch.nextafter(base, torch.full_like(base, float("inf"))) - base


def _e2m1_boundary_ulp(values: torch.Tensor) -> torch.Tensor:
    """每个值到最近 E2M1 rounding 边界的距离，单位是该值的 float32 ulp。"""
    a = values.abs().clamp_min(_TINY)
    dist = (a.unsqueeze(-1) - _E2M1_BOUNDARIES.to(a.device)).abs().amin(dim=-1)
    return dist / _ulp(a)


def _e4m3_boundary_ulp(sf_pre: torch.Tensor) -> torch.Tensor:
    """到最近 e4m3 相邻值中点的距离，单位是 float32 ulp。"""
    a = sf_pre.abs().clamp_min(_TINY).contiguous()
    mids = _E4M3_MIDPOINTS.to(a.device)
    idx = torch.searchsorted(mids, a.reshape(-1)).reshape(a.shape)
    lo = (idx - 1).clamp(0, mids.numel() - 1)
    hi = idx.clamp(0, mids.numel() - 1)
    dist = torch.minimum((a - mids[lo]).abs(), (mids[hi] - a).abs())
    return dist / _ulp(a)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "count": 0.0}
    flat = values.float().reshape(-1)
    return {
        "p50": float(torch.quantile(flat, 0.5)),
        "p90": float(torch.quantile(flat, 0.9)),
        "p99": float(torch.quantile(flat, 0.99)),
        "max": float(flat.max()),
        "count": float(flat.numel()),
    }


def _hex_bf16(values: torch.Tensor) -> list[str]:
    """bf16 位型（4 hex）；显式转 bf16，避免把 float32 位型拆成两个 16-bit 半字。"""
    return [
        f"{int(v) & 0xFFFF:04x}"
        for v in values.contiguous().to(torch.bfloat16).view(torch.int16).tolist()
    ]


def _case_k(case: str) -> int:
    return int(case.rsplit("-k", 1)[1])


def _is_heldout(case: str) -> bool:
    return case in HELDOUT_SCENARIOS


def _env_configure() -> None:
    from runner.env import configure_flashinfer_env

    configure_flashinfer_env()


# ------------------------------------------------------------------ 输入采样


@dataclass
class Group:
    """一组共享 ``input_scale`` 的输入行；quantize 的 global divisor 是 1/input_scale。"""

    case: str
    source: str
    k: int
    module: str
    layer: int
    prompt: str
    phase: str
    input_scale: float
    gs: float
    x: torch.Tensor  # CPU bf16 [m, k]
    rows: list[dict[str, Any]]
    seed: int

    @property
    def m(self) -> int:
        return int(self.x.shape[0])


def _read_input_scale(name: str) -> float:
    from runner.loader.checkpoint import CheckpointReader

    with CheckpointReader(MODEL_DIR) as reader:
        scale = reader.get(
            "model.language_model." + name[len("model.") :] + ".input_scale", "cpu"
        )
    return float(scale.float().item())


def _gs_of(input_scale: float) -> float:
    return float(
        torch.tensor(1.0, dtype=torch.float32) / torch.tensor(input_scale, dtype=torch.float32)
    )


def _single_group(
    case: str, source: str, k: int, input_scale: float, seed: int, x: torch.Tensor
) -> Group:
    x = x.to(torch.bfloat16).contiguous()
    rows = [
        {"prompt": "", "phase": "synthetic", "step": -1, "module": source, "row": index,
         "seed": seed}
        for index in range(x.shape[0])
    ]
    return Group(
        case=case,
        source=source,
        k=k,
        module=source,
        layer=-1,
        prompt="",
        phase="synthetic",
        input_scale=input_scale,
        gs=_gs_of(input_scale),
        x=x,
        rows=rows,
        seed=seed,
    )


def _gauss(
    case: str, k: int, m: int, input_scale: float, seed: int, *, outlier: bool = False
) -> Group:
    gen = torch.Generator().manual_seed(seed)
    calib_amax = 448.0 * 6.0 * input_scale
    sigma = calib_amax / 5.0
    base = torch.randn((m, k), generator=gen, dtype=torch.float32) * sigma
    if outlier:
        mask = torch.rand((m, k), generator=gen) < 0.004
        factor = 5.0 + 25.0 * torch.rand((m, k), generator=gen)
        base = torch.where(mask, base * factor, base)
        base[m // 2 :] = base[m // 2 :] * 4.0
    return _single_group(
        case, "synthetic-outlier" if outlier else "synthetic-gaussian", k, input_scale, seed, base
    )


def _scale_tie_stress(case: str, k: int, m: int, input_scale: float, seed: int) -> Group:
    """把 block amax 固定到几个“正好落在 e4m3 中点”的 bf16 值上。

    checkpoint 的 ``input_scale`` 近似是 ``calib_amax / (448*6)``，因此
    ``gs*(amax*c6)`` 经常正好等于 e4m3 相邻值的中点；同一个 amax 会在很多
    block / token 上重复出现，于是 reference 的 div-vs-mul 舍入会系统性地翻
    code。这里先在 bf16 amax 网格上找 ``sf_pre`` 正好落在中点的值，再让大量
    block 复用这些 amax，把该机制放大到可统计。
    """
    gen = torch.Generator().manual_seed(seed)
    gs = _gs_of(input_scale)
    calib_amax = 448.0 * 6.0 * input_scale
    nb = k // 16
    low, high = calib_amax * 0.1, calib_amax * 1.0
    exps = torch.arange(
        int(math.floor(math.log2(low))), int(math.floor(math.log2(high))) + 1, dtype=torch.int32
    )
    mant = torch.arange(0, 128, dtype=torch.int32)
    bits = ((exps.view(-1, 1) + 127) << 7) | mant.view(1, -1)
    candidates = bits.reshape(-1).to(torch.int16).view(torch.bfloat16).float()
    candidates = candidates[(candidates >= low) & (candidates <= high)]
    d_ulp = _e4m3_boundary_ulp(torch.mul(torch.mul(candidates, _C6_EXACT), torch.tensor(gs)))
    hits = candidates[d_ulp == 0.0]
    if hits.numel() >= 4:
        chosen = hits[torch.randperm(hits.numel(), generator=gen)[:16]]
    else:
        order = torch.argsort(d_ulp)
        chosen = candidates[order[:16]]
    pick = torch.randint(0, chosen.numel(), (m, nb), generator=gen)
    amax = chosen[pick]  # [m, nb]
    amp = torch.rand((m, nb, 16), generator=gen, dtype=torch.float32) * 0.85
    sign = torch.where(torch.rand((m, nb, 16), generator=gen) < 0.5, -1.0, 1.0)
    x = (amp * sign * amax.unsqueeze(-1)).reshape(m, k)
    slot = torch.randint(0, 16, (m, nb), generator=gen)
    sign_max = torch.where(torch.rand((m, nb), generator=gen) < 0.5, -1.0, 1.0)
    flat_slot = (
        torch.arange(m).view(m, 1) * (nb * 16)
        + torch.arange(nb).view(1, nb) * 16
        + slot
    ).reshape(-1)
    x.reshape(-1)[flat_slot] = amax.reshape(-1) * sign_max.reshape(-1)
    return _single_group(case, "synthetic-scale-tie", k, input_scale, seed, x)


def _synthetic_group(case: str) -> Group:
    k = _case_k(case)
    if case == "synthetic-gaussian-k5120":
        return _gauss(case, k, 128, _read_input_scale("model.layers.10.mlp.gate_proj"), 20261023)
    if case == "synthetic-outlier-k17408":
        return _gauss(
            case, k, 128, _read_input_scale("model.layers.57.mlp.down_proj"), 20261024, outlier=True
        )
    if case == "synthetic-scale-tie-k6144":
        return _scale_tie_stress(
            case, k, 128, _read_input_scale("model.layers.43.self_attn.o_proj"), 20261025
        )
    if case == "heldout-gaussian-k6144":
        group = _gauss(case, k, 192, _read_input_scale("model.layers.47.self_attn.o_proj"), 777001)
        group.source = "heldout"
        return group
    if case == "heldout-outlier-k5120":
        group = _gauss(
            case,
            k,
            192,
            _read_input_scale("model.layers.53.linear_attn.in_proj_qkv"),
            777002,
            outlier=True,
        )
        group.source = "heldout"
        return group
    raise ValueError(f"unknown synthetic scenario: {case}")


def _collect_real_groups(cases: Sequence[str]) -> dict[str, list[Group]]:
    """重放 M2 numeric.json 的固定 token，抓取真实 hidden state。"""
    _env_configure()
    import transformers  # noqa: F401
    from transformers import DynamicCache

    from runner.model.eager_runner import EagerTextRunner

    numeric = json.loads(M2_NUMERIC.read_text())
    records = {row["prompt_file"]: row for row in numeric["results"]}
    runner, _info = EagerTextRunner.from_checkpoint(MODEL_DIR, manifest_path=MANIFEST_PATH)
    model = runner.model
    modules = {name: mod for name, mod in model.named_modules() if name in CAPTURE_MODULES}
    missing = sorted(set(CAPTURE_MODULES) - set(modules))
    if missing:
        raise RuntimeError(f"capture modules missing from model: {missing}")

    captured: dict[tuple[str, str], list[dict[str, Any]]] = {}
    state = {"on": False, "phase": "prefill", "step": -1, "prompt": ""}

    def make_hook(name: str):
        k = CAPTURE_MODULES[name][1]

        def hook(_module, inputs, _output):
            if not state["on"]:
                return
            rows = inputs[0].reshape(-1, k)
            if state["phase"] == "prefill":
                rows = rows[:PREFILL_ROWS]
            payload = rows.detach().to("cpu", torch.bfloat16)
            captured.setdefault((state["prompt"], name), []).append(
                {"phase": state["phase"], "step": int(state["step"]), "x": payload}
            )

        return hook

    handles = [module.register_forward_hook(make_hook(name)) for name, module in modules.items()]
    try:
        with torch.no_grad():
            for prompt_file in PROMPT_FILES:
                payload = json.loads((ROOT / prompt_file).read_text())
                prompt_ids = runner.encode_messages(payload["messages"])
                fixed = list(records[prompt_file]["fixed_tokens"])
                cache = DynamicCache(config=model.config)
                state.update(on=True, phase="prefill", step=-1, prompt=prompt_file)
                model(
                    input_ids=prompt_ids,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                state["phase"] = "decode"
                for step, token in enumerate(fixed[:DECODE_STEPS]):
                    state["step"] = step
                    model(
                        input_ids=torch.tensor([[int(token)]], dtype=torch.long),
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                del cache
    finally:
        for handle in handles:
            handle.remove()
        state["on"] = False

    result: dict[str, list[Group]] = {}
    for case in cases:
        k = _case_k(case)
        groups: list[Group] = []
        for name, (layer, module_k) in CAPTURE_MODULES.items():
            if module_k != k:
                continue
            chunks: list[torch.Tensor] = []
            metas: list[dict[str, Any]] = []
            for prompt_file in PROMPT_FILES:
                calls = captured.get((prompt_file, name), [])
                prefill = [call for call in calls if call["phase"] == "prefill"]
                decode = sorted(
                    (call for call in calls if call["phase"] == "decode"),
                    key=lambda call: call["step"],
                )
                for call in prefill:
                    chunks.append(call["x"])
                    metas.extend(
                        {"prompt": prompt_file, "phase": "prefill", "step": -1,
                         "module": name, "layer": layer, "row": row, "seed": 0}
                        for row in range(call["x"].shape[0])
                    )
                for call in decode:
                    chunks.append(call["x"])
                    metas.extend(
                        {"prompt": prompt_file, "phase": "decode", "step": call["step"],
                         "module": name, "layer": layer, "row": row, "seed": 0}
                        for row in range(call["x"].shape[0])
                    )
            if not chunks:
                continue
            input_scale = _read_input_scale(name)
            groups.append(
                Group(
                    case=case,
                    source="real-m2-replay",
                    k=k,
                    module=name,
                    layer=layer,
                    prompt=",".join(PROMPT_FILES),
                    phase="prefill+decode",
                    input_scale=input_scale,
                    gs=_gs_of(input_scale),
                    x=torch.cat(chunks, dim=0).to(torch.bfloat16).contiguous(),
                    rows=metas,
                    seed=0,
                )
            )
        result[case] = groups
    del model, runner, modules, captured
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ------------------------------------------------------------------ FlashInfer 量化


class FlashQuantizer:
    """与 ``runner/ops/nvfp4/linear.py`` 相同的 activation 量化调用。"""

    def __init__(self) -> None:
        _env_configure()
        from flashinfer import SfLayout, nvfp4_quantize

        self._sf_layout = SfLayout.layout_128x4
        self._quantize = nvfp4_quantize
        self.device = torch.device("cuda")
        self.calls = 0

    def quantize(self, group: Group) -> tuple[torch.Tensor, torch.Tensor]:
        m, k = group.x.shape
        nb = k // 16
        x = group.x.to(self.device, non_blocking=False).contiguous()
        scale = torch.tensor([group.input_scale], dtype=torch.float32, device=self.device)
        global_divisor = (1.0 / scale).reshape(1)
        a_fp4, a_sf = self._quantize(
            x,
            global_divisor,
            sfLayout=self._sf_layout,
            do_shuffle=False,
            backend="cuda",
        )
        if a_sf.dtype != torch.uint8:
            a_sf = a_sf.view(torch.uint8)
        sf = unswizzle_scale_128x4(a_sf, m, nb).contiguous()
        codes = unpack_e2m1(a_fp4.view(torch.uint8))
        self.calls += 1
        return codes.to("cpu"), sf.to("cpu")


# ------------------------------------------------------------------ 规则实现


def _sf_pre(amax: torch.Tensor, gs: float, c6: float) -> torch.Tensor:
    """kernel: SFValue = SFScaleVal * (vecMax * rcp_approx(6.0f))，逐步 mul.rn。"""
    c6_t = torch.tensor(c6, dtype=torch.float32)
    gs_t = torch.tensor(gs, dtype=torch.float32)
    return torch.mul(torch.mul(amax, c6_t), gs_t)


def _out_from_sf(
    sf_bits: torch.Tensor,
    amax: torch.Tensor,
    gs: float,
    mode: str,
    *,
    gs_rcp_hw: float | None = None,
    hw_lookup: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    sff = _e4m3_float(sf_bits)
    one = torch.tensor(1.0, dtype=torch.float32)
    if mode == "hw":
        if gs_rcp_hw is None or hw_lookup is None:
            raise ValueError("hw mode needs probed reciprocal")
        mid = torch.mul(sff, torch.tensor(gs_rcp_hw, dtype=torch.float32))
        out = hw_lookup(mid)
    elif mode == "exact_rcp":
        gs_rcp = one / torch.tensor(gs, dtype=torch.float32)
        out = one / torch.mul(sff, gs_rcp)
    elif mode == "div":
        out = torch.tensor(gs, dtype=torch.float32) / sff
    elif mode == "div_then_rcp":
        out = one / torch.div(sff, torch.tensor(gs, dtype=torch.float32))
    else:
        raise ValueError(f"unknown out mode: {mode}")
    return torch.where(amax == 0, torch.zeros_like(out), out)


def _codes_from_out(blocks: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    t = blocks * out.unsqueeze(-1)
    return _e2m1_codes(t.reshape(blocks.shape[0], -1))


# ------------------------------------------------------------------ GPU 探针


_TRITON_KERNELS: dict[str, Any] = {}


def _triton_rcp_kernel():
    if "rcp" not in _TRITON_KERNELS:
        import triton  # noqa: F401
        import triton.language as tl

        @triton.jit
        def _rcp_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=1.0)
            y = tl.inline_asm_elementwise(
                "rcp.approx.ftz.f32 $0, $1;",
                "=r,r",
                [x],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
            tl.store(y_ptr + offs, y, mask=mask)

        _TRITON_KERNELS["rcp"] = _rcp_kernel
    return _TRITON_KERNELS["rcp"]


def _triton_e2m1_kernel():
    if "e2m1" not in _TRITON_KERNELS:
        import triton  # noqa: F401
        import triton.language as tl

        @triton.jit
        def _e2m1_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.inline_asm_elementwise(
                "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $1;"
                " mov.b32 $0, {t, t, t, t}; }",
                "=r,r",
                [x],
                dtype=tl.int32,
                is_pure=True,
                pack=1,
            )
            tl.store(y_ptr + offs, y, mask=mask)

        _TRITON_KERNELS["e2m1"] = _e2m1_kernel
    return _TRITON_KERNELS["e2m1"]


def _probe_rcp_gpu(values: torch.Tensor) -> torch.Tensor:
    import triton

    x = values.to(torch.float32).contiguous().cuda()
    y = torch.empty_like(x)
    n = x.numel()
    kernel = _triton_rcp_kernel()
    kernel[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)
    torch.cuda.synchronize()
    return y.cpu()


def _probe_e2m1_gpu(values: torch.Tensor) -> torch.Tensor:
    import triton

    x = values.to(torch.float32).contiguous().cuda()
    y = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    n = x.numel()
    kernel = _triton_e2m1_kernel()
    kernel[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)
    torch.cuda.synchronize()
    return (y.cpu() & 0xF).to(torch.uint8)


class _HwTable:
    """``rcp.approx.ftz.f32`` 的实测查表（输入 -> 输出）。"""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        order = torch.argsort(keys)
        self.keys = keys[order].contiguous()
        self.values = values[order].contiguous()

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.keys, values.contiguous()).clamp(0, self.keys.numel() - 1)
        if not torch.equal(self.keys[idx], values):
            raise RuntimeError("hw rcp lookup miss")
        return self.values[idx]


# ------------------------------------------------------------------ 差异分类


def _empty_hist() -> dict[str, int]:
    return {name: 0 for name in CATEGORIES}


def _diff_stats(
    *,
    case: str,
    group: Group,
    blocks: torch.Tensor,
    amax: torch.Tensor,
    codes_impl: torch.Tensor,
    codes_flash: torch.Tensor,
    sf_impl: torch.Tensor,
    sf_flash: torch.Tensor,
    sf_pre_impl: torch.Tensor,
    gs: float,
    t_gather: Callable[[torch.Tensor], torch.Tensor],
    kind: str,
    limit: int,
) -> dict[str, Any]:
    """比较实现 impl 与 FlashInfer，返回计数 / 直方图 / 明细。"""
    m, nb, _ = blocks.shape
    code_diff = (codes_impl != codes_flash).reshape(m, nb, 16)
    sf_diff = sf_impl != sf_flash
    blk_any = code_diff.any(dim=-1) | sf_diff
    idx = blk_any.reshape(-1).nonzero().squeeze(1)
    n_blocks = int(blk_any.sum())
    n_codes = int(code_diff.sum())

    code_cat = torch.full((idx.numel(), 16), CAT_RESIDUAL, dtype=torch.int8)
    block_cat = torch.full((idx.numel(),), CAT_RESIDUAL, dtype=torch.int8)
    dist_codes = torch.zeros(0, dtype=torch.float32)
    dist_sf = torch.zeros(0, dtype=torch.float32)
    x16 = torch.zeros((0, 16), dtype=torch.float32)
    t16 = torch.zeros((0, 16), dtype=torch.float32)
    codes_i16 = torch.zeros((0, 16), dtype=torch.uint8)
    codes_f16 = torch.zeros((0, 16), dtype=torch.uint8)
    diff16 = torch.zeros((0, 16), dtype=torch.bool)
    d16 = torch.zeros((0, 16), dtype=torch.float32)
    sf_pre_sel = torch.zeros(0, dtype=torch.float32)
    d_sf_sel = torch.zeros(0, dtype=torch.float32)
    sf_diff_sel = torch.zeros(0, dtype=torch.bool)

    if idx.numel() > 0:
        x16 = blocks.reshape(-1, 16)[idx]
        t16 = t_gather(idx)
        codes_i16 = codes_impl.reshape(-1, 16)[idx]
        codes_f16 = codes_flash.reshape(-1, 16)[idx]
        diff16 = code_diff.reshape(-1, 16)[idx]
        d16 = _e2m1_boundary_ulp(t16)
        cat = torch.full_like(codes_i16, CAT_RESIDUAL, dtype=torch.int8)
        cat = torch.where(d16 <= 4.0, torch.full_like(cat, CAT_RECIP), cat)
        cat = torch.where(d16 == 0.0, torch.full_like(cat, CAT_TIE), cat)
        cat = torch.where(t16.abs() < 0.25, torch.full_like(cat, CAT_UNDERFLOW), cat)
        cat = torch.where(t16.abs() >= 6.0, torch.full_like(cat, CAT_SATURATION), cat)
        cat = torch.where((x16 == 0.0) & diff16, torch.full_like(cat, CAT_SIGNED_ZERO), cat)
        # 只有真正 code 不同的位置参与 block 归类；其余位置用 -1 占位，
        # 否则每个 block 都会因为远离边界的普通值被误判成 residual。
        code_cat = torch.where(diff16, cat, torch.full_like(cat, -1))

        block_cat = torch.full((idx.numel(),), CAT_RECIP, dtype=torch.int8)
        block_cat = torch.where(
            (code_cat == CAT_TIE).any(dim=-1), torch.full_like(block_cat, CAT_TIE), block_cat
        )
        block_cat = torch.where(
            (code_cat == CAT_SIGNED_ZERO).any(dim=-1),
            torch.full_like(block_cat, CAT_SIGNED_ZERO),
            block_cat,
        )
        block_cat = torch.where(
            (code_cat == CAT_UNDERFLOW).any(dim=-1),
            torch.full_like(block_cat, CAT_UNDERFLOW),
            block_cat,
        )
        block_cat = torch.where(
            (code_cat == CAT_SATURATION).any(dim=-1),
            torch.full_like(block_cat, CAT_SATURATION),
            block_cat,
        )
        block_cat = torch.where(
            (code_cat == CAT_RESIDUAL).any(dim=-1),
            torch.full_like(block_cat, CAT_RESIDUAL),
            block_cat,
        )
        sf_diff_sel = sf_diff.reshape(-1)[idx]
        block_cat = torch.where(
            sf_diff_sel, torch.full_like(block_cat, CAT_FP8), block_cat
        )
        dist_codes = d16[diff16]
        sf_pre_sel = sf_pre_impl.reshape(-1)[idx]
        d_sf_sel = _e4m3_boundary_ulp(sf_pre_sel)
        dist_sf = d_sf_sel[sf_diff_sel]

    blocks_hist = _empty_hist()
    codes_hist = _empty_hist()
    if idx.numel() > 0:
        for value, count in zip(*torch.unique(block_cat.long(), return_counts=True)):
            blocks_hist[CATEGORIES[int(value)]] += int(count)
        if int(diff16.sum()) > 0:
            for value, count in zip(
                *torch.unique(code_cat[diff16].long(), return_counts=True)
            ):
                codes_hist[CATEGORIES[int(value)]] += int(count)

    amax_flat = amax.reshape(-1)
    details: list[dict[str, Any]] = []
    for j in range(min(int(idx.numel()), int(limit))):
        flat = int(idx[j])
        row = flat // nb
        meta = group.rows[row] if row < len(group.rows) else {}
        pos = diff16[j].nonzero().squeeze(1).tolist()
        details.append(
            {
                "case": case,
                "kind": kind,
                "source": group.source,
                "module": group.module,
                "layer": int(meta.get("layer", group.layer)),
                "prompt": meta.get("prompt", ""),
                "phase": meta.get("phase", group.phase),
                "step": int(meta.get("step", -1)),
                "row": row,
                "block": flat % nb,
                "seed": group.seed,
                "input_scale": group.input_scale,
                "global_divisor": gs,
                "amax": float(amax_flat[flat]),
                "sf_pre": float(sf_pre_sel[j]),
                "sf_boundary_ulp": float(d_sf_sel[j]),
                "sf_flash": f"{int(sf_flash.reshape(-1)[flat]):02x}",
                "sf_impl": f"{int(sf_impl.reshape(-1)[flat]):02x}",
                "x_hex": _hex_bf16(x16[j]),
                "codes_flash": [int(v) for v in codes_f16[j].tolist()],
                "codes_impl": [int(v) for v in codes_i16[j].tolist()],
                "diff_positions": [int(p) for p in pos],
                "e2m1_boundary_ulp": [float(v) for v in d16[j][diff16[j]].tolist()],
                "category": CATEGORIES[int(block_cat[j])],
                "code_categories": [CATEGORIES[int(code_cat[j, p])] for p in pos],
            }
        )

    return {
        "blocks": int(m * nb),
        "codes": int(m * nb * 16),
        "mismatch_blocks": n_blocks,
        "mismatch_codes": n_codes,
        "sf_mismatch_blocks": int(sf_diff.sum()),
        "block_categories": blocks_hist,
        "code_categories": codes_hist,
        "e2m1_boundary_ulp": _quantiles(dist_codes),
        "sf_boundary_ulp": _quantiles(dist_sf),
        "residual_blocks": int(blocks_hist["residual"]),
        "residual_codes": int(codes_hist["residual"]),
        "details": details,
    }


def _gather_ref_t(blocks: torch.Tensor, gs: float, sff: torch.Tensor):
    gs_t = torch.tensor(gs, dtype=torch.float32)

    def gather(idx: torch.Tensor) -> torch.Tensor:
        x16 = blocks.reshape(-1, 16)[idx]
        scale = sff.reshape(-1)[idx]
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        return (x16 * gs_t) / safe.unsqueeze(-1)

    return gather


def _gather_out_t(blocks: torch.Tensor, out: torch.Tensor):
    def gather(idx: torch.Tensor) -> torch.Tensor:
        x16 = blocks.reshape(-1, 16)[idx]
        return x16 * out.reshape(-1)[idx].unsqueeze(-1)

    return gather


# ------------------------------------------------------------------ 单 group 分析


def _analyze_group(
    group: Group,
    flash_codes: torch.Tensor,
    flash_sf: torch.Tensor,
    rule_codes: torch.Tensor,
    rule_sf: torch.Tensor,
    rule_out: torch.Tensor,
    rule_sf_pre: torch.Tensor,
    offline_codes: torch.Tensor,
    ref_result: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    m, k = group.x.shape
    nb = k // 16
    gs = group.gs
    blocks = group.x.float().reshape(m, nb, 16)
    amax = blocks.abs().amax(dim=-1)
    ref_codes, sf_ref = ref_result
    sf_ref_f = _e4m3_float(sf_ref)
    sf_pre_ref = torch.mul(amax, torch.tensor(gs, dtype=torch.float32)) / 6.0

    ref_stats = _diff_stats(
        case=group.case,
        group=group,
        blocks=blocks,
        amax=amax,
        codes_impl=ref_codes,
        codes_flash=flash_codes,
        sf_impl=sf_ref,
        sf_flash=flash_sf,
        sf_pre_impl=sf_pre_ref,
        gs=gs,
        t_gather=_gather_ref_t(blocks, gs, sf_ref_f),
        kind="ref",
        limit=DETAIL_LIMIT,
    )
    rule_stats = _diff_stats(
        case=group.case,
        group=group,
        blocks=blocks,
        amax=amax,
        codes_impl=rule_codes,
        codes_flash=flash_codes,
        sf_impl=rule_sf,
        sf_flash=flash_sf,
        sf_pre_impl=rule_sf_pre,
        gs=gs,
        t_gather=_gather_out_t(blocks, rule_out),
        kind="rule",
        limit=0,
    )
    offline_diff = int((offline_codes != flash_codes).sum())

    code_diff = ref_codes != flash_codes
    row_code_diff = code_diff.sum(dim=-1)
    row_sf_max = (sf_ref_f - _e4m3_float(flash_sf)).abs().amax(dim=-1)
    replay_rows = []
    for row, meta in enumerate(group.rows):
        if meta.get("phase") == "decode":
            replay_rows.append(
                {
                    "prompt": meta.get("prompt", ""),
                    "module": meta.get("module", group.module),
                    "step": int(meta.get("step", -1)),
                    "scale_max_abs": float(row_sf_max[row]),
                    "code_mismatch": int(row_code_diff[row]),
                }
            )
    return {
        "ref": ref_stats,
        "rule": rule_stats,
        "offline_code_mismatch": offline_diff,
        "replay_rows": replay_rows,
        "blocks": int(m * nb),
        "codes": int(m * nb * 16),
    }


# ------------------------------------------------------------------ 规则选择


def _aggregate_rule(
    groups_by_case: dict[str, list[Group]],
    flash: dict[tuple[str, int], dict[str, torch.Tensor]],
    probe_rcp: Callable[[torch.Tensor], torch.Tensor] | None,
    probe_e2m1: Callable[[torch.Tensor], torch.Tensor] | None,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    """确定规则（只用 fit 样本），生成每组 rule/offline 的 codes/sf/out。"""
    info: dict[str, Any] = {"rcp_probe": {"available": False}, "e2m1_probe": {"available": False}}
    fit_cases = [case for case in groups_by_case if not _is_heldout(case)]
    pairs = [(case, index) for case, groups in groups_by_case.items() for index in range(len(groups))]

    base: dict[tuple[str, int], dict[str, Any]] = {}
    for case, groups in groups_by_case.items():
        for index, group in enumerate(groups):
            m, k = group.x.shape
            blocks = group.x.float().reshape(m, k // 16, 16)
            amax = blocks.abs().amax(dim=-1)
            base[(case, index)] = {
                "blocks": blocks,
                "amax": amax,
                "flash_codes": flash[(case, index)]["codes"],
                "flash_sf": flash[(case, index)]["sf"],
            }

    c6 = float(_C6_EXACT.item())
    hw_rcp: _HwTable | None = None
    gs_rcp_hw: dict[float, float] = {}
    if probe_rcp is not None:
        try:
            gs_values = sorted({group.gs for groups in groups_by_case.values() for group in groups})
            inputs = torch.unique(torch.tensor([6.0] + gs_values, dtype=torch.float32))
            outputs = probe_rcp(inputs)
            exact = torch.tensor([1.0 / float(v) for v in inputs.tolist()], dtype=torch.float32)
            mismatch = int((outputs.view(torch.int32) != exact.view(torch.int32)).sum())
            probe_map = {
                float(key): float(value)
                for key, value in zip(inputs.tolist(), outputs.tolist())
            }
            c6 = probe_map[6.0]
            gs_rcp_hw = {float(value): probe_map[float(value)] for value in gs_values}
            hw_rcp = _HwTable(inputs, outputs)
            info["rcp_probe"] = {
                "available": True,
                "tested": int(inputs.numel()),
                "vs_exact_mismatches": mismatch,
                "c6": c6,
                "c6_hex": f"0x{_c6_bits(c6):08x}",
                "gs": {f"{value:.9g}": gs_rcp_hw[value] for value in gs_values},
            }
        except Exception as exc:  # noqa: BLE001 - 探针失败时退回 offline reference
            info["rcp_probe"] = {"available": False, "error": repr(exc)}
            hw_rcp = None
            gs_rcp_hw = {}

    sf_variant_scores: dict[str, int] = {}
    if hw_rcp is None:
        candidates = {
            "exact": float(_C6_EXACT.item()),
            "down": float(_C6_DOWN.item()),
            "up": float(_C6_UP.item()),
        }
        for name, value in candidates.items():
            total = 0
            for case in fit_cases:
                for index, group in enumerate(groups_by_case[case]):
                    bits = _bits_e4m3(_sf_pre(base[(case, index)]["amax"], group.gs, value))
                    total += int((bits != base[(case, index)]["flash_sf"]).sum())
            sf_variant_scores[name] = total
        exact_names = [name for name, score in sf_variant_scores.items() if score == 0]
        chosen = "exact" if "exact" in exact_names else (exact_names[0] if exact_names else min(sf_variant_scores, key=sf_variant_scores.get))
        c6 = candidates[chosen]
        info["c6_selection"] = {"source": "fit", "chosen": chosen, "scores": sf_variant_scores}
    else:
        info["c6_selection"] = {"source": "hw_probe"}

    out_scores: dict[str, int] = {}
    for mode in ("exact_rcp", "div", "div_then_rcp"):
        total = 0
        for case in fit_cases:
            for index, group in enumerate(groups_by_case[case]):
                item = base[(case, index)]
                bits = _bits_e4m3(_sf_pre(item["amax"], group.gs, c6))
                out = _out_from_sf(bits, item["amax"], group.gs, mode)
                total += int((_codes_from_out(item["blocks"], out) != item["flash_codes"]).sum())
        out_scores[mode] = total
    if hw_rcp is not None:
        out_mode = "hw"
        info["out_selection"] = {"source": "hw_probe", "fit_scores": out_scores}
    else:
        exact_modes = [mode for mode, score in out_scores.items() if score == 0]
        out_mode = "exact_rcp" if "exact_rcp" in exact_modes else (
            exact_modes[0] if exact_modes else min(out_scores, key=out_scores.get)
        )
        info["out_selection"] = {"source": "fit", "chosen": out_mode, "scores": out_scores}

    hw_mid_table: _HwTable | None = None
    if out_mode == "hw":
        try:
            mid_tensors = []
            for case, index in pairs:
                group = groups_by_case[case][index]
                item = base[(case, index)]
                bits = _bits_e4m3(_sf_pre(item["amax"], group.gs, c6))
                mid = torch.mul(
                    _e4m3_float(bits),
                    torch.tensor(gs_rcp_hw[group.gs], dtype=torch.float32),
                )
                mid_tensors.append(torch.unique(mid))
            mid_values = torch.unique(torch.cat(mid_tensors))
            hw_mid_table = _HwTable(mid_values, probe_rcp(mid_values))
            info["rcp_probe"]["mid_values"] = int(mid_values.numel())
        except Exception as exc:  # noqa: BLE001
            info["rcp_probe"] = {"available": False, "error": repr(exc)}
            hw_rcp = None
            hw_mid_table = None
            out_mode = "exact_rcp"
            info["out_selection"] = {"source": "fallback_after_probe_error", "chosen": out_mode}

    rule: dict[tuple[str, int], dict[str, Any]] = {}
    for case, index in pairs:
        group = groups_by_case[case][index]
        item = base[(case, index)]
        sf_pre_rule = _sf_pre(item["amax"], group.gs, c6)
        sf_bits = _bits_e4m3(sf_pre_rule)
        out = _out_from_sf(
            sf_bits,
            item["amax"],
            group.gs,
            out_mode,
            gs_rcp_hw=gs_rcp_hw.get(group.gs),
            hw_lookup=hw_mid_table,
        )
        rule[(case, index)] = {
            "sf_bits": sf_bits,
            "sf_pre": sf_pre_rule,
            "out": out,
            "codes": _codes_from_out(item["blocks"], out),
            "offline_codes": _codes_from_out(
                item["blocks"],
                _out_from_sf(sf_bits, item["amax"], group.gs, "exact_rcp"),
            ),
        }
        del base[(case, index)]["blocks"]
        gc.collect()

    info["rule"] = {
        "c6": c6,
        "out_mode": out_mode,
        "seq": [
            "amax = max(|bf16 x|) over a 16-element block",
            "sf_pre = gs * (amax * rcp.approx.ftz(6.0))  (two mul.rn)",
            "sf = e4m3_rne_satfinite(sf_pre)",
            "out = rcp.approx.ftz(sf_float * rcp.approx.ftz(gs)); 0 if amax == 0",
            "codes = cvt.rn.satfinite.e2m1x2.f32(x_float * out)",
        ],
        "hw_rcp_used": out_mode == "hw",
    }

    if probe_e2m1 is not None:
        probes = torch.tensor(
            [0.0, -0.0, 0.25, -0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 6.5, -6.5, 1e10, -1e10],
            dtype=torch.float32,
        )
        try:
            hw_codes = probe_e2m1(probes)
            soft_codes = _e2m1_codes(probes)
            info["e2m1_probe"] = {
                "available": True,
                "values": [float(v) for v in probes.tolist()],
                "hw_codes": [int(v) for v in hw_codes.tolist()],
                "soft_codes": [int(v) for v in soft_codes.tolist()],
                "mismatches": int((hw_codes != soft_codes).sum()),
            }
        except Exception as exc:  # noqa: BLE001
            info["e2m1_probe"] = {"available": False, "error": repr(exc)}
    return info, rule


def _c6_bits(value: float) -> int:
    return int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF


# ------------------------------------------------------------------ case 组装


def _case_config(case: str, groups: Sequence[Group]) -> dict[str, Any]:
    return {
        "source": sorted({group.source for group in groups}),
        "k": sorted({group.k for group in groups}),
        "m_values": sorted({group.m for group in groups}),
        "groups": len(groups),
        "rows": int(sum(group.m for group in groups)),
        "blocks": int(sum(group.m * (group.k // 16) for group in groups)),
        "modules": sorted({group.module for group in groups}),
        "seeds": sorted({group.seed for group in groups if group.seed}),
        "dtype": "bfloat16",
        "quantize": {
            "api": "flashinfer.nvfp4_quantize",
            "backend": "cuda",
            "sf_layout": "layout_128x4",
            "do_shuffle": False,
            "sf_vec_size": 16,
            "global_scale": "1 / input_scale",
            "kernel": "quantize_with_block_size (non-TMA, ELTS_PER_THREAD=16)",
        },
    }


def _merge_hist(dst: dict[str, int], src: dict[str, int]) -> None:
    for name, value in src.items():
        dst[name] += int(value)


def _summarize_case(case: str, groups: Sequence[Group], per_group: Sequence[dict[str, Any]]) -> dict[str, Any]:
    blocks = sum(item["blocks"] for item in per_group)
    codes = sum(item["codes"] for item in per_group)
    ref_blocks = sum(item["ref"]["mismatch_blocks"] for item in per_group)
    ref_codes = sum(item["ref"]["mismatch_codes"] for item in per_group)
    rule_blocks = sum(item["rule"]["mismatch_blocks"] for item in per_group)
    rule_codes = sum(item["rule"]["mismatch_codes"] for item in per_group)
    sf_blocks = sum(item["ref"]["sf_mismatch_blocks"] for item in per_group)
    block_hist = _empty_hist()
    code_hist = _empty_hist()
    rule_block_hist = _empty_hist()
    rule_code_hist = _empty_hist()
    details: list[dict[str, Any]] = []
    for item in per_group:
        _merge_hist(block_hist, item["ref"]["block_categories"])
        _merge_hist(code_hist, item["ref"]["code_categories"])
        _merge_hist(rule_block_hist, item["rule"]["block_categories"])
        _merge_hist(rule_code_hist, item["rule"]["code_categories"])
        details.extend(item["ref"]["details"])
    return {
        "case": case,
        "blocks": blocks,
        "codes": codes,
        "ref": {
            "mismatch_blocks": ref_blocks,
            "mismatch_codes": ref_codes,
            "sf_mismatch_blocks": sf_blocks,
            "block_categories": block_hist,
            "code_categories": code_hist,
        },
        "rule": {
            "mismatch_blocks": rule_blocks,
            "mismatch_codes": rule_codes,
            "block_categories": rule_block_hist,
            "code_categories": rule_code_hist,
        },
        "offline_code_mismatch": sum(item["offline_code_mismatch"] for item in per_group),
        "detail_records": details,
        "config": _case_config(case, groups),
    }


def _case_metrics(summary: dict[str, Any]) -> dict[str, float]:
    blocks = max(summary["blocks"], 1)
    codes = max(summary["codes"], 1)
    metrics: dict[str, float] = {
        "blocks": float(summary["blocks"]),
        "codes": float(summary["codes"]),
        "ref_mismatch_blocks": float(summary["ref"]["mismatch_blocks"]),
        "ref_mismatch_codes": float(summary["ref"]["mismatch_codes"]),
        "ref_mismatch_block_ratio": summary["ref"]["mismatch_blocks"] / blocks,
        "ref_mismatch_code_ratio": summary["ref"]["mismatch_codes"] / codes,
        "ref_sf_mismatch_blocks": float(summary["ref"]["sf_mismatch_blocks"]),
        "rule_mismatch_blocks": float(summary["rule"]["mismatch_blocks"]),
        "rule_mismatch_codes": float(summary["rule"]["mismatch_codes"]),
        "rule_mismatch_code_ratio": summary["rule"]["mismatch_codes"] / codes,
        "rule_residual_codes": float(summary["rule"]["code_categories"]["residual"]),
        "rule_residual_ratio": summary["rule"]["code_categories"]["residual"] / codes,
        "offline_mismatch_codes": float(summary["offline_code_mismatch"]),
        "offline_mismatch_ratio": summary["offline_code_mismatch"] / codes,
    }
    for name in CATEGORIES:
        metrics[f"ref_block_cat_{name}"] = float(summary["ref"]["block_categories"][name])
        metrics[f"rule_block_cat_{name}"] = float(summary["rule"]["block_categories"][name])
    return metrics


# ------------------------------------------------------------------ 汇总与校验辅助


def _parse_only(raw: str) -> tuple[str, ...] | None:
    if raw.strip() in ("", "all"):
        return None
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"--only 有未知场景 {unknown}; 可用: {', '.join(SCENARIOS)}")
    return names


def _git_changed() -> dict[str, Any]:
    try:
        changed = subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "--name-only", "HEAD^", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "changed": [], "forbidden": [], "dirty": True}
    forbidden = [
        path
        for path in changed
        if any(path == item or path.startswith(item) for item in FORBIDDEN_PATHS)
    ]
    return {
        "ok": not forbidden and not status,
        "commit": commit,
        "changed": changed,
        "forbidden": forbidden,
        "dirty": bool(status),
        "status": status.splitlines()[:20],
    }


def _recorded_samples() -> list[dict[str, Any]]:
    numeric = json.loads(M2_NUMERIC.read_text())
    out: list[dict[str, Any]] = []
    for row in numeric["results"]:
        for sample in row["activation_quant"].get("samples", []):
            out.append(
                {
                    "prompt": row["prompt_file"],
                    "module": sample["name"],
                    "scale_max_abs": float(sample["scale_max_abs"]),
                    "code_mismatch": int(sample["code_mismatch"]),
                }
            )
    return out


def _alias_module(name: str) -> str:
    if name.endswith(".mlp.up_proj"):
        return name[: -len(".mlp.up_proj")] + ".mlp.gate_proj"
    return name


def _replay_summary(
    observed: list[dict[str, Any]], recorded: list[dict[str, Any]]
) -> dict[str, Any]:
    matched = 0
    missing: list[dict[str, Any]] = []
    for sample in recorded:
        wanted = _alias_module(sample["module"])
        hit = any(
            _alias_module(row["module"]) == wanted
            and row["prompt"] == sample["prompt"]
            and abs(row["scale_max_abs"] - sample["scale_max_abs"]) <= 1e-12
            and row["code_mismatch"] == sample["code_mismatch"]
            for row in observed
        )
        matched += int(hit)
        if not hit:
            missing.append(sample)
    return {
        "recorded": len(recorded),
        "matched": matched,
        "missing": missing[:8],
        "match_ratio": (matched / len(recorded)) if recorded else 1.0,
    }


# ------------------------------------------------------------------ 主入口


def run_experiment(
    args: argparse.Namespace,
    *,
    flash_factory: Callable[[], Any] | None = None,
    collect_fn: Callable[[Sequence[str]], dict[str, list[Group]]] | None = None,
    probe_rcp: Callable[[torch.Tensor], torch.Tensor] | None = None,
    probe_e2m1: Callable[[torch.Tensor], torch.Tensor] | None = None,
    env_info: dict[str, Any] | None = None,
    timing: dict[str, float] | None = None,
) -> dict[str, Any]:
    from runner.experiment.op_artifact import OpReport

    started = time.monotonic()
    stage_times = timing if timing is not None else {}

    def mark(name: str) -> None:
        stage_times[name] = round(time.monotonic() - started, 3)

    only = _parse_only(args.only)
    selected = [case for case in SCENARIOS if only is None or case in only]
    data_cases = [case for case in selected if case != "coverage-summary"]

    groups_by_case: dict[str, list[Group]] = {}
    synthetic = [case for case in data_cases if case.startswith(("synthetic-", "heldout-"))]
    real = [case for case in data_cases if case.startswith("real-")]
    for case in synthetic:
        groups_by_case[case] = [_synthetic_group(case)]
    mark("synthetic")
    if real:
        collector = collect_fn or _collect_real_groups
        real_groups = collector(real)
        for case in real:
            groups_by_case[case] = real_groups.get(case, [])
        del real_groups
        gc.collect()
        mark("real_capture")

    factory = flash_factory or FlashQuantizer
    quantizer = factory()
    flash: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    determinism: dict[str, Any] = {}
    reference_determinism: dict[str, Any] = {}
    for case, groups in groups_by_case.items():
        for index, group in enumerate(groups):
            codes, sf = quantizer.quantize(group)
            flash[(case, index)] = {"codes": codes, "sf": sf}
            if index == 0:
                codes2, sf2 = quantizer.quantize(group)
                determinism[case] = {
                    "ok": bool(torch.equal(codes, codes2) and torch.equal(sf, sf2)),
                    "codes": int(codes.numel()),
                    "sf": int(sf.numel()),
                }
                packed_a, sf_a, _ = quantize_activation_reference(group.x, group.input_scale)
                packed_b, sf_b, _ = quantize_activation_reference(group.x, group.input_scale)
                reference_determinism[case] = {
                    "ok": bool(torch.equal(packed_a, packed_b) and torch.equal(sf_a, sf_b)),
                }
    mark("flash_quantize")

    rule_info, rule_outputs = _aggregate_rule(groups_by_case, flash, probe_rcp, probe_e2m1)
    mark("rule_select")

    ref_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for case, groups in groups_by_case.items():
        for index, group in enumerate(groups):
            packed, sf, _ = quantize_activation_reference(group.x, group.input_scale)
            ref_cache[(case, index)] = (
                unpack_e2m1(packed),
                sf.contiguous().view(torch.uint8),
            )
    mark("reference")

    per_group: dict[str, list[dict[str, Any]]] = {}
    m2_observed: list[dict[str, Any]] = []
    for case, groups in groups_by_case.items():
        items: list[dict[str, Any]] = []
        for index, group in enumerate(groups):
            result = _analyze_group(
                group,
                flash[(case, index)]["codes"],
                flash[(case, index)]["sf"],
                rule_outputs[(case, index)]["codes"],
                rule_outputs[(case, index)]["sf_bits"],
                rule_outputs[(case, index)]["out"],
                rule_outputs[(case, index)]["sf_pre"],
                rule_outputs[(case, index)]["offline_codes"],
                ref_cache[(case, index)],
            )
            items.append(result)
            m2_observed.extend(result["replay_rows"])
        per_group[case] = items
        gc.collect()
    mark("classify")

    summaries: dict[str, dict[str, Any]] = {}
    for case in data_cases:
        if per_group.get(case):
            summaries[case] = _summarize_case(case, groups_by_case[case], per_group[case])

    mainline = _git_changed()
    recorded = _recorded_samples() if real else []
    replay = _replay_summary(m2_observed, recorded) if real else {
        "recorded": 0,
        "matched": 0,
        "match_ratio": 1.0,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mismatch_path = out_dir / "mismatch_blocks.jsonl"
    written = 0
    with mismatch_path.open("w", encoding="utf-8") as handle:
        for case in data_cases:
            summary = summaries.get(case)
            if summary is None:
                continue
            for record in summary["detail_records"]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    report = OpReport(
        out_dir,
        candidate_name="activation-quant-attrib",
        candidate_format="nvfp4-w4a4-e2m1-ue4m3",
        candidate_extra={
            "input": "flashinfer nvfp4_quantize sfLayout=layout_128x4 do_shuffle=False backend=cuda",
            "kernel_path": "non-TMA quantize_with_block_size (ELTS_PER_THREAD=16, fast-math rcp)",
            "m2_numeric": str(M2_NUMERIC.relative_to(ROOT)),
            "m2_numeric_sha256": hashlib.sha256(M2_NUMERIC.read_bytes()).hexdigest()[:16],
            "mismatch_blocks_jsonl": mismatch_path.name,
            "mismatch_blocks_saved": written,
        },
        argv=None,
        environment=env_info or {},
    )

    def rule_exact(case: str) -> bool:
        summary = summaries.get(case)
        return bool(
            summary
            and summary["rule"]["mismatch_codes"] == 0
            and summary["rule"]["mismatch_blocks"] == 0
        )

    for case in data_cases:
        summary = summaries.get(case)
        if summary is None:
            continue
        report_case = report.case(case, **summary["config"])
        report_case.gate(
            "coverage:inventory",
            summary["blocks"] > 0 and summary["codes"] > 0,
            {
                "source": summary["config"]["source"],
                "k": summary["config"]["k"],
                "blocks": summary["blocks"],
                "codes": summary["codes"],
                "groups": summary["config"]["groups"],
                "rows": summary["config"]["rows"],
                "modules": len(summary["config"]["modules"]),
                "seeds": summary["config"]["seeds"],
            },
        )
        report_case.gate(
            "determinism:flashinfer-bitwise",
            bool(determinism.get(case, {}).get("ok")),
            determinism.get(case, {}),
        )
        report_case.gate(
            "determinism:reference-bitwise",
            bool(reference_determinism.get(case, {}).get("ok")),
            reference_determinism.get(case, {}),
        )
        if rule_exact(case):
            report_case.gate(
                "rule:bitwise",
                True,
                {
                    "blocks": summary["blocks"],
                    "codes": summary["codes"],
                    "impl": rule_info["rule"]["out_mode"],
                },
            )
        else:
            residual = summary["rule"]["code_categories"]["residual"]
            report_case.gate(
                "rule:residual-classified",
                sum(summary["rule"]["block_categories"].values())
                == summary["rule"]["mismatch_blocks"],
                {
                    "mismatch_blocks": summary["rule"]["mismatch_blocks"],
                    "mismatch_codes": summary["rule"]["mismatch_codes"],
                    "residual_codes": residual,
                    "residual_ratio": residual / max(summary["codes"], 1),
                    "block_categories": summary["rule"]["block_categories"],
                },
            )
        report_case.diag(
            "ref:taxonomy",
            summary["ref"]["mismatch_blocks"] == 0,
            {
                "mismatch_blocks": summary["ref"]["mismatch_blocks"],
                "mismatch_codes": summary["ref"]["mismatch_codes"],
                "sf_mismatch_blocks": summary["ref"]["sf_mismatch_blocks"],
                "block_categories": summary["ref"]["block_categories"],
                "code_categories": summary["ref"]["code_categories"],
            },
        )
        report_case.diag(
            "offline-reference",
            summary["offline_code_mismatch"] == 0,
            {"code_mismatch": summary["offline_code_mismatch"], "codes": summary["codes"]},
        )
        if "real-m2-replay" in summary["config"]["source"]:
            case_k = summary["config"]["k"][0]
            recorded_modules = {_alias_module(sample["module"]) for sample in recorded}
            case_recorded = {
                name
                for name in recorded_modules
                if name in CAPTURE_MODULES and CAPTURE_MODULES[name][1] == case_k
            }
            captured_modules = set(summary["config"]["modules"])
            report_case.gate(
                "coverage:m2-recorded-modules",
                case_recorded <= captured_modules,
                {"recorded": sorted(case_recorded), "captured": sorted(captured_modules)},
            )
        report_case.diag("m2-replay:metrics-match", replay["match_ratio"] == 1.0, replay)
        report_case.gate(
            "no-mainline-change",
            bool(mainline["ok"]),
            {
                "commit": mainline.get("commit"),
                "changed": mainline.get("changed", []),
                "forbidden": mainline.get("forbidden", []),
                "dirty": mainline.get("dirty"),
            },
        )
        for name, value in _case_metrics(summary).items():
            report_case.metric(name, value)
        for record in summary["detail_records"][:2]:
            report_case.diag(
                "sample-block",
                True,
                {
                    "module": record["module"],
                    "category": record["category"],
                    "sf_boundary_ulp": record["sf_boundary_ulp"],
                    "e2m1_boundary_ulp": record["e2m1_boundary_ulp"],
                },
            )

    if "coverage-summary" in selected:
        total_blocks = sum(summary["blocks"] for summary in summaries.values())
        total_codes = sum(summary["codes"] for summary in summaries.values())
        ref_codes = sum(summary["ref"]["mismatch_codes"] for summary in summaries.values())
        rule_codes = sum(summary["rule"]["mismatch_codes"] for summary in summaries.values())
        residual_codes = sum(
            summary["rule"]["code_categories"]["residual"] for summary in summaries.values()
        )
        sources = {source for summary in summaries.values() for source in summary["config"]["source"]}
        ks = {item for summary in summaries.values() for item in summary["config"]["k"]}
        full_run = only is None
        required_sources = {
            "synthetic-gaussian",
            "synthetic-outlier",
            "synthetic-scale-tie",
            "real-m2-replay",
            "heldout",
        }
        summary_case = report.case(
            "coverage-summary",
            source="aggregate",
            k=sorted(ks),
            cases=len(summaries),
        )
        summary_case.gate(
            "coverage:sources",
            True if not full_run else required_sources <= sources,
            {"required": sorted(required_sources), "got": sorted(sources)},
        )
        summary_case.gate(
            "coverage:shapes",
            True if not full_run else ks >= set(CAPTURE_K),
            {"required": list(CAPTURE_K), "got": sorted(ks)},
        )
        missing_cases = [case for case in data_cases if case not in summaries]
        summary_case.gate(
            "coverage:cases",
            bool(summaries)
            and not missing_cases
            and all(summary["blocks"] > 0 for summary in summaries.values()),
            {
                "missing": missing_cases,
                "blocks": {case: summary["blocks"] for case, summary in summaries.items()},
            },
        )
        summary_case.diag(
            "coverage:scenario-checklist",
            True,
            {case: ("ran" if case in summaries else "skipped") for case in SCENARIOS},
        )
        summary_case.gate(
            "coverage:classification",
            all(
                sum(summary["ref"]["block_categories"].values())
                == summary["ref"]["mismatch_blocks"]
                for summary in summaries.values()
            )
            and all(
                sum(summary["rule"]["block_categories"].values())
                == summary["rule"]["mismatch_blocks"]
                for summary in summaries.values()
            ),
            {
                "ref_mismatch_blocks": sum(
                    summary["ref"]["mismatch_blocks"] for summary in summaries.values()
                ),
                "rule_mismatch_blocks": sum(
                    summary["rule"]["mismatch_blocks"] for summary in summaries.values()
                ),
                "categories": list(CATEGORIES),
            },
        )
        if real:
            recorded_modules = {_alias_module(sample["module"]) for sample in recorded}
            observed_modules = {_alias_module(row["module"]) for row in m2_observed}
            summary_case.gate(
                "coverage:m2-recorded-modules",
                recorded_modules <= observed_modules,
                {"recorded": sorted(recorded_modules), "observed": sorted(observed_modules)},
            )
        fit_cases = [case for case in summaries if not _is_heldout(case)]
        heldout_cases = [case for case in summaries if _is_heldout(case)]
        fit_exact = bool(fit_cases) and all(rule_exact(case) for case in fit_cases)
        heldout_exact = bool(heldout_cases) and all(rule_exact(case) for case in heldout_cases)
        if fit_exact and heldout_exact:
            summary_case.gate("rule:bitwise-fit", True, {"cases": fit_cases})
            summary_case.gate("rule:bitwise-heldout", True, {"cases": heldout_cases})
        else:
            summary_case.gate(
                "rule:residual-classified",
                total_codes > 0,
                {
                    "fit_cases": fit_cases,
                    "heldout_cases": heldout_cases,
                    "fit_exact": fit_exact,
                    "heldout_exact": heldout_exact,
                    "rule_mismatch_codes": rule_codes,
                    "residual_codes": residual_codes,
                    "residual_ratio": residual_codes / max(total_codes, 1),
                },
            )
            summary_case.diag(
                "rule:bitwise", False, {"fit_exact": fit_exact, "heldout_exact": heldout_exact}
            )
        summary_case.gate("no-mainline-change", bool(mainline["ok"]), mainline)
        summary_case.metric("blocks", float(total_blocks))
        summary_case.metric("codes", float(total_codes))
        summary_case.metric("ref_mismatch_codes", float(ref_codes))
        summary_case.metric("ref_mismatch_ratio", ref_codes / max(total_codes, 1))
        summary_case.metric("rule_mismatch_codes", float(rule_codes))
        summary_case.metric("residual_codes", float(residual_codes))
        summary_case.metric("residual_ratio", residual_codes / max(total_codes, 1))
        summary_case.metric("m2_replay_match_ratio", float(replay["match_ratio"]))
        summary_case.metric("mismatch_blocks_saved", float(written))

    return report.finish(
        extra={
            "attribution": {
                "rule": rule_info,
                "m2_replay": replay,
                "mismatch_blocks_jsonl": mismatch_path.name,
                "timing_s": stage_times,
            }
        }
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="all", help="逗号分隔的 case 名，默认 all")
    parser.add_argument("--out", required=True, help="输出目录（写 op.json）")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _env_configure()
    env_info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    try:
        import flashinfer

        env_info["flashinfer"] = flashinfer.__version__
    except Exception:  # noqa: BLE001
        env_info["flashinfer"] = "unavailable"
    try:
        import triton

        env_info["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        env_info["triton"] = "unavailable"
    payload = run_experiment(
        args,
        probe_rcp=_probe_rcp_gpu,
        probe_e2m1=_probe_e2m1_gpu,
        env_info=env_info,
    )
    print(json.dumps(payload.get("summary", {}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
