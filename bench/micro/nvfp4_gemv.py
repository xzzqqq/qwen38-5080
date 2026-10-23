#!/usr/bin/env python3
"""NVFP4 M=1 GEMV（CUDA core SIMT）的 op gate + microbench。

设计
----
- 候选：``runner/ops/nvfp4/gemv.py`` + ``gemv_kernel.cu``，直接消费 FlashInfer
  ``nvfp4_quantize`` 的 ``a_fp4``/``a_sf`` 和 checkpoint 的 ``weight_packed`` 与
  **128x4 swizzled** ``weight_scale``（不生成 unsquizzled 副本、不 repack）；
- reference（gate 用，两条互相交叉验证）：
  1) **同一份量化后的 A**：解 FlashInfer 的 ``a_fp4``/``a_sf`` + checkpoint W 的
     128x4 swizzled scale，FP32 matmul，再乘 ``alpha = input_scale * weight_scale_2``；
  2) **M2 冻结语义**：``quantize_activation_reference`` 重新量化同一个 x 得到 a_deq，
     W 用 checkpoint 的**线性** scale（不经过 swizzle）与 ``weight_scale_2``；
     两条 reference 的一致性本身就是 layout/dtype 口径的 gate。
  注意 block scale 必须按 **UE4M3 数值**解释：``uint8`` view 再 ``.float()`` 会拿到
  整数字节值（0x38 → 56.0 而不是 1.0），2026-10-23 的失败就是这个问题；
- 容差在实现前冻结（[decisions](../decisions.md) M2 NVFP4）：
  ``nrmse <= 5e-3``、``cosine >= 0.99999``、``max_rel <= 1e-2``、输出 finite；
- 布局单独验：``layout_probe`` case 用两个合成权重（期望值从字节约定手算，不经过
  reference），分别验 E2M1 nibble 顺序和 128x4 swizzled scale 映射，并记录
  “反约定假设不成立”的判别力证据。

计时口径（重要）
----------------
单独一个 shape 的权重常常小于 L2（GB203 ≈ 64 MB），反复迭代时整块住进 L2，
测出来的“带宽”会超过 960 GB/s 物理上限（tensor-core 那条线在 job-000018 实测：
in_proj_qkv 1425 GB/s、gate_proj 2423 GB/s，全部不可信）。

所以本脚本的**每次计时迭代扫过该 shape 在模型里的全部真实实例**（in_proj_qkv ×48、
gate_proj / down_proj ×64、k_proj+v_proj ×64、lm_head ×1），逐层 launch，工作集
几百 MB 到 3.6 GB，远大于 L2；``mm_fp4`` 基线用**同一批实例、同一个计时边界**，
否则基线也会被 L2 抬高。主指标是 **聚合 ms / 聚合 GB/s**（与模型 8K 单步
20.7 ms / 14.399 GB = 696 GB/s 比），另报每实例 µs。单实例 hot 数字只作 diagnostic
（明确标注 L2-hot，不可与模型对比）。

只有跑完整 ``SCENARIOS`` 才是 gate 通过；``--only`` 子集只用于开发（coverage
gate 会失败）。

用法::

    .venv/bin/python bench/micro/nvfp4_gemv.py --out {output}/op
    .venv/bin/python bench/micro/nvfp4_gemv.py --only in_proj_qkv --out .tmp/gemv
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 与 scripts/m2_env.* 保持一致：FlashInfer JIT / nvcc 需要 CUDA 头文件。
CUDA_HOME = Path(os.environ.get("CUDA_HOME") or Path.home() / "micromamba/envs/cuda13")
PY_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
NVIDIA_INCLUDE = Path(sys.prefix) / "lib" / PY_TAG / "site-packages/nvidia/cu13/include"
NVIDIA_LIB = Path(sys.prefix) / "lib" / PY_TAG / "site-packages/nvidia/cu13/lib"
os.environ.setdefault("CUDA_HOME", str(CUDA_HOME))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT / ".tmp/flashinfer"))
os.environ["PATH"] = f"{CUDA_HOME / 'bin'}:{os.environ.get('PATH', '')}"
for _var, _paths in {
    "CPATH": [NVIDIA_INCLUDE, CUDA_HOME / "targets/x86_64-linux/include"],
    "LIBRARY_PATH": [
        NVIDIA_LIB,
        CUDA_HOME / "targets/x86_64-linux/lib",
        CUDA_HOME / "lib",
    ],
}.items():
    _merged: list[str] = []
    for _item in [*map(str, _paths), *os.environ.get(_var, "").split(os.pathsep)]:
        if _item and _item not in _merged:
            _merged.append(_item)
    os.environ[_var] = os.pathsep.join(_merged)

import torch  # noqa: E402

sys.path.insert(0, str(ROOT))

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize  # noqa: E402

from runner.experiment.op_artifact import OpReport  # noqa: E402
from runner.loader.checkpoint import CheckpointReader  # noqa: E402
from runner.ops.nvfp4.bytes import weight_bytes  # noqa: E402
from runner.ops.nvfp4.gemv import Nvfp4GemvWeights, auto_k_split  # noqa: E402
from runner.ops.nvfp4.reference import (  # noqa: E402
    e2m1_to_float,
    error_metrics,
    pack_e2m1,
    quantize_activation_reference,
    swizzle_scale_128x4,
    unpack_e2m1,
    unswizzle_scale_128x4,
)

# 产物 case 名；`layout_probe` 是布局探针，其余是真实权重 shape。
SCENARIOS = (
    "layout_probe",
    "in_proj_qkv",
    "gate_proj",
    "lm_head",
    "k_proj",
    "down_proj",
)

# coverage gate 要求必须齐全的 shape。
REQUIRED = ("in_proj_qkv", "gate_proj", "lm_head", "k_proj")

# 每个 case 覆盖的 checkpoint tensor 后缀；一个 shape 在模型里的所有实例都跑。
SHAPES = {
    "in_proj_qkv": (".linear_attn.in_proj_qkv.",),
    "gate_proj": (".mlp.gate_proj.",),
    "k_proj": (".self_attn.k_proj.", ".self_attn.v_proj."),
    "down_proj": (".mlp.down_proj.",),
    "lm_head": ("lm_head.",),
}
MODEL_DIR = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"

# 实现前冻结的容差（decisions: M2 NVFP4）。
TOL = {"nrmse": 5.0e-3, "cosine": 0.99999, "max_rel": 1.0e-2}

# split-K 粗扫（一组粗粒度参数，3 档）。
SPLIT_SWEEP = (1, 2, 4)

# 每个 case 常驻的权重量上限（16 GiB 卡；3.6 GB 已远超 L2，再大只浪费加载时间）。
MAX_CASE_WEIGHT_GB = 4.0


def timed(fn, warmup: int, iters: int) -> float:
    """CUDA event 计时，warmup 后取中位数（与 bench/micro/nvfp4_linear.py 同口径）。"""
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


def _u8(t: torch.Tensor) -> torch.Tensor:
    """packed FP4 数据：只关心字节。"""
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _scale_f8(t: torch.Tensor) -> torch.Tensor:
    """1 字节 block scale：按 UE4M3 解释（**不能** view 成 uint8 再 .float()，
    那会拿到整数字节值而不是 E4M3 的数值）。"""
    if t.dtype == torch.float8_e4m3fn:
        return t
    if t.element_size() != 1:
        raise TypeError(f"block scale 必须是 1 字节 dtype，得到 {t.dtype}")
    return t.view(torch.float8_e4m3fn)


def dequant_activation(a_fp4: torch.Tensor, a_sf: torch.Tensor, k: int) -> torch.Tensor:
    """用 frozen reference 的 primitive 解 FlashInfer 的 activation 量化结果。"""
    codes = unpack_e2m1(_u8(a_fp4).reshape(-1))
    values = e2m1_to_float(codes)
    sf = unswizzle_scale_128x4(_scale_f8(a_sf), 128, k // 16)[0].float()
    return values * sf.repeat_interleave(16)


def dequant_weight_scale(wsf_swizzled: torch.Tensor, n: int, k: int) -> torch.Tensor:
    return unswizzle_scale_128x4(_scale_f8(wsf_swizzled), n, k // 16).float()


def reference_out(
    a_scaled: torch.Tensor,
    weight: torch.Tensor,
    wsf_linear: torch.Tensor,
    alpha: torch.Tensor,
    *,
    chunk: int = 4096,
) -> torch.Tensor:
    """``(a4*a_sf) · (w4*w_sf) * alpha`` 的 FP32 分块 matmul。"""
    n = int(weight.shape[0])
    out = torch.empty(n, dtype=torch.float32, device=weight.device)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        values = e2m1_to_float(unpack_e2m1(weight[start:end]))
        values *= wsf_linear[start:end].repeat_interleave(16, dim=1)
        out[start:end] = values @ a_scaled
    return out * alpha


def quantize_activation(x: torch.Tensor, input_scale: torch.Tensor):
    return nvfp4_quantize(
        x,
        (1.0 / input_scale).reshape(1),
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )


def find_instances(reader: CheckpointReader, suffixes: tuple[str, ...]) -> list[str]:
    """checkpoint 里该 shape 的全部真实实例（按名字排序，去掉 ``weight`` 后缀）。"""
    prefixes: list[str] = []
    for name in sorted(reader.weight_map):
        if not name.endswith("weight"):
            continue
        for suffix in suffixes:
            if name.endswith(suffix + "weight"):
                prefixes.append(name[: -len("weight")])
                break
    return prefixes


def run_layout_probe(report: OpReport, args: argparse.Namespace) -> bool:
    """E2M1 nibble 顺序 + 128x4 swizzled scale 映射的独立探针。"""
    dev = torch.device("cuda")
    n, k = 128, 1024
    case = report.case(
        "layout_probe",
        n=n,
        k=k,
        kind="synthetic",
        nibble_order="low nibble = even k",
        scale_layout="128x4 swizzled (in place)",
        note="期望值由字节约定手算，不经过 reference；probe 计时是 L2-hot，仅 diagnostic",
    )
    ones16 = torch.ones(1, k // 16, dtype=torch.float8_e4m3fn, device=dev)
    alpha = torch.ones(1, dtype=torch.float32, device=dev)
    asf_ones = swizzle_scale_128x4(ones16)

    # --- 探针 1：nibble 顺序 -------------------------------------------------
    # W 字节 0x21：低 nibble(偶数 k)=code1(0.5)，高 nibble(奇数 k)=code2(1.0)；
    # a 字节 0x02：偶数 k=code2(1.0)，奇数 k=code0(0)；两侧 scale 都是 1.0。
    # 正确约定 -> 512 * (1.0 * 0.5) = 256；单侧 nibble 与约定不一致 -> 512。
    w_probe = torch.full((n, k // 2), 0x21, dtype=torch.uint8, device=dev)
    wsf_probe = swizzle_scale_128x4(
        torch.ones(n, k // 16, dtype=torch.float8_e4m3fn, device=dev)
    )
    a_probe = torch.full((1, k // 2), 0x02, dtype=torch.uint8, device=dev)
    holder = Nvfp4GemvWeights(w_probe, wsf_probe, alpha)
    out = holder.forward(a_probe, asf_ones).reshape(-1).float()
    expected = 0.5 * (k // 2)  # 256
    err = (out - expected).abs().max().item()
    finite = bool(torch.isfinite(out).all().item())
    probe_ms = timed(lambda: holder.forward(a_probe, asf_ones), args.warmup, args.iters)
    case.gate(
        "probe: E2M1 nibble 顺序（低 nibble = 偶数 k）",
        finite and err <= 0.5,
        {"expected": expected, "max_abs_err": err, "finite": finite},
    )
    # 判别力：单侧 nibble 与约定不一致（无论 a 还是 W 侧）都会得到 512；
    # 两侧同时反 nibble 是 k 轴重标号（pairing 不变），输出仍是 256，语义等价。
    swapped_expected = 1.0 * (k // 2)  # 512
    case.diag(
        "probe: nibble 顺序的判别力（单侧不一致 -> 512，实测必须离它 >100）",
        (out - swapped_expected).abs().min().item() > 100.0,
        {
            "swapped_expected": swapped_expected,
            "min_abs_vs_swapped": (out - swapped_expected).abs().min().item(),
            "joint_swap_note": "两侧同时反 nibble 是 k 轴重标号，输出不变（语义等价）",
        },
    )

    # --- 探针 2：128x4 swizzled scale 映射 -----------------------------------
    # W 全是 code2(1.0)；a 在 block 0..3 分别是 code 2/1/0/3，其余为 0；
    # scale[n][b] = [1.0, 1.25, 1.5, 2.0][(n+b) % 4]，E4M3 精确可表示。
    # 期望 out[n] = 16*1.0*s[n][0] + 16*0.5*s[n][1] + 0 + 16*1.5*s[n][3]。
    s_values = torch.tensor([1.0, 1.25, 1.5, 2.0], dtype=torch.float32, device=dev)
    idx = (
        torch.arange(n, device=dev)[:, None] + torch.arange(k // 16, device=dev)[None, :]
    ) % 4
    s_linear = s_values[idx]
    w_scale_probe = swizzle_scale_128x4(s_linear.to(torch.float8_e4m3fn))
    w_codes = torch.full((n, k // 2), 0x22, dtype=torch.uint8, device=dev)
    a_codes = torch.zeros(k, dtype=torch.uint8, device=dev)
    for block, code in enumerate((2, 1, 0, 3)):
        a_codes[block * 16 : (block + 1) * 16] = code
    a_by_block = torch.tensor([1.0, 0.5, 0.0, 1.5], dtype=torch.float32, device=dev) * 16.0
    expected_scale = sum(a_by_block[b] * s_linear[:, b] for b in range(4))
    holder_scale = Nvfp4GemvWeights(w_codes, w_scale_probe, alpha)
    out_scale = (
        holder_scale.forward(pack_e2m1(a_codes).reshape(1, -1), asf_ones)
        .reshape(-1)
        .float()
    )
    scale_err = (out_scale - expected_scale).abs().max().item()
    permuted = sum(a_by_block[b] * s_linear[:, (b + 1) % 4] for b in range(4))
    case.gate(
        "probe: 128x4 swizzled block scale 映射",
        bool(torch.isfinite(out_scale).all().item()) and scale_err <= 0.5,
        {"max_abs_err": scale_err, "scale_values": [1.0, 1.25, 1.5, 2.0]},
    )
    case.diag(
        "probe: scale 映射的判别力（错位假设下最小差值）",
        (out_scale - permuted).abs().min().item() > 0.6,
        {"min_abs_vs_permuted_expectation": (out_scale - permuted).abs().min().item()},
    )
    case.metric("probe_kernel_ms_hot", probe_ms)
    case.metric("nibble_probe_max_abs_err", err)
    case.metric("scale_probe_max_abs_err", scale_err)
    case.metric("n", float(n))
    case.metric("k", float(k))
    probe_ok = all(
        item["ok"] for item in case.gates if item["name"].startswith("probe:")
    )
    return probe_ok


def load_instances(
    reader: CheckpointReader, prefixes: list[str], k: int, n: int
) -> list[dict]:
    """按名字加载实例权重；超过显存上限就截断（仍然远超 L2）并记录。"""
    per_instance = weight_bytes(k, n)
    limit = int(MAX_CASE_WEIGHT_GB * 1e9 // per_instance)
    used = prefixes[: max(1, min(len(prefixes), limit))]
    out = []
    for index, prefix in enumerate(used):
        scale_linear = reader.get(prefix + "weight_scale", device="cuda").contiguous()
        out.append(
            {
                "name": prefix,
                "weight": reader.get(prefix + "weight", device="cuda"),
                "scale": swizzle_scale_128x4(scale_linear),
                # 只有第一个实例保留线性 scale：给 M2 冻结路径做交叉验证。
                "scale_linear": scale_linear if index == 0 else None,
                "scale_2": reader.get(prefix + "weight_scale_2", device="cuda"),
                "input_scale": reader.get(prefix + "input_scale", device="cuda"),
            }
        )
    return out


def run_shape(
    name: str, reader: CheckpointReader, report: OpReport, args: argparse.Namespace
) -> None:
    prefixes = find_instances(reader, SHAPES[name])
    if not prefixes:
        raise RuntimeError(f"checkpoint 里找不到 {name} 的实例")
    probe_weight = reader.get(prefixes[0] + "weight", device="cuda")
    n, k_half = (int(x) for x in probe_weight.shape)
    k = 2 * k_half
    del probe_weight
    torch.cuda.empty_cache()

    instances = load_instances(reader, prefixes, k, n)
    per_instance_bytes = weight_bytes(k, n)
    aggregate_bytes = per_instance_bytes * len(instances)
    l2_bytes = int(torch.cuda.get_device_properties(0).L2_cache_size)
    # 工作集 >= 4x L2 才认为聚合 GB/s 干净；否则（如 k_proj 只有 ~94 MB）
    # 绝对值会被 L2 抬高，但候选与 mm_fp4 受同样的影响，比值仍然可比。
    l2_clean = aggregate_bytes >= 4 * l2_bytes

    # activation：用第一个实例的 input_scale 量化一份，所有实例共用同一份 A。
    first_tensors = instances[0]
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261023)
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.5
    a_fp4, a_sf = quantize_activation(x, first_tensors["input_scale"])

    # reference 只对第一个实例算（同 shape 走同一份 kernel 代码路径）。
    #
    # 主 reference（gate 用）：解**同一份量化后的 A**（FlashInfer 的 a_fp4/a_sf）
    #   + checkpoint W 的 128x4 swizzled scale，两条都走 reference.py 的 primitive。
    # 交叉验证 reference（gate 用）：M2 冻结语义 —— 用 reference quantizer 重新量化
    #   同一个 x 得到 a_deq，W 用 checkpoint 的**线性** scale（不经过 swizzle），
    #   这样两条路径只要有一条把 layout/dtype 解释错就会对不上。
    alpha = (first_tensors["input_scale"] * first_tensors["scale_2"]).reshape(1).float()
    a_scaled = dequant_activation(a_fp4, a_sf, k)
    sf_linear_swizzled = dequant_weight_scale(first_tensors["scale"], n, k)
    ref = reference_out(a_scaled, first_tensors["weight"], sf_linear_swizzled, alpha)
    del sf_linear_swizzled

    a_ref_packed, a_ref_sf, a_ref_deq = quantize_activation_reference(
        x, first_tensors["input_scale"]
    )
    sf_linear_ckpt = _scale_f8(first_tensors["scale_linear"]).float()
    ref_frozen = reference_out(
        a_ref_deq.reshape(-1),
        first_tensors["weight"],
        sf_linear_ckpt,
        first_tensors["scale_2"].reshape(1).float(),
    )
    del sf_linear_ckpt
    torch.cuda.empty_cache()

    cross = error_metrics(ref, ref_frozen)
    a_ref_deq_flat = a_ref_deq.reshape(-1)
    a_code_mismatch = int(
        (
            unpack_e2m1(_u8(a_fp4).reshape(-1))
            != unpack_e2m1(_u8(a_ref_packed).reshape(-1))
        ).sum()
    )
    a_sf_mismatch = int(
        (_u8(a_sf) != swizzle_scale_128x4(a_ref_sf).view(torch.uint8)).sum()
    )
    a_decoded_rel = float(
        (a_scaled - a_ref_deq_flat).abs().max()
        / a_ref_deq_flat.abs().max().clamp_min(1e-12)
    )

    holders = [
        Nvfp4GemvWeights(
            item["weight"],
            item["scale"],
            (item["input_scale"] * item["scale_2"]).reshape(1).float(),
        )
        for item in instances
    ]
    policy = auto_k_split(n, sm_count=holders[0].sm_count)
    out = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")

    case = report.case(
        name,
        n=n,
        k=k,
        instances=len(instances),
        instances_total=len(prefixes),
        aggregate_weight_bytes=aggregate_bytes,
        per_instance_bytes=per_instance_bytes,
        working_set_gb=aggregate_bytes / 1e9,
        l2_multiple=aggregate_bytes / max(1, l2_bytes),
        l2_mib=l2_bytes / 1024**2,
        l2_clean=l2_clean,
        scale_layout="flashinfer-128x4-swizzled-inplace",
        nibble_order="low nibble = even k",
        accum="fp32",
        out_dtype="bf16",
        k_split_policy=policy,
        split_sweep=list(SPLIT_SWEEP),
        sm_count=holders[0].sm_count,
        timing="aggregate over all real instances, CUDA event median",
        numeric_instance=first_tensors["name"],
    )

    def sweep(split: int) -> None:
        for holder in holders:
            holder.forward(a_fp4, a_sf, out=out, k_split=split)

    measured: dict[int, dict[str, float]] = {}
    per_split_out: dict[int, torch.Tensor] = {}
    for split in SPLIT_SWEEP:
        out.fill_(float("nan"))
        holders[0].forward(a_fp4, a_sf, out=out, k_split=split)
        per_split_out[split] = out.reshape(-1).clone()
        metrics = error_metrics(out.reshape(-1), ref)
        ms = timed(lambda split=split: sweep(split), args.warmup, args.iters)
        measured[split] = {
            "agg_ms": ms,
            "nrmse": float(metrics["nrmse"]),
            "cosine": float(metrics["cosine"]),
            "max_rel": float(metrics["max_rel"]),
            "finite": float(bool(metrics["finite"])),
        }

    case.gate(
        "reference: 两条 A 反量化路径一致（nrmse <= 1e-2）",
        bool(cross["finite"]) and float(cross["nrmse"]) <= 1e-2,
        {
            "nrmse": float(cross["nrmse"]),
            "cosine": float(cross["cosine"]),
            "a_code_mismatch": a_code_mismatch,
            "a_sf_mismatch": a_sf_mismatch,
            "a_decoded_rel_maxdiff": a_decoded_rel,
            "paths": (
                "same-quantized-A (a_fp4/a_sf + swizzled W scale) vs "
                "M2 frozen (reference quantizer + checkpoint linear W scale)"
            ),
        },
    )

    picked = measured[policy]
    detail = {
        "k_split": policy,
        "nrmse": picked["nrmse"],
        "cosine": picked["cosine"],
        "max_rel": picked["max_rel"],
        "finite": bool(picked["finite"]),
        "tol": TOL,
        "vs": "同一份量化后的 A + checkpoint W dequant 的 FP32 matmul（第一个实例）",
    }
    for key, ok in (
        ("nrmse", picked["nrmse"] <= TOL["nrmse"]),
        ("cosine", picked["cosine"] >= TOL["cosine"]),
        ("max_rel", picked["max_rel"] <= TOL["max_rel"]),
        ("finite", bool(picked["finite"])),
    ):
        case.gate(f"numeric: {key}", bool(ok), detail)
    for split, item in measured.items():
        if split == policy:
            continue
        case.gate(
            f"numeric: k_split={split}",
            bool(item["finite"])
            and item["nrmse"] <= TOL["nrmse"]
            and item["cosine"] >= TOL["cosine"]
            and item["max_rel"] <= TOL["max_rel"],
            item,
        )

    # mm_fp4 baseline：同一批实例、同一份 a_fp4 / a_sf / 同一个计时边界。
    def call_mm_fp4_one(holder: Nvfp4GemvWeights):
        return mm_fp4(
            a_fp4,
            holder.weight_packed.T,
            a_sf,
            holder.weight_scale_orig.T,
            holder.alpha,
            torch.bfloat16,
            None,
            block_size=16,
            use_8x4_sf_layout=False,
            backend="cutlass",
            use_nvfp4=True,
        )

    def mm_fp4_sweep() -> None:
        for holder in holders:
            call_mm_fp4_one(holder)

    out_mm = call_mm_fp4_one(holders[0])
    metrics_mm = error_metrics(out_mm.reshape(-1), ref)
    out.fill_(float("nan"))
    holders[0].forward(a_fp4, a_sf, out=out, k_split=policy)
    vs_mm = error_metrics(out.reshape(-1), out_mm.reshape(-1))
    ms_mm = timed(mm_fp4_sweep, args.warmup, args.iters)
    # 单实例 L2-hot 数字只作 diagnostic（不可与模型 696 GB/s 比较）。
    gemv_hot = timed(
        lambda: holders[0].forward(a_fp4, a_sf, out=out, k_split=policy),
        args.warmup,
        args.iters,
    )
    mm_hot = timed(lambda: call_mm_fp4_one(holders[0]), args.warmup, args.iters)

    gemv_gbs = aggregate_bytes / (picked["agg_ms"] * 1e-3) / 1e9
    mm_gbs = aggregate_bytes / (ms_mm * 1e-3) / 1e9
    case.metric("gemv_agg_ms", picked["agg_ms"])
    case.metric("gemv_agg_gbs", gemv_gbs)
    case.metric("gemv_us_per_instance", picked["agg_ms"] * 1000.0 / len(holders))
    case.metric("mm_fp4_agg_ms", ms_mm)
    case.metric("mm_fp4_agg_gbs", mm_gbs)
    case.metric("mm_fp4_us_per_instance", ms_mm * 1000.0 / len(holders))
    case.metric("speedup_agg_vs_mm_fp4", ms_mm / picked["agg_ms"])
    case.metric("gemv_nrmse", picked["nrmse"])
    case.metric("gemv_cosine", picked["cosine"])
    case.metric("gemv_max_rel", picked["max_rel"])
    case.metric("mm_fp4_nrmse", float(metrics_mm["nrmse"]))
    case.metric("mm_fp4_cosine", float(metrics_mm["cosine"]))
    case.metric("cand_vs_mm_fp4_nrmse", float(vs_mm["nrmse"]))
    case.metric("cand_vs_mm_fp4_cosine", float(vs_mm["cosine"]))
    case.metric("ref_cross_nrmse", float(cross["nrmse"]))
    case.metric("a_code_mismatch", float(a_code_mismatch))
    case.metric("a_sf_mismatch", float(a_sf_mismatch))
    case.metric("a_decoded_rel_maxdiff", a_decoded_rel)
    case.metric("aggregate_gb", aggregate_bytes / 1e9)
    case.metric("l2_multiple", aggregate_bytes / max(1, l2_bytes))
    case.metric("gemv_ms_1inst_hot", gemv_hot)
    case.metric("gemv_gbs_1inst_hot", per_instance_bytes / (gemv_hot * 1e-3) / 1e9)
    case.metric("mm_fp4_ms_1inst_hot", mm_hot)
    case.metric("mm_fp4_gbs_1inst_hot", per_instance_bytes / (mm_hot * 1e-3) / 1e9)
    for split, item in measured.items():
        case.metric(f"gemv_agg_ms_k_split_{split}", item["agg_ms"])
        case.metric(
            f"gemv_agg_gbs_k_split_{split}",
            aggregate_bytes / (item["agg_ms"] * 1e-3) / 1e9,
        )
    # 全部 k_split 档必须给出同一个结果（只差 fp32 求和顺序；用 bf16 量级判据）。
    ref_absmax = float(ref.abs().max().clamp_min(1e-12))
    split_agreement: dict[int, float] = {}
    for split, values in per_split_out.items():
        if split == SPLIT_SWEEP[0]:
            continue
        split_agreement[split] = float(
            (values.float() - per_split_out[SPLIT_SWEEP[0]].float()).abs().max()
        )
    case.gate(
        "numeric: 各 k_split 档结果一致（只差 fp32 求和顺序）",
        all(value <= 1e-2 * ref_absmax for value in split_agreement.values()),
        {
            "baseline_k_split": SPLIT_SWEEP[0],
            "max_abs_diff": split_agreement,
            "threshold_abs": 1e-2 * ref_absmax,
        },
    )
    for split, value in split_agreement.items():
        case.metric(f"k_split_maxdiff_vs_{SPLIT_SWEEP[0]}_k_split_{split}", value)

    mm_ok = (
        bool(metrics_mm["finite"])
        and metrics_mm["nrmse"] <= TOL["nrmse"]
        and metrics_mm["cosine"] >= TOL["cosine"]
        and metrics_mm["max_rel"] <= TOL["max_rel"]
    )
    case.gate(
        "baseline: mm_fp4 与 reference 在冻结容差内（reference 自检）",
        bool(mm_ok),
        {
            key: float(metrics_mm[key])
            for key in ("nrmse", "cosine", "max_rel")
        }
        | {"tol": TOL, "expected_nrmse": 1.6e-3},
    )
    case.gate(
        "baseline: 候选与 mm_fp4 输出一致（同一份 A/W）",
        bool(vs_mm["finite"])
        and float(vs_mm["nrmse"]) <= 1e-3
        and float(vs_mm["cosine"]) >= 0.9999,
        {
            "nrmse": float(vs_mm["nrmse"]),
            "cosine": float(vs_mm["cosine"]),
            "max_rel": float(vs_mm["max_rel"]),
        },
    )
    case.diag(
        "diagnostic: 工作集 >= 4x L2（聚合 GB/s 才干净）",
        l2_clean,
        {
            "working_set_gb": aggregate_bytes / 1e9,
            "l2_mib": l2_bytes / 1024**2,
            "l2_multiple": aggregate_bytes / max(1, l2_bytes),
            "note": "不干净时绝对值被 L2 抬高，候选/mm_fp4 受同样影响，比值仍可比",
        },
    )
    case.diag(
        "baseline: mm_fp4 与同一 reference 的误差（第一个实例）",
        bool(metrics_mm["finite"]),
        {key: float(metrics_mm[key]) for key in ("nrmse", "cosine", "max_rel")},
    )
    print(
        f"{name}: n={n} k={k} inst={len(instances)}/{len(prefixes)} "
        f"work={aggregate_bytes / 1e9:.2f} GB "
        f"({aggregate_bytes / max(1, l2_bytes):.1f}x L2{' clean' if l2_clean else ' L2-AFFECTED'}) "
        f"k_split={policy} "
        f"gemv_agg={picked['agg_ms']:.3f} ms ({gemv_gbs:.1f} GB/s) "
        f"mm_fp4_agg={ms_mm:.3f} ms ({mm_gbs:.1f} GB/s) "
        f"speedup={ms_mm / picked['agg_ms']:.3f} nrmse={picked['nrmse']:.2e} "
        f"cos={picked['cosine']:.8f}"
        + "".join(
            f" ks{split}={item['agg_ms']:.3f}" for split, item in measured.items()
        )
        + f" | ref_cross={float(cross['nrmse']):.2e} "
        f"cand_vs_mm={float(vs_mm['nrmse']):.2e} "
        f"a_codes_mismatch={a_code_mismatch} a_sf_mismatch={a_sf_mismatch}"
        + f" | 1inst hot: gemv={gemv_hot:.4f} mm_fp4={mm_hot:.4f} ms",
        flush=True,
    )
    del holders, instances, out, out_mm
    torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="all",
        help=f"逗号分隔的 case 名，默认 all（{', '.join(SCENARIOS)}）",
    )
    parser.add_argument("--out", required=True, help="产物目录（写 op.json）")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    return parser.parse_args(argv)


def resolve_cases(only: str) -> list[str]:
    text = (only or "all").strip()
    if text == "all":
        return list(SCENARIOS)
    wanted = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in SCENARIOS]
    if unknown:
        raise SystemExit(f"未知 case: {unknown}；可选 {list(SCENARIOS)}")
    if not wanted:
        raise SystemExit("--only 不能为空")
    unique: list[str] = []
    for item in wanted:
        if item not in unique:
            unique.append(item)
    return unique


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run in full access")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    cases = resolve_cases(args.only)
    report = OpReport(
        args.out,
        "nvfp4-gemv-cuda-simt",
        "nvfp4-w4a4-m1-bf16",
        candidate_extra={
            "kernel": "cuda-core-simt-gemv (no tensor core)",
            "source": "runner/ops/nvfp4/gemv_kernel.cu",
            "scale_layout": "flashinfer-128x4-swizzled-inplace",
            "scale_answer": (
                "直接消费 NVFP4Linear 保留的 128x4 swizzled scale：按 "
                "tile*(K/16*128) + c*512 + (row%32)*16 + ((row/32)%4)*4 + d "
                "在 kernel 内直接寻址，不生成 unsquizzled 副本、不做 repack"
            ),
            "nibble_order": "low nibble = even k（checkpoint 与 nvfp4_quantize 一致）",
            "accum": "fp32",
            "split_k": "grid.y + kernel 内 counter/threadfence 归约，每层 1 个 launch",
            "tolerance": TOL,
            "reference": (
                "gate: dequant 同一份量化后的 A（a_fp4/a_sf）+ checkpoint W 的 128x4 "
                "swizzled scale；cross-check: M2 冻结路径（reference quantizer + 线性 "
                "W scale）。block scale 一律按 UE4M3 数值解释"
            ),
            "baseline": "FlashInfer mm_fp4 backend=cutlass, M=1",
            "timing": (
                "aggregate over all real instances per shape (working set >> L2), "
                "candidate 与 mm_fp4 同一边界；单实例数字只作 diagnostic"
            ),
        },
        environment={
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": "%d.%d" % torch.cuda.get_device_capability(0),
            "l2_mib": torch.cuda.get_device_properties(0).L2_cache_size / 1024**2,
            "cuda_home": os.environ.get("CUDA_HOME"),
        },
        steps=None,
    )

    probe_ok = False
    ran: list[str] = []
    reader = CheckpointReader(MODEL_DIR)
    try:
        for name in cases:
            if name == "layout_probe":
                probe_ok = run_layout_probe(report, args)
            else:
                run_shape(name, reader, report, args)
            ran.append(name)
    finally:
        reader.close()

    missing = [item for item in REQUIRED if item not in ran]
    for case in report.cases:
        case.gate(
            "coverage: 必测 shape 齐全 (in_proj_qkv/gate_proj/lm_head/k_proj)",
            not missing,
            {"required": list(REQUIRED), "ran": ran, "missing": missing},
        )
        case.gate(
            "coverage: nibble 顺序与 swizzled scale 布局分别验过",
            probe_ok,
            {"layout_probe": probe_ok},
        )
        case.gate(
            f"coverage: case {case.name} 属于冻结清单",
            case.name in SCENARIOS,
            {"scenarios": list(SCENARIOS)},
        )

    payload = report.finish(
        extra={
            "coverage": {
                "scenarios": list(SCENARIOS),
                "required": list(REQUIRED),
                "ran": ran,
                "missing": missing,
                "layout_probe": probe_ok,
            },
            "split_sweep": list(SPLIT_SWEEP),
        }
    )
    print(
        f"ok={payload['summary']['ok']} cases={len(payload['cases'])} "
        f"missing={missing} probe={probe_ok}",
        flush=True,
    )
    # 产物已写出：gate 失败由 validate_op 判成 gate_failed 并带上明细，
    # 所以这里返回 0（只有真正的运行错误才用非 0 退出码）。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
