#!/usr/bin/env python3
"""NVFP4 GEMV（M=1）microbench：swap-AB + native FP4 tensor core MMA vs FlashInfer ``mm_fp4``。

任务 C（Phase 2）的 op gate + 局部 microbench。判定口径与 M2 冻结容差一致：
与「同一份量化后的 A + checkpoint W 的 dequant 参考 matmul」比较，
``nrmse <= 5e-3``、``cosine >= 0.99999``、``max_rel <= 1e-2``、输出 finite。

数据流（候选）：

    checkpoint w U8 [N, K/2] + F8 [N, K/16] 线性 scale（不 repack、不 swizzle）
      -> runner/ops/nvfp4/gemm_tc.py: 每个 CTA 负责 BM 行，K 方向 BK 分块，
         tl.dot_scaled(e2m1 x e2m1, per-16 ue4m3 scale) -> mxf4nvf4 MMA，FP32 累加
      -> 取第 0 列 * alpha -> BF16 [1, N]

计时边界：只包住「一次 GEMV kernel launch」。activation quant、scale swizzle、
reference 计算都在计时区间外，并单独上报它们的成本（diagnostic）。

覆盖（每个 case 都跑）：
- ``coverage:required_shapes``：4 个必测 shape 齐全；
- ``coverage:nibble_layout``：用真实 W 的第 0 行，分别在偶数/奇数 k 上放一个
  单 nibble 的 activation，验低 nibble = 偶数下标的打包约定；
- ``coverage:scale_layout``：合成 W/A 的 per-16 block scale 图案（1/2/3/4 循环），
  验 scale 的 block 粒度、K 方向索引和行索引。

用法::

    .venv/bin/python bench/micro/nvfp4_gemv.py --out /tmp/gemv --only lm_head

注意：本脚本只应通过实验队列（GPU 主机上）执行，本地不要跑 CUDA。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 与 scripts/m2_env.* / bench/micro/nvfp4_linear.py 一致：让 FlashInfer JIT 找到
# CUDA / curand 头文件。只设环境变量，不建 CUDA context。
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
    _merged = []
    for _item in [*map(str, _paths), *os.environ.get(_var, "").split(os.pathsep)]:
        if _item and _item not in _merged:
            _merged.append(_item)
    os.environ[_var] = os.pathsep.join(_merged)

import torch  # noqa: E402

sys.path.insert(0, str(ROOT))

from runner.ops.nvfp4.bytes import forward_bytes, weight_bytes  # noqa: E402
from runner.ops.nvfp4.gemm_tc import (  # noqa: E402
    CONFIGS,
    DEFAULT_CONFIG,
    Config,
    gemv_tc,
    padding_bytes,
    workspace_bytes,
)
from runner.ops.nvfp4.reference import (  # noqa: E402
    dequant_nvfp4,
    error_metrics,
    quantize_activation_reference,
    swizzle_scale_128x4,
)
from runner.experiment.op_artifact import OpReport  # noqa: E402

# 必测 shape（AST 字面量：schema 提交期会做字面量求值来校验 --only）。
SCENARIOS = ("in_proj_qkv", "gate_proj", "lm_head", "k_proj")

# N = 2 * packed.shape[1]，K = 2 * packed.shape[1]；这里写逻辑 (N, K) 仅作校验。
SHAPES = {
    "in_proj_qkv": (10240, 5120),
    "gate_proj": (17408, 5120),
    "lm_head": (248320, 5120),
    "k_proj": (1024, 5120),
}

PRESETS = {
    "in_proj_qkv": (
        "model-00001-of-00002.safetensors",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.",
    ),
    "gate_proj": (
        "model-00001-of-00002.safetensors",
        "model.language_model.layers.0.mlp.gate_proj.",
    ),
    "lm_head": ("model-00002-of-00002.safetensors", "lm_head."),
    "k_proj": (
        "model-00001-of-00002.safetensors",
        "model.language_model.layers.11.self_attn.k_proj.",
    ),
}

MODEL_DIR = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"

# probe 用固定的 tile（N=64 必须被 block_m 整除），与候选配置无关。
PROBE_CFG = Config(block_m=64, block_k=256, block_n=16, num_warps=4, num_stages=3)

# M2 冻结容差（docs/decisions.md 2026-10-20 / 2026-10-21）。
TOL_NRMSE = 5e-3
TOL_COSINE = 0.99999
TOL_MAX_REL = 1e-2


def load_case(name: str) -> dict[str, torch.Tensor]:
    """从 checkpoint 读一个 shape 的 NVFP4 权重（CPU -> GPU）。"""
    from safetensors import safe_open

    shard, prefix = PRESETS[name]
    with safe_open(MODEL_DIR / shard, framework="pt", device="cpu") as f:
        tensors = {
            "weight": f.get_tensor(prefix + "weight"),
            "weight_scale": f.get_tensor(prefix + "weight_scale"),
            "weight_scale_2": f.get_tensor(prefix + "weight_scale_2"),
            "input_scale": f.get_tensor(prefix + "input_scale"),
        }
    n, k = SHAPES[name]
    got = tensors["weight"].shape
    if got != (n, k // 2):
        raise SystemExit(f"{name}: packed weight {tuple(got)} != {(n, k // 2)}")
    return {key: value.cuda() for key, value in tensors.items()}


def reference_out(
    w_packed: torch.Tensor,
    w_scale: torch.Tensor,
    w_scale_2: torch.Tensor,
    a_packed: torch.Tensor,
    a_scale: torch.Tensor,
    input_scale: torch.Tensor,
    *,
    chunk: int = 4096,
) -> torch.Tensor:
    """分块 dequant 参考：``a_dequant @ w_dequant^T``（FP32），避免整份 W 反量化。

    两个 global scale 都已经在 dequant 里（``input_scale`` / ``weight_scale_2``），
    所以结果等于候选的 ``acc * alpha``。
    """
    a_deq = dequant_nvfp4(a_packed, a_scale, input_scale)  # [1, K] fp32
    n = w_packed.shape[0]
    out = torch.empty(n, device=w_packed.device, dtype=torch.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        w_deq = dequant_nvfp4(w_packed[start:stop], w_scale[start:stop], w_scale_2)
        out[start:stop] = (a_deq @ w_deq.T).reshape(-1)
    return out


def timed(fn, warmup: int, iters: int) -> float:
    """CUDA event 计时，返回中位数（ms）。"""
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
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def _single_nibble(k: int, k_len: int, device: torch.device) -> torch.Tensor:
    """a_packed [1, K/2]：只在下标 k 放 E2M1 code 2（= 1.0），低 nibble = 偶数下标。"""
    packed = torch.zeros(1, k_len // 2, device=device, dtype=torch.uint8)
    byte = k // 2
    if k % 2 == 0:
        packed[0, byte] = 2
    else:
        packed[0, byte] = 2 << 4
    return packed


def nibble_layout_probe(tensors: dict[str, torch.Tensor], device: torch.device) -> tuple[bool, dict]:
    """用真实 W 的第 0 行验 nibble 顺序：偶数/奇数下标各放一个单 nibble activation。

    如果 kernel 把 nibble 顺序读反，偶数 probe 会命中 k+1 的权重，结果完全不同。
    """
    w = tensors["weight"][:64]
    ws = tensors["weight_scale"][:64]
    k_len = w.shape[1] * 2
    alpha = (tensors["input_scale"] * tensors["weight_scale_2"]).reshape(1).float()
    row = dequant_nvfp4(w[:1], ws[:1], tensors["weight_scale_2"])[0]
    ev = row[0::2].abs()
    od = row[1::2].abs()
    k_even = int(ev.argmax()) * 2
    k_odd = int(od.argmax()) * 2 + 1
    a_scale = torch.full((1, k_len // 16), 1.0, device=device, dtype=torch.float8_e4m3fn)
    detail: dict[str, object] = {"k_even": k_even, "k_odd": k_odd}
    ok = True
    for k, tag in ((k_even, "even"), (k_odd, "odd")):
        a_packed = _single_nibble(k, k_len, device)
        out = gemv_tc(w, ws, a_packed, a_scale, alpha, cfg=PROBE_CFG)
        expect = float(tensors["input_scale"]) * float(row[k])
        got = float(out[0, 0].float())
        rel = abs(got - expect) / max(abs(expect), 1e-12)
        detail[f"{tag}_expect"] = expect
        detail[f"{tag}_got"] = got
        detail[f"{tag}_rel"] = rel
        ok = ok and rel <= 1e-2 and abs(expect) > 0
    return ok, detail


def scale_layout_probe(k_len: int, device: torch.device) -> tuple[bool, dict]:
    """合成 per-16 block scale 图案，验 scale 的粒度 / K 索引 / 行索引。

    W、A 的 code 全是 2（= 1.0）；scale 在 1/2/3/4 之间按 block 和行循环。
    若 kernel 用 32 元素 block、错位或把行索引搞错，结果都对不上。
    """
    n = 64
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device, dtype=torch.float32)
    w_packed = torch.full((n, k_len // 2), 0x22, device=device, dtype=torch.uint8)
    a_packed = torch.full((1, k_len // 2), 0x22, device=device, dtype=torch.uint8)
    rows = torch.arange(n, device=device)[:, None]
    blocks = torch.arange(k_len // 16, device=device)[None, :]
    w_scale = vals[(rows + blocks) % 4].to(torch.float8_e4m3fn)
    a_scale = vals[blocks % 4].to(torch.float8_e4m3fn)
    alpha = torch.ones(1, device=device, dtype=torch.float32)
    out = gemv_tc(w_packed, w_scale, a_packed, a_scale, alpha, cfg=PROBE_CFG)
    ref = dequant_nvfp4(a_packed, a_scale, 1.0) @ dequant_nvfp4(w_packed, w_scale, 1.0).T
    m = error_metrics(out, ref)
    ok = m["nrmse"] <= TOL_NRMSE and m["cosine"] >= TOL_COSINE and m["finite"]
    return ok, m


def run_case(
    case: str,
    report: OpReport,
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    n, k = SHAPES[case]
    tensors = load_case(case)
    torch.cuda.reset_peak_memory_stats()
    w = tensors["weight"]
    ws = tensors["weight_scale"]
    ws2 = tensors["weight_scale_2"]
    in_s = tensors["input_scale"]

    torch.manual_seed(0)
    x = (torch.randn(1, k, device=device, dtype=torch.bfloat16) * 0.5).contiguous()

    # 同一份量化后的 A：reference 量化（纯 torch，与 checkpoint 语义一致）。
    a_packed, a_scale, _a_deq = quantize_activation_reference(x, in_s)
    # FlashInfer 的 activation quant（生产路径），用于 baseline 与可比性诊断。
    a_fi, a_sf = nvfp4_quantize(
        x,
        (1.0 / in_s).reshape(1),
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    try:
        a_fi_u8 = a_fi if a_fi.dtype == torch.uint8 else a_fi.view(torch.uint8)
        same_a = bool(torch.equal(a_packed.reshape(-1), a_fi_u8.reshape(-1)))
    except Exception:  # noqa: BLE001 - 子字节 dtype 不能 view 时只记不可比
        same_a = False
    alpha = (in_s * ws2).reshape(1).float()

    ref = reference_out(w, ws, ws2, a_packed, a_scale, in_s)

    case_obj = report.case(
        case,
        n=n,
        k=k,
        m=1,
        dtype="nvfp4-w4a4",
        weight_layout=f"checkpoint-packed-u8[{n},{k // 2}]-linear-scale[{n},{k // 16}]",
        kernel="triton-mxf4nvf4-swapab",
        configs=list(CONFIGS),
        default_config=DEFAULT_CONFIG,
        timing_boundary="one gemv kernel launch (CUDA event median)",
    )

    # ---- coverage: 布局分别验（shape 齐全在 main 里所有 case 跑完后统一判）----
    ok_nib, nib_detail = nibble_layout_probe(tensors, device)
    case_obj.gate("coverage:nibble_layout", ok_nib, nib_detail)
    ok_scale, scale_detail = scale_layout_probe(k, device)
    case_obj.gate("coverage:scale_layout", ok_scale, scale_detail)
    case_obj.diag("same_quantized_a_as_flashinfer", same_a, {"equal_packed_bytes": same_a})

    # ---- 候选：3 组粗粒度参数 ----
    ms: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    outs: dict[str, torch.Tensor] = {}
    for name, cfg in CONFIGS.items():
        try:
            fn = lambda cfg=cfg: gemv_tc(w, ws, a_packed, a_scale, alpha, cfg=cfg)  # noqa: B023
            fn()
            torch.cuda.synchronize()
            ms[name] = timed(fn, warmup, iters)
            outs[name] = gemv_tc(w, ws, a_packed, a_scale, alpha, cfg=cfg)
            metrics[name] = error_metrics(outs[name], ref)
            case_obj.diag(
                f"config:{name}",
                metrics[name]["nrmse"] <= TOL_NRMSE and metrics[name]["cosine"] >= TOL_COSINE,
                {"ms": ms[name], **metrics[name]},
            )
        except Exception as exc:  # noqa: BLE001 - 一个配置失败不应掩盖其它配置
            case_obj.diag(f"config:{name}", False, {"error": f"{type(exc).__name__}: {exc}"})
    if not ms:
        case_obj.gate("candidate_runs", False, {"error": "所有候选配置都失败"})
        case_obj.metric("ms_best", -1.0)  # OpCase 必须有有限 metric
        return {}

    best = min(ms, key=ms.get)
    best_m = metrics[best]

    # ---- 数值 gate（M2 冻结容差）----
    tag = f"[{best}]"
    case_obj.gate("output_finite", bool(best_m["finite"]), best_m)
    case_obj.gate(f"nrmse<={TOL_NRMSE}{tag}", best_m["nrmse"] <= TOL_NRMSE, best_m["nrmse"])
    case_obj.gate(f"cosine>={TOL_COSINE}{tag}", best_m["cosine"] >= TOL_COSINE, best_m["cosine"])
    case_obj.gate(f"max_rel<={TOL_MAX_REL}{tag}", best_m["max_rel"] <= TOL_MAX_REL, best_m["max_rel"])
    for name in ms:
        if name == best:
            continue
        m = metrics[name]
        case_obj.gate(
            f"numeric:{name}",
            m["finite"] and m["nrmse"] <= TOL_NRMSE and m["cosine"] >= TOL_COSINE
            and m["max_rel"] <= TOL_MAX_REL,
            m,
        )

    # ---- baseline：FlashInfer mm_fp4（cutlass，tensor core）----
    mm_kwargs = dict(
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cutlass",
        use_nvfp4=True,
    )
    base_ms = float("nan")
    try:
        # baseline 需要 128x4 swizzled scale：这是它的加载期一次性 repack 成本，
        # 单独计时（diagnostic），不进候选的计时区间。
        w_swizzled = swizzle_scale_128x4(ws)
        torch.cuda.synchronize()
        repack_start = torch.cuda.Event(enable_timing=True)
        repack_end = torch.cuda.Event(enable_timing=True)
        repack_start.record()
        swizzle_scale_128x4(ws)
        repack_end.record()
        torch.cuda.synchronize()
        baseline_repack_ms = float(repack_start.elapsed_time(repack_end))

        def baseline():
            return mm_fp4(
                a_fi,
                w.T,
                a_sf,
                w_swizzled.T,
                alpha,
                torch.bfloat16,
                None,
                **mm_kwargs,
            )

        base_out = baseline()
        torch.cuda.synchronize()
        base_ms = timed(baseline, warmup, iters)
        base_m = error_metrics(base_out, ref)
        case_obj.diag("baseline:mm_fp4_cutlass", True, {"ms": base_ms, **base_m})
    except Exception as exc:  # noqa: BLE001 - baseline 失败只记录，不改变候选判定
        baseline_repack_ms = float("nan")
        base_m = {}
        case_obj.diag(
            "baseline:mm_fp4_cutlass", False, {"error": f"{type(exc).__name__}: {exc}"}
        )

    # ---- 成本口径：repack / padding / workspace ----
    pad_bytes = padding_bytes(n, k, CONFIGS[best])
    ws_bytes = workspace_bytes(k)
    # 生产路径的 activation quant 成本（diagnostic，不计入候选计时）。
    q_ms = timed(
        lambda: nvfp4_quantize(
            x, (1.0 / in_s).reshape(1), sfLayout=SfLayout.layout_128x4,
            do_shuffle=False, backend="cuda",
        ),
        warmup,
        iters,
    )
    # 本方向不需要 repack：直接消费 checkpoint 的 packed 权重 + 线性 scale。
    repack_ms = 0.0
    effective_ms = ms[best] + repack_ms

    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    total_bytes = forward_bytes(1, k, n)

    def gbs(ms_value: float) -> float:
        return total_bytes / (ms_value * 1e-3) / 1e9 if ms_value > 0 else float("nan")

    def metric(name: str, value: float) -> None:
        """NaN / inf 不能进 metric（OpReport 会丢非有限值并 KeyError），直接跳过。"""
        if math.isfinite(float(value)):
            case_obj.metric(name, float(value))

    metric("ms_best", ms[best])
    metric("ms_mm_fp4", base_ms)
    metric("speedup_vs_mm_fp4", base_ms / ms[best])
    metric("gbs_best", gbs(ms[best]))
    metric("gbs_mm_fp4", gbs(base_ms))
    metric("frac_of_960_gbs_peak", gbs(ms[best]) / 960.0)
    metric("repack_ms", repack_ms)
    metric("effective_ms", effective_ms)
    metric("workspace_bytes", float(ws_bytes))
    metric("padding_bytes", float(pad_bytes))
    metric("peak_allocated_mib", float(peak_mib))
    metric("quant_fi_ms_diagnostic", q_ms)
    metric("nrmse", best_m["nrmse"])
    metric("cosine", best_m["cosine"])
    metric("max_rel", best_m["max_rel"])
    for name in ms:
        metric(f"ms_{name}", ms[name])
    if base_m:
        metric("baseline_repack_swizzle_ms", baseline_repack_ms)
        metric("baseline_nrmse", base_m["nrmse"])
        metric("baseline_cosine", base_m["cosine"])

    print(
        f"{case} N={n} K={k}: best={best} {ms[best]:.3f} ms "
        f"({gbs(ms[best]):.0f} GB/s) vs mm_fp4 {base_ms:.3f} ms "
        f"({gbs(base_ms):.0f} GB/s) speedup={base_ms / ms[best]:.2f} "
        f"nrmse={best_m['nrmse']:.2e} cos={best_m['cosine']:.8f} "
        f"pad={pad_bytes}B ws={ws_bytes}B same_A={same_a}",
        flush=True,
    )
    row = {
        "case": case,
        "best": best,
        "ms_best": ms[best],
        "ms": ms,
        "ms_mm_fp4": base_ms,
        "repack_ms": repack_ms,
        "effective_ms": effective_ms,
        "baseline_repack_ms": baseline_repack_ms,
        "padding_bytes": pad_bytes,
        "workspace_bytes": ws_bytes,
        "gbs_best": gbs(ms[best]),
        "gbs_mm_fp4": gbs(base_ms),
        "nrmse": best_m["nrmse"],
        "cosine": best_m["cosine"],
        "max_rel": best_m["max_rel"],
        "same_quantized_a": same_a,
    }
    del tensors, w, ws, w_swizzled, outs, ref, a_fi, a_sf, a_fi_u8, a_packed, a_scale
    torch.cuda.empty_cache()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--only", default="all", help="逗号分隔的 case 名；all 跑全部")
    parser.add_argument("--out", required=True, help="产物目录（op.json 写在这里）")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=15)
    args = parser.parse_args()

    names = list(SCENARIOS) if args.only.strip() == "all" else [
        part.strip() for part in args.only.split(",") if part.strip()
    ]
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        parser.error(f"未知 case: {unknown}；可用: {', '.join(SCENARIOS)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; 请通过实验队列在 GPU 主机上执行")

    device = torch.device("cuda")
    report = OpReport(
        args.out,
        candidate_name="nvfp4-gemv-tc-swapab",
        candidate_format="nvfp4-w4a4",
        candidate_extra={
            "wall": "M=1 GEMV by native FP4 MMA (mma.sync m16n8k64 mxf4nvf4)",
            "swap_ab": "W as MMA A operand, activation as B operand (broadcast)",
            "configs": {name: cfg.as_dict() for name, cfg in CONFIGS.items()},
            "default_config": DEFAULT_CONFIG,
            "tolerances": {
                "nrmse": TOL_NRMSE,
                "cosine": TOL_COSINE,
                "max_rel": TOL_MAX_REL,
            },
            "repack": "none (consumes checkpoint packed weight + linear per-16 scale)",
        },
        environment={
            "triton": __import__("triton").__version__,
            "torch": torch.__version__,
            "checkpoint": "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"
            "@5b7a687fc8211a5d631c8ca6a593dd37eb26ce33",
            "warmup": args.warmup,
            "iters": args.iters,
        },
    )

    rows = []
    for case in names:
        try:
            rows.append(run_case(case, report, warmup=args.warmup, iters=args.iters, device=device))
        except Exception as exc:  # noqa: BLE001 - 记录后继续下一个 case
            case_obj = report.case(case, n=SHAPES[case][0], k=SHAPES[case][1], error=str(exc))
            case_obj.gate(f"coverage:{case}", False, {"error": f"{type(exc).__name__}: {exc}"})
            case_obj.gate("case_runs", False, {"error": f"{type(exc).__name__}: {exc}"})
            case_obj.metric("ms_best", -1.0)
            print(f"{case}: FAILED {type(exc).__name__}: {exc}", flush=True)

    # coverage：所有 case 跑完后统一判必测 shape 是否齐全（缺一个即失败）。
    present = [item.name for item in report.cases]
    covered = set(SCENARIOS).issubset(set(present))
    for item in report.cases:
        item.gate(
            "coverage:required_shapes",
            covered,
            {"required": list(SCENARIOS), "present": present, "requested": names},
        )

    report.finish(extra={"cases_run": names, "cost_model": {
        "repack_bytes": 0,
        "repack_ms": 0.0,
        "note": "swap-AB kernel 直接消费 checkpoint 布局；不需要 swizzle/转置 repack，"
                "padding 由 padding_bytes() 上报，workspace 只有 K/2 + K/16 字节的 A。",
    }})
    out_dir = Path(args.out)
    (out_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\noutput: {out_dir}/op.json")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
