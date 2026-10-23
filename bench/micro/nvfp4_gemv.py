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

计时边界（这是本轮修正过的测量方法，见 job-000018 的两个方法学问题）：

- **避开 L2 复用**：GB203 的 L2 约 64 MB，k_proj(2.6 MB)/in_proj_qkv(29 MB)/
  gate_proj(50 MB) 整块住得进 L2，迭代之间会反复命中，测出的 GB/s 能到 1425/2423
  （超过 960 GB/s 物理峰值）。现在每个 case 准备 ``copies`` 份**不同地址**的权重副本，
  一个「计时单元」= 依次跑完所有副本，工作集 ≥ 256 MiB —— 远超 L2。lm_head 715 MB
  本身就是 11× L2，1 份即可。候选与 baseline **用同一套副本协议**。
- **graph replay 边界**：计时单元 capture 成 CUDA graph 后 replay，只包含 GPU 侧
  kernel + graph node 开销，不含 Python/launch 开销 —— 与真实 8K decode 的 graph
  replay 边界一致。否则 FlashInfer ``mm_fp4`` 每调用几十微秒的 host 开销会被算进
  基线，小 shape 上会把两边边界弄成不可比。
- **同一份 A**：候选和 baseline 都吃 FlashInfer ``nvfp4_quantize`` 的 packed bytes
  （候选的 block scale 由 128x4 swizzled 反变换得到）。PyTorch reference quantizer 与
  FlashInfer 的 E2M1 midpoint/FP8 scale 差异只作 diagnostic 记录。

activation quant、scale swizzle、reference 计算都在计时区间外，并单独上报成本。

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
    unswizzle_scale_128x4,
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

# 计时必须让每次迭代扫过远超 L2 的字节：GB203 的 L2 约 64 MB，小 shape 的权重
# （k_proj 2.6 MB、in_proj_qkv 29 MB、gate_proj 50 MB）会整块住进 L2，迭代之间
# 反复命中，测出来的 GB/s 会超过 960 GB/s 的物理峰值。做法：为每个 case 准备
# ``copies`` 份**不同地址**的权重副本，一次「计时单元」= 依次跑完所有副本，
# 工作集 = copies × 权重字节。
L2_TARGET_BYTES = 256 * 1024 * 1024
MAX_COPIES = 80

# 真实 decode 里每个 shape 的实例数（从 checkpoint index 数）：
# in_proj_qkv/out_proj/in_proj_z 各 48（linear attention 层），
# gate/up/down_proj 各 64（MLP），q/k/v/o_proj 各 16（full attention 层），lm_head 1。
REAL_INSTANCES = {
    "in_proj_qkv": 48,
    "gate_proj": 64,
    "lm_head": 1,
    "k_proj": 16,
}

# mm_fp4 口径的每步权重 + scale 字节（baseline profile: 401 次 = 20.7 ms/step）。
STEP_BYTES_MM_FP4 = 14.399e9


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
    """CUDA event 计时（单次 launch），返回中位数（ms）。只用于 quant 等小东西。"""
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


def capture_pass(fns, warmup_iters: int = 3):
    """把一次「计时单元」（copies 个 kernel）capture 成 CUDA graph。

    为什么用 graph：真实 8K decode 是 graph replay，Python/launch 开销不在计时里。
    直接 event 包 Python 调用会把 FlashInfer ``mm_fp4`` 每调用几十微秒的 host
    开销算进基线（小 shape 上尤其明显），使两边边界不一致。capture 后两边都只剩
    GPU 侧 kernel + graph node 开销。capture 失败返回 None，调用方回退到逐个 launch。
    """
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup_iters):
            for fn in fns:
                fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            for fn in fns:
                fn()
    except Exception:  # noqa: BLE001 - 有些 kernel 可能不可 capture
        torch.cuda.synchronize()
        return None
    torch.cuda.synchronize()
    return graph


def timed_pass(graph, fns, warmup: int, iters: int) -> float:
    """一次计时单元 = 整个 pass（graph replay 或逐个 launch），返回中位数（ms）。"""
    def unit() -> None:
        if graph is not None:
            graph.replay()
        else:
            for fn in fns:
                fn()

    for _ in range(warmup):
        unit()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        unit()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))


def copies_for(k: int, n: int) -> int:
    """让「一次计时单元」的工作集超过 L2 所需的权重副本数。"""
    per = weight_bytes(k, n)
    return max(1, min(MAX_COPIES, math.ceil(L2_TARGET_BYTES / per)))


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


def _unswizzle_a_scale(a_sf: torch.Tensor, k: int) -> tuple[torch.Tensor | None, str]:
    """把 FlashInfer 的 activation scale 还原成线性 [1, K/16]（用于候选 kernel）。

    ``nvfp4_quantize(sfLayout=layout_128x4)`` 返回的是 128x4 swizzled（M 补到 128）
    布局；候选 kernel 吃线性布局。两种形状都处理，失败时返回 None 让调用方回退。
    """
    cols = k // 16
    if a_sf.shape[-1] != cols:
        return None, f"cols {a_sf.shape[-1]} != {cols}"
    if a_sf.shape[0] == 1:
        return a_sf.reshape(1, cols).contiguous(), "linear"
    try:
        return unswizzle_scale_128x4(a_sf, 1, cols).contiguous(), f"128x4[{a_sf.shape[0]}]"
    except Exception as exc:  # noqa: BLE001
        return None, f"unswizzle failed: {type(exc).__name__}: {exc}"


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

    # ---- 同一份量化后的 A：候选和 baseline 用 FlashInfer 的 packed bytes ---
    # M2 记录过：PyTorch reference quantizer 与 FlashInfer 在 E2M1 midpoint / FP8
    # scale 上有个位数 code 的差异（decisions.md 2026-10-21）。为让「同一 A」成立，
    # 候选直接吃 FlashInfer 的 packed A + 反 swizzle 出来的 block scale；
    # 只有反 swizzle 失败时才回退到 reference 的 A（那时 same_quantized_a gate 会失败）。
    a_fi, a_sf = nvfp4_quantize(
        x,
        (1.0 / in_s).reshape(1),
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )
    a_fi_u8 = (
        a_fi if a_fi.dtype == torch.uint8 else a_fi.view(torch.uint8)
    ).reshape(1, k // 2).contiguous()
    a_scale_fi, a_scale_mode = _unswizzle_a_scale(a_sf, k)
    pair_exact = a_scale_fi is not None

    a_ref_packed, a_ref_scale, _ = quantize_activation_reference(x, in_s)
    if pair_exact:
        a_packed, a_scale = a_fi_u8, a_scale_fi
    else:
        a_packed = a_ref_packed.reshape(1, k // 2).contiguous()
        a_scale = a_ref_scale.reshape(1, k // 16).contiguous()

    # 诊断：reference quantizer 和 FlashInfer 各自产出的 A 差多少（不是巧合 gate）
    a_byte_diff = int((a_fi_u8.reshape(-1) != a_ref_packed.reshape(-1)).sum())
    a_scale_diff = int(
        (a_scale_fi.view(torch.uint8).reshape(-1) != a_ref_scale.view(torch.uint8).reshape(-1)).sum()
    )
    same_a_as_baseline = bool(
        pair_exact and torch.equal(a_packed.reshape(-1), a_fi_u8.reshape(-1))
    )

    alpha = (in_s * ws2).reshape(1).float()
    ref = reference_out(w, ws, ws2, a_packed, a_scale, in_s)

    # ---- L2 规避：copies 份不同地址的权重副本，一个计时单元扫完所有副本 ----
    copies = copies_for(k, n)
    per_bytes = weight_bytes(k, n)
    working_set = copies * per_bytes
    w_big = w.repeat(copies, 1)
    ws_lin_big = ws.repeat(copies, 1)
    out_bufs = [torch.empty(n, device=device, dtype=torch.bfloat16) for _ in range(copies)]

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
        copies=copies,
        working_set_mib=round(working_set / 1024**2, 1),
        timing_boundary="CUDA graph replay of one pass = copies GEMV launches "
                        "(pure GPU time, production graph-replay boundary)",
        a_scale_mode=a_scale_mode,
        instances=REAL_INSTANCES[case],
    )

    # ---- coverage: 布局分别验（shape 齐全在 main 里所有 case 跑完后统一判）----
    ok_nib, nib_detail = nibble_layout_probe(tensors, device)
    case_obj.gate("coverage:nibble_layout", ok_nib, nib_detail)
    ok_scale, scale_detail = scale_layout_probe(k, device)
    case_obj.gate("coverage:scale_layout", ok_scale, scale_detail)
    # 同一 A：候选与 baseline 必须吃同一份 packed A（可复现的比较前提）。
    case_obj.gate(
        "same_quantized_a_as_baseline",
        same_a_as_baseline,
        {
            "a_scale_mode": a_scale_mode,
            "reference_quantizer_byte_diff": a_byte_diff,
            "reference_quantizer_scale_diff": a_scale_diff,
        },
    )
    case_obj.diag(
        "reference_quantizer_matches_flashinfer",
        a_byte_diff == 0 and a_scale_diff == 0,
        {
            "byte_diff": a_byte_diff,
            "scale_diff": a_scale_diff,
            "packed_bytes": k // 2,
            "scales": k // 16,
            "note": "M2 已知差异（E2M1 midpoint / FP8 scale），只作诊断；"
                    "候选与 baseline 用的是 FlashInfer 的同一份 A",
        },
    )

    # ---- 候选：3 组粗粒度参数（graph replay 计时）----
    mm_kwargs = dict(
        block_size=16,
        use_8x4_sf_layout=False,
        backend="cutlass",
        use_nvfp4=True,
    )
    per_cfg: dict[str, dict[str, float]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for name, cfg in CONFIGS.items():
        try:
            cand0 = gemv_tc(
                w_big[0:n], ws_lin_big[0:n], a_packed, a_scale, alpha,
                cfg=cfg, out=out_bufs[0],
            )
            torch.cuda.synchronize()
            metrics[name] = error_metrics(cand0, ref)

            def cand_fns(cfg=cfg):
                return [
                    lambda kk=kk: gemv_tc(
                        w_big[kk * n:(kk + 1) * n],
                        ws_lin_big[kk * n:(kk + 1) * n],
                        a_packed,
                        a_scale,
                        alpha,
                        cfg=cfg,
                        out=out_bufs[kk],
                    )
                    for kk in range(copies)
                ]

            fns = cand_fns()
            graph = capture_pass(fns)
            ms_pass = timed_pass(graph, fns, warmup, iters)
            per_cfg[name] = {
                "ms_pass": ms_pass,
                "ms_inst": ms_pass / copies,
                "gbs": copies * forward_bytes(1, k, n) / (ms_pass * 1e-3) / 1e9,
                "gbs_inst": forward_bytes(1, k, n) / (ms_pass / copies * 1e-3) / 1e9,
            }
            case_obj.diag(
                f"config:{name}",
                metrics[name]["nrmse"] <= TOL_NRMSE and metrics[name]["cosine"] >= TOL_COSINE,
                {
                    "ms_pass": ms_pass,
                    "ms_per_instance": ms_pass / copies,
                    "timing": "graph" if graph is not None else "launches",
                    **metrics[name],
                },
            )
            del graph, fns
        except Exception as exc:  # noqa: BLE001 - 一个配置失败不应掩盖其它配置
            case_obj.diag(f"config:{name}", False, {"error": f"{type(exc).__name__}: {exc}"})
    if not per_cfg:
        case_obj.gate("candidate_runs", False, {"error": "所有候选配置都失败"})
        case_obj.metric("ms_best", -1.0)  # OpCase 必须有有限 metric
        return {}

    best = min(per_cfg, key=lambda name: per_cfg[name]["ms_inst"])
    best_m = metrics[best]

    # ---- 数值 gate（M2 冻结容差）----
    tag = f"[{best}]"
    case_obj.gate("output_finite", bool(best_m["finite"]), best_m)
    case_obj.gate(f"nrmse<={TOL_NRMSE}{tag}", best_m["nrmse"] <= TOL_NRMSE, best_m["nrmse"])
    case_obj.gate(f"cosine>={TOL_COSINE}{tag}", best_m["cosine"] >= TOL_COSINE, best_m["cosine"])
    case_obj.gate(f"max_rel<={TOL_MAX_REL}{tag}", best_m["max_rel"] <= TOL_MAX_REL, best_m["max_rel"])
    for name in per_cfg:
        if name == best:
            continue
        m = metrics[name]
        case_obj.gate(
            f"numeric:{name}",
            m["finite"] and m["nrmse"] <= TOL_NRMSE and m["cosine"] >= TOL_COSINE
            and m["max_rel"] <= TOL_MAX_REL,
            m,
        )

    # ---- baseline：FlashInfer mm_fp4（cutlass），同样 copies 份副本 + graph ----
    base_ms_pass = float("nan")
    baseline_repack_ms = float("nan")
    base_m: dict[str, float] = {}
    try:
        # baseline 需要 128x4 swizzled scale：加载期一次性 repack，单独计时。
        torch.cuda.synchronize()
        repack_start = torch.cuda.Event(enable_timing=True)
        repack_end = torch.cuda.Event(enable_timing=True)
        repack_start.record()
        ws_swz = swizzle_scale_128x4(ws)
        repack_end.record()
        torch.cuda.synchronize()
        baseline_repack_ms = float(repack_start.elapsed_time(repack_end))
        ws_swz_big = ws_swz.repeat(copies, 1)

        base0 = mm_fp4(
            a_fi, w_big[0:n].T, a_sf, ws_swz_big[0:n].T, alpha,
            torch.bfloat16, out_bufs[0].view(1, n), **mm_kwargs,
        )
        torch.cuda.synchronize()
        base_m = error_metrics(base0, ref)

        fns_b = [
            lambda kk=kk: mm_fp4(
                a_fi,
                w_big[kk * n:(kk + 1) * n].T,
                a_sf,
                ws_swz_big[kk * n:(kk + 1) * n].T,
                alpha,
                torch.bfloat16,
                out_bufs[kk].view(1, n),
                **mm_kwargs,
            )
            for kk in range(copies)
        ]
        graph_b = capture_pass(fns_b)
        base_ms_pass = timed_pass(graph_b, fns_b, warmup, iters)
        case_obj.diag(
            "baseline:mm_fp4_cutlass",
            True,
            {
                "ms_pass": base_ms_pass,
                "ms_per_instance": base_ms_pass / copies,
                "timing": "graph" if graph_b is not None else "launches",
                **base_m,
            },
        )
        del graph_b, fns_b, ws_swz_big
    except Exception as exc:  # noqa: BLE001 - baseline 失败只记录，不改变候选判定
        base_m = {}
        case_obj.diag(
            "baseline:mm_fp4_cutlass", False, {"error": f"{type(exc).__name__}: {exc}"}
        )

    # ---- 成本口径：repack / padding / workspace ----
    pad_bytes = padding_bytes(n, k, CONFIGS[best])
    ws_bytes = workspace_bytes(k)
    q_ms = timed(
        lambda: nvfp4_quantize(
            x, (1.0 / in_s).reshape(1), sfLayout=SfLayout.layout_128x4,
            do_shuffle=False, backend="cuda",
        ),
        warmup,
        iters,
    )
    repack_ms = 0.0  # 本方向不需要 repack
    effective_ms = per_cfg[best]["ms_inst"] + repack_ms

    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    inst = REAL_INSTANCES[case]
    agg_bytes = inst * per_bytes
    agg_ms_best = inst * per_cfg[best]["ms_inst"]
    agg_ms_base = inst * base_ms_pass / copies if math.isfinite(base_ms_pass) else float("nan")

    def gbs(ms_value: float, nbytes: float) -> float:
        return nbytes / (ms_value * 1e-3) / 1e9 if ms_value and ms_value > 0 else float("nan")

    def metric(name: str, value: float) -> None:
        """NaN / inf 不能进 metric（OpReport 会丢非有限值并 KeyError），直接跳过。"""
        if value is not None and math.isfinite(float(value)):
            case_obj.metric(name, float(value))

    metric("ms_best", per_cfg[best]["ms_inst"])
    metric("ms_pass_best", per_cfg[best]["ms_pass"])
    metric("gbs_best", per_cfg[best]["gbs"])
    metric("copies", float(copies))
    metric("working_set_mib", working_set / 1024**2)
    metric("instances", float(inst))
    metric("aggregate_bytes", float(agg_bytes))
    metric("aggregate_ms_best", agg_ms_best)
    metric("aggregate_gbs_best", gbs(agg_ms_best, agg_bytes))
    metric("ms_mm_fp4", base_ms_pass / copies if copies else float("nan"))
    metric("ms_pass_mm_fp4", base_ms_pass)
    metric("gbs_mm_fp4", gbs(base_ms_pass, copies * forward_bytes(1, k, n)))
    metric("aggregate_ms_mm_fp4", agg_ms_base)
    metric("aggregate_gbs_mm_fp4", gbs(agg_ms_base, agg_bytes))
    metric("speedup_vs_mm_fp4", (base_ms_pass / copies) / per_cfg[best]["ms_inst"]
           if math.isfinite(base_ms_pass) else float("nan"))
    metric("aggregate_speedup_vs_mm_fp4", agg_ms_base / agg_ms_best
           if math.isfinite(agg_ms_base) else float("nan"))
    metric("repack_ms", repack_ms)
    metric("effective_ms", effective_ms)
    metric("workspace_bytes", float(ws_bytes))
    metric("padding_bytes", float(pad_bytes))
    metric("peak_allocated_mib", float(peak_mib))
    metric("quant_fi_ms_diagnostic", q_ms)
    metric("nrmse", best_m["nrmse"])
    metric("cosine", best_m["cosine"])
    metric("max_rel", best_m["max_rel"])
    for name in per_cfg:
        metric(f"ms_{name}", per_cfg[name]["ms_inst"])
        metric(f"ms_pass_{name}", per_cfg[name]["ms_pass"])
    if base_m:
        metric("baseline_repack_swizzle_ms", baseline_repack_ms)
        metric("baseline_nrmse", base_m["nrmse"])
        metric("baseline_cosine", base_m.get("cosine", float("nan")))

    print(
        f"{case} N={n} K={k}: copies={copies} ws={working_set / 1024**2:.0f}MiB "
        f"best={best} {per_cfg[best]['ms_inst'] * 1e3:.1f} us/inst "
        f"({per_cfg[best]['gbs']:.0f} GB/s) vs mm_fp4 "
        f"{base_ms_pass / copies * 1e3:.1f} us/inst "
        f"({gbs(base_ms_pass, copies * forward_bytes(1, k, n)):.0f} GB/s) "
        f"inst={inst} agg={agg_ms_best:.3f} ms vs {agg_ms_base:.3f} ms "
        f"nrmse={best_m['nrmse']:.2e} same_A={same_a_as_baseline} "
        f"ref_a_byte_diff={a_byte_diff}",
        flush=True,
    )
    row = {
        "case": case,
        "best": best,
        "n": n,
        "k": k,
        "copies": copies,
        "working_set_mib": working_set / 1024**2,
        "instances": inst,
        "ms_inst_best": per_cfg[best]["ms_inst"],
        "ms_pass_best": per_cfg[best]["ms_pass"],
        "gbs_best": per_cfg[best]["gbs"],
        "ms_inst_mm_fp4": base_ms_pass / copies if math.isfinite(base_ms_pass) else None,
        "ms_pass_mm_fp4": base_ms_pass if math.isfinite(base_ms_pass) else None,
        "gbs_mm_fp4": gbs(base_ms_pass, copies * forward_bytes(1, k, n)),
        "per_config": {name: per_cfg[name] for name in per_cfg},
        "aggregate_bytes": agg_bytes,
        "aggregate_ms_best": agg_ms_best,
        "aggregate_ms_mm_fp4": agg_ms_base if math.isfinite(agg_ms_base) else None,
        "aggregate_gbs_best": gbs(agg_ms_best, agg_bytes),
        "aggregate_gbs_mm_fp4": gbs(agg_ms_base, agg_bytes),
        "repack_ms": repack_ms,
        "effective_ms": effective_ms,
        "baseline_repack_swizzle_ms": baseline_repack_ms,
        "padding_bytes": pad_bytes,
        "workspace_bytes": ws_bytes,
        "nrmse": best_m["nrmse"],
        "cosine": best_m["cosine"],
        "max_rel": best_m["max_rel"],
        "same_quantized_a": same_a_as_baseline,
        "reference_quantizer_byte_diff": a_byte_diff,
        "peak_allocated_mib": peak_mib,
    }
    del tensors, w, w_big, ws, ws_lin_big, out_bufs, ref, a_fi, a_sf, a_fi_u8
    del a_packed, a_scale, a_ref_packed, a_ref_scale
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

    # ---- 按真实实例数加权：4 个代表 shape 的聚合 ----
    good = [row for row in rows if row and row.get("aggregate_bytes")]
    total_bytes = sum(row["aggregate_bytes"] for row in good)
    total_ms_best = sum(row["aggregate_ms_best"] for row in good)
    total_ms_base = (
        sum(row["aggregate_ms_mm_fp4"] for row in good if row.get("aggregate_ms_mm_fp4"))
        if all(row.get("aggregate_ms_mm_fp4") for row in good)
        else None
    )
    gbs_best = total_bytes / (total_ms_best * 1e-3) / 1e9 if total_ms_best > 0 else None
    gbs_base = (
        total_bytes / (total_ms_base * 1e-3) / 1e9 if total_ms_base else None
    )
    aggregate = {
        "method": "每个计时单元 = copies 份不同地址权重副本的 graph replay，"
                  "工作集 > 256 MiB（远超 64 MB L2），再按真实实例数加权",
        "shapes": [
            {
                "case": row["case"],
                "instances": row["instances"],
                "bytes_per_instance": row["aggregate_bytes"] // row["instances"],
                "copies": row["copies"],
                "working_set_mib": round(row["working_set_mib"], 1),
                "ms_inst_best": row["ms_inst_best"],
                "ms_inst_mm_fp4": row.get("ms_inst_mm_fp4"),
                "gbs_best": row["gbs_best"],
                "gbs_mm_fp4": row.get("gbs_mm_fp4"),
                "aggregate_bytes": row["aggregate_bytes"],
                "aggregate_ms_best": row["aggregate_ms_best"],
                "aggregate_ms_mm_fp4": row.get("aggregate_ms_mm_fp4"),
            }
            for row in good
        ],
        "totals": {
            "aggregate_bytes": total_bytes,
            "aggregate_ms_best": total_ms_best,
            "aggregate_ms_mm_fp4": total_ms_base,
            "aggregate_gbs_best": gbs_best,
            "aggregate_gbs_mm_fp4": gbs_base,
            "aggregate_speedup": (total_ms_base / total_ms_best) if total_ms_base else None,
            "coverage_of_step_bytes": total_bytes / STEP_BYTES_MM_FP4,
        },
        "step_estimate": {
            "step_bytes_mm_fp4": STEP_BYTES_MM_FP4,
            "est_step_ms_best": (STEP_BYTES_MM_FP4 / (gbs_best * 1e9) * 1e3) if gbs_best else None,
            "est_step_ms_mm_fp4": (STEP_BYTES_MM_FP4 / (gbs_base * 1e9) * 1e3) if gbs_base else None,
            "baseline_profile_ms_per_step": 20.7,
            "caveat": "只有 4 个代表 shape（覆盖 step 权重字节的 37%）；未覆盖部分主要是 "
                      "up_proj/down_proj（64+64 个实例、各 ~50 MB，与 gate_proj 同类），"
                      "所以按同档 GB/s 外推。这是估计，不是实测整步。",
        },
    }

    report.finish(extra={"cases_run": names, "aggregate": aggregate, "cost_model": {
        "repack_bytes": 0,
        "repack_ms": 0.0,
        "note": "swap-AB kernel 直接消费 checkpoint 布局；不需要 swizzle/转置 repack，"
                "padding 由 padding_bytes() 上报，workspace 只有 K/2 + K/16 字节的 A。",
    }})
    print(
        f"\naggregate over {len(good)} shapes ({total_bytes / 1e9:.2f} GB = "
        f"{total_bytes / STEP_BYTES_MM_FP4:.0%} of step bytes): "
        f"{total_ms_best:.3f} ms candidate vs {total_ms_base if total_ms_base else float('nan'):.3f} ms "
        f"mm_fp4 ({gbs_best if gbs_best else float('nan'):.0f} vs "
        f"{gbs_base if gbs_base else float('nan'):.0f} GB/s); "
        f"est step {STEP_BYTES_MM_FP4 / (gbs_best * 1e9) * 1e3 if gbs_best else float('nan'):.1f} ms "
        f"vs baseline profile 20.7 ms",
        flush=True,
    )
    out_dir = Path(args.out)
    (out_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\noutput: {out_dir}/op.json")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
