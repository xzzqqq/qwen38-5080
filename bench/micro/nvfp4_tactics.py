#!/usr/bin/env python3
"""NVFP4 M=1 GEMV 的 per-shape backend/tactic 选型（op gate + microbench）。

要回答的问题
------------
模型 8K decode 每步 401 次 FlashInfer ``mm_fp4`` = 19.299 ms / 14.399 GB =
747 GB/s，而同一个 cutlass 后端在不同 shape 上的单 kernel 聚合带宽是
61–890 GB/s（k_proj 165、down_proj 745、gate_proj 813、lm_head 890）。
本脚本对每个真实 shape 试遍 SM120 上可用的 ``mm_fp4`` backend
（cutlass / cudnn / b12x / cute-dsl / auto）以及能拿到的 tactic 入口
（cutlass 的 32 档 CtaShape × swap_ab × DP/StreamK、cudnn 的 engine knob、
b12x 的 tile/prefetch 组合），按真实实例数加权给出整步估计：

    est_step_ms = Σ_shape (该 shape 全部真实实例的聚合 ms)
    est_gbs     = 14.399 GB / est_step_ms

设计
----
- **同一批实例**：每个 case 扫过 checkpoint 里该 shape 的**全部**真实实例
  （401 个 NVFP4 Linear 一一对应），不是单实例重复；
- **同一份 A/W**：所有 backend/tactic 都吃同一份 ``nvfp4_quantize`` 的
  ``a_fp4``/``a_sf`` 和同一批 checkpoint 权重 + 128x4 swizzled scale；
- **L2-free**：GB203 的 L2 约 64 MB。小 shape（k_proj/v_proj 各 16 × 2.6 MB）
  的真实实例工作集只有 ~42 MB，会整块住进 L2；对这些 case 复制 ``copies``
  份**不同地址**的权重副本，把工作集补到 ≥4×L2，计时结果再除以 ``copies``
  还原「真实实例数」的 ms。其余 shape 的真实实例工作集已经 ≥4×L2。
- **graph replay 边界**：一个计时单元 = 该 case 全部 (实例 × 副本) 的逐层
  launch，capture 成 CUDA graph 后 replay，CUDA event 取中位数；区间里没有
  Python/dispatch 开销，和真实 8K decode 的 graph replay 边界一致。
- **同一个 event 边界**：候选与 ``cutlass`` 基线都走同一套实例、同一份
  A/W、同一个 graph capture / event 计时函数。

正确性 gate（实现前冻结，见 decisions.md 的 M2 NVFP4 条目）
-----------------------------------------------------------
- 输出 finite；
- 与「同一份量化后的 A（a_fp4/a_sf）+ checkpoint W 的 128x4 swizzled scale
  反量化」的 FP32 matmul 比：``nrmse ≤ 5e-3``、``cosine ≥ 0.99999``、
  ``max_rel ≤ 1e-2``；
- 不同 backend 之间（同一份 A/W）``nrmse ≤ 1e-3``，并记录最大绝对差；
- 接线自检：``mm_fp4(backend="cutlass")`` 的默认 tactic 必须与某一档显式
  cutlass tactic 逐位相同（预期是 tactic 21 = CtaShape128x128x128/noSwap/DP），
  同时验证「直接调 cutlass module」的调用方式。

计时口径的两个已知坑（第 1 轮 B/C 两条线踩过）
----------------------------------------------
1. 工作集能放进 L2 时反复跑同一份权重会报出 >960 GB/s 的假数据 → 多副本；
2. ``mm_fp4`` 的 Python/dispatch 开销在小 shape 上恒为 60–68 µs → graph replay。

用法::

    .venv/bin/python bench/micro/nvfp4_tactics.py --out {output}/op
    .venv/bin/python bench/micro/nvfp4_tactics.py --only lm_head --out .tmp/tactics
    .venv/bin/python bench/micro/nvfp4_tactics.py --check-shapes   # 纯 CPU
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
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


# ------------------------------------------------------------------ 清单
# 产物 case 名；与 checkpoint 的 NVFP4 Linear 一一对应（AST 可求值的 tuple）。
SCENARIOS = (
    "in_proj_qkv",
    "in_proj_z",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
)

# case -> (checkpoint tensor 名后缀（base 名，含结尾的 '.'）, 期望实例数)
#
# 实例数来自 model.safetensors.index.json 的真实清单：400 个文本 NVFP4 Linear
# + lm_head = 401；权重 + scale = 14.399 GB（与 profile 的 14.399 GB 同口径）。
SHAPES: dict[str, tuple[tuple[str, ...], int]] = {
    "in_proj_qkv": ((".linear_attn.in_proj_qkv.",), 48),
    "in_proj_z": ((".linear_attn.in_proj_z.",), 48),
    "q_proj": ((".self_attn.q_proj.",), 16),
    "k_proj": ((".self_attn.k_proj.",), 16),
    "v_proj": ((".self_attn.v_proj.",), 16),
    "o_proj": ((".self_attn.o_proj.",), 16),
    "out_proj": ((".linear_attn.out_proj.",), 48),
    "gate_proj": ((".mlp.gate_proj.",), 64),
    "up_proj": ((".mlp.up_proj.",), 64),
    "down_proj": ((".mlp.down_proj.",), 64),
    "lm_head": (("lm_head.",), 1),
}

MODEL_DIR = ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"

# 实现前冻结的容差（decisions.md: M2 NVFP4）。
TOL = {"nrmse": 5.0e-3, "cosine": 0.99999, "max_rel": 1.0e-2}
# 不同 backend 之间的一致性阈值（本任务新增）。
AGREE_NRMSE = 1.0e-3

# profile 的整步参照（集成后 graph-7680 8 步，job-000028）。
PROFILE_STEP_MS = 19.299
PROFILE_TOTAL_GB = 14.399

# cutlass SM120 的 tactic 表顺序（fp4_gemm_cutlass_sm120.cu 的 getConfigs）：
# 8 个 tile × {swap_ab true/false} × {DP, StreamK}。tactic=-1 等价于
# 128x128x128 + swap_ab=false + DP，即 index 21（接线自检用）。
CUTLASS_TILES = (
    "128x32x64",
    "128x32x128",
    "128x64x64",
    "128x64x128",
    "128x128x64",
    "128x128x128",
    "256x128x64",
    "128x256x64",
)
CUTLASS_DEFAULT_TACTIC = 21
# 如果实现/枚举顺序变了，接线自检会失败；枚举顺序以 C++ getConfigs 为准。

# 计时单元的工作集目标：≥4×L2（GB203 L2 ≈ 64 MB）。
L2_TARGET_MULTIPLE = 4.0
MAX_COPIES = 8

# 每个 case 常驻的权重量上限（16 GiB 卡；超过就截断并记录，仍远超 L2）。
MAX_CASE_WEIGHT_GB = 4.0

# 各 backend 在 SM120 上的可用性（由 flashinfer 0.7.2 的 requirement 决定）：
# - cutlass / cudnn / b12x / cute-dsl 都支持 sm120；trtllm 与
#   cutedsl_low_latency 的 requirement 只到 SM100/103/107，不在本机可用列表；
# - b12x 需要 CUDA >= 12.9（NVFP4），cute-dsl 需要 CUDA 13 + cutlass-dsl；
# - 四者都支持 M=1、block_size=16、use_8x4_sf_layout=False（cute-dsl/b12x/
#   cudnn 会显式拒绝 8x4）。
DEFAULT_BACKENDS = ("cutlass", "b12x", "cute-dsl", "cudnn")


# ------------------------------------------------------------------ 纯 CPU 清单工具
def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise RuntimeError(f"{path} 不是合法的 safetensors 文件")
        (size,) = struct.unpack("<Q", raw)
        return json.loads(handle.read(size))


def checkpoint_manifest(model_dir: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """只读 safetensors header：name -> (dtype, shape)。不加载权重、不建 CUDA。"""
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    headers: dict[str, dict] = {}
    for shard in sorted(set(weight_map.values())):
        headers[shard] = _read_safetensors_header(model_dir / shard)
    return {
        name: (headers[shard][name]["dtype"], tuple(headers[shard][name]["shape"]))
        for name, shard in weight_map.items()
    }


def nvfp4_inventory(model_dir: Path) -> list[dict]:
    """checkpoint 里所有 NVFP4 Linear 的 (base, n, k)，按名字排序。"""
    manifest = checkpoint_manifest(model_dir)
    items: list[dict] = []
    for name, (dtype, shape) in manifest.items():
        if not name.endswith(".weight") or dtype != "U8":
            continue
        base = name[: -len("weight")]
        scale = base + "weight_scale"
        if scale not in manifest or manifest[scale][0] != "F8_E4M3":
            continue
        if len(shape) != 2 or len(manifest[scale][1]) != 2:
            continue
        n, k_half = shape
        items.append(
            {
                "name": base,
                "n": int(n),
                "k": 2 * int(k_half),
                "scale_shape": tuple(int(x) for x in manifest[scale][1]),
            }
        )
    items.sort(key=lambda item: item["name"])
    return items


def case_of(name: str) -> str | None:
    """tensor base 名（以 '.' 结尾）-> case 名。"""
    for case, (suffixes, _) in SHAPES.items():
        for suffix in suffixes:
            if name.endswith(suffix):
                return case
    return None


def inventory_summary(items: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    for item in items:
        case = case_of(item["name"])
        if case is None:
            continue
        weight = item["n"] * item["k"] // 2 + item["n"] * item["k"] // 16 + 8
        group = groups.setdefault(
            case,
            {
                "case": case,
                "instances": 0,
                "n": item["n"],
                "k": item["k"],
                "bytes": 0,
            },
        )
        if (group["n"], group["k"]) != (item["n"], item["k"]):
            raise RuntimeError(
                f"{case} 内出现不同 shape: {(group['n'], group['k'])} vs "
                f"{(item['n'], item['k'])}"
            )
        group["instances"] += 1
        group["bytes"] += weight
    return {
        "groups": groups,
        "total_instances": len(items),
        "total_bytes": sum(
            item["n"] * item["k"] // 2 + item["n"] * item["k"] // 16 + 8
            for item in items
        ),
    }


def check_shapes() -> int:
    """纯 CPU 自检：SCENARIOS / SHAPES 与 checkpoint 清单一致。"""
    items = nvfp4_inventory(MODEL_DIR)
    summary = inventory_summary(items)
    unmapped = [item["name"] for item in items if case_of(item["name"]) is None]
    groups = summary["groups"]
    problems: list[str] = []
    if unmapped:
        problems.append(f"未映射到 case 的 NVFP4 Linear: {unmapped[:5]} …({len(unmapped)})")
    for case, (_, expected) in SHAPES.items():
        group = groups.get(case)
        if group is None:
            problems.append(f"{case}: checkpoint 里没有实例")
            continue
        if group["instances"] != expected:
            problems.append(f"{case}: 实例数 {group['instances']} != 期望 {expected}")
    extra = sorted(set(groups) - set(SCENARIOS))
    if extra:
        problems.append(f"SHAPES 有、SCENARIOS 没有的 case: {extra}")
    missing = sorted(set(SCENARIOS) - set(groups))
    if missing:
        problems.append(f"SCENARIOS 有、checkpoint 没有的 case: {missing}")

    print(
        f"NVFP4 Linear: {summary['total_instances']} 个, "
        f"{summary['total_bytes'] / 1e9:.4f} GB"
    )
    for case in SCENARIOS:
        group = groups.get(case)
        if group is None:
            continue
        print(
            f"  {case:12s} n={group['n']:6d} k={group['k']:5d} "
            f"instances={group['instances']:3d} "
            f"bytes={group['bytes'] / 1e9:.4f} GB"
        )
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        return 1
    print("ok: SCENARIOS / SHAPES 与 checkpoint 清单一致")
    return 0


# ------------------------------------------------------------------ 运行时工具
def _u8(tensor):
    import torch

    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.view(torch.uint8)


def _scale_f8(tensor):
    """block scale 按 UE4M3 数值解释（不能 view 成 uint8 再 .float()）。"""
    import torch

    if tensor.dtype == torch.float8_e4m3fn:
        return tensor
    if tensor.element_size() != 1:
        raise TypeError(f"block scale 必须是 1 字节 dtype，得到 {tensor.dtype}")
    return tensor.view(torch.float8_e4m3fn)


def dequant_activation(a_fp4, a_sf, k: int):
    """用 frozen reference 的 primitive 解 FlashInfer 的 activation 量化结果。"""
    from runner.ops.nvfp4.reference import (
        e2m1_to_float,
        unswizzle_scale_128x4,
        unpack_e2m1,
    )

    codes = unpack_e2m1(_u8(a_fp4).reshape(-1))
    values = e2m1_to_float(codes)
    sf = unswizzle_scale_128x4(_scale_f8(a_sf), 128, k // 16)[0].float()
    return values * sf.repeat_interleave(16)


def dequant_weight_scale(swizzled, n: int, k: int):
    from runner.ops.nvfp4.reference import unswizzle_scale_128x4

    return unswizzle_scale_128x4(_scale_f8(swizzled), n, k // 16).float()


def reference_out(a_scaled, weight, wsf_linear, alpha, chunk: int = 4096):
    """``(a4*a_sf) · (w4*w_sf) * alpha`` 的 FP32 分块 matmul（reference）。"""
    import torch
    from runner.ops.nvfp4.reference import e2m1_to_float, unpack_e2m1

    n = int(weight.shape[0])
    out = torch.empty(n, dtype=torch.float32, device=weight.device)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        values = e2m1_to_float(unpack_e2m1(weight[start:end]))
        values *= wsf_linear[start:end].repeat_interleave(16, dim=1)
        out[start:end] = values @ a_scaled
    return out * alpha


def find_instances(reader, suffixes: tuple[str, ...]) -> list[str]:
    """checkpoint 里该 case 的全部真实实例（base 名，含结尾的 '.'）。"""
    prefixes: list[str] = []
    for name in sorted(reader.weight_map):
        if not name.endswith("weight"):
            continue
        base = name[: -len("weight")]
        if any(base.endswith(suffix) for suffix in suffixes):
            prefixes.append(base)
    return prefixes


class BackendConfig:
    """一个候选 (backend, tactic)。

    mode:
      - ``mm``: 走官方入口 ``mm_fp4(backend=family)``，tactic 为后端默认；
      - ``cutlass_tactic``: 直接调 cutlass module 的 ``fp4_gemm(..., tactic)``；
      - ``runner_tactic``: 调对应后端的 TunableRunner（cudnn/b12x）。
    """

    __slots__ = ("name", "family", "mode", "tactic", "integrable")

    def __init__(self, name: str, family: str, mode: str, tactic=None, *, integrable: bool):
        self.name = name
        self.family = family
        self.mode = mode
        self.tactic = tactic
        self.integrable = bool(integrable)

    def __repr__(self) -> str:  # pragma: no cover - 只用于打印
        return f"<{self.name} {self.mode} tactic={self.tactic}>"


def cutlass_tactic_label(index: int) -> str:
    tile = CUTLASS_TILES[index // 4] if index // 4 < len(CUTLASS_TILES) else "?"
    swap = "swapAB" if index % 4 in (0, 2) else "noSwap"
    sched = "streamK" if index % 4 in (2, 3) else "DP"
    return f"{tile}/{swap}/{sched}"


class Backends:
    """惰性构造各 backend 的 runner / module（首次使用时才 JIT）。"""

    def __init__(self):
        self._cutlass = None
        self._cudnn = None
        self._b12x = None
        self._cute = None
        self._workspace = None
        self.errors: dict[str, str] = {}

    def workspace(self):
        import torch
        from flashinfer.gemm.gemm_base import DEFAULT_WORKSPACE_SIZE

        if self._workspace is None:
            self._workspace = torch.empty(
                DEFAULT_WORKSPACE_SIZE, dtype=torch.uint8, device="cuda"
            )
        return self._workspace

    def get(self, family: str):
        import torch

        if family in self.errors:
            return None
        try:
            if family == "cutlass":
                if self._cutlass is None:
                    from flashinfer.gemm.gemm_base import (
                        get_gemm_sm120_module_cutlass_fp4,
                    )

                    self._cutlass = get_gemm_sm120_module_cutlass_fp4()
                return self._cutlass
            if family == "cudnn":
                if self._cudnn is None:
                    from flashinfer.gemm.gemm_base import (
                        _MM_FP4_TUNING_CONFIG_128x4,
                        _cudnn_gemm_fp4_runner,
                    )

                    self._cudnn = _cudnn_gemm_fp4_runner(_MM_FP4_TUNING_CONFIG_128x4)
                return self._cudnn
            if family == "b12x":
                if self._b12x is None:
                    from flashinfer.gemm.gemm_base import _b12x_gemm_fp4_runner

                    self._b12x = _b12x_gemm_fp4_runner(12, 0, True, torch.bfloat16, True)
                return self._b12x
            if family == "cute-dsl":
                if self._cute is None:
                    from flashinfer.gemm.gemm_base import _cute_dsl_gemm_fp4_runner

                    self._cute = _cute_dsl_gemm_fp4_runner(
                        12, 0, True, torch.bfloat16, True
                    )
                return self._cute
        except Exception as exc:  # noqa: BLE001 - 记录不可用原因，不中断
            self.errors[family] = f"{type(exc).__name__}: {exc}"
            return None
        raise ValueError(f"unknown backend family: {family}")

    def available(self, family: str) -> bool:
        return self.get(family) is not None


class CaseContext:
    """一个 case 的运行时数据：实例、A、reference、输出缓冲。"""

    __slots__ = (
        "case",
        "n",
        "k",
        "instances",
        "instances_total",
        "copies",
        "per_instance_bytes",
        "aggregate_bytes",
        "l2_bytes",
        "a_fp4",
        "a_u8",
        "a_sf",
        "a_sf_u8",
        "ref",
        "out",
        "out_num",
        "instance0",
    )


def _load_case(reader, case: str, l2_bytes: int, max_copies: int) -> CaseContext:
    """加载该 shape 的全部真实实例（+ 多副本）并算好 reference。"""
    import torch
    from flashinfer import SfLayout, nvfp4_quantize
    from runner.ops.nvfp4.bytes import weight_bytes
    from runner.ops.nvfp4.reference import swizzle_scale_128x4

    suffixes, expected = SHAPES[case]
    prefixes = find_instances(reader, suffixes)
    if not prefixes:
        raise RuntimeError(f"checkpoint 里找不到 {case} 的实例（后缀 {suffixes}）")
    if len(prefixes) != expected:
        raise RuntimeError(
            f"{case} 的实例数 {len(prefixes)} != SHAPES 期望 {expected}"
        )

    first = reader.get(prefixes[0] + "weight", device="cuda")
    n, k_half = (int(x) for x in first.shape)
    k = 2 * k_half
    del first

    per_instance_bytes = weight_bytes(k, n)
    aggregate_bytes = per_instance_bytes * len(prefixes)
    limit = int(MAX_CASE_WEIGHT_GB * 1e9 // per_instance_bytes)
    if len(prefixes) > limit:
        raise RuntimeError(
            f"{case}: 需要 {len(prefixes)} 个实例（{aggregate_bytes / 1e9:.2f} GB），"
            f"超过 MAX_CASE_WEIGHT_GB={MAX_CASE_WEIGHT_GB} 的上限；"
            "本任务要求覆盖全部真实实例，不能截断"
        )
    used = list(prefixes)
    copies = max(
        1,
        min(
            max_copies,
            int(math.ceil(L2_TARGET_MULTIPLE * l2_bytes / max(1, aggregate_bytes))),
        ),
    )

    instances = []
    for prefix in used:
        weight = reader.get(prefix + "weight", device="cuda")
        scale_linear = reader.get(prefix + "weight_scale", device="cuda")
        scale_swizzled = swizzle_scale_128x4(scale_linear)
        del scale_linear
        input_scale = reader.get(prefix + "input_scale", device="cuda").reshape(1).float()
        weight_scale_2 = (
            reader.get(prefix + "weight_scale_2", device="cuda").reshape(1).float()
        )
        alpha = (input_scale * weight_scale_2).reshape(1).float()
        replicas = [(weight, scale_swizzled)]
        for _ in range(copies - 1):
            replicas.append((weight.clone(), scale_swizzled.clone()))
        instances.append({"alpha": alpha, "replicas": replicas})

    instance0 = instances[0]
    w0, ws0 = instance0["replicas"][0]

    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261023)
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.5
    a_fp4, a_sf = nvfp4_quantize(
        x,
        (1.0 / reader.get(prefixes[0] + "input_scale", device="cuda")).reshape(1),
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cuda",
    )

    # reference：同一份量化后的 A + checkpoint W 的 128x4 swizzled scale。
    a_scaled = dequant_activation(a_fp4, a_sf, k)
    wsf_linear = dequant_weight_scale(ws0, n, k)
    ref = reference_out(a_scaled, w0, wsf_linear, instance0["alpha"])
    del wsf_linear

    ctx = CaseContext()
    ctx.case = case
    ctx.n = n
    ctx.k = k
    ctx.instances = instances
    ctx.instances_total = len(prefixes)
    ctx.copies = copies
    ctx.per_instance_bytes = per_instance_bytes
    ctx.aggregate_bytes = aggregate_bytes
    ctx.l2_bytes = l2_bytes
    ctx.a_fp4 = a_fp4
    ctx.a_u8 = a_fp4.view(torch.uint8)
    ctx.a_sf = a_sf
    ctx.a_sf_u8 = a_sf.view(torch.uint8)
    ctx.ref = ref
    ctx.out = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    ctx.out_num = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
    ctx.instance0 = instance0
    return ctx


def _runner_inputs(ctx: CaseContext, instance, out, backends: Backends) -> list:
    import torch

    weight, scale = instance["replicas"][0]
    return [
        ctx.a_fp4,
        weight.T,
        ctx.a_sf,
        scale.T,
        instance["alpha"],
        torch.bfloat16,
        out,
        16,
        True,
        backends.workspace(),
    ]


def _make_call(config: BackendConfig, ctx: CaseContext, backends: Backends):
    """把 (backend, tactic) 变成一个 ``call(weight, scale, alpha, out)``。"""
    import torch
    from flashinfer import mm_fp4

    if config.mode == "mm":
        backend = config.family

        def call(weight, scale, alpha, out):
            mm_fp4(
                ctx.a_fp4,
                weight.T,
                ctx.a_sf,
                scale.T,
                alpha,
                torch.bfloat16,
                out,
                block_size=16,
                use_8x4_sf_layout=False,
                backend=backend,
                use_nvfp4=True,
            )

        return call

    if config.mode == "cutlass_tactic":
        module = backends.get("cutlass")
        workspace = backends.workspace()
        tactic = config.tactic

        def call(weight, scale, alpha, out):
            module.fp4_gemm(
                ctx.a_u8,
                weight,
                ctx.a_sf_u8,
                _u8(scale),
                alpha,
                out,
                workspace,
                tactic,
            )

        return call

    if config.mode == "runner_tactic":
        runner = backends.get(config.family)
        tactic = config.tactic
        workspace = backends.workspace()
        out_dtype = torch.bfloat16

        def call(weight, scale, alpha, out):
            runner(
                inputs=[
                    ctx.a_fp4,
                    weight.T,
                    ctx.a_sf,
                    scale.T,
                    alpha,
                    out_dtype,
                    out,
                    16,
                    True,
                    workspace,
                ],
                tactic=tactic,
            )

        return call

    raise ValueError(f"unknown mode: {config.mode}")


def _sweep(ctx: CaseContext, call) -> None:
    for instance in ctx.instances:
        alpha = instance["alpha"]
        for weight, scale in instance["replicas"]:
            call(weight, scale, alpha, ctx.out)


def _time_graph(ctx: CaseContext, call, warmup: int, iters: int):
    """graph replay 计时（只算 GPU 时间）；返回 (median_ms, mode)。"""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _sweep(ctx, call)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=side):
            _sweep(ctx, call)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - capture 失败时退回 eager 计时
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for _ in range(warmup):
            _sweep(ctx, call)
        torch.cuda.synchronize()
        for start, end in zip(starts, ends):
            start.record()
            _sweep(ctx, call)
            end.record()
        torch.cuda.synchronize()
        samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        del side
        return statistics.median(samples), f"eager({type(exc).__name__})"

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    del graph, side
    return statistics.median(samples), "graph"


def _cutlass_configs(backends: Backends, args) -> list[BackendConfig]:
    configs = [BackendConfig("cutlass:default", "cutlass", "mm", None, integrable=True)]
    if "cutlass" in args.scan_tactics and backends.available("cutlass"):
        module = backends.get("cutlass")
        total = int(module.fp4_gemm_tactic_num())
        for index in range(total):
            configs.append(
                BackendConfig(
                    f"cutlass:t{index}({cutlass_tactic_label(index)})",
                    "cutlass",
                    "cutlass_tactic",
                    index,
                    integrable=False,
                )
            )
    return configs


def _mm_default(family: str, integrable: bool = True) -> BackendConfig:
    return BackendConfig(f"{family}:default", family, "mm", None, integrable=integrable)


def _runner_tactic_configs(family: str, ctx: CaseContext, backends: Backends, cap: int):
    runner = backends.get(family)
    if runner is None:
        return []
    inputs = _runner_inputs(ctx, ctx.instance0, ctx.out, backends)
    try:
        tactics = list(runner.get_valid_tactics(inputs, None))
    except Exception as exc:  # noqa: BLE001
        backends.errors[family] = f"get_valid_tactics: {type(exc).__name__}: {exc}"
        return []
    configs = []
    for index, tactic in enumerate(tactics[:cap]):
        configs.append(
            BackendConfig(
                f"{family}:t{index}", family, "runner_tactic", tactic, integrable=False
            )
        )
    return configs


def _measure(
    ctx: CaseContext,
    config: BackendConfig,
    report_case,
    backends: Backends,
    args,
    ref_state: dict,
) -> dict | None:
    """跑一个候选：数值检查 + graph 计时；不可用返回 None。"""
    import torch
    from runner.ops.nvfp4.reference import error_metrics

    try:
        call = _make_call(config, ctx, backends)
    except Exception as exc:  # noqa: BLE001
        backends.errors.setdefault(config.name, f"{type(exc).__name__}: {exc}")
        return None
    weight0, scale0 = ctx.instance0["replicas"][0]

    # 1) 数值：第一个实例的第一份副本，独立 out buffer。
    try:
        ctx.out_num.zero_()
        call(weight0, scale0, ctx.instance0["alpha"], ctx.out_num)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - 记录不可用，不中断
        backends.errors.setdefault(config.name, f"{type(exc).__name__}: {exc}")
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            raise
        return None

    metrics = error_metrics(ctx.out_num.reshape(-1), ctx.ref)
    if ref_state.get("base") is None:
        ref_state["base"] = ctx.out_num.reshape(-1).clone()
    agree = error_metrics(ctx.out_num.reshape(-1), ref_state["base"])

    # 2) 计时：graph replay 扫全部实例（× 副本）。
    try:
        measured_ms, timing_mode = _time_graph(ctx, call, args.warmup, args.iters)
    except Exception as exc:  # noqa: BLE001
        backends.errors.setdefault(
            config.name, f"timing: {type(exc).__name__}: {exc}"
        )
        return None

    agg_ms = measured_ms / ctx.copies
    numeric_ok = (
        bool(metrics["finite"])
        and float(metrics["nrmse"]) <= TOL["nrmse"]
        and float(metrics["cosine"]) >= TOL["cosine"]
        and float(metrics["max_rel"]) <= TOL["max_rel"]
    )
    row = {
        "name": config.name,
        "family": config.family,
        "mode": config.mode,
        "tactic": str(config.tactic),
        "integrable": config.integrable,
        "timing_mode": timing_mode,
        "agg_ms": agg_ms,
        "gbs": ctx.aggregate_bytes / (agg_ms * 1e-3) / 1e9,
        "us_per_instance": agg_ms * 1000.0 / max(1, ctx.instances_total),
        "nrmse": float(metrics["nrmse"]),
        "cosine": float(metrics["cosine"]),
        "max_rel": float(metrics["max_rel"]),
        "finite": bool(metrics["finite"]),
        "numeric_ok": numeric_ok,
        "agree_nrmse": float(agree["nrmse"]),
        "agree_max_abs": float(agree["max_abs"]),
        "agree_cosine": float(agree["cosine"]),
        "is_base": ref_state.get("base_name") is None,
    }
    if ref_state.get("base_name") is None:
        ref_state["base_name"] = config.name
        row["agree_nrmse"] = 0.0
        row["agree_max_abs"] = 0.0
        row["agree_cosine"] = 1.0

    _safe_metric(report_case, f"agg_ms[{config.name}]", agg_ms)
    _safe_metric(report_case, f"gbs[{config.name}]", row["gbs"])
    _safe_metric(report_case, f"us_per_instance[{config.name}]", row["us_per_instance"])
    _safe_metric(report_case, f"nrmse[{config.name}]", row["nrmse"])
    print(
        f"  {config.name:44s} agg={agg_ms:8.3f} ms ({row['gbs']:6.1f} GB/s) "
        f"{timing_mode:6s} nrmse={row['nrmse']:.2e} cos={row['cosine']:.8f} "
        f"agree={row['agree_nrmse']:.2e} maxabs={row['agree_max_abs']:.2e}"
        + ("" if row["numeric_ok"] else "  [NUMERIC-FAIL]"),
        flush=True,
    )
    return row


def _safe_metric(report_case, name: str, value) -> None:
    """写 metric；非有限值（NaN/Inf）跳过，避免 OpReport 的有限数值校验失败。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(number):
        report_case.metric(name, number)


def _numeric_gate(report_case, row: dict) -> None:
    report_case.gate(
        f"numeric: {row['name']} (nrmse/cosine/max_rel/finite)",
        row["numeric_ok"],
        {
            "nrmse": row["nrmse"],
            "cosine": row["cosine"],
            "max_rel": row["max_rel"],
            "finite": row["finite"],
            "tol": TOL,
            "vs": "同一份量化后的 A + checkpoint W dequant 的 FP32 matmul（第一个实例）",
        },
    )


def _agreement_gate(report_case, row: dict, base_name: str) -> None:
    report_case.gate(
        f"backend-agreement: {row['name']} vs {base_name} (nrmse <= 1e-3)",
        row["agree_nrmse"] <= AGREE_NRMSE,
        {
            "nrmse": row["agree_nrmse"],
            "max_abs": row["agree_max_abs"],
            "cosine": row["agree_cosine"],
            "threshold_nrmse": AGREE_NRMSE,
        },
    )


def _run_case(reader, case_name: str, report, args, backends: Backends) -> dict:
    import torch
    l2_bytes = int(torch.cuda.get_device_properties(0).L2_cache_size)
    ctx = _load_case(reader, case_name, l2_bytes, args.max_copies)
    working_set = ctx.aggregate_bytes * ctx.copies
    l2_multiple = working_set / max(1, l2_bytes)
    l2_clean = l2_multiple >= L2_TARGET_MULTIPLE

    report_case = report.case(
        ctx.case,
        n=ctx.n,
        k=ctx.k,
        instances=ctx.instances_total,
        instances_measured=len(ctx.instances),
        copies=ctx.copies,
        per_instance_bytes=ctx.per_instance_bytes,
        aggregate_weight_bytes=ctx.aggregate_bytes,
        working_set_gb=working_set / 1e9,
        l2_mib=l2_bytes / 1024**2,
        l2_multiple=l2_multiple,
        l2_clean=l2_clean,
        timing="per-case CUDA graph replay (all real instances x copies), CUDA event median",
        bytes_model="packed weight + block scale + 2 fp32 scalars",
        tolerance=TOL,
        agreement_tolerance_nrmse=AGREE_NRMSE,
        note=(
            "copies > 1 时计的是 copies 份不同地址副本，结果已除以 copies 还原"
            "真实实例数；候选与 cutlass 基线共用同一批实例/A/W/event 边界。"
            "A 只用第一个实例的 input_scale 量化一次并共用（计时口径；"
            "数值 gate 也只对第一个实例，alpha 仍用各实例自己的 global scale）"
        ),
    )

    ref_state: dict = {"base": None, "base_name": None}
    rows: list[dict] = []
    unavailable: dict[str, str] = {}

    # 分阶段构造（惰性）：cutlass 先测量（建立基线），cudnn/b12x 的 tactic
    # 枚举可能触发 JIT/图构建，放在后面；某个阶段构造失败不影响已有数据。
    stages: list[tuple[str, object]] = [
        ("cutlass", lambda: _cutlass_configs(backends, args)),
        ("b12x", lambda: [_mm_default("b12x")]),
        ("cute-dsl", lambda: [_mm_default("cute-dsl")]),
        ("cudnn", lambda: [_mm_default("cudnn")]),
        ("auto", lambda: [_mm_default("auto", integrable=False)]),
    ]
    if "b12x" in args.scan_tactics:
        stages.append(
            (
                "b12x-tactics",
                lambda: _runner_tactic_configs(
                    "b12x", ctx, backends, args.b12x_tactics
                ),
            )
        )
    if "cudnn" in args.scan_tactics:
        stages.append(
            (
                "cudnn-tactics",
                lambda: _runner_tactic_configs(
                    "cudnn", ctx, backends, args.cudnn_tactics
                ),
            )
        )

    for stage_name, build in stages:
        try:
            configs = build()
        except Exception as exc:  # noqa: BLE001 - 记录后继续下一阶段
            unavailable[f"stage:{stage_name}"] = f"{type(exc).__name__}: {exc}"
            continue
        for config in configs:
            row = _measure(ctx, config, report_case, backends, args, ref_state)
            if row is None:
                unavailable[config.name] = backends.errors.get(
                    config.name, backends.errors.get(config.family, "unavailable")
                )
                continue
            rows.append(row)

    if not rows:
        raise RuntimeError(f"{ctx.case}: 没有任何候选可用")

    graph_rows = [row for row in rows if row["timing_mode"].startswith("graph")] or rows
    integrable_rows = [
        row for row in graph_rows if row["integrable"] and row["numeric_ok"]
    ] or [row for row in graph_rows if row["integrable"]] or graph_rows
    best_integrable = min(integrable_rows, key=lambda row: row["agg_ms"])
    best_any = min(
        (row for row in graph_rows if row["numeric_ok"]),
        key=lambda row: row["agg_ms"],
        default=min(graph_rows, key=lambda row: row["agg_ms"]),
    )
    base_row = next((row for row in rows if row["name"] == "cutlass:default"), None)
    if base_row is None:
        base_row = next((row for row in rows if row["is_base"]), None)
    base_name = ref_state.get("base_name") or "cutlass:default"

    # 硬 gate：所有默认 backend（可接入的）+ 最终选中的 best_any。
    gated: set[str] = set()
    for row in rows:
        if row["mode"] == "mm":
            _numeric_gate(report_case, row)
            if not row["is_base"]:
                _agreement_gate(report_case, row, base_name)
            gated.add(row["name"])
    if best_any["name"] not in gated:
        _numeric_gate(report_case, best_any)
        _agreement_gate(report_case, best_any, base_name)
        gated.add(best_any["name"])

    # 接线自检：默认 cutlass tactic 必须与某档显式 tactic 逐位相同。
    cutlass_tactic_rows = [
        row
        for row in rows
        if row["mode"] == "cutlass_tactic" and row["agree_max_abs"] == 0.0
    ]
    tactic21 = next(
        (row for row in rows if row["name"].startswith(f"cutlass:t{CUTLASS_DEFAULT_TACTIC}(")),
        None,
    )
    report_case.gate(
        "wiring: cutlass 默认 tactic 与显式 tactic 逐位一致（直调 module 接线自检）",
        bool(cutlass_tactic_rows),
        {
            "bitwise_equal_tactics": [row["name"] for row in cutlass_tactic_rows],
            "expected_default": f"t{CUTLASS_DEFAULT_TACTIC}"
            f"({cutlass_tactic_label(CUTLASS_DEFAULT_TACTIC)})",
            "tactic21_agree_max_abs": None
            if tactic21 is None
            else tactic21["agree_max_abs"],
            "tactic21_agg_ms": None if tactic21 is None else tactic21["agg_ms"],
            "note": "逐位相同才算接线正确；ms 只作诊断",
        },
    )
    if tactic21 is not None and base_row is not None:
        _safe_metric(
            report_case,
            "tactic21_vs_default_agg_ms_diff",
            abs(tactic21["agg_ms"] - base_row["agg_ms"]),
        )

    report_case.diag(
        "diagnostic: 工作集 >= 4x L2（聚合 GB/s 才干净）",
        l2_clean,
        {
            "working_set_gb": working_set / 1e9,
            "l2_mib": l2_bytes / 1024**2,
            "l2_multiple": l2_multiple,
            "copies": ctx.copies,
            "note": "不干净时绝对值被 L2 抬高，但所有候选受同样影响，比值仍可比",
        },
    )
    for name, reason in unavailable.items():
        report_case.diag(f"unavailable: {name}", False, {"reason": reason})

    if base_row is not None:
        _safe_metric(report_case, "cutlass_default_agg_ms", base_row["agg_ms"])
        _safe_metric(report_case, "cutlass_default_gbs", base_row["gbs"])
        _safe_metric(
            report_case,
            "speedup_best_integrable_vs_cutlass",
            base_row["agg_ms"] / best_integrable["agg_ms"],
        )
        _safe_metric(
            report_case,
            "speedup_best_any_vs_cutlass",
            base_row["agg_ms"] / best_any["agg_ms"],
        )
    _safe_metric(report_case, "best_integrable_agg_ms", best_integrable["agg_ms"])
    _safe_metric(report_case, "best_integrable_gbs", best_integrable["gbs"])
    _safe_metric(report_case, "best_any_agg_ms", best_any["agg_ms"])
    _safe_metric(report_case, "best_any_gbs", best_any["gbs"])
    _safe_metric(report_case, "instances_total", float(ctx.instances_total))
    _safe_metric(report_case, "aggregate_gb", ctx.aggregate_bytes / 1e9)
    _safe_metric(report_case, "l2_multiple", l2_multiple)
    _safe_metric(report_case, "available_configs", float(len(rows)))
    _safe_metric(report_case, "unavailable_configs", float(len(unavailable)))

    for family in DEFAULT_BACKENDS:
        produced = any(row["family"] == family for row in rows)
        report_case.gate(
            f"coverage: backend {family} 可用性或不可用原因已记录",
            produced or family in backends.errors or any(
                name.startswith(family + ":") for name in unavailable
            ),
            {
                "produced_configs": sum(1 for row in rows if row["family"] == family),
                "error": backends.errors.get(family),
            },
        )
    report_case.gate(
        "coverage: 该 case 的实例数 == checkpoint 清单",
        ctx.instances_total == SHAPES[ctx.case][1],
        {
            "instances": ctx.instances_total,
            "expected": SHAPES[ctx.case][1],
            "suffixes": list(SHAPES[ctx.case][0]),
        },
    )
    report_case.gate(
        "coverage: 工作集 >= 4x L2（必要时已用多副本）",
        l2_clean,
        {
            "l2_multiple": l2_multiple,
            "copies": ctx.copies,
            "working_set_gb": working_set / 1e9,
        },
    )

    print(
        f"{ctx.case}: n={ctx.n} k={ctx.k} inst={ctx.instances_total} "
        f"copies={ctx.copies} work={working_set / 1e9:.2f} GB ({l2_multiple:.1f}x L2) "
        f"cutlass_default={base_row['agg_ms']:.3f} ms "
        f"best_integrable={best_integrable['name']} {best_integrable['agg_ms']:.3f} ms "
        f"({best_integrable['gbs']:.1f} GB/s) "
        f"best_any={best_any['name']} {best_any['agg_ms']:.3f} ms "
        f"({best_any['gbs']:.1f} GB/s)"
        if base_row is not None
        else f"{ctx.case}: 无 cutlass 基线",
        flush=True,
    )

    result = {
        "case": ctx.case,
        "n": ctx.n,
        "k": ctx.k,
        "instances": ctx.instances_total,
        "aggregate_bytes": ctx.aggregate_bytes,
        "baseline_ms": None if base_row is None else base_row["agg_ms"],
        "baseline_gbs": None if base_row is None else base_row["gbs"],
        "best_integrable": {key: best_integrable[key] for key in (
            "name", "family", "agg_ms", "gbs", "us_per_instance", "nrmse", "cosine",
            "max_rel", "numeric_ok",
        )},
        "best_any": {key: best_any[key] for key in (
            "name", "family", "agg_ms", "gbs", "us_per_instance", "nrmse", "cosine",
            "max_rel", "numeric_ok",
        )},
        "rows": rows,
        "unavailable": unavailable,
    }
    del ctx
    torch.cuda.empty_cache()
    return result


# ------------------------------------------------------------------ CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="all",
        help=f"逗号分隔的 case 名，默认 all（{', '.join(SCENARIOS)}）",
    )
    parser.add_argument("--out", default=None, help="产物目录（写 op.json）")
    parser.add_argument("--warmup", type=int, default=5, help="graph replay warmup 次数")
    parser.add_argument("--iters", type=int, default=20, help="graph replay 计时次数")
    parser.add_argument(
        "--max-copies",
        type=int,
        default=MAX_COPIES,
        help="小 shape 的多副本上限（把工作集补到 >=4x L2）",
    )
    parser.add_argument(
        "--scan-tactics",
        default="cutlass,cudnn,b12x",
        help="要扫描 tactic 的后端（逗号分隔；cutlass 无编译成本，cudnn/b12x 有 JIT）",
    )
    parser.add_argument(
        "--cudnn-tactics", type=int, default=8, help="cudnn engine knob 最多试几个"
    )
    parser.add_argument(
        "--b12x-tactics", type=int, default=3, help="b12x tile/prefetch tactic 最多试几个"
    )
    parser.add_argument(
        "--check-shapes",
        action="store_true",
        help="纯 CPU：只校验 checkpoint 清单与 SCENARIOS/SHAPES 一致",
    )
    return parser.parse_args(argv)


def _write_partial(report, results: list[dict], out_dir: Path) -> None:
    """每跑完一个 case 落一份 partial（防止后面的后端把进程弄崩时丢全部证据）。

    只是运行时的保险文件；队列校验只认 ``op.json``（finish 之后才写）。
    """
    payload = {
        "schema": "phase2-op/1-partial",
        "candidate": report.candidate_extra,
        "cases": [case.to_dict() for case in report.cases],
        "results": results,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "op.partial.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 - 保险文件失败不影响主流程
        pass


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
    if args.check_shapes:
        return check_shapes()
    if not args.out:
        raise SystemExit("--out 是必填参数（写 op.json）")
    args.scan_tactics = tuple(
        item.strip() for item in args.scan_tactics.split(",") if item.strip()
    )

    import torch
    from flashinfer import mm_fp4, nvfp4_quantize  # noqa: F401 - 提前触发 JIT 导入

    from runner.experiment.op_artifact import OpReport
    from runner.loader.checkpoint import CheckpointReader

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run in full access")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    cases = resolve_cases(args.only)
    report = OpReport(
        args.out,
        "nvfp4-mm-fp4-per-shape-backend",
        "nvfp4-w4a4-m1-bf16",
        candidate_extra={
            "backend_candidates": ["cutlass", "cudnn", "b12x", "cute-dsl", "auto"],
            "baseline": (
                "FlashInfer mm_fp4 backend=cutlass（tactic=-1 = CtaShape128x128x128/"
                "noSwap/DP，与显式 cutlass:t21 接线自检）"
            ),
            "backend_availability": (
                "SM120 上官方入口可用：cutlass / cudnn / b12x / cute-dsl / auto"
                "（auto 在 SM120 上 = b12x）；trtllm 与 cutedsl_low_latency 的 "
                "requirement 只到 SM100/103/107，不在本机可用列表"
            ),
            "tactic_entry": (
                "cutlass: module.fp4_gemm(..., tactic) 全量 32 档（8 tile x swap_ab x "
                "DP/StreamK）；cudnn: TunableRunner engine knob；b12x: tile/prefetch；"
                "cute-dsl: 小 M 专用 independent 计划（m=1 只有 1 档）"
            ),
            "timing": (
                "每个 case 扫全部真实实例（k/v 等小 shape 额外 copies 份不同地址副本，"
                "补到 >=4x L2），逐层 launch capture 成 CUDA graph 后 replay，CUDA "
                "event 取中位数；候选与 cutlass 基线共享同一批实例/A/W/event 边界"
            ),
            "reference": (
                "同一份量化后的 A（a_fp4/a_sf）+ checkpoint W 的 128x4 swizzled "
                "scale 反量化后的 FP32 matmul；容差为 M2 冻结值"
            ),
            "bytes_model": "packed weight + block scale + 2 fp32 scalars（不含 A/out）",
            "integration": (
                "本 commit 不接主线：静态表要由本 job 的 per-shape 结果决定；"
                "如果 best_integrable 有收益，下一步把 (N,K)->backend 表写死进 "
                "runner/ops/nvfp4/linear.py（M=1 才查表，prefill 仍 cutlass）"
            ),
        },
        environment={},
        steps=None,
    )
    report.environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": "%d.%d" % torch.cuda.get_device_capability(0),
        "l2_mib": torch.cuda.get_device_properties(0).L2_cache_size / 1024**2,
        "cuda_home": os.environ.get("CUDA_HOME"),
    }

    backends = Backends()
    results: list[dict] = []
    ran: list[str] = []
    reader = CheckpointReader(MODEL_DIR)
    try:
        for case in cases:
            print(f"=== {case} ===", flush=True)
            results.append(_run_case(reader, case, report, args, backends))
            ran.append(case)
            torch.cuda.empty_cache()
            _write_partial(report, results, Path(args.out))
    finally:
        reader.close()

    # 整步加权估计（按真实实例数；14.399 GB = 401 个 Linear 的 weight+scale）。
    total_bytes = sum(row["aggregate_bytes"] for row in results)
    total_instances = sum(row["instances"] for row in results)
    baseline_ms = sum(
        row["baseline_ms"] for row in results if row["baseline_ms"] is not None
    )
    baseline_complete = all(row["baseline_ms"] is not None for row in results)
    integrable_ms = sum(row["best_integrable"]["agg_ms"] for row in results)
    any_ms = sum(row["best_any"]["agg_ms"] for row in results)

    table: dict[str, str] = {}
    reasons: dict[str, str] = {}
    by_shape: dict[tuple[int, int], list[dict]] = {}
    for row in results:
        by_shape.setdefault((row["n"], row["k"]), []).append(row)
    for (n, k), rows in sorted(by_shape.items()):
        best = min(rows, key=lambda row: row["best_integrable"]["agg_ms"])[
            "best_integrable"
        ]
        key = f"{n}x{k}"
        table[key] = best["family"]
        reasons[key] = (
            f"{best['name']}: {best['agg_ms']:.3f} ms / {best['gbs']:.1f} GB/s "
            f"(instances={'+'.join(str(row['instances']) for row in rows)})"
        )

    report.candidate_extra["proposed_table"] = table
    report.candidate_extra["proposed_table_reasons"] = reasons
    report.candidate_extra["table_scope"] = (
        "M=1 decode 才查表；prefill（M>1）继续 cutlass，保持既有 bitwise 语义"
    )

    print("\n=== 整步加权估计 ===", flush=True)
    print(f"total={total_bytes / 1e9:.4f} GB instances={total_instances}")
    if baseline_complete and baseline_ms > 0:
        print(
            f"cutlass default: {baseline_ms:.3f} ms = "
            f"{total_bytes / (baseline_ms * 1e-3) / 1e9:.1f} GB/s"
        )
    else:
        print(f"cutlass default: incomplete ({baseline_ms:.3f} ms)")
    print(
        f"best integrable: {integrable_ms:.3f} ms = "
        f"{total_bytes / (integrable_ms * 1e-3) / 1e9:.1f} GB/s "
        f"(saves {baseline_ms - integrable_ms:.3f} ms)"
    )
    print(
        f"best any tactic: {any_ms:.3f} ms = "
        f"{total_bytes / (any_ms * 1e-3) / 1e9:.1f} GB/s "
        f"(saves {baseline_ms - any_ms:.3f} ms)"
    )
    for row in results:
        print(
            f"  {row['case']:12s} {row['n']:6d}x{row['k']:5d} inst={row['instances']:3d} "
            f"cutlass={row['baseline_ms']:.3f} "
            f"integrable={row['best_integrable']['agg_ms']:.3f}"
            f"({row['best_integrable']['family']}) "
            f"any={row['best_any']['agg_ms']:.3f}({row['best_any']['name']})"
        )

    missing = [case for case in SCENARIOS if case not in ran]
    baseline_gbs = (
        total_bytes / (baseline_ms * 1e-3) / 1e9
        if baseline_complete and baseline_ms > 0
        else 0.0
    )
    for case in report.cases:
        case.gate(
            "measurement: cutlass 基线整步估计在 [600, 960] GB/s（L2-free / 物理峰值校验）",
            bool(baseline_complete) and 600.0 <= baseline_gbs <= 960.0,
            {
                "cutlass_default_ms": baseline_ms,
                "cutlass_default_gbs": baseline_gbs,
                "profile_reference": {"ms": PROFILE_STEP_MS, "gbs": 747.0},
                "bounds_gbs": [600.0, 960.0],
                "note": (
                    "低于 600 说明测量协议异常（如 capture 失败退回 eager）；"
                    "高于 960 说明工作集被 L2 抬高，两者都不能作为选型证据"
                ),
            },
        )
        case.gate(
            "coverage: 必测 shape 清单齐全（401 个 NVFP4 Linear / 11 个 case）",
            not missing and len(results) == len(SCENARIOS),
            {"scenarios": list(SCENARIOS), "ran": ran, "missing": missing},
        )
        case.gate(
            "coverage: 该 case 属于冻结算例清单",
            case.name in SCENARIOS,
            {"scenarios": list(SCENARIOS)},
        )

    payload = report.finish(
        extra={
            "coverage": {
                "scenarios": list(SCENARIOS),
                "ran": ran,
                "missing": missing,
                "expected_instances": sum(v[1] for v in SHAPES.values()),
                "measured_instances": total_instances,
                "expected_weight_gb": PROFILE_TOTAL_GB,
                "measured_weight_gb": total_bytes / 1e9,
            },
            "aggregate": {
                "total_gb": total_bytes / 1e9,
                "instances": total_instances,
                "cutlass_default": {
                    "ms": baseline_ms,
                    "gbs": total_bytes / (baseline_ms * 1e-3) / 1e9
                    if baseline_ms > 0
                    else None,
                },
                "best_integrable": {
                    "ms": integrable_ms,
                    "gbs": total_bytes / (integrable_ms * 1e-3) / 1e9,
                    "saves_ms": baseline_ms - integrable_ms,
                    "table": table,
                },
                "best_any_tactic": {
                    "ms": any_ms,
                    "gbs": total_bytes / (any_ms * 1e-3) / 1e9,
                    "saves_ms": baseline_ms - any_ms,
                },
                "profile_reference": {
                    "ms": PROFILE_STEP_MS,
                    "gbs": PROFILE_TOTAL_GB / (PROFILE_STEP_MS * 1e-3) / 1e9,
                },
                "per_case": [
                    {
                        "case": row["case"],
                        "n": row["n"],
                        "k": row["k"],
                        "instances": row["instances"],
                        "cutlass_ms": row["baseline_ms"],
                        "cutlass_gbs": row["baseline_gbs"],
                        "best_integrable": row["best_integrable"]["name"],
                        "best_integrable_family": row["best_integrable"]["family"],
                        "best_integrable_ms": row["best_integrable"]["agg_ms"],
                        "best_integrable_gbs": row["best_integrable"]["gbs"],
                        "best_any": row["best_any"]["name"],
                        "best_any_ms": row["best_any"]["agg_ms"],
                        "best_any_gbs": row["best_any"]["gbs"],
                        "available_configs": len(row["rows"]),
                        "unavailable": row["unavailable"],
                    }
                    for row in results
                ],
            },
            "backend_errors": backends.errors,
            "configs": {
                "warmup": args.warmup,
                "iters": args.iters,
                "max_copies": args.max_copies,
                "scan_tactics": list(args.scan_tactics),
                "cudnn_tactics": args.cudnn_tactics,
                "b12x_tactics": args.b12x_tactics,
                "l2_target_multiple": L2_TARGET_MULTIPLE,
                "tolerance": TOL,
                "agreement_nrmse": AGREE_NRMSE,
            },
        }
    )
    print(
        f"\nok={payload['summary']['ok']} cases={len(payload['cases'])} missing={missing}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
