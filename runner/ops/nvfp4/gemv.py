"""NVFP4 M=1 GEMV：CUDA core（SIMT）实现，直接消费 128x4 swizzled scale。

设计
----
- 只服务 decode（M=1）。activation 沿用现行语义：FlashInfer
  ``nvfp4_quantize(x, 1/input_scale, sfLayout=layout_128x4, do_shuffle=False)``
  返回的 ``a_fp4`` / ``a_sf``，本模块不重新量化；
- 权重是 checkpoint 的 ``weight_packed`` [N, K/2] U8 与**已经 swizzled 的**
  block scale [N_pad, K/16]（``NVFP4Linear.weight_scale_swizzled``）。kernel 按
  swizzle 偏移直接读，**不生成 unsquizzled 副本、也不做 repack**：8K + graph
  常驻后只剩约 0.12 GB，全模型再留一份 linear scale 要 1.6 GB，不可行；原地
  repack 虽然同尺寸可行，但会让 layout 与 ``NVFP4Linear`` 现状分叉，留给接入轮决定；
- 计算全在 CUDA core：E2M1 用位运算解成 fp32，FP32 累加，输出 BF16 [1, N]；
- 每层一个 kernel launch（split-K 的部分和在 kernel 内归约），不额外增加
  每步的 launch 次数。kernel 设计见 ``gemv_kernel.cu`` 头部注释。

快速用法::

    w = Nvfp4GemvWeights(weight_packed, weight_scale_swizzled, alpha)
    y = w.forward(a_fp4, a_sf)          # -> bf16 [1, N]

第一次调用会编译同目录的 ``gemv_kernel.cu``（nvcc，几十秒），之后走
``TORCH_EXTENSIONS_DIR`` 缓存；缓存目录默认落在主仓库 ``.tmp/torch_extensions``。
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]

_ROWS_PER_TILE = 128
_BLOCK_THREADS = 512


def _extensions_dir() -> Path:
    """主仓库的 .tmp（job worktree 里的 .tmp 每次都会被删，缓存没有意义）。"""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(_HERE.parent), "rev-parse", "--git-common-dir"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - 不在 git 里就用 worktree 根目录
        out = ""
    if out:
        common = Path(out)
        if not common.is_absolute():
            common = (_HERE.parent / common).resolve()
        return common.parent / ".tmp" / "torch_extensions"
    return _ROOT / ".tmp" / "torch_extensions"


def _setup_build_env() -> None:
    """让 ``torch.utils.cpp_extension`` 找到 nvcc / CUDA 头文件（与 m2_env 同口径）。"""
    cuda_home = Path(
        os.environ.get("CUDA_HOME") or Path.home() / "micromamba/envs/cuda13"
    )
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    bin_dir = str(cuda_home / "bin")
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{path}" if path else bin_dir
    site = Path(torch.__file__).resolve().parents[1]
    for var, items in (
        (
            "CPATH",
            [
                str(site / "nvidia/cu13/include"),
                str(cuda_home / "targets/x86_64-linux/include"),
            ],
        ),
        (
            "LIBRARY_PATH",
            [
                str(site / "nvidia/cu13/lib"),
                str(cuda_home / "targets/x86_64-linux/lib"),
                str(cuda_home / "lib"),
            ],
        ),
    ):
        merged: list[str] = []
        for item in [*items, *os.environ.get(var, "").split(os.pathsep)]:
            if item and item not in merged:
                merged.append(item)
        os.environ[var] = os.pathsep.join(merged)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_extensions_dir()))


_EXT = None


def extension():
    """编译（或从缓存加载）CUDA 扩展；第一次调用会起 nvcc。"""
    global _EXT
    if _EXT is None:
        _setup_build_env()
        from torch.utils.cpp_extension import load

        _EXT = load(
            name="nvfp4_gemv_phase2",
            sources=[str(_HERE.with_name("gemv_kernel.cu"))],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=bool(int(os.environ.get("NVFP4_GEMV_VERBOSE", "0"))),
        )
    return _EXT


def auto_k_split(
    n: int, *, sm_count: int = 84, target_waves: int = 2, cap: int = 4
) -> int:
    """按行 tile 数选 split-K：让 block 数 >= 2×SM，同时把 split 控制在小范围。"""
    tiles = n // _ROWS_PER_TILE
    split = 1
    while split < cap and tiles * split < sm_count * target_waves:
        split *= 2
    return split


def _as_u8(t: torch.Tensor, what: str) -> torch.Tensor:
    if t.dtype != torch.uint8:
        if t.element_size() != 1:
            raise TypeError(f"{what} 必须是 1 字节 dtype，得到 {t.dtype}")
        t = t.view(torch.uint8)
    if not t.is_contiguous():
        raise ValueError(f"{what} 必须 contiguous")
    return t


@dataclass(frozen=True)
class GemvConfig:
    """粗粒度可调项；``k_split=0`` 表示按 ``auto_k_split`` 自动选。"""

    k_split: int = 0


class Nvfp4GemvWeights:
    """绑定一份 NVFP4 权重（packed U8 + 128x4 swizzled scale + alpha）。

    Args:
        weight_packed: U8 [N, K/2]，checkpoint 的 ``weight``。
        weight_scale_swizzled: 1 字节 dtype [ceil(N/128)*128, K/16]，必须是
            ``reference.swizzle_scale_128x4`` / FlashInfer
            ``block_scale_interleave`` 的输出（``NVFP4Linear`` 只保留这一份）。
        alpha: F32 [1]，``input_scale * weight_scale_2``。
        k_split: 0 = 自动；否则固定 split 数（调试 / 参数粗扫用）。
    """

    def __init__(
        self,
        weight_packed: torch.Tensor,
        weight_scale_swizzled: torch.Tensor,
        alpha: torch.Tensor,
        *,
        k_split: int = 0,
    ) -> None:
        if weight_packed.dim() != 2 or weight_packed.dtype != torch.uint8:
            raise TypeError(
                "weight_packed 必须是 U8 [N, K/2]，得到 "
                f"{weight_packed.dtype} {tuple(weight_packed.shape)}"
            )
        if not weight_packed.is_contiguous():
            raise ValueError("weight_packed 必须 contiguous")
        n, k_half = (int(x) for x in weight_packed.shape)
        k = 2 * k_half
        if n % _ROWS_PER_TILE != 0:
            raise ValueError(f"N 必须是 {_ROWS_PER_TILE} 的倍数，得到 {n}")
        if k % 64 != 0:
            raise ValueError(f"K 必须是 64 的倍数，得到 {k}")

        wsf = _as_u8(weight_scale_swizzled, "weight_scale_swizzled")
        if wsf.dim() != 2:
            raise ValueError(
                f"weight_scale_swizzled 必须是二维 swizzled 缓冲，得到 {tuple(wsf.shape)}"
            )
        rows_p, cols_p = (int(x) for x in wsf.shape)
        if rows_p != ((n + _ROWS_PER_TILE - 1) // _ROWS_PER_TILE) * _ROWS_PER_TILE:
            raise ValueError(
                f"swizzled scale 行数 {rows_p} 与 N={n} 不匹配（应补齐到 128 的倍数）"
            )
        if cols_p != k // 16:
            raise ValueError(f"swizzled scale 列数 {cols_p} != K/16={k // 16}")

        if alpha.numel() != 1 or alpha.dtype != torch.float32:
            raise TypeError(f"alpha 必须是 F32 标量，得到 {alpha.dtype} {alpha.numel()}")
        if weight_packed.device != wsf.device or weight_packed.device != alpha.device:
            raise ValueError("weight / scale / alpha 必须在同一 device")

        self.n = n
        self.k = k
        self.weight_packed = weight_packed
        self.weight_scale_u8 = wsf
        # 原 dtype（通常是 float8_e4m3fn）留一份引用，便于和 mm_fp4 用同一份 scale
        # 做 apples-to-apples 对比；kernel 只用上面的 uint8 view。
        self.weight_scale_orig = weight_scale_swizzled
        self.alpha = alpha.reshape(1).contiguous()
        self.k_split = int(k_split)
        self.device = weight_packed.device
        props = torch.cuda.get_device_properties(self.device)
        self.sm_count = int(props.multi_processor_count)
        self._workspace: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # k_split == 1 时 kernel 不用 workspace；给一个常驻的占位 tensor，
        # 避免每次 forward 都分配（microbench 的计时区间里不应有分配）。
        self._dummy = (
            torch.empty(1, dtype=torch.float32, device=self.device),
            torch.empty(1, dtype=torch.int32, device=self.device),
        )

    def resolve_k_split(self, k_split: int | None = None) -> int:
        if k_split:
            split = int(k_split)
        elif self.k_split:
            split = self.k_split
        else:
            split = auto_k_split(self.n, sm_count=self.sm_count)
        if split < 1:
            raise ValueError(f"k_split 必须 >= 1，得到 {split}")
        return split

    def _partials(self, k_split: int) -> tuple[torch.Tensor, torch.Tensor]:
        """split-K 部分和 + 归约 counter（kernel 用完会自己清零 counter）。"""
        if k_split == 1:
            return self._dummy
        cached = self._workspace.get(k_split)
        if cached is None:
            cached = (
                torch.empty(
                    (k_split, self.n), dtype=torch.float32, device=self.device
                ),
                torch.zeros(
                    self.n // _ROWS_PER_TILE, dtype=torch.int32, device=self.device
                ),
            )
            self._workspace[k_split] = cached
        return cached

    def forward(
        self,
        a_fp4: torch.Tensor,
        a_sf: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        k_split: int | None = None,
    ) -> torch.Tensor:
        """m = 1 的 GEMV。

        Args:
            a_fp4: ``nvfp4_quantize`` 的 packed activation，U8 [1, K/2] 或 [K/2]。
            a_sf: ``nvfp4_quantize`` 的 128x4 swizzled activation scale，
                [128, K/16]（M=1 时 padding 行不用）。
            out: 可选预分配输出 BF16 [1, N]（microbench 用来把分配开销排除在
                计时区间外）。
            k_split: 覆盖默认 split（None = 用构造时的策略）。
        """
        n, k = self.n, self.k
        a_u8 = _as_u8(a_fp4, "a_fp4")
        if a_u8.numel() != k // 2:
            raise ValueError(
                f"a_fp4 元素数 {a_u8.numel()} != K/2={k // 2}（只支持 M=1）"
            )
        sf_u8 = _as_u8(a_sf, "a_sf")
        if sf_u8.dim() != 2 or int(sf_u8.shape[1]) != k // 16:
            raise ValueError(
                f"a_sf 必须是 [rows, K/16={k // 16}] 的 swizzled 缓冲，"
                f"得到 {tuple(sf_u8.shape)}"
            )
        if int(sf_u8.shape[0]) % _ROWS_PER_TILE != 0:
            raise ValueError(
                f"a_sf 行数 {int(sf_u8.shape[0])} 必须是 {_ROWS_PER_TILE} 的倍数"
            )
        if self.device != a_u8.device:
            raise ValueError("activation 与权重不在同一 device")
        if out is None:
            out = torch.empty((1, n), dtype=torch.bfloat16, device=self.device)
        else:
            if out.shape != (1, n) or out.dtype != torch.bfloat16:
                raise ValueError(f"out 必须是 BF16 [1, {n}]，得到 {out.dtype} {tuple(out.shape)}")

        split = self.resolve_k_split(k_split)
        partials, counters = self._partials(split)
        extension().nvfp4_gemv(
            a_u8.reshape(-1),
            sf_u8,
            self.weight_packed,
            self.weight_scale_u8,
            self.alpha,
            out,
            partials,
            counters,
            split,
        )
        return out

    __call__ = forward

    def extra_repr(self) -> str:
        return (
            f"n={self.n}, k={self.k}, k_split={self.k_split or 'auto'}, "
            f"sm_count={self.sm_count}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.extra_repr()})"


__all__ = [
    "GemvConfig",
    "Nvfp4GemvWeights",
    "auto_k_split",
    "extension",
]
