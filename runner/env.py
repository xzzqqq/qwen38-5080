"""运行时环境：给 FlashInfer JIT 配好 CUDA/curand 头文件和 cache。

只设置进程环境变量，不 import torch/flashinfer；loader、runner、脚本在 import
FlashInfer 之前调用。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _prepend(var: str, paths: list[Path]) -> None:
    existing = [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    merged: list[str] = []
    for item in [*map(str, paths), *existing]:
        if item and item not in merged:
            merged.append(item)
    os.environ[var] = os.pathsep.join(merged)


def configure_cuda_allocator() -> None:
    """固定 PyTorch caching allocator 策略。

    M2 记录的 8K 容量数字（load 14.458 GB、prefill peak 15.620 GB、
    free 0.413 GB）只有开 ``expandable_segments`` 才能复现；默认 allocator
    在 7680+128 的 prefill 会因为 segment 碎片在 softmax 处 OOM。
    这是分配器策略，不改变任何数值；M3 的所有进程统一用它，并写进 provenance。
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def configure_flashinfer_env() -> None:
    cuda_home = Path(
        os.environ.get("CUDA_HOME") or Path.home() / "micromamba/envs/cuda13"
    )
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_include = Path(sys.prefix) / "lib" / py_tag / "site-packages/nvidia/cu13/include"
    nvidia_lib = Path(sys.prefix) / "lib" / py_tag / "site-packages/nvidia/cu13/lib"
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT / ".tmp/flashinfer"))
    _prepend("PATH", [cuda_home / "bin"])
    _prepend("CPATH", [nvidia_include, cuda_home / "targets/x86_64-linux/include"])
    _prepend(
        "LIBRARY_PATH",
        [
            nvidia_lib,
            cuda_home / "targets/x86_64-linux/lib",
            cuda_home / "lib",
        ],
    )
