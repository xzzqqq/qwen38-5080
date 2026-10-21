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
