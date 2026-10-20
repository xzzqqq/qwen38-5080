"""pytest 环境：让 FlashInfer JIT 找到 CUDA/curand 头文件和缓存目录。

不依赖调用者手动 export；测试进程启动时先设置，再 import flashinfer。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_HOME = Path(os.environ.get("CUDA_HOME") or Path.home() / "micromamba/envs/cuda13")
PY_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
NVIDIA_INCLUDE = Path(sys.prefix) / "lib" / PY_TAG / "site-packages/nvidia/cu13/include"
NVIDIA_LIB = Path(sys.prefix) / "lib" / PY_TAG / "site-packages/nvidia/cu13/lib"
CUDA_INCLUDE = CUDA_HOME / "targets/x86_64-linux/include"
CUDA_LIB = CUDA_HOME / "targets/x86_64-linux/lib"

os.environ.setdefault("CUDA_HOME", str(CUDA_HOME))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT / ".tmp/flashinfer"))
os.environ["PATH"] = f"{CUDA_HOME / 'bin'}:{os.environ.get('PATH', '')}"
for var, paths in {
    "CPATH": [NVIDIA_INCLUDE, CUDA_INCLUDE],
    "LIBRARY_PATH": [NVIDIA_LIB, CUDA_LIB, CUDA_HOME / "lib"],
}.items():
    existing = [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    merged = []
    for p in [*map(str, paths), *existing]:
        if p and p not in merged:
            merged.append(p)
    os.environ[var] = os.pathsep.join(merged)
