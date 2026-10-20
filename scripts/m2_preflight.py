#!/usr/bin/env python3
"""M2 预检：记录 Python/CUDA/GPU/Transformers/FlashInfer 能力，不加载 checkpoint。

设计：
- 只做只读探测和一次小尺寸 GPU 分配，不读权重、不建模型；
- 每项检查独立捕获异常，失败也写进报告，最后用 checks/ok 汇总；
- 结果同时写 JSON、文本摘要和 pip freeze，便于固定环境和复现。

用法：
    .venv/bin/python scripts/m2_preflight.py
    .venv/bin/python scripts/m2_preflight.py --output-dir results/m2/run-001
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _prepend_paths(var: str, paths: list[Path]) -> None:
    existing = [p for p in os.environ.get(var, "").split(os.pathsep) if p]
    merged: list[str] = []
    for item in [*map(str, paths), *existing]:
        if item and item not in merged:
            merged.append(item)
    os.environ[var] = os.pathsep.join(merged)


def configure_runtime_env() -> dict[str, str]:
    """选定 nvcc、CUDA 头文件和 FlashInfer cache。

    FlashInfer 的编译架构由 CUDA_HOME（或 which nvcc）决定。系统 nvcc 是
    12.4 时，SM 120 会直接被判为不支持；这里优先找项目约定的 cuda13 env。
    cuda13 的运行时头文件在 targets/x86_64-linux/include，curand 头文件
    在 pip 的 nvidia/cu13 包里，两者都要进 CPATH。
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        candidates = [
            Path.home() / "micromamba/envs/cuda13",
            Path("/usr/local/cuda"),
        ]
        for candidate in candidates:
            if (candidate / "bin/nvcc").exists():
                cuda_home = str(candidate)
                break
    venv = Path(sys.prefix)
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_include = venv / "lib" / py_tag / "site-packages/nvidia/cu13/include"
    nvidia_lib = venv / "lib" / py_tag / "site-packages/nvidia/cu13/lib"
    cuda_include = Path(cuda_home) / "targets/x86_64-linux/include" if cuda_home else None
    cuda_lib = Path(cuda_home) / "targets/x86_64-linux/lib" if cuda_home else None

    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        nvcc_bin = Path(cuda_home) / "bin"
        if nvcc_bin.exists():
            _prepend_paths("PATH", [nvcc_bin])
    _prepend_paths("CPATH", [p for p in (nvidia_include, cuda_include) if p])
    _prepend_paths(
        "LIBRARY_PATH",
        [
            p
            for p in (
                nvidia_lib,
                cuda_lib,
                Path(cuda_home) / "lib" if cuda_home else None,
            )
            if p
        ],
    )
    os.environ.setdefault(
        "FLASHINFER_WORKSPACE_BASE",
        str(Path.cwd() / ".tmp" / "flashinfer"),
    )
    return {
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CPATH": os.environ.get("CPATH", ""),
        "LIBRARY_PATH": os.environ.get("LIBRARY_PATH", ""),
        "FLASHINFER_WORKSPACE_BASE": os.environ.get(
            "FLASHINFER_WORKSPACE_BASE", ""
        ),
        "PATH": os.environ.get("PATH", ""),
    }


def collect_cuda_headers() -> dict[str, str]:
    """检查 JIT 需要的头文件是否都能在 CPATH 里找到。"""
    paths = [Path(p) for p in os.environ.get("CPATH", "").split(os.pathsep) if p]
    names = ["cublasLt.h", "curand_kernel.h", "cuda_fp16.h", "cuda_runtime_api.h"]
    found: dict[str, str] = {}
    for name in names:
        found[name] = next(
            (str(path / name) for path in paths if (path / name).exists()), ""
        )
    return found


def collect_nvcc() -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    info: dict[str, Any] = {"path": nvcc, "version": None, "ge_12_8": False}
    if not nvcc:
        return info
    proc = run_cmd([nvcc, "--version"])
    info["raw"] = proc
    text = (proc.get("stdout") or "") + (proc.get("stderr") or "")
    match = re.search(r"release (\d+)\.(\d+)", text)
    if match:
        major, minor = int(match.group(1)), int(match.group(2))
        info["version"] = f"{major}.{minor}"
        info["ge_12_8"] = (major, minor) >= (12, 8)
    return info


def run_cmd(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as e:  # noqa: BLE001 - 预检要记录任何失败
        return {"cmd": cmd, "error": f"{type(e).__name__}: {e}"}


def pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:
        return None


def import_module_safe(name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(name), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def collect_torch() -> dict[str, Any]:
    torch, err = import_module_safe("torch")
    if torch is None:
        return {"import_error": err}

    info: dict[str, Any] = {
        "version": getattr(torch, "__version__", None),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cudnn": None,
        "cuda_available": False,
        "alloc_test": None,
        "gpu": None,
    }
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception as e:  # noqa: BLE001
        info["cudnn_error"] = f"{type(e).__name__}: {e}"

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:  # noqa: BLE001
        info["cuda_available_error"] = f"{type(e).__name__}: {e}"

    if info["cuda_available"]:
        try:
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info()
            info["gpu"] = {
                "name": props.name,
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_mib": round(props.total_memory / 1024**2, 1),
                "free_memory_mib": round(free_b / 1024**2, 1),
                "total_memory_mib_mem_get_info": round(total_b / 1024**2, 1),
                "multi_processor_count": props.multi_processor_count,
            }
            # 最小分配/矩阵乘，验证 eager CUDA 路径真的能跑。
            x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
            y = x @ x
            torch.cuda.synchronize()
            info["alloc_test"] = {
                "ok": True,
                "dtype": str(y.dtype),
                "shape": list(y.shape),
                "sample": float(y.float().abs().mean().item()),
            }
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            info["gpu_error"] = f"{type(e).__name__}: {e}"
            info["alloc_test"] = {"ok": False, "error": info["gpu_error"]}

    return info


def collect_transformers() -> dict[str, Any]:
    transformers, err = import_module_safe("transformers")
    if transformers is None:
        return {"import_error": err}

    info: dict[str, Any] = {
        "version": getattr(transformers, "__version__", None),
        "qwen3_5_module": None,
        "qwen3_5_classes": [],
        "qwen3_5_config_keys": [],
        "qwen3_5_present": False,
    }

    try:
        info["qwen3_5_module"] = str(
            importlib.util.find_spec("transformers.models.qwen3_5")
        )
    except Exception as e:  # noqa: BLE001
        info["qwen3_5_module_error"] = f"{type(e).__name__}: {e}"

    try:
        info["qwen3_5_classes"] = sorted(
            name for name in dir(transformers) if "Qwen3_5" in name
        )
    except Exception as e:  # noqa: BLE001
        info["qwen3_5_classes_error"] = f"{type(e).__name__}: {e}"

    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        info["qwen3_5_config_keys"] = sorted(
            str(k) for k in CONFIG_MAPPING.keys() if "qwen3_5" in str(k).lower()
        )
    except Exception as e:  # noqa: BLE001
        info["qwen3_5_config_error"] = f"{type(e).__name__}: {e}"

    info["qwen3_5_present"] = bool(
        info["qwen3_5_module"] or info["qwen3_5_classes"] or info["qwen3_5_config_keys"]
    )
    return info


def collect_flashinfer() -> dict[str, Any]:
    info: dict[str, Any] = {
        "packages": {
            name: pkg_version(name)
            for name in (
                "flashinfer-python",
                "flashinfer-cubin",
                "flashinfer-jit-cache",
            )
        },
        "mm_fp4": None,
        "target_cuda_archs": [],
    }
    flashinfer, err = import_module_safe("flashinfer")
    if flashinfer is None:
        info["import_error"] = err
        return info

    info["version"] = getattr(flashinfer, "__version__", None)
    try:
        from flashinfer.compilation_context import CompilationContext

        info["target_cuda_archs"] = sorted(
            str(arch) for arch in CompilationContext().TARGET_CUDA_ARCHS
        )
    except Exception as e:  # noqa: BLE001
        info["target_cuda_archs_error"] = f"{type(e).__name__}: {e}"

    try:
        from flashinfer.gemm import mm_fp4

        doc = inspect.getdoc(mm_fp4) or ""
        info["mm_fp4"] = {
            "exists": True,
            "signature": str(inspect.signature(mm_fp4)),
            "doc_head": doc[:1200],
        }
    except Exception as e:  # noqa: BLE001
        info["mm_fp4"] = {"exists": False, "error": f"{type(e).__name__}: {e}"}
    return info


def git_info() -> dict[str, Any]:
    commit = run_cmd(["git", "rev-parse", "HEAD"])
    status = run_cmd(["git", "status", "--short"])
    return {
        "commit": commit.get("stdout") or None,
        "dirty": bool(status.get("stdout")),
        "status": status.get("stdout", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认 results/m2/<本地时间戳>",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = args.output_dir or Path("results/m2") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    env_info = configure_runtime_env()
    report: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": now_iso(),
        "cwd": str(Path.cwd()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "env": env_info,
        "cuda_headers": collect_cuda_headers(),
        "nvcc": collect_nvcc(),
        "git": git_info(),
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,compute_cap,display_active",
                "--format=csv,noheader",
            ]
        ),
        "torch": collect_torch(),
        "transformers": collect_transformers(),
        "flashinfer": collect_flashinfer(),
    }

    torch_info = report["torch"]
    gpu = torch_info.get("gpu") or {}
    capability = gpu.get("capability") or []
    flashinfer_archs = report["flashinfer"].get("target_cuda_archs") or []
    checks = {
        "torch_import": "import_error" not in torch_info,
        "cuda_available": bool(torch_info.get("cuda_available")),
        "capability_12_0": capability == [12, 0],
        "eager_alloc": bool((torch_info.get("alloc_test") or {}).get("ok")),
        "nvcc_ge_12_8": bool(report["nvcc"].get("ge_12_8")),
        "cuda_headers_present": all(report["cuda_headers"].values()),
        "qwen3_5_present": bool(report["transformers"].get("qwen3_5_present")),
        "flashinfer_import": "import_error" not in report["flashinfer"],
        "mm_fp4_present": bool((report["flashinfer"].get("mm_fp4") or {}).get("exists")),
        "flashinfer_arch_12": any(str(arch).startswith("(12") for arch in flashinfer_archs),
    }
    report["checks"] = checks
    report["ok"] = all(checks.values())

    # 保存 pip freeze，后面固定 requirements 时直接引用。
    freeze = run_cmd([sys.executable, "-m", "pip", "freeze"], timeout=120)
    (out_dir / "pip_freeze.txt").write_text(
        freeze.get("stdout", "") + "\n", encoding="utf-8"
    )

    (out_dir / "preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"run_id: {run_id}",
        f"python: {report['python']['executable']}",
        f"torch: {torch_info.get('version')} cuda={torch_info.get('cuda_runtime')}",
        f"gpu: {gpu.get('name')} capability={capability} "
        f"free={gpu.get('free_memory_mib')} MiB total={gpu.get('total_memory_mib')} MiB",
        f"nvcc: {report['nvcc'].get('version')} ({report['nvcc'].get('path')})",
        f"cuda headers: {report['cuda_headers']}",
        f"flashinfer archs: {flashinfer_archs}",
        f"transformers: {report['transformers'].get('version')} "
        f"qwen3_5={report['transformers'].get('qwen3_5_present')}",
        f"flashinfer: {report['flashinfer'].get('version')} "
        f"packages={report['flashinfer'].get('packages')}",
        "checks:",
    ]
    for name, ok in checks.items():
        lines.append(f"  {'PASS' if ok else 'FAIL'} {name}")
    if report["flashinfer"].get("mm_fp4"):
        lines.append(f"mm_fp4 signature: {report['flashinfer']['mm_fp4'].get('signature')}")
    lines.append(f"result: {'PASS' if report['ok'] else 'FAIL'}")
    lines.append(f"output: {out_dir}")
    summary = "\n".join(lines)
    (out_dir / "preflight.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
