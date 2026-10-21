"""M3 结果记录：provenance、run 目录、统计与汇总。

设计：
- 每个性能/验证 run 目录都带命令、commit、dirty diff、依赖版本、设备和
  checkpoint/workload hash，便于复现；
- 统计函数区分“run 间波动”和“step 延迟分布”，不混成一个指标；
- 不依赖 GPU，可在 CPU 上被脚本调用。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

CHECKPOINT = "models/Qwen3.8-27B-NVFP4-RTX5090"
# 只把源码/文档/输入算进 source manifest；results/models/.tmp 不进。
SOURCE_PREFIXES = ("runner/", "scripts/", "bench/", "docs/", "tests/")
SOURCE_FILES = ("Makefile", "requirements.txt", "README.md", "AGENTS.md", "CLAUDE.md", ".gitignore")
CHECKPOINT_REVISION = "5b7a687fc8211a5d631c8ca6a593dd37eb26ce33"
SOL_BANDWIDTH_BYTES = 960e9


def _run(cmd: list[str], cwd: Path | None = None, *, raw: bool = False) -> str:
    """跑命令取输出。raw=True 时只去掉尾部换行（porcelain 的行首空格有意义）。"""
    try:
        out = subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return ""
    return out.rstrip("\n") if raw else out.strip()


def git_state(root: Path) -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run(["git", "status", "--porcelain"], cwd=root, raw=True)
    diff = _run(["git", "diff"], cwd=root)
    staged = _run(["git", "diff", "--cached"], cwd=root)
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=root)
    untracked_files = [line for line in untracked.splitlines() if line.strip()]
    return {
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
        "diff_sha256": _sha256_text(diff + staged),
        "diff_bytes": len(diff) + len(staged),
        "untracked_files": untracked_files,
        "patch": diff + staged,
    }


def _is_source(rel: str) -> bool:
    return rel in SOURCE_FILES or rel.startswith(SOURCE_PREFIXES)


def source_manifest(root: Path) -> dict[str, Any]:
    """可复现的运行源码清单：改动 + untracked 的源码/文档/输入，逐个 hash。

    只有 dirty tree 才需要它；clean commit 由 commit id 唯一确定源码。
    """
    state = git_state(root)
    # 用 --name-only 取改动文件，避免解析 porcelain 的行首空格。
    modified = [
        line
        for line in (
            _run(["git", "diff", "--name-only"], cwd=root).splitlines()
            + _run(["git", "diff", "--cached", "--name-only"], cwd=root).splitlines()
        )
        if line.strip()
    ]
    candidates = sorted({*modified, *state["untracked_files"]})
    files = []
    for rel in candidates:
        if not _is_source(rel):
            continue
        path = root / rel
        if not path.is_file():
            continue
        data = path.read_bytes()
        files.append(
            {
                "path": rel,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    return {
        "dirty": state["dirty"],
        "commit": state["commit"],
        "patch_sha256": state["diff_sha256"],
        "patch_bytes": state["diff_bytes"],
        "files": files,
        "files_total_bytes": sum(f["bytes"] for f in files),
    }


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "transformers": None,
        "flashinfer": None,
        "cuda": None,
        "gpu_name": None,
        "gpu_total_mib": None,
        "driver": None,
        "clock_sm_mhz": None,
        "clock_max_mhz": None,
        "temperature_c": None,
        "power_w": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import flashinfer

        info["flashinfer"] = getattr(flashinfer, "__version__", "unknown")
    except Exception:
        pass
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,memory.total,clocks.sm,clocks.max.sm,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi:
        parts = [p.strip() for p in smi.split(",")]
        if len(parts) >= 6:
            info["driver"] = parts[0]
            info["gpu_total_mib"] = float(parts[1])
            info["clock_sm_mhz"] = float(parts[2])
            info["clock_max_mhz"] = float(parts[3])
            info["temperature_c"] = float(parts[4])
            info["power_w"] = float(parts[5])
    return info


def write_source_artifacts(out_dir: Path, patch: str, manifest: dict[str, Any]) -> None:
    """dirty tree 时把可恢复的 patch 和源码 manifest 落到 run dir。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "source_manifest.json", manifest)
    if manifest.get("dirty") and patch:
        (out_dir / "git_diff.patch").write_text(patch, encoding="utf-8")


def provenance(
    root: Path, extra: dict[str, Any] | None = None, out_dir: Path | None = None
) -> dict[str, Any]:
    state = git_state(root)
    patch = state.pop("patch", "")
    manifest = source_manifest(root)
    data = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git": state,
        "source_manifest": manifest,
        "environment": environment(),
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "sol_bandwidth_bytes": SOL_BANDWIDTH_BYTES,
    }
    if extra:
        data.update(extra)
    if out_dir is not None:
        out = Path(out_dir)
        save_json(out / "provenance.json", data)
        write_source_artifacts(out, patch, manifest)
    return data


def make_run_dir(root: Path, kind: str, tag: str = "") -> Path:
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{run_id}-{tag}" if tag else run_id
    path = root / "results/m3" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def step_stats(per_step_ms: Sequence[float]) -> dict[str, float]:
    """单次 run 内的 step 延迟分布（不混入 run 间波动）。"""
    arr = np.asarray(list(per_step_ms), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std(ddof=0)),
        "sum_s": float(arr.sum() / 1000.0),
        "tok_s": float(arr.size / (arr.sum() / 1000.0)),
    }


def run_stats(values: Sequence[float]) -> dict[str, float]:
    """run 间波动：每项一个聚合值（如每次 run 的 tok/s）。"""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
    }


def sol_report(ideal_bytes: float, measured_tok_s: float) -> dict[str, float]:
    """按每步理想字节算 SOL 与逻辑带宽。"""
    sol = SOL_BANDWIDTH_BYTES / ideal_bytes
    return {
        "ideal_bytes_per_step": ideal_bytes,
        "sol_tok_s": sol,
        "measured_tok_s": measured_tok_s,
        "pct_sol": measured_tok_s / sol,
        "logical_gb_s": ideal_bytes * measured_tok_s / 1e9,
    }


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def touch_run_log(path: Path, text: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "commands.txt").open("a", encoding="utf-8").write(text.rstrip() + "\n")


def elapsed(start: float) -> float:
    return time.perf_counter() - start


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")
