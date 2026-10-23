"""实验工具的 CPU 测试夹具：临时 git 仓库 + 可提交的 fake 实验脚本。

为什么要临时仓库：job 必须从**提交后的 commit** 的 detached worktree 执行，所以
fake 脚本也要被 commit。测试仓库里放同名路径（scripts/m3_check.py 等）的 fake
实现，产物的 schema 与真实脚本一致，走的是同一套校验器。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts/experiment.py"
PYTHON = str(ROOT / ".venv/bin/python")


FAKE_CHECK = '''#!/usr/bin/env python3
"""Fake m3_check：产物 schema 与真实脚本一致，行为由 FAKE_* 环境变量控制。"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import probe
except Exception:  # noqa: BLE001
    probe = None

SCENARIOS = ("graph-1024", "numeric-512")


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="all")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--fixtures", default="bench/workloads/m3-p0")
    p.add_argument("--smoke-max-new-tokens", type=int, default=2048)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", default=None)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    state = Path(os.environ.get("EXPERIMENT_STATE_DIR", "."))
    state.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("EXPERIMENT_JOB_ID", "?")
    with open(state / "fake_runs.log", "a", encoding="utf-8") as handle:
        handle.write(f"start {job_id} {time.time():.3f} steps={a.steps}\\n")

    # 用 flock 检测“两个 job 同时用 GPU”；进程被杀时内核自动释放。
    lock_fd = os.open(state / "fake_gpu.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("overlap detected", file=sys.stderr)
        return 1

    mode = os.environ.get("FAKE_CHECK_MODE", "ok")
    fixtures_dir = Path(a.fixtures)
    if not fixtures_dir.is_absolute():
        fixtures_dir = Path.cwd() / fixtures_dir
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())
    fixtures_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    only_session = os.environ.get("FAKE_SLEEP_SESSION")
    sleepy = (not only_session) or os.environ.get("EXPERIMENT_SESSION_ID") == only_session
    if sleepy and os.environ.get("FAKE_GRANDCHILD") == "1":
        child = subprocess.Popen(["sleep", "120"])
        (out / "grandchild.pid").write_text(str(child.pid))
    try:
        if os.environ.get("FAKE_STRAY") == "1":
            # 故意留一个继承 job 进程组的后台进程，测试执行器会清理它
            subprocess.Popen(["sleep", "120"])
        time.sleep(float(os.environ.get("FAKE_SLEEP", "0")) if sleepy else 0.0)
        if mode == "no_artifact":
            return 0
        names = list(SCENARIOS) if a.only.strip() in ("", "all") else [s.strip() for s in a.only.split(",")]
        results = []
        summaries = []
        for name in names:
            sub = out / name
            sub.mkdir(parents=True, exist_ok=True)
            gates = [
                {"level": "gate", "name": "coverage: 必测项齐全", "ok": True},
                {"level": "gate", "name": "fake gate", "ok": mode != "fail_gate"},
            ]
            diagnostics = [{"level": "diagnostic", "name": "fake diagnostic", "ok": mode != "diag_fail"}]
            report = {
                "scenario": name,
                "ok": all(g["ok"] for g in gates),
                "gate_count": len(gates),
                "gate_failures": [g for g in gates if not g["ok"]],
                "items": gates + diagnostics,
                "provenance": {
                    "git": {"commit": commit(), "dirty": False},
                    "argv": ["scripts/m3_check.py", "--worker", name, "--out", str(sub),
                             "--steps", str(a.steps)],
                },
                "fixtures": {"dir": a.fixtures, "manifest_sha256": fixtures_sha},
                "probe": getattr(probe, "MARKER", None) if probe else None,
            }
            (sub / "report.json").write_text(json.dumps(report))
            report["returncode"] = 0 if report["ok"] else 1
            results.append(report)
            summaries.append({
                "scenario": name,
                "ok": report["ok"],
                "gate_count": len(gates),
                "gate_failures": report["gate_failures"],
                "returncode": report["returncode"],
            })
        top = {
            "summary": {"run_dir": str(out), "ok": all(r["ok"] for r in results), "scenarios": summaries},
            "results": results,
            "probe": getattr(probe, "MARKER", None) if probe else None,
            "job_id": job_id,
        }
        (out / "provenance.json").write_text(json.dumps({
            "git": {"commit": commit(), "dirty": False},
            "environment": {"python": "fake", "torch": "fake"},
            "source_manifest": {"files": []},
        }))
        (out / "check.json").write_text(json.dumps(top))
        return 0 if all(r["ok"] for r in results) else 1
    finally:
        with open(state / "fake_runs.log", "a", encoding="utf-8") as handle:
            handle.write(f"end {job_id} {time.time():.3f}\\n")


if __name__ == "__main__":
    raise SystemExit(main())
'''


FAKE_BENCH = '''#!/usr/bin/env python3
"""Fake decode bench：按请求的 steps/repeats 写出完整样本；可用 FAKE_* 造旧配置。"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCENARIOS = {
    "graph-512-1024": {"execution": "graph", "depth": 512, "capacity": 1024},
    "cap-512-1024": {"execution": "static-cap", "depth": 512, "capacity": 1024},
}


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def stats(value, n):
    return {"n": n, "median": value, "mean": value, "min": value, "max": value, "std": 0.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="all")
    p.add_argument("--steps", type=int, default=511)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--repeats-non-graph", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--graph-warmup", type=int, default=3)
    p.add_argument("--fixtures", default="bench/workloads/m3-p0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--worker", default=None)
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    steps = int(os.environ.get("FAKE_BENCH_STEPS", a.steps))
    forced_repeats = os.environ.get("FAKE_BENCH_REPEATS")
    names = list(SCENARIOS) if a.only.strip() in ("", "all") else [s.strip() for s in a.only.split(",") if s.strip()]
    fixtures_dir = Path(a.fixtures)
    if not fixtures_dir.is_absolute():
        fixtures_dir = Path.cwd() / fixtures_dir
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())
    fixtures_sha = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    results = []
    for name in names:
        spec = SCENARIOS[name]
        repeats = int(forced_repeats) if forced_repeats is not None else (
            a.repeats_non_graph if spec["execution"] != "graph" and a.repeats_non_graph else a.repeats
        )
        samples = []
        for index in range(repeats):
            samples.append({
                "repeat": index,
                "model": {
                    "per_step_ms": [10.0] * steps,
                    "model_tok_s": 31.0,
                    "peak_allocated_gb": 1.0,
                    "peak_reserved_gb": 1.1,
                    "free_gb": 0.5,
                },
                "system": {"system_tok_s": 20.0},
                "request": {"request_tok_s": 18.0},
                "tokens": {
                    "generated": steps + 1,
                    "timed_steps": steps,
                    "matches_frozen_trace": True,
                },
                "prefill": {"prefill_s": 0.5},
                "gpu_before": {},
            })
        summary = {
            "model_tok_s": stats(31.0, repeats),
            "system_tok_s": stats(20.0, repeats),
            "request_tok_s": stats(18.0, repeats),
            "step_ms": {"n": steps * repeats, "p50_ms": 10.0, "p95_ms": 10.5},
            "prefill_s": {"median": 0.5},
            "ttft_s": {"median": 0.6},
            "peak_allocated_gb": 1.0,
            "peak_reserved_gb": 1.1,
            "free_gb_min": 0.5,
            "trace_match_all": True,
            "row_copy_ms": {"n": repeats, "median": 0.0},
        }
        results.append({
            "scenario": name,
            "spec": spec,
            "steps": steps,
            "repeats_requested": repeats,
            "run_config": {
                "schema": 2,
                "scenario": name,
                "execution": spec["execution"],
                "depth": spec["depth"],
                "capacity": spec["capacity"],
                "steps": steps,
                "repeats": repeats,
                "chunk_size": a.chunk_size,
                "graph_warmup": a.graph_warmup,
                "fixtures_sha256": fixtures_sha,
            },
            "environment_fingerprint": {"torch": "fake"},
            "samples": samples,
            "summary": summary,
            "provenance": {"git": {"commit": commit(), "dirty": False}},
            "ok": True,
        })
    matrix = {
        "schema": 2,
        "scenarios": names,
        "steps": a.steps,
        "repeats": a.repeats,
        "repeats_non_graph": a.repeats_non_graph,
        "chunk_size": a.chunk_size,
        "graph_warmup": a.graph_warmup,
        "fixtures_sha256": fixtures_sha,
    }
    (out / "provenance.json").write_text(json.dumps({
        "git": {"commit": commit(), "dirty": False},
        "run_config": matrix,
        "fixtures": a.fixtures,
        "environment_fingerprint": {"torch": "fake"},
        "environment": {"python": "fake"},
    }))
    complete = os.environ.get("FAKE_BENCH_COMPLETE", "1") == "1"
    (out / "bench.json").write_text(json.dumps({
        "complete": complete,
        "scenarios": names,
        "repeats": a.repeats,
        "steps": a.steps,
        "results": results,
        "table": "",
    }))
    (out / "summary.md").write_text("# fake bench\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


FAKE_MEMORY = '''#!/usr/bin/env python3
"""Fake m3_memory：stages/growth/determinism/peaks 字段与真实脚本一致。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCENARIOS = {"graph-512-1024": {"execution": "graph", "depth": 512, "capacity": 1024}}


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="all")
    p.add_argument("--steps", type=int, default=511)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--worker", default=None)
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    names = list(SCENARIOS) if a.only.strip() in ("", "all") else [s.strip() for s in a.only.split(",")]
    results = []
    for name in names:
        stages = [{"stage": stage, "peak_allocated_gb": 1.0, "peak_reserved_gb": 1.1,
                   "end_allocated_gb": 1.0, "end_reserved_gb": 1.1, "free_gb": 0.5}
                  for stage in ("load", "cache_init", "prefill_cold", "warmup_capture", "decode", "reset")]
        results.append({
            "scenario": name,
            "spec": SCENARIOS[name],
            "steps": a.steps,
            "stages": stages,
            "growth": {
                "allocated_delta_gb": 0.0,
                "reserved_delta_gb": 0.0,
                "cycle2_allocated_delta_gb": 0.0,
                "cycle2_reserved_delta_gb": 0.0,
                "kv_data_ptr_stable": True,
            },
            "peaks": {
                "load_peak_allocated_gb": 1.0,
                "max_peak_allocated_gb": 1.0,
                "max_peak_reserved_gb": 1.1,
                "min_free_gb": 0.5,
            },
            "determinism": {"prefill_logits_equal": True, "step_logits_equal": True},
            "graph": {"max_phase_peak_allocated_gb": 1.0},
            "ok": True,
        })
    (out / "provenance.json").write_text(json.dumps({
        "git": {"commit": commit(), "dirty": False},
        "run_config": {"scenarios": names, "steps": a.steps},
        "environment": {"python": "fake"},
    }))
    (out / "memory.json").write_text(json.dumps({"complete": True, "results": results, "table": ""}))
    (out / "summary.md").write_text("# fake memory\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


FAKE_PROFILE = '''#!/usr/bin/env python3
"""Fake profile 产物：timeline_summary + case 目录 + provenance。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CASES = (("graph-512", "graph", 512, 1024), ("eager-512", "eager", 512, 0))


def commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8)
    a = p.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    timeline = {}
    for name, execution, depth, capacity in CASES:
        case = out / name
        case.mkdir(parents=True, exist_ok=True)
        (case / "trace.nsys-rep").write_text("fake trace")
        (case / "gpu_trace.csv").write_text("Start (ns),Duration (ns)\\n1,2\\n")
        (case / "meta.json").write_text(json.dumps({
            "execution": execution,
            "depth": depth,
            "capacity": capacity,
            "steps": a.steps,
            "provenance": {"git": {"commit": commit(), "dirty": False}},
        }))
        timeline[name] = {
            "steps": a.steps,
            "kernels": 100,
            "per_step_kernels": 12.5,
            "per_step_kernel_us": 300.0,
            "per_step_wall_us": 1000.0,
            "tiny_kernel_count_pct": 50.0,
            "top": [{"name": "fake_kernel"}],
        }
    (out / "summary.md").write_text("# fake profile\\n")
    (out / "provenance.json").write_text(json.dumps({
        "git": {"commit": commit(), "dirty": False},
        "environment": {"python": "fake"},
    }))
    (out / "timeline_summary.json").write_text(json.dumps(timeline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


PROFILE_SH = '''#!/usr/bin/env bash
set -euo pipefail
OUT=""
STEPS="8"
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
if [ -z "$OUT" ]; then echo "--out is required" >&2; exit 2; fi
exec .venv/bin/python tools/fake_profile.py --out "$OUT" --steps "$STEPS"
'''


PROBE = 'MARKER = "commit-a"\n'


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def git_commit(repo: Path, message: str = "test") -> str:
    _git(repo, "add", "-A")
    proc = _git(repo, "commit", "-m", message)
    assert proc.returncode == 0, proc.stderr
    return head(repo)


def head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "bench/e2e").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)
    (root / ".gitignore").write_text(
        ".venv/\nmodels/\nresults/\n.tmp/\n__pycache__/\n", encoding="utf-8"
    )
    (root / "probe.py").write_text(PROBE, encoding="utf-8")
    (root / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    (root / "scripts/m3_check.py").write_text(FAKE_CHECK, encoding="utf-8")
    (root / "bench/e2e/decode.py").write_text(FAKE_BENCH, encoding="utf-8")
    (root / "scripts/m3_memory.py").write_text(FAKE_MEMORY, encoding="utf-8")
    (root / "scripts/m3_profile.sh").write_text(PROFILE_SH, encoding="utf-8")
    (root / "tools/fake_profile.py").write_text(FAKE_PROFILE, encoding="utf-8")
    (root / "bench/workloads/m3-p0").mkdir(parents=True)
    (root / "bench/workloads/m3-p0/manifest.json").write_text("{}", encoding="utf-8")
    assert _git(root, "init", "-b", "main").returncode == 0
    git_commit(root, "init")
    return root


# ------------------------------------------------------------------ 请求构造
def check_step(name: str = "check", only: str = "graph-1024", steps: int = 16) -> dict:
    return {
        "kind": "check",
        "argv": [
            ".venv/bin/python",
            "scripts/m3_check.py",
            "--only",
            only,
            "--steps",
            str(steps),
            "--out",
            f"{{output}}/{name}",
        ],
        "result": f"{{output}}/{name}/check.json",
    }


def bench_step(
    name: str = "bench",
    only: str = "graph-512-1024",
    steps: int = 4,
    repeats: int = 1,
    repeats_non_graph: int | None = None,
) -> dict:
    argv = [
        ".venv/bin/python",
        "bench/e2e/decode.py",
        "--only",
        only,
        "--steps",
        str(steps),
        "--repeats",
        str(repeats),
        "--out",
        f"{{output}}/{name}",
    ]
    if repeats_non_graph is not None:
        argv += ["--repeats-non-graph", str(repeats_non_graph)]
    return {
        "kind": "bench",
        "argv": argv,
        "result": f"{{output}}/{name}/bench.json",
    }


def memory_step(name: str = "memory", only: str = "graph-512-1024", steps: int = 4) -> dict:
    return {
        "kind": "memory",
        "argv": [
            ".venv/bin/python",
            "scripts/m3_memory.py",
            "--only",
            only,
            "--steps",
            str(steps),
            "--out",
            f"{{output}}/{name}",
        ],
        "result": f"{{output}}/{name}/memory.json",
    }


def profile_step(name: str = "profile", steps: int = 2) -> dict:
    return {
        "kind": "profile",
        "argv": [
            "bash",
            "scripts/m3_profile.sh",
            "--steps",
            str(steps),
            "--out",
            f"{{output}}/{name}",
        ],
        "result": f"{{output}}/{name}/timeline_summary.json",
    }


def make_request(
    session: str,
    commit: str,
    steps: list[dict],
    timeout_s: float = 60,
) -> dict:
    return {
        "session_id": session,
        "commit": commit,
        "timeout_s": timeout_s,
        "steps": steps,
    }


# ------------------------------------------------------------------ CLI 调用
def cli(repo: Path, *args: str, env: dict | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    command = [
        PYTHON,
        str(SCRIPT),
        "--root",
        str(repo),
        "--state-dir",
        str(repo / "results/experiments"),
        "--worktree-root",
        str(repo / ".tmp/worktrees"),
        *args,
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=timeout,
    )


def payload(proc: subprocess.CompletedProcess) -> dict:
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - 失败时给出可读信息
        raise AssertionError(
            f"stdout 不是 JSON: {proc.stdout!r}\nstderr: {proc.stderr!r}\n"
            f"exit={proc.returncode}"
        ) from exc


def submit(repo: Path, request: dict, env: dict | None = None) -> dict:
    path = repo / ".tmp" / f"request-{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request), encoding="utf-8")
    proc = cli(repo, "submit", "--request", str(path), env=env)
    data = payload(proc)
    assert proc.returncode == 0, (proc.returncode, data, proc.stderr)
    return data


def serve(repo: Path, *extra: str, env: dict | None = None, timeout: float = 180) -> subprocess.CompletedProcess:
    return cli(repo, "serve", "--no-gpu-guard", *extra, env=env, timeout=timeout)


def start_serve(
    repo: Path, *extra: str, env: dict | None = None, timeout: float = 180
) -> subprocess.Popen:
    command = [
        PYTHON,
        str(SCRIPT),
        "--root",
        str(repo),
        "--state-dir",
        str(repo / "results/experiments"),
        "--worktree-root",
        str(repo / ".tmp/worktrees"),
        "serve",
        "--no-gpu-guard",
        *extra,
    ]
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **(env or {})},
    )


def result(repo: Path, job_id: str, env: dict | None = None) -> dict:
    proc = cli(repo, "result", job_id, env=env)
    data = payload(proc)
    assert proc.returncode == 0, (proc.returncode, data, proc.stderr)
    return data


def status(repo: Path, env: dict | None = None) -> dict:
    proc = cli(repo, "status", env=env)
    data = payload(proc)
    assert proc.returncode == 0, (proc.returncode, data, proc.stderr)
    return data


def wait_job(repo: Path, job_id: str, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return cli(repo, "wait", job_id, *extra, env=env)


def wait_state(repo: Path, job_id: str, states: set[str], timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = result(repo, job_id)
        if data.get("state") in states:
            return data
        time.sleep(0.1)
    raise AssertionError(f"{job_id} 未进入 {states}，最后状态: {data.get('state')}")


def pid_gone(pid: int, timeout: float = 10) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            if stat.rsplit(") ", 1)[1].split()[0] == "Z":
                return True
        except (FileNotFoundError, IndexError):
            return True
        time.sleep(0.05)
    return False
