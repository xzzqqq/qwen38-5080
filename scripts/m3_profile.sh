#!/usr/bin/env bash
# M3 Nsight Systems 采集：eager / graph × depth 512 / 7680。
#
# 用法（在仓库根目录，bash）：
#   source scripts/m2_env.sh
#   scripts/m3_profile.sh                        # 用默认步数 8
#   scripts/m3_profile.sh --steps 16             # 换步数
#   scripts/m3_profile.sh --out /abs/run-dir     # 显式输出目录（实验队列用）
#   STEPS=16 scripts/m3_profile.sh               # 兼容旧写法
#
# 产物：<out>/<case>/ 下的 .nsys-rep、nsys stats CSV 和解释用 JSON；
# 不给 --out 时写 results/m3/<run-id>-m3-profile。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv}"
STEPS="${STEPS:-8}"
OUT_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out)   OUT_ARG="${2:?--out needs a directory}"; shift 2 ;;
    --steps) STEPS="${2:?--steps needs a number}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
RUN_ID="$(date +%Y%m%dT%H%M%S)"
if [ -n "$OUT_ARG" ]; then
  mkdir -p "$OUT_ARG"
  # 统一成绝对路径：provenance 与 summary 里不要出现依赖 cwd 的相对路径。
  OUT="$(cd "$OUT_ARG" && pwd)"
else
  OUT="$ROOT/results/m3/${RUN_ID}-m3-profile"
  mkdir -p "$OUT"
fi

# nsys 2023.4（随 CUDA 12.4 装的）看不到 sm_120 的 GPU：报告里没有 CUDA kernel data。
# 优先用 /opt/nvidia/nsight-systems-cli 里较新的 CLI（2026.2.1 起支持 Blackwell）。
pick_nsys() {
  if [ -n "${NSYS:-}" ]; then echo "$NSYS"; return; fi
  local best=""
  for candidate in /opt/nvidia/nsight-systems-cli/*/bin/nsys; do
    [ -x "$candidate" ] || continue
    if [ -z "$best" ] || [[ "$candidate" > "$best" ]]; then best="$candidate"; fi
  done
  if [ -n "$best" ]; then echo "$best"; return; fi
  command -v nsys || true
}
NSYS="$(pick_nsys)"
if [ -z "$NSYS" ] || [ ! -x "$NSYS" ]; then
  echo "nsys not found" >&2
  exit 1
fi
# provenance：commit、dirty patch、源码 manifest（nsys 采集也是结果的一部分）。
"$VENV/bin/python" - "$OUT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from runner import report as reporting

reporting.provenance(Path("."), {"kind": "m3-profile"}, out_dir=Path(sys.argv[1]))
PY

echo "using $NSYS" | tee "$OUT/nsys_version.txt"
"$NSYS" --version >> "$OUT/nsys_version.txt" 2>&1 || true

run_case() {
  local execution="$1" depth="$2" capacity="$3"
  local case_dir="$OUT/${execution}-${depth}"
  mkdir -p "$case_dir"
  echo "=== profiling $execution depth=$depth capacity=$capacity steps=$STEPS"
  # --cuda-graph-trace=node：把 graph replay 展开成每个 node 一行，才能数每步 node/kernel。
  "$NSYS" profile \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --sample=none --cpuctxsw=none \
    --force-overwrite=true \
    -o "$case_dir/trace" \
    "$VENV/bin/python" scripts/m3_profile.py \
      --execution "$execution" --depth "$depth" --capacity "$capacity" --steps "$STEPS" \
      --out "$case_dir/meta.json" \
    > "$case_dir/profile.log" 2>&1

  # 内核汇总 + 每次 kernel 的明细（算 <5µs 占比和直方图）
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    "$case_dir/trace.nsys-rep" > "$case_dir/kern_sum.csv" 2> "$case_dir/stats.err" || true
  "$NSYS" stats --report cuda_gpu_trace --format csv --force-export=true \
    "$case_dir/trace.nsys-rep" > "$case_dir/gpu_trace.csv" 2>> "$case_dir/stats.err" || true
  "$NSYS" stats --report cuda_gpu_mem_time_sum,nvtx_sum --format csv --force-export=true \
    "$case_dir/trace.nsys-rep" > "$case_dir/mem_nvtx_sum.csv" 2>> "$case_dir/stats.err" || true
}

run_case eager 512 0
run_case static-exact 512 1024
run_case static-cap 512 1024
run_case graph 512 1024
run_case eager 7680 0
run_case graph 7680 8192

"$VENV/bin/python" scripts/m3_profile_stats.py --dir "$OUT" | tee "$OUT/summary.md"
echo "output: $OUT"
