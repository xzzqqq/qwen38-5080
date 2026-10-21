#!/usr/bin/env bash
# M3 关闭流程：一条命令、可重入。任何一步失败即停；最后校验产物契约。
#
# 用法（仓库根目录，bash）：
#   source scripts/m2_env.sh
#   scripts/m3_close.sh                 # 全量（bench 11 场景 × 5 次，约 95 分钟）
#   REPEATS=3 scripts/m3_close.sh       # 只改 bench 重复次数
#   SKIP=bench,profile scripts/m3_close.sh
#
# 说明：
# - fixtures 已冻结时不重跑；缺失或 hash 不符才生成（check/bench 启动时会校验 hash）。
# - 每个阶段的产物都写到 results/m3/<run-id>-<kind>/，最后用 check_results 校验。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="${VENV:-$ROOT/.venv}"
REPEATS="${REPEATS:-5}"
REPEATS_NON_GRAPH="${REPEATS_NON_GRAPH:-}"
STEPS="${STEPS:-511}"
SKIP="${SKIP:-}"
FIXTURES="$ROOT/bench/workloads/m3-p0"

echo "== M3 close =="
echo "commit: $(git rev-parse HEAD)"
echo "dirty:  $(git status --porcelain | wc -l) 项"
echo "repeats: $REPEATS (bench), steps: $STEPS, skip: ${SKIP:-none}"
echo

skip() {
  case ",$SKIP," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

step() {
  echo "== $1 =="
  shift
  "$@"
}

if [ ! -f "$FIXTURES/manifest.json" ]; then
  step "fixtures" "$VENV/bin/python" scripts/m3_fixtures.py
else
  echo "== fixtures: 已存在，跳过（hash 由 check/bench 启动时校验）"
fi

if ! skip check; then
  step "check" "$VENV/bin/python" scripts/m3_check.py
fi

if ! skip memory; then
  step "memory" "$VENV/bin/python" scripts/m3_memory.py
fi

if ! skip bench; then
  # --resume：被中断时只补跑没完成的场景；源码变了会自动落到新目录。
  BENCH_ARGS=(--repeats "$REPEATS" --steps "$STEPS" --resume)
  if [ -n "$REPEATS_NON_GRAPH" ]; then
    BENCH_ARGS+=(--repeats-non-graph "$REPEATS_NON_GRAPH")
  fi
  step "bench" "$VENV/bin/python" bench/e2e/decode.py "${BENCH_ARGS[@]}"
fi

if ! skip profile; then
  step "profile" env STEPS=8 bash scripts/m3_profile.sh
fi

if ! skip sol; then
  latest_bench="$(ls -d "$ROOT"/results/m3/*-m3-bench 2>/dev/null | sort | tail -1 || true)"
  if [ -n "$latest_bench" ] && [ -f "$latest_bench/bench.json" ]; then
    step "sol" "$VENV/bin/python" scripts/m3_sol.py --bench "$latest_bench/bench.json"
  else
    step "sol" "$VENV/bin/python" scripts/m3_sol.py
  fi
fi

step "产物契约" "$VENV/bin/python" scripts/check_results.py --all --strict
echo
echo "M3 close 完成。结果在 results/m3/ 下，SOL 在 docs/sol.md。"
