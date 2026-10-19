#!/usr/bin/env bash
# M1 步骤 4：AR decode benchmark（depth 512 / 8192）。
#
# 注意：需要能访问 /dev/nvidia*（在 bwrap 沙箱外运行）。
# 输出：results/ar_512.json、results/ar_8k.json、results/gpu_monitor.csv、
#       results/ar_bench_times.txt
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

B=llama.cpp/build/bin/llama-bench
M=models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf

# 1 秒采样一次 GPU 状态，跑完 bench 后停掉。
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem \
  --format=csv -l 1 > results/gpu_monitor.csv 2>&1 &
MON=$!
sleep 1

echo "start $(date -Is) epoch=$(date +%s)" | tee results/ar_bench_times.txt

# -p 0：不测 prefill；-n 512：每次生成 512 token；-r 5：重复 5 次。
# -v：把模型加载和 buffer 信息写进 stderr 日志，便于步骤 5 分析。
run() {
  local depth=$1 out=$2 log=$3
  $B -m "$M" -ngl 999 -fa on -ctk f16 -ctv f16 \
    -p 0 -n 512 -d "$depth" -r 5 -v -o json > "$out" 2> "$log"
  echo "depth${depth} rc=$? $(date -Is) epoch=$(date +%s)" | tee -a results/ar_bench_times.txt
}

run 512  results/ar_512.json results/ar_512.log
run 8192 results/ar_8k.json  results/ar_8k.log

kill "$MON" 2>/dev/null
wait "$MON" 2>/dev/null
