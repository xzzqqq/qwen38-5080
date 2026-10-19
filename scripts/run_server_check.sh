#!/usr/bin/env bash
# M1 步骤 6：用 llama-server + --jinja 跑 math / code / chat 三条真实 prompt。
#
# 注意：需要能访问 /dev/nvidia*（在 bwrap 沙箱外运行）。
# 输出：results/server.log、results/payload_*.json、results/gen_*.json
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

S=llama.cpp/build/bin/llama-server
M=models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf

# greedy（temperature=0），三条都开 thinking。
python3 - <<'PY'
import json
prompts = {
    "math": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "code": "Write a Python function is_palindrome(s) that returns True if s is a palindrome, ignoring case and non-alphanumeric characters. Include 3 test cases.",
    "chat": "In three sentences, explain why LLM decode at batch size 1 is memory-bandwidth bound.",
}
for k, v in prompts.items():
    json.dump({"messages": [{"role": "user", "content": v}], "temperature": 0, "max_tokens": 2048},
              open(f"results/payload_{k}.json", "w"), ensure_ascii=False)
PY

$S -m "$M" -ngl 999 -c 8192 -fa on --jinja --port 8080 > results/server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT

for i in $(seq 1 180); do
    if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
        echo "server ready after ${i}s"
        break
    fi
    sleep 1
done

for k in math code chat; do
    echo "--- $k ---"
    curl -s --max-time 600 http://127.0.0.1:8080/v1/chat/completions \
        -H 'Content-Type: application/json' \
        --data-binary "@results/payload_${k}.json" \
        -o "results/gen_${k}.json"
    echo "$k curl rc=$?"
done
