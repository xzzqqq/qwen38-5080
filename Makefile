# 所有入口命令：环境检查、编译、测试、benchmark、profiling。
# GPU 命令必须在 full access 下运行（bwrap 沙箱看不到 /dev/nvidia*）。

.PHONY: m2-preflight m2-manifest m2-test m2-numeric m2-micro m2-smoke m2-memory m2-chunk \
        m3-fixtures m3-check m3-bench m3-memory m3-profile m3-sol m3-attn-effect

# M2 环境 / CUDA 头文件 / GPU 预检
m2-preflight:
	.venv/bin/python scripts/m2_preflight.py

# M2 checkpoint header -> tensor manifest + 初版显存账
m2-manifest:
	.venv/bin/python scripts/m2_manifest.py

# M2 op 对拍 + shape-stable 回归（需要 GPU）
m2-test:
	.venv/bin/python -m pytest tests/ops tests/model -q --disable-warnings

# M2 runner 数值验收：NVFP4 dequant / GDN FP32 / KV/state / lm_head（需要 GPU）
m2-numeric:
	.venv/bin/python scripts/m2_numeric.py --max-new-tokens 16

# M2 NVFP4Linear microbench（需要 GPU）
m2-micro:
	.venv/bin/python bench/micro/nvfp4_linear.py

# M2 三条真实 prompt 的 eager 生成（需要 GPU）
m2-smoke:
	.venv/bin/python scripts/m2_smoke.py

# M2 512 / 7680 depth + 512 decode 容量检查（需要 GPU）
m2-memory:
	.venv/bin/python scripts/m2_capacity.py

# M2 边界对齐的 full vs chunked prefill 一致性（需要 GPU）
m2-chunk:
	.venv/bin/python scripts/m2_chunk_consistency.py --length 256 --chunk 128

# ---- M3：decode 性能 baseline -------------------------------------------------

# P0 fixtures：固定深度 prompt + greedy trace + 三条真实 smoke（需要 GPU，约 3 分钟）
m3-fixtures:
	.venv/bin/python scripts/m3_fixtures.py

# P0 正确性：四模式 + padded reference + reset/边界/smoke（需要 GPU）
m3-check:
	.venv/bin/python scripts/m3_check.py

# P0 性能：四执行方式 × depth/capacity 矩阵，5 次重复，两套计时（需要 GPU，约 80 分钟）
m3-bench:
	.venv/bin/python bench/e2e/decode.py --repeats 5 --steps 511

# P0 分阶段显存：load / cache / prefill / capture / decode / reset（需要 GPU）
m3-memory:
	.venv/bin/python scripts/m3_memory.py

# Nsight Systems 时间线：eager / graph × 两档 depth（需要 GPU，约 10 分钟）
m3-profile:
	bash scripts/m3_profile.sh

# 实际字节账 + SOL，生成 docs/sol.md（需要 GPU 读权重）
m3-sol:
	.venv/bin/python scripts/m3_sol.py

# 诊断：padded vs unpadded attention 的 FP reduction 差异（小 GPU 占用）
m3-attn-effect:
	.venv/bin/python bench/micro/attn_length_effect.py
