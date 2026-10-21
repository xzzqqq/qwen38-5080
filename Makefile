# 所有入口命令：环境检查、编译、测试、benchmark、profiling。

.PHONY: m2-preflight m2-manifest m2-test m2-micro m2-smoke m2-memory m2-chunk

# M2 环境 / CUDA 头文件 / GPU 预检
m2-preflight:
	.venv/bin/python scripts/m2_preflight.py

# M2 checkpoint header -> tensor manifest + 初版显存账
m2-manifest:
	.venv/bin/python scripts/m2_manifest.py

# M2 op 对拍 + shape-stable 回归（需要 GPU）
m2-test:
	.venv/bin/python -m pytest tests/ops tests/model -q --disable-warnings

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
