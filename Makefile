# 所有入口命令：环境检查、编译、测试、benchmark、profiling。
# GPU 命令必须在 full access 下运行（bwrap 沙箱看不到 /dev/nvidia*）。

.PHONY: m2-preflight m2-manifest m2-test m2-numeric m2-micro m2-smoke m2-memory m2-chunk \
        m3-fixtures m3-check m3-bench m3-memory m3-profile m3-sol m3-attn-effect m3-quick \
        verify verify-run check-results \
        experiment-test experiment-status experiment-serve

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

# ---- 开发循环：产物契约 + 按改动推荐最小验证 ---------------------------------

# 校验 results/ 下 run dir 的文件/字段/计数契约（CPU，秒级）
check-results:
	.venv/bin/python scripts/check_results.py --all

# 按改动文件推荐最小验证档位（只打印）
verify:
	.venv/bin/python scripts/verify.py

# 直接执行推荐档位；关闭前用 verify-close 连全量一起跑
verify-run:
	.venv/bin/python scripts/verify.py --run

verify-close:
	.venv/bin/python scripts/verify.py --run --include-s5

# ---- GPU 实验队列：多 session 串行执行 check/bench/memory/profile ------------

# CPU 测试：请求校验、worktree 隔离、FIFO、超时/取消、重启核对、产物校验
experiment-test:
	.venv/bin/python -m pytest tests/experiment -q --disable-warnings

experiment-status:
	.venv/bin/python scripts/experiment.py status

# GPU 主机上的唯一执行器（全访问；常驻）
experiment-serve:
	.venv/bin/python scripts/experiment.py serve

# ---- M3：decode 性能 baseline -------------------------------------------------

# 快档（约 5 分钟）：改一点就跑这个。覆盖 graph capture + capacity guard +
# 分阶段显存 + 两套计时；改动涉及 reference/reset/boundary 时再加跑
# `--only numeric-512`。产物写到 .tmp/，不进 results/。
m3-quick:
	.venv/bin/python scripts/m3_check.py --only graph-1024 --steps 16 --out .tmp/m3-quick/check
	.venv/bin/python scripts/m3_memory.py --only graph-512-1024 --steps 32 --out .tmp/m3-quick/memory
	.venv/bin/python bench/e2e/decode.py --only graph-512-1024 --repeats 1 --steps 32 --out .tmp/m3-quick/bench

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
