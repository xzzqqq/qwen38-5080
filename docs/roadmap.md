# Qwen3.8-27B on RTX 5080：Roadmap

- 日期：2026-10-18
- 修订：2026-10-20，按 M1 实机结果调整性能参照、显存预算和 Phase 1–4 的验收标准

## 0. 修订摘要（2026-10-20）

- M1 只作外部参照：IQ4_XS 的 55.90 / 54.19 tok/s 对应逻辑带宽约 80%，不能直接外推到 NVFP4。
- 显存预算改用实测可用 16.61 GB（15841 MiB）；DFlash2 的 16.39–16.74 GB 是否装得下改为待验证。
- Baseline 明确为 Phase 1（M1–M3）：M1 外部参照，M2 正确实现，M3 自写 runner 的性能 baseline。
- Phase 2 以约 80% 带宽（FP8 KV@8K 约 51 tok/s）作为进入 spec 的门槛；55 tok/s 留作后续目标。
- Spec 提前验证容量、τ 和相对同一 target AR 的净加速；按有效输出速度选择 MTP / DFlash2。

## 1. 目标与范围

| 项 | 决定 |
|---|---|
| 目标 | 在单张 RTX 5080（16 GiB）上，把 Qwen3.8-27B 的 bs=1 decode tok/s 推到接近 SOL |
| 不在范围内 | TTFT 只记录、不优化；不做 batch>1 |
| context | prompt + 输出 ≤ 8192 |
| 显存预算 | 空闲可用 16.61 GB（15841 MiB，已扣 463 MiB 驱动保留）；显示器未接 5080 |
| 质量 | 不设目标，用 §3 的两条约束代替 |

## 2. Benchmark 规格

| 项 | 取值 |
|---|---|
| prompt 集 | 使用 DFlash2 card 用过的 5 个数据集：GSM8K / MATH-500 / HumanEval / MBPP / MT-Bench。每个取 20–50 条，选定后固定不变 |
| 固定深度对照 | `depth=512 / 7680, n_gen=512`，greedy；总长 1024 / 8192。用于和 SOL、M1 外部参照对齐。M1 的 8192→8704 超出本项目 8K 口径，只作外部参照 |
| 真实 workload | 开启 thinking；max_new_tokens 取 2048–4096，保证 prompt + 输出 ≤ 8K |
| 采样 | 开发时用 greedy。报告时再补一组 thinking 的推荐设置：temp 1.0 / top-p 0.95 / top-k 20 / min-p 0（DFlash2 card 和 unsloth card 推荐的都是这组） |
| 主指标 | decode tok/s（不含 prefill）；%SOL，分母取 960 GB/s。BF16 / FP8 KV 分开报告，并分别计算 SOL |
| 有效输出速度 | 输出 token 数 ÷ 总墙钟时间，计入 draft、verify、commit、sample 和空隙。spec 阶段以它为准，不直接套用 card 的 tok/s 作验收线 |
| 辅助指标 | τ = 实际输出 token 数 / verify 次数（含 correction / bonus token）；每个 step 的 draft / verify / commit / sample / 空隙拆分；各 kernel 达到的带宽；峰值显存 |

## 3. 约束

### 3.1 格式固定

格式是 SOL 定义的一部分，格式一变就要重算 SOL。

- 所有大的 linear 层和 lm_head：NVFP4 W4A4。元素是 E2M1，每 16 个元素共用一个 UE4M3 scale，外加一个 FP32 的全局 scale。
- GDN 的递归 state：FP32。
- KV：FP8；baseline 阶段可以先用 BF16。BF16 / FP8 两套 KV 分开报告速度，并分别计算 SOL。
- embedding：BF16，放在 host 内存；只优化 device 端 gather / H2D 传输，不把整份 embedding 复制到 GPU。
- 其他格式只能作为单独标注的变体，不替换主线。

### 3.2 实现正确

- **算子级**：每个自写算子都要和参考实现对拍。参考实现指用同一份 NVFP4 权重 dequant 后、用 PyTorch 跑出的结果。
  - W4A4 对拍必须使用相同的 activation quantization。
  - 量化（FP4 值 / scale）与布局转换（nibble 顺序 / scale 排列）分别验证，不能只测端到端。
- **端到端**：用同一串 token id 输入参考 runner 和自写 runner，逐层 hidden states、KV/GDN cache/state 和最终 logits 都要对齐。
  - 容差在集成前固定，不随结果调整。
- **spec decode**：
  - 回滚后的 state，要和只跑已接受前缀得到的 state 逐位一致；
  - greedy 下，每个被接受的 token 都要等于 verify 时 target logits 的 argmax。
- **GSM8K 正确率**：只用来发现 bug。
- **τ 不能当质量指标**：一个退化成复读的模型，τ 反而会很高。

## 4. SOL 与参照数字

可用显存：空闲 `Reserved=463 MiB`、`Free=15841 MiB`，实际可用 **16.61 GB**；总容量 16303 MiB，其中驱动保留 463 MiB。显示器未接在 5080 上（`display_active=Disabled`）。

| 项 | 数值 | 说明 |
|---|---|---|
| AR SOL（FP8 KV，8K） | 64.1 tok/s | 每 token 读 14.98 GB：权重 14.41 + KV 0.27 + GDN state 读写 0.30，再除以 960 GB/s |
| AR SOL（BF16 KV，8K） | ≈63.0 tok/s | 每 token 读 15.25 GB：权重 14.41 + KV 0.54 + state 0.30，再除以 960 GB/s |
| AR @ 80% 带宽（FP8 / BF16） | ≈51.2 / 50.4 tok/s | Phase 2 的务实目标；按实际字节计算 |
| AR @ 90% 带宽（FP8） | ≈57.7 tok/s | 假设值，不是实测上限 |
| AR 55 tok/s 所需带宽 | ≈85.8% | 14.98 × 55 ÷ 960；留作后续目标 |
| M1 实机参照（IQ4_XS，外部） | 55.90 / 54.19 tok/s | depth 512 / 8192，GPU 权重 13.35 GB；按平均有效 context 估算逻辑带宽 766 / 770 GB/s（约 80%） |
| 现有推理栈 AR（外推） | 46–50 tok/s | 5090 实测乘带宽比；只作量级参照，与 M1 的 IQ4_XS 不是同一格式 |
| DFlash2 SOL | 236–314 tok/s | 每个 step 17.4 ms；τ 取 4.10–5.46（card 数据，BF16，temp 1.0） |
| MTP 7 步 SOL | 166–222 tok/s；用 32K 截断词表最多 207–278 | τ 取 3.74–5.02 |
| 现有推理栈 DFlash2（外推） | 约 125–170 tok/s | 发布文章的 2.7–3.4× 乘 5080 的 AR 速度；只作外推参照 |
| 显存：baseline（NVFP4，BF16 KV，8K） | **15.62 GB allocated / 15.86 reserved**（M2 实测） | 7680+512 固定 token；8K 后 free **0.413 GB**；比 1b 的 15.93 GB 估算更紧；eager runner 诊断值 |
| 显存：带 DFlash2 | 16.39–16.74 GB | 对 16.61 GB 可用：余量 +0.22 ~ −0.13 GB；但 baseline 实测只剩 0.413 GB，drafter 容量更紧；**待 Phase 3 验证** |

口径说明：

1. **M1 的 tok/s 不能直接和 NVFP4 SOL 比**：IQ4_XS 的 GPU 权重 13.35 GB，比 NVFP4 预计的 14.41 GB 少约 1.06 GB（NVFP4 多约 8%）。8K 下按同样 80% 带宽，NVFP4 AR 约 51 tok/s。M1 证明的是 5080 上约 80% 逻辑带宽可达，不是自写 runner 的实现效率。
2. **逻辑带宽不是 DRAM 实测**：这里用“每 token 读字节模型 × tok/s”估算，不是 `ncu` 的 DRAM 流量。要实测 DRAM 字节见 §8。
3. **平均有效 context**：固定深度测试的 context 会从 depth 增长到 depth + n_gen，M1 带宽按 depth + n_gen/2 估算；SOL 表按满 8192 计，略保守。
4. **显存余量**：M1（IQ4_XS）server 1 Hz 采样峰值 14313 MiB，对 15841 MiB 可用余约 1.60 GB；若用 16303 MiB 总量会得到约 2.0 GB，漏扣了 463 MiB 保留。
5. **单位**：本文 GB = 10^9 字节；MiB 换算按 1 MiB = 1.048576 MB。
6. **优先级**：AR 从 80% 提到 85% 带宽只多约 6.25% 吞吐；Phase 2 不应为最后几个百分点推迟 spec。
7. **M2 1b 实测**：manifest 里文本 GPU 权重是 14.452 GB（2387 个 tensor，两个 shard CRC32 已校验），和表中 14.41 GB 的估算差约 0.3%；SOL 表暂保留估算，M3 按实际字节重算。

## 5. 关键决定

| 决定 | 理由 |
|---|---|
| 用 W4A4 | 显存只取决于权重位宽，A4 不额外省显存。好处在 verify：W4A4 能走原生 FP4 tensor core（`mxf4nvf4`），block 8 约 0.46 ms；W4A16 峰值要 3.6 ms |
| 自己写 runner，不改 vLLM / SGLang | 现成的 checkpoint 有 17.92 GB（embedding 和 vision 都是 BF16），原样装不进 16 GB |
| baseline 用 transformers 骨架 + FlashInfer NVFP4 GEMM 拼起来 | 模型结构的正确性直接继承自参考实现；之后逐个替换算子，每换一个就对拍一次。M2 先固定依赖版本和 checkpoint revision |
| 先做 MTP，再做 DFlash2 | MTP 是标准 decoder 层，能复用已有 kernel，可以先把 verify 和回滚单独做出来。最终按有效输出速度选择，不预设 DFlash2 一定胜出 |
| 回滚先用 lazy commit，ReplaySSM 留作进阶 | lazy commit 额外显存约为 0；保存整份快照在 block 8 时至少要 1.21 GB |
| 只做线性 draft chain | DeltaNet 做树验证时，每个分叉都要单独一条 state 路径 |
| 8K 下先不做层 offload | M2 实测峰值后再定；优先减少临时副本、复用 workspace、分块 prefill，仍不足再评估 offload |
| Phase 2 以约 80% 带宽为门槛 | 80→85% 只多 6.25% 吞吐；先把 spec 的容量、τ、净加速验证清楚，再决定是否回头抠 kernel |
| 精确测量按需进行（显存除外） | 性能方向用 SOL 判断；但显存峰值必须在 M2 和 Phase 3 实测 |
| embedding 放 host BF16 | 只优化 device 端 gather / H2D 传输；不把整份 embedding 搬到 GPU |

## 6. 里程碑

### 6.1 Phase 1：Baseline（M1–M3）

M1 只提供低成本外部参照；M2 建立正确实现；M3 建立自写 runner 的性能 baseline。M1 的数字不参与验收，只用于校准带宽效率。

| 里程碑 | Status | 内容 | 退出标准 | 粗估 |
|---|---|---|---|---|
| M1 实机参照 | ✅ 完成 2026-10-19 | 用 llama.cpp 跑 unsloth 的 `UD-IQ4_XS`（14.25 GB），用 `llama-bench` 测 depth 512 和 8192；固定 commit，脚本复现。详细清单见 `docs/plans/m1.md` | 已拿到：55.90 / 54.19 tok/s；逻辑带宽约 80%；可用显存 16.61 GB；真实生成正常。详见 [M1 journal](journal/2026-10-19-m1-llamacpp-reference.md) | 半天 |
| M2 拼装 runner 跑通 | 进行中（1a/1b/§2/§3 完成；chunked 一致性待查） | 执行清单见 [docs/plans/m2.md](plans/m2.md)。提前固定依赖版本和 checkpoint revision；用 transformers 里 qwen3_5 的文本模型做骨架；实现 NVFP4Linear（调用 FlashInfer `mm_fp4`）；embedding 放 CPU；只加载 `model.language_model.*` 和 `lm_head.*`，并列出所有保留高精度的 tensor（`in_proj_a/b`、`conv1d`、各个 norm 等）；核实 FP4 nibble / scale 的 tensor 布局、`alpha = input_scale × weight_scale_2`、FlashInfer backend、`M=1 / 8` 可用性和显存账；先用 eager 模式跑 | W4A4 对拍（相同 activation quantization）通过；量化与布局转换分别验证；runner 与参考 runner 数值对拍（同一 token 输入，逐层 hidden / cache/state / 最终 logits，容差先固定）；8K 容量验证通过；**不设 tok/s 门槛** | 2–3 天 |
| M3 decode baseline | 未开始 | 改用 static cache，自己写 decode loop 并录成 CUDA Graph；按 §2 写固定 harness（多次运行，报告中位数和波动，记录峰值显存）；按实际 tensor 布局和 KV 格式重算每步字节并算 SOL；抓第一张 Nsight Systems 时间线；分别报告模型执行速度和含 sampling 的有效输出速度 | 拿到 baseline tok/s 和时间分布；top 热点明确；加载 / repack / prefill / decode 的峰值显存有记录；BF16 / FP8 KV 分开报 | 1–2 天 |

### 6.2 优化阶段

优先级：AR 从 80% 提到 85% 带宽只多约 6.25% 吞吐。Phase 2 达到约 80% 且消除明显瓶颈后就进 spec，先用实测的容量、τ 和净加速判断后续 kernel 投入方向。

| Phase | Status | 内容 | 退出标准 |
|---|---|---|---|
| 2 AR 逼近 SOL | 未开始 | 按 M3 时间线上的热点逐个替换：NVFP4 GEMV（swap-AB + `mxf4nvf4`，并和 CUDA core 版本对照）、GDN decode、FP8 KV 的 decode attention、算子融合（norm + 激活量化，gate/up + SiLU·mul）、embedding 的 device 端 gather / H2D 传输优化（权重仍在 host BF16）、整个 step 录成一张 CUDA Graph | 逻辑带宽 ≥ 80%（按实际字节；FP8 KV@8K 约 51 tok/s，BF16 KV 约 50 tok/s）；明显瓶颈（launch / 同步 / 未融合）消除；55 tok/s（约 86% 带宽）留作后续独立目标，不设为本阶段门槛 |
| 3a spec decode：MTP | 未开始 | 从官方 BF16 checkpoint 取 MTP 权重（查 safetensors index，用 HTTP Range 只下载 `mtp.*`）并量化。实现 N=8 的 verify：小 N 的 W4A4 GEMM、N 个 query 的 attention、GDN verify 加 lazy commit。先做 greedy 接受，再做 rejection sampling | 回滚后 state 逐位正确；接受的 token 都等于 argmax；实测 τ（含 bonus / correction）、step 拆分、显存峰值、相对同一 target AR 的净加速；按有效输出速度决定是否进 3b |
| 3b spec decode：DFlash2 | 未开始 | 先做容量预演（drafter 权重 + codebook + KV injection 的峰值）；再用 RTN 把 drafter 量化到 NVFP4；实现 5 层 block drafter + fc + dyn-conv + selector + codebook gather；实现 KV injection | 实测 τ 并与 BF16 对照（card 均值 4.80）；量化后 τ 低于 BF16 的 75% 只作诊断，不直接当停止条件；最终按有效输出速度、显存和净加速在 MTP / DFlash2 间选择 |
| 4 spec 路径逼近 SOL | 未开始 | 按实测瓶颈优化 draft / verify / commit / sample / 同步；spec 循环全部在 device 端完成，不和 host 同步；整个 step 录成一张 graph；融合 verify；megakernel 仅在时间线证明层间空隙 / 权重预取是主因时采用 | 实测 τ 下 spec SOL 的 80–85%；最终指标是含 sampling 的有效输出速度；125–170 tok/s 仅作外推参照 |
| 5 可选 | — | 更长的 context（offload：offload 层均匀分散 + ring buffer）、state 改 BF16、KV 改 INT8、block 改 16、截断词表、prefill 优化；embedding 仍 host BF16 | — |

## 7. 风险与开放问题

M1 已回答：空闲可用 16.61 GB，显示器没有占用 5080。剩余显存风险集中在 NVFP4 baseline 的峰值，以及加 drafter 之后。

| 问题 | 什么时候能回答 | 兜底方案 |
|---|---|---|
| NVFP4 baseline 能否装进 16.61 GB | ✅ 已答（8K 实测 15.62 allocated / 15.86 reserved，free 0.413） | 余量很紧：减少临时副本、复用 workspace、分块 prefill；仍不足才评估 offload |
| chunked prefill 与单段是否数值一致 | M2 待查（300-token cos 0.929，mask/position 已排除） | 短 prompt 用单段；8K capacity 只验容量；必要时做 state 对拍/固定 chunk 边界 |
| 加 drafter 后是否还装得下 | Phase 3（先做容量预演） | 停在 MTP；drafter 权重 / codebook 放 host，或更激进地量化 |
| 量化后的 τ 与净加速 | Phase 3 | MTP；按有效输出速度选 |
| FlashInfer `mm_fp4` 在 `M=1 / 8` 的可用性 | ✅ 已答（M=1/8/128 + 尾块；backend `cutlass`） | 换 backend；或先用 CUDA core 版本把流程跑通 |
| `mm_fp4` 在 M=1 时的效率 | M3（§2 已给单 op microbench） | 只影响 baseline 的数字，Phase 2 会替换掉 |
| transformers 的 hybrid cache 能不能直接录进 CUDA Graph | M3 | 自己写静态的 state 管理 |
| 小 N 的 W4A4 GEMM 能跑到多少带宽 | Phase 2–3 | N=8 时理论上离算力上限还很远 |
| 逻辑带宽与 DRAM 实测是否一致 | 需要时 | `ncu` 的 `dram__bytes` / `dram__throughput` |

## 8. 按需测量

显存不是纯按需项：M2 必须测加载、repack、prefill、decode 四个阶段的峰值，Phase 3 加 drafter 后复核。其余测量按触发条件进行。

| 触发条件 | 测什么 | 工具 |
|---|---|---|
| M2 加载 / repack / prefill / decode | 各阶段显存峰值 | `torch.cuda.mem_get_info()` + `nvidia-smi` 采样 |
| Phase 3 加 drafter 后 | 峰值和回滚额外占用 | 同上 |
| 某个访存 kernel 卡在 960 GB/s 的 80–90% 上不去 | 纯读带宽的实际上限；DRAM 实际字节 | 写一个简单的 read kernel；或看 `ncu` 的 `dram__throughput` / `dram__bytes` |
| 出现 OOM，或加上 drafter 后显存吃紧 | 实际可用显存 | `torch.cuda.mem_get_info()` |
| step 总时间对不上各 kernel 时间之和 | launch、同步、kernel 之间的空隙 | Nsight Systems |
| 开始做 offload | H2D 带宽 | 到时再测 |

## 9. 参考

- NVFP4 checkpoint：https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 。格式为 W4A4，lm_head 也是 NVFP4；embedding、vision、`in_proj_a/b`、`conv1d`、norm 保留 BF16
- DFlash2：https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2 ；https://inco.ai/blog/dflash2/
- GGUF（M1 用）：https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- FlashInfer `mm_fp4`：https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html
- M1 记录：[journal](journal/2026-10-19-m1-llamacpp-reference.md)；原始结果 [ar_512.json](../results/ar_512.json)、[ar_8k.json](../results/ar_8k.json)、[gpu_monitor.csv](../results/gpu_monitor.csv)；复现 [run_ar_bench.sh](../scripts/run_ar_bench.sh)、[run_server_check.sh](../scripts/run_server_check.sh)；llama.cpp commit `f174553e1621c6a22a64e7bfcbb58d3a5c49e6fb`
