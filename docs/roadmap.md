# Qwen3.8-27B on RTX 5080：Roadmap

- 日期：2026-10-18

## 1. 目标与范围

| 项 | 决定 |
|---|---|
| 目标 | 在单张 RTX 5080（16 GiB）上，把 Qwen3.8-27B 的 bs=1 decode tok/s 推到接近 SOL |
| 不在范围内 | TTFT 只记录、不优化；不做 batch>1 |
| context | prompt + 输出 ≤ 8K |
| 质量 | 不设目标，用 §3 的两条约束代替 |

## 2. Benchmark 规格

| 项 | 取值 |
|---|---|
| prompt 集 | 使用 DFlash2 card 用过的 5 个数据集：GSM8K / MATH-500 / HumanEval / MBPP / MT-Bench。每个取 20–50 条，选定后固定不变 |
| 生成 | 开启 thinking；max_new_tokens 取 2048–4096，保证总长 ≤ 8K |
| 采样 | 开发时用 greedy。报告时再补一组 thinking 的推荐设置：temp 1.0 / top-p 0.95 / top-k 20 / min-p 0（DFlash2 card 和 unsloth card 推荐的都是这组） |
| 主指标 | decode tok/s（不含 prefill）；%SOL，分母取 960 GB/s，需要时再补一个实测的带宽上限 |
| 辅助指标 | τ（每次 verify 平均接受的 token 数）；每个 step 的时间拆分（draft / verify / commit / sample / 空隙）；各 kernel 达到的带宽；峰值显存 |

## 3. 约束

### 3.1 格式固定

格式是 SOL 定义的一部分，格式一变就要重算 SOL。

- 所有大的 linear 层和 lm_head：NVFP4 W4A4。元素是 E2M1，每 16 个元素共用一个 UE4M3 scale，外加一个 FP32 的全局 scale。
- GDN 的递归 state：FP32。
- KV：FP8；baseline 阶段可以先用 BF16。
- embedding：BF16，放在 host 内存。
- 其他格式只能作为单独标注的变体，不替换主线。

### 3.2 实现正确

- **算子级**：每个自写算子都要和参考实现对拍。参考实现指用同一份 NVFP4 权重 dequant 后、用 PyTorch 跑出的结果。
- **端到端**：用固定 prompt 跑，逐层 logits 和最终 logits 都要和参考 runner 对齐。
- **spec decode**：
  - 回滚后的 state，要和只跑已接受前缀得到的 state 逐位一致；
  - greedy 下，每个被接受的 token 都要等于 verify 时 target logits 的 argmax。
- **GSM8K 正确率**：只用来发现 bug。
- **τ 不能当质量指标**：一个退化成复读的模型，τ 反而会很高。

## 4. SOL 与参照数字

| 项 | 数值 | 说明 |
|---|---|---|
| AR SOL | 64.1 tok/s | 每个 token 读 14.98 GB：权重 14.41 + KV 0.27（8K，FP8）+ GDN state 读写 0.30，再除以 960 GB/s |
| AR 按 90% 带宽估算 | ≈ 58 tok/s | 这是假设值，不是实测上限 |
| 现有推理栈的 AR | 46–50 tok/s | 5090 上实测的 vLLM 86.8 / SparkInfer 92.9，乘以 0.536（两卡带宽比）折算 |
| DFlash2 SOL | 236–314 tok/s | 每个 step 17.4 ms；τ 取 4.10–5.46（card 数据，BF16，temp 1.0） |
| MTP 7 步 SOL | 166–222 tok/s；用 32K 截断词表最多 207–278 | τ 取 3.74–5.02 |
| 现有推理栈的 DFlash2 | 约 125–170 tok/s | 用发布文章里的 2.7–3.4×，乘以 5080 上的 AR 速度 |
| 显存：带 DFlash2 | 16.39–16.74 GB | 不接显示器时余量 +0.35 ~ +0.70 GB；回滚不能保存整份 state 快照 |
| 显存：baseline，不带 drafter | 约 16.0 GB | KV 用 BF16 |

## 5. 关键决定

| 决定 | 理由 |
|---|---|
| 用 W4A4 | 显存只取决于权重位宽，A4 不额外省显存。好处在 verify：W4A4 能走原生 FP4 tensor core（`mxf4nvf4`），block 8 约 0.46 ms；W4A16 峰值要 3.6 ms |
| 自己写 runner，不改 vLLM / SGLang | 现成的 checkpoint 有 17.92 GB（embedding 和 vision 都是 BF16），原样装不进 16 GB |
| baseline 用 transformers 骨架 + FlashInfer NVFP4 GEMM 拼起来 | 模型结构的正确性直接继承自参考实现；之后逐个替换算子，每换一个就对拍一次 |
| 先做 MTP，再做 DFlash2 | MTP 是标准 decoder 层，能复用已有 kernel，可以先把 verify 和回滚单独做出来。DFlash2 量化后如果 τ 掉得太多，就用 MTP 兜底，SOL 只低 12–30% |
| 回滚先用 lazy commit，ReplaySSM 留作进阶 | lazy commit 额外显存约为 0；保存整份快照在 block 8 时至少要 1.21 GB |
| 只做线性 draft chain | DeltaNet 做树验证时，每个分叉都要单独一条 state 路径 |
| 8K 下不做层 offload | 修正后的显存够用；offload 留到扩展 context 时再做 |
| 精确测量按需进行 | 用规格书算出的 SOL 已经足够判断方向；什么时候需要精确测量见 §8 |

## 6. 里程碑

### 6.1 Baseline

| 里程碑 | Status | 内容 | 退出标准 | 粗估 |
|---|---|---|---|---|
| M1 实机参照 | ✅ 完成 2026-10-19 | 用 llama.cpp 跑 unsloth 的 `UD-IQ4_XS`（14.25 GB），用 `llama-bench` 测 depth 512 和 8192 两种情况 | 拿到 5080 实机的 decode tok/s；确认显存余量和显示器占用多少显存。详细清单见 `docs/plans/m1.md` | 半天 |
| M2 拼装 runner 跑通 | 未开始 | 用 transformers 里 qwen3_5 的文本模型做骨架；实现 NVFP4Linear（调用 FlashInfer `mm_fp4`）；embedding 放 CPU；只加载 `model.language_model.*` 和 `lm_head.*`，并列出所有保留高精度的 tensor（`in_proj_a/b`、`conv1d`、各个 norm 等）；先用 eager 模式跑 | NVFP4Linear 和"dequant + matmul"对拍通过；能生成连贯的文本 | 1–2 天 |
| M3 decode baseline | 未开始 | 改用 static cache，自己写 decode loop 并录成 CUDA Graph；按 §2 写 benchmark harness（多次运行，报告中位数和波动，记录峰值显存）；按真实的 tensor 布局重算每步要读的字节数，算上高精度例外和 KV/state 的访问；抓第一张 Nsight Systems 时间线 | 拿到 baseline tok/s 和时间分布。tok/s 要同时报两个数：不含 sampling 的模型执行速度，以及含 sampling 的有效输出速度 | 1–2 天 |

### 6.2 优化阶段

| Phase | Status | 内容 | 退出标准 |
|---|---|---|---|
| 2 AR 逼近 SOL | 未开始 | 按时间线上的热点逐个替换：NVFP4 GEMV（swap-AB + `mxf4nvf4`，并和 CUDA core 版本对照）、GDN decode、FP8 KV 的 decode attention、算子融合（norm + 激活量化，gate/up + SiLU·mul）、embedding 改成 device 端 gather、整个 step 录成一张 CUDA Graph | ≥ 50 tok/s，争取 ≥ 55（即 ≥ 85% SOL）；大 kernel 达到 85% 以上的带宽 |
| 3a spec decode：MTP | 未开始 | 从官方 BF16 checkpoint 取 MTP 权重（查 safetensors index，用 HTTP Range 只下载 `mtp.*`）并量化。实现 N=8 的 verify：小 N 的 W4A4 GEMM、N 个 query 的 attention、GDN verify 加 lazy commit。先做 greedy 接受，再做 rejection sampling | 回滚后 state 逐位正确；接受的 token 都等于 argmax；τ 和 card 对照（BF16 均值 4.28） |
| 3b spec decode：DFlash2 | 未开始 | 用 RTN 把 drafter 量化到 NVFP4；实现 5 层 block drafter + fc + dyn-conv + selector + codebook gather；实现 KV injection | τ 和 card 对照（BF16 均值 4.80）。如果量化后 τ 不到 BF16 的 75%，就停在 MTP |
| 4 spec 路径逼近 SOL | 未开始 | spec 循环全部在 device 端完成，不和 host 同步；整个 step 录成一张 graph；融合 verify；最后用 megakernel 收尾（persistent kernel、层间细粒度同步、权重预取） | 先达到 125–170 tok/s，再争取到 spec SOL（用实测 τ 计算）的 80–85% |
| 5 可选 | — | 更长的 context（offload：offload 层均匀分散 + ring buffer）、state 改 BF16、KV 改 INT8、block 改 16、截断词表、prefill 优化 | — |

## 7. 风险与开放问题

| 问题 | 什么时候能回答 | 兜底方案 |
|---|---|---|
| 量化后的 τ | Phase 3 | MTP |
| 实际可用显存（显示器占用多少） | M1 | 不接显示器；codebook 放 host |
| FlashInfer `mm_fp4` 在 M=1 时的效率 | M3 | 只影响 baseline 的数字，Phase 2 会替换掉 |
| transformers 的 hybrid cache 能不能直接录进 CUDA Graph | M3 | 自己写静态的 state 管理 |
| 小 N 的 W4A4 GEMM 能跑到多少带宽 | Phase 2–3 | N=8 时理论上离算力上限还很远 |

## 8. 按需测量

| 触发条件 | 测什么 | 工具 |
|---|---|---|
| 某个访存 kernel 卡在 960 GB/s 的 80–90% 上不去 | 纯读带宽的实际上限 | 写一个简单的 read kernel；或看 `ncu` 的 `dram__throughput` |
| 出现 OOM，或加上 drafter 后显存吃紧 | 实际可用显存 | `torch.cuda.mem_get_info()` |
| step 总时间对不上各 kernel 时间之和 | launch、同步、kernel 之间的空隙 | Nsight Systems |
| 开始做 offload | H2D 带宽 | 到时再测 |

## 9. 参考

- NVFP4 checkpoint：https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 。格式为 W4A4，lm_head 也是 NVFP4；embedding、vision、`in_proj_a/b`、`conv1d`、norm 保留 BF16
- DFlash2：https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2 ；https://inco.ai/blog/dflash2/
- GGUF（M1 用）：https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
- FlashInfer `mm_fp4`：https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html
