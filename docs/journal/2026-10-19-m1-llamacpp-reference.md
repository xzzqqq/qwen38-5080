# 2026-10-19 M1：llama.cpp 实机参照

## 目的

在 5080 上用现成引擎拿一组参照数：单请求 AR decode 速度、显存占用、context 长度
对速度的影响；同时验证环境（驱动、CUDA、显存余量、桌面占用）和真实生成能力。

注意：本次用 `UD-IQ4_XS`（4.25 bpw），不是 NVFP4，**数字不作为本项目 baseline**。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 5080，16303 MiB，compute 12.0 |
| 空闲显存 | 0 MiB，`display_active=Disabled`（桌面没有占用 5080） |
| 驱动 | 580.105.08（open kernel module），用户态库同版本 |
| CUDA | 13.0.88（micromamba env `cuda13`，nvcc + cudart + cublas 13.0.2.14） |
| 编译器 | gcc 11.3.0；cmake 3.22.1 |
| CPU / RAM | i9-13900K / 125 GiB |

沙箱说明：DSH 的 bash 在 bwrap 里运行，`--dev /dev` 只提供最小设备，看不到
`/dev/nvidiactl`、`/dev/nvidia0`、`/dev/nvidia-uvm`。**所有 GPU 命令都要在
full access 模式下运行。**

## 模型

见 `results/model_info.txt`：`unsloth/Qwen3.8-27B-GGUF` 的
`Qwen3.8-27B-UD-IQ4_XS.gguf`，14.25 GB，SHA-256
`40fac405…99`，与 HF LFS 一致。

## llama.cpp

- commit：`f174553e1621c6a22a64e7bfcbb58d3a5c49e6fb`（2026-10-19）
- 编译：`GGML_CUDA=ON`，`CMAKE_CUDA_ARCHITECTURES=120a-real`，CUDA 来自 `cuda13` env
- 复现：`scripts/setup_llama_cpp.sh`（不入库，固定 commit 拉取）
- 配置记录：`results/llama_cpp_build.txt`、`results/llama_bench_help.txt`

## 命令

```bash
scripts/run_ar_bench.sh      # depth 512 / 8192，各 5 次，同时 1 Hz 采样 GPU
scripts/run_server_check.sh  # math / code / chat 三条真实 prompt
```

口径：`-ngl 999 -fa on -ctk f16 -ctv f16 -p 0 -n 512 -r 5`，默认
`n_batch=2048`、`n_ubatch=512`、`n_threads=8`。

## 结果：AR decode

| depth | 5 个样本 (tok/s) | 均值 ± sd |
|---|---|---|
| 512 | 55.898 / 55.922 / 55.905 / 55.883 / 55.883 | **55.90 ± 0.017** |
| 8192 | 54.168 / 54.195 / 54.194 / 54.191 / 54.189 | **54.19 ± 0.011** |

8K 比 512 慢 **3.1%**。两次都跑通，没有 OOM，没有降 KV 精度。

### GPU 状态（1 Hz 采样，`results/gpu_monitor.csv`，覆盖 bench 与 server 两段）

| 项 | 值 |
|---|---|
| `memory.used` 峰值 | **14275 MiB**（bench）/ **14313 MiB**（server prefill），约 2.0 GB 余量 |
| 功耗 | 中位 298 W，峰值 302 W（limit 360 W） |
| SM 时钟 | 中位 2812 MHz，峰值 2880 MHz |
| 显存时钟 | 15001 MHz（满速） |
| 利用率 | 100% |

没有被功耗或时钟节流。util 100% 而功耗只有 83% limit，是带宽受限的表现。

## buffer 与有效带宽

tensor 放置：`offloaded 66/66 layers to GPU`，output layer 也在 GPU，没有层在
CPU 上计算。

| buffer | depth 512（n_ctx 1024） | depth 8192（n_ctx 8704） |
|---|---|---|
| `CUDA0 model` | 12726.35 MiB | 12726.35 MiB |
| `CPU_Mapped model` | 521.00 MiB | 521.00 MiB |
| `CUDA0 KV` | 64.00 MiB | 544.00 MiB |
| `CUDA0 RS`（GDN state） | 149.62 MiB | 149.62 MiB |
| `CUDA0 compute` | 526.02 MiB | 533.52 MiB |

每 token 读字节数（只看 GPU 侧）：

- depth 512：13.35（权重）+ 0.067（KV）+ 0.314（RS 读写）≈ **13.73 GB**
  → 55.90 tok/s × 13.73 GB ≈ **767 GB/s ≈ 80% of 960**
- depth 8192：13.35 + 0.570 + 0.314 ≈ **14.23 GB**
  → 54.19 tok/s × 14.23 GB ≈ **771 GB/s ≈ 80% of 960**

两个 depth 的带宽利用率几乎相同，8K 的减速比例也和 KV 多出的 0.5 GB 吻合，
**可以确认 decode 是带宽受限**。

## 结果：真实生成（server）

`llama-server --jinja -c 8192`，greedy（temperature=0），max_tokens=2048。

| prompt | prompt tokens | 生成 tokens | prompt eval (TTFT) | eval tok/s | 结果 |
|---|---|---|---|---|---|
| math | 90 | 192 | 208.5 ms（431.6 tok/s） | 54.99 | 答案 72 ✅ |
| code | 42 | 179 | 149.0 ms（281.9 tok/s） | 55.11 | 函数 + 3 个测试正确 ✅ |
| chat | 30 | 279 | 144.6 ms（207.4 tok/s） | 55.13 | 正好三句 ✅ |

- 三条都 `finish_reason=stop`，thinking 正常结束，没有跑满 max_tokens。
- `reasoning_content` 正常返回，thinking token 计入 `completion_tokens`。
- `--jinja` 使用模型自带模板；日志显示 reasoning preserving 默认开启。
- server 的 eval 速度比 bench 低约 1.6%（含 sampling/HTTP 开销，context 也更长）。

## 结论

1. **参照数**：5080 上 llama.cpp + IQ4_XS 的 AR decode 是 55.9 tok/s（depth 512）
   和 54.2 tok/s（depth 8K）。
2. **比计划里"现有推理栈折算 46–50 tok/s"高 12–21%**，说明 5080 的 decode 上限
   比之前的折算更乐观。
3. **带宽利用率约 80%**：这给 M2/M3 一个可执行的标尺——自写 runner 如果低于
   80% 带宽，就还有优化空间。
4. **显存**：8K + f16 KV 峰值 14275 MiB，余约 2.0 GB。NVFP4 + drafter 预计
   16.4–16.7 GB，余量确实会非常紧张。
5. 真实生成正常：模板、thinking、停止、math 答案都对。

## 偏差与已知问题

- IQ4_XS 是 4.25 bpw、GPU 模型 buffer 13.35 GB，和 NVFP4 的 14.41 GB 不同；
  55.9 tok/s 不能和 AR SOL 64.1 直接比，只能比带宽利用率。
- llama-bench 用随机合成 token，不经过 tokenization / sampling；每个 token 后
  `llama_synchronize`，没有跨 token 的 CUDA Graph。
- `-d` 的 prefill 不计时；生成过程中 context 继续增长（8K 组实际在 8192→8704 上跑）。
- GPU 状态 1 秒采样一次，可能漏掉瞬时峰值，14275 MiB 不是严格峰值。
- `model_size` 14.24 GB 与 buffer 总和 13.89 GB 差约 0.35 GB，可能是对齐/元数据；
  带宽计算只用 buffer 口径。
- 可选步骤 7（IQ3_XXS 验证带宽受限）和 8（MTP）没做：步骤 7 的结论已由
  depth 512/8192 的对比间接支持；步骤 8 显存太紧。

## 后续

进 M2：transformers 骨架 + FlashInfer NVFP4 GEMM，先跑通 eager。
