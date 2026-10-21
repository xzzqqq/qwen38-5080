# 2026-10-20 M2 §3：eager 文本 runner

## 目的

把 NVFP4 权重装进 transformers 的 Qwen3_5 文本骨架，跑通 chunked prefill +
greedy decode，验证三条真实 prompt 和 8K 容量。

最终 M2 关闭记录见 [2026-10-21 M2 关闭](2026-10-21-m2-close.md)。

## Loader

- `Qwen3_5ForCausalLM` 先在 **meta** 设备构造，不分配整份 BF16 权重；
- 按 manifest 的 `is_nvfp4_weight` 找到 401 个量化 Linear，替换成 `NVFP4Linear`；
- `embed_tokens` 换成 `HostEmbedding`（CPU BF16，2.543 GB），只传 gather 后的行；
- `conv1d`、`in_proj_a/b`、norm、`A_log`/`dt_bias` 按 checkpoint dtype 逐 tensor 加载；
- `rotary_emb` 在 meta 上没有值，重建到目标设备；
- 加载后检查没有 meta tensor 残留。

加载结果：

| 项 | 值 |
|---|---|
| 加载耗时 | 1.36 s（page cache 已热） |
| 量化 Linear | 401 |
| 高精度 tensor | 449 |
| GPU allocated | 14.457 GB |
| GPU reserved | 14.556 GB |
| `mem_get_info` free | 1.728 GB |

## 真实生成

命令：

```bash
.venv/bin/python scripts/m2_smoke.py --max-new-tokens 512
```

| prompt | prompt tokens | gen | prefill | decode | finish | 结果 |
|---|---|---|---|---|---|---|
| math | 90 | 87 | 0.383 s | 7.153 s（12.16 tok/s） | eos | 答案 **72** |
| code | 84 | 239 | 0.106 s | 19.792 s（12.08 tok/s） | eos | `is_palindrome` + 3 个 assert |
| chat | 72 | 298 | 0.106 s | 24.777 s（12.03 tok/s） | eos | 正好三句 |

速度受两个 reference fallback 影响，不是性能 baseline：

- `causal_conv1d` 未安装，`causal_conv1d_fn/update` 走 PyTorch fallback；
- `flash-linear-attention` 未安装，GDN 走 reference chunk/recurrent 实现。

## 8K 容量

命令：

```bash
.venv/bin/python scripts/m2_capacity.py --depths 512 --n-decode 512
.venv/bin/python scripts/m2_capacity.py --depths 7680 --n-decode 512
```

| depth | cache_len | prefill | decode 512 | peak allocated | reserved | free |
|---|---|---|---|---|---|---|
| 512 | 1023 | 0.717 s | 45.979 s（11.11 tok/s） | 14.776 GB | 14.915 GB | 1.357 GB |
| 7680 | 8191 | 6.743 s | 45.677 s（11.19 tok/s） | **15.620 GB** | **15.859 GB** | **0.413 GB** |

a/b shape-stable 之后复测，峰值和 free 不变；decode 约 45–46 s，属于 reference
fallback 的诊断波动。

8K 不 OOM，没有 NaN/Inf；最后一步 token 没有再次 forward，所以 cache_len 是 8191。
baseline 只剩约 0.41 GB 空闲，DFlash2 的容量会更紧。

## 开放项：chunked prefill 数值不一致

300-token 固定序列：

| 路径 | logits vs 单段 |
|---|---|
| 单段 prefill | 基准 |
| `chunk=128` | max_abs 11.75，cos **0.929**，argmax 271 vs 37550 |

定位与证据：

1. **前缀对比**：单段 300 的前 128 个 hidden states vs 单独运行 128 tokens，从
   layer 1 就开始不同（max 0.125，cos 0.999993）。所以最初误差发生在跨段 state
   交接之前，不需要先怀疑 cache/GDN state。
2. **layer 0 实际子步骤**（同一份前 128 输入，M=300 vs M=128）：
   `input_norm` 完全一致 → `linear_attn` max 0.00519 → residual 0.00525 →
   `post_norm` 0.01465 → **MLP 0.04688** → layer0 输出 **0.125**。
3. **linear_attn 内部**（输入是同一份 normed tensor）：
   - `in_proj_qkv` / `in_proj_z`：**完全一致**（NVFP4 投影在此 shape 下没有分叉）
   - `in_proj_b`：max **0.03125**，mean 0.00148
   - `in_proj_a`：max **0.06250**，mean 0.00298
   - conv / q / k / v：一致
   - g：max 0.01055；beta：0.00391
   - GDN core：0.01563
   - norm / out_proj：0.00391 / 0.00519

结论：**最早的分叉是 BF16 `in_proj_a/b` 的 shape 相关 GEMM 舍入**；qkv/z 的
NVFP4 路径在这个案例里反而完全一致。之后 a/b → g/beta → GDN core → norm →
MLP 的 W4A4 activation quantization 逐步放大到 0.125。没有证据表明 GDN state
接错，也不能把首因归给 FP4 GEMM。

### shape-stable a/b 修复后的结果

把 96 个 `in_proj_a/b` 换成 `ShapeStableLinear(block_m=128)`：按固定 128 行分块，
尾块补零，再走同一 shape 的 `F.linear`。

- `in_proj_a/b` 的 full vs split 现在 **max=0**；
- 前 128 tokens 的逐层 hidden states 与单独 128 tokens **完全一致**；
- 按 chunk 检查：
  - `0:128`：全 exact；
  - `128:256`：第一个分叉在 hidden[24] = **layer 23（full_attention）**，
    max 0.01904；
  - `256:300`：layer 11 的分叉是前一段传播，不是新的首因。
- layer 23 的输入 exact；`attn_out` max 0.00586，residual 0.01563，
  post_norm 0.00781，MLP 0.01904。

### 最终定位：不是 cache/state，是 attention KV 长度不同

关键对照：

| 对照 | 结果 |
|---|---|
| full **256** vs `128+128`，逐层 hidden | **max=0** |
| full **256** vs `128+128`，最终 logits | **max=0，cos=1.0，argmax 相同** |
| full **300** vs `128+128+44` | 第 2 段 attention 的 KV 长度是 300 vs 256；仍有差异 |

full 300 的第 2 段 attention 会看到 44 个被 causal mask 的 future keys；chunked
的第 2 段只看到 256 个 key。有效 keys 相同，但 attention 的 reduction/softmax
长度不同，FP 结果不会 bitwise 相等。这不是 cache 接错，也不是 GDN state 错误。

因此 `make m2-chunk` 用 **边界对齐** 的 full 前缀做 reference：

```bash
make m2-chunk   # full 256 vs 128+128
```

结果：`max_abs=0.0`、`cos=1.0`、argmax 相同。chunked prefill 路径本身是正确的；
之前拿 full 300 当 `128+128+44` 的 bitwise reference 不成立。

## 结论

§3 的 loader、真实生成和 8K 容量完成；chunked prefill 的数值差异已定位并收敛：

- a/b 的 shape 依赖已修复；
- 边界对齐的 full vs chunked **完全一致**；
- 任意长度的 full 300 vs chunked 128+128+44 不应当作 bitwise 门槛，原因是
  attention 的有效 KV 长度不同。

M2 其余验收（reference runner、完整逐层对拍口径）仍待完成；不再用“短 prompt
单段”绕过，也不再把 full 300 vs chunked 作为失败判据。

## 产物

- `runner/loader/checkpoint.py`、`runner/loader/nvfp4_model.py`
- `runner/model/host_embedding.py`、`runner/model/eager_runner.py`
- `runner/ops/bf16/stable_linear.py`
- `tests/model/test_stable_linear.py`
- `scripts/m2_smoke.py`、`scripts/m2_capacity.py`、`scripts/m2_chunk_consistency.py`
- `results/m2/20261020T232439-smoke/`
- `results/m2/20261020T232552-capacity/`
- `results/m2/20261020T232643-capacity/`
- `results/m2/20261021T111128-chunk-consistency/`
- `results/m2/20261021T111213-capacity/`
