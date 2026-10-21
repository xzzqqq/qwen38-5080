# 2026-10-20 M2 §3：eager 文本 runner

## 目的

把 NVFP4 权重装进 transformers 的 Qwen3_5 文本骨架，跑通 chunked prefill +
greedy decode，验证三条真实 prompt 和 8K 容量。

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
| 512 | 1023 | 0.696 s | 42.238 s（12.10 tok/s） | 14.776 GB | 14.915 GB | 1.357 GB |
| 7680 | 8191 | 6.804 s | 42.538 s（12.01 tok/s） | **15.620 GB** | **15.859 GB** | **0.413 GB** |

8K 不 OOM，没有 NaN/Inf；最后一步 token 没有再次 forward，所以 cache_len 是 8191。
baseline 只剩约 0.41 GB 空闲，DFlash2 的容量会更紧。

## 开放项：chunked prefill 数值不一致

300-token 固定序列：

| 路径 | logits vs 单段 |
|---|---|
| 单段 prefill | 基准 |
| `chunk=128` | max_abs 11.75，cos **0.929**，argmax 271 vs 37550 |

定位：

- 显式 `position_ids` / `attention_mask` 不改变结果；
- mask 是 `(1,1,128,256)`，前 128 个 past 位置可见，causal 正确；
- 逐层比较从第 1 层（第一个 GDN 层）开始分叉：layer 1 max 0.125、cos 0.999996，
  之后逐层放大。

判断：不是 mask/position 逻辑错误，更像 **GDN chunk 边界 + FP4 GEMM 的 M 依赖累加**
带来的数值差异；但还没有用 state/逐层 reference 闭环。smoke prompt 都 <128，走单段，
不受影响；8K capacity 只验证不 OOM，不能当数值对拍。

## 结论

§3 的 loader、真实生成和 8K 容量完成；M2 的 runner 数值对拍仍有一个开放项。
下一步应先决定：

1. 用单段 prefill 做短 prompt 的数值 reference，chunked 只用于长上下文；
2. 还是把 GDN state/conv state 与单段结果逐层对拍，定位 chunk 边界的差异。

在解决前不把 M2 标完成。

## 产物

- `runner/loader/checkpoint.py`、`runner/loader/nvfp4_model.py`
- `runner/model/host_embedding.py`、`runner/model/eager_runner.py`
- `scripts/m2_smoke.py`、`scripts/m2_capacity.py`
- `results/m2/20261020T232439-smoke/`
- `results/m2/20261020T232552-capacity/`
- `results/m2/20261020T232643-capacity/`
