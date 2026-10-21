# 模型与 checkpoint

- 日期：2026-10-20
- 对应：M2 1b；原始记录见 `results/m2/20261020T175405-manifest/`

## 1. 模型结构

Qwen3.8-27B 文本模型：

| 项 | 值 |
|---|---|
| 层数 | 64：48 层 Gated DeltaNet（linear attention）+ 16 层 full attention |
| hidden | 5120 |
| FFN | 17408 |
| vocab | 248320 |
| full attention | GQA，24 query heads / 4 KV heads，head_dim 256 |
| linear attention | 16 key heads × 128，48 value heads × 128，conv kernel 4 |
| GDN state | FP32 |
| 视觉塔 | 27 层，hidden 1152；文本 runner 不加载 |

`config.json` 的 `layer_types` 是交错排列：每 4 层一个 full attention，其余是 linear attention。

## 2. Checkpoint 布局

| 项 | 值 |
|---|---|
| repo | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` |
| revision | `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33` |
| 本地目录 | `models/Qwen3.8-27B-NVFP4-RTX5090/`（`models/` 不入库） |
| 总大小 | 17.9158 GB（2 个 shard 的 data 字节之和） |
| tensor 数 | 2387 |
| shard | `model-00001-of-00002.safetensors` 9,972,777,720 B；`model-00002-of-00002.safetensors` 7,943,334,864 B |
| CRC32 | `617cc98f` / `e1291b6b`（均与 `crc32.txt` 匹配） |

分片内容：

| shard | tensor 数 | 内容 |
|---|---|---|
| `model-00001-of-00002` | 1312 | 979 个 `model.language_model.*` + 333 个 `model.visual.*` |
| `model-00002-of-00002` | 1075 | 1071 个 `model.language_model.*` + `lm_head` 的 4 个 tensor |

`model.safetensors.index.json` 记录每个 tensor 名到 shard 的映射，metadata 里的 `total_size` 是 17,915,815,528 B，和实际 data 字节一致。

**vision 和文本混在第一个 shard 里**，所以第一个 shard 仍要下载，只是加载时按 `model.visual.` 前缀跳过。第二个 shard 几乎全是文本权重和 `lm_head`。

## 3. 量化约定

`hf_quant_config.json` / `config.json` 的关键字段：

| 项 | 值 |
|---|---|
| `quant_algo` | `NVFP4` |
| 位宽 | 4 bit（E2M1） |
| group size | 16 |
| scale | UE4M3 / `F8_E4M3` |
| global scale | FP32 标量 |
| `kv_cache_quant_algo` | `FP8`（checkpoint 的建议；M2 仍用 BF16 KV） |
| `lm_head` | NVFP4 |
| 保留高精度 | vision、embedding、`conv1d`、`in_proj_a/b`、norm、`A_log`、`dt_bias` 等 |

文件里的实际存储形式：

| tensor | dtype / shape | 说明 |
|---|---|---|
| `...weight`（NVFP4） | `U8 [N, K/2]` | 两个 E2M1 打包进一个字节 |
| `...weight_scale` | `F8_E4M3 [N, K/16]` | 每 16 个逻辑元素一个 block scale |
| `...weight_scale_2` | `F32 []` | 全局 weight scale |
| `...input_scale` | `F32 []` | activation quant 的全局 scale |

因此逻辑 shape 要把 U8 权重的最后一维乘 2。M2 §2 已用独立 reference 验证：

- 低 nibble = 偶数下标，高 nibble = 奇数下标；
- `weight_scale` 在 checkpoint 是线性 `[N, K/16]`，送 `mm_fp4` 前要转成 128x4 swizzled；
- `weight_scale_2` 和 `input_scale` 都是**反量化乘数**；`alpha = input_scale × weight_scale_2`；
- activation 量化用 `1 / input_scale` 作为 global divisor；
- reference 量化与 FlashInfer `nvfp4_quantize` 的 code diff = 0，dequant 最大差约 `1.2e-7`。

`NVFP4Linear` 的 forward 是：`nvfp4_quantize` → `mm_fp4(backend="cutlass")` → BF16。
权重传 `weight_packed.T`，scale 传 swizzled 的转置；模块只保留 swizzled scale，
原始 linear scale 在构造后释放（否则全模型会多占约 1.6 GB）。实现见
`runner/ops/nvfp4/linear.py`。

## 4. Tensor manifest

生成方式：只读每个 safetensors 的 header（8 字节长度 + JSON header），不加载 tensor 数据；再用 `model.safetensors.index.json` 双向校验。

产物：

- `results/m2/20261020T175405-manifest/tensor_manifest.json`：2387 行，逐 tensor 记录 name / shard / dtype / shape / bytes / role / placement / logical params；
- `tensor_manifest.csv`：同内容的表格版；
- `manifest_summary.json`：汇总和显存账；
- `vram_account.json`：显存账单独一份；
- `summary.txt`：人读摘要。

### 分类汇总

| placement | tensor 数 | 字节 |
|---|---|---|
| GPU（文本 + lm_head） | 2053 | **14.452 GB** |
| host（embedding） | 1 | **2.543 GB** |
| skip（vision） | 333 | **0.921 GB** |

| role | tensor 数 | 字节 |
|---|---|---|
| NVFP4 文本权重 | 400 | 12.163 GB |
| NVFP4 文本 weight scale | 400 | 1.520 GB |
| `lm_head` NVFP4 权重 | 1 | 0.636 GB |
| `lm_head` weight scale | 1 | 0.079 GB |
| 高精度权重（含 `conv1d`/`in_proj_a/b`） | 144 | 0.051 GB |
| norm | 209 | 0.001 GB |
| `A_log` / `dt_bias` | 96 | ~0 |
| input/global scale | 801 | ~0 |

NVFP4 权重合计 401 个 tensor（400 文本 + `lm_head`），逻辑参数 25.60 B，存储 12.799 GB，**4.00 bits/param**（不含 scale）。这和 27B 的模型命名一致。

注意：`model.safetensors.index.json` 的 metadata 写 `total_parameters: 15,193,246,960`，但 manifest 按逻辑 shape 求和得到 **27.36 B**。前者看起来是打包后的元素计数或另一口径，**不能拿来做显存账**；显存账只用实际字节。M2 §2 没有改变这个结论。

## 5. 显存账（初版）

口径：GB = 10^9 字节；可用显存 15841 MiB = **16.610 GB**。

| 项 | 字节 / GB | 说明 |
|---|---|---|
| 文本 GPU 权重 | 14.452 | manifest 实测，含 scale / lm_head / 高精度 |
| KV BF16 @ 8K | 0.537 | 16 层 full attention，4 KV heads，head_dim 256 |
| KV FP8 @ 8K | 0.268 | 同上，1 byte |
| GDN recurrent state | 0.151 | 48 层 × 48 v_heads × 128 × 128 × FP32 |
| conv state | 0.008 | 48 层 × conv_dim 10240 × kernel 4 × FP32 |
| workspace | 0.500 | 初版占位；8K eager 实测见下 |
| CUDA context / allocator | 0.282 | preflight 中 torch 初始化后 free 减少 269.1 MiB |

估算：

| 组合 | 总占用 | 对 16.610 GB 的余量 |
|---|---|---|
| BF16 KV @ 8K | **15.929 GB** | **+0.681 GB** |
| FP8 KV @ 8K | **15.661 GB** | **+0.949 GB** |

结论：按当前估算，NVFP4 baseline 在 8K + BF16 KV 下**能装下**，但余量只有约 0.68 GB；workspace 初版占位；8K eager 实测见下。DFlash2 的 16.39–16.74 GB 估算仍未验证。

### M2 eager runner 实测（2026-10-20）

| 项 | 值 |
|---|---|
| 加载后（401 个 NVFP4Linear） | allocated 14.457 GB / reserved 14.556 GB |
| 8K（7680 prompt + 512 decode）peak | **allocated 15.620 GB / reserved 15.859 GB** |
| 8K 后 `mem_get_info()` free | **0.413 GB** |
| 7680 prefill | 6.59 s（eager；GDN/conv 走 reference fallback） |
| 512 decode | 44.01 s，11.61 tok/s（M2 关闭复测；诊断值，不是性能 baseline） |

这说明 baseline 能进 8K，但只剩约 0.4 GB 空闲；DFlash2 的容量会更紧。速度受 `causal_conv1d` / `flash-linear-attention` 未安装和 eager 路径影响，M2 不作为门槛。

## 6. 已知坑

- **分片混装**：vision 和文本都在 shard 1，不能因为“跳过 vision”就跳过整个文件；要下载 shard 1，按 tensor 前缀过滤加载。
- **embedding 不搬 GPU**：`model.language_model.embed_tokens.weight` 是 BF16，2.543 GB；M2 放 host，只传 gather 后的行。
- **NVFP4 的 nibble / scale layout**：已由 §2 验证。低 nibble 是偶数下标；`weight_scale` 线性布局要转 128x4 swizzled；scale 的 padding 行/列填 0。
- **`alpha`**：已确认 `input_scale × weight_scale_2`；`input_scale` 是反量化乘数，量化时取倒数。
- **FlashInfer JIT 头文件**：`cuda13` 的运行时头在 `targets/x86_64-linux/include`，`curand_kernel.h` 在 pip 的 `nvidia/cu13/include`；两者都要进 `CPATH`，并设 `LIBRARY_PATH`。用 `scripts/m2_env.sh` / `.fish` 或预检脚本自动配置。
- **KV 精度**：checkpoint 写 `kv_cache_quant_algo: FP8`，但 M2 明确用 BF16；两套要分开报 SOL。
- **HF 下载**：Xet 默认缓存写 `~/.cache`，沙箱只读；设置 `HF_HOME` / `HF_XET_CACHE` 到仓库 `.tmp`。直连超时时用 `HF_ENDPOINT=https://hf-mirror.com`。
- **metadata 参数数**：index 的 `total_parameters` 和逻辑 shape 不一致，不要混用。
- **chunked prefill 数值差异已解释**：96 个 BF16 `in_proj_a/b` 已换成
  `ShapeStableLinear(block_m=128)`，a/b full vs split max=0，前 128 hidden 也 exact。
  **边界对齐**的 full 256 vs `128+128` 逐层 hidden 和最终 logits 都 max=0、cos=1.0。
  full 300 vs `128+128+44` 仍不同，是因为第 2 段 attention 的 KV 长度 300 vs 256
  （full 多 44 个被 mask 的 future keys）；这是 attention reduction 长度差异，
  不是 cache/state 接错。正确 reference 用 `make m2-chunk`，不要拿任意 full 长度
  做 bitwise 基准。

## 7. 复现

```bash
# 1) 下载（固定 revision）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$PWD/.tmp/hf"
export HF_XET_CACHE="$PWD/.tmp/hf/xet"
.venv/bin/hf download gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --revision 5b7a687fc8211a5d631c8ca6a593dd37eb26ce33 \
  --local-dir models/Qwen3.8-27B-NVFP4-RTX5090 \
  --include "model-*.safetensors" --max-workers 1

# 2) manifest + 显存账（只读 header，不加载权重）
.venv/bin/python scripts/m2_manifest.py

# 3) 校验 CRC32
.venv/bin/python - <<'PY'
import zlib
from pathlib import Path
base = Path("models/Qwen3.8-27B-NVFP4-RTX5090")
for line in (base / "crc32.txt").read_text().splitlines():
    crc, name = line.split()
    h = 0
    with (base / name).open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h = zlib.crc32(chunk, h)
    print(name, f"{h & 0xffffffff:08x}", "OK" if f"{h & 0xffffffff:08x}" == crc else "MISMATCH")
PY

# 4) NVFP4Linear 对拍（需要 GPU；fish 用 source scripts/m2_env.fish）
source scripts/m2_env.sh
.venv/bin/python -m pytest tests/ops/test_nvfp4.py -q
.venv/bin/python bench/micro/nvfp4_linear.py
```
