# 2026-10-20 M2 1b：checkpoint 固定与 tensor manifest

## 目的

按固定 revision 拿到 NVFP4 checkpoint，确认文件完整；在不加载权重的前提下，
生成 tensor manifest 和初版显存账，为 M2 §2 的 NVFP4Linear 和 §3 的 runner 做准备。

## Checkpoint

| 项 | 值 |
|---|---|
| repo | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` |
| revision | `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33` |
| 本地目录 | `models/Qwen3.8-27B-NVFP4-RTX5090/` |
| shard 1 | 9,972,777,720 B，CRC32 `617cc98f` |
| shard 2 | 7,943,334,864 B，CRC32 `e1291b6b` |
| 总 data 字节 | 17,915,815,528 B = 17.916 GB |
| tensor 数 | 2387 |

下载命令：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$PWD/.tmp/hf"
export HF_XET_CACHE="$PWD/.tmp/hf/xet"
.venv/bin/hf download gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --revision 5b7a687fc8211a5d631c8ca6a593dd37eb26ce33 \
  --local-dir models/Qwen3.8-27B-NVFP4-RTX5090 \
  --include "model-*.safetensors" --max-workers 1
```

最终 `checkpoint_revision.txt` 里 `metadata_only: False`，requested 和 resolved
revision 都是同一个 commit。

## 遇到的问题

1. **HF Xet 缓存只读**：默认写 `~/.cache/huggingface`，沙箱里报
   `Read-only file system`。把 `HF_HOME` 和 `HF_XET_CACHE` 指到 `.tmp/hf`。
2. **Xet 下载卡住**：第一次后台下载停在 268 MB 的 `.incomplete`，进程消失。
   改用 `hf download` 并设置 `HF_HUB_DISABLE_XET=1` 后正常；最终由用户在前台执行。
3. **直连 HF 超时**：刷新 `checkpoint_revision.txt` 时直连 `model_info` 超时，
   设置 `HF_ENDPOINT=https://hf-mirror.com` 后成功。
4. **后台 job 子进程被回收**：pip 和第一次下载都出现“job 显示 running、实际进程消失”。
   大数据下载不要依赖后台 job，放前台或 tmux。

## 验收

- 两个 shard 的文件大小与 HF 文件树一致；
- 两个 CRC32 与 `crc32.txt` 完全一致；
- `model.safetensors.index.json` 的 2387 个 tensor 和两个 safetensors header
  双向一致，shard 归属一致；
- header 显示 `U8` / `F8_E4M3` / `F32` / `BF16` 四种存储 dtype。

CRC 校验命令和输出：

```text
model-00001-of-00002.safetensors bytes 9972777720 expected 617cc98f got 617cc98f OK
model-00002-of-00002.safetensors bytes 7943334864 expected e1291b6b got e1291b6b OK
```

## Tensor manifest

命令：

```bash
.venv/bin/python scripts/m2_manifest.py
```

产物在 `results/m2/20261020T175405-manifest/`：

- `tensor_manifest.json` / `.csv`
- `manifest_summary.json`
- `vram_account.json`
- `summary.txt`

关键数字：

| 项 | 值 |
|---|---|
| GPU 文本权重 | 14.452 GB（2053 个 tensor） |
| host embedding | 2.543 GB（1 个 tensor） |
| vision（skip） | 0.921 GB（333 个 tensor） |
| NVFP4 权重 | 401 个 tensor，25.60 B 逻辑参数，12.799 GB，4.00 bits/param |
| NVFP4 weight scale | 401 个 tensor，1.599 GB |
| 高精度权重 + norm + state | 0.052 GB |

分片：

- shard 1：979 个文本 + 333 个 vision；
- shard 2：1071 个文本 + `lm_head` 的 4 个 tensor。

所以 vision 不能靠“跳过整个 shard 1”来省下载；shard 1 仍要下，加载时按前缀过滤。

## 显存账

| 项 | GB |
|---|---|
| 可用显存 | 16.610 |
| 文本 GPU 权重 | 14.452 |
| KV BF16 @ 8K | 0.537 |
| GDN state + conv | 0.159 |
| workspace（占位） | 0.500 |
| CUDA context | 0.282 |
| **BF16 KV 总计** | **15.929**，余 **+0.681** |
| **FP8 KV 总计** | **15.661**，余 **+0.949** |

workspace 仍是占位值，必须用 M2 §4 实测的加载 / repack / prefill / decode 峰值替换。
当前结论：baseline 估算能装下，但余量只有约 0.68 GB，DFlash2 的容量仍待 Phase 3。

## 其他记录

- `model.safetensors.index.json` 的 metadata 写 `total_parameters: 15,193,246,960`，
  但按逻辑 shape 求和是 27.36 B。前者看起来是打包元素/另一口径；显存账只用字节，
  §2 再确认一次。
- `hf_quant_config.json` 的 `kv_cache_quant_algo: FP8` 只是 checkpoint 的建议；
  M2 按计划用 BF16 KV。

## 结论

1b 完成：checkpoint 已固定并校验，manifest 和初版显存账已生成。没有加载任何权重，
也没有测 GPU 峰值。

下一步 M2 §2：实现独立的 dequant reference 和 `NVFP4Linear`，先用实际 shape
对拍，再测 `mm_fp4` 的 M=1 / M=8 / M=128 可用性和耗时。

## 产物

- `models/Qwen3.8-27B-NVFP4-RTX5090/`（权重，不入库）
- `scripts/m2_fetch_checkpoint.py`
- `scripts/m2_manifest.py`
- `results/m2/20261020T175405-manifest/`
- `docs/model.md`
- 本文
