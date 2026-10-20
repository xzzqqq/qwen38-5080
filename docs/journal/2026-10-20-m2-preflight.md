# 2026-10-20 M2 1a：环境与后端预检

## 目的

在下载 NVFP4 checkpoint 之前，确认 M2 的 eager 栈能在 5080 上真正跑起来：
torch CUDA、nvcc、FlashInfer 编译架构、`qwen3_5` 模型定义和 `mm_fp4` 符号。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 5080，16303 MiB，capability `(12, 0)` |
| 驱动 | 580.105.08 |
| Python | 3.12.15（仓库内 `.venv`） |
| torch | 2.14.1+cu130，`torch.version.cuda == "13.0"` |
| transformers | 5.20.2，含 `qwen3_5` |
| FlashInfer | 0.7.2（仅 core；未装 cubin/jit-cache） |
| nvcc | 13.0.88，来自 `~/micromamba/envs/cuda13` |
| 安装源 | 清华 PyPI 镜像（torch 的 Linux wheel 已是 CUDA 13 构建） |

## 做了什么

1. 建项目内 `.venv`（Python 3.12），不碰系统 Python。
2. 安装 torch / transformers / safetensors / huggingface-hub / numpy / pytest /
   flashinfer-python，版本锁定到 `requirements.txt`。
3. 新增 `scripts/m2_preflight.py`，记录版本、GPU、nvcc、FlashInfer 架构和
   `mm_fp4` 签名，结果写 `results/m2/<run-id>/`。
4. 修了两个环境坑：
   - FlashInfer 用 `which nvcc` 找编译器；系统 nvcc 是 12.4，会把 SM 120 判成
     “requires CUDA >= 12.8”。预检脚本现在自动优先 `CUDA_HOME` 指向 cuda13。
   - FlashInfer 默认把 JIT cache 写到 `~/.cache/flashinfer`；现在通过
     `FLASHINFER_WORKSPACE_BASE` 放到仓库 `.tmp/flashinfer`，避免权限和污染。

## 结果

命令：

```bash
.venv/bin/python scripts/m2_preflight.py
```

9 项全部 PASS：

- `torch_import`
- `cuda_available`
- `capability_12_0`
- `eager_alloc`
- `nvcc_ge_12_8`
- `qwen3_5_present`
- `flashinfer_import`
- `mm_fp4_present`
- `flashinfer_arch_12`

关键输出：

```text
torch: 2.14.1+cu130 cuda=13.0
gpu: NVIDIA GeForce RTX 5080 capability=[12, 0] free=15571.9 MiB total=15840.1 MiB
nvcc: 13.0 (/home/xzq/micromamba/envs/cuda13/bin/nvcc)
flashinfer archs: ["(12, '0f')"]
mm_fp4 backend 选项: cudnn / trtllm / cutlass / cute-dsl /
  cutedsl_low_latency / b12x / cake / auto
```

原始记录：`results/m2/20261020T165501/`。

## 结论

环境层就绪，可以进入 1b：固定 checkpoint revision、建 tensor manifest、做初版显存账。

**没有做**：
- 没有调用真实 `mm_fp4` GEMM（只验证了符号和编译架构）；实际 backend / M=1 调用在
  §2 NVFP4Linear 对拍时做。
- 没有下载 checkpoint，也没有加载任何权重。
- 没有测 KV/state 的显存。

## 复现

```bash
# 预检脚本会自动选 cuda13 和仓库内 FlashInfer cache
.venv/bin/python scripts/m2_preflight.py
```

依赖见根目录 `requirements.txt`。当前代码 commit `9d70e2f`，工作区有未提交的
roadmap/M2 文档改动。
