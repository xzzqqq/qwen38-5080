"""把 NVFP4 checkpoint 装进 transformers 的 Qwen3_5 文本骨架。

设计：
- 先在 meta 设备构造 Qwen3_5ForCausalLM，避免分配整份 BF16 权重；
- 按 manifest 的 ``is_nvfp4_weight`` 找到 401 个量化 Linear，换成 NVFP4Linear；
- embedding 换 HostEmbedding（CPU BF16），其余 norm/conv1d/in_proj_a/b/A_log/dt_bias
  按 checkpoint dtype 逐 tensor 加载到 GPU；
- rotary_emb 在 meta 上没有值，重新构造到目标设备；
- 加载后检查没有 meta tensor 残留。

数据流：checkpoint tensor -> module buffer/parameter -> GPU（embedding 除外）。
预期瓶颈：加载阶段是 14.45 GB 权重的磁盘 + H2D 拷贝；不是稳态 decode 瓶颈。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

from runner.env import configure_flashinfer_env

configure_flashinfer_env()

from runner.model.gdn_patch import install_from_env as install_gdn_decode  # noqa: E402
from runner.model.host_embedding import HostEmbedding  # noqa: E402
from runner.model.norm_quant_patch import install_from_env as install_norm_quant  # noqa: E402
from runner.ops.attention import install as install_compact_attention  # noqa: E402
from runner.ops.bf16.stable_linear import ShapeStableLinear  # noqa: E402
from runner.ops.nvfp4.linear import NVFP4Linear, decode_backend_info  # noqa: E402

from .checkpoint import CheckpointReader

TEXT_PREFIX = "model.language_model."
LM_HEAD_PREFIX = "lm_head."
SHAPE_STABLE_BLOCK = 128


def find_manifest(root: Path) -> Path:
    candidates = sorted((root / "results/m2").glob("*-manifest/tensor_manifest.json"))
    if not candidates:
        raise FileNotFoundError("no results/m2/*-manifest/tensor_manifest.json")
    return candidates[-1]


def map_checkpoint_name(name: str) -> str | None:
    """checkpoint tensor 名 -> Qwen3_5ForCausalLM 的 module/parameter 名。"""
    if name.startswith("model.visual."):
        return None
    if name.startswith(TEXT_PREFIX):
        # Qwen3_5ForCausalLM.model 就是文本 Qwen3_5TextModel。
        return "model." + name[len(TEXT_PREFIX) :]
    if name.startswith(LM_HEAD_PREFIX):
        return "model.lm_head." + name[len(LM_HEAD_PREFIX) :]
    return None


def _set_module(model: nn.Module, module_path: str, module: nn.Module) -> None:
    # model.lm_head 是 Qwen3_5ForCausalLM 的直接属性，不在 TextModel 下。
    if module_path == "model.lm_head":
        model.lm_head = module
        return
    parent_path, _, attr = module_path.rpartition(".")
    parent = model.get_submodule(parent_path)
    setattr(parent, attr, module)


def load_nvfp4_text_model(
    model_dir: Path,
    *,
    manifest_path: Path | None = None,
    device: str = "cuda",
) -> tuple[Qwen3_5ForCausalLM, Qwen3_5TextConfig, dict[str, Any]]:
    root = Path.cwd()
    manifest_path = manifest_path or find_manifest(root)
    manifest = json.loads(Path(manifest_path).read_text())
    config_json = json.loads((Path(model_dir) / "config.json").read_text())
    text_config = Qwen3_5TextConfig(**config_json["text_config"])
    text_config._attn_implementation = "eager"
    text_config.use_cache = True

    t0 = time.time()
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(text_config)
    model.config._attn_implementation = "eager"
    model.config.use_cache = True
    # Phase 2 任务 A：q_len==1 的 decode attention 换成按 KV head 分组的紧凑实现
    # （不物化 repeat_kv）；prefill 由补丁内部回退到原 eager。四个执行方式
    # （eager / static-exact / static-cap / graph）因此仍共用同一条 decode 路径，
    # 同 shape bitwise gate 保持可比。
    attention_info = install_compact_attention()

    reader = CheckpointReader(Path(model_dir))
    handled: set[str] = set()
    quantized = 0

    # 1) 量化 Linear：manifest 里 is_nvfp4_weight 的 401 个。
    quant_rows = [
        row
        for row in manifest
        if row["is_nvfp4_weight"]
        and (row["name"].startswith(TEXT_PREFIX) or row["name"].startswith(LM_HEAD_PREFIX))
    ]
    for row in quant_rows:
        ckpt_name = row["name"]
        if not ckpt_name.endswith(".weight"):
            raise ValueError(f"unexpected quantized tensor: {ckpt_name}")
        prefix = ckpt_name[: -len(".weight")]
        model_path = (
            "model.lm_head" if prefix == "lm_head" else map_checkpoint_name(prefix)
        )
        if model_path is None:
            raise ValueError(f"cannot map quantized module: {prefix}")
        weight = reader.get(prefix + ".weight", "cpu").to(device)
        scale = reader.get(prefix + ".weight_scale", "cpu").to(device)
        scale2 = reader.get(prefix + ".weight_scale_2", "cpu").to(device)
        input_scale = reader.get(prefix + ".input_scale", "cpu").to(device)
        module = NVFP4Linear(
            weight,
            scale,
            scale2,
            input_scale,
            backend="cutlass",
        )
        _set_module(model, model_path, module)
        handled.update(
            {
                prefix + ".weight",
                prefix + ".weight_scale",
                prefix + ".weight_scale_2",
                prefix + ".input_scale",
            }
        )
        quantized += 1

    # 2) embedding 留 host，只把 gather 结果传 GPU。
    embed_name = TEXT_PREFIX + "embed_tokens.weight"
    embed = reader.get(embed_name, "cpu")
    _set_module(
        model,
        "model.embed_tokens",
        HostEmbedding(embed, torch.device(device)),
    )
    handled.add(embed_name)

    # 3) rotary_emb 的 inv_freq 不在 checkpoint 里，重建到目标设备。
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(text_config).to(device)

    # 4) 其余高精度 tensor：norm / conv1d / in_proj_a,b / A_log / dt_bias。
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    loaded_high_precision = 0
    skipped_unused = 0
    for ckpt_name in reader.weight_map:
        if ckpt_name in handled or ckpt_name.startswith("model.visual."):
            continue
        model_name = map_checkpoint_name(ckpt_name)
        if model_name is None or (model_name not in params and model_name not in buffers):
            skipped_unused += 1
            continue
        tensor = reader.get(ckpt_name, "cpu").to(device)
        parent_path, _, attr = model_name.rpartition(".")
        parent = model.get_submodule(parent_path)
        if model_name in params:
            setattr(parent, attr, nn.Parameter(tensor, requires_grad=False))
        else:
            setattr(parent, attr, tensor)
        loaded_high_precision += 1

    # 5) a/b 决定 GDN 的 gate；固定 block_m 让单段/分段 prefill 走同一 GEMM shape。
    wrapped_stable = 0
    for layer in model.model.layers:
        if not hasattr(layer, "linear_attn"):
            continue
        for name in ("in_proj_a", "in_proj_b"):
            linear = getattr(layer.linear_attn, name)
            setattr(
                layer.linear_attn,
                name,
                ShapeStableLinear.from_linear(linear, block_m=SHAPE_STABLE_BLOCK),
            )
            wrapped_stable += 1
    reader.close()

    meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    meta_buffers = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta_params or meta_buffers:
        raise RuntimeError(
            f"meta tensors remain: params={meta_params[:5]}, buffers={meta_buffers[:5]}"
        )

    model.eval()
    torch.set_grad_enabled(False)
    # Phase 2 任务 D：GDN decode 的 recurrent step 走融合 kernel（GDN_FUSED_DECODE=0 可关）。
    # 必须在 graph capture 之前接入：每个 state buffer 的 bf16 out 槽在首个 decode
    # step（warmup）里分配，之后地址固定，replay 依赖这一点。
    gdn_decode_info = install_gdn_decode(model)
    # Phase 2 任务 H：decoder layer 的 RMSNorm + activation quant 融合（只走单 token
    # decode；prefill 由 patch 内部回退）。scratch 缓冲在这里一次性分配，地址固定，
    # 之后的 graph capture/replay 依赖它（NORM_QUANT_FUSED=0 可关）。
    norm_quant_info = install_norm_quant(model)
    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info()
    info = {
        "load_s": time.time() - t0,
        "manifest": str(manifest_path),
        "quantized_linears": quantized,
        "high_precision_tensors": loaded_high_precision,
        "shape_stable_linears": wrapped_stable,
        "skipped_unused": skipped_unused,
        "gpu_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "gpu_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "gpu_free_gb": free_b / 1e9,
        "gpu_total_gb": total_b / 1e9,
        "decode_attention": attention_info,
        "gdn_decode": gdn_decode_info,
        # Phase 2 任务 F：M=1 decode 的 (N,K)->backend 静态表与依据。counters 是
        # 同一个 dict 的引用，产物最后序列化时能看到整个运行期间的调用计数。
        "nvfp4_decode": decode_backend_info(),
        # Phase 2 任务 H：norm+quant 融合的接线计数（fused_norms / fallbacks）。
        "norm_quant": norm_quant_info,
    }
    return model, text_config, info
