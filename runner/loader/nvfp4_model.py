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
from runner.ops.nvfp4.linear import (  # noqa: E402
    NVFP4Linear,
    build_decode_pair_from_checkpoint,
    decode_backend_info,
    decode_pair_specs,
)

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

    def _quant_parts(prefix: str) -> dict[str, torch.Tensor]:
        """一个量化模块的 4 个 checkpoint 张量（留在 CPU，交给配对/装载逻辑）。"""
        return {
            name: reader.get(f"{prefix}.{name}", "cpu")
            for name in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        }

    # 1a) M=1 合并调用（DecodePair）：**先**在 CPU 上把两半拼好、一次 H2D，再把合并
    #     buffer 的切片交给两个模块。顺序很重要：如果先分别装载再 ``torch.cat``，
    #     峰值是「两半 + 合并」= 2 倍，而 15.47 GiB 卡装载完 14.4 GB 权重后只剩
    #     ~1.7 GiB，且两半释放的 44.5 MiB segment 拼不出 89 MiB 的新分配 ——
    #     job-000055 的 eager 就是在 load 阶段这样 OOM 的（op 级不建整模型所以全绿）。
    #     同时注意：这里不能持有 ``dict(model.named_buffers())`` 之类的引用，否则
    #     旧 buffer 不会释放，合并又变回 2 倍峰值。
    pair_specs = decode_pair_specs(
        row["name"][: -len(".weight")] for row in quant_rows
    )
    decode_pairs: dict[str, Any] = {
        "pairs": 0,
        "skipped": 0,
        "kinds": {},
        "reasons": {},
        "strategy": "load-time merge (CPU cat -> one H2D, zero extra peak)",
    }
    paired_prefixes: set[str] = set()

    def _pair_skip(reason: str) -> None:
        decode_pairs["skipped"] += 1
        reasons = decode_pairs["reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    for leader_prefix, follower_prefix in pair_specs.items():
        leader_path = map_checkpoint_name(leader_prefix)
        follower_path = map_checkpoint_name(follower_prefix)
        if leader_path is None or follower_path is None:
            _pair_skip("unmapped")
            continue
        built = build_decode_pair_from_checkpoint(
            _quant_parts(leader_prefix),
            _quant_parts(follower_prefix),
            device=device,
            backend="cutlass",
        )
        if built is None:
            # 前置条件不满足（scalar 不一致 / N 不是 128 的倍数 / dtype …）：
            # 两半照常分别装载，M=1 走各自的共享路径，不影响 prefill 与数值语义。
            _pair_skip("precondition")
            continue
        leader_module, follower_module = built
        _set_module(model, leader_path, leader_module)
        _set_module(model, follower_path, follower_module)
        for prefix in (leader_prefix, follower_prefix):
            handled.update(
                {
                    prefix + ".weight",
                    prefix + ".weight_scale",
                    prefix + ".weight_scale_2",
                    prefix + ".input_scale",
                }
            )
            paired_prefixes.add(prefix)
        quantized += 2
        if leader_module._decode_pair is None:
            _pair_skip("attach")
            continue
        decode_pairs["pairs"] += 1
        key = f"{leader_prefix.rpartition('.')[2]}+{follower_prefix.rpartition('.')[2]}"
        decode_pairs["kinds"][key] = decode_pairs["kinds"].get(key, 0) + 1

    for row in quant_rows:
        ckpt_name = row["name"]
        if not ckpt_name.endswith(".weight"):
            raise ValueError(f"unexpected quantized tensor: {ckpt_name}")
        prefix = ckpt_name[: -len(".weight")]
        if prefix in paired_prefixes:
            continue
        model_path = (
            "model.lm_head" if prefix == "lm_head" else map_checkpoint_name(prefix)
        )
        if model_path is None:
            raise ValueError(f"cannot map quantized module: {prefix}")
        parts = _quant_parts(prefix)
        module = NVFP4Linear(
            parts["weight"].to(device),
            parts["weight_scale"].to(device),
            parts["weight_scale_2"].to(device),
            parts["input_scale"].to(device),
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
    # 这两个 dict 只为上面的查找而建；显式释放，避免把「已经被替换掉的旧 buffer」
    # 或后续 install 期间产生的临时引用留到 load 结束（job-000055 的 2 倍峰值教训）。
    del params, buffers

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

    # 6) M=1 合并调用（DecodePair）已经在 1a) 里随装载完成：k/v、gate/up、
    #    qkv/z 各自共享一块合并 buffer 与一个 persistent out。这里只做校验，
    #    保证「模型建好之后、graph capture 之前」配对确实生效（capture/replay
    #    依赖 pair.out 地址固定）。``install_decode_pairs`` 仍留给「模型已建好」
    #    的调用方，但它的 torch.cat 兜底路径会多一份峰值，生产 loader 不再使用。
    _modules_without_pair = 0
    for layer in model.model.layers:
        for holder_name, leader_name, follower_name in (
            ("self_attn", "k_proj", "v_proj"),
            ("mlp", "gate_proj", "up_proj"),
            ("linear_attn", "in_proj_qkv", "in_proj_z"),
        ):
            holder = getattr(layer, holder_name, None)
            if holder is None:
                continue
            for name in (leader_name, follower_name):
                module = getattr(holder, name, None)
                if module is not None and module._decode_pair is None:
                    _modules_without_pair += 1
    decode_pairs["modules_without_pair"] = _modules_without_pair

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
        # 第 4 轮：M=1 的合并调用（DecodePair）配对计数。
        "nvfp4_decode_pairs": decode_pairs,
        # Phase 2 任务 H：norm+quant 融合的接线计数（fused_norms / fallbacks）。
        "norm_quant": norm_quant_info,
    }
    return model, text_config, info
