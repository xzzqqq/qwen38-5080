"""Phase 2 任务 J：FP8 KV 的编码契约、参考实现和反量化热路径。

## 冻结方案（本轮起点，实现了 Coordinator 的冻结口径）

- 存储 dtype 是 ``torch.float8_e4m3fn``，K 与 V 都存 FP8；
- scale 是 per (layer, kv_head, token) 的 **float32 标量**，K/V 各一套；布局
  ``[capacity, kv_heads]``（行是 token、列是 kv head，行内连续）；
- 有效长度之外的槽位写 0（scale 也写 0），mask 语义不变。

## 数值契约（实现之前冻结，op gate 用 fp64 精确参考逐位验证）

量化（全部 float32 算术）::

    amax  = max_i |x_i|                     # 该 token 的 head_dim=256 个 dim
    scale = amax / 448                      # amax == 0 时 scale 写成 0
    q     = rne_e4m3(clamp(x / scale))      # scale == 0 时 q = 0

反量化（fp32 乘一次，再 cast bf16）::

    x = bf16(float32(q) * scale)

为什么反量化契约是“fp32 乘 + cast bf16”而不是 fp64 精确乘后 cast：decode 热路径
必须在一次 elementwise pass 里完成（见 ``_triton_dequant.py``），fp64 乘不进热路径。
这个契约本身仍然可被精确验证：q 只有 4 个有效位、scale 有 24 位，乘积最多 27 位，
fp64 可以精确表示，所以 ``float32(q64 * s64)`` 就是“fp32 乘法的正确舍入”；
``dequantize_exact`` 用它做 op gate，candidate 的 fp32 中间值必须与它 bitwise 相同，
之后的 bf16 cast 是确定性的。

## fp8 张量只存储/搬运/转换，不做算术

CPU 上 ``index_copy_`` / ``clamp`` / ``isfinite`` / ``equal`` 都没有 fp8 kernel，
GPU 上不假设它们有：cache 的写入、清零、拼接、pad 一律走 ``view(torch.uint8)``；
有限性检查先 ``.float()`` 再查。

## 数据流（decode 单层）

::

    cache.keys [1,4,L,256] fp8 + k_scale [L,4] fp32
      -> 任务 K 的 fused decode attention：K 的 scale 乘在 QK 点积后、V 的乘在 p 上
         （零 bf16 物化；``runner/ops/attention/decode_fused.py``）
      -> 只有 grouped_bmm 回退才 dequantize()：一次 pass 读 fp8、乘 scale、写
         bf16 固定 scratch（这一步计入 ``dequant_fallback``）

## 预期瓶颈

fused 路径的瓶颈是 fp8 KV 读带宽：每层读 ``2*4*L*256`` 字节，是 BF16 的一半，
且没有 scratch 写。``dequantize`` 的纯 torch 实现会拆成 2–3 个 elementwise
kernel（写出流量反而高于 BF16），所以它只用于回退与 prefill；per-call 时间、
有效带宽和回退代价由 ``scripts/phase2_fp8kv_check.py`` 实测。
"""

from __future__ import annotations

import os
import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
SCALE_DTYPE = torch.float32
# scale 的 side channel：cache 的 update 返回的 fp8 张量上挂 [L, H] 的 fp32 scale。
# 融合 decode kernel 可以直接消费 (K, V, scale)，不需要每步反量化整段前缀。
SCALE_ATTR = "kv_fp8_scale"

FORMAT_ENV = "PHASE2_KV_FORMAT"
FORMATS = ("fp8", "bf16")
DEFAULT_FORMAT = "fp8"
DEQUANT_ENV = "PHASE2_FP8_DEQUANT"
DEQUANT_BACKENDS = ("triton", "torch")
HEAD_DIM = 256

_TINY = torch.finfo(torch.float32).tiny

_TRITON = {
    "module": None,
    "state": "unknown",  # unknown / ready / failed / disabled
    "error": None,
    "calls": 0,
    "fallbacks": 0,
}
_SCRATCH: dict[str, torch.Tensor] = {}


class _LayoutUnsupported(Exception):
    """当前张量布局不在 Triton kernel 的覆盖范围内：静默回退 torch。"""


# --------------------------------------------------------------------- format
def kv_format(default: str = DEFAULT_FORMAT) -> str:
    """当前配置声明的 KV 格式（``PHASE2_KV_FORMAT``，默认 ``fp8``）。

    默认值在本分支是 fp8：plan §7 的验收配置就是 FP8 KV；BF16 对照通过
    ``PHASE2_KV_FORMAT=bf16`` 复现（四个执行方式仍是原来的 DynamicCache 语义）。
    """
    fmt = os.environ.get(FORMAT_ENV, default)
    if fmt not in FORMATS:
        raise ValueError(f"{FORMAT_ENV}={fmt!r} 必须是 {FORMATS} 之一")
    return fmt


def resolve_format(fmt: str | None) -> str:
    return kv_format() if fmt is None else kfmt(fmt)


def kfmt(fmt: str) -> str:
    if fmt not in FORMATS:
        raise ValueError(f"unknown KV format {fmt!r}; choose from {FORMATS}")
    return fmt


# ------------------------------------------------------------------- quantize
def _scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """amax [B,H,T] -> scale [B,H,T]（fp32，amax==0 时为 0）。"""
    scale = amax / FP8_MAX
    return torch.where(amax > 0, scale, torch.zeros_like(scale))


def quantize_with_scale(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """低层契约：用给定 scale ``[T, H]``（> 0）量化 ``x [1, H, T, D]``。

    只用于覆盖 clamp/饱和路径的 op gate 和将来需要外部 scale 的 kernel；正常写入
    走 ``quantize_kv``（scale 由该 token 的 amax 推导）。
    """
    if x.dim() != 4 or x.shape[0] != 1:
        raise ValueError(f"x 必须是 [1, H, T, D]，得到 {tuple(x.shape)}")
    if tuple(scale.shape) != (x.shape[2], x.shape[1]):
        raise ValueError(
            f"scale 必须是 [T, H]={x.shape[2], x.shape[1]}，得到 {tuple(scale.shape)}"
        )
    if bool((scale <= 0).any()):
        raise ValueError("quantize_with_scale 要求 scale > 0；零 scale 用 quantize_kv")
    s = scale.to(SCALE_DTYPE).transpose(0, 1).unsqueeze(-1)  # [1,H,T,1]
    return (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()


def quantize_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """per (kv_head, token) 量化 ``x [1, H, T, D]``。

    Returns:
        ``(q [1, H, T, D] fp8, scale [T, H] fp32)``；q 连续，scale 连续。
        全零 token 的 scale 是 0、code 是 0（反量化仍是 0）。
    """
    if x.dim() != 4 or x.shape[0] != 1:
        raise ValueError(f"x 必须是 [1, H, T, D]，得到 {tuple(x.shape)}")
    x32 = x.float()
    amax = x32.abs().amax(dim=-1)                        # [1,H,T]
    scale = _scale_from_amax(amax)                       # [1,H,T]
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (x32 / safe.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE).contiguous()
    scale = scale[0].transpose(0, 1).contiguous()        # [T,H]
    return q, scale


def reference_quantize_kv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``quantize_kv`` 的独立 reference：逐 kv head 用同一 fp32 契约重算。

    刻意不复用 ``quantize_kv`` 的内部 helper，也不做批量广播；op gate 要求两者
    的 fp8 code 与 scale **bitwise** 相同。e2e 的 FP8 padded reference 也调用它，
    所以“参考 cache 不调用候选量化器”。
    """
    if x.dim() != 4 or x.shape[0] != 1:
        raise ValueError(f"x 必须是 [1, H, T, D]，得到 {tuple(x.shape)}")
    _, heads, _, _ = x.shape
    q_parts: list[torch.Tensor] = []
    s_parts: list[torch.Tensor] = []
    for hi in range(heads):
        xh = x[0, hi].float()                            # [T,D]
        amax = xh.abs().amax(dim=-1)                     # [T]
        scale_h = amax / FP8_MAX
        safe = torch.where(scale_h > 0, scale_h, torch.ones_like(scale_h))
        qh = (xh / safe[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        q_parts.append(qh.contiguous().view(torch.uint8))
        s_parts.append(torch.where(amax > 0, scale_h, torch.zeros_like(scale_h)))
    q = torch.stack(q_parts, dim=0).unsqueeze(0).view(FP8_DTYPE)
    scale = torch.stack(s_parts, dim=1).contiguous()     # [T,H]
    return q, scale


# ----------------------------------------------------------------- dequantize
def _check_layout(q: torch.Tensor, scale: torch.Tensor) -> None:
    if q.dtype != FP8_DTYPE:
        raise ValueError(f"q 必须是 {FP8_DTYPE}，得到 {q.dtype}")
    if q.dim() != 4:
        raise ValueError(f"q 必须是 [1, H, L, D]，得到 {tuple(q.shape)}")
    if scale.dtype != SCALE_DTYPE:
        raise ValueError(f"scale 必须是 float32，得到 {scale.dtype}")
    if tuple(scale.shape) != (q.shape[2], q.shape[1]):
        raise ValueError(
            f"scale 必须是 [L, H]={(q.shape[2], q.shape[1])}，得到 {tuple(scale.shape)}"
        )


def _torch_dequant(q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    s = scale.transpose(0, 1).unsqueeze(-1)  # [1,H,L,1]
    out.copy_(q.float() * s)                 # 一次 fp32 乘，copy_ 里 cast bf16
    return out


def reference_dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """纯 torch 契约实现（e2e reference 用）：``bf16(float32(q) * scale)``。"""
    _check_layout(q, scale)
    s = scale.transpose(0, 1).unsqueeze(-1)
    return (q.float() * s).to(torch.bfloat16)


def dequantize_exact(q: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """op gate 的精确参考：fp64 乘一次，返回 ``(bf16, fp32)``。

    fp64 乘积是精确的（4 位 × 24 位），``float32(exact)`` 就是 fp32 乘法的正确
    舍入；candidate 的 fp32 中间值必须与它 bitwise 相同。
    """
    _check_layout(q, scale)
    s = scale.double().transpose(0, 1).unsqueeze(-1)
    exact = q.double() * s
    f32 = exact.to(torch.float32)
    return f32.to(torch.bfloat16), f32


def _dequant_backend() -> str:
    backend = os.environ.get(DEQUANT_ENV, "triton")
    if backend not in DEQUANT_BACKENDS:
        raise ValueError(f"{DEQUANT_ENV}={backend!r} 必须是 {DEQUANT_BACKENDS} 之一")
    return backend


def _triton_module():
    if _TRITON["module"] is not None:
        return _TRITON["module"]
    if _TRITON["state"] in ("failed", "disabled"):
        return None
    try:
        from . import _triton_dequant  # 延迟 import：CPU 检查不应该拉起 triton
    except Exception as exc:  # noqa: BLE001 - 任何 import 失败都回退
        _TRITON["state"] = "failed"
        _TRITON["error"] = f"{type(exc).__name__}: {exc}"
        return None
    _TRITON["module"] = _triton_dequant
    _TRITON["state"] = "ready"
    return _TRITON["module"]


def _triton_dequant(q: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    if not q.is_cuda:
        # CPU 纯检查（op 脚本的输入构造、单测）不尝试 Triton，也不把它标成失败。
        raise _LayoutUnsupported("非 CUDA 张量")
    module = _triton_module()
    if module is None:
        raise _LayoutUnsupported("triton 不可用")
    if q.shape[0] != 1 or q.shape[-1] != HEAD_DIM or out.dtype != torch.bfloat16:
        raise _LayoutUnsupported("shape/dtype 不在覆盖范围")
    heads, length = q.shape[1], q.shape[2]
    if scale.stride() != (heads, 1) or scale.storage_offset() != 0:
        raise _LayoutUnsupported("scale 不是连续的 [L,H]")
    stride_h = q.stride(1)
    if q.stride() != (heads * stride_h, stride_h, HEAD_DIM, 1) or q.storage_offset() != 0:
        raise _LayoutUnsupported("q 不是 [1,H,L,D] 的标准 view")
    if out.stride() != (heads * length * HEAD_DIM, length * HEAD_DIM, HEAD_DIM, 1):
        raise _LayoutUnsupported("out 不是连续的 [1,H,L,D]")
    module.dequant_fp8_bf16(q, scale, out)
    return out


def dequantize(
    q: torch.Tensor, scale: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """把 ``q [1,H,L,D]`` fp8 与 ``scale [L,H]`` 反量化成 bf16。

    decode 热路径用 ``out=dequant_scratch(...)`` 写入固定地址，避免每步分配；
    prefill 用 ``out=None`` 分配一块与 shape 完全一致的连续 bf16，保证不同执行
    方式看到同样的 bmm 输入（bitwise 可比）。
    """
    _check_layout(q, scale)
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    elif tuple(out.shape) != tuple(q.shape):
        raise ValueError(f"out shape {tuple(out.shape)} != q shape {tuple(q.shape)}")
    _TRITON["calls"] += 1
    if _dequant_backend() == "triton":
        try:
            return _triton_dequant(q, scale, out)
        except _LayoutUnsupported:
            _TRITON["fallbacks"] += 1
        except Exception as exc:  # noqa: BLE001 - 记录后回退，不让 e2e 因 kernel 挂掉
            _TRITON["state"] = "failed"
            _TRITON["error"] = f"{type(exc).__name__}: {exc}"
            _TRITON["fallbacks"] += 1
    return _torch_dequant(q, scale, out)


def dequant_kv(
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    out_k: torch.Tensor | None = None,
    out_v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """融合 kernel 的干净接口：``(K fp8, K scale, V fp8, V scale) -> (K bf16, V bf16)``。

    ``out_*`` 给定时写进固定 buffer；否则各分配一块连续 bf16。
    """
    return (
        dequantize(k, k_scale, out=out_k),
        dequantize(v, v_scale, out=out_v),
    )


# ------------------------------------------------------------- scale side channel
def attach_scale(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """把 ``scale [L,H]`` 挂到 cache update 返回的 fp8 张量上（side channel）。"""
    if tuple(scale.shape) != (tensor.shape[2], tensor.shape[1]):
        raise ValueError(
            f"scale 必须是 [L,H]={(tensor.shape[2], tensor.shape[1])}，"
            f"得到 {tuple(scale.shape)}"
        )
    tensor.kv_fp8_scale = scale
    return tensor


def kv_scale(tensor: torch.Tensor) -> torch.Tensor | None:
    """读取 fp8 张量上的 scale；不是 fp8 KV 时返回 ``None``。"""
    return getattr(tensor, SCALE_ATTR, None)


def dequant_scratch(role: str, shape, device: torch.device | str | None = None) -> torch.Tensor:
    """decode 热路径的固定 bf16 scratch（按 role 各一块）。

    只保留当前 shape：shape 变化时替换旧 buffer。graph 的 decode shape 固定
    （cap 长度），所以 capture 之后不会替换；可变长度的 eager/exact 路径不在
    graph 里，替换是安全的。
    """
    key = str(role)
    shape = tuple(int(x) for x in shape)
    target = torch.device(device) if device is not None else torch.device("cuda")
    buf = _SCRATCH.get(key)
    if buf is None or tuple(buf.shape) != shape or buf.device != target:
        buf = torch.empty(shape, dtype=torch.bfloat16, device=target)
        _SCRATCH[key] = buf
    return buf


def dequant_info() -> dict:
    """产物用：反量化后端状态（是否启用 Triton、失败原因、调用/回退计数）。"""
    module = _TRITON["module"]
    return {
        "requested": os.environ.get(DEQUANT_ENV, "triton"),
        "triton_state": _TRITON["state"],
        "triton_error": _TRITON["error"],
        "triton_available": _TRITON["state"] == "ready",
        "dequant_calls": int(_TRITON["calls"]),
        "dequant_fallbacks": int(_TRITON["fallbacks"]),
        "scratch": {key: tuple(value.shape) for key, value in _SCRATCH.items()},
    }


__all__ = [
    "DEFAULT_FORMAT",
    "DEQUANT_BACKENDS",
    "DEQUANT_ENV",
    "FORMAT_ENV",
    "FORMATS",
    "FP8_DTYPE",
    "FP8_MAX",
    "HEAD_DIM",
    "SCALE_ATTR",
    "SCALE_DTYPE",
    "attach_scale",
    "dequant_info",
    "dequant_kv",
    "dequant_scratch",
    "dequantize",
    "dequantize_exact",
    "kfmt",
    "kv_format",
    "kv_scale",
    "quantize_kv",
    "quantize_with_scale",
    "reference_dequantize",
    "reference_quantize_kv",
    "resolve_format",
]
