"""GDN gated RMSNorm（``Qwen3_5RMSNormGated``）+ out_proj 的 activation quant 融合。

问题
====
48 个 linear-attention 层的 decode 路径里，``Qwen3_5RMSNormGated.forward`` 是纯
PyTorch 展开，单层单步至少 12 个 kernel：

    x32      = hidden.to(fp32)                        # cast
    var      = x32.pow(2).mean(-1)                    # pow / mean 两个 kernel
    var      = var + eps                              # add（in-place）
    rstd     = torch.rsqrt(var)                       # rsqrt
    xn       = x32 * rstd                             # mul
    xb       = xn.to(bf16)                            # cast（先 round 回 BF16）
    normed   = weight * xb                            # bf16 mul（fp32 opmath）
    gate32   = gate.to(fp32)                          # cast
    s        = silu(gate32)                           # silu
    out      = normed * s                             # fp32 mul
    out      = out.to(bf16)                           # cast
    a_fp4, a_sf = nvfp4_quantize(out.reshape(1, 6144), 1 / out_proj.input_scale)

紧接着 ``out_proj``（NVFP4Linear, K=6144）又对同一个 ``[1,1,6144]`` 做一次
``quantize_with_block_size``。job-000042 的 profile 里，这批 tiny kernel（pow 49.7 /
mean 108.7 / rsqrt 52.3 / add 72.8 / mul 94.4 / bf16-mul 50.9 / cast 87 / silu 44.5 µs
每步，含 q_norm/k_norm 等分摊）加上 out_proj 的 48 次 quantize（约 76 µs）是本轮的
目标；本 kernel 用 **48 次 launch** 取代这一整段。

精确算术序列（与模块路径逐字节对齐）
====================================
``Qwen3_5RMSNormGated.forward``（transformers 5.20.2，权重在 checkpoint 里是 BF16）：

    x32   = hidden.to(fp32)                                # bf16 -> f32 精确
    var   = x32.pow(2).mean(-1, keepdim=True)              # 每行在 head_v_dim=128 上
    x32   = x32 * torch.rsqrt(var + eps)
    xb    = x32.to(bf16)                                   # **先** round 回 bf16
    normed= self.weight * xb                               # bf16 权重，bf16 opmath mul
    out   = normed * ACT2FN["silu"](gate.to(fp32))         # bf16 * f32 -> f32
    out   = out.to(bf16)                                   # 再 round 一次

与任务 H 的 ``Qwen3_5RMSNorm`` 的关键差异（三条，都是 bitwise 刀锋）：

1. **weight 直接乘，不是 zero-centered 的 ``1 + w``**；
2. **先在 fp32 里乘 rstd，再 cast 回 bf16，乘完 bf16 weight 后又 round 一次** ——
   也就是有两处独立的 BF16 舍入，而 ``Qwen3_5RMSNorm`` 只有一次；
3. **reduction 维是 ``head_v_dim = 128``（per v-head），不是 5120**，而且
   ``torch.mean`` 的 launch 配置随之完全不同（见下）。

``torch.mean`` 的 reduction 顺序（[48,128] contiguous FP32）
-----------------------------------------------------------
``mean`` 走 ``at::native::mean_kernel_cuda`` -> ``gpu_reduce_kernel`` + ``MeanOps``。
按 ``Reduce.cuh::setReduceConfig`` 对 ``num_outputs=48``、``inputs_per_output=128``、
最快维 stride=4B 的推导：

    reduction_on_fastest_striding_dimension = true
    dim0 = 128 >= 128 且 num_reduce_dims == 1  -> vectorize_input = true，dim0 = 128/4 = 32
    set_block_dimension<float>(32, 48): block_width = 32, block_height = 16（512 线程）
    input_mult  = {1, 0, 0}，step_input  = 32
    values_per_thread = div_up(128, 32) = 4 < warp_split_threshold(256) -> 不分 warp
    output_mult = {0, 1}，step_output = 16，grid.x = div_up(48, 16) = 3

于是每个 CUDA 线程（lane t = 0..31）**恰好 load 一个 float4**（元素 4t..4t+3，
16B 对齐，无 head/tail 处理），ValueList 的 4 个累加器按 ``value_list[0] += val[0]``
分开记，最后顺序合并：

    acc[t] = ((sq[4t] + sq[4t+1]) + sq[4t+2]) + sq[4t+3]
    sum    = offset 16,8,4,2,1 的五级「一半加一半」树（block_x_reduce 的 warp shfl_down）
    var    = mul_rn(sum, factor) + eps

**不能**用一次 ``tl.sum`` 代替那五级：Triton 的 warp 归约是 xor 1,2,4,...,16 的顺序，
括号化不同、FP32 结果会差 ulp。kernel 里就是连续的
``tl.sum(tl.reshape(v, (2, n)), axis=0)``（n = 16,8,4,2,1）。``factor`` 是
``MeanOps`` 的 ``static_cast<float>(num_outputs) / numel`` = 48/6144 = 2^-7（精确）。

silu 与除法
-----------
PyTorch 的 ``silu_kernel``（fp32 opmath）是 ``x / (1 + exp(-x))``；``exp`` 走
``::expf`` -> libdevice ``__nv_expf``，除法是 ``div.rn.f32``。Triton 的 ``a / b``
会落到 ``div.full.f32``，所以这里用 ``libdevice.div_rn`` + ``libdevice.exp``
（任务 I 的 conv kernel 已用同一对函数在 GPU 上验证过逐位一致）。

量化：按任务 E 冻结的规则 R2（与任务 H 的 ``norm_quant.py`` 完全同一套）
======================================================================
``amax`` 是每个 16-element block 的最大绝对值（6144 = 384 个 block，全部落在同一
行内），然后：

    sf_pre = mul_rn(gs, mul_rn(amax, c6))       # c6 = rcp.approx.ftz(6.0f)
    sf_u8  = cvt.rn.satfinite.e4m3x2.f32(sf_pre)
    out_s  = (amax == 0) ? 0 : rcp.approx.ftz(mul_rn(float(sf_u8), rcp.approx.ftz(gs)))
    code_i = cvt.rn.satfinite.e2m1x2.f32(mul_rn(y_i, out_s))

两个倒数都用 ``rcp.approx.ftz.f32``。这里直接复用 ``runner/ops/nvfp4/norm_quant.py``
里的 PTX 助手与 ``_emit_consumer``（连 ``a_sf`` 的 128x4 swizzle 落点
``(b//4)*512 + (b%4)`` 都是同一份代码），避免两份 R2 实现漂移。

数据流（单层、单 token，一个 program，6 个 chunk × 8 行）
========================================================
    core_attn_out [48,128] bf16, gate(z) [48,128] bf16, weight [128] bf16
      -> chunk c（8 行 = 64 个 16-block，1024 个元素）
           load x/g -> 每行 (32,4) 的 lane 结构 -> 5 级树 -> mul/add -> rsqrt.approx
           -> x*rstd -> bf16 -> *w -> bf16 -> *silu(gate) -> bf16
           -> 写 out_bf16[48,128]；按 64-block 一次 amax -> R2 -> 写 a_fp4/a_sf
    out_bf16 -> out_proj 的 ``_pending_quantized``（NVFP4Linear 的 _gemm(m=1) 快路径）

chunk 取 64 个 block（= 任务 H 的 ``CHUNK_BLOCKS``）是为了**原样复用**
``_emit_consumer`` 的 reshape/packing，量化语义不新写一份。

预期瓶颈
========
单次调用只读 24 KB（x + gate）、写 12 KB + 3 KB + 48 KB/消费者，算力/带宽都不是
瓶颈；收益来自 kernel 数量（单层 12+1 -> 1）。所以真正要压的是单 kernel 固定开销，
``num_warps`` 由 ``DEFAULT_NUM_WARPS`` 控制（op 脚本扫 2/4/8 只作诊断，接入固定 4）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from runner.ops.nvfp4.norm_quant import (
    BLOCK,
    C6,
    SF_GROUP_STRIDE,
    _emit_consumer,
    _mul_add_rn,
    _mul_rn,
    _rsqrt_approx,
)

# ---- 冻结的模型 shape（config.linear_num_value_heads / linear_value_head_dim）------
NUM_V_HEADS = 48  # = config.linear_num_value_heads
HEAD_V_DIM = 128  # = config.linear_value_head_dim
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 6144 = out_proj.in_features
N_BLOCKS = VALUE_DIM // BLOCK  # 384
SF_ROWS = 128  # 128x4 swizzle 的 rows pad（与 nvfp4_quantize 输出一致）
SF_COLS = N_BLOCKS  # 384；已是 4 的倍数

#: 每 chunk 的行数：8 行 = 64 个 16-block，与 ``_emit_consumer`` 的 CHUNK_BLOCKS 对齐。
CHUNK_ROWS = 8
CHUNK_BLOCKS = CHUNK_ROWS * (HEAD_V_DIM // BLOCK)  # 64
N_CHUNKS = NUM_V_HEADS // CHUNK_ROWS  # 6
LANES = HEAD_V_DIM // 4  # 32 = PyTorch mean 的 block_width
VEC = 4  # input_vec_size（float4）

#: ``MeanOps`` 的 factor = fl32(48 / 6144)；= 2^-7，精确可表示。
MEAN_FACTOR = 48.0 / 6144.0

DEFAULT_NUM_WARPS = 4
#: 粗粒度参数扫描（诊断用；接入固定 DEFAULT_NUM_WARPS）。
NUM_WARPS_CONFIGS: tuple[int, ...] = (2, 4, 8)

#: Triton 3.8 的 jit body 只能访问 ``tl.constexpr`` 形式的全局量。
C_ROWS = tl.constexpr(CHUNK_ROWS)
C_HEAD = tl.constexpr(HEAD_V_DIM)
C_LANES = tl.constexpr(LANES)
C_BLOCK = tl.constexpr(BLOCK)
C_BLOCKS = tl.constexpr(CHUNK_BLOCKS)
C_PAIR = tl.constexpr(BLOCK // 2)
C_GROUP_STRIDE = tl.constexpr(SF_GROUP_STRIDE)
C_N_CHUNKS = tl.constexpr(N_CHUNKS)


def mean_factor(value_dim: int = VALUE_DIM, num_rows: int = NUM_V_HEADS) -> float:
    """``MeanOps::project`` 的 factor = ``static_cast<float>(num_outputs) / numel``。

    必须用 f32 除法而不是 Python 的 ``num_rows / value_dim``：后者是 double 除法再
    舍入，双舍入在 48/6144 上恰好与 2^-7 相同，但换 shape 时不一定。
    """
    numerator = torch.tensor(float(num_rows), dtype=torch.float32)
    denominator = torch.tensor(float(value_dim), dtype=torch.float32)
    return float((numerator / denominator).item())


def silu_f32(gate: torch.Tensor) -> torch.Tensor:
    """``torch.nn.functional.silu`` 的 fp32 形式（只给 CPU 自测/参考用，不在 kernel 里）。"""
    return gate / (1.0 + torch.exp(-gate))


@triton.jit
def _gated_norm_quant_kernel(
    x_ptr,  # *bf16 [48,128] = core_attn_out.reshape(-1, 128)
    g_ptr,  # *bf16 [48,128] = z.reshape(-1, 128)
    w_ptr,  # *bf16 [128]    = norm.weight（**不是** 1 + w）
    out_ptr,  # *bf16 [48,128] 输出（normed * silu(gate)）
    a_ptr,  # *u8 [3072]      E2M1 code（低 nibble 是偶数下标）
    sf_ptr,  # *u8 [128,384]   128x4 swizzled，仅第 0 行有数据
    eps,  # f32 标量
    factor,  # f32 标量 = 48/6144
    c6,  # f32 标量 = rcp.approx.ftz(6.0f)
    gs,  # f32 标量 = fl32(1 / out_proj.input_scale)，per-consumer
):
    head = tl.arange(0, C_HEAD)  # (128,)
    weight = tl.load(w_ptr + head).to(tl.float32)
    blk = tl.arange(0, C_BLOCKS)  # (64,)
    pair = tl.arange(0, C_PAIR)  # (8,)

    for chunk in tl.static_range(C_N_CHUNKS):
        rows = chunk * C_ROWS + tl.arange(0, C_ROWS)  # (8,)
        offs = rows[:, None] * C_HEAD + head[None, :]  # (8,128)
        xv = tl.load(x_ptr + offs).to(tl.float32)

        # ---- 复刻 ``pow(2).mean(-1)``：每 lane 一个 float4，再五级「一半加一半」树
        v4 = tl.reshape(xv, (C_ROWS, C_LANES, 2, 2))
        even, odd = tl.split(v4)
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        q0 = _mul_rn(x0, x0)
        q1 = _mul_rn(x1, x1)
        q2 = _mul_rn(x2, x2)
        q3 = _mul_rn(x3, x3)
        # 线程内四个累加器按 0,1,2,3 顺序合并（Reduce.cuh 的 value_list combine）
        total = ((q0 + q1) + q2) + q3
        total = tl.sum(tl.reshape(total, (C_ROWS, 2, 16)), axis=1)
        total = tl.sum(tl.reshape(total, (C_ROWS, 2, 8)), axis=1)
        total = tl.sum(tl.reshape(total, (C_ROWS, 2, 4)), axis=1)
        total = tl.sum(tl.reshape(total, (C_ROWS, 2, 2)), axis=1)
        total = tl.sum(tl.reshape(total, (C_ROWS, 2, 1)), axis=1)  # (8,1)

        zero = tl.zeros((C_ROWS, 1), tl.float32)
        variance = _mul_add_rn(total, zero + factor, zero + eps)
        rstd = _rsqrt_approx(variance)  # (8,1)，rsqrt.approx.f32

        # ---- 归一化：先 cast 回 bf16，再乘 bf16 权重（bf16 opmath），再乘 silu(gate)
        xn = xv * rstd
        xb = xn.to(tl.bfloat16)
        nf = xb.to(tl.float32) * weight[None, :]
        nb = nf.to(tl.bfloat16)
        gv = tl.load(g_ptr + offs).to(tl.float32)
        # PyTorch silu_kernel（fp32 opmath）：x / (1 + exp(-x))；div.rn + __nv_expf
        yf = nb.to(tl.float32) * libdevice.div_rn(gv, 1.0 + libdevice.exp(-gv))
        yb = yf.to(tl.bfloat16)
        tl.store(out_ptr + offs, yb)

        # ---- 规则 R2：一个消费者（out_proj）的 sf / out scale / code
        yq = tl.reshape(yb.to(tl.float32), (C_BLOCKS, C_BLOCK))
        amax = tl.max(tl.abs(yq), axis=1)
        blocks = chunk * C_BLOCKS + blk
        a_off = blocks[:, None] * C_PAIR + pair[None, :]
        sf_off = (blocks // 4) * C_GROUP_STRIDE + (blocks % 4)
        _emit_consumer(yq, amax, gs, c6, a_ptr, sf_ptr, a_off, sf_off)


# ------------------------------------------------------------------ 宿主入口


def _as_u8(tensor: torch.Tensor) -> torch.Tensor:
    """1-byte dtype（uint8 / float8_e4m3fn / float4_e2m1x2）统一成 uint8 视图。"""
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.element_size() != 1:
        raise TypeError(f"expected a 1-byte dtype, got {tensor.dtype}")
    return tensor.view(torch.uint8)


def allocate_scratch(
    *,
    device: torch.device | str = "cuda",
    a_fp4_dtype: torch.dtype = torch.uint8,
    a_sf_dtype: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """分配 1 份 BF16 输出 + 1 份 ``(a_fp4, a_sf)`` 槽位。

    ``a_sf`` 的 padding（rows 1..127）必须是 0：用 ``zeros`` 分配，kernel 只写第 0 行
    的 384 个有效字节；之后各层复用同一块 buffer，padding 永远不会被改。
    """
    out = torch.empty(NUM_V_HEADS, HEAD_V_DIM, dtype=torch.bfloat16, device=device)
    a_fp4 = torch.empty(1, VALUE_DIM // 2, dtype=a_fp4_dtype, device=device)
    a_sf = torch.zeros(SF_ROWS, SF_COLS, dtype=a_sf_dtype, device=device)
    return out, a_fp4, a_sf


def gated_norm_fused(
    core_attn_out: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gs: float,
    out_bf16: torch.Tensor,
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> None:
    """一次融合 launch：gated RMSNorm（BF16 输出）+ out_proj 的 NVFP4 activation quant。

    Args:
        core_attn_out: BF16 ``[48, 128]`` contiguous（``core_attn_out.reshape(-1, 128)``）。
        gate: BF16 ``[48, 128]`` contiguous（``z.reshape(-1, 128)``）。
        weight: BF16 ``[128]`` contiguous，**就是** ``norm.weight``（直接乘）。
        eps: ``layer_norm_epsilon``（Python float，kernel 里按 f32 用）。
        gs: out_proj 的 global divisor ``fl32(1 / input_scale)``。
        out_bf16: BF16 ``[48, 128]`` 输出缓冲（= 模块 forward 的返回值）。
        a_fp4: U8 ``[1, 3072]``（或 ``[3072]``）输出。
        a_sf: U8 ``[128, 384]`` swizzled 输出，padding 必须已清零。
        num_warps: program 的 warp 数（只影响性能，不影响结果）。
    """
    if core_attn_out.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        raise TypeError(
            f"core_attn_out / gate 必须是 bf16，得到 {core_attn_out.dtype} / {gate.dtype}"
        )
    if weight.dtype != torch.bfloat16:
        raise TypeError(f"weight 必须是 bf16，得到 {weight.dtype}")
    if core_attn_out.numel() != VALUE_DIM or gate.numel() != VALUE_DIM:
        raise ValueError(
            f"kernel 冻结在 {NUM_V_HEADS}x{HEAD_V_DIM}={VALUE_DIM}，得到 "
            f"{tuple(core_attn_out.shape)} / {tuple(gate.shape)}"
        )
    if weight.numel() != HEAD_V_DIM:
        raise ValueError(f"weight 必须是 [{HEAD_V_DIM}]，得到 {tuple(weight.shape)}")
    for tensor in (core_attn_out, gate, weight):
        if not tensor.is_contiguous():
            raise ValueError("core_attn_out / gate / weight 必须 contiguous")
    if out_bf16.numel() != VALUE_DIM or out_bf16.dtype != torch.bfloat16:
        raise ValueError("out_bf16 必须是 bf16 的 [48, 128]")
    if out_bf16.data_ptr() == core_attn_out.data_ptr() or out_bf16.data_ptr() == gate.data_ptr():
        raise ValueError("out_bf16 不能与输入别名")

    _gated_norm_quant_kernel[(1,)](
        core_attn_out,
        gate,
        weight,
        out_bf16,
        _as_u8(a_fp4),
        _as_u8(a_sf),
        float(eps),
        MEAN_FACTOR,
        C6,
        float(gs),
        num_warps=int(num_warps),
    )


__all__ = [
    "CHUNK_BLOCKS",
    "CHUNK_ROWS",
    "DEFAULT_NUM_WARPS",
    "HEAD_V_DIM",
    "MEAN_FACTOR",
    "N_BLOCKS",
    "N_CHUNKS",
    "NUM_V_HEADS",
    "NUM_WARPS_CONFIGS",
    "SF_COLS",
    "SF_ROWS",
    "VALUE_DIM",
    "allocate_scratch",
    "gated_norm_fused",
    "mean_factor",
    "silu_f32",
]
