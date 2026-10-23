"""RMSNorm（zero-centered）+ NVFP4 activation quant 的融合 Triton kernel（M=1 decode）。

设计 / bitwise 契约
==================
现有路径是两步，本 kernel 必须与之**逐字节**相同（不是容差）：

    1) y_bf16 = Qwen3_5RMSNorm(x)          # transformers 5.20.2 的 eager forward
    2) a_fp4, a_sf = nvfp4_quantize(y_bf16.reshape(1, K), 1 / input_scale,
                                    sfLayout=layout_128x4, do_shuffle=False,
                                    backend="cuda")

因此 kernel 内部依次做三件事，每一步都要复刻原实现的机器算术：

**(a) RMSNorm 的 FP32 reduction 顺序（与 PyTorch ``mean(-1)`` 的 launch 一致）**

``Qwen3_5RMSNorm.forward`` 是 ``x.float() -> x * rsqrt(x.pow(2).mean(-1, keepdim=True)
+ eps) -> * (1 + weight.float()) -> .to(bf16)``，其中 ``mean`` 走
``at::native::mean_kernel_cuda`` -> ``gpu_reduce_kernel`` + ``MeanOps``。对
``[1,1,5120]`` contiguous FP32（num_outputs=1、inputs_per_output=5120、
reduction_on_fastest_striding_dimension=true、vectorize_input=true）实测/读源码得到的
launch 配置是：

    block = (block_width=512, block_height=1)，grid = (1, 1)，无 block_y / global reduce
    input_vec_size = vt0 = 4                # 每线程 4 个独立的 float4 累加器
    step_input    = block_width = 512       # 线程 tx 处理 vec 组 tx, tx+512, tx+1024
    values_per_thread = 10 < warp_split_threshold(16)  -> 只有一个 warp split，即上面那步
    mean factor   = fl32(1.0f / 5120)       # MeanOps::project: a * factor（一次 mul.rn）

于是每个线程的值是（``a+b+0`` 与 ``a+b`` 逐位相同，所以第三项缺失时补 0 是安全的）：

    acc_j(t)  = (x[4t+j] + x[4(t+512)+j]) + x[4(t+1024)+j]     # 第三项仅在 t < 256
    V[t]      = ((acc_0 + acc_1) + acc_2) + acc_3
    sum       = 9 级「一半加一半」树：offset = 256,128,64,32（block_x_reduce 的 shared
                memory 树）+ 16,8,4,2,1（warp shfl_down），每级都是 V[t] += V[t+offset]
    variance  = mul_rn(sum, factor) + eps   # 必须显式 mul.rn 再 add.rn；fma 会改舍入
    rstd      = rsqrt.approx.f32(variance)  # torch.rsqrt -> ::rsqrtf -> rsqrt.approx.f32

「一半加一半」树在 Triton 里就是连续的 ``tl.sum(tl.reshape(v, (2, n)), axis=0)``：
每次把 512 个值折半相加，等价于 offset 从 256 到 1 的 9 级配对。**不能**用一次
``tl.sum`` 代替：Triton 的 warp 归约是 xor 1,2,4,...,16 的顺序（相邻先加），和
PyTorch 的 16,8,4,2,1 是不同括号化，FP32 结果会差 ulp。

**(b) 归一化值必须先 round 回 BF16**

``.to(bf16)`` 是 ``cvt.rn.bf16.f32``（RNE）。量化输入若保持 FP32，block amax 与
code 都会变；这是语义变化，不允许。kernel 里 ``y.to(tl.bfloat16)`` 之后再用
``.to(tl.float32)`` 参与量化，保证量化看到的就是 BF16 值。

**(c) 量化按任务 E 冻结的规则 R2（逐条 PTX 语义）**

    gs      = fl32(1/input_scale)                       # per-tensor，per-consumer
    amax    = max_i |float32(y_i)|                      # 每个 16-element block
    sf_pre  = mul_rn(gs, mul_rn(amax, c6))              # c6 = rcp.approx.ftz(6.0f)
    sf_u8   = cvt.rn.satfinite.e4m3x2.f32(sf_pre)
    sff     = float32(sf_u8)                            # e4m3 -> f32 精确
    out     = (amax == 0) ? 0 : rcp.approx.ftz(mul_rn(sff, rcp.approx.ftz(gs)))
    code_i  = cvt.rn.satfinite.e2m1x2.f32(mul_rn(float32(y_i), out))

两个倒数都用 ``rcp.approx.ftz.f32``（用 ``1/x`` 会破坏 bitwise，E 实测残余 1.49e-4）。
``amax`` 与输入 scale 无关，所以同一份 norm 输出的多个消费者共享 norm/amax 计算，
只有 sf/code 各算一遍并各写一份 ``(a_fp4, a_sf)``。

布局与打包（与 ``nvfp4_quantize`` 一致）
=======================================
- ``a_fp4``：U8 ``[1, K/2]``，每字节两个 E2M1 code，**低 nibble 是偶数下标**；
- ``a_sf``：128x4 swizzled，rows pad 128、cols pad 4（K/16=320 已是 4 的倍数）。
  M=1 时只有第 0 行有数据，落点 ``off = (b // 4) * 512 + (b % 4)``（b 是 block 下标，
  512 = 32 * 4 * 4 是 swizzle 每 4 个 block 一组的跨度）；其余 127 行的 padding 恒 0，
  由调用方一次性清零后不再写。

数据流（单次调用，K=5120，一个 program，M=1）
============================================
    x[5120] BF16, w1[5120] FP32 (= 1 + norm.weight.float())
      -> 第一遍：(512,4) + (512,4) + (512,4,尾部 mask) 三次 coalesced load
         -> 4 累加器 -> 9 级树 -> mul/add -> rsqrt -> rstd
      -> 第二遍：5 个 64-block chunk（320 = 5 * 64），每个 chunk
         load x/w1 -> y_f32 -> y_bf16（同时写回 out_bf16）-> y_f32
         -> block amax（16 个一组）
         -> 每个消费者：sf -> e4m3 -> out scale -> 16 个 e2m1 code -> 打包写 a_fp4/a_sf

K 固定 5120（hidden_size），chunk 数 5 是静态的：核里没有运行期 block 循环边界，
寄存器压力被 1024-element 的 chunk 限制住。调用方（``runner/model/norm_quant_patch.py``）
在 hidden != 5120 时回退原路径。

预期瓶颈
========
单次调用只读 10 KB、写 ~10 KB + 2.5 KB/消费者，算力/带宽都不是瓶颈；收益来自
kernel 数量：现有路径每个 norm 是 ~8 个 elementwise/rsqrt/reduce kernel 加每个消费者
一次 ``quantize_with_block_size``，融合后是 1 个 kernel。所以真正要压的是单 kernel
的固定开销（一次 program、9 级树里的 layout 转换），num_warps 由 ``DEFAULT_NUM_WARPS``
控制，op 脚本会扫 2/4/8 三档报告（诊断用，接入固定 4）。
"""

from __future__ import annotations

import struct

import torch
import triton
import triton.language as tl

# ---- 冻结的常量（与 ``runner/ops/nvfp4/linear.py`` / 任务 E 的契约一致）----------
HIDDEN = 5120
BLOCK = 16
N_BLOCKS = HIDDEN // BLOCK  # 320
CHUNK_BLOCKS = 64  # 320 = 5 * 64，静态 chunk 数
N_CHUNKS = N_BLOCKS // CHUNK_BLOCKS  # 5
REDUCE_LANES = 512  # PyTorch mean 的 block_width
REDUCE_VEC = 4  # input_vec_size
REDUCE_SPLIT = REDUCE_LANES * REDUCE_VEC  # 2048：第二次迭代的偏移
REDUCE_TAIL = 2 * REDUCE_SPLIT  # 4096：第三次迭代的偏移（只有 t < 256）
SF_ROWS = 128  # 128x4 swizzle 的 rows pad
SF_GROUP_STRIDE = 512  # swizzle：每 4 个 block 一组，组跨度 512 字节
MAX_CONSUMERS = 3

#: ``rcp.approx.ftz.f32(6.0f)`` = 0x3e2aaaab（与 fl32(1/6) 零 ulp 差，E 已实测）。
_C6_BITS = 0x3E2AAAAB
C6 = struct.unpack("<f", struct.pack("<I", _C6_BITS))[0]

#: Triton 3.8 的 jit body 只能访问 ``tl.constexpr`` 形式的全局量（普通的 int 会被拒绝），
#: 所以下面给 kernel 再包一层；host 侧继续用上面的 int 常量。
C_LANES = tl.constexpr(REDUCE_LANES)
C_VEC = tl.constexpr(REDUCE_VEC)
C_SPLIT = tl.constexpr(REDUCE_SPLIT)
C_TAIL = tl.constexpr(REDUCE_TAIL)
C_BLOCK = tl.constexpr(BLOCK)
C_CHUNK_BLOCKS = tl.constexpr(CHUNK_BLOCKS)
C_PAIR = tl.constexpr(BLOCK // 2)
C_GROUP_STRIDE = tl.constexpr(SF_GROUP_STRIDE)

DEFAULT_NUM_WARPS = 4


def mean_factor(k: int = HIDDEN) -> float:
    """``MeanOps::project`` 的 factor = ``static_cast<float>(1) / k``（一次 f32 除法）。

    必须用 f32 除法而不是 Python 的 ``1.0 / k``：后者是 double 除法再舍入，双舍入
    在某些 k 上会和直接 f32 除法差 1 ulp。
    """
    one = torch.tensor(1.0, dtype=torch.float32)
    den = torch.tensor(float(k), dtype=torch.float32)
    return float((one / den).item())


MEAN_FACTOR = mean_factor(HIDDEN)


def global_divisor(input_scale: float) -> float:
    """``linear.py`` 的 ``1.0 / input_scale``，同样按 f32 除法算。"""
    one = torch.tensor(1.0, dtype=torch.float32)
    scale = torch.tensor(float(input_scale), dtype=torch.float32)
    return float((one / scale).item())


# ------------------------------------------------------------------ PTX 助手


@triton.jit
def _mul_rn(a, b):
    """``mul.rn.f32``：平方必须单独成一条 mul，不能被后面的加法收缩成 fma。

    ``x*x`` 与 ``fma(x, x, 0)`` 结果相同，但 ``fma(a, a, b)`` 与 ``mul(a,a)+b`` 差一次
    舍入；reduction 树里两者紧挨着，所以这里用 inline asm 把平方钉住。
    """
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_add_rn(a, b, c):
    """``mul.rn(a, b)`` 再 ``add.rn(., c)``：显式阻止 fma 收缩。"""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, $2; add.rn.f32 $0, t, $3; }",
        "=r,r,r,r",
        [a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rcp_approx_ftz(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _rsqrt_approx(x):
    return tl.inline_asm_elementwise(
        "rsqrt.approx.f32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e4m3_from_f32(x):
    """``cvt.rn.satfinite.e4m3x2.f32``；返回 int32，低 8 位是该值的 e4m3 code。"""
    return tl.inline_asm_elementwise(
        # cvt 的 .b16 目标里低字节是一个值的 e4m3 code；两个操作数相同，所以两个字节
        # 都是同一个 code，用 mov.b32 {t, t} 复制到 32 位寄存器后取低 8 位。
        "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f32 t, $1, $1;"
        " mov.b32 $0, {t, t}; }",
        "=r,r",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e2m1_from_f32(x):
    """``cvt.rn.satfinite.e2m1x2.f32``；返回 int32，低 4 位是该值的 E2M1 code。"""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $1, $1;"
        " mov.b32 $0, {t, t, t, t}; }",
        "=r,r",
        [x],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _e4m3_to_f32(code):
    """e4m3 code -> float32（精确，无近似）：kernel 里 ``float32(sf_u8)`` 的等价实现。

    e4m3 (bias 7)：normal = 2^(e-7) * (1 + m/8)，subnormal = 2^-6 * (m/8) = m * 2^-9。
    两者在 f32 里都精确可表示，所以这里用位构造 + 一次精确加法/乘法即可，不引入舍入。
    """
    sign = (code >> 7) & 1
    expo = (code >> 3) & 0xF
    mant = code & 0x7
    bits = ((expo + 120) << 23) | (mant << 20)
    # 0x7F / 0xFF（e4m3 的 NaN code）不会被 cvt.rn.satfinite 产生（它饱和到 0x7E=448），
    # 所以这里不额外处理 NaN。
    normal = tl.cast(bits, tl.float32, bitcast=True)
    subnormal = mant.to(tl.float32) * 0.001953125  # 2^-9
    magnitude = tl.where(expo == 0, subnormal, normal)
    return tl.where(sign == 0, magnitude, -magnitude)


@triton.jit
def _split4(v, N: tl.constexpr):
    """(N, 4) -> 4 个 (N,)：第 j 列就是原来的第 j 个元素（行优先展开）。"""
    even, odd = tl.split(tl.reshape(v, (N, 2, 2)))
    c0, c2 = tl.split(even)
    c1, c3 = tl.split(odd)
    return c0, c1, c2, c3


@triton.jit
def _emit_consumer(yf, amax, gs, c6, a_ptr, sf_ptr, a_off, sf_off):
    """规则 R2：一个消费者的 sf / out scale / code，并写 a_fp4 + a_sf。"""
    # 两个 mul.rn 的顺序就是 kernel 的 SFValue = SFScaleVal * (vecMax * rcp(6))；
    # 全是乘法，不会被 fma 收缩。
    sf_pre = gs * (amax * c6)
    sf_code = _e4m3_from_f32(sf_pre) & 0xFF
    sff = _e4m3_to_f32(sf_code)
    # gs 是标量 kernel 参数，用 tl.where 广播成与 amax 同形的向量，才能走 inline asm。
    gs_vec = tl.where(amax >= 0.0, gs, gs)
    gs_rcp = _rcp_approx_ftz(gs_vec)
    out_scale = _rcp_approx_ftz(sff * gs_rcp)
    out_scale = tl.where(amax == 0.0, 0.0, out_scale)

    codes = _e2m1_from_f32(yf * out_scale[:, None])
    lo, hi = tl.split(tl.reshape(codes, (C_CHUNK_BLOCKS, C_PAIR, 2)))
    packed = (lo & 0xF) | ((hi & 0xF) << 4)
    tl.store(a_ptr + a_off, packed.to(tl.uint8))
    tl.store(sf_ptr + sf_off, sf_code.to(tl.uint8))


@triton.jit
def _norm_quant_kernel(
    x_ptr,
    w1_ptr,
    out_ptr,
    eps,
    factor,
    c6,
    gs0,
    gs1,
    gs2,
    a0_ptr,
    sf0_ptr,
    a1_ptr,
    sf1_ptr,
    a2_ptr,
    sf2_ptr,
    NUM_CONSUMERS: tl.constexpr,
    N_CHUNKS_C: tl.constexpr,
):
    # ---------------- 第一遍：复刻 PyTorch ``x.pow(2).mean(-1)`` 的 reduction
    lane = tl.arange(0, C_LANES)
    cols = tl.arange(0, C_VEC)
    off = C_VEC * lane[:, None] + cols[None, :]

    # 平方：PyTorch ``x.pow(2)`` 是逐元素一次 mul.rn；这里用 inline asm 钉住，
    # 防止它和后面的 reduction 加法被收缩成 fma（会多/少一次舍入）。
    x0 = tl.load(x_ptr + off).to(tl.float32)
    x1 = tl.load(x_ptr + off + C_SPLIT).to(tl.float32)
    tail_mask = lane[:, None] < 256
    x2 = tl.load(x_ptr + off + C_TAIL, mask=tail_mask, other=0.0).to(tl.float32)
    v0 = _mul_rn(x0, x0)
    v1 = _mul_rn(x1, x1)
    v2 = _mul_rn(x2, x2)

    a0, a1, a2, a3 = _split4(v0, C_LANES)
    b0, b1, b2, b3 = _split4(v1, C_LANES)
    d0, d1, d2, d3 = _split4(v2, C_LANES)
    acc0 = (a0 + b0) + d0
    acc1 = (a1 + b1) + d1
    acc2 = (a2 + b2) + d2
    acc3 = (a3 + b3) + d3
    total = ((acc0 + acc1) + acc2) + acc3

    # 9 级「一半加一半」树（offset 256,128,...,1）；最后一级留成 (1,)，便于 inline asm
    total = tl.sum(tl.reshape(total, (2, 256)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 128)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 64)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 32)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 16)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 8)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 4)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 2)), axis=0)
    total = tl.sum(tl.reshape(total, (2, 1)), axis=0)

    # factor / eps 是标量 kernel 参数，先做成与 total 同形的张量，inline asm 只吃张量
    zero = tl.zeros((1,), tl.float32)
    variance = _mul_add_rn(total, zero + factor, zero + eps)
    # rstd 留成 (1,1)，广播到 (CHUNK_BLOCKS, BLOCK) 时不依赖 rank 对齐的细节
    rstd = tl.reshape(_rsqrt_approx(variance), (1, 1))

    # ---------------- 第二遍：归一化（写回 BF16）+ 每个消费者一份量化结果
    blk = tl.arange(0, C_CHUNK_BLOCKS)
    elems = tl.arange(0, C_BLOCK)
    pair = tl.arange(0, C_PAIR)

    for chunk in tl.static_range(N_CHUNKS_C):
        blocks = chunk * C_CHUNK_BLOCKS + blk
        offs = blocks[:, None] * C_BLOCK + elems[None, :]
        xv = tl.load(x_ptr + offs).to(tl.float32)
        w1 = tl.load(w1_ptr + offs)
        # (x * rstd) * w1：两次 mul.rn，与 eager 的 ``output * (1 + weight.float())`` 同序
        yf = (xv * rstd) * w1
        yb = yf.to(tl.bfloat16)
        tl.store(out_ptr + offs, yb)
        yqf = yb.to(tl.float32)

        amax = tl.max(tl.abs(yqf), axis=1)
        a_off = blocks[:, None] * C_PAIR + pair[None, :]
        sf_off = (blocks // 4) * C_GROUP_STRIDE + (blocks % 4)

        _emit_consumer(yqf, amax, gs0, c6, a0_ptr, sf0_ptr, a_off, sf_off)
        if NUM_CONSUMERS >= 2:
            _emit_consumer(yqf, amax, gs1, c6, a1_ptr, sf1_ptr, a_off, sf_off)
        if NUM_CONSUMERS >= 3:
            _emit_consumer(yqf, amax, gs2, c6, a2_ptr, sf2_ptr, a_off, sf_off)


# ------------------------------------------------------------------ 宿主入口


def _as_u8(tensor: torch.Tensor) -> torch.Tensor:
    """1-byte dtype（uint8 / float8_e4m3fn / float4_e2m1x2）统一成 uint8 视图给 kernel。"""
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.element_size() != 1:
        raise TypeError(f"expected a 1-byte dtype, got {tensor.dtype}")
    return tensor.view(torch.uint8)


def norm_quant_fused(
    x: torch.Tensor,
    w1: torch.Tensor,
    eps: float,
    gs_list: list[float] | tuple[float, ...],
    out_bf16: torch.Tensor,
    a_fp4: list[torch.Tensor] | tuple[torch.Tensor, ...],
    a_sf: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> None:
    """一次融合 launch：算 RMSNorm（BF16 输出）并为 2~3 个消费者各产一份 NVFP4 A。

    Args:
        x: BF16 ``[1, 1, K]``（或 ``[K]``），contiguous，K = 5120。
        w1: FP32 ``[K]`` contiguous，等于 ``1 + norm.weight.float()``（zero-centered）。
        eps: norm 的 eps（Python float，kernel 里按 f32 用）。
        gs_list: 每个消费者的 global divisor ``fl32(1/input_scale)``，长度 2 或 3。
        out_bf16: BF16 ``[1, 1, K]`` 输出缓冲（归一化值，已被 round 到 BF16）。
        a_fp4: 每个消费者一份 U8 ``[K/2]``（或 ``[1, K/2]``）输出。
        a_sf: 每个消费者一份 U8 ``[128, K/16]`` swizzled 输出，padding 必须已清零。
        num_warps: program 的 warp 数（只影响性能，不影响结果）。
    """
    n = len(gs_list)
    if n not in (2, 3):
        raise ValueError(f"n consumers must be 2 or 3, got {n}")
    if len(a_fp4) != n or len(a_sf) != n:
        raise ValueError("a_fp4 / a_sf length must match gs_list")
    if x.dtype != torch.bfloat16 or w1.dtype != torch.float32:
        raise TypeError(f"x must be bf16 and w1 f32, got {x.dtype} / {w1.dtype}")
    if x.numel() != HIDDEN or w1.numel() != HIDDEN:
        raise ValueError(f"kernel is frozen for K={HIDDEN}, got {x.numel()}")
    if not x.is_contiguous():
        x = x.contiguous()
    if not w1.is_contiguous():
        w1 = w1.contiguous()

    gs = list(gs_list) + [0.0] * (MAX_CONSUMERS - n)
    a_ptrs = [_as_u8(t) for t in a_fp4] + [None] * (MAX_CONSUMERS - n)
    sf_ptrs = [_as_u8(t) for t in a_sf] + [None] * (MAX_CONSUMERS - n)
    for index in range(MAX_CONSUMERS):
        if a_ptrs[index] is None:
            a_ptrs[index] = a_ptrs[0]
            sf_ptrs[index] = sf_ptrs[0]

    _norm_quant_kernel[(1,)](
        x,
        w1,
        out_bf16,
        float(eps),
        MEAN_FACTOR,
        C6,
        gs[0],
        gs[1],
        gs[2],
        a_ptrs[0],
        sf_ptrs[0],
        a_ptrs[1],
        sf_ptrs[1],
        a_ptrs[2],
        sf_ptrs[2],
        NUM_CONSUMERS=n,
        N_CHUNKS_C=N_CHUNKS,
        num_warps=int(num_warps),
    )


def allocate_scratch(
    *,
    n_slots: int = MAX_CONSUMERS,
    device: torch.device | str = "cuda",
    a_fp4_shape: tuple[int, ...] = (1, HIDDEN // 2),
    a_sf_shape: tuple[int, ...] = (SF_ROWS, HIDDEN // BLOCK),
    a_fp4_dtype: torch.dtype = torch.uint8,
    a_sf_dtype: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """分配 1 份 BF16 归一化输出 + ``n_slots`` 份 (a_fp4, a_sf) 槽位。

    shape/dtype 由调用方按 FlashInfer ``nvfp4_quantize`` 的实际输出传进来（不猜）。
    ``a_sf`` 的 padding（rows 1..127）必须是 0：这里用 ``zeros``，kernel 只写第 0 行
    的 320 个有效字节，之后各次调用复用同一块 buffer，padding 永远不会被改。
    """
    out = torch.empty(1, 1, HIDDEN, dtype=torch.bfloat16, device=device)
    a_fp4 = [
        torch.empty(a_fp4_shape, dtype=a_fp4_dtype, device=device)
        for _ in range(n_slots)
    ]
    a_sf = [
        torch.zeros(a_sf_shape, dtype=a_sf_dtype, device=device)
        for _ in range(n_slots)
    ]
    return out, a_fp4, a_sf


__all__ = [
    "C6",
    "DEFAULT_NUM_WARPS",
    "HIDDEN",
    "MEAN_FACTOR",
    "N_BLOCKS",
    "SF_ROWS",
    "allocate_scratch",
    "global_divisor",
    "mean_factor",
    "norm_quant_fused",
]
