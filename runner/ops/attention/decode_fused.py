"""融合 GQA decode attention（q_len==1）：split-K + online softmax 的单个 Triton kernel。

问题：``compact_decode.py`` 的 grouped bmm 已经把 ``repeat_kv`` 的 6 倍物化去掉，
但一层 decode attention 仍是 7 个 kernel（QK bmm、scaling、mask add、softmax、
cast、PV bmm、transpose/contiguous）。8K 下 QK bmm 50.7 µs + PV bmm 20 µs，
加上 softmax/copy 的 elementwise 后 16 层约 1.1–1.4 ms/step，而 KV 读的理论下限
只有 ``16 * 2 * 4 * 8192 * 256 * 2 B = 537 MB``（约 0.60 ms @900 GB/s），有效带宽
只有约 43%。

设计：
- **每个 kv head 只读一次**：program 以 ``(kv_head, split)`` 为粒度，一次处理共享
  该 kv head 的 ``GROUPS = 24 / 4 = 6`` 个 q head。若按 q head 切 program，K/V 会
  被读 6 遍（又回到 ``repeat_kv`` 的流量），所以 head 维必须合在一个 program 里。
  6 行 padding 到 ``BLOCK_M = 16`` 走 tensor core MMA（``tl.dot`` 的最小 M 是 16）；
  M 的 62% 浪费不影响性能，因为算术强度只有约 8 FLOP/B，瓶颈是 KV 读带宽。
- **split-K**：把 KV 长度切成 ``SPLITS`` 段，每段独立算 ``(m, l, acc)`` 写 fp32
  partial，第二个小 kernel 合并。``SPLITS`` 由长度决定（``cdiv(L, BLOCK_N*min_blocks)``，
  上限 32），8K 时 4 kv head × 32 = 128 个 CTA，撑满 84 个 SM；否则只有 4 个 CTA，
  带宽利用率极低。``SPLITS == 1``（很短的 L）直接归一化写 bf16 输出，省一个 kernel。
- **online softmax**：全程 fp32，log2 域（``qk_scale = scaling * log2(e)``，
  mask 也乘 ``log2(e)``），用 ``exp2``。``p`` 参与 PV 前转 bf16，与当前路径
  ``softmax(dtype=float32).to(bf16) @ v`` 的精度口径一致；区别只是这里不做
  ``scores`` 的 bf16 往返（bmm 输出的 bf16 舍入），并且归一化放在 fp32 累加之后，
  所以数值上不应比 bmm 路径差。
- **mask**：加性、可广播到 ``[1, 4, 1, L]``。不可见位置在 softmax 里 ``exp2`` 后
  是精确 0（bf16 的 ``finfo.min`` 乘 ``log2(e)`` 会下溢成 ``-inf``，同一结果），
  因此不可见 K/V 的取值不影响输出（bitwise 隔离）。
- **非连续 K/V**：按 ``stride(1)``（kv head）与 ``stride(2)``（token）寻址，直接读
  static cache 的 capacity 前缀 view，不做 reshape/copy。
- **FP8 KV 接口**：``k_scale`` / ``v_scale`` 是 per (token, kv_head) 的 fp32 scale
  （K/V 各一套，布局 ``[tokens, kv_heads]`` 或 ``[kv_heads, tokens]``）。
  scale 与 head_dim 无关，因此 K 的 scale 乘在 QK 点积之后、V 的 scale 乘在 ``p``
  上，数学上等价于 ``(k.float() * k_scale)`` 逐元素反量化，但每 block 只做
  ``BLOCK_N`` 次乘法而不是 ``BLOCK_N * 256`` 次。``scale=None`` 走 BF16。

数据流（单层、单步）：
    q [1,24,1,256] --按 kv head 分组--> Q[16,256]（前 6 行有效）
    for 每段 KV（BLOCK_N 个 token）:
        K[BLOCK_N,256] --trans--> Kᵀ[256,BLOCK_N]
        s = tl.dot(Q, Kᵀ) * qk_scale (+ k_scale) (+ mask*log2e)   # [16,BLOCK_N] fp32
        m_new = max(m, rowmax(s)); alpha = exp2(m - m_new)
        p = exp2(s - m_new); l = l*alpha + rowsum(p)
        acc = acc*alpha + bf16(p) @ V
    partial (m, l, acc) --combine(exp2 加权/归一化)--> out [1,24,1,256] bf16

预期瓶颈：KV 读带宽。8K 每层 K+V = 33.55 MB，16 层 537 MB/step，900 GB/s 下
0.60 ms；考虑 split-K 的 partial 流量（每层 24*32*256*4 B = 786 KB，可忽略）。
MMA 需求约 8 FLOP/B × 900 GB/s = 7.2 TFLOP/s，远低于 BF16 tensor core 峰值。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# 粗粒度参数（三组，microbench 里对比）：(BLOCK_N, num_warps, num_stages, min_blocks)
CONFIGS: dict[str, dict[str, int]] = {
    # 默认：64 token/block、2 stage；8K 时 32 split × 4 kv head = 128 CTA。
    "bn64-w8-s2": {"block_n": 64, "num_warps": 8, "num_stages": 2, "min_blocks": 2},
    # 更小的 block、更多 stage（每 split 8 个 block，流水更满，smem 更小）。
    "bn32-w4-s3": {"block_n": 32, "num_warps": 4, "num_stages": 3, "min_blocks": 4},
    # 更少的 split（8K 时 16 split = 64 CTA），每 split 8 个 block。
    "bn64-w8-s2-mb4": {"block_n": 64, "num_warps": 8, "num_stages": 2, "min_blocks": 4},
}
DEFAULT_CONFIG = "bn64-w8-s2"
MAX_SPLITS = 32

LOG2E = 1.4426950408889634
NEG = -1.0e38  # 位置无效的 score；exp2(-1e38) == 0，且不会和全 block 无效时相减出 NaN

# FP8 KV：e4m3 是唯一支持的 fp8 格式（与本项目 J 任务冻结的方案一致）。
FP8_DTYPES = (torch.float8_e4m3fn,)


# --------------------------------------------------------------------- kernels
@triton.jit
def _split_kernel(
    q_ptr,  # bf16 [H_Q, D]（head stride = q_stride_h）
    k_ptr,  # bf16|fp8 [H_KV, L, D]
    v_ptr,  # bf16|fp8 [H_KV, L, D]
    ks_ptr,  # fp32 per (token, kv_head)，HAS_KSCALE=False 时不读
    vs_ptr,  # fp32 per (token, kv_head)，HAS_VSCALE=False 时不读
    mask_ptr,  # bf16|fp32 add，HAS_MASK=False 时不读
    acc_ptr,  # fp32 [H_Q, SPLITS, D] partial（WRITE_OUT 时不写）
    m_ptr,  # fp32 [H_Q, SPLITS] 每 split 的 running max（log2 域）
    l_ptr,  # fp32 [H_Q, SPLITS] 每 split 的 softmax 分母
    out_ptr,  # bf16 [H_Q, D]，只在 WRITE_OUT 时写
    seq_len,  # 有效 KV 长度（runtime，避免按 L 重编译）
    q_stride_h,
    k_stride_h,
    k_stride_t,
    v_stride_h,
    v_stride_t,
    ks_stride_t,
    ks_stride_h,
    vs_stride_t,
    vs_stride_h,
    mask_stride_t,
    mask_stride_h,
    chunk,  # 每个 split 覆盖的 token 数（BLOCK_N 的整数倍，runtime）
    qk_scale,  # scaling * log2(e)
    LOG2E: tl.constexpr,
    NEG: tl.constexpr,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLITS: tl.constexpr,
    FP8_KV: tl.constexpr,
    HAS_KSCALE: tl.constexpr,
    HAS_VSCALE: tl.constexpr,
    HAS_MASK: tl.constexpr,
    WRITE_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    kv = pid // SPLITS
    sp = pid % SPLITS

    # PV 的输入 dtype：FP8 分支先把 V 反量化成 bf16（e4m3 是 bf16 子集，转换精确）；
    # BF16/FP16 分支跟随 V 的 dtype（fp16 只用于 CPU interpreter 单测）。
    pv_dtype = tl.bfloat16 if FP8_KV else v_ptr.dtype.element_ty

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < GROUPS
    rows = kv * GROUPS + offs_m

    # Q 只读一次；无效行填 0，它们算出的结果不会被写回。
    q = tl.load(
        q_ptr + rows[:, None] * q_stride_h + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), NEG, tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    start = sp * chunk
    for off in range(0, chunk, BLOCK_N):
        tok = start + off + tl.arange(0, BLOCK_N)
        tok_ok = tok < seq_len

        k = tl.load(
            k_ptr + kv * k_stride_h + tok[:, None] * k_stride_t + offs_d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        if FP8_KV:
            # fp8 -> bf16 是精确的（e4m3 是 bf16 的子集），反量化 scale 折到 score 上。
            k = k.to(tl.float32).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k))
        s = s * qk_scale
        if HAS_KSCALE:
            ksc = tl.load(
                ks_ptr + tok * ks_stride_t + kv * ks_stride_h,
                mask=tok_ok,
                other=0.0,
            ).to(tl.float32)
            s = s * ksc[None, :]
        if HAS_MASK:
            mk = tl.load(
                mask_ptr + tok * mask_stride_t + kv * mask_stride_h,
                mask=tok_ok,
                other=0.0,
            ).to(tl.float32)
            s = s + mk[None, :] * LOG2E
        s = tl.where(tok_ok[None, :], s, NEG)

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        # 全 block 位置无效时 m_new == m_i == NEG，exp2(0) 会给出 1；必须显式清零。
        p = tl.where(tok_ok[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(
            v_ptr + kv * v_stride_h + tok[:, None] * v_stride_t + offs_d[None, :],
            mask=tok_ok[:, None],
            other=0.0,
        )
        if HAS_VSCALE:
            vsc = tl.load(
                vs_ptr + tok * vs_stride_t + kv * vs_stride_h,
                mask=tok_ok,
                other=0.0,
            ).to(tl.float32)
            p = p * vsc[None, :]
            if FP8_KV:
                v = v.to(tl.float32).to(pv_dtype)
        acc = acc * alpha[:, None] + tl.dot(p.to(pv_dtype), v)
        m_i = m_new

    if WRITE_OUT:
        tl.store(
            out_ptr + rows[:, None] * D + offs_d[None, :],
            (acc / l_i[:, None]).to(out_ptr.dtype.element_ty),
            mask=row_ok[:, None],
        )
    else:
        part_off = rows[:, None] * (SPLITS * D) + sp * D + offs_d[None, :]
        tl.store(acc_ptr + part_off, acc, mask=row_ok[:, None])
        tl.store(m_ptr + rows * SPLITS + sp, m_i, mask=row_ok)
        tl.store(l_ptr + rows * SPLITS + sp, l_i, mask=row_ok)


@triton.jit
def _combine_kernel(
    acc_ptr,  # fp32 [H_Q, SPLITS, D]
    m_ptr,  # fp32 [H_Q, SPLITS]
    l_ptr,  # fp32 [H_Q, SPLITS]
    out_ptr,  # bf16 [H_Q, D]
    NEG: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    h = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, D)
    s_ok = offs_s < SPLITS

    m = tl.load(m_ptr + h * SPLITS + offs_s, mask=s_ok, other=NEG)
    l = tl.load(l_ptr + h * SPLITS + offs_s, mask=s_ok, other=0.0)
    acc = tl.load(
        acc_ptr + h * (SPLITS * D) + offs_s[:, None] * D + offs_d[None, :],
        mask=s_ok[:, None],
        other=0.0,
    )

    m_max = tl.max(m, axis=0)
    w = tl.exp2(m - m_max)
    denom = tl.sum(l * w, axis=0)
    out = tl.sum(acc * w[:, None], axis=0) / denom
    tl.store(out_ptr + h * D + offs_d, out.to(out_ptr.dtype.element_ty))


# ------------------------------------------------------------------- workspace
# partial 缓冲按 (device, splits, heads, head_dim) 复用：CUDA Graph capture 前会先跑
# eager warmup，所以分配发生在 capture 之外，replay 里只有固定地址的读写。
_WORKSPACE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def workspace(device: torch.device, splits: int, heads: int, head_dim: int):
    key = (str(device), splits, heads, head_dim)
    buf = _WORKSPACE.get(key)
    if buf is None:
        acc = torch.empty((heads, splits, head_dim), device=device, dtype=torch.float32)
        m = torch.empty((heads, splits), device=device, dtype=torch.float32)
        l = torch.empty((heads, splits), device=device, dtype=torch.float32)
        buf = (acc, m, l)
        _WORKSPACE[key] = buf
    return buf


# ------------------------------------------------------------------ host helpers
def split_count(seq_len: int, block_n: int, min_blocks: int, max_splits: int) -> int:
    """split 数：每个 split 至少 ``block_n * min_blocks`` 个 token，上限 max_splits。"""
    per = block_n * min_blocks
    return max(1, min(int(max_splits), triton.cdiv(int(seq_len), per)))


def split_chunk(seq_len: int, splits: int, block_n: int) -> int:
    """每个 split 的 token 数，向上取整到 BLOCK_N 的整数倍。"""
    return triton.cdiv(triton.cdiv(int(seq_len), splits), block_n) * block_n


def _scale_strides(scale: torch.Tensor, h_kv: int, seq_len: int, name: str):
    """per-token scale -> (stride_token, stride_head)。

    支持 ``[tokens, h_kv]``（J 冻结的布局）、``[h_kv, tokens]``、``[tokens]``
    （广播到所有 head）以及带 leading 1 的 3D 形状。
    """
    u = scale
    while u.dim() > 2 and u.shape[0] == 1:
        u = u[0]
    if u.dim() == 1:
        if u.shape[0] < seq_len:
            raise ValueError(f"{name} 长度 {u.shape[0]} < KV 长度 {seq_len}")
        return int(u.stride(0)), 0
    if u.dim() == 2:
        if u.shape[1] == h_kv or (u.shape[1] == 1 and u.shape[0] != h_kv):
            if u.shape[0] < seq_len:
                raise ValueError(f"{name} token 维 {u.shape[0]} < KV 长度 {seq_len}")
            return int(u.stride(0)), (int(u.stride(1)) if u.shape[1] > 1 else 0)
        if u.shape[0] == h_kv:
            if u.shape[1] < seq_len:
                raise ValueError(f"{name} token 维 {u.shape[1]} < KV 长度 {seq_len}")
            return int(u.stride(1)), int(u.stride(0))
    raise ValueError(f"{name} 形状 {tuple(scale.shape)} 不支持（需要 [tokens, kv] 或 [kv, tokens]）")


def _mask_strides(mask: torch.Tensor, h_kv: int, seq_len: int):
    """加性 mask -> (stride_token, stride_head)；支持 ``[.., 1, L]`` 与 ``[.., h_kv, L]``。"""
    u = mask
    while u.dim() > 1 and u.shape[0] == 1:
        u = u[0]
    if u.dim() > 2 and u.shape[1] == 1:
        u = u[:, 0]
    if u.dim() == 1:
        if u.shape[0] < seq_len:
            raise ValueError(f"mask 长度 {u.shape[0]} < KV 长度 {seq_len}")
        return int(u.stride(0)), 0
    if u.dim() == 2:
        if u.shape[-1] < seq_len or u.stride(-1) != 1:
            raise ValueError(
                f"mask 形状 {tuple(mask.shape)} 不支持：最后一维必须是长度 >= {seq_len} 的连续维"
            )
        return int(u.stride(-1)), (int(u.stride(0)) if u.shape[0] > 1 else 0)
    raise ValueError(f"mask 形状 {tuple(mask.shape)} 不支持")


def can_fuse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> bool:
    """fused kernel 的适用范围；不满足时上层回退 grouped_bmm（不改语义）。

    设备检查不在这里：patch 入口已经要求 ``query.is_cuda``，而 CPU 上的
    ``TRITON_INTERPRET=1`` 单元测试要能直接跑这个函数。
    """
    if query.dim() != 4 or query.shape[0] != 1 or query.shape[2] != 1:
        return False
    if key.dim() != 4 or key.shape != value.shape:
        return False
    if key.shape[0] != 1 or key.shape[1] != value.shape[1]:
        return False
    head_dim = query.shape[3]
    if head_dim != key.shape[3] or head_dim < 16 or (head_dim & (head_dim - 1)) != 0:
        return False
    if query.dtype not in (torch.bfloat16, torch.float16):
        return False
    if key.dtype != value.dtype:
        return False
    if key.dtype in FP8_DTYPES:
        # FP8 KV：必须给 per-token scale；q 仍是 bf16（模型口径）。
        if k_scale is None or v_scale is None or query.dtype != torch.bfloat16:
            return False
    elif key.dtype != query.dtype:
        return False
    if query.shape[1] % key.shape[1] != 0:
        return False
    if query.stride(3) != 1 or key.stride(3) != 1 or value.stride(3) != 1:
        return False
    if attention_mask is not None:
        try:
            _mask_strides(attention_mask, key.shape[1], key.shape[2])
        except ValueError:
            return False
    return True


# ------------------------------------------------------------------ entry point
def fused_gqa_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    *,
    config: str | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    min_blocks: int | None = None,
    max_splits: int = MAX_SPLITS,
) -> tuple[torch.Tensor, None]:
    """q_len==1 的融合 GQA decode attention。

    Args:
        query: ``[1, num_q_heads, 1, head_dim]`` BF16（``stride(-1) == 1``）。
        key/value: ``[1, num_kv_heads, L, head_dim]``，BF16 或 ``float8_e4m3fn``；
            允许是 capacity buffer 的非连续前缀 view（按 stride 寻址）。
        attention_mask: 加性 mask，可广播到 ``[1, num_kv_heads, 1, L]`` 或 None。
        scaling: 通常是 ``head_dim ** -0.5``。
        k_scale/value_scale: per (token, kv_head) 的 fp32 scale，``[tokens, kv_heads]``
            或 ``[kv_heads, tokens]``；BF16 KV 时为 None。

    Returns:
        ``(attn_output [1, num_q_heads, 1, head_dim] bf16, None)``。
        kernel 不物化 attention weights（HF 的 ``output_attentions=False`` 语义）；
        需要 weights 的调用方应走 grouped_bmm / eager。
    """
    if not can_fuse(query, key, value, attention_mask, k_scale, v_scale):
        raise ValueError(
            "fused_gqa_decode_attention 不支持这组输入："
            f"q={tuple(query.shape)}/{query.dtype} k={tuple(key.shape)}/{key.dtype} "
            f"v={tuple(value.shape)}/{value.dtype} mask={None if attention_mask is None else tuple(attention_mask.shape)}"
        )
    spec = CONFIGS[config or DEFAULT_CONFIG]
    block_n = int(block_n or spec["block_n"])
    num_warps = int(num_warps or spec["num_warps"])
    num_stages = int(num_stages or spec["num_stages"])
    min_blocks = int(min_blocks or spec["min_blocks"])

    batch, num_q_heads, q_len, head_dim = query.shape
    num_kv_heads = key.shape[1]
    seq_len = key.shape[2]
    groups = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(groups))

    fp8 = key.dtype in FP8_DTYPES
    has_kscale = k_scale is not None
    has_vscale = v_scale is not None
    ks_t = ks_h = vs_t = vs_h = mask_t = mask_h = 0
    if has_kscale:
        ks_t, ks_h = _scale_strides(k_scale, num_kv_heads, seq_len, "k_scale")
    if has_vscale:
        vs_t, vs_h = _scale_strides(v_scale, num_kv_heads, seq_len, "v_scale")
    if attention_mask is not None:
        mask_t, mask_h = _mask_strides(attention_mask, num_kv_heads, seq_len)

    splits = split_count(seq_len, block_n, min_blocks, max_splits)
    chunk = split_chunk(seq_len, splits, block_n)
    out = torch.empty(
        (batch, num_q_heads, q_len, head_dim), device=query.device, dtype=query.dtype
    )

    if splits == 1:
        # 单 split：不需要 partial 缓冲，直接归一化写输出。
        acc_buf = m_buf = l_buf = out
    else:
        acc_buf, m_buf, l_buf = workspace(query.device, splits, num_q_heads, head_dim)

    _split_kernel[(num_kv_heads * splits,)](
        query,
        key,
        value,
        k_scale if has_kscale else query,  # 未使用时只作为占位指针，不解引用
        v_scale if has_vscale else query,
        attention_mask if attention_mask is not None else query,
        acc_buf,
        m_buf,
        l_buf,
        out,
        seq_len,
        query.stride(1),
        key.stride(1),
        key.stride(2),
        value.stride(1),
        value.stride(2),
        ks_t,
        ks_h,
        vs_t,
        vs_h,
        mask_t,
        mask_h,
        chunk,
        scaling * LOG2E,
        LOG2E=LOG2E,
        NEG=NEG,
        GROUPS=groups,
        D=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        SPLITS=splits,
        FP8_KV=fp8,
        HAS_KSCALE=has_kscale,
        HAS_VSCALE=has_vscale,
        HAS_MASK=attention_mask is not None,
        WRITE_OUT=(splits == 1),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if splits > 1:
        _combine_kernel[(num_q_heads,)](
            acc_buf,
            m_buf,
            l_buf,
            out,
            NEG=NEG,
            D=head_dim,
            SPLITS=splits,
            BLOCK_S=triton.next_power_of_2(splits),
            num_warps=8,
            num_stages=1,
        )
    return out, None


def kernel_plan(seq_len: int, *, config: str | None = None, max_splits: int = MAX_SPLITS) -> dict:
    """给 op 产物/日志用的静态计划：split 数、chunk、grid、每 call 的 KV 字节。"""
    spec = CONFIGS[config or DEFAULT_CONFIG]
    splits = split_count(seq_len, spec["block_n"], spec["min_blocks"], max_splits)
    return {
        "config": config or DEFAULT_CONFIG,
        "block_n": spec["block_n"],
        "num_warps": spec["num_warps"],
        "num_stages": spec["num_stages"],
        "min_blocks": spec["min_blocks"],
        "splits": splits,
        "chunk": split_chunk(seq_len, splits, spec["block_n"]),
        "grid_split": 4 * splits,  # 本项目 4 个 kv head
        "combine_kernels": 0 if splits == 1 else 1,
    }


__all__ = [
    "CONFIGS",
    "DEFAULT_CONFIG",
    "FP8_DTYPES",
    "MAX_SPLITS",
    "can_fuse",
    "fused_gqa_decode_attention",
    "kernel_plan",
    "split_chunk",
    "split_count",
    "workspace",
]
