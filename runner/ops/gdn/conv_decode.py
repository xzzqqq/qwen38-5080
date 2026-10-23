"""GDN decode（``sequence_length == 1``）conv step 的融合 Triton kernel。

问题：``causal_conv1d_update`` 在 decode 时是纯 PyTorch 展开，单层单步至少 4 个
kernel：

    hidden_states_new = cat([conv_state, mixed_qkv], -1)      # 1 个 cat
    conv_state.copy_(hidden_states_new[..., -state_len:])    # 1 个 copy_
    out = F.conv1d(hidden_states_new, weight[None], ...)     # depthwise conv
    out = out[..., -1:]
    out = silu(out)                                          # 1 个 silu

48 层合计约 192 个 node/step（profile 里对应 ``CatArrayBatchedCopy`` /
``direct_copy`` / ``conv_depthwise2d`` / ``silu_kernel`` 各 48 个），每个都是
<2 µs 的 tiny kernel，代价主要是 launch/依赖链而不是带宽。

数据流（单层、C = conv_dim = 10240，state_len = conv_kernel = 4）：

    conv_state [1, C, 4] (bf16, 原地滚动)   x = mixed_qkv [1, C, 1] (bf16)
      cat 窗口 = [s1, s2, s3, x]（state 滚动之后，next state 恰好就是同一窗口）
      conv = sum_j weight[:, j] * win[:, j]        # fp32 累加后**先舍入到 bf16**
      out  = conv / (1 + exp(-conv))               # PyTorch silu 的 bf16 opmath 形式
      写回：state[:, :] = win（原地），out 写进调用方的 out 槽（默认就地覆盖 x）

关键语义（与 reference 的逐位对齐点，见 ``scripts/phase2_gdn_conv_check.py``）：
1. **state_len 是 conv_kernel_size（4），不是 kernel_size-1（3）**。prefill 的
   ``update_conv_state`` 把 conv_state 初始化为 ``[1, C, conv_kernel_size]``，所以
   decode 时 ``cat([state(4), x(1)])`` 长度是 5，conv 输出 2 个、取最后一个：
   ``out = w0*s1 + w1*s2 + w2*s3 + w3*x``，新 state = ``[s1, s2, s3, x]``。这个
   「窗口 == 新 state」的等式只在 state_len == kernel 时成立，host 侧会检查。
2. reference 是 ``F.conv1d`` 的 bf16 输出（fp32 累加、一次舍入到 bf16），**再**做
   silu。本 kernel 先把 conv 舍入到 bf16，再用 fp32 算 silu 并舍入，避免把
   「fp32 conv + fp32 silu」这种多出来的精度差带进 gate（会让 nrmse 变大 10× 以上）。
3. silu 用 ``x / (1 + exp(-x))``（PyTorch 的 bf16 opmath 形式）+ ``libdevice.exp``
   （``__nv_expf``，与 ``torch.exp`` 在 fp32 上逐位一致），不用 ``tl.sigmoid``
   （会落到 ``ex2.approx``）。
4. conv_state 必须**原地**更新：dtype/字节/data_ptr 不变，static cache 与 graph
   replay 依赖这一点；跨 program 没有写竞争（每 channel 只归一个 program）。

预期瓶颈：纯延迟与 launch 数。每层流量只有 state 读 80 KB + weight 80 KB +
x 20 KB、写 80 KB + out 20 KB = 280 KB（weight 48 层共 3.9 MB，跨 step 常驻 L2），
48 层合计约 13 MB/step，在 960 GB/s 上只有 ~14 µs；实际由 40–160 个 CTA 的
短 kernel 延迟（每层约 2 µs × 48）决定。收益来自 node 数 192 → 48。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# 粗粒度参数：(BLOCK（每 program 的 channel 数）, num_warps)。三组都保持
# 「每 channel 一个 program 元素」的布局，差别只在 CTA 数量/每线程向量宽度。
BLOCK_CONFIGS: tuple[tuple[int, int], ...] = ((128, 4), (256, 4), (64, 2))
DEFAULT_BLOCK = 128
DEFAULT_NUM_WARPS = 4

# 支持的 activation：None 表示不做激活，其余必须是 silu（本模型 hidden_act）。
_ACTIVATIONS = (None, "silu")


@triton.jit
def _gdn_conv_decode_kernel(
    x_ptr,  # *bf16 [C, 1]（channel 维 stride=1）
    state_ptr,  # *bf16 [C, K]，原地滚动
    w_ptr,  # *bf16 [C, K] depthwise conv weight
    out_ptr,  # *bf16 [C, 1]
    C: tl.constexpr,
    K: tl.constexpr,  # state_len == conv kernel size
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    APPLY_SILU: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid * BLOCK + tl.arange(0, BLOCK)
    j = tl.arange(0, K)

    # 窗口 [s1, s2, s3, x]：state 的第 1..K-1 列 + 当前 token。第 K 列被 mask
    # 掉（地址会跨到下一个 channel），再用 x 填上。
    if EVEN:
        win = tl.load(
            state_ptr + c[:, None] * K + (j[None, :] + 1),
            mask=j[None, :] < (K - 1),
            other=0.0,
        )
        x = tl.load(x_ptr + c)
        w = tl.load(w_ptr + c[:, None] * K + j[None, :])
    else:
        cmask = c < C
        win = tl.load(
            state_ptr + c[:, None] * K + (j[None, :] + 1),
            mask=cmask[:, None] & (j[None, :] < (K - 1)),
            other=0.0,
        )
        x = tl.load(x_ptr + c, mask=cmask, other=0.0)
        w = tl.load(w_ptr + c[:, None] * K + j[None, :], mask=cmask[:, None], other=0.0)

    win = tl.where(j[None, :] == (K - 1), x[:, None], win)

    # bf16 乘 bf16 的乘积在 fp32 里是精确的；reference（F.conv1d / addmm）也是
    # fp32 累加、最后一次性舍入到 bf16，所以先对齐这一次舍入再做 silu。
    acc = tl.sum(win.to(tl.float32) * w.to(tl.float32), 1)
    conv = acc.to(tl.bfloat16)

    val = conv.to(tl.float32)
    if APPLY_SILU:
        # PyTorch silu_kernel 的 bf16 opmath 形式：x / (1 + exp(-x))。除法用
        # div_rn（IEEE round-nearest）；`a / b` 在 Triton 里会落到 div.full.f32，
        # 与 reference 的 div.rn.f32 差 1 ulp —— bf16 舍入后几乎不可见，但没必要
        # 留这个差。
        val = libdevice.div_rn(val, 1.0 + libdevice.exp(-val))
    res = val.to(tl.bfloat16)

    if EVEN:
        tl.store(out_ptr + c, res)
        # 新 state 就是同一个窗口 [s1, s2, s3, x]：原地写回，地址不变。
        tl.store(state_ptr + c[:, None] * K + j[None, :], win)
    else:
        tl.store(out_ptr + c, res, mask=cmask)
        tl.store(
            state_ptr + c[:, None] * K + j[None, :], win, mask=cmask[:, None]
        )


def gdn_conv_decode_step(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str | None = "silu",
    out: torch.Tensor | None = None,
    block: int = DEFAULT_BLOCK,
    num_warps: int = DEFAULT_NUM_WARPS,
) -> torch.Tensor:
    """单层、单 token 的 GDN conv step（原地更新 ``conv_state``）。

    Args:
        hidden_states: BF16 ``[1, conv_dim, 1]``，即 ``in_proj_qkv`` 的输出
            transpose 之后、conv 之前的 ``mixed_qkv``；channel 维 stride 必须是 1。
        conv_state: BF16 ``[1, conv_dim, state_len]`` contiguous，**原地**滚动
            （dtype/字节/地址不变）。``state_len`` 必须等于 conv kernel size 且是
            2 的幂（本模型 4），否则调用方应回退到 torch 实现。
        weight: BF16 ``[conv_dim, kernel]``（``conv1d.weight.squeeze(1)``），
            contiguous。
        activation: ``"silu"`` 或 ``None``。
        out: 可选 BF16 ``[1, conv_dim, 1]`` 输出槽。默认 **就地写回**
            ``hidden_states``（``in_proj_qkv`` 的输出只被 conv 消费，模型随后把
            返回值赋给同一个变量，所以就地覆盖不改变任何可观察语义，也避免 graph
            里多一个固定地址的槽）。要给独立缓冲区时显式传 ``out``。
        block / num_warps: 见 ``BLOCK_CONFIGS``。

    Returns:
        BF16 ``[1, conv_dim, 1]``（``out`` 或 ``hidden_states`` 本身）。
    """
    if not hidden_states.is_cuda:
        raise ValueError("hidden_states 必须在 CUDA 上")
    if activation not in _ACTIVATIONS:
        raise ValueError(f"不支持的 activation {activation!r}，只支持 {_ACTIVATIONS}")
    if hidden_states.dtype is not torch.bfloat16:
        raise ValueError(f"hidden_states 必须是 bfloat16，得到 {hidden_states.dtype}")
    if hidden_states.dim() != 3 or hidden_states.shape[0] != 1 or hidden_states.shape[-1] != 1:
        raise ValueError(f"hidden_states 必须是 [1, C, 1]，得到 {tuple(hidden_states.shape)}")
    if hidden_states.stride(-2) != 1:
        raise ValueError(f"hidden_states 的 channel 维必须 stride=1，得到 {hidden_states.stride()}")
    if conv_state.dtype is not torch.bfloat16 or not conv_state.is_cuda:
        raise ValueError("conv_state 必须是 CUDA bfloat16")
    if conv_state.dim() != 3 or conv_state.shape[0] != 1:
        raise ValueError(f"conv_state 必须是 [1, C, state_len]，得到 {tuple(conv_state.shape)}")
    if not conv_state.is_contiguous():
        raise ValueError("conv_state 必须 contiguous")
    if weight.dtype is not torch.bfloat16 or not weight.is_cuda:
        raise ValueError("weight 必须是 CUDA bfloat16")
    if weight.dim() != 2 or not weight.is_contiguous():
        raise ValueError(f"weight 必须是 contiguous [C, K]，得到 {tuple(weight.shape)}")

    conv_dim = int(hidden_states.shape[1])
    state_len = int(conv_state.shape[-1])
    kernel_size = int(weight.shape[-1])
    if int(conv_state.shape[1]) != conv_dim or int(weight.shape[0]) != conv_dim:
        raise ValueError("conv_dim 在 hidden_states / conv_state / weight 之间不一致")
    if state_len != kernel_size:
        raise ValueError(
            f"state_len {state_len} != conv kernel {kernel_size}；"
            "本 kernel 只支持 state_len == kernel（窗口 == 新 state）"
        )
    if state_len < 2 or state_len & (state_len - 1) != 0:
        raise ValueError(f"state_len {state_len} 必须是 >=2 的 2 的幂")
    if conv_dim < block:
        raise ValueError(f"conv_dim {conv_dim} < block {block}")

    if out is None:
        out = hidden_states
    elif (
        out.dtype is not torch.bfloat16
        or out.shape != hidden_states.shape
        or not out.is_contiguous()
    ):
        raise ValueError("out 必须是 contiguous bf16 [1, C, 1]")

    grid = (triton.cdiv(conv_dim, block),)
    _gdn_conv_decode_kernel[grid](
        hidden_states,
        conv_state,
        weight,
        out,
        C=conv_dim,
        K=state_len,
        BLOCK=block,
        EVEN=(conv_dim % block == 0),
        APPLY_SILU=(activation == "silu"),
        num_warps=num_warps,
    )
    return out


__all__ = [
    "BLOCK_CONFIGS",
    "DEFAULT_BLOCK",
    "DEFAULT_NUM_WARPS",
    "gdn_conv_decode_step",
]
