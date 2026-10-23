"""NVFP4Linear：用 FlashInfer ``mm_fp4`` 实现的 W4A4 Linear。

设计：
- 权重在加载时保持 checkpoint 布局：U8 [N, K/2] + F8 [N, K/16] + F32 global；
- 构造时把 weight_scale 转成 128x4 swizzled 布局，并且只保留 swizzled 这一份，
  避免全模型多占约 1.6 GB 显存；
- forward：BF16 activation -> FlashInfer ``nvfp4_quantize`` -> ``mm_fp4`` -> BF16；
- **M=1 decode 的 per-shape backend 选型**（Phase 2 任务 F）：``mm_fp4`` 在不同
  shape 上效率差 1.5×，按 (N, K) 查一张模块级静态表，命中则用表里的 backend
  （当前全填 ``cute-dsl``），未命中 / 不可用回退 ``self.backend``（cutlass）。
  只在 M==1 生效；prefill（M>1）继续走 cutlass，保持 M2 冻结口径的 bitwise 语义。
  依据见 ``DECODE_BACKEND_TABLE`` 上方的注释（job-000035）。

数据流：
    x [M, K] BF16
      -> nvfp4_quantize(x, 1/input_scale) -> a_fp4 [M, K/2], a_sf [padM, K/16]
      -> mm_fp4(a_fp4, w_fp4.T, a_sf, w_sf_swizzled.T, alpha=input_scale*w_scale_2)
      -> y [M, N] BF16

预期瓶颈：M=1 时是权重带宽受限；M2 只做 correctness，性能在 M3/Phase 2。

环境要求：FlashInfer JIT 需要 nvcc 能找到 CUDA 头文件。运行时设置
``CUDA_HOME`` 和 ``CPATH``（含 ``nvidia/cu13/include``），见
``scripts/m2_env.sh`` / ``scripts/m2_env.fish``。
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import torch
from torch import nn

from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

from .reference import swizzle_scale_128x4

# ---------------------------------------------------------------- decode 选型
# M=1 decode 的 per-shape backend 静态表：(N, K) -> backend。
#
# 依据 2026-10-23 job-000035-461c30c4（bench/micro/nvfp4_tactics.py）：
# - 11 个 case 覆盖 checkpoint 全部 401 个 NVFP4 Linear（14.399 GB），每个 case 扫
#   全部真实实例、工作集 >=4x L2、CUDA graph replay 计时；
# - cute-dsl 11/11 全胜：加权 870 GB/s vs cutlass 760 GB/s（18.94 -> 16.55 ms/step，
#   -2.39 ms）；分 shape 870/836/843/504/829/832/882/882/884/911 GB/s；
# - 与 cutlass 输出**逐位一致**（11/11 nrmse=0.0、max_abs=0.0），所以换 backend
#   不改变数值语义；auto 在 SM120 上等于 b12x（更慢），cudnn 全面更慢。
#
# 口径：只在 M==1 查表；prefill（M>1）继续用 self.backend（checkpoint 的 cutlass）。
# 表按 shape 写全，未命中或 backend 抛错时回退 cutlass（每个进程只禁用一次）。
DECODE_BACKEND_TABLE: dict[tuple[int, int], str] = {
    (10240, 5120): "cute-dsl",   # in_proj_qkv x48
    (6144, 5120): "cute-dsl",    # in_proj_z x48
    (12288, 5120): "cute-dsl",   # q_proj x16
    (1024, 5120): "cute-dsl",    # k_proj / v_proj x32
    (5120, 6144): "cute-dsl",    # o_proj x16 + out_proj x48
    (17408, 5120): "cute-dsl",   # gate_proj / up_proj x128
    (5120, 17408): "cute-dsl",   # down_proj x64
    (248320, 5120): "cute-dsl",  # lm_head x1
    # M=1 decode 的合并调用（DecodePair）：两半共享同一 hidden_states + input_scale，
    # 合成一次 N=N1+N2 的 mm_fp4（128 列 tile 的 CTA 数翻倍，少一次 launch）。
    (2048, 5120): "cute-dsl",    # k_proj + v_proj（16 层）
    (16384, 5120): "cute-dsl",   # in_proj_qkv + in_proj_z（48 层）
    (34816, 5120): "cute-dsl",   # gate_proj + up_proj（64 层）
}

# 进程内的运行计数：给 check/bench 产物提供「JIT 编译只发生一次」的证据。
_DECODE_STATS: dict[str, Any] = {
    "calls": {},          # backend -> M=1 调用次数
    "first_call_ms": {},  # "NxK" -> 首次调用 wall ms（含 JIT 编译）
    "selected": {},       # "NxK" -> 实际使用的 backend
    "merged_calls": {},   # "N1+N2xK" -> 合并调用次数
    "merge_fallbacks": 0,
    "fallbacks": 0,
}
_DECODE_BACKEND_DISABLED: set[str] = set()


def decode_backend_for(n: int, k: int) -> str:
    """(N, K) -> decode backend；未命中返回 cutlass。"""
    return DECODE_BACKEND_TABLE.get((int(n), int(k)), "cutlass")


def decode_backend_info() -> dict[str, Any]:
    """静态表 + 依据，写进载入信息（``load`` diag / harness）。"""
    return {
        "table": {f"{n}x{k}": backend for (n, k), backend in DECODE_BACKEND_TABLE.items()},
        "scope": "只在 M==1（decode）查表；prefill（M>1）继续走 self.backend=cutlass",
        "fallback": "表未命中或 backend 抛错 -> cutlass（每个进程只禁用一次）",
        "source": (
            "2026-10-23 job-000035-461c30c4 / bench/micro/nvfp4_tactics.py："
            "11 个 shape = 401 个 NVFP4 Linear，L2-free 真实实例聚合 870 GB/s vs "
            "cutlass 760 GB/s（18.94 -> 16.55 ms/step）；cute-dsl 与 cutlass 输出逐位一致"
        ),
        # 故意是同一个 dict 的引用：check/bench 产物在最后序列化时能看到全程计数。
        "counters": _DECODE_STATS,
    }


def decode_backend_stats() -> dict[str, Any]:
    """当前进程的 decode backend 计数快照（给 harness / journal 用）。"""
    return {
        "calls": dict(_DECODE_STATS["calls"]),
        "first_call_ms": dict(_DECODE_STATS["first_call_ms"]),
        "selected": dict(_DECODE_STATS["selected"]),
        "merged_calls": dict(_DECODE_STATS["merged_calls"]),
        "merge_fallbacks": int(_DECODE_STATS["merge_fallbacks"]),
        "fallbacks": int(_DECODE_STATS["fallbacks"]),
        "disabled": sorted(_DECODE_BACKEND_DISABLED),
    }


class DecodePair:
    """把两个共享同一 ``hidden_states`` 的 NVFP4Linear 合并成一次 M=1 mm_fp4。

    为什么安全（2026-10-23 的 checkpoint 事实，见 journal）：
    - 三对模块（k/v、gate/up、in_proj_qkv/in_proj_z）在同一层里**共享同一个
      hidden_states**，且 pair 内 ``input_scale`` 与 ``weight_scale_2`` **逐位相同**
      （k/v 16/16、gate/up 64/64、qkv/z 48/48），所以两边量化出的 A 与 alpha 完全一致；
    - 合并权重 = 两半的 ``weight_packed`` 沿 N 拼接，scale = 两半线性 scale 拼接后
      统一 swizzle（N 都是 128 的倍数，拼接与 swizzle 可交换；op 里有逐字节 gate）；
    - 每个输出行只依赖自己那行权重，K 方向归约顺序与单调用相同，所以**逐位一致**。

    数据流（M=1）：
        leader: [a_fp4, a_sf] + W[N1+N2, K/2] + swizzled SF -> out[1, N1+N2]
                -> 返回 out[:, :N1]，把 out[:, N1:] 留给 follower
        follower: 校验 (x.data_ptr, numel) 后直接返回 out[:, N1:]（不再 launch）

    leader 用自己（=follower）的 ``input_scale`` 量化 A；若 norm+quant 融合已经给了
    ``_pending_quantized``，直接用那一份。follower 的 pending 槽位会被消费掉，
    保证 ``norm_quant_patch`` 的 ``unconsumed`` 计数保持 0。

    失败（合并 kernel 抛错）时整对永久禁用并回退到各自路径，不静默算错。
    """

    __slots__ = (
        "leader",
        "follower",
        "weight_packed",
        "weight_scale_swizzled",
        "alpha",
        "out_dtype",
        "n_leader",
        "n_follower",
        "backend",
        "out",
        "armed",
        "x_ptr",
        "x_numel",
        "disabled",
    )

    def __init__(
        self,
        leader: "NVFP4Linear",
        follower: "NVFP4Linear",
        weight_packed: torch.Tensor,
        weight_scale_swizzled: torch.Tensor,
    ) -> None:
        self.leader = leader
        self.follower = follower
        self.weight_packed = weight_packed
        self.weight_scale_swizzled = weight_scale_swizzled
        self.alpha = leader.alpha
        self.out_dtype = leader.out_dtype
        self.n_leader = leader.out_features
        self.n_follower = follower.out_features
        self.backend = decode_backend_for(
            self.n_leader + self.n_follower, leader.in_features
        )
        self.out = torch.empty(
            (1, self.n_leader + self.n_follower),
            dtype=self.out_dtype,
            device=leader.weight_packed.device,
        )
        self.armed = False
        self.x_ptr = 0
        self.x_numel = 0
        self.disabled = False

    # ------------------------------------------------------------------ 调用
    def forward(self, module, x, pending):
        """返回本模块的 [1, N] 输出；``None`` 表示走普通路径。"""
        if self.disabled:
            return None
        if module is self.follower:
            if x.numel() != x.shape[-1]:
                return None
            if (
                self.armed
                and self.x_ptr == x.data_ptr()
                and self.x_numel == x.numel()
            ):
                self.armed = False
                return self.out[:, self.n_leader :]
            _DECODE_STATS["merge_fallbacks"] += 1
            return None

        # leader：只处理单 token decode（M==1）。
        if x.numel() != x.shape[-1] or x.shape[-1] != self.leader.in_features:
            return None
        self.armed = False
        if pending is not None:
            a_fp4, a_sf = pending
        else:
            xx = x.reshape(-1, self.leader.in_features)
            if not xx.is_contiguous():
                xx = xx.contiguous()
            if xx.dtype != torch.bfloat16:
                xx = xx.to(torch.bfloat16)
            a_fp4, a_sf = nvfp4_quantize(
                xx,
                (1.0 / self.leader.input_scale).reshape(1),
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
                backend="cuda",
            )
        try:
            out = mm_fp4(
                a_fp4,
                self.weight_packed.T,
                a_sf,
                self.weight_scale_swizzled.T,
                self.alpha,
                self.out_dtype,
                self.out,
                block_size=16,
                use_8x4_sf_layout=False,
                backend=self.backend,
                use_nvfp4=True,
            )
        except Exception as exc:  # noqa: BLE001 - 整对永久回退，不静默算错
            self.disabled = True
            _DECODE_STATS["merge_fallbacks"] += 1
            warnings.warn(
                f"DecodePair {self.n_leader}+{self.n_follower} "
                f"(K={self.leader.in_features}) 合并失败，回退分开调用：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        self.x_ptr = x.data_ptr()
        self.x_numel = x.numel()
        self.armed = True
        label = f"{self.n_leader}+{self.n_follower}x{self.leader.in_features}"
        _DECODE_STATS["merged_calls"][label] = (
            _DECODE_STATS["merged_calls"].get(label, 0) + 1
        )
        return out[:, : self.n_leader]


def attach_decode_pair(leader: "NVFP4Linear", follower: "NVFP4Linear") -> bool:
    """把 leader/follower 合成一个 ``DecodePair``（M=1 生效；prefill 不变）。

    条件不满足返回 False（不报错）：同 in_features / 无 bias / 两个 scalar 逐位相同 /
    N 都是 128 的倍数（保证 scale 拼接 == 拼接后 swizzle）。成功时两半的
    ``weight_packed`` / ``weight_scale_swizzled`` 变成同一块 buffer 的视图，
    稳态显存不增加（拼接的瞬时峰值是每对一份）。
    """
    if not isinstance(leader, NVFP4Linear) or not isinstance(follower, NVFP4Linear):
        return False
    if leader is follower or leader.in_features != follower.in_features:
        return False
    if leader.bias is not None or follower.bias is not None:
        return False
    if leader.out_dtype != follower.out_dtype:
        return False
    if not torch.equal(leader.input_scale, follower.input_scale):
        return False
    if not torch.equal(leader.weight_scale_2, follower.weight_scale_2):
        return False
    n1, n2 = leader.out_features, follower.out_features
    if n1 % 128 or n2 % 128:
        return False
    if leader.weight_scale_swizzled.shape[0] != n1:
        return False
    if follower.weight_scale_swizzled.shape[0] != n2:
        return False
    weight = torch.cat([leader.weight_packed, follower.weight_packed], dim=0)
    scale = torch.cat(
        [leader.weight_scale_swizzled, follower.weight_scale_swizzled], dim=0
    )
    pair = DecodePair(leader, follower, weight, scale)
    leader.weight_packed = weight[:n1]
    leader.weight_scale_swizzled = scale[:n1]
    follower.weight_packed = weight[n1:]
    follower.weight_scale_swizzled = scale[n1:]
    leader._decode_pair = pair
    follower._decode_pair = pair
    return True


def install_decode_pairs(model: nn.Module) -> dict[str, Any]:
    """遍历 decoder layers，把三对共享 hidden_states 的 Linear 配成 DecodePair。

    返回计数（写进 load info / harness）；配不上的对跳过（``skipped``），
    不影响原有 per-module 路径。
    """
    stats: dict[str, Any] = {"pairs": 0, "skipped": 0, "kinds": {}}
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        return stats
    specs = (
        ("self_attn", "k_proj", "v_proj"),
        ("mlp", "gate_proj", "up_proj"),
        ("linear_attn", "in_proj_qkv", "in_proj_z"),
    )
    for layer in layers:
        for holder_name, leader_name, follower_name in specs:
            holder = getattr(layer, holder_name, None)
            if holder is None:
                continue
            leader = getattr(holder, leader_name, None)
            follower = getattr(holder, follower_name, None)
            if attach_decode_pair(leader, follower):
                stats["pairs"] += 1
                key = f"{leader_name}+{follower_name}"
                stats["kinds"][key] = stats["kinds"].get(key, 0) + 1
            else:
                stats["skipped"] += 1
    return stats


class NVFP4Linear(nn.Module):
    """替换 checkpoint 里的量化 Linear。

    Args:
        weight_packed: U8 [N, K/2]，checkpoint 的 ``weight``。
        weight_scale: F8_E4M3 [N, K/16]，checkpoint 的 ``weight_scale``。
        weight_scale_2: 标量 F32，checkpoint 的 ``weight_scale_2``。
        input_scale: 标量 F32，checkpoint 的 ``input_scale``。
        bias: 可选 BF16 [N]；M2 的 checkpoint 大多没有 bias。
        backend: prefill / 回退用的 FlashInfer backend；M2 固定 ``cutlass``。
        out_dtype: 输出 dtype，默认 BF16。
        decode_backend: M=1 decode 的 backend 覆盖。``None`` 表示查静态表
            ``DECODE_BACKEND_TABLE``（当前全填 cute-dsl）；显式传 ``"cutlass"``
            可关掉选型、完全回到 M2 路径。
    """

    def __init__(
        self,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        input_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        backend: str = "cutlass",
        out_dtype: torch.dtype = torch.bfloat16,
        decode_backend: str | None = None,
    ) -> None:
        super().__init__()
        if weight_packed.ndim != 2 or weight_packed.dtype != torch.uint8:
            raise TypeError(
                f"weight_packed must be uint8 [N, K/2], got {weight_packed.dtype} {tuple(weight_packed.shape)}"
            )
        if weight_scale.ndim != 2 or weight_scale.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"weight_scale must be float8_e4m3fn [N, K/16], got {weight_scale.dtype} {tuple(weight_scale.shape)}"
            )
        n, k_half = weight_packed.shape
        k = k_half * 2
        if weight_scale.shape != (n, k // 16):
            raise ValueError(
                f"weight_scale shape {tuple(weight_scale.shape)} != {(n, k // 16)}"
            )

        self.in_features = k
        self.out_features = n
        self.backend = backend
        self.decode_backend = decode_backend
        self.out_dtype = out_dtype

        self.register_buffer("weight_packed", weight_packed.contiguous())
        self.register_buffer("weight_scale_2", weight_scale_2.reshape(1).float())
        self.register_buffer("input_scale", input_scale.reshape(1).float())
        # 只保留 swizzled scale；原始 linear scale 在构造后释放，避免双份显存。
        self.register_buffer(
            "weight_scale_swizzled",
            swizzle_scale_128x4(weight_scale.contiguous()),
        )
        self.register_buffer("bias", None if bias is None else bias.contiguous())
        # alpha = input_scale * weight_scale_2 是模型常量：构造时算一次，省掉每次
        # forward 的一个小 kernel（f32 乘法，与逐次计算结果逐位相同）。
        self.register_buffer(
            "alpha", (self.input_scale * self.weight_scale_2).reshape(1).float()
        )
        # 融合路径的单次槽位：``norm_quant_patch`` 在调用 mixer 之前填入
        # (a_fp4, a_sf)，``forward`` 取走并清空；平时恒为 None。
        self._pending_quantized: tuple[torch.Tensor, torch.Tensor] | None = None
        # M=1 合并调用的 pair（loader 里 ``attach_decode_pair`` 设置）；prefill 不用。
        self._decode_pair: DecodePair | None = None

    # ------------------------------------------------------------------ 内核
    def _mm(
        self,
        a_fp4: torch.Tensor,
        a_sf: torch.Tensor,
        backend: str | None = None,
    ) -> torch.Tensor:
        """已量化好的 A + 128x4 swizzled SF -> [M, N] BF16（不含 bias/reshape）。"""
        return mm_fp4(
            a_fp4,
            self.weight_packed.T,
            a_sf,
            self.weight_scale_swizzled.T,
            self.alpha,
            self.out_dtype,
            None,
            block_size=16,
            use_8x4_sf_layout=False,
            backend=self.backend if backend is None else backend,
            use_nvfp4=True,
        )

    def _select_backend(self, m: int) -> str:
        """M==1（decode）按 (N,K) 查静态表；M>1（prefill）保持 self.backend。

        表只对本 checkpoint 的 SM120 有效（job-000035 的 870 vs 760 GB/s 依据见
        ``DECODE_BACKEND_TABLE``）。``decode_backend`` 显式覆盖优先，其次是查表。
        """
        if m != 1:
            return self.backend
        if self.decode_backend is not None:
            chosen = self.decode_backend
        elif self.backend == "cutlass":
            chosen = decode_backend_for(self.out_features, self.in_features)
        else:
            chosen = self.backend
        if chosen == self.backend or chosen in _DECODE_BACKEND_DISABLED:
            return self.backend
        return chosen

    def _gemm(self, a_fp4: torch.Tensor, a_sf: torch.Tensor, m: int) -> torch.Tensor:
        """量化好的 A -> [m, N]。decode 走选型 backend，失败则永久回退。"""
        backend = self._select_backend(m)
        if backend == self.backend:
            return self._mm(a_fp4, a_sf)
        label = f"{self.out_features}x{self.in_features}"
        # 首次调用包含 JIT 编译；只在非 capture 路径计时（warmup 在 capture 之前）。
        first = label not in _DECODE_STATS["first_call_ms"]
        measure = bool(first) and not torch.cuda.is_current_stream_capturing()
        if measure:
            torch.cuda.synchronize()
            started = time.perf_counter()
        try:
            out = self._mm(a_fp4, a_sf, backend)
        except Exception as exc:  # noqa: BLE001 - 不可用则永久回退，不静默失败
            _DECODE_BACKEND_DISABLED.add(backend)
            _DECODE_STATS["fallbacks"] += 1
            warnings.warn(
                f"NVFP4Linear: decode backend {backend!r} failed on "
                f"{label}, falling back to {self.backend!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            backend = self.backend
            out = self._mm(a_fp4, a_sf)
        if measure:
            torch.cuda.synchronize()
            _DECODE_STATS["first_call_ms"][label] = (
                time.perf_counter() - started
            ) * 1e3
        _DECODE_STATS["calls"][backend] = _DECODE_STATS["calls"].get(backend, 0) + 1
        _DECODE_STATS["selected"][label] = backend
        return out

    def _finish(self, out: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        if self.bias is not None:
            out = out + self.bias
        if len(shape) != 2:
            out = out.reshape(*shape[:-1], self.out_features)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pending, self._pending_quantized = self._pending_quantized, None
        if x.ndim < 2 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"x must be [..., {self.in_features}], got {tuple(x.shape)}"
            )
        # M=1 的合并调用（k/v、gate/up、in_proj_qkv/z）：一次 mm_fp4 出两半。
        # 返回 None 表示这一半走普通路径（prefill / pair 未命中 / 合并失败）。
        if self._decode_pair is not None:
            paired = self._decode_pair.forward(self, x, pending)
            if paired is not None:
                return self._finish(paired, x.shape)

        if pending is not None:
            # 输入已经由融合 kernel 量化过（同一 norm 输出的每个消费者各有一份）。
            # 融合路径只在 decode（M==1）生效，所以这里同样要按 decode 选型挑 backend。
            return self._finish(self._gemm(pending[0], pending[1], 1), x.shape)

        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        if not x.is_contiguous():
            x = x.contiguous()
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # checkpoint 的 input_scale 是反量化乘数；量化用它的倒数作为 global divisor。
        global_divisor = (1.0 / self.input_scale).reshape(1)
        a_fp4, a_sf = nvfp4_quantize(
            x,
            global_divisor,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )
        return self._finish(self._gemm(a_fp4, a_sf, x.shape[0]), original_shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"backend={self.backend!r}, decode_backend={self.decode_backend!r}"
        )
