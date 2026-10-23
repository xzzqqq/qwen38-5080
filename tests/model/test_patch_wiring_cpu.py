"""patch 接线的 CPU-only 守门测试（``CUDA_VISIBLE_DEVICES=""`` 必须全绿）。

为什么需要它
============
op gate 只测 kernel，``m3_check`` 的 bitwise gate 只比较执行方式 —— 两者都看不见
「patch 到底有没有被装上 / 配对逻辑是不是在 eager 下静默漏调」。第 4 轮的
job-000055 就是这样：op **222/222** 全绿，check 的 ``numeric-512`` 却在
``load_nvfp4_text_model`` 里 OOM（``install_decode_pairs`` 的 ``torch.cat`` 与
``dict(model.named_buffers())`` 的引用叠加，峰值 2 倍），eager 快照一条都没跑，
report.json 只剩 ``coverage: 必测项齐全`` 的 missing。这里用**纯 CPU** 把配对逻辑、
回退路径、计数器与几条历史 bug 一起钉死，不花 GPU、不建 CUDA context。

覆盖
====
1. 四种执行方式（eager / static-exact / static-cap / graph）的调用序列都能走通，
   且每一次 ``mm_fp4`` 都是预期的「合并 / 分开」宽度（不静默漏调）；
2. 前置条件不满足（顺序不对、data_ptr 不同、seq_len>1、配对数不匹配）时**整对回退**
   且计数正确；
3. 历史 bug 清单回归：3 槽 scratch 传给 2 消费者、gated 激活的操作数写反、
   GDN head-count 用 repeat 前的形状。

注意：真实 ``mm_fp4`` / ``nvfp4_quantize`` 在这里被 CPU 替身替换（只保留「每个
输出列只依赖自己那列权重」这一条性质，用来验证合并后切半与分开调用逐位一致），
所以本文件不验证数值正确性 —— 那是 op gate 与 m3_check 的职责。
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from runner.ops.nvfp4 import linear as L
from runner.ops.nvfp4.reference import swizzle_scale_128x4

K = 256
N1 = 128
N2 = 128
MERGED_WIDTH = N1 + N2
MODES = ("eager", "static-exact", "static-cap", "graph")


# --------------------------------------------------------------------- 替身
class _FakeMM:
    """CPU 版 ``mm_fp4``：out[0, j] = sum_k W[j, k]（只依赖自己那一列）。

    真 kernel 的每个输出行也只依赖自己那行权重，所以「合并调用后切半」与「分开两次
    调用」在这里应当**逐位相等**，可用来验证 DecodePair 的两半没有接反。
    """

    def __init__(self) -> None:
        self.widths: list[int] = []
        self.fail_on_width: int | None = None

    def __call__(self, a_fp4, wT, a_sf, w_sf, alpha, out_dtype, out, **kwargs):
        width = int(wT.shape[1])
        if self.fail_on_width is not None and width == self.fail_on_width:
            raise RuntimeError("fake mm_fp4 failure")
        self.widths.append(width)
        rows = int(a_fp4.shape[0])
        values = wT.to(torch.float32).sum(dim=0).reshape(1, width)
        result = values.repeat(rows, 1).to(out_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result


class _FakeQuantize:
    """CPU 版 ``nvfp4_quantize``：只记录 shape，占位数据交给替身 mm 忽略。"""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, x, divisor, **kwargs):
        self.shapes.append(tuple(x.shape))
        rows, k = int(x.shape[0]), int(x.shape[-1])
        a_fp4 = torch.zeros(rows, k // 2, dtype=torch.uint8)
        a_sf = torch.zeros(128, k // 16, dtype=torch.uint8)
        return a_fp4, a_sf


@pytest.fixture(autouse=True)
def fake_kernels(monkeypatch):
    """替换 kernel + 重置进程内计数 + 禁止任何 CUDA 探测。"""
    mm = _FakeMM()
    quant = _FakeQuantize()
    monkeypatch.setattr(L, "mm_fp4", mm)
    monkeypatch.setattr(L, "nvfp4_quantize", quant)
    # 配对路径不允许依赖 capture 状态；本地若有人调用它，直接让测试失败。
    def _forbid_capture(*args, **kwargs):
        raise AssertionError("配对路径不应探测 CUDA capture 状态")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", _forbid_capture)
    _reset_stats()
    yield mm, quant
    _reset_stats()


def _reset_stats() -> None:
    L._DECODE_STATS.update(
        {
            "calls": {},
            "first_call_ms": {},
            "selected": {},
            "merged_calls": {},
            "merge_fallbacks": 0,
            "merge_fallback_reasons": {},
            "merge_skips": {},
            "fallbacks": 0,
        }
    )
    L._DECODE_BACKEND_DISABLED.clear()


# --------------------------------------------------------------------- 构造
def _weight(n: int, k: int, seed: int) -> torch.Tensor:
    base = torch.arange(n * (k // 2), dtype=torch.int64).reshape(n, k // 2)
    return ((base + seed) % 251).to(torch.uint8)


def _parts(
    n: int,
    *,
    k: int = K,
    seed: int = 3,
    input_scale: float = 0.5,
    weight_scale_2: float = 2.0,
) -> dict[str, torch.Tensor]:
    return {
        "weight": _weight(n, k, seed),
        "weight_scale": torch.full((n, k // 16), 1.0, dtype=torch.float8_e4m3fn),
        "weight_scale_2": torch.tensor([weight_scale_2], dtype=torch.float32),
        "input_scale": torch.tensor([input_scale], dtype=torch.float32),
    }


def _linear(n: int, *, seed: int = 3, input_scale: float = 0.5, **kwargs) -> L.NVFP4Linear:
    parts = _parts(n, seed=seed, input_scale=input_scale)
    return L.NVFP4Linear(
        parts["weight"],
        parts["weight_scale"],
        parts["weight_scale_2"],
        parts["input_scale"],
        backend="cutlass",
        **kwargs,
    )


def _pair(*, input_scale: float = 0.5) -> tuple[L.NVFP4Linear, L.NVFP4Linear]:
    built = L.build_decode_pair_from_checkpoint(
        _parts(N1, seed=3, input_scale=input_scale),
        _parts(N2, seed=11, input_scale=input_scale),
        device="cpu",
    )
    assert built is not None
    return built


def _split_reference(weight_packed: torch.Tensor) -> torch.Tensor:
    """分开调用时替身 mm 会给出的输出：输出 j = W[j, :] 的行和（1 行解码）。"""
    return weight_packed.to(torch.float32).sum(dim=1).reshape(1, -1).to(torch.bfloat16)


def _pending() -> tuple[torch.Tensor, torch.Tensor]:
    """模拟 ``norm_quant_patch`` 放进 ``_pending_quantized`` 的一对 A。"""
    return (
        torch.zeros(1, K // 2, dtype=torch.uint8),
        torch.zeros(128, K // 16, dtype=torch.uint8),
    )


# ----------------------------------------------------------------- 占位模型
class _Producer(nn.Module):
    """一个 norm 的替身：把同一份量化结果 arm 给所有消费者，再按顺序调用。

    消费者按真实 HF 模块的命名挂成属性（``q_proj`` / ``k_proj`` / ``v_proj`` /
    ``gate_proj`` / ``up_proj`` / ``in_proj_qkv`` / ``in_proj_z``），这样
    ``install_decode_pairs`` 能像遍历真模型一样找到它们。
    """

    def __init__(self, consumers: dict[str, L.NVFP4Linear], *, fused: bool) -> None:
        super().__init__()
        for name, module in consumers.items():
            setattr(self, name, module)
        self.order = tuple(consumers)
        self.fused = fused

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        modules = [getattr(self, name) for name in self.order]
        if self.fused:
            for consumer in modules:
                consumer._pending_quantized = _pending()
        try:
            return [consumer(x) for consumer in modules]
        finally:
            for consumer in modules:
                assert consumer._pending_quantized is None, "pending 槽位没有被消费"


class _PlaceholderLayer(nn.Module):
    """极小的 decoder layer 占位：``layer_types`` 里的两种 block 都覆盖。"""

    def __init__(self, block_type: str, *, fused: bool) -> None:
        super().__init__()
        self.block_type = block_type
        if block_type == "full_attention":
            # q 独立；k/v 是 DecodePair 的 leader/follower。
            self.self_attn = _Producer(
                {
                    "q_proj": _linear(N1, seed=5, input_scale=0.7),
                    "k_proj": _linear(N1, seed=3, input_scale=0.7),
                    "v_proj": _linear(N2, seed=11, input_scale=0.7),
                },
                fused=fused,
            )
        elif block_type == "linear_attention":
            self.linear_attn = _Producer(
                {
                    "in_proj_qkv": _linear(N1, seed=7, input_scale=0.7),
                    "in_proj_z": _linear(N2, seed=13, input_scale=0.7),
                },
                fused=fused,
            )
        else:
            raise ValueError(block_type)
        self.mlp = _Producer(
            {
                "gate_proj": _linear(N1, seed=17, input_scale=0.9),
                "up_proj": _linear(N2, seed=19, input_scale=0.9),
            },
            fused=fused,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        mixer = self.self_attn if self.block_type == "full_attention" else self.linear_attn
        return mixer(x) + self.mlp(x)


class _PlaceholderModel(nn.Module):
    """``layer_types`` 含 linear_attention 与 full_attention 的两层占位模型。"""

    layer_types = ("linear_attention", "full_attention")

    def __init__(self, *, fused: bool) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_PlaceholderLayer(kind, fused=fused) for kind in self.layer_types]
        )

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        return [layer(x) for layer in self.model.layers]

    def pairs(self) -> list[tuple[L.NVFP4Linear, L.NVFP4Linear]]:
        found: list[tuple[L.NVFP4Linear, L.NVFP4Linear]] = []
        for layer in self.model.layers:
            for module in _layer_modules(layer):
                pair = module._decode_pair
                if pair is not None and pair.leader is module:
                    found.append((pair.leader, pair.follower))
        return found


def _placeholder(fused: bool = True) -> _PlaceholderModel:
    model = _PlaceholderModel(fused=fused)
    stats = L.install_decode_pairs(model)
    assert stats["pairs"] == 4, stats  # 每层 mlp 一对 + 两种 block 各一对
    return model


# --------------------------------------------------- 1. 四种执行方式的调用序列
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fused", [True, False])
def test_four_execution_modes_wire_the_same_pairing(mode, fused, fake_kernels):
    """四种执行方式都走同一条配对逻辑：不抛、不漏调、两半不接反。"""
    mm, quant = fake_kernels
    model = _placeholder(fused=fused)
    pairs = model.pairs()
    assert len(pairs) == 4

    steps = 3
    if mode == "eager":
        # eager：每一步的 hidden_states 是新张量（DynamicCache 路径）。
        feeder = lambda: torch.zeros(1, 1, K, dtype=torch.bfloat16)  # noqa: E731
    else:
        # static-exact / static-cap / graph：复用同一块 buffer（地址固定）。
        buffer = torch.zeros(1, 1, K, dtype=torch.bfloat16)
        feeder = lambda: buffer  # noqa: E731

    for _ in range(steps):
        outputs = model(feeder())
        for layer_outputs in outputs:
            for out in layer_outputs:
                assert out.shape == (1, 1, N1) or out.shape == (1, 1, N2)

    # 每个 pair 每步恰有一次合并调用（4 对 × steps），没有任何回退。
    assert L._DECODE_STATS["merge_fallbacks"] == 0, L._DECODE_STATS
    # 4 对的 (N1+N2, K) 相同，因此计数按同一个 label 累加。
    assert sum(L._DECODE_STATS["merged_calls"].values()) == len(pairs) * steps
    # 具体宽度：q 分开(128)，其余走合并(256)。
    assert mm.widths.count(MERGED_WIDTH) == len(pairs) * steps
    assert mm.widths.count(N1) == steps  # 每步只有 q 这一次单独调用

    # 没有融合 norm 时，每个 leader/follower 各自量化一次（不静默复用错张量）。
    if fused:
        assert quant.shapes == []
    else:
        assert len(quant.shapes) == 5 * steps

    # 两半输出与「分开调用」逐位一致（配对没有把 N1/N2 接反）。
    for leader, follower in pairs:
        x = torch.zeros(1, 1, K, dtype=torch.bfloat16)
        assert torch.equal(
            leader(x).reshape(-1), _split_reference(leader.weight_packed).reshape(-1)
        )
        assert torch.equal(
            follower(x).reshape(-1), _split_reference(follower.weight_packed).reshape(-1)
        )


# --------------------------------------------------------- 2. 前置条件与回退
def test_merged_halves_are_bitwise_equal_to_split_calls(fake_kernels):
    leader, follower = _pair()
    x = torch.zeros(1, 1, K, dtype=torch.bfloat16)
    merged_leader = leader(x)
    merged_follower = follower(x)
    assert torch.equal(
        merged_leader.reshape(-1), _split_reference(leader.weight_packed).reshape(-1)
    )
    assert torch.equal(
        merged_follower.reshape(-1), _split_reference(follower.weight_packed).reshape(-1)
    )
    assert L._DECODE_STATS["merge_fallbacks"] == 0


def test_follower_without_leader_falls_back_to_shared_path(fake_kernels):
    mm, quant = fake_kernels
    leader, follower = _pair()
    x = torch.zeros(1, 1, K, dtype=torch.bfloat16)
    out = follower(x)  # 顺序不对：leader 没跑，follower 先被调用
    assert out.shape == (1, 1, N2)
    assert torch.equal(out.reshape(-1), _split_reference(follower.weight_packed).reshape(-1))
    assert L._DECODE_STATS["merge_fallbacks"] == 1
    assert L._DECODE_STATS["merge_fallback_reasons"] == {"order": 1}
    assert L._DECODE_STATS["merged_calls"] == {}
    assert len(quant.shapes) == 1 and mm.widths == [N2]


def test_pointer_mismatch_falls_back_to_shared_path(fake_kernels):
    mm, _ = fake_kernels
    leader, follower = _pair()
    x_leader = torch.zeros(1, 1, K, dtype=torch.bfloat16)
    x_follower = torch.zeros(1, 1, K, dtype=torch.bfloat16)
    assert x_leader.data_ptr() != x_follower.data_ptr()
    assert leader(x_leader).shape == (1, 1, N1)
    out = follower(x_follower)  # 不同的输入张量：不能拿 leader 的合并结果
    assert out.shape == (1, 1, N2)
    assert L._DECODE_STATS["merge_fallbacks"] == 1
    assert L._DECODE_STATS["merge_fallback_reasons"] == {"ptr": 1}
    # leader 的合并调用已经发生，follower 回退后自己再算一次。
    assert mm.widths == [MERGED_WIDTH, N2]


def test_seq_len_gt1_skips_the_pair_without_counting_as_fallback(fake_kernels):
    mm, quant = fake_kernels
    leader, follower = _pair()
    x = torch.zeros(1, 4, K, dtype=torch.bfloat16)
    assert leader(x).shape == (1, 4, N1)
    assert follower(x).shape == (1, 4, N2)
    assert L._DECODE_STATS["merge_skips"] == {"seq_len": 1}
    assert L._DECODE_STATS["merge_fallbacks"] == 0
    assert L._DECODE_STATS["merged_calls"] == {}
    assert mm.widths == [N1, N2]
    assert quant.shapes == [(4, K), (4, K)]


def test_merged_kernel_error_disables_the_whole_pair(fake_kernels):
    mm, _ = fake_kernels
    leader, follower = _pair()
    mm.fail_on_width = MERGED_WIDTH
    x = torch.zeros(1, 1, K, dtype=torch.bfloat16)
    with pytest.warns(RuntimeWarning):
        out_leader = leader(x)
        out_follower = follower(x)
    assert out_leader.shape == (1, 1, N1)
    assert out_follower.shape == (1, 1, N2)
    assert torch.equal(out_leader.reshape(-1), _split_reference(leader.weight_packed).reshape(-1))
    assert torch.equal(
        out_follower.reshape(-1), _split_reference(follower.weight_packed).reshape(-1)
    )
    assert L._DECODE_STATS["merge_fallback_reasons"] == {"error": 1, "disabled": 1}
    # 再走一步：整对保持 disabled，两个模块都走共享路径，不再尝试合并。
    leader(x)
    follower(x)
    assert L._DECODE_STATS["merged_calls"] == {}
    assert L._DECODE_STATS["merge_fallback_reasons"] == {"error": 1, "disabled": 3}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda parts: parts.update(input_scale=torch.tensor([0.75])),
        lambda parts: parts.update(weight_scale_2=torch.tensor([3.0])),
        lambda parts: parts.update(weight=_weight(64, K, 3)),
        lambda parts: parts.update(weight=_weight(N1, K + 2, 3)),
        lambda parts: parts.update(weight_scale=torch.zeros(N1, K // 16, dtype=torch.uint8)),
        lambda parts: parts.update(weight=_weight(N1 - 1, K, 3)),
    ],
    ids=[
        "input_scale",
        "weight_scale_2",
        "n-not-multiple-128",
        "k-mismatch",
        "scale-dtype",
        "weight-shape",
    ],
)
def test_build_rejects_unsatisfied_preconditions(mutate):
    """任何一条前置条件不满足 -> 返回 None（调用方分别装载，不抛异常）。"""
    leader = _parts(N1, seed=3)
    follower = _parts(N2, seed=11)
    mutate(follower)
    assert L.build_decode_pair_from_checkpoint(leader, follower, device="cpu") is None


def test_install_skips_missing_half_and_counts_reason(fake_kernels):
    """配对数不匹配（只有 leader 没有 follower）：跳过并计数，不影响别的对。"""
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.k_proj = _linear(N1)
    layer.mlp = nn.Module()
    layer.mlp.gate_proj = _linear(N1, seed=17)
    layer.mlp.up_proj = _linear(N2, seed=19)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([layer])
    stats = L.install_decode_pairs(model)
    assert stats["pairs"] == 1
    assert stats["skipped"] == 1
    assert stats["reasons"] == {"type": 1}
    assert layer.mlp.gate_proj._decode_pair is not None


def test_install_is_idempotent_and_shares_storage(fake_kernels):
    model = _placeholder()
    pairs = model.pairs()
    assert len(pairs) == 4
    for leader, follower in pairs:
        assert (
            leader.weight_packed.untyped_storage().data_ptr()
            == follower.weight_packed.untyped_storage().data_ptr()
        )
        assert (
            leader.weight_scale_swizzled.untyped_storage().data_ptr()
            == follower.weight_scale_swizzled.untyped_storage().data_ptr()
        )
    # 第二次 install 不重复分配、也不重复计数。
    before = {
        id(module): module.weight_packed.untyped_storage().data_ptr()
        for layer in model.model.layers
        for module in _layer_modules(layer)
    }
    stats = L.install_decode_pairs(model)
    assert stats["pairs"] == 4
    assert stats["skipped"] == 0
    assert stats["reasons"] == {}
    after = {
        id(module): module.weight_packed.untyped_storage().data_ptr()
        for layer in model.model.layers
        for module in _layer_modules(layer)
    }
    assert before == after


def _layer_modules(layer: nn.Module) -> list[L.NVFP4Linear]:
    holder = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn")
    modules = [getattr(holder, name) for name in holder.order]
    return modules + [getattr(layer.mlp, name) for name in layer.mlp.order]


def test_decode_pair_specs_only_pairs_known_siblings():
    specs = L.decode_pair_specs(
        [
            "x.layers.0.self_attn.q_proj",
            "x.layers.0.self_attn.k_proj",
            "x.layers.0.self_attn.v_proj",
            "x.layers.0.mlp.gate_proj",
            "x.layers.0.mlp.up_proj",
            "x.layers.0.mlp.down_proj",
            "x.layers.0.linear_attn.in_proj_a",
            "x.layers.0.linear_attn.in_proj_qkv",
            "x.layers.0.linear_attn.in_proj_z",
        ]
    )
    assert specs == {
        "x.layers.0.self_attn.k_proj": "x.layers.0.self_attn.v_proj",
        "x.layers.0.mlp.gate_proj": "x.layers.0.mlp.up_proj",
        "x.layers.0.linear_attn.in_proj_qkv": "x.layers.0.linear_attn.in_proj_z",
    }


def test_weight_scale_swizzled_matches_cat_of_halves(fake_kernels):
    """合并 scale = 两半线性 scale 拼接后 swizzle（必须与两半各自 swizzle 再拼一致）。"""
    leader_parts = _parts(N1, seed=3)
    follower_parts = _parts(N2, seed=11)
    leader, follower = L.build_decode_pair_from_checkpoint(
        leader_parts, follower_parts, device="cpu"
    )
    sw = torch.cat([leader.weight_scale_swizzled, follower.weight_scale_swizzled], 0)
    expected = swizzle_scale_128x4(
        torch.cat([leader_parts["weight_scale"], follower_parts["weight_scale"]], 0)
    )
    assert torch.equal(sw, expected)
    assert leader.weight_scale_swizzled.shape == (N1, K // 16)


def test_constructor_requires_exactly_one_scale_source():
    parts = _parts(N1)
    with pytest.raises(TypeError):
        L.NVFP4Linear(
            parts["weight"],
            None,
            parts["weight_scale_2"],
            parts["input_scale"],
        )
    with pytest.raises(TypeError):
        L.NVFP4Linear(
            parts["weight"],
            parts["weight_scale"],
            parts["weight_scale_2"],
            parts["input_scale"],
            weight_scale_swizzled=torch.zeros(N1, K // 16, dtype=torch.float8_e4m3fn),
        )


# ------------------------------------------------- 3. 历史 bug 清单（回归）
@pytest.fixture
def norm_quant_state():
    """保存/恢复 ``norm_quant_patch`` 的全局状态，避免污染其它测试。"""
    from runner.model import norm_quant_patch as nq

    saved = dict(nq._STATE)
    nq._STATE.update(
        {
            "enabled": True,
            "scratch": None,
            "num_warps": 4,
            "counters": {
                "fused_norms": 0,
                "fused_consumers": 0,
                "unconsumed": 0,
                "fallbacks": {},
            },
            "warned": set(),
        }
    )
    yield nq
    nq._STATE.clear()
    nq._STATE.update(saved)


def _norm_quant_scratch(nq, slots: int) -> dict:
    return {
        "out": torch.zeros(1, nq.HIDDEN, dtype=torch.bfloat16),
        "a_fp4": [
            torch.zeros(1, nq.HIDDEN // 2, dtype=torch.uint8) for _ in range(slots)
        ],
        "a_sf": [
            torch.zeros(nq.SF_ROWS, nq.HIDDEN // 16, dtype=torch.uint8)
            for _ in range(slots)
        ],
    }


def _norm_quant(nq):
    norm = nn.Module()
    norm.weight = nn.Parameter(torch.zeros(nq.HIDDEN), requires_grad=False)
    norm.eps = 1e-6
    return nq._NormQuant(norm, "test")


def test_scratch_is_sliced_to_consumer_count(norm_quant_state, monkeypatch):
    """历史 bug（job-000036）：3 槽 scratch 被整份传给 2 消费者 kernel -> 长度不匹配。

    修复契约：每次按 ``count = len(consumers)`` 切片，kernel 只看到 2 个槽位。
    """
    nq = norm_quant_state
    captured = {}

    def fake_fused(x, w1, eps, gs, out, a_fp4, a_sf, *, num_warps=4):
        captured.update(gs=len(gs), a_fp4=len(a_fp4), a_sf=len(a_sf))

    monkeypatch.setattr(nq, "norm_quant_fused", fake_fused)
    nq._STATE["scratch"] = _norm_quant_scratch(nq, 3)
    consumers = [types.SimpleNamespace(_norm_quant_gs=0.5) for _ in range(2)]
    result = _norm_quant(nq).try_run(
        torch.zeros(1, 1, nq.HIDDEN, dtype=torch.bfloat16), consumers
    )
    assert result is not None
    assert captured == {"gs": 2, "a_fp4": 2, "a_sf": 2}
    assert nq._STATE["counters"]["fused_norms"] == 1
    assert nq._STATE["counters"]["fused_consumers"] == 2
    assert nq._STATE["counters"]["fallbacks"] == {}


def test_scratch_too_small_falls_back_whole_layer(norm_quant_state, monkeypatch):
    """槽位不够时整层回退（不走 kernel、不抛异常），并计数 ``scratch-slots``。"""
    nq = norm_quant_state
    called = []
    monkeypatch.setattr(
        nq, "norm_quant_fused", lambda *a, **k: called.append(1)
    )
    nq._STATE["scratch"] = _norm_quant_scratch(nq, 2)
    consumers = [types.SimpleNamespace(_norm_quant_gs=0.5) for _ in range(3)]
    assert (
        _norm_quant(nq).try_run(
            torch.zeros(1, 1, nq.HIDDEN, dtype=torch.bfloat16), consumers
        )
        is None
    )
    assert called == []
    assert nq._STATE["counters"]["fallbacks"] == {"scratch-slots": 1}
    assert nq._STATE["counters"]["fused_norms"] == 0


def test_consumer_count_mismatch_falls_back(norm_quant_state, monkeypatch):
    """消费者数量不是 2/3 时整层回退（job-000036 的另一个触发条件）。"""
    nq = norm_quant_state
    called = []
    monkeypatch.setattr(nq, "norm_quant_fused", lambda *a, **k: called.append(1))
    nq._STATE["scratch"] = _norm_quant_scratch(nq, 3)
    consumers = [types.SimpleNamespace(_norm_quant_gs=0.5) for _ in range(4)]
    assert (
        _norm_quant(nq).try_run(
            torch.zeros(1, 1, nq.HIDDEN, dtype=torch.bfloat16), consumers
        )
        is None
    )
    assert called == []
    assert nq._STATE["counters"]["fallbacks"] == {"consumer-count-mismatch": 1}


@pytest.fixture
def gated_norm_state():
    from runner.model import gated_norm_patch as gp

    saved = dict(gp._STATE)
    gp._STATE.update(
        {
            "enabled": True,
            "plans": {},
            "scratch": None,
            "num_warps": 4,
            "counters": {"fused": 0, "unconsumed": 0, "fallbacks": {}},
            "warned": set(),
        }
    )
    yield gp
    gp._STATE.clear()
    gp._STATE.update(saved)


def test_gated_activation_uses_the_gate_operand(gated_norm_state, monkeypatch):
    """历史 bug：gated 激活的操作数写反（``sigmoid/silu(gate)`` 写到了 normed 上）。

    本树里同形态的实现是 GDN gated norm（``normed * silu(gate)``）。这里从两个层面
    钉住：(a) 接线把 ``(hidden, gate)`` 按正确顺序传给融合 kernel；(b) kernel 源码里
    silu 作用在 ``gv``（gate）上、``nb``（normed）只是被乘的另一操作数。
    """
    gp = gated_norm_state
    captured = {}

    def fake_fused(x, gate, weight, eps, gs, out, a_fp4, a_sf, *, num_warps=4):
        captured.update(
            x=x, gate=gate, weight=weight, eps=eps, gs=gs, out=out,
            a_fp4=a_fp4, a_sf=a_sf,
        )
        out.zero_()

    monkeypatch.setattr(gp, "gated_norm_fused", fake_fused)
    monkeypatch.setattr(gp, "_skip_reason", lambda *args, **kwargs: None)
    out_proj = types.SimpleNamespace(_pending_quantized=None)
    plan = gp._Plan(
        out_proj, torch.ones(gp.HEAD_V_DIM, dtype=torch.bfloat16), 1e-6, 0.7, "test"
    )
    norm = object()
    gp._STATE["plans"] = {id(norm): plan}
    scratch = {
        "out": torch.zeros(gp.NUM_V_HEADS, gp.HEAD_V_DIM, dtype=torch.bfloat16),
        "a_fp4": [torch.zeros(1, gp.VALUE_DIM // 2, dtype=torch.uint8)],
        "a_sf": [torch.zeros(gp.SF_ROWS, gp.VALUE_DIM // 16, dtype=torch.uint8)],
    }
    gp._STATE["scratch"] = scratch
    hidden = torch.zeros(gp.NUM_V_HEADS, gp.HEAD_V_DIM, dtype=torch.bfloat16)
    gate = torch.zeros(gp.NUM_V_HEADS, gp.HEAD_V_DIM, dtype=torch.bfloat16)
    result = gp._patched_forward(norm, hidden, gate)
    assert captured["x"] is hidden
    assert captured["gate"] is gate
    assert captured["out"] is scratch["out"]
    assert result is scratch["out"]
    assert out_proj._pending_quantized == (scratch["a_fp4"][0], scratch["a_sf"][0])
    assert gp._STATE["counters"] == {"fused": 1, "unconsumed": 0, "fallbacks": {}}

    source = Path(inspect.getfile(gp)).parent.parent / "ops/gdn/gated_norm.py"
    body = source.read_text()
    assert "nb.to(tl.float32) * libdevice.div_rn(gv" in body, (
        "silu/sigmoid 必须作用在 gate(z) 上，normed 是被乘的另一操作数"
    )


def test_gdn_head_count_uses_post_repeat_qk_shape():
    """历史 bug（job-000040）：head-count 用 repeat 之前的 16 head 判断，decode 全回退。

    真实调用点传进来的是 ``repeat_interleave`` 之后的 ``[1, 1, 48, 128]``；这里断言
    该形状被接受、kernel 侧再用 ``[:, :, ::ratio, :]`` 取回 16 个 k head。
    """
    from runner.model import gdn_patch as gp

    ratio, v_heads, head_dim = 3, 48, 128
    state = torch.zeros(1, v_heads, head_dim, head_dim)
    value = torch.zeros(1, 1, v_heads, head_dim)
    post_repeat = torch.zeros(1, 1, v_heads, head_dim)
    assert gp._shape_plan(post_repeat, post_repeat, value, state, ratio) is None
    view = gp._qk_view(post_repeat, v_heads, ratio)
    assert view is not None and view.shape == (1, 1, v_heads // ratio, head_dim)

    pre_repeat = torch.zeros(1, 1, v_heads // ratio, head_dim)
    assert gp._qk_view(pre_repeat, v_heads, ratio) is pre_repeat

    # 用 16 head 冒充 48 head（旧 bug 的判据）：必须回退，不能崩。
    assert gp._shape_plan(pre_repeat, pre_repeat, value, state, 1) == "head-count"
