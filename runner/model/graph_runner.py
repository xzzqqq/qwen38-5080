"""M3 CUDA Graph：把 static-cap 的单 token decode 模型 step 录成一张 graph。

设计：
- 输入是一个固定地址的 ``inputs_embeds [1,1,5120]`` BF16 buffer；
  host 侧 embedding gather / H2D / sampling 都在 graph 外；
- graph 内包含 64 层 decoder、cache/state 原地更新和 lm_head
  （``logits_to_keep=1`` -> ``[1,1,248320]``），共 401 个 NVFP4 GEMM；
- 位置推进由 cache 里的 ``cumulative_length`` device tensor 完成，host 不写位置；
- capture 本身**不执行** kernel（stream capture 只记录），所以 capture 不会推进
  state；warmup 会推进，因此正式计时前必须原地 reset 并重新 prefill；
- graph-private pool 的生命周期跟 graph 绑定，必须计入显存峰值。

数据流（replay）：
    input_buf（host 已填好本步 embedding 行）
      -> replay() -> decoder 64 层 -> lm_head -> graph_out.logits（固定地址）

预期瓶颈：graph 只消除 CPU 提交/launch 开销，不融合内部 tiny kernel；
每步仍是数千个 kernel，权重读取 14.45 GB 仍是下限。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class CaptureInfo:
    warmup_iters: int
    capture_s: float
    logits_shape: tuple[int, ...]
    logits_data_ptr: int
    mem_before_allocated: int
    mem_before_reserved: int
    mem_after_allocated: int
    mem_after_reserved: int
    length_before: int
    length_after: int
    warmup_length_delta: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphDecodeStep:
    """把单 token decode step 录成 CUDA Graph。

    Args:
        model: 已加载、eval、no_grad 的 Qwen3_5ForCausalLM。
        input_buf: 固定地址输入，shape ``[1, 1, hidden]``，BF16，CUDA。
        warmup_iters: capture 前在 side stream 上的 warmup 次数（含 FlashInfer
            JIT / workspace / tactic 的第一次分配）。
    """

    def __init__(
        self,
        model,
        cache,
        input_buf: torch.Tensor,
        *,
        warmup_iters: int = 3,
        device: torch.device | str = "cuda",
    ) -> None:
        self.model = model
        # 必须显式传入 static cache：否则 model 会自己建一个 DynamicCache，
        # 变量长 buffer 会落在 graph-private pool 里，replay 写得进去但外部看不到。
        self.cache = cache
        self.input_buf = input_buf
        self.warmup_iters = int(warmup_iters)
        self.device = torch.device(device)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.out = None
        self.capture_info: CaptureInfo | None = None
        self._warmup_logits: list[torch.Tensor] = []
        self._warmup_length_delta = 0

    @property
    def logits(self) -> torch.Tensor:
        if self.out is None:
            raise RuntimeError("graph not captured yet")
        return self.out.logits

    @property
    def logits_ptr(self) -> int:
        return int(self.logits.data_ptr())

    def _forward(self):
        return self.model(
            inputs_embeds=self.input_buf,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )

    def warmup(self) -> None:
        """在 side stream 上跑 warmup；capture 前必须完成。

        PyTorch 要求 capture 前在非默认 stream 上做过 warmup，保证 cudnn/
        cublas/FlashInfer 的 workspace 和 tactic 已经确定。
        """
        length_before = int(self.cache.get_seq_length())
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(self.warmup_iters):
                self._warmup_logits.append(self._forward().logits)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize()
        delta = int(self.cache.get_seq_length()) - length_before
        # warmup 必须推进 static cache；否则说明 graph 用错了 cache（例如模型内部
        # 又建了一个 DynamicCache），capture 出来的东西对外部 state 不可见。
        if delta != self.warmup_iters:
            raise RuntimeError(
                f"graph warmup 没有推进 static cache：delta={delta} != {self.warmup_iters}；"
                "检查 GraphDecodeStep 是否传入了正确的 past_key_values"
            )
        self._warmup_length_delta = delta

    def capture(self) -> CaptureInfo:
        """录 graph。cache 里必须有已初始化的 KV / GDN / conv state。"""
        if self.graph is not None:
            raise RuntimeError("graph already captured")
        if not self._warmup_logits:
            self.warmup()
        self._warmup_logits.clear()

        length_before = int(self.cache.get_seq_length())
        torch.cuda.synchronize()
        mem_before = (
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )
        graph = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(graph):
            self.out = self._forward()
        torch.cuda.synchronize()
        capture_s = time.perf_counter() - t0
        mem_after = (
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
        )
        # capture 只记录不执行；这里再确认一次，避免把“capture 已推进位置”写错。
        length_after = int(self.cache.get_seq_length())
        self.graph = graph
        self.capture_info = CaptureInfo(
            warmup_iters=self.warmup_iters,
            capture_s=capture_s,
            logits_shape=tuple(self.out.logits.shape),
            logits_data_ptr=self.logits_ptr,
            mem_before_allocated=mem_before[0],
            mem_before_reserved=mem_before[1],
            mem_after_allocated=mem_after[0],
            mem_after_reserved=mem_after[1],
            length_before=length_before,
            length_after=length_after,
            warmup_length_delta=self._warmup_length_delta,
        )
        return self.capture_info

    def replay(self) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("graph not captured yet")
        self.graph.replay()
        return self.logits
