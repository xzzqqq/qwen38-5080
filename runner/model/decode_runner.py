"""M3 统一 decode runner：eager / static-exact / static-cap / graph。

四个执行方式共用同一 checkpoint、dtype、prefill chunk 和输入 staging：

    token id
      -> host BF16 embedding gather（CPU，整份 embedding 常驻 host）
      -> 复用 pinned 行 buffer
      -> 固定 GPU 输入 buffer ``inputs_embeds [1,1,5120]``
      -> 单步模型 forward（eager / static）或 CUDA Graph replay
      -> logits [1,1,248320]

计时边界（对应 docs/plans/m3.md §2.3）：
- 模型 decode tok/s：逐步 CUDA events，起 event 在 D2D copy 之后，只包模型提交区间；
- 系统 decode tok/s：墙钟包住 host gather / pinned copy / H2D / 模型 / sampling / D2H；
- 请求有效速度：见脚本层，包含 prefill 和 TTFT。

预期瓶颈：eager 受 CPU 提交（每步约万个 tiny kernel）限制；static-cap 在
capacity 远大于有效长度时把 ``repeat_kv`` 展开流量放大到 capacity；graph 消除
提交开销但不融合 kernel。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DynamicCache

from runner.model.graph_runner import GraphDecodeStep
from runner.model.static_cache import (
    build_static_cache,
    cache_stats,
    preallocate_kv,
    reset_cache,
)

EXECUTIONS = ("eager", "static-exact", "static-cap", "graph")
STATIC_EXECUTIONS = ("static-exact", "static-cap", "graph")


@dataclass
class PrefillStats:
    prompt_tokens: int
    prefill_s: float
    first_token_s: float
    first_token: int
    cache_len: int
    peak_allocated_gb: float
    peak_reserved_gb: float
    free_gb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prefill_s": self.prefill_s,
            "first_token_s": self.first_token_s,
            "first_token": self.first_token,
            "cache_len": self.cache_len,
            "peak_allocated_gb": self.peak_allocated_gb,
            "peak_reserved_gb": self.peak_reserved_gb,
            "free_gb": self.free_gb,
        }


class DecodeRunner:
    """一条请求生命周期 + 两套 decode 计时。

    Args:
        execution: ``eager`` / ``static-exact`` / ``static-cap`` / ``graph``。
        capacity: static 模式的 KV capacity（1024 或 8192）；eager 必须为 None。
        prefill_chunk: prefill 分块长度，四个模式一致（默认 128）。
        graph_warmup: capture 前的 side-stream warmup 次数。
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        execution: str,
        capacity: int | None = None,
        prefill_chunk: int = 128,
        max_context: int = 8192,
        graph_warmup: int = 3,
        device: str = "cuda",
    ) -> None:
        if execution not in EXECUTIONS:
            raise ValueError(f"execution must be one of {EXECUTIONS}, got {execution!r}")
        if execution == "eager":
            if capacity is not None:
                raise ValueError("eager 没有固定 backing capacity，不能指定 capacity")
        else:
            if capacity not in (1024, 8192):
                raise ValueError(f"capacity must be 1024 or 8192, got {capacity!r}")
        self.model = model
        self.tokenizer = tokenizer
        self.execution = execution
        self.capacity = capacity
        self.prefill_chunk = int(prefill_chunk)
        self.max_context = int(max_context)
        self.graph_warmup = int(graph_warmup)
        self.device = torch.device(device)
        self.stop_ids = {
            int(x)
            for x in (
                getattr(tokenizer, "eos_token_id", None),
                getattr(model.config, "eos_token_id", None),
            )
            if x is not None
        }

        hidden = int(model.config.hidden_size)
        self.input_buf = torch.zeros(1, 1, hidden, dtype=torch.bfloat16, device=self.device)
        self.pinned_row = torch.empty(1, 1, hidden, dtype=torch.bfloat16, pin_memory=True)
        self.host_weight = model.model.embed_tokens.weight.detach()
        if self.host_weight.device.type != "cpu":
            raise RuntimeError("embedding 必须在 host，M3 不把整份 embedding 搬上 GPU")

        self.cache = None
        self.graph_step: GraphDecodeStep | None = None
        self.graph_info: dict[str, Any] | None = None
        self._prepared = False

    # ------------------------------------------------------------------ setup
    def prepare(self) -> dict[str, Any]:
        """构建 cache；static 模式按 capacity 预分配 KV。只做一次。"""
        if self._prepared:
            return self.harness_info()
        if self.execution == "eager":
            self.cache = DynamicCache(config=self.model.config)
        else:
            mode = "exact" if self.execution == "static-exact" else "cap"
            self.cache = build_static_cache(self.model.config, self.capacity, mode)
            preallocate_kv(self.cache, self.model.config, device=self.device)
        if self.execution == "graph":
            self.graph_step = GraphDecodeStep(
                self.model,
                self.cache,
                self.input_buf,
                warmup_iters=self.graph_warmup,
                device=self.device,
            )
        self._prepared = True
        return self.harness_info()

    def harness_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "execution": self.execution,
            "capacity": self.capacity,
            "prefill_chunk": self.prefill_chunk,
            "max_context": self.max_context,
            "input_buffer_shape": tuple(self.input_buf.shape),
            "input_buffer_ptr": int(self.input_buf.data_ptr()),
            "input_buffer_dtype": str(self.input_buf.dtype),
            "pinned_staging_shape": tuple(self.pinned_row.shape),
            "host_embedding_bytes": self.host_weight.numel() * self.host_weight.element_size(),
        }
        if self.cache is not None and self.execution != "eager":
            info["cache"] = cache_stats(self.cache, check_finite=False).to_dict()
        if self.graph_info is not None:
            info["graph"] = dict(self.graph_info)
        return info

    # --------------------------------------------------------------- requests
    def reset(self) -> None:
        """原地 reset；graph 常驻时 pool 不销毁。"""
        if self.cache is None:
            return
        if self.execution == "eager":
            self.cache.reset()
        else:
            reset_cache(self.cache)

    def cache_length(self) -> int:
        return int(self.cache.get_seq_length())

    def host_gather(self, token_ids: list[int] | torch.Tensor) -> torch.Tensor:
        """CPU BF16 embedding gather（只取需要的行）。"""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.reshape(-1, 1)
        else:
            ids = torch.tensor([[int(t)] for t in token_ids], dtype=torch.long)
        return torch.nn.functional.embedding(ids, self.host_weight)

    def prefill(
        self,
        prompt_ids: torch.Tensor,
        *,
        reset_peak: bool = True,
        reserve_steps: int = 0,
    ) -> tuple[torch.Tensor, PrefillStats]:
        """reset 后重跑 prefill；返回最后位置 logits 和首 token 统计。

        Args:
            reset_peak: 是否在这里重置显存峰值。外层要观察“prefill+capture+prefill”
                的整体峰值时传 False，由外层统一 reset/读取。
            reserve_steps: 这次请求紧接着还要写多少 token；和 prompt 一起做
                capacity 检查，避免 decode 阶段才越界。
        """
        if not self._prepared:
            self.prepare()
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError(f"prompt_ids must be [1, S], got {tuple(prompt_ids.shape)}")
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len <= 0:
            raise ValueError("empty prompt")
        if prompt_len >= self.max_context:
            raise ValueError(
                f"prompt length {prompt_len} must be < max_context {self.max_context}"
            )
        if self.execution != "eager":
            if prompt_len + int(reserve_steps) > self.capacity:
                raise ValueError(
                    f"{self.execution} capacity overflow at prefill: prompt={prompt_len} "
                    f"reserve={reserve_steps} capacity={self.capacity}"
                )
        self.reset()
        torch.cuda.synchronize()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        prompt_ids = prompt_ids.to("cpu")
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        wall0 = time.perf_counter()
        t0.record()
        out = None
        for start in range(0, prompt_len, self.prefill_chunk):
            out = self.model(
                input_ids=prompt_ids[:, start : start + self.prefill_chunk],
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=1,
            )
        if out is None:
            raise RuntimeError("empty prompt")
        logits = out.logits[:, -1, :]
        t1.record()
        first_token = int(logits.argmax(dim=-1).item())  # 这里包含一次 D2H 同步
        torch.cuda.synchronize()
        stats = PrefillStats(
            prompt_tokens=prompt_len,
            prefill_s=t0.elapsed_time(t1) / 1000.0,
            first_token_s=time.perf_counter() - wall0,
            first_token=first_token,
            cache_len=self.cache_length(),
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
            peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
            free_gb=torch.cuda.mem_get_info()[0] / 1e9,
        )
        return logits, stats

    def ensure_graph(self, prompt_ids: torch.Tensor, *, reset_peak: bool = False) -> dict[str, Any]:
        """首个请求上完成 warmup + capture，然后恢复正式初态。

        warmup 会推进 ``graph_warmup`` 步；capture 只记录不执行。capture 完成后
        必须 reset 并重新 prefill，不能把 warmup 后的位置当成正式初态。

        Args:
            reset_peak: 默认 False，让外层能看到 prefill+warmup+capture+prefill
                的整体峰值；内部子阶段峰值记在 ``CaptureInfo`` 里。
        """
        if self.execution != "graph":
            return {}
        if self.graph_step is None:
            self.prepare()
        if self.graph_step.graph is not None:
            return dict(self.graph_info or {})
        # 每个子阶段各自 reset/读 peak：外层不能只看“阶段末 occupancy”。
        _, pre_before = self.prefill(prompt_ids, reset_peak=True)
        # warmup 会推进 graph_warmup 步；先确认 capacity 够，再提交。
        self.require_capacity(self.graph_warmup)
        info = self.graph_step.capture()
        _, pre_after = self.prefill(prompt_ids, reset_peak=True)
        self.graph_info = info.to_dict()
        self.graph_info.update(
            {
                "length_at_entry": info.length_at_entry,
                "length_before_capture": info.length_before_capture,
                "length_after_capture": info.length_after_capture,
                "prefill_before_peak_allocated_gb": pre_before.peak_allocated_gb,
                "prefill_before_peak_reserved_gb": pre_before.peak_reserved_gb,
                "prefill_after_peak_allocated_gb": pre_after.peak_allocated_gb,
                "prefill_after_peak_reserved_gb": pre_after.peak_reserved_gb,
            }
        )
        self.graph_info["max_phase_peak_allocated_gb"] = max(
            pre_before.peak_allocated_gb,
            info.warmup_peak_allocated / 1e9,
            info.capture_peak_allocated / 1e9,
            pre_after.peak_allocated_gb,
        )
        self.graph_info["max_phase_peak_reserved_gb"] = max(
            pre_before.peak_reserved_gb,
            info.warmup_peak_reserved / 1e9,
            info.capture_peak_reserved / 1e9,
            pre_after.peak_reserved_gb,
        )
        return dict(self.graph_info)

    # ------------------------------------------------------------------ steps
    def _step(self) -> torch.Tensor:
        """在固定输入 buffer 上跑一个 decode step，返回 logits（当前 stream）。"""
        if self.execution == "graph":
            if self.graph_step is None or self.graph_step.graph is None:
                raise RuntimeError("graph 还没 capture；先调用 ensure_graph")
            return self.graph_step.replay()
        out = self.model(
            inputs_embeds=self.input_buf,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return out.logits

    def capacity_headroom(self) -> int:
        """还能写多少 token（static 模式按 capacity，eager 按 max_context）。"""
        used = self.cache_length()
        limit = self.max_context if self.execution == "eager" else self.capacity
        return int(limit) - used

    def require_capacity(self, steps: int) -> None:
        """越界拒绝：容量不足时在提交 kernel 之前报错，而不是让 kernel 越界。"""
        used = self.cache_length()
        headroom = self.capacity_headroom()
        if headroom < steps:
            limit = self.max_context if self.execution == "eager" else self.capacity
            raise ValueError(
                f"{self.execution} capacity overflow: used={used} limit={limit} "
                f"needed={steps} headroom={headroom}"
            )

    def trace_decode(self, trace_ids: list[int], *, keep_logits: bool = True) -> dict[str, Any]:
        """按给定 token 逐步执行并可选保留 logits；只用于对拍，不走性能路径。"""
        logits_out: list[torch.Tensor] = []
        for token in trace_ids:
            self.require_capacity(1)
            row = self.host_gather([int(token)])
            self.stage_row(row, pinned=True)
            logits = self._step()[:, -1, :]
            if keep_logits:
                logits_out.append(logits.detach().to("cpu", torch.bfloat16))
        torch.cuda.synchronize()
        return {"logits": logits_out, "cache_len": self.cache_length()}

    def stage_row(self, row_cpu: torch.Tensor, *, pinned: bool) -> None:
        """把一行 embedding 放到固定 GPU 输入 buffer。"""
        if pinned:
            self.pinned_row.copy_(row_cpu)
            self.input_buf.copy_(self.pinned_row, non_blocking=True)
        else:
            self.input_buf.copy_(row_cpu.to(self.device, non_blocking=False))

    # ------------------------------------------------------- model-side timing
    def model_timed_decode(self, trace_ids: list[int], *, measure_copy: bool = True) -> dict[str, Any]:
        """用冻结 trace 驱动，逐步 CUDA events 计时。

        D2D copy 在 start event 之前，因此不计入模型时间；同 stream 顺序保证
        copy 先于对应 forward 完成。
        """
        steps = len(trace_ids)
        if steps <= 0:
            raise ValueError("trace_ids is empty")
        if self.execution == "graph" and (self.graph_step is None or self.graph_step.graph is None):
            raise RuntimeError("graph 还没 capture；先调用 ensure_graph")
        # 计时区间开始前一次性检查完整步数预算；循环里不再逐 step 同步。
        self.require_capacity(steps)

        rows = self.host_gather(trace_ids)  # [steps, 1, hidden] CPU
        rows_gpu = rows.to(self.device, non_blocking=False)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        wall0 = time.perf_counter()
        for i in range(steps):
            self.input_buf.copy_(rows_gpu[i : i + 1])
            starts[i].record()
            self._step()
            ends[i].record()
        t1.record()
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - wall0
        span_s = t0.elapsed_time(t1) / 1000.0
        per_step_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        result = {
            "steps": steps,
            "per_step_ms": per_step_ms,
            "sum_step_s": sum(per_step_ms) / 1000.0,
            "span_gpu_s": span_s,
            "wall_s": wall_s,
            "model_tok_s": steps / (sum(per_step_ms) / 1000.0),
            "span_tok_s": steps / span_s if span_s > 0 else 0.0,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "free_gb": torch.cuda.mem_get_info()[0] / 1e9,
            "cache_len_end": self.cache_length(),
        }
        if measure_copy:
            # 夹具 copy 成本：10 次 D2D，用来确认它远小于一步。
            torch.cuda.synchronize()
            c0 = torch.cuda.Event(enable_timing=True)
            c1 = torch.cuda.Event(enable_timing=True)
            c0.record()
            for i in range(10):
                self.input_buf.copy_(rows_gpu[i : i + 1])
            c1.record()
            torch.cuda.synchronize()
            result["row_copy_ms"] = c1.elapsed_time(c0) / 10.0
        return result

    # ------------------------------------------------------ system-side timing
    def system_decode(
        self,
        first_token: int,
        steps: int,
        *,
        stop_on_eos: bool = False,
        record_margins: bool = False,
    ) -> dict[str, Any]:
        """真实贪心闭环：host gather / pinned / H2D / 模型 / sampling / D2H 全计墙钟。"""
        if steps <= 0:
            raise ValueError("steps must be positive")
        if self.execution == "graph" and (self.graph_step is None or self.graph_step.graph is None):
            raise RuntimeError("graph 还没 capture；先调用 ensure_graph")
        # 闭环开始前一次性检查完整步数预算；循环里不再逐 step 同步。
        self.require_capacity(steps)
        tokens: list[int] = []
        margins: list[float] = []
        next_id = int(first_token)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        wall0 = time.perf_counter()
        finish_reason = "length"
        for _ in range(steps):
            row = self.host_gather([next_id])
            self.stage_row(row, pinned=True)
            logits = self._step()
            if record_margins:
                # top-2 差值用来区分「实现差异」和「near-tie 翻转」，只在对拍里用。
                row = logits[:, -1, :].float().flatten()
                top2 = torch.topk(row, 2)
                margins.append(float(top2.values[0].item() - top2.values[1].item()))
                next_id = int(top2.indices[0].item())
            else:
                next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            tokens.append(next_id)
            if stop_on_eos and next_id in self.stop_ids:
                finish_reason = "eos"
                break
        wall_s = time.perf_counter() - wall0
        return {
            "requested_steps": steps,
            "steps": len(tokens),
            "tokens": tokens,
            "margins": margins,
            "wall_s": wall_s,
            "system_tok_s": len(tokens) / wall_s if wall_s > 0 else 0.0,
            "finish_reason": finish_reason,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "free_gb": torch.cuda.mem_get_info()[0] / 1e9,
            "cache_len_end": self.cache_length(),
        }
