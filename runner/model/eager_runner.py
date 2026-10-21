"""Eager 文本 runner：chunked prefill + greedy decode。

设计：
- 复用 transformers 的 Qwen3_5ForCausalLM 和 DynamicCache；不写 static cache/CUDA Graph；
- prefill 按 chunk 调 model，cache/state 由 DynamicCache 维护；
- decode 每步只算最后一个位置的 logits（logits_to_keep=1）；
- embedding 在 host，其它权重在 GPU；
- 只做 M2 的正确性/容量，不追求 tok/s。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, DynamicCache

from runner.loader.nvfp4_model import load_nvfp4_text_model


@dataclass
class GenerationResult:
    prompt_tokens: int
    generated_tokens: int
    token_ids: list[int]
    text: str
    prefill_s: float
    decode_s: float
    decode_tok_s: float
    peak_allocated_gb: float
    peak_reserved_gb: float
    free_gb: float
    cache_len: int
    finish_reason: str
    meta: dict[str, Any] = field(default_factory=dict)


class EagerTextRunner:
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        *,
        device: str = "cuda",
        max_context: int = 8192,
        prefill_chunk: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_context = max_context
        self.prefill_chunk = prefill_chunk
        self.stop_ids = {
            int(x)
            for x in (
                getattr(tokenizer, "eos_token_id", None),
                getattr(model.config, "eos_token_id", None),
            )
            if x is not None
        }

    @classmethod
    def from_checkpoint(
        cls,
        model_dir: str | Path,
        *,
        device: str = "cuda",
        max_context: int = 8192,
        prefill_chunk: int = 128,
        manifest_path: Path | None = None,
    ) -> tuple["EagerTextRunner", dict[str, Any]]:
        model, _cfg, info = load_nvfp4_text_model(
            Path(model_dir),
            manifest_path=manifest_path,
            device=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        runner = cls(
            model,
            tokenizer,
            device=device,
            max_context=max_context,
            prefill_chunk=prefill_chunk,
        )
        return runner, info

    def _render_chat(self, messages: list[dict[str, str]]) -> str:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=True, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def encode_messages(self, messages: list[dict[str, str]]) -> torch.Tensor:
        text = self._render_chat(messages)
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids

    @torch.no_grad()
    def generate_ids(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        greedy: bool = True,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, S], got {tuple(input_ids.shape)}")
        prompt_len = int(input_ids.shape[1])
        budget = self.max_context - prompt_len
        if budget <= 0:
            raise ValueError(
                f"prompt length {prompt_len} already exceeds context {self.max_context}"
            )
        n_new = min(int(max_new_tokens), budget)
        input_ids = input_ids.to("cpu")

        cache = DynamicCache(config=self.model.config)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        out = None
        for start in range(0, prompt_len, self.prefill_chunk):
            chunk = input_ids[:, start : start + self.prefill_chunk]
            out = self.model(
                input_ids=chunk,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        if out is None:
            raise RuntimeError("empty prompt")
        logits = out.logits[:, -1, :]

        generated: list[int] = []
        decode_start = time.perf_counter()
        finish_reason = "length"
        for step in range(n_new):
            if greedy:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits.float(), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            token = int(next_id.item())
            generated.append(token)
            if stop_on_eos and token in self.stop_ids:
                finish_reason = "eos"
                break
            if step + 1 >= n_new:
                break
            out = self.model(
                input_ids=next_id.cpu(),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            logits = out.logits[:, -1, :]
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - decode_start

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        free_b, _ = torch.cuda.mem_get_info()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        result = GenerationResult(
            prompt_tokens=prompt_len,
            generated_tokens=len(generated),
            token_ids=generated,
            text=text,
            prefill_s=prefill_s,
            decode_s=decode_s,
            decode_tok_s=(len(generated) / decode_s) if decode_s > 0 else 0.0,
            peak_allocated_gb=peak_alloc / 1e9,
            peak_reserved_gb=peak_reserved / 1e9,
            free_gb=free_b / 1e9,
            cache_len=int(cache.get_seq_length()),
            finish_reason=finish_reason,
            meta={"prefill_chunk": self.prefill_chunk, "max_context": self.max_context},
        )
        del cache
        return result

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 128,
        greedy: bool = True,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        input_ids = self.encode_messages(messages)
        return self.generate_ids(
            input_ids,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
            stop_on_eos=stop_on_eos,
        )

    def generate_payload(self, payload_path: str | Path) -> GenerationResult:
        payload = json.loads(Path(payload_path).read_text())
        temperature = float(payload.get("temperature", 0))
        return self.generate_messages(
            payload["messages"],
            max_new_tokens=int(payload.get("max_tokens", 128)),
            greedy=temperature == 0,
        )


def result_to_dict(result: GenerationResult) -> dict[str, Any]:
    return {
        "prompt_tokens": result.prompt_tokens,
        "generated_tokens": result.generated_tokens,
        "token_ids": result.token_ids,
        "text": result.text,
        "prefill_s": result.prefill_s,
        "decode_s": result.decode_s,
        "decode_tok_s": result.decode_tok_s,
        "peak_allocated_gb": result.peak_allocated_gb,
        "peak_reserved_gb": result.peak_reserved_gb,
        "free_gb": result.free_gb,
        "cache_len": result.cache_len,
        "finish_reason": result.finish_reason,
        "meta": result.meta,
    }
