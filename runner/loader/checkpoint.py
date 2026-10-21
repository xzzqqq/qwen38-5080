"""按 tensor 名读取 safetensors checkpoint，支持 CPU/GPU 目标。"""

from __future__ import annotations

import json
from pathlib import Path

from safetensors import safe_open


class CheckpointReader:
    """打开 index 和 shard，按需读取 tensor。

    - 同一个 shard 的 safe_open handle 会复用；
    - GPU 读取时直接让 safetensors 把 tensor 放到 CUDA，避免 CPU 中转；
    - 只读，不修改 checkpoint。
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        self.index = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = self.index["weight_map"]
        self._handles: dict[tuple[str, str], object] = {}

    def _handle(self, shard: str, device: str):
        key = (shard, device)
        if key not in self._handles:
            self._handles[key] = safe_open(
                self.model_dir / shard, framework="pt", device=device
            )
        return self._handles[key]

    def get(self, name: str, device: str = "cpu"):
        if name not in self.weight_map:
            raise KeyError(f"tensor not in checkpoint: {name}")
        shard = self.weight_map[name]
        return self._handle(shard, device).get_tensor(name)

    def close(self) -> None:
        self._handles.clear()

    def __enter__(self) -> "CheckpointReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
