"""``runner/model/gated_norm_patch.py`` 的 CPU-only 接线自测（不建 CUDA context）。

为什么需要它：op gate 只测 kernel，m3_check 的 bitwise gate 只比较执行方式 ——
两者都看不见「patch 到底有没有被装上/有没有走错分支」。第 2 轮任务 H 就是因为
「文档串了但接线层签名不匹配」在 eager 的第 0 层才崩（job-000038）。这里用**纯 CPU**
把回退分支和计数器在不花 GPU 的前提下测掉：

- 没有 plan（norm 不是 48 个 GDN 层之一）-> 走原 forward，计 ``fallbacks.wiring``；
- 非 BF16 / 非 CUDA / 形状不对 -> 各自的原因计数，仍然走原 forward；
- ``install`` 在没有 GDN 层的模型上干净地抛错，``install_from_env`` 只记录不抛；
- ``uninstall`` 在没有安装时是 no-op。

注意：这里**不能**触碰 ``torch.cuda``（队列有 GPU 守卫，CPU 单测不允许带 CUDA context）。
"""

from __future__ import annotations

import pytest
import torch

from runner.model import gated_norm_patch as patch


@pytest.fixture(autouse=True)
def _reset() -> None:
    patch.uninstall()
    patch._STATE["orig_forward"] = lambda self, x, z: (x, z)  # noqa: SLF001
    patch._STATE["enabled"] = True  # noqa: SLF001
    patch._STATE["plans"] = {}  # noqa: SLF001
    patch._STATE["scratch"] = None  # noqa: SLF001
    patch._STATE["counters"] = {"fused": 0, "unconsumed": 0, "fallbacks": {}}  # noqa: SLF001
    patch._STATE["warned"] = set()  # noqa: SLF001
    yield
    patch._STATE["enabled"] = False  # noqa: SLF001
    patch._STATE["plans"] = {}  # noqa: SLF001
    patch._STATE["warned"] = set()  # noqa: SLF001


def _inputs(dtype=torch.bfloat16, device="cpu"):
    x = torch.zeros(48, 128, dtype=dtype, device=device)
    z = torch.zeros(48, 128, dtype=dtype, device=device)
    return x, z


def test_missing_plan_falls_back_to_original() -> None:
    x, z = _inputs()
    out = patch._patched_forward(object(), x, z)  # noqa: SLF001
    assert out[0] is x and out[1] is z
    assert patch._STATE["counters"]["fallbacks"] == {"wiring": 1}  # noqa: SLF001
    assert patch._STATE["counters"]["fused"] == 0  # noqa: SLF001


def test_dtype_and_device_and_shape_fallbacks() -> None:
    x, z = _inputs(dtype=torch.float32)
    plan = object()
    assert patch._skip_reason(x, z, plan) == "dtype"  # noqa: SLF001
    x, z = _inputs()  # bf16 但 CPU
    assert patch._skip_reason(x, z, plan) == "device"  # noqa: SLF001
    assert patch._skip_reason(x, z, None) == "wiring"  # noqa: SLF001


def test_disabled_wiring_falls_back() -> None:
    patch._STATE["enabled"] = False  # noqa: SLF001
    x, z = _inputs()
    out = patch._patched_forward(object(), x, z)  # noqa: SLF001
    assert out[0] is x
    assert patch._STATE["counters"]["fallbacks"] == {"wiring": 1}  # noqa: SLF001


def test_install_without_gdn_layers_raises_and_env_wrapper_does_not() -> None:
    holder = torch.nn.Module()
    with pytest.raises(RuntimeError):
        patch.install(holder)
    info = patch.install_from_env(holder)
    assert info["enabled"] is False
    assert "reason" in info


def test_uninstall_is_noop_when_not_installed() -> None:
    patch.uninstall()
    assert patch._STATE["installed"] is False  # noqa: SLF001
