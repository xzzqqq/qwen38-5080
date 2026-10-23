# Phase 2 第 4 轮修复：DecodePair 的 eager 接线与装载期 OOM

- 日期：2026-10-23
- 对应：[phase2 plan](../plans/phase2.md) 第 4 轮（合并共享输入的 decode projection）
- session：`phase2-tactics`
- worktree：`.tmp/worktrees/r4-gemv`，branch `exp/phase2-gemvtail`
- 失败 job：`job-000055-2aa2e4e3`（commit `7152a677`，op **222/222** 全绿，
  `m3_check --only numeric-512` 只报 `coverage: 必测项齐全` 缺 34 项）
- 候选 commit：`01e3ff85a895293ea961c0b9a09eed7f470d4d2f`
- 验证 job：`job-000056-4e8de313`（session `phase2-tactics`，3 步：op / check / bench）

## 1. 现象

`job-000055` 的 op 级（static / graph，kernel 对拍）**222/222 通过**，但 check 的
`numeric-512` 一个 eager 必测项都没有：

```
missing: eager: finite / eager: cache length / eager: forward 数 /
         eager->static-exact: logits / ... (34 项)
```

`report.json` 的 `error` 不是断言失败，而是 **load 阶段就抛了**：

```
File "runner/loader/nvfp4_model.py", line 197, in load_nvfp4_text_model
    decode_pairs = install_decode_pairs(model)
File "runner/ops/nvfp4/linear.py", line 281, in attach_decode_pair
    weight = torch.cat([leader.weight_packed, follower.weight_packed], dim=0)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 86.00 MiB.
GPU 0 has a total capacity of 15.47 GiB of which 85.88 MiB is free.
... Of the allocated memory 15.00 GiB is allocated by PyTorch
```

也就是说：**eager 快照不是算错，是根本没跑起来** —— 模型在 `load` 阶段就 OOM 了，
`load_model()` 抛异常被 worker 记进 `report.json.error`，场景检查一条都没执行。
op 级不建整模型（每个 case 只装该 shape 的实例），所以看不见这个失败。

## 2. 根因（两个叠加，缺一不可）

### 2.1 `dict(model.named_buffers())` 把旧 buffer 钉住了

`load_nvfp4_text_model` 在 step 4 建了：

```python
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())
```

这两个 dict 在函数返回前一直存活，而 `install_decode_pairs(model)` 就在它们之后调用。
`attach_decode_pair` 里 `torch.cat` 出新 buffer 后把模块的 buffer 换成新 buffer 的
视图，**旧 buffer 本该被释放**，但 `buffers` 里还握着引用 —— 于是每配一对就净增
一整份合并权重（k/v ≈ 5 MiB、qkv/z ≈ 42 MiB、gate/up ≈ 89 MiB），128 对累计
≈ 1.5 GiB，正好对上 `allocated 15.00 GiB`（baseline 14.46 GB + 1.5 GB）与
「86 MiB 分配失败时只剩 85.88 MiB」。

### 2.2 即使释放旧 buffer，`cat` 也拼不出连续 segment

PyTorch caching allocator 里两半各是一块 44.5 MiB 的 segment（大池独立 cudaMalloc）；
释放后它们不会合并，下一次 89 MiB 的 `torch.cat` 仍然要新开 segment。116→128 对
逐次把物理显存吃干（baseline 装完只剩 ~1.7 GiB）。所以「先分别装载再 cat」这条路
在 15.47 GiB 卡上**结构性地**不可行，不是加 `empty_cache` 能长期维持的。

## 3. 修复

### 3.1 loader：装载时零峰值配对（`runner/loader/nvfp4_model.py`）

- 用 `decode_pair_specs()` 从 manifest 的量化 prefix 里按**命名结构**找出 128 对
  （同 parent 下 k/v、gate/up、qkv/z），与层序和用例无关；
- 每一对**在两半都还不存在时**先用 CPU `torch.cat` 拼好、一次 H2D，再用
  `build_decode_pair_from_checkpoint()` 把合并 buffer 的切片交给两个 `NVFP4Linear`；
  稳态显存 == 原来两半之和，峰值只多一份 scale（11 MiB）；
- 前置条件（dtype/shape/k 一致、N%128、`input_scale` 与 `weight_scale_2` 逐位相同）
  在构造前检查；不满足就返回 `None`，两半照常分别装载（走共享路径），计数进
  `load_info.nvfp4_decode_pairs.reasons`；
- 删掉 `dict(named_buffers())` 在 install 之后仍然存活的引用（`del params, buffers`）；
- step 6 只做校验：统计 `modules_without_pair`，保证 capture 前配对已经生效
  （pair 的 persistent `out` 地址在 capture 前固定）。

`install_decode_pairs()` 保留给「模型已建好」的调用方与 CPU 自测；它的 `torch.cat`
兜底路径在显存不足时会捕获 OOM、跳过该对并 warning，不再把 load 弄崩。

### 3.2 `DecodePair`：四种执行方式同一条逻辑 + 回退计数

`NVFP4Linear.forward` 的配对入口与 cache 类型、是否 capture 完全无关，只依赖
「这次调用的 x 是不是同一份单 token 张量」：

- `_is_decode(x)`：`ndim>=2 且 numel == shape[-1] == in_features`；不满足时
  leader 记 `merge_skips["seq_len"]`、follower 静默走共享路径（prefill 不算回退）；
- follower 只有在 `armed` 且 `(data_ptr, numel)` 与 leader 一致时才取合并结果的
  后半，否则整对回退并记 `merge_fallbacks`，原因分 `order`（顺序不对）/`ptr`
  （中间换了输入）/`error`（合并 kernel 抛错，整对永久 disabled）/`disabled`；
- leader 在 `mm_fp4` 抛错时禁用整对、warning、返回 `None`，调用方继续用
  `pending` 或重新量化走共享路径 —— 不抛异常、不静默算错；
- 计数保留 `merged_calls` / `merge_fallbacks`，并新增
  `merge_fallback_reasons` / `merge_skips`；`load` diag 里能看到全程累计值
  （同一个 dict 引用），这是「配对真的命中了」的产物级证据。

四种执行方式的差异只有 x 缓冲是否复用、是否处于 capture 中；配对路径不读
`torch.cuda.is_current_stream_capturing()`（CPU 守门测试里把它替换成会抛异常的
桩，任何探测都会让测试红）。

## 4. 不花 GPU 的守门测试

新增 `tests/model/test_patch_wiring_cpu.py`（29 个 case，`CUDA_VISIBLE_DEVICES=""`
全绿）。CPU 替身替换 `mm_fp4` / `nvfp4_quantize`，替身保留「输出 j 只依赖 W 的
第 j 行」这一条性质，用来验证合并后切半与分开调用逐位一致。

1. **四种执行方式的调用序列**（eager / static-exact / static-cap / graph ×
   fused/unfused）：两层占位模型（`layer_types = linear_attention, full_attention`）
   覆盖 3 个 pair + 一个独立 q；断言每步每对恰好一次合并、宽度正确、无回退、
   `pending` 槽位全被消费、两半与分开调用逐位一致；
2. **前置条件不满足 → 整对回退且计数正确**：follower 先于 leader（`order`）、
   `data_ptr` 不同（`ptr`）、`seq_len>1`（`merge_skips`）、合并 kernel 抛错
   （`error` + 整对 disabled）、配对数不匹配（`install` 的 `reasons.type`）、
   6 组构造前置条件（scalar 不一致 / N 非 128 倍数 / K 不一致 / dtype / shape）；
3. **历史 bug 清单回归**：
   - *3 槽 scratch 传给 2 消费者*（job-000036）：`_NormQuant.try_run` 按
     `len(consumers)` 切片，kernel 只看到 2 个槽位；槽位不足/消费者数不是 2/3 时
     整层回退并计数，不抛异常；
   - *gated 激活操作数写反*（本树同形态实现是 GDN gated norm 的
     `normed * silu(gate)`）：接线断言 `(hidden, gate)` 顺序正确、out_proj 拿到
     pending；kernel 源码 tripwire 断言 silu 作用在 `gv`（gate）上、`nb` 是被乘的
     另一操作数；
   - *head-count 用 repeat 前的形状*（job-000040）：`gdn_patch._shape_plan`
     接受 repeat 之后的 `[1,1,48,128]`，`_qk_view` 再用 `[:, :, ::3, :]` 取回
     16 个 k head；用 16 head 冒充 48 head 时返回 `"head-count"` 回退而不是崩。

## 5. 验证阶梯

| 档位 | 命令 | 结果 |
|---|---|---|
| CPU 单测 | `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests -q` | 121 passed / 12 skipped |
| 守门测试 | `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/model/test_patch_wiring_cpu.py -q` | 29 passed |
| 产物契约 | `CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/check_results.py --all` | 6/6 |
| op + check + bench | 见下方 3 步请求（GPU 队列） | `job-000056-4e8de313`（提交时排队第 1；结果提交者后续读取） |

## 6. 提交记录

候选 commit：`01e3ff85a895293ea961c0b9a09eed7f470d4d2f`；
验证 job：`job-000056-4e8de313`（提交时队列位置 1，结果目录
`results/experiments/job-000056-4e8de313/`）；3 步 GPU job（session `phase2-tactics`，timeout 2400）：

```json
{"session_id": "phase2-tactics", "commit": "01e3ff85a895293ea961c0b9a09eed7f470d4d2f", "timeout_s": 2400,
 "steps": [
   {"kind": "op", "argv": [".venv/bin/python", "bench/micro/nvfp4_tactics.py", "--out", "{output}/op"], "result": "{output}/op/op.json"},
   {"kind": "check", "argv": [".venv/bin/python", "scripts/m3_check.py", "--only", "numeric-512,graph-1024", "--steps", "16", "--out", "{output}/check"], "result": "{output}/check/check.json"},
   {"kind": "bench", "argv": [".venv/bin/python", "bench/e2e/decode.py", "--only", "graph-8192,graph-512-1024", "--steps", "32", "--repeats", "1", "--out", "{output}/bench"], "result": "{output}/bench/bench.json"}
 ]}
```

判定口径：`numeric-512` 两组 bitwise 是硬 gate（上一轮缺的 eager 项必须全部出现）；
e2e 目标 `graph-8192 < 20.493 ms`（baseline `job-000050`，同协议 32×1）。

## 7. 已知限制

- 本轮**没有**在本地跑任何带 CUDA context 的命令；op / check / bench 全部委托 GPU
  队列，本地只覆盖纯 Python 接线与前置条件。
- `mm_fp4(..., out=pair.out)` 的 out 语义在 SM120 cute-dsl 上要求
  `out.shape == (m, n)`、BF16、16B 对齐且不与输入重叠；pair 的 `out` 是 capture
  前一次性分配的 `[1, N1+N2]` BF16，满足这些条件。若某个 backend 不支持 `out`，
  DecodePair 会走 `merge_fallbacks.error` 整对回退（数值仍正确，只是没有合并收益）。
- CPU 替身只验证「合并后切半 == 分开调用」的接线；真正的逐位 gate 仍由
  `bench/micro/nvfp4_tactics.py` 的 merged case 与 `m3_check` 负责。
