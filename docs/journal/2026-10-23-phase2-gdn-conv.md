# Phase 2 / 任务 I：GDN conv decode 片段融合（Triton kernel + 模型接入）

日期：2026-10-23。任务来源：[phase2 plan §3](../plans/phase2.md) 任务 D 的第二个边界
（conv step）；容差与口径见 [decisions 的 M2/M3 条目](../decisions.md)。
分支 `exp/phase2-gdn-conv`，基于第 1 轮集成 commit **`9e35873`**。

## 状态

| 项 | 结论 |
|---|---|
| 候选 | `runner/ops/gdn/conv_decode.py`（1 kernel/layer）+ `runner/model/gdn_patch.py` 追加 conv patch |
| 冻结 commit | `4a16d0e3cad5afaf7cd6913442ea3430525915a2` |
| 正确性 | op **524/524** gate 全过（4 个 case）；check graph-1024 **24/24**；既有 bitwise gate 一个没改 |
| 性能（同协议 32 步 × 1 次） | graph-8192 **27.781 → 27.514 ms**（−0.267 ms / −1.0%），系统 35.99 → **36.30 tok/s**（+0.9%）；graph-512-1024 **26.779 → 26.550 ms**（−0.229 ms / −0.9%），系统 37.29 → **37.62 tok/s**（+0.9%） |
| 显存 | 8K free 0.329 GB 不变；peak allocated +0.98 MB（48 个 conv out 槽 0.94 MB），reserved 15.848 GB 不变 |
| 结论 | **采纳**。局部 microbench 0.271 ms → e2e 0.267 / 0.229 ms，收益完整迁移 |
| job | `job-000030-c4bd4215`（v1，gate 失败，3.8 s）、`job-000034-486a8274`（v2，succeeded，180.5 s） |

产物：`results/experiments/job-000034-486a8274/output/{op,check,bench}`。
baseline 对照：`results/experiments/job-000029-16bb35a9/output/bench`（同脚本、同 fixtures、
`--steps 32 --repeats 1`，与 511×5 正式值差 <0.2%）。

## 1. 边界与设计

只替换 `causal_conv1d_update` 的 decode 分支，即
`use_precomputed_states and seq_len == 1 and not record_past` 那一个调用点。
recurrent step（第 1 轮已融合）、`in_proj_*`、out_proj、gated RMSNorm 都不在范围内；
prefill 走 `causal_conv1d_fn`（另一个函数），完全不受影响。

### 1.1 原路径：每层 4 个 tiny kernel

reference 是 transformers 在没有 `causal_conv1d` 时的纯 PyTorch 展开：

```
hidden_states_new = torch.cat([conv_state, mixed_qkv], -1)   # 1 个 CatArrayBatchedCopy
conv_state.copy_(hidden_states_new[..., -state_len:])        # 1 个 direct_copy
out = F.conv1d(hidden_states_new, weight[:, None], ...)      # conv_depthwise2d_forward_kernel_generic
out = out[..., -1:]
out = ACT2FN["silu"](out)                                    # 1 个 silu_kernel
```

`job-000028`（集成后 graph-7680，8 步）里这四类 kernel 各 48 个/step，累计约
**275 µs/step、192 个 node**：cat 48×1.2 µs + copy_ 48×~1.1 µs +
conv_depthwise 48×1.9 µs + silu 48×1.5 µs。单个 kernel 都 <2 µs，代价是
launch/依赖链而不是带宽。

### 1.2 融合 kernel（`runner/ops/gdn/conv_decode.py`）

一个 program 处理 `BLOCK` 个 channel（`grid = 10240/BLOCK`，默认 128 → 80 个 CTA）：

```
win   = [s1, s2, s3, x]                         # conv_state 的 1..3 列 + 当前 token
conv  = Σ_j weight[:,j] * win[:,j]              # fp32 累加（bf16 乘积在 fp32 里精确）
conv  = round_bf16(conv)                        # 先复刻 reference 的 F.conv1d 输出舍入
out   = conv / (1 + exp(-conv))                 # PyTorch silu 的 bf16 opmath 形式
conv_state[:, :] = win                          # 原地滚动（地址/dtype/字节不变）
out   → 独立 out 槽（[1,10240,1] bf16）
```

关键事实与对齐点：

1. **`state_len == conv_kernel == 4`（不是 3）**。prefill 的 `update_conv_state` 把
   conv_state 初始化成 `[1, C, conv_kernel_size]`，decode 时
   `cat([state(4), x(1)])` 长度是 5、conv 输出 2 个、reference 取最后一个：
   `out = w0*s1 + w1*s2 + w2*s3 + w3*x`，新 state 恰好等于同一个窗口 `[s1,s2,s3,x]`。
   这个「窗口 == 新 state」的等式只在 `state_len == kernel` 时成立，host 侧显式检查，
   不满足就回退（`state-len-kernel-mismatch`）。任务描述里写的 `[1,10240,3]` 与源码不符，
   以源码/实测为准。
2. **两次舍入**。reference 是 `F.conv1d` 的 bf16 输出（fp32 累加、一次舍入）**再**做 silu。
   kernel 先把 conv 舍入到 bf16、再用 fp32 算 silu 并舍入，实测与 reference 的 nrmse
   1.3e-9（实质逐位）；如果省掉中间那次舍入（fp32 conv + fp32 silu），误差会被放大到
   ~1e-3 量级，反而过不了 M2 的容差。
3. `libdevice.exp`（`__nv_expf`，与 `torch.exp` 在 fp32 上逐位一致）+ `div_rn`
   （`div.rn.ftz.f32`，对齐 nvcc 的 `div.rn.f32`）；不用 `tl.sigmoid`/`/`（会落到
   `ex2.approx` / `div.full.f32`）。PTX 里是 `fma.rn.f32.bf16`（bf16 操作数、fp32 累加），
   与「乘积精确、只舍入一次」的设计一致。
4. **conv_state 原地**：dtype/字节/`data_ptr` 不变（static cache 与 graph replay 依赖），
   每 channel 只归一个 program，跨 program 无写竞争。
5. **x 只读、out 走独立槽**。`mixed_qkv` 是 `in_proj_qkv(hidden).transpose(1,2)` 的 view
   （shape `[1,10240,1]`、strides `(10240,1,10240)`，PyTorch 仍判 contiguous，因为
   size-1 维不参与判定）。kernel 只依赖 channel 维 stride=1；channel stride≠1 的布局
   必须回退（`x-stride`）。输出写进按 `conv_state.data_ptr()` 缓存的 48 个槽
   （48 × 20 KB = **983040 B ≈ 0.94 MB**），capture 之前分配、之后地址固定。

理论瓶颈是延迟：每层流量只有 state 读 80 KB + weight 80 KB + x 20 KB、写 80 KB +
out 20 KB ≈ 280 KB，48 层约 13 MB/step（weight 48 层共 3.9 MB 跨 step 常驻 L2），
在 960 GB/s 上只有 ~14 µs；实测 48 层 0.047 ms，即 **0.98 µs/层**，基本是 kernel
launch/短 CTA 的延迟下限。

### 1.3 接入（`runner/model/gdn_patch.py` 追加）

- 接入点是模块级 `transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_update`。
  `forward` 在调用点做全局名字查找，本机 `kernels` 未安装所以 `use_kernelized_func`
  是 no-op，替换模块属性即可覆盖 eager / static-exact / static-cap /
  padded-reference / graph 五种执行方式，**没有复制模型源码**。
- `install_conv(model)` 与第 1 轮的 `install(model)`（recurrent）**分开装**，任何一条
  失败不牵连另一条；`install_from_env` 两条都装，`GDN_FUSED_DECODE=0` 整体关闭；
  `uninstall()` 同时恢复两个模块级函数。
- 命中条件（全部 host 侧检查，失败计数到 `conv.counters.fallbacks`）：单 token、
  `mixed_qkv` 是 CUDA bf16 `[1,10240,1]` 且 channel stride=1、conv_state 是 CUDA bf16
  `[1,10240,4]` contiguous、weight 是 contiguous bf16 `[10240,4]`、`bias is None`、
  `activation in (None, "silu")`、conv_dim 一致、`state_len == kernel` 且为 2 的幂。
- 计数是**活 dict**：`load.gdn_decode.conv.counters` 出现在 check.json 的 `load`
  diagnostic 和 bench.json 的 `results[*].load` 里，所以「capture 进去的到底是不是
  融合 kernel」有产物级证据（见 §3.2/§3.3）。

## 2. 冻结的 reference / 容差 / 覆盖

| 项 | 值 |
|---|---|
| reference | 未修改的 `causal_conv1d_update`（cat + copy_ + F.conv1d + silu） |
| 容差 | M2 冻结：`nrmse ≤ 1e-3`、`cosine ≥ 0.99999`、`max_rel ≤ 5e-3` |
| shape | layers=48, conv_dim=10240（2×16×128 + 48×128）, conv kernel=4, state_len=4, bf16 |
| state | 非零随机历史（`randn×0.1`，48 层 `max|state|` ≈ 0.4–0.6），不是全零 |
| 递推 | 连续 2 步；第 2 步双方各自从第 1 步写完的 state 出发（不共享 buffer） |
| 负对照 | 全零 state（≈「只看当前 token」）、state 错位一格、weight 反向，各自与**正确** reference 比 |
| 接线门 | 真实 config 的 **meta** GDN 模块 + 真实 CUDA 张量走 `causal_conv1d_update` 模块级入口 |

`scripts/phase2_gdn_conv_check.py` 的 4 个 case（op 524 gate）：

| case | 证什么 | 数字 |
|---|---|---|
| `conv-two-steps-48-layers` | 正确性（kernel 直调，48 层 ×2 步） | worst nrmse **1.29e-9** / max_rel 2.7e-8；conv_state 与 reference **逐位相同**；`out=None` 新分配且与 out 槽路径逐位一致；x 只读；silu 生效（vs 不做激活 nrmse 0.50） |
| `conv-negative-controls` | gate 的分辨力 | 正对照 nrmse 0；三个错实现 nrmse **1.447 / 1.253 / 0.878**，cosine −0.037 / 0.224 / 0.479（阈值 10×1e-3） |
| `patch-conv-wiring-real-shapes` | patch 真的被执行 | `fusion=96, reference=0, fallbacks={}`；48 槽 / 983040 B / 跨步 data_ptr 稳定；真实 transpose 布局 `strides (10240,1,10240)`；fp32 回退计数 + 逐位一致；`GDN_FUSED_DECODE=0` 关闭；uninstall 恢复 |
| `microbench-conv-baseline-vs-candidate` | 局部耗时 + 3 组粗参数 | graph replay 48 层 **0.318 → 0.047 ms**（6.74×）；eager 1.136 → 0.416 ms（2.73×） |

## 3. 结果

### 3.1 op（局部，48 层）

| 项 | baseline（cat + copy_ + conv1d + silu） | 候选 | 变化 |
|---|---|---|---|
| 每层（graph replay） | 6.63 µs | 0.98 µs | −85% |
| 48 层一步（graph replay） | 0.318 ms | **0.047 ms** | **−0.271 ms（6.74×）** |
| 48 层一步（eager） | 1.136 ms | 0.416 ms | −0.720 ms（2.73×） |

粗参数（48 层 eager step）：`block64-w2` 0.41597 < `block128-w4` 0.41683 <
`block256-w4` 0.41754 ms，三组差 <0.4%，说明已经到 launch/延迟下限；默认保留
`block128-w4`（80 个 CTA，与 5080 的 SM 数匹配）。

### 3.2 check（`job-000034`，graph-1024 × 16 步）

24/24 gate 通过；`load.gdn_decode.conv.counters = {fused: 960, reference: 0, fallbacks: {}}`。
960 = 48 层 × 16 步（eager 参考路径）+ 48 层 × (3 warmup + 1 capture)。
唯一 diagnostic 差异仍是 `graph: 逐步 argmax 与冻结 trace 一致` 的第 3 步
（top-2 margin 0.5625），与 M3/第 1 轮记录的 sub-ulp 放大同类，非 gate。

### 3.3 bench（同协议 32 步 × 1 次）

| 场景 | baseline `job-000029` | 候选 `job-000034` | 变化 |
|---|---|---|---|
| graph-8192 step p50 | 27.781 ms | **27.514 ms**（mean 27.518，min 27.500，max 27.606，std 0.018） | **−0.267 ms / −1.0%** |
| graph-8192 系统 | 35.95 tok/s | **36.30 tok/s** | +0.97% |
| graph-8192 模型 | 35.99 tok/s | 36.34 tok/s | +0.97% |
| graph-512-1024 step p50 | 26.779 ms | **26.550 ms**（mean 26.551，std 0.011） | **−0.229 ms / −0.9%** |
| graph-512-1024 系统 | 37.29 tok/s | **37.62 tok/s** | +0.88% |
| 8K free / peak allocated / reserved | 0.329 GB / 15.288 GB / 15.848 GB | 0.329 GB / 15.289 GB / 15.848 GB | free、reserved 不变；allocated +0.98 MB（out 槽） |
| `conv.counters`（bench worker） | — | `{fused: 192, reference: 0, fallbacks: {}}` | 48 × (3 warmup + 1 capture) |

`fused=192` 是「capture 进去的是融合 kernel」的直接证据：graph replay 不再经过 Python，
所以计数不随 32 步增长；若走 fallback，`reference` 会是 192 而 `fused` 为 0。
（`load.gdn_decode.conv.out_slots/out_bytes` 显示 0 是 **install 时刻的快照**，与第 1 轮
recurrent 的同名字段同一个已知限制；真正的 48 槽证据在 op 的 `coverage:out-slot-cache`。）

### 3.4 局部收益与端到端收益

| 深度 | microbench（graph replay） | e2e step p50 | 差 |
|---|---|---|---|
| 8192 | 0.271 ms | 0.267 ms | −0.004 ms（−1.5%） |
| 512 | 0.271 ms | 0.229 ms | −0.042 ms（−15%） |

两个 depth 的实测差 0.038 ms，与短协议噪声（std 0.011–0.018 ms）和图内 kernel 交错
差异同量级；方向与量级一致，**没有需要额外解释的部分**。这一点和上一轮不同：第 1 轮
e2e 比 microbench 多省 0.4–0.6 ms（当时记为未解释，怀疑是参考路径在真实 graph 里
的额外分配/交错成本），本轮 microbench 已经用「48 层完整 step + graph replay」作边界，
没有出现同方向偏差。

## 4. 设计取舍

1. **选 conv 而不是 g/β**。conv 的接入点是模块级函数（与第 1 轮同一类 seam，已证明
   可 patch、prefill 不受影响），而 g/β 的预计算只存在于 `forward` 的内联表达式里
   （`-A_log.exp()*softplus(a+dt_bias)`、`b.sigmoid()`），要接缝只能 shim
   `modeling.F.softplus` 或复制 forward —— 同一个方向里风险/收益比更差。
   g/β 的 node 数（每层约 5 个、合计 240 node/step）留到下一轮单独评估。
2. **放弃 in-place，改 48 个固定 out 槽**。第一版把 `out=None` 定义成就地覆盖 `x`
   （语义上等价：模型随后把返回值重新绑定到 `mixed_qkv`）。第一版 gate 失败后改成
   独立槽：`x` 严格只读、与 reference 的 buffer 语义一致、不再依赖「`in_proj_qkv`
   的输出没有别的消费者」这个隐式假设。代价是 0.94 MB 常驻（8K free 0.329 GB 下可忽略），
   换来的是一个可判定的契约和一条结构性 gate。
3. **先舍入 bf16 再做 silu**（见 §1.2 第 2 点）。这是本轮唯一一处「看起来可以更准、
   实际必须更钝」的选择：多带的 fp32 精度反而让节点过不了 M2 容差。
4. **`state_len == kernel` 才命中**。「窗口 == 新 state」的恒等式是 kernel 一次读写
   的基础；换成 `state_len = kernel-1` 的布局（任务描述假设的 3）需要额外一次 state
   读或不同的滚动写法，host 侧直接回退而不猜。

## 5. 限制与已知问题

- **`GDN_FUSED_DECODE=0` 只在 `install_from_env` 读一次**（与第 1 轮相同）：已经装上的
  进程中途改环境变量无效；两个 patch 一起关。
- **channel stride≠1 的 `mixed_qkv` 会回退**（`x-stride`，计入 fallbacks）。本模型
  `in_proj_qkv(...).transpose(1,2)` 满足 channel stride=1；若未来换 GEMM 布局或插入
  slice，`counters.fallbacks` 会立刻显示、性能退回 baseline，不会静默算错。
- **conv_state 约束**：必须是 CUDA bf16 `[1, conv_dim, state_len]` contiguous、
  `state_len == conv kernel` 且为 2 的幂；dtype/地址由 cache 保证。kernel 原地写回，
  不可换成 ring buffer（snapshot 与 bitwise gate 依赖固定布局）。
- **`x` 只读假设**：kernel 只读 `hidden_states`，输出写 out 槽；但 out 槽按
  `conv_state.data_ptr()` 缓存，若同一 conv_state 被复用到不同 conv_dim 的层会重建
  （info 里能看出槽数变化）。
- **依赖 Triton 3.8 + sm_120a**：`libdevice.exp`/`div_rn`/`fma.rn.f32.bf16`；换 GPU
  或 Triton 大版本需重编复测。首次编译 ~1 s（3 组参数各一次），已计入 op job 耗时。
- **out 槽在 capture 期间分配的风险**：现有所有入口都先 warmup（`GraphDecodeStep`
  强制 warmup 并校验 cache 前进），所以槽在 capture 之前建立；若未来有 runner 直接
  capture，槽会进 graph private pool（功能正确，只是地址不再属于普通 pool）。
- op 的 `provenance.commit` 是候选 commit，`dirty=false`；`results/experiments/` 不入库。

## 6. 失败记录（v1 → v2）

v1（`06e8cf2328059f7ce44b9f6662f0e6853b78bbdc`，`job-000030`）在 op 阶段
`gate_failed`，5 个失败项全部是**判据 bug**，kernel 本身在显式 out 槽路径上
nrmse 0（逐位等于 reference）：

1. **`numeric:in-place-path:*`（nrmse 0.69 / cosine 0.76）**：reference 用了候选
   **调用之后**的 state 快照当初值，两边差一整步。修法：reference 必须从调用前的
   快照出发；同时按上面的取舍放弃 in-place，这条 gate 换成
   `coverage:out-none-allocates-and-keeps-input` + `coverage:input-read-only` +
   `coverage:out-slot-cache`。
2. **三个负对照 nrmse 恒为 0**：我拿「reference 喂**同样的错输入**」当参照，两边
   一致地错，等于没有对照。修法：负对照必须与「正确输入的 reference」比；同一分布下
   预检 nrmse 0.86 / 1.21 / 1.44，实际 op 是 0.88 / 1.25 / 1.45（阈值 0.01）。
3. **修完之后加的 CPU dry-run 又抓到一次同类 bug**：`out=None` 的 reference 也用了
   调用后的快照。现在 `scripts/phase2_gdn_conv_check.py` 的 case 1/2 可以用「等价
   torch 实现冒充 kernel」在 CPU 上完整跑一遍并写 op.json（本次 204/204 + 102/102），
   这类判据 bug 不必再花 GPU job 才能发现。

**教训**：这两个 bug 都表现为 gate 失败而不是静默通过 —— 「独立 reference + 负对照 +
接线门」的组合是有效的；但判据本身也要有低成本的验证通道（CPU dry-run），
否则每修一次都要一次 GPU 往返。

## 7. 复现

```bash
# 局部 gate + microbench（≤5 分钟，首次编译约 1 s 另计）
.venv/bin/python scripts/phase2_gdn_conv_check.py --out {output}/op
# 执行方式 / cache / reset / 边界 / smoke
.venv/bin/python scripts/m3_check.py --only graph-1024 --steps 16 --out {output}/check
# 端到端（短协议，不能与 511 步 × 5 次的正式数字直接比较）
.venv/bin/python bench/e2e/decode.py --only graph-8192,graph-512-1024 \
    --steps 32 --repeats 1 --out {output}/bench
```

CPU 侧（不需要 GPU）：`python -m py_compile runner/ops/gdn/*.py runner/model/gdn_patch.py
scripts/phase2_gdn_conv_check.py`；Triton kernel 用 `triton.compile` +
`GPUTarget("cuda", 120, 32)` 离线编译成 sm_120 cubin（EVEN/SILU/BLOCK 4 个 variant 都过）；
`gdn_patch` 的接线逻辑与 `_conv_fallback_reason` 用 CPU 单测；case 1/2 的判据用
CPU dry-run（等价 torch 实现冒充 kernel）。

队列请求与产物（`session_id=phase2-gdn-conv`）：

```json
{"session_id": "phase2-gdn-conv",
 "commit": "4a16d0e3cad5afaf7cd6913442ea3430525915a2",
 "timeout_s": 2400,
 "steps": [
   {"kind": "op", "argv": [".venv/bin/python", "scripts/phase2_gdn_conv_check.py",
                           "--out", "{output}/op"], "result": "{output}/op/op.json"},
   {"kind": "check", "argv": [".venv/bin/python", "scripts/m3_check.py", "--only", "graph-1024",
                              "--steps", "16", "--out", "{output}/check"],
    "result": "{output}/check/check.json"},
   {"kind": "bench", "argv": [".venv/bin/python", "bench/e2e/decode.py", "--only",
                              "graph-8192,graph-512-1024", "--steps", "32", "--repeats", "1",
                              "--out", "{output}/bench"], "result": "{output}/bench/bench.json"}]}
```

## 8. 集成注意（交给 Coordinator）

- 接入文件：`runner/ops/gdn/conv_decode.py`（新）、`runner/ops/gdn/__init__.py`（导出
  conv 符号）、`runner/model/gdn_patch.py`（追加 `install_conv`/`_fused_conv`/`_conv_*`/
  `_CONV` 状态与 out 槽）、`scripts/phase2_gdn_conv_check.py`（新）。没有改
  `runner/experiment/*`、`bench/e2e/*`、`scripts/m3_check.py`、`Makefile`。
- 与第 1 轮（recurrent）在同一个文件里，但装/卸互相独立；两者共用
  `GDN_FUSED_DECODE` 开关。
- 与任务 H（norm+quant）不冲突；与任务 G（ShapeStableLinear）无关。
- 组合集成时：两个 op gate（`phase2_gdn_check.py` + `phase2_gdn_conv_check.py`）都要
  在组合 commit 上通过，`m3_check` 的三组 bitwise 是硬门；bench 用同协议对照。

## 9. 下一步建议（GDN 方向）

conv 已经只剩 **0.047 ms / 48 层（0.98 µs/层）**，再调参只在这个池子里，收益 <0.01 ms。
GDN 每层的剩余成本（`job-000028` profile）按「可融合边界」排序：

1. **gated RMSNorm + z-gate 的 sigmoid·mul**：`torch_recurrent` 之后 `core_attn_out`
   reshape 成 `[48,128]`，`self.norm(core_attn_out, z)` 里是 pow/mean/rsqrt/mul/cast +
   `sigmoid(gate)` + mul + cast，每层约 5–6 个 tiny kernel；`reduce_kernel` 全图 210 个/step
   （0.314 ms）里有 48 个是它。接入点是 **类方法** `Qwen3_5RMSNormGated.forward`
   （48 个模块实例，与 conv 同一种「模块级 seam」），可以一个 kernel 出结果；
   量级估计 0.1–0.15 ms/step。风险：norm 的 reduction 语义（`mean` vs `rsqrt` 的顺序、
   weight 乘法在 cast 之前还是之后）必须逐位复刻，否则会像 conv 的两次舍入一样被
   1 ulp 放大；需要独立 reference + 负对照（换权重的顺序 / 不做 gate）。
2. **g/β 预计算**：`-A_log.exp()*softplus(a+dt_bias)` 与 `b.sigmoid()` 每层约 5 个
   kernel（[48] 元素），240 node/step ≈ 0.19 ms kernel + 一部分 node 间隙。没有干净的
   函数 seam，需要用 `modeling.F` 的 shim 或复制 forward（第 1 轮拒绝过的风险），
   或者等 `in_proj_a/b` 的 epilogue（任务 G 的 ShapeStableLinear 侧）顺手算掉。
3. **`repeat_interleave(3, dim=2)`（96 个 tiny kernel ≈ 0.15 ms/step）**：第 1 轮就
   留在桌上；唯一干净的做法是 source-derived 的 forward patch + 一条「copy 与原生
   forward 逐位一致（关闭融合时）」的独立 gate，否则会重新打开「patch 没生效也全绿」
   的盲区（第 1 轮 §6.2 的教训）。优先级低于 1。
4. **不建议**再动 conv kernel：0.98 µs/层已经接近 kernel launch 的延迟下限，
   三组 block 参数差 <0.4%；除非把 48 层合成一个 kernel（需要跨层指针数组，接入面
   不支持）。
