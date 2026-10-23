# Phase 2：AR decode 优化计划

日期：2026-10-23。目标与格式约束沿用 [roadmap §2–3、§6.2](../roadmap.md)。

## 1. 目标与执行范围

- 在单张 RTX 5080 上优化 Qwen3.8-27B 的 bs=1 AR decode，交付后续 spec 可复用的 target runner。
- 阶段主指标是 **`graph-8192` 的系统 decode tok/s**，模型 tok/s 用于定位开销。
  最终验收配置为 **FP8 KV，达到 ≥80% 逻辑 SOL**。
- 先在 BF16 KV 上优化并验证实现，再接入 FP8 KV；两种格式分别记录性能和 SOL，
  BF16 保留作对照。FP8 接入不阻塞 GEMV、GDN 等独立优化。
- `SOL = 960e9 / B_mean`，`B_mean` 按实际格式及每步有效长度 `L_t` 计算；FP8 的 scale
  等必需存储也计入。capacity 空槽扫描、临时副本等额外流量单列，不能扩大分母使 %SOL 上升。
  当前字节模型对应 FP8 约 51 tok/s，最终以实际实现重算。
- 80% 是工程目标；未达到时如实记录差距，不能自行降低门槛或宣布进入 Phase 3。
- NVFP4 W4A4、GDN recurrent state FP32、conv state BF16、embedding host BF16 保持不变。
  工作范围为 decode、cache 与必要的 graph 集成；TTFT 只记录，spec 留给下一阶段。

**持续探索、集成和验证，直到达到 §7 的阶段验收条件。** 首轮按 §3 执行；每轮结束后，
根据结果和 §4 的前置条件确定下一批有限任务并继续。
仍执行 [Workflow](../optimization-workflow.md) 的停滞暂停规则：连续两轮完整探索无有效提升，
即暂停并在本地 Coordinator journal 记录报告，等待人工介入，即使尚未达到 80%。
用户指定的预算和停止指令同样生效。

## 2. 冻结基准与比较协议

| 项 | 固定值 |
|---|---|
| checkpoint | `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`，revision `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33` |
| correctness reference | M2 `ca1bd2e` 的 eager 语义；执行方式对照沿用 M3 的 padded reference / static / graph |
| 历史性能 baseline | `bff10d3dae0087552a697bc69b1617e8974ef9e6`；结果 `results/m3/20261022T003024-m3-bench/` |
| 首轮开发起点 | `ed52250923e3de90080b996d8cb057b0b2e84718`，包含实验队列；开始时先确认与历史 baseline 同口径可比 |
| 输入 | `bench/workloads/m3-p0/manifest.json` 及其固定 token fixtures，不随候选重新生成 |
| 正式测量 | depth 512 / 7680，生成 512 token，其中 511 次 decode 计时；chunk=128，完整 warmup 一遍，测量 5 次 |
| 必测 graph 场景 | `graph-512-1024`、`graph-512-8192`、`graph-8192`；系统 / 模型计时同时记录 |

历史系统速度分别为 **31.38 / 22.70 / 22.52 tok/s**（BF16 KV）；复现命令为
`source scripts/m2_env.sh` 后 `make m3-bench`，完整证据见
[M3 journal](../journal/2026-10-21-m3-decode-baseline.md) 和 [SOL 字节账](../sol.md)。
新实验通过队列提交对应脚本，协议见 [bench.md](../bench.md)。

每个候选与 baseline 用相同硬件环境、输入、格式、场景和计时边界比较。短测 baseline
按同一短协议另测，不能拿 32 步 / 1 次结果直接对比历史 511 步 / 5 次。
筛选看单步时间；正式确认用 5 次的 median 与离散度。收益接近波动时再做成对复测，
仍无法区分就记为“收益未确认”。microbench 加速不能替代端到端提升，也不能将各项 tok/s 收益相加。

## 3. 首轮探索

先为各任务冻结输入/输出、layout、dtype、reference、数值容差和必测样本，写入对应任务
journal 并提交，再开始候选 kernel。已有 gate 沿用；新增 attention gate 按独立 reference
与 baseline 的误差证据确定，不能看过候选结果再调容差。
同时补该路径 ≤5 分钟的 check / microbench 入口及队列产物校验，不扩建调度系统。

| 任务 | 假设与证据 | 首轮实现范围 | 必须回答的问题 |
|---|---|---|---|
| A：BF16 decode attention | M3 的长 context 热点包含 `repeat_kv` 物化及后续 matmul；紧凑 KV 可减少复制和中间结果 | 保持 BF16 KV，直接读取 4 个 KV head；正确处理有效长度、mask 和 graph replay | 消除展开后，8K 和短 context 的端到端收益、workspace 与峰值显存如何变化？ |
| B：CUDA core NVFP4 GEMV | 当前 `M=1` 调用 GEMM，专用路径可能降低无效工作 | 保留当前 activation quantization；先覆盖 `in_proj_qkv`、`gate_proj`、`lm_head` 的代表 shape | unpack/dequant、访存和 reduction 的代价是否优于现有 `cutlass`？ |
| C：tensor core NVFP4 GEMV | native FP4 与 swap-AB 是另一实现方向，不能仅凭算力或带宽估算选型 | 与 B 使用同一量化输入、权重和计时边界；粗测 `mxf4nvf4` 路径 | 是否胜过 B / baseline？repack、padding、workspace 的成本是否抵消收益？ |
| D：GDN decode | conv / recurrent 路径存在多次小 kernel 和中间 tensor | 从 conv 或 recurrent 中选一个边界明确的片段替换；保留 FP32 recurrent state | 非零历史 state、连续多步下是否正确？减少的 kernel 是否反映到整步时间？ |
| E：activation quant 差异定位 | M3 将独立 quant reference 的差异留作 norm/quant 优化前置项 | 保存失败的 16-element block、input_scale、双方 scale/code、token/layer 和舍入边界距离 | 差异具体来自什么规则？能否冻结融合 kernel 所需的 quant reference？ |

这些任务可独立形成证据；B/C 共用接口，A/D 的 runner 接入统一集成。E 不修改主线
量化语义，也不阻塞沿用当前 quantizer 的 B/C。热点证据与原始 trace 见
[M3 journal §5、§9](../journal/2026-10-21-m3-decode-baseline.md)。

首轮每个实现任务限 **一个实现方向、最多 3 组粗粒度参数、一次正确性修复后重测**；E 限一次
样本归因与验证。未完成或无收益也返回结论。先比较实现，再决定下一轮是否细调参数。
支持代表 shape 不等于已覆盖全模型；接入前补齐实际 shape，或明确未覆盖 shape 回退 baseline。

通过局部 gate 且有收益证据的候选再跑定向端到端确认。B/C 首轮选一个方案接入，
暂不搜索逐 shape 混合策略。将互不冲突的胜出候选组成一个 integration commit，完成一次
组合 gate、三个 graph 场景的正式测量和 8K 显存检查。失败则保留单项证据与失败定位，
在轮末据此决定下一轮的修复或探索任务。

## 4. 后续方向与触发条件

| 方向 | 何时开始 | 实验要求 |
|---|---|---|
| FP8 KV attention | A 的紧凑 KV 路径验证后 | 先冻结 FP8 编码、scale、写入/读取与独立 reference，再实现；BF16 / FP8 单独报告，重算 SOL 与显存 |
| norm + activation quant、gate/up + SiLU·mul 融合 | E 解释差异，且待融合接口稳定后 | 每次只选一个融合边界，单列量化语义变化；先证明正确性，再比较整步收益 |
| 参数细调或新假设 | 集成后重新 profile，仍有可解释的性能差距 | 从新的主要瓶颈提出有限候选；可测试替代实现，不要求沿首轮方案一直调参 |
| 常量预计算、embedding gather/H2D | 时间线证明仍有相关开销时 | `alpha/divisor` 预计算需同设备/dtype 且 bitwise 中性；embedding 保持 host BF16 |

## 5. 验证要求

| 级别 | 检查 | 判定 |
|---|---|---|
| gate | 每个新算子的独立 reference | NVFP4 使用相同量化后的 A/W 做 dequant 对拍；GDN 沿用 [既定容差](../decisions.md)，覆盖真实非零 state 与多步更新；attention 覆盖短/长有效长度、mask、边界和 graph 重放 |
| gate | 量化与布局 | FP4 code/scale、nibble 顺序、scale 布局分别验；FP8 新增独立 quant/dequant reference，不能只看最终 logits |
| gate | 同 token 端到端 | 逐层 hidden、KV/GDN state、logits；reference 固定且不调用候选算子。新增数值容差在实现前冻结；同实现 static→graph、reset 确定性继续要求 bitwise |
| gate | 覆盖与结构 | required 清单完整，reference count 非零且符合预期，finite / dtype / cache length / capacity guard 正确；遗漏必测路径即失败 |
| gate | 8K 完整生命周期 | load、repack、prefill、warmup、capture、graph 常驻后 reset→7680 prefill→511 decode→再次 reset/prefill；无 OOM、stale state 或持续增长，记录分阶段 peak/free |
| gate | 三条真实 smoke | 沿用 math / code / chat 的 EOS、长度和内容检查 |
| diagnostic | 自由生成分叉、FP8 相对 BF16 的量化误差、profile、kernel 带宽 | 保存结果与解释；不能替代上述 gate，也不要求不同格式的 greedy trace 相同 |

现有 `m3_check.py` 主要验证执行方式，不能单独证明新 kernel 正确；新增路径必须进入
独立算子 gate 和端到端 coverage。发生 reduction/layout 变化时，保留原 reference，
说明可比性；不能把原 bitwise failure 直接改成 diagnostic 来接纳候选。

## 6. 反馈阶梯与执行预算

下表 GPU 命令均是队列 job 的步骤，Python 入口使用 `.venv/bin/python`；
`{output}` 由工具展开。耗时是预算，按首个实际 job 修正。

| 档位 | 命令 / 入口 | 耗时预算 | 证明什么 |
|---|---|---|---|
| 静态 | `make verify`、`git diff --check` | <10 秒 | 推荐验证范围、改动格式；不证明数值正确 |
| 新算子 | 各任务新增 check / microbench 的单 shape 入口（待实现并登记准确命令） | 每条 ≤5 分钟；首次编译另列 | 新路径正确性与局部耗时；覆盖实际 shape、边界和 state |
| graph 冒烟 | `scripts/m3_check.py --only graph-1024 --steps 16 --out {output}/check` | 1–3 分钟 | 接入后 graph、cache、计时路径可运行；结合算子 gate 判断 |
| 端到端筛选 | `bench/e2e/decode.py --only graph-8192 --steps 32 --repeats 1 --out {output}/bench` | ≤5 分钟 | 8K 方向性反馈；通过后再测短 context，不作正式提升结论 |
| 单场景确认 | `bench/e2e/decode.py --only graph-8192 --steps 511 --repeats 5 --out {output}/bench` | 5–15 分钟 | 8K 系统收益与噪声；其他 graph 场景分别提交同协议 |
| 8K 显存 | `scripts/m3_memory.py --only graph-8192 --steps 511 --out {output}/memory` | 5–15 分钟 | graph 常驻的完整生命周期；新增 repack/workspace 也需纳入 |
| 定位瓶颈 | `scripts/m3_profile.sh --steps 8 --out {output}/profile` | 5–15 分钟 | kernel/空隙变化；只在集成后或结果无法解释时运行 |
| 阶段关闭 | 完整 check + memory + bench + profile + 实际 SOL 字节账 | 预计 2–3 小时 / 格式，新增项另计 | 候选达到性能目标后执行，确认阶段完整覆盖 |

短测 job 的 `timeout_s=600`，单场景正式测量或含首次编译的 job 为 `1800`；
记录每轮 GPU 执行耗时及其中的编译耗时，用实测成本调整后续任务批次。
job 超时按失败处理，在本轮汇总原因并决定下一步。用户若指定总预算，提交前确认
剩余预算足以容纳该 job；预算不足时暂停并记录已有结果与未完成项。
每条新增路径先提供 ≤5 分钟反馈入口；新脚本须接入
`runner/experiment/artifacts.py`，候选选择与格式必须写入产物并校验。

阶段关闭沿用 M3 全量入口：`scripts/m3_check.py`、`scripts/m3_memory.py`、
`bench/e2e/decode.py --steps 511 --repeats 5`、`scripts/m3_profile.sh`，各带独立 `--out`，
按场景拆 job。新增算子 gate、FP8 场景与 SOL 更新入口在对应任务中补齐后再关闭；
当前 `scripts/m3_sol.py` 尚未注册到队列，不能直接作为已支持的步骤提交。
任何裁剪都记录缺项，不把短档结果视作全量完成。

## 7. 交付与阶段验收

每轮交付：候选 commit / job / 结果路径，单项与组合收益，正确性和显存结论，失败原因，
下一轮建议。记录到 `docs/journal/`；原始产物留在 `results/experiments/<job-id>/`。
每个结论附实际命令、commit、checkpoint revision 和可比配置；Coordinator 日志按 Workflow 记录。

Phase 2 完成需同时满足：

1. FP8 主线通过全部 gate，BF16 对照仍可复现；所有新增路径都有 ≤5 分钟验证入口。
2. FP8 `graph-8192` 系统速度达到 ≥80% 实际逻辑 SOL；三个 graph 场景均正式测量，
   没有未解释的性能回退。
3. 组合后的时间线与字节账解释主要耗时；已发现的明显复制、同步、launch 或融合问题
   已处理，或有实验说明继续投入的收益有限。未达到速度目标仍算未完成。
4. 完整 gate、显存、性能和剩余瓶颈有可复现记录，冻结供下一阶段使用的 integration commit。
