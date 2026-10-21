# <里程碑编号>：<标题> 执行计划

- 对应：[roadmap](../roadmap.md) §<x>；日期：<YYYY-MM-DD>；状态：<计划 / 实施中 / 完成>
- 目标：<一句话，可验收>
- 非目标：<明确不做的部分>

## 1. 反馈阶梯（必填）

每个交付路径都要能在小 loop 里验证；全量只在关闭时跑。写清楚每档证明什么、
耗时多少，以及"新增路径由哪一档兜底"。

| 档位 | 命令 | 耗时预算 | 证明什么 |
|---|---|---|---|
| S0 静态 | `py_compile` / `git diff --check` / `make check-results` | < 10 s | 语法、产物契约 |
| S1 CPU | `pytest tests/...` / 纯 CPU 语义测试 | < 10 s | 单元数值、报告语义 |
| S2 单场景 | `<harness> --only <case> --steps <n>` | 1–3 min | 改动路径的最小闭环 |
| S3 快档 | `make <milestone>-quick` | ≤ 5 min | 端到端冒烟（含显存/计时） |
| S4 定向全量 | `<harness> --only <affected>` | 10–30 min | 受影响的完整指标 |
| S5 关闭全量 | `make verify-close` | 预算内 | gate + diagnostic 全量 |

DoD：**新增路径必须有一个 ≤5 分钟的档位覆盖它**；没有就先把 harness 的可裁剪
开关（`--only` / `--steps` / `--repeats`）补上，再写实现。

## 2. 验收（gate / diagnostic 分开）

| 级别 | 检查 | 判定 |
|---|---|---|
| gate | <必测项> | <通过条件> |
| diagnostic | <观察项> | <只解释，不作门槛> |

必测项要写进脚本的 `required` 清单（缺项 = gate failure），不能只检查"至少有一个 gate"。

## 3. 产物、预算与复现

| 产物 | 路径 | 预算 |
|---|---|---|
| <run dir> | `results/<milestone>/<run-id>-<kind>/` | <分钟 / GB> |

- 每个 run dir 必须带 `provenance.json`、`source_manifest.json`（dirty 时加
  `git_diff.patch`），并用 `make check-results` 校验。
- 长跑（>10 min）先估算时间，写进计划的预算列；超预算时缩小场景集，不要干等。

## 4. 关闭命令（一条命令、可重入）

```bash
source scripts/m2_env.sh
make verify-close          # 依次跑 check / memory / bench / profile / sol
make check-results         # 校验产物契约
```

失败重入：长跑必须支持 `--resume` 或按场景重跑（`--only`），不要重头再来。

## 5. 已知限制与后续项

- <未做的项，以及为什么可以留到后面>
