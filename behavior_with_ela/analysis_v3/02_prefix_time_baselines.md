# 02 · Task 9B：Prefix-only / Fixed-0.20 / Behavior@0.20 对照（本轮最关键 baseline）

- 日期：2026-08-29
- 问题：v2 的收益来自 state-dependent Behavior 信息，还是主要来自 prefix 身份 + 固定 0.2B 切换时点？
- 方法：四个新 baseline 全部只用 train grouped-family OOF 拟合（validation 不参与任何规则或阈值拟合）；对照面板补齐 fixed-0.30、time-only、continue-current、prefix 条件单次切换上界（历史字段 `best_observed_one_switch`，见 Task 9G 命名说明）。不执行 objective evaluation。
- 产物：`analysis_v3/task2/{baseline_policy_panel,paired_delta_vs_v2_summary}.parquet`；run 级明细与 OOF 表在 `results/analysis_v3/task2/`。

## 1. 四个新 baseline 的定义

| 名称 | 输入 | 决策规则 |
|---|---|---|
| B1a `prefix_only_table` | 仅 prefix 身份 | 按 (prefix, candidate) 的 train 函数平衡平均增益表 + first-trigger 阈值（OOF 扫出 0.1451 口径下为 -0.0678） |
| B1b `prefix_plus_fe_table` | prefix + FE | 按 (prefix, FE, candidate) 增益表 + first-trigger 阈值（0.7102） |
| B2 `fixed_020_mapping` | 仅 prefix，仅 FE=2000 | 每个 prefix 取平均增益最大的候选（增益>0 才切），由训练数据学得 |
| B3 `behavior_at_020` | 28 维 Behavior，仅 FE=2000 | 与 v2 完全相同的回归载体，但只在 0.2B 决策一次（OOF 阈值 -0.0082） |

B2 学到的映射（train 函数平衡平均增益，非手写）：**pso→cmaes (+3.086)、shade→cmaes (+1.895)、cmaes→不切**。

## 2. 统一比较（函数平衡）

| 策略 | train OOF gain | train 归一化 regret | val gain | val 归一化 regret | train/val switch rate | train/val G<0 |
|---|---:|---:|---:|---:|---:|---:|
| prefix-only（B1a） | 1.3137 | 0.2565 | **1.8504** | 0.1636 | 0.667 / 0.667 | 0.231 / 0.103 |
| prefix+FE（B1b） | 1.3137 | 0.2565 | **1.8504** | 0.1636 | 0.667 / 0.667 | 0.231 / 0.103 |
| Fixed-0.20 mapping（B2） | 1.3137 | 0.2565 | **1.8504** | 0.1636 | 0.667 / 0.667 | 0.231 / 0.103 |
| Behavior@0.20（B3） | 1.3137 | 0.2565 | **1.8504** | 0.1636 | 0.667 / 0.667 | 0.231 / 0.103 |
| **Full dynamic v2** | **1.3141** | 0.2564 | **1.8504** | 0.1636 | 0.667 / 0.667 | 0.231 / 0.103 |
| Fixed-0.30（既有） | 1.0076 | 0.2943 | 1.5094 | 0.2411 | 0.667 / 0.667 | 0.232 / 0.156 |
| time-only | 0.4352 | 0.6700 | 0.8411 | 0.5973 | 1.000 / 1.000 | 0.350 / 0.387 |
| continue-current | 0.0000 | 0.4210 | 0.0000 | 0.5394 | 0 / 0 | 0 / 0 |
| prefix 条件单次切换上界 | 2.2096 | 0.0000 | 2.2080 | 0.0000 | 0.671 / 0.785 | 0 / 0 |

注：所有新 baseline 的 regret 分母统一重算为全机会 best-observed（B3 的 replay 只见 FE=2000 机会，若直接取其内部 regret 会低估分母；重算后与 v2、phase-1 面板严格同口径）。完整面板（terminal loss、success rate、目标分布、切换 FE 分位）见 `baseline_policy_panel.parquet`。

## 3. 最核心检验：$\Delta_{Behavior}$

$$\Delta_{Behavior}=\mathrm{Perf}(v2)-\mathrm{Perf}(\text{prefix-only}/\text{fixed-0.20})$$

| 对照 | train OOF 逐 run 配对差（函数平衡） | validation 配对差 | v2 更好 run 占比（train/val） |
|---|---:|---:|---:|
| v2 − prefix-only | **+0.0003** | **0.0000** | 0.12% / 0.00% |
| v2 − prefix+FE | +0.0003 | 0.0000 | 0.12% / 0.00% |
| v2 − Fixed-0.20 | +0.0003 | 0.0000 | 0.12% / 0.00% |
| v2 − Behavior@0.20 | +0.0003 | 0.0000 | 0.12% / 0.00% |
| v2 − Fixed-0.30 | +0.3064 | +0.3410 | 34.1% / 46.5% |
| v2 − time-only | +0.8789 | +1.0093 | 49.4% / 59.1% |

train OOF 上 +0.0003 的微小差距来自 v2 在 6 个 run 上的非 2000 时点切换（3 次 @2200、1 次 @2520、1 次 @3000）与 2 次 pso→shade 切换；validation 上 v2 与三个 baseline 的 540 个 run 决策**完全一致**（差值恰为 0）。

## 4. 结论

1. **v2 ≈ Prefix-only ≈ Fixed-0.20 ≈ Behavior@0.20，在两个 split 上同时成立。** Behavior 特征在当前阈值与 first-trigger 规则下没有提供 prefix 身份之外的可行动信息；后续动态决策机会同样没有贡献（v2 与只在 0.2B 决策一次的 B3 决策完全一致）。
2. v2 相对 Fixed-0.30 与 time-only 的优势是真实的（+0.31/+0.34 与 +0.88/+1.01），但其来源是"时点选择（0.2B）+ 目标选择（cmaes）"，这两者都可以由一张三行查表表达。
3. 按工作单判据：**不能把当前 1.8504 的 validation 收益归因于 Behavior**。研究定位必须修改——不是 "strongly dynamic behavior scheduling"，而是 **trajectory-conditioned early algorithm selection**（且主要是 prefix 条件的早期切换）。
4. 附带发现：B1a/B1b/B2/B3 四条实现路径（增益表、含时间表、固定映射、完整回归）收敛到同一策略，说明该策略在当前数据上是稳定的学习结果，不是某次实现的偶然产物。
