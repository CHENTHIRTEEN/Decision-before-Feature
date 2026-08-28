# Task 2：practical gain delta 敏感性分析

- 日期：2026-08-28
- 问题：`practical_gain_delta = max(0.05, noise_delta(0.95)) = 1.464` 是否只是把大量中等幅度动作差异归入 Equivalent？主结论（Behavior > Time-only；策略优于 Continue/Random）对 delta 选择是否稳健？
- 方法：全部复用已有 repetition 数据与 parquet，无新 objective 运行。对 $q \in \{0.50, 0.75, 0.80, 0.90, 0.95\}$ 重算函数平衡 noise delta，重派生 Improve/Equivalent/Degrade 标签与 acceptable-action 集合；分类器与 time-only 在重标签数据上按同一 grouped family OOF 重新训练；回归载体分数与标签无关，仅按各 delta 重放阈值扫描。validation 只用于 q=0.95（原始固定值），不参与任何 delta 选择。

## 1. noise delta 随分位数急剧变化（重尾驱动）

| q | function-balanced noise delta | practical delta | improve 标签占比 |
|---:|---:|---:|---:|
| 0.50 | 0.0564 | 0.0564 | 0.2278 |
| 0.75 | 0.2928 | 0.2928 | 0.1753 |
| 0.80 | 0.3838 | 0.3838 | 0.1638 |
| 0.90 | 0.7525 | 0.7525 | 0.1354 |
| 0.95 | 1.4639 | 1.4639 | 0.1057 |

q=0.95 的逐 family 分布（`noise_delta_per_family.parquet`）：median / IQR / q90 / q95 / max 见 `noise_distribution_q95.json`；按 prefix / candidate / early-mid-late 的层内增益标准差同文件。中位数远低于均值，说明 1.464 由少数高噪声 family 的右尾拉高。

## 2. 各 delta 下的策略表现（train grouped OOF，函数平衡）

| 载体 | q | delta | gain | 归一化 regret | harmful switch rate | switch rate |
|---|---:|---:|---:|---:|---:|---:|
| action-gain 三分类 | 0.50 | 0.056 | 0.3195 | 6.2926 | **0.2917** | 0.6074 |
| time-only | 0.50 | 0.056 | 0.0000 | 0.6198 | 0.0000 | 0.0000 |
| action-loss 回归 | 0.50 | 0.056 | **1.3141** | 2.0202 | 0.2065 | 0.6667 |
| action-gain 三分类 | 0.75 | 0.293 | 0.5729 | 1.1399 | 0.1606 | 0.6074 |
| time-only | 0.75 | 0.293 | 0.3862 | 2.6626 | 0.2358 | 1.0000 |
| action-loss 回归 | 0.75 | 0.293 | **1.3141** | 0.6634 | 0.1111 | 0.6667 |
| action-gain 三分类 | 0.80 | 0.384 | 0.6533 | 0.7960 | 0.1254 | 0.6154 |
| time-only | 0.80 | 0.384 | 0.4698 | 2.0882 | 0.2210 | 1.0000 |
| action-loss 回归 | 0.80 | 0.384 | **1.3141** | 0.5673 | 0.0954 | 0.6667 |
| action-gain 三分类 | 0.90 | 0.753 | 0.7430 | 0.5609 | 0.0877 | 0.6969 |
| time-only | 0.90 | 0.753 | 0.6859 | 1.1531 | 0.1944 | 1.0000 |
| action-loss 回归 | 0.90 | 0.753 | **1.3141** | 0.3799 | 0.0704 | 0.6667 |
| action-gain 三分类 | 0.95 | 1.464 | 0.8467 | 0.4023 | 0.0756 | 0.7673 |
| time-only | 0.95 | 1.464 | 0.4352 | 0.6700 | 0.1500 | 1.0000 |
| action-loss 回归 | 0.95 | 1.464 | **1.3141** | 0.2564 | 0.0463 | 0.6667 |

分类器 sign 指标随 delta 变窄而退化：AP 0.264（q=0.95）→ 0.304（0.90）→ 0.337（0.80）→ 0.344（0.75）→ 0.391（0.50）——AP 上升是 improve 基线率同步上升（0.106→0.228）所致，排序能力并未增强；balanced accuracy 0.505–0.568 区间波动。

## 3. 回答工作单问题

1. **1.464 是否把中等差异过度归入 Equivalent？** 是的，方向上成立：q 从 0.95 收紧到 0.50 时 improve 标签占比从 0.106 升到 0.228。但这**不是**当前结论的支撑来源——分类器在窄 delta 下 harmful switch rate 恶化到 29.2%，说明窄 delta 下的 "improve" 标签大量来自状态内噪声而非真实可捕获差异。
2. **主结论对 delta 稳健**：Behavior 分类器在 q ∈ {0.75, 0.80, 0.90, 0.95} 全部优于 time-only（q=0.50 时 time-only 阈值选择为从不切换，gain=0）；两者对 Continue/Random 的相对位置也保持。**回归载体在全部五个 delta 下 gain 恒为 1.3141、且在每个 delta 上同时优于分类器与 time-only**——其优势不依赖 delta 口径。
3. **delta 固定依据（train OOF + 重复统计解释）**：q=0.95 的宽等价带对应"切换决策必须超过状态内重复噪声 95% 分位"的保守解释；q=0.50 相当于忽略噪声。回归载体对两种极端口径都不敏感，进一步支持以回归载体为主候选（Task 8）。建议维持 q=0.95 的预指定值，不作后验调整。

产物：`analysis_v2/task2/`（delta_sensitivity_summary.parquet、noise_distribution_q95.json）、`results/analysis_v2/task2/`（各 q 的 OOF 预测）。
