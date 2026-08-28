# Task 6：action uncertainty 诊断（树级方差与预测边际）

- 日期：2026-08-28
- 背景：Phase 2 的 M4 bootstrap uncertainty 度量的是"descriptor 估计稳定性"而非"动作决策不确定性"，其 validation 退化（−0.167）不能解读为"uncertainty 无用"。本诊断改用动作决策本身的不确定性来源：A1 回归集成的树级结构。仅诊断，不实现 Query Gate。
- 数据：BBOB train 25,469 个 state；A1 集成 200 棵树的全树预测（无新 objective 运行）。

## 1. 决策与错误率（delta=1.464，阈值 0.1451）

| 统计 | 数值 |
|---|---:|
| 决策切换率 | 0.6777 |
| harmful switch rate（切换后真实增益 < −delta） | 0.0291 |
| missed helpful switch rate（未切换但存在 > +delta 的真实增益） | 0.0024 |

## 2. 不确定性特征识别高风险决策的能力（AUC，风险=1）

| 目标 | 特征 | AUC（低值→风险） | 风险态均值 | 安全态均值 |
|---|---|---:|---:|---:|
| harmful switch | switch gain margin（|预测切换增益|） | **0.779** | 0.751 | 1.660 |
| harmful switch | top1-top2 loss margin | 0.656 | 0.734 | 1.185 |
| harmful switch | 树级 argmin 分歧率 | 0.134（即高分歧→风险 AUC≈0.866） | 0.183 | 0.050 |
| harmful switch | best-switch 树方差 | 0.631 | 1.382 | 1.644 |
| missed helpful switch | switch gain margin | 0.581 | 1.185 | 1.635 |

完整表：`uncertainty_auc.csv`；按 margin 十分位的 harmful rate：`harmful_rate_by_margin_decile.csv`。

## 3. 结论

1. **动作级不确定性确实携带风险信号**：有害切换显著集中于小 |预测切换增益| 处（AUC 0.779），且树间 argmin 分歧越高越危险（AUC ≈ 0.87）。这直接回答了工作单问题："action uncertainty 能否识别高风险决策？"——能，尤其是增益边际与树间分歧。
2. 有趣的反直觉点：best-switch 树方差在风险态反而更低（1.38 vs 1.64，低方差→风险 AUC 0.63）——有害切换多发生在"集成一致但一致地错"的状态，与 SHADE→CMAES 难例（Task 3）互相印证：这类状态的错误是系统性的，不是采样噪声。
3. 对后续的含义：若未来重启 ELA/Query 研究，Query 的触发器应优先考虑"增益边际小 / 树分歧大"的 action uncertainty，而不是 descriptor bootstrap uncertainty。当前阶段按工作单要求暂停该方向。

产物：`analysis_v2/task6/`（summary.json、uncertainty_auc.csv、harmful_rate_by_margin_decile.csv）。
