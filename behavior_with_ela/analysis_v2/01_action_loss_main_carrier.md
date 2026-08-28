# Task 1：behavior_action_loss_regression_v2 主候选载体

- 日期：2026-08-28
- 定义：28 维 Behavior 特征 → 多输出 `RandomForestRegressor(200, depth 8, sqrt)` → $[\hat L_{PSO}, \hat L_{SHADE}, \hat L_{CMAES}]$ → 预测增益
  $$\hat G(s,a)=\hat L(s,\text{continue})-\hat L(s,a)$$
  → first-trigger 切换/继续决策（阈值经 train grouped family OOF 确定）。
- 与 Phase 1 `behavior_action_loss_rf` 同构，本报告将其形式化为主候选并补充完整指标；三分类模型保留为对照。
- 产物：`analysis_v2/task1/`（summary.json、score_metrics_oof.csv、unified_policy_panel.parquet）与 `results/analysis_v2/task1/`（OOF、阈值表、run 级明细）。

## 1. 回归质量（train grouped family OOF，BBOB switch 行 50,938）

| 候选动作 | 行数 | loss MAE | loss RMSE | 增益 Spearman |
|---|---:|---:|---:|---:|
| pso | 16,575 | 3.3353 | 4.3490 | +0.3173 |
| shade | 17,099 | 4.1456 | 4.8711 | +0.4912 |
| cmaes | 17,264 | 4.3884 | 4.9665 | **−0.2576** |

- **pairwise action ranking accuracy（两 switch 候选内排序正确率）：0.7607**
- **top-1 action accuracy（含 continue 的三动作具体最优命中）：0.6453**
- 注意：cmaes 候选行的增益 Spearman 为 **−0.258**（排序方向整体反转），与三分类模型中 CMA-ES AP 低于基线率的现象一致（详见 03 报告的 prefix 分解）。

## 2. 策略指标（函数平衡）

| split | gain | 归一化 regret | switch rate | acceptable rate | harmful switch rate | switch FE p10/p50/p90 |
|---|---:|---:|---:|---:|---:|---|
| train OOF | 1.3141 | 0.2564 | 0.6667 | 0.8568 | 0.0463 | 2000 / 2000 / 2000 |
| validation | 1.8504 | 0.1636 | 0.6667 | 0.9593 | 0.0000 | 2000 / 2000 / 2000 |

注意两个行为特征：
1. **first-trigger 几乎全部发生在首个决策机会（FE=2000）**：分数分布稳定使阈值在最早的监控点即被越过。该策略行为上接近"开局即按预测切换一次"。
2. **验证集 harmful switch rate = 0**：在 practical delta = 1.464 口径下无有害切换。

## 3. 十方法统一比较（函数平衡；完整表见 unified_policy_panel.parquet）

| 策略 | train OOF gain | train 归一化 regret | val gain | val 归一化 regret |
|---|---:|---:|---:|---:|
| continue-current | 0.0000 | 0.4210 | 0.0000 | 0.5394 |
| random one-switch | −0.3684 | 1.0229 | −0.3201 | 1.1383 |
| random-matched | −0.1913 | 0.9083 | −0.1363 | 1.0059 |
| fixed-0.30 one-switch | 1.0076 | 0.2943 | 1.5094 | 0.2411 |
| time-only | 0.4352 | 0.6700 | 0.8411 | 0.5973 |
| action-gain 三分类（Phase 1 主模型） | 0.8467 | 0.4023 | 1.4481 | 0.2561 |
| **behavior_action_loss_regression_v2** | **1.3141** | **0.2564** | **1.8504** | **0.1636** |
| to-switch-style RF | 0.9219 | 0.5297 | 1.4821 | 0.3241 |
| best-observed one-switch（逐状态最佳动作，仅诊断上界） | 2.2096 | 0.0000 | 2.2080 | 0.0000 |

SBS 为静态全程单算法参照（cmaes，训练派生），口径为全程 terminal log10 gap 而非单次切换增益：train mean log10 gap −5.192（success rate 0.406），validation −5.061（0.333）；VBS 相应为 −5.837 / −5.073。完整面板见 `unified_policy_panel.parquet`。

## 4. 结论

1. 回归载体在两个 split 上均全面优于三分类载体（val gain +0.40，regret −0.09）、优于 fixed-0.30 与 to-switch 风格基线，且验证集 zero harmful switch。
2. 相对逐状态最佳动作上界（validation 2.208），v2 捕获约 **83.8%** 的单次切换可得增益（1.8504/2.2080）。
3. 支持将 `behavior_action_loss_regression_v2` 列为正式主候选（Task 8 裁决）。
