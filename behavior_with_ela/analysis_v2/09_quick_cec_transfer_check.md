# Task 9 前置：CEC2017 三函数快速转移实测（开发性诊断）

- 日期：2026-08-28
- 性质：**开发性 sanity check**，验证 `behavior_action_loss_regression_v2` 在线适配备端到端可用并给出初步转移信号；不是正式 Task 9 评价（后者需全部 28 函数与完整策略面板）。未据此调整任何模型或阈值。
- 设置：CEC2017 10D，F1（单峰）/ F10（多峰）/ F29（组合）× 10 seeds × 3 个起始算法（prefix=all），FE_total=10000，reflect，在线真实执行 one-switch 策略。每策略 90 runs。参照：continue-current 静态全程（同 prefix 同 seed 配对）。
- 实现：`online.py::predict_switch_scores` 新增回归载体分支（协议 `behavior_action_loss_regression_v2`：单回归器输出三算法预测损失，得分 = 预测 continue 损失 − 预测候选损失，阈值 0.145087）；三分类路径未改动。

## 结果（函数平衡；逐函数见 `quick_cec_summary.csv`）

| 策略 | mean log10 gap | 相对 continue 增益 | switch rate | 首次切换 FE（中位） |
|---|---:|---:|---:|---:|
| **v2 regression** | **−1.6030** | **+1.8647** | 0.6667 | 2000 |
| 三分类 classifier | −1.0913 | +1.3531 | 0.7000 | 2000 |
| continue-current | — | 0（参照） | 0 | — |

逐函数（v2 vs 三分类，mean log10 gap）：F1 **−8.79 vs −7.29**；F10 0.671 vs 0.691；F29 3.313 vs 3.323。两策略在全部三个函数上均优于 continue-current。

## 观察与注意

1. **方向与离线一致**：v2 在三个函数上全部不劣于三分类、在 F1 上明显更强，与 Task 1/8 的载体排序一致。
2. **切换时点**：两策略的首次切换都发生在首个决策机会（FE=2000）——与离线 first-trigger 行为相同；策略实质是"开局按预测切换一次"。
3. F10/F29 在 10000 FE 内 success rate 为 0（gap 仍为正），属于难题上的相对比较，绝对水平不代表收敛能力。
4. 本测试的函数/种子规模不支持统计结论；正式 CEC2017 评价仍按总报告 §11 的流程执行（全部 28 函数 + 完整策略面板，且不得据其回调模型）。

产物：`results/online/cec2017_quick3/`（两策略 outcomes/opportunities、静态参照）、`results/analysis_v2/task9_quick_cec/v2_online_bundle.joblib`、`analysis_v2/task9_quick_cec/quick_cec_summary.csv`。
