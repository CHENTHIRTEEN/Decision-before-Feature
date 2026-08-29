# 15f · Problem-ID Diagnostic 解释修正（Task 13.1L）

- 日期：2026-08-29。本报告只做解释降级声明，不删除任何旧 artifact，不重跑该实验。

## 1. 为什么旧诊断不能解释为"模型知道测试 problem identity"

Task 13 的 13R 诊断把 `[problem, current, FE]` 与 `+B` 作为 one-hot 特征在 leave-cv_group-out 下评价。对任一 held-out cv_group：

- 该 group 的 problem_id（如 `bbob_f016_i02_d10`）对应的 dummy 列在训练集中**恒为 0**；
- 模型在训练时从未见过该 problem 的任何正例，测试时该输入维度不携带"已识别该问题"的信息；
- 因此 `D_problem_*` 的 OOF 数字**不能**解释为"problem identity 已知时 Behavior 是否仍有增量"的检验。

## 2. 处置

| 项 | 处置 |
|---|---|
| `problem_id_diagnostic.parquet` 与 `oof_problem_id_diagnostic_predictions.parquet` | **保留不删除**，标记 `INVALID AS KNOWN-PROBLEM DIAGNOSTIC / LEGACY ONLY` |
| Task 13 总报告 14i §3 中"problem identity 已知时仍 +0.043/+0.056"一句 | **撤销该解释**：该数值仍然成立（作为 one-hot 特征消融的 legacy 记录），但其含义是"加入 problem 哑变量列后拟合结构的变化"，而非已知问题身份下的增量 |
| 正式替代证据 | **within-problem LOSO（Task 13N）+ 本轮 100 次 permutation（15e）**：直接固定 (problem, current, FE) 并对 seed 外推，才是"Behavior 是否超出 problem identity"的合法检验 |

## 3. 对 Task 13 结论的影响

- Verdict B（B1 GENUINE STATE VALUE）**不受影响**：其证据来自 13N 的 within-problem LOSO 与 shuffle null，与被降级的 one-hot 诊断无关；
- Task 13 总报告其余数字不引用该诊断；
- 本轮总报告（Task 13.1）将以 100 次 permutation 的 p 值作为 within-problem 信号的最终稳健性口径（15e）。
