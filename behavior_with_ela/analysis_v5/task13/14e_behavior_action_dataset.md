# 14e · Behavior-Action Dataset 构造（Task 13F）

- 日期：2026-08-29。产物：`results/analysis_v5/task13/behavior_action_dataset_task13.parquet`（1890 行 × 49 列）。

## 1. 构造方式

以 `state_id` 为主键做三次 one-to-one 精确 join：

1. Task 12 solver 语义损失矩阵：`loss_shade / loss_lshade / loss_cso`（current=X 时为 continue，否则为 switch-to-X 的 1000-FE 真实 outcome）；
2. Task 13C replay 的 28 列 selector Behavior（bf_*，经 `behavior_replay_alignment.parquet` 验证逐位对齐后拼接）；
3. Task 13B（max 规则）的 practical 集合：`switch_required / A_ND_members / A_ND_size / switch_target`。

另含 `continue_loss`（Task 12 continue 分支真实 outcome）、`raw_best_action`（逐状态最佳动作，仅辅助对照）、`FE_ratio`、全部元数据列。**raw loss 矩阵完整保留，未做任何删除。**

## 2. 内容与完整性

| 检查 | 结果 |
|---|---|
| 行数 | 1890 = Task 12 Stage-2 全部 states（bbob 1350 + mabbob 540） |
| 主键唯一性 | state_id 唯一，无缺失 |
| bf_* 列数 | 28（selector 契约） |
| switch-required 基率 | 0.2598（与 13B max 规则一致） |

## 3. 命名说明（契约对照）

工作单使用 $B^{global}$/bg_* 记号；仓库正式契约（`behavior/features.py`）的无成熟度全局行为列为 **bf_\***（28 列 selector 集）。本数据集与全部下游模型**保持仓库正式列名 bf_\***，两者映射关系为 bg_*(工作单记号) ≡ bf_*(本仓库契约)；maturity 3 列与 diagnostic 3 列留在特征表中但不进入任何模型。

## 4. 下游使用

- 主监督目标（13G）：多输出连续动作损失 $Y(s)=[L(s,\text{SHADE}),L(s,\text{L-SHADE}),L(s,\text{CSO})]$，部署动作为 $\hat a(s)=\arg\min_a \hat L_a(s)$；practical 集合仅用于风险指标（K5/K6）与分层，不压缩成分类标签；
- 评价全部使用真实 Task 12 action outcome（零新增 action-label FE）。
