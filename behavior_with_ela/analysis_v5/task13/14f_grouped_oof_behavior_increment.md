# 14f · Grouped-OOF Behavior 增量（Task 13G–M）

- 日期：2026-08-29。协议：leave-cv_group-out（BBOB 10 / MA 12 折，与 Task 12.1 的 $L_{current+FE}^{OOF}$ 同一 split 哲学），suite 内拟合；主 carrier 为项目正式 pipeline（WeightedMedianImputer→StandardScaler→RandomForestRegressor(200, depth 8, sqrt, fixed seed)），Ridge(alpha=1) 为唯一低复杂度对照；评价全部使用 Task 12 真实 1000-FE action outcome；**new action-label FE = 0**。
- 产物：`oof_policy_performance.parquet`、`oof_increment_gains.parquet`、`oof_prediction_diagnostics.parquet`、行级 `results/analysis_v5/task13/{oof_action_loss_predictions, oof_policy_rows}.parquet`。

## 1. 主表（fb realized loss；K1 selected-action loss）

| carrier | suite | M3 current | M4 FE | **M0 current+FE** | M1 B | **M2 current+FE+B** | $\Delta_B=L_{M0}-L_{M2}$ | 95% CI（paired cv_group bootstrap 2000） |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RF（主） | BBOB | −1.5767 | −1.5212 | −1.5634 | −1.6054 | **−1.6107** | **+0.0473** | [−0.0503, +0.1743] |
| RF（主） | MA | −4.5068 | −4.4756 | −4.5219 | −4.5732 | **−4.5731** | **+0.0513** | [+0.0025, +0.1059] |
| Ridge | BBOB | −1.5767 | −1.5676 | −1.6012 | −1.5952 | −1.6231 | +0.0219 | [−0.0676, +0.1445] |
| Ridge | MA | −4.5091 | −4.5139 | −4.5299 | −4.5231 | −4.5265 | −0.0034 | [−0.0217, +0.0153] |

- 主假设检验（13M，$H_1: L_{M2}<L_{M0}$）：**MA（RF）拒绝 H0**（CI 下界 +0.0025>0）；BBOB（RF）点估计为正但 CI 穿 0；Ridge 两个 suite 均不能拒绝。
- K3 gain vs continue（辅助）：M2 = +0.0052 / +0.0450（两 suite 为正；M0 在 BBOB 为 −0.042，即 current+FE 策略比盲目 continue 更差的 suite 存在）。
- K4 regret to observed statewise best（辅助，含 winner's-curse 警示）：M0 0.196/0.168 → M2 0.148/0.117。
- **M1≈M2**：$\Delta_{M2-M1}$ ≈ 0（BBOB −0.005 [−0.015,+0.003]、MA +0.000 [−0.034,+0.046]）——current+FE 叠加在 Behavior 之上**不再提供增量**；Behavior 单独已包含全部可提取信号。

## 2. 与 Task 12.1 基线的衔接（§33 问 7）

Task 12.1 经验 argmin 基线 $L_{current+FE}^{OOF}$=−1.5856/−4.5298；本轮 M0：RF −1.5634/−4.5219（carrier 拟合噪声使其差 0.022/0.008）、Ridge −1.6012/**−4.5299**（MA 几乎精确复现）。split 哲学一致；主比较 $\Delta_B$ 采用**同 carrier 内配对**（M0 与 M2 同折同拟合），不受该 carrier 偏移影响。

## 3. 预测诊断（13L，仅辅助）

| 量（RF/M2） | BBOB | MA |
|---|---:|---:|
| per-action Spearman（shade/lshade/cso） | 0.494/0.487/0.480 | 0.855/0.852/0.834 |
| MAE / RMSE | 2.289/3.380 | 1.934/2.556 |
| raw top-1 accuracy（chance 0.333） | 0.489 | 0.498 |
| pairwise ordering accuracy（chance 0.500） | 0.653 | 0.659 |

M0 的对应 Spearman 仅 0.14–0.26；M4（FE only）≈0；M3（current only）为负——**时间相位单独无信息，current 单独是弱反指示**，Behavior 是预测力的主要来源。

## 4. 判读

Behavior 在 grouped-OOF 下**方向一致地**优于 current+FE（RF 两 suite 点估计 +0.047/+0.051），其中 MA 达到显著（CI>0）、BBOB 受限于 fold 方差（CI 穿 0）；Ridge 无法稳定提取该信号（低复杂度载体的能力边界，非协议失败）。K5/K6 风险指标见 14i。按 §18 判定为 **A2 CONDITIONAL**（点估计正、MA 显著、BBOB CI 穿 0）。
