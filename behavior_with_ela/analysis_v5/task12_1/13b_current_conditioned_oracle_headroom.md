# 13b · Current-conditioned Oracle Ladder 与性能差分解（Task 12.1B/C/D/E）

- 日期：2026-08-29
- 数据：同一 Stage-2 state set（1890 states × 3 solver cells，1000-FE solver 语义 log10 损失），零新增 objective evaluations。
- 产物：`oracle_ladder_current_conditioned.parquet`、`oracle_headroom_current_conditioned_bootstrap.parquet`（2000 次 cv_group 重采样 95% CI）、`oracle_inflation_repeated_subset.parquet`。
- 记号：$L_{statewise}$ = 逐状态最佳动作（state-wise best observed action）fb 损失；上下文经验策略（descriptive context policy）为"在全量开发数据上按 fb 均值逐组选最优动作"，仅作诊断上限，部署不可用。

## 1. 完整 Ladder（fb log10 损失，越低越好）

| suite | $L_{SBS}$ | $L_{current}$ desc | $L_{current+FE}$ desc | $L_{problem}$ desc | $L_{problem+FE}$ desc | $L_{problem+current+FE}$ desc | $L_{problem+current+FE}$ LOSO-seed | $L_{current}$ OOF | $L_{current+FE}$ OOF | $L_{statewise}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BBOB | −1.5501 | −1.6055 | −1.6117 | −1.6078 | −1.6453 | −1.7059 | −1.6573 | −1.5791 | −1.5856 | −1.7590 |
| MA | −4.4956 | −4.5397 | −4.5719 | −4.5312 | −4.6048 | −4.6401 | −4.5859 | −4.5245 | −4.5298 | −4.6902 |

- B1/B4/B5/B7 与 Task 12 报告数值一致（$L_{SBS}$、$L_{problem+FE}$、$L_{statewise}$ 复现误差 <0.002），确认同口径。
- C2（LOSO-seed）：descriptive problem+current+FE 从 −1.7059/−4.6401 退到 −1.6573/−4.5859，**seed 内过拟合约 0.049/0.054**。
- C3（leave-cv_group-out OOF）：**$L_{current+FE}^{OOF}$ = −1.5856 / −4.5298**，这是下一阶段 Behavior test 应优先超过的可部署 simple baseline；它已比 SBS 好 +0.035/+0.034。

## 2. 性能差分解（D1–D5，fb，95% CI）

| 量 | 定义 | BBOB | MA |
|---|---|---:|---:|
| $\Delta_{portfolio}$ | $L_{SBS}-L_{statewise}$ | **+0.2088** [0.1249, 0.3082] | **+0.1946** [0.1222, 0.2805] |
| $\Delta_{deploy-residual}$ | $L_{current+FE}^{OOF}-L_{statewise}$ | **+0.1734** [0.0792, 0.2941] | **+0.1604** [0.1130, 0.2050] |
| $\Delta_{context-residual}$ | $L_{problem+current+FE}^{desc}-L_{statewise}$ | **+0.0531** [0.0340, 0.0740] | **+0.0501** [0.0288, 0.0782] |
| $\Delta_{problem-info}$ | $L_{current+FE}^{OOF}-L_{pcf}^{LOSO-seed}$ | +0.0716 [0.0060, 0.1653] | +0.0560 [0.0074, 0.1051] |
| $\Delta_{dynamic}^{old}$ | $L_{problem+FE}-L_{statewise}$（Task 12 原指标） | +0.1137 [0.0727, 0.1597] | +0.0854 [0.0565, 0.1217] |

判读：

1. **旧 $\Delta_{dynamic}^{old}$ 大部分被 current identity 吃掉**：conditioning 从 problem+FE 升级为 problem+current+FE 后，BBOB 0.114→0.053（−54%）、MA 0.085→0.050（−41%）。旧指标对 Behavior 价值的指向性下调，自本轮起仅作对照（对 Behavior 价值解释而言 superseded）。
2. **可部署口径下（只知 current+FE）**，剩余逐状态性能差 0.173/0.160，CI 均不含 0——这是 Behavior 增量测试形式上要竞争的空间。
3. **但 $\Delta_{context-residual}$ 仅 ≈0.05**（两 suite 一致），且 95% CI 上界 0.074/0.078 低于 suite 短程噪声 δ95（0.088/0.084）的单次量级；它与 §3 的逐状态最佳动作乐观偏差同量级，解释时必须成对阅读。

## 3. 逐状态最佳动作的 winner's-curse 诊断（Task 12.1E，仅诊断、不外推校正）

重复子集（argmin cell 恰好被抽中重复，R=3）上比较 replicate-0 与 median-of-repeats：

| suite | n（重复 argmin cell states） | bias fb | bias pooled | 分层覆盖 |
|---|---:|---:|---:|---|
| BBOB | 137 | **+0.1023** | +0.0928 | 9 个 current×FE 层 n=11–20，全部 INSUFFICIENT；极端层 lshade@FE4000 fb +0.85 |
| MA | 57 | **+0.0258** | +0.0215 | 9 层 n=2–10，全部 INSUFFICIENT |

- $Bias_{oracle}=L_{oracle}^{median}-L_{oracle}^{r0}\ge 0$ 在 BBOB 上点估计 ≈0.10，与 $\Delta_{context-residual}$（0.053）同量级、约为 $\Delta_{deploy-residual}$（0.173）的六成；MA 上较小（0.026）。分层估计高度不稳定（含 −0.11～+0.85 的离群层），**不得据此对全数据集做校正外推**（按工作单 §E 只作 `oracle inflation diagnostic` 报告）。
- 含义（保守读法）：$L_{statewise}$ 作为逐状态上限本身带 0～约 0.10（BBOB）量级的乐观偏差，因此 $\Delta_{deploy-residual}$ 的真值区间应理解为约 [0.07, 0.17]（BBOB）、[0.13, 0.16]（MA，curse 较小）；$\Delta_{context-residual}$ 与 curse 偏差不可区分，即"完整上下文之后剩余的真实逐状态价值"目前**无法与 0 可靠区分**。
- 根因：R=3、10% 抽样的重复覆盖不足以稳定识别该偏差。若后续要以可接受成本钉住它，需对少量 states 扩重复（新增 FE，另行授权），或接受它作为已知不确定性带入 verdict（本轮采取后者，见总报告 V2）。

## 4. 对 Behavior GO 门的影响

工作单 §27 门（$\Delta_{deploy-residual}$ 明显非零）在名义口径下满足（两 suite CI>0）；但依据 §3，该量的点估计含逐状态最佳动作的乐观成分，BBOB 上不确定区间宽。对应总报告 verdict 取 **V2 CONDITIONAL P1**：允许进入 Behavior Incremental Test，但定位为 CONDITIONAL DEVELOPMENT TEST，且测试必须以 $L_{current+FE}^{OOF}$（而非任何 oracle）为超越目标。
