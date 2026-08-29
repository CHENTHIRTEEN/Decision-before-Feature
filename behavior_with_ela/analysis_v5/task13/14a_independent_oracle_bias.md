# 14a · Independent Winner's-Curse 再诊断（Task 13A）

- 日期：2026-08-29；零新增 objective evaluations（复用 Task 12 重复分支）。
- 动机：Task 12.1 的偏差估计 $\mathrm{median}(r_0,r_1,r_2)-r_0$ 中，$r_0$ 既参与 winner selection 又进入参考值，可能低估偏差。本诊断改用与选择完全独立的参考值：
$$
L_{ind}(s)=\frac{r_1+r_2}{2},\qquad Bias_{ind}(s)=L_{ind}(s)-r_0(s),
$$
其中 $r_0$ winner cell 同 Task 12.1（逐状态最佳动作的 replicate-0 argmin），且该 cell 必须同时拥有 $r_1,r_2$。
- 产物：`independent_oracle_bias.parquet`（汇总）、行级 `results/analysis_v5/task13/independent_oracle_bias_states.parquet`。

## 1. 总体结果（fb = cv_group 平衡；bootstrap 2000 次）

| suite | n | pooled mean | fb mean | 95% CI | 覆盖判定 | Task 12.1 median 估计（对照） |
|---|---:|---:|---:|---|---|---:|
| BBOB | 137 | +0.1003 | **+0.1118** | [0.0203, 0.2234] | ok | +0.1023 |
| MA | 57 | +0.0323 | **+0.0336** | [−0.0028, 0.0720] | ok | +0.0258 |

## 2. 分层（current × FE，n=2–20，全部 INSUFFICIENT，仅列作诊断）

- BBOB：偏差集中在 lshade-current 层（lshade@4000 fb +0.88、lshade@2000 +0.13、lshade@6000 +0.12），shade/cso 层 |bias| ≤ 0.03；
- MA：各层在 −0.09～+0.21 间波动（lshade@6000 为 −0.21），方向不一致。

## 3. 结论

1. **独立估计比 Task 12.1 的 median 估计更大**（BBOB fb +0.102→+0.112；MA +0.026→+0.034），与"$r_0$ 污染导致低估"的假设方向一致，但幅度差异小（≈0.005–0.008）；
2. 偏差的主体不确定性仍在重复覆盖（两 suite 总体 n≥30 为 ok，但 18 个 current×FE 层全部 n<30），BBOB 总体 CI 下界 >0、MA 的 CI 触 0；
3. 维持 Task 12.1 的处理原则：**只作 oracle inflation diagnostic，不外推校正全数据集的逐状态最佳动作参考**。对 Task 13 的含义：以 $L_{current+FE}^{OOF}$（无 min-selection 偏差）为主基线是对的；对 oracle 相关的辅助指标（K4 regret）解释时保留 curse 警示。
