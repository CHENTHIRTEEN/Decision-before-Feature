# Decision-before-Feature：Task 13.1-H 零 FE 校准卫生修正 总报告

- 日期：2026-08-29；HEAD 基线 = Task 13.1 提交 `a688fdf`。
- 定位：完全零新增 FE 的 hygiene patch——把 Task 13.1 部署噪声尺度从"全部 development repetitions pooled"改为 **fold-local（每个 held-out cv_group 只用训练 groups 估计）**，重跑预注册 κ 网格并对照。
- 报告：`15i_fold_local_noise_calibration.md`、`15j_fold_local_margin_policy_reanalysis.md`；代码 `task13_1h_analysis.py`；行级表 `results/analysis_v5/task13_1/{fold_local_margin_policy_rows, task13_1h_resource_ledger}.parquet`。

## 一、泄漏性质声明

原 pooled $\delta_a^{pool}$ 使每个 OOF fold 的阈值尺度含有 held-out group 的重复变异性。这**不是** action-label leakage（标签仍全部来自既有 outcome），也**不是**模型训练 leakage（M2 OOF 预测未变），而是 **OOF calibration leakage**——影响的是 threshold 校准的纯度，因此必须在 Task 14A 消耗确认数据前修正。

## 二、修正内容与复现

1. H1：对 22 个 cv_group 各估计 $\delta_{a,95}^{(-g)}$（同一 replicate-deviation/Q95/函数平衡/solver-cell 语义，仅排除 held-out group）；
2. H2：pair scale = max（主）/ sum（保守），由 fold-local δ 合成；
3. H3：重跑预注册 κ∈{0, 0.5, 1.0, 1.5, 2.0} × {max, sum}（无新增、无挑选）；
4. 复现检查：fold 表的 pooled 列重建后与已提交 `margin_policy_summary.parquet` 全部一致（|diff| ≤ 1e-9）——复现通过，未触发 STOP。

## 三、结果（细节见 15i/15j）

1. **尺度稳定性**：排除任一 group 只改变 δ 约 2–4%（cso 0.067–0.080、lshade 0.101–0.121、shade 0.070–0.078 vs pooled 0.076/0.115/0.074；平均绝对差 0.002–0.004）；
2. **策略曲线几乎不变**：κ=0.5（max）fold-local：switch 0.417/0.396、harmful 0.073/0.078、gain vs Continue +0.020/+0.046、vs Lookup +0.040/+0.044；κ=1.0：harmful 0.051/0.043；
3. **pooled vs fold-local 配对差**：|point| ≤ 0.008、方向不定，CI 除两处贴 0 的小幅度区间外均含 0——无系统性劣化。

## 四、Hygiene Verdict 与 13 问回答

$$
\boxed{\text{Verdict：H1 NEGLIGIBLE}}
$$

| # | 问题 | 回答 |
|---|---|---|
| 1 | 每 fold 的 δ_a | 见 15i 表（22 fold × 3 solver 明细） |
| 2 | fold-local vs pooled 差异 | 平均绝对差 cso 0.003 / lshade 0.004 / shade 0.002（≈3–4%） |
| 3 | κ=0.5 switch/harmful 仍低于 raw M2？ | **是**（0.417/0.396 vs 0.642/0.591；0.073/0.078 vs 0.136/0.137） |
| 4 | κ=1.0 风险优势保持？ | **是**（harmful 0.051/0.043；switch 0.324/0.259） |
| 5 | gain vs Continue 方向一致？ | **是**（κ=0.5/1.0 两 suite 均为正） |
| 6 | gain vs lookup 方向一致？ | **是**（κ=0.5：+0.040/+0.044；κ=1.0：+0.033/+0.030） |
| 7 | pooled-vs-fold loss 差 CI 含 0？ | 基本含 0；两处小幅贴 0（幅度 ≤0.005），最大差 0.008 且方向不定 |
| 8 | R1 是否仍成立 | **是**（κ=0.5/1.0 max 的全部 R1 条件逐项复核通过） |
| 9 | κ=0.5 保留？ | **是**：performance-oriented pre-fixed candidate（性能导向的预先固定候选点） |
| 10 | κ=1.0 保留？ | **是**：risk-oriented pre-fixed candidate（风险导向的预先固定候选点） |
| 11 | hygiene verdict | **H1 NEGLIGIBLE** |
| 12 | 是否允许进入 Task 14A | **是（GO 不变）** |

两个 operating points（κ=0.5 与 κ=1.0）共同进入后续确认协议；Task 14A **不得**用 confirmation data 再调 κ。本轮不写"κ=0.5 is selected"。


> 措辞说明：工作单原文的候选点标签依 AGENTS.md §0.3 的用语规范改写为 "pre-fixed candidate"（预先固定候选点），含义不变。

## 五、成本与停止声明

| 项 | 值 |
|---|---:|
| new objective FE | **0** |
| wall time | ≈2 min（`task13_1h_resource_ledger.parquet`） |

按工作单 §14 链条：读取 Task 13.1 → 复现 pooled → fold-local 尺度 → 固定 κ 网格 → pooled 对照 → 配对 bootstrap → hygiene verdict → **STOP**。未自动执行 Task 14A；ProgressForecast 维持 PG3 NO-GO；CEC2017 PAUSED / CEC2022 HELD OUT。

## 十二、下一步建议

执行 **Task 14A**（GO 已三次确认：Task 13.1 §28、本轮 H1 后不变）。推荐延续 Task 13.1 总报告 §十一 的 prompt，并追加一条约束：**margin policy 的部署常数一律使用 fold-local 语义估计（本 patch 的 `task13_1h_analysis.py` 流程），Task 14A 的确认数据不得参与任何 δ 或 κ 的估计。**
