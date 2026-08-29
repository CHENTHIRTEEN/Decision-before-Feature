# 13e · Progress-Gate 结构可行性（Task 12.1J–P）：真实 Future Progress 是否富集 Switch Opportunity

- 日期：2026-08-29
- 定位：**不训练任何 Progress Predictor**；直接使用真实已实现的 1000-FE current-solver progress，即 oracle-progress 的结构上限。若连真实 progress 都无法富集 switch opportunity，则 $B_t\rightarrow\hat R$ 无门价值。
- 产物：`realized_progress_1000.parquet`（行级）、`progress_switch_association.parquet`、`progress_trigger_quantile_sensitivity.parquet`、`progress_noise_diagnostic.parquet`。

## 1. $\ell_t$ 的恢复（J1）：零 FE

Stage-2 表未存 checkpoint gap。两条逐位恒等式验证（1890/1890，diff=0.0）：

1. Stage-1 FE=10000 mark ≡ Stage-2 `terminal_loss_current_run`；
2. Stage-1 FE=t+1000 mark ≡ Stage-2 continue 分支（replicate 0）loss（t∈{2000,4000,6000}）。

故 $\ell_t$ 直接取自 Stage-1 marks（replicate 0），**未执行 replay、未消耗任何 FE**（`replay_alignment.parquet`、`alignment_summary.json`）。

## 2. Realized Progress（J2/K）

$$R_{t,1000}^{current}=10\,(\ell_t-\ell_{t+1000}^{continue})\ \ge 0,\qquad Z(s)=1[current\notin A_{ND}(s)]$$

- R 分布：中位数 BBOB 1.71 / MA 4.64（R 单位=每 1000-FE 的 log10-gap 下降×10）；四分位距 [0.11, 12.4] / [0.31, 10.2]；43%/35% 的 states R<1（近乎停滞）。
- **Low Progress ≠ Switch** 全程保持：progress 只作为 switch-opportunity 检测的候选信号，target 恒为 $Z$，从不预测 target solver。

## 3. Threshold-Free 诊断（L）

| suite | AUROC($-R\rightarrow Z$) | 95% CI | AP | AP 95% CI | base rate $P(Z{=}1)$ | Spearman($-R,G_{best-switch}$) | Spearman($-R,G_{practical}$) |
|---|---:|---|---:|---|---:|---:|---:|
| BBOB | **0.489** | [0.374, 0.622] | 0.230 | [0.153, 0.342] | 0.258 | +0.366 | **−0.035** |
| MA | **0.516** | [0.406, 0.644] | 0.251 | [0.185, 0.363] | 0.265 | +0.361 | **+0.006** |

（function-balanced cv_group bootstrap，2000 次。）

- 两个 suite 的 AUROC 置信区间都覆盖 0.5，AP ≈ base rate：**真实 progress 对 switch opportunity 无判别力**。
- 唯一存在的关联是与 raw best-switch gain 的秩相关（≈0.36），但换用扣除 pairwise 噪声的 $G_{practical}$ 后**归零**——该关联完全落在实用噪声尺度（δ95≈0.08–0.13）之内，不可利用。
- 分层：信号集中于 shade/lshade currents（Spearman 0.31–0.42）与 FE=4000（0.53/0.60）；**cso-current 上 Spearman≈0.01–0.09**，各 current 分层 AUROC 0.32–0.66（n=180/450，噪声范围内波动）——progress 信号不是某个 current/phase 的稳定代理，而是处处接近无信号。

## 4. 固定触发率敏感性（M，预注册 q∈{0.20,0.30,0.40}）

suite-global 分位（current_FE 分层结果同型，见 parquet；MA 分层 q=0.30 enrichment 最高仅 1.12）：

| suite | q | trigger rate | recall | precision | enrichment | missed-switch | call reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| BBOB | 0.20 | 0.200 | 0.075 | 0.096 | **0.37** | 0.925 | 0.80 |
| BBOB | 0.30 | 0.300 | 0.221 | 0.190 | **0.74** | 0.779 | 0.70 |
| BBOB | 0.40 | 0.400 | 0.345 | 0.222 | **0.86** | 0.655 | 0.60 |
| MA | 0.20 | 0.200 | 0.105 | 0.139 | **0.52** | 0.895 | 0.80 |
| MA | 0.30 | 0.300 | 0.280 | 0.247 | **0.93** | 0.720 | 0.70 |
| MA | 0.40 | 0.400 | 0.427 | 0.282 | **1.07** | 0.573 | 0.60 |

**低 progress 触发不仅不富集、在 BBOB 上反而显著贫化**（bottom-20% 的 switch opportunity 率只有 base rate 的 0.37 倍）；q=0.40 时 recall 也仅 0.34/0.43，missed-switch ≥0.57。30–40% 触发率下保留大部分 practical switch opportunities 的门判据不满足。

## 5. 高进展对照（N，exhaustive）

Task 12 对全部 states 做了完整 branching，故高进展对照无需抽样匹配：

- $P(Z{=}1\mid HighProgress)$：BBOB 0.28–0.30（base 0.258）、MA 0.25–0.30（base 0.265）——**高进展 states 携带 switch opportunity 的频率与整体相同**；
- $E[G_{practical}\mid HighProgress]$：0.14–0.17（全部 states 上亦同量级）。

即 progress gate 若上线，会把"当前仍有进展但其它 solver 明显更好"的 states 以与其它 states 完全相同的比例挡在门外——没有任何选择性收益。

## 6. Progress 噪声（P，仅诊断、不预定标）

重复 continue 分支（R=3）给出的 $R^{(r)}$ 波动（fb $Q_{50}/Q_{95}$，R 单位）：BBOB shade 2.03/3.23、lshade 2.08/8.54、cso 0.26/0.97；MA shade 1.41/1.54、lshade 1.48/1.94、cso 0.20/0.62。lshade 的 progress 短程随机性最大。**本轮不预定标最终 $\delta_{progress}$**。

## 7. Verdict（O1）

对照 PG1/PG2/PG3 判据：AUROC/AP 不高于 base rate（✗ PG1）；低 progress 分位不富集反而贫化（✗）；高进展 missed-switch 与 base rate 相同（✗）。

$$
\boxed{\text{ProgressForecast structural verdict：PG3 NO-GO}}
$$

在 $H_g=1000$FE、$P_{balanced}$ 三动作、practical（δ-corrected）语义、当前 state 分布下，**即使真实 Future Progress 完全已知，也不构成有价值的 Switch Opportunity Gate**。据此：$B_t\rightarrow\hat R$ 的预测器即便训练出来也没有门价值——后续 pipeline 不应再为 ProgressForecast 预留位置；此结论为结构性的，且由 oracle-progress 上限证得，强于"先训练再评价"。

边界声明：本 verdict 限于上述 horizon/动作空间/实用语义/训练域（BBOB-train 子集 + selected MA-BBOB）；若未来动作空间、horizon（如 500FE sensitivity）或 state 分布改变，需重新做同型结构核查而非沿用本结论。
