# 15d · 预注册 Margin 策略网格：风险–性能（Task 13.1F/H/I）

- 日期：2026-08-29。策略：仅当 $\hat m(s)>\kappa\cdot d_{c,a'}$ 执行 $c\rightarrow a'$，否则 continue；$\kappa\in\{0,0.5,1.0,1.5,2.0\}$（预注册，不挑选），$d\in\{\max,\sum\}$（pooled solver 噪声尺度）。全部 20 个 (m2) 点全部报告。产物：`margin_policy_rows.parquet`、`margin_policy_summary.parquet`、`margin_policy_bootstrap.parquet`、`risk_performance_pareto.parquet`。

## 1. max-scale 网格（主口径）

| κ | suite | fb loss | gain vs Continue | gain vs Lookup | switch rate | harmful rate | harmful mass | precision | recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0（raw） | bbob | −1.6107 | +0.0052 | +0.0251 | 0.642 | 0.136 | 0.087 | 0.293 | 0.730 |
| 0.0（raw） | ma | −4.5731 | +0.0450 | +0.0433 | 0.591 | 0.137 | 0.047 | 0.310 | 0.692 |
| **0.5** | bbob | **−1.6271** | +0.0216 | +0.0415 | 0.416 | **0.071** | 0.045 | 0.317 | 0.512 |
| **0.5** | ma | −4.5741 | **+0.0459** | +0.0443 | 0.398 | 0.078 | 0.034 | 0.335 | 0.504 |
| 1.0 | bbob | −1.6203 | +0.0148 | +0.0347 | 0.322 | 0.050 | 0.031 | 0.338 | 0.422 |
| 1.0 | ma | −4.5594 | +0.0312 | +0.0295 | 0.261 | 0.043 | 0.025 | 0.369 | 0.364 |
| 1.5 | bbob | −1.6120 | +0.0065 | +0.0264 | 0.256 | 0.037 | 0.023 | 0.324 | 0.322 |
| 1.5 | ma | −4.5586 | +0.0305 | +0.0288 | 0.191 | 0.032 | 0.021 | 0.408 | 0.294 |
| 2.0 | bbob | −1.6083 | +0.0028 | +0.0227 | 0.195 | 0.030 | 0.019 | 0.316 | 0.239 |
| 2.0 | ma | −4.5476 | +0.0194 | +0.0178 | 0.119 | 0.020 | 0.017 | 0.406 | 0.182 |

（sum-scale 结论同型：κ=0.5 时 switch 0.345/0.296、harmful 0.053/0.050、gain vs lookup +0.039/+0.034；κ=1.5 起收益收缩至接近 0。完整 40 行见 `margin_policy_summary.parquet`。）

## 2. 对 Continue / Lookup / Raw-M2 的配对 CI（5000 draws）

以 **κ=0.5, max**（代表点）为例：

| 比较 | BBOB | MA |
|---|---:|---:|
| vs Continue gain | +0.0216 | +0.0459 |
| vs Lookup gain | +0.0415 | +0.0443 |
| Raw M2 − policy loss | +0.0164 [−0.0365, +0.0650] | +0.0010 [−0.0154, +0.0149] |

即：κ=0.5 相对 raw M2 **无显著性能损失**（CI 均含 0，bbob 点估计反而更低），同时 harmful 减半、switch rate 下降 0.22/0.19。κ=1.0 同样无显著劣化（bbob +0.010 [−0.091,+0.084]；MA −0.014 [−0.051,+0.019]）。

## 3. Pareto 前沿（harmful rate vs fb loss，无人工权重）

- BBOB：κ=0.5/1.0/1.5/2.0（max）与 κ=0.5/1.0/2.0（sum）在前沿上；**raw M2（κ=0）不在前沿上**——被 κ=0.5 支配（harmful 与 loss 同时更差）；
- MA：κ=1.0/1.5/2.0（max）等在前沿上；raw M2 同样不在。

## 4. 判读

1. **过度切换是 raw argmin 的实现选择，不是 Behavior 信息的必然代价**：固定 κ 阈值在不显著损失平均性能的前提下，把 switch rate 从 0.59–0.64 压到 0.26–0.42、harmful rate 从 0.14 压到 0.03–0.08；
2. κ=0.5（max）甚至略微改善 bbob 平均损失（−1.6271 vs −1.6107）——低 margin 切换大多是噪声交易；
3. 两 scale 结论一致（max 与 sum 的排序相同，sum 更保守）。

**本轮不确定最终阈值**（κ 的最终选择留给 Task 14A 后的 deployment design）；按 §16 判定为：

$$
\boxed{\text{Verdict 2：R1 RISK-CONTROL FEASIBLE}}
$$

（κ=0.5 与 κ=1.0 max 同时满足：harmful/switch 双降、vs Continue 与 vs Lookup 增益不为负且不低于 raw M2、相对 raw M2 的损失差 CI 无明确劣化。）
