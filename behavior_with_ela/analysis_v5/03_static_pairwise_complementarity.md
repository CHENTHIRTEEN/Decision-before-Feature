# 03 · Static Pairwise Complementarity（Task 12C）与 05 · Stage 1 筛选

- 日期：2026-08-30
- 指标：$DCM_{ij}=\frac{S_{ij}+|C_{ij}|}{2}$，$S=P(A_i\sim A_j)$（practical tie，±δ_FE,suite），$C=P(A_i\succ A_j)-P(A_j\succ A_i)$。**DCM 越低越互补**；高 |C| 表示单边支配。
- 产物：`analysis_v5/task11/../task12/static_pairwise_dcm.parquet`（21 对 × 2 suite × 4 FE）。

## 1. BBOB FE=6000 的代表性关系（δ95≈1.87 的宽 practical 带）

| pair | P(tie) | P(a≻b) | P(b≻a) | DCM |
|---|---:|---:|---:|---:|
| shade ↔ cso | 0.685 | 0.200 | 0.115 | **0.385（最低）** |
| pso ↔ cso | 0.811 | 0.063 | 0.126 | 0.437 |
| de ↔ cso | 0.756 | 0.067 | 0.178 | 0.433 |
| lshade ↔ cso | 0.678 | 0.278 | 0.044 | 0.456 |
| de ↔ ga | 0.607 | **0.393** | **0.000** | 0.500（单边） |
| lshade ↔ ga | 0.378 | **0.622** | 0.000 | 0.500（单边） |

最低 DCM 的三对都涉及 cso（cso 的优势区在早期，与 shade/lshade 的中后期区错开）；DE–GA 呈完全单边（DE 支配 GA，见 04 报告）；MA-BBOB 的 DCM 矩阵同型（`static_pairwise_dcm.parquet`）。

## 2. Stage 1 规则筛选结果（Pareto/rule-based，无加权总分）

| 候选 | 判定 | 依据 |
|---|---|---|
| shade | **KEEP** | 独占胜 6.7–14.6%、跨 3 个 BBOB family + MA、marginal VBS +0.86、与 cso/lshade 均 DCM<0.45 |
| lshade | **KEEP（DOMINANT-CONTROL 观察）** | 独占胜 12.6–15.6%、跨 4 families + MA、marginal VBS +1.21；standalone 最强，支配风险持续监测（Stage 2 实测支配率 0.65，未失控） |
| cso | **KEEP** | 早期独占区（FE=2000：win 28.2%、mass 0.39）、跨 2 个 BBOB family + MA、与 shade DCM 最低 0.385 |
| pso | REJECT-WEAK | excl-win ≤1.8%、mass ≤0.03、marginal VBS +0.03、仅 1–2 family |
| de | REJECT-WEAK | excl-win ≤0.5%、mass ≤0.034、仅 1 family |
| lbestpso | REJECT-WEAK | excl-win ≤0.5%、mass ≈0 |
| ga | REJECT-WEAK | 全 FE/suite excl-win = 0、mass = 0 |

跨 family 核验：FE=6000/10000 的 exclusive states 分布——shade 15–29 states/3 families（BBOB）；lshade 27–44/4；cso 11/2（6000）；pso、de 仅 1–3 states/1 family；MA 侧 pso/de 为 0。

**进入 Stage 2 的候选 = {shade, lshade, cso}（恰 3 个）**，同时作为预注册组合 P4（自动最优 3 元子集）；P5（4 元子集）因 KEEP 数不足 4 而不存在。预注册组合 P1 {DE,GA,PSO}、P2 {SHADE,GA,PSO}、P3 {L-SHADE,GA,lbestPSO} 均含 REJECT-WEAK 成员（GA、lbestPSO、PSO、DE），**无法由 KEEP 集构造**，其成员淘汰证据已如实记录（DE-GA 判定见 04 报告）。
