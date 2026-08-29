# 05 · Stage 1 Candidate Screening 汇总

- 日期：2026-08-30
- 详细证据：02（strength）、03（DCM）、04（H-DEGA）。本报告只汇总裁决。

## Stage 1 裁决表

| 候选 | 裁决 | 一句话依据 |
|---|---|---|
| shade | **KEEP** | 中后期独占区（6.7–14.6%），跨 3 BBOB families + MA，marginal VBS +0.86 |
| lshade | **KEEP（DOMINANT-CONTROL 观察）** | 最强 standalone（fb −5.68 @terminal），marginal VBS +1.21，跨 4 families；支配率 15.6% 需在 Stage 2 复测 |
| cso | **KEEP** | 唯一的早期阶段专家（FE=2000：win 28.2%、mass 0.39），跨 2 families；与 shade 的 DCM 全矩阵最低（0.385） |
| pso | REJECT-WEAK | excl-win ≤1.8%、mass ≤0.03、1–2 families |
| de | REJECT-WEAK | excl-win ≤0.5%、mass ≤0.034、1 family；且与 GA 构成单边支配对 |
| lbestpso | REJECT-WEAK | excl-win ≤0.5%、mass ≈0、全域无独立优势区 |
| ga | REJECT-WEAK | 全 FE/suite excl-win = 0；H-DEGA 判定 NOT REPLICATED |

- KEEP 集大小 = 3 → **Stage 2 只使用 {shade, lshade, cso}**（工作单 §17.1：禁止为凑 4 个保留弱算法）。
- 预注册组合处置：P1/P2/P3 因成员被 REJECT-WEAK 而不可构造（证据如上，未删除）；P4 = {shade, lshade, cso}（Stage 1 自动最优 3 元子集，恰为全集）；P5（4 元）不存在。
- DOMINANT-CONTROL 复测结论：lshade 在 Stage 2 dynamic 状态上的 practical 支配率为 0.65（BBOB）/ 0.64（MA）——显著但非 Task 11 式近全支配（见 08 报告）。
