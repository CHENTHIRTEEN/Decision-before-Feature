# 09 · Dynamic Pairwise Complementarity Matrix（Task 12N）

- 日期：2026-08-30
- 产物：`analysis_v5/task12/dynamic_pairwise_dcm.parquet`（natural 与 dynamic 两套矩阵并列）。

## 1. Dynamic DCM（1000-FE cross-action，practical δ95 suite 口径）

| suite | pair | P(tie) | P(a≻b) | P(b≻a) | DCM |
|---|---|---:|---:|---:|---:|
| bbob | shade ↔ lshade | 0.520 | 0.203 | 0.277 | **0.297** |
| bbob | shade ↔ cso | 0.436 | 0.398 | 0.167 | 0.333 |
| bbob | lshade ↔ cso | 0.435 | 0.431 | 0.134 | 0.366 |
| mabbob | shade ↔ lshade | 0.507 | 0.241 | 0.252 | **0.259** |
| mabbob | shade ↔ cso | 0.348 | 0.450 | 0.202 | 0.298 |
| mabbob | lshade ↔ cso | 0.354 | 0.454 | 0.193 | 0.307 |

## 2. natural_DCM vs dynamic_DCM（BBOB，FE=6000 natural 为参照）

| pair | natural DCM | dynamic DCM | 变化 |
|---|---:|---:|---|
| shade ↔ lshade | 0.441 | **0.297** | 互补性增强 |
| shade ↔ cso | 0.385 | **0.333** | 增强 |
| lshade ↔ cso | 0.456 | **0.366** | 增强 |

**所有三对在 dynamic cross-action 下都比 natural trajectory 更互补**，且全部呈双向结构（两个方向的 P 都 ≥0.13，无单边支配对）。原因：switch 动作继承了 incumbent 轨迹的进展（population transfer）， newcomer 的不同动力学接管后产生 natural run 中不存在的优势区——这正是动态组合价值区别于静态组合价值的形式。

## 3. 对照 Task 11 的教训

Task 11 的 {pso, shade, cmaes} mature 状态上不存在任何 DCM<0.5 的双向对（cmaes 单边支配）；本轮三对全部 DCM<0.37 且双向。**pairwise DCM 判据（工作单 §27：不显示大量重复）满足**。

BBOB 与 MA-BBOB 的矩阵同型（MA 的 DCM 略更低），无需分池处理（30 报告的分 suite 要求已满足：两 suite 分列）。
