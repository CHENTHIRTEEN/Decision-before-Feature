# 11 · Portfolio Pareto Selection（Task 12O）

- 日期：2026-08-30
- 本报告与 12 报告合并撰写于 `11_portfolio_pareto_selection_and_12_cmaes_addback.md`（§1 = Portfolio Pareto Selection，§2 = CMAES Add-Back Control）。

## 1. 子集指标（重列）

| suite | subset | SBS | Δ_portfolio | Δ_dynamic | max dominance | practical entropy |
|---|---|---|---:|---:|---:|---:|
| bbob | shade+lshade+cso | lshade | +0.209 | +0.114 | 0.650 | 1.254 |
| mabbob | shade+lshade+cso | lshade | +0.195 | +0.085 | 0.637 | 1.306 |

## 2. Pareto 判据逐条核对

1. 无明显弱 solver ✔（三者 exclusive-win 区见 05 报告）；
2. 无近全支配 ✔（max practical dominance 0.65/0.64，对照 Task 11 cmaes 的 0.92+）；
3. 多 solver 跨 family practical exclusive-win ✔；
4. VBS–SBS headroom 明显 ✔（+0.19~+0.21，CI 远离 0）；
5. Δ_dynamic 非零 ✔（+0.114 / +0.085；MA 贴噪声边界已标注）；
6. within-problem best action 变化 ✔（P(varies)=0.60/0.58，低于 tie-null → 结构性）；
7. DCM 无重复 ✔（全部 ≤0.37 且双向）；
8. 规模最小 ✔（3 个，KEEP 集即 3 个）。

## 3. 预先固定

$$P_{balanced}=\{\text{SHADE},\ \text{L-SHADE},\ \text{CSO}\}$$（在 12 报告的 add-back 控制执行前预先固定。）
