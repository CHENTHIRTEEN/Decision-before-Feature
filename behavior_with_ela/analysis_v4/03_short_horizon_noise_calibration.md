# 03 · Short-Horizon Noise Calibration（Task 11H）

- 日期：2026-08-30
- 方法：460 个确定性抽样 checkpoint（≈9.8%，与结果无关）× 3 动作 × R=3 replicate（正式 RNG fork 语义）。每个 state-action 的 gain 序列 $g_r=L_r(s,\text{continue})-L_r(s,a)$（同一 replicate 内配对），取 $|g_r-\mathrm{median}_r(g)|$，按函数平衡分位数得 $\delta_{h,50}/\delta_{h,95}$。无新增采样方案，全部复用正式 repetition 语义。
- 产物：`analysis_v4/task11/short_horizon_noise_deltas.parquet`；明细 `results/analysis_v4/task11/short_horizon_noise_deviations.parquet`。

## 1. 分 horizon 噪声 delta

| horizon | suite | $\delta_{50}$（函数平衡 / pooled） | $\delta_{95}$（函数平衡 / pooled） |
|---|---|---|---|
| 500 | BBOB | 0 / 0 | 0.0746 / 0.0614 |
| 500 | MA-BBOB | 0 / 0 | 0.0708 / 0.0859 |
| 1000 | BBOB | 0 / 0 | 0.0890 / 0.0652 |
| 1000 | MA-BBOB | 0 / 0 | 0.1151 / 0.1263 |
| terminal | BBOB | 0 / 0 | 0.2923 / 0.1580 |
| terminal | MA-BBOB | 0 / 0 | 0.2426 / 0.2485 |

（pooled = 本轮两 suite 混合的函数平衡均值，作为跨 suite 判定用 $\delta_{500,95}\approx0.073$、$\delta_{1000,95}\approx0.098$、$\delta_{T,95}\approx0.267$。）

## 2. 结论与边界

1. **禁止沿用 terminal delta 的要求得到数据支持**：短 horizon 的噪声远小于 terminal（δ95 约 0.07–0.13 vs 0.24–0.29）。若误用旧 $\delta_{95}\approx1.464$（其来自不同协议与 prefix/动作全域），会把几乎全部真实差异判为等价。
2. **$\delta_{50}=0$ 的含义**：一半以上的 state-action 在 3 个 replicate 内 gain 完全一致（大量分支在短 horizon 内 best 无任何改善，重复确定性地复现同一损失）。中位偏差为 0 不代表噪声不存在，而是分布重尾——95 分位才是有效判据。因此本报告的 practical 判定一律使用 $\delta_{h,95}$，并在 04/05 报告同时给出 raw margin。
3. **与旧 terminal delta 的关系**：本轮对 terminal 的复检（0.24–0.29）显著低于旧协议的 1.464，原因包括域不同（全部为 mature post-handoff cmaes 状态，无 pso/shade-prefix 大 gain 方差）、5 seeds、以及不同重复结构。本轮审计内的 practical 判定一律使用本轮自估 delta；不与旧值混用。
4. 样本充分性：460 states × 2 switch 动作 × 3 replicate 覆盖全部 route × FE × family/definition；BBOB 与 MA 的 δ95 相互接近（0.061–0.089 vs 0.071–0.126），未见需要标记 "short-horizon practical delta unstable" 的证据。
