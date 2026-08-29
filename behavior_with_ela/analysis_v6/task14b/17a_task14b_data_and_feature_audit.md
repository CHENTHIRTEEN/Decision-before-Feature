# 17a · 数据与特征审计（Task 14B-A）

- 日期：2026-08-29；零新增 objective FE。数据集：`post_handoff_behavior_action_dataset_task14b.parquet`（3780 行）。机器可读版：`17a_feature_audit.json`。

## 1. 一致性核查（全部通过，未触发 STOP）

| 检查 | 结果 |
|---|---|
| states = 3780；bg/bs 行 = 3780/3780 | ✅ 一一对应，无丢失/重复 |
| 6 方向 × 630、source FE 三档各 1260、bbob 2700 / ma 1080 | ✅ |
| action labels 与 Task 14A 完全一致 | ✅（同源 parquet 直取） |
| bg / bs 各 28 列 selector 契约 | ✅ |
| NaN / Inf（28+28 列） | 0 / 0 |
| constant 列 | 无 |
| bg↔bs 完全相同列 | 5 列（fe_ratio、diversity_mean_pairwise、covariance_spectral_concentration、elite_concentration、covariance_effective_rank_w05）——慢变特征在 1000-FE 段内几乎不变，属构造性重合 |
| bg↔bs 逐特征相关（mean/min/max） | 0.446 / −0.004 / 1.000 |

## 2. 泄漏检查

- **B_segment**：segment recorder 于 handoff 时刻重建（首观察 fe=segment_start），特征仅来自 [t, t+1000] 段内 update 历史（segment 相对窗口 20/50/100 FE，anchor 全部位于段内）——不含 handoff 前信息；
- **B_global**：全局 recorder 自 0 连续累积，窗口 fe_total=10000，特征仅使用 ≤ t+1000 的轨迹；
- 两类特征均不含 next-action outcome / t+1000 后信息 / benchmark reference；
- segment 特征**不泄漏 source algorithm**：其窗口完全位于 B 段内（与 bg 的高相关来自慢变特征的自然延续，不是 reset 泄漏）。

## 3. 数值稳定性预处理（如实声明）

post-handoff 段内提取的行为特征存在**除法长尾**：531 个 cell |v|>1e6（最大 6.7e13），以及若干 1e-12 以下的非规格化碎片。后者会使预先固定 carrier 的 StandardScaler 在 4-row within-route 折上产生 float32 溢出（本轮实际触发）。预处理：

- |v|>1e6 裁剪至 ±1e6（影响 ≤531/211,680 cells ≈0.25%）；
- |v|<1e-12 且非零 快照为 0（消除非规格化方差）。

自然域特征天然 ≤~1e5，该预处理不改变跨任务可比性的主体；受影响 cell 明细在审计 JSON 中。

## 4. 时间代理列

`bg_fe_ratio` 与 `bs_fe_ratio` 同为全局 FE 比例（二者恒等）；segment 窗口为 segment 相对口径，其余 27 列描述段内形状。
