# 14d · Behavior 特征审计（Task 13E）

- 日期：2026-08-29。对象：replay 产出的正式 Behavior 特征（`results/analysis_v5/task13/behavior_global_features.parquet`，1890 行）。机器可读版：`13e_behavior_feature_audit.json`。
- 契约来源：`behavior/features.py`（逐列读取实现，非旧文档）。B$^{global}$ = `SELECTOR_BEHAVIOR_FEATURE_COLUMNS` 28 列；提取器另产出 3 列 maturity（`bf_search_maturity`、`bf_search_maturity_linear`、`bf_explore_exploit_ratio`）与 3 列 diagnostic，**均不进入本轮模型**（与 Task 9–11 的正式 feature contract 一致：无成熟度行为 28 列）。

## 1. 数量、dtype、缺失与退化

| 检查项 | 结果 |
|---|---|
| Behavior 特征数量 | 28（selector 契约精确匹配） |
| dtype | 全部 float64 |
| NaN（28 列内） | **0**（1890 行全部完整；全量提取中的 111 个 NaN 位于 3 个 diagnostic 列，未进入模型） |
| Inf | 0 |
| constant / near-constant | 无（最大同值占比 < 99%） |
| per-solver 可用性 | shade 630/630、lshade 630/630、cso 630/630 全特征可用 |
| per-FE 可用性 | FE=2000/4000/6000 各 630 行全特征可用 |

## 2. 泄漏检查（结论：全部特征仅使用 ≤ t 的轨迹信息）

- 提取链：NativeUpdateWindowRecorder 只在**完整 update 后**观察；w02/w05/w10 窗口 anchor 为不晚于目标的最近完整 update；`effective_window_fe_*` 元数据证实各窗口实际终点 ≤ checkpoint FE；
- 特征定义（improvement rate/frequency、diversity、centroid/elite shift、covariance、stagnation、distance decay、fitness distribution 变化）全部为窗口内历史统计或当前快照统计；
- 不含 $t{+}1000$ 信息、不含 action outcome、不含 benchmark reference gap 类字段；
- 唯一的显式时间类特征：**`bf_fe_ratio`**（FE/FE_total，决策时刻已知量，非泄漏，但按工作单 §22 列为 time-like，供 13P 敏感性剔除检验）；3 个 maturity 列不在 28 列内，不参与。

## 3. 量纲与求解器可得性说明

- 28 列量纲差异极大（率类 0–100+，shaped 距离类 0–1，covariance ratio 可达 40+）——正式 carrier 内置 StandardScaler 处理，不做手工重标定；
- L-SHADE 的窗口端点分位数特征使用收缩种群扩展的网格估计器（见 14c §3），等长端点路径逐位不变（回归验证 diff=0.0）；
- 无缺失 ⇒ 正式 carrier 的 WeightedMedianImputer 在本轮为恒等变换（保留以严格复用正式 pipeline）。

## 4. 结论

28 列无成熟度 Behavior 特征在全部 1890 states 上完整、无泄漏、契约一致，**允许构造 behavior-action dataset 并进入 OOF 训练评价**。
