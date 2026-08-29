# 06 · Dynamic Cross-Action Protocol 与 07 · Dynamic Noise Calibration（Task 12F–G/J）

- 日期：2026-08-30
- 实现与采集：`analysis_v5/task12_stage2.py`；产物 `results/portfolio_screening/task12/stage2/shards/` 与 `results/analysis_v5/task12/`。

## 1. Stage 2 问题集（预注册、outcome-blind，Task 12G）

- **BBOB**：5 个 broad family 各取编号最小的 2 个 train 函数 → {f1,f2}, {f6,f7}, {f10,f11}, {f15,f16}, {f20,f21} × instances 1–3 = 30 问题。
- **MA-BBOB**：24 个 selected definitions 排序后隔一取一（stride 2）→ 12 definitions × instance 1。
- seeds 1–5。分层规则在看任何候选 outcome 之前固定。

## 2. 状态构造与全 fork（Task 12H/I）

- 每条 (problem, seed, current-solver) 轨迹：natural run 0→10000，在 FE∈{2000,4000,6000} 克隆 checkpoint（candidate-current states）。
- 状态数：42 问题 × 5 seeds × 3 candidates × 3 checkpoints = **1,890**。
- 每 checkpoint 分叉全部动作：continue + switch 到其余 2 个 KEEP 候选（每动作 1000 FE 主 horizon）；**CMA-ES add-back 分支（+1 动作）同样在同一 checkpoint 上执行但写入隔离表**，选择分析对其不可见，直至 P_balanced 预先固定（12 报告）。
- 分支总数：base 5,670 + repetition 1,890 = 7,560；add-back 2,302（含重复）。FE：分支 6.78M + add-back 2.30M。

## 3. Dynamic noise calibration（Task 12J）

10% state-action 对（确定性 SeedSequence 抽样，与 outcome 无关）× R=3：

| suite | action | δ50（函数平衡） | δ95（函数平衡） | pooled δ95 |
|---|---|---:|---:|---:|
| bbob | continue | 0 | 0.0874 | 0.0857 |
| bbob | lshade | 0 | 0.1012 | 0.0964 |
| bbob | shade | 0 | 0.0754 | 0.0683 |
| bbob | cso | 0 | 0.0863 | 0.0562 |
| mabbob | continue | 0 | 0.0852 | 0.0866 |
| mabbob | lshade | 0 | 0.0580 | 0.0658 |
| mabbob | shade | 0 | 0.0670 | 0.0671 |
| mabbob | cso | 0 | 0.1253 | 0.1348 |

- 主 practical 判定使用 **suite-specific δ95**（BBOB ≈ 0.0876，MA ≈ 0.0839，为各动作的平均），pooled 作敏感性。
- 各算法 δ95 同量级（0.058–0.125）：新候选没有表现出系统性更高/更低的随机性；δ50=0 的重尾解释同 Task 11（大量分支在 1000 FE 内 best 不动）。
- 未沿用 Task 11 的 0.098：实测 MA 的 cso（0.125）与 bbob 的 lshade（0.101）等已略超/略低于该值，逐 suite 标定是必要的。
