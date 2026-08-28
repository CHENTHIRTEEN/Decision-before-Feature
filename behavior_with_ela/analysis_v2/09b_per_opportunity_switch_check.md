# Task 9 前置：per-opportunity 切换规则下的 CEC 三函数实测（协议变更）

- 日期：2026-08-29
- 协议变更（用户决策）：放弃"每 run 至多切换一次"的 first-trigger 语义，改为**每个决策机会只要最优候选预测增益超过阈值即切换**（无切换次数上限、无滞回）。`online.py::run_one_switch_policy` 新增 `switch_rule ∈ {first_trigger, per_opportunity}`，per_opportunity 支持 run 内多次种群转移；FE 账目与全局原生更新计数采用与 `repeated_das.py` 相同的成熟模式（转移事件流按切换序号区分）。
- 设置：与 09 报告相同（F1/F10/F29 × 10 seeds × 3 prefix，在线真实执行，每策略 90 runs）；静态 continue 参照复用。

## 结果（函数平衡；逐函数见 `multiswitch_summary.csv`）

| 策略 | 切换规则 | mean log10 gap | 相对 continue 增益 | switch rate | 平均切换次数 | 最多切换 |
|---|---|---:|---:|---:|---:|---:|
| v2 regression | first_trigger | −1.6061 | +1.868 | 0.667 | 0.67 | 1 |
| **v2 regression** | **per_opportunity** | **−1.6061** | **+1.868** | 0.667 | **0.67** | **1** |
| 三分类 classifier | first_trigger | −1.1581 | +1.420 | 0.700 | 0.70 | 1 |
| 三分类 classifier | **per_opportunity** | **−1.5788** | **+1.841** | 0.700 | **1.20** | **5** |

全部 180 runs completed，FE 账目精确（`evaluation_count == FE_total`），零失败。

## 解读

1. **回归载体对切换规则不敏感**：v2 在 per_opportunity 下从不二次切换（切换后预测增益始终低于阈值），策略行为与 first_trigger 完全相同——其连续型优势估计在切换后自然收敛于"继续"是更安全的。这是回归载体的一种稳健性。
2. **协议变更主要惠及三分类载体**：分类器主动多次切换（最多 5 次），mean gap 从 −1.158 改善到 −1.579，几乎追平 v2（−1.606）。新规则下 v2 仍在三个函数上全部领先（F1 −8.62 vs −8.55；F10 0.5041 vs 0.5042；F29 3.2996 vs 3.3052），但差距明显缩小。
3. **工程要点（已解决）**：转移后优化器的评估计数与原生更新序号都从头开始，与记录器的单调性约束冲突。one-switch 模式（转移后停止监控）不会暴露该问题；multi 模式改用与 `repeated_das.py` 一致的全局计数与绝对 FE 记账后解决。
4. **需要注意的分布偏移**：首次切换之后，行为特征基于跨算法混合的原生更新历史计算（与单算法离线训练分布不同）。多次切换的后续决策是在该偏移下做出的；本测试中分类器的多次切换整体有益（+0.42），但这点是后续 Repeated DAS 研究必须显式分析的对象。
5. first_trigger 语义保留为默认值，历史结果可复现（本次 FE 记账口径变化使 first_trigger 数值与 09 报告存在 ±0.07 以内的微小差异——旧口径用优化器自身计数，新口径用全局预算计数，后者与离线/repeated_das 一致）。

产物：`results/online/cec2017_quick3_multiswitch/`（两策略 outcomes/opportunities）、`analysis_v2/task9_quick_cec/multiswitch_summary.csv`。

## 补充（2026-08-29）：多次切换 run 的实际动作序列

机会行新增 `initial_prefix_algorithm` 归因列、结果行新增 `switch_chain` 后的精确重建：

- 多次切换 run 共 26 个（2 次 14、3 次 8、4 次 1、5 次 3），**最终算法全部为 cmaes**。
- 3 个 5 次切换 run 全部是 F10 上的 **cmaes↔shade 振荡**：
  - F10 seed=3（初始 pso）：pso --@2000--> cmaes --@2120--> shade --@2200--> cmaes --@2400--> shade --@2600--> cmaes
  - F10 seed=4（初始 pso）：pso --@2000--> cmaes --@2200--> shade --@2400--> cmaes --@2600--> shade --@2800--> cmaes
  - F29 seed=6（初始 pso）：pso --@2000--> cmaes --@2200--> shade --@2400--> cmaes --@2520--> shade --@2600--> cmaes
- F1 上的典型模式是开局 pso --@2000--> shade --@2200--> cmaes（10 个 pso-initial run 中 10/10 如此）；F29 上还出现末段切换（如 cmaes --@5720--> shade --@6000--> cmaes）。
- 振荡（shade↔cmaes 在相邻机会间来回）正是无滞回、无最小停留时间的逐点触发策略的预期病理，也是 Repeated DAS 协议引入 dwell 时间与滞回边际的动机；v2 回归载体在相同规则下从不二次切换，无振荡。
