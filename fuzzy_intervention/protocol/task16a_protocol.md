# Task16A 预先指定实验协议

## 研究定位

本实验检验 Continue、轻量 Perturb 和 Switch 是否构成三个具有实际性能互补性的动作，并检验紧凑行为探针与预算阶段是否关联于粗粒度干预区域。该模块不检验模糊控制器效果，也不支持“行为可精确预测下一求解器价值”的表述。

Task15A 的 I3 结论保持不变。本模块与主 Decision 数据、标签和 Selector 结果严格隔离。

## 实验域

- dimension：10；
- total budget：10000 FE；
- boundary handling：reflect；
- algorithms：SHADE、L-SHADE、CSO，仅用于本独立探索性模块；
- development seeds：1–5；
- problems：Task14A 的 BBOB 30 个问题与 selected MA-BBOB 12 个问题；
- natural source checkpoints：2000、4000、6000、8000 FE；
- action horizon：1000 FE，Perturb 评价计入该预算。

每个 problem × current algorithm × seed × checkpoint 都保留，不按探针值筛选，预计 2520 个自然状态。

## 动作

每个状态运行五个具体动作：`continue`、`perturb_targeted`、`perturb_random` 和两个不同目标算法的 `switch_*`。两个 Switch 结果分别进入实际非支配集合；它们的较小 loss 只派生为逐状态最佳已观测 Switch 参考。

## 轻量扰动

- subset fraction q = 0.25；
- k = max(1, ceil(0.25 × (N - 1)))；
- 当前最优个体不进入候选集合；
- Targeted 排序依次为个体停滞年龄降序、近期生产性进展升序、稳定 agent_id；
- Random 从同一候选集合均匀抽取相同 k；
- unit-cube Gaussian perturbation sigma = 0.05；
- 使用 reflect 映射回 [0, 1]；
- 无条件替换所选个体并重新评价；
- 保持当前算法及求解器级自适应状态；扰动评价不写入 SHADE/L-SHADE success history 或 archive；
- 两种 Perturb 使用同一 kernel 和相同状态修正规则。

CSO 的个体速度与位置一一对应。对被替换个体，将其速度置为零，这是维持位置—速度一致性的最小修正；未选个体速度和求解器级参数不变。SHADE/L-SHADE 不需要额外个体状态修正。L-SHADE 使用 checkpoint 的实际 N，不恢复初始种群规模。

## FE 口径

Perturb 先用 k FE 评价替换位置，再让当前算法继续 1000-k FE。Continue 与 Switch 均使用 1000 FE。优化器接口支持精确 FE 数，因此 `actual_action_FE` 必须等于 1000；原生 update 是否在动作终点完整结束另存元数据，不改变终点 best-so-far loss。

## 随机流

所有新增随机流只由显式整数 seed、suite code、function、instance、dimension、checkpoint、current algorithm、action、repetition 和 event 交给 `numpy.random.SeedSequence`。不同动作、Random subset 和 perturb vector 使用不同 stream code；不使用 Python `hash()`，不使用摘要函数。

## 结果指标

正式 loss 为终点 best-so-far gap 的截断 log10：

`log10(clip(max(best_fitness - reference_value, 0), 1e-12, 1e20))`。

reference value 只用于离线结果评价，不进入 probe、Targeted subset 或任何在线决策量。

## 噪声与实际性能差

在读取首个动作结果前，以显式 SeedSequence 为每个 state × concrete action 独立选择 10% 组合。被选组合运行 3 次独立 repetition，其他组合运行 1 次。每个动作的 95% noise scale 由同一 state-action 的 repetition 差异估计。

主 pairwise threshold 为两动作 noise scale 的较大值；敏感性阈值为平方和开方。若 `L_a < L_b - delta_ab`，记 a 对 b 具有超过噪声尺度的较低 loss。五个具体动作分别进入实际非支配集合。

## 预先指定判据

Perturb structural floor：至少一个 suite 同时满足 Targeted Perturb 非支配率 ≥ 0.15、Targeted Perturb 优于 Continue 的比例 ≥ 0.10；另一个 suite 的两项均不得低于 0.05。

Perturb–Switch 双向互补性：pooled 域中，Targeted Perturb 优于两个 Switch 的比例 ≥ 0.08，且至少一个 Switch 优于 Targeted Perturb 的比例 ≥ 0.08。

Targeted 对 Random 的 paired gain 定义为 `L_random - L_targeted`。主区间以 cv_group/problem 为重采样单位，5000 draws，95% 区间；BBOB、MA-BBOB 分开，pooled 只作辅助。

## 停止边界

Task16A 完成 A/T/P 与 F1/F2/F3 结论后停止。本阶段不得实现 Type-1、Interval Type-2、membership tuning、规则搜索、闭环控制、RL、动作分类器、ELA、ProgressForecast、新求解器、seeds 6–10、CEC 或工程问题。

