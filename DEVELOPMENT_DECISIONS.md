# Decision-before-Feature 开发前裁决

本文仅保留当前仍生效的项目级裁决。已废弃的历史推导、事故复盘长文和过时口径已删除。

## 1. 最高优先级

- `AGENTS.md` 是最高优先级活动规范。
- `PROJECT_HANDOFF.md` 仅作当前状态摘要。
- 若 `docs/` 与活动规范冲突，以 `AGENTS.md` 和本文件为准。

## 2. 当前生效裁决

- `g_fe_selected_path` 是主功效；`g_fe` 仅作最佳已观测动作诊断；`runtime` / `wall-clock` 仅作为独立资源与计时维度。
- `action_loss` 统一解释为严格等总 FE 预算下的 FE-indexed optimization loss。
- 任何科学标签不得由 wall-clock time 定义。
- 活动 Decision Model 只使用当前协议é¢åæå®的特征、标签与 threshold 口径。
- Selection Reference 的 action-loss 回归、Utility labels、Decision labels 与在线评估必须共享一致的数据契约。
- 所有 runtime / wall-clock 对比必须使用真实 replay，不得用 component runtime 合成完整路径时间。
- 所有正式实验前必须完成数据契约、字段一致性、replay plan、artifact 路由与 failure materialization 的核对。
- 旧 `u_query_joint_lamT_*`、`performance_gain_norm`、`time_cost_norm` 和其他时间主标签口径仅可作为诊断，不得作为活动目标。
- 正式 optimizer seeds é¢åæå®为 1–5（2026-08-21 由 30 下调，BBOB-train / BBOB-validation / CEC2017 一致执行）。该下调属开发期采样设计变更、非精度驱动；主 CI 的 seed 层配对 bootstrap 在每个 static problem 内仅 5 个重抽单元，不得声称功效充分。历史 30-seed 产物只能作超集来源，进入正式链路前必须按 seeds 1–5 过滤或按新配置重新生成。
- 正式数据集配置整合为单一 train/validation 入口（2026-08-21）：combined 配置以 `suites` 段内嵌各 suite，公共规格（维度/seeds/FE/边界/endpoint 常数）由配置结构强制跨 suite 一致，suite 段只允许覆盖 suite/存储 split/functions/instances/family protocol/manifest 路径/output；磁盘存储 split 名（bbob_train/mabbob_formal/bbob_validation/mabbob_validation）是下游é¢åæå®读取契约，不随逻辑整合改名。`results/` 可整体清空重采；MA-BBOB manifest 为确定性协议产物，由 generate/select CLI 重建。
- BBOB、CEC 与 MA-BBOB 同为正式实验基准函数，不得区别对待（2026-08-21 裁决）：正式采集规格统一为 seeds 1–5、`FE_total = 1000 × dimension`、population 40、`boundary_handling = reflect`、同一 endpoint/timeout/floor-cap 常数、同一 shard 校验器与 intake 标准；clip 仅限显式声明 `boundary_handling: clip` 的敏感性分析。suite 之间只允许两类设计差异：数据角色（train/validation/外部集合归属）与问题结构（维度集合：BBOB/MA-BBOB 10/20/40D、CEC 10/30/50D；instances：BBOB 1/2/3、MA-BBOB 每 definition 自带 per-component instance 向量故外层只用 instance 1、CEC instance 1）。BBOB/CEC 的 `boundary_handling` 由硬编码改为配置驱动，`make_bbob_problem` 默认 clip、`make_cec_problem` 默认 reflect，正式配置必须显式声明；`phase1-check-config` 要求 BBOB 配置显式声明边界处理、train/validation 成对校验强制双方 reflect。
- MA-BBOB 分两个互斥角色：`mabbob_formal`（24 定义，components ⊆ BBOB-train）仅作 Selector 训练增强；`mabbob_validation`（18 定义，candidate 201–218，components 与 dense 权重支持集 ⊆ 六个 validation 函数）仅作 evaluation-only 评价集，不得进入任何 fitting split。两侧采集规格统一为 10/20/40D、instance 1、seeds 1–5、reflect（2026-08-21 裁决统一 seeds，此前 formal 为 [1,2]）。BBOB-validation estimand 为两层 50/50 预指定组成（六原函数等权层 × 18 定义等权层），组成固定、不得事后调整，两层同源性必须在报告中声明。MA-BBOB 运行配置必须显式携带 `manifest_path` 与 `boundary_handling=reflect`；缺 manifest 的 mabbob 采集一律无效。
- MA-BBOB 任一池的 dense 权重支持集必须严格限制在该池的成分宇宙内并以 entry 声明的 components 锚定，不得使用跨全集固定向量（2026-08-21 修正 train 池 `DENSE_PROFILES` 缺陷后生效）；泄漏检查按权重支持集而非仅 components 元数据执行。
- transferred-CMA-ES 交接状态必须做正则化协方差重构（2026-08-22 裁决）：population 固定 40 与维度无关，`dimension >= population_size - 1`（正式 d40）时转移种群中心化样本协方差结构性秩亏，旧实现（裸样本协方差 + `1e-12·I` + 绝对特征值地板 `1e-30`）使 `C^{-1/2}` 放大零空间方向并使 σ 更新 `math.exp` 溢出（生产数据 d40 transferred-cmaes 失败率 12.4%，全部 `OverflowError`）。é¢åæå®修复参数：样本协方差特征值相对下限 `max(λ)·1e-10`（按 WS-CMA-ES warm-start 协方差正则与 pycma 条件数防护口径）；`σ = sqrt(median floored eigenvalue)` 并 clip 到 `[1e-6, 0.3]·mean span`；σ 更新指数 clip `±50`（对健康运行为恒等操作）。原生 CMA-ES 路径逐代 bit 级不变为该修复的强制验收条件。同日追加（σ 发散防护）：d40 下每代 rank-μ 更新持续注入秩亏成分，仅靠交接期正则不足以阻止 σ 指数发散（可冲至 1e71 甚至溢出为 inf/nan 种群），故 `_update_cmaes_eigensystem` 与初始特征分解的特征值地板统一为相对下限 `max(λ)·1e-10`，且 σ 更新后施加绝对上限 `3×mean span`（远高于健康 native σ，实测原生 max σ ≤ 1.2×初始 0.3·span、原生 max cond(C) ≈ 1.4e3，两防护对原生路径均为逐代恒等操作，bit 级不变重新验收通过）；σ 触顶后 CMA-ES 自适应可正常回落恢复。
- 对 `Problem` 做计时/追踪包装时必须透传 `boundary_handling`（2026-08-22 裁决）：`Problem` dataclass 默认 `clip`，包装对象丢失该属性会使 action-loss 生成与在线策略评估的优化器推进静默回退 clip，造成同一 run 内 prefix（reflect 轨迹）与 continuation（clip）边界语义不一致。`selection_reference.action_losses._TimedObjective.wrapped_problem`、`decision.online_controller_evaluate._tracked_problem` 必须透传该属性；一致性检查器构造 problem 时必须从 config 显式透传。修复前生成的全部 selection_reference action-loss 产物按 clip continuation 失效，必须以 reflect 全量重生成。

## 3. 已废弃内容

以下内容已从活动裁决中删除，不再保留为当前工作依据：

- 早期重建式 continuation 口径
- 以时间加权 Utility 作为主标签的旧方案
- 逐状态 threshold 的旧模型选择方式
- 旧版 validation 参与选模或阈值拟合的口径
- 用合成 timings 替代实测 replay 的做法
- 任何将 runtime 当作主科学标签组成项的旧表述
