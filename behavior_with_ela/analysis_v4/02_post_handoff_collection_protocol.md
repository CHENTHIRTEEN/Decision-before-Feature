# 02 · Mature Post-Handoff 采集协议与代码审计（Task 11B–E，protocol v1）

- 日期：2026-08-30
- 协议名：`mature_post_handoff_action_horizon_audit_v1`（覆盖并取代已暂停的旧 Task 11 skeleton 的 natural-state 假设；`analysis_v4/common.py` 仅保留通用常量，不再主导本轮设计）。
- 实现：`behavior_with_ela/post_handoff_audit.py`；分析：`analysis_v4/task11_audit.py`。

## 1. Routes（Task 11B，第一轮只做三条 core routes）

| route | 起点 | handoff | 之后 | 语义 |
|---|---|---|---|---|
| R0_native_cmaes | cmaes @ FE=0 | 无 | 原生续跑 | native control（source=current=cmaes，handoff_performed=False，segment_start=0） |
| R1_pso_to_cmaes | pso @ FE=0 | FE=2000 population transfer | cmaes | 与 Task 9 已学习的 initial routing 一致 |
| R2_shade_to_cmaes | shade @ FE=0 | FE=2000 population transfer | cmaes | 同上 |

选择依据：与 learned initial routing（pso/shade→cmaes、cmaes→continue）一致；不使用任何 CEC 结果挑选；隔离 source-history 效应；控制首轮 FE 成本。CMAES→SHADE / CMAES→PSO 留待 11N 判定后（见 07/08 报告与总报告）。

## 2. Mature states（Task 11C）

checkpoint $FE\in\{3000,4000,5000,6000\}$（均落在 40-FE 原生更新边界）；R1/R2 的 dwell = FE−2000 = 1000/2000/3000/4000；R0 的 segment_start=0。每条 route 跑到 6000 FE 为止，分支成本单独记账。

## 3. Checkpoint 真实性（Task 11D）

每个 checkpoint 由真实优化器执行到达：`initialize_optimizer_state` → 40-FE 步进 `advance_optimizer_state` → `clone_optimizer_state`（deepcopy 含 RNG/covariance/evolution paths）捕获。transfer 使用正式 `initialize_transferred_optimizer_state`（NO_QUERY_TRANSFER_EVENT 语义）。未做任何 summary 级人工构造。

## 4. Global / Segment Behavior（Task 11E）

- Global：独立 `NativeUpdateWindowRecorder` 从 FE=0 累积；checkpoint 处 `build()` 得 window_statistics + native_update_history（与正式 recorder 相同的截断窗口语义），用正式 `extract_behavior_rows` 提取 → `bg_*`（28 列）。
- Segment：第二条独立 recorder，R1/R2 在 handoff 后以转移后初始种群重启（segment 计数置 1），R0 从 0 开始（segment≡global）→ `bs_*`（28 列）。
- 未修改任何 Behavior 数学定义；短历史下不可定义的特征按原 extractor 行为输出缺失值，未用 global 历史填充。

## 5. 三动作 fork 与多 horizon（Task 11F/G）

每个 checkpoint 分叉 $\mathcal A=\{\text{continue cmaes}, \text{switch pso}, \text{switch shade}\}$：continue 用 `clone_optimizer_state`（完整原生状态续跑），switch 用正式 population transfer。**每条分支连续运行** checkpoint→+500→+1000→terminal，在同一分支上记录三个损失（500/1000 取首个 ≥ 标记的 40-FE 更新边界，即 +520/+1000；两分支同一规则，等 FE 可比；terminal 精确到 10000）。得 $L_{500},L_{1000},L_T$ 与 $G_h=L_h(s,\text{continue})-L_h(s,a)$，$a^\star_h=\arg\min_a L_h$（并列时按 continue 优先排序）。

## 6. Short-horizon repetition（Task 11H 采样部分）

按 `SeedSequence([2026083001, suite_code, function, instance, seed, route_code, FE])` 的确定性抽样（与 outcome/action winner/Behavior 无关；AGENTS 0.2 禁止 hash，故以 SeedSequence 实现"deterministic hash sampling"的意图）抽取约 10% checkpoint；被抽中者三个动作各执行 $R=3$ 个 replicate，复用正式 RNG fork 语义：continue replicate $r>0$ = `CONTINUATION_REPETITION_STREAM_OFFSET + NATIVE_STREAMS[cmaes]`，transfer replicate $r>0$ = `TRANSFER_REPETITION_EVENT_OFFSET + r`。

## 7. 规模与成本账本（Task 11 域）

| 项 | 值 |
|---|---:|
| 域 | BBOB train 18 函数 × 3 instances + selected MA-BBOB 24 definitions × 1 instance；seeds 1–5 |
| route-runs | 1,170（每 route-run 6,000 FE） |
| base route FE | 7,020,000 |
| branch FE（含 repetition） | 114,936,000 |
| **总新增 objective evaluations** | **≈121.96M FE** |
| states / checkpoints | 4,680（BBOB 3,240 + MA 1,440） |
| branches | 16,800（base 14,040 + repetition 2,760） |
| sampled checkpoints | 460（9.8%） |
| wall time | 总计 4,018 core-秒（8 workers ≈ 8.4 分钟墙钟） |
| peak RSS | ≈160 MB/worker |
| validation / CEC 参与 | 0（训练域之外零接触） |

## 8. 代码审计清单（工作单三十）核对结果

| # | 检查 | 结果 |
|---|---|---|
| 1 | same-action 分支保留完整 native state | PASS（`clone_optimizer_state` deepcopies 协方差/演化路径/step-size/RNG） |
| 2 | cross-action 仅用正式 population transfer | PASS（`initialize_transferred_optimizer_state`，事件号沿用正式语义） |
| 3 | multi-horizon 为同一分支连续运行 | PASS（单次 `advance_optimizer_state` + 更新回调记录标记，非三个随机分支） |
| 4 | 三 horizon 同一初始 checkpoint | PASS（`checkpoint_states[fe]` 单一深拷贝） |
| 5 | 全局 FE 严格单调 | PASS（循环按 40-FE 预算推进；checkpoint 断言精确命中） |
| 6 | 分支内 FE 不重复计费 | PASS（tracker 计数 == 剩余预算，否则 RuntimeError） |
| 7 | segment recorder 在 handoff 后重置 | PASS（新 recorder + 计数置 1 + 初始种群观测） |
| 8 | global recorder 保留完整 history | PASS（独立 recorder，未与 segment 共用） |
| 9 | repetition 复用正式 RNG fork | PASS（与 `action_dataset._evaluate_action` 相同 stream/event 常量） |
| 10 | validation/CEC 未参与生成 | PASS（仅 train config 的 bbob_train/mabbob_train suite） |
| 11 | Task 9/10 artifacts 未被覆盖 | PASS（只读引用；新产物全部在新目录） |

已知偏差记录：(a) 500/1000 FE 标记落在 40-FE 更新边界之间时记录于首个 ≥ 标记的边界（+500→+520），两分支同规则；(b) 抽样以 SeedSequence 实现（AGENTS 0.2 禁止 hash）。
