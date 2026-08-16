# 服务器会话启动指南（2026-08-16）

> 本文件用于在服务器上开启新对话时，让 AI 助手快速了解当前项目状态和下一步任务。

---

## 1. 你在哪里

这是 **Decision-before-Feature** 项目：在连续黑盒优化中，研究"是否值得在运行中执行一次 Landscape Query 来改善后续算法选择"。

项目根目录：`/path/to/Decision-before-Feature`

---

## 2. 当前进展（2026-08-16）

### 方案 A 已实现

我们刚完成了方案 A（等总 FE 性能功效）的核心代码实现。主要改动：

1. **主标签从时间加权 Utility 改为 G_FE**
   - `G_FE = log((E_skip + epsilon_p) / (E_query + epsilon_p))`
   - runtime 不再进入主标签，仅作独立资源维度
   - 旧 `u_query_joint_lamT_*` 保留为敏感性分析

2. **新增模块**
   - `utility_labels/efficacy.py` — G_FE 计算
   - `decision/practical_delta.py` — `delta_practical` 阈值估计
   - `decision/pilot_coverage.py` — Pilot 覆盖检查（`CoverageMass` / `CoverageRun`）
   - `decision/opportunity_ablation.py` — 四种 opportunity 频率消融
   - `decision/schedule_threshold.py` — schedule × threshold 联合冻结
   - `decision/conformal.py` — conformal 预测区间
   - `decision/skip_defer_query.py` — Skip–Defer–Query 三向决策
   - `configs/phase1_pilot_bbob.yaml` — Pilot 配置（F1/F3/F15/F24, 10D, 3 seeds）

3. **修改的文件**
   - `utility_labels/generation.py` — 接入 G_FE 计算
   - `utility_labels/fields.py` — 新增 `EFFICACY_COLUMNS`
   - `decision/model_protocol.py` — 主选择指标改为 `mean_g_fe`
   - `PROJECT_HANDOFF.md` / `DEVELOPMENT_DECISIONS.md` / `README.md` — 同步更新

### 当前采样协议

正式协议仍为 `phase1_dynamic_budget_event_v1`：
- 监测网格 0.20–0.60, step 0.01
- 12 个预算里程碑
- 行为事件触发（improvement_resume / stagnation_onset / rank_change / elite_migration / diversity_recovery）
- 每条 run 12–18 个状态
- 每条 run 至多一次 Query（run-level first-trigger）

方案 A 要求旧 [0.20, 0.60] 范围必须在新 G_FE 标签下重新验证，Pilot 配置已准备好但尚未运行。

---

## 3. 下一步任务（按优先级）

### Step 1: 跑 Pilot 覆盖检查（GO/NO-GO 前置）

```bash
uv run phase1-plan-shards --config configs/phase1_pilot_bbob.yaml
uv run phase1-collect-batch --config configs/phase1_pilot_bbob.yaml
uv run phase1-check-trajectory-shards --config configs/phase1_pilot_bbob.yaml
uv run optimizer-state-check
uv run behavior-permutation-check
```

Pilot 完成后：
- 对 0.10–0.70 扩展网格上的候选点生成 G_FE 标签
- 用 `decision/pilot_coverage.py` 计算 `CoverageMass` 和 `CoverageRun`
- 判断 [0.20, 0.60] 是否满足 `CoverageMass >= 0.95` 且 `CoverageRun >= 0.90`

### Step 2: 验证 G_FE 标签计算链路

```bash
uv run query-sample-batch --config configs/phase1_pilot_bbob.yaml
uv run query-extract-cheap --config configs/phase1_pilot_bbob.yaml
uv run selection-reference-evaluate-actions --query-id descriptor_cheap_invariant
uv run selection-reference-build --query-id descriptor_cheap_invariant
```

检查：
- `g_fe` 符号正确（Query 更好时 > 0）
- `epsilon_p` 随问题尺度协变
- `g_fe_bounded` 落在 [-1, 1]

### Step 3: 跑 opportunity frequency 消融

在 Pilot 数据上比较 milestones-only / milestones+events / equal-count / dense。

### Step 4: 验证 trajectory reservoir

确认 zero-FE trajectory descriptor 在采集阶段已同步保留。

### Step 5: 验证 Skip-Defer-Query

在 Pilot OOF 预测上测试三向决策和 conformal 区间。

---

## 4. 必读文档（按顺序）

1. `AGENTS.md` — 项目最高优先级约束
2. `README.md` — 项目概览和命令列表
3. `PROJECT_HANDOFF.md` — 交接记录和 blocker 清单
4. `DEVELOPMENT_DECISIONS.md` — 开发裁决
5. `Decision-before-Feature_Efficacy-first与增量价值理论实验方案.md` — 方案 A/B/Efficacy-first 理论方案
6. `Decision-before-Feature_方案A_采样频率修订协议.md` — 采样频率修订协议

---

## 5. 关键约束

- **不得启动正式 72 shards**，直到 Pilot 覆盖检查通过
- **不得用 BBOB-validation / CEC 结果**选择采样范围、threshold 或 delta_practical
- **不得从其他目录**寻找旧代码、数据或结果
- **所有判断只使用当前仓库**

---

## 6. 当前仍然未闭合的 blocker

参见 `PROJECT_HANDOFF.md` 第 10 节的 19 个 blocker。最关键的几个：

1. offline decision-state-to-terminal runner 尚未核对
2. BBOB-validation 的完整 instance-aware online endpoint 未实现
3. ERT suite-level consumer 未闭合
4. `dimension_stratified_T0` 未实现
5. CEC2022 / 工程集合的正式配置未冻结

---

## 7. 一句话总结

> 方案 A 的代码已经写好，现在需要在服务器上跑 Pilot 验证采样范围和 G_FE 标签链路，然后再决定是否启动正式全量采集。
