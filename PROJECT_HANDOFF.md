# Decision-before-Feature 项目交接记录

本文仅保留当前活动研究协议、仓库状态与后续工作指引。旧版本推导、已历史结果、过时口径和历史长篇说明已删除。

## 当前状态

- 主协议已切换为 selected-path 方案：`g_fe_selected_path` 作为主功效，`g_fe` 仅作最佳已观测动作诊断，`runtime` 仅作为独立资源/计时维度。
- `action_loss` 统一按严格等总 FE 预算下的 FE-indexed optimization loss 解释，不得用 wall-clock 定义科学标签。
- 2026-08-24 当前执行范围进一步收敛：只生成 query-adjusted 主路径的 `g_fe_selected_path`、Decision dataset、四候选训练摘要与在线评价输入。Behavior-only、pre-run AAS、matched-acquisition、五路径分解和 runtime 标签暂不生成；完整路径 replay 不再是当前训练前置条件，只保留为后续部署资源评价入口。
- 2026-08-25 旧的 `g_fe` 主标签结果已由 selected-path 主标签重算取代。新主标签字段为 `g_fe_selected_path`，主二元标签为 `g_fe_selected_path_gt_zero`；旧 `g_fe` / `g_fe_gt_zero` 仅保留为最佳已观测动作诊断。新四候选 nested CV-group OOF 仍选择 Logistic Regression，正式 `oof_g_fe_selected_path_first_trigger` threshold 为 `0.8778368490787811`；validation first-trigger 为 3/540 runs，validation 用于 model/threshold fit 的行数为 0。产物位于 `outputs/recompute_20260825_selected_path/decision/`，模型协议检查状态为 `ok`。
- 2026-08-25 selected-path online validation 已完成：`decision-validation-online-evaluate` 对 540 条 trajectory 逐机会复算训练后的 Logistic Regression，严格执行 selected-path OOF threshold 的 first-trigger；3 条 run 各 query 一次，query-full `dimension_aware_hybrid_selector` 参与触发后的路径，540 条完整 online policy paths 均完成。科学端点写入 `outputs/recompute_20260825_selected_path/validation_online_evaluation/validation_online_policy_outcomes.parquet`，其中不含 runtime / wall-clock 列；deployment metrics 单独保存，明确不进入标签或科学效用。该目录的 baseline comparison 与 `complete_path_timings` 缺口见 `baseline_and_timing_gap_report.md`。
- 2026-08-25 Search Maturity 形式消融已完成：下游 Selector 改为不使用三个成熟度字段的 28 列 B2+Motion Behavior；Decision 分别使用 `bf_search_maturity`、`bf_search_maturity_linear`、`bf_explore_exploit_ratio` 三种 29 列行为输入，并固定使用 `RandomForestRegressor(n_estimators=200,max_depth=8,max_features=sqrt)`。BBOB-train nested function-family OOF first-trigger mean `g_fe_selected_path` 排序为 Linear `-0.0026170294`、Explore/Exploit ratio `-0.0029384686`、Search Maturity `-0.0051970518`，当前优选 `bf_search_maturity_linear`。三组 validation online replay 均为 540 runs、0 query calls，clean 结果分别位于 `outputs/recompute_20260825_maturity_ablation/*/validation_online_evaluation_clean/`；该 validation 结果不用于形式反选。
- 2026-08-21：正式 optimizer seeds 由 30 降为 5（`configs/phase1_bbob_train.yaml`、`phase1_bbob_validation.yaml`、`phase1_cec2017_test.yaml` 与 README / 主规范 / Pipeline 协议的规模数字已按 5/30 同步）。已有 BBOB-train dimension 10 轨迹是 30-seed 超集，正式消费前需按 seeds 1–5 子集化或重新生成；`phase1-check-trajectory-shards` 以 5-seed 配置校验旧分片会报 seed coverage mismatch。
- 当前工作区存在未提交改动（`selection_reference/` 与本次协议修改）。

## 当前活动模块

- `trajectory/`：轨迹与最终性能底层数据。
- `behavior/`：行为特征与提取逻辑。
- `landscape_queries/`：query 采样与特征提取。
- `selection_reference/`：statewise action-loss 参考与 selector 训练。
- `utility_labels/`：utility / efficacy 标签生成与校验。
- `decision/`：模型协议、训练、评估与控制器。
- `experiments/cli/`：正式采集、配置、分片与 Selection Reference 检查入口。
- `benchmarks/`：BBOB / CEC benchmark 适配。

## 仍需完成的事项

- 完整路径 replay runner 仅在后续部署资源评价重新启用；当前主 Decision 标签链不等待该 consumer。
- 继续完成 BBOB-validation、CEC2017 / CEC2022 和工程集合的正式评估闭环。
- 继续清理历史文档中残留的联合效用 / 操作性增量命名、`performance_gain_norm`、`time_cost_norm` 与时间主标签口径。
- Stage-A、`utility_labels/*` 与 `decision/*` 已适配整合配置：按存储 split 读取 BBOB/MA-BBOB，并映射为逻辑 train/validation 角色。当前 `train_derived_sbs` 只从配置的训练角色聚合，不读取 validation。

## 2026-08-24 代码与旧产物清理

- 删除不在正式运行链中的 pilot、tiny check、边界试跑、早期三向决策辅助模块、旧外部预测器和重复分析脚本，共 25 个源码、配置或旧会话文件。
- 删除 `docs/archive/`、`results/archive/`、`tmp/`、旧 Selector 诊断结果、Python 缓存、包安装元数据和论文编译中间文件；未创建副本。
- 保留正式数据质量检查、核心一致性检查、活动 Selector 选择链及其当前结果。

## 2026-08-24 下游 Selector 正式入口整合

- `selection_reference/model.py` 现同时训练旧多输出 RF 与六个 one-vs-one RF，并按 10D/20D 使用多输出 RF、40D 使用 pairwise aggregation 的预先指定规则生成主选择结果；`dimension` 仅用于分层路由，不进入基础特征。
- `selection-reference-build` 统一承担正式训练、cross-family OOF、held-out 预测、全维度 pairwise 敏感性、旧 RF 基线和评价汇总，默认输出 `selection_reference.parquet`、`pairwise_aggregation_sensitivity.parquet`、`formal_multioutput_rf_baseline.parquet`、`selector_evaluation_summary.parquet` 与 `statewise_selector.joblib`。
- 已删除 `experiments/analysis/` 中三个重复 Selector 训练与评价脚本。旧 `statewise_selector.joblib` 保持兼容读取；旧诊断目录仅作历史比较。下一轮 Utility、Decision dataset 与在线评价输入必须从新 hybrid 主表重新生成，不得复用旧 RF 派生标签。

## 2026-08-22 transferred-CMA-ES 正则化与 boundary 透传两项修复

- **修复 1（d40 交接溢出）**：`_transferred_cmaes_state` 旧实现用转移种群裸中心化样本协方差重构 C。population 固定 40 与维度无关，d40 时 N−1=39 < d，协方差结构性秩亏；`C += 1e-12·I` 与绝对特征值地板 `1e-30` 不足以阻止 `C^{-1/2}` 放大零空间方向，σ 更新 `math.exp` 抛 `OverflowError`。修复前生产基线：mabbob_validation d40 transferred-cmaes 504/4074 失败（12.4%，全 OverflowError）；d10/d20 与 bbob_validation d20 均为 0。修复参数（é¢åæå®入 DECISIONS）：特征值相对下限 `max(λ)·1e-10`、`σ=sqrt(median floored eig)` clip `[1e-6,0.3]·mean span`、σ 更新指数 clip `±50`（参考 WS-CMA-ES warm-start 协方差正则与 pycma TolUpX 类 σ 爆炸防护）。
- **修复 2（action losses 用 clip，用户发现）**：`_TimedObjective.wrapped_problem()` 与 `_tracked_problem()` 重建 `Problem` 时漏传 `boundary_handling`，dataclass 默认 `clip`，导致 action-loss 生成与在线策略评估的全部 4 条动作路径（continue + 3 transfers，全算法全维度）在 clip 下推进，与 prefix 轨迹的 reflect 语义不一致——修复前全部 selection_reference action-loss 产物因此失效。已修复两处包装透传（验证 wrapped/tracked 均为 reflect、4000 FE 行为检查零边界堆积）；`selection_reference_consistency.py` 三处 `make_problem` 改为从 config 显式透传。
- **最小核对（修复 1）**：原生路径 7 run（mabbob c201/c210 d40、c201 d10、bbob f5 d20 × seeds，251–1001 代）新旧实现逐代 bit 级一致，指数 clip 从未触发；交接实验 d40 每版本 45 次（6 函数 × 2 seeds × {de,pso,shade} × ckpt 0.2，加 c201 ckpt 0.05）旧实现 11 次 OverflowError、新实现 0 次；d10/d20 共 36 次双实现均无失败（无回归）。
- **修复 3（transferred cmaes σ 发散，同日晚追加）**：d40 首轮重生成（20:46 版本）日志出现 `state.py:556 RuntimeWarning: overflow encountered in multiply`。根因：d40 下每一代 rank-μ 更新持续注入秩亏成分（40 样本 − 1 = 39 < 40），`_update_cmaes_eigensystem` 的绝对地板 `1e-30` 使 `C^{-1/2}` 每代放大零空间方向，σ 以指数 clip 上限（×e^50/代）发散（实测复现案例 σ 第 1 代即 >1e3、峰值 7.3e71；仅冲过 ~1e308 的 run 触发 overflow warning，溢出 inf 经 reflect 变 nan 种群空转，其 action_loss 记发散前 best，属乐观偏差）。修复：特征值地板统一为相对下限 `max(λ)·1e-10`（`_update_cmaes_eigensystem` 与初始特征分解），σ 更新后施加绝对上限 `3×mean span`（`CMAESState.sigma_upper_bound`，默认 inf，native 与 transferred 初始化设置）。实测原生路径 max σ ≤ 3.6、max cond(C) ≤ 1.4e3，两防护对原生为逐代恒等操作，7 条原生 run bit 级不变复验通过；36 次 d40 交接复测 0 失败、0 overflow warning、0 非有限 run、σ 全部 ≤ 上限，且触顶 run 的 σ 可自适应回落（复现案例终值 0.032）。**数据影响：20:46 版 d40 为"修复 1+2 无修复 3"产物，需再次重生成**；中间产物已在 2026-08-24 清理，d10/d20 未受影响（rank-μ 满秩，无发散）。
- **重生成完成与最终布局（2026-08-24 验收）**：三修复后全部 action losses 已重建，`results/selection_reference/descriptor_cheap_invariant/` 下四个 split 各一个全维度单文件（`query_adjusted_budget.parquet`，取代 per-dim 分文件布局；下游 `read_action_loss_data` 按显式路径读取不受影响，此后新增 action-loss 产物应沿用单文件布局）。验收数字（全部 `action_status=ok`、transferred-cmaes 失败 0、无非有限 loss、prefixes={de,pso,cmaes,shade}×seeds 1–5、`lhs_50d`、reflect）：bbob_train 49,377 states / 3,240 runs（= 18 函数 × 3 instances × 3 维度 × 4 prefix × 5 seeds，与采集量一致）；mabbob_formal 22,056 states / 1,440 runs；mabbob_validation 16,441 states（d10=5,629、d20=5,460、d40=5,352，与修复前 states 数一致；d40 子集为 2026-08-23 01:40 版本经逐位一致合并）；bbob_validation 16,520 states / 1,080 runs。修复前失效产物与中间版本已在 2026-08-24 清理。

## 2026-08-22 采集事故与恢复（manifest 静默回退）

- 事故：交接后 `results/` 被再次整体清空（含 `results/archive/` 与 4 个 MA-BBOB manifest；archive 为已撤回历史封存，**不可再生**，`docs/30_results` 中指向 `results/archive/withdrawn_20260811/...` 的引用自此悬空）。随后用户启动采集：bbob 两组不受影响（有效）；**mabbob_formal 72 分片在 manifest 缺失下静默回退 legacy 定义**（c001 reference=2.7807e-11 ≠ manifest 的 2.3101e-12，实为 `_random_weights` 构造），mabbob_validation 因 candidate 201+ 越出 legacy 范围而全 run 失败暴露。
- 修复：`benchmarks/factory.py` 硬化——config 声明 `manifest_path` 但文件缺失 → `FileNotFoundError`；mabbob 无任何 manifest → `ValueError`（落实 DECISIONS"缺 manifest 的 mabbob 采集一律无效"）；manifest 中无该 candidate → `ValueError`。三项均已负例测试。
- 恢复：无效的 mabbob_formal 72 分片与空的 mabbob_validation 目录已删除；4 个 manifest 已确定性重建；check-config 对通过；c1/c201 problem 构建核对为 manifest 定义。bbob_train（54）与 bbob_validation（18）分片经抽查有效保留。
- 教训入协议：**任何 results 清空后，必须先执行两步 manifest 重建（generate ×2 + select ×2）再启动采集**；factory 守卫保证此后 manifest 缺失会立即硬失败而非静默产出无效数据。

## 2026-08-21 重置与整合配置

- 用户裁决：清空 `results/` 从头开始；正式配置整合为 `configs/phase1_train.yaml` 与 `configs/phase1_validation.yaml`（bbob+mabbob per-suite 段 + 公共规格块）；被取代的分套件配置与旧 pilot 配置已删除；保留 `phase1_cec2017_test.yaml` 与 `prospective_suites.yaml`。
- 新增 `experiments/phase1_batch_common.expand_suite_configs/load_suite_configs`；`phase1-collect-batch`、`phase1-plan-shards`、`phase1-check-config`（按 suite 配对，mabbob 对检查 candidate 不相交）、`phase1-check-trajectory-shards`、`behavior-extract-batch`、`query-sample-batch` 均支持整合配置；全仓默认配置路径已改指向新配置名。
- results 已清空并确定性重建 4 个 manifest（train 42 池/24 选择、validation 23 池/18 选择）；冒烟验证：整合配置一条命令采集 bbob f001（60 runs）+ mabbob c001（20 runs）dim10 并通过分片校验。
- 当前 `selection_reference/action_losses` 与 `selection-reference-check` 已支持整合配置；`selection-reference-build` 使用显式产物路径。`decision/*` 与 `utility_labels/*` 仍需完成整合配置适配。

## 2026-08-21 基准统一裁决（BBOB/CEC/MA-BBOB 不区别对待）

- 用户裁决：BBOB 与 MA-BBOB（含 CEC）同为正式实验基准函数，不得区别对待；正式规格统一为 seeds 1–5、FE=1000×D、reflect、同 endpoint 常数与校验器，详见 `DEVELOPMENT_DECISIONS.md`。
- 实施中发现并修复实质不一致：`benchmarks/bbob.py` 此前硬编码 `boundary_handling="clip"`，与 CEC（reflect）、mabbob（reflect）及 `mabbob.md`"正式默认 reflect"裁决矛盾。现已改为配置驱动：`make_bbob_problem`/`make_cec_problem` 接受 boundary 参数，`runtime_problem_config` 对所有 suite 透传，三个 BBOB/CEC 正式配置已显式 `boundary_handling: reflect`，`phase1-check-config` 强制 BBOB 配置显式声明、train/validation 对双方 reflect。
- **数据影响**：已按 5 seeds 采完的 BBOB train（54 分片）/validation（18 分片）均在 clip 下生成，与统一口径不一致，需 `--overwrite` 全量重采（确定性 seeds 1–5 重跑，runs 数不变：train 3,240 + validation 1,080）；30-seed 旧 dim-10 数据同为 clip，本来就待替换。CEC2017 尚未采集，无影响；mabbob 两侧本来就是 reflect，无需重采。

## 2026-08-21 验证侧 MA-BBOB 扩充（mabbob_validation）

- 新增验证成分池（23 定义，candidate 201–223，`--pool validation`）与 18 定义正式子集（6+6+3+3，`--split validation`）：`configs/phase1_mabbob_validation.yaml`，seeds 1–5、10/20/40D、instance 1、reflect。
- 2026-08-21 追加裁决：`mabbob_formal` 同样扩展为 10/20/40D，且 seeds 统一为 1–5（与 `mabbob_validation` 和主协议一致；1,440 runs），两侧 mabbob 配置的维度/seeds 校验统一要求 [10,20,40] 与 [1..5]，manifest 以 10 为参考维度、20/40D 的 xopt 由 xopt_mode+xopt_seed 确定性再生（已 20/40D 真跑核验）。seeds 统一后 formal 的 72 个分片需 `--overwrite` 全量重采（旧 2-seed 行确定性重算不变、另加 seeds 3–5）；query 采样按 problem×design 生成，不受 optimizer seeds 影响、无需重跑；formal 侧已提取的 behavior 需 `--overwrite` 重提。
- BBOB-validation estimand 扩充为两层 50/50（六原函数层 × 18 定义层），50/50 预指定、两层同源声明见主规范 §14.1；`mabbob_validation` evaluation-only，不进任何 fitting split。
- 已修复 MA-BBOB 采集接线缺陷：`runtime_problem_config` 统一透传 `manifest_path`/`boundary_handling`（此前会静默回退 legacy 定义 + clip）；`factory.problem_bounds` 对 mabbob 直接返回 [-5,5]^d；`phase1_check_trajectory_shards`、`action_losses._parse_problem_id`、query 采样 allowlist、`model.py` split 推断均已支持 mabbob。
- 验收：c201×10D 冒烟 20 runs 通过，reference 与 manifest 逐位一致，reflect 生效（clip 边界堆积 113 vs reflect 0），分片校验 exit 0。剩余正式量：18 定义 × 3 维 × 4 算法 × 5 seeds = 1,080 runs ≈ 75.6M FE。
- train 池 dense 缺陷已修正（2026-08-21）：原 `DENSE_PROFILES` 为固定 24 槽向量（`dominant_trace` 的 0.9 权重锚在 F1 槽位而非 entry 自身 components、`geometric_decay` 锚在 F1/F2/F3、`uniform_24/balanced_dense` 给全部 24 函数含 6 个 validation 函数各 1/24），与 entry 声明的 components 不一致。现统一为 scoped `_dense_weights`：支持集严格限制在池自身成分宇宙内（train 池 18 函数 / validation 池 6 函数）并以 entry components 锚定。42 池与 24 定义选择 manifest 已重生成：`selected_candidate_ids` 与旧版完全一致，仅 6 个 dense 条目（37–42）权重向量变化；`phase1_check_config` 的泄漏检查已升级到权重支持集级别，旧缺陷输入会被拒绝。旧版清单已在 2026-08-24 清理。

## 文档指引

- 活动协议以 `AGENTS.md` 为最高优先级。
- 设计与训练协议以 `docs/10_protocols/Decision-before-Feature Decision Model设计与训练协议.md` 为准。
- 最小字段规范以 `docs/10_protocols/Decision-before-Feature_最小ActionLoss字段规范.md` 为准。
- 结果概览见 `docs/30_results/phase1_current_results.md`。
