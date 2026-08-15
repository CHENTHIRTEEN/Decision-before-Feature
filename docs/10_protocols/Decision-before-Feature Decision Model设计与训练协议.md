# Decision-before-Feature Decision Model 设计与训练协议

> 唯一活动协议（2026-08-15 修订）。旧 18 模型搜索、预制完整-train Utility 上的伪 outer OOF、逐状态 threshold、四组 T0/B1/B2/B3 消融和 validation 参与比较的口径全部退出。本文件定义待实现与待运行的设计，不声称模型或正式结果已经生成。BBOB-validation 已被历史开发读取，只是已见内部评价集；CEC2017 已有历史在线诊断，只是已见外部开发集。

## 1. 研究对象与信息时序

Decision Model 在 query 获取之前工作。给定一条由 fold-specific SBS 产生的完整 native trajectory，它在有序 decision opportunities 上判断是否第一次执行当前固定 query。一个 run 最多触发一次；query 触发后后续机会在该 policy 下不可达。

当前 query sampling 是固定整数 `sample_seed` 定义的确定性算子：base seed、stream code、function number、instance、dimension 与 sample-design code 一起进入 `numpy.random.SeedSequence`。每个 static problem × sample design 只产生一个 problem-keyed LHS realization，optimizer seed、decision state 与 action 共用它。本轮不增加 LHS replicates，也不新增 action losses；所有 query 结论严格条件于这组固定 realizations，不估计对重复 query sampling randomness 的期望。

令 Stage-A 预指定单次科学 outcome 的每条路径非负 benchmark-reference raw gap 先按 suite 配置在
`[1e-12, 1e20]` 截断，再记其十进制对数为
`ell_skip`、`ell_q` 与 `ell_b`。令 `T_skip`、`T_q`、`T_b` 为同一
complete state 和 RNG 起点到 terminal 的 Stage-B 三次真实 timing-only replay 的 censored future-path wall-clock 中位数：completed repetition 取 raw observed time，timed-out/failed repetition 取 `max(raw observed time, role timeout)`；共享 prefix 视为 sunk cost。Stage-B 逐次保存 raw/censored time、status/effective FE/timeout/completion；Stage-A completed 时完成项复现其 gap、`observed_first_hit_FE`、`target_hit_observed` 与 `endpoint_success`，Stage-A 未完成时完成项只检查彼此一致且不得覆盖失败端点；状态混合保留为 instability，且不得选择性补跑。raw observed median 只作诊断。FE=0→terminal policy wall-clock 另报且不进入 Utility。主预测目标是：

\[
U_{query}^{joint}(s_t)=
(\ell_{skip}-\ell_q)
-\lambda_T(\log_{10}T_q-\log_{10}T_{skip}),
\]

即 Query + full Selector 联合路径相对 native SBS continuation 的效用。该量含 query acquisition、Selector error、action transition、remaining-budget 和 continuation effects，不是 query descriptors 的独立边际价值。

另保留：

- `U_behavior_only_full_budget`：Behavior-only Selector 相对 Skip 的 full-budget 效用；
- `query_operational_increment`：Query path 相对 full-budget Behavior-only path 的净增量；必须分别在全 eligible states 与同一 Proposed first-trigger states 报告；
- query-adjusted state-only/query-only/full Selector：只作信息来源诊断。

主操作性情景固定为 `lambda_time=1, lambda_memory=0`，表示 performance gap 与 runtime 的十进制数量级变化等权；`lambda_time={0,0.25,0.5,1,2}` 是完整 sensitivity，不得按 train、validation 或 external result 改选。旧 raw-gap max-scale、相对时间差或一次计时 Utility 全部失效。

`I_q` 同时包含 query FE/runtime、sample best、较短 continuation budget 与 Selector 差异，不是纯信息效应或因果 estimand。若同一分析范围 `U_query_joint>0` 但 `I_q<=0`，只能支持联合路径优于 SBS，不能支持 query acquisition 优于 Behavior-only。正式五路径用 `query_matched_state_only` 与 `sampling_only_continue_current` 进一步分解 descriptor-use、state-only-vs-sampling 与 sampling-direct 操作性增量，并要求逐行加法一致；`query_feature_predictive_increment_log10_gap` 另保留为排除 sample best 的预测诊断。

## 2. Decision 输入

允许输入仅为 query 前可得、对 population 行排列不变的 Behavior：

- improvement rate/frequency；
- diversity 与分布变化；
- set-motion summaries；
- fitness-distribution change；
- stagnation、distance decay、convergence；
- Search Maturity 的确定性派生基函数。

禁止进入 Decision X：

- query features 或 query sample values；
- function ID、BBOB function number、benchmark suite；
- dimension；
- prefix/default/selected algorithm ID；
- optimizer-specific parameter、internal state 或 RNG state；
- benchmark reference value、known optimum、任何 gap；
- observed action losses、best observed action、Selector prediction/regret；
- nominal milestone/event label、window metadata。

`FE_ratio` 通过 `bf_fe_ratio` 进入；后者必须逐行等于实际整数 `FE/FE_total`。算法身份可作为 metadata 分层，但“algorithm-identity-free”不等于行为分布对 prefix optimizer 不敏感，结论限定于主 SBS prefixes。

## 3. 冻结输入组与 Search Maturity

| 组 | 输入 | 字段数 |
|---|---|---:|
| T0 | `bf_fe_ratio` | 1 |
| B1 | core permutation-invariant Behavior | 19 |
| B2 | B1 + longitudinal set dynamics | 25 |
| B2+Motion | B2 + 3 set-motion fields | 28 |
| B2+Maturity | B2 + 3 deterministic maturity fields | 28 |
| B3 | B2 + Motion + Maturity | 31 |

`B2+Motion` 和 `B2+Maturity` 是兄弟组，分别识别 set-motion 与确定性 Maturity 变换的预测增量；B3 同时包含二者。Search Maturity 不增加原始信息，不称为 latent state、因果中介或已验证阶段。`all_candidates` 只是 B3 兼容别名，不是第七组；3 个 diagnostic-only 字段不进入任一组。

T0 主比较只用 12 个预算 milestones，称为 `milestone_only_T0`。B3 与 T0 的直接对比也在同一 milestone rows 上重算。event-only opportunity 的出现依赖 Behavior，因此完整动态 schedule 上的 `schedule_conditioned_T0` 只能作 sensitivity。

T0 仍是强制主 baseline，不并入或替换 Decision X。另增加估计性静态上下文诊断 `dimension_stratified_T0`：在 BBOB 的 10D、20D、40D 内分别执行同一 `cv_group_id = function_id` OOF/threshold 链，每层模型仍只输入 `bf_fe_ratio`。当前 `FE_total=1000d`，所以 `log(dimension)` 与 `log(FE_total)` 完全共线；主 cheap query 的 planned query-cost ratio 恒为 0.05，也不能形成可识别输入。只有 B3 相对 `milestone_only_T0` 与 `dimension_stratified_T0` 的有限集效应方向均一致时，才可写“Behavior 的预测信息超过单纯调用阶段，且该观察不能仅由已评估维度层解释”；该诊断不进入主 Decision X，也不外推到未训练维度。

## 4. 活动模型

活动候选只包括：

1. LDA classification：拟合 `1[U_query_joint_lamT_1 > 0]`；
2. Logistic Regression classification：`C=1`、balanced class weights、L-BFGS，拟合同一标签；
3. Ridge regression：`alpha=1`，拟合连续 `u_query_joint_lamT_1`。

三个候选都使用 Pipeline 内的 train-fold median imputation 与 standard scaling。Random Forest、XGBoost、LightGBM、MLP、SVM、kernel approximation 或额外 feature engineering 不进入活动 Decision Model 搜索。Selection Reference 的固定多输出 Random Forest 是不同组件。

同名模型家族另拟合 `U_behavior_only_full_budget`，用于 `self_thresholded_behavior_only`；不重新选择模型家族。LDA/Logistic score 只表示分类判别分数，不解释为 Utility magnitude；连续 Utility RMSE 只对 Ridge 定义。

## 5. 完整 outer-fold-specific 嵌套

模型选择的证据单位是 BBOB-train outer holdout function。每个 outer fold 必须独立执行整条学习链：

1. 从 outer-fit functions 的 `FE=FE_total` outcomes 计算 `SBS_outer`；
2. 仅使用 `SBS_outer` prefixes 形成 outer-fit/outer-holdout 主 population；
3. 在 outer-fit functions 内部 cross-fit Query Selector 与 Behavior-only Selector，生成不含同 function in-sample Selector prediction 的 outer-fit Decision labels；
4. 由两套 Stage-A action matrices、selected replay plan、Stage-B 三次 state-to-terminal timing-only replay 和 Stage-A 截断 `log10_gap` 生成三类 Utility；
5. 对每个活动 Decision candidate，在 outer-fit functions 内做 inner function folds；每个 inner fold 必须只用 inner-fit functions 重新计算 `SBS_inner`，在 inner-fit 内 cross-fit 并重拟合两类 Selector，生成 inner-fit Decision labels，再拟合 preprocessing 与 Decision；
6. inner-holdout Utility 与 score 只能由 `SBS_inner` 和 inner-fit 上游组件生成。拼接这些端到端 inner OOF rows，分别按 Query-joint 与 Behavior-only run-level first-trigger objective 冻结 outer threshold；T0 只读 milestones；
7. 用 `SBS_outer` 在全部 outer-fit functions 上重建 outer-fit labels，并拟合两类 Selector、Decision 与 preprocessing；
8. 只在组件冻结后，对 outer holdout function 生成 Utility、score、first-trigger policy outcome 与所有 baseline 指标。

outer 或 inner holdout 均不得参与其评价链中的 SBS、Selector、Utility label、imputation、scaling、model、threshold、Random calibration、score-neighborhood 或 feature-group decision。不得先用完整 BBOB-train 生成 Utility labels 再仅对 Decision Model 分 folds，并把所得分数称为端到端 nested OOF。

外层 fold 数和内层 fold 数只由 BBOB-train function-ID groups 与预设 GroupKFold 规则确定；同一 transformed instance 或 seed 不得跨 function group 作为独立 holdout。该设计统一称为 `cv_group_id = function_id` 或 function-level split。代码中字段 `cv_group_id` 等于 `function_id`（如 `bbob_f001`），是 CV 分组键；字段 `family`（如 `bbob_separable_f01_f05`）是景观 taxonomy 字段，不用于 CV 分组，也不支持跨 landscape-family 泛化表述。

## 6. First-trigger threshold 与模型选择

对 run `r` 的有序机会 `s_{r1},...,s_{rK}`，score 为 `z_{rj}`。给定 threshold `tau`：

\[
J_r(\tau)=\min\{j:z_{rj}>\tau\}.
\]

集合为空时 run Utility 为 0；否则只使用 `U(s_{rJ_r})`。threshold objective 是所有 outer-fit inner-OOF runs 经 run → static problem → fixed dimension stratum → function 聚合后的 mean first-trigger Utility。若并列，先选调用 runs 更少的 threshold；仍并列选数值更大的 threshold。

三候选主选择只看 B3 上拼接 outer holdouts 后的 function-balanced mean first-trigger `u_query_joint_lamT_1`。并列顺序固定 LDA → Logistic Regression → Ridge。AUROC、Average Precision、Spearman、Ridge RMSE、T0、validation 或 external result 不改写选择。

拼接 train outer OOF 只用于预设候选选择与开发期诊断。选择最大 OOF 候选后，同一 OOF 不能称为 selected procedure 的无偏 estimate。BBOB-validation 已被历史模型比较、调参、消融和采样设计读取，只能给 selected procedure 与 milestone-only B3--T0 的已见内部有限集估计；删除或撤回旧产物不能恢复“未见”状态。CEC2017 同样只作已见外部开发集估计。确认性外部证据只能来自本次协议冻结后才生成并首次运行的 CEC2022 与工程集合；三候选两两 outer-OOF 对比仍只是选模诊断。

选择模型名后，同一名字用于 T0、B1、B2、B2+Motion、B2+Maturity、B3 和 Behavior-only fit；每组仍按自身 train-only OOF scores 冻结 threshold，但不重新选模型或主 feature group。

最终部署使用完整 BBOB-train 重新执行相同 function-OOF 链，冻结：

- full-train SBS；
- Query Selector 与 Behavior-only Selector；
- selected B3 Decision model；
- `oof_utility_first_trigger` threshold；
- `oof_behavior_utility_first_trigger` threshold；
- Proposed OOF run-level call rate及 first-trigger `FE_ratio` 经验分布，供 `matched_rate_random` 使用。

BBOB-validation、CEC2017、CEC2022 与工程问题均只加载这些 BBOB-train frozen components。证据角色不同：BBOB-validation 是已见内部评价集，CEC2017 是已见外部开发集；CEC2022 与工程集合只有在函数/问题范围、维度、预算、重复、reference/constraint rule、失败端点、runner/factory 与分析 contrasts 全部冻结且未查看任何 outcome 后，才是确认性外部评价。当前 CEC2022 缺少冻结配置，工程集合还缺少配置、factory 与 constraint endpoint，二者均是正式确认性运行 blocker。

主训练改为 `cluster_balanced_fit`。在每个 fit fold 内，先使 functions 等权，再使每个 function 内的固定 dimension strata 等权、每个 function × dimension 内的 static problems 等权、每个 static problem 内的 optimizer runs 等权，最后把每个 run 的权重等分到其合格 states；row weights 只在该 fit fold 内计算并归一化为均值 1。`sample_weight=1` 的 state-row 等权拟合降为 `row_weighted_fit` sensitivity。nested first-trigger evaluation 不会自动修正训练权重；在 imputation/scaling/三候选 estimator 尚未对同一 cluster-balanced fit population 完成兼容实现前，不得冻结正式模型。

## 7. Policy 指标

模型选择、threshold、validation、baseline 和主策略指标全部使用 trajectory first-trigger。主指标包括：

- run-level mean joint Utility；
- matched-trigger query operational increment；
- run-level call/trigger/handoff rates；
- first-call precision 与 non-beneficial first-call Utility；
- first-trigger utility capture；
- final `log10_gap`、target-hit rate、endpoint-success rate、ERT；
- decision-state future-path ratio、FE=0→terminal policy wall-clock 与 peak memory；
- coverage 与失败敏感性。

utility capture 对所有策略共享同一 run-level hindsight opportunity reference：在 native SBS/default trajectory 的全部预定义合格机会中取 `H_r=max_t max(0,U_t)`。策略分子只取其 first-trigger state 的 `max(0,U)`，未触发为 0。该分母不随策略触发时点改变，也不是可部署 policy；聚合时报告加权分子、分母和二者比值，并单报 `H_r=0` 的 run 比例。逐状态 AUROC/AP/Spearman 与 state-level capture 只能标为 auxiliary score diagnostics。

## 8. 统计规则

function 是最高聚合层。BBOB-validation 的 F5/F9/F13/F14/F19/F24、固定 dimensions 与 instances 1/2/3 是已见固定有限集合，不是 function 或 transformed-instance 超总体样本；其 estimand 是这些固定单元的等权有限集均值。10,000 次条件配对 bootstrap 始终保留全部 6 functions、全部 fixed dimensions 与全部 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds。RQ1 对每个抽中 seed/run 保留完整有序 state 簇。function-resampling 只作函数组成敏感性，不进入主 95% CI，也不能恢复确认性或产生超总体区间。CEC2017 按同样原则只给已见外部开发集有限集估计。

ERT 不进入通用“run-level 数值先求差再取算术均值”的 bootstrap。每个 policy 在每个 `function × dimension` stratum 内以 `ERT=N_FE/N_hit` 重算：每个 bootstrap replicate 固定全部 static problems，只在每个 problem 内联合配对重抽 optimizer runs，分别重算 treatment/reference 的 FE numerator 与 hit count，再形成 `log10(ERT_treatment/ERT_reference)`；随后对固定 dimensions 等权得到 function effect，最后对固定 functions 等权。单方 `N_hit=0` 的 stratum 保留为有符号无穷，双方 `N_hit=0` 记为显式 undefined mass；不得静默删除 stratum 或 replicate。区间使用扩展实数分位数，将 undefined mass 保守分配到两侧尾部，并分开保存 finite/unbounded/undefined-observed 状态、undefined mass 与各类零命中计数。`interval_established` 只表示 observed contrast 和扩展实数边界是否有定义，不得因为一个偶发 bootstrap replicate 出现零命中就改变。绝对 ERT 逐 `function × dimension` 报告；不同维度的 raw FE 不得先池化成一个总体 ERT。若另报总体绝对量，只允许使用预先定义的 budget-normalized ERT 有限集汇总，且不能替代主 log-ratio。

Utility ±0.01、`log10_gap` ±0.05、runtime ratio `[0.95,1.05]`、call/target-hit-rate 差 ±0.05 只称为“项目内预设 operational tolerance”，没有独立领域依据时不得称 confirmatory equivalence。BBOB-validation 与 CEC2017 只用 95% 条件区间逐项描述相对 tolerance 的位置；差异不显著不表示等价，Utility 中的 endpoint 抵消也不能建立任一端点等价。若未来 untouched external suite 预先声明 simultaneous intervals，其 family、interval level 与解释必须在首次 outcome 前冻结，但当前第一篇论文不据这些项目内 tolerance 作确认性等价声明。

RQ2 的主要科学 contrast 仍是冻结模型家族的 milestone-only B3--T0；其 BBOB-validation 数值只是已见内部有限集估计，确认性证据等待 untouched CEC2022 与工程集合。RQ3--RQ5 均改为估计性分析，以逐 function/problem effects、固定有限集均值和条件 95% CI 为主。双侧 sign-flip/Holm 仅可作为明确依赖“固定 function effects 的 signs 可交换”假设的辅助敏感性：六函数 exact raw p 最小 0.03125，RQ3 与 RQ5 各自六 contrasts 的最小 Holm-adjusted p 均为 0.1875，不能作为 RQ 成败判据。RQ4 按 suite 与 endpoint 分开，不把四个 suites 组成一个 Holm family；某 suite 内若有多个 contrasts，必须在首次 outcome 前单独冻结 family。未拒绝不表示无效或等价，任何 raw/adjusted p 都不支持函数超总体推断。前瞻外部评价同样以预设有限 suite 的效应量、区间、coverage 与失败敏感性为主。

## 9. 失败与产物资格

Decision score 缺失或非有限时，该机会按 No-query；若 run 尚未触发，可在下一可达机会重新检查。query 触发后的 query/Selector failure 保留 FE 和 wall-clock，fallback 为 query-adjusted native `continue_current`。所有计划 run 进入 coverage denominator。BBOB train/validation 与 CEC2017 固定 raw-gap floor/cap `1e-12/1e20`、success target `1e-8`、单 state-action path timeout `3600 s`，并在 Stage-A 每次 objective evaluation 记录 first hit；Stage-A timeout 计科学失败且保留，ERT 失败项计完整 planned budget。Stage-B timeout/failure 只进入 timing 状态、instability 与 failure sensitivity，不重定义科学 endpoint。

每个 suite × endpoint 同时报 attempted denominator/coverage、complete-pair estimate 与双向极端 failure sensitivity。未闭合 pair 的 favorable/adverse 赋值为 gap floor/cap、`target_hit_observed` 1/0、ERT 未命中项的 full planned budget、runtime 取该 suite complete pairs 最小正值/timeout；Utility 从同一组极端 endpoint 重算。`endpoint_success` 若作为单独 endpoint，也按 1/0 分配但不得替代 ERT 的 target-hit 定义。已按 cap/timeout 保留的科学 path failure 是观测，不当作 missing pair。效应方向、区间相对 operational tolerance 的位置任一改变，或 coverage<95%，对应结论未建立。

`experiments/hierarchical_inference.py::paired_run_effects` 中的缺失配对删除只定义 complete-pair estimate，不是 attempted-population result。任何 consumer 必须另接计划 denominator、coverage 与上述双向 failure sensitivity；该 consumer 未闭合前，通用 paired interval 不能单独进入 suite 结论。

`paired_ert_strata` 与 `paired_hierarchical_ert_log10_ratio_interval` 当前也只是专用 ERT 计算核，尚无 suite 汇总 caller 将 ratio、attempted denominator/coverage、双向 failure sensitivity 与正式报告表接在同一分析链中。该 consumer 未实现前，专用 ERT 输出同样不得进入 suite 结论。

旧重建式 trajectory、旧 Selection Reference、旧 Utility 数值、完整-train 预制 Utility 上的 Decision-only OOF、逐状态 threshold、旧 `oof_utility`、四组消融和 validation 参与的产物均无正式证据资格。只有通过第 5 节完整 outer/inner 上游链、整数 FE state coverage、Stage-A/Stage-B 字段分离、三次预定计时及 endpoint consistency/instability 检查和 first-trigger policy reconstruction 的新产物可进入模型选择或论文结果。

## 10. 生成顺序

```text
outer-fit complete-budget outcomes -> SBS_outer
-> two action-loss matrices once
-> cross-fitted Query/Behavior-only Selectors -> fold-role selected replay plan
-> Stage-B three-repeat decision-state future-path timing only
-> Stage-A log-gap + Stage-B median log-runtime joint/behavior/operational-increment Utility
-> inner-fold-specific SBS/Selectors/Utility/Decision OOF + first-trigger thresholds
-> outer-fit final components
-> one outer-holdout evaluation
-> concatenated outer OOF model selection
-> full-train OOF threshold and final refit
-> frozen seen-set estimation + untouched external confirmation after suite closure
```
