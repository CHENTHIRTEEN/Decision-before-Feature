# Decision-before-Feature

本项目研究一个前置于特征计算的资源决策问题：在黑盒优化过程中，是否值得为当前搜索状态执行一项预先定义的 landscape-analysis query。项目不设计新的优化算法，而是用离线轨迹和监督学习估计固定 query 的状态依赖效用。

## 当前状态

截至 2026-08-14，四种优化器已改为完整状态推进：同一算法在 checkpoint 间保留内部状态与 RNG；只有 selector 确实切换算法时，才执行一次显式的 population transfer 初始化。可用 `uv run optimizer-state-check` 在真实 BBOB 问题上检查连续运行与多次 checkpoint 保存/恢复的一致性。

Behavior extractor 同时已改为 permutation-invariant 的种群集合统计：跨窗口的空间变化使用经验 Wasserstein、centroid shift 和协方差谱集中度，fitness 变化使用经验分位数分布；不再把 population 行号解释为跨代个体身份。运行时逐次记录完整原生 update 的轻量窗口统计，正式 behavior state 的 w02/w05/w10 anchor 不再从稀疏输出状态中选择；若名义 FE 不能整除一次原生 update，则取不晚于目标位置的最近完整 update，误差严格小于一次 update，并保存 `effective_window_ratio_*`、`effective_window_fe_*` 与 `effective_native_updates_*`。所有 rate/slope 使用实际 `ΔFE/FE_total`，这些窗口字段只作 metadata，不进入 Decision 输入。

正式状态采样已冻结为 `phase1_dynamic_budget_event_v1`：在 `0.20–0.60` 上按 `0.01` 候选网格监测，保留 12 个预定义预算里程碑，并依据 improvement resume、stagnation onset、effective-rank change、elite migration 与 diversity recovery 补充事件状态；每个跨过至少一个 0.01 监测网格的完整原生 update 只判定一次事件。同一 update 跨过多个监测点时，若包含预算里程碑，则里程碑与事件合并为一行，且该行不消耗 event-only 配额、最小间隔锚点或 `event_index_in_phase`；若不含里程碑，则以最新跨过的监测点作为名义节点。每个 run 输出 12–18 个状态。`FE_ratio` 始终是实际 `FE/FE_total`，名义里程碑另存 `budget_milestone_ratio`，状态连接使用整数 `FE` 而非浮点 ratio。完整预算终值另存为每个 `problem_id × algorithm × seed` 在 `FE=FE_total` 恰好一行的 `final_performance.parquet`；该表与 `0.20–0.60` decision trajectory 分离，不能把 `0.60` 的最后一个 decision state 当作完整预算终值。

唯一活动协议把 Selection Reference 定义为逐共享状态候选动作损失回归：每个 state 对 `continue_current` 和其余三个 portfolio algorithm 分别进行真实 continuation，`remaining_budget_ratio` 作为连续输入；不再按静态 problem label 和 nearest performance bucket 选择算法。正式链路同时生成 Query-adjusted 与 `behavior_only_full_budget` 两套动作预算，固定多输出 Random Forest 预测相对 `continue_current` 的 continuation-only `clipped_log10_gap_advantage_vs_continue_current`；旧 `statewise_minmax_observed_action_loss` 只作 target sensitivity。预指定 Stage-A 单次科学运行固定 terminal gap、`observed_first_hit_FE`、`target_hit_observed`、`path_completed`、`endpoint_success`、planned/effective FE 与失败状态；主 Query 科学端点另计入真实 query sample best/first hit，sample 不进入 optimizer population。Stage-B 对五条 decision-state-to-terminal 路径各进行三次 replay，只提供 future-path wall-clock 中位数和重复级计时诊断，不改写 Stage-A 科学端点。该双矩阵、fold-specific selected replay plan、五路径计时和完整嵌套尚未产生可用正式产物。

三档 query 提取器已实现统一 `unit_cube_x__median_iqr_y_v1` 前处理：cheap/standard 共享 `lhs_50d`，broad 使用 `lhs_100d`；隔离 pflacco 1.2.2 提取器对预处理后的 X/y 计算冻结 whitelist，终点评价字段只作 metadata。活动 query ID 为 `descriptor_cheap_invariant`、`pflacco_standard_invariant`、`pflacco_broad_invariant`。实现完成不等于正式数据检查或结果完成；72 个正式 trajectory shards 尚未启动。

Decision Model 的活动候选固定为 LDA、Logistic Regression 与 Ridge。每个 outer fold 单独计算 `SBS_outer` 并拟合该 fold 的 Selectors/Utility/Decision；每个 inner fold 又只用 inner-fit functions 重算 `SBS_inner`、Selectors 与 Utility，端到端 inner OOF first-trigger outcomes 冻结 outer threshold。完整 BBOB-train 的部署 threshold 与 Random calibration 也必须来自 grouped-by-function OOF 上游链。BBOB-validation 已被旧模型比较、调参与消融查看，只能作已见内部评价集；CEC2017 也已有 preliminary/targeted 结果，只能作已见外部开发集。Random Forest、XGBoost、LightGBM、MLP 及分类特征工程搜索已退出 Decision Model 活动调参路径；Selection Reference 中固定的 Random Forest action-loss regression 不受此约束。

此前生成的 BBOB trajectory 使用了重建式 continuation，旧 behavior 含有依赖行号对应关系的字段，旧 landscape 表又把 16 个自定义描述符笼统称为 ELA。旧 utility labels、Decision dataset、模型、baseline 和成本—性能结果因此全部撤回；必须从 trajectory 开始按依赖顺序重新生成。已有 CEC2017 在线结果同样不能用于外部结论。

这些撤回产物已全部移出活动结果路径，本机仅在 `results/archive/` 下封存且不进入 Git。当前不存在可复用的正式模型、checkpoint 或论文结果表。

当前结果的完整口径见 [docs/30_results/phase1_current_results.md](docs/30_results/phase1_current_results.md)，跨对话状态见 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。

## 冻结实验协议

- 训练：BBOB 10D / 20D / 40D。
- 已见内部评价集：BBOB-validation 10D / 20D / 40D，按 function ID 与 BBOB-train 隔离；它不再承担独立确认性评价。
- 已见外部开发集：CEC2017。其函数口径仍待核对，不能承担确认性结论。
- 未触及外部确认候选：CEC2022 与工程问题。只有在本轮协议冻结后、且 functions/problems、预算、端点、失败、约束规则、顶层有限集单位与固定 strata 权重均先写入当前项目再首次生成 outcome，才可承担未触及外部确认评价；当前实现尚不满足这些条件。CEC suite 以 function 为顶层有限集单位；工程集合以预先命名的 engineering problem 为顶层单位，不强行套用 BBOB function/dimension 层级。
- 算法池：DE、PSO、CMA-ES、SHADE。
- 主采样协议：`phase1_dynamic_budget_event_v1`；12 个必选预算里程碑为 `0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.60`，事件状态使每个 run 总计 12–18 行。
- 主 query 固定为 `descriptor_cheap_invariant`：14 个自定义低成本描述符，使用 `lhs_50d`，即 5% 总 FE。统一 median/IQR preprocessing 后恒为 0/1 的 `descriptor_y_median`、`descriptor_y_iqr` 已从活动 whitelist 删除；query ID、采样和 action losses 不变。
- `pflacco_standard_invariant`（37 维，`lhs_50d`）与 `pflacco_broad_invariant`（52 维，`lhs_100d`，10% 总 FE）用于当前固定的配置稳健性实验；从本次冻结起不得根据 BBOB-validation、CEC2017 或后续外部结果改选主 query。
- Decision 输入仅来自 permutation-invariant 的算法无关搜索行为；function、dimension、algorithm、query feature 和优化器内部状态只能作为 metadata 或分层报告字段。
- Decision feature-group 比较固定为 `T0/B1/B2/B2+Motion/B2+Maturity/B3=1/19/25/28/28/31`。`B2+Motion` 与 `B2+Maturity` 是同维度兄弟组，用于区分 set-motion 与确定性 Maturity 变换；B3 同时包含二者。RQ2 的主 time-only 比较只在 12 个预定义预算里程碑上进行：`milestone-only B3 - milestone-only T0` 必须逐 state key 使用相同行。动态 Proposed 与 `milestone_only_T0` 若进入 RQ3，只是包含机会调度、特征、拟合分数和 threshold 差异的整政策比较，不能归因为 Behavior 增量；事件机会上的 T0 只能称为 schedule-conditioned sensitivity。`all_candidates` 仅是 B3 的兼容别名。
- Decision Model 活动候选严格为 LDA、Logistic Regression、Ridge；主选择指标为 grouped-by-function outer OOF 的 run-level first-trigger mean joint utility。train outer OOF 只承担候选选择和开发期诊断，不能在选择后继续充当 selected procedure 的无偏性能估计；BBOB-validation 与 CEC2017 均已被查看，也不能恢复该资格。所选 procedure 与 RQ2 的 B3--T0 目前没有独立确认性结果，只有未来按冻结协议首次运行的 CEC2022/工程集合可提供相应证据。AUROC、Average Precision、Spearman 为逐状态辅助指标，连续 Utility RMSE 只对 Ridge 定义。
- 一个 run 最多执行一次 query。阈值、模型比较、call rate、precision、utility capture 与 policy utility 全部按最早越阈值状态计算；首次调用后的后续状态不可达，不进入该 run 的策略效用。
- policy utility capture 对所有策略共享 native SBS/default trajectory 上全部预定义机会的 run-level hindsight maximum；策略分子只取其 first-trigger state，未触发为 0。该参照不随策略触发时点改变，也不是可部署 policy。
- BBOB-validation 不参与 SBS、Selector、preprocessing、模型、候选选择或 threshold 拟合；部署阈值模式固定为 `oof_utility_first_trigger`。
- 主联合策略标签为 `u_query_joint_lamT_1`，回答 Query/full-Selector operational path 相对 native SBS 的净差；`query_operational_increment_lamT_1` 比较 Query 与 full-budget Behavior-only 路径，包含 query FE、runtime、sample best、较短 continuation budget与 Selector 差异，不是纯信息效应或因果 estimand。必须同时报告全 eligible-state 与 Proposed-triggered `I_q`；若 `U_joint>0` 但 `I_q<=0`，只能支持联合路径优于 SBS，不能支持 query acquisition 优于 Behavior-only。五路径还包含 `query_matched_state_only` 与 `sampling_only_continue_current`：前者在相同 acquisition、sample endpoint、query-adjusted action matrix 和预算下移除 descriptors，后者执行相同 acquisition 后原生继续当前算法；分别生成 `query_descriptor_use_increment_lamT_*`、`query_state_only_vs_sampling_increment_lamT_*` 和 `query_sampling_direct_increment_lamT_*`，并与 `u_query_joint_lamT_*` 逐行加法一致。`query_feature_predictive_increment_log10_gap` 继续作为同一 action matrix 上排除 sample best 的 OOF 预测诊断。五路径分解不作因果解释。
- Utility 性能项只使用预指定 Stage-A 单次科学运行按 suite floor/cap 处理后的 `log10_gap` 差；时间项使用 Stage-B 三次 decision-state-to-terminal future-path 的删失 wall-clock 中位数：completed repetition 使用实际时间，timed-out/failed repetition 使用 `max(raw observed runtime, role timeout)`。原始观测中位数只作诊断。主 `log10` ratio 逐行满足 `query_operational_increment = joint - behavior_only`。共享 optimizer prefix 已经发生，视为 sunk cost，不进入状态条件 Utility 时间。`lambda_time=1`、`lambda_memory=0` 表示 gap 与未来路径 runtime 的十进制数量级变化等权；`lambda_time={0,0.25,0.5,1,2}` 只作预设敏感性分析。旧 raw-gap max-scale、线性相对时间、一次计时及把快速失败原始耗时作为主成本的 Utility 全部失效。
- 主效用之外分别报告 Stage-A 终端 `log10_gap`、Stage-B decision-state future-path runtime ratio 和 FE=0→terminal 在线政策 wall-clock。前两者分别提供科学性能端点与到达状态后的条件计时，后者评估完整政策运行；三者不得混称或互相替代。Stage-B 使用固定机器/线程/预加载进程；每条 selected future path 从同一复制 state/RNG 真实执行预定 3 次，canonical order 按 `cyclic_complete_path_v1` 循环移位。每次保存 repetition/order、raw/censored 组件与完整未来路径时间、`timing_replay_status in {completed,timed_out,failed}`、`observed_first_hit_FE`、`target_hit_observed`、`target_hit_before_failure`、`path_completed`、`endpoint_success`、effective FE 与 timeout。`first_hit_FE/success` 若保留，只能分别作为 observed first hit/target-hit-observed 的兼容别名；正式 ERT 按曾达到 target 计算，而完整路径且命中的 endpoint 另由 `endpoint_success` 表示。路径身份、completed replays 内部结果和 Stage-A→completed-replay 结果分别使用独立一致性字段；Stage-B 内部状态混合与 Stage-A/Stage-B completion 不稳定也分字段保存。主时间使用删失中位数，raw observed median 与旧 `failure_worst_case` 仅作诊断/兼容。不得复制计时，也不得在观察时间或状态后选择性补跑、删改重复、改顺序或改 lambda。
- `benchmark_reference_value` 和所有 gap 字段只用于离线标签和最终评价，不进入 Behavior、Selection Reference 输入或 Decision X；使用已知最优值计算离线标签并不意味着在线优化器知道最优值。
- 第一篇论文主 probe/default 固定为 fold-specific SBS。SBS 使用相应 fit functions 的完整预算 `log10_gap`，按 run → static problem（function × dimension × instance）→ fixed dimension stratum → function 等权聚合，选择均值最低的算法；并列按 `de,pso,cmaes,shade`。这与主性能端点和 function 顶层权重一致，不再用平均 rank 丢弃效应量。No-query 原生继续该 SBS 的完整 checkpoint state。
- Query 后选择当前 prefix 时原生继续；选择其他算法时采用一次 checkpoint population transfer；query 采样点不并入后续优化 population。
- 多 prefix 行单独用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化，不进入主 Decision 数据。
- 标签显式保存 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 和 `skip_switches_from_prefix`，不再生成含义模糊的 selected-vs-default 字符串分层。
- `no_query_algorithm` 显式保存 No-query 分支算法并等于 `default_algorithm`；`handoff_type` 显式保存 Query-selected action 的 transition 类型并等于 `query_transition_mode`；`handoff_required` 等价于 `handoff_type == population_transfer_initialization`。
- 逐状态最小 action loss 称为 `best observed action`，只用于潜在性能差与 selector regret 诊断，不称为 oracle，也不进入 Decision 输入。
- 静态 VBS 在每个 `function × instance × dimension` problem 内先对每个算法的完整预算 clipped `log10_gap` 跨 optimizer seeds 取均值，选择均值最低算法（并列按 `de,pso,cmaes,shade`），再用该算法的逐 seed paired outcomes 汇总；不得逐 seed 选择最小算法，也不得用逐状态 `best observed action` 替代。
- Baseline 额外固定为：`matched_rate_random` 只用 BBOB-train 端到端 OOF Proposed 冻结 run-level 调用率与 trigger-FE 经验分布，每 run 预抽目标 ratio 并在第一个不早于目标的在线机会触发；30 个 Random streams 的 outcomes 先在同一 run 内平均，再进入 problem/function 聚合，不能当作 30 个独立 runs。`pre_run_aas_fe0` 是 FE=0、query-only、sample-isolated 的 pre-run AAS：仅用 query features 选择初始算法，query sample 不初始化 optimizer population；关系记账为 `prefix_algorithm=selected_algorithm`、`selected_equals_prefix=true`、`handoff_required=false`、`handoff_type=fresh_optimizer_initialization`，而 `default_algorithm=no_query_algorithm=SBS_fold`。fresh initialization 只记到 `runtime_fresh_initialization`，`runtime_handoff` 保留给已有 population 的 transfer initialization。`Always Query` 在首个在线机会调用，三者不得混称。该 baseline 及所有 portfolio 结论仅适用于仓内冻结实现、参数与 `population_size=40`。
- BBOB train/validation 与 CEC2017 固定 failure cap `1e20`、取 `log10` 前 raw-gap floor/cap `1e-12/1e20`、success target `1e-8`、`action_timeout_seconds=3600`、`timing_replay_timeout_seconds=3600`、`policy_timeout_seconds=3600` 与逐 objective evaluation first hit；三种 timeout 分别约束 Stage-A action continuation、Stage-B decision-state timing replay 和 FE=0 full-policy path，不得混用。Stage-A timeout/failed path 的 final-gap endpoint 仍按失败 cap 保留，但若失败前已经达到 target，`target_hit_observed=true` 且 ERT 使用该 observed first hit；`endpoint_success=false` 继续明确完整路径没有完成。未达到 target 的 ERT 项计完整 planned budget；effective FE 逐行保留。query sample 不进入 population，但 sample best/first hit 进入 operational Query Stage-A endpoint与 ERT；另报 continuation-only gap和 sample-best contribution。Stage-B timeout/completion 只进入计时状态、不稳定性与失败敏感性，不改写 Stage-A gap 或 path completion。
- 正式结果保留所有计划运行和 query failure。Decision score 缺失时该机会按 No-query 处理；query 已触发后若特征或 selector 失败，query FE 与时间仍计入，并按预设 fallback 继续当前算法；不得删除失败行后重新计算调用率。每个 suite/endpoint 同时报 attempted denominator、complete-pair 与双向极端 failure sensitivity：gap 用 floor/cap；`target_hit_observed` 用 1/0；ERT 的 adverse 未命中项计完整 planned budget，favorable 命中项使用在已知 prefix、已消耗 query FE 与路径时间原点下最早可行的 objective-evaluation index；runtime adverse 用 timeout、favorable 用该 suite complete pairs 的最小正 runtime。若最早可行 hit FE 或最小正 runtime 无法由正式行重建，相应 sensitivity 为 undefined，结论未建立。Utility 由同一组极端端点重算。方向、区间相对 operational tolerance 的位置任一改变或 coverage<95% 时，结论未建立；已按 cap 保留的科学 path failure 不当作 missing pair。
- 单个 problem-keyed LHS 使主结果成为“固定 query realization 下的条件 estimand”，不能泛化到 sampling randomness。多个 LHS 只有在查看结果前加入并纳入配对/层级分析，才能作为 sampling-robustness sensitivity；standard 与 broad 同时改变 representation、sample size、sample realization 和预算，差异不能拆分归因。
- RQ1 的目标分布统一按 state → run → static problem → fixed dimension stratum → function 聚合；policy endpoint 从 run 层开始。活动协议把 function/dimension/static-problem/run-balanced 权重升为主拟合，run 权重再均分给其 states；现有 estimator wiring 尚未实现该权重，因而仍是正式运行 blocker。旧 unweighted row fit 只可作敏感性，nested first-trigger evaluation 本身不会修正训练权重。
- BBOB-validation 的可报告 estimand 是 F5/F9/F13/F14/F19/F24、固定 dimensions 与 instances 1/2/3 上的等权有限集均值，只能作条件性内部开发估计。主区间固定保留这 6 个 functions、全部 dimensions 与全部 static problems，仅在每个固定 static problem 内配对重抽 optimizer seeds；RQ1 对每个抽中 seed/run 保留完整有序 state 序列。重抽 function 只作函数组成敏感性，不进入主 CI，也不产生 function 或 transformed-instance 超总体推断。当前 3 instances × 30 seeds 没有仓内精度依据，不得声称功效充分；CEC2022/工程问题的重复数须在查看其 outcome 前另行给出精度目标并冻结。
- Utility ±0.01、`log10_gap` ±0.05、runtime ratio [0.95,1.05]、call/target-hit rate ±0.05 仅称“项目内预设 operational tolerance”。主条件区间和预设 family 的描述性 simultaneous intervals 只说明相对这些边界的位置，不能把任意项目边界升级为领域通用等价界；若要评价 endpoint-success rate，必须单独预设其边界，不能复用 target-hit rate 的名称。
- RQ2 的主要科学 contrast 是冻结模型家族的 milestone-only B3--T0；三候选两两 outer-OOF 比较只作选模诊断。六个固定且已见 functions 不是函数总体的随机样本，双侧 sign-flip 还额外要求 function-effect signs 可交换；其 raw/adjusted p 只可作为假设敏感的辅助描述。RQ3 与 RQ5 各有六个预设辅助 contrasts，在六函数下最小 Holm-adjusted p 均为 0.1875，数学上不可能在 0.05 下拒绝。RQ4 按 suite 与 endpoint 分别给有限集估计，不把四个 suites 冒充一个四 contrast Holm family。三者都以有限集效应量、逐 function/problem 结果和条件区间为主。
- 当前证据包最多支持“是否调用一个固定 query 是可学习的资源决策”这一条件性主张。若三档 query、完整在线政策和未查看外部集合未闭合，不得扩张为“任意 ELA/landscape analysis 都应被优化”或跨表示的普遍结论。

## 计算规模与运行前停止条件

BBOB train + validation 的基础轨迹共有 25,920 runs、至少 311,040 mandatory states 与 0.6048B 基础 FE。12 个必选 milestones 的平均 prefix ratio 是 0.35，因此 mandatory-only future path 平均为 0.65B；旧“所有 states 均位于 0.60B”的成本口径系统性低估 1.625 倍。只含 mandatory states、跨 Stage-A matrices 共享且不复用基础 trajectory 时，Stage A 为 main cheap **37.467612B FE**、三档 **53.434836B FE**；若逐行证明基础 trajectory 终值完全同义，可降为 **32.750172B / 48.717396B FE**。当前 main producer 仍分别执行两个额外 Skip，Stage A 为 **46.902492B FE**；三档当前实现量等待实际调用图枚举。Query sample artifact 仍按 `problem × sample_design` 生成一次，不能按 state/prefix 重复计入。

Utility 必须两阶段生成：阶段 A 把 action matrices 和 FE=0 outcomes 各运行一次，以这些预指定运行固定科学 gap/observed first hit/path completion/planned/effective FE 与失败状态并拟合 fold-specific Selectors；Selector 冻结并枚举 5 outer × 4 inner 加 full-train→已见 BBOB-validation roles 后，阶段 B 才对实际 selected decision-state future paths真实重跑三次。按当前 fold 覆盖推导，train state 有 22 个 replay roles，BBOB-validation state 有 1 个。仅 12 个 mandatory milestones 时，跨 matrices 共享但不复用基础 trajectory 的总量为 main cheap **215.709732B FE**、三档 **350.202636B FE**；进一步复用基础 trajectory 时为 **210.992292B / 345.485196B FE**；保持当前 main producer 时主 query 为 **225.144612B FE**。event-only states、完整在线政策、失败、外部 suites 和额外 query replicates 全部未计入，所以这些只是 mandatory-only 算术情景，不是严格下界或资源承诺。

Replay planner 已有枚举能力，但当前没有 offline decision-state-to-terminal runner、物化实测的 fold-role-complete replay plan、fold-role→Selector artifact 路由、真实 evaluator timing 或已确认资源排期；Stage-A Skip 共享/复用也未兑现。当前 CEC2017 online evaluator 的 7 条固定政策加 30 条 Random、每条 1 次科学运行加 3 次 timing replay，按现配置另需约 **11.5884B planned FE**，尚未并入前述 BBOB replay 总量。它只支持 CEC、固定 `instance=1`；若覆盖 BBOB-validation 全部 instances，同样政策集合约需 **5.5944B FE**，但当前实现不可执行。`configs/phase1_cec2017_test.yaml` 仍列 F1--F29（包含 F2、排除 F30），项目内尚无依据证明这符合所用实现与官方口径；必须在查看 policy outcomes 前核对冻结。CEC2022/工程问题还缺 functions/problems、预算、reference、gap floor/cap、success target、三种 timeout、first-hit 与 constraint rule。上述均为正式运行 blocker。

## 正式入口

正式配置只有：

- `configs/phase1_bbob_train.yaml`
- `configs/phase1_bbob_validation.yaml`
- `configs/phase1_cec2017_test.yaml`

主要命令可通过 `uv run <command> --help` 查看参数：

Selection Reference、Utility、Decision、baseline 与外部评价入口均显式区分 `query_id`。Decision 分析命令使用 `--query-id` 从 `results/decision/{query_id}/` 推导默认输入和输出；显式传入的 artifact 仍必须通过 query 协议核对。

- 数据采集：`phase1-plan-shards`、`phase1-collect-batch`、`phase1-check-trajectory-shards`、`optimizer-state-check`、`behavior-permutation-check`
- Query 与标签：`query-sample-batch`、`query-extract-cheap`、隔离的 `tools/pflacco_query/extract.py`、`query-consistency`、`selection-reference-check`、`selection-reference-evaluate-actions`、`selection-reference-build`、`utility-labels-generate-batch`
- Decision 模型：`decision-train-full`、`decision-check-model-protocol`、`decision-compare-feature-groups`
- 决策与基线分析：`decision-threshold-sweep`、`decision-compare-controller-baselines`、`decision-build-static-vbs`
- 外部评价：`decision-online-controller-evaluate`

`decision-materialize-training-data` 与 `decision-check-training-pipeline-contract` 仅保留兼容性退出提示，不会生成任何活动产物。当前仓库还没有 offline decision-state-to-terminal replay runner，因此不存在可执行的完整正式运行链。

## 目录

```text
configs/               正式实验配置
benchmarks/            BBOB、CEC benchmark 适配
optimizers/            DE、PSO、CMA-ES、SHADE 的完整状态与原生 continuation
trajectory/            轨迹字段、记录与数据质量检查
behavior/              算法无关行为特征
landscape_queries/     三档 query 规格、LHS 样本、cheap 提取和一致性检查
tools/pflacco_query/   Python 3.11 + pflacco 1.2.2 隔离提取环境
selection_reference/   逐状态候选动作损失与离线算法选择参考
utility_labels/        共享前缀配对续跑与效用标签
decision/              Decision 数据、模型、baseline 与外部评价
experiments/           正式分片采集和配置检查
docs/                  主规格、协议、结果摘要、扩展和历史归档
results/               本机生成结果；默认不提交 Git
```

三档结果按 `query_id` 隔离：`results/selection_reference/{query_id}/`、`results/utility_labels/{query_id}/` 和 `results/decision/{query_id}/`。旧 `results/ela/` 以及缺少 `query_id`、`query_protocol`、`sample_design_id` 的标签或模型不属于活动读取契约。

## 文档优先级

1. `AGENTS.md`
2. `DEVELOPMENT_DECISIONS.md`
3. `docs/00_master/`
4. `docs/10_protocols/`
5. `docs/30_results/`

`docs/archive/` 只保存研究脉络，不是当前协议或运行入口。
