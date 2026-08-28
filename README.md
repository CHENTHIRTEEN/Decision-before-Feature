# Decision-before-Feature

本项目研究一个前置于特征计算的资源决策问题：在黑盒优化过程中，是否值得为当前搜索状态执行一项预先定义的 landscape-analysis query。项目不设计新的优化算法，而是用离线轨迹和监督学习估计固定 query 的状态依赖效用。

## 当前状态

截至 2026-08-16，仓库中的核心实现已经比上一版交接记录更完整。四种优化器已经支持完整状态推进与 checkpoint 恢复；Behavior extractor 已经切换为 permutation-invariant 的种群集合统计；`phase1_dynamic_budget_event_v1` 的动态采样、`final_performance.parquet` 的逐 run 终值表、Query/Selection Reference/Utility/Decision 的主链路、以及在线控制器评估入口都已经在源码中落地。

更具体地说：

- `optimizers/` 已支持 native continuation 与跨算法 `population_transfer_initialization`；
- `trajectory/` 已支持最终性能表、window statistics、事件采样验证和 failure endpoint 记录；
- `behavior/` 已支持 permutation-invariant window 特征与多层 feature group；
- `selection_reference/` 已支持 query-full / state-only / query-only / behavior-only-full-budget / pre-run AAS 的 selector 构建与持久化；
- `utility_labels/` 已支持 Query-joint、Behavior-only、query-adjusted state-only、sampling-only 和完整 three-repeat timing replay 的 paired Utility 生成；
- `decision/` 已支持三候选模型、cluster-balanced fit、nested CV-group OOF threshold 与 replay plan 物化；
- `benchmarks/cec.py`、`benchmarks/factory.py` 已支持 `cec2017` 与 `cec2022`；
- `decision/online_controller_evaluate.py` 已能执行在线政策评估、Stage-A/Stage-B timing replay、matched-rate Random 与 baseline 汇总。

这些实现意味着项目已经“可以跑”，但还不意味着“可以正式结束实验”。当前仍缺少或尚未闭合的部分主要是：offline decision-state-to-terminal runner 的正式核对、BBOB-validation 的完整内部评价链路、72 个正式 BBOB trajectory shard 的生成与验收、ERT suite-level consumer、工程问题 factory/constraint/config，以及对 CEC2017 F2/F30 与外部确认集重复数的最终é¢åæå®。

如果你要开始数据生成，现在更适合先做的是：核对配置、确认 replay plan、检查产物目录、然后分阶段启动正式运行，而不是直接一口气跑完整协议。

Behavior extractor 同时已改为 permutation-invariant 的种群集合统计：跨窗口的空间变化使用经验 Wasserstein、centroid shift 和协方差谱集中度，fitness 变化使用经验分位数分布；不再把 population 行号解释为跨代个体身份。运行时逐次记录完整原生 update 的轻量窗口统计，正式 behavior state 的 w02/w05/w10 anchor 不再从稀疏输出状态中选择；若名义 FE 不能整除一次原生 update，则取不晚于目标位置的最近完整 update，误差严格小于一次 update，并保存 `effective_window_ratio_*`、`effective_window_fe_*` 与 `effective_native_updates_*`。所有 rate/slope 使用实际 `ΔFE/FE_total`，这些窗口字段只作 metadata，不进入 Decision 输入。

正式状态采样已é¢åæå®为 `phase1_dynamic_budget_event_v1`：在 `0.20–0.60` 上按 `0.01` 候选网格监测，保留 12 个预定义预算里程碑，并依据 improvement resume、stagnation onset、effective-rank change、elite migration 与 diversity recovery 补充事件状态；每个跨过至少一个 0.01 监测网格的完整原生 update 只判定一次事件。同一 update 跨过多个监测点时，若包含预算里程碑，则里程碑与事件合并为一行，且该行不消耗 event-only 配额、最小间隔锚点或 `event_index_in_phase`；若不含里程碑，则以最新跨过的监测点作为名义节点。每个 run 输出 12–18 个状态。`FE_ratio` 始终是实际 `FE/FE_total`，名义里程碑另存 `budget_milestone_ratio`，状态连接使用整数 `FE` 而非浮点 ratio。完整预算终值另存为每个 `problem_id × algorithm × seed` 在 `FE=FE_total` 恰好一行的 `final_performance.parquet`；该表与 `0.20–0.60` decision trajectory 分离，不能把 `0.60` 的最后一个 decision state 当作完整预算终值。

唯一活动协议把 Selection Reference 定义为逐共享状态候选动作选择组件：每个 state 对 `continue_current` 和其余三个 portfolio algorithm 分别进行真实 continuation，`remaining_budget_ratio` 作为连续输入；不再按静态 problem label 和 nearest performance bucket 选择算法。当前正式下游 Selector 为 `dimension_aware_hybrid_selector`：10D/20D 使用多输出 Random Forest，40D 使用六个 one-vs-one RF classifier 组成的 `pairwise_aggregation_rf_classifier`。当前标签链只读取 Query-adjusted 四动作矩阵，主目标变换是相对 `continue_current` 的 `clipped_log10_gap_advantage_vs_continue_current`；旧 `statewise_minmax_observed_action_loss` 不得生成主 selected action 或 Decision labels。主 Decision 标签字段为 `g_fe_selected_path`，只由 Stage-A 的 `skip` 与 query-full selected path 科学端点生成等总 FE 功效；`g_fe` 仅作最佳已观测动作诊断。Behavior-only、pre-run AAS、五路径分解与完整路径计时均延后，不是当前训练前置条件。

三档 query 提取器已实现统一 `unit_cube_x__median_iqr_y_v1` 前处理：cheap/standard 共享 `lhs_50d`，broad 使用 `lhs_100d`；隔离 pflacco 1.2.2 提取器对预处理后的 X/y 计算é¢åæå® whitelist，终点评价字段只作 metadata。活动 query ID 为 `descriptor_cheap_invariant`、`pflacco_standard_invariant`、`pflacco_broad_invariant`。实现完成不等于正式数据检查或结果完成；72 个正式 trajectory shards 尚未启动。

Decision Model 的活动候选固定为 LDA、Logistic Regression 与 Ridge。每个 outer fold 单独计算 `SBS_outer` 并拟合该 fold 的 Selectors/Utility/Decision；每个 inner fold 又只用 inner-fit functions 重算 `SBS_inner`、Selectors 与 Utility，端到端 inner OOF first-trigger outcomes é¢åæå® outer threshold。完整 BBOB-train 的部署 threshold 与 Random calibration 也必须来自 grouped-by-function OOF 上游链。BBOB-validation 已被旧模型比较、调参与消融查看，只能作已见内部评价集；CEC2017 也已有 preliminary/targeted 结果，只能作已见外部开发集。Random Forest、XGBoost、LightGBM、MLP 及分类特征工程搜索已退出 Decision Model 活动调参路径；Selection Reference 中的 `dimension_aware_hybrid_selector`、旧多输出 RF 基线与 pairwise aggregation sensitivity 属于不同组件，不受此约束。

此前生成的 BBOB trajectory 使用了重建式 continuation，behavior 含有依赖行号对应关系的字段，landscape 表又把 16 个自定义描述符笼统称为 ELA。当前活动 utility labels、Decision dataset、模型、baseline 和成本—性能结果需要从 trajectory 开始按依赖顺序重新生成；已有 CEC2017 在线结果不作为外部结论。

历史产物已从活动结果路径中删除。当前不存在可直接复用的正式模型、checkpoint 或论文结果表。

当前结果的完整口径见 [docs/30_results/phase1_current_results.md](docs/30_results/phase1_current_results.md)，跨对话状态见 [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md)。

## é¢åæå®实验协议

- BBOB、CEC 与 MA-BBOB 同为正式实验基准函数，不得区别对待（2026-08-21 裁决）：正式采集规格统一为 seeds 1–5、`FE_total = 1000 × dimension`、population 40、`boundary_handling = reflect`（clip 仅限显式声明的敏感性分析）、同一 endpoint/timeout/floor-cap 常数与校验器；suite 间只允许数据角色与问题结构（维度集合、instances）的设计差异。`configs/phase1_cec2017_test.yaml` 的维度按其 suite é¢åæå®为 10/30/50D。
- 训练：BBOB 10D / 20D / 40D。
- 已见内部评价集：BBOB-validation 10D / 20D / 40D，按 function ID 与 BBOB-train 隔离。
- 已见外部开发集：CEC2017。其函数口径仍待核对。
- CEC2022 与工程问题在首次生成 outcome 前需补齐并é¢åæå® functions/problems、预算、端点、失败、约束规则、顶层有限集单位与固定 strata 权重；CEC suite 以 function 为顶层有限集单位，工程集合以预先命名的 engineering problem 为顶层单位，不强行套用 BBOB function/dimension 层级。
- 算法池：DE、PSO、CMA-ES、SHADE。
- 主采样协议：`phase1_dynamic_budget_event_v1`；12 个必选预算里程碑为 `0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.60`，事件状态使每个 run 总计 12–18 行。
- 主 query 固定为 `descriptor_cheap_invariant`：14 个自定义低成本描述符，使用 `lhs_50d`，即 5% 总 FE。统一 median/IQR preprocessing 后恒为 0/1 的 `descriptor_y_median`、`descriptor_y_iqr` 已删除；query ID、采样和 action losses 保持不变。
- `pflacco_standard_invariant`（37 维，`lhs_50d`）与 `pflacco_broad_invariant`（52 维，`lhs_100d`，10% 总 FE）用于当前固定的配置稳健性实验；主 query 保持不变。
- Decision 输入仅来自 permutation-invariant 的算法无关搜索行为；function、dimension、algorithm、query feature 和优化器内部状态作为 metadata 或分层报告字段。
- 当前成熟度消融固定为三种单字段 Decision 形式：`B2+Motion+SearchMaturity`、`B2+Motion+SearchMaturityLinear`、`B2+Motion+ExploreExploitRatio`，每组 29 个 `bf_*` 行为特征；下游 Selector 固定使用不含成熟度的 28 个 B2+Motion 行为特征。Decision 候选固定为 RF 回归器，形式由 BBOB-train nested function-family OOF first-trigger mean `g_fe_selected_path` 决定。
- Decision Model 活动候选严格为 LDA、Logistic Regression、Ridge；主选择指标为 grouped-by-function outer OOF 的 run-level first-trigger mean joint utility。train outer OOF 只承担候选选择和开发期诊断，不能在选择后继续充当 selected procedure 的无偏性能估计；BBOB-validation 与 CEC2017 只作为已见开发集。所选 procedure 与 RQ2 的 B3--T0 目前没有独立确认性结果，只有未来按é¢åæå®协议首次运行的 CEC2022/工程集合可提供相应证据。AUROC、Average Precision、Spearman 为逐状态辅助指标，连续 Utility RMSE 只对 Ridge 定义。
- 一个 run 最多执行一次 query。阈值、模型比较、call rate、precision、utility capture 与 policy utility 全部按最早越阈值状态计算；首次调用后的后续状态不可达，不进入该 run 的策略效用。
- policy utility capture 对所有策略共享 native SBS/default trajectory 上全部预定义机会的 run-level hindsight maximum；策略分子只取其 first-trigger state，未触发为 0。该参照不随策略触发时点改变，也不是可部署 policy。
- BBOB-validation 不参与 SBS、Selector、preprocessing、模型、候选选择或 threshold 拟合；部署阈值模式固定为 `oof_g_fe_selected_path_first_trigger`。
- 当前主标签是严格等总 FE 的 `g_fe_selected_path = log((E_skip + epsilon_p)/(E_query_selected + epsilon_p))`；只使用 Query-adjusted Stage-A 科学端点，不把 wall-clock/runtime 作为标签组成项。`g_fe`（最佳已观测动作）仅作诊断。Behavior-only、pre-run AAS、matched-acquisition 与五路径分解当前均不生成，也不进入 Decision dataset、阈值拟合或模型选择。
- runtime 相关标签在当前阶段停用。完整路径 replay runner 仅保留为后续部署资源评价入口；任何 runtime、Pareto 或收敛时间比较若未来重新启用，仍必须在线完整路径实测，不能用 component runtime 拼接。
- `benchmark_reference_value` 和所有 gap 字段只用于离线标签和最终评价，不进入 Behavior、Selection Reference 输入或 Decision X；使用已知最优值计算离线标签并不意味着在线优化器知道最优值。
- 第一篇论文主 probe/default 固定为 fold-specific SBS。SBS 使用相应 fit functions 的完整预算 `log10_gap`，按 run → static problem（function × dimension × instance）→ fixed dimension stratum → function 等权聚合，选择均值最低的算法；并列按 `de,pso,cmaes,shade`。这与主性能端点和 function 顶层权重一致，不再用平均 rank 丢弃效应量。No-query 原生继续该 SBS 的完整 checkpoint state。
- Query 后选择当前 prefix 时原生继续；选择其他算法时采用一次 checkpoint population transfer；query 采样点不并入后续优化 population。
- 多 prefix 行单独用于 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 泛化，不进入主 Decision 数据。
- 标签显式保存 `selected_equals_default`、`selected_equals_prefix`、`handoff_required` 和 `skip_switches_from_prefix`，不再生成含义模糊的 selected-vs-default 字符串分层。
- `no_query_algorithm` 显式保存 No-query 分支算法并等于 `default_algorithm`；`handoff_type` 显式保存 Query-selected action 的 transition 类型并等于 `query_transition_mode`；`handoff_required` 等价于 `handoff_type == population_transfer_initialization`。
- 逐状态最小 action loss 称为 `best observed action`，只用于潜在性能差与 selector regret 诊断，不称为 oracle，也不进入 Decision 输入。
- 静态 VBS 在每个 `function × instance × dimension` problem 内先对每个算法的完整预算 clipped `log10_gap` 跨 optimizer seeds 取均值，选择均值最低算法（并列按 `de,pso,cmaes,shade`），再用该算法的逐 seed paired outcomes 汇总；不得逐 seed 选择最小算法，也不得用逐状态 `best observed action` 替代。
- Baseline 额外固定为：`matched_rate_random` 只用 BBOB-train 端到端 OOF Proposed é¢åæå® run-level 调用率与 trigger-FE 经验分布，每 run 预抽目标 ratio 并在第一个不早于目标的在线机会触发；30 个 Random streams 的 outcomes 先在同一 run 内平均，再进入 problem/function 聚合，不能当作 30 个独立 runs。`pre_run_aas_fe0` 是 FE=0、query-only、sample-isolated 的 pre-run AAS：仅用 query features 选择初始算法，query sample 不初始化 optimizer population；关系记账为 `prefix_algorithm=selected_algorithm`、`selected_equals_prefix=true`、`handoff_required=false`、`handoff_type=fresh_optimizer_initialization`，而 `default_algorithm=no_query_algorithm=SBS_fold`。fresh initialization 只记到 `runtime_fresh_initialization`，`runtime_handoff` 保留给已有 population 的 transfer initialization。`Always Query` 在首个在线机会调用，三者不得混称。该 baseline 及所有 portfolio 结论仅适用于仓内é¢åæå®实现、参数与 `population_size=40`。
- BBOB train/validation 与 CEC2017 固定 failure cap `1e20`、取 `log10` 前 raw-gap floor/cap `1e-12/1e20`、success target `1e-8`、`action_timeout_seconds=3600`、`timing_replay_timeout_seconds=3600`、`policy_timeout_seconds=3600` 与逐 objective evaluation first hit；三种 timeout 分别约束 Stage-A action continuation、Stage-B decision-state timing replay 和 FE=0 full-policy path，不得混用。Stage-A timeout/failed path 的 final-gap endpoint 仍按失败 cap 保留，但若失败前已经达到 target，`target_hit_observed=true` 且 ERT 使用该 observed first hit；`endpoint_success=false` 继续明确完整路径没有完成。未达到 target 的 ERT 项计完整 planned budget；effective FE 逐行保留。query sample 不进入 population，但 sample best/first hit 进入 operational Query Stage-A endpoint与 ERT；另报 continuation-only gap和 sample-best contribution。Stage-B timeout/completion 只进入计时状态、不稳定性与失败敏感性，不改写 Stage-A gap 或 path completion。
- 正式结果保留所有计划运行和 query failure。Decision score 缺失时该机会按 No-query 处理；query 已触发后若特征或 selector 失败，query FE 与时间仍计入，并按预设 fallback 继续当前算法；不得删除失败行后重新计算调用率。每个 suite/endpoint 同时报 attempted denominator、complete-pair 与双向极端 failure sensitivity：gap 用 floor/cap；`target_hit_observed` 用 1/0；ERT 的 adverse 未命中项计完整 planned budget，favorable 命中项使用在已知 prefix、已消耗 query FE 与路径时间原点下最早可行的 objective-evaluation index；runtime adverse 用 timeout、favorable 用该 suite complete pairs 的最小正 runtime。若最早可行 hit FE 或最小正 runtime 无法由正式行重建，相应 sensitivity 为 undefined，结论未建立。Utility 由同一组极端端点重算。方向、区间相对 operational tolerance 的位置任一改变或 coverage<95% 时，结论未建立；已按 cap 保留的科学 path failure 不当作 missing pair。
- 单个 problem-keyed LHS 使主结果成为“固定 query realization 下的条件 estimand”，不能泛化到 sampling randomness。多个 LHS 只有在查看结果前加入并纳入配对/层级分析，才能作为 sampling-robustness sensitivity；standard 与 broad 同时改变 representation、sample size、sample realization 和预算，差异不能拆分归因。
- RQ1 的目标分布统一按 state → run → static problem → fixed dimension stratum → function 聚合；policy endpoint 从 run 层开始。活动协议把 function/dimension/static-problem/run-balanced 权重升为主拟合，run 权重再均分给其 states；现有 estimator wiring 尚未实现该权重，因而仍是正式运行 blocker。旧 unweighted row fit 只可作敏感性，nested first-trigger evaluation 本身不会修正训练权重。
- BBOB-validation 的可报告 estimand（2026-08-21 扩充）采用两层 50/50 预指定组成：0.5 ×（F5/F9/F13/F14/F19/F24 六函数等权子均值，内部口径与旧 estimand 同构、乘 2 可还原）+ 0.5 ×（18 个 `mabbob_validation` 定义的等权子均值）。`mabbob_validation` 为 evaluation-only 集合：components 仅含六个 validation 函数、dense 权重支持集不越界，10/20/40D、instance 1、seeds 1–5、reflect；不得进入任何 fitting split。主区间固定保留两层全部 functions/definitions、全部 dimensions 与 static problems，仅在每个固定 static problem 内配对重抽 optimizer seeds；不重抽 function/definition 或层组成。重抽 function 只作函数组成敏感性，不进入主 CI，也不产生 function 或 transformed-instance 超总体推断。当前 3 instances × 5 optimizer seeds（2026-08-21 由 30 seeds 下调，属开发期采样设计变更，非精度驱动）没有仓内精度依据，不得声称功效充分；主 CI 的 seed 层配对 bootstrap 在每个 static problem 内仅有 5 个重抽单元，区间会相应变宽。CEC2022/工程问题的重复数须在查看其 outcome 前另行给出精度目标并é¢åæå®。
- Utility ±0.01、`log10_gap` ±0.05、runtime ratio [0.95,1.05]、call/target-hit rate ±0.05 仅称“项目内预设 operational tolerance”。主条件区间和预设 family 的描述性 simultaneous intervals 只说明相对这些边界的位置，不能把任意项目边界升级为领域通用等价界；若要评价 endpoint-success rate，必须单独预设其边界，不能复用 target-hit rate 的名称。
- RQ2 的主要科学 contrast 是é¢åæå®模型家族的 milestone-only B3--T0；三候选两两 outer-OOF 比较只作选模诊断。六个固定且已见 functions 不是函数总体的随机样本，双侧 sign-flip 还额外要求 function-effect signs 可交换；其 raw/adjusted p 只可作为假设敏感的辅助描述。RQ3 与 RQ5 各有六个预设辅助 contrasts，在六函数下最小 Holm-adjusted p 均为 0.1875，数学上不可能在 0.05 下拒绝。RQ4 按 suite 与 endpoint 分别给有限集估计，不把四个 suites 冒充一个四 contrast Holm family。三者都以有限集效应量、逐 function/problem 结果和条件区间为主。
- 当前证据包最多支持“是否调用一个固定 query 是可学习的资源决策”这一条件性主张。若三档 query、完整在线政策和未查看外部集合未闭合，不得扩张为“任意 ELA/landscape analysis 都应被优化”或跨表示的普遍结论。

## 计算规模与运行前停止条件

BBOB train + validation 的基础轨迹共有 4,320 runs（5 seeds）、至少 51,840 mandatory states 与 0.1008B 基础 FE。12 个必选 milestones 的平均 prefix ratio 是 0.35，因此 mandatory-only future path 平均为 0.65B；旧“所有 states 均位于 0.60B”的成本口径系统性低估 1.625 倍。只含 mandatory states、跨 Stage-A matrices 共享且不复用基础 trajectory 时，Stage A 为 main cheap **6.244602B FE**、三档 **8.905806B FE**；若逐行证明基础 trajectory 终值完全同义，可降为 **5.458362B / 8.119566B FE**。当前 main producer 仍分别执行两个额外 Skip，Stage A 为 **7.817082B FE**；三档当前实现量等待实际调用图枚举。Query sample artifact 仍按 `problem × sample_design` 生成一次，不能按 state/prefix 重复计入。

Utility 必须两阶段生成：阶段 A 把 action matrices 和 FE=0 outcomes 各运行一次，以这些预指定运行固定科学 gap/observed first hit/path completion/planned/effective FE 与失败状态并拟合 fold-specific Selectors；Selector é¢åæå®并枚举 5 outer × 4 inner 加 full-train→已见 BBOB-validation roles 后，阶段 B 才对实际 selected decision-state future paths真实重跑三次。按当前 fold 覆盖推导，train state 有 22 个 replay roles，BBOB-validation state 有 1 个。仅 12 个 mandatory milestones 时，跨 matrices 共享但不复用基础 trajectory 的总量为 main cheap **35.951622B FE**、三档 **58.367106B FE**；进一步复用基础 trajectory 时为 **35.165382B / 57.580866B FE**；保持当前 main producer 时主 query 为 **37.524102B FE**。event-only states、完整在线政策、失败、外部 suites 和额外 query replicates 全部未计入，所以这些只是 mandatory-only 算术情景，不是严格下界或资源承诺。

Replay planner 已有枚举能力，但当前没有 offline decision-state-to-terminal runner、物化实测的 fold-role-complete replay plan、fold-role→Selector artifact 路由、真实 evaluator timing 或已确认资源排期；Stage-A Skip 共享/复用也未兑现。当前 CEC2017 online evaluator 的 7 条固定政策加 30 条 Random、每条 1 次科学运行加 3 次 timing replay，按现配置（5 seeds）另需约 **1.9314B planned FE**，尚未并入前述 BBOB replay 总量。它只支持 CEC、固定 `instance=1`；若覆盖 BBOB-validation 全部 instances，同样政策集合约需 **0.9324B FE**，但当前实现不可执行。`configs/phase1_cec2017_test.yaml` 仍列 F1--F29（包含 F2、排除 F30），项目内尚无依据证明这符合所用实现与官方口径；必须在查看 policy outcomes 前核对é¢åæå®。CEC2022/工程问题还缺 functions/problems、预算、reference、gap floor/cap、success target、三种 timeout、first-hit 与 constraint rule。上述均为正式运行 blocker。

## 正式入口

正式配置只有：

- `configs/phase1_train.yaml`：整合训练集（BBOB-train 18 函数 + `mabbob_formal` 24 定义）
- `configs/phase1_validation.yaml`：整合验证集（BBOB-validation 6 函数 + `mabbob_validation` 18 定义，evaluation-only，两层 50/50 estimand）
- `configs/phase1_cec2017_test.yaml`：已见外部开发集（单 suite）

整合配置内嵌 `suites` 段：顶层 `dataset` 是逻辑 train/validation 标签，公共规格（维度/seeds/FE/边界/endpoint 常数）对全部 suite 强制一致，suite 段只能覆盖 suite/存储 `split`/functions/instances/family protocol/manifest 路径/output；磁盘分片布局与下游读取契约（`bbob_train`/`mabbob_formal` 等存储 split 目录名）不变。采集入口、正式数据质量检查、`selection-reference-evaluate-actions` 与 `selection-reference-check` 均已支持整合配置；`selection-reference-build` 直接读取显式 action-loss、Behavior 与 query-feature 产物路径。仍需适配整合配置的消费者是 `utility_labels/*` 与 `decision/*`。MA-BBOB manifest 是确定性协议产物，位于 `results/mabbob_diversity_pilot/`，results 清空后由 generate/select CLI 重建。

主要命令可通过 `uv run <command> --help` 查看参数：

Selection Reference、Utility、Decision、baseline 与外部评价入口均显式区分 `query_id`。Decision 分析命令使用 `--query-id` 从 `results/decision/{query_id}/` 推导默认输入和输出；显式传入的 artifact 仍必须通过 query 协议核对。当前主标签是等总 FE 的 `g_fe_selected_path`，`g_fe` 仅作最佳已观测动作诊断，runtime 只作为独立资源维度报告。

- 数据采集：`phase1-plan-shards`、`phase1-collect-batch`、`phase1-check-config`、`phase1-check-trajectory-shards`
- Query 与标签：`query-sample-batch`、`query-extract-cheap`、隔离的 `tools/pflacco_query/extract.py`、`query-consistency`、`selection-reference-check`、`selection-reference-evaluate-actions`、`selection-reference-build`、`utility-labels-generate-batch`
- Decision 模型：`decision-train-full`（直接消费四 split query-adjusted action-loss，生成 `g_fe_selected_path`、Decision dataset、RF 回归训练摘要与在线评价输入）、`decision-check-model-protocol`、`decision-compare-feature-groups`
- 决策与基线分析：`decision-threshold-sweep`、`decision-compare-controller-baselines`、`decision-build-static-vbs`
- 外部评价：`decision-online-controller-evaluate`

`decision-materialize-training-data` 与 `decision-check-training-pipeline-contract` 仅保留兼容性退出提示，不会生成任何活动产物。当前仓库还没有已核对的 offline decision-state-to-terminal replay runner，因此完整正式运行链仍不能直接端到端执行。

`selection-reference-build` 是下游 Selector 的唯一正式训练与评价入口。默认在 `results/selection_reference/{query_id}/` 生成 hybrid 主表 `selection_reference.parquet`、全维度 pairwise 敏感性表 `pairwise_aggregation_sensitivity.parquet`、旧多输出 RF 基线表 `formal_multioutput_rf_baseline.parquet`、汇总表 `selector_evaluation_summary.parquet` 和同时包含两类组件的 `statewise_selector.joblib`。旧模型文件仍可读取，但新的 Utility、Decision dataset 与在线评价输入必须从新主表重新生成。

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
docs/                  主规格、协议、结果摘要与扩展
results/               本机生成结果；默认不提交 Git
```

三档结果按 `query_id` 隔离：`results/selection_reference/{query_id}/`、`results/utility_labels/{query_id}/` 和 `results/decision/{query_id}/`。旧 `results/ela/` 以及缺少 `query_id`、`query_protocol`、`sample_design_id` 的标签或模型不属于活动读取契约。

## 文档优先级

1. `AGENTS.md`
2. `DEVELOPMENT_DECISIONS.md`
3. `docs/00_master/`
4. `docs/10_protocols/`
5. `docs/30_results/`
