# selection_reference 泛化失败的数据质量假设

## 0. 文档目的

本文基于当前 `results/decision/min_support/` 下已有诊断结果，整理 `selection_reference` 在 held-out validation family 上泛化失败的五类数据质量假设，并为每一类假设设计最小验证实验。

本文只讨论固定下游 `selection_reference` 组件的诊断问题。它不改变原始 `utility_labels`，不覆盖已有 `selection_reference`，不修改正式 phase1 配置，也不把 ELA-based selector 本身写作本文创新点。所有后续实验都应输出到独立诊断目录，并明确标注为 sensitivity 或 diagnostic result。

## 1. 当前总体判断

当前证据支持一个谨慎判断：`selection_reference` 的 validation 失配很可能与数据质量和覆盖结构有关，但不能简化为单一原因。已有 selector 消融显示，当前 `RandomForestClassifier` 在 train 上 `selected_algorithm = vbs_algorithm` 达到 `100%`，但在 validation 上只有约 `41.8%-42.0%`；限制深度的 RandomForest 没有改善；nearest-bucket smoothing 基本没有改善；SBS 和按 `remaining_budget_ratio` 的 stage-wise majority selector 可以提高 validation `selected=VBS`，但 proxy `P_ELA` 与 proxy `U_ELA` 不一定同步改善。这说明问题不是简单的“模型太复杂”或“VBS 一致率越高越好”，而是训练覆盖、bucket 映射、标签来源、行为特征表达和模型归纳偏置共同作用。

在目标 validation family 中，`bbob_f005`、`bbob_f019`、`bbob_f024` 是主要错配区域。`bbob_f005` 在多个诊断 split 中 `selected=VBS` 为 `0%`，但当前 RF 的 proxy `U_ELA` 有时反而优于 SBS；`bbob_f019` 中 SBS/stage majority 能显著提高 `selected=VBS`；`bbob_f024` 则呈现明显的 dimension 和 stage 依赖，`selected=VBS` 与 proxy utility 不完全一致。因此，后续诊断应同时报告 `selected=VBS`、`P_ELA`、`U_ELA`，不能只使用单一一致率指标。

## 2. 假设矩阵

| 编号 | 数据质量假设 | 当前证据 | 影响路径 | 最小验证实验 | 输出位置 | 判据 |
|---|---|---|---|---|---|---|
| H1 | 训练覆盖不足 | train changed rows 中 `U_ELA>0` 极少，validation changed rows 中 `U_ELA>0` 明显更多；validation 机会集中在 `FE_ratio=0.30-0.552`。 | selector 在 train 上学到的决策边界缺少目标 family/stage 的支撑，validation 上容易退化为错误算法偏好。 | 在不改变原始 labels 的前提下，构造 train-support coverage report，并用已有 late-stage extension 或 fe-transition extension 训练诊断副本。 | `results/decision/min_support/selection_reference_generalization_data_quality/coverage_support/` | 若补充覆盖后 target family 的 `selected=VBS`、proxy `P_ELA` 或真实 `U_ELA` 捕获率改善，且非目标区域误调用没有同步放大，则支持 H1。 |
| H2 | performance bucket 稀疏 | `0.25 -> 0.30` 阶段出现 `stage_025_to_030_performance_bucket_change_rate=1.0` 与 `selected_algorithm_change_rate=0.75`；0.25 映射到 `0.75/0.752` bucket 时全选 cmaes，而 0.30 映射到 `0.60` bucket 时出现 de/shade。 | 稀疏 bucket 的 nearest mapping 使相邻 stage 的 `selected_algorithm` 发生离散跳变，`P_ELA` 继而出现与真实行为状态不连续的标签变化。 | 只复用已有 selection_reference，比较 current nearest、lower bucket、upper bucket、linear/interpolated bucket 的 selected_algorithm 与 utility proxy。 | `results/decision/min_support/selection_reference_generalization_data_quality/bucket_sparsity/` | 若相邻 bucket 替换能复现或消除 f005/f024 的 selected_algorithm 跳变，则支持 H2；若平滑后 `U_ELA` 仍不改善，则说明 bucket 稀疏不是唯一原因。 |
| H3 | VBS 标签噪声 | validation labels 中 `U_ELA>0` 共 79 行，其中 same_algorithm 有 52 行但 positive utility sum 只占 `26.06%`；changed_algorithm 有 27 行但 positive utility sum 占 `73.94%`。 | `selected_algorithm == default_algorithm` 时的 `U_ELA>0` 更像共享前缀后的 continuation randomness control；若与 changed_algorithm 等权进入 selector 或 Decision 诊断，会污染“算法切换带来收益”的解释。 | 分开报告 same_algorithm 与 changed_algorithm 的 VBS 一致率、`P_ELA`、`U_ELA`，并训练只读诊断副本：full-label、changed-only、changed-weighted。 | `results/decision/min_support/selection_reference_generalization_data_quality/vbs_label_noise/` | 若 changed-only 或 changed-weighted 诊断副本在 changed_algorithm validation 上改善，而 same_algorithm 区域没有被过度解释为 selector gain，则支持 H3。 |
| H4 | ELA feature 表达不足 | f024 行为可分性诊断显示现有 behavior features 对 late-stage target holdout 的排序能力有限：RF top-20% capture `0.3077`，Logistic top-20% capture `0.3846`，best single-feature F1 `0.5946`。候选 `cf_elite_centroid_shift_norm` 曾提示存在额外算法无关信息，但 two-feature guard 在非目标区域 precision 下降。 | 当前 selector 依赖 ELA features 选择算法，而 Decision Model 只能观察算法无关 behavior features。若 ELA selector 需要的结构信息没有被 behavior features 表达，Decision Model 很难预测何时 ELA selector 会选对。 | 不改正式 extractor，先做候选算法无关 population/fitness 行为特征的离线可分性诊断；比较 existing behavior、existing+candidate、candidate-only 的 target holdout 排序和误调用成本。 | `results/decision/min_support/selection_reference_generalization_data_quality/behavior_feature_expression/` | 若 candidate features 提高 f024 target holdout 的 `U_ELA>0` capture，同时 non-holdout precision 不明显下降，则支持 H4；若训练集改善但 holdout 不改善，则更像覆盖不足或标签噪声。 |
| H5 | 模型容量与归纳偏置不匹配 | 当前 RF train `selected=VBS` 为 `100%`，validation 约 `41.8%-42.0%`；depth-limited RF 未改善；SBS/stage majority 虽提高部分 validation 一致率，但 proxy utility 不稳定。 | 高容量模型可能记住训练 family 的 bucket-feature-label 组合；低容量模型又可能只学到 SBS 或 stage majority，无法捕捉局部算法选择机会。 | 使用同一训练/validation 文件训练严格受控的诊断 selector：RF depth grid、class-balanced RF、multinomial logistic、kNN/nearest-bucket、stage-wise majority，并统一报告 target family。 | `results/decision/min_support/selection_reference_generalization_data_quality/model_capacity/` | 若所有模型在 f005/f019/f024 上呈现相似失败，支持数据覆盖或标签问题；若只有高容量 RF train-perfect 且 validation 失配，支持容量/归纳偏置问题；若低容量模型改善 `selected=VBS` 但 `U_ELA` 变差，则不能把 H5 单独定为主因。 |

## 3. H1：训练覆盖不足

训练覆盖不足是当前最强的候选解释之一。`problem_attribution_matrix.md` 已指出，在 `changed_algorithm` 条件下，train 中只有 `9 / 1040` 行满足 `U_ELA>0`，比例为 `0.87%`；validation 中则有 `245 / 1900` 行满足 `U_ELA>0`，比例为 `12.89%`。这些 validation 机会主要集中在 `FE_ratio=0.30-0.552`，而当前 train 对该区域缺少足够的效用标签支撑。

最小验证实验不应重新定义标签，也不应把 validation family 加入正式训练。建议先做只读 support map：按 `family`、`dimension`、`FE_ratio`、`default_algorithm`、`selected_algorithm`、`label_source` 统计 train 与 validation 的 row count、`U_ELA>0` count、positive utility sum、selected=VBS rate。然后用已有 `late_stage_coverage_extension` 或 `fe_transition` 扩展作为诊断训练副本，观察 target holdout seeds 上的变化。

判断标准是：若扩展覆盖后，诊断模型在 target validation family 上提高 `U_ELA>0` 捕获率或降低 `P_ELA`，并且非目标 validation 的误调用成本没有同步放大，则训练覆盖不足是主要原因之一。若扩展 train 内部可学习但 holdout 不改善，则问题更可能转向同 family 内迁移不足、bucket 稀疏或行为特征表达不足。

## 4. H2：performance bucket 稀疏

当前 `selection_reference` 的明显不连续来自 performance bucket 映射。`selector_transition_diagnostic_summary.json` 显示，`FE_ratio=0.25` 到 `0.30` 的 transition 中，performance bucket 变化率为 `1.0`，selected_algorithm 变化率为 `0.75`。`performance_bucket_sensitivity_summary.json` 进一步说明，0.25 当前映射到 `0.75/0.752` bucket，因此目标 problem/dimension 全选 cmaes；若映射到下方 `0.60` bucket，则会复现 0.30 的 de/shade selections。反过来，把 0.30 映射到上方 `0.75/0.752` bucket，会移除 selected_algorithm changes。

最小验证实验只应复用已有 selection_reference 输出，不重新训练 selector，不生成 alternate utility labels。具体做法是构造 bucket scenario table：current nearest、lower neighbor、upper neighbor、linear interpolation、majority-over-neighbors。每个 scenario 报告 selected_algorithm change rate、selected=VBS、bucket proxy `P_ELA`、bucket proxy `U_ELA`，并对 `bbob_f005`、`bbob_f024` 单独列出。

判断标准是：若 selected_algorithm 的阶段跳变可以由相邻 bucket 替换复现或消除，则 bucket 稀疏成立。若平滑能提高 selected=VBS 但不能改善 `P_ELA` 或 `U_ELA`，则该假设只能解释标签不连续，不能单独解释最终 utility 失败。

## 5. H3：VBS 标签噪声

VBS 标签噪声在这里主要指固定下游组件中的标签来源混杂，而不是说原始数据错误。`label_source_check_summary.json` 显示，在 validation 的 2400 行 label 中，`U_ELA>0` 共 79 行；其中 same_algorithm 有 52 行，positive utility sum 为 `4.2132`，占 `26.06%`；changed_algorithm 有 27 行，positive utility sum 为 `11.9531`，占 `73.94%`。same_algorithm 的 `U_ELA>0` 应解释为共享前缀后的续跑随机差异参照，而不应归因于 ELA selector 改变算法带来的收益。

最小验证实验应把 same_algorithm 与 changed_algorithm 分开，而不是删除任何原始行。可以训练三个诊断副本：full-label 使用原始权重，changed-only 只在训练损失中使用 `selected_algorithm != default_algorithm` 的行，changed-weighted 对 changed_algorithm 增加权重但保留 same_algorithm 作为参照。评估时同时报告两类 source 的 `selected=VBS`、`P_ELA`、`U_ELA` 和误调用成本。

判断标准是：若 changed-only 或 changed-weighted 在 changed_algorithm validation 上提高 `U_ELA` 捕获，而 same_algorithm 上的 apparent gain 不再主导结论，则支持 VBS 标签来源混杂假设。若三种训练方式差异很小，则标签噪声不是当前 selector 泛化失败的主导因素。

## 6. H4：ELA feature 表达不足

当前 selection_reference 是 ELA-based algorithm selection；它学习的是在给定场景下更适宜的算法。Decision Model 则不能读取 ELA features，只能读取算法无关行为。因此，若 ELA selector 的成功依赖当前 behavior features 没有表达的 landscape 信息，Decision Model 即使不过拟合，也难以稳定预测 ELA selector 何时会选对。

已有 f024 诊断给出混合证据。`f024_behavior_separability_summary.json` 中，现有 behavior features 在 f024 target holdout 上仍有一定排序能力，但不充分：RF top-20% positive capture rate 为 `0.3077`，Logistic 为 `0.3846`，best existing single-feature F1 为 `0.5946`。候选特征诊断曾提示 `cf_elite_centroid_shift_norm` 可能补充 population/fitness 形态信息，但 two-feature guard 在 non-holdout 或 extension_train 上 precision 降至约 `0.30`，说明简单规则仍可能过拟合局部 problem。

最小验证实验不应把 ELA feature、function id、algorithm id 加入 Decision 输入。建议只做候选算法无关行为特征的离线诊断：existing behavior、existing plus candidate、candidate only 三组输入，使用相同 split、相同 target rows、相同 threshold policy，报告 f024 target holdout 与 non-target validation 的 score ranking、top-k `U_ELA>0` capture、precision、proxy utility 和真实 utility。

判断标准是：若候选行为特征在 target holdout 上提升 `U_ELA>0` capture，且 non-target validation 的误调用成本没有明显增加，则支持表达不足假设。若只在训练扩展行上改善而 holdout 不改善，则该结果不能作为正式特征扩展依据，只能说明候选特征仍受覆盖不足或局部标签噪声影响。

## 7. H5：模型容量与归纳偏置不匹配

当前 RF 的训练表现具有明显 train-perfect 迹象：在 selector 消融中，current RF 和 depth-limited RF 在 train 上均达到 `selected=VBS 100%`，但 validation 只有约 `41.8%-42.0%`。不过 depth-limited RF 没有改善 validation，SBS/stage-wise majority 虽能提升 validation selected=VBS，却常伴随 proxy utility 变差。这说明“RF 过拟合训练 family”是可能原因，但不是充分解释；模型容量问题需要与标签稀疏、bucket 离散和 proxy utility 分开验证。

最小验证实验应使用相同的 train/validation labels 与相同的 selection_reference 输入，训练或复用一组容量受控 selector：RF depth grid、min_samples_leaf grid、class-balanced RF、multinomial logistic regression、kNN 或 nearest-bucket selector、stage-wise majority selector、SBS。每个 selector 统一输出 train、validation、`bbob_f005`、`bbob_f019`、`bbob_f024` 的 selected=VBS、`P_ELA`、`U_ELA`、selected_algorithm distribution 和 family-stage confusion。

判断标准是：若高容量 RF 维持 train-perfect 而 validation 明显低于低容量模型，且低容量模型同时改善 `U_ELA`，则支持模型容量/归纳偏置假设。若低容量模型只提高 selected=VBS 但 `U_ELA` 变差，则不能把 selected=VBS 作为改进证据。若所有模型在同一 family/stage 失败，则更支持 H1、H2 或 H4。

## 8. 建议执行顺序

1. 先做 H1 coverage support map，因为它不需要重训、不改变任何标签，能最快确认 train/validation 机会区域是否错位。
2. 同步做 H2 bucket scenario table，因为当前 0.25 到 0.30 的 selected_algorithm 跳变已经有直接证据。
3. 再做 H3 label-source 分层训练诊断，把 same_algorithm 作为 continuation randomness control 单独报告。
4. 若 H1-H3 仍不能解释 f024 failure，再做 H4 candidate behavior feature 诊断。
5. 最后做 H5 capacity grid，并把模型容量结论限定为在前四类数据质量条件下的剩余解释。

## 9. 证据索引

本文引用的当前项目内证据文件如下：

- `results/decision/min_support/problem_attribution_matrix.md`
- `results/decision/min_support/selection_reference_selector_ablation/selector_ablation_diagnostic.md`
- `results/decision/min_support/selector_transition_diagnostic/selector_transition_diagnostic_summary.json`
- `results/decision/min_support/performance_bucket_sensitivity/performance_bucket_sensitivity_summary.json`
- `results/decision/min_support/label_source_check/label_source_check_summary.json`
- `results/decision/min_support/f024_behavior_separability/f024_behavior_separability_summary.json`
- `docs/20_extensions/min_support_late_stage_coverage_extension_plan.md`

## 10. 写作边界

在论文或项目主文档中，应把上述内容表述为固定下游 `selection_reference` 的泛化风险与数据质量诊断，而不是新的算法选择方法。当前证据允许写成：“selection_reference 的失配可能由训练覆盖不足、performance bucket 稀疏、label-source 混杂、behavior feature 表达不足和模型容量共同造成；因此本文把该组件作为固定下游参考，并对其风险进行分层诊断。”不能写成已证明某一个原因单独导致全部失败。
