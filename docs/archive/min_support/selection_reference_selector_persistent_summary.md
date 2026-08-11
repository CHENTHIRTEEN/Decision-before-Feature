# selection_reference selector 持久化汇总（历史）

## 1. 文档目的

本文持久化当前项目中与 `selection_reference` selector 相关的讨论、诊断和文献定位。它只整理当前项目内已经形成的结论，不修改原始 `utility_labels`、`selection_reference`、正式 feature extractor 或 phase1 配置。

核心定位：

> `selection_reference` 是固定下游 ELA-based algorithm selection 组件，用于构造 offline ELA utility labels；它不是本文创新点。

本文创新点仍然是：

    search behavior -> decide whether to execute ELA

而不是：

    ELA features -> propose a new algorithm selector

## 2. 当前 selector 的实现口径

当前 `selection_reference` 对应 direct supervised classifier：

    ELA features -> RandomForestClassifier -> selected_algorithm

其决策目标是学习当前问题场景下的适宜算法，即预测 `selected_algorithm` 或 problem-stage `vbs_algorithm`。它与 `U_ELA` 感知器的相似点是二者都依赖算法行为或下游算法表现形成监督信号；区别是：

- `selection_reference` 在执行 ELA 后，用 ELA features 决定选择哪个优化算法；
- Decision-before-Feature controller 在执行 ELA 前，用低成本算法无关 search behavior 决定是否值得执行 ELA；
- 因此，`selection_reference` 是下游参考组件，Decision model 才是本文研究对象。

## 3. U_ELA>0 后的算法选择解释

`U_ELA>0` 表示：

$$
U_{ELA}=(P_{skip}-P_{ELA})-\lambda C_{ELA}>0
$$

也就是说，在已有 label 构造口径下，执行 ELA 并调用 `selection_reference` 的路径相对 skip-ELA 路径有正效用。

需要注意：

- `U_ELA>0` 不等于 selector 一定切换了算法；
- same_algorithm 行中也可能出现 `U_ELA>0`，这类行更像共享前缀后的 continuation comparison，不应解释为“ELA selector 切换算法带来收益”；
- changed_algorithm 行才更接近“执行 ELA 后 selector 选择了不同优化算法”的效用来源；
- 因此报告 selector 质量时必须同时给出 `same_algorithm` 与 `changed_algorithm` 分层。

## 4. H1-H5 诊断结论

当前 H1-H5 总结为：

| 假设 | 结论强度 | 主要结论 |
|---|---|---|
| H1 训练覆盖不足 | 强 | validation changed_algorithm `U_ELA>0` rate 高于 train，说明 held-out family 中机会区域缺少训练支撑。 |
| H2 bucket 稀疏 | 强 | 相邻 performance bucket 可复现或消除 selected_algorithm 阶段跳变，说明 nearest-bucket 映射存在离散敏感性。 |
| H3 VBS 标签来源混杂 | 中等 | same_algorithm 与 changed_algorithm 的 `U_ELA>0` 来源不同，应分层解释，不能把 same_algorithm 的 gain 写成算法切换收益。 |
| H4 behavior feature 表达不足 | 中等偏弱 | candidate behavior feature 对 f024 target holdout 有排序信号，但 threshold policy 在 holdout 上未稳定捕获正效用。 |
| H5 模型容量与归纳偏置 | 中等 | RF train-perfect 与 validation mismatch 存在，但低容量模型提升 `selected=VBS` 不必然提升 `U_ELA`。 |

关键数值已经汇总在：

- `results/decision/min_support/selection_reference_generalization_data_quality/selection_reference_generalization_h1_h5_total_report.md`

由于 `results/` 目录按项目规则不进入 git，稳定文档入口为：

- `docs/20_extensions/selection_reference_generalization_data_quality_hypotheses.md`
- `docs/20_extensions/selection_reference_selector_persistent_summary.md`

## 5. H5 对 selector 过拟合判断的约束

当前 H5 说明：

- 当前 RF 与 `rf_depth_none_leaf_1` 在 train 上 `selected=VBS=1.000000`，validation 为 `0.418750`；
- `rf_depth_8_leaf_5` 将 validation `selected=VBS` 提高到 `0.518750`，但 precision 仅 `0.117647`，proxy `U_ELA=-0.019384`；
- stage-wise majority 的 validation `selected=VBS=0.581250`，但 `U_ELA>0` capture 为 `0.000000`；
- logistic 在 `bbob_f024` 上 `selected=VBS=0.687500`，但 validation overall proxy `U_ELA=-0.111535`。

因此：

> 可以说当前 RF 有 train-perfect 和 held-out mismatch 迹象；不能说泛化失败只由 RF 过拟合造成。数据覆盖、bucket 稀疏和 label-source 混杂是更强的数据质量解释。

## 6. ELA-based selector 文献谱系

文献中根据 ELA 做 selector 的方法至少包括：

| 类别 | 代表工作 | 与当前项目关系 |
|---|---|---|
| cost-sensitive classification | Bischl et al. (2012) | 早期直接出处，使用 ELA features 与 cost-sensitive learning。 |
| direct classification | Kerschke and Trautmann (2019) | 当前 `RandomForestClassifier` selector 对应这一类。 |
| performance regression | Kerschke and Trautmann (2019); Jankovic and Doerr (2020); Jankovic et al. (2021) | 预测每个算法性能，再选预测最优算法；更贴近 $P_{ELA}$ 和 $U_{ELA}$。 |
| pairwise regression | Kerschke and Trautmann (2019) | 对算法对预测 performance difference，再聚合选择。 |
| algorithm configuration | Belkhir et al. (2017); Prager et al. (2020) | 用 ELA 选择同一算法框架下的配置或模块，不是当前 portfolio selection 主线。 |
| Deep-ELA / learned representation | van Stein et al. (2023); Seiler et al. (2025) | 用深度表示替代或补充手工 ELA features，属于未来扩展方向。 |
| MO-ELA | Preuss et al. (2026) | 面向多目标优化，当前项目只作为 related work 边界。 |
| benchmarking risks | Tanabe (2022); Kerschke and Trautmann (2019); Jankovic et al. (2021) | 支持把 selector 质量作为数据条件和评价风险，而非无条件假设稳定泛化。 |

详细分类和 DOI/arXiv 入口已经写入：

- `docs/10_protocols/Decision-before-Feature Algorithm Portfolio与Selection Reference设计.md`

## 7. 论文可写结论与内部诊断边界

可以写入论文的结论：

1. 当前 `selection_reference` 属于既有 ELA-based per-instance algorithm selection 范式。
2. 当前项目采用 direct supervised classifier 作为固定下游组件，不把 selector 作为本文创新点。
3. held-out family 上存在 selector 泛化风险，且 H1-H5 显示风险与训练覆盖、bucket 稀疏、label-source 混杂和模型归纳偏置有关。
4. `selected=VBS` 不能单独代表 utility 改善，必须同时报告 `P_ELA`、`U_ELA`、`U_ELA>0` capture 和 precision。

应保留为内部诊断或 appendix sensitivity 的内容：

1. H4 candidate behavior feature 暂不能作为正式 feature extractor 修改依据；
2. RF depth/min_samples_leaf grid 暂不能作为正式替代 selector；
3. stage-wise majority 虽提高 validation `selected=VBS`，但不能写成更优 selector；
4. H2 neighboring-bucket utility 只是 proxy，不是新生成 utility labels；
5. changed-only 训练副本只能说明 label-source 敏感性，不是正式组件。

## 8. 当前 git 持久化范围

应纳入 git 的 selector 相关内容：

- selection reference 文档和文献谱系；
- H1-H5 数据质量假设文档；
- H1/H2、H3、H4、H5 的最小诊断脚本；
- 与 selector 文献和定位相关的 master 文档更新。

不应纳入 git 的内容：

- `results/` 下的 parquet/json/md 诊断输出，除非后续明确决定用小型 Markdown 报告作为论文附录材料；
- 原始 phase1 trajectories、utility labels、selection reference parquet；
- checkpoint 或大规模中间数据。

## 9. 后续建议

下一步建议是将本文件压缩成论文正文的 related work 与 limitation 两处文字：

- Related Work: ELA-based algorithm selection 方法谱系；
- Methods / Limitations: fixed downstream `selection_reference` 的角色与泛化风险；
- Appendix: H1-H5 sensitivity and data-quality diagnostics。
