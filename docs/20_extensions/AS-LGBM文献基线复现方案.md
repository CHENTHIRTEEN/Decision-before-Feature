# AS-LGBM 文献基线复现方案

## 资料与任务边界

本实验依据用户明确指定的两份资料：

1. 论文：`Guo 等 - 2025 - Automated algorithm selection for black-box optimization using light gradient boosting machine.pdf`；
2. 代码仓库：<https://github.com/HandingWangXDGroup/AS-LGBM.git>。

论文和仓库中的说明属于复现资料，不改变 Decision-before-Feature 的主实验协议。本实现是独立文献基线，不进入主 Decision Model 候选，不修改主 Selector 的四算法组合，也不写入主 Decision dataset。

## 已核对的算法流程

AS-LGBM 的静态输入—输出关系为：

```text
问题实例
  -> 500 个 LHS 样本
  -> 61 个低成本 ELA 特征
  -> 10 个算法各运行 30 次
  -> Soft-ERT
  -> 选择 Soft-ERT 最小的算法作为分类标签
  -> 多分类 LightGBM
```

仓库 `Benchmarks.py` 中的算法顺序为 `ABC, ACO, CMA-ES, CSO, DE, FEP, GA, PSO, SA, RAND`。失败运行以 `-1` 记录；Soft-ERT 计算时每个失败运行加入 `10001`，分母为未失败运行数与 1 的较大值。生成标签后，仓库将失败运行值替换为 `10000`，并用预测算法和标签算法的 30 次运行结果进行 Wilcoxon rank-sum 检验；算法不同但 `p > 0.05` 时计入 acceptable accuracy。

论文正文给出的通用 LightGBM 设置为学习率 `0.01`、最大深度 `5`、early stopping `200`、最大迭代 `2000`。公开 notebook 存在数据集特定设置差异，例如 BBOB 单独使用学习率 `0.001`，部分数据集显式设置 L1/L2 正则化；入口默认采用论文正文的通用设置，所有差异通过命令行参数显式指定。

论文文字写明五折交叉验证，但公开 notebook 的主流程使用 `train_test_split(test_size=0.2, random_state=42)`，并把该留出集同时传给 LightGBM 的验证接口。入口保留两个模式：`paper_holdout` 用于贴近公开 notebook，`five_fold` 用于贴近论文文字。两种模式都应把评价集参与 early stopping 的事实写入结果解释，不能把它当作独立、无偏的泛化评价。

## 当前项目中的可复现性判断

当前项目现有产物不能直接支持论文的精确复现，原因是：

- 主项目优化器池是 DE、PSO、CMA-ES、SHADE，共 4 个算法，而论文基线需要 10 个算法；
- 主项目的廉价 query 是 14 个自定义 descriptor，和论文的 61 个低成本 ELA 特征不是同一特征表；
- 主项目的 action-loss 是共享状态级的候选动作结果，论文标签需要每个静态问题实例、每个算法 30 次独立运行的原始性能序列；
- 当前工作树没有论文所需的 61 维静态 ELA 表和 300 列运行结果表。

因此，不能用当前四算法状态级产物替代论文输入，也不能据此报告论文结果。缺失数据需要按论文输入契约重新生成或由用户提供。

## 实现与输入契约

实现位于 `baselines/as_lgbm_reproduction.py`，只依赖静态表文件：

- 特征表：每行一个实例，61 个数值特征；CSV 可带首列实例编号，Parquet 可通过 `--feature-columns` 指定；
- 性能表：每行一个实例，按算法顺序排列的 `10 × 30 = 300` 个运行值；CSV 可带首列实例编号，Parquet 可通过 `--performance-columns` 指定；
- 性能表也可以直接指定 RGI 生成器输出的 `performance_runs/` 分片目录，读取器会按 `part-*.parquet` 合并；
- 数据筛选遵循公开仓库的非零、有限值和绝对值上限规则；
- 输出保存为指定目录下的 `summary.json`、`predictions.csv` 和每折 LightGBM 文本模型。

示例命令：

```bash
uv run python -m baselines.as_lgbm_reproduction \
  --features <ela_61_table.csv> \
  --performance <performance_300_table.csv> \
  --output-dir outputs/as_lgbm_paper_baseline/bbob \
  --evaluation-mode paper_holdout
```

Parquet 输入应显式传入列名，例如：

```bash
uv run python -m baselines.as_lgbm_reproduction \
  --features <features.parquet> \
  --performance <performance.parquet> \
  --feature-columns feature_001,feature_002,...,feature_061 \
  --performance-columns run_001,run_002,...,run_300 \
  --output-dir outputs/as_lgbm_paper_baseline/dataset
```

## 新增实验说明

- 研究问题：在论文所定义的静态实例、ELA 特征、Soft-ERT 标签条件下，能否运行并核对 AS-LGBM 选择器流程。
- baseline：AS-LGBM；对照可在同一静态输入和同一 Soft-ERT 标签上另行运行 SBS、RF 或 SVM，但本次实现不把它们加入主 Decision Model。
- 数据泄漏：特征只来自实例采样，Soft-ERT 和标签只用于离线训练目标；然而公开 notebook 将留出评价集用于 early stopping，因此 `paper_holdout` 和 `five_fold` 的分数必须标注该评价集用途，不能当作独立确认性结果。
- 结果保存：每次运行的模型、预测逐行结果、标签分布、Soft-ERT 口径、超参数和各折指标保存在用户指定输出目录；当前未生成科学结果，因为所需输入数据尚缺。

## RGI 生成与十算法运行入口

根据论文和公开仓库，RGI 不是从已有 BBOB/CEC 函数表中读取，而是由随机表达式树生成。当前项目的独立基线实现位于：

- `baselines/rgi_generation.py`：SeedSequence 驱动的 full 表达式树、RPN 序列化和目标函数解码；
- `baselines/rgi_optimizers.py`：ABC、ACO、CMA-ES、CSO、DE、FEP、GA、PSO、SA、RAND 十个算法的独立运行器；
- `baselines/rgi_batch.py`：按实例分片生成、运行和保存结果。

默认参数为论文中给出的 `count=200000`、树深度 `5--8`、`dimension=10`、变量范围 `[-10, 10]`、`population_size=100`、`FE_total=10000` 和每算法 `30` 次运行。正式生成前可先由用户指定单独输出目录并检查资源安排；本轮只完成代码，不启动 200,000 实例运行。

随机流使用当前项目的 `numpy.random.SeedSequence` 规范，显式区分 `root_seed`、`instance_id`、stream code、repeat、algorithm index 和 event。表达式不通过 `eval` 执行，而是由 JSON RPN 解码器处理。仓库数值编码和权重原样保留；代码中实际存在 15 个算子编码（`10--24`），与论文文字中的“14 个算子”不一致，已在元数据中记录。

默认输出布局为：

```text
<output-dir>/
├── rgi_instances/part-*.parquet       # 每个实例一行，含 RPN 和树统计
├── algorithm_runs/part-*.parquet      # 每个实例×算法×重复一行
├── performance_runs/part-*.parquet    # 每个实例一行，300 个运行列
└── rgi_metadata.json
```

启动入口：

```bash
uv run as-lgbm-generate-rgi \
  --output-dir outputs/as_lgbm_rgi \
  --count 200000 \
  --batch-size 100
```

`performance_runs` 默认保存每次运行的 `final_best`。如果已经预先定义了目标值，需要与论文的目标判定规则一致后，显式使用 `--performance-metric target_fe --target-value <value>`；此时保存首次达到目标的 FE，未达到目标保存 `-1`，可直接对应现有 `soft_ert` 读取器。两种口径均在 `algorithm_runs` 中同时保留 `best_value`、`final_value`、`first_hit_fe`、`effective_fe`、`target_hit` 和失败信息，不能把没有预先定义目标时的最终目标函数值称为 Soft-ERT。

该 RGI 数据生产器只属于 AS-LGBM 文献基线，不修改主项目 `DE、PSO、CMA-ES、SHADE` 算法池，也不进入当前 Decision-before-Feature 的 Decision dataset 或 Selector。
