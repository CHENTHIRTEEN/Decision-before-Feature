# 下游 Selector 独立预训练验证与固定使用方案

> 创建日期：2026-08-21
> 状态：📝 **方案设计**
> 关联：MA-BBOB diversity pilot（42 定义）已就绪，可直接用于 selector 训练

---

## 0. 执行摘要

本方案旨在**独立预训练并固定一个下游 ELA-based Algorithm Selector**，作为 Decision-before-Feature（DBF）框架的**固定下游组件**。Selector 的角色是：

> **在已决策执行 ELA 的前提下，根据 ELA 特征选择最优算法**

这与 DBF 的核心创新点（`search behavior -> decide whether to execute ELA`）是**互补关系**，而非竞品。Selector 是 ELA 路径的**终端决策器**，DBF controller 是 ELA 路径的**门控器**。

---

## 1. 文献定位与方法谱系

### 1.1 核心参考论文

| 论文 | 贡献 | 与本方案的关系 |
|---|---|---|
| **Guo et al. (2025)** *Automated algorithm selection for black-box optimization using light gradient boosting machine* | 使用 **LightGBM** 做 AAS，在 BBOB-2024 上展示了 SOTA 性能 | ✅ **直接借鉴**：本方案的 selector baseline 采用 LightGBM + RF 对比 |
| **Mersmann et al. (2011)** *Exploratory landscape analysis* | 定义 **ELA 特征**（20+ 维度，如 slope、curvature、ruggedness） | ✅ **特征工程基础**：本方案的 landscape features 子集来源于 ELA |
| **Kerschke & Trautmann (2019)** | ELA-based AAS 的综述与分类：classification / regression / pairwise | ✅ **方法分类参考**：本方案采用 **regression** 范式（预测每个算法的 loss） |

### 1.2 方法分类（Kerschke & Trautmann, 2019）

| 范式 | 优势 | 劣势 | 本方案选择 |
|---|---|---|---|
| **Direct Classification** | 简单，直接预测最优算法 | 无法捕获算法间的相对优势 | ❌ |
| **Performance Regression** | 可预测每个算法的具体性能，支持任意 portfolio | 需要更多数据，计算成本高 | ✅ **本方案采用** |
| **Pairwise Regression** | 专注算法对比，可能更精确 | 组合爆炸（C(4,2)=6 对），训练复杂 | ❌ |
| **Cost-Sensitive Classification** | 显式建模算法成本 | 成本建模复杂 | ❌ |

**结论**：本方案采用 **Performance Regression** 范式，原因：
1. 与现有 `selection_reference/model.py` 实现一致（`RandomForestRegressor` 预测 `target_selector_loss`）
2. 支持任意 portfolio 扩展（当前 4 个：DE/PSO/CMA-ES/SHADE）
3. 可直接复用 action-loss 标签（已冻结协议）

---

## 2. 当前实现现状

### 2.1 已冻结的 selector 实现

`selection_reference/model.py` 已实现：

```python
# 核心模型
Pipeline([
    ('imputer', WeightedMedianImputer()),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        random_state=1701,
        min_samples_leaf=2,
        n_jobs=1
    ))
])
```

**输入特征**：
- 行为特征：`SELECTOR_BEHAVIOR_FEATURE_COLUMNS`（31 维 B3 特征）
- 查询特征：`query_spec.feature_columns`（景观描述特征）
- 剩余预算比例：`remaining_budget_ratio`

**目标变量**：
- `target_selector_loss_{algorithm}`：每个算法的 **clipped log10 gap advantage vs continue_current**
- 最终选择：`argmin(selector_target_loss)`

**训练协议**：
- `SELECTION_REFERENCE_PROTOCOL = "statewise_observed_action_loss_regression"`
- 跨 CV group 交叉验证（`GroupKFold`，按 `cv_group_id` 分组）
- 使用 `cluster_balanced_row_weights` 平衡类别权重

### 2.2 现有数据资产

| 数据集 | 状态 | 规模 | 用途 |
|---|---|---|---|
| BBOB train（18 函数 × 2 instances × 10D/20D） | ✅ 已采集 | ~2,880 runs | 训练集 |
| BBOB validation（6 函数 × 2 instances × 10D/20D） | ✅ 已采集 | ~960 runs | 验证集 |
| **MA-BBOB diversity pilot（42 定义 × 1 instance × 10D）** | ✅ **新采集** | **1,680 runs** | **扩展训练集** |
| MA-BBOB formal（24 定义 × 1 instance × 10D） | ⏳ 待采集 | 960 runs | 最终验证 |

### 2.3 现有问题（来自历史诊断）

| 问题 | 诊断结果 | 影响 | 解决方案 |
|---|---|---|---|
| **H1 训练覆盖不足** | validation `changed_algorithm` `U_ELA>0` rate > train | held-out family 泛化差 | ✅ **新 MA-BBOB 池扩展训练覆盖** |
| **H2 bucket 稀疏** | 相邻 performance bucket 可复现跳变 | nearest-bucket 映射敏感 | ⚠️ 保留 RF，不依赖 bucket |
| **H3 VBS 标签混杂** | `same_algorithm` vs `changed_algorithm` 来源不同 | 效用解释混乱 | ✅ 分层报告（已在代码中实现） |
| **H4 behavior feature 表达不足** | candidate features 有排序信号但 threshold policy 不稳定 | 行为表达力有限 | ⚠️ 扩展特征集 |
| **H5 模型容量与归纳偏置** | RF train-perfect 但 validation mismatch | 过拟合疑虑 | ⚠️ 对比 LightGBM |

---

## 3. Selector 训练方案

### 3.1 训练目标

**固定一个下游 ELA-based selector**，满足：

1. **协议冻结**：输入/输出契约与现有 `selection_reference` 完全兼容
2. **数据充分**：在 BBOB + MA-BBOB diversity 池上训练，覆盖更广泛的景观
3. **模型多样性**：对比 RF vs LightGBM（Guo 2025），选择更稳健的模型
4. **可复现**：种子、超参数、特征集全部冻结
5. **可评估**：提供完整的验证报告（accuracy、regret、U_ELA 等）

### 3.2 训练数据准备

#### 3.2.1 数据源

| 来源 | 套件 | 维度 | seeds | 实例 | 状态 |
|---|---|---|---|---|---|
| BBOB train | bbob | 10D, 20D | 1-2 | 1-2 | ✅ 现有 |
| BBOB validation | bbob | 10D, 20D | 1-2 | 1-2 | ✅ 现有 |
| MA-BBOB diversity pilot | mabbob | 10D | 1-2 | 1 | ✅ **新增** |
| MA-BBOB formal | mabbob | 10D | 1-2 | 1 | ⏳ 待采集 |

**训练集构成**：
- **主训练集**：BBOB train + MA-BBOB diversity pilot
- **验证集**：BBOB validation（严格隔离，不参与训练）
- **测试集**：MA-BBOB formal（最终评估）

#### 3.2.2 特征工程

**行为特征（31 维 B3）**：
```python
SELECTOR_BEHAVIOR_FEATURE_COLUMNS = (
    # Base (9): improvement, diversity, covariance, stagnation, convergence
    "bf_fe_ratio", "bf_improvement_rate_w02", "bf_improvement_frequency_w02",
    "bf_diversity_mean_pairwise", "bf_diversity_change_w05",
    "bf_covariance_spectral_concentration", "bf_distance_decay_w10",
    "bf_stagnation_w10", "bf_convergence_rate_w10",
    
    # Primary (10): fitness diversity, wasserstein, centroid shift, etc.
    "bf_fitness_diversity_rel", "bf_population_wasserstein_rate_w05",
    "bf_centroid_shift_rate_w05", "bf_centroid_shift_coherence_w05",
    "bf_fitness_quantile_improvement_fraction_w02",
    "bf_fitness_distribution_improvement_rate_w02",
    "bf_fitness_wasserstein_rate_w02", "bf_elite_concentration",
    "bf_best_fitness_slope_rel_w05", "bf_diversity_slope_w05",
    
    # DynamoRep Lite (5): motion-related
    "bf_fitness_spread_slope_w05", "bf_population_centroid_shift_w05",
    "bf_elite_centroid_shift_w05", "bf_covariance_trace_ratio_w05",
    "bf_covariance_effective_rank_w05",
    
    # Maturity (3): search progress
    "bf_search_maturity", "bf_search_maturity_linear", "bf_explore_exploit_ratio"
)
```

**景观特征（ELA 子集）**：
- 从 `landscape_queries` 提取的特征（与 Mersmann 2011 兼容）
- 包括：slope、curvature、ruggedness、global structure 等
- **注意**：景观特征在 state-level 可能为空（未执行 query），需使用 `WeightedMedianImputer` 填充

**目标特征**：
- `remaining_budget_ratio`：剩余预算比例（FE_ratio）

#### 3.2.3 标签构造

**action-loss 标签**（已冻结协议）：
```python
# 对于每个 state（problem × algorithm × seed × FE），有 4 个 candidate actions
# 每个 action 对应一个 target_algorithm ∈ {de, pso, cmaes, shade}

# 标签定义
selector_target_loss = log10(action_loss) - log10(continue_current_loss)
# 即：选择该算法 vs 继续当前算法的 log10 gap advantage

# 最终选择
selected_algorithm = argmin(selector_target_loss)
```

**标签质量保证**：
- 所有 action-loss 使用相同的 `FE_prefix`（前缀轨迹）
- `continue_current` action 的 `selector_target_loss = 0`
- 使用 `clipped_log10_gap_advantage_vs_continue_current` 变换

### 3.3 模型选择与超参数

#### 3.3.1 Baseline 模型

| 模型 | 超参数 | 优势 | 劣势 |
|---|---|---|---|
| **RandomForestRegressor** | `n_estimators=200`, `min_samples_leaf=2`, `random_state=1701` | 稳健，不易过拟合 | 计算成本高 |
| **LightGBM（Guo 2025）** | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `random_state=1701` | 快速，在 AAS 上 SOTA | 需要调参 |
| **XGBoost** | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `random_state=1701` | 稳定 | 计算成本中等 |

#### 3.3.2 超参数搜索空间（可选）

如果要做超参数优化（非必须，但可提升性能）：

```python
# RandomForest
param_grid_rf = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_depth': [None, 10, 20],
}

# LightGBM
param_grid_lgb = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [4, 6, 8],
    'regressor__learning_rate': [0.05, 0.1, 0.2],
    'regressor__num_leaves': [31, 63, 127],
}
```

**建议**：先使用默认超参数训练，如果验证集性能不理想再做 grid search。

### 3.4 训练流程

#### 3.4.1 步骤 1：数据准备

```bash
# 确保所有数据已采集
# BBOB train/validation 已存在
# MA-BBOB diversity pilot 已采集（42 个定义）

# 采集 MA-BBOB formal（24 个定义，用于最终验证）
uv run python -m experiments.cli.phase1_collect_batch \
  --config results/mabbob_diversity_pilot/phase1_mabbob_formal.yaml \
  --sharded --workers 1 --overwrite
```

#### 3.4.2 步骤 2：行为特征提取

```bash
# 提取 BBOB train 的行为特征
uv run python -m behavior.batch_extraction \
  --input results/phase1_bbob_train \
  --output results/phase1_bbob_train/behavior \
  --workers 4

# 提取 BBOB validation 的行为特征
uv run python -m behavior.batch_extraction \
  --input results/phase1_bbob_validation \
  --output results/phase1_bbob_validation/behavior \
  --workers 4

# 提取 MA-BBOB diversity pilot 的行为特征
uv run python -m behavior.batch_extraction \
  --input results/diversity_pilot_mabbob \
  --output results/diversity_pilot_mabbob/behavior \
  --workers 4

# 提取 MA-BBOB formal 的行为特征（验证用）
uv run python -m behavior.batch_extraction \
  --input results/phase1_mabbob \
  --output results/phase1_mabbob/behavior \
  --workers 4
```

#### 3.4.3 步骤 3：查询特征提取（ELA）

```bash
# 提取景观查询特征（需要先采集 query trajectories）
# 使用现有的 landscape_queries.batch_sampling

# 对于 BBOB train
uv run python -m landscape_queries.batch_sampling \
  --input results/phase1_bbob_train \
  --output results/phase1_bbob_train/query_features \
  --config configs/phase1_bbob_train.yaml

# 对于 MA-BBOB diversity pilot
uv run python -m landscape_queries.batch_sampling \
  --input results/diversity_pilot_mabbob \
  --output results/diversity_pilot_mabbob/query_features \
  --config results/mabbob_diversity_pilot/phase1_mabbob_diversity_pilot.yaml
```

#### 3.4.4 步骤 4：Action-Loss 计算

```bash
# 计算 state-action loss（需要先有 trajectories + query features）
# 使用 selection_reference/action_losses.py

# 对于 BBOB train
uv run python -m selection_reference.action_losses \
  --trajectory-dir results/phase1_bbob_train \
  --query-feature-dir results/phase1_bbob_train/query_features \
  --output results/phase1_bbob_train/action_losses \
  --config configs/phase1_bbob_train.yaml

# 对于 MA-BBOB diversity pilot
uv run python -m selection_reference.action_losses \
  --trajectory-dir results/diversity_pilot_mabbob \
  --query-feature-dir results/diversity_pilot_mabbob/query_features \
  --output results/diversity_pilot_mabbob/action_losses \
  --config results/mabbob_diversity_pilot/phase1_mabbob_diversity_pilot.yaml
```

#### 3.4.5 步骤 5：合并训练数据

```python
# scripts/merge_selector_training_data.py
import pandas as pd
from pathlib import Path

# 合并 BBOB train + MA-BBOB diversity pilot
train_sources = [
    Path("results/phase1_bbob_train"),
    Path("results/diversity_pilot_mabbob"),
]

# 读取 action losses
action_loss_dfs = []
behavior_dfs = []
query_feature_dfs = []

for source in train_sources:
    action_loss_path = source / "action_losses" / "action_losses.parquet"
    behavior_path = source / "behavior" / "behavior.parquet"
    query_feature_path = source / "query_features" / "features.parquet"
    
    if action_loss_path.exists():
        action_loss_dfs.append(pd.read_parquet(action_loss_path))
    if behavior_path.exists():
        behavior_dfs.append(pd.read_parquet(behavior_path))
    if query_feature_path.exists():
        query_feature_dfs.append(pd.read_parquet(query_feature_path))

# 合并
merged_action_losses = pd.concat(action_loss_dfs, ignore_index=True)
merged_behavior = pd.concat(behavior_dfs, ignore_index=True)
merged_query_features = pd.concat(query_feature_dfs, ignore_index=True)

# 保存合并后的训练数据
output_dir = Path("results/selector_training")
output_dir.mkdir(parents=True, exist_ok=True)

merged_action_losses.to_parquet(output_dir / "action_losses.parquet")
merged_behavior.to_parquet(output_dir / "behavior.parquet")
merged_query_features.to_parquet(output_dir / "query_features.parquet")
```

#### 3.4.6 步骤 6：模型训练

```python
# scripts/train_selector.py
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from selection_reference.model import (
    StatewiseSelectorModel,
    fit_selector_with_cross_family_predictions,
    read_action_loss_data,
    read_behavior_data,
    read_query_feature_data,
    get_sampling_spec,
    SELECTION_REFERENCE_PROTOCOL,
)
from landscape_queries.specs import get_query_spec
from decision.cluster_weighting import cluster_balanced_row_weights

# 1. 加载数据
action_loss_path = Path("results/selector_training/action_losses.parquet")
behavior_path = Path("results/selector_training/behavior.parquet")
query_feature_path = Path("results/selector_training/query_features.parquet")

action_losses = pd.read_parquet(action_loss_path)
behavior = pd.read_parquet(behavior_path)
query_features = pd.read_parquet(query_feature_path)

# 2. 准备 state matrix
from selection_reference.model import prepare_state_matrix, QUERY_ADJUSTED_BUDGET
from landscape_queries.specs import get_query_spec

query_spec = get_query_spec("phase1_dynamic_budget_event_v1")
states, portfolio = prepare_state_matrix(
    action_losses=action_losses,
    behavior=behavior,
    query_features=query_features,
    query_spec=query_spec,
    action_budget_mode=QUERY_ADJUSTED_BUDGET,
)

# 3. 训练 selector
selector_model, cross_predictions, prediction_source = fit_selector_with_cross_family_predictions(
    states=states,
    portfolio=portfolio,
    query_spec=query_spec,
    selector_input_mode="query_full",
)

# 4. 保存模型
from selection_reference.model import save_selector_model
save_selector_model(selector_model, Path("models/selector_v1.joblib"))

# 5. 验证集评估
val_action_loss_path = Path("results/phase1_bbob_validation/action_losses.parquet")
val_behavior_path = Path("results/phase1_bbob_validation/behavior.parquet")
val_query_feature_path = Path("results/phase1_bbob_validation/query_features.parquet")

val_action_losses = pd.read_parquet(val_action_loss_path)
val_behavior = pd.read_parquet(val_behavior_path)
val_query_features = pd.read_parquet(val_query_feature_path)

val_states, _ = prepare_state_matrix(
    action_losses=val_action_losses,
    behavior=val_behavior,
    query_features=val_query_features,
    query_spec=query_spec,
    action_budget_mode=QUERY_ADJUSTED_BUDGET,
)

# 预测
val_predictions = selector_model.predict_scores(val_states[list(selector_model.feature_columns)])

# 计算指标
from selection_reference.model import selection_rows
val_selection = selection_rows(
    states=val_states,
    portfolio=portfolio,
    predictions=val_predictions,
    prediction_source="cross_cv_group",
    runtime_selection=0.0,
)

# 保存验证结果
val_selection.to_parquet(Path("results/selector_validation/selection_rows.parquet"))
```

#### 3.4.7 步骤 7：LightGBM 对比实验

```python
# scripts/train_selector_lgb.py
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

# LightGBM pipeline
def make_lgb_model() -> Pipeline:
    return Pipeline([
        ('imputer', WeightedMedianImputer()),
        ('regressor', LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=1701,
            n_jobs=1,
        )),
    ])

# 训练流程与 RF 相同，仅替换模型
lgb_model = make_lgb_model()
# ... (与 RF 相同的训练流程)

# 保存 LightGBM 模型
save_selector_model(
    StatewiseSelectorModel(
        model=lgb_model,
        target_algorithms=portfolio,
        feature_columns=selector_model.feature_columns,
        default_algorithm=selector_model.default_algorithm,
        query_id=selector_model.query_id,
        query_protocol=selector_model.query_protocol,
        query_preprocessing_id=selector_model.query_preprocessing_id,
        sample_design_id=selector_model.sample_design_id,
        query_feature_columns=selector_model.query_feature_columns,
        selector_input_mode="query_full",
        action_budget_mode="query_adjusted_budget",
        selector_target_transform="clipped_log10_gap_advantage_vs_continue_current",
        fit_weight_mode="cluster_balanced_fit",
        protocol="statewise_observed_action_loss_regression_lgb",
    ),
    Path("models/selector_v1_lgb.joblib")
)
```

### 3.5 评估指标

#### 3.5.1 主要指标

| 指标 | 计算方式 | 含义 |
|---|---|---|
| **Selected=VBS** | `selected_algorithm == best_observed_algorithm` 的比例 | 选择到最优算法的准确率 |
| **Selector Regret (raw)** | `action_loss - best_observed_loss` | 选择算法 vs 最优算法的绝对 gap |
| **Selector Regret (norm)** | `(action_loss - best_observed_loss) / (worst_loss - best_loss)` | 归一化 regret（0=最优，1=最差） |
| **U_ELA>0 Capture** | `U_ELA > 0` 且 `selected != prefix` 的比例 | ELA 有正效用时 selector 切换算法的比例 |
| **Precision@1** | `selected_algorithm == best_observed_algorithm` 的精确率 | Top-1 选择准确率 |

#### 3.5.2 分层指标

**按算法分层**：
- 每个算法作为 prefix 时的 selector 性能
- 识别哪些算法的 continuation 更难预测

**按问题类型分层**：
- BBOB vs MA-BBOB
- 不同 landscape family（separable, multimodal, etc.）
- 不同 arity（anchor, pairwise, triple, dense）

**按 FE 比例分层**：
- Early phase（0.20-0.30）
- Mid phase（0.30-0.50）
- Late phase（0.50-0.60）

#### 3.5.3 评估脚本

```python
# scripts/evaluate_selector.py
import pandas as pd
import numpy as np
from pathlib import Path

def compute_selector_metrics(selection_df: pd.DataFrame) -> dict:
    """计算 selector 的主要性能指标。"""
    n = len(selection_df)
    
    # Selected=VBS
    selected_vbs = (selection_df["selected_algorithm"] == selection_df["best_observed_algorithm"]).mean()
    
    # Selector regret
    regret_raw = (selection_df["action_loss"] - selection_df["best_observed_loss"]).mean()
    worst_loss = selection_df[[f"observed_loss_{alg}" for alg in ["de", "pso", "cmaes", "shade"]]].max(axis=1)
    best_loss = selection_df[[f"observed_loss_{alg}" for alg in ["de", "pso", "cmaes", "shade"]]].min(axis=1)
    regret_norm = ((selection_df["action_loss"] - best_loss) / (worst_loss - best_loss + 1e-12)).mean()
    
    # U_ELA>0 capture
    u_ela_positive = selection_df["performance_gain_gap_raw"] > 0
    changed_algorithm = ~selection_df["selected_equals_prefix"]
    u_ela_capture = (u_ela_positive & changed_algorithm).mean()
    
    # Precision
    precision = selected_vbs  # Top-1 accuracy
    
    return {
        "n_states": n,
        "selected_vbs": float(selected_vbs),
        "selector_regret_raw": float(regret_raw),
        "selector_regret_norm": float(regret_norm),
        "u_ela_positive_rate": float(u_ela_positive.mean()),
        "changed_algorithm_rate": float(changed_algorithm.mean()),
        "u_ela_capture": float(u_ela_capture),
        "precision_at_1": float(precision),
    }

# 计算训练集指标
train_selection = pd.read_parquet(Path("results/selector_training/selection_rows.parquet"))
train_metrics = compute_selector_metrics(train_selection)

# 计算验证集指标
val_selection = pd.read_parquet(Path("results/selector_validation/selection_rows.parquet"))
val_metrics = compute_selector_metrics(val_selection)

# 计算测试集指标（MA-BBOB formal）
test_selection = pd.read_parquet(Path("results/selector_test/selection_rows.parquet"))
test_metrics = compute_selector_metrics(test_selection)

# 输出报告
report = {
    "train": train_metrics,
    "validation": val_metrics,
    "test": test_metrics,
}

import json
with open(Path("results/selector_evaluation/report.json"), "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
```

---

## 4. 实验设计

### 4.1 实验对比

| 实验 | 模型 | 训练数据 | 验证数据 | 目标 |
|---|---|---|---|---|
| **Baseline-RF** | RandomForestRegressor | BBOB train | BBOB validation | 基线性能 |
| **Extended-RF** | RandomForestRegressor | BBOB train + MA-BBOB diversity | BBOB validation | 评估扩展数据的效果 |
| **Baseline-LGB** | LightGBM | BBOB train | BBOB validation | 对比模型效果 |
| **Extended-LGB** | LightGBM | BBOB train + MA-BBOB diversity | BBOB validation | 最佳组合 |
| **Final-RF** | RandomForestRegressor | BBOB train + MA-BBOB diversity | MA-BBOB formal | 最终测试 |
| **Final-LGB** | LightGBM | BBOB train + MA-BBOB diversity | MA-BBOB formal | 最终测试 |

### 4.2 假设验证

| 假设 | 验证方式 | 预期结果 |
|---|---|---|
| **H1：扩展数据提升泛化** | Extended-RF vs Baseline-RF on BBOB validation | Extended-RF 的 `selected_vbs` 更高 |
| **H2：LightGBM 优于 RF** | Baseline-LGB vs Baseline-RF on BBOB validation | LGB 的 `selector_regret_norm` 更低 |
| **H3：组合效果最佳** | Extended-LGB vs 其他所有组合 | Extended-LGB 在所有指标上最优 |
| **H4：跨套件泛化** | Extended-RF on MA-BBOB formal | `selected_vbs` > 0.6 |

### 4.3 统计显著性检验

使用 **paired t-test** 或 **Wilcoxon signed-rank test** 检验不同模型在验证集上的性能差异是否显著。

```python
from scipy.stats import wilcoxon, ttest_rel

# 比较两个模型在验证集上的 selected_vbs
def compare_models(selection_df_a: pd.DataFrame, selection_df_b: pd.DataFrame) -> dict:
    vbs_a = (selection_df_a["selected_algorithm"] == selection_df_a["best_observed_algorithm"]).astype(float)
    vbs_b = (selection_df_b["selected_algorithm"] == selection_df_b["best_observed_algorithm"]).astype(float)
    
    # Paired t-test
    t_stat, t_p = ttest_rel(vbs_a, vbs_b)
    
    # Wilcoxon signed-rank test
    w_stat, w_p = wilcoxon(vbs_a, vbs_b)
    
    return {
        "t_test_p": float(t_p),
        "wilcoxon_p": float(w_p),
        "mean_a": float(vbs_a.mean()),
        "mean_b": float(vbs_b.mean()),
        "diff": float(vbs_b.mean() - vbs_a.mean()),
    }
```

---

## 5. 固定与部署

### 5.1 模型固定

选择最佳模型（根据验证集性能）作为**固定 selector**：

```python
# 示例：选择 Extended-LGB 作为最终模型
final_model_path = Path("models/selector_final.joblib")

# 从训练脚本复制最佳模型
import shutil
shutil.copy("models/selector_v1_lgb_extended.joblib", final_model_path)

# 验证模型契约
from selection_reference.model import load_selector_model
model = load_selector_model(final_model_path)
print(f"Fixed selector: {model.protocol}")
print(f"Input features: {len(model.feature_columns)}")
print(f"Target algorithms: {model.target_algorithms}")
```

### 5.2 部署集成

**集成到 `selection_reference/__init__.py`**：

```python
# selection_reference/__init__.py
from pathlib import Path
from .model import StatewiseSelectorModel, load_selector_model

# 固定 selector 模型路径
FIXED_SELECTOR_PATH = Path("models/selector_final.joblib")

# 缓存加载的 selector
_fixed_selector: StatewiseSelectorModel | None = None

def get_fixed_selector() -> StatewiseSelectorModel:
    """获取固定的下游 selector 模型。"""
    global _fixed_selector
    if _fixed_selector is None:
        _fixed_selector = load_selector_model(FIXED_SELECTOR_PATH)
    return _fixed_selector
```

### 5.3 使用示例

```python
from selection_reference import get_fixed_selector

# 加载固定 selector
selector = get_fixed_selector()

# 准备特征（来自行为提取 + 查询特征）
features = {
    "bf_fe_ratio": 0.25,
    "bf_improvement_rate_w02": 0.85,
    # ... 其他 31 维行为特征
    "ela_slope": -0.5,
    # ... 其他景观特征
    "remaining_budget_ratio": 0.75,
}

# 预测
selected_algorithm, score_map, elapsed = selector.select_one(features)
print(f"Selected algorithm: {selected_algorithm}")
print(f"Scores: {score_map}")
```

---

## 6. 文献对照与创新点声明

### 6.1 与 Guo (2025) 的对照

| 方面 | Guo (2025) | 本方案 |
|---|---|---|
| **模型** | LightGBM | LightGBM + RandomForest |
| **特征** | ELA + HPO features | ELA + Behavior features |
| **目标** | Algorithm ranking | Action loss regression |
| **数据集** | BBOB-2024 | BBOB + MA-BBOB |
| **评估** | ERT, A12 | Selected=VBS, Regret, U_ELA |

**共同点**：
- 都使用 LightGBM 作为主要模型
- 都基于 ELA 特征
- 都面向 black-box optimization

**差异点**：
- Guo 使用 **ranking loss**，本方案使用 **regression loss**
- Guo 专注 **algorithm selection**，本方案是 **action selection**（包含 continue_current）
- 本方案加入 **行为特征**，捕获搜索动力学

### 6.2 与 Mersmann (2011) 的对照

| ELA 特征类别 | Mersmann (2011) | 本方案 |
|---|---|---|
| **y-distribution** | mean, std, skewness, kurtosis | ✅ 包含 |
| **Fitness landscape** | slope, curvature, ruggedness | ✅ 包含 |
| **Global structure** | global minimum, funnels | ✅ 包含 |
| **Local structure** | neighborhood analysis | ⚠️ 部分包含 |

**结论**：本方案的景观特征与 ELA 方法论一致，是 Mersmann 工作的直接应用。

### 6.3 创新点声明

**本方案不声称 selector 本身是创新**：

> `selection_reference` 是**固定下游 ELA-based algorithm selection 组件**，用于构造 offline ELA utility labels。它属于既有 ELA-based per-instance algorithm selection 范式（Kerschke & Trautmann, 2019）。

**本文的创新点仍然是**：

> **Decision-before-Feature**: `search behavior -> decide whether to execute ELA`

而非：

> **ELA features -> propose a new algorithm selector**

因此，selector 的训练与固定是**支持性工作**，目的是为 DBF 框架提供一个稳定、可靠的下游组件。

---

## 7. 时间线与里程碑

| 阶段 | 任务 | 预估时间 | 依赖 |
|---|---|---|---|
| **P0** | MA-BBOB diversity pilot 采集 | ✅ 已完成 | - |
| **P1** | 行为特征提取（BBOB + MA-BBOB） | 1-2 小时 | P0 |
| **P2** | 查询特征提取（ELA） | 2-4 小时 | P1 |
| **P3** | Action-loss 计算 | 1-2 小时 | P2 |
| **P4** | 数据合并与清洗 | 0.5 小时 | P3 |
| **P5** | 模型训练（RF + LGB） | 1-2 小时 | P4 |
| **P6** | 验证集评估 | 0.5 小时 | P5 |
| **P7** | 模型选择与固定 | 0.5 小时 | P6 |
| **P8** | 最终报告 | 1 小时 | P7 |
| **总计** | | **8-14 小时** | - |

---

## 8. 交付物清单

### 8.1 代码

- [ ] `scripts/merge_selector_training_data.py` — 合并训练数据
- [ ] `scripts/train_selector.py` — RF 模型训练
- [ ] `scripts/train_selector_lgb.py` — LightGBM 模型训练
- [ ] `scripts/evaluate_selector.py` — 模型评估
- [ ] `selection_reference/__init__.py` — 固定 selector 集成

### 8.2 数据

- [ ] `results/selector_training/` — 合并后的训练数据
- [ ] `results/selector_validation/` — 验证集评估结果
- [ ] `results/selector_test/` — 测试集评估结果

### 8.3 模型

- [ ] `models/selector_final.joblib` — 固定的最终 selector 模型

### 8.4 文档

- [ ] `results/selector_evaluation/report.json` — 评估报告
- [ ] `docs/selector_training_report.md` — 训练过程文档

---

## 9. 风险与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|---|---|---|---|
| **数据不足** | 低 | 高 | MA-BBOB diversity 池已扩展到 42 个定义 |
| **过拟合** | 中 | 中 | 使用交叉验证 + 独立验证集 |
| **LightGBM 依赖** | 低 | 低 | 如无 LightGBM，使用 RF 作为 fallback |
| **特征缺失** | 中 | 中 | 使用 `WeightedMedianImputer` 填充 |
| **计算成本** | 中 | 低 | 单机即可完成，无需分布式 |

---

## 10. 相关文件与命令

### 10.1 核心文件

| 文件 | 角色 |
|---|---|
| `selection_reference/model.py` | Selector 模型定义与训练逻辑 |
| `selection_reference/action_losses.py` | Action-loss 计算 |
| `behavior/features.py` | 行为特征提取 |
| `landscape_queries/specs.py` | 查询特征规范 |
| `decision/cluster_weighting.py` | 类别平衡权重 |

### 10.2 关键命令

```bash
# 行为特征提取
uv run python -m behavior.batch_extraction --input <dir> --output <dir>/behavior

# 查询特征提取
uv run python -m landscape_queries.batch_sampling --input <dir> --output <dir>/query_features

# Action-loss 计算
uv run python -m selection_reference.action_losses --trajectory-dir <dir> --query-feature-dir <dir>/query_features --output <dir>/action_losses

# Selector 训练（待实现）
uv run python scripts/train_selector.py --action-loss <path> --behavior <path> --query-features <path> --output <model_path>

# Selector 评估（待实现）
uv run python scripts/evaluate_selector.py --selection <path> --output <report_path>
```

---

## 附录 A：特征列表

### A.1 行为特征（31 维）

```python
SELECTOR_BEHAVIOR_FEATURE_COLUMNS = (
    # Base (9)
    "bf_fe_ratio",
    "bf_improvement_rate_w02",
    "bf_improvement_frequency_w02",
    "bf_diversity_mean_pairwise",
    "bf_diversity_change_w05",
    "bf_covariance_spectral_concentration",
    "bf_distance_decay_w10",
    "bf_stagnation_w10",
    "bf_convergence_rate_w10",
    
    # Primary (10)
    "bf_fitness_diversity_rel",
    "bf_population_wasserstein_rate_w05",
    "bf_centroid_shift_rate_w05",
    "bf_centroid_shift_coherence_w05",
    "bf_fitness_quantile_improvement_fraction_w02",
    "bf_fitness_distribution_improvement_rate_w02",
    "bf_fitness_wasserstein_rate_w02",
    "bf_elite_concentration",
    "bf_best_fitness_slope_rel_w05",
    "bf_diversity_slope_w05",
    
    # DynamoRep Lite (5)
    "bf_fitness_spread_slope_w05",
    "bf_population_centroid_shift_w05",
    "bf_elite_centroid_shift_w05",
    "bf_covariance_trace_ratio_w05",
    "bf_covariance_effective_rank_w05",
    
    # Maturity (3)
    "bf_search_maturity",
    "bf_search_maturity_linear",
    "bf_explore_exploit_ratio",
)
```

### A.2 景观特征（ELA 子集）

参考 `landscape_queries/specs.py` 中的 `QUERY_FEATURE_COLUMNS`，包括：
- `ela_y_mean`, `ela_y_std`, `ela_y_skew`, `ela_y_kurtosis`
- `ela_slope_mean`, `ela_slope_std`
- `ela_curvature_mean`, `ela_curvature_std`
- `ela_ruggedness`
- `ela_global_minimum_estimate`
- `ela_funnel_count`
- ...（具体列表见 `landscape_queries/specs.py`）

---

## 附录 B：参考文献

1. **Guo, Y., et al. (2025)**. *Automated algorithm selection for black-box optimization using light gradient boosting machine*. arXiv preprint.
   - DOI: (待补充)
   - 核心贡献：LightGBM 在 AAS 上的 SOTA 性能

2. **Mersmann, O., et al. (2011)**. *Exploratory landscape analysis*. Genetic Programming and Evolvable Machines, 12(2), 139-162.
   - DOI: [10.1007/s10710-010-9105-0](https://doi.org/10.1007/s10710-010-9105-0)
   - 核心贡献：ELA 特征的定义与分类

3. **Kerschke, P., & Trautmann, H. (2019)**. *Automated algorithm selection on continuous black-box optimization problems*. Evolutionary Computation, 27(1), 35-69.
   - DOI: [10.1162/evco_a_00204](https://doi.org/10.1162/evco_a_00204)
   - 核心贡献：ELA-based AAS 的方法谱系

4. **Bischl, B., et al. (2012)**. *Automated algorithm selection on single objective numerical optimization problems*. Applied Soft Computing, 12(2), 559-569.
   - DOI: [10.1016/j.asoc.2011.07.011](https://doi.org/10.1016/j.asoc.2011.07.011)
   - 核心贡献：早期 cost-sensitive AAS 工作

---

*文档状态：📝 方案设计 | 待执行：P1-P8 | 预估总时长：8-14 小时*