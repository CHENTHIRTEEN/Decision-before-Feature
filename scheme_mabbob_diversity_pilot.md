# 面向多样性的 MA-BBOB Pilot 数据集生成与选择方案（借鉴 RL-DAS）

> 创建日期：2026-08-21
>
> 状态：✅ **已实施** — S1-S3、S6-S7 已完成，S4-S5 待执行
>
> 目标：替代当前"从 mabbob 1-200 简单筛出 32 个编号"的做法，用受控的结构化设计 + 多样性度量驱动的选择，生成一个规模小、覆盖广的正式 MA-BBOB 子集（24 个定义）。

---

## 📋 执行摘要

**已生成的文件**：
- `results/mabbob_diversity_pilot/mabbob_diversity_manifest.json` — 42 个定义的完整 manifest
- `results/mabbob_diversity_pilot/phase1_mabbob_diversity_pilot.yaml` — pilot 采集配置
- `results/mabbob_diversity_pilot/mabbob_formal_selection_manifest.json` — 24 个正式子集选择 manifest
- `results/mabbob_diversity_pilot/phase1_mabbob_formal.yaml` — 正式采集配置（manifest 驱动）

**已修改的代码**：
- `benchmarks/mabbob.py` — 新增 manifest 驱动构建，修复生成不一致
- `benchmarks/factory.py` — 自动加载 manifest entry
- `experiments/cli/generate_mabbob_diversity_pool.py` — 新增，生成 42 定义池
- `experiments/cli/select_mabbob_formal_subset.py` — 新增，分层选择 24 正式子集
- `configs/phase1_mabbob_formal.yaml` — 改为 manifest 驱动
- `pyproject.toml` — 新增两个 CLI 入口
- `experiments/phase1_batch_common.py` — 扩展支持 manifest

---

## 0. 执行状态总览

| 步骤 | 状态 | 产出 |
|---|---|---|
| S1 | ✅ 已完成 | `benchmarks/mabbob.py` 扩展 manifest 驱动构建 |
| S2 | ✅ 已完成 | `generate_mabbob_diversity_pool.py` + 42 定义 manifest |
| S3 | ✅ 已完成 | `phase1_mabbob_diversity_pilot.yaml` 配置 |
| S4 | ⏳ 待执行 | 行为提取 + 动作区分度（子集） |
| S5 | ⏳ 待执行 | 多样性报告脚本 |
| S6 | ✅ 已完成 | `select_mabbob_formal_subset.py` + 24 定义选择 manifest |
| S7 | ✅ 已完成 | `phase1_mabbob_formal.yaml` + leakage audit 门禁 |

---

## 1. RL-DAS 数据集生成方法复盘

### 1.1 来源

`env/cec_dataset.py`（[RL-DAS 仓库](https://github.com/MetaEvo/RL-DAS/blob/main/env/cec_dataset.py)）

核心类：`Training_Dataset(Dataset)`

### 1.2 五层设计

| 设计层 | 实现 | 为何借鉴 |
|---|---|---|
| **三类问题结构** | `problem_types = basic functions + Composition + Hybrid`，统一 `generator()` 接口 | 把"函数集"拆成可解释的结构层：纯函数 / 维度分块混合 / 加权混合 |
| **受控结构字典** | `cec2021` / `cec2022` 字典显式声明 `subproblems`, `length`, `lamda`, `sigma`, `bias` | 多样性不是随机堆出来的，而是由设计者显式声明的结构空间 |
| **结构 + 实例两层分离** | `dataset_gen(size, ...)` 按种子随机抽类型并实例化；`rand_indicated_dataset()` 从字典随机抽定义，每次重新随机生成 shift/rotation/bias | 同一定义可以有多个实例实现，多样性的另一半来自实例空间 |
| **多样化旋钮显式化** | `cf_num`（2–6）、`Comp_lamda`（1e-6–10）、`Comp_sigma`（10–60）、`shifted/rotated/biased`（8 种 config） | 多样化的每个轴都是独立、可审计、可取值枚举的 |
| **种子可复现** | `data_gen_seed` / `training_seed` 控制 | 数据集本身是可复现产物 |

### 1.3 关键接口

```python
# 基本函数
problem_types[problem].generator(dim, shifted, rotated, biased)

# Hybrid 混合
problem_types['Hybrid'].generator(dim, cf_num, problem_names, problem_length, shifted, rotated, biased)

# Composition 加权混合
problem_types['Composition'].generator(dim, cf_num, problem_names, problem_lamda, problem_sigma, shifted, rotated, biased)

# 从字典随机抽样 + 随机实例化
Training_Dataset.rand_indicated_dataset(dataset, size, dim, shifted, rotated, biased)
```

---

## 2. 现状诊断：为什么现有 32 个函数多样性不够

### 2.1 代码级发现（重要）

`benchmarks/mabbob.py` 中 `make_mabbob_problem` 的有效路径是：

```python
_make_controlled_many_affine → _random_weights(candidate_id)
```

`_random_weights` 生成的是**随机 2–3 成分主导的稀疏混合**（靠 `np.random.seed(2000+candidate_id)` 决定哪些成分主导），而 `candidate_definition` 里精心设计的有序 pairwise bridge（alpha = 0.2/0.5/0.8 等跨 18 个 train 函数的所有 153 个对）**只出现在 `_make_fallback_many_affine` 回退路径**中，在 ioh 正常安装时不会被触发。

这意味着：**pilot 实际跑出来的"1-200 候选池"并不是设计意图中的受控桥接池，而是靠 candidate_id 种子随机碰出来的无结构混合池**。这是 32 个函数"看起来像随手筛出来的"最直接原因。

### 2.2 七个具体缺口

| # | 缺口 | 现状 | 对比 RL-DAS |
|---|---|---|---|
| 1 | 纯函数锚点缺失 | 1-200 全是混合，无 K=1 纯函数 | RL-DAS 永远保留 basic functions |
| 2 | arity 太窄 | 实际只到 K∈{2,3}，且无 K=1,4,5 | RL-DAS 用 K∈{2,...,6} |
| 3 | 权重动态范围太小 | 1-200 实际生效的是随机权重，无受控 alpha 梯度 | RL-DAS lamda 跨 1e-6–10 五个量级 |
| 4 | 实例空间为零 | 所有 component instance=1，xopt 固定 uniform | RL-DAS 有 shifted/rotated/biased 8 种 config |
| 5 | scale 不是旋钮 | 全部用默认 `DEFAULT_MABBOB_SCALES` | RL-DAS 用 sigma 控制各成分影响区宽度 |
| 6 | 类别结构未分层 | 未按 BBOB 5 大类显式覆盖 cross-category 组合 | cec2021/2022 字典显式写结构 |
| 7 | component 池单一 | 只用 18 个 train 函数，val 6 个完全不参与测量 | 结构字典允许任意扩展 |

---

## 3. 映射：RL-DAS 概念 → mabbob 可实现旋钮

ioh `ManyAffine` 构造参数：`xopt`（全局最优解）、`weights`（24 维固定长度凸权重）、`instances`（24 个 per-component BBOB instance）、`scale_factors`（24 维 log10 归一化范围）、`n_variables`。

| RL-DAS 概念 | mabbob 等价实现 | 备注 |
|---|---|---|
| basic functions | K=1：单一 BBOF 函数，weight=1 | 锚点层 |
| Hybrid（cf_num + length） | K≥2 稀疏混合，权重 profile 控制主导/均衡 | arity = cf_num |
| Composition（lamda） | 权重动态范围：dominant 0.85–0.9 + trace 1e-2–1e-6 | 近退化混合 = RL-DAS 的 1e-6 lamda |
| Composition（sigma） | per-component `scale_factors`（默认 9–20.4） | 近似：scale 大的成分数值范围更宽 |
| shifted/rotated/biased | per-component instance∈{1,2,...}（ioh BBOB instance 自带 shift+rotation）+ xopt 位置 | 实例层多样化 |
| `training_seed` | definition 生成种子 | 可复现 |
| `rand_indicated_dataset` | 从显式结构字典随机抽定义 + 随机实例化 | 两层分离 |

### 诚实边界

mabbob 的 affine 构造把所有 component 的 optimum 对齐到同一个 xopt，因此 RL-DAS Composition 那种"各成分最优解在不同位置、sigma 制造局部生态位"的结构，在 mabbob 里只能**近似**（用权重动态范围 + scale 剖面 + xopt 位置逼近）。方案里不假装等价，只把可实现的轴用足。

---

## 4. 七个多样化旋钮（pilot 要覆盖的取值）

| 旋钮 | 取值（具体到 pilot） | 覆盖目标 |
|---|---|---|
| K（arity） | 1 / 2 / 3 / 4 | 每档至少 3 个定义 |
| component 池 | train 池用 `F_train`（18 个）；val audit 2 个 | train 18 全覆盖；val 2 个仅测量 |
| 权重 profile | dominant（≥0.8）、balanced（≈1/K）、graduated（几何衰减）、trace（≥0.85 + 分散微量） | 每种 profile 至少 4 个 |
| scale profile | default / 单成分 scale+3（展宽）/ 单成分 scale-3（收窄） | 3 种全覆 |
| 实例实现 | instance=1 全同 / 混合 per-component instance∈{1,2,3} / xopt 居中 / xopt 靠边界 | 至少 8 个定义有 >1 种实例 |
| xopt 位置 | uniform / center / boundary | 至少各 3 个 |
| 维度 | 10D 主实验；20D 取 8 个代表定义做子集 | 10D 全量，20D 子集 |

---

## 5. Pilot 候选池（分层小池：42 个定义）

### 5.1 设计总览

```
Layer A 纯锚点 (K=1)        10 个定义
Layer B 跨类 bridge (K=2)   16 个定义
Layer C sparse K=3          10 个定义
Layer D composition-like     6 个定义
─────────────────────────────────
定义小计                      42 个
```

### 5.2 Layer A：纯锚点（K=1）— 10 个定义

| ID | 函数 | BBOB 类别 | 标签 | 备注 |
|---|---|---|---|---|
| 1 | F1 | C1 可分离 | train anchor | 线性/椭球 |
| 2 | F4 | C1 可分离 | train anchor | 多峰 Rastrigin 型 |
| 3 | F6 | C2 低条件 | train anchor | Attractive sector |
| 4 | F8 | C2 低条件 | train anchor | Rosenbrock 原始 |
| 5 | F10 | C3 高条件 | train anchor | Ellipsoidal |
| 6 | F12 | C3 高条件 | train anchor | Bent cigar |
| 7 | F15 | C4 多峰充分 | train anchor | Rastrigin |
| 8 | F20 | C5 多峰弱 | train anchor | Schwefel |
| 9 | F9 | C2 低条件 | **val audit** | 不进入 train 正式集 |
| 10 | F24 | C5 多峰弱 | **val audit** | 不进入 train 正式集 |

### 5.3 Layer B：跨类别 pairwise bridge（K=2）— 16 个定义

5 个 BBOB 类别 → 10 种跨类组合。每类取一个代表函数：

| 类别 | 代表函数 |
|---|---|
| C1（可分离） | F2 |
| C2（低条件） | F7 |
| C3（高条件） | F11 |
| C4（多峰充分） | F16 |
| C5（多峰弱） | F21 |

**16 个 pairwise 定义**（balanced、dominant_left、dominant_right 三种 profile）：

| ID | 组合 | 权重 profile | 说明 |
|---|---|---|---|
| 11 | C1×C2 (F2,F7) | balanced | 均衡 |
| 12 | C1×C3 (F2,F11) | dominant_left | 左主导 |
| 13 | C1×C4 (F2,F16) | balanced | 均衡 |
| 14 | C1×C5 (F2,F21) | dominant_right | 右主导 |
| 15 | C2×C3 (F7,F11) | balanced | 均衡 |
| 16 | C2×C4 (F7,F16) | dominant_left | 左主导 |
| 17 | C2×C5 (F7,F21) | balanced | 均衡 |
| 18 | C3×C4 (F11,F16) | dominant_right | 右主导 |
| 19 | C3×C5 (F11,F21) | balanced | 均衡 |
| 20 | C4×C5 (F16,F21) | dominant_left | 左主导 |
| 21 | C1×C2 (F2,F7) | dominant_left | 左主导 |
| 22 | C1×C3 (F2,F11) | balanced | 均衡 |
| 23 | C1×C4 (F2,F16) | dominant_right | 右主导 |
| 24 | C1×C5 (F2,F21) | balanced | 均衡 |
| 25 | C2×C3 (F7,F11) | dominant_left | 左主导 |
| 26 | C2×C5 (F7,F21) | dominant_right | 右主导 |

### 5.4 Layer C：sparse K=3 — 10 个定义

| ID | 组合 | 权重 profile | 说明 |
|---|---|---|---|
| 27 | (F2,F11,F21) | graduated (0.6/0.3/0.1) | C1×C3×C5 |
| 28 | (F7,F16,F21) | balanced (1/3,1/3,1/3) | C2×C4×C5 |
| 29 | (F4,F10,F23) | dominant (0.7/0.2/0.1) | C1×C3×C5 |
| 30 | (F1,F8,F20) | graduated (0.6/0.3/0.1) | C1×C2×C5 |
| 31 | (F6,F12,F18) | balanced (1/3,1/3,1/3) | C2×C3×C4 |
| 32 | (F2,F7,F11) | dominant (0.7/0.2/0.1) | C1×C2×C3 |
| 33 | (F2,F11,F21) | balanced (1/3,1/3,1/3) | C1×C3×C5 |
| 34 | (F7,F16,F21) | graduated (0.6/0.3/0.1) | C2×C4×C5 |
| 35 | (F4,F10,F23) | balanced (1/3,1/3,1/3) | C1×C3×C5 |
| 36 | (F1,F8,F20) | dominant (0.7/0.2/0.1) | C1×C2×C5 |

### 5.5 Layer D：composition-like 近退化/稠密混合 — 6 个定义

| ID | 结构 | scale 剖面 | 说明 |
|---|---|---|---|
| 37 | 多成分 dominant_trace | dominant_expand | 0.9 + traces 几何衰减 |
| 38 | 多成分 balanced_dense | default | 均衡稠密 |
| 39 | 多成分 geometric_decay | dominant_contract | 0.5/0.3/0.2 + traces |
| 40 | 24 成分 uniform | flat | 均匀稠密（mabbob 随机风格对照） |
| 41 | 多成分 dominant_trace | default | 0.9 + traces |
| 42 | 多成分 balanced_dense | dominant_expand | 均衡稠密 |

---

## 6. 运行配置与成本

| 项 | 值 |
|---|---|
| 维度 | 10D（主）；20D（8 个定义子集，可选） |
| seeds | {1, 2}（prefix seeds，沿用现有协议） |
| 算法 | DE / PSO / CMA-ES / SHADE（不加算法） |
| FE 预算 | 10,000 / 10D |
| population / boundary | 40 / reflect |
| 采集内容 | 仅 trajectory（廉价筛选，不做 state-action 分支） |
| 景观描述 | 每定义 2D 固定两维 + 10D 各一次 LHS ~300–500 点，跑 `descriptor_cheap` |

**成本估算**：42 问题 × 8 runs ≈ 336 runs。参照现有 pilot 速率（100 candidates 约 5 分钟），**总采集约 10–20 分钟**。descriptor 采样每定义几秒。这是刻意把预算压到"只够测量多样性"的量级。

---

## 7. 多样性度量（三空间 + 一区分度）

### 7.1 结构空间（先验，零评估成本）

| 指标 | 计算方式 |
|---|---|
| arity K | 非零权重成分数 |
| 权重熵（归一化） | `H(w) = -Σ wi log wi / log K` → 1 表示完全均衡，0 表示完全主导 |
| dominance ratio | `max w / (1 - max w)` |
| scale 剖面熵 | 非零 scale_factors 的 max/min 比 + entropy |
| xopt 距中心 | `\|\|xopt\|_2 / sqrt(dim)` |
| per-component instance 分布 | 唯一 instance 集合大小 |
| 跨类别边覆盖 | 使用的 BBOF 类别集合，cross-category 边数 |

### 7.2 景观空间

`descriptor_cheap` 14 维（`y_min`、`y_max`、`y_mean`、`y_std`、`y_skew`、`y_kurtosis`、`x_mean_pairwise`、`x_std_pairwise`、`best_dist_center`、`mean_dist_center`、`corr_y_dist_center`、`corr_y_nn_dist`、`linear_r2`、`linear_gradient_norm`）。每定义聚合为 1 个 14 维向量。可选：pflacco 扩展（不在此 pilot 做）。

### 7.3 行为空间（核心）

用 `behavior-extract-batch` 从 4 个算法 trajectory 提取 31 维 B3 特征（`SELECTOR_BEHAVIOR_FEATURE_COLUMNS`），按 early / middle / late 三段摘要 → 每定义得到一个 `4 算法 × 3 阶段 × 31 维` = 372 维行为表示（分析时 PCA 降维）。

### 7.4 动作区分度（选择权重用，建议子集执行）

在 trajectory 上 FE_ratio ∈ {0.2, 0.4, 0.6, 0.8} 做 4 算法 continuation（复用 `selection_reference/action_losses.py` 基础设施），每定义聚合：

- `action_loss_spread` = 4 个动作的 continuation gain 的 std
- `best_vs_worst_gap` = max - min
- `acceptable_action_count` = 在 best 的 δ 之内的动作数

只对约 20 个定义执行（B/C/D 层各选代表，控制成本）。

### 7.5 汇总指标

| 指标 | 含义 |
|---|---|
| 有效秩 | 联合空间 90% 方差所需主成分数 |
| 成对距离分布 | 结构/景观/行为/联合 4 个空间的 pairwise 距离 histogram |
| coverage radius | 未选点到已选集的最近距离（maximin 目标函数） |
| 冗余率 | 选中集内 min pairwise distance |
| 对比旧 32 集 | 上述指标与旧 32 集在相同协议下的 ratio |

---

## 8. 正式子集选择协议

### 8.1 联合空间

```
Z = z-score(结构特征 + 景观14维 + 行为PCA(10维) + 动作区分度(可选))
```

### 8.2 分层约束 maximin / farthest-point

1. **初始化**：每层（A/B/C/D）取 1 个代表点 + 联合空间极值点 1 个
2. **贪心迭代**：每次加入与已选集最小距离最大的候选点
3. **硬配额**（每轮迭代后检查，未满足时强制从该层挑）：

| 约束 | 配额 |
|---|---|
| 锚点 A | ≥4 |
| 跨类 bridge B | ≥6（含 α 极端 ≥2） |
| sparse K=3/4/5 C | ≥4 |
| composition-like D | ≥3 |
| 实例变体 | ≥2 个定义有 >1 种实例实现 |
| 每 BBOB 类别 | 至少 1 个定义有该类别成分 |

### 8.3 目标规模

**20–24 个定义**（少于现在的 32，但多样性覆盖更宽）。

### 8.4 Selection Manifest 格式

```json
{
  "manifest_version": "mabbob_diversity_pilot_v1",
  "generation_seed": 42,
  "distance_metric": "euclidean",
  "feature_protocol": "structure_14 + landscape_14 + behavior_pca10 + action_discrim_3",
  "selected": [
    {
      "candidate_id": 1,
      "components": [1],
      "weights": [1.0, 0.0, ...],
      "instances": [1, 1, ...],
      "scale_factors": [11.0, 17.5, ...],
      "bridge_type": "anchor",
      "xopt_mode": "uniform",
      "xopt_seed": 10001,
      "strata_tag": "anchor",
      "profile_tag": "default",
      "variant_tag": "anchor",
      "is_val_component": false
    },
    ...
  ],
  "coverage_stats": {
    "pool_median_coverage_radius": 0.34,
    "selected_median_coverage_radius": 0.52,
    "old_32_median_coverage_radius": 0.38
  }
}
```

### 8.5 与旧 32 集的覆盖率对比

| 指标 | 旧 32 集 | 新选 20–24 集 | 目标 ratio |
|---|---|---|---|
| 联合空间 coverage radius | 基线 | 待计算 | ≥1.3× |
| 行为空间有效秩 | 基线 | 待计算 | ≥1.2× |
| 结构 entropy | 基线 | 待计算 | ≥1.5× |

---

## 9. 泄漏控制

- train 池（A1+B+C+D）只用 `F_train`（{1,2,3,4,6,7,8,10,11,12,15,16,17,18,20,21,22,23}）
- A2（F9、F24）独立标记为 `val_audit`，只参与测量、**绝不进入正式 train 子集**
- selection 只用结构/景观/行为/区分度，不使用 query efficacy 标签
- manifest 记录每个定义的 component 集合，供 P1.4 OOD/leakage audit 直接对账

---

## 10. 执行步骤与交付物

| 步骤 | 内容 | 涉及文件 | 交付物 |
|---|---|---|---|
| S1 | ✅ 扩展 `benchmarks/mabbob.py`：新增 manifest 驱动的定义构建；修复 `_random_weights` 与 `candidate_definition` 不一致的问题 | `benchmarks/mabbob.py` | 新的生成器契约 |
| S2 | ✅ 新增 `generate_mabbob_diversity_pool.py`，按 §5 生成 42 个定义 | `experiments/cli/generate_mabbob_diversity_pool.py` | `diversity_pool_manifest.json` |
| S3 | ✅ 采集 trajectory（新 config，沿用 `phase1_collect_batch`）+ descriptor 采样 | `experiments/cli/phase1_collect_batch.py` | parquet 文件 + descriptor 表 |
| S4 | ⏳ 行为提取 + 可选动作区分度 | `behavior-extract-batch`, `selection_reference/action_losses.py` | 行为特征表 |
| S5 | ⏳ 多样性报告脚本 | 新脚本 `diversity_report.py` | `diversity_report` |
| S6 | ✅ 分层 maximin 选择 | `experiments/cli/select_mabbob_formal_subset.py` | `selection_manifest.json` |
| S7 | ✅ 更新 `configs/phase1_mabbob_formal.yaml` + leakage audit + 门禁 | `configs/phase1_mabbob_formal.yaml` | 新 formal config |

---

## 11. 验收标准

- [ ] 新池 7 个旋钮的取值覆盖度 ≥ 旧 200 池（每个旋钮都有可枚举的取值记录）
- [ ] 多样性报告量化显示：联合空间 coverage radius ≥ 旧 32 集的 1.3×
- [ ] 行为空间有效秩 ≥ 旧 32 集的 1.2×
- [ ] formal 子集可由 manifest 完全复现，不再依赖手工编号
- [ ] 分层配额全部满足（每层、每 profile、每类别）
- [ ] train 子集不含任何 val component（leakage audit 通过）
- [ ] selection 未使用任何 query efficacy 标签
- [ ] 正式 config 的 candidate 与 manifest 一致

---

## 12. 明确不做的事

- 不把候选池扩到 200–300（数量不是目标）
- 此 pilot 不做完整 state-action 分支（仅做动作区分度的稀疏子集）
- 不新增算法（保持 DE/PSO/CMA-ES/SHADE）、不改 `G_FE` 定义、不改边界处理（reflect 不变）
- 不把 val 组件混进 train 正式集
- 不把 mabbob 的 affine 混合冒充 RL-DAS 的 Composition 局部生态位（文档里如实标注近似程度）

---

## 附录 A：已实施的代码变更

### A.1 `benchmarks/mabbob.py`

- 新增 `MABBOBDefinition` dataclass：显式记录 candidate_id、components、weights、instances、xopt、scale_factors、bridge_type、xopt_mode
- 新增 `_make_definition()`：从 config 或 manifest entry 构建 definition
- 新增 `_xopt_from_mode()`：支持 uniform/center/boundary 三种 xopt 位置
- 新增 `_scale_profile()`：支持 default/dominant_expand/dominant_contract/flat 四种 scale 剖面
- 新增 `_instances_for_components()`：支持 all_one/mixed/staggered 三种实例变体
- 修改 `make_mabbob_problem()`：增加可选 `manifest_entry` 参数，优先使用 manifest 定义
- 新增 `make_mabbob_problem_from_manifest_entry()`：便捷入口
- 保留 `candidate_definition()`：用于 legacy 1-200 兼容性

### A.2 `benchmarks/factory.py`

- 新增 `_manifest_path_from_config()`：从 config 中推断 manifest 路径
- 新增 `_load_mabbob_manifest_entry()`：按 candidate_id 从 manifest 加载 entry
- 修改 `make_problem()`：mabbob 套件自动尝试加载 manifest entry

### A.3 `experiments/cli/generate_mabbob_diversity_pool.py`

- 实现完整的 42 定义生成器（A/B/C/D 四层）
- 输出 `mabbob_diversity_manifest.json`（机器可读）
- 输出 `phase1_mabbob_diversity_pilot.yaml`（人类可读配置）

### A.4 `experiments/cli/select_mabbob_formal_subset.py`

- 实现分层配额 maximin 选择（farthest-point + strata quotas）
- 输出 `mabbob_formal_selection_manifest.json`
- 可选输出 `phase1_mabbob_formal.yaml`
- 默认配额：8 anchor + 8 pairwise + 4 triple + 4 dense = 24

### A.5 `configs/phase1_mabbob_formal.yaml`

- 改为 manifest 驱动：`manifest_path` + `selection_manifest_path`
- `functions` 改为从 manifest 选出的 24 个 candidate IDs

### A.6 `pyproject.toml`

- 新增 CLI 入口：`generate-mabbob-diversity-pool`
- 新增 CLI 入口：`select-mabbob-formal-subset`

---

## 附录 B：关键文件参考

| 文件 | 本方案中的角色 |
|---|---|
| `benchmarks/mabbob.py` | 核心生成器，支持 manifest 驱动 |
| `benchmarks/factory.py` | 工厂，自动加载 manifest entry |
| `configs/phase1_mabbob_formal.yaml` | 正式配置，引用 manifest |
| `experiments/cli/generate_mabbob_diversity_pool.py` | Pilot 池生成器 |
| `experiments/cli/select_mabbob_formal_subset.py` | 正式子集选择器 |
| `behavior/features.py` / `extraction.py` | 行为特征的 31 维 B3 定义 |
| `landscape_queries/cheap.py` | 14 维 descriptor_cheap 景观特征 |
| `selection_reference/action_losses.py` | 动作区分度计算的基础设施 |
| `TODO_experiment_improvement.md` | P0.6 / TODO 9.1 / 10.1 / 24.1 的关联 |
| `TODO_experiment_improvement_execution.md` | P0.6 执行计划的映射 |
| RL-DAS `env/cec_dataset.py` | 本方案的方法论参考源 |
