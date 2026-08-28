# 03 · Task 9C：v2 与 SBS 的完整 policy-level 对比

- 日期：2026-08-29
- 问题：v2 是否优于 SBS = CMA-ES（train 派生的静态最优单算法，从 FE=0 跑满 10000 FE）？真实部署场景（初始算法 = SBS）下 v2 是否仍有正收益？
- 定义：$G_{v2|SBS}=L_{SBS}^{terminal}-L_{v2}^{terminal}$（log10 口径，正值表示 v2 更好）。SBS 终点取自 prefix=cmaes 的 continue-current 端点（同一评测管线内的配对测量），并与 `static_portfolio_summary` 的独立 SBS 统计做了数值核对。
- 场景：**all-prefix**（诊断口径：三个起始算法合并）与 **cmaes-start**（正式部署主口径）。不执行 objective evaluation。
- 产物：`analysis_v3/task3/{per_function_paired_gain,bootstrap_ci,scenario_summary.json,sbs_consistency_check.json,cmaes_start_selected_algorithm}.parquet`；run 级配对表在 `results/analysis_v3/task3/`。

## 1. SBS 参照一致性核对

| split | continue 行推得的 SBS 均值 | static summary 独立值 | 绝对差 |
|---|---:|---:|---:|
| train OOF | -5.1375 | -5.1920 | 0.0545 |
| validation | -5.0497 | -5.0613 | 0.0116 |

差异来源：continue 端点是 checkpoint 处重执行的中位重复测量，static 值来自原始连续轨迹；量级远小于策略差距，不影响结论。配对分析统一使用前者。

## 2. 主表（函数平衡）

| 场景 | split | runs | v2 terminal loss | SBS terminal loss | 配对增益 $G_{v2|SBS}$ | 95% bootstrap CI（函数级） |
|---|---|---:|---:|---:|---:|---|
| **cmaes-start（部署主口径）** | train OOF | 540 | -5.1375 | -5.1375 | **0.0000** | [0, 0] |
| **cmaes-start（部署主口径）** | validation | 180 | -5.0497 | -5.0497 | **0.0000** | [0, 0] |
| all-prefix（诊断） | train OOF | 1620 | -4.5132 | -5.1375 | **-0.6243** | [-0.939, -0.325] |
| all-prefix（诊断） | validation | 540 | -4.5731 | -5.0497 | **-0.4766** | [-0.861, -0.216] |

- **cmaes-start：v2 与 SBS 的 720 个 run（两 split）全部完全等价**——v2 从不为 cmaes-start 轨迹触发切换（Task 9A），端点与 SBS 逐 run 相同；success rate 也相同（train 0.398 / val 0.333）。
- **all-prefix：v2 显著劣于 SBS。** 以 $\pm\delta_{50}=0.0564$ 为等价带：train 上 v2 更差 35.2%、等价 51.8%、更好 13.0%；validation 上更差 46.3%、等价 43.7%、更好 10.0%（原始符号口径 v2 更差占 37.7% / 49.3%）。函数级 bootstrap CI 不含 0。机制：以弱算法消耗前 20% 预算再切回 cmaes，相对直接跑 cmaes 是净机会成本。
- VBS–SBS gap closed fraction：all-prefix 口径下 train 为 **-0.893**（v2 向反方向拉开 SBS–VBS 差距的 89%）；validation 上 VBS 与 SBS 几乎重合（-5.073 vs -5.061，分母 0.0117），该比值不稳定（-20.5），只作记录不作解释。cmaes-start 口径下为 0.000。

## 3. 逐函数配对增益（validation，all-prefix）

| 函数 | 配对增益均值 | v2 更好 run 占比 | v2 / SBS 均值 terminal loss |
|---|---:|---:|---|
| bbob_f005 | -0.385 | 0.0% | -11.62 / -12.00 |
| bbob_f009 | **-1.375** | 8.9% | -2.62 / -3.99 |
| bbob_f013 | -0.417 | 20.0% | -5.09 / -5.51 |
| bbob_f014 | -0.400 | 2.2% | -9.33 / -9.73 |
| bbob_f019 | -0.241 | 14.4% | -0.13 / -0.37 |
| bbob_f024 | -0.043 | 22.2% | +1.34 / +1.30 |

validation 的 6 个函数上配对增益全部为负：没有一个函数族支持"v2 优于 SBS"。损失最大的 bbob_f009 正是弱起始算法浪费预算的典型场景。

## 4. 对部署判断的含义

1. **不能把 v2 作为"优于 SBS 的 deployable DAS"**：all-prefix 口径的 1.8504 "gain over continue" 是对"随机/弱算法开局"的救火收益，混入了 prefix 构成（2/3 弱起始），不属于相对 SBS 的策略优势。
2. **CMAES-start 无明显灾难性恶化**（与 SBS 完全相同），判据 4 成立；但正收益也不存在——v2 在强求解器轨迹上没有识别出任何转移机会（Task 9A/B）。
3. 按工作单的分支规则：由于 "CMAES-start v2 > SBS" 不成立且 "v2 ≈ Fixed-0.20" 成立，若进入 CEC2017，v2 的研究定位应为 **behavior-based recovery / continuation selection from arbitrary solver trajectories + trajectory-conditioned early algorithm selection**，不得声称 dynamic scheduling 或超越 SBS。
