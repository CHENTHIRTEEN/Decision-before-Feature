# Decision-before-Feature：Phase 1 / Phase 2 深度诊断与下一阶段方法固定报告

- 日期：2026-08-28
- 范围：Task 0–8 全部完成；Task 9（CEC2017）与 Task 10（CEC2022）按要求未启动、未运行任何新 objective evaluation。
- 命名说明：按项目开发规范（AGENTS.md 0.3），原工作单中的 "audit" 一词统一改为一致性检查（consistency check），总报告文件名中的相应词以"方法固定"表述。

---

## 1. 协议一致性检查（Task 0，详见 00 报告）

16 项运行时断言 PASS、1 项结构性 WARNING（train OOF 阈值选择乐观性，由独立 validation 缓解）、0 FAIL。state 键唯一、动作矩阵完整、continue 为原生续跑、切换统一 population transfer、重复配对正确、family OOF 真 leave-family-out、MA-BBOB 无组件泄漏、validation 未参与任何 fit、first-trigger 每 run 一次且取最早越过。Phase 1/2 数据无需重新生成。

## 2. 主候选载体对比（Task 1，详见 01 报告）

| 候选 | train OOF gain | val gain | val 归一化 regret | 排序/命中 |
|---|---:|---:|---:|---|
| A 三分类 action-gain | 0.8467 | 1.4481 | 0.2561 | — |
| **B action-loss 回归 v2** | **1.3141** | **1.8504** | **0.1636** | pairwise 0.761，top-1 0.645 |
| C prefix-aware 回归 | 1.3148 | 1.8504 | 0.1636 | 与 B 无可辨差异 |

B 捕获逐状态最佳动作上界的约 83.8%（1.8504/2.2080）；验证集 harmful switch rate 为 0。

## 3. practical delta 敏感性（Task 2，详见 02 报告）

noise delta 由 q=0.50 的 0.056 膨胀到 q=0.95 的 1.464（重尾驱动）。Behavior > Time-only 在全部 q 成立；**回归载体的 gain 在所有 q 下恒为 1.3141 且逐 delta 占优**；窄 delta 使分类器 harmful switch 恶化到 29.2%。1.464 确实把中等差异归入 Equivalent，但这不是主结论的成因；维持 q=0.95 预指定值。

## 4. CMA-ES 误差分析与 prefix 诊断（Task 3，详见 03 报告）

CMA-ES AP 低于基线率由 **SHADE→CMAES** 单对驱动（AP 0.209 vs 基线率 0.296；PSO→CMAES 反而 0.340 > 0.286）。prefix-aware 在 train OOF 大幅提升（0.994 vs 0.847）但完全不迁移（validation 1.449 vs 1.448）；对回归载体同样无增益（C≈B）。结论：当前求解器上下文是训练族内的记忆信号，不构成可迁移信息。回归载体下 cmaes 行增益 Spearman −0.258 是同一难例的回归版表现，列为首要误差分析对象。

## 5. Local Landscape 分组消融与冗余（Task 4/5，详见 04/05 报告）

- 逐组配对增量（train OOF，vs M1）：L1 +0.0345、L2 +0.0005、L3 −0.0156、L4 +0.0120；validation 上 L4 组合明显有害（1.287 vs 1.453）、L2 单独崩落（0.729）。无任何组建立稳健 conditional increment。
- M2-alone 现象由 L1 分布特征解释（M2_L1 validation 1.429，接近 M1）。
- 冗余机制：lf→bf 跨组预测 R² 中位 0.475（46% 的 bf 目标可被 lf 以 R²>0.5 预测）；A1 vs A3 在 1.6% 的 state 上分歧，分歧处强制采用 A3 真实增益差 mean −2.58（338:68 劣于 A1）。
- 论文可写结论：**search behavior 与 trajectory-derived landscape descriptors 都编码了求解器-问题交互信息，但在当前动态选择设定下其可行动信息大量重叠**。该结论由上述分析支持，非预设。

## 6. 不确定性（Task 6，详见 06 报告）

descriptor bootstrap uncertainty（M4）度量的是估计稳定性而非决策不确定性，暂停扩展。action-level 诊断显示：有害切换集中于小增益边际（AUC 0.779）与高树间分歧（AUC≈0.87），且"集成一致但一致地错"的系统性错误与 SHADE→CMAES 难例互证。若未来重启信息获取研究，Query 触发器应基于 action uncertainty。

## 7. MA-BBOB 增广（Task 7，详见 07 报告）

增广有效：validation gain 1.448 vs 1.198（B-only）、归一化 regret 0.256 vs 0.319；MA definition 分组 OOF AP 0.604 / ρ 0.683。保留增广协议。

## 8. Phase 1 主方法裁决（Task 8，详见 08 报告）

**主方法固定为 `behavior_action_loss_regression_v2`**：28 维 algorithm-agnostic Behavior → 多输出损失回归 → 预测增益 → 阈值 0.1451 的 first-trigger 切换规则（阈值经 train grouped family OOF 确定）。三分类保留为对照/可解释性载体；prefix-aware 为诊断记录；历史结果全部保留。**Phase 1 判定：GO**。

## 9. 数据复用与新运行说明

- **Task 0–8 全部复用已有 parquet 与已训练模型，新 objective evaluations = 0**，无新增 FE 消耗，不触碰原协议。
- 所有重分析写入新目录（`analysis_v2/` 与 `results/analysis_v2/`），未覆盖任何历史结果。

## 10. 结论变化清单

| 原结论 | 状态 |
|---|---|
| Behavior > Time-only；策略优于 Continue/Random | **加强**（对 delta 口径稳健，Task 2） |
| behavior_action_loss_rf 强于三分类 | **加强并升级**：正式化为主方法 v2（Task 1/8） |
| MA-BBOB 增广价值未知 | **新证据支持保留**（Task 7） |
| 三分类作为主载体 | **被替代**（降级为对照，Task 8） |
| M3/M3+ 不确定性增量存疑 | **确认为阴性并给出机制**（重叠 + 过拟合，Task 4/5） |
| descriptor bootstrap uncertainty | **暂停**（度量对象错位，Task 6） |
| prefix 信息价值 | **新的诊断性结论**：训练族内有用、不迁移（Task 3） |
| 跨维度泛化、工程问题、CEC 迁移本身 | **尚无法判断**（协议限定 10D/10k FE，待 Task 9/10） |

## 11. CEC2017 是否 GO

**建议 GO（进入 Task 9 的准备）**，前提是先完成一个小的兼容性工作项：当前在线策略实现（`behavior_with_ela/online*.py`）消费的是三分类 bundle 接口，需要为回归载体加一个在线决策适配器（预测损失 → 增益 → 阈值 0.1451 的 first-trigger 判定），并纳入在线对比面板。除该适配器与既有在线重放外，不需要任何新 objective 数据生成。

进入 CEC2017 前需固定的完整清单（均已有值）：Behavior 特征 28 维（`SELECTOR_BEHAVIOR_FEATURE_COLUMNS`）、无预处理（仅中位数插补）、portfolio [pso, shade, cmaes]、模型类 RandomForestRegressor(200, depth 8, sqrt)、practical delta 1.464、决策阈值 0.1451、监控机会与事件定义（`phase1_dynamic_budget_event_v1`）、one-switch 规则、population 40、FE 10000、reflect 边界、CEC2017 题集 F1, F3–F29（10D，instances [1]）。CEC2017 只评价、不训练；不得据其结果回调模型后重报同一套结果。

下一阶段精确命令（按工作单要求仅列出、不执行）：

```bash
# (a) 实现 regression 载体的在线适配器后，注册 v2 策略进在线对比面板
#     (修改 behavior_with_ela/online_baselines.py / online_comparison.py 增加一个策略名)
# (b) CEC2017 在线评价（全部策略，含 SBS/continue/fixed/random-matched/time-only/三分类/v2 回归/traditional AAS）
.venv/bin/python -m behavior_with_ela.online_comparison \
  --config configs/behavior_with_ela_cec2017.yaml \
  --output behavior_with_ela/results/online/cec2017_v2
# (c) 汇总报告生成后写入 behavior_with_ela/analysis_v2/09_cec2017_transfer.md
```

## 12. 本轮交付物清单

- 报告：`behavior_with_ela/analysis_v2/00…08` 共 9 份 .md + 本总报告
- 分析代码：`behavior_with_ela/analysis_v2/task0_check.py`、`common.py`、`task1…task7` 脚本
- 数据：`behavior_with_ela/results/analysis_v2/{task0…task8}/`（OOF、run 明细、阈值表、汇总 parquet/json/csv；不入 git）
