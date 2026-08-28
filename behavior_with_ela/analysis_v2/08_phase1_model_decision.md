# Task 8：Phase 1 主方法裁决

- 日期：2026-08-28
- 输入：Task 0–7 全部结果（协议一致性检查、v2 载体评测、delta 敏感性、prefix 消融、L 组消融、冗余分析、action uncertainty、MA-BBOB 增广消融）。
- 原则：优先函数平衡策略增益；其次归一化单次切换 regret；再次 harmful switch rate；跨函数稳健性；模型简洁性；校准/不确定性可用性。不以单一 validation mean gain 选择。

## 1. 三个候选的汇总（函数平衡）

| 候选 | train OOF gain | train 归一化 regret | val gain | val 归一化 regret | val harmful switch |
|---|---:|---:|---:|---:|---:|
| A. Behavior action-gain 三分类 | 0.8467 | 0.4023 | 1.4481 | 0.2561 | 0.0756* |
| **B. Behavior 多输出 action-loss 回归（v2）** | **1.3141** | **0.2564** | **1.8504** | **0.1636** | **0.0000** |
| C. Prefix-aware action-loss 回归 | 1.3148 | 0.2563 | 1.8504 | 0.1636 | 0.0000 |

*train OOF 口径。

附加证据：
- **delta 稳健性**（Task 2）：B 的 gain 在 q∈{0.50,…,0.95} 全部恒为 1.3141，且在每个 delta 上同时优于 A 与 time-only；A 在窄 delta 下 harmful switch 恶化到 29.2%。
- **简洁性**：B 与 C 为单模型多输出；A 需 3 个逐动作分类器。C 相对 B 无任何 split 上的可辨增量（train +0.0007，validation 逐项相同），却引入与算法池耦合的 3 个输入维度。
- **不确定性可用性**（Task 6）：B 的连续损失输出天然提供增益边际与树级分歧，二者对有害切换的识别 AUC 分别 0.779 / ≈0.87；A 的类别概率无此结构。
- **跨函数稳健性**：B 在 train OOF 与 validation 的方向一致且 validation 更强（1.314→1.850），未见反转。
- **辅助发现**：B 的 cmaes 候选行增益 Spearman 为 −0.258（Task 1），弱点与 A 的 CMA-ES AP 问题同源（SHADE→CMAES 对，Task 3），是后续误差分析的明确对象，但不足以改变载体的排序。

## 2. 裁决

**主方法 = 候选 B：`behavior_action_loss_regression_v2`**

$$B_t \rightarrow [\hat L_{PSO}, \hat L_{SHADE}, \hat L_{CMAES}], \qquad \hat G(s,a)=\hat L(s,\text{continue})-\hat L(s,a)$$

切换规则：在监控决策机会上，若 $\max_a \hat G(s,a) > \theta = 0.1451$（train grouped family OOF 上经函数平衡增益最大化确定）则切换到 $\arg\max_a \hat G(s,a)$，否则继续；每 run 至多切换一次。

- 三分类模型降级为对照/可解释性/标签噪声分析载体，全部历史结果保留。
- C 降级为诊断记录（prefix 上下文在训练族内有价值、不迁移）。
- 主任务表述调整为：**continuous action-loss prediction + practical action selection**。

## 3. Phase 1 最终判定：**GO**

依据：协议一致性检查通过（0 FAIL）；Behavior 优于 time-only 对 delta 口径稳健（Task 2）；策略优于 Continue/Random 在两个 split 成立；MA-BBOB 增广有正贡献（Task 7）；风险点是 cmaes/shade→cmaes 排序弱点与 train OOF 阈值乐观性（已由独立 validation 缓解）。

## 4. Phase 2 最终结论

在 Behavior 条件下，trajectory-derived local landscape 的 conditional increment 未获建立：M3−M1 在 train OOF +0.031、validation −0.028；action-loss 载体上 A3−A1 决策完全一致；分歧处 A1 系统性更优（Task 5）。M2-alone 现象主要由 L1 分布特征解释（Task 4），机制为 lf→bf 的信息重叠（Task 5）。descriptor bootstrap uncertainty 按 Task 6 的论证暂停扩展。
