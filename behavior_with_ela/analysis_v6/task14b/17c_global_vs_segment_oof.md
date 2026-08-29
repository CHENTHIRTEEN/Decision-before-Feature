# 17c · Global vs Segment OOF 主比较（Task 14B H1/H2）

- 日期：2026-08-29。主比较口径：MGS vs MG（segment 在 global 之外的增量，工作单 §16）；carrier 预先固定 RF；fb paired bootstrap 5000 draws。

## 1. H1（Post-Handoff Behavior Increment vs 简单上下文/lookup）

| 比较 | BBOB | MA | 判定 |
|---|---:|---:|---|
| MGS vs Lookup | −0.0433 [−0.073, −0.021] | −0.0232 [−0.058, +0.021] | **拒绝 H1**：bbob 显著为负 |
| MGS vs M0 | −0.0422 [−0.067, −0.021] | −0.0192 [−0.058, +0.035] | 同上 |
| MG vs M0 | −0.0292 [−0.047, −0.011] | −0.0241 [−0.058, +0.023] | MG 同样为负 |

## 2. H2（Segment Increment Beyond Global）

| 比较 | BBOB | MA |
|---|---:|---:|
| MGS vs MG | −0.0129 [−0.035, +0.002] | +0.0049 [−0.009, +0.018] |

两 suite CI 均含 0，方向不一致（bbob 负 / ma 正）——**segment 在 global 之外无可执行增量**。

## 3. BG-only / BS-only 诊断（非主比较）

| 模型 | BBOB | MA |
|---|---:|---:|
| BG_only | −1.9596 | −5.0228 |
| BS_only | **−1.9413** | **−5.0033** |

BS-only 略优于 BG-only——但这只说明 segment 表示在当前模型下"拟合更顺"，不构成独立信息证据（工作单 §16 的核心告诫）；二者加合（MGS）并不优于单独任何一个。

## 4. 结论

$$
\boxed{\text{H1 与 H2 均不成立}}
$$

post-handoff 域内 Behavior（无论 global、segment 或并集）不能解释 route/context 之外的 next-action value；相应的 Task 14A 状态级残差 Δ_post≈0.105 在被测特征/载体下**不可被 Behavior 捕获**。
