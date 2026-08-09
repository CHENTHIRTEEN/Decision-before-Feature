
# Decision-before-Feature 当前实验结果分析与改进任务

## 角色

你现在作为本项目的科研实验工程代理。

项目目标：

研究 Decision-before-Feature：

> 利用低成本、算法无关的搜索行为信息，预测当前状态下执行 ELA 的净价值，从而决定是否调用 ELA-based Algorithm Selection。

当前核心不是提高普通回归精度，而是：

> 优化 ELA 调用决策质量。

---

# 1. 当前实验发现（必须先理解）

当前已有实验结果：

## primary

结果：

- ELA call rate:
  0.004120
- Utility capture:
  0.020836
- Precision:
  0.074906

---

## primary_with_maturity

加入 Search Maturity 后：

- ELA call rate:
  0.026620
- Utility capture:
  0.343901
- Precision:
  0.267246

相比 primary：

新增调用样本：

平均 utility:

0.095817

移除调用样本：

平均 utility:

-0.054178

净增：

summed utility:

170.900046

---

## all_candidates

加入全部候选 feature：

Ridge Spearman:

primary_with_maturity:

0.428243

all_candidates:

0.480589

进一步分析：

移除：

bf_best_distance_fitness_corr

后：

Spearman:

0.426603

基本回到 maturity 水平。

移除：

bf_population_overlap_w05

后：

Spearman:

0.484538

因此：

rank correlation 提升主要来自：

bf_best_distance_fitness_corr。

但是：

该 feature 更接近 landscape proxy。

不能直接作为主模型核心证据。

---

# 2. 当前核心判断

不要简单优化：

预测：

U_ELA

的 RMSE。

因为：

Decision-before-Feature 最终目标不是：

预测数值。

而是：

选择正确的 ELA 调用集合。

因此：

评价重点：

1. 哪些状态被调用；
2. 调用状态的平均 utility；
3. 总体 captured utility；
4. 成本收益。

---

# 3. 任务一：重新审计当前 Decision Objective

请分析：

当前模型是否存在：

## 问题1：

threshold 过于保守。

当前：

ELA call rate 极低：

0.4% -> 2.6%

请检查：

是否大量正 utility 状态被遗漏。

输出：

- Utility distribution；
- Positive utility ratio；
- Prediction score distribution；
- Threshold sensitivity。

---

# 4. 实现 Threshold Sweep

当前可能：

固定：

\hat U > 0

导致调用过少。

请实现：

threshold sweep。

遍历：

\[
\tau
\]

例如：

-0.5 到 0.5。

对于每个 threshold：

计算：

## Decision metrics

- ELA call rate
- precision
- recall
- utility capture
- summed utility
- average selected utility

绘制：

1. threshold vs summed utility
2. threshold vs call rate
3. precision-recall style utility curve

目标：

找到：

最大：

\[
\sum_i U_i I(\hat U_i>\tau)
\]

而不是最大：

RMSE。

---

# 5. 实现 Utility-aware Training

当前模型可能优化：

MSE。

但是：

MSE 与最终决策目标不一致。

请增加实验：

## Baseline

普通 regression：

\[
L=
(U-\hat U)^2
\]

---

## Weighted regression

尝试：

\[
L=
w_i(U_i-\hat U_i)^2
\]

其中：

方案：

\[
w_i=|U_i|
\]

\[
w_i=max(U_i,0)
\]

根据 utility uncertainty 设置。

比较：

- RMSE
- Spearman
- summed utility
- call quality

---

# 6. 实现 Two-stage Decision Model

当前：

单模型：

Behavior

↓

U_ELA

请增加：

## Stage 1

分类：

\[
P(U_{ELA}>\delta)
\]

判断：

是否值得调用。

---

## Stage 2

条件回归：

\[
E[U|U>\delta]
\]

只预测正 utility 样本。

最终：

组合：

\[
ExpectedUtility
===============

P(U>\delta)
\times
E[U|U>\delta]
\]

比较：

single regression

vs

two-stage model。

---

# 7. Search Maturity 深入分析

当前：

primary_with_maturity

有效。

不要直接认为：

Maturity feature 更强。

请分析：

它到底提升：

## Prediction ability

还是：

## Decision boundary quality

需要输出：

加入 maturity 前后：

- calibration curve
- ranking change
- threshold curve

---

# 8. Search Maturity Ablation

实现：

## M0

无 maturity

## M1

当前：

\[
M=ES(1-XS)
\]

## M2

Convergence-progress maturity:

\[
M=
(1-D_t/D_0)
\times IR_t
\]

## M3

Weighted maturity:

\[
M=w_1ES+w_2ST+w_3(1-H)
\]

比较：

重点指标：

不是 RMSE。

重点：

- summed utility
- utility capture
- precision
- call rate

---

# 9. FDC Proxy 分析

当前：

bf_best_distance_fitness_corr

提升 Spearman。

但是：

该 feature 接近 landscape proxy。

不要直接进入主模型。

请实现三个实验：

## Behavior-only

不包含 FDC。

## Behavior + FDC

加入：

bf_best_distance_fitness_corr。

## ELA Feature

完整 ELA。

比较：

预测性能和决策性能。

目标：

判断：

FDC 是否只是廉价 landscape proxy。

---

# 10. 不允许直接把 FDC 放入最终模型

除非满足：

1. 明确说明它不是 ELA feature；
2. 证明无需额外函数评价；
3. 证明跨算法；
4. 证明 OOD 泛化；
5. 证明不会破坏 algorithm-agnostic claim。

否则：

仅作为：

diagnostic feature。

---

# 11. 增加 Decision-aware Evaluation

所有实验必须报告：

不要只报告：

- RMSE
- Spearman

必须报告：

## Decision Metrics

- ELA call rate
- Precision:

\[
\frac{TP}{TP+FP}
\]

- Recall
- Utility capture:

\[
\frac{
\sum U_{selected}
}{
\sum U_{positive}
}
\]

- Summed utility

---

# 12. 保持主线

禁止：

把项目变成：

feature engineering benchmark。

核心问题：

不是：

哪个 feature 最强。

而是：

> Search behavior 是否包含关于 ELA acquisition value 的信息？

---

# 13. 推荐最终模型比较

至少比较：

## Model A

Progress only

输入：

- FE ratio
- improvement

---

## Model B

Behavior core

加入：

- diversity
- entropy
- stagnation
- success rate

---

## Model C

Behavior + maturity

---

## Model D

Behavior + maturity + FDC

仅作为诊断。

---

# 14. 最终论文推荐结论方向

如果实验支持：

论文不要写：

"maturity improves prediction"

而写：

> Search maturity improves the quality of ELA invocation decisions by identifying high-value analysis states rather than merely reducing regression error.

---

# 15. 开发要求

执行顺序：

Step 1:

审计已有结果。

Step 2:

实现 threshold sweep。

Step 3:

实现 utility-aware evaluation。

Step 4:

实现 maturity ablation。

Step 5:

实现 two-stage decision。

Step 6:

实现 FDC diagnostic experiment。

禁止：

- 修改已有原始实验结果；
- 修改 benchmark split；
- 使用 test 数据调 threshold；
- 将 FDC 偷渡进入主模型；
- 增加未经记录的新 feature。

所有新增实验必须保存：

- config；
- seed；
- split；
- model；
- metrics；
- output path。

目标：

找到最符合 Decision-before-Feature 科学问题的决策机制，而不是单纯追求预测指标最高。
