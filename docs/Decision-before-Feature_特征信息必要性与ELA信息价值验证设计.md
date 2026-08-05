# Decision-before-Feature 特征信息必要性与ELA信息价值验证设计

## 1. 文档定位

本文档补充 Decision-before-Feature 研究中的核心问题：

> 如何证明完整ELA特征信息并非始终值得获取？

注意：

本文不主张"ELA特征没有价值"。

更准确的表述：

> 对于部分未知黑盒优化状态，获取完整Landscape
> Analysis信息带来的增量决策收益不足以抵消其获取成本。

------------------------------------------------------------------------

# 2. 两类"不需要"

## 2.1 不需要执行ELA（主要研究问题）

问题：

> 当前搜索状态是否值得执行Landscape Analysis？

比较：

-   Skip ELA
-   Full ELA

目标：

证明部分状态：

$$ U_{ELA}\leq0 $$

------------------------------------------------------------------------

## 2.2 不需要完整ELA特征（扩展问题）

问题：

> 执行ELA后，是否需要完整Feature Set？

比较：

-   Full ELA
-   Compact ELA

该方向属于Progressive ELA，可作为后续扩展。

------------------------------------------------------------------------

# 3. ELA Utility定义

不能只比较性能。

需要考虑成本：

$$ U_{ELA}=PerformanceGain-\lambda Cost $$

其中：

PerformanceGain：

-   ELA路径性能
-   Skip路径性能

Cost：

-   采样FE
-   Feature计算时间
-   Runtime

------------------------------------------------------------------------

# 4. 反事实分叉协议

必须使用共享搜索前缀。

流程：

## Step 1

运行优化算法到checkpoint。

保存：

-   population
-   fitness
-   trajectory
-   behavior state

------------------------------------------------------------------------

## Step 2

两个分支：

### Skip ELA

继续默认优化：

$$ P_{skip} $$

### Run ELA

执行：

ELA → Algorithm Selection → 优化

得到：

$$ P_{ELA} $$

------------------------------------------------------------------------

## Step 3

计算：

$$ U_{ELA} $$

作为Oracle标签。

------------------------------------------------------------------------

# 5. "大多数不需要"的统计证明

不能简单统计checkpoint数量。

原因：

同一轨迹中的checkpoint高度相关。

采用分层统计：

Function Family → Function Instance → Dimension → Algorithm → Seed →
Checkpoint

定义：

$$ \pi_{not}=P(U_{ELA}\leq\delta) $$

其中：

($\delta$) 为实际收益阈值。

要求：

$$ LCB_{95\%}(\pi_{not})>0.5 $$

即：

在95%置信下，大多数状态不需要ELA。

------------------------------------------------------------------------

# 6. 不应使用简单显著性检验

错误：

p\>0.05代表等价。

推荐：

-   Bootstrap confidence interval
-   TOST equivalence test

------------------------------------------------------------------------

# 7. 主要分析窗口

不建议使用50%、100%预算状态证明"不需要ELA"。

因为后期天然更容易没有收益。

主要窗口：

-   1%
-   2%
-   5%
-   10%
-   20%

50%、100%用于Search Maturity分析。

------------------------------------------------------------------------

# 8. 信息无用和成本过高的区别

必须区分：

## 情况A

ELA信息有效，但是成本太高：

$$ PerformanceGain>0 $$

但是：

$$ U_{ELA}<0 $$

------------------------------------------------------------------------

## 情况B

ELA信息增量有限。

比较：

Behavior-only

vs

Behavior+ELA

如果忽略成本后提升仍有限，则说明ELA信息对该决策增量有限。

------------------------------------------------------------------------

# 9. Full ELA与Compact ELA

如果研究特征冗余：

不要简单删除低SHAP特征。

原因：

ELA Feature高度相关。

推荐：

Feature Group分析：

-   Distribution
-   Geometry
-   Information Content
-   Meta-model

寻找：

$$ F^* $$

满足：

$$
Performance(F^*)\geq Performance(F_{full})-\epsilon
$$

同时降低计算成本。

------------------------------------------------------------------------

# 10. 推荐实验

## Experiment 1

ELA Utility Distribution

证明存在negative utility区域。

## Experiment 2

Need ELA比例分析

按：

-   function family
-   dimension
-   algorithm

统计。

## Experiment 3

Behavior-only vs Behavior+ELA

分析ELA信息增量。

## Experiment 4

Full ELA vs Compact ELA

分析Feature冗余。

------------------------------------------------------------------------

# 11. 与Decision-before-Feature关系

逻辑：

Search Behavior

↓

判断是否值得获取信息

↓

ELA Utility

↓

Decision-before-Feature

------------------------------------------------------------------------

核心不是：

减少Feature数量。

而是：

> 在获取信息之前判断信息获取是否具有净价值。

------------------------------------------------------------------------

# 12. 第一篇论文主张

建议主张：

1.  ELA并非always beneficial；
2.  搜索行为能够预测ELA Utility；
3.  Decision-before-Feature减少无效ELA调用；
4.  在降低分析成本的同时保持优化性能。

特征压缩和Progressive ELA作为后续研究方向。
