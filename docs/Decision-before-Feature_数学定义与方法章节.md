# Decision-before-Feature 数学定义与方法章节

## 1. 方法定位

本文研究的问题不是传统的"如何利用ELA选择算法"，而是进一步研究：

> 在执行昂贵Landscape Analysis之前，是否应该执行Landscape Analysis。

因此，将问题定义为Analysis Selection Problem。

传统Automated Algorithm Selection流程：

    Optimization Problem
            |
            v
    ELA Feature Extraction
            |
            v
    Algorithm Selection
            |
            v
    Optimizer

本文提出：

    Optimization Problem

    |
            v

    Cheap Search Behavior Observation

    |
            v

    Decision-before-Feature

|                       |
| --------------------- |
|                       |
| Skip ELA      Run ELA |

    |              |

    Default      Algorithm Selection

    Optimizer

核心思想：

Feature extraction本身也是一种需要优化的资源消耗。

---

# 2. Problem Formulation

## 2.1 Optimization Problem

设黑盒优化问题为：

$$
p \in \mathcal{P}
$$

目标：

$$
\min f_p(x)
$$

其中：

- (p)：问题实例
- (x)：决策变量
- (f_p)：未知目标函数

---

## 2.2 Traditional Algorithm Selection

传统方法：

首先计算：

$$
\phi(p)
$$

其中：

$$
\phi
$$

表示ELA feature extractor。

然后：

$$
A^*=S(\phi(p))
$$

其中：

(S)为算法选择模型。

---

## 2.3 Limitation

ELA存在额外成本：

$$
C_{ELA}
$$

包括：

- Function Evaluation Cost
- CPU Time
- Memory Cost

因此：

不能默认：

$$
ELA(p)>0
$$

---

# 3. Analysis Selection Problem

定义决策变量：

$$
d\in\{0,1\}
$$

其中：

$$
d=1
$$

表示执行ELA。

$$
d=0
$$

表示跳过ELA。

目标：

最大化：

$$
U(d,p)
$$

即：

$$
U=Performance-Cost
$$

---

# 4. ELA Utility Label

为了训练Decision Module，需要构造监督信号。

## 4.1 No-analysis performance

直接优化：

$$
P_{skip}
$$

表示：

不执行ELA时的性能。

---

## 4.2 Analysis-based performance

执行：

$$
Problem \rightarrow ELA \rightarrow Algorithm
Selection
$$

得到：

$$
P_{ELA}
$$

---

## 4.3 Utility

定义：

$$
G=P_{skip}-P_{ELA}
$$

表示ELA带来的收益。

成本：

$$
C=C_{ELA}
$$

最终：

$$
U_{ELA}=G-\lambda C
$$

其中：

($\lambda$)控制性能和成本权衡。

决策：

$$
U_{ELA}>0
$$

执行ELA。

否则：

跳过ELA。

---

# 5. Search Behavior State Representation

Decision-before-Feature不能使用ELA feature作为输入，否则产生循环。

因此输入必须来自低成本搜索行为。

定义：

$$
s_t
$$

表示时间t的搜索状态。

组成：

$$
s_t= \[B_t,P_t\]
$$

其中：

B：

Behavior information。

P：

Progress information。

---

# 6. Behavior Features

## 6.1 Diversity

描述种群空间分布。

例如：

平均距离：

$$
D_t
$$

---

## 6.2 Improvement Rate

描述优化收益：

$$
I_t= \frac{f_{best}(t-k)-f_{best}(t)}{k}
$$

---

## 6.3 Directional Entropy

描述搜索方向不确定性。

高entropy：

偏探索。

低entropy：

偏开发或停滞。

---

## 6.4 Stagnation

描述连续无改善时间。

---

## 6.5 Communication Behavior

适用于群智能算法：

描述个体之间的信息传播。

---

# 7. Search Maturity

## 7.1 Definition

Search Maturity定义：

> 当前搜索过程是否已经产生足够的信息，使进一步Landscape
> Analysis具有价值。

区别：

  概念                       含义

---

  Convergence                是否接近最优
  Exploration/Exploitation   当前搜索行为
  Search Maturity            分析价值是否形成

---

# 7.2 Two-factor formulation

搜索成熟度不是单调增长。

因此定义两个因素。

## Exploration Stabilization

$$
ES_t
$$

表示探索是否由随机状态转向结构化。

---

## Exploitation Saturation

$$
XS_t
$$

表示开发是否过度。

---

定义：

$$
M_t=ES_t(1-XS_t)
$$

解释：

低成熟：

信息不足。

中成熟：

最适合分析。

高成熟：

可能已经过度收缩。

---

# 8. Decision Module

定义：

$$
\hat U=f_\theta(s_t)
$$

输入：

搜索行为状态。

输出：

ELA Utility。

训练目标：

$$
\min_{\theta}(U_{ELA}-\hat U)^2
$$

---

# 9. Algorithm

    Input:
    Black-box problem p

    1. Run cheap probing optimization

    2. Collect search behavior

    3. Extract behavior state

    4. Predict ELA utility

    5. If utility > 0:

    Execute ELA

    Run algorithm selection

    Else:

    Run default optimizer

    6. Return solution

---

# 10. Expected Contributions

1. 提出Analysis Selection Problem。
2. 提出Decision-before-Feature框架。
3. 建立搜索行为与ELA价值之间关系。
4. 提供资源感知的自动算法选择流程。
