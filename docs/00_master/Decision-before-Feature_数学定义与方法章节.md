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
s_t=[B_t,P_t]
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
r_t=\frac{FE_t}{FE_{max}}
$$

令anchor检查点a满足：

$$
r_a \le r_t-w
$$

则：

$$
I_t^{(w)}=
\frac{f_{best}(a)-f_{best}(t)}
{\max(|f_{best}(a)|,\epsilon)}
\cdot
\frac{1}{r_t-r_a}
$$

---

## 6.3 Directional Entropy

描述搜索方向不确定性。

基于FE-ratio窗口内当前population相对于anchor population的位移方向计算。

高entropy：

偏探索。

低entropy：

偏开发或停滞。

---

## 6.4 Stagnation

描述连续无改善时间。

令：

$$
r_{last}
$$

表示最近一次best-fitness严格改善对应的FE ratio，则：

$$
S_t^{(w)}=
\frac{\min(\max(r_t-r_{last},0),w)}{w}
$$

---

## 6.5 Distance Decay

描述population向当前population-best收缩的程度。

令：

$$
d_t=\frac{1}{N}\sum_i\frac{\|x_i-x_{best,t}\|_2}{\sqrt d}
$$

则：

$$
DD_t^{(w)}=
\frac{d_a-d_t}{\max(d_a,\epsilon)}
$$

其中：

$$
x_{best,t}
$$

是当前population中的best solution，不是真实全局最优。

---

## 6.6 Convergence Rate

描述种群diversity下降速度：

$$
CR_t^{(w)}=
\frac{D_a-D_t}{\max(D_a,\epsilon)}
\cdot
\frac{1}{r_t-r_a}
$$

---

## 6.7 Relation to Prior Behavioral Metrics

上述指标来自已有metaheuristic behavior analysis的启发，但不是逐式复现。

与 *How do metaheuristics exploit?* 的区别：

- 该论文指标用于late-stage exploitation behavior diagnostics/control；
- 本文指标用于预测ELA Utility；
- 该论文可使用iteration窗口、exploitation phase划分和reference optimum；
- 本文使用FE-ratio窗口，不使用真实全局最优、function identity或ELA feature。

与 *Determining Metaheuristic Similarity Using Behavioral Analysis* 的区别：

- 该论文构造whole-run behavioral footprint，用于algorithm similarity clustering；
- 本文构造checkpoint-level behavior state，用于Analysis Selection；
- 本文仅保留diversity change、improvement、convergence、distance/locality等可低成本改写的行为语义；
- 本文不使用whole-run knee-point extraction、STN locality指标、IN communication指标、DBSCAN clustering或frequency map。

论文表述应采用：

> budget-normalized behavior states inspired by prior behavioral analysis metrics

而不是：

> the same exploitation or similarity metrics as prior work.

---

## 6.8 Communication Behavior

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
