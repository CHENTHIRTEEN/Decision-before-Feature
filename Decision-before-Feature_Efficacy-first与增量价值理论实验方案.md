# Decision-before-Feature：ELA 性能功效、增量信息价值与 Efficacy-first Analysis Selection 理论及实验方案

> 版本：v1.0
> 文档性质：论文级研究规范草案
> 适用范围：连续单目标黑盒优化、运行中一次性 Landscape Query、下游算法选择与 warm-start / population transfer
> 公式规范：所有数学公式均采用标准 Markdown `$...$` 与 `$$...$$` 格式

---

## 摘要

本文将原先含义混杂的 $U_{\mathrm{ELA}}$ 拆分为三个不同层次：

1. **ELA 性能功效（ELA Efficacy）**：在相同总函数评价预算下，执行 Landscape Query 是否改善最终优化结果；
2. **ELA 相对于已有 Behavior 信息的增量价值（Incremental Value of ELA given Behavior）**：在 Behavior-only Selector 已经能够利用轨迹信息进行算法决策的前提下，额外 ELA 信息还带来多少决策收益；
3. **ELA 调用适宜度（Query Desirability）**：综合预测功效、预测不确定性、剩余预算和实际偏好后，在线策略是否应当 Query、Skip 或 Defer。

本文建议：

- **方案 A“等总 FE 的性能功效”作为第一篇论文的主科学标签**；
- **方案 B“相对于 Behavior-only Selector 的增量价值”作为关键机制验证和更强的第二层贡献**；
- **Efficacy-first Analysis Selection 作为完整方法框架**；
- wall-clock、内存和调用率继续严格记录，但主要作为独立资源维度、Pareto 评价或敏感性分析，不再强制进入唯一主标签；
- 一型或区间二型模糊控制可以作为 Efficacy-first 的可解释决策层，但不应替代共享前缀配对续跑所观测到的真实功效。

一句话概括：

> 本文不再试图用一个依赖硬件与主观权重的标量同时代表“ELA 是否有效”和“ELA 是否值得”，而是先客观测量功效，再测量相对于已有信息的增量价值，最后由可解释、可校准的策略决定是否调用。

---

# 1. 研究定位与证据等级

## 1.1 证据等级

本文采用以下标记区分文献支持与项目原创内容：

| 标记                         | 含义                                                 |
| ---------------------------- | ---------------------------------------------------- |
| **D：直接支持**        | 文献明确提出或验证了相同概念、方法或现象             |
| **M：方法学支持**      | 文献支持实验原则、统计工具或相近方法，但不是同一问题 |
| **I：思想来源**        | 文献提供可迁移的思路，本项目进行了重新定义           |
| **O：项目原创/待验证** | 当前研究提出的定义、命题或假设，必须由实验验证       |

以下内容属于 **O**：

- Analysis Selection Problem；
- Decision-before-Feature；
- ELA Efficacy 的具体定义；
- Behavior 条件下 ELA 增量价值的具体实现；
- Efficacy-first Analysis Selection；
- Search Maturity 作为分析适宜度变量；
- Skip–Defer–Query 三向分析策略；
- 本文提出的全部项目定理、命题和分解式。

## 1.2 与已有工作的边界

已有 trajectory-based / per-run algorithm selection 工作已经证明：

- 可以先运行默认优化器；
- 可以从已有轨迹中提取 ELA 或时间序列特征；
- 可以预测候选算法的后续性能；
- 可以在运行中切换一次算法并 warm-start。

因此，本项目不能把“运行中算法选择”“轨迹特征”或“一次切换”本身作为主要新颖性。

本项目的新问题是：

> 在已有低成本 Behavior 信息的条件下，是否仍值得支付额外 FE 获取独立 Landscape Query，并据此改善后续算法决策？

---

# 2. 统一符号、状态与实验基础

## 2.1 基本符号

| 符号                            | 含义                                    |
| ------------------------------- | --------------------------------------- |
| $p$                           | 黑盒优化问题实例                        |
| $d$                           | 问题维度                                |
| $B$                           | 总函数评价预算，即`FE_total`          |
| $t$                           | 当前决策机会时刻                        |
| $b_t$                         | 当前共享前缀已消耗 FE                   |
| $s_t$                         | 当前完整优化器状态                      |
| $B_t$                         | 从轨迹前缀提取的算法无关 Behavior 表示  |
| $q$                           | 固定 Landscape Query 配置               |
| $c_q$                         | Query 消耗的 FE                         |
| $\Phi_t^{(q)}$                | Query 返回的景观特征                    |
| $\mathcal A$                  | 后续候选动作/算法集合                   |
| $\pi_0$                       | No-query 策略                           |
| $\pi_B$                       | Behavior-only Selector                  |
| $\pi_{B+\Phi}$                | Behavior + Query Selector               |
| $L$                           | 越小越好的终端优化损失                  |
| $f^\star$                     | benchmark 已知最优目标值或合法参考值    |
| $\omega$                      | continuation 随机性或 semantic RNG 分支 |
| $\delta_{\mathrm{practical}}$ | 最小实际意义功效阈值                    |

## 2.2 共享前缀配对续跑

每个状态样本来自同一个完整 checkpoint：

$$
s_t=
\left(
X_t,
F_t,
\text{best}_t,
\text{optimizer internal state},
\text{RNG state}
\right).
$$

从同一个 $s_t$ 分出两条路径。

### No-query 路径

$$
\text{remaining FE}_{0}=B-b_t.
$$

策略可以是：

- 原生继续当前默认算法；
- 或 Behavior-only Selector 选择后续动作。

### Query 路径

$$
\text{remaining FE}_{q}=B-b_t-c_q.
$$

Query 先消耗 $c_q$ 次 FE，再由下游 Selector 选择后续动作。

### 配对原则

- 两条路径共享相同前缀；
- 同算法 continuation 使用完整原生状态；
- 跨算法使用é¢åæå®的 handoff 协议；
- 候选 continuation 使用语义分离但可复现的随机流；
- 所有事实字段先保存，派生功效和价值在下游计算。

## 2.3 One-shot online policy

一条 run 有候选决策机会：

$$
t=1,\ldots,T_r.
$$

给定 score $z_{r,t}$ 和阈值 $\tau$，首次触发时刻为：

$$
t_r(\tau)
=
\min
\left\{
t:z_{r,t}>\tau
\right\}.
$$

若集合为空，则整条 run 不执行 Query。

所有模型选择、阈值校准和在线评价必须采用这个 run-level first-trigger 语义，不能把每个状态行视作相互独立的 Query 决策。

---

# 3. 方案 A：等总 FE 的性能功效

## 3.1 核心问题

方案 A 不问：

> Query 花了多少程序运行时间？

它首先只问：

> 在相同总 FE 下，Query 路径是否得到更好的最终解？

这对应黑盒优化中最稳定、最可复现的固定预算问题。

Query 的函数评价成本已经通过减少后续优化 FE 进入 Query 路径，因此不需要再次用程序运行时间为其定价。

## 3.2 思路来源

### 直接与方法学来源

1. **固定预算黑盒优化评价**：BBOB/COCO 与 trajectory-based algorithm selection 均使用固定函数评价预算下的 target precision 或最终损失作为性能标准。
2. **Trajectory-based Algorithm Selection**：先使用一部分预算形成轨迹，再预测不同算法在剩余预算下的性能；论文使用 log target precision 进行回归与比较。
3. **Per-run Algorithm Selection**：以固定 A1/A2 FE 预算评价切换后的算法性能，说明“剩余预算”是算法选择价值的重要组成。
4. **DynamoRep**：指出独立特征采样会减少实际优化预算，支持将 Query FE 的机会成本直接计入最终优化结果，而不是另行重复扣除。

### 本项目原创扩展

- 将 Query 和 No-query 置于严格等总 FE 的共享前缀配对续跑中；
- 将“ELA 是否有效”定义为终端损失差；
- 将 wall-clock 从主功效标签中剥离；
- 采用 benchmark-reference gap 与 log-ratio 获得跨函数尺度稳定性。

## 3.3 终端损失定义

对于已知最优值的 benchmark：

$$
E_{\mathrm{skip}}
=
\max
\left(
f_{\mathrm{skip}}-f^\star,
0
\right),
$$

$$
E_{\mathrm{query}}
=
\max
\left(
f_{\mathrm{query}}-f^\star,
0
\right).
$$

为避免零误差导致对数不可用，定义问题尺度协变的稳定项：

$$
\epsilon_p
=
\eta
\max
\left(
E_{\mathrm{prefix}},
S_p,
\epsilon_0
\right),
$$

其中：

- $E_{\mathrm{prefix}}$ 是共享前缀 best-so-far gap；
- $S_p$ 是合法的、仅用于离线评价的问题尺度；
- $\eta$ 是预先é¢åæå®的小常数；
- $\epsilon_0$ 防止数值下溢。

推荐主功效定义：

$$
\boxed{
G_{\mathrm{FE}}(s_t,q)
=
\log
\frac{
E_{\mathrm{skip}}+\epsilon_p
}{
E_{\mathrm{query}}+\epsilon_p
}
}
$$

解释：

- $G_{\mathrm{FE}}>0$：Query 路径更好；
- $G_{\mathrm{FE}}=0$：两条路径等效；
- $G_{\mathrm{FE}}<0$：Query 路径更差；
- $G_{\mathrm{FE}}=\log 10$：Query 路径误差约改善一个数量级，忽略 $\epsilon_p$ 时成立。

可同时保存有界敏感性指标：

$$
G_{\mathrm{bounded}}
=
\frac{
E_{\mathrm{skip}}-E_{\mathrm{query}}
}{
\max
\left(
E_{\mathrm{skip}},
E_{\mathrm{query}},
\epsilon_p
\right)
}.
$$

其中：

$$
G_{\mathrm{bounded}}\in[-1,1].
$$

## 3.4 观测功效与期望功效

单次配对续跑得到：

$$
g_{r,t,\omega}^{(q)}.
$$

若每个状态执行多组 continuation 随机流，可定义条件期望功效：

$$
\bar G_{\mathrm{FE}}(s_t,q)
=
\mathbb E_{\omega}
\left[
G_{\mathrm{FE}}(s_t,q,\omega)
\mid s_t
\right].
$$

实际数据中建议同时保存：

- 每个配对 continuation 的原始功效；
- 状态级均值；
- 状态级标准误；
- 正功效概率；
- 最小实际功效概率。

## 3.5 最小实际意义标签

数学上 $G_{\mathrm{FE}}>0$ 不一定具有实际意义。

定义：

$$
Y_{\mathrm{eff}}
=
\mathbb I
\left[
G_{\mathrm{FE}}>
\delta_{\mathrm{practical}}
\right].
$$

$\delta_{\mathrm{practical}}$ 必须只用训练数据和预注册规则确定。

建议：

$$
\delta_{\mathrm{practical}}
=
\max
\left(
\delta_{\mathrm{domain}},
\delta_{\mathrm{noise}}
\right).
$$

其中：

- $\delta_{\mathrm{domain}}$：研究者预先定义的最小有意义改善；
- $\delta_{\mathrm{noise}}$：共享状态重复 continuation 所估计的随机波动边界。

一种训练集估计形式：

$$
\delta_{\mathrm{noise}}
=
Q_{1-\alpha}
\left(
\frac{
\left|
G_{\mathrm{FE}}^{(1)}
-
G_{\mathrm{FE}}^{(2)}
\right|
}{2}
\right).
$$

该阈值不能使用 BBOB validation、CEC 或工程测试问题调节。

---

## 3.6 项目定理 A1：等总 FE 内生化 Query FE 机会成本

### 定理

设 No-query 和 Query 两条路径共享总预算 $B$ 与前缀消耗 $b_t$，Query 消耗 $c_q>0$ 次函数评价，则：

$$
B_{\mathrm{noquery,opt}}=B-b_t,
$$

$$
B_{\mathrm{query,opt}}=B-b_t-c_q.
$$

若终端损失分别由以上实际预算运行得到，则 $G_{\mathrm{FE}}$ 已经包含 Query FE 的机会成本，不应再额外减去同一笔 $c_q$。

### 证明

Query 路径相对于 No-query 路径少获得 $c_q$ 次后续优化评价。任何由这 $c_q$ 次评价缺失造成的终端性能损失已经体现在 $E_{\mathrm{query}}$ 中。因此：

$$
E_{\mathrm{query}}
=
E
\left(
s_t,
\pi_q,
B-b_t-c_q
\right)
$$

已经是扣除 Query FE 后的结果。若再在功效中扣除 $\lambda_{\mathrm{FE}}c_q$，则同一机会成本同时进入终端损失和显式成本项，构成双重计费。证毕。

### 适用边界

若另设“Query 获得额外 FE、不减少优化预算”的扩展实验，则需要单独显式计入额外 FE 成本，但不得与等总 FE 主协议混用。

---

## 3.7 项目定理 A2：硬件独立性

### 定理

若同一问题、同一状态、同一 FE 预算和同一随机协议在不同硬件上产生相同的终端损失 $E_{\mathrm{skip}}$ 与 $E_{\mathrm{query}}$，则 $G_{\mathrm{FE}}$ 不随 CPU、线程数、系统负载或语言实现改变。

### 证明

$G_{\mathrm{FE}}$ 仅由：

$$
E_{\mathrm{skip}},
E_{\mathrm{query}},
\epsilon_p
$$

决定，不包含 wall-clock 变量。只要上述量保持不变，硬件变化不会改变 $G_{\mathrm{FE}}$。证毕。

### 解释

这正是方案 A 相对于时间加权 Utility 的主要优点。程序时间仍应报告，但不再决定“ELA 是否有效”的客观标签。

---

## 3.8 项目定理 A3：正仿射变换不变性

### 定理

设目标函数进行正仿射变换：

$$
f'(x)=a f(x)+b,
\qquad a>0,
$$

对应最优值：

$$
f^{\star\prime}=a f^\star+b.
$$

若稳定项同步满足：

$$
\epsilon'_p=a\epsilon_p,
$$

则：

$$
G'_{\mathrm{FE}}=G_{\mathrm{FE}}.
$$

### 证明

有：

$$
E'_{\mathrm{skip}}
=
f'_{\mathrm{skip}}-f^{\star\prime}
=
a
\left(
f_{\mathrm{skip}}-f^\star
\right)
=
aE_{\mathrm{skip}},
$$

同理：

$$
E'_{\mathrm{query}}=aE_{\mathrm{query}}.
$$

因此：

$$
G'_{\mathrm{FE}}
=
\log
\frac{
aE_{\mathrm{skip}}+a\epsilon_p
}{
aE_{\mathrm{query}}+a\epsilon_p
}
=
\log
\frac{
E_{\mathrm{skip}}+\epsilon_p
}{
E_{\mathrm{query}}+\epsilon_p
}
=
G_{\mathrm{FE}}.
$$

证毕。

### 意义

该定理避免仅因 benchmark 添加 bias 或整体缩放，ELA 功效标签就发生变化。

---

## 3.9 项目定理 A4：功效符号与性能优劣等价

### 定理

若 $\epsilon_p>0$ 固定，则：

$$
G_{\mathrm{FE}}>0
\iff
E_{\mathrm{query}}<E_{\mathrm{skip}}.
$$

### 证明

对数函数严格单调递增，因此：

$$
G_{\mathrm{FE}}>0
$$

等价于：

$$
\frac{
E_{\mathrm{skip}}+\epsilon_p
}{
E_{\mathrm{query}}+\epsilon_p
}>1,
$$

进一步等价于：

$$
E_{\mathrm{skip}}>E_{\mathrm{query}}.
$$

证毕。

---

## 3.10 项目命题 A5：配对续跑的方差优势

设两条终端损失估计为随机变量 $X$ 和 $Y$，配对差为：

$$
D=X-Y.
$$

则：

$$
\operatorname{Var}(D)
=
\operatorname{Var}(X)
+
\operatorname{Var}(Y)
-
2\operatorname{Cov}(X,Y).
$$

若共享前缀和语义配对随机流使：

$$
\operatorname{Cov}(X,Y)>0,
$$

则配对差的方差低于完全独立续跑的差值方差。

这支持：

- 共享完整 checkpoint；
- 统一 continuation 重复；
- common-random-number 风格的 semantic RNG 分支；
- 以 run/state 为重采样单位，而不是把多个 checkpoint 当作独立样本。

---

## 3.11 方案 A 的监督学习形式

### 连续回归

$$
\widehat G_{\mathrm{FE}}
=
g_\theta(B_t).
$$

### 概率分类

$$
\widehat p_t
=
P
\left(
G_{\mathrm{FE}}>
\delta_{\mathrm{practical}}
\mid B_t
\right).
$$

### 分位数回归

$$
\widehat q_\alpha(B_t)
=
Q_\alpha
\left(
G_{\mathrm{FE}}\mid B_t
\right).
$$

推荐：

- 主任务：连续功效回归；
- 主决策：有实际意义功效概率或下分位数；
- 辅助任务：正负功效分类；
- 不再把 RMSE 作为唯一模型选择标准。

---

## 3.12 方案 A 实验问题

### RQ-A1

在等总 FE 下，Landscape Query 是否存在显著的状态依赖正负功效？

报告：

- $G_{\mathrm{FE}}$ 分布；
- $P(G_{\mathrm{FE}}>0)$；
- $P(G_{\mathrm{FE}}>\delta_{\mathrm{practical}})$；
- family、dimension、stage、query 类型分层。

### RQ-A2

Behavior 是否包含可预测 $G_{\mathrm{FE}}$ 的信息？

比较：

- T0：FE ratio；
- B1：Invariant Behavior Core；
- B2：Longitudinal Dynamics；
- B3：Behavior + Readiness/Maturity。

### RQ-A3

Behavior-aware gate 是否能以较低 Query 调用率捕获大部分正功效？

主要指标：

- run-level first-trigger mean efficacy；
- Query call rate；
- helpful-call precision；
- positive-opportunity run recall；
- efficacy capture；
- harmful early-trigger cost。

### RQ-A4

结论是否跨 function family、dimension、probe algorithm 和 benchmark 泛化？

---

## 3.13 方案 A 基线

1. Never Query；
2. First-opportunity Always Query；
3. Fixed-checkpoint Query；
4. Call-rate-matched Random；
5. Trigger-time-matched Random；
6. Time-only gate；
7. Progress-only gate；
8. Behavior gate；
9. Behavior + Maturity gate；
10. 描述性 best-observed first-trigger 上界。

---

## 3.14 方案 A 统计方案

- 分析单位：run，而不是状态行；
- 同一 run 最多一次 Query；
- family-level grouped bootstrap；
- seed/instance 配对比较；
- 报告效应量和置信区间；
- 对 Proposed 与 Always Query 进行等价性分析；
- 对多个 query 类型和多个模型进行多重比较校正；
- validation 和 external test 不参与模型、阈值或 $\delta_{\mathrm{practical}}$ 选择。

---

## 3.15 方案 A 的优势与局限

### 优势

- 不依赖程序运行时间；
- 等总 FE，预算公平；
- 可跨硬件复现；
- 符合 fixed-budget 黑盒优化评价；
- 标签含义清晰；
- 易于连接 Skip–Defer–Query。

### 局限

- 不能单独描述 CPU 或内存资源；
- 已知最优值只适用于 benchmark；
- 工程问题需要 best-known 或其他参考尺度；
- 单次 continuation 仍有随机噪声；
- 若 Query 计算时间极大但 FE 很少，方案 A 可能高估实际部署价值，因此仍需独立资源评价。

---

# 4. 方案 B：相对于 Behavior-only Selector 的增量价值

## 4.1 核心问题

方案 A 比较：

```text
No-query continuation
vs
Query + Selector
```

但已有 Behavior 本身已经携带问题—算法交互信息。Trajectory-based AS、Per-run AS 和 DynamoRep 均表明轨迹信息可以用于问题识别、性能预测或算法选择。

因此更严格的问题是：

> 在 Behavior-only Selector 已经能够选择后续动作的前提下，额外 ELA 信息还增加了多少决策价值？

这就是条件增量价值。

## 4.2 两个下游策略

### Behavior-only Selector

$$
\pi_B:
(B_t,r_t)
\mapsto
a,
$$

其中 $r_t$ 为连续剩余预算比例。

### Behavior + ELA Selector

$$
\pi_{B+\Phi}:
(B_t,\Phi_t^{(q)},r_t)
\mapsto
a.
$$

两者必须：

- 使用相同动作集合；
- 使用相同 action-loss table；
- 使用相同模型容量或进行容量敏感性；
- 使用相同 function-family cross-fitting；
- 不使用当前函数族的 in-sample prediction；
- 不使用 test benchmark 调参。

## 4.3 实现实证增量价值

在等总 FE 下定义：

$$
\boxed{
V_{\Phi\mid B}^{\mathrm{real}}
=
L
\left(
\pi_B(B_t);
B-b_t
\right)
-
L
\left(
\pi_{B+\Phi}(B_t,\Phi_t);
B-b_t-c_q
\right)
}
$$

解释：

- $V_{\Phi\mid B}^{\mathrm{real}}>0$：额外 ELA 信息足以抵消 Query FE，并改善 Behavior-only 决策；
- $V_{\Phi\mid B}^{\mathrm{real}}=0$：ELA 无增量价值；
- $V_{\Phi\mid B}^{\mathrm{real}}<0$：ELA 的信息增益不足，或 Selector / handoff / FE 成本导致净伤害。

推荐名称：

```text
incremental_ela_value_given_behavior
```

避免继续用含义模糊的 `u_ela`。

---

## 4.4 理想 Expected Value of Sample Information

设候选动作损失向量为：

$$
\mathbf L
=
(L_a)_{a\in\mathcal A}.
$$

只观察 Behavior 时的 Bayes 风险：

$$
R_B^\star
=
\min_{a\in\mathcal A}
\mathbb E
\left[
L_a\mid B_t
\right].
$$

在观察 ELA 特征后：

$$
R_{B+\Phi}^\star
=
\mathbb E_{\Phi\mid B_t}
\left[
\min_{a\in\mathcal A}
\mathbb E
\left[
L_a\mid B_t,\Phi
\right]
\right].
$$

理想增量信息价值定义为：

$$
\boxed{
V_{\Phi\mid B}^{\star}
=
R_B^\star-R_{B+\Phi}^\star
}
$$

这是 ELA 版本的 Expected Value of Sample Information 思想：

> 在看到额外信息之前，预期它能使最优决策的风险降低多少？

该定义暂不包含 Query FE，用于刻画纯信息潜力。实际部署价值再扣除 Query 造成的机会成本和现实模型误差。

---

## 4.5 项目定理 B1：理想增量信息价值非负

### 定理

若动作集合在获取信息前后相同，且不存在信息获取成本，则：

$$
V_{\Phi\mid B}^{\star}\ge0.
$$

### 证明

在观察 $\Phi$ 后，决策者至少可以忽略 $\Phi$，继续使用仅依赖 $B_t$ 的最优动作。因此，更丰富信息下的最优条件风险不可能高于忽略信息的风险。

形式上，对任意固定动作 $a$：

$$
\min_{a'}
\mathbb E[L_{a'}\mid B_t,\Phi]
\le
\mathbb E[L_a\mid B_t,\Phi].
$$

对 $\Phi\mid B_t$ 取期望：

$$
\mathbb E_{\Phi\mid B_t}
\left[
\min_{a'}
\mathbb E[L_{a'}\mid B_t,\Phi]
\right]
\le
\mathbb E[L_a\mid B_t].
$$

对右侧再取最优动作：

$$
R_{B+\Phi}^\star
\le
R_B^\star.
$$

因此：

$$
V_{\Phi\mid B}^{\star}
=
R_B^\star-R_{B+\Phi}^\star
\ge0.
$$

证毕。

### 解释

如果实测增量价值为负，不表示“信息理论上具有负价值”，而通常意味着：

- Query 消耗了 FE；
- Selector 没有充分利用信息；
- 模型估计误差；
- handoff 损失；
- OOD 分布偏移；
- Query 表示不稳定。

---

## 4.6 项目定理 B2：条件冗余时增量价值为零

### 定理

若：

$$
\mathbf L
\perp
\Phi
\mid
B_t,
$$

即在给定 Behavior 后，ELA 特征与动作损失向量条件独立，则：

$$
V_{\Phi\mid B}^{\star}=0.
$$

### 证明

条件独立意味着：

$$
\mathbb E[L_a\mid B_t,\Phi]
=
\mathbb E[L_a\mid B_t]
$$

对所有 $a$ 成立。因此：

$$
R_{B+\Phi}^\star
=
\mathbb E_{\Phi\mid B_t}
\left[
\min_a
\mathbb E[L_a\mid B_t]
\right]
=
\min_a
\mathbb E[L_a\mid B_t]
=
R_B^\star.
$$

故：

$$
V_{\Phi\mid B}^{\star}=0.
$$

证毕。

### 科学意义

该定理对应本研究最关键的可证伪假设：

> ELA 是否在 Behavior 之外提供与后续动作损失相关的额外信息？

如果答案是否定的，Decision-before-Feature 的合理结论应是“多数场景无需独立 ELA”，而不是继续寻找更复杂模型来拯救预设结论。

---

## 4.7 项目定理 B3：信息嵌套下的价值单调性

### 定理

设 Query $\Phi_1$ 的信息可由更丰富 Query $\Phi_2$ 完全恢复，即：

$$
\sigma(B_t,\Phi_1)
\subseteq
\sigma(B_t,\Phi_2).
$$

在相同动作集合、相同后续预算且不计 Query 成本时：

$$
V_{\Phi_2\mid B}^{\star}
\ge
V_{\Phi_1\mid B}^{\star}.
$$

### 证明

$\Phi_2$ 提供的可用决策信息不少于 $\Phi_1$。在观察 $\Phi_2$ 后，决策者可以模拟仅使用 $\Phi_1$ 的策略，因此：

$$
R_{B+\Phi_2}^\star
\le
R_{B+\Phi_1}^\star.
$$

两边从 $R_B^\star$ 中相减，得到：

$$
V_{\Phi_2\mid B}^{\star}
\ge
V_{\Phi_1\mid B}^{\star}.
$$

证毕。

### 重要限制

真实系统中 broad Query 通常消耗更多 FE，因此：

$$
V_{\Phi_2\mid B}^{\mathrm{real}}
$$

不一定大于：

$$
V_{\Phi_1\mid B}^{\mathrm{real}}.
$$

这正是比较 cheap、standard、broad Query 时必须同时报告“纯信息潜力”和“实际等总 FE 价值”的原因。

---

## 4.8 项目定理 B4：现实增量价值分解

定义：

- $R_B^\star$：Behavior 信息下、完整剩余预算的理想 Bayes 风险；
- $R_{B+\Phi,0}^\star$：Behavior + ELA 信息下、若不扣 Query FE 时的理想 Bayes 风险；
- $R_{B+\Phi,q}^\star$：Behavior + ELA 信息下、扣除 Query FE 后的理想 Bayes 风险；
- $\mathrm{Reg}_B$：现实 Behavior-only Selector 相对于 $R_B^\star$ 的 regret；
- $\mathrm{Reg}_{B+\Phi}$：现实 Query Selector 相对于 $R_{B+\Phi,q}^\star$ 的 regret；
- $C_{\mathrm{FE}}$：Query FE 造成的理想风险增量。

令：

$$
C_{\mathrm{FE}}
=
R_{B+\Phi,q}^\star
-
R_{B+\Phi,0}^\star.
$$

理想纯信息价值：

$$
V_{\Phi\mid B}^\star
=
R_B^\star
-
R_{B+\Phi,0}^\star.
$$

现实损失：

$$
L_B^{\mathrm{real}}
=
R_B^\star+\mathrm{Reg}_B,
$$

$$
L_{B+\Phi}^{\mathrm{real}}
=
R_{B+\Phi,q}^\star
+
\mathrm{Reg}_{B+\Phi}.
$$

则：

$$
\boxed{
V_{\Phi\mid B}^{\mathrm{real}}
=
V_{\Phi\mid B}^{\star}
-
C_{\mathrm{FE}}
+
\mathrm{Reg}_B
-
\mathrm{Reg}_{B+\Phi}
}
$$

### 证明

直接代入：

$$
V_{\Phi\mid B}^{\mathrm{real}}
=
L_B^{\mathrm{real}}
-
L_{B+\Phi}^{\mathrm{real}}
$$

得到：

$$
=
R_B^\star+\mathrm{Reg}_B
-
R_{B+\Phi,q}^\star
-
\mathrm{Reg}_{B+\Phi}.
$$

又因为：

$$
R_{B+\Phi,q}^\star
=
R_{B+\Phi,0}^\star
+
C_{\mathrm{FE}},
$$

所以：

$$
V_{\Phi\mid B}^{\mathrm{real}}
=
R_B^\star-R_{B+\Phi,0}^\star
-
C_{\mathrm{FE}}
+
\mathrm{Reg}_B
-
\mathrm{Reg}_{B+\Phi},
$$

即：

$$
V_{\Phi\mid B}^{\mathrm{real}}
=
V_{\Phi\mid B}^{\star}
-
C_{\mathrm{FE}}
+
\mathrm{Reg}_B
-
\mathrm{Reg}_{B+\Phi}.
$$

证毕。

### 关键解释

该分解揭示一个容易被忽略的问题：

> 如果 Behavior-only Selector 本身很差，Query pipeline 相对于它的增量价值会被人为放大。

因此，方案 B 必须使用可信、严格 cross-fitted、模型容量公平的 Behavior-only Selector，不能故意设置一个弱基线来制造 ELA 增量价值。

---

## 4.9 方案 B 的数据生成

每个共享状态至少需要完整 action-loss matrix：

```text
state_id
prefix_algorithm
remaining_budget_ratio
loss_continue_current
loss_DE
loss_PSO
loss_CMAES
loss_SHADE
```

然后分别训练：

### Selector-B

输入：

```text
Behavior
remaining_budget_ratio
```

输出：

```text
predicted action losses
```

### Selector-B+Q

输入：

```text
Behavior
Query features
remaining_budget_ratio
```

输出：

```text
predicted action losses
```

两者都应预测每个动作的损失，而不是只预测最佳算法硬标签。

## 4.10 Cross-fitting 协议

推荐 nested grouped cross-fitting：

1. 外层按 function family 留出；
2. 内层训练 Selector-B 与 Selector-B+Q；
3. 为外层 holdout 生成动作预测；
4. 从同一 action-loss table 读取两个 Selector 实际选择动作的 observed loss；
5. 计算 $V_{\Phi\mid B}^{\mathrm{real}}$；
6. 所有 Decision Controller 标签均来自 out-of-fold 下游策略；
7. validation 和 external test 只使用完整 BBOB train 重训后的é¢åæå®模型。

不得：

- 用当前函数族拟合 Selector 后再为其自身生成标签；
- 用 validation 选择 Query 特征；
- 用 best-observed action 代替现实 Selector；
- 把 Selector 分类准确率当作最终优化价值。

---

## 4.11 方案 B 实验问题

### RQ-B1

Behavior-only Selector 是否已经具有实用算法选择能力？

比较：

- SBS；
- Behavior-only Selector；
- Query Selector；
- best-observed action 上界。

### RQ-B2

ELA 是否在 Behavior 之外提供显著增量价值？

报告：

$$
V_{\Phi\mid B}^{\mathrm{real}}
$$

的：

- 总体分布；
- 正值比例；
- family/dimension/stage 分层；
- query 类型分层；
- OOD 分层。

### RQ-B3

增量价值来自哪里？

分解：

- 理想信息潜力；
- Query FE 机会成本；
- Behavior-only Selector regret；
- Query Selector regret；
- handoff 后 observed loss。

### RQ-B4

更广的 Query 是否提供更多纯信息，但未必提供更高实际价值？

比较：

```text
trajectory descriptor
descriptor_cheap
pflacco_standard
pflacco_broad
```

### RQ-B5

Behavior 与 ELA 的互补性是否跨 benchmark 保持？

---

## 4.12 方案 B 结果解释矩阵

| $G_{\mathrm{FE}}$ | $V_{\Phi\mid B}^{\mathrm{real}}$ | 解释                                                                |
| ------------------: | ---------------------------------: | ------------------------------------------------------------------- |
|              $>0$ |                             $>0$ | Query 相对简单 continuation 有效，且相对 Behavior-only 仍有增量价值 |
|              $>0$ |                           $\le0$ | Query pipeline 有效，但收益可由 Behavior-only Selector替代          |
|            $\le0$ |                             $>0$ | 理论上不应稳定大量出现，应检查两种基线预算与定义是否一致            |
|            $\le0$ |                           $\le0$ | Query 无实际价值或有害                                              |

另一个重要矩阵：

| $V_{\Phi\mid B}^{\star}$ | $V_{\Phi\mid B}^{\mathrm{real}}$ | 解释                                                                 |
| -------------------------: | ---------------------------------: | -------------------------------------------------------------------- |
|                         高 |                                 高 | ELA 有信息且现实 Selector 能利用                                     |
|                         高 |                              低/负 | ELA 有潜力，但 Query FE 或 Selector regret 吞噬收益                  |
|                         低 |                                 低 | Behavior 已包含主要决策信息                                          |
|                         低 |                                 高 | 可能由 Behavior-only Selector 过弱造成虚高，应诊断$\mathrm{Reg}_B$ |

---

## 4.13 方案 B 的优势与风险

### 优势

- 理论问题更精确；
- 直接回应“既然已有 Behavior，为何还需要 ELA？”；
- 与 Active Feature Acquisition、EVSI 和 cost-sensitive decision making 接轨；
- 可解释不同 Query 的增量信息；
- 有利于形成 TEVC 级方法论贡献。

### 风险

- 实验链条更复杂；
- 需要两套公平 Selector；
- Selector 容量差异会污染增量价值；
- cross-fitting 成本高；
- Behavior-only Selector 若过强，ELA 价值可能很低，但这仍是有意义的科学结论。

---

# 5. 研究方案一：Efficacy-first Analysis Selection

## 5.1 核心思想

Efficacy-first 不再先定义：

$$
U=G-\lambda_T C_T.
$$

而是按顺序处理：

```text
第一层：测量等总 FE 下的客观性能功效 G_FE
第二层：预测 G_FE 的分布与不确定性
第三层：根据最小实际效应和风险偏好决定 Query / Skip / Defer
第四层：独立报告 runtime、memory、call rate 和 Pareto
```

其核心假设是：

$$
\boxed{
\text{Low-cost search behavior contains information about the future efficacy of acquiring landscape information.}
}
$$

## 5.2 方法结构

```text
Shared optimization prefix
        ↓
Permutation-invariant Behavior B_t
        ↓
Efficacy distribution model
        ↓
Predicted gain + uncertainty
        ↓
Skip / Defer / Query
        ↓
If Query:
    Landscape Query
    → downstream selector
    → continue/switch
Else:
    continue without Query
```

## 5.3 预测目标

### 条件均值

$$
\mu_t
=
\mathbb E
\left[
G_{\mathrm{FE}}
\mid B_t
\right].
$$

### 条件分位数

$$
q_{\alpha,t}
=
Q_\alpha
\left(
G_{\mathrm{FE}}\mid B_t
\right).
$$

### 有意义功效概率

$$
p_t
=
P
\left(
G_{\mathrm{FE}}>
\delta_{\mathrm{practical}}
\mid B_t
\right).
$$

推荐同时输出：

```text
predicted_mean_efficacy
predicted_lower_quantile
predicted_upper_quantile
probability_meaningful_gain
```

## 5.4 三向决策规则

设预测区间为：

$$
[\ell_t,u_t].
$$

定义：

$$
d_t=
\begin{cases}
\text{Query},
&
\ell_t>
\delta_{\mathrm{practical}},
\\[6pt]
\text{Skip},
&
u_t<0,
\\[6pt]
\text{Defer},
&
\text{otherwise}.
\end{cases}
$$

解释：

- **Query**：即使保守估计仍有实际正功效；
- **Skip**：即使乐观估计也无正功效；
- **Defer**：证据不足，继续低成本 Probe，在下一个 decision opportunity 再判断。

这与当前一次性 first-trigger 机制兼容：

- Defer 不消耗 Query；
- 一旦 Query，后续不再决策；
- Skip 可以是当前机会 Skip，而非整条 run 永久 Skip；
- 到最后机会仍 Defer 时，默认 No-query。

## 5.5 Practical equivalence band

定义两个边界：

$$
\delta_{\mathrm{skip}}\le0,
$$

$$
\delta_{\mathrm{query}}>0.
$$

可取：

$$
\delta_{\mathrm{skip}}=0,
$$

$$
\delta_{\mathrm{query}}
=
\delta_{\mathrm{practical}}.
$$

中间区域：

$$
[\delta_{\mathrm{skip}},\delta_{\mathrm{query}}]
$$

是“效果不明确或不足以支持昂贵动作”的模糊/等价区。

这比简单硬标签：

$$
G_{\mathrm{FE}}>0
$$

更稳健。

---

## 5.6 项目定理 E1：完美预测下阈值策略的最优性

### 定理

设在当前状态只有两个动作：

- No-query 的相对收益为 $0$；
- Query 的相对收益为 $G_{\mathrm{FE}}-\delta_{\mathrm{practical}}$。

若 $G_{\mathrm{FE}}$ 可被完美观测，则最优决策为：

$$
\text{Query}
\iff
G_{\mathrm{FE}}>
\delta_{\mathrm{practical}}.
$$

### 证明

若：

$$
G_{\mathrm{FE}}-\delta_{\mathrm{practical}}>0,
$$

Query 收益高于 No-query 的 $0$，故选择 Query。

若：

$$
G_{\mathrm{FE}}-\delta_{\mathrm{practical}}\le0,
$$

No-query 不劣于 Query，故选择 No-query。证毕。

### 解释

$\delta_{\mathrm{practical}}$ 不是程序时间权重，而是“至少要好到什么程度才值得改变流程”的实际意义边界。

---

## 5.7 项目定理 E2：有效预测区间下的保守 Query 风险界

### 定理

设预测区间满足边际覆盖：

$$
P
\left(
G_{\mathrm{FE}}\in[\ell_t,u_t]
\right)
\ge
1-\alpha.
$$

采用规则：

$$
\text{Query}
\iff
\ell_t>
\delta_{\mathrm{practical}}.
$$

则“触发 Query 但真实功效不超过阈值”的事件概率满足：

$$
P
\left(
\ell_t>
\delta_{\mathrm{practical}},
G_{\mathrm{FE}}\le
\delta_{\mathrm{practical}}
\right)
\le
\alpha.
$$

### 证明

若：

$$
\ell_t>
\delta_{\mathrm{practical}}
\ge
G_{\mathrm{FE}},
$$

则必有：

$$
G_{\mathrm{FE}}<\ell_t,
$$

即真实值落在预测区间之外。因此该事件是区间不覆盖事件的子集，其概率不超过 $\alpha$。证毕。

### 重要限制

- 这是边际保证，不等于每个 function family 的条件保证；
- 如果校准集与测试分布发生严重 OOD，保证可能失效；
- run 内多个相关 decision opportunities 需要使用 run-level 或 trajectory-level conformal 设计；
- validation/test 不得用于重新校准区间。

---

## 5.8 项目定理 E3：Defer 区域的一致收缩

### 定理

假设随搜索进行：

1. 功效预测区间宽度满足：

$$
u_t-\ell_t\rightarrow0;
$$

2. 区间中心一致收敛到真实功效；
3. 真实功效不恰好等于决策边界：

$$
G_{\mathrm{FE}}
\notin
\left\{
0,
\delta_{\mathrm{practical}}
\right\}.
$$

则随着信息积累，策略最终以概率趋近 1 离开 Defer 区域，并作出正确的 Query 或 Skip 决策。

### 证明思路

当区间宽度趋于 0 且中心趋于真实值时：

- 若 $G_{\mathrm{FE}}>\delta_{\mathrm{practical}}$，最终下界也超过 $\delta_{\mathrm{practical}}$；
- 若 $G_{\mathrm{FE}}<0$，最终上界也低于 $0$；
- 只有真实值位于边界或等价带时，Defer 可能长期存在。

证毕。

### 科学意义

Defer 不是含糊其辞，而是一种合理的信息累积策略。其有效性必须通过“区间是否随轨迹增长而收缩”进行实证验证。

---

## 5.9 项目命题 E4：模型与阈值必须按 run-level first-trigger 目标选择

对 run $r$，策略贡献为：

$$
C_r(\tau)
=
\begin{cases}
G_{r,t_r(\tau)},
&
\text{若触发},
\\
0,
&
\text{否则}.
\end{cases}
$$

正式模型选择目标应为：

$$
\boxed{
J(\theta,\tau)
=
\frac{1}{R}
\sum_{r=1}^{R}
C_r(\tau)
}
$$

而不是：

$$
\frac{1}{N}
\sum_{r,t}
G_{r,t}
\mathbb I
\left[
z_{r,t}>\tau
\right].
$$

后者允许同一 run 被重复 Query，并让状态更多的 run 获得更大权重，与真实 one-shot 部署不一致。

---

# 6. Efficacy-first 的模型设计

## 6.1 第一阶段建议保持简单

主候选：

- Ridge/Elastic Net：连续功效回归；
- Logistic Regression：有意义正功效概率；
- LDA：稀疏正功效状态的线性判别；
- Gradient Boosting：仅作非线性上界；
- Quantile Regression / Quantile Gradient Boosting：上下分位数；
- Conformal wrapper：预测区间与拒绝/Defer。

模型创新不是论文主角。模型选择依据是：

- run-level first-trigger mean efficacy；
- harmful-call cost；
- Query call rate；
- efficacy capture；
- OOD stability。

## 6.2 多任务形式

可以联合预测：

$$
\widehat G_{\mathrm{FE}},
$$

$$
\widehat P
\left(
G_{\mathrm{FE}}>
\delta_{\mathrm{practical}}
\right),
$$

$$
\widehat{\sigma}_G
$$

或上下分位数。

联合目标示例：

$$
\mathcal L
=
\mathcal L_{\mathrm{reg}}
+
\beta_1
\mathcal L_{\mathrm{cls}}
+
\beta_2
\mathcal L_{\mathrm{quantile}}.
$$

第一版不建议直接使用复杂深度网络。先证明问题结构，再考虑模型容量。

---

# 7. 与一型/二型模糊控制的接口

## 7.1 迁移 Roy 等人的思想

Roy、Beauthier 与 Mayer 的 Fuzzy PSO 工作提供以下可迁移原则：

1. 模糊控制输入应是少量、非冗余、具有明确行为意义的 probes；
2. 对原始 fitness 使用 rank、clamping 或归一化，避免尺度问题；
3. 使用简单 membership functions；
4. 使用 Takagi–Sugeno–Kang 加权平均降低推理开销；
5. 规则和隶属函数不完全依赖专家手写，而是在训练 benchmark 上系统优化；
6. 训练、验证和真实问题测试应分离；
7. 控制器复杂度增加会带来过拟合风险。

这些原则可以用于 Efficacy-first 的决策层，但不能替代真实 $G_{\mathrm{FE}}$ 标签。

## 7.2 一型模糊 Efficacy Gate

推荐最多使用 4 个输入：

$$
x_1=\widehat G_{\mathrm{FE}},
$$

$$
x_2=u_t-\ell_t,
$$

$$
x_3=\text{remaining FE ratio},
$$

$$
x_4=\text{maturity or selector disagreement}.
$$

输出：

$$
D_{\mathrm{query}}\in[0,1].
$$

示例规则：

```text
IF predicted efficacy is HIGH
AND uncertainty is LOW
AND remaining budget is HIGH
THEN query desirability is HIGH

IF predicted efficacy is MEDIUM
AND uncertainty is HIGH
THEN query desirability is MEDIUM

IF predicted efficacy is LOW
OR remaining budget is LOW
THEN query desirability is LOW
```

决策：

$$
\begin{cases}
\text{Query},
&
D_{\mathrm{query}}\ge\eta_H,
\\
\text{Skip},
&
D_{\mathrm{query}}\le\eta_L,
\\
\text{Defer},
&
\eta_L<D_{\mathrm{query}}<\eta_H.
\end{cases}
$$

## 7.3 模糊软标签

还可以直接把真实功效转成软隶属度：

$$
\mu_{\mathrm{query}}(G)
=
\begin{cases}
0,
&
G\le\delta_0,
\\[4pt]
\dfrac{G-\delta_0}{\delta_1-\delta_0},
&
\delta_0<G<\delta_1,
\\[8pt]
1,
&
G\ge\delta_1.
\end{cases}
$$

其中：

- $\delta_0$：可忽略功效；
- $\delta_1$：明确实际有效功效。

这比简单地把 $G=0.0001$ 与 $G=-0.0001$ 强制分为两个类别更稳健。

## 7.4 区间二型模糊何时值得使用

只有当训练数据表明以下现象时，再引入 Interval Type-2：

- 不同 family folds 学到的 membership 边界变化明显；
- OOD 状态大量落在 Type-1 边界附近；
- Query/Skip 规则对维度或 benchmark 高度敏感；
- Type-1 harmful-call cost 较高；
- 隶属度定义本身存在稳定的区间不确定性。

可以在每个外层 fold 学习边界：

$$
\delta_0^{(k)},
\delta_1^{(k)},
$$

再用分位数形成 Footprint of Uncertainty：

$$
\delta_j
\in
\left[
Q_{0.1}
\left(
\delta_j^{(k)}
\right),
Q_{0.9}
\left(
\delta_j^{(k)}
\right)
\right].
$$

区间二型模糊的区间不是概率置信区间。建议：

- conformal interval 负责统计不确定性；
- fuzzy system 负责决策语义；
- 两者不要混为同一概念。

---

# 8. Efficacy-first 完整实验方案

## 8.1 Phase 0：最小真实 Pilot

范围建议：

```text
BBOB: F1, F3, F15, F24
D = 10
instances = 1
seeds = 1,2,3
algorithms = DE, PSO, CMA-ES, SHADE
```

检查：

- optimizer state continuation；
- action-loss matrix；
- $f^\star$ gap 计算；
- 正仿射变换不变性；
- Behavior permutation invariance；
- Query feature preprocessing；
- run-level first-trigger 一致性；
- Selector-B 与 Selector-B+Q cross-fitting；
- $G_{\mathrm{FE}}$ 和 $V_{\Phi\mid B}$ 可重算性。

## 8.2 Phase 1：BBOB 内部论文实验

### Train

- 18 个 function families；
- 10D、20D、40D；
- instances 1–3；
- seeds 1–30。

### Validation

- 完全未见的 6 个 function families；
- 相同维度；
- 不用于模型、阈值、$\delta$ 或模糊规则选择。

### Query 类型

1. trajectory descriptor，0 额外 FE；
2. independent descriptor cheap；
3. pflacco standard；
4. pflacco broad。

### 决策机会

- é¢åæå®的 milestone + event opportunities；
- milestone-only 消融；
- equal-count fixed opportunities；
- 每条 run 最多一次 Query。

## 8.3 Phase 2：外部泛化

顺序建议：

1. CEC2017；
2. CEC2022；
3. MA-BBOB；
4. Nevergrad/YABBOB 类型问题。

外部测试只使用 BBOB train 最终模型，不重调：

- Behavior normalization；
- Selector；
- Controller；
- conformal calibration；
- fuzzy membership；
- threshold；
- decision schedule。

## 8.4 Phase 3：有限预算/昂贵优化

预算：

$$
B\in
\{100D,300D,500D,1000D\}.
$$

Query FE：

$$
c_q\in
\{5D,10D,20D,50D\}.
$$

研究：

- Query FE 占比增加时功效如何变化；
- cheap/standard/broad Query 的 break-even 区域；
- Efficacy-first 是否自动减少昂贵 Query；
- Behavior-only 是否在极低预算下更占优势。

不要只用 `sleep()` 模拟昂贵函数作为主证据。可将其作为 wall-clock accounting sanity check。

## 8.5 Phase 4：工程问题

选择 3–5 个可复现问题，要求：

- black-box；
- 评估成本明确；
- 有合法 best-known/reference；
- 预算可é¢åæå®；
- 约束处理可统一。

工程问题只用于外部验证，不参与模型或模糊规则训练。

---

# 9. 完整基线矩阵

| 基线                           | 额外 Query FE | 是否用 Behavior | 是否用 ELA |
| ------------------------------ | ------------: | --------------: | ---------: |
| Uninterrupted SBS              |             0 |              否 |         否 |
| Traditional AAS at start       |          固定 |              否 |         是 |
| After-probe Always Query       |          固定 |   仅生成 prefix |         是 |
| Fixed-checkpoint Query         |          固定 |           否/弱 |         是 |
| Call-rate-matched Random       |          匹配 |              否 |       条件 |
| Trigger-time-matched Random    |          匹配 |              否 |       条件 |
| Behavior-only Selector         |             0 |              是 |         否 |
| Trajectory-descriptor Selector |             0 |              是 | 轨迹描述符 |
| Efficacy-first Controller      |          条件 |              是 |       条件 |
| Fuzzy Efficacy Gate            |          条件 |              是 |       条件 |
| Best-observed first-trigger    |          诊断 |            全知 |       诊断 |

---

# 10. 核心指标

## 10.1 功效指标

- mean $G_{\mathrm{FE}}$；
- median $G_{\mathrm{FE}}$；
- $P(G_{\mathrm{FE}}>0)$；
- $P(G_{\mathrm{FE}}>\delta_{\mathrm{practical}})$；
- log-regret reduction；
- final performance；
- anytime AUC。

## 10.2 Run-level gate 指标

对 run $r$：

$$
G_r^{+}
=
\max
\left(
0,
\max_t G_{r,t}
\right).
$$

### Call rate

$$
\mathrm{CallRate}
=
\frac{
\#\{\text{called runs}\}
}{
R
}.
$$

### Helpful precision

$$
\mathrm{Precision}
=
\frac{
\#\{r:\text{called and }G_{r,t_r}>\delta\}
}{
\#\{\text{called runs}\}
}.
$$

### Positive opportunity recall

$$
\mathrm{Recall}
=
\frac{
\#\{r:\text{called and }G_{r,t_r}>\delta\}
}{
\#\{r:G_r^{+}>\delta\}
}.
$$

### Efficacy capture

$$
\mathrm{Capture}
=
\frac{
\sum_r
\max
\left(
G_{r,t_r},
0
\right)
}{
\sum_r
G_r^{+}
}.
$$

### Mean first-trigger efficacy

$$
J
=
\frac{1}{R}
\sum_r
\begin{cases}
G_{r,t_r},
&
\text{called},
\\
0,
&
\text{not called}.
\end{cases}
$$

## 10.3 增量价值指标

- mean $V_{\Phi\mid B}^{\mathrm{real}}$；
- positive incremental value rate；
- ideal information value proxy；
- Selector-B regret；
- Selector-B+Q regret；
- Query FE opportunity cost；
- increment capture。

## 10.4 独立资源指标

- Query FE；
- Query call rate；
- query sampling runtime；
- feature computation runtime；
- selection runtime；
- handoff runtime；
- end-to-end net wall-clock；
- peak RSS；
- Pareto front。

这些指标独立报告，不要求用固定 $\lambda_T$ 压成唯一标签。

---

# 11. 消融实验

## 11.1 Behavior 消融

```text
T0: FE ratio
B1: invariant behavior
B2: longitudinal dynamics
B3: behavior + readiness/maturity
```

## 11.2 Query 消融

```text
trajectory descriptor
descriptor cheap
pflacco standard
pflacco broad
```

## 11.3 Decision 消融

```text
hard threshold
probability threshold
quantile lower bound
conformal interval
type-1 fuzzy
interval type-2 fuzzy
```

## 11.4 Opportunity 消融

```text
milestones only
milestones + events
equal-count fixed grid
dense grid
```

## 11.5 Selector 消融

```text
SBS
Behavior-only
Query-only
Behavior + Query
best-observed action
```

## 11.6 Maturity 消融

```text
without maturity
maturity only
behavior + maturity
matched nonlinear interaction baseline
```

---

# 12. 数据契约建议

原始数据表应保存事实，而不是只保存某个最终 Utility。

## 12.1 必存事实

```text
state key
complete prefix metadata
query_id
query_fe
remaining_fe

raw final objective:
p_noquery_raw
p_behavior_selector_raw
p_behavior_query_selector_raw
p_best_observed_raw

benchmark_reference_value
loss_noquery
loss_behavior_selector
loss_behavior_query_selector
loss_best_observed

all action losses
selected action by each selector
selector predictions
selector margins / entropy

runtime components
memory components

behavior fields
query feature fields
```

## 12.2 下游派生字段

```text
g_fe_log
g_fe_bounded
meaningful_efficacy_label

incremental_value_given_behavior
ideal_information_value_proxy
selector_regret_behavior
selector_regret_query
query_fe_opportunity_cost

conformal lower/upper bounds
query desirability
fuzzy memberships
```

## 12.3 禁止进入 Decision X

- function ID；
- family；
- algorithm ID；
- benchmark reference value；
- action losses；
- ELA features；
- selected action；
- VBS/best-observed action；
- future information；
- Utility/efficacy label；
- wall-clock target。

---

# 13. GO / NO-GO 条件

## 13.1 方案 A 继续推进条件

至少满足：

1. $G_{\mathrm{FE}}$ 存在稳定正负异质性；
2. B1/B2/B3 明显优于 Time-only；
3. run-level first-trigger mean efficacy 为正；
4. Proposed 相对 Always Query 大幅降低调用率；
5. Proposed 与 Always Query 在终端性能上等价或更优；
6. held-out families 和至少一个 external suite 上成立。

## 13.2 方案 B 继续推进条件

至少满足：

1. Behavior-only Selector 优于 SBS；
2. Behavior + Query Selector 在严格 OOF 下进一步降低 action loss；
3. $V_{\Phi\mid B}^{\mathrm{real}}$ 在非平凡比例状态中为正；
4. 增量价值不是由弱 Behavior baseline 人为制造；
5. cheap/standard/broad Query 的价值—成本关系具有可解释结构。

## 13.3 模糊控制升级条件

Type-1 Fuzzy 进入主比较的条件：

- 少量 probes 足以表达决策；
- 规则跨 family 稳定；
- 性能不显著落后于统计模型；
- 解释性和 harmful-call 控制有明显优势。

Interval Type-2 进入正式扩展的条件：

- Type-1 membership 边界在 folds 间明显不稳定；
- OOD 时边界不确定性显著；
- 区间二型在 external test 上改善风险—覆盖曲线；
- 额外复杂度具有可测收益。

---

# 14. 三者关系与最终推荐

| 层次                        | 核心问题                                   | 推荐地位                        |
| --------------------------- | ------------------------------------------ | ------------------------------- |
| 方案 A：$G_{\mathrm{FE}}$ | ELA 在等总 FE 下是否改善结果？             | **第一篇论文主标签**      |
| 方案 B：$V_{\Phi\mid B}$  | ELA 相对已有 Behavior 是否有增量信息？     | **关键机制验证/第二贡献** |
| Efficacy-first              | 如何根据预测功效和不确定性在线调用？       | **完整方法框架**          |
| Fuzzy Gate                  | 如何形成可解释的 Skip–Defer–Query 规则？ | 增强方法或后续扩展              |
| Runtime/Pareto              | 是否在实际资源上划算？                     | 独立评价，不强制进入主标签      |

最终推荐流程：

```text
Offline:
shared-prefix paired continuation
        ↓
calculate G_FE
        ↓
train Behavior efficacy model
        ↓
train Behavior-only and Behavior+Query selectors
        ↓
calculate V_{Phi|B}
        ↓
calibrate uncertainty and first-trigger policy

Online:
observe Behavior
        ↓
predict efficacy distribution
        ↓
Skip / Defer / Query
        ↓
if Query:
    obtain ELA
    call downstream selector
    continue/switch
```

---

# 15. 推荐论文贡献表述

## Contribution 1：问题定义

提出 Analysis Selection Problem，将“是否获取景观信息”从默认预处理步骤转化为一个状态条件决策问题。

## Contribution 2：功效定义

提出硬件无关、等总 FE 的 ELA Efficacy：

$$
G_{\mathrm{FE}}.
$$

并通过共享前缀配对续跑建立离线监督标签。

## Contribution 3：增量信息价值

提出：

$$
V_{\Phi\mid B},
$$

用于衡量 ELA 在已有搜索行为信息之上的增量决策价值，并给出信息潜力、Query FE 与 Selector regret 分解。

## Contribution 4：Efficacy-first Controller

构建预测功效分布与不确定性的 Skip–Defer–Query 控制器，并在 run-level first-trigger 语义下进行训练、校准和评价。

## Contribution 5：广泛验证

在 function-family OOD、dimension OOD、cross-probe、cross-benchmark、有限预算和工程问题上验证结论。

---

# 16. 推荐标题候选

## 主论文标题候选 1

**Decision-before-Feature: Efficacy-First Landscape Analysis Selection in Black-Box Optimization**

## 主论文标题候选 2

**When Is Landscape Information Worth Acquiring? Behavior-Aware Efficacy Prediction for Automated Algorithm Selection**

## 主论文标题候选 3

**Beyond Always-On ELA: State-Conditioned Incremental Value of Landscape Information in Black-Box Optimization**

---

# 17. 实施路线

## Step 1：é¢åæå®客观功效

- 使用 benchmark-reference gap；
- 实现 $G_{\mathrm{FE}}$；
- 验证平移/缩放不变性；
- 取消 `lamT=1` 作为唯一主标签。

## Step 2：重建下游 Selector 对照

- Behavior-only Selector；
- Behavior + Query Selector；
- 相同 action-loss table；
- nested family cross-fitting。

## Step 3：生成两类标签

```text
g_fe
incremental_value_given_behavior
```

## Step 4：训练 Efficacy-first 模型

- regression；
- meaningful-gain classification；
- quantile/conformal interval；
- first-trigger threshold。

## Step 5：一型模糊增强

- 选 3–4 个 probes；
- TSK 输出 desirability；
- train-only meta-optimization；
- 与 Logistic/Quantile/Conformal 比较。

## Step 6：外部验证

- BBOB held-out families；
- CEC2017/2022；
- MA-BBOB；
- limited FE；
- engineering cases。

---

# 18. 参考文献与思路来源

## [R1] Exploratory Landscape Analysis

Mersmann, O., Bischl, B., Trautmann, H., Preuss, M., Weihs, C., & Rudolph, G. (2011). *Exploratory Landscape Analysis*. GECCO 2011, 829–836. DOI: 10.1145/2001576.2001690.

**支持内容**：ELA 的基本范式和数值景观描述。
**不直接支持**：本文的 ELA Efficacy、增量价值或 Analysis Selection。

## [R2] Cost-sensitive ELA-based Algorithm Selection

Bischl, B., Mersmann, O., Trautmann, H., & Preuss, M. (2012). *Algorithm Selection Based on Exploratory Landscape Analysis and Cost-Sensitive Learning*. GECCO 2012, 313–320. DOI: 10.1145/2330163.2330209.

**支持内容**：算法选择可以考虑特征/运行成本。
**不支持**：任何固定 $\lambda_T=1$，也不直接定义本项目 Query Utility。

## [R3] Automated Algorithm Selection Survey

Kerschke, P., Hoos, H. H., Neumann, F., & Trautmann, H. (2019). *Automated Algorithm Selection: Survey and Perspectives*. Evolutionary Computation, 27(1), 3–45. DOI: 10.1162/evco_a_00242.

**支持内容**：AAS、SBS、VBS、特征—算法映射背景。

## [R4] Trajectory-based Algorithm Selection with Warm-starting

Jankovic, A., Vermetten, D., Kostovska, A., de Nobel, J., Eftimov, T., & Doerr, C. (2022). *Trajectory-based Algorithm Selection with Warm-starting*. IEEE CEC 2022. DOI: 10.1109/CEC55065.2022.9870222.

**支持内容**：

- 独立特征采样会产生计算与 FE 开销；
- 可以利用已有轨迹计算 landscape features；
- 可以预测候选算法固定预算性能；
- log target precision 比原始 target 更适合性能回归；
- 可以运行中切换一次并 warm-start；
- 固定切换点的自适应化是后续方向。

## [R5] Per-run Algorithm Selection with Warm-Starting

Kostovska, A., Jankovic, A., Vermetten, D., de Nobel, J., Wang, H., Eftimov, T., & Doerr, C. (2022). *Per-run Algorithm Selection with Warm-Starting Using Trajectory-Based Features*. PPSN 2022, LNCS 13398, 46–60. DOI: 10.1007/978-3-031-14714-2_4.

**支持内容**：

- 从默认算法轨迹提取 ELA/时间序列特征；
- 预测后续算法固定预算表现；
- Behavior/trajectory 信息可以用于在线 per-run AS；
- ELA 与算法内部时间序列可能互补；
- 跨 benchmark 表示覆盖不足会导致泛化下降。

## [R6] DynamoRep

Cenikj, G., Petelin, G., Doerr, C., Korošec, P., & Eftimov, T. (2023). *DynamoRep: Trajectory-Based Population Dynamics for Classification of Black-box Optimization Problems*. arXiv:2306.05438.

**支持内容**：

- 轨迹统计包含问题—算法交互信息；
- 简单纵向 population statistics 可低成本提取；
- 独立特征采样减少实际优化预算；
- 把所有轨迹点静态混合会丢失 longitudinality；
- 轨迹表示可成为 Behavior-only Selector 的基础。

## [R7] Fuzzy Hyperparameter Control

Roy, N., Beauthier, C., & Mayer, A. (2025). *Hyperparameter Control Using Fuzzy Logic: Evolving Policies for Adaptive Fuzzy Particle Swarm Optimization Algorithm*. Evolutionary Computation, 33(2), 279–308. DOI: 10.1162/evco_a_00353.

**支持内容**：

- 使用少量 optimizer probes 构造模糊反馈；
- probes 应非冗余、稳定、归一化；
- TSK/Sugeno 推理可低成本实现；
- 规则和 membership functions 可通过 benchmark-driven meta-optimization 学习；
- 控制器复杂度过高可能过拟合；
- 训练、验证、现实问题应分开。

**不直接支持**：用模糊逻辑定义 ELA 的真实功效；本文只能借鉴其控制结构。

## [R8] Acquisition Conditioned Oracle

Valancius, M., Lennon, M., & Oliva, J. (2024). *Acquisition Conditioned Oracle for Nongreedy Active Feature Acquisition*. ICML 2024, PMLR 235, 48957–48975.

**支持内容**：

- Active Feature Acquisition 是逐实例、序贯获取额外信息的问题；
- 获取成本应与预测/决策改善权衡；
- 非贪心信息获取可能优于简单逐特征贪心。

## [R9] Active-Acquisition POMDP

Li, Y., & Oliva, J. (2025). *Towards Cost Sensitive Decision Making*. AISTATS 2025, PMLR 258, 3601–3609.

**支持内容**：

- 决策者可在部分观测状态下主动获取额外信息；
- 信息获取与任务决策可形成分层策略；
- 成本敏感信息获取可以作为序贯决策问题。

## [R10] Stochastic Encodings for Active Feature Acquisition

Norcliffe, A. L. I., Lee, C., Imrie, F., van der Schaar, M., & Lio, P. (2025). *Stochastic Encodings for Active Feature Acquisition*. ICML 2025, PMLR 267, 46784–46814.

**支持内容**：

- AFA 是实例级序贯决策；
- 可对未观测信息的多种可能实现进行推理；
- 简单贪心信息增益可能是短视的。

## [R11] Expected Value of Sample Information

Ades, A. E., Lu, G., & Claxton, K. (2004). *Expected Value of Sample Information Calculations in Medical Decision Modeling*. Medical Decision Making, 24(2). DOI: 10.1177/0272989X04263162.

**支持内容**：

- 新信息的价值应由信息到来前后最优决策的期望收益差定义；
- EVSI 属于 preposterior decision analysis；
- 本项目将其思想迁移到 ELA 信息获取。

## [R12] Conformal Regression with Reject Option

Johansson, U., Sönströd, C., & Boström, H. (2024). *Conformal Regression with Reject Option*. PMLR 230, 277–294.

**支持内容**：

- 回归模型可以在不确定时拒绝作出预测；
- conformal interval 可与 coverage/rejection 机制结合；
- 支持 Efficacy-first 中的 Defer 思路。

## [R13] Selective Regression

Shah, A., Bu, Y., Lee, J. K., Das, S., Panda, R., Sattigeri, P., & Wornell, G. W. (2022). *Selective Regression under Fairness Criteria*. ICML 2022, PMLR 162, 19598–19615.

**支持内容**：

- selective regression 通过降低 coverage 换取更可靠预测；
- 支持将“不确定时不决策”视为正式学习问题。

## [R14] COCO

Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., & Brockhoff, D. (2020). *COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting*. Optimization Methods and Software, 36, 114–144. DOI: 10.1080/10556788.2020.1808977.

**支持内容**：

- FE 是连续黑盒优化的核心成本尺度；
- 支持 fixed-budget、运行轨迹和配对 benchmark 评价。

## [R15] Sampling Sensitivity of ELA

Renau, Q., Doerr, C., Dreo, J., & Doerr, B. (2020). *Exploratory Landscape Analysis Is Strongly Sensitive to the Sampling Strategy*. PPSN 2020, 139–153. DOI: 10.1007/978-3-030-58115-2_10.

**支持内容**：

- ELA 特征依赖采样策略和样本规模；
- Query 配置必须版本化、é¢åæå®并进行敏感性分析。

---

# 19. 最终裁决

当前最稳妥的论文级路线是：

1. **主标签从时间加权 $U_{\mathrm{ELA}}$ 改为等总 FE 的 $G_{\mathrm{FE}}$；**
2. **把 runtime 从标签必选项改为独立资源维度；**
3. **实现 Behavior-only Selector，计算 $V_{\Phi\mid B}$；**
4. **Controller 预测功效分布，而不是只预测一个混合 Utility；**
5. **使用 Skip–Defer–Query，按 run-level first-trigger 评价；**
6. **一型 TSK 模糊控制作为可解释增强；**
7. **只有在数据证明 membership 边界不稳定后才使用区间二型模糊。**

这一路线能够同时回答三个层次的问题：

$$
\text{ELA 是否有效？}
$$

$$
\text{ELA 是否提供 Behavior 之外的增量信息？}
$$

$$
\text{当前状态下是否应当执行 ELA？}
$$

三者不再被迫塞进同一个依赖硬件和任意权重的 $U_{\mathrm{ELA}}$ 中。
