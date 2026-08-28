# Decision-before-Feature 新讨论补充：算法池、10D Benchmark 体系与 CEC 测评设计

> 文档性质：对前一版《Decision-before-Feature_行为景观协同动态算法选择_改进方案.md》的补充
> 
> 目标：将近期讨论形成的算法池收缩、BBOB + selected MA-BBOB 的 10D 主实验、RGI 暂缓、CEC 10D 跨 benchmark 测评等决策统一整理。

---

## 1. 本轮讨论形成的核心收缩

下一阶段不再同时推进多维、RGI、Soft-ERT、Repeated DAS 和大规模算法池，而是先集中验证：

$$
\boxed{
\text{Behavior} \rightarrow \text{candidate action gain}
}
$$

即：仅利用算法运行过程中已经产生的 algorithm-agnostic Behavior，是否能够判断当前状态下“继续当前算法”还是“切换到另一个候选算法”更有利。

第一阶段建议统一：

- 维度：10D；
- 主训练：BBOB 10D + selected MA-BBOB 10D；
- 算法池：PSO、SHADE、CMA-ES；
- 主任务：one-switch dynamic algorithm selection；
- 主标签：相对于 `continue_current` 的 terminal action gain；
- 外部测评：CEC 10D；
- RGI：暂不加入；
- Soft-ERT：暂不作为第一阶段主标签；
- extra landscape Query：暂不作为第一阶段主方法。

---

## 2. 算法池改为 {PSO, SHADE, CMA-ES}

建议主算法池预先指定为：

$$
\boxed{
\mathcal A_{core}
=
\{\mathrm{PSO},\mathrm{SHADE},\mathrm{CMA\mbox{-}ES}\}
}
$$

不再保留普通 DE。

主要理由不是预先规定：

- PSO = early；
- SHADE = middle；
- CMA-ES = late。

这种严格阶段绑定缺少足够普适证据，也会限制 selector。

更稳妥的理解是三者属于明显不同的搜索机制：

- PSO：社会信息与个体记忆驱动的大尺度群体移动；
- SHADE：基于差分向量与 success-history parameter adaptation 的自适应搜索；
- CMA-ES：利用协方差与步长适应学习搜索分布和局部几何。

三者具备较强机制互补性，适合作为小而异质的核心 portfolio。

---

## 3. 搜索阶段作为分析变量，而不是硬约束

可以事后按照 FE ratio 分层：

$$
\text{early}: FE/B\in[0.1,0.3),
$$

$$
\text{middle}: FE/B\in[0.3,0.6),
$$

$$
\text{late}: FE/B\in[0.6,0.9].
$$

然后统计：

$$
P(a^\star=a\mid phase).
$$

更有价值的结果不是简单得到：

$$
PSO \rightarrow SHADE \rightarrow CMAES,
$$

而是发现类似：

- PSO 在高 diversity、低 progress 状态更有优势；
- SHADE 在 moderate diversity、steady improvement 状态更有优势；
- CMA-ES 在 population contraction、局部结构更清晰的状态更有优势。

这使研究从 time-based phase selection 转向真正的 state-based dynamic algorithm selection。

---

## 4. 暂不扩成更大的正式算法池

算法数为 $K$ 时，有向跨算法 transition 数大约为：

$$
K(K-1).
$$

当 $K=3$ 时：

$$
3\times2=6.
$$

当 $K=6$ 时：

$$
6\times5=30.
$$

而当前每个 decision state 还需要真实 continuation、action gain 以及部分 repetition，因此大算法池会显著增加计算成本和 handoff 复杂度。

所以第一阶段保持三算法更合理。

---

## 5. 但可以额外做 Portfolio Sufficiency Study

主算法池使用三算法，不代表永远不扩展。

可以准备候选池：

$$
\mathcal A_{candidate}
=
\{
PSO,
lbest\mbox{-}PSO,
SHADE,
L\mbox{-}SHADE,
CMAES,
IPOP\mbox{-}CMAES
\}.
$$

对于当前 portfolio $P$ 和候选算法 $a'$，定义：

$$
L_P^\star(s)
=
\min_{a\in P}L(s,a),
$$

以及新增算法的边际价值：

$$
\Delta(a'\mid P)
=
\mathbb E_s
\left[
L_P^\star(s)-L_{P+a'}^\star(s)
\right].
$$

再定义独占优势率：

$$
U(a')
=
P
\left[
L(s,a')
<
\min_{a\in P}L(s,a)
-
\delta_{practical}
\right].
$$

只有新增算法带来稳定且具有实际意义的增量时，才扩充正式 portfolio。

因此更大的算法集合应该是一个 portfolio-sufficiency experiment，而不是主方法默认配置。

---

## 6. 第一阶段只做 10D

建议统一：

$$
\boxed{d=10}
$$

暂时不做 20D / 40D。

原因是当前最重要的问题是验证：

$$
\boxed{
\text{Behavior} \rightarrow \text{Action Gain}
}
$$

如果同时改变：

- problem distribution；
- dimension；
- portfolio；
- label；
- Query protocol；

结果变化后很难解释来源。

因此第一阶段固定 10D，更适合建立清晰证据链。

---

## 7. BBOB + selected MA-BBOB 作为主训练体系

第一阶段继续使用：

$$
\boxed{
\text{BBOB}_{10D}
+
\text{selected MA-BBOB}_{10D}
}
$$

而不是改用 RGI。

### BBOB 的角色

BBOB 作为 canonical landscape anchor：

- 有明确标准函数；
- 社区认知度高；
- 可以做 function-level OOD；
- 有合法 reference optimum；
- 便于定义 fixed-budget action loss；
- 适合分析 selector 在标准 landscape 上的行为。

### MA-BBOB 的角色

selected MA-BBOB 作为：

$$
\boxed{
\text{controlled structural bridge / training augmentation}
}
$$

它的价值不是单纯增加训练行数，而是提供 BBOB anchor 之间的混合景观，减少模型只记忆少量标准函数模式的风险。

---

## 8. 第一阶段暂不使用 RGI

AS-LGBM 原论文使用了大规模随机表达式树生成 RGI，目的主要是构造大规模静态 AAS 训练集。

当前 DAS 的数据单位不同：

$$
(state, action, horizon, repetition).
$$

一个函数本身就会产生大量 decision-state / action outcome。

因此当前更需要的是：

$$
\text{problem-distribution coverage},
$$

而不是继续堆函数数量。

BBOB + selected MA-BBOB 已经足以用于第一阶段验证 Behavior → Action Gain。

因此：

$$
\boxed{
\text{Phase 1 暂不使用 RGI}
}
$$

---

## 9. 后续 RGI 的合理角色

如果 Phase 1 成立，再加入 RGI。

推荐分工：

$$
\boxed{
BBOB = canonical anchors
}
$$

$$
\boxed{
MA\mbox{-}BBOB = controlled interpolation
}
$$

$$
\boxed{
RGI = broad distribution expansion
}
$$

RGI 可进一步用于：

- broader OOD training augmentation；
- unknown-optimum experiments；
- Dynamic Soft-ERT extension。

不建议直接照搬原论文完整生成 200,000 个 RGI 并全部跑动态 continuation。

---

## 10. 第一阶段建议配置

建议预先指定为：

| 项目 | 建议 |
|---|---|
| Dimension | 10D only |
| BBOB train | 当前 18 functions |
| BBOB validation | 当前 6 held-out functions |
| BBOB instances | 1, 2, 3 |
| MA-BBOB | 当前筛出的 24 definitions |
| Algorithms | PSO / SHADE / CMA-ES |
| FE_total | 10,000 |
| Population size | 40 |
| Decision opportunities | 当前 dynamic-budget-event protocol |
| Actions per state | 3 |
| Main horizon | terminal |
| Main label | relative terminal log-gap action gain |
| RGI | 不进入 Phase 1 |
| Soft-ERT | 不作为 Phase 1 主标签 |
| Extra Query | 不作为 Phase 1 主方法 |

---

## 11. Base seeds 建议

原项目 BBOB 使用 30 seeds。

对于新的 state-action continuation 数据，30 seeds 会造成很大的乘法膨胀。

第一轮 feasibility 建议先：

$$
\boxed{
10\ base\ seeds
}
$$

BBOB train：

$$
18
\times
3\ instances
\times
10\ seeds
\times
3\ algorithms
=
1620
$$

条基础 trajectories。

BBOB validation：

$$
6
\times
3
\times
10
\times
3
=
540.
$$

MA-BBOB：

$$
24
\times
10
\times
3
=
720.
$$

总基础 trajectories：

$$
2880.
$$

如果每条 trajectory 平均产生约 15 个 decision states：

$$
2880\times15
=
43200
$$

个 states。

每个 state 三个动作：

$$
43200\times3
=
129600
$$

个 state-action outcomes。

对于第一阶段 tabular behavior-based action prediction 已经足够。

---

## 12. Repetition 不建议全量三重复

所有 state-action outcome 第一遍先：

$$
R=1.
$$

再对预先定义的 subset 做：

$$
R=3.
$$

建议重复的状态包括：

- observed action margin 较小；
- action gain 接近 practical-equivalence 区域；
- event-triggered states；
- function × prefix algorithm × phase 中预先抽取的固定比例；
- easy / medium / hard 分层样本。

这样可以估计：

- action-gain variance；
- sign-flip rate；
- practical-equivalence threshold；

同时避免总成本直接乘三。

---

## 13. 主标签改为相对 action gain

对状态 $s$ 和动作 $a$，定义 terminal loss：

$$
L_{s,a}
=
\log_{10}
\left(
\operatorname{clip}
(
f^{best}_{s,a}-f^\star
)
\right).
$$

以 `continue_current` 为参照：

$$
\boxed{
G_{s,a}
=
L_{s,continue}
-
L_{s,a}
}
$$

解释：

- $G_{s,a}>0$：候选 action 优于继续当前算法；
- $G_{s,a}=0$：近似等效；
- $G_{s,a}<0$：候选 action 更差。

主任务建议做三分类：

$$
Y_{s,a}
=
\begin{cases}
Improve,&G_{s,a}>\delta,\\
Equivalent,&|G_{s,a}|\le\delta,\\
Degrade,&G_{s,a}<-\delta.
\end{cases}
$$

第一阶段不强调精确回归 action gain。

---

## 14. 第一阶段只使用 terminal horizon

为了控制变量，第一阶段先只使用：

$$
\boxed{
H=B_{remaining}
}
$$

即从当前 state 执行动作后一直运行到完整总预算。

先回答：

> 当前这个动作最终值不值得？

如果主结论成立，再增加：

$$
H=0.1B,\quad H=0.2B
$$

研究短期和中期动作偏好。

---

## 15. 第一阶段主模型

输入：

$$
B_t
$$

只使用当前已经实现的 algorithm-agnostic permutation-invariant Behavior。

输出：

$$
P(Y_{t,a}=Improve).
$$

三算法情况下，每个 state 只有三个动作：

- continue current；
- switch candidate 1；
- switch candidate 2。

因此动作空间非常清晰。

---

## 16. 主 baseline

至少包括：

1. Continue-current；
2. SBS；
3. Random switch；
4. Fixed-time one-switch；
5. Time-only selector；
6. Existing Behavior-only action-loss selector；
7. Behavior-based action-gain classifier；
8. Current G_FE Query Gate pipeline，作为旧方法 baseline。

---

## 17. 评价指标

不能只看 classification accuracy。

### Action-level

- macro F1；
- balanced accuracy；
- Average Precision；
- Brier score；
- calibration。

### Selection-level

定义：

$$
Regret_s
=
L_{s,selected}
-
\min_aL_{s,a}.
$$

报告：

- mean regret；
- median regret；
- normalized regret；
- acceptable-action rate。

### Optimization-level

- terminal log-gap；
- success rate；
- relative gain over continue-current；
- relative gain over SBS；
- VBS–SBS gap closed。

---

## 18. Acceptable Action Set

不强制唯一 best action。

定义：

$$
\mathcal A_{acc}(s)
=
\left\{
a:
L_{s,a}
-
L_s^\star
\le
\delta_{practical}
\right\},
$$

其中：

$$
L_s^\star
=
\min_aL_{s,a}.
$$

如果多个算法在实际意义上接近，选中任意一个都视为 acceptable。

---

## 19. BBOB + MA-BBOB 训练权重

不要把所有 state rows 直接拼接后等权。

建议先保证：

$$
w_{BBOB\ block}=0.5,
$$

$$
w_{MA\mbox{-}BBOB\ block}=0.5.
$$

然后：

- BBOB 内按 function 等权；
- MA-BBOB 内按 definition 等权；
- 再向下平衡 seed / prefix algorithm / state。

这样可以避免某个 benchmark source 因为生成更多 state rows 而主导拟合。

---

## 20. MA-BBOB component leakage

如果某个 BBOB function 被作为 validation / outer holdout：

$$
f_k,
$$

那么包含 $f_k$ 作为 component 的 MA-BBOB definition 不应进入该 fold 的训练。

否则 held-out BBOB function 会通过 MA-BBOB 间接进入训练。

当前项目已经有 component guard / selection manifest 的设计，建议继续保留。

---

## 21. CEC 10D 作为跨 benchmark 测评

BBOB + selected MA-BBOB 10D 完成训练和内部验证后，下一阶段直接使用：

$$
\boxed{
CEC_{10D}
}
$$

检验跨 benchmark generalization。

这样研究问题非常清楚：

> 在维度固定为 10D 的条件下，Behavior → Action Gain 的规律能否从 BBOB/MA-BBOB 泛化到 CEC？

---

## 22. CEC 测评必须保持训练流程不变

进入 CEC 后保持以下内容不变：

- Behavior extractor；
- preprocessing；
- action-gain predictor；
- practical threshold；
- decision threshold；
- portfolio；
- population size；
- FE budget；
- handoff protocol；
- state sampling；
- action rule。

完整逻辑：

```text
BBOB + selected MA-BBOB
        ↓
train / grouped OOF / model selection
        ↓
freeze
        ↓
CEC 10D trajectories
        ↓
Behavior
        ↓
predicted action gain
        ↓
selected action
        ↓
terminal outcome
```

CEC outcome 不应再次用于：

- 选特征；
- 选模型；
- 调 threshold；
- 调算法参数。

---

## 23. CEC2017 与 CEC2022 的角色

由于当前项目历史上已经使用过 CEC2017 做 preliminary / development evaluation，因此建议：

### CEC2017 10D

作为：

$$
\boxed{
external\ benchmark\ development / transfer\ evaluation
}
$$

### CEC2022 10D

如果在首次正式 outcome 之前预先指定所有协议，更适合作为：

$$
\boxed{
final\ held\mbox{-}out\ confirmation
}
$$

因此建议整体链路：

$$
\boxed{
\begin{aligned}
Train &: BBOB_{10D}+MA\mbox{-}BBOB_{10D},\\
Internal\ OOD &: held\mbox{-}out\ BBOB_{10D},\\
External\ development &: CEC2017_{10D},\\
Final\ confirmation &: CEC2022_{10D}.
\end{aligned}
}
$$

---

## 24. CEC 上继续使用同一三算法 portfolio

保持：

$$
\boxed{
\{PSO,SHADE,CMAES\}
}
$$

并统一：

- 10D；
- population size = 40；
- FE budget = 10,000；
- 同一 boundary handling；
- 同一 population transfer；
- 同一 decision opportunities；
- 同一 Behavior 定义。

这样 BBOB → CEC 的主要变化就是：

$$
\boxed{
problem\ distribution
}
$$

而不是实验其他因素。

---

## 25. 第一阶段仍建议 one-switch

第一篇工作先限制：

$$
\boxed{
N_{switch}\le1
}
$$

原因：

- 真实 action outcome 容易定义；
- 便于比较 fixed-time switch；
- 更容易隔离 Behavior prediction 的作用；
- 不需要解决复杂 context restoration；
- 避免 policy tree 爆炸。

只有证明：

$$
Behavior\mbox{-}guided\ one\mbox{-}switch
>
continue\mbox{-}current
$$

并且优于：

- Random switch；
- fixed-time switch；
- time-only selector；

再进入 repeated DAS。

---

## 26. 后续扩展顺序

建议按以下顺序推进：

### Phase 1

$$
Behavior
\rightarrow
Action\ Gain
$$

BBOB + selected MA-BBOB，10D，one-switch。

### Phase 2

$$
Behavior
+
Trajectory\mbox{-}derived\ Local\ Landscape
$$

检查在线景观信息在 Behavior 之外是否还有增量价值。

### Phase 3

$$
Adaptive\ Query
$$

把独立 landscape sampling 改成 information-acquisition action。

### Phase 4

$$
Dynamic\ Soft\mbox{-}ERT
$$

用于 unknown-optimum extension。

### Phase 5

$$
Repeated\ DAS
$$

允许多次切换。

### Phase 6

RGI / engineering problems / cross-dimension。

---

## 27. 当前最推荐的最小实验版本

```text
Dimension:
    10D

Training:
    BBOB train 18 functions
    + selected MA-BBOB 24 definitions

Internal validation:
    6 held-out BBOB functions

External transfer:
    CEC2017 10D

Final confirmation:
    CEC2022 10D

Portfolio:
    PSO
    SHADE
    CMA-ES

Budget:
    10,000 FE

Population:
    40

Base seeds:
    10

Decision opportunities:
    current dynamic-budget-event protocol

Actions:
    continue current
    switch candidate 1
    switch candidate 2

Main label:
    relative terminal log-gap action gain

Main task:
    Improve / Equivalent / Degrade

Main input:
    Behavior only

Main policy:
    one-switch

RGI:
    not used in Phase 1

Soft-ERT:
    not used as Phase 1 primary label

Extra Query:
    not used as Phase 1 primary method
```

---

## 28. 这一版的核心科学问题

可以浓缩成：

$$
\boxed{
\textbf{
Can algorithm-agnostic search behavior predict which solver should be used next,
and does this rule transfer from BBOB/MA-BBOB to CEC?
}
}
$$

中文：

> **算法无关的在线搜索行为，能否预测当前状态下下一步应使用哪个 solver，并且这种决策规律能否从 BBOB/MA-BBOB 泛化到 CEC？**

---

## 29. 最终建议

当前阶段最重要的是收缩变量。

建议正式采用：

$$
\boxed{
BBOB_{10D}
+
selected\ MA\mbox{-}BBOB_{10D}
\rightarrow
Behavior\mbox{-}based\ one\mbox{-}switch\ DAS
\rightarrow
CEC_{10D}
}
$$

暂时不把以下内容塞进第一阶段：

- RGI；
- 20D / 40D；
- online landscape；
- Query VOI；
- Dynamic Soft-ERT；
- repeated DAS；
- larger formal portfolio。

先证明最核心的：

$$
\boxed{
Behavior \rightarrow Action Gain
}
$$

再扩展其余模块。这样证据链更干净，失败时也更容易知道到底是哪一环出了问题。
