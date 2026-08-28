# Decision-before-Feature 投稿前技术审计报告

> 审计对象：`CHENTHIRTEEN/Decision-before-Feature`  
> 审计基准：`main` 分支，提交 `27859afc68a0b7b69f5e8647ed0b8800b8781d04`  
> 审计日期：2026-08-15  
> 目标定位：IEEE / Elsevier 旗下演化计算、智能优化与算法选择方向高水平 SCI  
> 审计性质：研究设计、实验协议、代码—文档一致性、理论支撑与可复现性静态审计

---

## 0. 审计范围、证据标准与限制

本报告审查了以下内容：

- 顶层研究规范、论文结构、数学定义、实验协议和结果占位文档；
- BBOB、CEC2017 及预留 CEC2022 配置；
- Behavior 特征、动态状态采样、独立景观查询、Utility 标签、Selection Reference、Decision Model、在线控制器等核心实现；
- DE、PSO、CMA-ES、SHADE 的仓内实现及 population transfer 规则；
- 当前分支状态、结果状态、软件依赖和复现实务；
- 与 ELA、算法选择、主动特征获取、嵌套交叉验证、BBOB/COCO、CMA-ES、SHADE 等相关的外部文献。

报告中的判断分为三类：

- **已核验事实**：可以从当前提交的代码、配置或文档直接确认；
- **方法学推论**：由已核验事实和标准统计/决策理论推得，报告中会说明推理边界；
- **修改建议**：面向高水平期刊投稿提出的改进方案，不冒充已经得到的实验结论。

### 0.1 运行验证限制

本次审计通过 GitHub 连接器逐文件检查了仓库，但当前执行环境无法完成仓库克隆，因而：

- **没有实际运行单元测试、端到端实验或数值复现**；
- 对代码问题的判断属于静态审查，其中明确标为“疑似缺陷”或“需动态验证”的内容，不应直接当作已复现的软件错误；
- 仓库当前没有正式实验结果可供验证。`docs/30_results/phase1_current_results.md` 明确说明旧结果已经撤销，正式 72 个 trajectory shard 尚未开始。

这一区分很重要。阅读几千行代码后宣布“实现绝对正确”，通常只是科研版的占卜。

---

# 1. 总体结论

## 1.1 一句话判断

**研究问题具有明显的新颖性和高水平期刊潜力，但当前版本不适合直接启动完整实验，更不适合投稿。主要障碍不是想法不足，而是活动协议、论文正文与代码之间出现了若干关键语义漂移；同时，算法实现有效性、外部确认性评估和约两千亿量级的计算方案尚未闭环。**

## 1.2 当前成熟度评估

| 维度 | 评价 | 说明 |
|---|---:|---|
| 研究问题新颖性 | 较强 | “先判断是否值得获取独立景观信息，再执行查询”区别于传统 ELA-AAS、trajectory-based selection 和一般动态算法选择 |
| 数学问题定义 | 中上 | 已区分 Skip、Query、Behavior-only、运行时间、终点 gap 和 first-trigger，但主比较对象仍需重新裁决 |
| 防泄漏设计 | 很强 | complete state、RNG state、fold-specific selector、train-only threshold、特征白名单等设计明显高于常见实验 |
| 实验可证伪性 | 中等 | 已主动承认负结果可能，但 RQ、路径和敏感性过多，核心假设容易被大量附属分析淹没 |
| 代码—文档一致性 | 当前较弱 | CV 分组、五路径、selector target、加权拟合、LDA 定义、CEC2017 配置等存在实质不一致 |
| 算法基线可信度 | 当前不足 | 四个优化器均为仓内实现，尚无与可信实现或 COCO 基准结果的能力审计 |
| 统计设计 | 中上 | 分层权重、嵌套 OOF、效应量意识较好；但状态级依赖、确认性样本和不确定性目标仍需收紧 |
| 外部泛化证据 | 当前不足 | BBOB validation 已被检查，CEC2017 已进入开发协议，真正 prospective 的外部确认尚未é¢åæå® |
| 计算可行性 | 高风险 | 仓库估算主路径超过 2100 亿 FE，三档查询约 3450–3500 亿 FE，尚未见小规模实测资源审计 |
| 可复现软件工程 | 当前不足 | 有锁文件和大量契约检查，但没有 CI、正式测试目录、LICENSE、CITATION.cff 和发布级 manifest |

## 1.3 面向投稿的现实判断

- **IEEE Transactions on Evolutionary Computation**：题目方向相符，但只有在修复协议一致性、完成算法能力审计、建立更干净的理论主线、获得真正 prospective 的外部证据后，才具有合理冲击价值。
- **Swarm and Evolutionary Computation**：与该研究的“算法选择 + 行为特征 + 资源感知决策”组合非常契合，是当前设计更自然的高水平目标。
- **Information Sciences / Applied Soft Computing 等**：取决于最终理论深度、外部数据和结果强度，也可能是合理目标。
- 当前不能根据设计文档本身预判“达到 top 期刊录用标准”。高水平审稿人不会因为协议长达几十页便自动感动，他们通常会先寻找一个足以否决全文的语义漏洞。

---

# 2. 当前方案真正做得好的地方

这些优点应保留，而不是在重构中一并推倒。

## 2.1 研究问题不是简单的“ELA + 分类器”

核心问题不是：

> 根据景观特征选择算法。

而是：

> 在已经观察到优化行为的状态上，是否值得为一个固定、独立、需要付费的景观查询消耗资源？

这使研究对象从“算法选择”转向“信息获取决策”，与以下研究形成清楚区分：

1. 传统的预运行 ELA-based AAS；
2. 利用已有优化轨迹直接选算法的 per-run algorithm selection；
3. 搜索过程中不断切换算子/算法的 dynamic algorithm selection；
4. 基于景观特征调节参数、种群或算子的 adaptive optimization；
5. 主动特征获取中一般的“是否购买额外信息”问题。

这个定位是目前最值得保护的创新核心。

## 2.2 对查询成本的处理方向正确

当前协议明确：

- 独立 LHS 查询消耗真实 FE；
- 查询 FE 从后续优化预算中扣除；
- 查询样本不直接并入 optimizer population；
- 查询样本中的最优点和 first hit 仍进入 operational Query endpoint；
- 不再对已经减少的优化 FE 重复扣费。

这与近期关于 ELA 预算的研究结论一致：特征采样预算会显著影响算法选择收益，且必须把 sample best 与被占用的优化预算一并考虑，而不能只比较“选对算法的比例”。

## 2.3 complete-state 反事实分叉设计严谨

仓库没有把“当前种群”错误地当作完整算法状态，而是显式保留：

- 当前 population 与 fitness；
- 历史窗口；
- 算法内部状态；
- RNG state；
- 已消耗 FE；
- native continuation 与 population-transfer initialization 的区别。

同一状态上的候选 continuation 使用复制的完整状态或明确定义的转移规则，这比常见的“重新初始化后假装是同一个状态”严谨得多。

## 2.4 信息可用时间和泄漏边界定义清楚

Decision gate 排除了：

- query descriptors；
- function identity；
- algorithm identity；
- optimizer-specific parameters；
- known optimum gap；
- action outcomes；
- 未来状态。

同时，fold-specific Selector、Decision OOF、训练集阈值和外部测试相互隔离。这个方向完全正确。

## 2.5 first-trigger 与可达状态意识较好

部署策略规定：

- 一个 run 最多触发一次；
- 触发后后续离线状态对该策略不可达；
- 评估使用 first-trigger，而不是把每个状态当作独立分类样本随意计数。

这是动态决策研究中经常被忽略的关键点。

## 2.6 Stage A 与 Stage B 的区分值得保留

- Stage A 决定科学终点、gap、first hit、completion 和 failure；
- Stage B 的三次 replay 只用于 wall-clock；
- timed-out/failed replay 使用 censoring，防止“快速失败反而获得低时间成本”；
- observed hit、path completion 和 endpoint success 被分别记录。

这套设计复杂，但逻辑本身是合理的。

## 2.7 对统计能力的自我约束很诚实

仓库已经指出：BBOB validation 只有 6 个 function，若以 function 为独立分析单位，即使所有函数同向，双侧 sign-flip 最小 $p$ 值为

$$
p_{\min}=\frac{2}{2^6}=0.03125.
$$

若同时对 6 个主要比较做 Holm 校正，最小调整后值为

$$
p_{\min}^{\mathrm{Holm}}=6\times 0.03125=0.1875.
$$

因此这 6 个函数不可能承担常规的多重校正确认性显著性叙事。仓库选择强调效应量、区间和有限集合估计，这个判断是正确的。

---

# 3. 必须在正式大规模运行前解决的 P0 问题

## 3.1 P0 总表

| 编号 | 问题 | 严重性 | 直接后果 |
|---|---|---:|---|
| P0-01 | 论文声称按 function ID 分组，代码实际按 5 类 landscape family 分组 | 致命 | OOF、阈值、泛化问题和所有结果的含义改变 |
| P0-02 | 主 gate 学习 Query vs Skip，但论文核心语言经常声称判断“额外查询是否值得” | 致命 | 可出现 Query 优于 Skip、却劣于 Behavior-only，仍被标为值得查询 |
| P0-03 | 五路径活动协议与论文正文中的三路径/“缺少 matched path”表述冲突 | 致命 | 数学定义、结果表、贡献和代码不是同一研究 |
| P0-04 | Selector 主 target 已改为相对 continue-current 的 log-gap 差，正文仍写 statewise min–max | 高 | 训练目标、预测符号和可解释性全部错位 |
| P0-05 | CEC2017 配置包含已移除的 F2，同时遗漏 F30 | 高 | 外部测试套件定义错误 |
| P0-06 | 仓内 SHADE 供体索引疑似不满足经典 distinctness 约束，四算法均未做能力审计 | 高 | Selector 可能学到实现缺陷而非算法互补性 |
| P0-07 | 当前物理执行计划约 2100–3500 亿 FE，未完成小规模实测资源审计 | 高 | 项目可能在得到首个可用主结果前耗尽算力和开发周期 |
| P0-08 | 仓库无正式 CI/测试体系，本次也未能动态执行 | 高 | 复杂协议缺少可持续的机器验证 |
| P0-09 | BBOB validation 和 CEC2017 已进入开发视野，真正 prospective 外部确认集未é¢åæå® | 高 | 不能把它们包装成独立确认性证据 |
| P0-10 | 论文、TBD 和代码对 cluster weighting、LDA、CEC2022 状态的描述不一致 | 高 | 读者无法判断究竟哪套实现产生结果 |

---

# 4. P0-01：交叉验证分组语义发生实质错误

## 4.1 已核验事实

论文 Introduction 明确写道：

> nested grouped-by-function out-of-fold model selection  
> The grouping unit is function ID.

但当前代码链路中：

1. `benchmarks/bbob.py` 将 BBOB 的 `family` 映射为 5 个经典 landscape groups；
2. `decision/train_full_decision_model.py`、`decision/nested_learning.py` 和 `selection_reference/model.py` 使用 `family` 作为 GroupKFold 的 grouping 字段；
3. 因此实际做的是“按 5 个 landscape family 分组”，而不是“按 18 个训练 function ID 分组”。

这不是变量命名问题，而是实验设计改变。

## 4.2 为什么严重

两种 CV 回答的问题完全不同：

### 按 function ID 分组

测试模型能否迁移到**未见过的函数**，同时训练集仍可包含相同高层景观家族的其他函数。

### 按五类 landscape family 分组

测试模型能否迁移到**整个未见过的高层景观组**。折数最多只有 5，难度、训练样本组成和阈值估计都明显不同。

它还会改变：

- outer fold 数量；
- inner fold 的可行性；
- SBS 的 fold-specific 计算；
- query selector 和 behavior-only selector 的训练集；
- Decision label；
- threshold；
- OOF utility；
- 最终 first-trigger policy。

换句话说，若不修复，后面即使所有表格数值都“很好看”，论文也无法准确解释它们代表什么。

## 4.3 推荐修复

新增三个不可混用的字段：

```text
function_id
landscape_family
cv_group_id
```

主实验固定：

```text
cv_group_id = function_id
```

并将“留一 landscape family”作为更严格的独立鲁棒性分析。

必须加入自动契约：

```python
assert set(train.function_id).isdisjoint(set(test.function_id))
```

该检查需要覆盖：

- SBS 派生；
- query-adjusted Selector；
- behavior-only Selector；
- matched state-only Selector；
- Decision Model；
- threshold selection；
- feature-group comparison；
- stacking/diagnostic model。

**修复后必须重建所有 OOF artifact。旧 artifact 不允许通过改 metadata 继续使用。**

---

# 5. P0-02：主决策标签与“是否值得获取查询”的科学问题没有完全对齐

## 5.1 当前三个关键量

当前方案定义：

### Query 相对 native Skip 的联合效用

$$
U_q^{\mathrm{joint}}(S)
=
(\ell_s-\ell_q)
-\lambda_T\left(\log_{10}T_q-\log_{10}T_s\right).
$$

### Behavior-only 相对 native Skip 的效用

$$
U_b(S)
=
(\ell_s-\ell_b)
-\lambda_T\left(\log_{10}T_b-\log_{10}T_s\right).
$$

### Query 相对 Behavior-only 的 operational increment

$$
I_q(S)
=
U_q^{\mathrm{joint}}(S)-U_b(S).
$$

当前 gate 的二元标签主要是：

$$
y_q(S)=\mathbf{1}\left[U_q^{\mathrm{joint}}(S)>0\right].
$$

## 5.2 逻辑问题

Behavior features 在查询前已经可用，Behavior-only Selector 也是一个现实可执行的 no-query policy。于是可能出现：

$$
U_q^{\mathrm{joint}}(S)>0,
\qquad
I_q(S)\le 0.
$$

解释是：

- Query path 比“什么都不选、原算法继续”更好；
- 但 Query path 不如“仅凭现有 Behavior 做算法选择”。

这时将状态标为“值得购买额外查询”并不符合通常的 value-of-information 语义。查询只是优于一个较弱的 no-query comparator，而不是优于最佳可用 no-query 决策。

## 5.3 两个可行方向

### 方向 A：保留当前二元 gate，但收窄论文主张

明确研究问题是：

> 在 fold-specific SBS 的 native continuation 与 Query-plus-Selector 之间，何时应选择后者？

此时不能把结果概括为“是否值得进行特征提取”，也不能声称估计了 query 的纯信息价值。

### 方向 B：采用更符合主动信息获取的主问题

令可用路径为：

- $s$：native continuation；
- $b$：Behavior-only selection；
- $q$：acquire Query and select。

定义条件期望成本：

$$
C_k(S)
=
\mathbb{E}
\left[
\ell_k+\lambda_T\log_{10}T_k
\mid S
\right],
\qquad
k\in\{s,b,q\}.
$$

Bayes 最优路径为：

$$
k^*(S)=\arg\min_{k\in\{s,b,q\}} C_k(S).
$$

对应的 query acquisition 决策为：

$$
d^*(S)
=
\mathbf{1}\left[k^*(S)=q\right].
$$

若论文坚持二元 query/no-query gate，no-query comparator 应取最佳可用 no-query 策略：

$$
d^*(S)
=
\mathbf{1}
\left[
\mathbb{E}[C_q(S)\mid S]
<
\min\left\{
\mathbb{E}[C_s(S)\mid S],
\mathbb{E}[C_b(S)\mid S]
\right\}
\right].
$$

在 Behavior-only 被固定为主要替代路径时，可简化为：

$$
d^*(S)
=
\mathbf{1}
\left[
\mathbb{E}[I_q(S)\mid S]>0
\right].
$$

## 5.4 推荐

面向高水平投稿，建议选择方向 B。原因不是它“更复杂”，而是它使论文标题、理论、标签和实际决策完全一致：

> 现有信息足以作出决策时不购买查询；只有查询相对最佳 no-query 决策带来正净增益时才购买。

当前 $U_q^{\mathrm{joint}}$ 仍可保留为重要分解量，但不应继续独占主标签。

---

# 6. P0-03：活动五路径协议与论文正文冲突

## 6.1 活动协议已经是五路径

当前最高层字段规范和代码已包含：

1. `skip`：native continuation；
2. `behavior_only_full_budget`；
3. `sampling_only_continue_current`；
4. `query_matched_state_only`；
5. `query_joint`。

其中三条 query-acquisition path 共享同一独立样本和相同 query FE：

- sampling-only：采样后继续当前算法；
- matched state-only：采样后不看 descriptors，只用 state-only Selector；
- full query：采样、看 descriptors、用 full Selector。

这是一个很好的 operational decomposition。

## 6.2 正文仍在描述旧协议

`docs/40_manuscript/sections/03_problem_formulation.tex` 仍写：

> protocol does not include a matched-acquisition state-only fourth path

`06_results.tex` 和 `07_discussion.tex` 也仍围绕“缺少 matched-acquisition path，无法拆分 descriptor contribution”的旧限制组织。

这与活动代码和字段规范直接矛盾。

## 6.3 建议采用的五路径分解

令：

- $s$：Skip；
- $b$：Behavior-only；
- $m$：sampling-only；
- $o$：query-matched state-only；
- $q$：full query。

对任一路径 $k$ 定义相对 Skip 的效用：

$$
U_k
=
(\ell_s-\ell_k)
-\lambda_T\log_{10}\frac{T_k}{T_s}.
$$

则 full Query 相对 Skip 可作如下 operational 分解：

### 查询 descriptors 使用增量

$$
D_q=U_q-U_o.
$$

### 在已付查询成本条件下，state-only selection 相对继续原算法的增量

$$
S_q=U_o-U_m.
$$

### 单纯查询采样、sample-best、预算变化和时间开销的联合增量

$$
A_q=U_m-U_s=U_m.
$$

因此：

$$
U_q=D_q+S_q+A_q.
$$

Behavior-only 另行比较：

$$
U_b
=
(\ell_s-\ell_b)
-\lambda_T\log_{10}\frac{T_b}{T_s}.
$$

并可报告：

$$
I_q=U_q-U_b.
$$

## 6.4 解释边界

上述分解是**固定实现和固定路径下的 operational decomposition**，不是随机试验意义上的因果效应。必须避免使用：

- causal effect of descriptors；
- pure value of information；
- unbiased contribution；

更准确的词是：

- matched operational increment；
- pathway decomposition；
- observed resource-adjusted difference。

---

# 7. P0-04：Selector target 的正文与代码不是同一个目标

## 7.1 当前正文

Problem formulation 仍将主 target 写为 statewise min–max：

$$
\widetilde L_{s,a}
=
\frac{L_{s,a}-\min_c L_{s,c}}
{\max\left(\max_c L_{s,c}-\min_c L_{s,c},10^{-12}\right)}.
$$

## 7.2 当前活动代码与字段规范

当前主 target 已经改为相对 `continue_current` 的 clipped log-gap 差，min–max 仅作为敏感性分析。

推荐统一写成：

$$
y_{s,a}
=
\ell_{s,a}-\ell_{s,a_0},
\qquad
a_0=\texttt{continue\_current},
$$

其中：

$$
\ell_{s,a}
=
\log_{10}
\left[
\min\left(
\max(g_{s,a},g_{\min}),
g_{\max}
\right)
\right].
$$

此时：

- $y_{s,a}<0$：动作 $a$ 优于 continue-current；
- $y_{s,a}=0$：与 continue-current 相同；
- $y_{s,a}>0$：动作 $a$ 更差。

## 7.3 命名建议

当前字符串 `clipped_log10_gap_advantage_vs_continue_current` 容易造成符号误读，因为“advantage”通常暗示越大越好，而这里越小越好。

建议改为：

```text
relative_clipped_log10_gap_vs_continue_current
```

或定义相反符号：

$$
A_{s,a}=\ell_{s,a_0}-\ell_{s,a},
$$

此时才满足“advantage 越大越好”。

## 7.4 为什么相对基线 target 更合理

相对于 statewise min–max，它具有以下优势：

- 保留不同状态间的绝对效果尺度；
- continue-current 固定为 0，解释清楚；
- 不会因为某状态中一个极差失败动作而压缩其他动作差异；
- 与最终决策“是否切换”更直接对齐。

但仍需保留 min–max、raw log gap 或 rank target 作为敏感性，因为 clipping 和 failure cap 可能形成大量平局。

---

# 8. P0-05：CEC2017 配置错误

## 8.1 已核验事实

当前 `configs/phase1_cec2017_test.yaml` 使用：

```yaml
functions: [1, 2, 3, ..., 29]
```

官方 CEC2017 bound-constrained suite 后来因数值不稳定移除了 Problem 2，正式保留的是 29 个问题：

```text
F1, F3, F4, ..., F29, F30
```

因此当前配置：

- 不应包含 F2；
- 应包含 F30。

## 8.2 修复

配置应改为：

```yaml
functions:
  [1, 3, 4, 5, 6, 7, 8, 9, 10,
   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
   21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
```

并新增：

```yaml
suite_definition: cec2017_bound_constrained_final_29
suite_source: official_cec2017_definition
implementation: opfunu
implementation_version: ...
```

## 8.3 必须做的 evaluator 审计

不能只依赖“库能返回数值”：

1. 对每个函数检查 bounds、dimension 支持和 reference value；
2. 对已知 optimum 或官方测试点，与官方实现/定义进行点值核对；
3. 检查 F30 是否可用；
4. 明确 OPFUNU 对 shift/rotation data 的来源和版本；
5. 保存 package version、data hash 和函数映射表；
6. 为 `function_id -> implementation class` 建立自动测试。

---

# 9. P0-06：优化器实现有效性和公平性尚未证明

## 9.1 固定 population size 不是自动公平

当前四个算法统一：

```text
population_size = 40
```

其优点是：

- 状态形状一致；
- population transfer 简化；
- 每个 native update 的 FE 量级一致。

但它不意味着算法能力公平。尤其：

- CMA-ES 的常用默认 $\lambda$ 随维度变化；
- SHADE 的常见配置、memory size 和 population size 有其标准设置；
- PSO 和 DE 的参数也影响组合互补性；
- 统一种群规模可能人为改变 SBS 和切换收益。

因此论文应把它称为“用于本研究状态接口的é¢åæå® portfolio configuration”，而不是暗示这些是各算法的通用标准版本。

## 9.2 SHADE 供体索引存在静态疑点

当前 `_start_shade_generation` 中：

- $r_1$ 从排除当前个体 $i$ 的 population 中抽取；
- $r_2$ 从 population 与 archive 的 union 中抽取；
- 循环只保证 $r_2\ne i$；
- 未显式保证 $r_2\ne r_1$。

经典 current-to-pbest/1 需要相应索引互异。若 $r_2=r_1$ 且 $r_2$ 指向 population 中同一向量，则差分项退化：

$$
x_{r_1}-x_{r_2}=0.
$$

这会改变变异分布和算法能力。

这项判断来自静态代码，不等同于已运行复现的 bug；但在正式实验前必须通过单元测试确认并修正。

## 9.3 四算法必须先做 consistency check

至少包括：

### A. 独立运行能力

在 BBOB 上比较仓内实现与可信实现或公开参考数据：

- DE；
- PSO；
- `pycma` 或 modular CMA-ES；
- 可信 SHADE 实现。

指标包括：

- AOCC / ECDF；
- fixed-budget log gap；
- ERT；
- failure rate；
- 不同维度的性能趋势。

### B. 算法不变量

- 完整状态 clone 后 native continuation 必须逐评价完全一致；
- 相同 seed 和配置必须可复现；
- 分批推进与一次推进相同 FE 后结果一致；
- partial generation 的 exact FE accounting 正确；
- population transfer 不应重复评价已有 population；
- 每个算法的边界处理被准确记录。

### C. 参数敏感性

至少做：

- population size 20 / 40 / 80；
- CMA-ES native/default $\lambda$；
- SHADE memory size；
- 一组常用 DE/PSO 参数。

主结论若只在 population size 40 下成立，应明确限定。

## 9.4 算法命名必须具体

建议 artifact 中使用：

```text
DE/rand/1/bin_F0.5_CR0.9_pop40_clip
PSO_w0.72_c1.49_c2.49_vmax0.2_pop40_clip
CMAES_repo_pop40_clip
SHADE_repo_H5_pop40_clip
```

正文可使用简写，但补充材料必须给出准确身份。

---

# 10. Population transfer 是重要混杂因素

## 10.1 当前转移不对称

从某一算法状态切换到另一算法时：

- DE：使用当前 population，重置 DE 内部状态；
- PSO：使用当前 positions，但速度重新随机初始化，personal best 由当前种群构造；
- CMA-ES：由当前 population 估计 mean、covariance 和 sigma；
- SHADE：使用当前 population，但 memory 和 archive 重置。

因此不同 target algorithm 获得的信息量和初始化质量不同。

## 10.2 可能的后果

Selector 可能学到的是：

- 哪种转移构造最稳；
- 某种算法最容易从别人的 population 恢复；
- 某种算法因内部记忆重置而系统吃亏；

而不完全是“该算法适合当前问题”。

这不使研究无效，但 estimand 必须写清：

> 选择的是“算法 + é¢åæå®转移规则”组成的 action。

## 10.3 推荐敏感性

至少比较：

1. native continuation；
2. 当前 population transfer；
3. fresh restart，使用相同 remaining FE；
4. 可行时的 algorithm-specific warm start。

并按以下维度报告：

- source prefix algorithm；
- target algorithm；
- handoff type；
- dimension；
- decision phase。

若 transfer rule 决定大部分收益，应将其作为主要发现，而不是藏在补充材料里。

---

# 11. P0-07：计算方案在当前形态下过重

## 11.1 基础规模

BBOB train + validation 配置包含：

- 24 个 function；
- 3 个 instance；
- 3 个 dimension；
- 30 个 seed；
- 4 个 algorithm。

基础 run 数：

$$
24\times 3\times 3\times 30\times 4
=
25{,}920.
$$

仅原始优化轨迹 FE：

$$
24\times 3\times 30\times 4
\times(10{,}000+20{,}000+40{,}000)
=
604{,}800{,}000.
$$

每 run 至少 12 个状态，因此最少状态数：

$$
25{,}920\times 12
=
311{,}040.
$$

事件状态会继续增加。

仓库主规范估计：

- cheap 主查询完整路径约 2109.9–2251.4 亿 FE；
- 三档查询合计约 3454.9–3502.0 亿 FE；

且未完全包含外部套件、异常重跑和所有诊断。

## 11.2 主要浪费来自“同一物理路径被 fold role 重复执行”

不同 fold-specific model 可能选择同一 action，但当前复杂流水线容易为每个角色重复执行：

- 相同 state；
- 相同 query sample；
- 相同 action budget；
- 相同 target algorithm；
- 相同 transition rule；
- 相同 RNG state。

科学终点并不会因为“哪个模型选择了这个 action”而改变。

## 11.3 推荐缓存键

建立物理路径缓存：

```text
(
  state_id,
  query_id,
  sample_realization_id,
  action_budget_mode,
  target_algorithm,
  transition_mode,
  path_type
)
```

Stage A：

- 每个唯一 action outcome 只执行一次；
- fold-specific Selector 只从 action matrix 中选取 outcome；
- 不为不同模型重复运行同一个 continuation。

Stage B：

- 优化/查询路径时间按唯一物理路径缓存；
- selector inference 时间单独测量；
- 可在分层子集上用完整 end-to-end replay 验证“组件加和”近似；
- 若坚持所有状态完整 replay，则必须先证明资源预算可承担。

## 11.4 推荐的分阶段策略

### Stage 0：契约与小型 smoke test

- 4 个 BBOB function；
- 2 个 dimension；
- 2 个 instance；
- 5 个 seed；
- cheap query；
- 所有五路径；
- 所有算法；
- 全部测试和 schema verification。

目标不是论文结果，而是验证：

- 实际 FE 与理论账本一致；
- wall-clock、峰值内存和 artifact 体积；
- failure rate；
- 每状态物理路径数量；
- 缓存命中率；
- 完整流水线能否结束。

### Stage 1：cheap-query 主实验

只对 `descriptor_cheap_invariant` 运行完整主协议。

### Stage 2：query configuration 敏感性

`pflacco_standard` 和 `pflacco_broad` 先在按 function family、dimension、phase 分层的子集上运行。只有 cheap 主结果表明研究问题存在信号后，再扩展。

### Stage 3：prospective 外部确认

在é¢åæå®外部配置后一次性运行，不再调阈值和模型。

## 11.5 更根本的简化选择

若 wall-clock replay 继续成为主要计算瓶颈，可以考虑：

- 主 utility 使用 budget-adjusted terminal gap；
- wall-clock 与内存作为并列政策终点，而不进入训练标签；
- 或使用参数化 objective-cost 模型。

这会牺牲部分“单一净效用”叙事，却显著提高可复现性和可解释性。

---

# 12. Wall-clock utility 的外部有效性问题

## 12.1 BBOB 上的 wall-clock 主要反映什么

BBOB objective 通常极便宜。此时测得的路径时间会大量反映：

- Python 循环；
- NumPy/SciPy 实现；
- 自定义优化器代码效率；
- 序列化和特征计算；
- CPU、线程和缓存状态。

真实昂贵黑盒问题中，objective evaluation 可能占绝大多数时间。于是同一个 $\lambda_T$ 在两种场景中含义不同。

## 12.2 参数化时间模型

可将路径时间写为：

$$
T_k(c_f)
=
N_kc_f+C_k^{\mathrm{ctrl}},
$$

其中：

- $N_k$：路径实际 objective evaluations；
- $c_f$：单次 objective evaluation 的外部成本；
- $C_k^{\mathrm{ctrl}}$：优化器、特征、模型与调度 CPU 成本。

对应效用：

$$
U_k(c_f)
=
(\ell_s-\ell_k)
-\lambda_T
\log_{10}
\frac{N_kc_f+C_k^{\mathrm{ctrl}}}
{N_sc_f+C_s^{\mathrm{ctrl}}}.
$$

建议至少报告几个成本区间：

```text
c_f = 0
c_f = 10^-4 s
c_f = 10^-3 s
c_f = 10^-2 s
c_f = 10^-1 s
c_f = 1 s
```

也可以通过受控 sleep/emulated objective 验证端到端时间。

## 12.3 $\lambda_T=1$ 不是理论常数

当前：

$$
U_k(\lambda_T)=\Delta_{\ell,k}-\lambda_T\Delta_{T,k}.
$$

若 $\Delta_{T,k}\ne 0$，break-even 权重为：

$$
\lambda_T^*
=
\frac{\Delta_{\ell,k}}{\Delta_{T,k}}.
$$

应报告：

- $\Delta_\ell$；
- $\Delta_T$；
- $\lambda_T^*$；
- Pareto dominance；
- utility 随 $\lambda_T$ 的曲线。

这样读者可以根据自己的成本偏好判断，而不是被迫接受 $\lambda_T=1$ 的单位交换率。

---

# 13. 动态状态采样：实现严谨，但阈值缺少迁移依据

## 13.1 优点

当前动态采样包含：

- 0.20–0.60 的监测网格；
- 12 个 mandatory milestones；
- 每 phase 最多两个 event-only 状态；
- event-only 最小间隔；
- threshold + rearm hysteresis；
- native-update 对齐；
- 实际 FE 与名义 ratio 分离。

比普通的“每隔 10 代取样”严谨得多。

## 13.2 问题

以下阈值属于研究者定义的操作常数：

- stagnation；
- rank change；
- elite migration；
- diversity recovery。

é¢åæå®它们可以防止事后调参，但不能证明：

- 对 DE、PSO、CMA-ES、SHADE 同样合理；
- 对 10D、20D、40D 同样合理；
- 对 BBOB 与 CEC 同样合理。

另外，event-triggered states 的出现概率本身与问题和算法有关。把 mandatory 和 event states 混合后，状态分布不再是均匀时间网格。

## 13.3 推荐主次关系

建议：

- **主分析**：mandatory milestones；
- **次分析**：milestone + event augmentation；
- 分别报告两类状态；
- policy endpoint 始终以 run 为单位；
- 状态级建模使用 run/function 分层权重；
- 对阈值做有限的预注册敏感性，而不是在 validation 上选择。

---

# 14. Query 设计审计

## 14.1 单个 checkpoint 只有一个 query realization

Query sample seed 由 run seed、function、instance、dimension 和 sample design 共同生成。因而不同 run seed 对应不同 LHS，这比“每个问题永远一个固定样本”更合理。

但在同一 complete state 上，只有一个 query realization。于是 Utility label 同时包含：

- 状态差异；
- optimizer 随机性；
- query sample 随机性。

无法直接估计“同一状态下是否会因不同 LHS 而改变标签”。

### 建议

对分层抽取的 checkpoint，额外运行 2–5 个独立 query realization，估计：

- descriptor 方差；
- selected action 方差；
- $U_q$ 方差；
- 标签翻转率；
- first-trigger 稳定性。

这应是噪声审计，不必扩展到全部状态。

## 14.2 broad query 同时改变了四件事

`pflacco_broad_invariant` 相比 cheap/standard 同时改变：

1. sample size：$50d\rightarrow100d$；
2. feature family；
3. sample realization；
4. FE 和 wall-clock 成本。

因此只能解释为“query configuration 的总体差异”，不能声称差异来自“更丰富 descriptors”。

### 建议的因子化子实验

在子集上至少比较：

| Sample | Features |
|---|---|
| 50d | cheap |
| 50d | standard |
| 50d | broad-compatible subset |
| 100d | cheap |
| 100d | standard |
| 100d | broad |

尽可能在同一 100d 样本的前 50d / 全 100d 上计算，以降低采样随机性。

## 14.3 trajectory-reservoir query 的定位

仓库还实现了从已评价点 reservoir 计算 cheap descriptors 的零额外 FE 路径。它不等于独立 Query，因为：

- 数据来自 optimizer trajectory；
- 采样分布依赖算法；
- 没有独立空间填充设计；
- “零额外 FE”不代表零计算成本。

建议将其定位为：

> reuse-existing-evaluations representation baseline

而不是独立查询的替代定义。

---

# 15. Behavior 特征与“算法无关”主张需要更谨慎

## 15.1 应区分两种含义

### 特征定义不使用算法身份

当前主特征确实排除了 algorithm ID 和 optimizer-specific parameter。

### 特征分布不依赖算法

这并不成立。轨迹是算法产生的，因此 Behavior 分布必然依赖 prefix algorithm。

更准确的表述是：

> algorithm-identity-free, permutation-invariant behavioral representation

而不是：

> algorithm-invariant behavior。

## 15.2 主样本只来自 fold-specific SBS prefix

这会限制结论：

> gate 对 SBS-prefix state 是否有效。

不能直接扩展为：

> gate 对任意优化算法状态都有效。

### 建议

- 主文明确限定 SBS-prefix；
- 增加 all-prefix 诊断；
- 做 leave-one-prefix-algorithm-out；
- 或按 prefix algorithm 报告性能。

## 15.3 几何缩放一致性

大量距离特征已将位置缩放到 unit cube，这是正确的。

但静态检查显示 `bf_covariance_spectral_concentration` 的某条计算路径直接使用 raw population。对 BBOB/CEC 等每维同尺度 box 影响可能有限，但对各向异性工程边界会破坏尺度不变性。

建议：

- 所有 covariance / distance 特征统一在 unit cube 上计算；
- 添加坐标平移、尺度变换和个体排列不变量测试；
- 若不修改，则缩窄“invariant”命名。

---

# 16. 传统 AAS baseline 当前被有意“隔离”，但实践上偏弱

## 16.1 当前设计

pre-run AAS：

- 先支付 Query sample；
- 根据 descriptors 选算法；
- 优化器随后 fresh initialization；
- query sample 不用于 warm start。

这能隔离“描述信息用于选算法”的效果，但不代表实践中最强的预运行 AAS。

## 16.2 审稿风险

trajectory/warm-starting 文献已经表明，可以复用先前评价或状态来启动后续算法。审稿人可能认为：

> 你让自己的 Query path 计入 sample-best，却让 traditional AAS 丢弃全部样本信息，从而人为削弱 baseline。

## 16.3 推荐双 baseline

同时保留：

1. **isolated feature-only AAS**：当前实现，用于机制隔离；
2. **practical sample-reuse AAS**：将 sample best、elite subset 或合法 warm-start 信息交给后续优化器。

两者必须分别命名，不能混作一个 baseline。

---

# 17. 统计分析方案需要进一步收紧

## 17.1 明确三个不同 estimand

### 状态级效用 estimand

描述在离线 eligible state 分布上的平均 Utility。状态不是独立观测。

### first-trigger policy estimand

每个 run 产生至多一个决策，比较政策层面的：

- terminal gap；
- success/ERT；
- wall-clock；
- query-use rate；
- regret。

这是主部署 estimand。

### function-level 泛化 estimand

以 function 为等权单位，回答对é¢åæå® function 集合的平均效果。它不同于 seed-level 置信区间。

## 17.2 推荐权重

状态级训练可以使用：

$$
w_i
\propto
\frac{1}
{n_{\mathrm{function}(i)}
 n_{\mathrm{problem}(i)}
 n_{\mathrm{run}(i)}}.
$$

但论文应同时报告：

- equal-state；
- equal-run；
- equal-function；

它们代表不同问题，不能只挑结果最好的一种。

## 17.3 Bootstrap 边界

对 seed 做 bootstrap 只能反映：

- stochastic run variation；

不能代表：

- 新 function 的泛化不确定性。

对固定 6 个 validation functions，function bootstrap 也只是有限集合重采样，不会神奇地创造一个真实 function superpopulation。

## 17.4 10,000 次 bootstrap 不是样本量论证

重采样次数增加只减少 Monte Carlo 误差，不增加信息量。正式实验前应从 pilot 估计方差并进行 precision planning，例如设定：

- 最小有意义 effect；
- 目标 CI 半宽；
- 需要的 function / generated problem / seed 数。

## 17.5 Multiplicity

不建议把以下所有项都写成并列确认性假设：

- 6 feature groups；
- 3 query configurations；
- 5 $\lambda_T$；
- 3 model families；
- 多个 baseline；
- 多套件；
- 多维度；
- 多阶段。

推荐：

### Primary

1. query vs best no-query policy 的 first-trigger policy utility；
2. behavior features vs time-only 的增量；
3. prospective external suite 的同方向复制。

### Secondary / sensitivity

其余 query、模型、feature group、$\lambda_T$ 和分解项。

## 17.6 “无显著差异”不是“等价”

若目标之一是：

> 减少查询且性能不变差，

应使用预先定义的 smallest effect size of interest 或非劣/等价框架，而不是依赖 $p>0.05$。

---

# 18. 外部泛化和确认性设计

## 18.1 当前 BBOB validation 不再是盲 holdout

仓库已经检查并讨论了这 6 个函数。它仍可作为：

- development holdout；
- finite-set validation；
- 模型选择后的诊断集；

但不应再称为完全独立确认性测试。

## 18.2 CEC2017 同样已经进入开发协议

CEC2017 配置、函数和代码已经被反复讨论，且当前配置还需要修复，因此更适合作为 development external benchmark。

## 18.3 立即é¢åæå®真正 prospective 的外部集

推荐两条互补路线：

### 路线 A：CEC2022

优点：

- 与 CEC2017 不同；
- 函数数目适中；
- 易于形成传统 benchmark 对照。

但它仍是固定小套件，function-level 推断能力有限。

### 路线 B：MA-BBOB 或其他生成器

优点：

- 可生成大量未见函数；
- 更适合评估 selector 的 OOD 泛化；
- 可做 precision planning；
- 能避免只在 24 个 BBOB 类型上反复开发。

MA-BBOB 文献已经显示，从原始 BBOB 训练的 selector 对 affine combinations 可能泛化很差，因此它是有压力的外部测试，而不是方便获得漂亮结果的附属集。

## 18.4 工程问题不能仓促加入

当前算法和协议面向 box-constrained continuous optimization。大量工程问题具有：

- 显式约束；
- 非同尺度变量；
- 离散/混合变量；
- 不可行评价；
- 噪声；
- 未知 optimum。

在没有统一 constraint handling、reference endpoint、transfer rule 和 failure policy 前，不应把“engineering problems”作为装饰性附录。

可选择：

1. 先只使用真正的无约束/box-constrained 工程代理问题；
2. 或é¢åæå®一个统一的 feasibility/penalty 规则，并承认研究对象已扩展。

---

# 19. 理论支撑：什么是合理的，什么尚未成立

## 19.1 最合适的理论框架

该研究最自然地属于：

- cost-sensitive information acquisition；
- active feature acquisition；
- state-conditional value of information；
- contextual decision under acquisition cost。

而不是必须强行包装为：

- reinforcement learning；
- optimal stopping；
- causal inference。

## 19.2 一个可以正式写入论文的基本命题

设状态 $S$ 下只有两个动作：

- $d=0$：不查询；
- $d=1$：执行查询。

令查询相对于 no-query 的随机净效用为 $U(S)$，且决策者风险中性。条件期望效用为：

$$
\mathbb{E}[dU(S)\mid S].
$$

则 Bayes 最优一阶段决策为：

$$
d^*(S)
=
\mathbf{1}
\left[
\mathbb{E}[U(S)\mid S]>0
\right].
$$

### 简证

当 $d=0$ 时条件效用为 0；当 $d=1$ 时条件效用为 $\mathbb{E}[U(S)\mid S]$。逐状态选择二者较大者即可。

这为“连续 utility regression + 零阈值”提供了理论依据。

## 19.3 该命题不能证明什么

它不能自动证明：

- 训练出的 Ridge/LDA/Logistic 得到了真实条件期望；
- OOD 下阈值仍最优；
- first-trigger sequential policy 全局最优；
- 当前 Utility scalarization 是所有用户的真实偏好；
- query realization 的随机性被充分建模。

## 19.4 不应轻易称为 optimal stopping

当前策略在一系列机会点上重复应用一个 classifier，并在首次超过阈值时触发。若要声称是 optimal stopping，需要定义：

- state transition；
- 不触发后未来机会的价值；
- 触发后吸收状态；
- Bellman recursion；
- continuation value。

例如：

$$
V_t(S_t)
=
\max
\left\{
Q_t^{\mathrm{query}}(S_t),
\mathbb{E}[V_{t+1}(S_{t+1})\mid S_t,d_t=0]
\right\}.
$$

当前模型没有估计第二项，因此更准确的称呼是：

> sequentially deployed one-shot acquisition gate with a first-trigger rule。

## 19.5 “纯信息价值”不成立

Query path 同时包含：

- sample best；
- reduced continuation budget；
- feature computation；
- Selector；
- population transfer；
- wall-clock；
- failure behavior。

因此 $U_q$ 是 operational utility，不是纯信息价值。五路径能更好地分解，但仍不是随机化因果估计。

---

# 20. 文档—代码不一致矩阵

| 主题 | 文档状态 | 代码状态 | 建议 |
|---|---|---|---|
| CV grouping | 论文称 function ID | 主要代码用 `family`，BBOB 中仅 5 组 | P0 修复字段与所有 artifact |
| 路径数 | 正文仍按 Skip/Query/Behavior，并称缺 matched path | 活动协议已有五路径 | 重写 Problem Formulation、RQ、Results、Discussion |
| Selector target | 正文主公式为 statewise min–max | 主代码为 relative clipped log gap | 统一公式、符号和命名 |
| cluster-balanced weighting | 多处写“尚未接入/blocker” | weighted imputer、weighted scaler、weighted LDA 和 fit route 已实现并被调用 | 改为“已实现，待端到端动态验证” |
| LDA | 正文类似 sklearn LDA | 自定义 `WeightedLinearDiscriminantAnalysis` | 给出算法定义并做等权一致性测试 |
| Logistic | 正文提 balanced class weights | 代码主要使用 cluster sample weight，`class_weight=None` | 决定唯一正式配置 |
| CEC2017 | F1–F29 | 官方最终 29 题不含 F2、含 F30 | 修复配置和 evaluator 映射 |
| CEC2022 | 文档称 factory 未实现 | `benchmarks/factory.py` 已可路由 OPFUNU | blocker 改为é¢åæå®配置、端点和验证链 |
| matched acquisition | 讨论称不存在 | utility fields/generation 已包含 | 删除旧限制文字 |
| query feature increment | 旧字段名仍残留于部分代码/文档 | 活动协议已有新路径分解 | 清理 deprecated schema |
| Result 状态 | 论文结果章节有结构占位 | 正式结果尚未运行 | 所有结果语气保持 future/placeholder |
| timing role | 部分旧文档混合 action timing 与 complete-path replay | 新代码区分 Stage A/Stage B | 以活动字段规范为唯一来源 |
| family 字段 | 同时承担景观家族和 CV group | 语义混用 | 分拆三个字段 |
| Query 术语 | 部分文档泛称 ELA | 实际只有三套固定 query | 所有主张限定到 evaluated query |

---

# 21. 软件与可复现性不足

## 21.1 当前已有基础

- `pyproject.toml`；
- `uv.lock`；
- 固定 seed stream；
- 大量 schema 与 protocol validation；
- 配置文件；
- 分 shard 执行；
- artifact 字段规范；
- commit-based 开发记录。

这些是良好基础。

## 21.2 当前缺失

静态检查未发现：

- `.github/workflows`；
- 正式 `tests/`；
- `LICENSE`；
- `CITATION.cff`；
- 发布版 protocol manifest；
- 一键 tiny smoke test；
- 自动生成的 config-to-manuscript table；
- 发布 tag 和 immutable result manifest。

## 21.3 必须新增的自动测试

### 状态与预算

- clone continuation bitwise/evaluation-wise equality；
- exact FE accounting；
- partial generation；
- first-hit FE；
- failure cap 与 status 不混淆；
- query sample FE + continuation FE = total path FE。

### 特征

- population permutation invariance；
- coordinate translation/scale invariance；
- behavior window 对齐；
- query feature列白名单；
- non-finite 处理。

### 学习与分组

- function-ID fold 零重叠；
- train-only preprocessing；
- weighted route 真正被调用；
- unit weights 下 custom LDA 与标准实现相符；
- Selector target 符号；
- first-trigger reachability。

### 算法

- SHADE $i,r_1,r_2$ distinctness；
- transfer 不重复评价；
- same algorithm 只 native clone；
- target-specific handoff 元数据；
- trusted implementation competency regression。

### 路径

- 五路径 Utility 加法关系；
- Stage A/Stage B 角色隔离；
- 缓存键不碰撞；
- stale schema 拒绝；
- censored time 规则。

### Benchmark

- CEC2017 final-29 映射；
- BBOB reference value；
- CEC2022 evaluator coverage；
- bounds/reference hash。

## 21.4 建议的发布 manifest

每次正式 run 写入：

```json
{
  "git_commit": "...",
  "dirty_tree": false,
  "python_version": "...",
  "dependency_lock_hash": "...",
  "config_hash": "...",
  "protocol_hash": "...",
  "benchmark_data_hash": "...",
  "machine_id": "...",
  "cpu": "...",
  "threads": 1,
  "seed_registry_version": "...",
  "artifact_schema_version": "..."
}
```

论文中的算法表、函数列表、预算和 feature count 应从同一 YAML/manifest 自动生成。手工复制的数字迟早会分裂成多个“唯一真相”，人类对此有一种近乎宗教式的热爱。

---

# 22. 建议重构后的核心研究问题

将当前 RQ 收敛为四个主问题。

## RQ1：Query acquisition 是否有增量价值？

> 相对于最佳可用 no-query policy，固定独立 query 在哪些状态具有正的资源调整增量？

主量（令 Skip 的相对效用 $U_s(S)=0$）：

$$
I_q(S)=U_q(S)-\max\{0,U_b(S)\}.
$$

若以成本最小化表示，则改为相应 cost difference。

## RQ2：Pre-query Behavior 能否预测该增量？

比较：

- intercept / prior；
- time-only；
- dimension-stratified time-only；
- Behavior groups；
- Behavior + uncertainty diagnostics。

主要评价：

- first-trigger policy utility；
- regret；
- query-use rate；
- calibration/coverage；
- OOD degradation。

## RQ3：收益来自查询链路的哪一部分？

使用五路径 operational decomposition：

- sampling direct effect；
- state-only selection；
- descriptor increment；
- Query vs Behavior-only；
- transfer contribution。

## RQ4：是否能迁移到真正未参与开发的函数分布？

- BBOB：开发与内部估计；
- CEC2017：开发外部集；
- prospective CEC2022 / MA-BBOB：确认；
- engineering：只有在协议扩展完成后加入。

其余 feature stability、maturity association、模型系数等放入 secondary analysis。

---

# 23. 推荐的收敛版实验方案

## 23.1 Phase A：协议修复，不产生论文结果

1. 修复 function-ID CV；
2. 修复 CEC2017；
3. 统一五路径；
4. 统一 Selector target；
5. 统一 weighting/LDA/logistic 文档；
6. 修复或验证 SHADE；
7. 建立 CI 与 smoke config；
8. é¢åæå®外部 prospective suite；
9. 生成 protocol manifest；
10. 禁止读取旧 artifact。

## 23.2 Phase B：算法能力审计

对四算法完成：

- BBOB competency；
- trusted implementation comparison；
- 参数/种群敏感性；
- failure audit；
- transfer audit。

若某算法明显不合格，先修实现，再重新é¢åæå®版本。

## 23.3 Phase C：小规模端到端 pilot

回答：

- 每个 state 实际生成多少物理路径；
- FE 账本是否吻合；
- 状态/路径失败率；
- query feature failure；
- 模型训练耗时；
- artifact 大小；
- timing 方差；
- 同状态 query realization 标签翻转率；
- 预计完整资源。

提前设置 go/no-go：

```text
unexpected implementation failure rate < 0.1%
path identity mismatch = 0
fold leakage = 0
FE accounting mismatch = 0
non-finite primary feature rate < prespecified threshold
wall-clock coefficient of variation < prespecified threshold
```

## 23.4 Phase D：BBOB development 主实验

- cheap query 为唯一主配置；
- function-ID grouped nested CV；
- mandatory milestone 为主；
- event augmentation 为次；
- first-trigger 为主政策评价；
- equal-function aggregation；
- five-path decomposition；
- Behavior-only 作为核心 no-query comparator；
- query realization noise 子实验；
- population transfer 敏感性。

## 23.5 Phase E：query configuration sensitivity

在é¢åæå®子集上比较 standard/broad，不以其结果重新选择主模型或阈值。

## 23.6 Phase F：prospective external confirmation

é¢åæå®后只执行一次：

- 不重新训练超参数；
- 不重新调 threshold；
- 不修改 feature group；
- 不根据 CEC/MA-BBOB 结果回头改主协议；
- 每 suite 单独报告，不把异质 suite 粗暴混成一个平均数。

---

# 24. 建议的主结果表

## Table 1：协议和资源

- suite；
- functions/problems；
- dimensions；
- seeds；
- runs；
- states；
- physical path executions；
- FE；
- wall-clock；
- failures；
- peak RSS。

## Table 2：算法能力审计

- algorithm；
- trusted reference；
- AOCC/ERT difference；
- failure rate；
- parameter configuration；
- pass/fail。

## Table 3：主 first-trigger policy

- Skip；
- Behavior-only；
- Always Query；
- Time-only gate；
- Decision-before-Feature；
- matched random；
- pre-run AAS isolated；
- pre-run AAS sample-reuse。

指标：

- terminal log gap；
- success/ERT；
- query-use rate；
- future-path time；
- FE0-to-terminal time；
- regret；
- 95% interval。

## Table 4：五路径分解

- $U_m$；
- $U_o-U_m$；
- $U_q-U_o$；
- $U_q-U_b$；
- 按 function/dimension/phase 分层。

## Table 5：OOD

每个外部 suite 独立报告：

- absolute policy performance；
- relative to SBS；
- relative to Behavior-only；
- query-use rate；
- coverage；
- failure；
- calibration shift。

## Figure 建议

1. 五路径结构与信息可用时间；
2. Utility 分解瀑布图；
3. first-trigger 时间分布；
4. per-function paired effect；
5. utility vs $\lambda_T$；
6. utility vs objective evaluation cost $c_f$；
7. query realization label stability；
8. OOD calibration/coverage。

---

# 25. 论文贡献建议重写

建议压缩为三项，不把每个工程细节都列成贡献。

## Contribution 1：问题形式化

> We formulate whether to acquire a fixed independent landscape query at an intermediate black-box optimization state as a cost-sensitive, state-conditional information-acquisition decision.

## Contribution 2：泄漏安全的 operational protocol

> We introduce a complete-state, budget-matched five-path protocol that separates native continuation, behavior-only selection, sampling, state-only selection after acquisition, and descriptor-informed selection while retaining failures and measured resource costs.

## Contribution 3：跨函数与 prospective 外部证据

> We evaluate whether pre-query permutation-invariant behavior predicts query acquisition value under function-grouped nested learning and a prospectively predefined external benchmark.

不要在没有结果前写：

- consistently improves；
- generalizes robustly；
- reduces cost without loss；
- first method to solve；
- universally decides whether ELA is worthwhile。

更安全的新颖性表述：

> Among the literature covered by our documented search, we did not identify the same combination of intermediate-state pre-query gating, independent paid landscape acquisition, and matched five-path outcome accounting.

同时在附录记录：

- 检索日期；
- 数据库；
- query；
- 纳入/排除标准。

---

# 26. 逐级修改清单

## 26.1 P0：正式实验前

- [ ] 将所有 CV grouping 改为显式 `function_id`，并增加 leave-family-out 次分析；
- [ ] 决定主 gate 是 Query vs Skip，还是 Query vs best no-query；
- [ ] 将论文全文改为活动五路径；
- [ ] 统一 Selector target 与符号；
- [ ] 修复 CEC2017 F2/F30；
- [ ] 验证并修复 SHADE donor distinctness；
- [ ] 完成四算法 competency audit；
- [ ] 更新 weighting/LDA/logistic/CEC2022 状态；
- [ ] 建立 CI 和 tiny end-to-end test；
- [ ] é¢åæå® prospective external suite；
- [ ] 完成资源 pilot；
- [ ] 实现物理路径缓存与去重；
- [ ] 旧 artifact 全部 schema-invalid。

## 26.2 P1：主实验前

- [ ] query realization noise 子实验；
- [ ] milestone-only 主分析；
- [ ] all-prefix 或 leave-prefix-out 诊断；
- [ ] transfer/restart 敏感性；
- [ ] practical sample-reuse AAS；
- [ ] objective-cost 参数化；
- [ ] $\lambda_T^*$ 与 Pareto 分析；
- [ ] equal-state/equal-run/equal-function estimand 并列；
- [ ] simulation-based precision planning；
- [ ] systemic implementation failure abort rule。

## 26.3 P2：投稿前

- [ ] LICENSE；
- [ ] CITATION.cff；
- [ ] release tag；
- [ ] immutable result manifest；
- [ ] 容器或完整环境说明；
- [ ] 自动生成论文配置表；
- [ ] external artifact DOI/Zenodo；
- [ ] systematic novelty search appendix；
- [ ] 删除过时 TBD 和重复协议；
- [ ] 修正文献综述中的重复词，如 `nested nested`；
- [ ] 统一所有术语：query、acquisition、Behavior-only、state-only、sample-best、endpoint。

---

# 27. Go / No-Go 检查表

只有全部满足，才建议启动完整主实验。

## Protocol

- [ ] 唯一活动规范已确定；
- [ ] 论文公式与代码 target 一致；
- [ ] 五路径名称和字段一致；
- [ ] function ID / family / CV group 不再复用同一字段；
- [ ] external suite 已é¢åæå®。

## Code

- [ ] CI 通过；
- [ ] tiny end-to-end 通过；
- [ ] 无 fold overlap；
- [ ] FE accounting mismatch 为 0；
- [ ] state replay mismatch 为 0；
- [ ] SHADE distinctness 通过；
- [ ] trusted optimizer audit 通过。

## Data

- [ ] 所有 artifact 有 commit/config/protocol hash；
- [ ] 旧 artifact 被拒绝；
- [ ] failure schema 完整；
- [ ] query sample realization 可追踪；
- [ ] path cache 无碰撞。

## Statistics

- [ ] 主 estimand 已é¢åæå®；
- [ ] 主 comparator 已é¢åæå®；
- [ ] primary/secondary 分离；
- [ ] weighting estimand 已说明；
- [ ] precision target 已给出；
- [ ] validation 不被包装成未检查 holdout。

## Resources

- [ ] pilot 实测总 FE；
- [ ] pilot 实测 wall-clock；
- [ ] pilot 实测存储；
- [ ] 完整运行有硬预算；
- [ ] 可从 checkpoint 恢复；
- [ ] 重跑政策预先定义。

---

# 28. 最终评判

## 28.1 是否值得继续

**值得。**

该工作最有价值的不是又设计了一组 Behavior features，而是提出了一个此前在黑盒优化文献中尚未被充分形式化的问题：

> 已经拥有搜索行为信息时，是否仍应为独立的景观查询付费？

这条主线有理论联系、有清楚的实际意义，也能产生负结果仍有价值的论文。

## 28.2 当前最大风险

最大风险不是模型性能不够，而是：

1. 实际代码回答的不是论文写的问题；
2. 巨量计算先于协议定稿；
3. 自定义优化器缺陷被学习器当作规律；
4. BBOB 内部开发结果被误写成外部泛化；
5. 路径分解过多，核心 acquisition claim 反而模糊。

## 28.3 最优修改方向

不建议继续增加模型、query 或 feature。当前最需要的是：

- 收紧主比较对象；
- 消除语义漂移；
- 证明 portfolio 实现可信；
- 把 3500 亿 FE 的物理执行压缩到真正必要的唯一反事实；
- é¢åæå®一个真正 prospective 的外部测试；
- 让五路径服务于一个主问题，而不是让主问题被五路径拖走。

完成这些修改后，该项目会从“极其复杂的研究基础设施”转化为“问题清楚、证据可审计、能够被反驳也能够被复现的方法学论文”。这才是冲击高水平期刊真正需要的形态。

---

# 29. 主要参考文献

1. Mersmann, O., Bischl, B., Trautmann, H., Preuss, M., Weihs, C., & Rudolph, G. (2011). **Exploratory Landscape Analysis**. GECCO. [DOI](https://doi.org/10.1145/2001576.2001690)
2. Kerschke, P., Hoos, H. H., Neumann, F., & Trautmann, H. (2019). **Automated Algorithm Selection: Survey and Perspectives**. *Evolutionary Computation*, 27(1), 3–45. [DOI](https://doi.org/10.1162/evco_a_00242)
3. Kerschke, P., & Trautmann, H. (2019). **Automated Algorithm Selection on Continuous Black-Box Problems by Combining Exploratory Landscape Analysis and Machine Learning**. *Evolutionary Computation*, 27(1), 99–127. [DOI](https://doi.org/10.1162/evco_a_00236)
4. Saar-Tsechansky, M., Melville, P., & Provost, F. (2009). **Active Feature-Value Acquisition**. *Management Science*, 55(4), 664–684. [DOI](https://doi.org/10.1287/mnsc.1080.0952)
5. Jankovic, A., Vermetten, D., Kostovska, A., de Nobel, J., Eftimov, T., & Doerr, C. (2022). **Trajectory-based Algorithm Selection with Warm-starting**. [arXiv:2204.06397](https://arxiv.org/abs/2204.06397)
6. Kostovska, A., Jankovic, A., Vermetten, D., de Nobel, J., Wang, H., Eftimov, T., & Doerr, C. (2022). **Per-run Algorithm Selection with Warm-starting using Trajectory-based Features**. PPSN. [arXiv:2204.09483](https://arxiv.org/abs/2204.09483)
7. van der Blom, K., & Vermetten, D. (2026). **On the Influence of the Feature Computation Budget on Per-Instance Algorithm Selection for Black-Box Optimization**. [arXiv:2605.04954](https://arxiv.org/abs/2605.04954)
8. Cenikj, G., Petelin, G., Seiler, M., Cenikj, N., & Eftimov, T. (2025). **Landscape Features in Single-Objective Continuous Optimization: Have We Hit a Wall in Algorithm Selection Generalization?** *Swarm and Evolutionary Computation*. [DOI](https://doi.org/10.1016/j.swevo.2025.101894)
9. Petelin, G., & Cenikj, G. (2025). **The Pitfalls of Benchmarking in Algorithm Selection: What We Are Getting Wrong**. GECCO, 1181–1189. [DOI](https://doi.org/10.1145/3712256.3726336)
10. Vermetten, D., Ye, F., Bäck, T., & Doerr, C. (2025). **MA-BBOB: A Problem Generator for Black-Box Optimization Using Affine Combinations and Shifts**. *ACM Transactions on Evolutionary Learning and Optimization*, 5(1). [DOI](https://doi.org/10.1145/3673908)
11. Hansen, N., Auger, A., Ros, R., Mersmann, O., Tušar, T., & Brockhoff, D. (2021). **COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting**. *Optimization Methods and Software*, 36(1), 114–144. [DOI](https://doi.org/10.1080/10556788.2020.1808977)
12. Hansen, N. (2016). **The CMA Evolution Strategy: A Tutorial**. [arXiv:1604.00772](https://arxiv.org/abs/1604.00772)
13. Tanabe, R., & Fukunaga, A. (2013). **Success-History Based Parameter Adaptation for Differential Evolution**. IEEE CEC, 71–78. [DOI](https://doi.org/10.1109/CEC.2013.6557555)
14. Varma, S., & Simon, R. (2006). **Bias in Error Estimation When Using Cross-Validation for Model Selection**. *BMC Bioinformatics*, 7, 91. [DOI](https://doi.org/10.1186/1471-2105-7-91)
15. Cawley, G. C., & Talbot, N. L. C. (2010). **On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation**. *Journal of Machine Learning Research*, 11, 2079–2107.
16. Renau, Q., Dréo, J., Doerr, C., & Doerr, B. (2021). **Towards Explainable Exploratory Landscape Analysis: Extreme Feature Selection for Classifying BBOB Functions**. EvoApplications.
17. Prager, R. P., & Trautmann, H. (2024). **Pflacco: Feature-Based Landscape Analysis of Continuous and Constrained Optimization Problems in Python**. *Evolutionary Computation*, 32(3), 211–216. [DOI](https://doi.org/10.1162/evco_a_00341)
18. Suganthan, P. N., et al. **CEC2017 Special Session and Competition on Single Objective Real-Parameter Numerical Optimization: Final Bound-Constrained Problem Definition and Revisions**. [Official competition page](https://sites.google.com/view/suganthan-p-n/cec-benchmark/cec17-special-session)
19. Hansen, N., et al. (2022). **Anytime Performance Assessment in Blackbox Optimization Benchmarking**. *IEEE Transactions on Evolutionary Computation*, 26, 1293–1305. [DOI](https://doi.org/10.1109/TEVC.2022.3210897)

> 注：最终论文应以仓库 `references.bib` 中经过 DOI/Crossref 核验的记录作为唯一书目来源，不建议从本报告手工复制后再维护第二套参考文献。

---

# 30. 仓库证据索引

本报告重点核验的文件包括：

```text
README.md
AGENTS.md
DEVELOPMENT_DECISIONS.md
PROJECT_HANDOFF.md

configs/phase1_bbob_train.yaml
configs/phase1_bbob_validation.yaml
configs/phase1_cec2017_test.yaml

docs/00_master/Decision-before-Feature Master Research Specification.md
docs/10_protocols/Decision-before-Feature_phase1_utility_label_column_spec.md
docs/30_results/phase1_current_results.md

docs/40_manuscript/sections/01_introduction.tex
docs/40_manuscript/sections/02_related_work.tex
docs/40_manuscript/sections/03_problem_formulation.tex
docs/40_manuscript/sections/04_method.tex
docs/40_manuscript/sections/05_experimental_setup.tex
docs/40_manuscript/sections/06_results.tex
docs/40_manuscript/sections/07_discussion.tex
docs/40_manuscript/sections/08_reproducibility.tex
docs/40_manuscript/sections/09_conclusion.tex
docs/40_manuscript/TBD_REQUIREMENTS.md
docs/40_manuscript/references.bib

benchmarks/bbob.py
benchmarks/cec.py
benchmarks/factory.py

trajectory/sampling.py
trajectory/window_statistics.py
trajectory/query.py

behavior/features.py
behavior/streaming.py
behavior/validation.py

landscape_queries/specs.py
landscape_queries/sampling.py
landscape_queries/consistency.py

optimizers/settings.py
optimizers/registry.py
optimizers/state.py

selection_reference/action_losses.py
selection_reference/model.py

utility_labels/fields.py
utility_labels/generation.py

decision/cluster_weighting.py
decision/model_protocol.py
decision/nested_learning.py
decision/train_full_decision_model.py
decision/online_controller_evaluate.py

experiments/phase1_batch_common.py
```

---

## 附：本报告不作出的保证

- 不保证静态审查发现了所有代码缺陷；
- 不把尚未运行的结果写成经验结论；
- 不保证某一目标期刊一定录用；
- 不把 operational path difference 称为因果效应；
- 不以协议长度代替科学贡献；
- 不建议在修复 P0 问题前启动完整正式运行。
