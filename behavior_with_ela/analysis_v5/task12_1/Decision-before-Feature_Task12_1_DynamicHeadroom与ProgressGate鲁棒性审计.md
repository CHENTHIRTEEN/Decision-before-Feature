# Decision-before-Feature：Task 12.1 动态选择余量稳健性复核 × Progress Gate 结构可行性 总报告

- 日期：2026-08-29
- 性质：Task 12 与 Behavior Incremental Test 之间的方法学复核；**不训练 Behavior Selector / Progress Predictor，不筛选 portfolio，零新增 action-label FE（state-reconstruction FE 亦为 0）**。
- 报告组：`analysis_v5/task12_1/13a–13g`；轻量表 `analysis_v5/task12_1/`；行级表 `results/analysis_v5/task12_1/`；分析代码 `analysis_v5/task12_1_analysis.py`。
- $P_{balanced}=\{\text{SHADE},\text{L-SHADE},\text{CSO}\}$ 保持预先固定，未因本轮结果重选。

## 一、两个核心问题的回答

**Question A：修正 current context、practical tie 与 noise 之后，$\{SHADE, L\text{-}SHADE, CSO\}$ 是否仍有真实 dynamic 性能差？**

**有，但显著小于 Task 12 的名义读数，且其"逐状态"部分的绝对幅度受 winner's-curse 不确定性限制。** 关键数字（fb log10 损失，95% CI 经 cv_group bootstrap）：

| 量 | BBOB | MA | 说明 |
|---|---:|---:|---|
| $\Delta_{portfolio}=L_{SBS}-L_{statewise}$ | +0.209 [0.125, 0.308] | +0.195 [0.122, 0.281] | 静态/上下文组合价值，稳健 |
| $\Delta_{deploy-residual}=L_{current+FE}^{OOF}-L_{statewise}$ | **+0.173 [0.079, 0.294]** | **+0.160 [0.113, 0.205]** | 可部署 simple baseline 之外的逐状态空间（名义） |
| $\Delta_{context-residual}=L_{problem+current+FE}^{desc}-L_{statewise}$ | +0.053 [0.034, 0.074] | +0.050 [0.029, 0.078] | 完整 simple context 之后的残差 |
| $\Delta_{dynamic}^{old}$（Task 12 原指标） | +0.114 | +0.085 | 被 current identity 吸收 54%/41%，**superseded** |

- 旧 practical 标签的 tie 偏置被证实并修正：50.3% states 的 tie set >1，24.2% 的标签被 `tied[0]` 移离 raw argmin；修正后 **switch-required rate = 26%（0.258/0.265）**，三种噪声语义下稳定（0.254–0.278）。
- 三对互补性在 pairwise conservative δ 下**全部保持双向**（方向概率 ≥0.13，DCM 0.25–0.37，随语义变化 ≤0.02）。
- **必须如实声明的保留项**：逐状态最佳动作参考存在乐观偏差（winner's curse）：重复子集诊断 BBOB +0.102（n=137）、MA +0.026（n=57），分层估计不稳定（n=2–20）。按该诊断的保守读法，$\Delta_{deploy-residual}$ 的真值区间约 [0.07, 0.17]（BBOB），而 $\Delta_{context-residual}$（≈0.05）与该偏差不可区分——**"完整上下文之后剩余的真实逐状态价值"目前无法与 0 可靠区分**。未对全数据集做外推校正（重复覆盖不足）。

**Question B：如果连真实 Future Progress 都已知，它是否足以作为有价值的 Switch Opportunity Gate？**

**否（结构性 NO-GO）。** 使用真实已实现 progress（oracle 上限，非模型）：

- AUROC($-R\rightarrow Z$)：BBOB 0.489 [0.374, 0.622]、MA 0.516 [0.406, 0.644]——CI 均覆盖 0.5；AP ≈ base rate（0.230 vs 0.258；0.251 vs 0.265）。
- 唯一可见的关联 Spearman($-R, G_{best-switch}$)≈0.37 在扣除 pairwise 噪声后归零（$G_{practical}$：−0.035/+0.006）。
- 低 progress 触发（q=0.2/0.3/0.4）**不富集反而贫化** switch opportunity（BBOB enrichment 0.37/0.74/0.86；recall ≤0.43；missed ≥0.57）。
- 高进展对照（exhaustive）：$P(Z{=}1\mid HighProgress)$=0.25–0.30 ≈ base rate——gate 对"仍有进展但别的 solver 更好"的 states 没有任何选择性。

$$\boxed{\text{ProgressForecast structural verdict：PG3 NO-GO}}$$

在 $H_g=1000$FE、$P_{balanced}$、practical 语义、当前训练域 state 分布下，$B_t\rightarrow\hat R$ 即使预测完美也没有门价值；后续 pipeline 不应再为 ProgressForecast 预留位置。

## 二、RQ1–RQ5 逐条回答

1. **RQ1（current 之后还剩多少）**：可部署口径（current+FE OOF）之外名义残差 0.173/0.160；完整上下文（problem+current+FE）之后 0.053/0.050，且与逐状态参考的乐观偏差不可区分。旧 $\Delta_{dynamic}^{old}=0.114/0.085$ 中 54%/41% 被 current identity 吸收。
2. **RQ2（去 tie 后动作空间是否非退化）**：非退化。switch-required 26%、$P(c\in A_{ND})$=0.74、$|A_{ND}|{>}1$ 占 49–53%、$H(A_{op})$=1.22–1.24 bits、无 $A_{ND}$ 空集、无近全支配。
3. **RQ3（pairwise noise 后互补性是否成立）**：成立。三对 × 两 suite 全部双向；直接配对重复不足（$N_{paired}$=5–17 < 30，全部 INSUFFICIENT），按预注册 fallback 使用保守 $\max(\delta_i,\delta_j)$，legacy/pooled 作敏感性。
4. **RQ4（ProgressForecast 结构上是否有意义）**：无（PG3，见 Question B）。
5. **RQ5（Task 12 数据能否测 segment Behavior）**：不能。states 全部 natural（segment_start=0），$B^{segment}\approx B^{global}$ 构造使然；真正 segment 价值 untested 直到 post-handoff states 存在。且 states 现在连 $B^{global}$ 都没有（需先做带 recorder 的状态重建，≈3.78M FE 账目单列；对齐可行性已被两条逐位恒等式证明）。

## 三、最终 verdict

$$
\boxed{\text{Task 12 Verdict：V2 CONDITIONAL P1}}
$$

成立依据：无 single-solver 近全支配（P_continue=0.74，target joint ≤0.21）；pairwise DCM 全部双向；current-preserving switch-required rate 非平凡（26%）；$\Delta_{deploy-residual}>0$（两 suite CI>0）；$\Delta_{context-residual}>0$；两 suite 方向一致无反转；结论不依赖 `tied[0]`。**未满足 V1 的条件 8**：逐状态参考的乐观偏差（BBOB 点估计 ≈0.10）与 $\Delta_{context-residual}$ 同量级、且不确定性大，因此"main residual 是否被 curse 解释"无法排除——这正是 V2 与 V1 的分界。

$$
\boxed{\text{Progress verdict：PG3 NO-GO（结构上限意义下）}}
$$

按 §27 GO 门：verdict ∈ {ROBUST, CONDITIONAL} 且 $\Delta_{deploy-residual}$ 名义明显非零 → **允许进入 Behavior Incremental Test，但仅作为 CONDITIONAL DEVELOPMENT TEST**，且：

1. 超越目标必须定义为 $\text{Performance}(current+FE+B)>\text{Performance}(current+FE^{OOF})$；
2. 前置步骤：带正式 recorder 的确定性状态重建（3.78M FE 上限，账目单列，replay 后必须通过逐位对齐审计）；
3. 不得进入 ProgressForecast full pipeline（PG3）；
4. 建议同时预留一个低成本重复扩充设计（对少量 states R=3→≥5 或扩 seeds），用于钉住 winner's-curse 幅度——该动作涉新增 FE，需另行授权，不默认执行。

## 四、§36 的 29 个问题逐条回答

| # | 问题 | 回答 |
|---|---|---|
| 1 | Stage 2 是否全部 natural states | 是（1890/1890，source==current==route） |
| 2 | 是否没有 post-handoff segment states | 是，没有（handoff 全 False，dwell==FE） |
| 3 | 旧 practical best 是否有 candidate-order tie bias | 有：50.3% tie set>1，24.2% 标签被移离 raw argmin |
| 4 | 去 tie 后 switch-required rate | 0.258（BBOB）/ 0.265（MA）；legacy 0.261/0.278，pooled 0.254/0.261 |
| 5 | current-preserving 转移是否仍有切换结构 | 有：条件 target 集中（lshade→cso 0.64–0.70 等），且集中在 FE=2000（0.35/0.43） |
| 6 | pairwise DCM 是否对噪声语义稳健 | 稳健：三语义 DCM 变化 ≤0.02，全部保持双向 |
| 7 | $L_{current+FE}$ | desc −1.6117/−4.5719；**OOF −1.5856/−4.5298** |
| 8 | $L_{problem+current+FE}$ | desc −1.7059/−4.6401；LOSO-seed −1.6573/−4.5859 |
| 9 | 旧 $\Delta_{dynamic}^{old}$ 被 current 吃掉多少 | BBOB 0.114→0.053（54%）、MA 0.085→0.050（41%） |
| 10 | $\Delta_{deploy-residual}$ | +0.173 [0.079,0.294] / +0.160 [0.113,0.205]（curse 修正后约 [0.07,0.17]/[0.13,0.16]） |
| 11 | $\Delta_{context-residual}$ | +0.053 [0.034,0.074] / +0.050 [0.029,0.078]，与 curse 偏差不可区分 |
| 12 | statewise oracle 是否有明显 winner's-curse | 是：BBOB +0.102（n=137）、MA +0.026（n=57）；分层不稳（n=2–20），仅作诊断未外推 |
| 13 | 三 solver 是否仍 non-degenerate dynamic portfolio | 是（26% switch-required、无支配失控、DCM 双向、H_op 1.22–1.24 bits） |
| 14 | Task 12 Verdict | **V2 CONDITIONAL P1**（V1 条件 8 未满足） |
| 15 | 真实 progress 值与 switch opportunity 是否相关 | 实用语义下不相关（raw 秩相关 0.37 全部落在噪声内） |
| 16 | AUROC/AP | AUROC 0.489/0.516（CI 含 0.5）；AP 0.230/0.251 ≈ base rate 0.258/0.265 |
| 17 | bottom 20/30/40% 覆盖多少 switch opportunity | recall 0.075/0.221/0.345（BBOB）、0.105/0.280/0.427（MA） |
| 18 | missed-switch rates | 0.925/0.779/0.655（BBOB）、0.895/0.720/0.573（MA） |
| 19 | theoretical branch reduction | 0.80/0.70/0.60（=1−ρ，代价是漏掉上行的多数机会） |
| 20 | 高进展 states 中还有多少 useful switch | $P(Z{=}1\mid HP)$=0.25–0.30 ≈ base；$E[G_{practical}\mid HP]$=0.14–0.17 |
| 21 | Progress structural verdict | **PG3 NO-GO** |
| 22 | Task 12 states 能否测真 segment Behavior | 不能（segment_start=0，$B^{seg}\approx B^{glob}$ 构造使然） |
| 23 | 是否需要 replay 才能提取 global Behavior | 是（states 无 bg_*；progress 审计本身不需要 replay） |
| 24 | 若 replay 是否与旧 state 完全对齐 | 对齐可行性已证明：两条恒等式 1890/1890 逐位相等（diff=0.0） |
| 25 | CMAES add-back 新语义下 | **ONE-STEP ADD-BACK: NO COLLAPSE**（两 suite；ratio 0.826/0.651） |
| 26 | repeated CMAES-current 是否仍 unresolved | 是，仍 unresolved（无 CMAES-current 成熟重复状态） |
| 27 | 是否允许进入 Behavior Incremental Test | 允许，但仅 CONDITIONAL DEVELOPMENT TEST（先做带 recorder 的状态重建；基线=current+FE OOF） |
| 28 | 是否允许立即训练 ProgressForecast | **NO**（本轮协议禁止；且结构 verdict 为 PG3，训练亦无门价值） |
| 29 | 下一阶段 | Behavior Incremental Test（CONDITIONAL）：状态重建 replay+recorder → grouped-OOF 的 M0 vs M1 vs M2，目标 Performance(current+FE+B) > Performance(current+FE^OOF) |

## 五、成本账本

| 项 | 值 |
|---|---:|
| 新增 action-label FE | **0** |
| state-reconstruction（natural replay）FE | **0**（$\ell_t$ 取自 Task 12 Stage-1 marks；两条恒等式 1890/1890 逐位验证） |
| 本轮分析 wall time / peak RSS | ≈15 s / ≈216 MB（`results/analysis_v5/task12_1/task12_1_resource_ledger.parquet`） |

## 六、停止声明

按工作单 §38 链条：语义核查 → current-conditioned ladder → OOF simple baseline → winner's-curse 诊断 → pairwise noise → set-valued practical actions → current-preserving 重分析 → robust DCM → checkpoint gap 恢复（零 FE）→ realized progress → progress 结构审计 → 高进展对照 → CMAES one-step add-back → Behavior 数据就绪 → 双 verdict，全部完成，**STOP**。未训练任何模型、未新增分支、未跑 CEC/validation、未重选 portfolio、未修改 Task 9–12 产物。

## 七、下一步建议

1. **（推荐）Behavior Incremental Test（CONDITIONAL DEVELOPMENT TEST）**：先执行带正式 recorder 的确定性状态重建 replay（≈3.78M FE，单列账目，逐位对齐审计），再在同一 Stage-2 域上做 grouped-OOF 的 M0/M1/M2 对比；基线 $L_{current+FE}^{OOF}$=−1.5856/−4.5298。
2. **（可选、需授权）重复扩充**：对 1890 states 的一个预注册子集把 R=3 提到 R≥5（或扩 seeds），专门用于钉住逐状态最佳动作的 winner's-curse 幅度；这是把 V2 升级为 V1 判定的最短路径。
3. **不做**：ProgressForecast（PG3）；portfolio 重选；CEC/validation；post-handoff 采集。

### 下一步 prompt（可直接复制开新对话）

```
你正在继续 GitHub 项目 Decision-before-Feature（目录 behavior_with_ela/）。
先读 AGENTS.md、analysis_v5/Task12 总报告、analysis_v5/task12_1/ 全部 13a-13g 报告
与总报告 Decision-before-Feature_Task12_1_DynamicHeadroom与ProgressGate鲁棒性审计.md。
当前状态：P_balanced={shade,lshade,cso} 不变；Task 12.1 verdict = V2 CONDITIONAL P1
（Δ_deploy-residual=+0.173/+0.160，但 winner's-curse 诊断 BBOB +0.102 未排除）；
Progress verdict = PG3 NO-GO（真实 progress 对 practical switch opportunity 无判别力，
AUROC≈0.5）；states 无 bg_*/bs_*；ℓ_t 已由 stage-1 marks 恢复（两条恒等式逐位对齐）。

本轮任务：Behavior Incremental Test（CONDITIONAL DEVELOPMENT TEST）。
1. 先做确定性状态重建 replay：shade/lshade/cso 自然跑 0→6000FE
   （42 problems × 5 seeds ≈3.78M FE 上限，账目单列 state-reconstruction FE），
   同跑启用正式 recorder 保存 FE∈{2000,4000,6000} 的 bg_28；
   必须复现 analysis_v5/task12_1/replay_alignment.parquet 的两条恒等式
   （terminal 与 continue 逐位相等），任何失配即 STOP；
2. 复用 results/analysis_v5/task12/{dynamic_solver_loss_matrix}.parquet 的 1000-FE
   标签，按 state_id 与 replayed Behavior 对齐；
3. 模型一律 RandomForestRegressor(200, max_depth 8, max_features=sqrt)，
   特征组：M0=current+FE one-hot；M1=bg_28；M2=[current,FE,bg_28]；
   grouped OOF 按 cv_group；禁止 segment 行为（不得把 bg 复制为 bs）；
4. 主评价：fb policy loss、相对 M0-OOF 基线（-1.5856/-4.5298）的策略增益、
   practical 达标率（pairwise conservative δ）；BBOB 与 MA 分列；
5. 判定：仅当某 Behavior 组在两个 suite 同时优于 M0 且增益 CI>0 才算
   INCREMENT CONFIRMED；否则 NO INCREMENT（并按 Task12.1 报告口径降级处理）；
6. 输出 analysis_v5/task12_2/ 报告与轻量表、资源账本（区分 replay FE 与分析）；
7. 禁止：ProgressForecast（PG3）、portfolio 重选、调参搜索、CEC、validation、
   删除阴性结果、把 statewise oracle 当基线。
```
