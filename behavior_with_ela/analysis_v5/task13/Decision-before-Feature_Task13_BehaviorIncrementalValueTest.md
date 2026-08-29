# Decision-before-Feature：Task 13 Behavior 增量价值验证 总报告

- 日期：2026-08-29
- 协议：`task13_behavior_incremental_value_test_v1`；replay 3.78M FE（state-reconstruction，单列账本）；**new action-label FE = 0**；未训练 Progress Predictor；未筛 portfolio；未跑 CEC/validation。
- 报告组：`analysis_v5/task13/14a–14i`；轻量表 `analysis_v5/task13/`；行级表 `results/analysis_v5/task13/`；代码 `analysis_v5/task13/{task13_ab_diagn, task13_replay, task13_analysis}.py`。
- 预先固定事实沿用：$P_{balanced}=\{\text{SHADE},\text{L-SHADE},\text{CSO}\}$、$H_a=1000$FE、current-preserving practical 语义（max 规则主、sum 敏感性）、Stage-2 states 全部 natural（本轮只测 $B^{global}$）。

## 一、两个核心问题的回答

**Question 1（deployment）：$[current,FE,B]$ 是否真正优于 $current+FE$？**

**方向上是，强度为 CONDITIONAL（A2）。** 主 carrier（正式 RF pipeline）grouped-OOF 真实 policy loss：

| suite | $L_{M0}^{OOF}$ | $L_{M1}^{OOF}$ | $L_{M2}^{OOF}$ | $\Delta_B$ | 95% CI |
|---|---:|---:|---:|---:|---|
| BBOB | −1.5634 | −1.6054 | −1.6107 | **+0.0473** | [−0.0503, +0.1743]（穿 0） |
| MA | −4.5219 | −4.5732 | −4.5731 | **+0.0513** | [+0.0025, +0.1059]（不含 0） |

- MA 拒绝 $H_0$；BBOB 点估计为正但折间方差大；Ridge 对照不能提取该信号（载体能力边界）。
- **M1 ≈ M2**：current+FE 叠加在 Behavior 上无额外增量（Δ≈0，CI 含 0）——可提取信号几乎全部在 Behavior 中。
- 风险面：M2 的 practical harmful rate 0.136/0.137（M0 为 0.065/0.044）——以更高单步风险换平均收益，须如实进入部署权衡。

**Question 2（更严格）：固定 problem、current、FE 后，Behavior 是否仍能区分不同 state？**

**是（B1 GENUINE STATE VALUE，RF 正式载体）。** within-problem LOSO（378 组 × 5 seeds）：

| suite | $L_{W0}$ | $L_{W2}$ | $\Delta_{within}$ | 95% CI | 组内置换 null q97.5 |
|---|---:|---:|---:|---|---:|
| BBOB | −1.6573 | −1.6760 | **+0.0188** | [+0.0045, +0.0343] | +0.0072 |
| MA | −4.5859 | −4.6020 | **+0.0161** | [+0.0052, +0.0303] | +0.0079 |

真实增量为置换 null 上界的 2.3–2.6 倍；收益覆盖 60%/83% 的 problems。Ridge 对照不支持（4 行训练下退化），已如实声明。

两级 shuffle 对照（O1/O2）与 time-proxy 剔除（bf_fe_ratio，变化 ≤0.006 方向混合）均不破坏结论——增量是**真实的轨迹形状信息**，不是 shuffle 伪影、不是时间/成熟度代理、也不是隐式 problem identifier（problem identity 已知时仍 +0.043/+0.056）。

## 二、最终双层 Verdict

$$
\boxed{\text{Verdict A（Deployment Increment）：A2 CONDITIONAL}}
$$
依据：MA 显著为正、BBOB 点估计正但 CI 穿 0（§18 A2 的第三种情形）；两 suite 方向一致（RF），无 A3 的方向冲突；harmful rate 升高已记录但平均损失更优。

$$
\boxed{\text{Verdict B（Genuine Dynamic State Increment）：B1 GENUINE STATE VALUE}}
$$
依据：$\Delta_{within}$ 两 suite CI>0、超置换 null、跨多数 problems；载体边界（Ridge）与 BBOB 增量质量集中度（top-3=73%）作为限定条件一并声明。

**按 §31：Case 1（A1/A2 + B1）成立 → 下一阶段为 Post-Handoff Sequential Confirmation。**

## 三、§33 的 33 个问题逐条回答

| # | 问题 | 回答 |
|---|---|---|
| 1 | independent winner's-curse bias | BBOB fb **+0.112** [0.020, 0.223]（n=137）；MA **+0.034** [−0.003, 0.072]（n=57） |
| 2 | 是否比 Task 12.1 更大 | 是（0.102→0.112；0.026→0.034），与 r0 污染假设方向一致，差异小；仍仅作诊断、不外推 |
| 3 | max/quad/sum 的 switch-required | 0.258/0.265 → 0.226/0.252 → **0.187/0.211** |
| 4 | 非退化性对保守 δ 是否稳健 | **是（STABLE）**：sum 下 ≥0.187≥0.10，三对全双向（最小方向概率 0.106），无 $A_{ND}$ 空集 |
| 5 | replay 是否 100% 对齐 | **是**：1890/1890 键与 gap 全对，max diff = 0.0（容差 1e-12）；L-SHADE 实际种群 {33,26,18} 已记录 |
| 6 | Behavior 是否 future leakage | **否**：全部特征源自 ≤t 的完整 update 历史（窗口 anchor ≤ checkpoint）；无 t+1000/outcome/reference 字段 |
| 7 | M0 是否复现 Task 12.1 | 同 split 哲学；经验基线 −1.5856/−4.5298；M0-Ridge −1.6012/**−4.5299**（MA 精确复现），M0-RF −1.5634/−4.5219（carrier 拟合噪声 ≤0.022）；主比较用同 carrier 配对，不受偏移影响 |
| 8 | M1 OOF | RF：−1.6054 / −4.5732 |
| 9 | M2 OOF | RF：**−1.6107 / −4.5731** |
| 10 | $\Delta_B$ | BBOB **+0.0473**；MA **+0.0513** |
| 11 | 95% CI 是否跨 0 | BBOB 跨 0；MA 不跨（+0.0025 起） |
| 12 | M2 harmful rate | 0.136 / 0.137（M0：0.065/0.044；δ=pairwise max 规则）——升高，风险项 |
| 13 | per-action Spearman | M2：BBOB 0.494/0.487/0.480；MA 0.855/0.852/0.834（M0 仅 0.14–0.26；M4≈0；M3 为负） |
| 14 | 是否少数 problem 驱动 | 覆盖广（within：60%/83% problems 为正）；BBOB 增量质量偏集中（top-3=73%），MA 相对分散，已记录 |
| 15 | $\Delta_{within}$ | RF：**+0.0188** [0.0045, 0.0343] / **+0.0161** [0.0052, 0.0303] |
| 16 | 固定 problem/current/FE 后是否有价值 | **是（B1）**，且超组内置换 null 2.3–2.6×；Ridge 不支持为载体边界 |
| 17 | shuffle 后增量是否消失 | **是**：O1 真实 0.047/0.051 ≫ null q97.5 0.004/0.011；O2 真实 0.019/0.016 ≫ q97.5 0.007/0.008 |
| 18 | 去 time-like 后是否保留 | **是**：剔除 bf_fe_ratio 后 RF 变化 ≤0.006、两 suite 方向相反（非该代理驱动） |
| 19 | Behavior 是否主要充当 problem identifier | **否**：problem identity 已知时仍 +0.043/+0.056（RF 诊断，NON-DEPLOYABLE）；within-problem 亦正 |
| 20 | Verdict A | **A2 CONDITIONAL** |
| 21 | Verdict B | **B1 GENUINE STATE VALUE** |
| 22 | ProgressForecast | **仍 PG3 NO-GO**（Task 12.1 结构性结论，本轮未推翻也不需推翻） |
| 23 | 是否允许 post-handoff sequential confirmation | **是**（Case 1：A2+B1） |
| 24 | 是否允许测试 true segment Behavior | **是**——但只能在 post-handoff states 上（本轮仍无 segment 语义，未伪造） |
| 25 | 是否允许 restart/reset controls | **是**，与 post-handoff 采集同轮预注册执行（SHADE/L-SHADE population-preserving reset） |
| 26 | 正式 CEC | **继续 PAUSED**；CEC2022 继续 held out |
| 27 | 下一阶段 | **Post-Handoff Sequential Confirmation**（详见 §五） |

## 四、成本账本

| 项 | 值 |
|---|---:|
| new action-label FE | **0** |
| state-reconstruction FE | **3,780,000**（=预注册上限；42×5×3×6000） |
| 分析 wall time / peak RSS | 见 `results/analysis_v5/task13/task13_resource_ledger.parquet`（replay ≈3 min @8 workers，近似值） |

方法学声明（必须随论文方法学一并披露）：正式窗口模块为支持收缩种群（L-SHADE NP 40→4）做了最小扩展——量化容差改用 anchor 邻域实际 update 间隔、不等长端点分位数用共同概率网格；**NP 恒定路径经回归验证逐位不变**（bbob_f001 重提取 1274 行 × 全部 bf_* 列 diff=0.0）。详见 14c §3。

## 五、下一阶段：Post-Handoff Sequential Confirmation（Case 1）

1. 构造真实 post-handoff states：6 个方向（SHADE↔L-SHADE、SHADE↔CSO、L-SHADE↔CSO）× checkpoint × 1000-FE 分支，预先固定 Stage-2 问题集与 seed；
2. 首次合法测试 $B^{segment}$：$B^{global}$ vs $B^{segment}$ vs $[B^{global},B^{segment}]$（基线仍为 current+FE OOF）；
3. 同轮预注册执行 SHADE reset control / L-SHADE reset control（population-preserving，分离 restart 效应）；
4. ProgressForecast 不复活（PG3 negative structural ablation 保留）；正式 CEC 继续暂停。

### 下一步 prompt（可直接复制开新对话）

```
你正在继续 GitHub 项目 Decision-before-Feature（目录 behavior_with_ela/）。
先读 AGENTS.md、analysis_v5/Task12/Task12.1/Task13 三份总报告与
analysis_v5/task13/14a-14i。当前状态：P_balanced={shade,lshade,cso} 不变；
Task13 verdict：A2 CONDITIONAL（Δ_B=+0.047/+0.051，MA CI>0、BBOB 穿 0）+
B1 GENUINE STATE VALUE（Δ_within=+0.019/+0.016 双 CI>0，超 shuffle null）；
replay 对齐 1890/1890 diff=0；ProgressForecast=PG3 NO-GO。

本轮任务：Post-Handoff Sequential Confirmation。
1. 构造真实 post-handoff states：在 Task 12 Stage-2 同一 problem×seed×
   checkpoint 集上，按 6 个方向（shade↔lshade, shade↔cso, lshade↔cso）
   执行 population-transfer handoff 后运行 1000-FE 分支（continue 与
   switch-to-other-two），并沿 handoff 后轨迹在 FE∈{t+1000} 克隆
   post-handoff checkpoint（segment_start=t）；
2. 同轮预注册执行 reset controls：SHADE-current 与 L-SHADE-current 的
   population-preserving reset 分支（分离 restart/memory-reset 效应）；
3. 正式 recorder 记录 handoff 后的 B_global 与真实 segment 语义的
   B_segment（segment_start=handoff 点，禁止用 global 复制冒充）；
4. 评价（零新 action-label FE 之外的最小分支成本须在资源账本单列）：
   基线 current+FE-OOF vs B_global vs B_segment vs 并集；
   grouped leave-cv_group-out，RF 正式 carrier + Ridge 对照；
   全部使用真实 1000-FE outcome；BBOB/MA 分列 + paired bootstrap CI；
5. 判定：B_segment 或并集须在两 suite 同时优于 B_global 且 CI>0 才算
   SEGMENT INCREMENT CONFIRMED；否则 NO SEGMENT INCREMENT；
6. 输出 analysis_v5/task14/ 报告组与轻量表、资源账本（区分分支 FE 与
   分析），遵守 AGENTS.md §0.3 的学术措辞规范与 §0.2 的工程机制禁令；
7. 禁止：ProgressForecast、portfolio 重选、调参搜索、新算法、CEC、
   validation、删除阴性结果。
```

## 六、停止声明

按工作单 §35 链条全部执行完毕：独立 curse 诊断 → delta 敏感性（STABLE）→ 确定性 replay → 逐位对齐（1890/1890, diff=0）→ 特征审计 → dataset → grouped-OOF → within-problem LOSO → shuffle controls → time-proxy → 双 verdict（A2 + B1）→ **STOP**。未自动进入 post-handoff 采集、segment 测试、reset-control FE、ProgressForecast、validation 或 formal CEC。
