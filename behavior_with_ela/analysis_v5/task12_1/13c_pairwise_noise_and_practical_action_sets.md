# 13c · Pairwise Practical Noise 与 Set-Valued Practical Action 定义（Task 12.1F/G）

- 日期：2026-08-29
- 数据：Task 12 重复分支（557 cell × R=3）、add-back 重复分支（206 cell × R=3）、1890-state solver 矩阵。零新增 objective evaluations。
- 产物：`pairwise_noise_deltas.parquet`、`cmaes_action_noise.parquet`、`practical_action_sets.parquet`（行级，1890×3 语义）。

## 1. 直接 pairwise delta（F1/F2）：全部 INSUFFICIENT

对每个 suite × pair，统计"两个 solver cell 都被重复抽样"的 states，用 replicate 配对差 $D_{ij}^{(r)}=L_i^{(r)}-L_j^{(r)}$ 的函数平衡 $Q_{0.95}(|D-\mathrm{median}\,D|)$ 估计：

| suite | pair | $N_{paired}$ | 直接 $\delta_{ij,95}$ | 判定（预注册 $N\ge30$） |
|---|---|---:|---:|---|
| BBOB | shade\|lshade | 10 | 0.442 | **INSUFFICIENT** |
| BBOB | shade\|cso | 17 | 0.475 | **INSUFFICIENT** |
| BBOB | lshade\|cso | 12 | 0.803 | **INSUFFICIENT** |
| MA | shade\|lshade | 8 | 0.162 | **INSUFFICIENT** |
| MA | shade\|cso | 5 | 0.377 | **INSUFFICIENT** |
| MA | lshade\|cso | 7 | 0.220 | **INSUFFICIENT** |

$N_{paired}\approx 0.1^2\times 630$ 的量级，5–17 全部低于预注册下限 30；直接估计（0.16–0.80）样本过少、数值不稳定，**不用于任何判定**。

## 2. 主判据：保守 pair-specific fallback（F3）

$$\delta_{ij,95}^{cons}(s)=\max\big(\delta_{95}(s,\text{cell }i),\ \delta_{95}(s,\text{cell }j)\big),$$

其中 cell 的 raw action 为 `continue`（cell==current 时）或 `switch-to-a`，其 $\delta_{95}$ 取 Task 12 已标定的 (suite × action) 函数平衡值：BBOB continue 0.0874 / shade 0.0754 / lshade 0.1012 / cso 0.0863；MA continue 0.0852 / shade 0.0670 / lshade 0.0580 / cso 0.1253。CMA-ES 作为动作的 $\delta_{95}$ 由 add-back 重复 cell 用同一估计器补齐：**BBOB 0.1626、MA 0.1112**（`cmaes_action_noise.parquet`）。

敏感性保留：legacy suite 均值 δ95（BBOB 0.0876 / MA 0.0839）与 pooled action-pair max。**未为获得更多 switch 选择最小阈值**；三种语义下关键结论一致性见 13d。

## 3. Set-Valued Practical Action（G）

删除 `best_practical = tied[0]`；pairwise practical superiority

$$a\succ_\delta b \iff L(s,a)<L(s,b)-\delta_{ab,95}^{cons}(s),\qquad A_{ND}(s)=\{a:\nexists b,\ b\succ_\delta a\}.$$

- **空集处理（预注册）**：不可传递支配可致 $A_{ND}(s)=\varnothing$；实测三种语义、两 suite、3/4 动作空间下 **P(A_ND 为空)=0**（1890×全部设定无一一例）， operational 回退从未触发。
- current-preserving 规则：$c\in A_{ND}(s)\Rightarrow$ operational action = `continue`；仅 $c\notin A_{ND}(s)$ 记为 switch opportunity（$Z(s)=1$），唯一 target 取 $A_{ND}(s)$ 内 raw loss 最小者（仅作 operational 汇总，科学分析保留完整 $A_{ND}$）。

## 4. 三种语义下的基本量（3 动作空间）

| 语义 | suite | $P(c\in A_{ND})$ | switch-required | optional-switch（$c\in A_{ND}\wedge|A_{ND}|{>}1$） | $E|A_{ND}|$ |
|---|---|---:|---:|---:|---:|
| pairwise（主） | BBOB | 0.742 | **0.258** | 0.464 | 1.88 |
| pairwise（主） | MA | 0.735 | **0.265** | 0.430 | 1.78 |
| legacy | BBOB | 0.739 | 0.261 | 0.456 | 1.86 |
| legacy | MA | 0.722 | 0.278 | 0.407 | 1.75 |
| pooled | BBOB | 0.746 | 0.254 | 0.476 | 1.91 |
| pooled | MA | 0.739 | 0.261 | 0.435 | 1.79 |

结论：**current-preserving 语义下 switch-required rate ≈26%（两 suite、三语义一致）**；约 43–48% 的 states 处于"current 可保留但集合非平凡"的 optional 区。Task 12 由 tied[0] 得到的"practical best 落在其它 solver"份额（如 lshade→shade 43%）混杂了 tie 顺序偏置，不能与 0.26 直接比较；二者差异恰说明旧口径把大量 within-δ 平局误计为切换。
