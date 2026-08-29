# 15e · Within-Problem Permutation 扩展至 100 次（Task 13.1J/K）

- 日期：2026-08-29。协议：保持 Task 13 O2 原定义——在 (problem, current, FE) 组内置换 Behavior，重跑 4 train / 1 test LOSO；正式 RF carrier；**$N_{perm}=100$（预注册主结果，不提前停止）**。W2 腿与 Task 13 完全同 carrier/折/特征（W1 腿与 W0–W2 null 无关，省略）。实现：`task13_1_perm.py`（8 进程并行，约 35 分钟）。
- 产物：`within_problem_permutation_100.parquet`、`within_problem_permutation_summary.parquet`、`within_problem_permutation_observed.parquet`。

## 1. 观测值复核

同代码重算未置换 $\Delta_{within}$：BBOB **+0.01877**、MA **+0.01609**——与 Task 13 提交的 within-problem 结果（+0.0188/+0.0161）完全一致（实现一致性验证通过）。

## 2. 100 次 permutation null 与经验 p 值

| suite | $\Delta_{within}^{obs}$ | null mean | null std | q95 | q97.5 | **empirical one-sided p** |
|---|---:|---:|---:|---:|---:|---:|
| BBOB | +0.01877 | +0.00107 | 0.00485 | +0.00860 | +0.00928 | **0.0099**（=1/101） |
| MA | +0.01609 | +0.00007 | 0.00466 | +0.00763 | +0.00908 | **0.0099**（=1/101） |

两个 suite 的观测增量**超过全部 100 个置换样本**（p = (1+0)/(1+100) = 0.0099 ≤ 0.05），约为 q97.5 的 2 倍。Task 13 基于 10 次 permutation 的初步结论在正式 100 次协议下**成立且更稳健**。

## 3. 措辞边界（Task 13.1K）

每个 (problem, current, FE) 组只有 5 个 seeds，每个 LOSO fold 的训练集只有 4 行；置换 null 排除的是"组内 B–seed 对应关系可被任意替换"的零假设，不排除更弱的替代解释（如 4-NN 式记忆效应的具体机制）。因此合法表述为：

> **Behavior contains within-problem state-discriminative signal under the fixed RF carrier**（固定 RF 载体下，Behavior 含有可区分同问题内部不同搜索状态的信号，p≈0.01，两 suite 一致）。

不得表述为"已在每个 problem 内学到普遍可泛化的 state policy"。

## 4. 实现修正声明

首轮运行发现 slim 实现的 W0 均值误含 held-out 状态（5 行而非 4 行），导致观测值符号翻转（−0.030）；已修正为严格 4-train 均值并**全量重跑 100 次**，修正后观测值与 Task 13 逐位一致。该错误仅存在于本轮临时实现，不影响 Task 13 已提交的任何 artifact。
