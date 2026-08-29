# 05 · Action-Horizon Disagreement（Task 11J）

- 日期：2026-08-30
- 问题：$a^\star_{500}/a^\star_{1000}/a^\star_T$ 是否为同一个动作？terminal-commitment 标签能否近似 segment-control 标签？

## J0. 总体 disagreement（4,680 states）

| 量 | 值 |
|---|---:|
| $P(a^\star_{500}\neq a^\star_{1000})$ | 0.062 |
| $P(a^\star_{1000}\neq a^\star_{T})$ | **0.164** |
| $P(a^\star_{500}\neq a^\star_{T})$ | 0.175 |
| practical：best@1000 在 terminal 上超出 $\delta_{T,95}$ | 0.076 |
| practical：best@T 在 1000 上超出 $\delta_{1000,95}$ | 0.063 |
| pattern「best@1000 = switch 且 best@T = continue」 | 200 states = 4.3%（跨 19/42 个 function group） |

## J1. Transition matrix（best@1000 → best@T，行归一）

| | continue | pso | shade |
|---|---:|---:|---:|
| continue | 0.917 | 0.058 | 0.025 |
| pso | 0.462 | 0.505 | 0.033 |
| shade | 0.579 | 0.057 | 0.363 |

（完整计数表 `results/analysis_v4/task11/multi_horizon_action_outcomes.parquet` 可复核。）1000-FE 层选择 switch 的状态，到 terminal 有约一半回到 continue。

## J2/J3. 判读

1. argmin 层面 16.4% 的 1000-vs-terminal 分歧确实存在、跨多数 function group 出现（19/42 有 switch→continue pattern），且 practical 口径下仍有 6–8% 超出各自 horizon 的 δ95——**不是纯粹的 argmin tie 伪影**。
2. 但其**可利用价值被上界封顶**：即使逐状态完美地按各 horizon 选择最优动作，相对 always-continue 的可得增益也只有 0.013（1000）/ 0.042（terminal）log10（07 报告），比噪声 δ95 低一个数量级。也就是说，分歧存在、但几乎不带可兑现的性能。
3. 与 Task 10 的关系（RQ5）：该 4.3% 的 pattern 规模太小、margin 低于噪声，**不足以支持"repeated DAS 的价值来自 segment-level complementarity"**这一解释在训练域成立。Task 10 在三个 CEC 开发函数上观察到的 repeated 收益（classifier +0.42 over first-trigger、dwell1000 再 +0.08）更可能是跨 suite 差异与频繁重评估的局部收益，不能外推为训练域上存在可学习的 segment action 结构。

## Verdict B（Action horizon）：**H2 MODERATE HORIZON DEPENDENCE**

- 不满足 H1（HORIZON-INVARIANT）：16% 的 raw 分歧与 6–8% 的 practical 分歧不算"很低"；
- 不满足 H3（STRONG）：H3 要求 1000 horizon 的可得增益明显高于 terminal 且 practical disagreement 显著——实测两个 horizon 的可得增益都接近零（0.013 vs 0.042），practical 分歧 6–8% 属中等；
- 结论：**horizon 标签的选择问题在本 portfolio 下是次要矛盾**——即便未来换成 $G_{1000}(s,a)$ 作为 repeated Selector 主标签，其可得价值仍受 04 报告的退化结论支配。当前不构成把 terminal 标签换成 segment 标签的性能理由（换标签的收益上界 0.013–0.042 log10，低于噪声）。
