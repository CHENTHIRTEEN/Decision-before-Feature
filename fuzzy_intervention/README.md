# Fuzzy Intervention

本目录承载 Continue–Perturb–Switch 动作空间的独立探索性实验。它不修改主 Decision-before-Feature 的算法池、Decision 标签、Selector 特征或训练程序，也不向主结果目录写入数据。

Task16A 只评价三类干预动作是否具有非退化的实际性能互补性，以及预先指定的 Productivity、Entropy、Stagnation 与预算阶段是否呈现粗粒度结构。本阶段不训练分类器或控制器，不实现模糊规则，不使用 ELA，不进入验证集、CEC、工程问题或 seeds 6–10。

目录约定：

- `protocol/`：先于动作结果确定的实验定义；
- `probes/`：P/H/S 的合法在线计算与无结果归一化；
- `interventions/`：固定轻量扰动与状态一致性修正；
- `experiments/task16a_action_space/`：自然状态生成与五动作分支；
- `analysis/task16a/`：分组统计、区间估计与结论生成；
- `results/task16a/`：Task16A 独立重结果。

