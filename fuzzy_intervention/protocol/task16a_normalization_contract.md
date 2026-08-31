# Task16A Probe Normalization Contract

Task16A 只进行结构分析，不生成部署规则。

对 P/H/S 分别在 development natural source states 中按 current algorithm 计算经验分布秩，映射到 [0, 1]。不按 problem、function 或 suite 单独归一化，不读取任何动作结果。

对每个算法和 probe，稳定排序采用原始值、problem_id、seed、source_FE、state_id；相同原始值使用平均秩，以保留 ties。归一化列分别为 `probe_productivity_rank`、`probe_entropy_rank`、`probe_stagnation_rank`。

固定 tertile：LOW 为 rank ≤ 1/3，MED 为 1/3 < rank < 2/3，HIGH 为 rank ≥ 2/3。边界不根据结果调整，也不解释为未来 membership function。

预先指定 regime：

- R1：P=HIGH 且 S=LOW；
- R2：P=LOW 且 S=HIGH；
- R3：R2 且 H=LOW；
- R4：R2 且 H=HIGH。

Maturity 分层：Early/Mid 为 M ∈ {0.2, 0.4}，Late 为 M ∈ {0.6, 0.8}。

