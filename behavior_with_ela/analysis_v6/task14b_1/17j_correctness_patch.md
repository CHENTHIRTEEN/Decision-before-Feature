# 17j · Task 14B 正确性修正

本轮未改变 Task 14A action outcomes、portfolio、horizon、commitment、模型或数据划分；新增 action-label FE = 0。W0 改为 held-out seed 之外的 4-seed train-only 选择；Global/Segment permutation 分别只置换对应 block；A_ND dominance 改为由较低 loss 动作标记另一动作为 dominated；absolute loss 直接按 row-level realized values 计算。
