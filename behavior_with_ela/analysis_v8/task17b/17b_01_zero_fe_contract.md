Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_01 零 FE 与统计契约

- Primary domain：post-handoff。
- Primary representation：Global28；Compact6 仅作敏感性。
- 两个 replicate layer 固定按 key 0、1；key 2 仅作敏感性。
- `E_res = A_local - A_noise`；`M_res = median(D_local) - median(D_noise)`。
- D3 的操作判据在查看机制结果前固定为：至少两对 U1 rate 比总体 local rate 下降至少 0.10，且与 noise rate 的绝对差不超过 0.10。
- 粘贴任务中缺失公式左端的定义，按相邻文字还原为归一化绝对差；未引入额外统计量。
