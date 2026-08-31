Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_05 Pairwise-margin reliability

| action_pair   |   states |   cv_groups |   median_D_noise |   P_D_noise_gt1 |   replicate_margin_spearman |   exact_category_agreement |   directional_reversal_rate |   tie_winner_transition_rate | reliability_class   | third_layer_reliability_class   | repetition_weight_sensitive   |
|:--------------|---------:|------------:|-----------------:|----------------:|----------------------------:|---------------------------:|----------------------------:|-----------------------------:|:--------------------|:--------------------------------|:------------------------------|
| lshade|cso    |       40 |          13 |           0.9464 |          0.4750 |                      0.8520 |                     0.7750 |                      0.0000 |                       0.2250 | NR3 LOW RELIABILITY | NR3 LOW RELIABILITY             | False                         |
| shade|cso     |       46 |          12 |           1.2658 |          0.5435 |                      0.8147 |                     0.8696 |                      0.0217 |                       0.1087 | NR3 LOW RELIABILITY | NR3 LOW RELIABILITY             | False                         |
| shade|lshade  |       40 |          14 |           0.7417 |          0.4500 |                      0.5115 |                     0.7000 |                      0.0500 |                       0.2500 | NR3 LOW RELIABILITY | NR3 LOW RELIABILITY             | False                         |