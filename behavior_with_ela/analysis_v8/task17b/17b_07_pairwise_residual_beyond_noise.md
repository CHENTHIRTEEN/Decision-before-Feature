Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_07 Pairwise residual beyond noise

| action_pair   |   n_noise_states |   n_local_pairs |   n_all_pairs |   A_noise |   A_local |   A_all |   E_res |   E_res_ci_low |   E_res_ci_high |   M_res |   M_res_ci_low |   M_res_ci_high |   noise_to_local_ratio |   F_capture |   F_residual |
|:--------------|-----------------:|----------------:|--------------:|----------:|----------:|--------:|--------:|---------------:|----------------:|--------:|---------------:|----------------:|-----------------------:|------------:|-------------:|
| shade|lshade  |               40 |              40 |           400 |    0.4500 |    0.3500 |  0.5375 | -0.1000 |        -0.1905 |          0.0000 | -0.0521 |        -1.6881 |          0.2647 |                 1.2857 |      2.1429 |      -1.1429 |
| shade|cso     |               46 |              44 |           440 |    0.5435 |    0.6818 |  0.6841 |  0.1383 |        -0.0500 |          0.3158 |  1.1930 |        -0.2580 |          1.9448 |                 0.7971 |      0.0162 |       0.9838 |
| lshade|cso    |               40 |              40 |           400 |    0.4750 |    0.5500 |  0.5850 |  0.0750 |        -0.1087 |          0.2812 |  0.3444 |        -0.9151 |          1.3581 |                 0.8636 |      0.3182 |       0.6818 |

Primary `matched_support` 将 cross-state rows 限定到包含对应 T2 repetition state 的 exact decision strata；`all_available_support` 仅作敏感性。
