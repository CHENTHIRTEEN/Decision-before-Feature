Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# 17b_11 Noise-ceiling interpretation

这里的 repetition geometry 只称 empirical noise-ceiling reference，不称严格的 information-theoretic upper bound。同状态重复同时覆盖后续求解随机性，以及 CSO transfer 时由 action RNG 生成的初始 velocity。

| action_pair   |   A_noise |   A_local |   noise_to_local_ratio |   E_res |   E_res_ci_low |   E_res_ci_high |
|:--------------|----------:|----------:|-----------------------:|--------:|---------------:|----------------:|
| shade|lshade  |    0.4500 |    0.3500 |                 1.2857 | -0.1000 |        -0.1905 |          0.0000 |
| shade|cso     |    0.5435 |    0.6818 |                 0.7971 |  0.1383 |        -0.0500 |          0.3158 |
| lshade|cso    |    0.4750 |    0.5500 |                 0.8636 |  0.0750 |        -0.1087 |          0.2812 |