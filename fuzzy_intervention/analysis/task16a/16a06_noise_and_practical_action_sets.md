# 16a06 Noise 与实际非支配集合

每个被选 state-action 有 3 次独立运行；主阈值为两动作 95% noise scale 的较大值。主阈值下 E|A_ND|=4.0720；平方和开方阈值下 E|A_ND|=4.3245。

| action           |   repeated_state_action_pairs |   absolute_repetition_differences |   delta_action_95 |   quantile |
|:-----------------|------------------------------:|----------------------------------:|------------------:|-----------:|
| continue         |                           247 |                               494 |          0.561663 |       0.95 |
| perturb_random   |                           255 |                               510 |          0.569991 |       0.95 |
| perturb_targeted |                           210 |                               420 |          0.50722  |       0.95 |
| switch_cso       |                           179 |                               358 |          0.509594 |       0.95 |
| switch_lshade    |                           184 |                               368 |          0.502872 |       0.95 |
| switch_shade     |                           159 |                               318 |          0.345392 |       0.95 |
