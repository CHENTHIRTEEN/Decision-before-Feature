# 17a1_01 Zero-FE and Statistical Contract

Task17A.1 是零 FE 的统计正确性与估计对象复核。目的不是获得更漂亮的显著性，而是判断 Task17A 哪些机制结论在统一统计口径后仍然成立。

## 执行范围

只读取 Task17A 已生成的 states、pairs、representation distances、decision distances、NN 与随机 pair 对照；未导入或调用 objective、optimizer、ELA、特征提取器或学习模型。

## 随机与统计参数

|   master_seed |   bootstrap_draws |   permutations |   random_neighbor_controls | bootstrap_unit   |   new_objective_fe |
|--------------:|------------------:|---------------:|---------------------------:|:-----------------|-------------------:|
|    2026083101 |              5000 |            100 |                        100 | cv_group_id      |                  0 |

## 一致性检查

| check_id                              | passed   | detail                                                           |   new_objective_fe |
|:--------------------------------------|:---------|:-----------------------------------------------------------------|-------------------:|
| C01_shared_alignment_statistic        | True     | point, grouped bootstrap, and permutation call one core function |                  0 |
| C02_cv_group_block_resampling         | True     | resampling unit is cv_group_id without problem_id subdivision    |                  0 |
| C03_one_local_ava10_pair_per_stratum  | True     | every domain-representation-stratum contributes one pair         |                  0 |
| C04_two_local_ava20_pairs_per_stratum | True     | every domain-representation-stratum contributes two pairs        |                  0 |
| C05_no_cross_domain_pairs             | True     | both state endpoints match the pair domain                       |                  0 |
| C06_zero_new_objective_fe             | True     | all result tables report zero newly evaluated objective FE       |                  0 |
| C07_state_and_pair_counts             | True     | 1890 plus 3780 states and 3780 plus 7560 pairs                   |                  0 |
| C08_draw_counts                       | True     | all requested grouped draws and permutations are present         |                  0 |
