# 17a1_03 Data Identity and Group Semantics

Task17A.1 是零 FE 的统计正确性与估计对象复核。目的不是获得更漂亮的显著性，而是判断 Task17A 哪些机制结论在统一统计口径后仍然成立。

## 身份结果

| domain       |   states |   strata |   pairs |   states_per_stratum |   pairs_per_stratum |   new_objective_fe |
|:-------------|---------:|---------:|--------:|---------------------:|--------------------:|-------------------:|
| natural      |     1890 |      378 |    3780 |                    5 |                  10 |                  0 |
| post_handoff |     3780 |      756 |    7560 |                    5 |                  10 |                  0 |

## cv_group_id 语义

| suite   |   n_cv_groups |   min_problems_per_group |   max_problems_per_group |
|:--------|--------------:|-------------------------:|-------------------------:|
| bbob    |            10 |                        3 |                        3 |
| mabbob  |            12 |                        1 |                        1 |

## 解释

BBOB 的 cv_group_id 是函数编号，同一函数的三个实例属于同一 grouped dependence 单元；MA-BBOB 的 cv_group_id 是候选函数编号，当前每组一个实例。既有 Task12-15 grouped-OOF 也直接按 cv_group_id 留组。
