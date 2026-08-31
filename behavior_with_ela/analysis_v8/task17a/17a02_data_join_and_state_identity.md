# 17a02_data_join_and_state_identity

## 精确拼接

| domain       |   source_rows |   joined_rows |   unique_states | exact_one_to_one   |   new_objective_fe | natural_issd_status         |
|:-------------|--------------:|--------------:|----------------:|:-------------------|-------------------:|:----------------------------|
| natural      |          1890 |          1890 |            1890 | True               |                  0 | skipped_no_zero_fe_artifact |
| post_handoff |          3780 |          3780 |            3780 | True               |                  0 | not_applicable              |

## 状态键

自然域 stratum=(problem_id,current_algorithm,source_FE)；交接后域 stratum=(problem_id,route,source_FE)。每个 stratum 均保留 seeds 1–5，因而各自产生 10 个 state pairs。
