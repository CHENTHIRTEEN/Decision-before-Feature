# 17a1_14 Final Verdict

Task17A.1 是零 FE 的统计正确性与估计对象复核。目的不是获得更漂亮的显著性，而是判断 Task17A 哪些机制结论在统一统计口径后仍然成立。

## 分类

**C1 ROBUST PARTIAL DECISION ALIGNMENT WITH SUBSTANTIAL RESIDUAL ALIASING**

## 异质性

| domain       | suite   |   n_pairs |   decision_distance_gt1_rate |   cv_group_balanced_gt1_rate |   cv_group_balanced_ci_low |   cv_group_balanced_ci_high |   n_cv_groups |   weighting_difference | heterogeneity_verdict   | bootstrap_unit   |   bootstrap_draws |   new_objective_fe |
|:-------------|:--------|----------:|-----------------------------:|-----------------------------:|---------------------------:|----------------------------:|--------------:|-----------------------:|:------------------------|:-----------------|------------------:|-------------------:|
| natural      | bbob    |      2700 |                     0.702222 |                     0.702222 |                   0.593704 |                    0.805556 |            10 |            1.11022e-16 | DH1 NONTRIVIAL          | cv_group_id      |              5000 |                  0 |
| natural      | mabbob  |      1080 |                     0.730556 |                     0.730556 |                   0.661111 |                    0.79537  |            12 |            1.11022e-16 | DH1 NONTRIVIAL          | cv_group_id      |              5000 |                  0 |
| natural      | pooled  |      3780 |                     0.710317 |                     0.717677 |                   0.65774  |                    0.776431 |            22 |            0.00735931  | DH1 NONTRIVIAL          | cv_group_id      |              5000 |                  0 |
| post_handoff | bbob    |      5400 |                     0.691667 |                     0.691667 |                   0.578699 |                    0.799074 |            10 |            0           | DH1 NONTRIVIAL          | cv_group_id      |              5000 |                  0 |
| post_handoff | mabbob  |      2160 |                     0.749537 |                     0.749537 |                   0.668056 |                    0.826389 |            12 |            1.11022e-16 | DH1 NONTRIVIAL          | cv_group_id      |              5000 |                  0 |
| post_handoff | pooled  |      7560 |                     0.708201 |                     0.723232 |                   0.654958 |                    0.789649 |            22 |            0.0150313   | DH1 NONTRIVIAL          | cv_group_id      |              5000 |                  0 |

## 科学表述

Observable search behavior shows statistically detectable within-context alignment with observed alternate-action decision geometry, while substantial action-value aliasing remains among locally nearest observable states.

## 执行边界

本轮不支持直接加入 solver internal state，不训练新 selector，不增加 seeds 6-10，不运行 CEC，不执行闭环。新增 objective FE = 0；下一步只能依据 Task17A.1 分类重新设计。

## 机器可读字段

```json
{
  "heterogeneity_verdict": "DH1 NONTRIVIAL",
  "final_verdict": "C1 ROBUST PARTIAL DECISION ALIGNMENT WITH SUBSTANTIAL RESIDUAL ALIASING",
  "shift_verdict": "DS2 NO ROBUST SHIFT",
  "ladder_verdict": "LR2 PARTIAL OR INCONSISTENT",
  "all_base_pooled_ca1": true,
  "no_suite_significantly_inverse": true,
  "substantial_post_local_aliasing": true,
  "observable_proximity_reduces_aliasing": true,
  "scientific_statement": "Observable search behavior shows statistically detectable within-context alignment with observed alternate-action decision geometry, while substantial action-value aliasing remains among locally nearest observable states.",
  "behavior_sufficient_for_dynamic_selection": false,
  "new_selector_allowed": false,
  "solver_internal_state_allowed": false,
  "seeds_6_10_allowed": false,
  "cec_allowed": false,
  "next_step_requires_task17a1_based_redesign": true,
  "new_objective_fe": 0
}
```
