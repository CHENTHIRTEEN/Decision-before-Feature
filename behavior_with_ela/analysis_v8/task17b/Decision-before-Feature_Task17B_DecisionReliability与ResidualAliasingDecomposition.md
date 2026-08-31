Task17B 是零新增目标函数评估的机制分解。所有正式区间均按 `cv_group_id` 分组 bootstrap 5000 次；`new_objective_FE = 0`。

# Decision-before-Feature Task17B：Decision Reliability 与 Residual Aliasing Decomposition

## 结论

**D2 NOISE-LIMITED DECISION GEOMETRY**。

## Repetition topology

| component_type    | action_or_pair   |   t1_cells |   t2_states |   t3_states |   cv_groups | outcome_blind_selection   | formal_support   |
|:------------------|:-----------------|-----------:|------------:|------------:|------------:|:--------------------------|:-----------------|
| T1_action         | shade            |        394 |           0 |           6 |          22 | YES                       | True             |
| T1_action         | lshade           |        396 |           0 |           6 |          22 | YES                       | True             |
| T1_action         | cso              |        352 |           0 |           6 |          22 | YES                       | True             |
| T2_pair           | shade|lshade     |        790 |          40 |           6 |          14 | YES                       | True             |
| T2_pair           | shade|cso        |        746 |          46 |           6 |          12 | YES                       | True             |
| T2_pair           | lshade|cso       |        748 |          40 |           6 |          13 | YES                       | True             |
| T3_full_signature | shade|lshade|cso |       1142 |           0 |           6 |           3 | YES                       | False            |

- Source state：相同 population、fitness、best、FE；continue 为全状态 clone；transfer 由同一 source state 初始化。
- RNG：仅 semantic RNG coordinates 改变；stream separation 由代码确认，不作有限样本独立性的过度主张。
- Natural：T1 可作描述；三个 pooled T2 均低于 30 states，因此不承担 secondary mechanism verdict。
- new objective FE：0。

## Action noise

| action   |   P_N_gt1 |   median |    q90 |     q95 |   repeated_states |   cv_groups |
|:---------|----------:|---------:|-------:|--------:|------------------:|------------:|
| cso      |    0.3617 |   0.3441 | 6.3879 | 11.1504 |               352 |          22 |
| lshade   |    0.3577 |   0.4203 | 3.8305 |  5.6442 |               396 |          22 |
| shade    |    0.3841 |   0.5727 | 4.2440 |  6.2579 |               394 |          22 |

按 `P(N_a>1)`，随机波动最大的是 shade（0.3841）。suite 明细保存在 parquet 与分报告中。

| suite   | action   |   P_N_gt1 |   median |    q90 |   repeated_states |   cv_groups | repetition_weight_sensitive   |
|:--------|:---------|----------:|---------:|-------:|------------------:|------------:|:------------------------------|
| bbob    | cso      |    0.3425 |   0.2319 | 6.7262 |               254 |          10 | False                         |
| mabbob  | cso      |    0.4116 |   0.5665 | 5.9163 |                98 |          12 | True                          |
| bbob    | lshade   |    0.3486 |   0.3846 | 4.0718 |               284 |          10 | False                         |
| mabbob  | lshade   |    0.3810 |   0.5188 | 2.9856 |               112 |          12 | True                          |
| bbob    | shade    |    0.3858 |   0.5505 | 4.4118 |               292 |          10 | False                         |
| mabbob  | shade    |    0.3791 |   0.6269 | 3.5130 |               102 |          12 | True                          |

BBOB 中 SHADE 的 `P(N_a>1)` 最大，MA-BBOB 中 CSO 最大；action-level 排序并非跨 suite 完全稳定。all-pairs 与每 state 固定 key 0/1 的最大-action 结论在 pooled 层一致，未标记 repetition-weight sensitive。

## Pairwise reliability

| action_pair   |   P_D_noise_gt1 |   replicate_margin_spearman |   exact_category_agreement |   directional_reversal_rate |   tie_winner_transition_rate | reliability_class   | third_layer_reliability_class   | repetition_weight_sensitive   |
|:--------------|----------------:|----------------------------:|---------------------------:|----------------------------:|-----------------------------:|:--------------------|:--------------------------------|:------------------------------|
| lshade|cso    |          0.4750 |                      0.8520 |                     0.7750 |                      0.0000 |                       0.2250 | NR3 LOW RELIABILITY | NR3 LOW RELIABILITY             | False                         |
| shade|cso     |          0.5435 |                      0.8147 |                     0.8696 |                      0.0217 |                       0.1087 | NR3 LOW RELIABILITY | NR3 LOW RELIABILITY             | False                         |
| shade|lshade  |          0.4500 |                      0.5115 |                     0.7000 |                      0.0500 |                       0.2500 | NR3 LOW RELIABILITY | NR3 LOW RELIABILITY             | False                         |

| suite   | action_pair   |   states |   cv_groups | formal_support   |   P_D_noise_gt1 |   replicate_margin_spearman | reliability_class   |
|:--------|:--------------|---------:|------------:|:-----------------|----------------:|----------------------------:|:--------------------|
| bbob    | lshade|cso    |       32 |           9 | True             |          0.5000 |                      0.8827 | NR3 LOW RELIABILITY |
| mabbob  | lshade|cso    |        8 |           4 | False            |          0.3750 |                      0.7857 | NR3 LOW RELIABILITY |
| bbob    | shade|cso     |       38 |           9 | True             |          0.5789 |                      0.7602 | NR3 LOW RELIABILITY |
| mabbob  | shade|cso     |        8 |           3 | False            |          0.3750 |                      0.9048 | NR3 LOW RELIABILITY |
| bbob    | shade|lshade  |       24 |           9 | True             |          0.3333 |                      0.5913 | NR3 LOW RELIABILITY |
| mabbob  | shade|lshade  |       16 |           5 | True             |          0.6250 |                      0.2500 | NR3 LOW RELIABILITY |

## Residual beyond noise

| action_pair   |   A_noise |   A_local |   A_all |   E_res |   E_res_ci_low |   E_res_ci_high |   M_res |   F_capture |   F_residual |
|:--------------|----------:|----------:|--------:|--------:|---------------:|----------------:|--------:|------------:|-------------:|
| shade|lshade  |    0.4500 |    0.3500 |  0.5375 | -0.1000 |        -0.1905 |          0.0000 | -0.0521 |      2.1429 |      -1.1429 |
| shade|cso     |    0.5435 |    0.6818 |  0.6841 |  0.1383 |        -0.0500 |          0.3158 |  1.1930 |      0.0162 |       0.9838 |
| lshade|cso    |    0.4750 |    0.5500 |  0.5850 |  0.0750 |        -0.1087 |          0.2812 |  0.3444 |      0.3182 |       0.6818 |

| suite   | action_pair   | formal_support   |   A_noise |   A_local |   A_all |   E_res |   E_res_ci_low |   E_res_ci_high |   M_res |
|:--------|:--------------|:-----------------|----------:|----------:|--------:|--------:|---------------:|----------------:|--------:|
| bbob    | shade|lshade  | True             |    0.3333 |    0.2500 |  0.5167 | -0.0833 |        -0.2143 |          0.0909 | -0.1309 |
| mabbob  | shade|lshade  | True             |    0.6250 |    0.5000 |  0.5687 | -0.1250 |        -0.2222 |          0.0000 | -1.1797 |
| bbob    | shade|cso     | True             |    0.5789 |    0.6111 |  0.6500 |  0.0322 |        -0.1714 |          0.1548 |  0.7802 |
| mabbob  | shade|cso     | False            |    0.3750 |    1.0000 |  0.8375 |  0.6250 |         0.5000 |          0.7500 |  2.1565 |
| bbob    | lshade|cso    | True             |    0.5000 |    0.4688 |  0.5687 | -0.0312 |        -0.2308 |          0.1667 | -0.2809 |
| mabbob  | lshade|cso    | False            |    0.3750 |    0.8750 |  0.6500 |  0.5000 |         0.5000 |          0.5000 |  1.2621 |

至少 2/3 pairs 的 beyond-noise residual：NO。BBOB 与 MA-BBOB 逐对结果见 17b_10；低支持 MA cells 不承担正式结论。

## Full signature

T3=6 states / 3 groups，未达到 50 / 8，故 AVA_noise、AVA_local,T3、E_res_full 与 R_D 均不计算。

## High-confidence states

U0=0.5471，U1=0.4529，U2=0.2243，U3=0.1902。

| subset   | action_pair   |   A_local |   median_D_local |   winner_mismatch_rate |   strata_with_pairs |
|:---------|:--------------|----------:|-----------------:|-----------------------:|--------------------:|
| U1       | shade|lshade  |    0.7197 |           2.4203 |                 0.3418 |            471.0000 |
| U1       | shade|cso     |    0.7643 |           2.8569 |                 0.3418 |            471.0000 |
| U1       | lshade|cso    |    0.7367 |           2.3083 |                 0.3418 |            471.0000 |
| U3       | shade|lshade  |    0.7035 |           1.9653 |                 0.2261 |            199.0000 |
| U3       | shade|cso     |    0.7839 |           2.6968 |                 0.2261 |            199.0000 |
| U3       | lshade|cso    |    0.7085 |           2.2517 |                 0.2261 |            199.0000 |

U1-repeat 因 T3 不足不可构造；U1/U3 local aliasing 未作为正式 ground truth。U1 与 U3 的三对 local rates 均未下降到 same-state floor，现有结果不支持 apparent aliasing 主要由 practical near-tie states 驱动。

## Stop-loss

- 最终：D2 NOISE-LIMITED DECISION GEOMETRY
- Task17C：NO
- new feature：NO
- new selector：NO
- fuzzy：NO
- seeds 6-10：NO
- CEC：NO
- 追加 repetitions：NO
- paper consolidation：EVALUATE_ONLY
- 主线 STOP：YES

## 图表

- `figures/17b_figure_a_same_local_all.png`
- `figures/17b_figure_b_margin_repeatability.png`
- `figures/17b_figure_c_capture_residual.png`
