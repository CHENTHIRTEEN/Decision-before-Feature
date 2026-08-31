# 数值稳定性检查

阈值：near-constant 的众数比例 ≥0.995；极端尾部比例定义为 |value|>1e6。

| feature_name                                 |   missing_rate |   nonfinite_rate |   near_constant_rate |   tail_instability |   max_abs_finite | screened_core   | near_constant_flag   | numeric_instability_flag   |
|:---------------------------------------------|---------------:|-----------------:|---------------------:|-------------------:|-----------------:|:----------------|:---------------------|:---------------------------|
| bf_fe_ratio                                  |              0 |                0 |           0.111111   |          0         |      0.7         | False           | False                | False                      |
| bf_improvement_rate_w02                      |              0 |                0 |           0.332804   |          0         |     28.8702      | False           | False                | False                      |
| bf_improvement_frequency_w02                 |              0 |                0 |           0.369577   |          0         |      1           | False           | False                | False                      |
| bf_diversity_mean_pairwise                   |              0 |                0 |           0.00026455 |          0         |      1.25061     | False           | False                | False                      |
| bf_diversity_change_w05                      |              0 |                0 |           0.0272487  |          0         |     85.2313      | False           | False                | False                      |
| bf_covariance_spectral_concentration         |              0 |                0 |           0.0206349  |          0         |      0.999977    | False           | False                | False                      |
| bf_distance_decay_w10                        |              0 |                0 |           0.0166667  |          0         |   1323.97        | False           | False                | False                      |
| bf_stagnation_w10                            |              0 |                0 |           0.279365   |          0         |      1           | True            | False                | False                      |
| bf_convergence_rate_w10                      |              0 |                0 |           0.0166667  |          0         |  14229.2         | False           | False                | False                      |
| bf_fitness_diversity_rel                     |              0 |                0 |           0.0505291  |          0         |      5.33682     | True            | False                | False                      |
| bf_population_wasserstein_rate_w05           |              0 |                0 |           0.0465608  |          0         |     17.0637      | False           | False                | False                      |
| bf_centroid_shift_rate_w05                   |              0 |                0 |           0.0272487  |          0         |     16.4805      | False           | False                | False                      |
| bf_centroid_shift_coherence_w05              |              0 |                0 |           0.0465608  |          0         |      1           | True            | False                | False                      |
| bf_fitness_quantile_improvement_fraction_w02 |              0 |                0 |           0.598413   |          0         |      1           | False           | False                | False                      |
| bf_fitness_distribution_improvement_rate_w02 |              0 |                0 |           0.0486772  |          0         |    205.208       | True            | False                | False                      |
| bf_fitness_wasserstein_rate_w02              |              0 |                0 |           0.0486772  |          0         |    205.208       | False           | False                | False                      |
| bf_elite_concentration                       |              0 |                0 |           0.00026455 |          0         |      1.75169     | True            | False                | False                      |
| bf_best_fitness_slope_rel_w05                |              0 |                0 |           0.124339   |          0         |     32.3601      | False           | False                | False                      |
| bf_diversity_slope_w05                       |              0 |                0 |           0.0103175  |          0         |     15.183       | False           | False                | False                      |
| bf_fitness_spread_slope_w05                  |              0 |                0 |           0.0460317  |          0         |    307.54        | False           | False                | False                      |
| bf_population_centroid_shift_w05             |              0 |                0 |           0.0272487  |          0         |      0.830616    | True            | False                | False                      |
| bf_elite_centroid_shift_w05                  |              0 |                0 |           0.034127   |          0         |      0.759869    | False           | False                | False                      |
| bf_covariance_trace_ratio_w05                |              0 |                0 |           0.0198413  |          0         |   8554.18        | False           | False                | False                      |
| bf_covariance_effective_rank_w05             |              0 |                0 |           0.0386243  |          0         |      8.9709      | False           | False                | False                      |
| bf_diversity_recovery_w05                    |              0 |                0 |           0.807937   |          0         |     85.2313      | False           | False                | False                      |
| bf_population_chamfer_distance_w05           |              0 |                0 |           0.0272487  |          0         |      0.640462    | False           | False                | False                      |
| bf_covariance_trace_change_w05               |              0 |                0 |           0.0272487  |          0         |   8553.18        | False           | False                | False                      |
| bf_covariance_effective_rank_change_w05      |              0 |                0 |           0.0507937  |          0.0010582 |      4.94655e+12 | False           | False                | True                       |
