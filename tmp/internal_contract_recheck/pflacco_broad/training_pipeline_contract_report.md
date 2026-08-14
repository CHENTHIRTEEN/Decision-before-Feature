# Decision Model training pipeline contract check

## Scope

- This check loads the materialized Decision dataset and validates preprocessing contract only.
- No Decision Model estimator was trained.
- X columns: `bf_fe_ratio, bf_improvement_rate_w02, bf_improvement_frequency_w02, bf_diversity_mean_pairwise, bf_diversity_change_w05, bf_covariance_spectral_concentration, bf_distance_decay_w10, bf_stagnation_w10, bf_convergence_rate_w10, bf_fitness_diversity_rel, bf_population_wasserstein_rate_w05, bf_centroid_shift_rate_w05, bf_centroid_shift_coherence_w05, bf_fitness_quantile_improvement_fraction_w02, bf_fitness_distribution_improvement_rate_w02, bf_fitness_wasserstein_rate_w02, bf_elite_concentration, bf_best_fitness_slope_w05, bf_diversity_slope_w05, bf_fitness_spread_slope_w05, bf_population_centroid_shift_w05, bf_elite_centroid_shift_w05, bf_covariance_trace_ratio_w05, bf_covariance_effective_rank_w05, bf_diversity_recovery_w05, bf_population_chamfer_distance_w05, bf_covariance_trace_change_w05, bf_covariance_effective_rank_change_w05, bf_search_maturity, bf_search_maturity_linear, bf_explore_exploit_ratio`.
- Target column: `u_query_lamT_1`.

## Split summary

| split | rows | target_null_count | target_finite | u_gt_zero_rate |
| --- | --- | --- | --- | --- |
| bbob_train | 70033 | 0 | True | 0 |
| bbob_validation | 24601 | 0 | True | 0 |

## X legality

| check | passed | detail |
| --- | --- | --- |
| x_columns_equal_behavior_feature_columns | True | bf_fe_ratio,bf_improvement_rate_w02,bf_improvement_frequency_w02,bf_diversity_mean_pairwise,bf_diversity_change_w05,bf_covariance_spectral_concentration,bf_distance_decay_w10,bf_stagnation_w10,bf_convergence_rate_w10,bf_fitness_diversity_rel,bf_population_wasserstein_rate_w05,bf_centroid_shift_rate_w05,bf_centroid_shift_coherence_w05,bf_fitness_quantile_improvement_fraction_w02,bf_fitness_distribution_improvement_rate_w02,bf_fitness_wasserstein_rate_w02,bf_elite_concentration,bf_best_fitness_slope_w05,bf_diversity_slope_w05,bf_fitness_spread_slope_w05,bf_population_centroid_shift_w05,bf_elite_centroid_shift_w05,bf_covariance_trace_ratio_w05,bf_covariance_effective_rank_w05,bf_diversity_recovery_w05,bf_population_chamfer_distance_w05,bf_covariance_trace_change_w05,bf_covariance_effective_rank_change_w05,bf_search_maturity,bf_search_maturity_linear,bf_explore_exploit_ratio |
| metadata_columns_absent_from_x | True |  |
| target_columns_absent_from_x | True |  |
| forbidden_exact_columns_absent_from_x | True |  |
| forbidden_name_fragments_absent_from_x | True |  |

## Imputer fit contract

- Imputer: `SimpleImputer(strategy='median')`.
- Fit split: `bbob_train`.
- Fit rows: `70033`.
- Validation rows used for fit: `0`.

| feature | fit_split | fit_rows | validation_rows_used_for_fit | strategy | median_matches_train_split | train_null_count | validation_null_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf_fe_ratio | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_improvement_rate_w02 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_improvement_frequency_w02 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_diversity_mean_pairwise | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_diversity_change_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_covariance_spectral_concentration | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_distance_decay_w10 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_stagnation_w10 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_convergence_rate_w10 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_fitness_diversity_rel | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_population_wasserstein_rate_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_centroid_shift_rate_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_centroid_shift_coherence_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_fitness_quantile_improvement_fraction_w02 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_fitness_distribution_improvement_rate_w02 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_fitness_wasserstein_rate_w02 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_elite_concentration | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_best_fitness_slope_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_diversity_slope_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_fitness_spread_slope_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_population_centroid_shift_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_elite_centroid_shift_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_covariance_trace_ratio_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_covariance_effective_rank_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_diversity_recovery_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_population_chamfer_distance_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_covariance_trace_change_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_covariance_effective_rank_change_w05 | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_search_maturity | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_search_maturity_linear | bbob_train | 70033 | 0 | median | True | 0 | 0 |
| bf_explore_exploit_ratio | bbob_train | 70033 | 0 | median | True | 0 | 0 |

## Scaler fit contract

- Scaler: `StandardScaler()`.
- Fit split: `bbob_train`.
- Fit rows: `70033`.
- Validation rows used for fit: `0`.

| feature | fit_split | fit_rows | validation_rows_used_for_fit | mean_matches_train_split | var_matches_train_split | scaler_scale |
| --- | --- | --- | --- | --- | --- | --- |
| bf_fe_ratio | bbob_train | 70033 | 0 | True | True | 0.120799 |
| bf_improvement_rate_w02 | bbob_train | 70033 | 0 | True | True | 1.02053 |
| bf_improvement_frequency_w02 | bbob_train | 70033 | 0 | True | True | 0.262133 |
| bf_diversity_mean_pairwise | bbob_train | 70033 | 0 | True | True | 0.13076 |
| bf_diversity_change_w05 | bbob_train | 70033 | 0 | True | True | 1.18471 |
| bf_covariance_spectral_concentration | bbob_train | 70033 | 0 | True | True | 0.17424 |
| bf_distance_decay_w10 | bbob_train | 70033 | 0 | True | True | 11.8629 |
| bf_stagnation_w10 | bbob_train | 70033 | 0 | True | True | 0.352811 |
| bf_convergence_rate_w10 | bbob_train | 70033 | 0 | True | True | 96.0808 |
| bf_fitness_diversity_rel | bbob_train | 70033 | 0 | True | True | 0.261882 |
| bf_population_wasserstein_rate_w05 | bbob_train | 70033 | 0 | True | True | 3.53452 |
| bf_centroid_shift_rate_w05 | bbob_train | 70033 | 0 | True | True | 3.02891 |
| bf_centroid_shift_coherence_w05 | bbob_train | 70033 | 0 | True | True | 0.217746 |
| bf_fitness_quantile_improvement_fraction_w02 | bbob_train | 70033 | 0 | True | True | 0.347862 |
| bf_fitness_distribution_improvement_rate_w02 | bbob_train | 70033 | 0 | True | True | 3.33744 |
| bf_fitness_wasserstein_rate_w02 | bbob_train | 70033 | 0 | True | True | 3.66749 |
| bf_elite_concentration | bbob_train | 70033 | 0 | True | True | 0.121466 |
| bf_best_fitness_slope_w05 | bbob_train | 70033 | 0 | True | True | 41780.9 |
| bf_diversity_slope_w05 | bbob_train | 70033 | 0 | True | True | 1.1466 |
| bf_fitness_spread_slope_w05 | bbob_train | 70033 | 0 | True | True | 1.20254 |
| bf_population_centroid_shift_w05 | bbob_train | 70033 | 0 | True | True | 0.152381 |
| bf_elite_centroid_shift_w05 | bbob_train | 70033 | 0 | True | True | 0.15506 |
| bf_covariance_trace_ratio_w05 | bbob_train | 70033 | 0 | True | True | 23.0778 |
| bf_covariance_effective_rank_w05 | bbob_train | 70033 | 0 | True | True | 7.17614 |
| bf_diversity_recovery_w05 | bbob_train | 70033 | 0 | True | True | 1.07368 |
| bf_population_chamfer_distance_w05 | bbob_train | 70033 | 0 | True | True | 0.150019 |
| bf_covariance_trace_change_w05 | bbob_train | 70033 | 0 | True | True | 23.0756 |
| bf_covariance_effective_rank_change_w05 | bbob_train | 70033 | 0 | True | True | 1.14073e+11 |
| bf_search_maturity | bbob_train | 70033 | 0 | True | True | 0.0635656 |
| bf_search_maturity_linear | bbob_train | 70033 | 0 | True | True | 0.0702562 |
| bf_explore_exploit_ratio | bbob_train | 70033 | 0 | True | True | 0.394649 |

## Transform checks

| stage | rows | finite_values | remaining_nan_count |
| --- | --- | --- | --- |
| imputer_transform_train | 70033 | True | 0 |
| imputer_transform_validation | 24601 | True | 0 |
| scaler_transform_train | 70033 | True | 0 |
| scaler_transform_validation | 24601 | True | 0 |
