# Classifier Feature Engineering and Tuning Report

## Scope

- Dataset: existing phase1 refined sampling materialized BBOB Decision dataset.
- Feature source: `primary_with_maturity` with 23 behavior-only columns.
- No validation rows were used for feature selection, model fitting, or threshold fitting.
- Interaction LinearSVC candidates are recorded as skipped because the solver was not tractable on the selected interaction matrix in repeated runs.

## Input Contract

| check | passed | detail |
| --- | --- | --- |
| source_columns_equal_primary_with_maturity | True | bf_fe_ratio,bf_improvement_rate_w02,bf_improvement_frequency_w02,bf_diversity_mean_pairwise,bf_diversity_change_w05,bf_directional_entropy_w05,bf_distance_decay_w10,bf_stagnation_w10,bf_convergence_rate_w10,bf_fitness_diversity,bf_fitness_diversity_rel,bf_movement_magnitude,bf_movement_diversity,bf_direction_consistency_w05,bf_success_rate_w02,bf_improvement_variance_w02,bf_best_improvement_ratio_w02,bf_elite_concentration,bf_best_fitness_slope_w05,bf_diversity_slope_w05,bf_search_maturity,bf_search_maturity_linear,bf_explore_exploit_ratio |
| source_columns_exclude_metadata | True |  |
| source_columns_subset_of_behavior_feature_columns | True | bf_fe_ratio,bf_improvement_rate_w02,bf_improvement_frequency_w02,bf_diversity_mean_pairwise,bf_diversity_change_w05,bf_directional_entropy_w05,bf_distance_decay_w10,bf_stagnation_w10,bf_convergence_rate_w10,bf_fitness_diversity,bf_fitness_diversity_rel,bf_movement_magnitude,bf_movement_diversity,bf_direction_consistency_w05,bf_success_rate_w02,bf_improvement_variance_w02,bf_best_improvement_ratio_w02,bf_elite_concentration,bf_best_fitness_slope_w05,bf_diversity_slope_w05,bf_search_maturity,bf_search_maturity_linear,bf_explore_exploit_ratio |
| validation_rows_used_for_feature_selection_model_or_threshold_fit | True | 0 |

## Baseline Validation

| base_model_name | role | feature_scheme | decision_ela_call_rate | decision_mean_utility | utility_capture_rate | precision_u_gt_zero_under_calls | unhelpful_call_cost_sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lda_classifier | baseline | raw_primary_with_maturity | 0.0378395 | 0.00432497 | 0.561033 | 0.312806 | 136.611 |
| softmax_logistic_classifier | baseline | raw_primary_with_maturity | 0.0372531 | 0.00408656 | 0.54352 | 0.306545 | 139.047 |
| linear_svm_classifier | baseline | raw_primary_with_maturity | 0.0359722 | 0.00398438 | 0.529237 | 0.306735 | 135.055 |

## Top Validation Candidates

| base_model_name | feature_scheme | params_json | decision_mean_utility | utility_capture_rate | precision_u_gt_zero_under_calls | unhelpful_call_cost_sum |
| --- | --- | --- | --- | --- | --- | --- |
| lda_classifier | raw_primary_with_maturity | {"shrinkage": 0.5, "solver": "lsqr"} | 0.0043582 | 0.56099 | 0.315074 | 134.426 |
| lda_classifier | raw_primary_with_maturity | {"shrinkage": "auto", "solver": "lsqr"} | 0.00435564 | 0.56099 | 0.314685 | 134.591 |
| lda_classifier | raw_primary_with_maturity | {"shrinkage": 0.3, "solver": "lsqr"} | 0.00435564 | 0.56099 | 0.314685 | 134.591 |
| lda_classifier | raw_primary_with_maturity | {"shrinkage": 0.7, "solver": "lsqr"} | 0.00435483 | 0.560947 | 0.315226 | 134.612 |
| lda_classifier | raw_primary_with_maturity | {"shrinkage": 0.1, "solver": "lsqr"} | 0.00434404 | 0.561033 | 0.314344 | 135.375 |
| lda_classifier | raw_primary_with_maturity | {"shrinkage": null, "solver": "svd"} | 0.00432497 | 0.561033 | 0.312806 | 136.611 |
| lda_classifier | raw_primary_with_maturity | {"shrinkage": null, "solver": "svd"} | 0.00432497 | 0.561033 | 0.312806 | 136.611 |
| lda_classifier | raw_primary_with_maturity | {"shrinkage": null, "solver": "lsqr"} | 0.00432497 | 0.561033 | 0.312806 | 136.611 |
| linear_svm_classifier | raw_primary_with_maturity | {"C": 0.03, "class_weight": null} | 0.00423074 | 0.559628 | 0.306731 | 141.673 |
| linear_svm_classifier | raw_primary_with_maturity | {"C": 1.0, "class_weight": null} | 0.00422779 | 0.559628 | 0.306608 | 141.865 |
| linear_svm_classifier | raw_primary_with_maturity | {"C": 0.1, "class_weight": null} | 0.00422779 | 0.559628 | 0.306608 | 141.865 |
| linear_svm_classifier | raw_primary_with_maturity | {"C": 0.3, "class_weight": null} | 0.00422779 | 0.559628 | 0.306608 | 141.865 |
| linear_svm_classifier | raw_primary_with_maturity | {"C": 10.0, "class_weight": null} | 0.00422779 | 0.559628 | 0.306608 | 141.865 |
| linear_svm_classifier | raw_primary_with_maturity | {"C": 3.0, "class_weight": null} | 0.00422779 | 0.559628 | 0.306608 | 141.865 |
| softmax_logistic_classifier | raw_primary_with_maturity | {"C": 0.03, "class_weight": "balanced", "penalty": "l2"} | 0.00411919 | 0.545538 | 0.307246 | 138.432 |

## Skipped Validation Candidates

| base_model_name | feature_scheme | params_json | status | skip_reason |
| --- | --- | --- | --- | --- |
| linear_svm_classifier | engineered_interaction_selected | {"C": 0.03, "class_weight": "balanced"} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 0.03, "class_weight": null} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 0.1, "class_weight": "balanced"} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 0.1, "class_weight": null} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 0.3, "class_weight": "balanced"} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 0.3, "class_weight": null} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 1.0, "class_weight": "balanced"} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 1.0, "class_weight": null} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 3.0, "class_weight": "balanced"} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 3.0, "class_weight": null} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 10.0, "class_weight": "balanced"} | skipped | skipped_slow_solver |
| linear_svm_classifier | engineered_interaction_selected | {"C": 10.0, "class_weight": null} | skipped | skipped_slow_solver |

## Output Prediction Models

| output_model_name | candidate_id | base_model_name | model_family | estimator | feature_scheme | role | params_json | validation_decision_mean_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lda_classifier__baseline | lda_classifier__raw_primary_with_maturity__baseline__shrinkage_null__solver_svd | lda_classifier | lda | LinearDiscriminantAnalysis() | raw_primary_with_maturity | baseline | {"shrinkage": null, "solver": "svd"} | 0.00432497 |
| lda_classifier__best_candidate | lda_classifier__raw_primary_with_maturity__best_candidate__shrinkage_0p5__solver_lsqr | lda_classifier | lda | LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.5) | raw_primary_with_maturity | best_candidate | {"shrinkage": 0.5, "solver": "lsqr"} | 0.0043582 |
| softmax_logistic_classifier__baseline | softmax_logistic_classifier__raw_primary_with_maturity__baseline__C_1p0__class_weight_balanced__penalty_l2 | softmax_logistic_classifier | softmax_logistic | LogisticRegression(C=1.0,class_weight='balanced') | raw_primary_with_maturity | baseline | {"C": 1.0, "class_weight": "balanced", "penalty": "l2"} | 0.00408656 |
| softmax_logistic_classifier__best_candidate | softmax_logistic_classifier__raw_primary_with_maturity__best_candidate__C_0p03__class_weight_balanced__penalty_l2 | softmax_logistic_classifier | softmax_logistic | LogisticRegression(C=0.03,class_weight='balanced') | raw_primary_with_maturity | best_candidate | {"C": 0.03, "class_weight": "balanced", "penalty": "l2"} | 0.00411919 |
| linear_svm_classifier__baseline | linear_svm_classifier__raw_primary_with_maturity__baseline__C_1p0__class_weight_balanced | linear_svm_classifier | linear_svm | LinearSVC(C=1.0,class_weight='balanced',tol=1e-2) | raw_primary_with_maturity | baseline | {"C": 1.0, "class_weight": "balanced"} | 0.00398438 |
| linear_svm_classifier__best_candidate | linear_svm_classifier__raw_primary_with_maturity__best_candidate__C_0p03__class_weight_null | linear_svm_classifier | linear_svm | LinearSVC(C=0.03,class_weight=None,tol=1e-2) | raw_primary_with_maturity | best_candidate | {"C": 0.03, "class_weight": null} | 0.00423074 |

## Selected Baseline and Best Candidates

| model_name | feature_scheme | decision_ela_call_rate | decision_mean_utility | utility_capture_rate | precision_u_gt_zero_under_calls | unhelpful_call_cost_sum |
| --- | --- | --- | --- | --- | --- | --- |
| lda_classifier__best_candidate | raw_primary_with_maturity | 0.0374691 | 0.0043582 | 0.56099 | 0.315074 | 134.426 |
| lda_classifier__baseline | raw_primary_with_maturity | 0.0378395 | 0.00432497 | 0.561033 | 0.312806 | 136.611 |
| linear_svm_classifier__best_candidate | raw_primary_with_maturity | 0.038287 | 0.00423074 | 0.559628 | 0.306731 | 141.673 |
| softmax_logistic_classifier__best_candidate | raw_primary_with_maturity | 0.0372685 | 0.00411919 | 0.545538 | 0.307246 | 138.432 |
| softmax_logistic_classifier__baseline | raw_primary_with_maturity | 0.0372531 | 0.00408656 | 0.54352 | 0.306545 | 139.047 |
| linear_svm_classifier__baseline | raw_primary_with_maturity | 0.0359722 | 0.00398438 | 0.529237 | 0.306735 | 135.055 |

## Selected Models on Known Weak Layers

| model_name | layer | group | rows | decision_mean_utility | utility_capture_rate | precision_u_gt_zero_under_calls | unhelpful_call_cost_sum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| linear_svm_classifier__baseline | FE_ratio | FE_ratio=0.25 | 6480 | -0.00244869 | 0.00553063 | 0.00392157 | 16.0637 |
| softmax_logistic_classifier__best_candidate | FE_ratio | FE_ratio=0.25 | 6480 | -0.00251263 | 0.00553063 | 0.00378788 | 16.478 |
| lda_classifier__best_candidate | FE_ratio | FE_ratio=0.25 | 6480 | -0.00251348 | 0.00152768 | 0.00371747 | 16.3415 |
| lda_classifier__baseline | FE_ratio | FE_ratio=0.25 | 6480 | -0.00256607 | 0.00152768 | 0.003663 | 16.6823 |
| linear_svm_classifier__best_candidate | FE_ratio | FE_ratio=0.25 | 6480 | -0.00257499 | 0.00152768 | 0.00369004 | 16.7401 |
| softmax_logistic_classifier__baseline | FE_ratio | FE_ratio=0.25 | 6480 | -0.0026151 | 0 | 0 | 16.9459 |
| linear_svm_classifier__baseline | FE_ratio | FE_ratio=0.28 | 6480 | -0.00238456 | 0.011171 | 0.0205761 | 15.8112 |
| softmax_logistic_classifier__baseline | FE_ratio | FE_ratio=0.28 | 6480 | -0.00245731 | 0.0182779 | 0.0268199 | 16.5112 |
| softmax_logistic_classifier__best_candidate | FE_ratio | FE_ratio=0.28 | 6480 | -0.00246475 | 0.0182779 | 0.026616 | 16.5594 |
| lda_classifier__best_candidate | FE_ratio | FE_ratio=0.28 | 6480 | -0.00249214 | 0.0182779 | 0.0259259 | 16.7369 |
| lda_classifier__baseline | FE_ratio | FE_ratio=0.28 | 6480 | -0.00250089 | 0.0182779 | 0.0258303 | 16.7936 |
| linear_svm_classifier__best_candidate | FE_ratio | FE_ratio=0.28 | 6480 | -0.00259541 | 0.0182779 | 0.0252708 | 17.4061 |
| lda_classifier__best_candidate | FE_ratio | FE_ratio=0.3 | 6480 | -0.00262151 | 0.0115469 | 0.0148148 | 17.3529 |
| lda_classifier__baseline | FE_ratio | FE_ratio=0.3 | 6480 | -0.00269889 | 0.0115469 | 0.0144928 | 17.8543 |
| linear_svm_classifier__baseline | FE_ratio | FE_ratio=0.3 | 6480 | -0.00313024 | 0.0115469 | 0.0135593 | 20.6494 |
| linear_svm_classifier__best_candidate | FE_ratio | FE_ratio=0.3 | 6480 | -0.00321783 | 0.0115469 | 0.0133333 | 21.217 |
| softmax_logistic_classifier__baseline | FE_ratio | FE_ratio=0.3 | 6480 | -0.00326242 | 0.0115469 | 0.013245 | 21.506 |
| softmax_logistic_classifier__best_candidate | FE_ratio | FE_ratio=0.3 | 6480 | -0.00326242 | 0.0115469 | 0.013245 | 21.506 |
| lda_classifier__best_candidate | FE_ratio | FE_ratio=0.35 | 6480 | -0.00272591 | 0.0242013 | 0.0185185 | 18.3038 |
| linear_svm_classifier__baseline | FE_ratio | FE_ratio=0.35 | 6480 | -0.00276416 | 0.0242013 | 0.0184502 | 18.5516 |
| softmax_logistic_classifier__best_candidate | FE_ratio | FE_ratio=0.35 | 6480 | -0.002835 | 0.0242013 | 0.0181159 | 19.0107 |
| lda_classifier__baseline | FE_ratio | FE_ratio=0.35 | 6480 | -0.00283651 | 0.0242013 | 0.0181159 | 19.0205 |
| softmax_logistic_classifier__baseline | FE_ratio | FE_ratio=0.35 | 6480 | -0.00287483 | 0.0242013 | 0.0179856 | 19.2688 |
| linear_svm_classifier__best_candidate | FE_ratio | FE_ratio=0.35 | 6480 | -0.00296062 | 0.0242013 | 0.0177305 | 19.8247 |
| lda_classifier__best_candidate | dimension | dimension=40 | 21600 | -0.00312742 | 0.330085 | 0.110012 | 97.3463 |
| lda_classifier__baseline | dimension | dimension=40 | 21600 | -0.00320424 | 0.33044 | 0.11084 | 99.0376 |
| linear_svm_classifier__baseline | dimension | dimension=40 | 21600 | -0.00325 | 0.330085 | 0.107879 | 99.994 |
| softmax_logistic_classifier__best_candidate | dimension | dimension=40 | 21600 | -0.00333006 | 0.330085 | 0.106332 | 101.723 |
| softmax_logistic_classifier__baseline | dimension | dimension=40 | 21600 | -0.00334509 | 0.330085 | 0.106079 | 102.048 |
| linear_svm_classifier__best_candidate | dimension | dimension=40 | 21600 | -0.0034271 | 0.330085 | 0.104583 | 103.819 |
| lda_classifier__best_candidate | label_source | label_source=same_algorithm | 48240 | -0.00269532 | 0.0268075 | 0.0181818 | 133.664 |
| linear_svm_classifier__baseline | label_source | label_source=same_algorithm | 48240 | -0.00272572 | 0.0200323 | 0.014384 | 134.21 |
| lda_classifier__baseline | label_source | label_source=same_algorithm | 48240 | -0.00273544 | 0.0268075 | 0.0179856 | 135.599 |
| softmax_logistic_classifier__best_candidate | label_source | label_source=same_algorithm | 48240 | -0.00278341 | 0.0237792 | 0.016898 | 137.502 |
| softmax_logistic_classifier__baseline | label_source | label_source=same_algorithm | 48240 | -0.00279904 | 0.0223354 | 0.0163043 | 138.06 |
| linear_svm_classifier__best_candidate | label_source | label_source=same_algorithm | 48240 | -0.00283882 | 0.0268075 | 0.0176263 | 140.587 |

## Output Files

| name | path |
| --- | --- |
| tuning_candidate_summary | results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/tuning_candidate_summary.parquet |
| selected_feature_summary | results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/selected_feature_summary.parquet |
| output_selection_summary | results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/output_selection_summary.parquet |
| validation_predictions | results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/validation_predictions.parquet |
| split_decision_summary | results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/split_decision_summary.parquet |
| split_metric_summary | results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/split_metric_summary.parquet |
| report | results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/classifier_feature_engineering_tuning_report.md |
| summary | results/decision/phase1_refined_sampling/classifier_feature_engineering_tuning/classifier_feature_engineering_tuning_summary.json |
