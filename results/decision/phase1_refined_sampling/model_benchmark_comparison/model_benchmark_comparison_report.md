# BBOB Decision Model Benchmark Comparison

## Scope

- Dataset: existing phase1 refined sampling materialized BBOB Decision dataset.
- Train split: `bbob_train`; validation split: `bbob_validation`.
- No validation rows were used to fit preprocessing, model parameters, or thresholds.
- Feature group: `primary_with_maturity` with 23 behavior-only input columns.
- Classification models are evaluated as decision-score models using probability or decision-function scores.

## Input Contract

| check | passed | detail |
| --- | --- | --- |
| x_columns_subset_of_behavior_feature_columns | True | bf_fe_ratio,bf_improvement_rate_w02,bf_improvement_frequency_w02,bf_diversity_mean_pairwise,bf_diversity_change_w05,bf_directional_entropy_w05,bf_distance_decay_w10,bf_stagnation_w10,bf_convergence_rate_w10,bf_fitness_diversity,bf_fitness_diversity_rel,bf_movement_magnitude,bf_movement_diversity,bf_direction_consistency_w05,bf_success_rate_w02,bf_improvement_variance_w02,bf_best_improvement_ratio_w02,bf_elite_concentration,bf_best_fitness_slope_w05,bf_diversity_slope_w05,bf_search_maturity,bf_search_maturity_linear,bf_explore_exploit_ratio |
| fit_split_is_train_only | True | bbob_train rows=194400; validation rows used for fit=0 |
| metadata_retained_for_reporting_only | True | split,problem_id,family,dimension,prefix_algorithm,seed,FE,FE_ratio,default_algorithm,selected_algorithm,label_source |

## Models

| model_name | model_family | objective | fit_seconds | validation_prediction_seconds |
| --- | --- | --- | --- | --- |
| dummy_mean_regression | dummy | regression | 0.294925 | 0.00504771 |
| ridge_regression | ridge | regression | 0.308408 | 0.00619717 |
| linear_regression | linear | regression | 0.321873 | 0.00647596 |
| bayesian_ridge_regression | bayesian_ridge | regression | 0.345082 | 0.00603487 |
| elastic_net_regression | elastic_net | regression | 0.368434 | 0.00680217 |
| lda_classifier | lda | classification | 0.387908 | 0.00722317 |
| softmax_logistic_classifier | softmax_logistic | classification | 0.419411 | 0.00694646 |
| rbf_nystroem_ridge_regression | kernel_ridge_approx | regression | 0.729467 | 0.111074 |
| xgboost_regression | xgboost | regression | 0.838192 | 0.0179513 |
| linear_svm_classifier | linear_svm | classification | 0.92237 | 0.00699471 |
| lightgbm_regression | lightgbm | regression | 1.97181 | 0.0728736 |
| hist_gradient_boosting_regression | hist_gradient_boosting | regression | 2.39938 | 0.0911098 |
| mlp_classifier | mlp_classifier | classification | 3.6769 | 0.0164749 |
| mlp_regression | mlp | regression | 5.84591 | 0.0162223 |
| extra_trees_regression | extra_trees | regression | 21.587 | 0.408433 |
| rbf_nystroem_svm_regression | kernel_svm_approx | regression | 38.2779 | 0.0988778 |
| random_forest_regression | random_forest | regression | 107.858 | 0.292884 |
| linear_svm_regression | linear_svm | regression | 131.366 | 0.00680733 |

## Best Rows

| criterion | metric | status | model_name | model_family | objective | threshold_mode | top_k_fraction | rmse | spearman | average_precision_u_gt_zero | decision_mean_utility | decision_ela_call_rate | utility_capture_rate | precision_u_gt_zero_under_calls | top_k_u_gt_zero_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lowest_validation_rmse_regression_models | rmse | available | rbf_nystroem_svm_regression | kernel_svm_approx | regression |  |  | 0.128917 | 0.409774 | 0.0778069 |  |  |  |  |  |
| highest_validation_spearman | spearman | available | linear_svm_regression | linear_svm | regression |  |  | 0.405941 | 0.456383 | 0.107243 |  |  |  |  |  |
| highest_validation_average_precision | average_precision_u_gt_zero | available | softmax_logistic_classifier | softmax_logistic | classification |  |  |  | 0.0912398 | 0.176193 |  |  |  |  |  |
| highest_train_utility_threshold_decision_mean_utility | decision_mean_utility | available | lda_classifier | lda | classification | train_utility |  |  |  |  | 0.00432497 | 0.0378395 | 0.561033 | 0.312806 |  |
| highest_top10_threshold_decision_mean_utility | decision_mean_utility | available | elastic_net_regression | elastic_net | regression | top_10 |  |  |  |  | 0.00122191 | 0.0968827 | 0.603956 | 0.177445 |  |
| highest_top10_ranking_utility_capture_rate | utility_capture_rate | available | softmax_logistic_classifier | softmax_logistic | classification |  | 0.1 |  |  |  |  |  | 0.646442 |  | 0.190123 |

## Validation Score Metrics

| model_name | objective | rmse | spearman | roc_auc_u_gt_zero | average_precision_u_gt_zero |
| --- | --- | --- | --- | --- | --- |
| softmax_logistic_classifier | classification |  | 0.0912398 | 0.811795 | 0.176193 |
| linear_svm_classifier | classification |  | 0.0902687 | 0.812831 | 0.174539 |
| lda_classifier | classification |  | 0.0886929 | 0.815529 | 0.174294 |
| mlp_classifier | classification |  | 0.152787 | 0.816007 | 0.144729 |
| elastic_net_regression | regression | 0.135859 | 0.426694 | 0.695811 | 0.138241 |
| bayesian_ridge_regression | regression | 0.216378 | 0.428043 | 0.694141 | 0.138059 |
| ridge_regression | regression | 0.218232 | 0.428243 | 0.693874 | 0.138034 |
| linear_regression | regression | 0.218243 | 0.428249 | 0.693872 | 0.138034 |
| extra_trees_regression | regression | 0.129706 | 0.413006 | 0.688625 | 0.112177 |
| linear_svm_regression | regression | 0.405941 | 0.456383 | 0.650193 | 0.107243 |
| mlp_regression | regression | 4.57365 | 0.353666 | 0.635742 | 0.0855751 |
| rbf_nystroem_ridge_regression | regression | 0.132005 | 0.349092 | 0.631819 | 0.0793805 |
| rbf_nystroem_svm_regression | regression | 0.128917 | 0.409774 | 0.631424 | 0.0778069 |
| xgboost_regression | regression | 0.155501 | 0.291338 | 0.606825 | 0.0734987 |
| random_forest_regression | regression | 0.151269 | 0.359711 | 0.576385 | 0.07242 |
| lightgbm_regression | regression | 0.139698 | 0.325951 | 0.583107 | 0.0681709 |
| hist_gradient_boosting_regression | regression | 0.144001 | 0.325659 | 0.572986 | 0.0654612 |
| dummy_mean_regression | regression | 0.133385 |  | 0.5 | 0.0577932 |

## Validation Decision: Train Utility Threshold

| model_name | objective | decision_ela_call_rate | decision_mean_utility | positive_row_capture_rate | utility_capture_rate | precision_u_gt_zero_under_calls |
| --- | --- | --- | --- | --- | --- | --- |
| lda_classifier | classification | 0.0378395 | 0.00432497 | 0.204806 | 0.561033 | 0.312806 |
| softmax_logistic_classifier | classification | 0.0372531 | 0.00408656 | 0.197597 | 0.54352 | 0.306545 |
| linear_svm_classifier | classification | 0.0359722 | 0.00398438 | 0.190921 | 0.529237 | 0.306735 |
| elastic_net_regression | regression | 0.027037 | 0.00271719 | 0.126569 | 0.351474 | 0.270548 |
| ridge_regression | regression | 0.0266204 | 0.00265025 | 0.123097 | 0.343901 | 0.267246 |
| linear_regression | regression | 0.0266204 | 0.00265025 | 0.123097 | 0.343901 | 0.267246 |
| bayesian_ridge_regression | regression | 0.0266204 | 0.00264921 | 0.123097 | 0.343901 | 0.267246 |
| hist_gradient_boosting_regression | regression | 0.000848765 | 4.3892e-05 | 0.00320427 | 0.00820873 | 0.218182 |
| lightgbm_regression | regression | 0.000910494 | 3.93234e-05 | 0.00320427 | 0.0081533 | 0.20339 |
| dummy_mean_regression | regression | 0 | 0 | 0 | 0 | 0 |
| linear_svm_regression | regression | 0.000216049 | -6.39529e-06 | 0 | 0 | 0 |
| extra_trees_regression | regression | 0.0194753 | -4.86351e-05 | 0.0606142 | 0.0973395 | 0.179873 |
| xgboost_regression | regression | 0.00268519 | -0.000139192 | 0.000801068 | 0.00203911 | 0.0172414 |
| rbf_nystroem_svm_regression | regression | 0.00123457 | -0.000182569 | 0.00160214 | 0.00147626 | 0.075 |
| random_forest_regression | regression | 0.00407407 | -0.000406905 | 0.0104139 | 0.0104833 | 0.147727 |
| mlp_classifier | classification | 0.0150463 | -0.00108361 | 0.0200267 | 0.0326018 | 0.0769231 |
| rbf_nystroem_ridge_regression | regression | 0.0251852 | -0.00208032 | 0.0379172 | 0.0222853 | 0.0870098 |
| mlp_regression | regression | 0.0511111 | -0.00384137 | 0.0923899 | 0.0620194 | 0.104469 |

## Validation Decision: Top 10% Train-Score Threshold

| model_name | objective | decision_ela_call_rate | decision_mean_utility | positive_row_capture_rate | utility_capture_rate | precision_u_gt_zero_under_calls |
| --- | --- | --- | --- | --- | --- | --- |
| elastic_net_regression | regression | 0.0968827 | 0.00122191 | 0.297463 | 0.603956 | 0.177445 |
| bayesian_ridge_regression | regression | 0.0973457 | 0.00118754 | 0.297463 | 0.603647 | 0.176601 |
| linear_regression | regression | 0.0974537 | 0.00118504 | 0.29773 | 0.603914 | 0.176564 |
| ridge_regression | regression | 0.0974537 | 0.00118504 | 0.29773 | 0.603914 | 0.176564 |
| linear_svm_regression | regression | 0.0949074 | 8.03867e-05 | 0.23765 | 0.483278 | 0.144715 |
| dummy_mean_regression | regression | 0 | 0 | 0 | 0 | 0 |
| extra_trees_regression | regression | 0.0537809 | -0.0003453 | 0.15514 | 0.262503 | 0.166714 |
| random_forest_regression | regression | 0.0125309 | -0.000904166 | 0.0264352 | 0.0278016 | 0.121921 |
| hist_gradient_boosting_regression | regression | 0.0236728 | -0.00129343 | 0.0152203 | 0.0259098 | 0.0371578 |
| lightgbm_regression | regression | 0.031358 | -0.00150296 | 0.0259012 | 0.0451794 | 0.0477362 |
| xgboost_regression | regression | 0.0585802 | -0.00323304 | 0.0859813 | 0.0945898 | 0.0848261 |
| linear_svm_classifier | classification | 0.12179 | -0.00362039 | 0.375968 | 0.669653 | 0.178409 |
| softmax_logistic_classifier | classification | 0.126034 | -0.00407066 | 0.389319 | 0.681355 | 0.178523 |
| rbf_nystroem_svm_regression | regression | 0.0838426 | -0.00533225 | 0.114553 | 0.0596552 | 0.0789619 |
| mlp_classifier | classification | 0.146867 | -0.00550549 | 0.396529 | 0.644712 | 0.156037 |
| lda_classifier | classification | 0.142778 | -0.00660507 | 0.388251 | 0.688416 | 0.157155 |
| mlp_regression | regression | 0.118611 | -0.00798142 | 0.204005 | 0.163288 | 0.0994015 |
| rbf_nystroem_ridge_regression | regression | 0.127747 | -0.00979261 | 0.187183 | 0.0981968 | 0.0846823 |

## Validation Top 10% Ranking

| model_name | objective | top_k_u_gt_zero_rate | lift_vs_base_rate | positive_row_capture_rate | utility_capture_rate | top_k_mean_observed_utility |
| --- | --- | --- | --- | --- | --- | --- |
| softmax_logistic_classifier | classification | 0.190123 | 3.28972 | 0.328972 | 0.646442 | -0.016508 |
| linear_svm_classifier | classification | 0.188889 | 3.26836 | 0.326836 | 0.639315 | -0.0164181 |
| lda_classifier | classification | 0.174846 | 3.02537 | 0.302537 | 0.631325 | -0.0236593 |
| elastic_net_regression | regression | 0.174537 | 3.02003 | 0.302003 | 0.605578 | 0.0101969 |
| bayesian_ridge_regression | regression | 0.174537 | 3.02003 | 0.302003 | 0.605437 | 0.0103538 |
| ridge_regression | regression | 0.174383 | 3.01736 | 0.301736 | 0.605282 | 0.0103609 |
| linear_regression | regression | 0.174383 | 3.01736 | 0.301736 | 0.605282 | 0.0103609 |
| mlp_classifier | classification | 0.166358 | 2.8785 | 0.28785 | 0.541574 | -0.0235327 |
| linear_svm_regression | regression | 0.143364 | 2.48064 | 0.248064 | 0.493224 | -0.00146576 |
| extra_trees_regression | regression | 0.131019 | 2.26702 | 0.226702 | 0.339948 | -0.0241728 |
| mlp_regression | regression | 0.102932 | 1.78104 | 0.178104 | 0.139202 | -0.0687529 |
| xgboost_regression | regression | 0.0759259 | 1.31375 | 0.131375 | 0.12759 | -0.0575207 |
| random_forest_regression | regression | 0.0705247 | 1.22029 | 0.122029 | 0.112593 | -0.0543657 |
| lightgbm_regression | regression | 0.0530864 | 0.918558 | 0.0918558 | 0.106209 | -0.0523144 |
| hist_gradient_boosting_regression | regression | 0.0444444 | 0.769025 | 0.0769025 | 0.0806568 | -0.0550444 |
| rbf_nystroem_ridge_regression | regression | 0.0864198 | 1.49533 | 0.149533 | 0.0761169 | -0.0786213 |
| rbf_nystroem_svm_regression | regression | 0.0773148 | 1.33778 | 0.133778 | 0.0711497 | -0.0629213 |
| dummy_mean_regression | regression | 0.000154321 | 0.00267023 | 0.000267023 | 4.13467e-06 | -0.108408 |

## Train Decision: Train Utility Threshold

| model_name | objective | decision_ela_call_rate | decision_mean_utility | positive_row_capture_rate | utility_capture_rate | precision_u_gt_zero_under_calls |
| --- | --- | --- | --- | --- | --- | --- |
| extra_trees_regression | regression | 0.0608333 | 0.012107 | 0.837346 | 0.963004 | 0.844072 |
| random_forest_regression | regression | 0.0632202 | 0.0118873 | 0.840869 | 0.955856 | 0.815622 |
| lightgbm_regression | regression | 0.0409362 | 0.0064461 | 0.44149 | 0.596001 | 0.661347 |
| hist_gradient_boosting_regression | regression | 0.0437294 | 0.005963 | 0.44753 | 0.572705 | 0.627573 |
| xgboost_regression | regression | 0.0338374 | 0.00384516 | 0.316668 | 0.398995 | 0.573883 |
| mlp_regression | regression | 0.0386986 | 0.00188626 | 0.225988 | 0.280026 | 0.358102 |
| mlp_classifier | classification | 0.0198302 | 0.00144381 | 0.183542 | 0.198188 | 0.567575 |
| rbf_nystroem_ridge_regression | regression | 0.0136728 | 0.000348966 | 0.0715544 | 0.0770779 | 0.320918 |
| lda_classifier | classification | 0.017572 | 0.000305927 | 0.0925258 | 0.104125 | 0.322892 |
| softmax_logistic_classifier | classification | 0.0120525 | 0.000254472 | 0.0656824 | 0.0770271 | 0.334187 |
| linear_svm_classifier | classification | 0.0107767 | 0.000186039 | 0.0582166 | 0.0673376 | 0.331265 |
| rbf_nystroem_svm_regression | regression | 0.00158436 | 0.000123569 | 0.0112407 | 0.014335 | 0.435065 |
| linear_svm_regression | regression | 0.000102881 | 3.9771e-06 | 0.000671085 | 0.00100419 | 0.4 |
| elastic_net_regression | regression | 0.00037037 | 2.05952e-06 | 0.00159383 | 0.00152892 | 0.263889 |
| bayesian_ridge_regression | regression | 0.000339506 | 1.4914e-06 | 0.00150994 | 0.00141788 | 0.272727 |
| linear_regression | regression | 0.000339506 | 1.4914e-06 | 0.00150994 | 0.00141788 | 0.272727 |
| ridge_regression | regression | 0.000339506 | 1.4914e-06 | 0.00150994 | 0.00141788 | 0.272727 |
| dummy_mean_regression | regression | 0 | 0 | 0 | 0 | 0 |

## Target Models: Validation Decision by Label Source

| model_name | group | rows | decision_mean_utility | utility_capture_rate | precision_u_gt_zero_under_calls | unhelpful_call_cost_sum |
| --- | --- | --- | --- | --- | --- | --- |
| lda_classifier | label_source=changed_algorithm | 16560 | 0.0248922 | 0.680561 | 0.940051 | 1.01154 |
| softmax_logistic_classifier | label_source=changed_algorithm | 16560 | 0.0241446 | 0.66013 | 0.940633 | 0.987077 |
| linear_svm_classifier | label_source=changed_algorithm | 16560 | 0.0235312 | 0.643167 | 0.945355 | 0.845339 |
| ridge_regression | label_source=changed_algorithm | 16560 | 0.0152638 | 0.416589 | 0.971175 | 0.179149 |
| ridge_regression | label_source=same_algorithm | 48240 | -0.00167976 | 0.0190282 | 0.0180534 | 83.6168 |
| linear_svm_classifier | label_source=same_algorithm | 48240 | -0.00272572 | 0.0200323 | 0.014384 | 134.21 |
| lda_classifier | label_source=same_algorithm | 48240 | -0.00273544 | 0.0268075 | 0.0179856 | 135.599 |
| softmax_logistic_classifier | label_source=same_algorithm | 48240 | -0.00279904 | 0.0223354 | 0.0163043 | 138.06 |

## Target Models: Validation Decision by Dimension

| model_name | group | rows | decision_mean_utility | utility_capture_rate | precision_u_gt_zero_under_calls | unhelpful_call_cost_sum |
| --- | --- | --- | --- | --- | --- | --- |
| lda_classifier | dimension=10 | 21600 | 0.00654241 | 0.603366 | 0.346863 | 15.7693 |
| softmax_logistic_classifier | dimension=10 | 21600 | 0.00617906 | 0.570035 | 0.347368 | 14.9398 |
| linear_svm_classifier | dimension=10 | 21600 | 0.00568752 | 0.524526 | 0.337625 | 13.709 |
| ridge_regression | dimension=10 | 21600 | 0.00389351 | 0.374065 | 0.288991 | 13.2874 |
| lda_classifier | dimension=20 | 21600 | 0.00963674 | 0.585985 | 0.481663 | 21.804 |
| linear_svm_classifier | dimension=20 | 21600 | 0.00951563 | 0.578169 | 0.483271 | 21.3524 |
| softmax_logistic_classifier | dimension=20 | 21600 | 0.00942571 | 0.575021 | 0.474847 | 22.0594 |
| ridge_regression | dimension=20 | 21600 | 0.005958 | 0.378893 | 0.376396 | 19.9958 |
| ridge_regression | dimension=40 | 21600 | -0.00190077 | 0.104763 | 0.0810811 | 50.5127 |
| lda_classifier | dimension=40 | 21600 | -0.00320424 | 0.33044 | 0.11084 | 99.0376 |
| linear_svm_classifier | dimension=40 | 21600 | -0.00325 | 0.330085 | 0.107879 | 99.994 |
| softmax_logistic_classifier | dimension=40 | 21600 | -0.00334509 | 0.330085 | 0.106079 | 102.048 |

## Target Models: Validation Decision by FE Ratio

| model_name | group | rows | decision_mean_utility | utility_capture_rate | precision_u_gt_zero_under_calls | unhelpful_call_cost_sum |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_regression | FE_ratio=0.2 | 6480 | 0 | 0 | 0 | -0 |
| softmax_logistic_classifier | FE_ratio=0.2 | 6480 | 0 | 0 | 0 | -0 |
| lda_classifier | FE_ratio=0.2 | 6480 | 0 | 0 | 0 | -0 |
| linear_svm_classifier | FE_ratio=0.2 | 6480 | 0 | 0 | 0 | -0 |
| linear_svm_classifier | FE_ratio=0.25 | 6480 | -0.00244869 | 0.00553063 | 0.00392157 | 16.0637 |
| ridge_regression | FE_ratio=0.25 | 6480 | -0.00246258 | 0.00152768 | 0.0037594 | 16.0117 |
| lda_classifier | FE_ratio=0.25 | 6480 | -0.00256607 | 0.00152768 | 0.003663 | 16.6823 |
| softmax_logistic_classifier | FE_ratio=0.25 | 6480 | -0.0026151 | 0 | 0 | 16.9459 |
| linear_svm_classifier | FE_ratio=0.28 | 6480 | -0.00238456 | 0.011171 | 0.0205761 | 15.8112 |
| ridge_regression | FE_ratio=0.28 | 6480 | -0.0024256 | 0.0162891 | 0.0228137 | 16.2418 |
| softmax_logistic_classifier | FE_ratio=0.28 | 6480 | -0.00245731 | 0.0182779 | 0.0268199 | 16.5112 |
| lda_classifier | FE_ratio=0.28 | 6480 | -0.00250089 | 0.0182779 | 0.0258303 | 16.7936 |
| ridge_regression | FE_ratio=0.3 | 6480 | -0.0022006 | 0.0115469 | 0.0163265 | 14.6254 |
| lda_classifier | FE_ratio=0.3 | 6480 | -0.00269889 | 0.0115469 | 0.0144928 | 17.8543 |
| linear_svm_classifier | FE_ratio=0.3 | 6480 | -0.00313024 | 0.0115469 | 0.0135593 | 20.6494 |
| softmax_logistic_classifier | FE_ratio=0.3 | 6480 | -0.00326242 | 0.0115469 | 0.013245 | 21.506 |
| ridge_regression | FE_ratio=0.35 | 6480 | -0.00223341 | 0.0242013 | 0.021097 | 15.1124 |
| linear_svm_classifier | FE_ratio=0.35 | 6480 | -0.00276416 | 0.0242013 | 0.0184502 | 18.5516 |
| lda_classifier | FE_ratio=0.35 | 6480 | -0.00283651 | 0.0242013 | 0.0181159 | 19.0205 |
| softmax_logistic_classifier | FE_ratio=0.35 | 6480 | -0.00287483 | 0.0242013 | 0.0179856 | 19.2688 |
| lda_classifier | FE_ratio=0.4 | 6480 | 0.0147495 | 0.5978 | 0.766423 | 8.53538 |
| softmax_logistic_classifier | FE_ratio=0.4 | 6480 | 0.0145366 | 0.590579 | 0.766667 | 8.65782 |
| linear_svm_classifier | FE_ratio=0.4 | 6480 | 0.0143467 | 0.582456 | 0.763158 | 8.47335 |
| ridge_regression | FE_ratio=0.4 | 6480 | 0.0136171 | 0.531268 | 0.851163 | 4.28614 |
| ridge_regression | FE_ratio=0.45 | 6480 | 0.00291094 | 0.561442 | 0.180233 | 9.52468 |
| lda_classifier | FE_ratio=0.45 | 6480 | 0.00181652 | 0.62728 | 0.136531 | 19.9454 |
| linear_svm_classifier | FE_ratio=0.45 | 6480 | 0.00180927 | 0.626329 | 0.133588 | 19.9444 |
| softmax_logistic_classifier | FE_ratio=0.45 | 6480 | 0.00177202 | 0.626589 | 0.134831 | 20.1988 |
| lda_classifier | FE_ratio=0.5 | 6480 | 0.0192279 | 0.857414 | 0.848148 | 5.0046 |
| softmax_logistic_classifier | FE_ratio=0.5 | 6480 | 0.0189857 | 0.847166 | 0.844697 | 5.02485 |
| linear_svm_classifier | FE_ratio=0.5 | 6480 | 0.0186925 | 0.833351 | 0.844961 | 4.83676 |
| ridge_regression | FE_ratio=0.5 | 6480 | 0.0117563 | 0.513922 | 0.909091 | 1.50047 |
| lda_classifier | FE_ratio=0.55 | 6480 | 0.0046775 | 0.63321 | 0.354244 | 20.1472 |
| softmax_logistic_classifier | FE_ratio=0.55 | 6480 | 0.00457364 | 0.606941 | 0.367589 | 18.7269 |
| linear_svm_classifier | FE_ratio=0.55 | 6480 | 0.00449342 | 0.59809 | 0.374486 | 18.5414 |
| ridge_regression | FE_ratio=0.55 | 6480 | 0.00294105 | 0.298182 | 0.405405 | 4.70262 |
| lda_classifier | FE_ratio=0.6 | 6480 | 0.0133806 | 0.806258 | 0.659259 | 12.6277 |
| softmax_logistic_classifier | FE_ratio=0.6 | 6480 | 0.0122073 | 0.741132 | 0.652174 | 12.2069 |
| linear_svm_classifier | FE_ratio=0.6 | 6480 | 0.0112296 | 0.689519 | 0.642857 | 12.1837 |
| ridge_regression | FE_ratio=0.6 | 6480 | 0.00459922 | 0.256436 | 0.767123 | 1.79083 |

## Output Files

| name | path |
| --- | --- |
| model_input_contract | results/decision/phase1_refined_sampling/model_benchmark_comparison/model_input_contract.parquet |
| model_fit_summary | results/decision/phase1_refined_sampling/model_benchmark_comparison/model_fit_summary.parquet |
| decision_thresholds | results/decision/phase1_refined_sampling/model_benchmark_comparison/decision_thresholds.parquet |
| train_predictions | results/decision/phase1_refined_sampling/model_benchmark_comparison/train_predictions.parquet |
| validation_predictions | results/decision/phase1_refined_sampling/model_benchmark_comparison/validation_predictions.parquet |
| split_metric_summary | results/decision/phase1_refined_sampling/model_benchmark_comparison/split_metric_summary.parquet |
| split_decision_summary | results/decision/phase1_refined_sampling/model_benchmark_comparison/split_decision_summary.parquet |
| split_ranking_summary | results/decision/phase1_refined_sampling/model_benchmark_comparison/split_ranking_summary.parquet |
| best_summary | results/decision/phase1_refined_sampling/model_benchmark_comparison/model_benchmark_best_summary.parquet |
| report | results/decision/phase1_refined_sampling/model_benchmark_comparison/model_benchmark_comparison_report.md |
| summary | results/decision/phase1_refined_sampling/model_benchmark_comparison/model_benchmark_comparison_summary.json |
