# 语义族映射

| concept_id                   | semantic_family                | primary_aggregate_representative             | sensitivity_aggregate_representative   | issd_primitive   | screened_core   | selection_basis                                                                                                                    | rationale                                                                                                |
|:-----------------------------|:-------------------------------|:---------------------------------------------|:---------------------------------------|:-----------------|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|
| movement                     | Movement                       | bf_population_centroid_shift_w05             | bf_population_chamfer_distance_w05     | movement         | True            | legality > interpretability > numerical stability > algorithm agnosticism > scale/population normalization > cost > fold stability | centroid displacement is a simple domain-scaled population movement summary                              |
| direction_coordination       | Direction / Coordination       | bf_centroid_shift_coherence_w05              | bf_covariance_spectral_concentration   | direction        | True            | legality > interpretability > numerical stability > algorithm agnosticism > scale/population normalization > cost > fold stability | coherence separates aligned displacement from changing population motion                                 |
| progress_contribution        | Progress / Contribution        | bf_fitness_distribution_improvement_rate_w02 | bf_improvement_rate_w02                | progress         | True            | legality > interpretability > numerical stability > algorithm agnosticism > scale/population normalization > cost > fold stability | distribution-level improvement is closer to individual productive progress than best-only change         |
| fitness_stagnation           | Stagnation                     | bf_stagnation_w10                            | bf_improvement_frequency_w02           | stagnation       | True            | legality > interpretability > numerical stability > algorithm agnosticism > scale/population normalization > cost > fold stability | the existing stagnation rate is explicit and online, while the ISSD primitive uses per-agent fitness age |
| relative_position_dispersion | Relative Position / Dispersion | bf_elite_concentration                       | bf_diversity_mean_pairwise             | elite_distance   | True            | legality > interpretability > numerical stability > algorithm agnosticism > scale/population normalization > cost > fold stability | elite concentration describes the location of the population relative to its current high-quality region |
| fitness_distribution         | Fitness Distribution           | bf_fitness_diversity_rel                     | bf_fitness_wasserstein_rate_w02        | rank             | True            | legality > interpretability > numerical stability > algorithm agnosticism > scale/population normalization > cost > fold stability | relative fitness spread is scale-aware; ISSD uses normalized individual rank                             |

| feature_name                                 | semantic_family                                | screened_core   |
|:---------------------------------------------|:-----------------------------------------------|:----------------|
| bf_fe_ratio                                  | Temporal Context (not a search-state behavior) | False           |
| bf_improvement_rate_w02                      | Progress / Contribution                        | False           |
| bf_improvement_frequency_w02                 | Stagnation                                     | False           |
| bf_diversity_mean_pairwise                   | Relative Position / Dispersion                 | False           |
| bf_diversity_change_w05                      | Relative Position / Dispersion                 | False           |
| bf_covariance_spectral_concentration         | Direction / Coordination                       | False           |
| bf_distance_decay_w10                        | Relative Position / Dispersion                 | False           |
| bf_stagnation_w10                            | Stagnation                                     | True            |
| bf_convergence_rate_w10                      | Progress / Contribution                        | False           |
| bf_fitness_diversity_rel                     | Fitness Distribution                           | True            |
| bf_population_wasserstein_rate_w05           | Movement                                       | False           |
| bf_centroid_shift_rate_w05                   | Movement                                       | False           |
| bf_centroid_shift_coherence_w05              | Direction / Coordination                       | True            |
| bf_fitness_quantile_improvement_fraction_w02 | Progress / Contribution                        | False           |
| bf_fitness_distribution_improvement_rate_w02 | Progress / Contribution                        | True            |
| bf_fitness_wasserstein_rate_w02              | Fitness Distribution                           | False           |
| bf_elite_concentration                       | Relative Position / Dispersion                 | True            |
| bf_best_fitness_slope_rel_w05                | Progress / Contribution                        | False           |
| bf_diversity_slope_w05                       | Relative Position / Dispersion                 | False           |
| bf_fitness_spread_slope_w05                  | Fitness Distribution                           | False           |
| bf_population_centroid_shift_w05             | Movement                                       | True            |
| bf_elite_centroid_shift_w05                  | Movement                                       | False           |
| bf_covariance_trace_ratio_w05                | Relative Position / Dispersion                 | False           |
| bf_covariance_effective_rank_w05             | Relative Position / Dispersion                 | False           |
| bf_diversity_recovery_w05                    | Relative Position / Dispersion                 | False           |
| bf_population_chamfer_distance_w05           | Movement                                       | False           |
| bf_covariance_trace_change_w05               | Direction / Coordination                       | False           |
| bf_covariance_effective_rank_change_w05      | Relative Position / Dispersion                 | False           |
