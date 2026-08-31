# Decision-before-Feature Task 15A 总报告

Stage A verdict：**S-A1 COMPACT CORE IDENTIFIED**；Stage B verdict：**I3 MICRO-BEHAVIOR NO-GO**。

现有 Behavior：28 维；screened concepts：6；ISSD-Q18：18 维。

| suite   |     L_W0 |     L_WA |     L_WI |    L_WAI |   delta_within_I |   delta_aggregate_to_combined |
|:--------|---------:|---------:|---------:|---------:|-----------------:|------------------------------:|
| bbob    | -2.01761 | -2.01579 | -2.01349 | -2.01521 |      -0.0041188  |                  -0.000581036 |
| mabbob  | -5.08469 | -5.08723 | -5.08709 | -5.08729 |       0.00240513 |                   6.17107e-05 |

| suite   |   observed_delta_WA_minus_WAI |    null_mean |   empirical_p |
|:--------|------------------------------:|-------------:|--------------:|
| bbob    |                  -0.000581036 | -0.000794267 |      0.415842 |
| mabbob  |                   6.17107e-05 | -0.00409667  |      0.138614 |

资源账本：`results/analysis_v7/task15a/task15a_resource_ledger.parquet`；new action-label FE=0。

下一步建议：若 I1 成立，只进入预先定义的 Search-Role Composition 机制分析；若 I2，先做限定于已有 primitive 的机制分析；若 I3，停止继续扩展 algorithm-agnostic trajectory behavior representation。
