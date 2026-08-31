# 17a1_05 Corrected Permutation Null

Task17A.1 是零 FE 的统计正确性与估计对象复核。目的不是获得更漂亮的显著性，而是判断 Task17A 哪些机制结论在统一统计口径后仍然成立。

## 定义

每次在各 stratum 的五个 state 间置换三维 decision signature，重算十个 decision distances，再执行层内 average-rank normalization；observable ranks 保持不变。

## 结果

| domain       | suite   | representation    |    null_mean |       q95 |     q97_5 |
|:-------------|:--------|:------------------|-------------:|----------:|----------:|
| natural      | bbob    | compact6          |  0.000900338 | 0.0390113 | 0.0417484 |
| natural      | bbob    | global28          |  0.00208302  | 0.0367404 | 0.0401829 |
| natural      | mabbob  | compact6          |  0.00234143  | 0.0553631 | 0.0613447 |
| natural      | mabbob  | global28          |  0.00342545  | 0.0576061 | 0.0619444 |
| natural      | pooled  | compact6          |  0.00130666  | 0.0330567 | 0.0364582 |
| natural      | pooled  | global28          |  0.00246903  | 0.0319467 | 0.0389988 |
| post_handoff | bbob    | compact6          |  0.000621686 | 0.0259638 | 0.029604  |
| post_handoff | bbob    | compact_issd24    |  0.000773313 | 0.0267096 | 0.0304586 |
| post_handoff | bbob    | global28          | -9.63983e-05 | 0.0234134 | 0.028501  |
| post_handoff | bbob    | issd18            | -1.52621e-05 | 0.0236899 | 0.0300348 |
| post_handoff | bbob    | segment_matched28 | -0.0010995   | 0.0226075 | 0.0270521 |
| post_handoff | bbob    | segment_old28     |  0.000949858 | 0.0229379 | 0.0264539 |
| post_handoff | mabbob  | compact6          |  0.000927987 | 0.041506  | 0.0486192 |
| post_handoff | mabbob  | compact_issd24    |  0.00308469  | 0.0435007 | 0.0467752 |
| post_handoff | mabbob  | global28          | -0.000761486 | 0.0383791 | 0.045227  |
| post_handoff | mabbob  | issd18            | -0.00010106  | 0.0447061 | 0.0503759 |
| post_handoff | mabbob  | segment_matched28 | -0.00210101  | 0.0425838 | 0.0544321 |
| post_handoff | mabbob  | segment_old28     | -0.00247185  | 0.0305157 | 0.0371177 |
| post_handoff | pooled  | compact6          |  0.000681893 | 0.0197159 | 0.0230645 |
| post_handoff | pooled  | compact_issd24    |  0.00141607  | 0.0255208 | 0.0313487 |
| post_handoff | pooled  | global28          | -0.000290948 | 0.0208088 | 0.0230097 |
| post_handoff | pooled  | issd18            | -3.32789e-05 | 0.0219931 | 0.030045  |
| post_handoff | pooled  | segment_matched28 | -0.00137544  | 0.0190378 | 0.022296  |
| post_handoff | pooled  | segment_old28     | -2.94033e-05 | 0.0190016 | 0.0231076 |
