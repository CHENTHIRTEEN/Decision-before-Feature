# 16A.1-08 Maturity monotonicity

> Task16A.1 为零 FE 的事后诊断性复核，不替代 Task16A 预先指定的正式结论。

Pooled R2 Spearman rho=-0.3118 [-0.5138, -0.1174]。

| scope_type        | scope_value   |   num_states |   spearman_rho |   ci95_low |   ci95_high |   bootstrap_finite_draws |
|:------------------|:--------------|-------------:|---------------:|-----------:|------------:|-------------------------:|
| pooled            | all           |          402 |     -0.311776  |  -0.51384  | -0.117438   |                     5000 |
| suite             | bbob          |          312 |     -0.342004  |  -0.589208 | -0.0958522  |                     5000 |
| suite             | mabbob        |           90 |     -0.220352  |  -0.54716  | -0.0247361  |                     5000 |
| current_algorithm | cso           |          202 |     -0.427682  |  -0.702819 | -0.139636   |                     5000 |
| current_algorithm | lshade        |          124 |     -0.10962   |  -0.300621 |  0.0235631  |                     4993 |
| current_algorithm | shade         |           76 |     -0.128689  |  -0.319304 |  0.00446539 |                     5000 |
| suite_current     | bbob|cso      |          159 |     -0.507608  |  -0.808145 | -0.196899   |                     5000 |
| suite_current     | bbob|lshade   |           90 |     -0.152913  |  -0.578941 |  0.116823   |                     4908 |
| suite_current     | bbob|shade    |           63 |     -0.142272  |  -0.360996 |  0.0430052  |                     5000 |
| suite_current     | mabbob|cso    |           43 |     -0.185911  |  -0.622074 |  0.210672   |                     5000 |
| suite_current     | mabbob|lshade |           34 |     -0.0876149 |  -0.214479 |  0          |                     4608 |
| suite_current     | mabbob|shade  |           13 |     -0.520416  |  -0.552771 | -0.520416   |                     3750 |

结论：**M1 STRUCTURED**。
