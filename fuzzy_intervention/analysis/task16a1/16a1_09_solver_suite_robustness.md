# 16A.1-09 Solver/suite robustness

> Task16A.1 为零 FE 的事后诊断性复核，不替代 Task16A 预先指定的正式结论。

| analysis           | scope_type        | scope_value   |   point_estimate |    ci95_low |   ci95_high |   valid_cells_or_states |   sign | ci_excludes_zero   |
|:-------------------|:------------------|:--------------|-----------------:|------------:|------------:|------------------------:|-------:|:-------------------|
| A_Z_I_R2_minus_R1  | suite             | bbob          |      -0.0293084  | -0.105714   |  0.0380952  |                       6 |     -1 | False              |
| A_Z_I_R2_minus_R1  | suite             | mabbob        |       0.0544444  | -0.0833333  |  0.2625     |                       3 |      1 | False              |
| A_Z_I_R2_minus_R1  | current_algorithm | cso           |       0.0280952  | -0.0357143  |  0.120141   |                       7 |      1 | False              |
| A_Z_I_R2_minus_R1  | current_algorithm | lshade        |       0          |  0          |  0          |                       1 |      0 | False              |
| A_Z_I_R2_minus_R1  | current_algorithm | shade         |      -0.209184   | -0.5        |  0.107143   |                       1 |     -1 | False              |
| A_Z_I_R2_minus_R1  | source_FE         | 2000          |      -0.0142857  | -0.05       |  0          |                       1 |     -1 | False              |
| A_Z_I_R2_minus_R1  | source_FE         | 4000          |       0.0416667  |  0          |  0.125      |                       2 |      1 | False              |
| A_Z_I_R2_minus_R1  | source_FE         | 6000          |       0.0738095  |  0          |  0.216667   |                       2 |      1 | False              |
| A_Z_I_R2_minus_R1  | source_FE         | 8000          |      -0.0572959  | -0.222222   |  0.075      |                       4 |     -1 | False              |
| B1_G_I_R2_minus_R1 | suite             | bbob          |      -0.0285509  | -0.183872   | -0.00358417 |                       6 |     -1 | True               |
| B1_G_I_R2_minus_R1 | suite             | mabbob        |      -0.042062   | -0.235149   | -0.00151543 |                       3 |     -1 | True               |
| B1_G_I_R2_minus_R1 | current_algorithm | cso           |      -0.0295822  | -0.14704    | -0.0051635  |                       7 |     -1 | True               |
| B1_G_I_R2_minus_R1 | current_algorithm | lshade        |      -0.0881653  | -0.0881653  | -0.0881653  |                       1 |     -1 | True               |
| B1_G_I_R2_minus_R1 | current_algorithm | shade         |      -0.00225106 | -0.00225106 |  0.0167226  |                       1 |     -1 | False              |
| M_R2_spearman_G_SP | suite             | bbob          |      -0.342004   | -0.589208   | -0.0958522  |                     312 |     -1 | True               |
| M_R2_spearman_G_SP | suite             | mabbob        |      -0.220352   | -0.54716    | -0.0247361  |                      90 |     -1 | True               |
| M_R2_spearman_G_SP | current_algorithm | cso           |      -0.427682   | -0.702819   | -0.139636   |                     202 |     -1 | True               |
| M_R2_spearman_G_SP | current_algorithm | lshade        |      -0.10962    | -0.300621   |  0.0235631  |                     124 |     -1 | False              |
| M_R2_spearman_G_SP | current_algorithm | shade         |      -0.128689   | -0.319304   |  0.00446539 |                      76 |     -1 | False              |
