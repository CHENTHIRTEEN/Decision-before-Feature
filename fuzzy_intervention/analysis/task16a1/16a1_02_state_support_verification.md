# 16A.1-02 State support verification

> Task16A.1 为零 FE 的事后诊断性复核，不替代 Task16A 预先指定的正式结论。

24 个 suite×solver×FE 基础 cell 中，R1/R2 各至少 5 states 的有效 cell 为 9。

| suite   | current_algorithm   |   source_FE |   num_states |   n_R1 |   n_R2 |   n_R3 |   n_R4 | support_R1_R2   | support_R3_R4   |
|:--------|:--------------------|------------:|-------------:|-------:|-------:|-------:|-------:|:----------------|:----------------|
| bbob    | cso                 |        2000 |          150 |     31 |      6 |      0 |      3 | SUPPORTED       | LOW_SUPPORT     |
| bbob    | cso                 |        4000 |          150 |     16 |     36 |      0 |     25 | SUPPORTED       | LOW_SUPPORT     |
| bbob    | cso                 |        6000 |          150 |     13 |     58 |      0 |     42 | SUPPORTED       | LOW_SUPPORT     |
| bbob    | cso                 |        8000 |          150 |      8 |     59 |      3 |     27 | SUPPORTED       | LOW_SUPPORT     |
| bbob    | lshade              |        2000 |          150 |     17 |      0 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| bbob    | lshade              |        4000 |          150 |      1 |      1 |      1 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| bbob    | lshade              |        6000 |          150 |      1 |     34 |      5 |     17 | LOW_SUPPORT     | SUPPORTED       |
| bbob    | lshade              |        8000 |          150 |      5 |     55 |     11 |     35 | SUPPORTED       | SUPPORTED       |
| bbob    | shade               |        2000 |          150 |     21 |      1 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| bbob    | shade               |        4000 |          150 |      3 |      1 |      1 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| bbob    | shade               |        6000 |          150 |      2 |     16 |      5 |      8 | LOW_SUPPORT     | SUPPORTED       |
| bbob    | shade               |        8000 |          150 |      5 |     45 |      7 |     12 | SUPPORTED       | SUPPORTED       |
| mabbob  | cso                 |        2000 |           60 |     23 |      2 |      0 |      1 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | cso                 |        4000 |           60 |     14 |      6 |      2 |      2 | SUPPORTED       | LOW_SUPPORT     |
| mabbob  | cso                 |        6000 |           60 |      9 |     16 |      2 |      4 | SUPPORTED       | LOW_SUPPORT     |
| mabbob  | cso                 |        8000 |           60 |      7 |     19 |      9 |      4 | SUPPORTED       | LOW_SUPPORT     |
| mabbob  | lshade              |        2000 |           60 |     20 |      0 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | lshade              |        4000 |           60 |      3 |      2 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | lshade              |        6000 |           60 |      1 |      8 |      1 |      5 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | lshade              |        8000 |           60 |      1 |     24 |      6 |     14 | LOW_SUPPORT     | SUPPORTED       |
| mabbob  | shade               |        2000 |           60 |     26 |      0 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | shade               |        4000 |           60 |      4 |      1 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | shade               |        6000 |           60 |      1 |      5 |      1 |      1 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | shade               |        8000 |           60 |      0 |      7 |      0 |      7 | LOW_SUPPORT     | LOW_SUPPORT     |
