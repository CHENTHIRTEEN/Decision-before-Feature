# Decision-before-Feature Task16A.1：Zero-FE Joint Regime Verification

> Task16A.1 为零 FE 的事后诊断性复核，不替代 Task16A 预先指定的正式结论。

## Regime / maturity

1. R1/R2 cell counts：

| suite   | current_algorithm   |   source_FE |   n_R1 |   n_R2 |   n_R3 |   n_R4 | support_R1_R2   | support_R3_R4   |
|:--------|:--------------------|------------:|-------:|-------:|-------:|-------:|:----------------|:----------------|
| bbob    | cso                 |        2000 |     31 |      6 |      0 |      3 | SUPPORTED       | LOW_SUPPORT     |
| bbob    | cso                 |        4000 |     16 |     36 |      0 |     25 | SUPPORTED       | LOW_SUPPORT     |
| bbob    | cso                 |        6000 |     13 |     58 |      0 |     42 | SUPPORTED       | LOW_SUPPORT     |
| bbob    | cso                 |        8000 |      8 |     59 |      3 |     27 | SUPPORTED       | LOW_SUPPORT     |
| bbob    | lshade              |        2000 |     17 |      0 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| bbob    | lshade              |        4000 |      1 |      1 |      1 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| bbob    | lshade              |        6000 |      1 |     34 |      5 |     17 | LOW_SUPPORT     | SUPPORTED       |
| bbob    | lshade              |        8000 |      5 |     55 |     11 |     35 | SUPPORTED       | SUPPORTED       |
| bbob    | shade               |        2000 |     21 |      1 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| bbob    | shade               |        4000 |      3 |      1 |      1 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| bbob    | shade               |        6000 |      2 |     16 |      5 |      8 | LOW_SUPPORT     | SUPPORTED       |
| bbob    | shade               |        8000 |      5 |     45 |      7 |     12 | SUPPORTED       | SUPPORTED       |
| mabbob  | cso                 |        2000 |     23 |      2 |      0 |      1 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | cso                 |        4000 |     14 |      6 |      2 |      2 | SUPPORTED       | LOW_SUPPORT     |
| mabbob  | cso                 |        6000 |      9 |     16 |      2 |      4 | SUPPORTED       | LOW_SUPPORT     |
| mabbob  | cso                 |        8000 |      7 |     19 |      9 |      4 | SUPPORTED       | LOW_SUPPORT     |
| mabbob  | lshade              |        2000 |     20 |      0 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | lshade              |        4000 |      3 |      2 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | lshade              |        6000 |      1 |      8 |      1 |      5 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | lshade              |        8000 |      1 |     24 |      6 |     14 | LOW_SUPPORT     | SUPPORTED       |
| mabbob  | shade               |        2000 |     26 |      0 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | shade               |        4000 |      4 |      1 |      0 |      0 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | shade               |        6000 |      1 |      5 |      1 |      1 | LOW_SUPPORT     | LOW_SUPPORT     |
| mabbob  | shade               |        8000 |      0 |      7 |      0 |      7 | LOW_SUPPORT     | LOW_SUPPORT     |

2. 有效 R1/R2 cells：9/24。
3. Raw pooled R2-R1：-0.0908。
4. Standardized R2-R1：-0.0014。
5. BBOB standardized CI：-0.0293 [-0.1057, 0.0381]。
6. MA-BBOB standardized CI：0.0544 [-0.0833, 0.2625]。
7. Solver standardized differences：SHADE=-0.2092，L-SHADE=0.0000，CSO=0.0281。
8. Part A verdict：A-NULL。

## Continuous advantage

9. G_P median：R1=-0.0066，R2=0.0000。
10. G_S median：R1=0.0156，R2=0.0000。
11. G_I standardized R2-R1：-0.0331。
12. G_I CI：[-0.1200, -0.0072]。
13. R3-R4 G_P：0.0290。
14. R3-R4 G_S：-0.0076。
15. 跨 suite 稳定：NO。
16. 跨 solver 稳定：NO。
17. Part B verdict：B-NONE。

## Noise

18. Global delta95：

| action           |   delta95 |   n_repeated_pairs |
|:-----------------|----------:|-------------------:|
| continue         |  0.561663 |                247 |
| perturb_random   |  0.569991 |                255 |
| perturb_targeted |  0.50722  |                210 |
| switch_cso       |  0.509594 |                179 |
| switch_lshade    |  0.502872 |                184 |
| switch_shade     |  0.345392 |                159 |

19. Suite/current/FE/current×FE ranges：

| action           | conditioning   |   supported_cells |   delta95_min |   delta95_max |   R_delta |   coefficient_of_variation |
|:-----------------|:---------------|------------------:|--------------:|--------------:|----------:|---------------------------:|
| continue         | FE             |                 4 |    0.309917   |      0.638596 |   2.06054 |                  0.240385  |
| continue         | current        |                 3 |    0.155081   |      0.645154 |   4.16012 |                  0.471713  |
| continue         | current_FE     |                11 |    0.00188562 |      1.33505  | 708.015   |                  0.766194  |
| continue         | suite          |                 2 |    0.466271   |      0.561663 |   1.20458 |                  0.0927991 |
| perturb_random   | FE             |                 4 |    0.410775   |      0.631403 |   1.5371  |                  0.193914  |
| perturb_random   | current        |                 3 |    0.458851   |      0.595422 |   1.29764 |                  0.114603  |
| perturb_random   | current_FE     |                11 |    0.0906668  |      1.90955  |  21.0611  |                  0.879069  |
| perturb_random   | suite          |                 2 |    0.561834   |      0.580342 |   1.03294 |                  0.0162047 |
| perturb_targeted | FE             |                 4 |    0.413972   |      0.59381  |   1.43442 |                  0.129777  |
| perturb_targeted | current        |                 3 |    0.323467   |      0.586651 |   1.81363 |                  0.233061  |
| perturb_targeted | current_FE     |                11 |    0.162793   |      0.649428 |   3.9893  |                  0.311017  |
| perturb_targeted | suite          |                 2 |    0.491574   |      0.535255 |   1.08886 |                  0.0425394 |
| switch_cso       | FE             |                 4 |    0.306499   |      0.801037 |   2.61351 |                  0.417814  |
| switch_cso       | current        |                 2 |    0.360644   |      0.75537  |   2.09451 |                  0.353693  |
| switch_cso       | current_FE     |                 8 |    0.231612   |      1.24861  |   5.39097 |                  0.65449   |
| switch_cso       | suite          |                 2 |    0.504775   |      0.733882 |   1.45388 |                  0.184964  |
| switch_lshade    | FE             |                 4 |    0.419194   |      0.501431 |   1.19618 |                  0.0688302 |
| switch_lshade    | current        |                 2 |    0.205243   |      0.634644 |   3.09216 |                  0.511261  |
| switch_lshade    | current_FE     |                 8 |    0.0135317  |      0.677328 |  50.0548  |                  0.610071  |
| switch_lshade    | suite          |                 2 |    0.28781    |      0.537072 |   1.86606 |                  0.302179  |
| switch_shade     | FE             |                 4 |    0.123963   |      0.477687 |   3.85346 |                  0.402993  |
| switch_shade     | current        |                 2 |    0.301568   |      0.389784 |   1.29252 |                  0.127599  |
| switch_shade     | current_FE     |                 8 |    0.0628366  |      0.55377  |   8.81286 |                  0.542044  |
| switch_shade     | suite          |                 2 |    0.33667    |      0.50647  |   1.50435 |                  0.20139   |

20. Cell repeated-pair support：见 `task16a1_noise_cells.parquet`。
21. R_delta：按 action 与 conditioning 完整报告。
22. Noise verdict：N1 STRONG HETEROGENEITY。
23. Local-threshold sensitivity：已构造，仅作敏感性分析；raw pooled 与两个 suite 的 R2-R1 方向均未改变；低支持 noise cell 的阈值未被使用。
24. 低支持 cell 是否影响 local 解释：低支持 cell-specific threshold 从未进入 sensitivity；仅使用 current×FE、current 或 global 中支持充分的估计。

## Maturity

25. R2 四个 maturity 的 G_SP median：M=0.2: 0.0165 (n=9)；M=0.4: 0.0532 (n=47)；M=0.6: 0.0000 (n=137)；M=0.8: 0.0000 (n=209)。
26. R3：M=0.2: NA (n=0)；M=0.4: 0.3556 (n=4)；M=0.6: 0.0000 (n=14)；M=0.8: 0.0000 (n=36)。
27. R4：M=0.2: 0.0432 (n=4)；M=0.4: 1.5508 (n=27)；M=0.6: 0.0000 (n=77)；M=0.8: 0.0000 (n=99)。
28. Pooled R2 Spearman：-0.3118 [-0.5138, -0.1174]。
29. BBOB/MA：BBOB=-0.3420，MA-BBOB=-0.2204。
30. Solver：SHADE=-0.1287，L-SHADE=-0.1096，CSO=-0.4277。
31. Maturity verdict：M1 STRUCTURED。

## Final

32. Task16A primary F3 是否保持：YES。
33. 当前 Perturb A3 是否保持：YES。
34. Task16A.1 verdict：J2 WEAK / AMBIGUOUS STRUCTURE。
35. 是否允许直接运行 Type-1：NO。
36. 是否允许 Interval Type-2：NO。
37. 若为 J1，是否只能先设计 Task16A.2：YES。
38. 若为 J3，search-regime intervention line 是否停止：YES。
39. new objective FE 是否为 0：YES。

## 资源

- reused_task16a_states：2520；
- reused_action_outcomes：15068；
- reused_repetition_rows：3702；
- analysis_cpu_seconds：29.522；
- wall_seconds：30.907；
- peak_rss_mb：232.266。

## 解释边界

该结论只描述 Task16A 既有开发数据中的事后诊断结构，不改变 Task16A 的正式结论，也不评价任何模糊控制器。
