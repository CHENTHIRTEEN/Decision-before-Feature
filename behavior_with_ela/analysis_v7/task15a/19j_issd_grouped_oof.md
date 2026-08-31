# ISSD grouped OOF

| model                | carrier   | suite   |   realized_fb_loss |   gain_vs_continue_fb |   switch_rate |    n |
|:---------------------|:----------|:--------|-------------------:|----------------------:|--------------:|-----:|
| M0_context           | rf        | bbob    |           -1.98352 |            0.0143456  |      0.216667 | 2700 |
| M0_context           | rf        | mabbob  |           -5.04312 |            0.0108789  |      0.217593 | 1080 |
| M0_context           | ridge     | bbob    |           -1.9555  |           -0.0136703  |      0.194444 | 2700 |
| M0_context           | ridge     | mabbob  |           -5.03408 |            0.00183997 |      0.185185 | 1080 |
| M_ISSD               | rf        | bbob    |           -1.95614 |           -0.0130265  |      0.514074 | 2700 |
| M_ISSD               | rf        | mabbob  |           -5.02631 |           -0.00593083 |      0.40463  | 1080 |
| M_ISSD               | ridge     | bbob    |           -1.95667 |           -0.0125036  |      0.513333 | 2700 |
| M_ISSD               | ridge     | mabbob  |           -4.99661 |           -0.0356327  |      0.488889 | 1080 |
| M_combined           | rf        | bbob    |           -1.96316 |           -0.00601061 |      0.538148 | 2700 |
| M_combined           | rf        | mabbob  |           -5.02699 |           -0.00524536 |      0.505556 | 1080 |
| M_combined           | ridge     | bbob    |           -1.96362 |           -0.00554852 |      0.50037  | 2700 |
| M_combined           | ridge     | mabbob  |           -5.0007  |           -0.0315392  |      0.493519 | 1080 |
| M_full_legacy        | rf        | bbob    |           -1.95749 |           -0.0116815  |      0.622593 | 2700 |
| M_full_legacy        | rf        | mabbob  |           -5.02358 |           -0.00866523 |      0.60463  | 1080 |
| M_full_legacy        | ridge     | bbob    |           -1.93079 |           -0.0383799  |      0.630741 | 2700 |
| M_full_legacy        | ridge     | mabbob  |           -4.99897 |           -0.0332665  |      0.656481 | 1080 |
| M_lookup             | lookup    | bbob    |           -1.98352 |            0.0143456  |      0.216667 | 2700 |
| M_lookup             | lookup    | mabbob  |           -5.04312 |            0.0108789  |      0.217593 | 1080 |
| M_screened_aggregate | rf        | bbob    |           -1.96294 |           -0.00622817 |      0.644444 | 2700 |
| M_screened_aggregate | rf        | mabbob  |           -5.01572 |           -0.0165203  |      0.575    | 1080 |
| M_screened_aggregate | ridge     | bbob    |           -1.94359 |           -0.025583   |      0.611481 | 2700 |
| M_screened_aggregate | ridge     | mabbob  |           -5.00025 |           -0.0319856  |      0.628704 | 1080 |
| Always Continue      | fixed     | bbob    |           -1.96917 |            0          |      0        | 2700 |
| Always Continue      | fixed     | mabbob  |           -5.03224 |            0          |      0        | 1080 |

| suite   | carrier   | upper                | lower      |         gain |      ci_low |      ci_high |
|:--------|:----------|:---------------------|:-----------|-------------:|------------:|-------------:|
| bbob    | rf        | M0_context           | M_ISSD     | -0.027372    | -0.0587152  | -0.000499985 |
| bbob    | rf        | M_screened_aggregate | M_ISSD     | -0.0067983   | -0.0406087  |  0.017707    |
| bbob    | rf        | M_screened_aggregate | M_combined |  0.000217568 | -0.0301533  |  0.0236889   |
| bbob    | ridge     | M0_context           | M_ISSD     |  0.00116671  | -0.0462578  |  0.0696913   |
| bbob    | ridge     | M_screened_aggregate | M_ISSD     |  0.0130794   | -0.00529241 |  0.0351238   |
| bbob    | ridge     | M_screened_aggregate | M_combined |  0.0200344   |  0.00290819 |  0.0382123   |
| mabbob  | rf        | M0_context           | M_ISSD     | -0.0168097   | -0.0454486  |  0.0197566   |
| mabbob  | rf        | M_screened_aggregate | M_ISSD     |  0.0105895   | -0.0199335  |  0.0390218   |
| mabbob  | rf        | M_screened_aggregate | M_combined |  0.0112749   | -0.0136835  |  0.0358368   |
| mabbob  | ridge     | M0_context           | M_ISSD     | -0.0374726   | -0.0580698  | -0.0162298   |
| mabbob  | ridge     | M_screened_aggregate | M_ISSD     | -0.00364707  | -0.0224133  |  0.0108868   |
| mabbob  | ridge     | M_screened_aggregate | M_combined |  0.000446353 | -0.020274   |  0.0158876   |
