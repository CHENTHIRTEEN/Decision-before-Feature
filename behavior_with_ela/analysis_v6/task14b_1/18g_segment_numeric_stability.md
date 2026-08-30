# 18g · Segment numeric stability

 representation  n_rows  n_feature_cells  raw_max_absolute_value  raw_abs_gt_1e6_cells  raw_abs_gt_1e4_cells  raw_nonzero_abs_lt_1e12_cells  post_clip_max_absolute_value  post_clip_abs_gt_1e6_cells  post_clip_abs_gt_1e4_cells
    segment_old    3780           105840            6.690430e+13                   527                  1345                            292                     1000000.0                           0                        1345
segment_matched    3780           105840            4.294629e+13                   656                  1463                            551                     1000000.0                           0                        1463

Matched 沿用既有 `|v|>1e6` clip 与 `0<|v|<1e-12` 归零规则。
