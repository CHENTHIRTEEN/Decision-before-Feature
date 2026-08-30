# 18f · Scale-matched Segment protocol

使用相同 `WINDOW_RATIOS={0.02,0.05,0.10}` 和 global `FE_total=10000`，严格得到 200/500/1000 FE。3780/3780 states 从 Task 14A source checkpoint 确定性重放至同一 1000-FE mature endpoint；最长窗口 anchor 为 handoff point，未使用 pre-handoff 信息，新增 action-label FE=0。

replay endpoint 的 best/log gap 最大绝对差为 `0.000e+00`，低于 1e-12。
