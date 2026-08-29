# 15g · Transition 风险分层（Task 13.1N）

- 日期：2026-08-29。对象：raw M2（κ=0）实际执行的切换，按 6 个方向分层。产物：`transition_risk_stratification.parquet`。**工作单约束：本分层只作 Task 14A 诊断输入，不得据此删除任何方向；Task 14A 若执行仍保持预注册 6 方向。**

## 1. 六方向风险表（raw M2 实际切换）

| suite | transition | count | mean realized gain | harmful rate | switch-required precision |
|---|---|---:|---:|---:|---:|
| BBOB | cso → lshade | 312 | −0.057 | 0.135 | 0.269 |
| BBOB | cso → shade | 109 | **+0.114** | 0.037 | 0.284 |
| BBOB | shade → lshade | 148 | **−0.203** | **0.304** | 0.250 |
| BBOB | shade → cso | 96 | +0.048 | 0.292 | 0.333 |
| BBOB | lshade → shade | 90 | −0.128 | 0.322 | 0.222 |
| BBOB | lshade → cso | 112 | **+0.440** | 0.313 | **0.446** |
| MA | cso → lshade | 146 | +0.073 | 0.151 | 0.274 |
| MA | cso → shade | 24 | +0.043 | 0.000 | 0.208 |
| MA | shade → lshade | 57 | −0.101 | **0.509** | 0.211 |
| MA | shade → cso | 35 | +0.214 | 0.314 | **0.514** |
| MA | lshade → shade | 19 | +0.015 | 0.105 | 0.211 |
| MA | lshade → cso | 38 | **+0.281** | 0.263 | **0.526** |

## 2. 风险集中模式

1. **harmful 集中在 shade-current 的两个方向**：shade→lshade（0.30/0.51）与 shade→cso（0.29/0.31）——SHADE 轨迹上模型预测的切换在约三至五成情况下实际劣于 continue；bbob lshade→shade（0.32）同属高风险组；
2. **高价值方向是 →cso 与 cso→shade**：lshade→cso 增益 +0.44/+0.28 且 switch-required precision 最高（0.45/0.53）；cso→shade 增益为正且 harmful 最低（0.04/0.00）；
3. cso→lshade 是量最大的方向（312/146 次）但 bbob 上期望增益为负（−0.057）——这是 raw M2 harmful mass 的重要来源，也是 margin 阈值能显著改善整体风险的原因之一（15c/15d）。

## 3. 对 Task 14A 的诊断含义（不改变预注册）

- 6 方向保留；但 14A 的 post-handoff action-space 审计应**分方向报告**，重点核对 shade-current 两方向是否在真实 handoff 后仍然表现出高 harmful（若 handoff 后消失，说明部分风险是 natural-state 特有的预测伪影；若保持，则是真实的转移成本）；
- margin 阈值与 per-direction 风险不匹配的问题（cso→lshade 的负增益）留待 Task 14B 的 policy 设计，本轮不做方向级阈值。
