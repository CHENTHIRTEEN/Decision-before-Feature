# 16f · 吸收态审计（Task 14A Absorbing-State Audit）

- 日期：2026-08-29。定义：对每个 current $B$，$P(B\in A_{ND}\mid current=B)>0.90$ 且 route+FE 之外残差 $|L_{P3}-L_{P4}|<0.05$ 时标记 ABSORBING-STATE RISK（描述性阈值，不自动判死）。产物：`post_handoff_absorbing_audit.parquet`。

## 1. 按 current

| current | P(current∈A_ND) | switch-required | L_P3 | L_P4 statewise | residual | flag |
|---|---:|---:|---:|---:|---:|---|
| lshade | 0.880 | 0.120 | −3.5187 | −3.6120 | 0.093 | False |
| shade | 0.825 | 0.175 | −3.7873 | −3.9035 | 0.116 | False |
| cso | 0.768 | 0.232 | −3.6643 | −3.7723 | 0.108 | False |

## 2. 按 route（最收敛的三条）

| route | P(current∈A_ND) | switch-required | residual |
|---|---:|---:|---:|
| cso→lshade | 0.898 | 0.102 | ≈0.088 |
| lshade→shade | 0.789 | 0.211 | 0.137 |
| cso→shade | 0.860 | 0.140 | 0.096 |

## 3. 结论

- **无任何 current/route 触发吸收态标记**：最收敛的 lshade-current（0.880）与 cso→lshade route（0.898）都低于 0.90 阈值，且全部残差 ≥0.088（为 post-handoff 噪声 δ95 的 1–9 倍、fb CI 显著非零）；
- RQ2 的回答：**不存在 $P(\text{continue }B)\approx 1$ 的吸收态**；最收敛处仍有 10% 的状态需要切换、残差空间完整；
- 监控建议：14B 若发现 lshade-current 或 cso→lshade 的比例进一步上升，应重跑本审计（阈值 0.90/0.05 保持预注册语义）。
