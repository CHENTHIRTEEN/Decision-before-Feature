# 16d · Post-Handoff Next-Action Space（Task 14A 主分析）

- 日期：2026-08-29。数据：3780 mature post-handoff states × {continue, switch×2} × 1000-FE 真实 outcome。产物：`post_handoff_practical_action_sets.parquet`、`post_handoff_dcm.parquet`、`post_handoff_oracle_headroom.parquet`、`post_handoff_strata.parquet`。

## 1. Practical 动作空间（三口径）

| rule | suite | switch-required | P(current∈A_ND) | unique winner | P(空集) | E\|A_ND\| |
|---|---|---:|---:|---:|---:|---:|
| max（主） | pooled | **0.2243** | 0.7757 | 0.4529 | 0 | 1.91 |
| max | bbob | 0.2300 | 0.7700 | 0.4504 | 0 | 1.94 |
| max | ma | 0.2102 | 0.7898 | 0.4593 | 0 | 1.82 |
| quadrature | pooled | 0.2061 | 0.7939 | 0.4063 | 0 | 1.98 |
| sum | pooled | **0.1757** | 0.8243 | 0.3418 | 0 | 2.08 |

**switch-required 在最保守 sum 口径下仍 ≥0.157**——非退化。

## 2. Pairwise DCM（max 口径，全部双向）

| pair | P(tie) | P(a≻b) | P(b≻a) | DCM |
|---|---:|---:|---:|---:|
| shade↔lshade | 0.592 | 0.169 | 0.239 | 0.331 |
| shade↔cso | 0.442 | 0.394 | 0.164 | 0.336 |
| lshade↔cso | 0.445 | 0.402 | 0.153 | 0.347 |

## 3. Oracle Ladder（P0–P4，fb log10 loss）

| suite | P0 Continue | P1 Current-only | P2 Route | P3 Route+srcFE | P4 Statewise | **Δ_post = P3−P4** |
|---|---:|---:|---:|---:|---:|---:|
| pooled | −3.6399 | −3.6399 | −3.6454 | −3.6568 | −3.7626 | **+0.1058** |
| bbob | −1.9692 | −1.9692 | −1.9750 | −1.9884 | −2.0930 | +0.1046 |
| ma | −5.0322 | −5.0322 | −5.0373 | −5.0471 | −5.1539 | +0.1068 |

- P1=P0：**current-only 经验策略退化为 Always Continue**（每 current 的 fb 最优动作都是 continue）；
- route 只再吃 0.005，route+source-FE 只再吃 0.011——**粗粒度 handoff history 几乎不解释 next-action**；
- **Δ_post ≈ +0.105（两 suite 一致，量级与 natural Δ_dynamic 0.114/0.085 相当）**：状态级 next-action 空间完整保留。

## 4. 分层

| stratum | switch-required | 说明 |
|---|---:|---|
| route cso→lshade | **0.102** | 最接近吸收（P∈A_ND=0.898，未越 0.90 阈） |
| route lshade→cso | **0.316** | 最动态 |
| route 其余 | 0.138–0.211 | — |
| current lshade / shade / cso | 0.120 / 0.175 / 0.232 | lshade-current 最收敛 |
| source FE 2000/4000/6000 | 0.206 / 0.168 / 0.153 | 越晚越趋向 continue |
