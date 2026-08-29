# 16a · Task 14A 协议与成本（Protocol & Cost）

- 日期：2026-08-29。前置门：Task 13.1-H hygiene = **H1 NEGLIGIBLE** ∈ {H1,H2} → GO 确认。
- Portfolio 预先固定 $\mathcal A=\{\text{SHADE},\text{L-SHADE},\text{CSO}\}$；6 方向全部保留；问题域 = Task 12 Stage-2 development set（BBOB 30 + MA 12，10D，instances 不变）；Stage A seeds 1–5（**seeds 6–10 未触碰**，保持 confirmation status）。

## 1. 协议要点

1. **Source checkpoint**：每个 (problem, seed, source) 自然轨迹 0→6000 FE（正式全局 window recorder），在 FE∈{2000,4000,6000} 克隆；
2. **真实 handoff**：population-only transfer（`initialize_transferred_optimizer_state`，语义 RNG 事件）；强制 1000 FE commitment（期内无决策）→ mature post-handoff state（全局 FE = t+1000，segment_start=t，segment_age=1000）；
3. **checkpoint 保存**：best/population/fitness/target 内部状态/RNG/实际 NP（L-SHADE 缩减后 NP 如 36/26/33 被完整保留）+ 全局与 segment 两个 recorder；
4. **Next-action fork**：{continue B, switch A, switch C} × 1000 FE（主 horizon；terminal 不跑）；
5. **Repetition**：post-handoff state-action 对的 10%（outcome-blind SeedSequence 抽样）× R=3；
6. **Reset controls（必做）**：current=SHADE / L-SHADE 的 population-preserving reset（保留 population/fitness/best/evaluations/缩减 schedule；重置 success-history、archive、adaptive memory；fresh 语义 RNG 事件；禁止恢复 NP=40 或重启缩减 schedule——`dataclasses.replace` 逐字段实现，schedule 字段原样保留）；
7. **Behavior**：B_global（全局 recorder，0→t+1000 连续累积）与 B_segment（segment recorder 于 handoff 重建，segment 相对窗口 w02/w05/w10 = 20/50/100 FE）各 3780 条正式提取（本轮只记录不训练）。

## 2. FE 账本（分类列支，reset 不混入普通 branch）

| 阶段 | FE |
|---|---:|
| source natural（630 × 6000） | 3,780,000 |
| handoff commitment（3780 × 1000） | 3,780,000 |
| next-action branch（13,624 × 1000） | 13,624,000 |
| repetition（含于 branch 行，2,284 额外 replicate） | （已计入 branch） |
| reset control（2,520 × 1000） | 2,520,000 |
| **总计** | **23,704,000** |

wall time ≈195 s（8 workers）；峰值 RSS ≈218 MB（`task14a_collection_ledger.parquet`）。

## 3. Commitment 后 attained gap

| source FE | 2000 | 4000 | 6000 |
|---|---:|---:|---:|
| mean log10 gap after commitment | −0.726 | −2.323 | −3.612 |

按 route：lshade→shade 最低（−2.880），shade→cso 最高（−1.737）——早期 handoff 的 target 尚未成熟，符合预期。
