# 16g · SHADE / L-SHADE Reset Controls（Task 14A 核心混杂审计）

- 日期：2026-08-29。设计：每个 current=SHADE / L-SHADE 的 post-handoff checkpoint 上同轮执行 R0 native continue、R1 population-preserving reset、R2 switch（fork 已含）。$G_{reset}=L_{native}-L_{reset}$，$G_{switch}=L_{native}-L_{switch}$，$\Delta_{solver\text{-}specific}=L_{reset}-L_{switch}$（>0 ⇒ 换 solver 优于原地重置记忆）。配对 cv_group bootstrap 5000 draws。产物：`reset_control_outcomes.parquet`、`reset_effect_summary.parquet`。

## 1. SHADE-current（reset-SHADE vs switch-L-SHADE）

| suite | G_reset | 95% CI | G_switch | 95% CI | **Δ solver-specific** | 95% CI |
|---|---:|---|---:|---|---:|---|
| pooled | −0.0437 | [−0.0855, +0.0009] | −0.0189 | [−0.0688, +0.0307] | **+0.0248** | [+0.0076, +0.0414] |
| bbob | −0.0678 | [−0.1260, −0.0134] | −0.0412 | [−0.1137, +0.0296] | +0.0267 | [+0.0028, +0.0536] |
| ma | −0.0237 | [−0.0778, +0.0418] | −0.0004 | [−0.0632, +0.0717] | +0.0233 | [+0.0021, +0.0443] |

## 2. L-SHADE-current（reset-L-SHADE vs switch-SHADE）

| suite | G_reset | 95% CI | G_switch | 95% CI | Δ solver-specific | 95% CI |
|---|---:|---|---:|---|---:|---|
| pooled | −0.0885 | [−0.1498, −0.0325] | −0.1046 | [−0.1633, −0.0536] | −0.0161 | [−0.0337, +0.0042] |
| bbob | −0.1030 | [−0.2141, −0.0033] | −0.1200 | [−0.2272, −0.0240] | −0.0169 | [−0.0277, −0.0076] |
| ma | −0.0763 | [−0.1350, −0.0201] | −0.0918 | [−0.1494, −0.0416] | −0.0154 | [−0.0462, +0.0197] |

## 3. 判读（RQ3）

1. **成熟 post-handoff 状态上"切换收益"本身消失**：G_switch 点估计 pooled 为 −0.019（SHADE-current）/ −0.105（L-SHADE-current）——continue 优于两个替代动作；自然状态上 shade→lshade 的正切换增益**不会延续到成熟换挡后状态**；
2. **reset 是有害操作**：G_reset 全部为负（重置 1000-FE 成熟的记忆/Archive 使 SHADE 差 ~0.04、L-SHADE 差 ~0.09）——success-history 在成熟段承载了真实信息；
3. **reset 无法解释切换收益**：Δ_solver-specific（SHADE-current）= +0.025 CI>0——即使决定离开 SHADE，换到 L-SHADE 也优于原地重置记忆；L-SHADE-current 上 reset 与 switch 无显著差。因此 natural 状态观察到的切换增益**不可能是 reset 伪影**（reset 从不产生正增益）；
4. **schedule 混杂排除**：reset 分支逐行保留 `reduction_max_fe`/`max_evaluations`/当前 NP（population_size_preserved 列），未重启缩减 schedule；
5. CSO：无 SHADE 式历史记忆，按工作单不做 pseudo-reset（本轮未发现其长期内部状态问题）。

$$
\boxed{\text{Reset-Confound Verdict：RC1 SOLVER-SPECIFIC EFFECT}}
$$

（附注：post-handoff 域内更准确的表述是——continue 占优、reset 有害、solver 间差异存在但方向依 current 而定；不得把自然状态的方向性互补直接外推到成熟换挡后状态。）
