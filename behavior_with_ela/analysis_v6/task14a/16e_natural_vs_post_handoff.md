# 16e · Natural vs Post-Handoff 对照 与 16f · 吸收态审计

- 日期：2026-08-29。产物：`natural_vs_post_handoff_metrics.parquet`、`post_handoff_absorbing_audit.parquet`。

## 1. Dynamic complementarity 是否 survives handoff？

| 量（max 口径） | natural（Task 12.1/13.1） | post-handoff（本轮） | 结论 |
|---|---:|---:|---|
| switch-required（bbob / ma） | 0.258 / 0.265 | 0.230 / 0.210 | 轻度下降，**仍明显非零** |
| sum 口径 switch-required | 0.187 / 0.211 | 0.183 / 0.157 | 仍非零 |
| DCM 范围 | 0.254–0.370（全双向） | 0.331–0.347（全双向） | **保持双向**，离散度收窄 |
| 状态级残差 | Δ_dynamic 0.114/0.085（vs problem+FE） | Δ_post 0.105/0.107（vs route+srcFE） | **量级相当** |

结论：**complementarity survives handoff**——成熟 post-handoff 状态的 next-action 空间没有坍塌；上下文（route + source FE）只解释残差的约 15%。

## 2. 吸收态审计（按 current）

| current | P(current∈A_ND) | switch-required | residual（P3−P4） | absorbing flag（>0.90 且 \|residual\|<0.05） |
|---|---:|---:|---:|---|
| lshade | **0.880** | 0.120 | 0.093 | False |
| shade | 0.825 | 0.175 | 0.116 | False |
| cso | 0.768 | 0.232 | 0.108 | False |

- 无 current 触发 0.90 吸收阈值；lshade 最接近（0.880）但残差 0.093 远大于噪声；
- 分 route：cso→lshade 的 P=0.898 亦未越阈，且残差 ≈0.088；
- source FE 越晚越收敛（0.206→0.153）——晚期换挡后空间更"安静"，但到 6000 仍有 0.153。

**ABSORBING-STATE RISK：未触发。**
