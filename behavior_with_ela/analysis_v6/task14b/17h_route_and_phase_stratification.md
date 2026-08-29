# 17h · Route 与 Source-FE 分层（Task 14B Route Stratification）

- 日期：2026-08-29。RF 载体、真实 outcome；M0/MG/MS/MGS 分层 fb loss 与 segment 增量。产物：`route_phase_stratification.parquet`。**约束：不得据此删方向。**

## 1. 按 6 方向

| route | switch-required | L_M0 | MG gain (M0−MG) | MS gain | MGS gain | segment increment (MG−MGS) | harmful(MGS) | switch(MGS) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shade→lshade | 0.138 | −3.8223 | −0.0281 | −0.0443 | −0.0367 | +0.0076 | 0.152 | 0.556 |
| shade→cso | 0.148 | −3.2764 | −0.0332 | −0.0480 | −0.0399 | +0.0067 | 0.181 | 0.656 |
| lshade→shade | 0.211 | −4.4259 | −0.0229 | −0.0284 | −0.0297 | −0.0068 | 0.171 | 0.714 |
| lshade→cso | 0.316 | −4.0522 | −0.0270 | −0.0293 | −0.0271 | +0.0001 | 0.135 | 0.635 |
| cso→shade | 0.140 | −3.1486 | −0.0277 | −0.0359 | −0.0293 | +0.0016 | 0.156 | 0.675 |
| cso→lshade | 0.102 | −3.2151 | −0.0300 | −0.0395 | −0.0354 | −0.0054 | 0.124 | 0.594 |

（MG gain 为正 = MG 比 M0 更差；全方向 MG/MS/MGS 均无正增益。）

## 2. 判读

- **六个方向全部无 Behavior 正增益**：MG gain 全部为 +0.023～+0.033（负向），segment increment 在 −0.007～+0.008 之间无方向一致的正值——A3 NO-GO 不是个别 route 拉低的结果；
- lshade→cso（Task 14A 中最动态的 route，switch-required 0.316）增量同样 ≈0；
- 最收敛的 cso→lshade（0.102）harmful 最低（0.124）——route 差异主要体现在切换基率上，而非 Behavior 可提取性上。

## 3. 按 source FE（phase）

| source FE | switch-required | L_M0 | MGS gain | segment increment |
|---:|---:|---:|---:|---:|
| 2000 | 0.206 | −2.5413 | −0.0244 | −0.0064 |
| 4000 | 0.168 | −3.7645 | −0.0436 | −0.0081 |
| 6000 | 0.153 | −4.6645 | −0.0406 | +0.0006 |

**Segment Behavior 的价值没有集中在早期 post-handoff 状态**（2000 处增量亦为负）——工作单 §30 要求的"如只在某阶段成立须如实报告"在此不适用：任何阶段都不成立。
