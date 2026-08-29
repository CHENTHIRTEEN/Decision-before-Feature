# 15j · Fold-Local Margin Policy 复跑与 pooled 对照（Task 13.1-H/H3–H6）

- 日期：2026-08-29。κ 网格严格沿用预注册 {0, 0.5, 1.0, 1.5, 2.0}；margins 沿用已提交的 RF-M2 OOF（A1 已逐位复现）；harmful 判据仍为 state-aware pairwise max δ。产物：`fold_local_margin_policy_rows.parquet`、`fold_local_margin_policy_summary.parquet`、`fold_local_vs_pooled_bootstrap.parquet`。

## 1. pooled 结果复现（停止条件第 2 步）

fold 表中 κ=0 与全部 pooled 列重建后，20 个 (scale, κ, suite) 的 fb loss 与已提交 `margin_policy_summary.parquet` 全部一致（|diff| ≤ 1e-9）——复现通过。

## 2. fold-local 主表（max scale）

| κ | suite | fb loss | gain vs Continue | gain vs Lookup | switch | harmful | recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0（raw） | bbob | −1.6107 | +0.0052 | +0.0251 | 0.642 | 0.136 | 0.730 |
| 0.0（raw） | ma | −4.5731 | +0.0450 | +0.0433 | 0.591 | 0.137 | 0.692 |
| 0.5 | bbob | −1.6253 | +0.0198 | +0.0397 | 0.417 | **0.073** | 0.514 |
| 0.5 | ma | −4.5740 | **+0.0459** | +0.0442 | 0.396 | **0.078** | 0.504 |
| 1.0 | bbob | −1.6184 | +0.0129 | +0.0328 | 0.324 | **0.051** | 0.422 |
| 1.0 | ma | −4.5596 | +0.0315 | +0.0298 | 0.259 | **0.043** | 0.364 |
| 1.5 | bbob | −1.6119 | +0.0064 | +0.0263 | 0.258 | 0.037 | 0.322 |
| 1.5 | ma | −4.5579 | +0.0298 | +0.0281 | 0.185 | 0.032 | 0.280 |
| 2.0 | bbob | −1.6080 | +0.0025 | +0.0224 | 0.196 | 0.030 | 0.239 |
| 2.0 | ma | −4.5462 | +0.0180 | +0.0163 | 0.113 | 0.020 | 0.182 |

（sum scale 同型：κ=0.5 时 switch 0.349/0.294、harmful 0.053/0.050、gain vs lookup +0.043/+0.033。）

## 3. pooled vs fold-local 配对比较（5000 draws，L_fold − L_pooled）

| scale | κ | suite | fold−pooled | 95% CI |
|---|---:|---|---:|---|
| max | 0.5 | bbob | +0.00175 | [−0.00012, +0.00534] |
| max | 0.5 | ma | +0.00006 | [+0.00000, +0.00019] |
| max | 1.0 | bbob | +0.00191 | [+0.00001, +0.00457] |
| max | 1.0 | ma | −0.00027 | [−0.00444, +0.00359] |
| sum | 0.5 | bbob | −0.00377 | [−0.01132, +0.00000] |
| sum | 0.5 | ma | +0.00149 | [+0.00000, +0.00366] |
| sum | 1.5 | ma | −0.00823 | [−0.02470, +0.00000] |

（其余 11 行 |point| ≤ 0.0015、CI 均含 0；κ=0 恒等于 0。完整表见 parquet。）最大绝对差 0.008（sum κ=1.5 MA），且方向不定——**无系统性劣化**。

## 4. 判读（H5/H6）

- κ=0.5（fold-local）：switch 0.417/0.396、harmful 0.073/0.078，仍显著低于 raw M2（0.642/0.591、0.136/0.137）；gain vs Continue +0.020/+0.046、vs Lookup +0.040/+0.044，两 suite 方向一致且为正；
- κ=1.0（fold-local）：harmful 0.051/0.043，风险优势保持，增益为正；
- pooled vs fold-local 的 loss 差幅度 ≤0.008 且无方向一致的劣化（两点 CI 恰好贴 0、幅度 ≤0.005，属可忽略量级）。

$$
\boxed{\text{Hygiene verdict：H1 NEGLIGIBLE}}
$$

R1 结论在完全 OOF 的 calibration 语义下成立；κ=0.5 保留为 performance-oriented frozen candidate，κ=1.0 保留为 risk-oriented frozen candidate（两个 operating points 均进入后续确认协议，最终 κ 不在本轮选择）。**Task 14A 允许进入（GO 不变）**。
