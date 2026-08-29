# 15a · 输入与 OOF 复现核查（Task 13.1A/B 门）

- 日期：2026-08-29；HEAD = `c55563f`（Task 13 提交，与工作单要求一致）；零新增 objective FE。
- 机器可读版：`15a_input_audit.json`。

## 1. 输入产物清单核查

Task 13 全部 8 个输入 parquet 与 Task 12.1 `oracle_ladder_current_conditioned.parquet` 均存在且可读（清单见工作单 §6）。

## 2. A1：RF-M2 OOF 逐 state 复现

从 Task 13 固定代码路径（同一 `run_grouped_oof`、同一正式 RF carrier、同一 cv_group 折）重新产生 M2 OOF，与已提交 `oof_policy_rows.parquet` 逐 state 对比：

| 检查 | 结果 |
|---|---|
| 三个动作预测 max abs diff | **0.0**（逐位一致） |
| selected action | 1890/1890 一致 |
| realized loss | 1890/1890 一致（atol 1e-12） |

复现通过（固定 random_state 的确定性成立），未触发 STOP。

## 3. B1：empirical current+FE OOF lookup 复现 Task 12.1

严格复用 Task 12.1 的 leave-cv_group-out 经验查表（不用 RF）：

| suite | 本轮 fb lookup | Task 12.1 ladder 全精度值 | abs diff |
|---|---:|---:|---:|
| BBOB | −1.585625 | −1.585625 | **0.0** |
| MA | −4.529830 | −4.529830 | **0.0** |

工作单引用的 −1.5856/−4.5298 为四位舍入显示，全精度一致。复现通过，未触发 STOP。

## 4. 部署用 pooled 噪声尺度（Task 13.1E 前置）

由 Task 12/13 训练域重复分支（两 suite 合并、按 cv_group 函数平衡、Q95）得到 **train-domain 预先固定的部署常数（train-domain pre-specified constants）**：

| solver | $\delta_a^{pool}$ | 组数 | 重复 cell 数 |
|---|---:|---:|---:|
| shade | 0.0740 | 22 | 179 |
| lshade | 0.1152 | 22 | 186 |
| cso | 0.0760 | 22 | 192 |

与 suite 无关、与 problem 无关，仅由 solver 决定，可安全作为部署常数。
