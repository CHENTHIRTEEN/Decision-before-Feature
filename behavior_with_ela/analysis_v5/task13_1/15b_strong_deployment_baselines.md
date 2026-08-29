# 15b · 强部署基线阶梯与配对比较（Task 13.1B/C）

- 日期：2026-08-29。同一 1890 states、function-balanced、paired cv_group bootstrap（5000 draws，95% CI）。行级：`strong_baseline_policy_rows.parquet`；比较表：`strong_baseline_pairwise_bootstrap.parquet`。

## 1. 基线阶梯（fb realized loss，RF carrier）

| 基线 | BBOB | MA | 说明 |
|---|---:|---:|---|
| B0 Always Continue | **−1.6055** | **−4.5282** | 最朴素但很强：bbob 上优于 RF-M0 |
| B1 Empirical current+FE lookup | **−1.5856** | **−4.5298** | Task 12.1 精确复现（diff=0） |
| B2 RF-M0 | −1.5634 | −4.5219 | Task 13 的 ablation 基线（最弱） |
| B3 M1 Behavior-only | −1.6054 | −4.5732 | — |
| B4 Raw M2 | −1.6107 | −4.5731 | argmin 直接部署 |

**关键事实：Always Continue 与 empirical lookup 都强于 RF-M0（bbob 上分别强 +0.042/+0.022）。** Task 13 以 RF-M0 为参照的 Δ_B=+0.047/+0.051 高估了相对"最强简单策略"的优势。

## 2. 配对比较（C1–C4，5000 draws）

| 比较 | BBOB | MA |
|---|---:|---:|
| C1 Continue − M2 | +0.0052 [−0.098, +0.146] | +0.0450 [−0.0067, +0.104] |
| **C2 Lookup − M2** | **+0.0251** [−0.073, +0.159] | **+0.0433** [+0.0042, +0.085] |
| C3 M0 − M2 | +0.0473 [−0.048, +0.173] | +0.0513 [+0.0035, +0.108] |
| C4 Lookup − M1 | +0.0198 [−0.071, +0.146] | +0.0434 [+0.0023, +0.089] |

## 3. Task 13 的 +0.047/+0.051 分解（§33 问 7）

$\Delta_B(M0\rightarrow M2)$ = (M0 − lookup) + (lookup − M2)：

- BBOB：+0.047 = **0.022（RF-M0 弱于 lookup）** + 0.025（M2 真正超出 lookup）——约 47% 来自基线偏弱；
- MA：+0.051 = 0.008 + **0.043**——主体（84%）在相对 lookup 后仍保留，且 CI>0。

## 4. Verdict 1（Behavior vs Strong Baselines）

Raw M2 点估计在全部四个比较方向为正，但只有 MA 相对 lookup/M0 的 CI 不含 0；bbob 全部穿 0，且相对 Continue 仅近似打平（+0.005）。

$$
\boxed{\text{Verdict 1：S2 CONDITIONAL}}
$$

M2 相对最强简单基线保留正向期望（MA 证据较强、bbob 不足），但 **raw argmin 不构成可部署的支配性结论**；风险修正后的 margin 策略见 15d——在那里 M2 家族对 lookup 的增益反而更大（κ=0.5 max：+0.041/+0.044）。
