# 14h · Shuffle Controls 与 Time-Proxy 敏感性（Task 13O/P）

- 日期：2026-08-29。全部为零 FE 置换/敏感性实验；置换流由显式 SeedSequence 派生。
- 产物：`shuffle_control_results.parquet`、`time_proxy_sensitivity.parquet`。

## 1. O1：在 (current, FE) 层内置换 Behavior，重拟合 M2（grouped-OOF）

| carrier | suite | repeats | shuffle 后 $\Delta$ 均值 | q025 | q975 | **真实 $\Delta_B$（对照）** |
|---|---|---:|---:|---:|---:|---:|
| RF | BBOB | 100 | −0.0074 | −0.0204 | +0.0044 | **+0.0473** |
| RF | MA | 100 | −0.0062 | −0.0268 | +0.0110 | **+0.0513** |
| Ridge | BBOB | 100 | −0.0393 | −0.0533 | −0.0255 | +0.0219 |
| Ridge | MA | 100 | −0.0178 | −0.0449 | +0.0046 | −0.0034 |

RF 的真实增量（+0.047/+0.051）**远超** shuffle 分布上界（+0.004/+0.011，经验 p<1/100）：M2 的增量不是 (current, FE) 层内任何 Behavior-agnostic 结构的伪影。Ridge 在 MA 上真实增量落于 null 之内，与其 OOF 无增量的结论自洽。

## 2. O2：在 (problem, current, FE) 组内置换 Behavior，within-problem LOSO

| carrier | suite | repeats | shuffle 后 $\Delta_{within}$ 均值 | q025 | q975 | **真实 $\Delta_{within}$** |
|---|---|---:|---:|---:|---:|---:|
| RF | BBOB | 10 | −0.0024 | −0.0082 | +0.0072 | **+0.0188** |
| RF | MA | 10 | +0.0011 | −0.0049 | +0.0079 | **+0.0161** |
| Ridge | BBOB | 100 | −0.0439 | −0.0643 | −0.0221 | −0.0042 |
| Ridge | MA | 100 | −0.0305 | −0.0613 | −0.0001 | +0.0119 |

真实 within 增量为 RF null 上界的 2.3–2.6 倍——**genuine state signal 可信**（RF repeats=10 为计算预算预先声明；Ridge 100 repeats 供对照）。

## 3. Time-Proxy 敏感性（13P）

time-like 特征清单（28 列 selector 契约内）：仅 **`bf_fe_ratio`**（3 个 maturity 列不在契约内，未参与）。比较 M2 完整 vs 去除 bf_fe_ratio：

| carrier | suite | M2 完整 | M2 −time | 差 |
|---|---|---:|---:|---:|
| RF | BBOB | −1.6107 | −1.6045 | −0.0062（fe_ratio 有正贡献） |
| RF | MA | −4.5731 | −4.5794 | +0.0063（去除后反而更好） |
| Ridge | 两 suite | — | 与完整版相同 | 0（Ridge 线性下 bf_fe_ratio 被显式 FE_ratio 吸收） |

剔除后增量不消失（两 suite 变化 ≤0.006、方向相反）——**Behavior 增量不是线性 time/maturity 代理驱动的**；结合特征重要性（14i：time-like 特征未进 top），更支持真实 trajectory 形状信息。

## 4. 结论

两级 shuffle null 均无法解释观测增量；time-like 剔除不破坏增量。M2（RF）的 deployment 增量与 within-problem 状态分辨力均为**真实的 Behavior 信息**，而非混杂构造。
