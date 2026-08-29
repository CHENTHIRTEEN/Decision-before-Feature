# 14g · Within-Problem Genuine State Test（Task 13N）

- 日期：2026-08-29。设计：固定 (problem, current, FE)（378 组 × 5 seeds），leave-one-seed-out（4 train / 1 test 轮换 5 次）：
  - **W0** 组内均值基线（每动作预测=训练 4 seeds 均值）；
  - **W1** 仅 $B^{global}$（28 列）；
  - **W2** [current, FE, $B^{global}$]（组内 current/FE 恒定 ⇒ 额外信息实质全部来自 Behavior）。
- 判据 $\Delta_{within}=L_{W0}-L_{W2}$；fb + cv_group paired bootstrap（2000）。

## 1. 主结果

| carrier | suite | $L_{W0}$ | $L_{W1}$ | $L_{W2}$ | $\Delta_{within}$ | 95% CI | $L_{W0}-L_{W1}$ | 95% CI |
|---|---|---:|---:|---:|---:|---|---:|---|
| RF（正式 carrier） | BBOB | −1.6573 | −1.6768 | **−1.6760** | **+0.0188** | [+0.0045, +0.0343] | +0.0195 | [+0.0058, +0.0336] |
| RF | MA | −4.5859 | −4.6003 | **−4.6020** | **+0.0161** | [+0.0052, +0.0303] | +0.0144 | [+0.0026, +0.0300] |
| Ridge | BBOB | −1.6573 | −1.6531 | −1.6531 | −0.0042 | [−0.0176, +0.0104] | — | — |
| Ridge | MA | −4.5859 | −4.5978 | −4.5978 | +0.0119 | [−0.0368, +0.0825] | — | — |

## 2. 与 shuffle null 的对照（O2，见 14h）

RF 载体下组内置换 Behavior 的 null 分布（10 repeats）：均值 −0.002/+0.001，q97.5 = +0.0072/+0.0079——真实 $\Delta_{within}$（0.019/0.016）为 null 上界的 **2.3–2.6 倍**，远超置换噪声。

## 3. 收益分布（防"少数问题驱动"）

- W2 更优的 problem 占比：BBOB 18/30（60%）、MA 10/12（83%）；
- 增量质量集中度：top-3 problem 占总增量 73%/74%——覆盖面广，但 BBOB 的增量质量偏集中（如实记录）。

## 4. 判读

**固定 problem、current、FE 之后，Behavior 仍能区分不同 state**：正式 RF carrier 下两个 suite 的 $\Delta_{within}$ 均 CI>0，且显著高于组内置换 null；Ridge（4 行训练样本下基本退化为均值估计）无法提取该信号，属载体能力边界而非数据缺失。收益覆盖多数 problem，非孤立问题驱动（质量集中度已记录）。

$$
\boxed{\text{Verdict B：B1 GENUINE STATE VALUE（RF 正式载体；Ridge 对照不支持的边界已如实声明）}}
$$
