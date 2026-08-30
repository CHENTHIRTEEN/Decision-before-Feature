# Decision-before-Feature：Task 14B.1 Correctness × Scale-matched Segment Behavior Audit

Part A：**CA1 CORRECTIONS DO NOT CHANGE NO-GO**。Part B：**SB3 SCALE DOES NOT RESCUE SEGMENT**。最终：**F3 FINAL GENERIC-BEHAVIOR NO-GO**。

## Part A corrected result

| suite | Continue | lookup | M0 | MG | MS-old | MGS-old |
|---|---:|---:|---:|---:|---:|---:|
| bbob | -1.969171 | -1.984669 | -1.983517 | -1.954284 | -1.942101 | -1.941348 |
| mabbob | -5.032240 | -5.047094 | -5.043119 | -5.019019 | -5.005823 | -5.023923 |

Within-route corrected：

| suite | L_W0 | L_WG | L_WS-old | L_WGS-old | Δglobal | Δsegment |
|---|---:|---:|---:|---:|---:|---:|
| bbob | -2.017612 | -2.014543 | -2.014583 | -2.014453 | -0.003069 | -0.000090 |
| mabbob | -5.084686 | -5.086835 | -5.086364 | -5.085736 | +0.002149 | -0.001099 |

Old Segment-only permutation p：BBOB 0.702970，MA 0.772277；Global permutation p：BBOB 0.217822，MA 0.089109。原 W0 held-out leakage、P2 整行置换、A_ND dominance 方向和 Continue/lookup absolute 报告口径均已修正；Task 14A `switch_required` 逐 state 一致。

## Scale-matched grouped OOF

| suite | M0 | MG | MS-old | MS-matched | MGS-old | MGS-matched |
|---|---:|---:|---:|---:|---:|---:|
| bbob | -1.983517 | -1.954284 | -1.942101 | -1.939975 | -1.941348 | -1.945953 |
| mabbob | -5.043119 | -5.019019 | -5.005823 | -5.014945 | -5.023923 | -5.021413 |

| suite | MGS-matched vs M0 | MGS-matched vs lookup | MGS-matched vs MG |
|---|---|---|---|
| bbob | -0.037564 [-0.058543,-0.017567] | -0.038716 [-0.062789,-0.017937] | -0.008331 [-0.023174,+0.000828] |
| mabbob | -0.021706 [-0.059521,+0.036774] | -0.025681 [-0.059566,+0.021253] | +0.002394 [-0.009506,+0.016039] |

Matched within-route：

| suite | L_W0 | L_WG | L_WS-matched | L_WGS-matched | Δglobal | Δsegment-matched |
|---|---:|---:|---:|---:|---:|---:|
| bbob | -2.017612 | -2.014543 | -2.015462 | -2.014826 | -0.003069 | +0.000283 |
| mabbob | -5.084686 | -5.086835 | -5.086169 | -5.086816 | +0.002149 | -0.000019 |

Matched Segment-only permutation：

| suite | observed | null mean | empirical p |
|---|---:|---:|---:|
| bbob | +0.000283 | +0.000199 | 0.504950 |
| mabbob | -0.000019 | +0.000196 | 0.534653 |

结论限定于 tested 10D balanced portfolio、1000-FE post-handoff commitment、fixed RF/Ridge carriers 与 200/500/1000-FE generic trajectory behavior descriptors；不能外推为 Behavior 在所有 switching 情形均无用。seeds 6–10、closed-loop repeated DAS、CEC2017/CEC2022 均不进入；ProgressForecast 维持 PG3 NO-GO。
