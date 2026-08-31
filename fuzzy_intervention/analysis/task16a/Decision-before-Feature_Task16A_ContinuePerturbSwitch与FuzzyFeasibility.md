# Decision-before-Feature Task16A：Continue–Perturb–Switch 与 Fuzzy Feasibility

## Action Space

1. 实际 source states 数：2520。
2. Continue/Switch 复用旧 FE：0。
3. 新 Perturb FE：5,970,000。
4. Targeted Perturb practical ND rate：0.7879。
5. Random Perturb practical ND rate：0.7955。
6. Targeted Perturb 优于 Continue rate：0.0179。
7. Switch 优于 Continue rate：0.0646。
8. Perturb 优于两个 Switch rate：0.0341。
9. Switch 优于 Perturb rate：0.1475。
10. E|A_ND|：4.0720。
11. 三层动作是否 non-degenerate：NO。
12. Action-space verdict：A3 PERTURB NO-GO。

## Targeting

13. Targeted vs Random paired gain：0.0088 [-0.0084, 0.0240]。
14. BBOB CI：0.0140 [-0.0022, 0.0364]。
15. MA-BBOB CI：0.0045 [-0.0222, 0.0233]。
16. high-stagnation subset：-0.0022 [-0.0217, 0.0127]。
17. harmful rate：Targeted=0.1303，Random=0.1301。
18. Targeting verdict：T2 TARGETING INCONCLUSIVE。

## Probe Structure

19. R1 intervention-required rate：0.1222。
20. R2 intervention-required rate：0.0314。
21. Delta_intervention：-0.1230 [-0.2436, -0.0134]。
22. R3：Perturb beneficial=0.0357，Switch beneficial=0.0913。
23. R4：Perturb beneficial=0.0000，Switch beneficial=0.0171。
24. 跨 suite 同方向：YES；BBOB 与 MA-BBOB 均为 R2<R1，与预期富集方向相反。
25. 是否由单一 solver 驱动：NO；SHADE、L-SHADE、CSO 的 Delta_intervention 均小于 0。
26. Probe verdict：P3 NO PROBE STRUCTURE。

## Maturity

27. R2 Early/Mid median G：0.0478。
28. R2 Late median G：0.0000。
29. Delta_M：-0.0478 [-1.6007, 0.0000]。
30. Maturity 是否改变 intervention preference：未形成跨 suite 一致差异。
31. 是否只改变 overall loss scale：本实验报告相对差 G；若 Delta_M 不稳定，不能排除主要是 loss scale 变化。

## Final

32. 最终 verdict：F3 THREE-LEVEL INTERVENTION NO-GO。
33. 是否允许 Task16B Type-1：NO。
34. 是否允许 Interval Type-2：NO。
35. 是否允许 membership tuning：NO。
36. 是否允许 seeds 6–10：NO。
37. 是否允许 CEC：NO。
38. 是否允许 closed-loop：NO。
39. Task15A I3 是否仍成立：YES。
40. 是否可以声称 Behavior 精确预测最佳 solver：NO。

平方和开方 noise threshold 的最终结论为：F3 THREE-LEVEL INTERVENTION NO-GO。

## 科学解释边界

本结论只覆盖所测 development setting、固定 q/sigma/kernel 与 1000 FE horizon。Task16A 没有训练或评价任何模糊控制器。
