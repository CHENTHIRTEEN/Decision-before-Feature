# Decision-before-Feature 维度与泛化实验设计

> 唯一活动说明（2026-08-15）。本文只定义 dimension 分层与外部泛化边界。BBOB-validation 已被旧模型比较、调参与消融查看，只作已见内部评价；CEC2017 也已有 preliminary/targeted 结果，只作已见外部开发评价。二者均不用于当前调参、选模、特征组选择或 threshold，也不能承担独立确认。CEC2017 的 F2/F30 口径、CEC2022 与工程问题配置仍是运行前 blocker。

## 1. 主 BBOB 设置

| split | functions | dimensions | instances | seeds | role |
|---|---|---|---|---|---|
| BBOB-train | 1,2,3,4,6,7,8,10,11,12,15,16,17,18,20,21,22,23 | 10,20,40 | 1,2,3 | 1--30 | 完整 nested function OOF、选模、threshold/Random calibration 与最终重拟合 |
| BBOB-validation | 5,9,13,14,19,24 | 10,20,40 | 1,2,3 | 1--30 | 已见固定六函数内部评价 |

当前 COCO `bbob` suite 不支持 50D，因此 BBOB 50D/100D 不进入主 validation。若未来选择支持这些维度的 suite，必须作为新的 extension，独立冻结 functions、bounds、budget、targets 和 failure rules。

## 2. Dimension 的信息边界

Dimension 必须保存为 metadata，用于：

- run/problem/function 键；
- 分层效应、coverage、missingness 与失败分析；
- Behavior/query/runtime 分布移动；
- action loss shard 的科学端点与 canonical loss 追踪。
- 外部 suite 的范围说明。

Dimension 不进入 Decision X，也不进入 Selector 的 query 前 Behavior 编码。禁止通过 function ID、algorithm ID、known optimum/gap 或 dimension 让模型记忆 benchmark。输入中不含 dimension 只表示没有显式 identity 字段，不能自动证明已学到 dimension-invariant 规律。

## 3. 预算与行为归一化

主 BBOB 预算固定为

$$
B=1000D,
$$

即 10D/20D/40D 分别为 10,000/20,000/40,000 FE。Population size 固定为 40。Behavior 使用实际 $FE/B$ 和 native-update 窗口；空间距离按 bounds 与 $\sqrt D$ 规范化。所有分层仍报告 raw FE、actual window FE/ratio 和 runtime，防止无量纲输入掩盖 dimension-dependent cost。

## 4. 泛化层次

### 4.1 Held-out BBOB functions

主内部泛化采用 function-ID grouped split，不是随机 instance/seed split，也不是经典 landscape-family taxonomy。BBOB-validation estimand 是 6 个已见固定 functions 的等权有限集均值；逐 function effects、条件 95% CI 与失败率必须完整展示。该结果不属于未查看确认集，也不把这六函数当作函数超总体样本。

### 4.2 Dimension 分层

在 BBOB train OOF 和冻结 validation 内分别报告 10D、20D、40D 的：

- $G_{\mathrm{FE}}$（方案 A 主标签）、$U_q^{joint}$、$U_b$、$I_q$（旧口径兼容）；
- terminal/continuation-only `log10_gap`、target-hit rate、endpoint-success rate、ERT；
- first-trigger call/trigger/handoff；
- query/sample/optimization/runtime 组成；
- coverage 与 query/selector/action/optimizer failure。

Dimension 分层是 heterogeneity analysis，不把每个 function--dimension 组合升级为独立顶层样本。

### 4.3 Cross-benchmark evaluation

BBOB-train 冻结的 SBS、Selectors、Decision preprocessing/model、feature group、threshold 和 Random calibration 原样用于外部 suite。已见 BBOB-validation、已见 CEC2017、前瞻 CEC2022 与前瞻工程问题分别报告，不池化为单一“OOD”分数，也不把任一外部 suite 用于重新训练。只有在本轮冻结后先完成配置/端点/分析规则、再首次生成 outcome 的外部集合可承担独立确认。

CEC2017 当前维度、seeds 和预算写为 10D/30D/50D、30 seeds、$1000D$；但 `configs/phase1_cec2017_test.yaml` 使用 F1--F29，即包含 F2、排除 F30。项目内尚无依据确认该函数集与所用实现/官方口径一致，必须在运行前核对并冻结，不能静默改配置或先看结果。

CEC2022 与工程问题尚未冻结具体 functions/problems、dimension、bounds、budget、success target、gap floor/cap、timeout、first-hit 和 constraint-handling rule，因此当前不能执行或声称覆盖。

### 4.4 Cross-prefix/algorithm robustness

主 Decision population 只使用 `prefix_algorithm == default_algorithm == SBS_fold`。DE、PSO、CMA-ES、SHADE 的全 prefix 数据只进入预定义 cross-probe robustness、leave-one-probe-out 与 algorithm-agnostic 分层，不得混入主训练或主结果。不同 prefix 的 Skip 若离开 prefix，必须显式记录 population transfer。

该分析最多支持“输入不显式含算法 identity 且关联在所评估 prefixes 上稳定”；不能证明模型只学习了抽象行为或对未见优化器普遍泛化。

## 5. 统计单位与等价

Function 是最高聚合层。层级固定为 run → static problem → fixed dimension stratum → function。BBOB-validation 的 10,000 次条件 bootstrap 保留全部六个已见固定 functions、dimensions 与 instances 1/2/3 对应的 static problems，只在每个固定 static problem 内配对重抽 optimizer seeds。Dimension 与 static problems 均为固定 strata；function-resampling 只作函数组成敏感性，不进入主 CI，也不产生 transformed-instance 超总体推断。

Utility $\pm0.01$、`log10_gap` $\pm0.05$、runtime ratio $[0.95,1.05]$、call/target-hit rate $\pm0.05$ 只称项目内 operational tolerance。条件 CI 仅逐项描述相对边界的位置；未来未查看评价集若作等价判断，须先冻结有领域含义的边界与 simultaneous interval。不同 dimension 的 Utility 抵消不能替代 terminal performance 或 runtime 的端点判断。方案 A 下主功效以 `G_FE` 为主，旧 Utility 仅作兼容。

## 6. 可支持的结论

只有在对应效应与区间完整时，才可写：

- 冻结 policy 在所评估维度上的效应稳定或表现出 dimension dependence；
- 冻结 BBOB-train procedure 在某个具体 external suite 上保持、减弱、反转或未建立效应；
- 某 query 配置的 failure/runtime 随 dimension 改变。

不得写：

- 随机 instance split 证明 function-level generalization；
- 不输入 dimension 即证明 dimension-invariant learning；
- CEC 天然代表所有 OOD；
- 部分 suite 成功即证明无条件跨 benchmark 泛化；
- 未冻结或未运行的 50D/100D、CEC2022 或工程问题已被覆盖。

## 7. 执行顺序

资源与排期确认后，按完整科学子矩阵分阶段执行：

1. main `descriptor_cheap_invariant` BBOB train/validation；
2. `pflacco_standard_invariant` / `pflacco_broad_invariant` 配置稳健性；
3. 函数集口径核对后的 CEC2017；
4. 配置和 constraint rule 冻结后的 CEC2022；
5. 配置冻结后的工程问题。

阶段性结果必须注明实际覆盖，不能冒充尚未完成的全协议结论。
