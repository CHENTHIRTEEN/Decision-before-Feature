# 01 · Candidate Implementations and Protocol（Task 12A）

- 日期：2026-08-30
- 协议：`complementarity_driven_dynamic_portfolio_screening_v1`；所有候选参数在产生任何 outcome 前按 canonical 默认值写死（`optimizers/state.py`），本轮零调参。

## 1. 候选池与参数（预先固定）

| 候选 | 名字 | 参数（全部 canonical / 项目既有） | 实现 |
|---|---|---|---|
| PSO | `pso` | w=0.72, c1=c2=1.49, vmax=0.2·span, global-best, N=40 | 项目既有 |
| lbestPSO | `lbestpso` | 同上常数；闭环环形 informants={self±1}（size 3），lbest 替换 gbest | 新增 `PSOLocalState` |
| DE/rand/1/bin | `de` | **项目既有 DE 即 canonical rand/1/bin**：F=0.5, CR=0.9, binomial crossover, N=40 | 项目既有 |
| SHADE | `shade` | success-history memory=5, archive≤N, Cauchy-F/Normal-CR, pbest 2/N–0.2, N=40 | 项目既有 |
| L-SHADE | `lshade` | 同 SHADE + 线性种群缩减 N: 40→4（canonical N_min）按 FE/10000 线性；实际种群大小与 archive 已记录 | 新增 `LShadeState` |
| GA | `ga` | 实数编码 generational GA：binary tournament + SBX(η_c=15, pc=0.9) + polynomial mutation(η_m=20, pm=1/d)，无精英保留（Deb canonical），N=40 | 新增 `GAState` |
| CSO | `cso` | Cheng & Jin 2015：随机配对竞争，败者 v'=r1∘v+r2∘(x_win−x_lo)+φ·r3∘(x̄−x_lo)，φ=0.5，仅败者评估（N/2 每代），N=40 | 新增 `CSOState` |

CMA-ES 的角色（工作单 §6）：不属于 balanced 筛选池；保留为 strong external baseline / SBS reference / **add-back dominance control**（其分支数据被隔离采集，仅在 P_balanced 预先固定后分析，见 12 报告）。

## 2. 统一协议

- d=10，B=10000，N=40（L-SHADE 初始 40、按 canonical 缩减并记录实际大小），boundary=reflect，seeds 1–5（预定义前 5 个，不得按结果更换）。
- 动态筛选主 horizon：**H_a=1000 FE**；terminal 仅作记录。
- Population-only transfer：跨算法只传 population+fitness，source-specific memory（velocity/covariance/archive/success-history）按目标算法语义重初始化，transfer RNG 复用现有 semantic fork（`TRANSFER_STREAMS`）；same-algorithm continue = `clone_optimizer_state` 全状态。
- Repetition：确定性 SeedSequence 抽样（AGENTS 0.2 禁止 hash，等价实现"deterministic hash sampling"意图）10%，R=3，沿用正式 RNG fork 语义（continue=offset stream，transfer=event offset）。
- 单元测试（12A）：以自检脚本承担（AGENTS 0.2 禁止 pytest/test 目录）。已验证：FE 记账精确、best 单调、有限值、同种子确定性、异种子敏感性、L-SHADE 缩减（33@2000→4@10000、archive≤N）、clone 独立性、全部新算法 transfer 正确。已知必要修复：L-SHADE 缩减后 pbest 区间下界 `min(2/N, 0.2)`（N<10 时原公式区间反转）。

## 3. 状态声明

DE 的 F=0.5 与文献常见 0.5–0.8 一致；该参数自 Phase 1 起就是项目正式值，本轮未改动。所有新算法实现先通过自检、再进入任何数据采集。
