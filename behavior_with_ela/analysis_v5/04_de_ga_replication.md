# 04 · H-DEGA：Guo 2025 的 DE–GA 互补性假设独立复检

- 日期：2026-08-30
- 假设来源：Guo et al., SWEVO 2025 在其 10 算法 / 生成式 instance space 上报告 DE 与 GA 为最互补对之一。本轮将其视为**外部假设**，在当前 BBOB train + selected MA-BBOB（10D，B=10000，canonical 预先固定参数）动态 state setting 中独立检验，不预设成立。
- 产物：`analysis_v5/task12/h_dega_static.json`；static DCM 表。

## 1. 结果（static natural-run 口径；DE=rand/1/bin F=0.5/CR=0.9，GA=canonical SBX+PM）

| FE | suite | P(tie) | P(DE≻GA) | P(GA≻DE) | DCM | DE excl-win | GA excl-win |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2000 | bbob | — | — | — | — | 0.000 | 0.000 |
| 4000 | bbob | — | — | — | — | 0.000 | 0.000 |
| 6000 | bbob | 0.607 | 0.393 | **0.000** | 0.500 | 0.000 | 0.000 |
| 6000 | mabbob | 0.756 | 0.114 | 0.000 | 0.443 | 0.000 | 0.000 |
| 10000 | bbob | — | — | — | — | 0.000 | 0.000 |

（完整四 FE × 两 suite 表在 `h_dega_static.json`；de/ga 在全部 FE × suite 的 practical exclusive win 均为 0——两者都过不了弱性门槛，但相对排序始终 DE ≥ GA。）

## 2. 判定

**NOT REPLICATED（在当前设置中）**：

1. 互补性的前提是**双向**优势区（低 $S$ 且 $C\approx0$）；实测 $P(GA\succ DE)=0$ 处处成立，DE–GA 是单边支配对（C=+0.393 @6000 BBOB），DCM=0.50 为矩阵中的最高档。
2. GA 在本 problem distribution 上不仅不与 DE 互补，甚至无法在噪声之外击败任何候选（全部 FE/suite exclusive win = 0）。
3. 需要记录的边界：本项目 GA 为 canonical 预先固定实现（SBX η=15、pm=1/d、无精英），Guo 等的 GA 实现与调优未公开到可直接复刻的程度；本判定表述为"**canonical 预先固定 GA 在本 distribution 上未能复现 Guo 的 DE–GA 互补性**"，不排除调优后 GA 表现不同——但按协议本轮禁止调参，且"standalone 弱 + 未调优"本身就是 portfolio 筛选要淘汰的对象。
4. 方法论收获：静态 instance-level 互补性结论**不能**迁移到动态 state-action setting，必须像本轮这样在本域重新检验。
