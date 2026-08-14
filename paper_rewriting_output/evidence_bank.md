# Evidence Bank

## Evidence classes

| ID | Evidence item | Current status | Supports now | Does not support now |
|---|---|---|---|---|
| M1 | Formal state-conditional decision, shared complete state, query/no-query paths, and equal-total-FE budgets | Defined in current Problem Formulation | The task and estimand are precisely specified | Any direction or magnitude of query utility |
| M2 | Utility \(U_q=G-\lambda_T C_T-\lambda_M C_M\), primary \(\lambda_T=1\), \(\lambda_M=0\), and five time-weight sensitivity levels | Defined in current Problem Formulation | Coherent resource accounting and no duplicate FE charge | A favorable resource trade-off or insensitivity to cost weights |
| M3 | Complete-native-update windows and 31 model-eligible permutation-invariant behavior variables in B1--B3 | Defined in current Method | A pre-query algorithm-agnostic representation has been operationalized | Predictive value beyond FE ratio or transport across optimizers |
| M4 | Query-specific multi-output action-loss Selection Reference over four unique actions | Defined in current Method | Separation of acquisition and downstream action selection | High selector accuracy, low regret, or optimality |
| M5 | LDA, Logistic Regression, and Ridge candidates; B3-only nested family-OOF selection; group-specific train OOF thresholds | Defined in current Method | A leakage-resistant selection protocol is fixed | Which model wins or whether any model is useful |
| M6 | BBOB training/validation family partition and external evaluation roles | Defined in Experimental Setup | Training and evaluation information boundaries are fixed | Successful BBOB-validation, CEC, or engineering transfer |
| M7 | Required policy baselines and RQ-aligned metrics/statistical plan | Defined in Experimental Setup | The comparisons capable of testing the contribution are specified | Any superiority, equivalence, or practical benefit |
| L1 | Jankovic et al. 2022, trajectory-based AS with warm-starting | Project-local full text; verified bibliographic record | Closest direct-action antecedent and trajectory reuse | Present query-utility or external performance |
| L2 | Kostovska et al. 2022, per-run AS using trajectory features and CMA-ES state | Project-local full text; verified bibliographic record | Optimizer-internal versus trajectory representations; transfer risk | Present gate validity or generalization |
| L3 | Cenikj et al. 2023, DynamoRep | Project-local full text; verified bibliographic record | Longitudinal population representations can encode interaction | Query-value prediction or algorithm-agnostic validity |
| L4 | Hayward and Engelbrecht 2025, behavioral similarity | Project-local full text; verified bibliographic record | Common behavioral vocabulary and construct cautions | Suitability of known-optimum or identity-dependent measures as gate inputs |
| L5 | Mbasso et al. 2026, exploitation behavior diagnostics | Project-local full text; verified bibliographic record | Distance decay, directional entropy, stagnation, and practical-effect caution | Search Maturity validity or present CEC performance |
| E1 | Cenikj et al. 2025 and 2026 Elsevier landscape representation studies | Zotero full text verified; cited in current manuscript | Split sensitivity, BBOB dependence, representation and reporting boundaries | Present family-split effectiveness or transfer |
| E2 | Korošec and Eftimov 2024; Ochoa et al. 2021 | Zotero full text verified; cited in current manuscript | Native search histories provide problem--algorithm information | Cost-free representation or utility prediction |
| E3 | Malan and Engelbrecht 2013; Guo et al. 2025; Kerschke and Trautmann 2019 | Zotero/local full text verified; cited in current manuscript | Landscape/AAS workflow, heterogeneous acquisition demands, SBS/VBS lineage | Unconditional value of ELA at a particular state |
| E4 | Sallam et al. 2017; Zhou et al. 2024; parameter-control and MetaBBO reviews | Zotero full text verified; cited in current manuscript | Boundary between information acquisition and optimizer control | Classification of this method as online parameter/operator control |

## Pending formal evidence products

| Product | RQ role | Required before the paper may claim | Present manuscript locations |
|---|---|---|---|
| P1--P5 | RQ1 | Distribution and components of primary-query utility, including the proportion with \(U_q\le0\) and family-level intervals | `TBD-ABS-RQ1`, four RQ1 TBDs, `TBD-DISC-01`, `TBD-CONC-RQ1` |
| P2, P5, P6, P8, P10 | RQ2 | Selected B3 model family, B3-versus-T0 effect, auxiliary metrics, and frozen validation performance | `TBD-ABS-RQ2`, four RQ2 TBDs, `TBD-DISC-02`, `TBD-CONC-RQ2` |
| P5--P7, P10 | RQ3 | Equal-budget policy effects on utility, final loss, runtime, query use, and non-beneficial calls | `TBD-ABS-RQ3`, four RQ3 TBDs, `TBD-DISC-03`, `TBD-CONC-RQ3` |
| P6--P8, P10 | RQ4 | Suite-specific BBOB-validation, CEC2017, CEC2022, and engineering effects with coverage and failures | `TBD-ABS-RQ4`, four RQ4 TBDs, `TBD-DISC-04`, `TBD-CONC-RQ4` |
| P6, P9, P10 | RQ5 | T0--B3 ablation, model-appropriate stability, maturity--utility associations, and scope | `TBD-ABS-RQ5`, four RQ5 TBDs, `TBD-DISC-05`, `TBD-CONC-RQ5` |
| P1--P10 | Sensitivity and validity | Robustness across three prespecified queries and time weights, selector diagnostics, relationship strata, and completeness/failure disclosures | `TBD-SENS-DIAG-01`, `TBD-DISC-06` |
| P7, P10 | Resource reproducibility | Hardware/software measurement conditions and same-system behavior, query, inference, selection, wall-time, and peak-memory values | `TBD-REPRO-RESOURCE-01` |

## Evidence-use rule

Literature supports motivation, terminology, comparison axes, and design choices. Method definitions support only statements about what the protocol is. Neither source class can fill a Results, Abstract, Discussion, or Conclusion outcome slot. Formal empirical language must wait for the named P1--P10 products and must report direction, magnitude, uncertainty, and scope.

