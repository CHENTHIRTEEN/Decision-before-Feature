# SOTA Gap Map

| Candidate Contribution | What SOTA Already Does | User Evidence | Real Gap | Claim Strength | Risk |
|---|---|---|---|---|---|
| Formalize landscape-query acquisition as a state-conditional decision | ELA-based AAS computes landscape features and then selects an optimizer; cost-sensitive systems may count feature FE | Frozen utility definition, equal-total-FE design, shared-state query and no-query paths | Existing studies do not decide whether a prespecified independent query should be acquired before its features exist | **Strong conceptual claim.** The task distinction is directly supported by the formal protocol and literature comparison | Reviewers may view it as feature-cost-sensitive selection unless decision timing and downstream selector separation are explicit |
| Predict net query utility from pre-query algorithm-agnostic behavior | Trajectory AS uses trajectory ELA and sometimes optimizer-specific state to choose a second optimizer; trajectory representations classify problems or describe search | Frozen T0/B1/B2/B3 inputs, exclusions, Decision Model candidates, time-only comparator | No reviewed paper maps only pre-query, algorithm-agnostic behavior to the net utility of acquiring a separate fixed query | **Strong method claim; empirical effectiveness TBD.** | Behavior may add little beyond FE ratio; RQ2 and Time-only comparison must determine this |
| Define query utility under one coherent resource account | Traditional AAS may report feature-sampling FE; behavior studies may profile monitoring time; prior per-run work emphasizes reused evaluations | Equal total FE, query continuation budget reduction, time and memory terms, no double subtraction | Sampling opportunity cost and residual computation cost are rarely combined with downstream action quality in a state-level query/skip label | **Strong formulation claim.** Numerical trade-offs remain TBD | Weight choices may dominate decisions; sensitivity to time prices and query configurations is required |
| Generate labels from matched shared states and completed candidate continuations | Prior per-run studies regress candidate performance after a fixed CMA-ES prefix | Native optimizer-state replay, four-action continuation table, statewise normalized action loss | Prior reviewed work does not construct query and no-query outcomes from the same complete online state while preserving native continuation | **Strong protocol claim.** | Replay errors or incomplete optimizer state would invalidate labels; scientific consistency checks must precede result interpretation |
| Separate the query gate from the downstream action-loss selector | AAS predicts an optimizer directly; adaptive operator work selects a search mechanism | Frozen Selection Reference and separate Decision Model | Direct action selection cannot reveal whether landscape information was worth acquiring; the two prediction tasks need separate targets and evaluation | **Strong architectural claim.** | Poor downstream selection can make the query appear useless; state-only/query-only/full selector diagnostics are necessary |
| Make state-transition semantics explicit | Warm-starting studies show that handoff design changes second-stage performance | Native continuation for prefix action, population transfer for other actions, explicit relationship fields | Prior results cannot be compared fairly when continuing the current optimizer and changing optimizer are conflated | **Strong methodological claim.** | Population transfer may favor some algorithms; report by selected/default/prefix relations and keep initialization fixed |
| Use nested function-family out-of-fold model selection and threshold fitting | Many studies use random runs or instances; recent Elsevier work shows harder problem splits can remove apparent gains | Frozen nested training-family OOF protocol; validation and external suites excluded from fitting | The reviewed closest per-run studies do not select the gate model and decision threshold under held-out function families | **Strong evaluation-design claim.** | Few families may yield high uncertainty; family-level results and interval construction must be transparent |
| Evaluate a gate against time alone and fixed acquisition policies | Traditional AAS, SBS, and VBS are common; per-run work compares selector to individual algorithms | Time-only, Never Query, Always Query, Random Analysis, Traditional AAS, SBS, VBS | Existing comparisons do not isolate whether a query gate learns behavior beyond invocation stage while accounting for query use | **Strong baseline-design claim.** Performance superiority TBD | Baselines may expose no gain; that is a valid empirical result and must not be softened |
| Test transport only after freezing on BBOB training families | Kostovska et al. show BBOB-to-YABBOB degradation; representation studies warn about split sensitivity and BBOB dependence | Frozen BBOB validation, CEC2017, CEC2022, and engineering conditions | Evidence for cross-suite transport of pre-query utility prediction is absent | **Evaluation objective only; no current generalization claim.** | Distribution shift may cause failure; external suites must not affect preprocessing, model, features, or threshold |
| Connect Search Maturity to query utility without making a causal claim | Behavioral papers describe exploration, exploitation, diversity, entropy, stagnation, and similarity | Frozen behavior taxonomy, maturity-oriented windows, RQ5 analyses | Prior behavioral work does not establish when those summaries correspond to the utility of an independent landscape query | **Exploratory explanatory contribution; evidence TBD.** | Correlation, coefficients, and discriminant directions do not identify causation; interpretations must remain descriptive |
| Evaluate three prespecified query configurations without turning them into feature selection | Representation surveys show large variation across feature families, sampling, dimension, and cost | `descriptor_cheap` primary query; two predefined pflacco robustness queries | A gate may be configuration-dependent, so a single-query result cannot represent landscape analysis in general | **Bounded robustness claim after formal results.** | Comparing configurations after seeing outcomes could become post-hoc selection; all three run through the same frozen pipeline |
| Report query policy value as final optimization performance and resource use, not classification accuracy alone | Algorithm-selection work often reports prediction or selection accuracy alongside optimizer performance | RQ3 policy evaluation, call rate, final loss, utility, runtime, memory, ERT targets | Correct classification need not improve optimization when errors have unequal loss and acquisition has cost | **Strong evaluation rationale.** | Utility scales may obscure final performance; report both raw policy outcomes and normalized decision quantities |

## Gap Summary

### What is already established

The literature establishes that landscape features can inform algorithm and operator choices; trajectory data can describe problem--algorithm interaction; optimizer behavior can be summarized with progress, diversity, entropy, stagnation, distance, network, and interaction measures; and warm-started selection can exploit complementary algorithms. It also establishes that feature extraction consumes resources, that state transfer matters, and that apparent selection gains can weaken under more demanding problem splits or new benchmark suites.

### The unresolved question

None of the 27 reviewed and cited works asks the present decision question at the same information time: before a prespecified independent landscape query is executed, can low-cost algorithm-agnostic behavior identify states in which the query's expected downstream performance difference justifies its function-evaluation and computation cost?

The closest trajectory-based papers always construct a trajectory representation and then choose a second optimizer. Their target is candidate performance, their switch time is fixed, and one version includes CMA-ES-specific internal state. The behavior-analysis papers either cluster whole-run behavior or modify optimizer internals. The ELA-based AAS papers assume the landscape representation is already acquired for the selection decision. These are important antecedents, but they have different decision objects, labels, information boundaries, and state transitions.

### The defensible contribution boundary

The present paper may contribute:

1. a formal state-conditional query-acquisition problem;
2. a matched shared-state offline label that combines downstream action quality with explicit query cost;
3. a strict pre-query, algorithm-agnostic input boundary;
4. separation of the query gate from a fixed continuous action-loss selector;
5. nested function-family out-of-fold model and threshold selection;
6. RQ-aligned comparison against time-only, fixed, random, traditional AAS, SBS, and VBS references; and
7. a prespecified test of where the frozen procedure does and does not transport.

Only items 1--5 can currently be described as protocol or method contributions. Performance, value, cost reduction, behavior usefulness, and transport remain empirical TBDs.

### Claim limits

- The main claim concerns `descriptor_cheap`, not all ELA or all landscape representations.
- BBOB validation, CEC2017, CEC2022, and engineering performance have not yet been established.
- Behavioral association with query utility does not imply a causal role for Search Maturity.
- Reusing an optimizer trajectory removes no more than the need for a separate sample in the studies that do so; representation computation still has cost.
- A completed minimum-loss action is a retrospective comparison quantity, not a deployable ideal selector.
- Published numerical findings from the reviewed sources cannot substitute for this project's RQ1--RQ5 outputs.

### Principal reviewer objections to pre-empt

1. **“This is ordinary cost-sensitive AAS.”** Show that the gate acts before query features exist and has a different target from the downstream selector.
2. **“The controller only learns time.”** Make the Time-only comparison central to RQ2.
3. **“The label advantages a particular handoff.”** Define native continuation and population transfer exactly, and report relationship strata.
4. **“The split leaks related functions.”** Keep all selection and threshold fitting within nested training-family folds.
5. **“The result is specific to one descriptor set.”** Name the primary query precisely and use the two predefined robustness configurations without selecting among them post hoc.
6. **“External benchmarks were used during development.”** State and enforce their evaluation-only role; do not claim transport before results exist.
7. **“The manuscript is an implementation report.”** Organize Experimental Setup around RQs, estimands, contrasts, and statistical interpretation; move local storage and execution details out of the article.
