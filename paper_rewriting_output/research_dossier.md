# Research Dossier

## Scope and evidence basis

This dossier is prepared for an English-language Elsevier journal manuscript in evolutionary computation, swarm intelligence, and continuous black-box optimization. The current evidence set contains 27 cited works: five independent full-text papers in `docs/90_literature`, ten Elsevier journal papers documented in the manuscript literature notes, and twelve additional Zotero-indexed neighboring studies. Literature establishes motivation, terminology, and design constraints; it does not supply evidence for the pending Decision-before-Feature experiments.

The five independent project-local papers are:

1. Jankovic et al. (2022), *Trajectory-based Algorithm Selection with Warm-starting*, IEEE CEC.
2. Kostovska et al. (2022), *Per-run Algorithm Selection with Warm-Starting Using Trajectory-Based Features*, PPSN XVII.
3. Hayward and Engelbrecht (2025), *Determining Metaheuristic Similarity Using Behavioral Analysis*, IEEE Transactions on Evolutionary Computation.
4. Mbasso et al. (2026), *How do metaheuristics exploit? A particle-level behavioral study across the CEC 2025 benchmark functions*, Evolutionary Intelligence.
5. Cenikj et al. (2023), *DynamoRep: Trajectory-Based Population Dynamics for Classification of Black-box Optimization Problems*, GECCO.

The ten Elsevier journal sources cover landscape representation and generalization, a representation survey, Opt2Vec, search trajectory networks, fitness-landscape characterization, landscape-informed population adaptation, MetaBBO, ELA-based algorithm selection, parameter control, and landscape-based operator selection.

## Venue Requirements

- **Article form.** Present a research article with a focused scientific question, a formal problem definition, a method that can be reconstructed from the paper, a benchmark design aligned with the claims, and results organized by research question. The manuscript should read as a journal argument, not as an execution log.
- **Elsevier presentation.** Retain the current CAS journal structure, numbered sections, compact mathematical notation, numbered tables and figures, and a complete bibliography. The source should use real citation commands and stable bibliographic records.
- **Methodological completeness.** Define the information available at the decision time, the fixed query, the query and no-query paths, the shared state, the downstream action set, state transition rules, utility, cost terms, model-selection protocol, threshold fitting, and evaluation metrics.
- **Cost-aware comparison.** Charge policies under the same total function-evaluation budget. Report function-evaluation use separately from feature-computation time and memory. Do not imply that reusing trajectory observations removes feature-computation cost.
- **Evaluation design.** Keep BBOB training families separate from frozen BBOB validation families. Model selection, preprocessing, feature choices, and threshold fitting use BBOB training families only. CEC2017, CEC2022, and engineering problems are separate external evaluations, not tuning resources.
- **Statistical reporting.** Report effect sizes, uncertainty intervals, multiplicity handling, paired comparisons where supported by the design, and practical-effect criteria. Statistical detectability must not be substituted for practically meaningful improvement.
- **Baselines.** Include Never Query, Always Query, Random Analysis, Traditional AAS, Time-only Controller, SBS, and VBS. On shared online states, call the minimum over completed candidate continuations the *best observed action*, because it is a retrospective comparison rather than an available ideal rule.
- **Claim discipline.** Current text may define the protocol and explain its relationship to prior work. It must not state that BBOB validation, CEC, or engineering generalization has succeeded before the corresponding formal outputs exist. Withdrawn project results cannot be cited as evidence.
- **Reproducibility prose.** Describe scientific objects and procedures: benchmark releases, function families, dimensions, instances, seeds, budgets, optimizer versions, state variables, feature definitions, model candidates, random-stream construction, and analysis rules. Local paths, filenames, file formats, shell commands, shard counts as storage details, and project handoff language do not belong in the article body.
- **Availability statement.** If code and data are later released, state the repository and version at submission time. Until then, do not present a local working directory as an archival research resource.

## Review Criteria

### 1. Novelty of the decision problem

A reviewer should be able to distinguish this paper from algorithm selection, trajectory representation, adaptive operator selection, and parameter control. The central decision is whether to acquire a prespecified landscape-descriptor query at the current state. It is not which optimizer or operator to use after features already exist.

### 2. Information-time consistency

Every predictor input must be available before the query is executed. Function identifiers, algorithm identifiers, algorithm-specific internal parameters, and query features are excluded from the gate. The paper must make this temporal information boundary explicit and apply it consistently in equations, tables, and interpretation.

### 3. Fair performance and cost accounting

Query and no-query paths must start from the same replayed optimizer state. Query sampling function evaluations reduce the remaining optimization budget under the primary equal-total-FE design and cannot be deducted a second time. Additional feature time and memory enter utility only if they have not already entered final performance loss.

### 4. Valid state transitions

Native continuation of the prefix optimizer preserves its complete state. Choosing another portfolio algorithm uses the frozen population-transfer initialization. The manuscript should report `selected_equals_default`, `selected_equals_prefix`, and `handoff_required` as scientific relationships, without replacing them with an ambiguous changed-algorithm label.

### 5. Label and selector validity

Labels must be generated from completed candidate continuations at the same state. The downstream Selection Reference predicts continuous action losses over four unique actions: native continuation plus the other three portfolio algorithms. The Decision Model predicts query utility, not function family, algorithm identity, query features, or the hindsight-best action.

### 6. Generalization protocol

Random run or instance partitions within each function are insufficient for the paper's main claim. The reviewer should see nested function-family out-of-fold selection on BBOB training families, a threshold derived from complete training-family out-of-fold scores, frozen BBOB validation, and separately reported external benchmarks. External success remains an empirical question.

### 7. Strength of baselines and alternative explanations

The Time-only Controller is necessary to test whether behavior adds information beyond the evaluation stage. Traditional AAS tests whether conditional query use improves on always paying for the fixed query. SBS and VBS contextualize portfolio complementarity. State-only, query-only, and full Selection Reference diagnostics distinguish downstream selection quality from gate quality.

### 8. Interpretability without causal overreach

Coefficients, discriminant directions, maturity associations, and subgroup differences may explain model behavior descriptively. Shared-state candidate continuations do not by themselves establish a causal effect. Search Maturity is a measured latent organization of behavior, not a claimed cause unless a formal causal design is added.

### 9. Construct precision

The main query is the 16-dimensional `descriptor_cheap` configuration. It cannot stand for all ELA, all pflacco features, or all landscape analysis. The two pflacco configurations are prespecified robustness conditions. Conclusions must name the evaluated query and cost setting.

### 10. Evidence traceability

Each claim in Abstract, Results, Discussion, and Conclusion must map to a formal RQ output. Until those outputs exist, substantive outcome sentences remain marked TBD. Citations motivate the question and justify methods; they do not fill missing project results.

## Accepted Paper Patterns

### Contribution-first opening

Strong papers in the source set open from a concrete limitation in an established workflow. The relevant pattern is: landscape-informed selection can help, but information acquisition consumes resources; trajectory observations are available earlier; therefore the value of acquiring a separate query must be evaluated conditionally. The opening should reach the Decision-before-Feature question before surveying every neighboring literature stream.

### Progressive narrowing of related work

The most useful ordering is:

1. landscape characterization and ELA-based algorithm selection;
2. trajectory-based per-run selection and warm-starting;
3. behavior characterization and dynamic diagnostics;
4. adaptive operators and broader automated optimization;
5. the unresolved transition from action selection to query selection.

Each subsection should end with a precise difference, not a generic statement that prior work is limited.

### Formal separation of task components

Accepted algorithm-selection studies clearly separate instance representation, performance prediction, portfolio selection, and evaluation. This paper should add a preceding layer and keep all layers distinct:

\[
\text{pre-query behavior}\rightarrow\text{query decision}\rightarrow
\text{fixed query, if called}\rightarrow\text{action-loss selector}\rightarrow
\text{state transition}.
\]

### RQ-aligned experimental narrative

The Experimental Setup should state what each RQ tests, what comparison identifies it, and which statistic answers it. Dataset construction and implementation details support that design but should not become the organizing spine of the section.

### Boundary-aware external evaluation

Kostovska et al. report that a BBOB-trained selector transferred poorly to YABBOB when trajectory coverage differed. Cenikj et al. show that demanding problem splits can remove apparent gains over SBS. A credible paper therefore treats held-out families and new suites as attempts to test transport, not as guaranteed validation.

### Results as validation of contribution promises

- RQ1 validates the need for conditional acquisition by estimating the distribution of query utility.
- RQ2 validates the pre-query prediction contribution against time alone and across T0--B3.
- RQ3 validates resource-aware policy performance against fixed and traditional alternatives.
- RQ4 evaluates the boundary of transport to frozen validation and external suites.
- RQ5 explains when decisions differ using prespecified descriptive analyses.

### Limitations close to the relevant claim

The source papers are strongest when limitations accompany results: portfolio dominance, warm-start dependence, benchmark shift, reliance on known optima, fixed dimension, restricted algorithm families, or incomplete real-world evidence. This manuscript should place equivalent qualifications beside each result rather than postponing all caveats to Discussion.

## Comparative synthesis of the literature set

| Literature stream | Decision object | Information available when the decision is made | Cost treatment | Training label or analytical target | State transition | Generalization boundary | Difference from this paper |
|---|---|---|---|---|---|---|---|
| Jankovic et al. (CEC 2022) | Choose one second-stage optimizer | ELA features computed from a fixed CMA-ES prefix | Reuses prefix evaluations; no query-versus-skip net utility | Fixed-budget log performance per candidate optimizer | One algorithm-specific warm start | BBOB 5D with instance-group evaluation; CEC transfer was future work | Always constructs the trajectory representation and chooses an optimizer; no pre-query gate |
| Kostovska et al. (PPSN 2022) | Choose one of six second-stage variants | Trajectory ELA, CMA-ES internal-state time series, or both | Avoids a separate ELA sample; computation and acquisition are not jointly priced in a gate utility | Candidate performance regression | One warm-started switch from CMA-ES | BBOB 5D/10D; BBOB-to-YABBOB performance weakened | Includes algorithm-specific internal state and no query/skip label |
| Hayward and Engelbrecht (TEVC 2025) | No online action; cluster behavioral similarity | Whole-run behavior, including known-optimum, interaction, locality, and effort measures | FE and ERT are characteristics, not an acquisition decision | Behavioral profile and cluster co-membership | None | Five relatively simple static functions at 20D; fixed population and budget assumptions | Descriptive whole-run characterization, not state-conditional query utility prediction |
| Mbasso et al. (2026) | Trigger optimizer-specific exploitation intervention | Particle movement, directional entropy, stagnation, and distance to a reference | Reports monitoring overhead; not query acquisition utility | Behavioral diagnostic and optimizer performance | Modifies PSO, DE, or GWO internals | CEC 2025; broader noisy, dynamic, and real-world transport not established | Controls an optimizer after observing behavior; does not decide whether to acquire landscape descriptors |
| ELA and landscape surveys | Characterize a landscape or summarize representations | Independent sample, walk, trajectory, or learned representation | Documents sampling and computation differences | Landscape or representation property | None | Strong BBOB concentration and heterogeneous protocols | Establishes representation costs but not a state-specific acquisition policy |
| Guo et al. (2025) and related ELA-based AAS | Select a portfolio algorithm | Landscape sample and computed ELA features | Counts feature sampling FE; the query is still executed for every decision | Algorithm performance or class label | Starts or selects an optimizer | Evaluation depends on generated instances and split design | Traditional AAS is the downstream comparator, not the proposed gate |
| Sallam et al. (2017) and Zhou et al. (2024) | Select an operator or adapt population organization | Landscape measurements plus optimizer history | Discusses overhead but optimizes solver behavior | Operator reward or benchmark performance | Changes the running optimizer | Algorithm- and benchmark-specific studies | Their decision objects are search mechanisms, not information acquisition |
| Jankovi\'c and Doerr (2019), adaptive landscape analysis | Observe local descriptor evolution during CMA-ES | Descriptors computed on 2,000 additional points at target levels | Extra sample evaluations are required even though descriptor computation needs no further FEs | Feature evolution and preliminary problem discrimination | No query gate or paired transition comparison | Three BBOB functions in 5D; selector coupling remained future work | Repeated local ELA observation is not a pre-sample acquisition decision |
| Pei et al. (2025), AOS survey | Select search operators within a metaheuristic | Stateless credit history or state features from solution, problem, and process | Cost is tied to operator allocation and feature/state construction | Operator reward or state-to-operator policy | Changes the active search operator | Method- and problem-dependent survey corpus | The action is an operator, not independent information acquisition |
| Petelin and Cenikj (2025), benchmarking pitfalls | Evaluate algorithm-selection meta-models | Problem instances, features, algorithm ranks, and performance targets | Exposes misleading instance partitions and scale-sensitive errors | Pairwise ranking or performance-regression quality | None | BBOB/COCO methodology examples | Motivates function-family and task-aligned evaluation but cannot validate the present split or utility |
| Ochoa et al. (2021), Opt2Vec, and trajectory survey work | Describe or learn from optimizer trajectories | Visited points or populations | Reuses evaluations; representation cost remains | Network statistic or problem-classification target | None | Task- and benchmark-dependent | Supports trajectory information as input but not its sufficiency for query utility |
| Cenikj et al. (2025, 2026) | Evaluate representations and selection generalization | Static and learned landscape representations | Highlights protocol sensitivity | Algorithm selection performance | Selects an algorithm | Difficult problem splits and cross-benchmark evidence remain challenging | Supplies the reason for strict splits, not evidence that the proposed gate generalizes |
| MetaBBO and parameter-control reviews | Optimize parameters, components, structures, or update rules | Task and training feedback determined by each framework | Broad resource discussion | Optimizer reward or performance | Alters search configuration or learned optimizer | Depends on task distribution and training design | Decision-before-Feature uses offline supervised query-utility prediction and does not learn an optimizer update rule |

## Constraints for This Paper

### Frozen scientific scope

- The paper studies whether the evaluated fixed landscape-descriptor query should run; it does not propose a new optimizer.
- The main query is `descriptor_cheap`: 16 descriptors, `lhs_50d`, and a 5% FE allocation. `pflacco_standard` and `pflacco_broad` are prespecified robustness conditions.
- The optimizer portfolio is DE, PSO, CMA-ES, and SHADE.
- Decision inputs are algorithm-agnostic behavior and the allowed continuous budget context. No function ID, algorithm ID, query feature, or algorithm-specific parameter enters the gate.
- Active Decision Model candidates are LDA, Logistic Regression, and Ridge. The model family is selected only by nested BBOB-training function-family out-of-fold decision utility.
- The Selection Reference is a multi-output Random Forest regressor over four unique actions and statewise normalized observed action losses. It is a fixed downstream reference, not the main algorithmic contribution.
- BBOB training uses 10D, 20D, and 40D and function-family splits. BBOB validation is evaluation-only. CEC2017, CEC2022, and engineering problems are separate external conditions.

### Required prose changes

- Rewrite the experimental section around scientific choices and RQs. Remove local results directories, local filenames, storage paths, shell commands, intermediate-table names, and project workflow language from manuscript prose.
- Describe data persistence only when it affects the scientific contract, for example the state key or variables required to reproduce native continuation. Do not narrate local artifact organization.
- Present reproducibility through benchmark identifiers, versions, seeds, update boundaries, state transition definitions, model fitting, and analysis rules.
- Distinguish trajectory ELA, optimizer-internal time series, algorithm-agnostic behavior, and independent landscape queries. They are not interchangeable feature names.
- Use *query* or the exact configuration name when referring to the evaluated operation; do not generalize the main result to all ELA.

### Prohibited evidential shortcuts

- Do not transfer numerical results from any cited paper to the present method.
- Do not use withdrawn Phase 1 results as support for model choice, threshold quality, behavior usefulness, or external performance.
- Do not state that BBOB validation or CEC/engineering transport has been established before the formal RQ4 outputs exist.
- Do not infer that algorithm-agnostic behavior is predictive merely because trajectories classify problems or cluster algorithms.
- Do not state that trajectory reuse has zero cost.
- Do not call algorithm-specific internal state algorithm-agnostic.
- Do not equate the best completed action at a state with a deployable ideal rule.

### Present evidence boundary

At the current project state, the manuscript can claim that the problem, protocol, implementation interfaces, and planned evaluations are defined. It can explain why the question is distinct and scientifically motivated. Performance, calibration, cost savings, model superiority, explanatory associations, and transport to held-out or external problems remain TBD until their formal products are generated.
