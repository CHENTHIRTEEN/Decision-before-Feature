# Style Profile

## Corpus

| ID | Title | Venue / year | Role | Why selected |
|---|---|---|---|---|
| X1 | *Trajectory-based Algorithm Selection with Warm-starting* | IEEE CEC / 2022 | Closest task | Direct trajectory-selection and warm-start comparison |
| X2 | *Per-run Algorithm Selection with Warm-Starting Using Trajectory-Based Features* | PPSN / 2022 | Closest expanded task | Clear decomposition of prefix, features, prediction, and switch |
| X3 | *DynamoRep* | GECCO companion/arXiv version / 2023 | Trajectory representation | Concise cost/problem--solver-interaction motivation |
| X4 | *Determining Metaheuristic Similarity Using Behavioral Analysis* | IEEE TEVC / 2025 | Behavior characterization | Concept-first behavioral taxonomy and explicit limitations |
| X5 | *How do metaheuristics exploit?* | Evolutionary Intelligence / 2026 | Behavioral diagnostics | Effect-size and generalization restraint |
| X6 | *Landscape features ... algorithm selection generalization?* | Swarm and Evolutionary Computation / 2025 | Target-scene exemplar | Split-aware evaluation and bounded generalization claims |

## Global Style

- **Typical paper arc:** accepted workflow and concrete limitation \(\rightarrow\) precise missing decision \(\rightarrow\) formal separation of information, prediction, and action \(\rightarrow\) RQ-aligned evidence \(\rightarrow\) bounded interpretation.
- **Reader expertise assumed:** familiar with continuous black-box optimization, evolutionary/swarm algorithms, ELA, AAS, fixed-budget evaluation, and supervised learning; unfamiliar with the new query-acquisition estimand.
- **Claim strength:** methodological contributions can be stated directly; empirical effectiveness requires a named formal output, effect estimate, interval, and scope. External transport is never assumed.
- **Technical density:** high in formulation and method, moderate in Introduction/Related Work, and quantitative in Results. One principal idea per paragraph.
- **Citation style:** use the Elsevier CAS bibliography mechanism and real `\cite{}` commands. Place citations immediately after the claim they support; do not attach a long citation cluster to an entire paragraph.
- **Figure/table emphasis:** use the architecture figure to establish information timing and use RQ tables/figures to carry comparisons. Captions must be interpretable without local filenames or execution history.

## Section Profiles

### Abstract

| Dimension | Corpus consensus | Rule for this paper |
|---|---|---|
| Move sequence | problem context; exact gap; method; evaluation design; principal evidence; bounded implication | Preserve this six-move sequence; until evidence exists, keep the RQ1--RQ5 outcome sentences as explicit TBDs |
| Numerical specificity | only task-defining numbers or final headline estimates | The 16-descriptor primary query may be named; no preliminary or withdrawn performance number |
| Opening pattern | established workflow plus concrete shortcoming | Open with landscape-assisted selection and query cost, not a generic history of optimization |
| Closing pattern | implication limited to tested task | Do not conclude about all ELA; name evaluated queries and benchmarks |
| Forbidden moves | unsupported priority, broad generalization, metric-only ending | No “first,” “proves,” “generalizes,” or “negligible overhead” without formal support |

### Introduction

| Dimension | Corpus consensus | Rule for this paper |
|---|---|---|
| Paragraph target | 7--9 substantive paragraphs | Keep the current eight-move arc but tighten repeated boundary statements |
| Move sequence | portfolio heterogeneity; ELA workflow; acquisition cost; trajectory information; generalization risk; exact question; contributions; paper map | Reach the pre-query question by the middle of the section |
| Gap statement style | contrast decision objects and information timing | State that reviewed AAS selects an action after representation acquisition, whereas this gate acts before descriptors exist |
| Contribution placement | after formal task distinction, before paper organization | Use four compact contributions; label empirical value and transport as questions, not accomplishments |
| Citation density | one focused cluster per factual move | Cite local warm-starting and DynamoRep antecedents explicitly, not only broad surveys |

### Related Work

| Dimension | Corpus consensus | Rule for this paper |
|---|---|---|
| Organization | task streams that progressively approach the gap | Order: static landscape/AAS; acquisition cost; per-run trajectory selection; trajectory/behavior representations; adaptive search/MetaBBO; generalization; exact gap |
| Comparison unit | information source, target, action, cost, and transition | End every subsection with at least one explicit difference on these axes |
| Synthesis | closest work receives the most detailed comparison | Give Jankovic and Kostovska a dedicated subsection; include DynamoRep, Hayward, and Mbasso by exact methodological role |
| Novelty wording | corpus-scoped, not universal | Use “Among the literature reviewed here, we did not identify ...” |
| Forbidden moves | citation catalogue and straw-man comparison | Acknowledge what each antecedent solves before naming the remaining question |

### Problem Formulation and Method

| Dimension | Corpus consensus | Rule for this paper |
|---|---|---|
| Procedure depth | sufficient to reconstruct state, representation, target, fit, and deployment | Preserve all frozen mathematical definitions and explain the rationale immediately around them |
| Equation detail | equations define estimands and transformations; prose defines direction and scope | After every core equation, state which sign is preferable and what is unavailable at decision time |
| Subsection rhythm | motivation sentence; definition; operational consequence; boundary | Apply consistently to state, query paths, behavior, Selection Reference, and Decision Model |
| Rationale placement | before or immediately after each design choice | Explain why native state, permutation invariance, train-only preprocessing, and family OOF are necessary |
| Terminology | stable standard terms | Use *query acquisition*, *decision utility*, *native continuation*, and *population-transfer initialization* consistently |

### Experimental Setup

| Dimension | Corpus consensus | Rule for this paper |
|---|---|---|
| Organizing spine | research question, comparison, analysis population, measure, inference | Replace report-like generation and artifact narration with RQ estimands and contrasts |
| Benchmark detail | functions/families, dimensions, instances, seeds, budgets, portfolio, split | Retain exact frozen values; remove shard counts unless scientifically needed for sample-size interpretation |
| Baselines | named with decision semantics and common budget | Explain Never/Always/Random/Traditional AAS/Time-only/SBS/VBS as scientific contrasts |
| Cost reporting | sampling FEs separate from computation and memory | State that query FEs reduce the remaining budget; report time and memory separately; do not double-charge FEs |
| External tests | frozen and suite-specific | BBOB validation, CEC2017, CEC2022, and engineering problems must be described separately |

### Results

| Dimension | Corpus consensus | Rule for this paper |
|---|---|---|
| Ordering | scientific question, principal estimate, uncertainty, comparison, interpretation | Retain RQ1--RQ5; open each subsection with the estimand rather than data-generation status |
| Quantitative detail | effect direction, magnitude, uncertainty, replication unit, and failures | Every claim maps to its table/figure/statistical product and a TBD identifier until available |
| Comparison language | metric-specific and scoped | Say “lower final error,” “higher mean utility,” or “fewer calls with \(U_q\le0\),” never a vague “better” |
| Interpretation depth | brief in Results, fuller in Discussion | Do not infer causation from coefficients, discriminant directions, correlations, or paired continuations |
| Placeholder style | structural, not project-status narrative | Remove “regeneration,” “withdrawn,” “artifact,” and “at drafting time”; retain only neutral evidence requirements |

### Discussion and Conclusion

| Dimension | Corpus consensus | Rule for this paper |
|---|---|---|
| Opening move | answer the main question with scope and condition | Until outputs exist, keep the outcome-dependent sentences as TBD and discuss only the formulation |
| Summary/interpretation | interpret mechanisms and limits rather than repeat every metric | Relate behavior to utility descriptively and preserve disagreement among decision and auxiliary metrics |
| Limitation style | limitations placed beside the claim they constrain | Name query configuration, portfolio, transfer rule, split, extraction failures, and benchmark coverage |
| Closing move | practical/scientific implication bounded to evidence | Configuration consistency may support robustness across three queries, not universal value of landscape analysis |

### Reproducibility

| Dimension | Corpus consensus | Rule for this paper |
|---|---|---|
| Scientific objects | benchmark release, dimensions, budgets, algorithms, seeds, feature definitions, fit/evaluation rules | Describe these directly and align them with equations and RQs |
| Environment | versions and resource-measurement conditions | Report Python/library versions, hardware, threads, cache, and peak-memory method when formal measurements exist |
| Randomization | deterministic run indexing and explicit seed construction | Retain the integer `SeedSequence` design without narrating local commands |
| Data description | variables sufficient to recompute estimands | Describe state keys and retained quantities conceptually; omit Parquet paths and directory trees |
| Forbidden moves | local paths, CLI lists, artifact lifecycle, handoff status | No result folder names, configuration filenames, shell commands, regeneration story, or “current active directory” language |

## Phrase and Skeleton Bank

These are abstract rhetorical skeletons, not source sentences.

| Function | Reusable skeleton | Constraint |
|---|---|---|
| Establish workflow gap | `Although [representation] can support [downstream decision], acquiring it consumes [resources] before its state-specific value is known.` | Name the fixed query rather than all landscape analysis when narrowing |
| State closest-work difference | `[Prior work] uses [trajectory information] to choose [action] after [representation condition]; here, the decision is whether to acquire an independent query before its descriptors exist.` | Acknowledge prior contribution first |
| Present estimand | `We evaluate the query by comparing [shared-state no-query path] with [query-and-selector path] under [common budget], then subtracting [non-duplicated cost].` | Do not call this causal without an identification design |
| Introduce evaluation | `RQk is assessed by [contrast] on [replication unit], using [effect/interval] and [failure/coverage measure].` | Avoid local output names |
| Report result | `[Policy A] yielded [direction and magnitude] in [metric] relative to [Policy B] ([interval]), over [scope].` | Use only after formal output exists |
| Bound generalization | `The estimate applies to [suite/query/portfolio]; [missing or divergent evidence] prevents a broader transport claim.` | Keep suites separate |

## Claim Strength Rules

| Claim type | Allowed verbs/frames | Avoid |
|---|---|---|
| Formal definition or implemented method | defines, formulates, constructs, restricts, separates, freezes | proves, guarantees |
| Direct formal result | yielded, differed by, was associated with, the interval indicated | universally improves, validates in general |
| Interpretation | suggests, is consistent with, may reflect | causes, demonstrates mechanism |
| Literature-scoped novelty | among the literature reviewed here, we did not identify | first-ever, unprecedented |
| Limitation | is limited to, remains untested, did not support, coverage was incomplete | minor issue, negligible limitation |
| Pending evidence | will be evaluated; remains TBD pending [named product] | preliminary numbers, expected success |

## Terminology and Collocations

| Concept | Preferred term | Avoid | Basis |
|---|---|---|---|
| Operation being gated | fixed landscape-descriptor query; query acquisition | ELA in general; feature bundle | Frozen problem definition |
| Query comparator | no-query path / query path | treatment/control; counterfactual | No causal estimand is defined |
| Retrospective action minimum | best observed action | oracle | Uses completed candidate continuations |
| Switching state | population-transfer initialization | warm start without qualification | Frozen transition semantics differ by action |
| Predictor inputs | algorithm-agnostic behavior; pre-query behavior | optimizer internals | Information boundary |
| Cost | query FEs, query time, selection time, peak memory | free features; negligible overhead | Resource accounting |
| Validation | frozen BBOB evaluation; external evaluation | validated generalization | Outcomes pending |
| Interpretation | predictive association; descriptive stratification | causal driver | Study design |

## Style Constraints for Rewriting

- Introduction: approximately 900--1,200 words, 7--9 paragraphs.
- Related Work: 6--7 task-centered subsections, each ending in a comparison or gap sentence.
- Problem Formulation and Method: preserve equation-level completeness; paragraphs generally 90--170 words.
- Experimental Setup: organize around RQs, populations, contrasts, measures, and statistical analysis; no local execution narrative.
- Results: one estimate-centered narrative unit per table or figure; no claim before its TBD requirement is satisfied.
- Reproducibility: scientific procedure and environment only; move repository-specific instructions outside the article.
- Sentences: prefer 18--32 words; split sentences carrying more than one qualification and one methodological consequence.
- Citation density: typically 1--3 citations per substantive background paragraph, concentrated at the exact supported claim.
- Maximum claim strength: protocol-level novelty plus evidence-bounded empirical conclusions.
- Words/phrases to avoid: `first`, `proves`, `oracle` for observed minima, `counterfactual`, `zero cost`, `negligible`, `generalizes` without scope, `artifact`, `shard`, `regeneration`, `at drafting time`, local path names, and CLI commands.

## Section Blueprint Inputs

| Section | Obligatory moves | Closest exemplar | Paragraph target | Evidence placement rule | Opening/closing rule |
|---|---|---|---:|---|---|
| Abstract | context, cost gap, task, method, formal evidence, bounded implication | X2 + X6 | 1 block | Numerical claims only from formal RQ outputs | Open with workflow limitation; close at evaluated scope |
| Introduction | standard workflow, cost, trajectory signal, generalization risk, exact decision, contributions, RQs | X1 + X3 + X6 | 7--9 | Local antecedents in trajectory paragraphs; Elsevier sources in cost/generalization paragraphs | State Decision-before-Feature before contribution list |
| Related Work | six comparison streams and corpus-scoped gap | X2 + X4 | 6--7 subsections | Each prior-work claim immediately cited; no citations used as project evidence | End every stream with an axis of difference |
| Formulation | state, decision, paths, utility, label, decomposition | X2 | 4--5 subsections | Project definitions control; literature is contextual only | Open with estimand; close with claim boundary |
| Method | offline/online split, native windows, behavior, maturity, selector, gate | X3 + X4 | 6 subsections | Rationale adjacent to each equation/table | Close with frozen deployment sequence |
| Experimental Setup | RQs, populations/splits, queries, opportunities, baselines, metrics, statistics | X6 | 7--8 subsections | Values from frozen protocol; citations only justify design | Open with RQ map; close with prospective statistical choices |
| Results | RQ1--RQ5 estimates, uncertainty, answer, sensitivity | X6 | 5 RQ units + sensitivity | Formal products only; exact TBD mapping | Open with neutral reporting logic, not project status |
| Discussion/Conclusion | answers, mechanism interpretations, limitations, bounded implication | X5 + X6 | 6 discussion units + concise conclusion | Every outcome maps back to formal result | Separate held-out and external scope |
| Reproducibility | randomization, data construction, environment, resource measurement, statistical reconstruction | X3 | 4--5 subsections | Scientific definitions and versions, not local storage | Close with availability/resource TBD if needed |

## Compact transfer table

| Section | Style rule | Planned template row | Result template row | Why this transfers |
|---|---|---|---|---|
| Introduction | Concrete resource gap precedes method naming | I-ELA-cost / I-pre-query-gap | -- | X1 and X3 make the design response necessary before describing it |
| Related Work | Compare decision object and information time | RW-trajectory-AS / RW-behavior / RW-gap | -- | X2 separates representations and actions clearly |
| Method | Concept before metric, consequence after equation | M-behavior / M-utility / M-selector | -- | X4's category-first structure improves interpretability |
| Experimental Setup | RQ, contrast, replication unit, statistic | E-RQ-map / E-baselines / E-statistics | -- | X6 makes evaluation design part of the contribution |
| Results | Estimate, interval, scope, interpretation | -- | RQ1--RQ5 table/figure/stat/claim rows | X5 and X6 distinguish detectability, practical effect, and transport |
| Reproducibility | Reconstruct scientific procedure, not local execution | REP-randomization / REP-resources | -- | X3 gives concise methodological availability without making it the article's narrative |

## Non-Negotiable Rules

1. The pre-query Decision Model never receives query descriptors, function ID, algorithm ID, or optimizer-specific internals.
2. Keep query acquisition separate from downstream action selection in every section and figure.
3. Do not state that BBOB validation, CEC2017, CEC2022, or engineering-problem generalization has succeeded before formal RQ4 products exist.
4. Do not use withdrawn results, local result files, or project execution status as manuscript evidence.
5. Query FEs reduce the continuation budget and are not deducted twice; time and memory must be reported according to the frozen weighting.
6. Report effect size, uncertainty, scope, and practical criterion; a nonsignificant comparison is not equivalence.
7. Claims concern the evaluated query configurations, not landscape analysis universally.
8. Use corpus-scoped novelty wording and preserve association-versus-causation boundaries.

