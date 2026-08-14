# Confirmed Contribution

## Core Contribution

| Field | Content |
|---|---|
| Main contribution statement | The paper defines a state-conditional, cost-adjusted decision problem for executing a prespecified landscape-descriptor query before its descriptors exist, and specifies an offline protocol that learns this decision from pre-query algorithm-agnostic behavior while separating query acquisition from downstream portfolio selection. |
| Contribution type | new method |
| One-sentence reviewer payoff | Landscape information becomes an optional, explicitly costed operation whose value can be evaluated at the search state where the acquisition decision is actually made, rather than an unconditional prerequisite of algorithm selection. |

## Why This Contribution Is Needed

| Field | Content |
|---|---|
| Field problem | Continuous black-box optimizers have complementary strengths, and landscape representations can support portfolio decisions, but the objective evaluations and computation used to acquire those representations compete with the optimization budget. |
| Specific gap | The reviewed ELA-based selection studies choose an action after descriptors have been acquired, whereas trajectory-based studies construct a representation and then choose a second-stage optimizer. They do not decide, before an independent query is run, whether its expected downstream benefit offsets its acquisition cost. |
| Concrete challenge | A valid label must compare query and no-query paths from the same complete optimizer state, avoid double-counting query FEs, expose downstream-selector error and state-transition effects, and train a gate without leaking query descriptors, function identity, algorithm identity, or held-out benchmark information. |
| Why prior work leaves it unresolved | Jankovic and Kostovska always construct trajectory representations and predict second-stage algorithm performance; static ELA-based AAS assumes acquired descriptors; behavior-characterization work describes whole-run dynamics or controls optimizer internals. Their decision objects, information times, labels, and transitions therefore differ from the query-acquisition problem defined here. |

## How This Paper Responds

| Field | Content |
|---|---|
| Design response | The method restores a shared complete native state, evaluates a no-query continuation and four query-adjusted actions, fits a query-specific action-loss Selection Reference, constructs normalized utility with equal-total-FE and explicit residual computation costs, and trains LDA, Logistic Regression, or Ridge gates using nested BBOB-training function-family OOF evaluation and train-only threshold fitting. |
| Evidence required | RQ1 must establish the query-utility distribution and its components; RQ2 must compare the three B3 model candidates and the frozen selected family against T0; RQ3 must compare all required policies under equal FE and measured resources; RQ4 must report frozen held-out and suite-specific external evaluations with coverage; RQ5 must report prespecified input-group and association analyses; all require effect sizes, intervals, and the prospectively fixed inferential choices listed in the TBD requirements. |
| Evidence available | The current project provides a complete formal problem definition, query and transition semantics, behavior variables, model and threshold protocol, benchmark split, baselines, RQ-aligned table and figure structures, reproducibility rules, and verified literature support for the motivation and task distinction. These materials support method and protocol claims only. |
| Evidence missing | Formal products P1--P10 and the corresponding RQ1--RQ5 estimates are not yet available. In particular, predictive value beyond T0, end-to-end resource benefit, BBOB-validation performance, CEC2017/CEC2022 or engineering transport, and Search-Maturity associations remain unsupported outcomes. |

## Claim Boundary

| Field | Content |
|---|---|
| Strong claims allowed | The paper may state that it formulates the pre-query acquisition decision, defines matched-state cost-adjusted labels, enforces a pre-query algorithm-agnostic input boundary, separates the gate from a fixed downstream selector, and specifies train-only nested function-family model and threshold selection. |
| Claims to soften or avoid | Do not state that the query is usually avoidable, behavior adds value beyond FE ratio, the policy improves loss or resource use, BBOB validation is successful, external benchmarks generalize, Search Maturity is causal, or the result applies to all ELA. Preserve the 38 evidence-specific TBD items until their formal products exist. |
| Novelty risk | A reviewer may interpret the work as ordinary feature-cost-aware AAS or trajectory-based warm-started selection. The honest distinction is corpus-scoped: among the literature reviewed here, no study was identified with the same pre-descriptor decision time, independent fixed query, matched query/skip utility, and separate downstream selector. |
| Significance risk | A reviewer may regard gating one fixed query as narrow. The response is that the bounded query makes acquisition value measurable and falsifiable; broader relevance depends on the two prespecified configuration analyses and cannot be asserted before those results are available. |

