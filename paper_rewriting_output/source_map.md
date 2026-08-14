# Source Map

## Controlling research question

The paper asks whether a prespecified, independently sampled landscape-descriptor query should be executed at a given black-box optimization state. The decision is made **before** query descriptors exist, using only algorithm-agnostic behavior already available from the native optimizer history and the permitted budget context. Query value is defined by its downstream effect under a common total function-evaluation (FE) budget, with residual computation cost accounted for separately. Under the frozen phase-one estimand, query FEs reduce the continuation budget, query and selection time enter the primary utility through \(\lambda_T\), and memory remains a measured resource dimension while \(\lambda_M=0\). No source below is evidence that the proposed rule works; only the formal RQ1--RQ5 outputs can establish that.

## Source classes and roles

| Source class | Materials | Evidential role in this paper | Prohibited use |
|---|---|---|---|
| Project protocol and manuscript | Current `AGENTS.md`, project handoff/method documents, and `docs/40_manuscript` | Authoritative definitions of the decision time, query paths, utility, state transitions, model candidates, splits, baselines, and TBD boundaries | May not be silently changed to match a cited method |
| Project-local full-text literature | Five PDFs in `docs/90_literature` | Closest antecedents for trajectory-based selection, trajectory representation, behavior characterization, and state-sensitive search diagnostics | Published outcomes may not be imported as results of Decision-before-Feature |
| Verified Elsevier literature notes | Ten full-text Zotero readings summarized in `docs/40_manuscript/literature_reading_notes.md` | Landscape representation, feature cost, AAS, adaptive search, generalization risk, and target-journal rhetoric | Cannot establish the proposed controller's held-out or external performance |
| Current Results placeholders | RQ1--RQ5 structures and `TBD_REQUIREMENTS.md` | Define the empirical products needed for each claim | Placeholders and withdrawn runs are not evidence |

## Project-local full-text literature

| ID | Work | What it contributes to the argument | Exact distinction from this paper | Primary manuscript use |
|---|---|---|---|---|
| L1 | Jankovic et al. (2022), *Trajectory-based Algorithm Selection with Warm-starting* | Shows a per-run selection workflow in which ELA features are computed from a fixed optimizer prefix and a second-stage algorithm is selected with warm-starting | The representation is always constructed; the target is second-stage algorithm performance, not whether a separate query should be acquired; query-versus-skip utility and joint FE/time/memory accounting are absent | Introduction; Related Work on trajectory-based selection; distinction between action selection and query acquisition |
| L2 | Kostovska et al. (2022), *Per-run Algorithm Selection with Warm-Starting Using Trajectory-Based Features* | Compares trajectory ELA with CMA-ES internal-state time series and studies transfer from BBOB to YABBOB | Uses a fixed CMA-ES prefix, optimizer-specific variables, and a direct switch recommendation; its cross-suite difficulty motivates evaluation but is not evidence about the proposed frozen pipeline | Related Work on warm-starting and information types; motivation for held-out-family and cross-suite evaluation |
| L3 | Cenikj et al. (2023), *DynamoRep: Trajectory-Based Population Dynamics for Classification of Black-box Optimization Problems* | Demonstrates that longitudinal population summaries can represent problem--algorithm interaction without an independent objective-function sample; explicitly discusses representation size and computation | Solves problem classification, creates dimension- and population-size-dependent representations, and does not predict net query utility or enforce an algorithm-agnostic gate | Related Work on trajectory representations; rationale for lightweight history summaries; caution against substituting classification accuracy for decision utility |
| L4 | Hayward and Engelbrecht (2025), *Determining Metaheuristic Similarity Using Behavioral Analysis* | Provides a common behavioral vocabulary for exploration, exploitation, locality, communication, and evaluation effort across metaheuristics | Uses whole-run behavioral profiles and some quantities tied to known optima, individual identities, or interaction structures; it neither defines a pre-query online state nor a query decision | Related Work on behavior characterization; support for interpretable algorithm-comparative summaries; input-boundary caveat |
| L5 | Mbasso et al. (2026), *How do metaheuristics exploit?* | Uses distance-to-reference decay, directional entropy, and stagnation as interpretable behavioral diagnostics and distinguishes statistical detectability from practical importance | Triggers optimizer-specific interventions and evaluates PSO/DE/GWO on CEC 2025; it does not establish the present Search Maturity construct, utility label, or external generalization | Related Work on behavioral diagnostics; RQ5 terminology and non-causal interpretation; practical-effect reporting |

## Elsevier literature mapped from verified Zotero readings

| ID | Work | Supported role | Claim boundary |
|---|---|---|---|
| E1 | Cenikj et al. (2025), *Landscape features in single-objective continuous optimization: Have we hit a wall in algorithm selection generalization?* | Demonstrates why demanding problem splits and SBS comparisons are necessary | Does not validate this project's family split, threshold, or external tests |
| E2 | Cenikj et al. (2026), *A survey of features used for representing black-box single-objective continuous optimization* | Taxonomy of static, trajectory, algorithm, and interaction representations; sampling/computation and reporting requirements | Does not show that the frozen behavior variables predict utility or that trajectory features are cost-free |
| E3 | Korošec and Eftimov (2024), *Opt2Vec* | Dynamic population states can encode problem--algorithm interaction | Problem classification is not query-utility prediction; learned inputs are not automatically algorithm-agnostic |
| E4 | Ochoa et al. (2021), *Search trajectory networks* | Native search histories can support algorithm-behavior analysis without independent objective sampling | Network construction has computation cost and its metrics are not the frozen behavior variables |
| E5 | Malan and Engelbrecht (2013), fitness-landscape characterization survey | Landscape measures differ in assumptions, sampling, search dependence, and practical relevance | Supplies no query budget, utility weight, or threshold for this study |
| E6 | Zhou et al. (2024), landscape-adaptive multi-population ABC | Example of landscape information controlling a solver mechanism | Its CEC and engineering outcomes are not Decision-before-Feature generalization evidence |
| E7 | Yang et al. (2025), MetaBBO review | Locates information acquisition among broader automated-optimization objects | The proposed method remains offline supervised learning, not online learned optimization |
| E8 | Guo et al. (2025), ELA-based AAS with LightGBM | Representative sample--ELA--selector pipeline, SBS baseline, and FE-aware feature acquisition | Does not justify adding LightGBM to the frozen gate candidates or equating low-cost ELA with statewise positive utility |
| E9 | Gomes Pereira de Lacerda et al. (2021), parameter-control review | Separates tuning, numerical parameter control, and operator selection | Decision-before-Feature is none of these controller-training tasks |
| E10 | Sallam et al. (2017), landscape-based adaptive operator selection for DE | Shows how landscape and operator-history information can control search and why overhead/transfer matter | Algorithm-specific operator control does not establish an algorithm-agnostic query-acquisition rule |

## Additional Zotero neighboring studies used in the manuscript

| ID | Work | Supported role | Claim boundary |
|---|---|---|---|
| Z1 | Jankovi\'c and Doerr (2019), *Adaptive Landscape Analysis* | Establishes that local ELA descriptors can be observed repeatedly during CMA-ES and that the descriptor sample itself consumes evaluations | Does not decide before sampling whether to acquire descriptors and does not define paired query/skip utility |
| Z2 | Pei et al. (2025), adaptive operator-selection survey | Distinguishes stateless credit-driven and state-based feature-driven operator selection | Both categories allocate search operators; neither treats independent descriptor acquisition as the action |
| Z3 | Petelin and Cenikj (2025), algorithm-selection benchmarking pitfalls | Motivates problem-oriented partitions and task-aligned, scale-aware evaluation | Does not validate this study's function-family split, normalization, utility target, or external transport |

## Manuscript-unit map

| Manuscript unit | Controlling project evidence | Literature anchors | Required rhetorical function |
|---|---|---|---|
| Abstract | Formal problem and method; RQ1--RQ5 outputs when available | No literature-dependent result claim | Problem--gap--method--evidence--bounded implication; all unavailable outcomes remain TBD |
| Introduction | Frozen decision timing, query definition, cost accounting, and five RQs | E5, E2, E8 for ELA/cost; L1--L3, E3--E4 for trajectory information; E1 for generalization risk | Move from obligatory landscape acquisition to the unresolved value-of-query decision |
| Related Work | Decision object, information time, label, transition, split, and cost distinctions | All L sources; E1--E10 and Z1--Z3 by subsection | Compare tasks explicitly rather than list methods; end each stream with the remaining gap |
| Problem Formulation | Shared state, query/no-query paths, utility, action relations | Literature only motivates terms | Define the estimand without local implementation narrative or empirical outcome language |
| Method | Native histories, behavior variables, Selection Reference, Decision Model, threshold | L3--L5 and E3--E4 for representation context; L1--L2 for action-selection contrast | Explain why each design element is needed and preserve the pre-query information boundary |
| Experimental Setup | Frozen benchmarks, queries, RQs, baselines, measures, inference plan | E1--E2 for split/reporting norms; L2 for transfer risk; E8 for AAS/SBS context | Present estimands, contrasts, and analysis populations rather than files, commands, or generation logs |
| Results | Formal RQ1--RQ5 products only | Citations may contextualize, never fill values | Report effect, interval, direction, scope, and failures; keep held-out and external suites separate |
| Discussion/Conclusion | Formal results plus prespecified limitations | Closest antecedents for comparison after evidence exists | Interpret without causal overreach or universal ELA claims; distinguish evaluated query configurations |
| Reproducibility | Scientific randomization, versions, state alignment, feature definitions, transition and analysis rules | E2 for reporting completeness | Enable reconstruction without describing the local directory, filenames, storage layout, or shell interface |

## Source discrepancies and exclusions

- The current `source_index.md` was produced before the present contents of `docs/90_literature` stabilized: its `REF005` path is no longer present, while the current DynamoRep PDF is not indexed there. This map treats the five files now present in `docs/90_literature` as the local full-text corpus; the index should be refreshed by the owning research task rather than silently treating the stale row as a sixth paper.
- The two warm-starting papers are related but independent publications with different titles, experiments, and DOIs. They should be cited separately.
- No material under an `archive` directory, no withdrawn Phase 1 result, and no project-external historical experiment is used.
- The closest reviewed literature supports the motivation and task distinction. It does not support an absolute priority claim. The defensible wording is: “Among the literature reviewed here, we did not identify a method that makes this same pre-query, cost-adjusted decision.”
