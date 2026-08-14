# Motivation Options After Research

## Status

The user has already established the controlling motivation: before computing descriptors from a prespecified independent landscape query, predict whether its expected downstream value justifies its resource cost using only pre-query, algorithm-agnostic search behavior. The options below are **not competing research directions**. They are three phrasings of the same paper spine for wording confirmation and later section-level use.

The frozen resource interpretation must remain explicit. Query FEs are charged through the reduced continuation budget; query and selection time enter the primary utility; memory is measured and reported, while the current primary phase-one label retains \(\lambda_M=0\). A nonzero memory penalty would require a separately fixed sensitivity setting rather than an unannounced change to the primary estimand.

## Option A — Decision-theoretic phrasing (recommended for the paper spine)

**Motivation.** Landscape descriptors may improve a downstream portfolio decision, but their state-specific value is unknown until after the resources needed to acquire them have been spent. The paper therefore asks whether a prespecified landscape-descriptor query should be executed at the current search state, using only algorithm-agnostic behavior observable before the query and evaluating the decision by cost-adjusted downstream optimization utility.

**Why this phrasing works.** It foregrounds the new decision object and information time. It distinguishes the paper from both conventional AAS, which selects an algorithm after features exist, and trajectory-based per-run selection, which constructs a trajectory representation and directly recommends a second optimizer.

## Option B — Resource-accounting phrasing

**Motivation.** A landscape query is worthwhile only when the downstream performance difference it enables compensates for the optimization budget consumed by its samples and the remaining computation and memory burden. Decision-before-Feature treats this trade-off as a state-conditional prediction problem: estimate query utility from pre-query search behavior, then acquire descriptors only when the frozen score and threshold support the call.

**Why this phrasing works.** It is strongest for Experimental Setup and Discussion because it places equal-total-FE accounting, measured time, peak memory, and policy performance in one argument. It must retain the current \(\lambda_M=0\) boundary when describing the primary label.

## Option C — Trajectory-information phrasing

**Motivation.** A cheap optimizer prefix already contains information about progress, diversity, entropy, stagnation, and distance dynamics. Rather than using those observations to replace landscape descriptors or directly choose an optimizer, the paper tests a narrower question: whether this pre-query behavior is sufficient to predict when an independent fixed descriptor query is likely to repay its acquisition cost.

**Why this phrasing works.** It connects most directly to Jankovic et al., Kostovska et al., DynamoRep, search-trajectory representations, and behavioral-analysis work. It also makes the Time-only Controller essential: behavior is useful only if it contributes decision value beyond elapsed FE ratio.

## Shared non-negotiable content

Whichever wording is used in a given section, it must preserve all of the following:

1. **Decision object:** execute or skip a fixed query; not choose an optimizer, operator, or parameter.
2. **Information time:** the gate acts before query descriptors are generated.
3. **Inputs:** algorithm-agnostic native-history behavior and permitted budget context only.
4. **Target:** downstream query-versus-skip utility under matched states and a common total FE budget.
5. **Costs:** query FEs, measured query/selection time, and reported memory burden without double counting or silently changing \(\lambda_M\).
6. **Architecture:** the query gate and the fixed downstream action-loss selector remain separate supervised tasks.
7. **Evaluation:** model and threshold selection use BBOB-training function families only; BBOB validation and external suites are evaluation-only.
8. **Claim boundary:** the current contribution is the formal problem and protocol. Predictive advantage, resource benefit, BBOB held-out performance, CEC transport, engineering-problem transport, and Search Maturity associations remain empirical TBDs.

## Recommended use by section

- Use **Option A** as the controlling Introduction and contribution wording.
- Use **Option C** to bridge trajectory/behavior literature into the precise Related Work gap.
- Use **Option B** to organize Experimental Setup, Results, and Discussion around FE, time, memory, final loss, and query-call quality.

## Corpus-scoped gap statement

Among the literature reviewed here, we did not identify a method that, before descriptors from a prespecified independent landscape query exist, uses only algorithm-agnostic search behavior to predict the query's downstream utility after coherent FE and computational-cost accounting. This statement is limited to the reviewed corpus and is not an absolute priority claim.
