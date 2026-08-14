Create a publication-ready scientific architecture figure for an Elsevier journal article:

“Decision-before-Feature: Deciding Whether to Execute a Landscape Descriptor Query in Black-box Optimization”

SCIENTIFIC MESSAGE

The proposed framework treats the execution of a predefined landscape descriptor query as a decision problem. Before computing query features, it observes low-cost, algorithm-agnostic search behavior from the current optimizer trajectory. An offline-trained Decision Model produces a decision score and determines whether the fixed query should be executed. If the query is executed, a fixed query-specific Selection Reference selects a portfolio action under the remaining budget.

The main query is “descriptor_cheap”, a predefined 16-dimensional low-cost landscape descriptor query. Do not describe it as complete ELA or complete pflacco.

The framework uses offline trajectory collection and supervised learning. It does not perform online controller training or reinforcement learning.

OVERALL COMPOSITION

Use a clean horizontal scientific workflow with an aspect ratio of approximately 2.1:1, suitable for a full-width Elsevier manuscript figure.

Organize the figure into two large horizontal regions:

A. Offline Data Construction and Model Training
B. Online Decision-before-Feature Process

Use a continuous left-to-right reading direction.

Separate the two regions with soft pastel background bands rather than heavy borders. Connect the trained offline components to the online process using vertical dashed arrows labeled “Frozen model transfer”.

Use short labels, generous spacing, orthogonal connectors, and no crossing arrows.

REGION A — OFFLINE DATA CONSTRUCTION AND MODEL TRAINING

Create four visually connected subsections.

SUBSECTION A1 — TRAJECTORY AND BEHAVIOR DATA

Draw:

“BBOB Training Problems”
Small text:
“10D · 20D · 40D”
“Function-family split”

→

“Offline Portfolio Trajectories”
Show four equal small tags:
“DE”
“PSO”
“CMA-ES”
“SHADE”

→

“Complete Native-update Histories”
Small text:
“Dynamic budget/event states”
“w02 · w05 · w10”

Use a small trajectory motif showing multiple complete optimizer updates. Make clear that the behavior windows are derived from complete native-update histories rather than sparse formal checkpoints.

→

“Algorithm-agnostic, Permutation-invariant Behavior State”

Inside this module, organize the behavior information into five compact rows:

“Progress and improvement”
“Population diversity”
“Population-set dynamics”
“Fitness-distribution change”
“Stagnation and distance decay”

Add a small muted annotation:

“Decision inputs: behavior only”
“Excluded: query features, function ID, algorithm ID, and optimizer internals”

SUBSECTION A2 — PAIRED OFFLINE CONTINUATIONS

From a common module titled:

“Shared Complete Checkpoint State”

create two parallel branches.

Upper branch:

“No Query”
→
“Native Continuation of the Default Solver”
→
“No-query Performance P_skip”

Lower branch:

“Run Fixed Landscape Descriptor Query”
→
“16-D descriptor_cheap Features”
→
“Query-specific Selection Reference”
→
“Selected-action Continuation”
→
“Query-path Performance p_query”

Place a shared note beneath both branches:

“Same complete checkpoint state”
“Equal total function-evaluation budget”

Do not merge query samples into the continuation population.

SUBSECTION A3 — FIXED SELECTION REFERENCE

Place a separate pale-gray or pale-blue inset below the query branch:

“Fixed Downstream Selection Reference”

Show its internal flow:

“Four Unique Portfolio Actions”
→
“Observed Statewise Action Losses”
→
“Multi-output RandomForestRegressor”
→
“Predicted Normalized Action Losses”
→
“Minimum-loss Action”

Inside the four-action module, write:

“continue_current”
“+ the other three portfolio algorithms”

Show the model inputs:

“Query features”
“Behavior state”
“Continuous remaining-budget ratio”

Show the target label:

“Statewise Min–max Observed Action Loss”

Optionally display the compact mathematical annotation:

L̃(s_t,a)
=
[L(s_t,a) − min_b L(s_t,b)]
/
max[max_b L(s_t,b) − min_b L(s_t,b), 10⁻¹²]

Add a muted subtitle:

“Fixed downstream component — not a Decision Model candidate”

SUBSECTION A4 — QUERY UTILITY AND DECISION MODEL

Merge “P_skip” and “p_query” into a highlighted module:

“Offline Query-utility Labels”

Display:

U_query(s_t)
=
[P_skip − p_query]
− λ_T C_T(s_t)
− λ_M C_M(s_t)

Add a short footnote:

“Query sampling FEs are represented by the reduced continuation budget.”

Connect:

“Behavior State”
+
“Query-utility Labels”

→

“Supervised Decision Dataset”

→

three equal Decision Model candidate cards:

“LDA”
“Logistic Regression”
“Ridge”

→

“Nested Function-family OOF Model Selection”

Small text:

“Decision utility as the primary criterion”

→

“Full BBOB-train Family-OOF Scores”

→

“Freeze Decision Threshold θ_OOF”

→

a highlighted final training output:

“Frozen Decision Controller”
containing:
“Preprocessing”
“Selected Decision Model”
“θ_OOF”

SEARCH MATURITY REPRESENTATION

From “Behavior State”, create two model-input routes:

1. A solid direct route:
“Direct Behavior Representation”
→ Decision Model candidates

2. A secondary dashed route:
“Search Maturity Representation”
“M_t = ES_t(1 − XS_t)”
“Maturity-aware evaluated variant”
→ Decision Model candidates

Make clear that Search Maturity is a behavior-derived representation under evaluation, not a mandatory proven stage, not equivalent to convergence, and not necessarily monotonic over time.

HELD-OUT EVALUATION

At the far right of Region A, add a light-gray dashed box:

“Held-out Evaluation Only”

List:

“BBOB Validation”
“CEC2017”
“CEC2022”
“Engineering Problems”

Only allow arrows from the frozen models toward this evaluation box. No arrow may return from held-out data to preprocessing, model selection, feature selection, or threshold fitting.

REGION B — ONLINE DECISION-BEFORE-FEATURE PROCESS

Create a visually simpler and more prominent deployment workflow.

Draw:

“Unseen Black-box Problem”

→

“Default / Prefix Optimizer Probe”

→

“Complete Native-update History”

→

“Algorithm-agnostic Behavior State”

→

“Frozen Decision Model”

→

“Decision Score s(x)”

→

a prominent diamond:

“s(x) > θ_OOF ?”

Create two clearly separated branches.

LOWER NO-QUERY BRANCH

Label:
“No: s(x) ≤ θ_OOF”

→

“No Query”

→

“Native Continuation of Current / Default Optimizer”

→

“Optimization with Remaining Budget”

→

“Final Solution”

UPPER RUN-QUERY BRANCH

Label:
“Yes: s(x) > θ_OOF”

→

“Run Fixed Landscape Descriptor Query”

→

“Query Features”

→

“Frozen Query-specific Selection Reference”

Feed two additional side inputs into the Selection Reference:

“Behavior State”
“Continuous Remaining-budget Ratio”

Then draw:

“Predicted Losses for Four Unique Actions”

→

“Select Minimum-loss Action”

→

a smaller decision diamond:

“Selected Action = Prefix Algorithm?”

Yes branch:

“Native Optimizer-state Continuation”

No branch:

“One Population-transfer Initialization”
→
“Run Selected Optimizer”

Merge both branches into:

“Optimization with Remaining Budget”

→

“Final Solution”

Add a small record box near the final action:

“Selection Relations”

Inside it list:

“selected_equals_default”
“selected_equals_prefix”
“handoff_required = not selected_equals_prefix”

Small note:

“handoff_required ↔ population_transfer_initialization”

ELSEVIER-ORIENTED VISUAL STYLE

Use a polished but restrained visual style suitable for an Elsevier optimization, engineering, or artificial-intelligence journal:

- flat editable vector artwork;
- white main background;
- softly tinted section bands;
- rounded rectangular cards with subtle 3–5 mm corner radii;
- no shadows or only an extremely subtle flat separation;
- dark charcoal text instead of pure black;
- muted teal for search behavior;
- soft blue for offline data and trained components;
- restrained terracotta or amber for Query Utility and the Decision-before-Feature decision;
- muted green for optimizer continuation and final output;
- light gray-blue for the fixed Selection Reference and held-out evaluation;
- color-blind-safe palette;
- distinguish modules through both color and border style so the figure remains interpretable in grayscale;
- use a slightly stronger central highlight around “Decision-before-Feature”;
- allow only small abstract scientific glyphs, such as trajectory lines, population dots, or budget bars;
- no decorative illustrations;
- use Arial, Helvetica, Source Sans, or a similar sans-serif font;
- use Times or STIX for mathematical expressions;
- maintain readable text after reduction to approximately 190 mm width;
- use consistent 1.0–1.3 pt connector lines;
- use solid arrows for data and runtime flow;
- use dashed arrows for frozen-model transfer and evaluated variants;
- align all modules to a common grid;
- reserve clear connector lanes;
- avoid more than three visual hierarchy levels;
- do not include an Elsevier logo or imitate any journal branding;
- export as editable SVG, PDF, or EPS, with a 600 dpi preview.

No 3D rendering, gradients, glossy effects, strong shadows, neon colors, photographs, cartoons, robots, brains, neural-network decorations, chips, circuit boards, or laboratory imagery.

Do not use “ELA Utility”, “Run ELA”, “Skip ELA”, “P_ELA”, or “Estimated ELA Utility”. Use “Query Utility”, “Run Fixed Landscape Descriptor Query”, “No Query”, “p_query”, and “Decision Score s(x)”.

Do not describe descriptor_cheap as complete ELA or complete pflacco. Do not include directional entropy as an active behavior input.

Do not place query features, remaining_budget_ratio, function ID, algorithm ID, optimizer-specific parameters, or optimizer internal state into the Decision Model.

Do not show Random Forest, XGBoost, LightGBM, MLP, or SVM as active Decision Model candidates. Random Forest belongs only to the fixed downstream Selection Reference.

Do not draw online training, reinforcement learning, model updates during deployment, random instance splits, or test-data feedback into training.

Do not draw five actions. Use exactly four unique actions: continue_current plus the other three portfolio algorithms.

Do not call the minimum observed action an oracle or VBS. Do not use counterfactual terminology. Do not label population-transfer initialization as native continuation, restart, or generic warm start.

Do not subtract query sampling FEs twice. Do not depict Search Maturity as monotonic, causally established, equivalent to convergence, or a mandatory stage.

Avoid crowded text, excessive formulas, crossing arrows, decorative icons, fabricated numerical results, performance claims, and journal logos.