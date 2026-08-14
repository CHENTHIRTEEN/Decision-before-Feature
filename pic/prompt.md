Design a publication-ready scientific architecture diagram for an IEEE Transactions paper titled:

“Decision-before-Feature for Resource-aware Algorithm Selection in Black-box Optimization”

Scientific message:
The framework decides whether a predefined landscape descriptor query is worth executing before computing its query features. It uses low-cost, algorithm-agnostic search behavior collected from the current optimization trajectory, applies an offline-trained decision model, and executes the fixed query only when the predicted decision score exceeds an out-of-fold threshold. The method uses offline trajectory collection and supervised learning; it is not online reinforcement learning or online controller training.

FIGURE LAYOUT
Create a horizontal two-panel workflow with an aspect ratio of approximately 2:1, optimized for an IEEE double-column width of 178 mm.

Use two horizontal swimlanes:

(a) Offline Supervision and Model Construction
(b) Online Decision-before-Feature Inference

The reading order must be strictly left-to-right. Use orthogonal arrows, aligned modules, ample white space, no connector crossings, and no long prose inside boxes. All labels inside the figure must be in concise academic English.

PANEL (a): OFFLINE SUPERVISION AND MODEL CONSTRUCTION

Draw the following main flow:

1. “BBOB Training Problems”
   Small labels:
   “10D / 20D / 40D”
   “Function-family split”

2. “Native Optimizer Trajectories”
   Show four equal portfolio members:
   “DE”, “PSO”, “CMA-ES”, “SHADE”

3. “Dynamic Budget/Event States”
   Include:
   “Complete optimizer state”
   “Complete native-update history”
   “w02 / w05 / w10”

   Visually indicate that behavior windows are derived from consecutive complete native optimizer updates, not sparse checkpoints.

4. “Algorithm-agnostic, Permutation-invariant Behavior State”
   List only compact behavior categories:
   “Progress and improvement”
   “Diversity”
   “Population-set dynamics”
   “Fitness-distribution change”
   “Stagnation and distance decay”

   Add a small gray annotation:
   “Decision inputs: behavior only”
   “Excluded: query features, function ID, algorithm ID, optimizer internals”

5. From each “Shared Complete Checkpoint State”, construct two parallel offline continuation branches:

   Upper branch:
   “No Query”
   → “Native Continuation of Default Solver”
   → “No-query Performance P_skip”

   Lower branch:
   “Run Fixed Landscape Descriptor Query”
   → “16-D descriptor_cheap Features”
   → “Query-specific Selection Reference”
   → “Selected-action Continuation”
   → “Query-path Performance p_query”

   Annotate that both branches start from the same complete checkpoint state and use the same total function-evaluation budget.

6. Beside the query branch, place a visually separate light-gray module titled:

   “Fixed Downstream Selection Reference”

   Show:
   “Four Unique Actions”
   “continue_current + three other portfolio algorithms”

   Input:
   “Query features + behavior state + continuous remaining budget”

   Model:
   “Multi-output RandomForestRegressor”

   Target:
   “Statewise Min–max Observed Action Loss”

   Selection rule:
   “Select action with minimum predicted normalized loss”

   Make it visually explicit that this Random Forest is a fixed downstream Selection Reference, not a Decision Model candidate and not the main methodological contribution.

7. Merge P_skip and p_query into:

   “Offline Query-utility Labels”

   Display the exact formula:

   U_query(s_t)
   = [P_skip − p_query]
   − λ_T C_T(s_t)
   − λ_M C_M(s_t)

   Add a small note:
   “Query sampling FEs are represented by the reduced continuation budget; do not subtract them twice.”

8. Connect:
   “Behavior State + Query-utility Label”
   → “Supervised Decision Dataset”

   Then show three equal Decision Model candidates:
   “LDA”
   “Logistic Regression”
   “Ridge”

   Connect them to:
   “Nested Function-family OOF Model Selection”
   Small label:
   “Decision utility as primary criterion”

   Then:
   “Full BBOB-train Family-OOF Scores”
   → “Freeze θ_OOF”

   Final offline output:
   “Frozen Preprocessing”
   “Selected Decision Model”
   “OOF Decision Threshold θ_OOF”

9. Show Search Maturity as a behavior-derived evaluated representation rather than an obligatory proven stage:

   From “Behavior State”, draw:
   - a solid direct path to the Decision Model candidates;
   - a secondary dashed path through:

     “Search Maturity Representation”
     “M_t = ES_t(1 − XS_t)”
     “Maturity-aware evaluated variant”

   Do not depict Search Maturity as monotonic convergence or as an already verified causal mechanism.

10. On the far right, place a dashed gray box:

   “Held-out Evaluation Only”

   Include:
   “BBOB Validation”
   “CEC2017”
   “CEC2022”
   “Engineering Problems”

   Allow arrows only from the frozen model toward this box. Do not draw any arrow returning from evaluation data to preprocessing, model selection, or threshold fitting.

PANEL (b): ONLINE DECISION-BEFORE-FEATURE INFERENCE

Draw the following deployment flow:

1. “Unseen Black-box Problem”

2. “Default / Prefix Optimizer Probe”

3. “Complete Native-update History”

4. “Algorithm-agnostic Behavior State”

5. “Frozen Decision Model”

6. “Decision Score s(x)”

7. A prominent diamond decision node:
   “s(x) > θ_OOF ?”

Create two clearly separated branches:

NO BRANCH:
“s(x) ≤ θ_OOF”
→ “No Query”
→ “Native Continuation of Current / Default Optimizer”
→ “Optimization with Remaining Budget”
→ “Final Solution”

YES BRANCH:
“s(x) > θ_OOF”
→ “Run Fixed Landscape Descriptor Query”
→ “Query Features”
→ “Frozen Query-specific Selection Reference”

Feed the following side inputs into the Selection Reference:
“Behavior State”
“Continuous Remaining-budget Ratio”

Then:
“Predicted Normalized Losses for Four Unique Actions”
→ “Select Minimum-loss Action”

Add a second decision node:
“Selected Action = Prefix Algorithm?”

YES:
“Native Optimizer-state Continuation”

NO:
“One Population-transfer Initialization”
→ “Run Selected Optimizer”

Merge both paths into:
“Optimization with Remaining Budget”
→ “Final Solution”

Add a small output-record box:
“Selection Relations”

List:
“selected_equals_default”
“selected_equals_prefix”
“handoff_required = not selected_equals_prefix”

Add:
“handoff_required ↔ population_transfer_initialization”

VISUAL STYLE

Use a restrained IEEE Transactions visual language:

- clean flat 2D vector diagram;
- white background;
- dark navy outlines and primary arrows;
- muted teal for algorithm-agnostic behavior and Search Maturity;
- muted amber for Query Utility and the Decision-before-Feature decision node;
- dark green for optimizer actions and final optimization;
- light gray for the fixed Selection Reference, annotations, and evaluation-only modules;
- color-blind-safe palette with sufficient grayscale contrast;
- small corner radii;
- standard diamonds only for decisions;
- uniform 0.8–1.2 pt strokes;
- Helvetica, Arial, or Source Sans for labels;
- Times or STIX style for mathematical notation;
- no more than three visual hierarchy levels;
- readable after reduction to 178 mm width;
- precise alignment and balanced margins;
- solid arrows for runtime or data flow;
- dashed arrows for trained-model transfer and evaluated variants;
- output as clean editable SVG/PDF-style vector artwork or 600 dpi raster artwork.

Do not use 3D objects, gradients, shadows, glow, glass effects, neon colors, photographs, cartoon characters, brains, robots, neural-network decorations, chips, circuit boards, or laboratory imagery.

Do not use the labels “ELA Utility”, “Run ELA”, “Skip ELA”, “P_ELA”, or “Estimated ELA Utility”. Use “Query Utility”, “Run Fixed Landscape Descriptor Query”, “No Query”, “p_query”, and “Decision Score s(x)”.

Do not describe descriptor_cheap as complete ELA or complete pflacco. Do not include directional entropy as an active behavior input.

Do not place query features, function ID, algorithm ID, optimizer-specific parameters, or optimizer internal state into the Decision Model input.

Do not present Random Forest, XGBoost, LightGBM, MLP, or SVM as Decision Model candidates. Random Forest belongs only to the fixed downstream Selection Reference.

Do not connect remaining_budget_ratio to the Decision Model. It is an input to the Selection Reference.

Do not draw online model training, reinforcement learning, model-parameter feedback loops, or test-data feedback into training.

Do not allow BBOB Validation, CEC2017, CEC2022, or engineering problems to influence preprocessing, feature selection, model selection, or threshold fitting.

Do not draw five actions as continue_current plus all four algorithms. There must be exactly four unique actions: continue_current plus the other three portfolio algorithms.

Do not call the minimum observed action loss an oracle or VBS. Do not use counterfactual terminology. Do not label cross-algorithm switching as native continuation, restart, or generic warm start; use “population-transfer initialization”.

Do not subtract query sampling FEs twice. Do not depict Search Maturity as monotonically increasing, equivalent to convergence, causally established, or an obligatory proven module.

Do not include fabricated performance values, accuracy numbers, causal claims, crowded text, curved crossing arrows, or decorative icons.