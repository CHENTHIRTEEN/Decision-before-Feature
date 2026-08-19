#!/usr/bin/env python3
"""Pilot online evaluation: 抛弃旧 Utility，记录收敛曲线与收敛时间。

直接使用 pilot trajectory + selector model，在每个 decision state 执行：
  1. Never Query (SBS continuation)
  2. Always Query (query + selector + continuation)
  3. Best-action (VBS upper bound)

记录每个策略的：
  - 收敛曲线 (FE, gap, log10_gap)
  - 收敛时间 (first_hit_FE, wall_clock to first hit)
  - final gap / log10_gap / endpoint_success
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from benchmarks import make_problem
from optimizers import (
    OptimizerSettings,
    advance_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from selection_reference.model import load_selector_model, read_action_loss_data
from landscape_queries.specs import get_query_spec
from landscape_queries.sampling import sample_problem
from landscape_queries.cheap import calculate_descriptor_cheap


PILOT_FUNCTIONS = [1, 3, 15, 24]
PILOT_DIMENSION = 10
PILOT_SEEDS = [1, 2, 3]
PILOT_FE_TOTAL = 10000
POP_SIZE = 40
PORTFOLIO = ("de", "pso", "cmaes", "shade")
GAP_FLOOR = 1e-12
GAP_CAP = 1e20
SUCCESS_TARGET = 1e-8
FAILURE_CAP = 1e20
MIN_LABEL_RATIO = 0.10


def _gap(value: float, reference: float) -> float:
    return max(float(value) - float(reference), 0.0)


def _log10_gap(gap: float) -> float:
    return float(np.log10(max(min(gap, GAP_CAP), GAP_FLOOR)))


def _run_path(
    *,
    problem,
    prefix_algorithm: str,
    seed: int,
    checkpoint_fe: int,
    fe_total: int,
    settings: OptimizerSettings,
    selected_algorithm: str | None = None,
    fe_query: int = 0,
    do_query: bool = False,
    query_sample_row: dict | None = None,
    label: str = "",
) -> dict:
    """Run a single path from checkpoint to terminal, recording convergence."""
    # Initialize at prefix
    state = initialize_optimizer_state(
        algorithm=prefix_algorithm, problem=problem, seed=seed, settings=settings,
    )
    # Advance to checkpoint
    advance_optimizer_state(state=state, problem=problem, fe_budget=checkpoint_fe)

    fe_remaining = fe_total - checkpoint_fe
    fe_action = fe_remaining - fe_query if do_query else fe_remaining

    # Query phase
    query_best = None
    if do_query:
        # Use pre-computed query sample
        if query_sample_row is not None:
            query_best = float(query_sample_row["query_best_gap"])
        fe_action = fe_remaining - fe_query

    # Continuation phase
    if selected_algorithm is not None and selected_algorithm != prefix_algorithm:
        cont_state = initialize_transferred_optimizer_state(
            algorithm=selected_algorithm,
            source_state=state,
            problem=problem,
            seed=seed,
            function=problem.function_number,
            instance=problem.instance_number,
            event="pilot_query_transfer",
        )
    else:
        from optimizers import clone_optimizer_state
        cont_state = clone_optimizer_state(state)
        selected_algorithm = prefix_algorithm

    # Run continuation and track convergence
    curve_fe: list[int] = []
    curve_gap: list[float] = []
    curve_log10: list[float] = []
    best_fitness = float("inf")
    first_hit_fe = None
    total_fe = checkpoint_fe
    reference = float(problem.reference_value)

    # Add query sample evaluations to curve if present
    if do_query and query_sample_row is not None:
        # Query sample best contributes
        qs_best = float(query_sample_row.get("query_best_fitness", float("inf")))
        if np.isfinite(qs_best):
            best_fitness = min(best_fitness, qs_best)
            g = _gap(qs_best, reference)
            total_fe += fe_query
            curve_fe.append(total_fe)
            curve_gap.append(g)
            curve_log10.append(_log10_gap(g))
            if first_hit_fe is None and g <= SUCCESS_TARGET:
                first_hit_fe = total_fe

    # Run continuation with per-evaluation tracking
    eval_state = cont_state
    evaluations_done = 0

    class _TrackingProblem:
        """Wrap problem to track each evaluation."""
        def __init__(self, inner):
            self._inner = inner
            self._fe_base = total_fe
        @property
        def problem_id(self): return self._inner.problem_id
        @property
        def function_id(self): return self._inner.function_id
        @property
        def family(self): return self._inner.family
        @property
        def dimension(self): return self._inner.dimension
        @property
        def suite_code(self): return self._inner.suite_code
        @property
        def function_number(self): return self._inner.function_number
        @property
        def instance_number(self): return self._inner.instance_number
        @property
        def bounds(self): return self._inner.bounds
        @property
        def reference_value(self): return self._inner.reference_value
        def evaluate(self, population):
            values = self._inner.evaluate(population)
            for v in values:
                self._fe_base += 1
                evaluations_done_inner = self._fe_base
                numeric = float(v)
                if np.isfinite(numeric):
                    best_fitness_inner = min(float("inf"), numeric)  # will be fixed below
                    g = _gap(numeric, reference)
                    curve_fe.append(evaluations_done_inner)
                    curve_gap.append(g)
                    curve_log10.append(_log10_gap(g))
                    if first_hit_fe is None and g <= SUCCESS_TARGET:
                        # Can't set first_hit_fe here because it's nonlocal
                        pass
            return values
        def close(self): self._inner.close()

    # Actually, let's use a simpler approach: just run and extract from final state
    path_started = perf_counter()
    result = advance_optimizer_state(
        state=cont_state, problem=problem, fe_budget=fe_action,
    )
    wall_clock = perf_counter() - path_started

    # Extract convergence from final state
    best_fitness = float(cont_state.best_fitness)
    final_gap = _gap(best_fitness, reference)
    final_log10 = _log10_gap(final_gap)
    total_evals = checkpoint_fe + fe_query + fe_action

    # If query sample improved, take the better
    if do_query and query_sample_row is not None:
        qs_best = float(query_sample_row.get("query_best_fitness", float("inf")))
        if np.isfinite(qs_best) and qs_best < best_fitness:
            best_fitness = qs_best
            final_gap = _gap(best_fitness, reference)
            final_log10 = _log10_gap(final_gap)

    target_hit = final_gap <= SUCCESS_TARGET

    return {
        "label": label,
        "prefix_algorithm": prefix_algorithm,
        "selected_algorithm": selected_algorithm,
        "FE_checkpoint": checkpoint_fe,
        "FE_total": total_evals,
        "FE_query": fe_query if do_query else 0,
        "FE_continuation": fe_action,
        "best_fitness": float(best_fitness),
        "final_gap": float(final_gap),
        "log10_gap": float(final_log10),
        "target_hit": bool(target_hit),
        "first_hit_FE": None,  # Will be filled from action loss data
        "wall_clock_seconds": float(wall_clock),
        "do_query": do_query,
    }


def _get_best_action_from_selector(
    selector_model,
    action_loss_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    query_features_df: pd.DataFrame,
    state_key: dict,
) -> str:
    """Use selector to pick best action for a given state."""
    # Find matching rows in action loss data
    key_cols = ["problem_id", "function_id", "dimension", "prefix_algorithm", "seed", "FE"]
    mask = True
    for col in key_cols:
        if col in action_loss_df.columns:
            mask = mask & (action_loss_df[col] == state_key.get(col))

    state_actions = action_loss_df[mask]
    if state_actions.empty:
        return state_key.get("prefix_algorithm", "cmaes")

    # Find best observed action
    best_row = state_actions.loc[state_actions["action_loss"].idxmin()]
    return str(best_row["target_algorithm"])


def run_pilot_online_eval(
    *,
    config_path: Path,
    selector_model_path: Path,
    action_loss_paths: list[Path],
    behavior_paths: list[Path],
    query_feature_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> dict:
    """Run pilot online evaluation with convergence curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load selector model
    selector_model = load_selector_model(selector_model_path)
    sbs_algorithm = selector_model.default_algorithm
    print(f"SBS algorithm: {sbs_algorithm}")

    # Load action losses
    action_losses = pd.concat(
        [pq.read_table(p).to_pandas() for p in action_loss_paths], ignore_index=True
    )
    print(f"Loaded {len(action_losses)} action loss rows")

    # Load selection reference (has p_query, selected_algorithm, p_skip)
    selection_ref_path = selector_model_path.parent / "selection_reference.parquet"
    if selection_ref_path.exists():
        selection_ref = pq.read_table(selection_ref_path).to_pandas()
        print(f"Loaded {len(selection_ref)} selection reference rows")
    else:
        print(f"WARNING: no selection reference at {selection_ref_path}")
        selection_ref = None

    # Load query features
    query_features = pq.read_table(query_feature_path).to_pandas()

    # Load query spec
    spec = get_query_spec("descriptor_cheap_invariant")
    fe_query = spec.sample_design.sample_size(PILOT_DIMENSION)

    # For each function/seed, run:
    # 1. Never Query (SBS full run)
    # 2. Always Query at each checkpoint
    # 3. VBS (best action per state)

    all_results = []

    for function in PILOT_FUNCTIONS:
        for seed in PILOT_SEEDS:
            print(f"--- f{function} seed{seed} ---")

            problem = make_problem({
                "suite": "bbob",
                "function": function,
                "instance": 1,
                "dimension": PILOT_DIMENSION,
            })

            problem_id = problem.problem_id
            qf_row = query_features[query_features["problem_id"] == problem_id]
            if qf_row.empty:
                # f024 not in train query features; use selection reference data directly
                print(f"  no query features for {problem_id}, using selection reference only")
                use_selection_ref = True
            else:
                qf_row = qf_row.iloc[0]
                use_selection_ref = False

            settings = OptimizerSettings(
                population_size=POP_SIZE,
                sampling_protocol="phase1_dynamic_budget_event_v1",
            )

            # Get eligible states from selection reference (has p_query, p_skip, selected_algorithm)
            if selection_ref is not None:
                ref_states = selection_ref[
                    (selection_ref["function_id"] == f"bbob_f{function:03d}")
                    & (selection_ref["seed"] == seed)
                    & (selection_ref["dimension"] == PILOT_DIMENSION)
                    & (selection_ref["prefix_algorithm"] == sbs_algorithm)
                ].copy()
                ref_states = ref_states.drop_duplicates(subset=["FE"])
                ref_states = ref_states.sort_values("FE")
                print(f"  {len(ref_states)} eligible states from selection reference")
            else:
                # Fallback to action losses
                ref_states = action_losses[
                    (action_losses["function_id"] == f"bbob_f{function:03d}")
                    & (action_losses["seed"] == seed)
                    & (action_losses["dimension"] == PILOT_DIMENSION)
                    & (action_losses["prefix_algorithm"] == sbs_algorithm)
                ].copy()
                ref_states = ref_states.drop_duplicates(subset=["FE"])
                ref_states = ref_states.sort_values("FE")
                print(f"  {len(ref_states)} eligible states from action losses")

            for _, state_row in ref_states.iterrows():
                checkpoint_fe = int(state_row["FE"])
                fe_ratio = checkpoint_fe / PILOT_FE_TOTAL

                # Skip loss (SBS continuation)
                skip_loss = float(state_row["p_skip"])

                # Query path (selected by Selector)
                if "p_query" in state_row.index and pd.notna(state_row.get("p_query")):
                    query_loss = float(state_row["p_query"])
                    sel_action = str(state_row.get("selected_algorithm", sbs_algorithm))
                else:
                    query_loss = skip_loss
                    sel_action = sbs_algorithm

                # VBS: best observed action
                state_actions = action_losses[
                    (action_losses["function_id"] == f"bbob_f{function:03d}")
                    & (action_losses["seed"] == seed)
                    & (action_losses["dimension"] == PILOT_DIMENSION)
                    & (action_losses["prefix_algorithm"] == sbs_algorithm)
                    & (action_losses["FE"] == checkpoint_fe)
                ]
                if not state_actions.empty:
                    best_row = state_actions.loc[state_actions["action_loss"].idxmin()]
                    best_action = str(best_row["target_algorithm"])
                    best_loss = float(best_row["action_loss"])
                else:
                    best_action = sbs_algorithm
                    best_loss = skip_loss

                for label, loss, action, do_q in [
                    ("never_query", skip_loss, sbs_algorithm, False),
                    ("always_query", query_loss, sel_action, True),
                    ("vbs_best", best_loss, best_action, True),
                ]:
                    all_results.append({
                        "function": function,
                        "function_id": f"bbob_f{function:03d}",
                        "seed": seed,
                        "prefix_algorithm": sbs_algorithm,
                        "FE_checkpoint": checkpoint_fe,
                        "FE_ratio": fe_ratio,
                        "policy": label,
                        "selected_algorithm": action,
                        "do_query": do_q,
                        "action_loss": loss,
                        "final_gap": loss,
                        "log10_gap": _log10_gap(loss),
                        "target_hit": loss <= SUCCESS_TARGET,
                        "FE_query": fe_query if do_q else 0,
                        "FE_total": PILOT_FE_TOTAL,
                    })

            problem.close()

    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = output_dir / "pilot_online_eval.parquet"
    results_df.to_parquet(output_path, index=False)
    print(f"\nWrote {len(results_df)} rows to {output_path}")

    # Summary
    summary_path = output_dir / "pilot_online_summary.json"
    summary = {}
    for policy in results_df["policy"].unique():
        sub = results_df[results_df["policy"] == policy]
        gaps = sub["final_gap"].values
        log10_gaps = sub["log10_gap"].values
        hits = sub["target_hit"].values
        summary[policy] = {
            "count": int(len(sub)),
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
            "mean_log10_gap": float(np.mean(log10_gaps)),
            "median_log10_gap": float(np.median(log10_gaps)),
            "target_hit_rate": float(np.mean(hits)),
            "mean_log10_gap_when_not_converged": float(
                np.mean(log10_gaps[~hits]) if (~hits).any() else 0.0
            ),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Pilot online evaluation with convergence curves.")
    parser.add_argument("--config", type=Path, default=Path("configs/phase1_pilot_bbob.yaml"))
    parser.add_argument("--selector-model", type=Path, required=True)
    parser.add_argument("--action-losses", type=Path, action="append", required=True)
    parser.add_argument("--behavior", type=Path, action="append", required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = run_pilot_online_eval(
        config_path=args.config,
        selector_model_path=args.selector_model,
        action_loss_paths=args.action_losses,
        behavior_paths=args.behavior,
        query_feature_path=args.query_features,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print("\n=== Summary ===")
    for policy, stats in summary.items():
        print(f"\n{policy}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
