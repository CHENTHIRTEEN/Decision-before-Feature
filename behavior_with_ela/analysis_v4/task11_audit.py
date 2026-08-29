"""Task 11 audit analysis: aggregate shards, calibrate short-horizon noise,
and run the action-space / horizon / upper-bound / source-history audits.

Steps: aggregate -> noise -> audit. Labels come only from the collected
multi-horizon branch outcomes; no model is trained in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SHARDS = ROOT / "behavior_with_ela/results/post_handoff/task11/shards"
HEAVY = ROOT / "behavior_with_ela/results/analysis_v4/task11"
LIGHT = ROOT / "behavior_with_ela/analysis_v4/task11"
ACTIONS = ("continue", "pso", "shade")
HORIZONS = ("500", "1000", "terminal")
BOOTSTRAP_STREAM = 2026083002


def json_write(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float))


def load_shards() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    states, branches, ledgers = [], [], []
    for shard in sorted(SHARDS.iterdir()):
        if not shard.is_dir():
            continue
        states.append(pd.read_parquet(shard / "states.parquet"))
        branches.append(pd.read_parquet(shard / "branches.parquet"))
        ledgers.append(pd.read_parquet(shard / "ledger.parquet"))
    return (
        pd.concat(states, ignore_index=True),
        pd.concat(branches, ignore_index=True),
        pd.concat(ledgers, ignore_index=True),
    )


def step_aggregate() -> None:
    HEAVY.mkdir(parents=True, exist_ok=True)
    LIGHT.mkdir(parents=True, exist_ok=True)
    states, branches, ledgers = load_shards()
    base = branches.loc[branches["replicate_id"].eq(0)].copy()

    # multi-horizon action outcome table (one row per state-action)
    outcome = base.copy()
    for horizon in HORIZONS:
        continue_loss = (
            outcome.loc[outcome["candidate_action"].eq("continue")]
            .set_index("state_id")[f"loss_{horizon}"]
            .rename("continue_loss")
        )
        outcome = outcome.merge(
            continue_loss, on="state_id", how="left", validate="many_to_one"
        )
        outcome[f"gain_{horizon}"] = (
            outcome["continue_loss"] - outcome[f"loss_{horizon}"]
        )
        outcome = outcome.drop(columns=["continue_loss"])
    for horizon in HORIZONS:
        pivot = outcome.pivot_table(
            index="state_id", columns="candidate_action", values=f"loss_{horizon}",
            aggfunc="first",
        )
        best = pivot.idxmin(axis=1)
        order = {action: index for index, action in enumerate(ACTIONS)}
        tie_min = pivot.min(axis=1)
        counts = pivot.eq(tie_min, axis=0).sum(axis=1)
        outcome[f"best_{horizon}"] = outcome["state_id"].map(
            best.to_dict()
        )
        outcome[f"tie_count_{horizon}"] = outcome["state_id"].map(counts.to_dict())
    outcome.to_parquet(HEAVY / "multi_horizon_action_outcomes.parquet", index=False)

    states.to_parquet(HEAVY / "mature_post_handoff_states.parquet", index=False)
    behavior_columns = ["state_id", "route", "source_algorithm", "FE", "dwell_FE"]
    behavior_columns += [c for c in states.columns if c.startswith(("bg_", "bs_"))]
    states[behavior_columns].to_parquet(
        HEAVY / "mature_state_behavior.parquet", index=False
    )
    branches.to_parquet(HEAVY / "short_horizon_repetitions.parquet", index=False)
    ledgers.to_parquet(HEAVY / "task11_resource_ledger.parquet", index=False)

    summary = {
        "states": int(len(states)),
        "branch_rows": int(len(branches)),
        "base_branches": int(len(base)),
        "sampled_states": int(states["sampled_for_repetition"].sum()),
        "ledger_total_base_fe": int(ledgers["base_route_fe"].sum()),
        "ledger_total_branch_fe": int(ledgers["branch_fe"].sum()),
        "wall_seconds_total": float(ledgers["wall_seconds"].sum()),
        "max_rss_mb": float(ledgers["max_rss_mb"].max()),
    }
    json_write(summary, LIGHT / "collection_summary.json")
    print(json.dumps(summary, indent=1))


def state_table() -> pd.DataFrame:
    states = pd.read_parquet(HEAVY / "mature_post_handoff_states.parquet")
    outcome = pd.read_parquet(HEAVY / "multi_horizon_action_outcomes.parquet")
    best = outcome.loc[outcome["candidate_action"].eq(outcome[f"best_terminal"])]
    pivots = {}
    for horizon in HORIZONS:
        pivots[horizon] = outcome.pivot_table(
            index="state_id", columns="candidate_action", values=f"gain_{horizon}",
            aggfunc="first",
        )
    return states, outcome, pivots


def conditional_entropy(frame: pd.DataFrame, group_columns: list[str], horizon: str) -> float:
    total = len(frame)
    value = 0.0
    for _, group in frame.groupby(group_columns, sort=False):
        counts = group[f"best_{horizon}"].value_counts().to_numpy(dtype=float)
        probabilities = counts / counts.sum()
        value += (len(group) / total) * float(
            -(probabilities * np.log2(probabilities)).sum()
        )
    return float(value)


def entropy_of(counts: pd.Series) -> float:
    values = counts.to_numpy(dtype=float)
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def function_balanced(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("cv_group_id")[column].mean().mean())


def step_noise() -> None:
    HEAVY.mkdir(parents=True, exist_ok=True)
    LIGHT.mkdir(parents=True, exist_ok=True)
    branches = pd.read_parquet(HEAVY / "short_horizon_repetitions.parquet")
    rows = []
    for horizon in HORIZONS:
        pivot_loss = branches.pivot_table(
            index=["state_id", "replicate_id"], columns="candidate_action",
            values=f"loss_{horizon}", aggfunc="first",
        ).reset_index()
        pivot_loss[f"gain_pso"] = pivot_loss["continue"] - pivot_loss["pso"]
        pivot_loss[f"gain_shade"] = pivot_loss["continue"] - pivot_loss["shade"]
        long = pivot_loss.melt(
            id_vars=["state_id", "replicate_id"],
            value_vars=["gain_pso", "gain_shade"],
            var_name="candidate_action", value_name="gain",
        )
        long["candidate_action"] = long["candidate_action"].str.replace("gain_", "")
        grouped = long.groupby(["state_id", "candidate_action"])["gain"]
        medians = grouped.transform("median")
        long["absolute_deviation"] = (long["gain"] - medians).abs()
        states = pd.read_parquet(HEAVY / "mature_post_handoff_states.parquet")[
            ["state_id", "cv_group_id", "suite"]
        ]
        long = long.merge(states, on="state_id", how="left", validate="many_to_one")
        long["horizon"] = horizon
        rows.append(long)
    deviations = pd.concat(rows, ignore_index=True)
    deviations.to_parquet(HEAVY / "short_horizon_noise_deviations.parquet", index=False)
    summary_rows = []
    for horizon in HORIZONS:
        subset = deviations.loc[deviations["horizon"].eq(horizon)]
        for suite_name, group in subset.groupby("suite", sort=False):
            per_function = group.groupby("cv_group_id")["absolute_deviation"]
            quantiles = per_function.quantile([0.50, 0.95]).unstack()
            summary_rows.append(
                {
                    "horizon": horizon,
                    "suite": suite_name,
                    "states_actions": int(len(group)),
                    "delta_50_function_balanced": float(quantiles[0.50].mean()),
                    "delta_95_function_balanced": float(quantiles[0.95].mean()),
                    "pooled_delta_50": float(group["absolute_deviation"].quantile(0.50)),
                    "pooled_delta_95": float(group["absolute_deviation"].quantile(0.95)),
                }
            )
    table = pd.DataFrame(summary_rows)
    table.to_parquet(LIGHT / "short_horizon_noise_deltas.parquet", index=False)
    print(table.to_string())


def _deltas() -> dict[tuple[str, str], float]:
    table = pd.read_parquet(LIGHT / "short_horizon_noise_deltas.parquet")
    pooled = table.groupby("horizon")[["pooled_delta_50", "pooled_delta_95"]].mean()
    return {
        (horizon, kind): float(pooled.loc[horizon, f"pooled_delta_{kind}"])
        for horizon in HORIZONS
        for kind in ("50", "95")
    }


def step_audit() -> None:
    LIGHT.mkdir(parents=True, exist_ok=True)
    states = pd.read_parquet(HEAVY / "mature_post_handoff_states.parquet")
    outcome = pd.read_parquet(HEAVY / "multi_horizon_action_outcomes.parquet")
    deltas = _deltas()
    base_states = states[
        ["state_id", "suite", "cv_group_id", "current_log10_gap"]
    ]
    frame = outcome.drop(columns=["suite", "cv_group_id"], errors="ignore").merge(
        base_states, on="state_id", how="left", validate="many_to_one"
    )
    if frame["route"].isna().any():
        raise RuntimeError("outcome rows missing state metadata")

    # ---- I1 best-action distributions ----
    distribution_rows = []
    for horizon in HORIZONS:
        for columns in (
            ["route"], ["source_algorithm"], ["FE"], ["suite"], ["cv_group_id"],
        ):
            table = (
                frame.groupby([*columns, f"best_{horizon}"], sort=False)
                .size().reset_index(name="count")
            )
            totals = frame.groupby(columns, sort=False).size().rename("total")
            table = table.merge(totals.reset_index(), on=columns, how="left")
            table["horizon"] = horizon
            table["group_type"] = "|".join(columns)
            distribution_rows.append(table)
    distributions = pd.concat(distribution_rows, ignore_index=True)
    distributions.to_parquet(LIGHT / "best_action_distributions.parquet", index=False)

    # ---- I2 entropies ----
    entropy_rows = []
    for horizon in HORIZONS:
        entry = {
            "horizon": horizon,
            "H_best": entropy_of(frame[f"best_{horizon}"].value_counts()),
            "H_best_given_route": conditional_entropy(frame, ["route"], horizon),
            "H_best_given_route_FE": conditional_entropy(frame, ["route", "FE"], horizon),
            "H_best_given_source_current_FE": conditional_entropy(
                frame, ["source_algorithm", "current_algorithm", "FE"], horizon
            ),
        }
        entropy_rows.append(entry)
    entropies = pd.DataFrame(entropy_rows)
    entropies.to_parquet(LIGHT / "conditional_entropies.parquet", index=False)

    # ---- I3 practical sets + I4 escape rates ----
    practical_rows = []
    escape_rows = []
    for horizon in HORIZONS:
        pivot = frame.pivot_table(
            index="state_id", columns="candidate_action", values=f"gain_{horizon}",
            aggfunc="first",
        )
        meta = frame.drop_duplicates("state_id").set_index("state_id")
        best_gain = pivot.max(axis=1)
        acceptable = pivot.ge(best_gain - deltas[(horizon, "95")], axis=0)
        acceptable50 = pivot.ge(best_gain - deltas[(horizon, "50")], axis=0)
        escape = pivot.drop(columns=["continue"]).max(axis=1) > deltas[(horizon, "95")]
        escape50 = pivot.drop(columns=["continue"]).max(axis=1) > deltas[(horizon, "50")]
        table = pd.DataFrame(
            {
                "acceptable_count_95": acceptable.sum(axis=1),
                "acceptable_count_50": acceptable50.sum(axis=1),
                "escape_95": escape,
                "escape_50": escape50,
                "suite": meta["suite"],
                "route": meta["route"],
                "source_algorithm": meta["source_algorithm"],
                "FE": meta["FE"],
                "cv_group_id": meta["cv_group_id"],
            }
        )
        for horizon_label, column in (("95", "acceptable_count_95"), ("50", "acceptable_count_50")):
            practical_rows.append(
                {
                    "horizon": horizon,
                    "delta": horizon_label,
                    "mean_acceptable_count": float(table[column].mean()),
                    "unique_best_share": float((table[column] == 1).mean()),
                    "states": int(len(table)),
                }
            )
        for column, keys in (
            ("escape_95", "95"),
            ("escape_50", "50"),
        ):
            group_table = table.groupby(["route"], sort=False)[column].agg(["mean", "size"])
            for route, row in group_table.iterrows():
                escape_rows.append(
                    {
                        "horizon": horizon,
                        "delta": keys,
                        "route": route,
                        "escape_rate_function_balanced": function_balanced(
                            table.loc[table["route"].eq(route)], column
                        ),
                        "escape_rate_pooled": float(row["mean"]),
                        "states": int(row["size"]),
                    }
                )
    pd.DataFrame(practical_rows).to_parquet(
        LIGHT / "practical_action_sets.parquet", index=False
    )
    pd.DataFrame(escape_rows).to_parquet(LIGHT / "escape_rates.parquet", index=False)

    # ---- J horizon disagreement ----
    best = frame.drop_duplicates("state_id")[
        ["state_id", "cv_group_id", "route", "suite", "FE",
         "best_500", "best_1000", "best_terminal"]
    ]
    disagreement = {
        "P_best500_ne_best1000": float((best["best_500"] != best["best_1000"]).mean()),
        "P_best1000_ne_best_terminal": float(
            (best["best_1000"] != best["best_terminal"]).mean()
        ),
        "P_best500_ne_best_terminal": float(
            (best["best_500"] != best["best_terminal"]).mean()
        ),
        "states": int(len(best)),
    }
    for horizon_left, horizon_right in (("500", "1000"), ("1000", "terminal")):
        confusion = (
            best.groupby([f"best_{horizon_left}", f"best_{horizon_right}"], sort=False)
            .size().reset_index(name="count")
        )
        confusion["left"] = horizon_left
        confusion["right"] = horizon_right
        disagreement[f"confusion_{horizon_left}_to_{horizon_right}"] = confusion.to_dict(
            orient="records"
        )
    # practical disagreement: best@1000 not practically acceptable at terminal and vice versa
    gains_terminal = frame.pivot_table(
        index="state_id", columns="candidate_action", values="gain_terminal", aggfunc="first"
    )
    gains_1000 = frame.pivot_table(
        index="state_id", columns="candidate_action", values="gain_1000", aggfunc="first"
    )
    best1000 = best.set_index("state_id")["best_1000"]
    best_terminal = best.set_index("state_id")["best_terminal"]
    def _lookup(pivot: pd.DataFrame, index: pd.Series, columns: pd.Series) -> np.ndarray:
        values = pivot.to_numpy(dtype=float)
        row_positions = pivot.index.get_indexer(index)
        column_positions = pivot.columns.get_indexer(columns)
        if (row_positions < 0).any() or (column_positions < 0).any():
            raise RuntimeError("lookup found unmatched state-action pairs")
        return values[row_positions, column_positions]

    terminal_of_best1000 = _lookup(gains_terminal, best1000.index, best1000)
    terminal_best_gain = gains_terminal.max(axis=1)
    disagreement["P_best1000_not_acceptable_at_terminal_95"] = float(
        (
            terminal_best_gain - terminal_of_best1000 > deltas[("terminal", "95")]
        ).mean()
    )
    of_best_terminal = _lookup(gains_1000, best_terminal.index, best_terminal)
    best1000_gain = gains_1000.max(axis=1)
    disagreement["P_best_terminal_not_acceptable_at_1000_95"] = float(
        (best1000_gain - of_best_terminal > deltas[("1000", "95")]).mean()
    )
    pattern = best.loc[
        best["best_1000"].isin(["pso", "shade"]) & best["best_terminal"].eq("continue")
    ]
    disagreement["count_1000switch_terminal_continue"] = int(len(pattern))
    disagreement["share_1000switch_terminal_continue"] = float(
        len(pattern) / max(len(best), 1)
    )
    disagreement["pattern_functions"] = int(pattern["cv_group_id"].nunique()) if len(pattern) else 0
    json_write(disagreement, LIGHT / "horizon_disagreement.json")

    # ---- K empirical information upper bounds ----
    bound_rows = []
    for horizon in HORIZONS:
        loss_column = f"loss_{horizon}"
        state_best = frame.groupby("state_id")[loss_column].min().rename("state_best")
        frame_work = frame.merge(state_best, on="state_id", how="left", validate="many_to_one")
        states_unique = frame_work.drop_duplicates("state_id")
        continue_loss = function_balanced(
            frame_work.loc[frame_work["candidate_action"].eq("continue")], loss_column
        )
        for label, columns in (
            ("current_only", ["current_algorithm"]),
            ("route", ["source_algorithm", "current_algorithm"]),
            ("route_fe", ["source_algorithm", "current_algorithm", "FE"]),
        ):
            # function-balanced mean loss per (group, action): mean over
            # cv_group of the within-group state mean, then the empirical
            # best action per group (ties -> action name order)
            group_action_cv = (
                frame_work.groupby([*columns, "candidate_action", "cv_group_id"], sort=False)[
                    loss_column
                ]
                .mean()
                .rename("cv_loss")
                .reset_index()
            )
            group_action = (
                group_action_cv.groupby([*columns, "candidate_action"], sort=False)["cv_loss"]
                .mean()
                .rename("group_action_loss")
                .reset_index()
            )
            chosen = (
                group_action.sort_values(
                    ["group_action_loss", "candidate_action"], kind="mergesort"
                )
                .groupby(columns, sort=False)
                .head(1)
                .rename(columns={"candidate_action": "chosen_action"})
            )
            assigned = frame_work.merge(
                chosen, on=columns, how="left", validate="many_to_one"
            )
            policy_rows = assigned.loc[
                assigned["candidate_action"].eq(assigned["chosen_action"])
            ]
            policy_loss = function_balanced(policy_rows, loss_column)
            state_best_loss = function_balanced(states_unique, "state_best")
            bound_rows.append(
                {
                    "horizon": horizon,
                    "information": label,
                    "chosen_actions": json.dumps(
                        {
                            "|".join(str(part) for part in (keys if isinstance(keys, tuple) else (keys,))): str(value)
                            for keys, value in chosen.set_index(columns)[
                                "chosen_action"
                            ].items()
                        }
                    ),
                    "function_balanced_policy_loss": policy_loss,
                    "function_balanced_state_best": state_best_loss,
                    "state_wise_gain_remaining": policy_loss - state_best_loss,
                    "continue_reference_loss": continue_loss,
                    "policy_gain_over_continue": continue_loss - policy_loss,
                }
            )
    bounds = pd.DataFrame(bound_rows)
    bounds.to_parquet(LIGHT / "empirical_upper_bounds.parquet", index=False)

    # ---- L source-history effect + M transfer vs native ----
    state_meta = frame.drop_duplicates("state_id")[
        ["state_id", "route", "source_algorithm", "current_algorithm", "FE",
         "cv_group_id", "suite", "current_log10_gap", "action_margin"]
        if "action_margin" in frame.columns
        else ["state_id", "route", "source_algorithm", "current_algorithm", "FE",
              "cv_group_id", "suite", "current_log10_gap"]
    ]
    source_rows = []
    for horizon in HORIZONS:
        pivot = frame.pivot_table(
            index="state_id", columns="candidate_action",
            values=f"gain_{horizon}", aggfunc="first",
        )
        escape_state = (
            pivot.drop(columns=["continue"]).max(axis=1) > deltas[(horizon, "95")]
        ).rename("escape_95").reset_index()
        best_state = pivot.idxmax(axis=1).rename("practical_best_95").reset_index()
        merged = state_meta.merge(escape_state, on="state_id", how="left").merge(
            best_state, on="state_id", how="left"
        )
        cont = frame.loc[frame["candidate_action"].eq("continue")].pivot_table(
            index="state_id", columns="candidate_action",
            values=f"loss_{horizon}", aggfunc="first",
        )
        merged = merged.merge(
            cont[["continue"]].rename(columns={"continue": "continue_loss"}),
            on="state_id", how="left",
        )
        group_table = merged.groupby(["route", "FE"], sort=False).agg(
            states=("state_id", "size"),
            mean_current_log10_gap=("current_log10_gap", "mean"),
            mean_continue_loss=("continue_loss", "mean"),
            escape_rate_95=("escape_95", "mean"),
        )
        practical_best = merged.groupby(["route", "FE"], sort=False)[
            "practical_best_95"
        ].agg(lambda values: float((values == "continue").mean())).rename("practical_best_continue_share")
        group_table = group_table.join(practical_best).reset_index()
        group_table.insert(0, "horizon", horizon)
        source_rows.append(group_table)
    pd.concat(source_rows, ignore_index=True).to_parquet(
        LIGHT / "route_by_fe_summary.parquet", index=False
    )

    # paired route-vs-native gain differences with function-level bootstrap
    bootstrap_rows = []
    for horizon in HORIZONS:
        gains = frame.pivot_table(
            index=["state_id"], columns="candidate_action",
            values=f"gain_{horizon}", aggfunc="first",
        )
        gains = gains.join(
            states.set_index("state_id")[["route", "cv_group_id", "FE"]]
        )
        native = gains.loc[gains["route"].eq("R0_native_cmaes")]
        for route in ("R1_pso_to_cmaes", "R2_shade_to_cmaes"):
            transfer = gains.loc[gains["route"].eq(route)]
            key_columns = ["cv_group_id", "FE"]
            paired = native.set_index(key_columns)[["continue", "pso", "shade"]].add_suffix("_native").join(
                transfer.set_index(key_columns)[["continue", "pso", "shade"]].add_suffix("_transfer"),
                how="inner",
            )
            if paired.empty:
                continue
            for action in ACTIONS:
                delta = paired[f"{action}_transfer"] - paired[f"{action}_native"]
                per_function = delta.groupby(paired.index.get_level_values(0)).mean()
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [BOOTSTRAP_STREAM, HORIZONS.index(horizon), len(per_function)]
                    ).generate_state(4)
                )
                draws = [
                    float(per_function.to_numpy()[rng.integers(0, len(per_function), size=len(per_function))].mean())
                    for _ in range(2000)
                ]
                bootstrap_rows.append(
                    {
                        "horizon": horizon,
                        "contrast": f"{route}_minus_native",
                        "action": action,
                        "states": int(len(delta)),
                        "mean_delta": float(delta.mean()),
                        "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                    }
                )
    pd.DataFrame(bootstrap_rows).to_parquet(
        LIGHT / "transfer_vs_native_bootstrap.parquet", index=False
    )

    # ---- N secondary SHADE-route evidence ----
    shade_rows = []
    for horizon in ("1000", "terminal"):
        subset = frame
        rates = []
        for group, group_frame in subset.groupby("cv_group_id", sort=False):
            pivot = group_frame.pivot_table(
                index="state_id", columns="candidate_action",
                values=f"gain_{horizon}", aggfunc="first",
            )
            if "shade" not in pivot.columns:
                continue
            rates.append(
                {
                    "cv_group_id": group,
                    "suite": group_frame["suite"].iloc[0],
                    "states": int(len(pivot)),
                    "P_best_shade": float(
                        (pivot["shade"] >= pivot.max(axis=1) - 1e-12).mean()
                    ),
                    "escape_share": float(
                        (
                            pivot[["pso", "shade"]].max(axis=1)
                            > deltas[(horizon, "95")]
                        ).mean()
                    ),
                }
            )
        shade_table = pd.DataFrame(rates)
        shade_table.insert(0, "horizon", horizon)
        shade_rows.append(shade_table)
    pd.concat(shade_rows, ignore_index=True).to_parquet(
        LIGHT / "shade_route_evidence.parquet", index=False
    )

    # ---- O progress labels from the continue branch ----
    progress = frame.loc[frame["candidate_action"].eq("continue")][
        ["state_id", "cv_group_id", "route", "FE", "current_log10_gap", "loss_500", "loss_1000"]
    ].copy()
    progress["progress_500_current"] = (
        progress["current_log10_gap"] - progress["loss_500"]
    ) / 0.05
    progress["progress_1000_current"] = (
        progress["current_log10_gap"] - progress["loss_1000"]
    ) / 0.10
    progress.to_parquet(HEAVY / "progress_labels_current.parquet", index=False)
    print(
        json.dumps(
            {
                key: value
                for key, value in disagreement.items()
                if not key.startswith("confusion")
            },
            indent=1,
            default=str,
        )
    )
    print(entropies.to_string())
    print(bounds.to_string())
    print(pd.DataFrame(practical_rows).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=("aggregate", "noise", "audit"))
    args = parser.parse_args()
    if args.step == "aggregate":
        step_aggregate()
    elif args.step == "noise":
        step_noise()
    elif args.step == "audit":
        step_audit()


if __name__ == "__main__":
    main()
