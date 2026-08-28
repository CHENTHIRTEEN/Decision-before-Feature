"""Task 9B: prefix-identity and fixed-timing baselines for the v2 carrier.

B1a prefix-only empirical action-value table (no Behavior, all opportunities);
B1b prefix+FE table (timing-aware, still no Behavior);
B2  fixed-0.20 learned mapping (single decision at FE=2000, no Behavior);
B3  Behavior@0.20 (same regression carrier, decisions restricted to FE=2000).

All learned components use train grouped-family OOF only; validation is never
used for rule or threshold fitting. No objective evaluation is executed.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import common  # noqa: F401
from common import (
    json_dumps,
    load_train_val,
    noise_deltas,
    policy_metrics,
    save_heavy_table,
    save_table,
    target_distribution,
)

sys.path.insert(0, str(common.ROOT))

from behavior_with_ela.baselines import (  # noqa: E402
    _action_loss_family_oof_predictions,
    _fit_action_loss_model,
    _policy_row,
    _predict_action_loss_rows,
    _run_reference,
)
from behavior_with_ela.model import (  # noqa: E402
    fit_first_trigger_threshold,
    replay_first_trigger,
)

TASK = "task2"
V2_RUN_TASK = "task1"
FIXED_FE = 2000


def gain_table(rows: pd.DataFrame, *, keyed_by_fe: bool) -> pd.DataFrame:
    switch = rows.loc[~rows["action_equals_prefix"].astype(bool)]
    group_columns = ["suite", "cv_group_id", "prefix_algorithm"]
    if keyed_by_fe:
        group_columns = ["suite", "cv_group_id", "prefix_algorithm", "FE"]
    group_columns = [*group_columns, "candidate_action"]
    function_means = switch.groupby(group_columns, sort=False)[
        "action_gain_vs_continue"
    ].mean()
    frame = function_means.reset_index(name="mean_gain")
    suite_columns = [column for column in group_columns if column != "cv_group_id"]
    suite_means = frame.groupby(suite_columns, sort=False)["mean_gain"].mean()
    block_columns = [column for column in suite_columns if column != "suite"]
    block = suite_means.reset_index(name="table_gain").groupby(
        block_columns, sort=False, as_index=False
    )["table_gain"].mean()
    return block


def table_lookup(
    table: pd.DataFrame,
    fallback: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    keyed_by_fe: bool,
) -> tuple[np.ndarray, float]:
    key_columns = ["prefix_algorithm", "candidate_action"]
    if keyed_by_fe:
        key_columns = ["prefix_algorithm", "FE", "candidate_action"]
    merged = rows.merge(table[key_columns + ["table_gain"]], on=key_columns, how="left")
    fallback_rate = 0.0
    if keyed_by_fe:
        missing = merged["table_gain"].isna().to_numpy()
        fallback_rate = float(missing.mean())
        if missing.any():
            fallback_merge = rows.loc[missing].merge(
                fallback[["prefix_algorithm", "candidate_action", "table_gain"]],
                on=["prefix_algorithm", "candidate_action"],
                how="left",
            )
            merged.loc[missing, "table_gain"] = fallback_merge[
                "table_gain"
            ].to_numpy(dtype=float)
        if merged["table_gain"].isna().any():
            raise RuntimeError("prefix table lookup left unmatched rows")
    return merged["table_gain"].to_numpy(dtype=float), fallback_rate


def table_oof_scores(
    train: pd.DataFrame,
    *,
    keyed_by_fe: bool,
) -> tuple[pd.DataFrame, dict]:
    bbob = train.loc[train["suite"].astype(str).eq("bbob")].copy()
    families = tuple(sorted(set(bbob["family"].astype(str))))
    frames = []
    fallback_counts = {}
    for fold_number, heldout_family in enumerate(families, start=1):
        heldout_functions = set(
            bbob.loc[bbob["family"].astype(str).eq(heldout_family), "function_id"]
            .astype(str)
        )
        train_mask = ~(
            train["suite"].astype(str).eq("bbob")
            & train["family"].astype(str).eq(heldout_family)
        )
        from behavior_with_ela.model import _ma_overlaps_heldout

        ma_safe = ~train.apply(
            lambda row: _ma_overlaps_heldout(row, heldout_functions), axis=1
        )
        fold_train = train.loc[train_mask & ma_safe]
        table = gain_table(fold_train, keyed_by_fe=keyed_by_fe)
        fallback = gain_table(fold_train, keyed_by_fe=False)
        fold_eval = bbob.loc[
            bbob["family"].astype(str).eq(heldout_family)
            & ~bbob["action_equals_prefix"].astype(bool)
        ].copy()
        values, fallback_rate = table_lookup(
            table, fallback, fold_eval, keyed_by_fe=keyed_by_fe
        )
        fold_eval["predicted_improve_probability"] = values
        fold_eval["predicted_action_class"] = np.where(
            values > 0.0, "improve", "equivalent"
        )
        fold_eval["oof_fold"] = fold_number
        fallback_counts[fold_number] = fallback_rate
        frames.append(
            fold_eval[
                [
                    "problem_id", "prefix_algorithm", "seed", "FE",
                    "decision_opportunity_index", "candidate_action",
                    "predicted_action_class", "predicted_improve_probability",
                    "oof_fold",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True), fallback_counts


def table_validation_scores(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    keyed_by_fe: bool,
) -> pd.DataFrame:
    table = gain_table(train, keyed_by_fe=keyed_by_fe)
    fallback = gain_table(train, keyed_by_fe=False)
    switch = validation.loc[
        validation["suite"].astype(str).eq("bbob")
        & ~validation["action_equals_prefix"].astype(bool)
    ].copy()
    values, _ = table_lookup(table, fallback, switch, keyed_by_fe=keyed_by_fe)
    switch["predicted_improve_probability"] = values
    switch["predicted_action_class"] = np.where(values > 0.0, "improve", "equivalent")
    return switch[
        [
            "problem_id", "prefix_algorithm", "seed", "FE",
            "decision_opportunity_index", "candidate_action",
            "predicted_action_class", "predicted_improve_probability",
        ]
    ]


def fit_table_policy(
    *,
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    delta: float,
    default_algorithm: str,
    keyed_by_fe: bool,
) -> dict:
    oof, fallback_counts = table_oof_scores(train, keyed_by_fe=keyed_by_fe)
    thresholds, selected, train_runs = fit_first_trigger_threshold(
        action_rows=train,
        action_predictions=oof,
        practical_delta=delta,
    )
    train_runs["policy_name"] = name
    validation_scores = table_validation_scores(
        train, validation, keyed_by_fe=keyed_by_fe
    )
    validation_runs = replay_first_trigger(
        action_rows=validation,
        action_predictions=validation_scores,
        threshold=selected,
        practical_delta=delta,
        default_algorithm=default_algorithm,
    )
    validation_runs["policy_name"] = name
    return {
        "name": name,
        "threshold": float(selected),
        "thresholds": thresholds,
        "train_runs": train_runs,
        "validation_runs": validation_runs,
        "fallback_counts": fallback_counts,
    }


def fit_fixed_020_mapping(rows: pd.DataFrame) -> pd.DataFrame:
    switch = rows.loc[
        ~rows["action_equals_prefix"].astype(bool)
        & rows["FE"].astype(int).eq(FIXED_FE)
    ]
    if switch.empty:
        raise ValueError("no FE=2000 switch rows available for the fixed-0.20 mapping")
    function_means = switch.groupby(
        ["suite", "cv_group_id", "prefix_algorithm", "candidate_action"],
        sort=False,
    )["action_gain_vs_continue"].mean()
    suite_means = (
        function_means.reset_index(name="mean_gain")
        .groupby(["suite", "prefix_algorithm", "candidate_action"], sort=False)[
            "mean_gain"
        ]
        .mean()
    )
    block = suite_means.reset_index(name="train_mean_gain")
    order = {algorithm: index for index, algorithm in enumerate(common.PORTFOLIO)}
    block["candidate_order"] = block["candidate_action"].map(order).astype(int)
    selected_rows = []
    for prefix in common.PORTFOLIO:
        choices = block.loc[
            block["prefix_algorithm"].astype(str).eq(prefix)
        ].sort_values(
            ["train_mean_gain", "candidate_order"],
            ascending=[False, True],
            kind="mergesort",
        )
        best = choices.iloc[0]
        if float(best["train_mean_gain"]) <= 0.0:
            selected_rows.append(
                {
                    "prefix_algorithm": prefix,
                    "selected_algorithm": prefix,
                    "train_mean_gain": 0.0,
                    "switch_FE": FIXED_FE,
                }
            )
        else:
            selected_rows.append(
                {
                    "prefix_algorithm": prefix,
                    "selected_algorithm": str(best["candidate_action"]),
                    "train_mean_gain": float(best["train_mean_gain"]),
                    "switch_FE": FIXED_FE,
                }
            )
    return pd.DataFrame(selected_rows)


def fixed_020_policy(
    actions: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    default_algorithm: str,
    practical_delta: float,
    policy_name: str,
) -> pd.DataFrame:
    bbob = actions.loc[actions["suite"].astype(str).eq("bbob")].copy()
    reference = _run_reference(bbob).set_index(
        ["problem_id", "prefix_algorithm", "seed"], drop=False
    )
    target_by_prefix = dict(
        zip(mapping["prefix_algorithm"], mapping["selected_algorithm"], strict=True)
    )
    rows = []
    for run_key, run in bbob.groupby(
        ["problem_id", "prefix_algorithm", "seed"], sort=False
    ):
        ref = reference.loc[run_key]
        prefix = str(ref["prefix_algorithm"])
        target = str(target_by_prefix[prefix])
        if target == prefix:
            rows.append(
                _policy_row(
                    reference=ref,
                    policy_name=policy_name,
                    selected_algorithm=prefix,
                    selected_gain=0.0,
                    selected_log10_loss=float(ref["continue_log10_loss"]),
                    selected_fe=None,
                    selected_opportunity=None,
                    default_algorithm=default_algorithm,
                    practical_delta=practical_delta,
                )
            )
            continue
        candidates = run.loc[
            run["candidate_action"].astype(str).eq(target)
            & run["FE"].astype(int).eq(FIXED_FE)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "fixed-0.20 policy requires exactly one target row at FE=2000"
            )
        selected = candidates.iloc[0]
        rows.append(
            _policy_row(
                reference=ref,
                policy_name=policy_name,
                selected_algorithm=target,
                selected_gain=float(selected["action_gain_vs_continue"]),
                selected_log10_loss=float(selected["log10_action_loss"]),
                selected_fe=int(selected["FE"]),
                selected_opportunity=int(selected["decision_opportunity_index"]),
                default_algorithm=default_algorithm,
                practical_delta=practical_delta,
            )
        )
    return pd.DataFrame(rows)


def recompute_regret_against_full_best(
    runs: pd.DataFrame,
    full_best: pd.DataFrame,
    *,
    practical_delta: float,
) -> pd.DataFrame:
    """Uniform regret denominator: the full-opportunity best observed one-switch gain.

    B3 restricts predictions to FE=2000, so its replay can only see a restricted
    best; all new baselines are re-scored against the full opportunity set to stay
    comparable with the v2 policy and the phase-1 panel.
    """
    reference = full_best[
        ["problem_id", "prefix_algorithm", "seed", "best_observed_one_switch_gain"]
    ].rename(columns={"best_observed_one_switch_gain": "full_best_gain"})
    merged = runs.merge(
        reference,
        on=["problem_id", "prefix_algorithm", "seed"],
        how="left",
        validate="many_to_one",
    )
    if merged["full_best_gain"].isna().any():
        raise RuntimeError("missing full-opportunity best for regret recomputation")
    best = merged["full_best_gain"].to_numpy(dtype=float)
    selected = merged["selected_action_gain"].to_numpy(dtype=float)
    merged["best_observed_one_switch_gain"] = best
    merged["one_switch_regret"] = best - selected
    merged["normalized_regret_denominator"] = np.maximum(
        best, max(float(practical_delta), 1e-12)
    )
    merged["normalized_one_switch_regret"] = (
        best - selected
    ) / merged["normalized_regret_denominator"].to_numpy(dtype=float)
    merged["acceptable_policy"] = (best - selected) <= float(practical_delta)
    return merged


def policy_panel_entry(
    runs: pd.DataFrame,
    deltas: dict[str, float],
) -> dict:
    return {
        "run_count": int(len(runs)),
        **policy_metrics(runs, deltas),
    }


def paired_delta_vs_v2(
    v2_runs: pd.DataFrame,
    baseline_runs: pd.DataFrame,
    *,
    split: str,
    baseline_name: str,
) -> pd.DataFrame:
    key = ["problem_id", "prefix_algorithm", "seed"]
    v2 = v2_runs[key + ["selected_action_gain", "cv_group_id"]].rename(
        columns={"selected_action_gain": "v2_gain"}
    )
    base = baseline_runs[key + ["selected_action_gain"]].rename(
        columns={"selected_action_gain": "baseline_gain"}
    )
    merged = v2.merge(base, on=key, how="inner", validate="one_to_one")
    merged["gain_delta_v2_minus_baseline"] = (
        merged["v2_gain"] - merged["baseline_gain"]
    )
    merged["evaluation_split"] = split
    merged["baseline_policy"] = baseline_name
    return merged


def main() -> None:
    config, validation_config, bundle, delta, train, validation = load_train_val()
    deltas = noise_deltas(config)
    default_algorithm = str(bundle["default_algorithm"])

    v2_train_runs, v2_validation_runs, v2_threshold = common.load_v2_first_trigger_runs()

    print(f"[{TASK}] B1a prefix-only table policy", flush=True)
    b1a = fit_table_policy(
        name="prefix_only_table",
        train=train,
        validation=validation,
        delta=delta,
        default_algorithm=default_algorithm,
        keyed_by_fe=False,
    )
    print(f"[{TASK}] B1b prefix+FE table policy", flush=True)
    b1b = fit_table_policy(
        name="prefix_plus_fe_table",
        train=train,
        validation=validation,
        delta=delta,
        default_algorithm=default_algorithm,
        keyed_by_fe=True,
    )

    print(f"[{TASK}] B2 fixed-0.20 learned mapping", flush=True)
    mapping_oof_frames = []
    mapping_train_parts = []
    bbob_train = train.loc[train["suite"].astype(str).eq("bbob")]
    families = tuple(sorted(set(bbob_train["family"].astype(str))))
    from behavior_with_ela.model import _ma_overlaps_heldout

    for fold_number, heldout_family in enumerate(families, start=1):
        heldout_functions = set(
            bbob_train.loc[
                bbob_train["family"].astype(str).eq(heldout_family), "function_id"
            ].astype(str)
        )
        train_mask = ~(
            train["suite"].astype(str).eq("bbob")
            & train["family"].astype(str).eq(heldout_family)
        )
        ma_safe = ~train.apply(
            lambda row: _ma_overlaps_heldout(row, heldout_functions), axis=1
        )
        fold_mapping = fit_fixed_020_mapping(train.loc[train_mask & ma_safe])
        fold_mapping["oof_fold"] = fold_number
        mapping_oof_frames.append(fold_mapping)
    mapping_oof = pd.concat(mapping_oof_frames, ignore_index=True)
    mapping_full = fit_fixed_020_mapping(train)
    train_runs_b2 = fixed_020_policy(
        train,
        mapping_full,
        default_algorithm=default_algorithm,
        practical_delta=delta,
        policy_name="fixed_020_mapping",
    )
    validation_runs_b2 = fixed_020_policy(
        validation,
        mapping_full,
        default_algorithm=default_algorithm,
        practical_delta=delta,
        policy_name="fixed_020_mapping",
    )
    print(mapping_full.to_string(), flush=True)

    print(f"[{TASK}] B3 Behavior@0.20 regression", flush=True)
    train_f2k = train.loc[train["FE"].astype(int).eq(FIXED_FE)].copy()
    validation_f2k = validation.loc[validation["FE"].astype(int).eq(FIXED_FE)].copy()
    oof_b3 = _action_loss_family_oof_predictions(train_f2k, config)
    thresholds_b3, selected_b3, train_runs_b3 = fit_first_trigger_threshold(
        action_rows=train_f2k,
        action_predictions=oof_b3,
        practical_delta=delta,
    )
    train_runs_b3["policy_name"] = "behavior_at_020"
    model_b3 = _fit_action_loss_model(train_f2k, config, fold_number=90_002)
    validation_scores_b3 = _predict_action_loss_rows(
        model=model_b3,
        action_rows=validation_f2k,
        practical_delta=delta,
    )
    validation_runs_b3 = replay_first_trigger(
        action_rows=validation,
        action_predictions=validation_scores_b3,
        threshold=selected_b3,
        practical_delta=delta,
        default_algorithm=default_algorithm,
    )
    validation_runs_b3["policy_name"] = "behavior_at_020"

    heavy_dir = common.V3_HEAVY / TASK
    heavy_dir.mkdir(parents=True, exist_ok=True)
    save_heavy_table(b1a["train_runs"], "runs_prefix_only_train.parquet", TASK)
    save_heavy_table(b1a["validation_runs"], "runs_prefix_only_validation.parquet", TASK)
    save_heavy_table(b1b["train_runs"], "runs_prefix_plus_fe_train.parquet", TASK)
    save_heavy_table(b1b["validation_runs"], "runs_prefix_plus_fe_validation.parquet", TASK)
    save_heavy_table(train_runs_b2, "runs_fixed_020_train.parquet", TASK)
    save_heavy_table(validation_runs_b2, "runs_fixed_020_validation.parquet", TASK)
    save_heavy_table(train_runs_b3, "runs_behavior_at_020_train.parquet", TASK)
    save_heavy_table(validation_runs_b3, "runs_behavior_at_020_validation.parquet", TASK)
    save_heavy_table(mapping_oof, "fixed_020_mapping_oof.parquet", TASK)

    # unified panel with the existing phase-1 policy runs
    phase1_train = pd.read_parquet(
        common.RESULTS / "baselines/phase1/train_policy_runs.parquet"
    )
    phase1_validation = pd.read_parquet(
        common.RESULTS / "baselines/phase1/validation_policy_runs.parquet"
    )
    v2_train_runs["policy_name"] = "behavior_action_loss_regression_v2"
    v2_validation_runs["policy_name"] = "behavior_action_loss_regression_v2"
    new_train = pd.concat(
        [
            recompute_regret_against_full_best(
                b1a["train_runs"], v2_train_runs, practical_delta=delta
            ),
            recompute_regret_against_full_best(
                b1b["train_runs"], v2_train_runs, practical_delta=delta
            ),
            recompute_regret_against_full_best(
                train_runs_b2, v2_train_runs, practical_delta=delta
            ),
            recompute_regret_against_full_best(
                train_runs_b3, v2_train_runs, practical_delta=delta
            ),
            v2_train_runs,
        ],
        ignore_index=True,
    )
    new_validation = pd.concat(
        [
            recompute_regret_against_full_best(
                b1a["validation_runs"], v2_validation_runs, practical_delta=delta
            ),
            recompute_regret_against_full_best(
                b1b["validation_runs"], v2_validation_runs, practical_delta=delta
            ),
            recompute_regret_against_full_best(
                validation_runs_b2, v2_validation_runs, practical_delta=delta
            ),
            recompute_regret_against_full_best(
                validation_runs_b3, v2_validation_runs, practical_delta=delta
            ),
            v2_validation_runs,
        ],
        ignore_index=True,
    )
    panel_rows = []
    for split, runs in (
        ("bbob_train_oof", pd.concat([new_train, phase1_train], ignore_index=True)),
        (
            "bbob_validation",
            pd.concat([new_validation, phase1_validation], ignore_index=True),
        ),
    ):
        for policy, group in runs.groupby("policy_name", sort=False):
            entry = policy_panel_entry(group, deltas)
            distribution = target_distribution(group)
            panel_rows.append(
                {
                    "evaluation_split": split,
                    "policy_name": str(policy),
                    **entry,
                    "share_selected_cmaes": float(
                        distribution.loc[
                            distribution["selected_algorithm"].eq("cmaes"),
                            "share_per_total_runs",
                        ].sum()
                    ),
                    "share_selected_shade": float(
                        distribution.loc[
                            distribution["selected_algorithm"].eq("shade"),
                            "share_per_total_runs",
                        ].sum()
                    ),
                    "share_selected_pso": float(
                        distribution.loc[
                            distribution["selected_algorithm"].eq("pso"),
                            "share_per_total_runs",
                        ].sum()
                    ),
                    "share_stay_with_prefix": float(
                        distribution.loc[
                            distribution["selected_algorithm"].eq("stay_with_prefix"),
                            "share_per_total_runs",
                        ].sum()
                    ),
                }
            )
    panel = pd.DataFrame(panel_rows)
    save_table(panel, "baseline_policy_panel.parquet", TASK)

    delta_rows = []
    for name, runs_by_split in (
        ("prefix_only_table", {"bbob_train_oof": b1a["train_runs"], "bbob_validation": b1a["validation_runs"]}),
        ("prefix_plus_fe_table", {"bbob_train_oof": b1b["train_runs"], "bbob_validation": b1b["validation_runs"]}),
        ("fixed_020_mapping", {"bbob_train_oof": train_runs_b2, "bbob_validation": validation_runs_b2}),
        ("behavior_at_020", {"bbob_train_oof": train_runs_b3, "bbob_validation": validation_runs_b3}),
        ("fixed_030_transition", {"bbob_train_oof": None, "bbob_validation": None}),
        ("time_only_action_gain", {"bbob_train_oof": None, "bbob_validation": None}),
    ):
        for split, baseline_runs in runs_by_split.items():
            if baseline_runs is None:
                source = phase1_train if split == "bbob_train_oof" else phase1_validation
                baseline_runs = source.loc[
                    source["policy_name"].astype(str).eq(name)
                ]
            paired = paired_delta_vs_v2(
                v2_train_runs if split == "bbob_train_oof" else v2_validation_runs,
                baseline_runs,
                split=split,
                baseline_name=name,
            )
            delta_rows.append(paired)
    paired_all = pd.concat(delta_rows, ignore_index=True)
    save_heavy_table(paired_all, "paired_delta_vs_v2_runs.parquet", TASK)
    paired_rows = []
    for (name, split), group in paired_all.groupby(
        ["baseline_policy", "evaluation_split"], sort=False
    ):
        paired_rows.append(
            {
                "baseline_policy": str(name),
                "evaluation_split": str(split),
                "run_count": int(len(group)),
                "function_balanced_mean_gain_delta": common.function_balanced(
                    group["gain_delta_v2_minus_baseline"],
                    group["cv_group_id"],
                ),
                "share_runs_v2_better": float(
                    (group["gain_delta_v2_minus_baseline"] > 0.0).mean()
                ),
                "share_runs_v2_worse": float(
                    (group["gain_delta_v2_minus_baseline"] < 0.0).mean()
                ),
            }
        )
    paired_summary = pd.DataFrame(paired_rows)
    save_table(paired_summary, "paired_delta_vs_v2_summary.parquet", TASK)

    thresholds_payload = {
        "v2_regression_threshold": float(v2_threshold),
        "prefix_only_threshold": float(b1a["threshold"]),
        "prefix_plus_fe_threshold": float(b1b["threshold"]),
        "behavior_at_020_threshold": float(selected_b3),
        "prefix_plus_fe_oof_fallback_rate": b1b["fallback_counts"],
    }
    save_table(thresholds_payload, "thresholds.json", TASK)
    print(f"[{TASK}] done", flush=True)
    print(
        panel[
            [
                "evaluation_split", "policy_name", "function_balanced_mean_gain",
                "function_balanced_mean_normalized_regret", "switch_rate",
                "harmful_below_zero_rate",
            ]
        ].to_string(),
        flush=True,
    )
    print(json_dumps(thresholds_payload), flush=True)


if __name__ == "__main__":
    main()
