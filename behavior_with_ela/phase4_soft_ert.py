from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from behavior_with_ela.action_dataset import action_shard_paths
from behavior_with_ela.model import (
    STATE_KEY,
    apply_practical_gain_delta,
    read_action_datasets,
)
from behavior_with_ela.protocol import ExperimentConfig, load_experiment_config


SOFT_ERT_PROTOCOL = "dynamic_portfolio_relative_soft_ert_v1"


def build_dynamic_soft_ert(
    *,
    config_path: str | Path,
    phase1_model_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    bundle = joblib.load(phase1_model_path)
    practical_delta = float(bundle["practical_gain_delta"])
    repetitions = read_action_repetitions(config)
    terminal = apply_practical_gain_delta(
        read_action_datasets(config),
        practical_delta,
    )
    rows = dynamic_soft_ert_rows(
        repetitions=repetitions,
        terminal_actions=terminal,
        config=config,
    )
    summary = summarize_soft_ert(rows)
    output = Path(output_dir)
    action_path = output / "soft_ert_action_dataset.parquet"
    summary_path = output / "soft_ert_summary.parquet"
    metadata_path = output / "soft_ert_summary.json"
    paths = (action_path, summary_path, metadata_path)
    if any(path.exists() for path in paths) and not overwrite:
        raise FileExistsError(f"Soft-ERT outputs already exist; pass --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in paths:
            path.unlink(missing_ok=True)
    rows.to_parquet(action_path, index=False)
    summary.to_parquet(summary_path, index=False)
    result = {
        "soft_ert_protocol": SOFT_ERT_PROTOCOL,
        "target_slack_fraction": config.soft_ert.target_slack_fraction,
        "practical_FE_ratio": config.soft_ert.practical_FE_ratio,
        "timeout_penalty_offset_FE": config.soft_ert.timeout_penalty_offset_FE,
        "state_action_rows": int(len(rows)),
        "states": int(rows[list(STATE_KEY)].drop_duplicates().shape[0]),
        "known_optimum_used": False,
        "main_log_gap_label_replaced": False,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def read_action_repetitions(config: ExperimentConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for suite in config.suites:
        for function in suite.functions:
            path = action_shard_paths(config, suite, function)[0]
            if not path.exists():
                raise FileNotFoundError(
                    f"missing action repetitions {path}; regenerate action data with improvement events"
                )
            frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("no action repetition data was found")
    return pd.concat(frames, ignore_index=True)


def dynamic_soft_ert_rows(
    *,
    repetitions: pd.DataFrame,
    terminal_actions: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    required = {
        *STATE_KEY,
        "candidate_action",
        "action_equals_prefix",
        "replicate_id",
        "planned_action_FE",
        "action_status",
        "action_improvement_offsets",
        "action_improvement_values",
    }
    missing = sorted(required.difference(repetitions.columns))
    if missing:
        raise ValueError(f"action repetitions are missing Soft-ERT fields: {missing}")
    terminal_columns = [
        *STATE_KEY,
        "candidate_action",
        "suite",
        "family",
        "cv_group_id",
        "function_id",
        "component_functions",
        "state_id",
        "action_gain_vs_continue",
        "action_gain_class",
        "log10_action_loss",
    ]
    terminal = terminal_actions[terminal_columns].copy()
    output_rows: list[dict[str, Any]] = []
    for state_key, state_rows in repetitions.groupby(list(STATE_KEY), sort=False):
        horizons = state_rows["planned_action_FE"].astype(int).unique()
        if len(horizons) != 1:
            raise ValueError("one Soft-ERT state must use a common terminal horizon")
        horizon = int(horizons[0])
        histories = [_history(row) for _, row in state_rows.iterrows()]
        prefix_values = {history[1][0] for history in histories}
        if len(prefix_values) != 1:
            raise ValueError("action histories disagree on the shared prefix best value")
        prefix_best = float(next(iter(prefix_values)))
        soft_best = float(min(np.min(values) for _, values in histories))
        target = float(
            soft_best
            + config.soft_ert.target_slack_fraction * (prefix_best - soft_best)
        )
        action_summaries: list[dict[str, Any]] = []
        for action, action_rows in state_rows.groupby("candidate_action", sort=False):
            hits: list[int | None] = []
            for _, row in action_rows.sort_values("replicate_id").iterrows():
                offsets, values = _history(row)
                reached = np.flatnonzero(values <= target + EPS_FOR_TARGET)
                hits.append(None if len(reached) == 0 else int(offsets[int(reached[0])]))
            success_count = sum(value is not None for value in hits)
            penalty = horizon + config.soft_ert.timeout_penalty_offset_FE
            contribution = sum(
                penalty if value is None else int(value) for value in hits
            )
            soft_ert = float(contribution / max(success_count, 1))
            first = action_rows.iloc[0]
            action_summaries.append(
                {
                    "candidate_action": str(action),
                    "action_equals_prefix": bool(first["action_equals_prefix"]),
                    "soft_ert": soft_ert,
                    "soft_ert_success_count": int(success_count),
                    "soft_ert_repetition_count": int(len(hits)),
                    "soft_ert_hit_rate": float(success_count / len(hits)),
                    "soft_ert_first_hit_offsets": hits,
                    "all_action_paths_completed": bool(
                        action_rows["action_status"].astype(str).eq("completed").all()
                    ),
                }
            )
        continue_rows = [row for row in action_summaries if row["action_equals_prefix"]]
        if len(continue_rows) != 1:
            raise RuntimeError("Soft-ERT state requires one continue-current action")
        continue_ert = float(continue_rows[0]["soft_ert"])
        practical_fe = float(config.soft_ert.practical_FE_ratio * horizon)
        state_payload = dict(zip(STATE_KEY, state_key, strict=True))
        for row in action_summaries:
            gain = continue_ert - float(row["soft_ert"])
            row.update(
                {
                    **state_payload,
                    "soft_ert_protocol": SOFT_ERT_PROTOCOL,
                    "soft_ert_horizon_FE": horizon,
                    "soft_ert_prefix_best": prefix_best,
                    "soft_ert_observed_best": soft_best,
                    "soft_ert_target": target,
                    "soft_ert_target_slack_fraction": (
                        config.soft_ert.target_slack_fraction
                    ),
                    "soft_ert_timeout_penalty_FE": (
                        horizon + config.soft_ert.timeout_penalty_offset_FE
                    ),
                    "soft_ert_practical_FE": practical_fe,
                    "soft_ert_continue": continue_ert,
                    "soft_ert_gain_vs_continue": float(gain),
                    "soft_ert_gain_class": _gain_class_FE(gain, practical_fe),
                }
            )
            output_rows.append(row)
    output = pd.DataFrame(output_rows).merge(
        terminal,
        on=[*STATE_KEY, "candidate_action"],
        how="inner",
        validate="one_to_one",
    )
    if len(output) != len(output_rows):
        raise RuntimeError("Soft-ERT rows do not align with terminal action rows")
    terminal_sign = output["action_gain_class"].astype(str).map(
        {"degrade": -1, "equivalent": 0, "improve": 1}
    )
    soft_sign = output["soft_ert_gain_class"].astype(str).map(
        {"degrade": -1, "equivalent": 0, "improve": 1}
    )
    output["terminal_soft_ert_class_agreement"] = terminal_sign.eq(soft_sign)
    output["terminal_soft_ert_direction_agreement_non_equivalent"] = np.where(
        (terminal_sign != 0) & (soft_sign != 0),
        terminal_sign == soft_sign,
        None,
    )
    return output.sort_values(
        [*STATE_KEY, "candidate_action"],
        kind="mergesort",
    ).reset_index(drop=True)


def summarize_soft_ert(rows: pd.DataFrame) -> pd.DataFrame:
    switch = rows.loc[~rows["action_equals_prefix"].astype(bool)].copy()
    summaries = []
    for suite, suite_rows in switch.groupby("suite", sort=False):
        function_rows = (
            suite_rows.groupby("cv_group_id", as_index=False)
            .agg(
                class_agreement=("terminal_soft_ert_class_agreement", "mean"),
                terminal_gain=("action_gain_vs_continue", "mean"),
                soft_ert_gain=("soft_ert_gain_vs_continue", "mean"),
                soft_ert_hit_rate=("soft_ert_hit_rate", "mean"),
            )
        )
        summaries.append(
            {
                "suite": str(suite),
                "function_count": int(len(function_rows)),
                "state_action_rows": int(len(suite_rows)),
                "function_balanced_class_agreement": float(
                    function_rows["class_agreement"].mean()
                ),
                "function_balanced_terminal_gain": float(
                    function_rows["terminal_gain"].mean()
                ),
                "function_balanced_soft_ert_gain": float(
                    function_rows["soft_ert_gain"].mean()
                ),
                "function_balanced_soft_ert_hit_rate": float(
                    function_rows["soft_ert_hit_rate"].mean()
                ),
            }
        )
    return pd.DataFrame(summaries)


def _history(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(row["action_improvement_offsets"], dtype=int).reshape(-1)
    values = np.asarray(row["action_improvement_values"], dtype=float).reshape(-1)
    if len(offsets) == 0 or len(offsets) != len(values):
        raise ValueError("action improvement history has inconsistent lengths")
    if offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("action improvement offsets must start at zero and increase")
    if not np.isfinite(values).all() or np.any(np.diff(values) >= 0.0):
        raise ValueError("action improvement values must be finite and strictly decrease")
    return offsets, values


def _gain_class_FE(value: float, delta: float) -> str:
    if value > delta:
        return "improve"
    if value < -delta:
        return "degrade"
    return "equivalent"


EPS_FOR_TARGET = 1e-12


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Dynamic Soft-ERT action labels as an unknown-optimum sensitivity."
    )
    parser.add_argument("--config", default="configs/behavior_with_ela_train.yaml")
    parser.add_argument(
        "--phase1-model",
        default="results/behavior_with_ela/model/behavior_action_gain/models.joblib",
    )
    parser.add_argument(
        "--output",
        default="results/behavior_with_ela/sensitivity/soft_ert/train",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = build_dynamic_soft_ert(
        config_path=args.config,
        phase1_model_path=args.phase1_model,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(
        f"built {summary['state_action_rows']} Dynamic Soft-ERT action rows "
        f"across {summary['states']} states"
    )


if __name__ == "__main__":
    main()
