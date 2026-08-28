"""Shared helpers for the analysis_v3 deployment diagnostics (Task 9A-9H)."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "behavior_with_ela" / "results"
V3 = ROOT / "behavior_with_ela" / "analysis_v3"
V3_HEAVY = RESULTS / "analysis_v3"
V2_HEAVY = RESULTS / "analysis_v2"

from behavior_with_ela.model import (  # noqa: E402
    apply_practical_gain_delta,
    estimate_noise_gain_delta,
    read_action_datasets,
    read_action_repetitions,
)
from behavior_with_ela.phase2 import _run_metrics  # noqa: E402
from behavior_with_ela.protocol import load_experiment_config  # noqa: E402

TRAIN_CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
VALIDATION_CONFIG = ROOT / "configs/behavior_with_ela_validation.yaml"
PHASE1_MODEL = RESULTS / "model/behavior_action_gain/models.joblib"

PORTFOLIO = ("pso", "shade", "cmaes")
SUCCESS_LOG10_TARGET = -8.0  # log10(success_gap_target) of the train/validation configs
BOOTSTRAP_STREAM = 2026082901
NOISE_DELTA_STREAM = 2026082902


def load_train_val():
    import joblib

    config = load_experiment_config(TRAIN_CONFIG)
    validation_config = load_experiment_config(VALIDATION_CONFIG)
    bundle = joblib.load(PHASE1_MODEL)
    delta = float(bundle["practical_gain_delta"])
    train = apply_practical_gain_delta(read_action_datasets(config), delta)
    validation = apply_practical_gain_delta(
        read_action_datasets(validation_config), delta
    )
    return config, validation_config, bundle, delta, train, validation


def load_v2_first_trigger_runs() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    train_runs = pd.read_parquet(V2_HEAVY / "task1/train_first_trigger_runs.parquet")
    validation_runs = pd.read_parquet(
        V2_HEAVY / "task1/validation_first_trigger_runs.parquet"
    )
    threshold = float(train_runs["threshold"].iloc[0])
    return train_runs, validation_runs, threshold


def load_v2_oof_predictions() -> pd.DataFrame:
    return pd.read_parquet(V2_HEAVY / "task1/oof_predictions.parquet")


def noise_deltas(config) -> dict[str, float]:
    """Function-balanced within-state gain noise deltas at the preset quantiles."""
    cache = V3_HEAVY / "noise_deltas.json"
    if cache.exists():
        return {key: float(value) for key, value in json.loads(cache.read_text()).items()}
    repetitions = read_action_repetitions(config)
    table: dict[str, float] = {}
    for label, quantile in (("delta_50", 0.50), ("delta_95", 0.95)):
        _, value = estimate_noise_gain_delta(repetitions, quantile=quantile)
        table[label] = float(value)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(table, indent=2))
    return table


def harmful_rates(runs: pd.DataFrame, deltas: dict[str, float]) -> dict[str, float]:
    triggered = runs.loc[runs["switch_triggered"].astype(bool)]
    if triggered.empty:
        return {
            "harmful_below_zero_rate": 0.0,
            "harmful_below_delta_50_rate": 0.0,
            "harmful_below_delta_95_rate": 0.0,
            "triggered_runs": 0,
        }
    gains = triggered["selected_action_gain"].to_numpy(dtype=float)
    return {
        "harmful_below_zero_rate": float((gains < 0.0).mean()),
        "harmful_below_delta_50_rate": float(
            (gains < -float(deltas["delta_50"])).mean()
        ),
        "harmful_below_delta_95_rate": float(
            (gains < -float(deltas["delta_95"])).mean()
        ),
        "triggered_runs": int(len(triggered)),
    }


def switch_fe_stats(runs: pd.DataFrame) -> dict[str, float]:
    values = runs.loc[
        runs["switch_triggered"].astype(bool), "selected_FE"
    ].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {}
    return {
        "switch_fe_p10": float(np.quantile(values, 0.10)),
        "switch_fe_p50": float(np.quantile(values, 0.50)),
        "switch_fe_p90": float(np.quantile(values, 0.90)),
    }


def target_distribution(runs: pd.DataFrame) -> pd.DataFrame:
    group = runs.loc[runs["switch_triggered"].astype(bool)]
    total = max(len(group), 1)
    shares = group["selected_algorithm"].value_counts(normalize=True)
    rows = [
        {
            "selected_algorithm": str(algorithm),
            "selected_share": float(shares.get(algorithm, 0.0)),
            "selected_runs": int((group["selected_algorithm"] == algorithm).sum()),
        }
        for algorithm in PORTFOLIO
    ]
    stay = runs.loc[~runs["switch_triggered"].astype(bool)]
    rows.append(
        {
            "selected_algorithm": "stay_with_prefix",
            "selected_share": float(len(stay) / max(len(runs), 1)),
            "selected_runs": int(len(stay)),
        }
    )
    frame = pd.DataFrame(rows)
    frame["triggered_total"] = int(len(group))
    frame["run_total"] = int(len(runs))
    frame["share_per_total_runs"] = frame["selected_runs"] / max(len(runs), 1)
    return frame


def policy_metrics(
    runs: pd.DataFrame,
    deltas: dict[str, float],
) -> dict[str, float]:
    return {
        **_run_metrics(runs),
        **harmful_rates(runs, deltas),
        **switch_fe_stats(runs),
        "success_rate": float(
            (
                runs["selected_terminal_log10_loss"].to_numpy(dtype=float)
                <= SUCCESS_LOG10_TARGET + 1e-12
            ).mean()
        ),
    }


def function_balanced(values: pd.Series, groups: pd.Series) -> float:
    frame = pd.DataFrame({"value": values, "group": groups})
    return float(frame.groupby("group")["value"].mean().mean())


def sbs_reference(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (problem_id, seed) SBS terminal log10 loss from cmaes continue rows."""
    cmaes = runs.loc[runs["prefix_algorithm"].astype(str).eq("cmaes")]
    reference = cmaes[
        [
            "problem_id",
            "function_id",
            "cv_group_id",
            "seed",
            "continue_terminal_log10_loss",
        ]
    ].rename(
        columns={"continue_terminal_log10_loss": "sbs_terminal_log10_loss"}
    ).sort_values(["problem_id", "seed"], kind="mergesort").reset_index(drop=True)
    if reference.duplicated(["problem_id", "seed"]).any():
        raise RuntimeError("SBS reference contains duplicate problem-seed rows")
    return reference


def gain_over_sbs(runs: pd.DataFrame) -> pd.DataFrame:
    reference = sbs_reference(runs)
    merge_columns = ["problem_id", "function_id", "cv_group_id", "seed"]
    reference = reference[merge_columns + ["sbs_terminal_log10_loss"]]
    merged = runs.merge(reference, on=merge_columns, how="left", validate="many_to_one")
    if merged["sbs_terminal_log10_loss"].isna().any():
        raise RuntimeError("policy runs are missing an SBS pairing")
    merged["gain_over_sbs"] = (
        merged["sbs_terminal_log10_loss"].to_numpy(dtype=float)
        - merged["selected_terminal_log10_loss"].to_numpy(dtype=float)
    )
    return merged


def save_table(table, name: str, task: str) -> Path:
    target_dir = V3 / task
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    if name.endswith(".parquet"):
        table.to_parquet(path, index=False)
    elif name.endswith(".json"):
        path.write_text(json_dumps(table))
    else:
        table.to_csv(path, index=False)
    return path


def save_heavy_table(table, name: str, task: str) -> Path:
    target_dir = V3_HEAVY / task
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    table.to_parquet(path, index=False)
    return path


def json_dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=float)
