"""Shared helpers for the analysis_v2 diagnostics (Task 1-8)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "behavior_with_ela" / "results"
V2 = ROOT / "behavior_with_ela" / "analysis_v2"
V2_HEAVY = RESULTS / "analysis_v2"

from behavior_with_ela.model import (  # noqa: E402
    apply_practical_gain_delta,
    fit_first_trigger_threshold,
    read_action_datasets,
    replay_first_trigger,
)
from behavior_with_ela.phase2 import _run_metrics  # noqa: E402
from behavior_with_ela.protocol import load_experiment_config  # noqa: E402

TRAIN_CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
VALIDATION_CONFIG = ROOT / "configs/behavior_with_ela_validation.yaml"
PHASE1_MODEL = RESULTS / "model/behavior_action_gain/models.joblib"

PORTFOLIO = ("pso", "shade", "cmaes")


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


def add_prefix_onehot(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for algorithm in PORTFOLIO:
        result[f"prefix_{algorithm}"] = (
            result["prefix_algorithm"].astype(str).eq(algorithm)
        ).astype(float)
    return result


def policy_block(runs: pd.DataFrame) -> dict:
    return _run_metrics(runs)


def harmful_switch_rate(runs: pd.DataFrame, delta: float) -> float:
    triggered = runs.loc[runs["switch_triggered"].astype(bool)]
    if triggered.empty:
        return 0.0
    return float(
        (triggered["selected_action_gain"].to_numpy(dtype=float) < -float(delta)).mean()
    )


def switch_fe_quantiles(runs: pd.DataFrame) -> dict:
    triggered = runs.loc[runs["switch_triggered"].astype(bool), "selected_FE"]
    if triggered.empty:
        return {}
    values = triggered.to_numpy(dtype=float)
    return {
        "switch_fe_p10": float(np.quantile(values, 0.10)),
        "switch_fe_p50": float(np.quantile(values, 0.50)),
        "switch_fe_p90": float(np.quantile(values, 0.90)),
    }


def per_candidate_scores(predictions: pd.DataFrame, delta: float) -> pd.DataFrame:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score

    rows = []
    for candidate, group in predictions.groupby("candidate_action", sort=True):
        truth = group["action_gain_class"].astype(str)
        binary = truth.eq("improve").astype(int)
        prob = group["predicted_improve_probability"].to_numpy(dtype=float)
        gain_true = group["action_gain_vs_continue"].to_numpy(dtype=float)
        gain_pred = (
            group["predicted_action_gain"].to_numpy(dtype=float)
            if "predicted_action_gain" in group.columns
            else prob
        )
        finite = np.isfinite(gain_pred) & np.isfinite(gain_true)
        rho = (
            float(spearmanr(gain_pred[finite], gain_true[finite]).statistic)
            if finite.sum() > 2
            else float("nan")
        )
        rows.append(
            {
                "candidate_action": str(candidate),
                "rows": int(len(group)),
                "improve_prevalence": float(binary.mean()),
                "improve_ap": (
                    float(average_precision_score(binary, prob))
                    if binary.nunique() > 1
                    else float("nan")
                ),
                "gain_spearman": rho,
                "mean_true_gain": float(gain_true.mean()),
                "mean_predicted_gain": float(np.mean(gain_pred)),
            }
        )
    return pd.DataFrame(rows)


def save_table(table: pd.DataFrame, name: str, task: str) -> Path:
    target_dir = V2 / task
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / name
    if name.endswith(".parquet"):
        table.to_parquet(path, index=False)
    elif name.endswith(".json"):
        path.write_text(json_dumps(table))
    else:
        table.to_csv(path, index=False)
    return path


def json_dumps(obj) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False, default=float)
