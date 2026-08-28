"""Task 0 protocol consistency checks on real Phase 1/2 artifacts.

Run from project root: .venv/bin/python behavior_with_ela/analysis_v2/task0_check.py
Writes machine-readable evidence to behavior_with_ela/results/analysis_v2/task0/.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "behavior_with_ela" / "results"
OUT = RESULTS / "analysis_v2" / "task0"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from behavior_with_ela.model import RUN_KEY, STATE_KEY  # noqa: E402

PORTFOLIO = ["pso", "shade", "cmaes"]
FINDINGS: list[dict] = []


def check(name: str, passed: bool, detail: str, status: str = "FAIL") -> None:
    FINDINGS.append(
        {
            "check": name,
            "status": "PASS" if passed else status,
            "detail": detail,
        }
    )
    print(f"[{FINDINGS[-1]['status']}] {name}: {detail}")


def warn(name: str, detail: str) -> None:
    FINDINGS.append({"check": name, "status": "WARNING", "detail": detail})
    print(f"[WARNING] {name}: {detail}")


def load_action_frames(split_pattern: str) -> pd.DataFrame:
    parts = [
        pd.read_parquet(f)
        for f in sorted((RESULTS / "actions").glob(split_pattern))
    ]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ds = load_action_frames("*/*/dimension_10/action_gain_dataset.parquet")
    reps = load_action_frames("*/*/dimension_10/action_repetitions.parquet")

    # 1. state key uniqueness and complete action matrix
    key = [*STATE_KEY, "candidate_action"]
    dup = int(ds.duplicated(key).sum())
    check("state-action key uniqueness (135k rows)", dup == 0, f"duplicate rows={dup}")
    sizes = ds.groupby(list(STATE_KEY), sort=False)["candidate_action"].agg(
        ("nunique", "size")
    )
    complete = sizes["nunique"].eq(3).all() and sizes["size"].eq(3).all()
    check(
        "action matrix completeness per state",
        bool(complete),
        f"states={len(sizes)}, all have exactly 3 candidates: {bool(complete)}",
    )
    obs = set(ds["candidate_action"].astype(str))
    check(
        "candidate set equals portfolio",
        obs == set(PORTFOLIO),
        f"observed candidates={sorted(obs)}",
    )

    # 2. continue-current uses native continuation of the same run
    cont = ds.loc[ds["action_equals_prefix"].astype(bool)]
    sw = ds.loc[~ds["action_equals_prefix"].astype(bool)]
    cont_native = cont["handoff_type"].astype(str).eq("native_optimizer_state").all()
    check(
        "continue-current handoff is native optimizer state",
        bool(cont_native),
        f"continue rows={len(cont)}, all native: {bool(cont_native)}",
    )
    cont_label_ok = cont["action"].astype(str).eq("continue_current").all()
    switch_label_ok = (
        sw["action"].astype(str) == sw["candidate_action"].astype(str)
    ).all()
    check(
        "action labels: continue_current for same-prefix rows, candidate identity for switches",
        bool(cont_label_ok and switch_label_ok),
        f"continue rows={len(cont)} labeled continue_current: {bool(cont_label_ok)}; "
        f"switch rows={len(sw)} labeled by candidate: {bool(switch_label_ok)}",
    )

    # 3. cross-algorithm switching uses population transfer
    transfer_ok = sw["handoff_type"].astype(str).eq("population_transfer_initialization").all()
    check(
        "switch actions use population transfer initialization",
        bool(transfer_ok),
        f"switch rows={len(sw)}, all population_transfer_initialization: {bool(transfer_ok)}",
    )

    # 4. repetitions pairing
    rep_counts = reps.groupby([*STATE_KEY, "candidate_action"], sort=False)[
        "replicate_id"
    ].agg(["count", "nunique"])
    pairing_ok = rep_counts["count"].isin([1, 3]).all() and rep_counts["nunique"].eq(
        rep_counts["count"]
    ).all()
    check(
        "repetition replicate ids are paired 1 or 3 without gaps",
        bool(pairing_ok),
        f"repetition state-actions={len(rep_counts)}",
    )
    plan = ds.set_index(key)["action_repetitions"].groupby(level=list(range(len(key)))).first()
    rep_actual = reps.groupby([*STATE_KEY, "candidate_action"]).size()
    plan, rep_actual = plan.align(rep_actual, join="inner")
    planned_match = bool((plan == rep_actual).all())
    check(
        "repetition counts match planned_action_repetitions",
        planned_match,
        f"compared={len(plan)}, match={planned_match}",
    )

    # 5. BBOB family OOF is leave-family-out
    oof = pd.read_parquet(
        RESULTS / "model/behavior_action_gain/oof_action_predictions.parquet"
    )
    fold_ok = True
    for fold, g in oof.groupby("oof_fold"):
        if not g["family"].astype(str).eq(str(g["heldout_family"].iloc[0])).all():
            fold_ok = False
    check(
        "every OOF prediction row belongs to its held-out family",
        bool(fold_ok),
        f"folds={sorted(oof['oof_fold'].unique().tolist())}",
    )
    check(
        "OOF rows are BBOB only",
        bool(oof["suite"].astype(str).eq("bbob").all()),
        f"rows={len(oof)}",
    )
    coverage_ok = len(oof) == int(
        (
            ds["suite"].astype(str).eq("bbob")
            & ds["split"].astype(str).eq("bbob_train")
            & ~ds["action_equals_prefix"].astype(bool)
        ).sum()
    )
    check(
        "OOF covers all BBOB switch rows exactly",
        bool(coverage_ok),
        f"oof={len(oof)}",
    )

    # 6. MA-BBOB component guard: each mabbob row excluded from folds whose held-out
    # functions appear in its components
    bbob = ds.loc[ds["suite"].astype(str).eq("bbob")]
    families = sorted(bbob["family"].astype(str).unique())
    heldout_map = {
        fam: set(bbob.loc[bbob["family"].astype(str).eq(fam), "function_id"].astype(str))
        for fam in families
    }
    ma = ds.loc[ds["suite"].astype(str).eq("mabbob"), ["function_id", "component_functions"]]
    guard_ok = True
    for _, row in ma.iterrows():
        components = {
            f"bbob_f{int(v):03d}" for v in np.asarray(row["component_functions"]).reshape(-1)
        }
        for fam, funcs in heldout_map.items():
            if components & funcs and str(row["function_id"]) in funcs:
                guard_ok = False
    overlap_rows = 0
    for _, row in ma.iterrows():
        components = {
            f"bbob_f{int(v):03d}" for v in np.asarray(row["component_functions"]).reshape(-1)
        }
        for fam, funcs in heldout_map.items():
            if components & funcs:
                overlap_rows += 1
                break
    check(
        "MA-BBOB definitions never carry held-out BBOB function identity",
        bool(guard_ok),
        f"mabbob definitions={ma['function_id'].nunique()}, definitions with overlapping components (rows)={overlap_rows} (excluded per fold by _ma_overlaps_heldout)",
    )

    # 7. validation isolation
    summary = json.loads(
        (RESULTS / "model/behavior_action_gain/training_summary.json").read_text()
    )
    check(
        "validation rows used for fit/threshold = 0",
        summary["validation_rows_used_for_model_fit"] == 0
        and summary["validation_rows_used_for_threshold_fit"] == 0,
        json.dumps(
            {
                "fit": summary["validation_rows_used_for_model_fit"],
                "threshold": summary["validation_rows_used_for_threshold_fit"],
            }
        ),
    )

    # 8. function balancing semantics
    runs = pd.read_parquet(
        RESULTS / "model/behavior_action_gain/oof_first_trigger_runs.parquet"
    )
    manual = runs.groupby("cv_group_id")["selected_action_gain"].mean().mean()
    reported = summary["first_trigger_metrics"]["function_balanced_mean_gain"]
    check(
        "function-balanced mean = mean of per-family means",
        abs(manual - reported) < 1e-9,
        f"manual={manual:.9f} reported={reported:.9f}",
    )

    # 9. first-trigger: one row per run, earliest crossing
    run_unique = not runs.duplicated(list(RUN_KEY)).any()
    check(
        "first-trigger produces exactly one row per run",
        bool(run_unique),
        f"runs={len(runs)}, unique={bool(run_unique)}",
    )
    trig = runs.loc[runs["switch_triggered"].astype(bool)]
    thr = float(summary["decision_threshold"])
    state_best = (
        oof.sort_values(
            ["predicted_improve_probability"],
            ascending=False,
            kind="mergesort",
        )
        .groupby([*RUN_KEY, "FE", "decision_opportunity_index"], sort=False)
        .head(1)
    )
    eligible = state_best.loc[state_best["predicted_improve_probability"] > thr]
    earliest = eligible.groupby(list(RUN_KEY), sort=False)["FE"].min()
    sel_fe = trig.set_index(list(RUN_KEY))["selected_FE"]
    aligned, compared = 0, 0
    for rkey, fe in earliest.items():
        if rkey in sel_fe.index:
            compared += 1
            aligned += int(int(sel_fe.loc[rkey]) == int(fe))
    check(
        "selected FE is the earliest threshold-crossing state (BBOB OOF)",
        compared > 0 and aligned == compared,
        f"compared={compared}, aligned={aligned}, threshold={thr:.6f}",
    )

    # 10. threshold selection optimism on train OOF (structural)
    thresholds = pd.read_parquet(
        RESULTS / "model/behavior_action_gain/threshold_summary.parquet"
    )
    sel_row = thresholds.loc[thresholds["selected_threshold"]].iloc[0]
    quantiles = thresholds["function_balanced_mean_gain"].quantile([0.5, 0.9, 1.0])
    warn(
        "train OOF policy metrics carry threshold-selection optimism",
        (
            f"selected threshold {sel_row['threshold']:.4f} maximises function_balanced_mean_gain "
            f"over {len(thresholds)} candidates (median={quantiles[0.5]:.4f}, max={quantiles[1.0]:.4f}); "
            "mitigation: Phase 1 headline claims also reported on untouched BBOB validation"
        ),
    )

    # 11. action-loss target matrix ordering
    from behavior_with_ela.baselines import _action_loss_state_matrix

    states, targets = _action_loss_state_matrix(ds, tuple(PORTFOLIO))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(states), size=25, replace=False)
    order_ok = True
    for si in sample_idx:
        row0 = states.iloc[int(si)]
        srows = ds.loc[
            (ds["problem_id"] == row0["problem_id"])
            & (ds["prefix_algorithm"] == row0["prefix_algorithm"])
            & (ds["seed"] == row0["seed"])
            & (ds["FE"] == row0["FE"])
            & (ds["decision_opportunity_index"] == row0["decision_opportunity_index"])
        ].set_index("candidate_action")["log10_action_loss"]
        if not all(np.isclose(targets[int(si), i], srows[a]) for i, a in enumerate(PORTFOLIO)):
            order_ok = False
            break
    check(
        "action-loss target matrix follows [pso, shade, cmaes] order",
        bool(order_ok),
        f"states={len(states)}, target shape={targets.shape}, 25 random states verified={bool(order_ok)}",
    )

    # 12. A1 vs A3 validation identity (deeper)
    vr = pd.read_parquet(
        RESULTS
        / "model/local_landscape_increment_action_loss/validation_first_trigger_runs.parquet"
    )
    piv = vr.pivot_table(
        index=list(RUN_KEY),
        columns="phase2_feature_group",
        values=["selected_algorithm", "selected_FE"],
        aggfunc="first",
    )
    same_action = (
        piv[("selected_algorithm", "A1_behavior")]
        == piv[("selected_algorithm", "A3_behavior_local_landscape")]
    ).mean()
    same_fe = (
        piv[("selected_FE", "A1_behavior")].fillna(-1)
        == piv[("selected_FE", "A3_behavior_local_landscape")].fillna(-1)
    ).mean()
    check(
        "A1 vs A3 validation first-trigger decisions identical (re-verified)",
        same_action == 1.0 and same_fe == 1.0,
        f"same action={same_action:.4f}, same FE={same_fe:.4f}",
    )
    a1_oof = pd.read_parquet(
        RESULTS
        / "model/local_landscape_increment_action_loss/train_oof_action_predictions.parquet"
    )
    a1 = a1_oof.loc[a1_oof["phase2_feature_group"].eq("A1_behavior")]
    a3 = a1_oof.loc[a1_oof["phase2_feature_group"].eq("A3_behavior_local_landscape")]
    m = a1.merge(
        a3[[*STATE_KEY, "candidate_action", "predicted_action_gain"]],
        on=[*STATE_KEY, "candidate_action"],
        suffixes=("_a1", "_a3"),
    )
    state_key = [*STATE_KEY]
    top1 = (
        m.sort_values("predicted_action_gain_a1", ascending=False, kind="mergesort")
        .groupby(state_key, sort=False)
        .head(1)
        .set_index(state_key)["candidate_action"]
    )
    top3 = (
        m.sort_values("predicted_action_gain_a3", ascending=False, kind="mergesort")
        .groupby(state_key, sort=False)
        .head(1)
        .set_index(state_key)["candidate_action"]
    )
    paired = pd.DataFrame({"a1": top1, "a3": top3}).dropna()
    disagreement = float((paired["a1"] != paired["a3"]).mean())
    print(
        f"[INFO] train OOF state-level top-1 disagreement A1 vs A3: {disagreement:.4f} "
        f"over {len(paired)} states"
    )
    FINDINGS.append(
        {
            "check": "A1 vs A3 train OOF state-level top-1 disagreement",
            "status": "INFO",
            "detail": (
                f"disagreement={disagreement:.4f} over {len(paired)} states; "
                f"score mean|diff|={float((m['predicted_action_gain_a1'] - m['predicted_action_gain_a3']).abs().mean()):.4f}"
            ),
        }
    )

    out = OUT / "task0_findings.json"
    out.write_text(json.dumps(FINDINGS, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
