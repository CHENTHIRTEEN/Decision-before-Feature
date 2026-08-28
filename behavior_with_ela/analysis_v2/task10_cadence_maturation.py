"""Task 10: decision cadence and post-handoff maturation diagnostic.

Strictly controlled online scheduling study on the three CEC2017 development
functions (F1/F10/F29, 10 seeds, 3 initial prefixes). Models, thresholds and
Behavior are unchanged; only the online switch scheduling differs:

P0 first_trigger            one-switch reference (both carriers)
P1 raw_per_opportunity      no dwell / hysteresis / cap (both carriers)
P2 dwell_500                minimum_dwell_fe = 500
P3 dwell_1000               minimum_dwell_fe = 1000
P4 dwell_1500               minimum_dwell_fe = 1500
P5 dwell_1000 + hysteresis  classifier only (top1-top2 probability margin 0.10)

Steps:
  ablation    execute the policy grid
  analysis    policy comparison, switch intervals, chattering, segment summary
  maturation  forced-B continuation vs raw policy at equal FE (+ resume terminal)
  curves      post-switch progress curves
  confound    transfer (reinit) vs native-state restoration for A->B->A runs

Outputs: results/online/cec2017_quick3_scheduler_ablation/<carrier>__<policy>/
and aggregated tables under results/analysis_v2/task10/.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd

import common  # noqa: F401
from common import json_dumps

sys.path.insert(0, str(common.ROOT))

from behavior.features import extract_behavior_rows  # noqa: E402
from behavior_with_ela.online import (  # noqa: E402
    _selected_suites,
    predict_switch_scores,
    run_one_switch_policy,
)
from behavior_with_ela.protocol import (  # noqa: E402
    load_experiment_config,
    make_experiment_problem,
)
from optimizers import (  # noqa: E402
    NO_QUERY_TRANSFER_EVENT,
    OptimizerSettings,
    advance_optimizer_state,
    clone_optimizer_state,
    initialize_optimizer_state,
    initialize_transferred_optimizer_state,
)
from trajectory.recorder import TrajectoryRecorder  # noqa: E402

TASK = "task10_cadence_maturation"
FUNCTIONS = (1, 10, 29)
CEC_CONFIG = common.ROOT / "configs/behavior_with_ela_cec2017.yaml"
ABLATION_ROOT = common.RESULTS / "online" / "cec2017_quick3_scheduler_ablation"
TASK10_ROOT = common.RESULTS / "analysis_v2" / "task10"
STATIC_CONTINUE = common.RESULTS / "online/cec2017_quick3/static_continue.parquet"
V2_BUNDLE = common.V2_HEAVY / "task9_quick_cec" / "v2_online_bundle.joblib"
CARRIERS = {
    "v2_regression": V2_BUNDLE,
    "action_gain_classifier": common.PHASE1_MODEL,
}
CLASSIFIER = "action_gain_classifier"
POLICIES = {
    "p0_first_trigger": dict(
        switch_rule="first_trigger", minimum_dwell_fe=0, hysteresis_margin=0.0
    ),
    "p1_raw_per_opportunity": dict(
        switch_rule="per_opportunity", minimum_dwell_fe=0, hysteresis_margin=0.0
    ),
    "p2_dwell_500": dict(
        switch_rule="per_opportunity", minimum_dwell_fe=500, hysteresis_margin=0.0
    ),
    "p3_dwell_1000": dict(
        switch_rule="per_opportunity", minimum_dwell_fe=1000, hysteresis_margin=0.0
    ),
    "p4_dwell_1500": dict(
        switch_rule="per_opportunity", minimum_dwell_fe=1500, hysteresis_margin=0.0
    ),
    "p5_dwell_1000_hysteresis_10": dict(
        switch_rule="per_opportunity", minimum_dwell_fe=1000, hysteresis_margin=0.10
    ),
}
CONFIG_MATRIX = tuple(
    (carrier, policy)
    for carrier in CARRIERS
    for policy in POLICIES
    if not (policy.startswith("p5") and carrier != CLASSIFIER)
)
HORIZONS = (500, 1000, 1500)
CURVE_OFFSETS = (200, 500, 1000)
PREMATURE_BOUND = 1000


def config_dir(carrier: str, policy: str):
    return ABLATION_ROOT / f"{carrier}__{policy}"


def function_number(function_id: str) -> int:
    return int(str(function_id).rsplit("f", 1)[-1])


# ---------------------------------------------------------------------------
# step: ablation
# ---------------------------------------------------------------------------

def step_ablation() -> None:
    from behavior_with_ela.online import evaluate_one_switch_online

    for carrier, policy in CONFIG_MATRIX:
        print(f"[{TASK}] running {carrier}__{policy}", flush=True)
        evaluate_one_switch_online(
            config_path=CEC_CONFIG,
            model_path=CARRIERS[carrier],
            output_dir=config_dir(carrier, policy),
            only_functions=FUNCTIONS,
            initial_algorithm="all",
            workers=8,
            record_segments=not POLICIES[policy]["switch_rule"].startswith("first"),
            scheduling_label=f"{carrier}__{policy}",
            overwrite=True,
            **POLICIES[policy],
        )


# ---------------------------------------------------------------------------
# shared loading helpers
# ---------------------------------------------------------------------------

def load_outcomes() -> pd.DataFrame:
    frames = []
    for carrier, policy in CONFIG_MATRIX:
        frame = pd.read_parquet(
            config_dir(carrier, policy) / "online_policy_outcomes.parquet"
        )
        frame.insert(0, "carrier", carrier)
        frame.insert(1, "policy", policy)
        frames.append(frame)
    outcomes = pd.concat(frames, ignore_index=True)
    outcomes["run_id"] = (
        outcomes["problem_id"].astype(str)
        + ":"
        + outcomes["prefix_algorithm"].astype(str)
        + ":seed"
        + outcomes["seed"].astype(int).astype(str)
    )
    return outcomes


def load_all_segments() -> pd.DataFrame:
    frames = []
    for carrier, policy in CONFIG_MATRIX:
        path = config_dir(carrier, policy) / "online_segments.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame.insert(0, "carrier", carrier)
        frame.insert(1, "policy", policy)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_sequences(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in outcomes.iterrows():
        fes = [int(value) for value in row["switch_fe_sequence"]]
        targets = [str(value) for value in row["switch_target_sequence"]]
        rows.append(
            {
                "carrier": row["carrier"],
                "policy": row["policy"],
                "function_id": row["function_id"],
                "prefix_algorithm": str(row["prefix_algorithm"]),
                "seed": int(row["seed"]),
                "run_id": row["run_id"],
                "log10_gap": float(row["log10_gap"]),
                "success": bool(row["success"]),
                "switch_count": int(row["switch_count"]),
                "switch_fes": fes,
                "targets": targets,
                "algorithm_sequence": [str(row["prefix_algorithm"])] + targets,
            }
        )
    return pd.DataFrame(rows)


def intervals_of(fes: list[int]) -> list[int]:
    return [int(fes[index] - fes[index - 1]) for index in range(1, len(fes))]


def reversals_of(sequence: list[str], fes: list[int]) -> list[dict]:
    events = []
    for k in range(len(fes) - 1):
        if k + 2 >= len(sequence):
            continue
        left, middle, right = sequence[k], sequence[k + 1], sequence[k + 2]
        if left == right and middle != left:
            events.append(
                {
                    "pair": f"{middle}<->{left}",
                    "reversal_interval": int(fes[k + 1] - fes[k]),
                }
            )
    return events


def quantile_block(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "share_below_200": float((values < 200).mean()),
        "share_below_500": float((values < 500).mean()),
        "share_below_1000": float((values < 1000).mean()),
    }


# ---------------------------------------------------------------------------
# step: analysis (10D/10E/10H)
# ---------------------------------------------------------------------------

def interval_chattering_table(seqs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (carrier, policy), group in seqs.groupby(["carrier", "policy"], sort=False):
        all_intervals: list[int] = []
        reversal_intervals: list[int] = []
        reversal_by_pair: dict[str, list[int]] = {}
        runs_reversal_h500: set[str] = set()
        runs_reversal_h1000: set[str] = set()
        for _, row in group.iterrows():
            fes = list(row["switch_fes"])
            all_intervals.extend(intervals_of(fes))
            for event in reversals_of(list(row["algorithm_sequence"]), fes):
                interval = event["reversal_interval"]
                reversal_intervals.append(interval)
                reversal_by_pair.setdefault(event["pair"], []).append(interval)
                if interval < 500:
                    runs_reversal_h500.add(row["run_id"])
                if interval < PREMATURE_BOUND:
                    runs_reversal_h1000.add(row["run_id"])
        block = (
            quantile_block(np.asarray(all_intervals, dtype=float))
            if all_intervals
            else {"n": 0}
        )
        rows.append(
            {
                "carrier": carrier,
                "policy": policy,
                "runs": int(len(group)),
                **{f"interval_{key}": value for key, value in block.items()},
                "chattering_run_rate_h500": len(runs_reversal_h500) / max(len(group), 1),
                "chattering_run_rate_h1000": len(runs_reversal_h1000) / max(len(group), 1),
                "reversal_event_count_h500": int(
                    sum(1 for v in reversal_intervals if v < 500)
                ),
                "reversal_event_count_h1000": int(
                    sum(1 for v in reversal_intervals if v < PREMATURE_BOUND)
                ),
                "mean_reversal_interval": (
                    float(np.mean(reversal_intervals)) if reversal_intervals else None
                ),
                "reversal_pairs": json_dumps(
                    {
                        pair: {
                            "count": len(values),
                            "median_interval": float(np.median(values)),
                            "min": int(np.min(values)),
                        }
                        for pair, values in sorted(reversal_by_pair.items())
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def policy_comparison_table(outcomes: pd.DataFrame, seqs: pd.DataFrame) -> pd.DataFrame:
    static = pd.read_parquet(STATIC_CONTINUE)
    cont = static[
        ["function_id", "seed", "prefix_algorithm", "log10_gap"]
    ].rename(columns={"log10_gap": "continue_log10_gap"})
    merged = outcomes.merge(
        cont,
        on=["function_id", "seed", "prefix_algorithm"],
        how="left",
        validate="many_to_one",
    )
    if merged["continue_log10_gap"].isna().any():
        raise RuntimeError("missing continue-current reference rows")
    merged["gain_vs_continue"] = merged["continue_log10_gap"] - merged["log10_gap"]
    seqs_indexed = seqs.set_index(["carrier", "policy", "run_id"], drop=False)

    rows = []
    for (carrier, policy), group in merged.groupby(["carrier", "policy"], sort=False):
        intervals: list[int] = []
        reversals: list[dict] = []
        runs_reversal_h1000: set[str] = set()
        first_fes: list[int] = []
        second_fes: list[int] = []
        final_algorithms: list[str] = []
        transitions: list[tuple[str, str]] = []
        for _, row in group.iterrows():
            seq_row = seqs_indexed.loc[(carrier, policy, row["run_id"])]
            fes = list(seq_row["switch_fes"])
            sequence = list(seq_row["algorithm_sequence"])
            if fes:
                first_fes.append(fes[0])
            if len(fes) > 1:
                second_fes.append(fes[1])
            intervals.extend(intervals_of(fes))
            for event in reversals_of(sequence, fes):
                reversals.append(event)
                if event["reversal_interval"] < PREMATURE_BOUND:
                    runs_reversal_h1000.add(row["run_id"])
            source = sequence[0]
            for target in seq_row["targets"]:
                transitions.append((source, target))
                source = target
            final_algorithms.append(sequence[-1])
        block = (
            quantile_block(np.asarray(intervals, dtype=float)) if intervals else {"n": 0}
        )
        final = pd.Series(final_algorithms).value_counts(normalize=True)
        transition_counts = (
            pd.Series([f"{s}->{t}" for s, t in transitions]).value_counts()
            if transitions
            else pd.Series(dtype=int)
        )
        rows.append(
            {
                "carrier": carrier,
                "policy": policy,
                "runs": int(len(group)),
                "mean_terminal_log10_gap": float(group["log10_gap"].mean()),
                "median_terminal_log10_gap": float(group["log10_gap"].median()),
                "function_balanced_gain_vs_continue": float(
                    group.groupby("function_id")["gain_vs_continue"].mean().mean()
                ),
                "switch_run_rate": float((group["switch_count"] > 0).mean()),
                "mean_switch_count": float(group["switch_count"].mean()),
                "median_switch_count": float(group["switch_count"].median()),
                "max_switch_count": int(group["switch_count"].max()),
                "median_first_switch_fe": (
                    float(np.median(first_fes)) if first_fes else None
                ),
                "median_second_switch_fe": (
                    float(np.median(second_fes)) if second_fes else None
                ),
                "switch_interval_median": block.get("median"),
                "premature_switch_rate_below_500": block.get("share_below_500"),
                "premature_switch_rate_below_1000": block.get("share_below_1000"),
                "success_rate": float(group["success"].mean()),
                "final_algorithm_distribution": json_dumps(
                    {key: float(value) for key, value in final.items()}
                ),
                "transition_counts": json_dumps(
                    {
                        key: int(value)
                        for key, value in sorted(transition_counts.items())
                    }
                ),
            }
        )
    comparison = pd.DataFrame(rows)
    chatter = interval_chattering_table(seqs)[
        ["carrier", "policy", "chattering_run_rate_h500", "chattering_run_rate_h1000"]
    ]
    return comparison.merge(chatter, on=["carrier", "policy"], how="left")


def step_analysis() -> None:
    TASK10_ROOT.mkdir(parents=True, exist_ok=True)
    outcomes = load_outcomes()
    seqs = run_sequences(outcomes)
    comparison = policy_comparison_table(outcomes, seqs)
    intervals = interval_chattering_table(seqs)
    segments = load_all_segments()
    comparison.to_parquet(TASK10_ROOT / "policy_comparison.parquet", index=False)
    intervals.to_parquet(TASK10_ROOT / "switch_interval_chattering.parquet", index=False)
    if not segments.empty:
        segments.to_parquet(TASK10_ROOT / "segment_summary.parquet", index=False)
        lifetime_rows = []
        for (carrier, policy), group in segments.groupby(["carrier", "policy"], sort=False):
            lifetime_rows.append(
                {
                    "carrier": carrier,
                    "policy": policy,
                    "segments": int(len(group)),
                    **{
                        f"lifetime_{key}": value
                        for key, value in quantile_block(
                            group["segment_lifetime_FE"].to_numpy(dtype=float)
                        ).items()
                    },
                    "improvement_median": float(
                        group["segment_log10_improvement"].median()
                    ),
                    "improvement_mean": float(
                        group["segment_log10_improvement"].mean()
                    ),
                }
            )
        pd.DataFrame(lifetime_rows).to_parquet(
            TASK10_ROOT / "segment_lifetime.parquet", index=False
        )
    for columns, name in (
        (["carrier", "policy", "function_id"], "policy_by_function"),
        (["carrier", "policy", "prefix_algorithm"], "policy_by_prefix"),
    ):
        table = (
            outcomes.groupby(columns, sort=False)
            .agg(
                runs=("log10_gap", "size"),
                mean_log10_gap=("log10_gap", "mean"),
                median_log10_gap=("log10_gap", "median"),
                mean_switch_count=("switch_count", "mean"),
                success_rate=("success", "mean"),
            )
            .reset_index()
        )
        table.to_parquet(TASK10_ROOT / f"{name}.parquet", index=False)
    print(
        comparison[
            [
                "carrier", "policy", "mean_terminal_log10_gap",
                "function_balanced_gain_vs_continue", "mean_switch_count",
                "premature_switch_rate_below_500", "chattering_run_rate_h1000",
            ]
        ].to_string(),
        flush=True,
    )


# ---------------------------------------------------------------------------
# instrumented reruns (maturation / curves)
# ---------------------------------------------------------------------------

def _instrumented_worker(job: dict) -> dict:
    config = load_experiment_config(CEC_CONFIG)
    bundle = joblib.load(CARRIERS[job["carrier"]])
    suite = _selected_suites(config, ("cec2017",))[0]
    params = dict(POLICIES[job["policy"]])
    outcome, _, _, _, checkpoints = run_one_switch_policy(
        config=config,
        suite=suite,
        function=int(job["function"]),
        instance=1,
        seed=int(job["seed"]),
        prefix_algorithm=str(job["prefix"]),
        bundle=bundle,
        gap_checkpoints=tuple(job["checkpoints"]),
        **params,
    )
    return {
        "job": {key: job[key] for key in job if key != "checkpoints"},
        "terminal_log10_gap": float(outcome["log10_gap"]),
        "switch_fes": [int(v) for v in outcome["switch_fe_sequence"]],
        "checkpoints": checkpoints,
    }


def _checkpoint_value(checkpoints: list[dict], fe: int) -> tuple[int, float] | None:
    """Recorded value for a checkpoint: first update boundary >= fe."""
    for row in checkpoints:
        if int(row["checkpoint_FE"]) == int(fe):
            return int(row["recorded_FE"]), float(row["log10_gap"])
    raise RuntimeError(f"checkpoint {fe} missing")


def step_maturation() -> None:
    outcomes = load_outcomes()
    classifier_p1 = outcomes.loc[
        outcomes["carrier"].eq(CLASSIFIER)
        & outcomes["policy"].eq("p1_raw_per_opportunity")
    ]
    cases = []
    for _, row in classifier_p1.iterrows():
        fes = [int(v) for v in row["switch_fe_sequence"]]
        targets = [str(v) for v in row["switch_target_sequence"]]
        sequence = [str(row["prefix_algorithm"])] + targets
        for k in range(len(fes) - 1):
            interval = fes[k + 1] - fes[k]
            if interval < PREMATURE_BOUND:
                cases.append(
                    {
                        "run_id": row["run_id"],
                        "function": function_number(row["function_id"]),
                        "function_id": row["function_id"],
                        "seed": int(row["seed"]),
                        "prefix": str(row["prefix_algorithm"]),
                        "t0": fes[k],
                        "t1": fes[k + 1],
                        "interval": interval,
                        "forced_algorithm": targets[k],
                        "transition_pair": f"{sequence[k]}->{targets[k]}",
                    }
                )
    print(f"[{TASK}] maturation cases: {len(cases)}", flush=True)

    jobs = []
    for index, case in enumerate(cases):
        case_id = f"case{index:03d}"
        case["case_id"] = case_id
        horizon_targets = [case["t0"] + h for h in HORIZONS if case["t0"] + h <= 10000]
        raw_job = dict(case)
        raw_job.update(
            carrier=CLASSIFIER,
            policy="p1_raw_per_opportunity",
            checkpoints=horizon_targets,
            dwell=0,
            case_id=case_id,
        )
        jobs.append(raw_job)
        for h in HORIZONS:
            if case["t0"] + h > 10000:
                continue
            dwell_job = dict(case)
            dwell_job.update(
                carrier=CLASSIFIER,
                policy=f"p{HORIZONS.index(h) + 2}_dwell_{h}",
                checkpoints=[case["t0"] + h],
                dwell=h,
                case_id=case_id,
            )
            jobs.append(dwell_job)

    results = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_instrumented_worker, job) for job in jobs]
        job_by_id = {(job["case_id"], job["dwell"]): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            job = job_by_id[(result["job"]["case_id"], result["job"]["dwell"])]
            results[(result["job"]["case_id"], result["job"]["dwell"])] = {
                "checkpoints": result["checkpoints"],
                "terminal_log10_gap": result["terminal_log10_gap"],
            }

    branch_rows = []
    summary_rows = []
    for index, case in enumerate(cases):
        case_id = case["case_id"]
        raw = results[(case_id, 0)]
        recorded = float(
            classifier_p1.loc[classifier_p1["run_id"].eq(case["run_id"]), "log10_gap"].iloc[0]
        )
        if abs(raw["terminal_log10_gap"] - recorded) > 1e-12:
            raise RuntimeError(
                f"raw replay mismatch for {case['run_id']}: "
                f"{raw['terminal_log10_gap']} vs {recorded}"
            )
        for h in HORIZONS:
            checkpoint = case["t0"] + h
            if checkpoint > 10000:
                continue
            dwell = results[(case_id, h)]
            raw_recorded_fe, raw_gap = _checkpoint_value(raw["checkpoints"], checkpoint)
            forced_recorded_fe, forced_gap = _checkpoint_value(
                dwell["checkpoints"], checkpoint
            )
            branch_rows.append(
                {
                    **{key: case[key] for key in case if key != "case_id"},
                    "h": h,
                    "checkpoint_requested": checkpoint,
                    "raw_recorded_fe": raw_recorded_fe,
                    "forced_recorded_fe": forced_recorded_fe,
                    "raw_log10_gap_at_h": raw_gap,
                    "forced_b_log10_gap_at_h": forced_gap,
                    "m_h_raw_minus_forced": raw_gap - forced_gap,
                    "raw_terminal_log10_gap": raw["terminal_log10_gap"],
                    "resume_terminal_log10_gap": dwell["terminal_log10_gap"],
                }
            )
    branches = pd.DataFrame(branch_rows)
    branches.to_parquet(TASK10_ROOT / "maturation_branches.parquet", index=False)
    for h in HORIZONS:
        subset = branches.loc[branches["h"].eq(h)]
        if subset.empty:
            continue
        m = subset["m_h_raw_minus_forced"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "h": h,
                "cases": int(len(subset)),
                "mean_m_h": float(m.mean()),
                "median_m_h": float(np.median(m)),
                "fraction_m_h_above_zero": float((m > 0).mean()),
                "fraction_m_h_below_zero": float((m < 0).mean()),
                "mean_resume_terminal_gain_over_raw": float(
                    (
                        subset["raw_terminal_log10_gap"]
                        - subset["resume_terminal_log10_gap"]
                    ).mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(TASK10_ROOT / "maturation_summary.parquet", index=False)
    by_algorithm = (
        branches.groupby(["h", "forced_algorithm"], sort=False)
        .agg(
            cases=("m_h_raw_minus_forced", "size"),
            mean_m_h=("m_h_raw_minus_forced", "mean"),
            median_m_h=("m_h_raw_minus_forced", "median"),
            fraction_above_zero=("m_h_raw_minus_forced", lambda v: float((v > 0).mean())),
        )
        .reset_index()
    )
    by_algorithm.to_parquet(TASK10_ROOT / "maturation_by_target_algorithm.parquet", index=False)
    by_pair = (
        branches.groupby(["h", "transition_pair"], sort=False)
        .agg(
            cases=("m_h_raw_minus_forced", "size"),
            mean_m_h=("m_h_raw_minus_forced", "mean"),
            median_m_h=("m_h_raw_minus_forced", "median"),
        )
        .reset_index()
    )
    by_pair.to_parquet(TASK10_ROOT / "maturation_by_transition_pair.parquet", index=False)
    by_function = (
        branches.groupby(["h", "function_id"], sort=False)
        .agg(cases=("m_h_raw_minus_forced", "size"), mean_m_h=("m_h_raw_minus_forced", "mean"))
        .reset_index()
    )
    by_function.to_parquet(TASK10_ROOT / "maturation_by_function.parquet", index=False)
    print(summary.to_string(), flush=True)
    print(by_algorithm.to_string(), flush=True)


# ---------------------------------------------------------------------------
# step: curves (10G)
# ---------------------------------------------------------------------------

def step_curves() -> None:
    outcomes = load_outcomes()
    rows = []
    for carrier in CARRIERS:
        subset = outcomes.loc[
            outcomes["carrier"].eq(carrier)
            & outcomes["policy"].eq("p1_raw_per_opportunity")
            & (outcomes["switch_count"] > 0)
        ]
        jobs = []
        for _, row in subset.iterrows():
            fes = [int(v) for v in row["switch_fe_sequence"]]
            checkpoints = sorted(
                {
                    fe + offset
                    for fe in fes
                    for offset in (*CURVE_OFFSETS,)
                    if fe + offset <= int(row["FE_total"])
                }
                | set(fes)
            )
            jobs.append(
                {
                    "carrier": carrier,
                    "policy": "p1_raw_per_opportunity",
                    "run_id": row["run_id"],
                    "function": function_number(row["function_id"]),
                    "function_id": row["function_id"],
                    "seed": int(row["seed"]),
                    "prefix": str(row["prefix_algorithm"]),
                    "targets": [str(v) for v in row["switch_target_sequence"]],
                    "switch_fes": fes,
                    "recorded_log10_gap": float(row["log10_gap"]),
                    "checkpoints": checkpoints,
                }
            )
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_instrumented_worker, jobs))
        for job, result in zip(jobs, results):
            if abs(result["terminal_log10_gap"] - job["recorded_log10_gap"]) > 1e-12:
                raise RuntimeError(f"replay mismatch for {job['run_id']}")
            lookup = {int(row["checkpoint_FE"]): row for row in result["checkpoints"]}
            for switch_index, fe in enumerate(job["switch_fes"]):
                base = lookup[fe]
                entry = {
                    "carrier": carrier,
                    "run_id": job["run_id"],
                    "function_id": job["function_id"],
                    "prefix": job["prefix"],
                    "target_algorithm": job["targets"][switch_index],
                    "switch_fe": fe,
                    "log10_gap_at_switch": float(base["log10_gap"]),
                }
                for offset in CURVE_OFFSETS:
                    mark = fe + offset
                    if mark in lookup:
                        entry[f"recorded_fe_plus_{offset}"] = int(
                            lookup[mark]["recorded_FE"]
                        )
                        entry[f"log10_gap_plus_{offset}"] = float(
                            lookup[mark]["log10_gap"]
                        )
                        entry[f"progress_p{offset}"] = float(
                            base["log10_gap"] - float(lookup[mark]["log10_gap"])
                        )
                    else:
                        entry[f"log10_gap_plus_{offset}"] = None
                        entry[f"progress_p{offset}"] = None
                rows.append(entry)
    curves = pd.DataFrame(rows)
    curves.to_parquet(TASK10_ROOT / "post_switch_progress_curves.parquet", index=False)
    agg_rows = []
    for (carrier, target), group in curves.groupby(["carrier", "target_algorithm"], sort=False):
        entry = {
            "carrier": carrier,
            "target_algorithm": target,
            "switches": int(len(group)),
        }
        for offset in CURVE_OFFSETS:
            values = group[f"progress_p{offset}"].dropna().to_numpy(dtype=float)
            entry[f"mean_p{offset}"] = float(values.mean()) if values.size else None
            entry[f"median_p{offset}"] = float(np.median(values)) if values.size else None
        if {"progress_p200", "progress_p500"}.issubset(group.columns):
            paired = group.dropna(subset=["progress_p200", "progress_p500"])
            entry["share_p500_above_p200"] = float(
                (paired["progress_p500"] > paired["progress_p200"]).mean()
            )
        agg_rows.append(entry)
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_parquet(TASK10_ROOT / "progress_curve_summary.parquet", index=False)
    print(aggregate.to_string(), flush=True)


# ---------------------------------------------------------------------------
# step: confound (transfer-restart vs native-state restoration)
# ---------------------------------------------------------------------------

def _replay_classifier_raw(
    *,
    function: int,
    seed: int,
    prefix: str,
    save_switch: int | None = None,
    intercept_switch: int | None = None,
) -> dict:
    """Exact replay of the classifier raw per-opportunity policy.

    Mirrors run_one_switch_policy with switch_rule=per_opportunity, dwell=0,
    hysteresis=0. When ``save_switch`` is set, a deep copy of the source state
    is captured just before that transfer executes. When ``intercept_switch``
    is set, that switch is predicted but NOT executed; the replay returns the
    pre-transfer state together with the state that would have been replaced.
    """
    config = load_experiment_config(CEC_CONFIG)
    bundle = joblib.load(CARRIERS[CLASSIFIER])
    suite = _selected_suites(config, ("cec2017",))[0]
    problem = make_experiment_problem(
        suite,
        function=function,
        instance=1,
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    evaluation_count = 0
    global_native_updates = 1
    current_algorithm = prefix
    switch_fes: list[int] = []
    switch_targets: list[str] = []
    saved_state = None
    intercept_result = None
    recorder = TrajectoryRecorder(sampling_protocol=config.sampling_protocol)

    def observe_evaluation(point: np.ndarray, value: float) -> None:
        nonlocal evaluation_count
        evaluation_count += 1

    def observe_update(updated) -> None:
        nonlocal global_native_updates
        global_native_updates += 1
        recorder.observe(
            problem=problem,
            algorithm=current_algorithm,
            seed=seed,
            fe=evaluation_count,
            fe_total=config.fe_total,
            native_updates=global_native_updates,
            population=updated.population,
            fitness=updated.fitness,
            best_fitness=updated.best_fitness,
        )

    try:
        settings = OptimizerSettings(
            population_size=config.population_size,
            sampling_protocol=config.sampling_protocol,
            boundary_handling=config.boundary_handling,
        )
        state = initialize_optimizer_state(
            algorithm=prefix,
            problem=problem,
            seed=seed,
            settings=settings,
            on_evaluation=observe_evaluation,
        )
        recorder.observe(
            problem=problem,
            algorithm=prefix,
            seed=seed,
            fe=state.evaluations,
            fe_total=config.fe_total,
            native_updates=1,
            population=state.population,
            fitness=state.fitness,
            best_fitness=state.best_fitness,
        )
        while evaluation_count < config.fe_total:
            previous_records = len(recorder.records)
            advance_optimizer_state(
                state=state,
                problem=problem,
                fe_budget=min(
                    config.population_size,
                    config.fe_total - evaluation_count,
                ),
                on_native_update=observe_update,
                on_evaluation=observe_evaluation,
            )
            if len(recorder.records) == previous_records:
                continue
            record = recorder.records[-1]
            behavior = extract_behavior_rows([asdict(record)])[0]
            scores, _ = predict_switch_scores(
                bundle=bundle,
                behavior=behavior,
                prefix_algorithm=current_algorithm,
            )
            candidates = [
                algorithm for algorithm in config.algorithms if algorithm != current_algorithm
            ]
            selected = max(
                candidates,
                key=lambda algorithm: (
                    float(scores[algorithm]),
                    -config.algorithms.index(algorithm),
                ),
            )
            if float(scores[selected]) <= float(bundle["decision_threshold"]):
                continue
            executed = len(switch_fes) + 1
            if intercept_switch is not None and executed == intercept_switch:
                intercept_result = {
                    "current_state": state,
                    "t1": int(record.FE),
                    "return_target": selected,
                    "event_if_executed": NO_QUERY_TRANSFER_EVENT + executed,
                }
                break
            if save_switch is not None and executed == save_switch:
                saved_state = clone_optimizer_state(state)
            switch_fes.append(int(record.FE))
            switch_targets.append(selected)
            state = initialize_transferred_optimizer_state(
                algorithm=selected,
                source_state=state,
                problem=problem,
                seed=seed,
                function=function,
                instance=1,
                event=NO_QUERY_TRANSFER_EVENT + executed,
            )
            current_algorithm = selected
        if intercept_result is None:
            reference = problem.reference_value
            final_gap = min(
                max(float(state.best_fitness) - float(reference), 0.0),
                config.failure_loss_cap,
            )
            intercept_result = {
                "final_gap": final_gap,
                "log10_gap": float(
                    np.log10(np.clip(final_gap, config.log10_gap_floor, config.log10_gap_cap))
                ),
            }
    finally:
        problem.close()
    return {
        "switch_fes": switch_fes,
        "switch_targets": switch_targets,
        "saved_state": saved_state,
        **intercept_result,
    }


def step_confound() -> None:
    outcomes = load_outcomes()
    classifier_p1 = outcomes.loc[
        outcomes["carrier"].eq(CLASSIFIER)
        & outcomes["policy"].eq("p1_raw_per_opportunity")
    ]
    subset = classifier_p1.loc[classifier_p1["switch_count"] >= 2]
    selected = []
    for _, row in subset.iterrows():
        fes = [int(v) for v in row["switch_fe_sequence"]]
        sequence = [str(row["prefix_algorithm"])] + [
            str(v) for v in row["switch_target_sequence"]
        ]
        for k in range(len(fes) - 1):
            if k + 2 < len(sequence) and sequence[k] == sequence[k + 2] and sequence[k + 1] != sequence[k]:
                selected.append(
                    {
                        "run_id": row["run_id"],
                        "function": function_number(row["function_id"]),
                        "function_id": row["function_id"],
                        "seed": int(row["seed"]),
                        "prefix": str(row["prefix_algorithm"]),
                        "save_switch_1based": k + 1,
                        "intercept_switch_1based": k + 2,
                        "algorithm_a": sequence[k],
                        "algorithm_b": sequence[k + 1],
                        "recorded_log10_gap": float(row["log10_gap"]),
                    }
                )
                break
    selected = selected[:5]
    rows = []
    for case in selected:
        full = _replay_classifier_raw(
            function=case["function"],
            seed=case["seed"],
            prefix=case["prefix"],
        )
        if abs(full["log10_gap"] - case["recorded_log10_gap"]) > 1e-12 or list(
            full["switch_fes"]
        ) != _recorded_switch_fes(classifier_p1, case["run_id"]):
            raise RuntimeError(f"replay mismatch for confound case {case['run_id']}")
        fork = _replay_classifier_raw(
            function=case["function"],
            seed=case["seed"],
            prefix=case["prefix"],
            save_switch=case["save_switch_1based"],
            intercept_switch=case["intercept_switch_1based"],
        )
        config = load_experiment_config(CEC_CONFIG)
        suite = _selected_suites(config, ("cec2017",))[0]
        problem = make_experiment_problem(
            suite,
            function=case["function"],
            instance=1,
            dimension=config.dimension,
            boundary_handling=config.boundary_handling,
        )
        remaining = config.fe_total - fork["t1"]
        branches = {}
        # branch 1: what the raw policy actually does - population transfer reinit of A
        transferred = initialize_transferred_optimizer_state(
            algorithm=case["algorithm_a"],
            source_state=fork["current_state"],
            problem=problem,
            seed=case["seed"],
            function=case["function"],
            instance=1,
            event=fork["event_if_executed"],
        )
        advance_optimizer_state(
            state=transferred,
            problem=problem,
            fe_budget=remaining,
        )
        reference = float(problem.reference_value)
        branches["transfer_reinit_a"] = float(
            np.log10(
                np.clip(
                    min(
                        max(float(transferred.best_fitness) - reference, 0.0),
                        config.failure_loss_cap,
                    ),
                    config.log10_gap_floor,
                    config.log10_gap_cap,
                )
            )
        )
        # branch 2: restore the original native A state captured when A was left
        restored = clone_optimizer_state(fork["saved_state"])
        advance_optimizer_state(
            state=restored,
            problem=problem,
            fe_budget=remaining,
        )
        branches["restore_native_a"] = float(
            np.log10(
                np.clip(
                    min(
                        max(float(restored.best_fitness) - reference, 0.0),
                        config.failure_loss_cap,
                    ),
                    config.log10_gap_floor,
                    config.log10_gap_cap,
                )
            )
        )
        problem.close()
        rows.append(
            {
                **{key: case[key] for key in case if key != "recorded_log10_gap"},
                "t0": int(fork["switch_fes"][-1]) if fork["switch_fes"] else None,
                "t1": int(fork["t1"]),
                "log10_gap_transfer_reinit_a": branches["transfer_reinit_a"],
                "log10_gap_restore_native_a": branches["restore_native_a"],
                "restoration_gain": branches["restore_native_a"]
                - branches["transfer_reinit_a"],
            }
        )
        print(f"[{TASK}] confound case done: {case['run_id']}", flush=True)
    table = pd.DataFrame(rows)
    TASK10_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(TASK10_ROOT / "transfer_restart_confound.parquet", index=False)
    print(table.to_string(), flush=True)


def _recorded_switch_fes(outcomes: pd.DataFrame, run_id: str) -> list[int]:
    row = outcomes.loc[outcomes["run_id"].eq(run_id)].iloc[0]
    return [int(v) for v in row["switch_fe_sequence"]]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        required=True,
        choices=("ablation", "analysis", "maturation", "curves", "confound"),
    )
    args = parser.parse_args()
    TASK10_ROOT.mkdir(parents=True, exist_ok=True)
    if args.step == "ablation":
        step_ablation()
    elif args.step == "analysis":
        step_analysis()
    elif args.step == "maturation":
        step_maturation()
    elif args.step == "curves":
        step_curves()
    elif args.step == "confound":
        step_confound()


if __name__ == "__main__":
    main()
