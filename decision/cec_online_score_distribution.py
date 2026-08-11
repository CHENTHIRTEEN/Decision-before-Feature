from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from behavior.features import extract_behavior_rows
from benchmarks import make_problem
from decision.model_protocol import FROZEN_THRESHOLD_MODE, decision_scores, resolve_model_name
from decision.query_contract import decision_query_root, validate_query_payload
from decision.online_controller_evaluate import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_SAMPLING_PROTOCOL,
    DEFAULT_TRAIN_CONFIG_PATH,
    SAMPLING_PROTOCOLS,
    _checkpoint_plan,
    _checkpoint_ratios,
    _decision_check_frequency,
    _model_family,
    _model_path,
    _threshold,
)
from experiments.phase1_batch_common import as_int_list, fe_total_for_dimension, load_config, selected_dimensions, selected_functions
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS, get_query_spec
from optimizers import OptimizerSettings, advance_optimizer_state, initialize_optimizer_state
from selection_reference.common import read_performance, single_best_solver
from trajectory.records import TrajectoryRecord


DEFAULT_NEAR_ZERO_THRESHOLD = -0.05
TOP_SCORE_ROWS = 200


def diagnose_online_score_distribution(
    *,
    query_id: str,
    config_path: Path,
    train_config_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    model_name: str,
    sampling_protocol: str,
    near_zero_threshold: float,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    only_seeds: list[int] | None,
    max_runs: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    _check_output_paths(output_dir, overwrite)
    config = load_config(config_path)
    if str(config["suite"]).lower() not in {"cec2017", "cec2022"}:
        raise ValueError("online score distribution currently expects a CEC suite")
    train_config = load_config(train_config_path)
    default_algorithm = single_best_solver(read_performance(train_config, None, None))
    controller = _load_score_controller(training_summary_path, model_name, query_id)
    checkpoint_ratios = _checkpoint_ratios(config, sampling_protocol)
    decision_check_frequency = _decision_check_frequency(sampling_protocol)
    functions = selected_functions(config, only_functions)
    dimensions = selected_dimensions(config, only_dimensions)
    seeds = _selected_seeds(config, only_seeds)
    checkpoint_plan = {
        dimension: _checkpoint_plan(config, dimension, checkpoint_ratios)
        for dimension in dimensions
    }

    rows = []
    run_counter = 0
    started = perf_counter()
    for function in functions:
        for dimension in dimensions:
            fe_total = fe_total_for_dimension(config, dimension)
            for seed in seeds:
                if max_runs is not None and run_counter >= max_runs:
                    break
                rows.extend(
                    _score_one_run(
                        config=config,
                        function=function,
                        dimension=dimension,
                        seed=seed,
                        fe_total=fe_total,
                        checkpoint_plan=checkpoint_plan[dimension],
                        default_algorithm=default_algorithm,
                        controller=controller,
                        sampling_protocol=sampling_protocol,
                        decision_check_frequency=decision_check_frequency,
                    )
                )
                run_counter += 1
            if max_runs is not None and run_counter >= max_runs:
                break
        if max_runs is not None and run_counter >= max_runs:
            break
    runtime_seconds = perf_counter() - started
    score_rows = pd.DataFrame(rows)
    if score_rows.empty:
        raise ValueError("online score distribution produced no rows")
    score_rows["score_ge_near_zero"] = score_rows["decision_score"] > near_zero_threshold
    score_summary = _score_summary(score_rows, near_zero_threshold)
    opportunity_rows = score_rows[score_rows["score_ge_near_zero"] | score_rows["score_ge_zero"]].copy()
    top_score_rows = score_rows.sort_values("decision_score", ascending=False).head(TOP_SCORE_ROWS).copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(score_rows, output_dir / "online_score_rows")
    _write_frame(score_summary, output_dir / "online_score_summary")
    _write_frame(opportunity_rows, output_dir / "online_trigger_opportunity_rows")
    _write_frame(top_score_rows, output_dir / "online_top_score_rows")

    summary = {
        "experiment": "cec_online_controller_score_distribution",
        "query_id": query_id,
        "query_protocol": get_query_spec(query_id).protocol,
        "sample_design_id": get_query_spec(query_id).sample_design_id,
        "research_question": (
            "Which CEC online decision-check states receive Decision scores near or above zero under the frozen "
            "BBOB-trained controller?"
        ),
        "config": str(config_path),
        "train_config": str(train_config_path),
        "training_summary": str(training_summary_path),
        "model_name": controller["model_name"],
        "model_family": controller["model_family"],
        "sampling_protocol": sampling_protocol,
        "decision_check_frequency": decision_check_frequency,
        "checkpoint_ratios": [float(value) for value in checkpoint_ratios],
        "default_algorithm": default_algorithm,
        "thresholds": {
            "zero": 0.0,
            FROZEN_THRESHOLD_MODE: float(controller["oof_utility_threshold"]),
            "near_zero": near_zero_threshold,
        },
        "rows": int(len(score_rows)),
        "base_runs": int(run_counter),
        "runtime_seconds": float(runtime_seconds),
        "outputs": {
            "score_rows": str(output_dir / "online_score_rows.parquet"),
            "score_summary": str(output_dir / "online_score_summary.parquet"),
            "trigger_opportunity_rows": str(output_dir / "online_trigger_opportunity_rows.parquet"),
            "top_score_rows": str(output_dir / "online_top_score_rows.parquet"),
            "report": str(output_dir / "online_score_distribution_report.md"),
            "summary": str(output_dir / "online_score_distribution_summary.json"),
        },
        "data_leakage_check": {
            "external_rows_used_for_controller_fit": 0,
            "external_rows_used_for_threshold_fit": 0,
            "external_query_features_used_as_controller_input": False,
            "function_id_algorithm_id_or_optimizer_internal_parameters_used_as_controller_input": False,
            "controller_inputs_are_behavior_features_only": True,
            "query_branch_executed": False,
            "utility_labels_regenerated": False,
        },
        "scope_notes": [
            "This diagnostic scores online default-probe checkpoint states only.",
            "It does not run the fixed query, selection reference, or post-decision query continuation.",
            "Near-zero rows use the descriptive threshold supplied by --near-zero-threshold.",
        ],
    }
    summary_path = output_dir / "online_score_distribution_summary.json"
    report_path = output_dir / "online_score_distribution_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report_path.write_text(
        _markdown_report(
            summary=summary,
            score_summary=score_summary,
            opportunity_rows=opportunity_rows,
            top_score_rows=top_score_rows,
        ),
        encoding="utf-8",
    )
    print(f"wrote online score rows to {output_dir / 'online_score_rows.parquet'}")
    print(f"wrote online score distribution report to {report_path}")
    return summary


def _check_output_paths(output_dir: Path, overwrite: bool) -> None:
    outputs = (
        output_dir / "online_score_rows.csv",
        output_dir / "online_score_rows.parquet",
        output_dir / "online_score_summary.csv",
        output_dir / "online_score_summary.parquet",
        output_dir / "online_trigger_opportunity_rows.csv",
        output_dir / "online_trigger_opportunity_rows.parquet",
        output_dir / "online_top_score_rows.csv",
        output_dir / "online_top_score_rows.parquet",
        output_dir / "online_score_distribution_report.md",
        output_dir / "online_score_distribution_summary.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"online score distribution outputs already exist; pass --overwrite: {existing[0]}")


def _load_score_controller(training_summary_path: Path, model_name: str, query_id: str) -> dict[str, Any]:
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    validate_query_payload(summary, query_id=query_id, artifact="Decision training summary")
    model_name = resolve_model_name(summary, model_name)
    feature_columns = [str(column) for column in summary.get("feature_columns", [])]
    if not feature_columns:
        raise ValueError("training summary does not define feature_columns")
    model_path = _model_path(summary, model_name)
    return {
        "model": joblib.load(model_path),
        "model_name": model_name,
        "model_family": _model_family(summary, model_name),
        "feature_columns": feature_columns,
        "model_path": model_path,
        "zero_threshold": 0.0,
        "oof_utility_threshold": _threshold(summary, model_name, FROZEN_THRESHOLD_MODE),
        "query_id": str(summary["query_id"]),
        "query_protocol": str(summary["query_protocol"]),
        "sample_design_id": str(summary["sample_design_id"]),
    }


def _selected_seeds(config: dict, only_seeds: list[int] | None) -> list[int]:
    seeds = as_int_list(config, "seeds")
    if only_seeds is None:
        return seeds
    requested = set(int(seed) for seed in only_seeds)
    missing = sorted(requested.difference(seeds))
    if missing:
        raise ValueError(f"requested seeds are not in config: {missing}")
    return [seed for seed in seeds if seed in requested]


def _score_one_run(
    *,
    config: dict,
    function: int,
    dimension: int,
    seed: int,
    fe_total: int,
    checkpoint_plan: list[tuple[float, int]],
    default_algorithm: str,
    controller: dict[str, Any],
    sampling_protocol: str,
    decision_check_frequency: str,
) -> list[dict[str, Any]]:
    suite = str(config["suite"]).lower()
    problem = make_problem({"suite": suite, "function": function, "instance": 1, "dimension": dimension})
    try:
        population_size = int(config["population_size"])
        trajectory_rows: list[dict[str, Any]] = []
        settings = OptimizerSettings(population_size=population_size, checkpoint_ratios=(1.0,))
        started = perf_counter()
        current_state = initialize_optimizer_state(
            algorithm=default_algorithm,
            problem=problem,
            seed=seed,
            settings=settings,
        )
        runtime_probe = perf_counter() - started
        current_fe = int(current_state.evaluations)
        rows = []
        for ratio, checkpoint_fe in checkpoint_plan:
            delta = checkpoint_fe - current_fe
            if delta <= 0:
                continue
            continuation = advance_optimizer_state(state=current_state, problem=problem, fe_budget=delta)
            runtime_probe += continuation.runtime_seconds
            current_fe = checkpoint_fe
            trajectory_record = TrajectoryRecord.from_arrays(
                problem_id=problem.problem_id,
                family=problem.family,
                dimension=problem.dimension,
                algorithm=default_algorithm,
                seed=seed,
                fe=current_fe,
                fe_total=fe_total,
                native_updates=int(current_state.generation),
                population=current_state.population,
                fitness=current_state.fitness,
                best_fitness=current_state.best_fitness,
                fe_ratio=ratio,
            )
            trajectory_rows.append(trajectory_record.__dict__)
            behavior_row = extract_behavior_rows([row.copy() for row in trajectory_rows])[-1]
            score = _score_behavior(controller, behavior_row)
            rows.append(
                {
                    "split": _split_name(config),
                    "query_id": controller["query_id"],
                    "query_protocol": controller["query_protocol"],
                    "sample_design_id": controller["sample_design_id"],
                    "suite": suite,
                    "problem_id": problem.problem_id,
                    "family": problem.family,
                    "function": int(function),
                    "dimension": int(dimension),
                    "seed": int(seed),
                    "default_algorithm": default_algorithm,
                    "FE": int(current_fe),
                    "FE_total": int(fe_total),
                    "FE_ratio": float(ratio),
                    "best_fitness": float(current_state.best_fitness),
                    "decision_score": float(score),
                    "score_margin_to_zero": float(score),
                    "score_margin_to_oof_utility": float(score - controller["oof_utility_threshold"]),
                    "score_ge_zero": bool(score > 0.0),
                    "score_ge_oof_utility": bool(score > controller["oof_utility_threshold"]),
                    "sampling_protocol": sampling_protocol,
                    "decision_check_frequency": decision_check_frequency,
                    "checkpoint_index": int(len(trajectory_rows)),
                    "checkpoint_count": int(len(checkpoint_plan)),
                    "runtime_probe_cumulative": float(runtime_probe),
                }
            )
        return rows
    finally:
        problem.close()


def _score_behavior(controller: dict[str, Any], behavior_row: dict[str, Any]) -> float:
    frame = pd.DataFrame([{column: behavior_row[column] for column in controller["feature_columns"]}])
    score = float(decision_scores(controller["model"], frame)[0])
    if not np.isfinite(score):
        raise ValueError("controller produced non-finite score")
    return score


def _score_summary(score_rows: pd.DataFrame, near_zero_threshold: float) -> pd.DataFrame:
    frame = score_rows.copy()
    frame["score_ge_near_zero"] = frame["decision_score"] > near_zero_threshold
    layers = {
        "overall": [],
        "function": ["function"],
        "dimension": ["dimension"],
        "function_dimension": ["function", "dimension"],
        "FE_ratio": ["FE_ratio"],
        "function_FE_ratio": ["function", "FE_ratio"],
        "dimension_FE_ratio": ["dimension", "FE_ratio"],
    }
    rows = []
    for layer, columns in layers.items():
        groups = [((), frame)] if not columns else frame.groupby(columns, dropna=False, sort=True)
        for values, subset in groups:
            if columns and not isinstance(values, tuple):
                values = (values,)
            group = dict(zip(columns, values, strict=False)) if columns else {}
            rows.append(_summary_row(subset, layer, group))
    return pd.DataFrame(rows).sort_values(["layer", "group"]).reset_index(drop=True)


def _summary_row(frame: pd.DataFrame, layer: str, group: dict[str, Any]) -> dict[str, Any]:
    scores = frame["decision_score"].to_numpy(dtype=float)
    return {
        "layer": layer,
        "group": _group_label(group),
        "function": group.get("function"),
        "dimension": group.get("dimension"),
        "FE_ratio": group.get("FE_ratio"),
        "rows": int(len(frame)),
        "run_count": int(frame[["problem_id", "dimension", "seed"]].drop_duplicates().shape[0]),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_q90": float(np.quantile(scores, 0.90)),
        "score_q95": float(np.quantile(scores, 0.95)),
        "score_q99": float(np.quantile(scores, 0.99)),
        "score_ge_near_zero_rows": int(frame["score_ge_near_zero"].sum()),
        "score_ge_near_zero_rate": float(frame["score_ge_near_zero"].mean()),
        "score_ge_zero_rows": int(frame["score_ge_zero"].sum()),
        "score_ge_zero_rate": float(frame["score_ge_zero"].mean()),
        "score_ge_oof_utility_rows": int(frame["score_ge_oof_utility"].sum()),
        "score_ge_oof_utility_rate": float(frame["score_ge_oof_utility"].mean()),
    }


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stem.with_suffix(".parquet"))


def _markdown_report(
    *,
    summary: dict[str, Any],
    score_summary: pd.DataFrame,
    opportunity_rows: pd.DataFrame,
    top_score_rows: pd.DataFrame,
) -> str:
    overall = score_summary[score_summary["layer"] == "overall"]
    function_dimension = score_summary[score_summary["layer"] == "function_dimension"].sort_values(
        ["score_ge_zero_rows", "score_max"], ascending=False
    )
    fe_ratio = score_summary[score_summary["layer"] == "FE_ratio"].sort_values("FE_ratio")
    top_columns = [
        "problem_id",
        "function",
        "dimension",
        "seed",
        "FE_ratio",
        "FE",
        "decision_score",
        "score_ge_zero",
        "score_ge_oof_utility",
        "best_fitness",
    ]
    return "\n".join(
        [
            "# CEC online score distribution report",
            "",
            "## Scope",
            "",
            "- Scores are computed along the online default-probe trajectory at each decision-check checkpoint.",
            "- Fixed-query and selection-reference branches are not executed.",
            f"- Near-zero threshold: `{summary['thresholds']['near_zero']}`.",
            f"- Frozen train-OOF threshold: `{summary['thresholds'][FROZEN_THRESHOLD_MODE]}`.",
            "",
            "## Overall",
            "",
            _markdown_table(overall),
            "",
            "## FE-ratio summary",
            "",
            _markdown_table(
                fe_ratio[
                    [
                        "FE_ratio",
                        "rows",
                        "score_mean",
                        "score_max",
                        "score_q95",
                        "score_ge_near_zero_rows",
                        "score_ge_zero_rows",
                        "score_ge_oof_utility_rows",
                    ]
                ]
            ),
            "",
            "## Function-dimension opportunity summary",
            "",
            _markdown_table(
                function_dimension[
                    [
                        "function",
                        "dimension",
                        "rows",
                        "score_max",
                        "score_q95",
                        "score_ge_near_zero_rows",
                        "score_ge_zero_rows",
                        "score_ge_oof_utility_rows",
                    ]
                ].head(40)
            ),
            "",
            "## Top score rows",
            "",
            _markdown_table(top_score_rows[top_columns].head(40)),
            "",
            "## Near-zero or above-zero rows",
            "",
            _markdown_table(opportunity_rows[top_columns].head(80)),
            "",
            "## Output files",
            "",
            _markdown_table(pd.DataFrame([{"name": key, "path": value} for key, value in summary["outputs"].items()])),
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "all"
    return ", ".join(f"{key}={value}" for key, value in group.items())


def _split_name(config: dict) -> str:
    return Path(config["output"]).stem.removesuffix("_trajectories")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CEC online controller score distribution by checkpoint.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG_PATH)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sampling-protocol", choices=SAMPLING_PROTOCOLS, default=DEFAULT_SAMPLING_PROTOCOL)
    parser.add_argument("--near-zero-threshold", type=float, default=DEFAULT_NEAR_ZERO_THRESHOLD)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--only-seed", type=int, action="append", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    query_root = decision_query_root(args.query_id)
    split = _split_name(load_config(args.config))
    diagnose_online_score_distribution(
        query_id=args.query_id,
        config_path=args.config,
        train_config_path=args.train_config,
        training_summary_path=args.training_summary
        or query_root
        / "feature_group_ablation/primary_with_maturity/full_decision_model_training_summary.json",
        output_dir=args.output_dir or query_root / split / "online_score_distribution",
        model_name=args.model_name,
        sampling_protocol=args.sampling_protocol,
        near_zero_threshold=args.near_zero_threshold,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        only_seeds=args.only_seed,
        max_runs=args.max_runs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
