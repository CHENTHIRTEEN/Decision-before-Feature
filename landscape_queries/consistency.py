from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from landscape_queries.batch_features import FEATURE_METADATA_COLUMNS
from landscape_queries.batch_sampling import SAMPLE_KEY_COLUMNS
from landscape_queries.specs import (
    LANDSCAPE_QUERY_SPECS,
    PFLACCO_GROUP_COLUMNS,
    SAMPLE_DESIGN_SPECS,
    get_query_spec,
    get_sample_design_spec,
    validate_frozen_query_specs,
)


SAMPLE_ENDPOINT_COLUMNS = {
    "benchmark_reference_value",
    "success_gap_target",
    "query_success",
    "query_first_hit_offset",
    "query_best_gap",
}


def check_landscape_query_consistency(
    *,
    sample_paths: list[Path],
    feature_paths: list[Path],
    action_loss_paths: list[Path],
) -> dict[str, int]:
    validate_frozen_query_specs()
    samples = _read_tables(sample_paths)
    _check_feature_path_schemas(feature_paths)
    features = _read_tables(feature_paths)
    action_losses = _read_tables(action_loss_paths)
    if not samples.empty:
        _check_samples(samples)
    if not features.empty:
        _check_features(features, samples)
    if not action_losses.empty:
        _check_action_losses(action_losses)
    return {
        "sample_rows": int(len(samples)),
        "feature_rows": int(len(features)),
        "action_loss_rows": int(len(action_losses)),
    }


def _read_tables(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_dir():
            files = sorted(path.rglob("*.parquet"))
            if not files:
                raise ValueError(f"no parquet files found under {path}")
            frames.extend(pq.read_table(file).to_pandas() for file in files)
        else:
            frames.append(pq.read_table(path).to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _check_feature_path_schemas(paths: list[Path]) -> None:
    missing_endpoint_metadata = SAMPLE_ENDPOINT_COLUMNS.difference(FEATURE_METADATA_COLUMNS)
    if missing_endpoint_metadata:
        raise ValueError(
            "query endpoint fields must be retained as reporting metadata: "
            f"{sorted(missing_endpoint_metadata)}"
        )
    for spec in LANDSCAPE_QUERY_SPECS.values():
        leaked_features = SAMPLE_ENDPOINT_COLUMNS.intersection(spec.feature_columns)
        if leaked_features:
            raise ValueError(
                f"{spec.query_id} feature whitelist contains final-evaluation endpoint fields: "
                f"{sorted(leaked_features)}"
            )
    files = []
    for path in paths:
        files.extend(sorted(path.rglob("features.parquet"))) if path.is_dir() else files.append(path)
    for path in files:
        schema_names = set(pq.read_schema(path).names)
        if "query_id" not in schema_names:
            raise ValueError(
                f"feature artifact is missing query_id and predates the active landscape-query protocol: {path}"
            )
        table = pq.read_table(path, columns=["query_id"])
        query_ids = set(table.column("query_id").to_pylist())
        if len(query_ids) != 1:
            raise ValueError(f"feature file must contain exactly one query_id: {path}")
        spec = get_query_spec(str(next(iter(query_ids))))
        observed = schema_names
        expected = set(FEATURE_METADATA_COLUMNS) | set(spec.feature_columns)
        if observed != expected:
            raise ValueError(
                f"{spec.query_id} feature file schema does not exactly match its whitelist; "
                f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
            )


def _check_samples(samples: pd.DataFrame) -> None:
    required = {
        *SAMPLE_KEY_COLUMNS,
        "FE_total",
        "sampling_protocol",
        "sample_seed",
        "sample_size",
        "FE_query",
        "FE_query_planned",
        "runtime_query_sampling",
        "runtime_query_evaluation",
        "runtime_sampling_evaluation",
        "benchmark_reference_value",
        "success_gap_target",
        "query_success",
        "query_first_hit_offset",
        "query_best_gap",
        "lower_bounds",
        "upper_bounds",
        "X",
        "y",
        "sample_status",
        "sample_path_completed",
        "sample_planned_FE",
        "sample_effective_FE",
        "sample_observed_first_hit_FE",
        "sample_target_hit_observed",
        "sample_target_hit_before_failure",
        "sample_endpoint_success",
        "sample_timed_out",
        "sample_failure_type",
        "sample_failure_message",
        "sample_failure",
    }
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"sample data are missing columns: {sorted(missing)}")
    if samples.duplicated(list(SAMPLE_KEY_COLUMNS)).any():
        raise ValueError("sample keys must be unique")
    if not set(samples["sample_status"].astype(str)).issubset({"ok", "failed"}):
        raise ValueError("sample_status must be ok or failed")
    for row in samples.to_dict(orient="records"):
        design = get_sample_design_spec(str(row["sample_design_id"]))
        if str(row["sampling_protocol"]) != design.protocol:
            raise ValueError("sample row uses an unsupported sampling protocol")
        expected_size = design.sample_size(int(row["dimension"]))
        completed = bool(row["sample_path_completed"])
        if completed != (str(row["sample_status"]) == "ok"):
            raise ValueError("sample_status and sample_path_completed disagree")
        effective_fe = int(row["sample_effective_FE"])
        if (
            int(row["sample_size"]) != expected_size
            or int(row["FE_query_planned"]) != expected_size
            or int(row["sample_planned_FE"]) != expected_size
            or int(row["FE_query"]) != effective_fe
            or not 0 <= effective_fe <= expected_size
            or (completed and effective_fe != expected_size)
        ):
            raise ValueError("sample planned/effective FE accounting is inconsistent")
        x = np.asarray(row["X"], dtype=float).reshape(-1, int(row["dimension"]))
        y = np.asarray(row["y"], dtype=float)
        if x.shape != (effective_fe, int(row["dimension"])) or y.shape != (effective_fe,):
            raise ValueError("saved X or y shape is inconsistent with its sample design")
        if completed and (not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("completed saved X and y must be finite")
        lower = np.asarray(row["lower_bounds"], dtype=float).reshape(-1)
        upper = np.asarray(row["upper_bounds"], dtype=float).reshape(-1)
        dimension = int(row["dimension"])
        bounds_available = lower.shape == (dimension,) and upper.shape == (dimension,)
        if completed and (
            lower.shape != (dimension,)
            or upper.shape != (dimension,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(lower >= upper)
        ):
            raise ValueError("saved query bounds are invalid")
        if bounds_available and (
            np.any(x < lower - 1e-12) or np.any(x > upper + 1e-12)
        ):
            raise ValueError("saved query sample contains points outside the benchmark bounds")

        reference_raw = row["benchmark_reference_value"]
        reference = None if pd.isna(reference_raw) else float(reference_raw)
        target = float(row["success_gap_target"])
        best_gap = float(row["query_best_gap"])
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("query endpoint reference and success target must be finite")
        if not np.isfinite(best_gap) or best_gap < 0.0:
            raise ValueError("query_best_gap must be finite and non-negative")
        gaps = (
            np.maximum(y - reference, 0.0)
            if reference is not None and np.isfinite(reference)
            else np.full(len(y), np.inf, dtype=float)
        )
        finite_gaps = gaps[np.isfinite(gaps)]
        if finite_gaps.size:
            expected_best_gap = float(np.min(finite_gaps))
            if not np.isclose(best_gap, expected_best_gap, rtol=0.0, atol=1e-12):
                raise ValueError("query_best_gap is inconsistent with saved objective values")
        hits = np.flatnonzero(gaps <= target)
        expected_offset = int(hits[0] + 1) if hits.size else None
        raw_offset = row["query_first_hit_offset"]
        observed_offset = None if pd.isna(raw_offset) else int(raw_offset)
        observed_success = bool(row["query_success"])
        if observed_offset is not None and not 1 <= observed_offset <= int(row["FE_query"]):
            raise ValueError("query_first_hit_offset must lie in [1, FE_query]")
        if observed_success != (observed_offset is not None):
            raise ValueError("query_success and query_first_hit_offset are inconsistent")
        if observed_offset != expected_offset:
            raise ValueError("query_first_hit_offset is inconsistent with saved objective values")
        if bool(row["sample_target_hit_observed"]) != observed_success:
            raise ValueError("sample_target_hit_observed is inconsistent")
        if bool(row["sample_target_hit_before_failure"]) != (
            observed_success and not completed
        ):
            raise ValueError("sample_target_hit_before_failure is inconsistent")
        if bool(row["sample_endpoint_success"]) != (observed_success and completed):
            raise ValueError("sample_endpoint_success is inconsistent")
        if not completed and not str(row["sample_failure_type"]):
            raise ValueError("failed sample row requires failure context")
        runtime_parts = (
            float(row["runtime_query_sampling"]),
            float(row["runtime_query_evaluation"]),
        )
        if any(not np.isfinite(value) or value < 0.0 for value in runtime_parts):
            raise ValueError("query sampling runtimes must be finite and non-negative")
        if not np.isclose(
            float(row["runtime_sampling_evaluation"]),
            sum(runtime_parts),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("runtime_sampling_evaluation must equal sampling plus objective evaluation")


def _check_features(features: pd.DataFrame, samples: pd.DataFrame) -> None:
    required_metadata = set(FEATURE_METADATA_COLUMNS)
    missing = required_metadata.difference(features.columns)
    if missing:
        raise ValueError(f"feature data are missing protocol columns: {sorted(missing)}")
    query_ids = set(features["query_id"].astype(str))
    unknown = query_ids.difference(LANDSCAPE_QUERY_SPECS)
    if unknown:
        raise ValueError(f"feature data contain unknown query ids: {sorted(unknown)}")
    key = ["split", "problem_id", "function_id", "family", "function", "instance", "dimension", "sample_design_id"]
    for query_id, frame in features.groupby("query_id", sort=True):
        spec = get_query_spec(str(query_id))
        missing_features = set(spec.feature_columns).difference(frame.columns)
        if missing_features:
            raise ValueError(f"{query_id} feature whitelist is missing columns: {sorted(missing_features)}")
        if set(frame["query_protocol"].astype(str)) != {spec.protocol}:
            raise ValueError(f"{query_id} query_protocol does not match the frozen spec")
        if set(frame["query_preprocessing_id"].astype(str)) != {spec.preprocessing_id}:
            raise ValueError(f"{query_id} query_preprocessing_id does not match the frozen spec")
        if set(frame["sample_design_id"].astype(str)) != {spec.sample_design_id}:
            raise ValueError(f"{query_id} uses the wrong sample design")
        expected_columns_json = json.dumps(list(spec.feature_columns), ensure_ascii=False)
        if set(frame["query_feature_columns"].astype(str)) != {expected_columns_json}:
            raise ValueError(f"{query_id} query_feature_columns does not match the whitelist")
        if frame.duplicated(key).any():
            raise ValueError(f"{query_id} feature sample keys must be unique")
        runtime = frame["runtime_sampling_evaluation"].astype(float) + frame[
            "runtime_feature_computation"
        ].astype(float)
        if not np.allclose(frame["runtime_query"].astype(float), runtime, rtol=0.0, atol=1e-12):
            raise ValueError(f"{query_id} runtime_query is not sampling + feature computation")
        if not np.array_equal(
            frame["runtime_query_feature_computation"].astype(float).to_numpy(),
            frame["runtime_feature_computation"].astype(float).to_numpy(),
        ):
            raise ValueError(f"{query_id} query feature-computation runtime alias is inconsistent")
        if (frame["additional_function_evaluations"].astype(int) != 0).any():
            raise ValueError(f"{query_id} must not perform additional objective evaluations")
        for row in frame.to_dict(orient="records"):
            _check_feature_row_status(row, spec)
        train = frame[frame["split"].astype(str) == "bbob_train"]
        if not train.empty:
            usable_train = train[train["feature_status"].astype(str).eq("ok")]
            all_missing = [
                column for column in spec.feature_columns if usable_train[column].isna().all()
            ]
            if all_missing:
                raise ValueError(f"{query_id} has entirely missing BBOB-train feature columns: {all_missing}")

    if not samples.empty:
        sample_meta = samples[
            key
            + [
                "sample_seed",
                "sample_size",
                "FE_query",
                "runtime_query_sampling",
                "runtime_query_evaluation",
                "runtime_sampling_evaluation",
            ]
        ]
        joined = features.merge(sample_meta, on=key, how="left", suffixes=("_feature", "_sample"), indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError("every feature row must map to exactly one saved X,y sample row")
        for column in (
            "sample_seed",
            "sample_size",
            "FE_query",
            "runtime_query_sampling",
            "runtime_query_evaluation",
            "runtime_sampling_evaluation",
        ):
            left = joined[f"{column}_feature"].astype(float)
            right = joined[f"{column}_sample"].astype(float)
            if not np.array_equal(left.to_numpy(), right.to_numpy()):
                raise ValueError(f"feature-to-sample metadata mismatch: {column}")

    cheap = features[
        features["query_id"].astype(str) == "descriptor_cheap_invariant"
    ]
    standard = features[
        features["query_id"].astype(str) == "pflacco_standard_invariant"
    ]
    if not cheap.empty and not standard.empty:
        shared_key = ["split", "problem_id", "function_id", "family", "function", "instance", "dimension", "sample_design_id"]
        left = cheap[shared_key + ["sample_seed", "sample_size", "FE_query"]].sort_values(shared_key).reset_index(drop=True)
        right = standard[shared_key + ["sample_seed", "sample_size", "FE_query"]].sort_values(shared_key).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError("descriptor_cheap and pflacco_standard do not share identical lhs_50d sample keys and FE")


def _check_feature_row_status(row: dict, spec) -> None:
    group_status = json.loads(str(row["feature_group_status"]))
    if set(group_status) != set(spec.feature_groups):
        raise ValueError(f"{spec.query_id} group status does not cover the frozen groups")
    nonfinite = json.loads(str(row["feature_nonfinite"]))
    if not isinstance(nonfinite, dict) or set(nonfinite).difference(spec.feature_groups):
        raise ValueError(f"{spec.query_id} feature_nonfinite must be grouped by frozen feature group")
    failures = json.loads(str(row["feature_failure"]))
    if not isinstance(failures, list):
        raise ValueError(f"{spec.query_id} feature_failure must be a JSON list")

    failed_groups = []
    for group in spec.feature_groups:
        columns = spec.feature_columns if group == "descriptor_cheap" else PFLACCO_GROUP_COLUMNS[group]
        status = group_status[group]
        required_status = {"status", "runtime_seconds", "nonfinite_columns", "warnings", "error"}
        if not isinstance(status, dict) or set(status) != required_status:
            raise ValueError(f"{spec.query_id} group {group} has an incomplete status record")
        runtime = float(status["runtime_seconds"])
        if not np.isfinite(runtime) or runtime < 0.0:
            raise ValueError(f"{spec.query_id} group {group} runtime must be finite and non-negative")
        recorded_nonfinite = [str(column) for column in status["nonfinite_columns"]]
        if recorded_nonfinite != [str(column) for column in nonfinite.get(group, [])]:
            raise ValueError(f"{spec.query_id} group {group} non-finite records disagree")
        if set(recorded_nonfinite).difference(columns):
            raise ValueError(f"{spec.query_id} group {group} records non-whitelisted non-finite columns")
        null_columns = [column for column in columns if pd.isna(row[column])]
        status_name = str(status["status"])
        if status_name in {"failed", "not_computed_sample_failed"}:
            failed_groups.append(group)
            if not str(status["error"]) or set(null_columns) != set(columns):
                raise ValueError(f"{spec.query_id} failed group {group} must record an error and null all columns")
        elif status_name == "ok":
            if str(status["error"]) or set(null_columns) != set(recorded_nonfinite):
                raise ValueError(f"{spec.query_id} group {group} silently replaced or omitted a non-finite value")
        else:
            raise ValueError(f"{spec.query_id} group {group} has an unknown status")

    expected_status = "failed" if failed_groups else "ok"
    if str(row["feature_status"]) != expected_status or bool(failures) != bool(failed_groups):
        raise ValueError(f"{spec.query_id} aggregate feature failure status is inconsistent")
    observed_count = sum(not pd.isna(row[column]) for column in spec.feature_columns)
    if int(row["feature_count"]) != observed_count:
        raise ValueError(f"{spec.query_id} feature_count does not match finite output columns")


def _check_action_losses(action_losses: pd.DataFrame) -> None:
    required = {
        "sample_design_id",
        "FE_query",
        "FE_total",
        "dimension",
        "performance_value_mode",
        "performance_loss_mode",
        "benchmark_reference_value",
        "p_skip",
        "p_skip_raw",
        "loss_skip",
        "action_loss",
        "action_loss_raw",
        "p_query_raw",
        "loss_query",
        "p_query",
        "transition_mode",
        "runtime_no_query_handoff",
        "runtime_no_query_optimization",
        "runtime_handoff",
        "runtime_action_optimization",
        "action_loss_protocol",
    }
    missing = required.difference(action_losses.columns)
    if missing:
        raise ValueError(f"action-loss data are missing query-budget columns: {sorted(missing)}")
    for design_id, frame in action_losses.groupby("sample_design_id", sort=True):
        design = get_sample_design_spec(str(design_id))
        expected = frame["dimension"].astype(int) * design.sample_size_per_dimension
        if not np.array_equal(frame["FE_query"].astype(int).to_numpy(), expected.to_numpy()):
            raise ValueError(f"{design_id} action losses use an inconsistent query FE budget")
    runtime_columns = (
        "runtime_no_query_handoff",
        "runtime_no_query_optimization",
        "runtime_handoff",
        "runtime_action_optimization",
    )
    runtimes = action_losses[list(runtime_columns)].to_numpy(dtype=float)
    if not np.isfinite(runtimes).all() or (runtimes < 0.0).any():
        raise ValueError("action-loss runtimes must be finite and non-negative")
    native = action_losses["transition_mode"].astype(str) == "native_optimizer_state"
    if not bool((action_losses.loc[native, "runtime_handoff"].astype(float) == 0.0).all()):
        raise ValueError("native action continuations must have zero handoff runtime")
    if set(action_losses["performance_value_mode"].astype(str)) != {"raw_objective"}:
        raise ValueError("action losses must store raw objective values explicitly")
    if set(action_losses["performance_loss_mode"].astype(str)) != {"known_optimum_gap"}:
        raise ValueError("action losses must use known-optimum-gap losses")
    reference = action_losses["benchmark_reference_value"].to_numpy(dtype=float)
    action_raw = action_losses["action_loss_raw"].to_numpy(dtype=float)
    action_gap = action_losses["action_loss"].to_numpy(dtype=float)
    skip_raw = action_losses["p_skip_raw"].to_numpy(dtype=float)
    skip_gap = action_losses["p_skip"].to_numpy(dtype=float)
    if not np.allclose(action_gap, np.maximum(action_raw - reference, 0.0), rtol=0.0, atol=1e-12):
        raise ValueError("action_loss is not the known-optimum gap of action_loss_raw")
    if not np.allclose(skip_gap, np.maximum(skip_raw - reference, 0.0), rtol=0.0, atol=1e-12):
        raise ValueError("p_skip is not the known-optimum gap of p_skip_raw")
    if not np.allclose(action_losses["p_query_raw"], action_raw, rtol=0.0, atol=1e-12):
        raise ValueError("p_query_raw must equal action_loss_raw")
    if not np.allclose(action_losses["loss_query"], action_gap, rtol=0.0, atol=1e-12):
        raise ValueError("loss_query must equal action_loss")
    if not np.allclose(action_losses["p_query"], action_gap, rtol=0.0, atol=1e-12):
        raise ValueError("p_query must equal action_loss")
    if not np.allclose(action_losses["loss_skip"], skip_gap, rtol=0.0, atol=1e-12):
        raise ValueError("loss_skip must equal p_skip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check frozen landscape-query protocols without pytest.")
    parser.add_argument("--samples", type=Path, action="append", default=None)
    parser.add_argument("--features", type=Path, action="append", default=None)
    parser.add_argument("--action-losses", type=Path, action="append", default=None)
    args = parser.parse_args()
    summary = check_landscape_query_consistency(
        sample_paths=args.samples or [],
        feature_paths=args.features or [],
        action_loss_paths=args.action_losses or [],
    )
    print(
        "validated frozen query specs; "
        f"samples={summary['sample_rows']}, features={summary['feature_rows']}, "
        f"action_losses={summary['action_loss_rows']}"
    )


if __name__ == "__main__":
    main()
