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
        "runtime_query_sampling",
        "runtime_query_evaluation",
        "runtime_sampling_evaluation",
        "X",
        "y",
        "sample_status",
    }
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"sample data are missing columns: {sorted(missing)}")
    if samples.duplicated(list(SAMPLE_KEY_COLUMNS)).any():
        raise ValueError("sample keys must be unique")
    if set(samples["sample_status"].astype(str)) != {"ok"}:
        raise ValueError("consistency inputs contain failed sample rows")
    for row in samples.to_dict(orient="records"):
        design = get_sample_design_spec(str(row["sample_design_id"]))
        if str(row["sampling_protocol"]) != design.protocol:
            raise ValueError("sample row uses an unsupported sampling protocol")
        expected_size = design.sample_size(int(row["dimension"]))
        if int(row["sample_size"]) != expected_size or int(row["FE_query"]) != expected_size:
            raise ValueError("sample_size and FE_query must match the frozen dimension multiplier")
        if int(row["FE_query"]) != int(round(design.fe_ratio * int(row["FE_total"]))):
            raise ValueError("sample FE does not match the frozen percentage of FE_total")
        x = np.stack([np.asarray(point, dtype=float) for point in row["X"]], axis=0)
        y = np.asarray(row["y"], dtype=float)
        if x.shape != (expected_size, int(row["dimension"])) or y.shape != (expected_size,):
            raise ValueError("saved X or y shape is inconsistent with its sample design")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("saved X and y must be finite")
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
    key = ["split", "problem_id", "family", "function", "instance", "dimension", "sample_design_id"]
    for query_id, frame in features.groupby("query_id", sort=True):
        spec = get_query_spec(str(query_id))
        missing_features = set(spec.feature_columns).difference(frame.columns)
        if missing_features:
            raise ValueError(f"{query_id} feature whitelist is missing columns: {sorted(missing_features)}")
        if set(frame["query_protocol"].astype(str)) != {spec.protocol}:
            raise ValueError(f"{query_id} query_protocol does not match the frozen spec")
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
        bbob = frame[frame["split"].astype(str).isin({"bbob_train", "bbob_validation"})]
        if not bbob.empty and (bbob["feature_status"].astype(str) != "ok").any():
            raise ValueError(f"{query_id} has group-level extraction failure on BBOB train/validation")
        train = frame[frame["split"].astype(str) == "bbob_train"]
        if not train.empty:
            all_missing = [column for column in spec.feature_columns if train[column].isna().all()]
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

    cheap = features[features["query_id"].astype(str) == "descriptor_cheap"]
    standard = features[features["query_id"].astype(str) == "pflacco_standard"]
    if not cheap.empty and not standard.empty:
        shared_key = ["split", "problem_id", "family", "function", "instance", "dimension", "sample_design_id"]
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
        if status_name == "failed":
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
        ratios = frame["FE_query"].astype(float) / frame["FE_total"].astype(float)
        if not np.allclose(ratios, design.fe_ratio, rtol=0.0, atol=1e-12):
            raise ValueError(f"{design_id} action losses use an inconsistent FE ratio")
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
