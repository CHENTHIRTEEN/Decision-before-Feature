from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.phase1_batch_common import load_config, make_shards, split_name
from landscape_queries.specs import LANDSCAPE_QUERY_SPECS
from utility_labels.fields import UTILITY_VALUE_COLUMNS
from utility_labels.generation import generate_utility_labels
from utility_labels.validation import validate_utility_label_file


TARGET_COLUMN = "u_query_joint_lamT_1"


def utility_label_shard_path(
    output_root: Path,
    split: str,
    function_id: str,
    dimension: int,
) -> Path:
    return output_root / split / function_id / f"dimension_{dimension}" / "utility_labels.parquet"


def generate_utility_label_shards(
    *,
    query_id: str,
    config_paths: list[Path],
    query_selection_reference_path: Path,
    behavior_selection_reference_path: Path,
    query_adjusted_behavior_reference_path: Path,
    sampling_only_reference_path: Path,
    complete_path_timings_path: Path,
    output_root: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    overwrite: bool,
    workers: int,
) -> dict[str, int]:
    tasks = []
    skipped = 0
    for config_path in config_paths:
        config = load_config(config_path)
        split = split_name(config)
        for shard in make_shards(config, only_functions, only_dimensions):
            output = utility_label_shard_path(
                output_root, split, shard.function_id, shard.dimension
            )
            if output.exists() and not overwrite:
                print(f"skip existing query utility shard {output}")
                skipped += 1
                continue
            tasks.append(
                {
                    "query_id": query_id,
                    "config_path": str(config_path),
                    "query_selection_reference_path": str(query_selection_reference_path),
                    "behavior_selection_reference_path": str(behavior_selection_reference_path),
                    "query_adjusted_behavior_reference_path": str(query_adjusted_behavior_reference_path),
                    "sampling_only_reference_path": str(sampling_only_reference_path),
                    "complete_path_timings_path": str(complete_path_timings_path),
                    "output_path": str(output),
                    "function": shard.function,
                    "dimension": shard.dimension,
                }
            )
    written = 0
    rows = 0
    if workers <= 1:
        summaries = [_generate_one_shard(task) for task in tasks]
    else:
        summaries = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_one_shard, task) for task in tasks]
            for future in as_completed(futures):
                summaries.append(future.result())
    for summary in summaries:
        written += 1
        rows += int(summary["rows"])
        print(
            f"wrote {summary['rows']} rows to {summary['output']} "
            f"({float(summary['elapsed_seconds']):.1f}s)"
        )
    return {"written_shards": written, "skipped_existing_shards": skipped, "rows": rows}


def summarize_utility_label_shards(
    *,
    query_id: str,
    config_paths: list[Path],
    output_root: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    report_dir: Path,
) -> dict[str, int | str]:
    shard_rows = []
    frames = []
    for config_path in config_paths:
        config = load_config(config_path)
        split = split_name(config)
        for shard in make_shards(config, only_functions, only_dimensions):
            path = utility_label_shard_path(
                output_root, split, shard.function_id, shard.dimension
            )
            row = {
                "query_id": query_id,
                "split": split,
                "function_id": shard.function_id,
                "family": shard.family,
                "dimension": shard.dimension,
                "path": str(path),
                "exists": path.exists(),
                "rows": 0,
                "validation_status": "missing",
                "validation_error": "",
            }
            if path.exists():
                try:
                    validated = validate_utility_label_file(path)
                    frame = pq.read_table(path).to_pandas()
                    frames.append(frame)
                    row["rows"] = int(validated["rows"])
                    row["validation_status"] = "ok"
                except Exception as exc:
                    row["validation_status"] = "failed"
                    row["validation_error"] = f"{type(exc).__name__}: {exc}"
            shard_rows.append(row)
    shards = pd.DataFrame(shard_rows)
    labels = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = _utility_summary(labels, query_id=query_id)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_frame(shards, report_dir / "utility_label_shard_summary")
    _write_frame(summary, report_dir / "utility_distribution_summary")
    report_path = report_dir / "utility_label_consistency_report.md"
    report_path.write_text(
        _markdown_report(query_id=query_id, shards=shards, summary=summary),
        encoding="utf-8",
    )
    return {
        "expected_shards": int(len(shards)),
        "validated_shards": int((shards["validation_status"] == "ok").sum()) if not shards.empty else 0,
        "rows": int(len(labels)),
        "report": str(report_path),
    }


def _generate_one_shard(task: dict[str, Any]) -> dict[str, Any]:
    started = perf_counter()
    summary = generate_utility_labels(
        query_id=str(task["query_id"]),
        config_path=Path(task["config_path"]),
        query_selection_reference_path=Path(task["query_selection_reference_path"]),
        behavior_selection_reference_path=Path(task["behavior_selection_reference_path"]),
        query_adjusted_behavior_reference_path=Path(
            task["query_adjusted_behavior_reference_path"]
        ),
        sampling_only_reference_path=Path(task["sampling_only_reference_path"]),
        complete_path_timings_path=Path(task["complete_path_timings_path"]),
        output_path=Path(task["output_path"]),
        only_functions=[int(task["function"])],
        only_dimensions=[int(task["dimension"])],
        max_labels=None,
        overwrite=True,
    )
    validate_utility_label_file(summary["output"])
    return {**summary, "elapsed_seconds": perf_counter() - started}


def _utility_summary(labels: pd.DataFrame, *, query_id: str) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(columns=["query_id", "split", "rows", "call_rate", "mean_u_query_joint_lamT_1"])
    rows = []
    for split, frame in labels.groupby("split", sort=True):
        target = frame[TARGET_COLUMN].astype(float)
        rows.append(
            {
                "query_id": query_id,
                "split": str(split),
                "rows": int(len(frame)),
                "call_rate": float((target > 0.0).mean()),
                "mean_u_query_joint_lamT_1": float(target.mean()),
                "mean_performance_gain_norm": float(frame["performance_gain_norm"].mean()),
                "mean_selector_regret_raw": float(frame["selector_regret_raw"].mean()),
                **{f"mean_{column}": float(frame[column].mean()) for column in UTILITY_VALUE_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), stem.with_suffix(".parquet"))


def _markdown_report(*, query_id: str, shards: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        f"# {query_id} utility label consistency",
        "",
        f"- expected shards: {len(shards)}",
        f"- validated shards: {int((shards['validation_status'] == 'ok').sum()) if not shards.empty else 0}",
        f"- rows: {int(shards['rows'].sum()) if not shards.empty else 0}",
        "",
        "Engineering consistency checks are not used as publication evidence.",
        "",
    ]
    if not summary.empty:
        lines.append(_markdown_table(summary))
        lines.append("")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    def format_value(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    columns = list(frame.columns)
    headers = [str(column) for column in columns]
    rows = [[format_value(row[column]) for column in columns] for _, row in frame.iterrows()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate query-specific utility labels by split/family/dimension.")
    parser.add_argument("--query-id", choices=sorted(LANDSCAPE_QUERY_SPECS), required=True)
    parser.add_argument("--config", type=Path, action="append", default=None)
    parser.add_argument("--query-selection-reference", type=Path, default=None)
    parser.add_argument("--behavior-selection-reference", type=Path, default=None)
    parser.add_argument("--query-adjusted-behavior-reference", type=Path, default=None)
    parser.add_argument("--sampling-only-reference", type=Path, default=None)
    parser.add_argument("--complete-path-timings", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    config_paths = args.config or [
        Path("configs/phase1_bbob_train.yaml"),
        Path("configs/phase1_bbob_validation.yaml"),
    ]
    query_selection = args.query_selection_reference or Path("results/selection_reference") / args.query_id / "selection_reference.parquet"
    behavior_selection = args.behavior_selection_reference or Path("results/selection_reference/behavior_only_full_budget/selection_reference.parquet")
    state_only_selection = args.query_adjusted_behavior_reference or Path("results/selection_reference") / args.query_id / "state_only_selection_reference.parquet"
    sampling_only_selection = args.sampling_only_reference or Path("results/selection_reference") / args.query_id / "sampling_only_continue_current.parquet"
    output_root = args.output_root or Path("results/utility_labels") / args.query_id
    report_dir = args.report_dir or Path("results/utility_labels") / args.query_id / "quality"
    if not args.summarize_only:
        generate_utility_label_shards(
            query_id=args.query_id,
            config_paths=config_paths,
            query_selection_reference_path=query_selection,
            behavior_selection_reference_path=behavior_selection,
            query_adjusted_behavior_reference_path=state_only_selection,
            sampling_only_reference_path=sampling_only_selection,
            complete_path_timings_path=args.complete_path_timings,
            output_root=output_root,
            only_functions=args.only_function,
            only_dimensions=args.only_dimension,
            overwrite=args.overwrite,
            workers=args.workers,
        )
    summarize_utility_label_shards(
        query_id=args.query_id,
        config_paths=config_paths,
        output_root=output_root,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        report_dir=report_dir,
    )


if __name__ == "__main__":
    main()
