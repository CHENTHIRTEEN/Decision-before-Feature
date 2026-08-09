from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.phase1_batch_common import load_config, make_shards
from utility_labels.fields import UTILITY_VALUE_COLUMNS
from utility_labels.generation import FE_ANALYSIS_RATIO, generate_utility_labels
from utility_labels.validation import validate_utility_label_file


TARGET_COLUMN = "u_ela_lamT_1"
EPS = 1e-12


def utility_label_shard_path(output_root: Path, split: str, family: str, dimension: int) -> Path:
    return output_root / split / family / f"dimension_{dimension}" / "utility_labels.parquet"


def generate_utility_label_shards(
    *,
    config_paths: list[Path],
    behavior_root: Path,
    ela_root: Path,
    selection_reference_path: Path,
    output_root: Path,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    overwrite: bool,
    workers: int,
) -> dict[str, int]:
    tasks = []
    skipped_existing = 0
    for config_path in config_paths:
        config = load_config(config_path)
        split = _split_name(config)
        for shard in make_shards(config, only_functions, only_dimensions):
            output_path = utility_label_shard_path(output_root, split, shard.family, shard.dimension)
            if output_path.exists() and not overwrite:
                print(f"skip existing utility label shard {output_path}")
                skipped_existing += 1
                continue
            tasks.append(
                {
                    "config_path": str(config_path),
                    "behavior_root": str(behavior_root),
                    "ela_root": str(ela_root),
                    "selection_reference_path": str(selection_reference_path),
                    "output_path": str(output_path),
                    "function": int(shard.function),
                    "dimension": int(shard.dimension),
                    "split": split,
                    "family": shard.family,
                }
            )

    written = 0
    rows = 0
    started = perf_counter()
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            summary = _generate_one_shard(task)
            written += 1
            rows += int(summary["rows"])
            print(_format_shard_summary(summary))
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            futures = {executor.submit(_generate_one_shard, task): task for task in tasks}
            for future in as_completed(futures):
                summary = future.result()
                written += 1
                rows += int(summary["rows"])
                print(_format_shard_summary(summary))
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()

    elapsed = perf_counter() - started
    print(
        "finished "
        f"{written} written shards, "
        f"{skipped_existing} existing shards skipped, "
        f"{rows} utility label rows, "
        f"{elapsed:.1f}s elapsed"
    )
    return {
        "written_shards": written,
        "skipped_existing_shards": skipped_existing,
        "rows": rows,
    }


def summarize_utility_label_shards(
    *,
    config_paths: list[Path],
    output_root: Path,
    selection_reference_path: Path,
    selector_proxy_path: Path | None,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    report_dir: Path,
) -> dict[str, int | str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = []
    label_frames = []
    expected_shards = 0
    existing_shards = 0
    validated_shards = 0
    expected_rows_total = 0
    observed_rows_total = 0

    for config_path in config_paths:
        config = load_config(config_path)
        split = _split_name(config)
        for shard in make_shards(config, only_functions, only_dimensions):
            expected_shards += 1
            output_path = utility_label_shard_path(output_root, split, shard.family, shard.dimension)
            expected_rows = _expected_eligible_rows(config, shard.output_path)
            expected_rows_total += expected_rows
            row = {
                "split": split,
                "family": shard.family,
                "dimension": int(shard.dimension),
                "path": str(output_path),
                "expected_rows": int(expected_rows),
                "exists": bool(output_path.exists()),
                "rows": 0,
                "coverage_rate": 0.0,
                "validation_status": "missing",
                "validation_error": "",
            }
            if output_path.exists():
                existing_shards += 1
                try:
                    validation = validate_utility_label_file(output_path)
                    frame = pq.read_table(output_path).to_pandas()
                    frame["label_source"] = np.where(
                        frame["selected_algorithm"].astype(str) == frame["default_algorithm"].astype(str),
                        "same_algorithm",
                        "changed_algorithm",
                    )
                    label_frames.append(frame)
                    row["rows"] = int(validation["rows"])
                    row["coverage_rate"] = float(row["rows"] / expected_rows) if expected_rows else 0.0
                    row["validation_status"] = "ok"
                    validated_shards += 1
                    observed_rows_total += int(validation["rows"])
                except Exception as exc:
                    row["validation_status"] = "failed"
                    row["validation_error"] = f"{type(exc).__name__}: {exc}"
            shard_rows.append(row)

    shard_summary = pd.DataFrame(shard_rows)
    labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    label_summary = _label_summary(labels)
    selector_quality = _selector_quality(labels, selection_reference_path)
    proxy_consistency = _proxy_observed_consistency(labels, selector_proxy_path)

    _write_frame(shard_summary, report_dir / "utility_label_shard_coverage.parquet")
    _write_frame(label_summary, report_dir / "utility_label_source_summary.parquet")
    _write_frame(selector_quality, report_dir / "observed_selector_quality_summary.parquet")
    _write_frame(proxy_consistency, report_dir / "proxy_observed_consistency_summary.parquet")
    shard_summary.to_csv(report_dir / "utility_label_shard_coverage.csv", index=False)
    label_summary.to_csv(report_dir / "utility_label_source_summary.csv", index=False)
    selector_quality.to_csv(report_dir / "observed_selector_quality_summary.csv", index=False)
    proxy_consistency.to_csv(report_dir / "proxy_observed_consistency_summary.csv", index=False)

    report_path = report_dir / "utility_label_data_quality_report.md"
    report_path.write_text(
        _markdown_report(
            shard_summary=shard_summary,
            label_summary=label_summary,
            selector_quality=selector_quality,
            proxy_consistency=proxy_consistency,
            expected_shards=expected_shards,
            existing_shards=existing_shards,
            validated_shards=validated_shards,
            expected_rows_total=expected_rows_total,
            observed_rows_total=observed_rows_total,
        ),
        encoding="utf-8",
    )
    print(f"wrote utility label data quality report to {report_path}")
    return {
        "expected_shards": expected_shards,
        "existing_shards": existing_shards,
        "validated_shards": validated_shards,
        "expected_rows": expected_rows_total,
        "observed_rows": observed_rows_total,
        "report": str(report_path),
    }


def _generate_one_shard(task: dict[str, Any]) -> dict[str, int | str | float]:
    started = perf_counter()
    summary = generate_utility_labels(
        config_path=Path(task["config_path"]),
        behavior_root=Path(task["behavior_root"]),
        ela_root=Path(task["ela_root"]),
        selection_reference_path=Path(task["selection_reference_path"]),
        output_path=Path(task["output_path"]),
        only_functions=[int(task["function"])],
        only_dimensions=[int(task["dimension"])],
        max_labels=None,
    )
    validation = validate_utility_label_file(task["output_path"])
    return {
        "split": str(task["split"]),
        "family": str(task["family"]),
        "dimension": int(task["dimension"]),
        "rows": int(validation["rows"]),
        "output": str(summary["output"]),
        "elapsed_seconds": perf_counter() - started,
    }


def _format_shard_summary(summary: dict[str, int | str | float]) -> str:
    return (
        f"wrote {summary['rows']} utility label rows to {summary['output']} "
        f"({summary['split']} {summary['family']} d{summary['dimension']}, "
        f"{float(summary['elapsed_seconds']):.1f}s)"
    )


def _expected_eligible_rows(config: dict, trajectory_path: Path) -> int:
    table = pq.read_table(trajectory_path, columns=["dimension", "FE", "FE_ratio"])
    rows = table.to_pylist()
    count = 0
    for row in rows:
        dimension = int(row["dimension"])
        fe_total = _fe_total(config, dimension)
        fe_prefix = int(row["FE"])
        fe_analysis = int(FE_ANALYSIS_RATIO * fe_total)
        ratio = float(row["FE_ratio"])
        if ratio >= 0.12 and ratio < 1.0 and fe_prefix + fe_analysis < fe_total:
            count += 1
    return count


def _label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "split",
                "label_source",
                "rows",
                "u_gt_zero_rows",
                "u_gt_zero_rate",
                "positive_utility_sum",
                "utility_sum",
                "mean_u_ela_lamT_1",
                "mean_performance_gain_norm",
            ]
        )
    frames = []
    layers = {
        "overall": ["split"],
        "label_source": ["split", "label_source"],
        "family_label_source": ["split", "family", "label_source"],
        "dimension_label_source": ["split", "dimension", "label_source"],
        "fe_ratio_label_source": ["split", "FE_ratio", "label_source"],
        "family_dimension_fe_ratio_label_source": ["split", "family", "dimension", "FE_ratio", "label_source"],
    }
    for layer, columns in layers.items():
        rows = []
        for values, frame in labels.groupby(columns, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            row = {"layer": layer, **dict(zip(columns, values, strict=False))}
            utility = frame[TARGET_COLUMN].astype(float)
            positive = utility > 0.0
            row.update(
                {
                    "rows": int(len(frame)),
                    "u_gt_zero_rows": int(positive.sum()),
                    "u_gt_zero_rate": float(positive.mean()) if len(frame) else 0.0,
                    "positive_utility_sum": float(utility[positive].sum()),
                    "utility_sum": float(utility.sum()),
                    "mean_u_ela_lamT_1": float(utility.mean()),
                    "mean_performance_gain_norm": float(frame["performance_gain_norm"].mean()),
                }
            )
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _selector_quality(labels: pd.DataFrame, selection_reference_path: Path) -> pd.DataFrame:
    columns = [
        "layer",
        "split",
        "label_source",
        "rows",
        "selection_reference_join_rate",
        "selected_matches_vbs_rate",
        "sbs_matches_vbs_rate",
        "mean_observed_u_ela_lamT_1",
        "mean_observed_gain_norm",
    ]
    if labels.empty or not selection_reference_path.exists():
        return pd.DataFrame(columns=columns)
    reference = pq.read_table(selection_reference_path).to_pandas()
    reference = reference[
        [
            "split",
            "problem_id",
            "dimension",
            "remaining_budget_ratio",
            "vbs_algorithm",
            "sbs_algorithm",
            "selector_status",
        ]
    ].copy()
    labels = labels.copy()
    labels["remaining_budget_ratio"] = (labels["FE_ela_optimization"] / labels["FE_total"]).round(6)
    reference["remaining_budget_ratio"] = reference["remaining_budget_ratio"].round(6)
    joined = labels.merge(
        reference,
        on=["split", "problem_id", "dimension", "remaining_budget_ratio"],
        how="left",
    )
    joined["selection_reference_joined"] = joined["vbs_algorithm"].notna()
    joined["selected_matches_vbs"] = joined["selected_algorithm"].astype(str) == joined["vbs_algorithm"].fillna("").astype(str)
    joined["sbs_matches_vbs"] = joined["default_algorithm"].astype(str) == joined["vbs_algorithm"].fillna("").astype(str)

    frames = []
    for layer, group_columns in {
        "overall": ["split"],
        "label_source": ["split", "label_source"],
        "family_label_source": ["split", "family", "label_source"],
        "fe_ratio_label_source": ["split", "FE_ratio", "label_source"],
    }.items():
        rows = []
        for values, frame in joined.groupby(group_columns, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            row = {"layer": layer, **dict(zip(group_columns, values, strict=False))}
            row.update(
                {
                    "rows": int(len(frame)),
                    "selection_reference_join_rate": float(frame["selection_reference_joined"].mean()),
                    "selected_matches_vbs_rate": float(frame["selected_matches_vbs"].mean()),
                    "sbs_matches_vbs_rate": float(frame["sbs_matches_vbs"].mean()),
                    "mean_observed_u_ela_lamT_1": float(frame[TARGET_COLUMN].mean()),
                    "mean_observed_gain_norm": float(frame["performance_gain_norm"].mean()),
                }
            )
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _proxy_observed_consistency(labels: pd.DataFrame, selector_proxy_path: Path | None) -> pd.DataFrame:
    columns = [
        "layer",
        "split",
        "label_source",
        "rows",
        "proxy_join_rate",
        "observed_proxy_gain_corr",
        "same_gain_sign_rate",
        "observed_gain_positive_rate",
        "proxy_gain_positive_rate",
    ]
    if labels.empty or selector_proxy_path is None or not selector_proxy_path.exists():
        return pd.DataFrame(columns=columns)
    proxy = pq.read_table(selector_proxy_path).to_pandas()
    required = {
        "split",
        "problem_id",
        "dimension",
        "remaining_budget_ratio",
        "current_proxy_gain_vs_sbs_norm",
    }
    if not required.issubset(proxy.columns):
        return pd.DataFrame(columns=columns)
    labels = labels.copy()
    labels["remaining_budget_ratio"] = (labels["FE_ela_optimization"] / labels["FE_total"]).round(6)
    proxy = proxy[list(required)].copy()
    proxy["remaining_budget_ratio"] = proxy["remaining_budget_ratio"].round(6)
    joined = labels.merge(proxy, on=["split", "problem_id", "dimension", "remaining_budget_ratio"], how="left")
    joined["proxy_joined"] = joined["current_proxy_gain_vs_sbs_norm"].notna()

    frames = []
    for layer, group_columns in {
        "overall": ["split"],
        "label_source": ["split", "label_source"],
        "family_label_source": ["split", "family", "label_source"],
        "fe_ratio_label_source": ["split", "FE_ratio", "label_source"],
    }.items():
        rows = []
        for values, frame in joined.groupby(group_columns, dropna=False, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            row = {"layer": layer, **dict(zip(group_columns, values, strict=False))}
            available = frame[frame["proxy_joined"]]
            observed = available["performance_gain_norm"].astype(float)
            proxy_gain = available["current_proxy_gain_vs_sbs_norm"].astype(float)
            if len(available) > 1 and float(observed.std()) > EPS and float(proxy_gain.std()) > EPS:
                corr = float(observed.corr(proxy_gain))
            else:
                corr = np.nan
            same_sign = np.sign(observed.to_numpy()) == np.sign(proxy_gain.to_numpy())
            row.update(
                {
                    "rows": int(len(frame)),
                    "proxy_join_rate": float(frame["proxy_joined"].mean()) if len(frame) else 0.0,
                    "observed_proxy_gain_corr": corr,
                    "same_gain_sign_rate": float(np.mean(same_sign)) if len(available) else np.nan,
                    "observed_gain_positive_rate": float((observed > 0.0).mean()) if len(available) else np.nan,
                    "proxy_gain_positive_rate": float((proxy_gain > 0.0).mean()) if len(available) else np.nan,
                }
            )
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _markdown_report(
    *,
    shard_summary: pd.DataFrame,
    label_summary: pd.DataFrame,
    selector_quality: pd.DataFrame,
    proxy_consistency: pd.DataFrame,
    expected_shards: int,
    existing_shards: int,
    validated_shards: int,
    expected_rows_total: int,
    observed_rows_total: int,
) -> str:
    lines = [
        "# phase1 refined sampling utility label data quality",
        "",
        "本报告只汇总 observed utility label shards；bucket proxy 只用于一致性诊断，不替代 observed label。",
        "",
        f"- expected shards: {expected_shards}",
        f"- existing shards: {existing_shards}",
        f"- validated shards: {validated_shards}",
        f"- expected rows: {expected_rows_total}",
        f"- observed rows: {observed_rows_total}",
        f"- row coverage: {(observed_rows_total / expected_rows_total if expected_rows_total else 0.0):.6f}",
        "",
        "## Shard Coverage By Split",
    ]
    if not shard_summary.empty:
        for split, frame in shard_summary.groupby("split", sort=True):
            lines.append(
                f"- {split}: shards={int(frame['exists'].sum())}/{len(frame)}, "
                f"rows={int(frame['rows'].sum())}/{int(frame['expected_rows'].sum())}, "
                f"coverage={(frame['rows'].sum() / frame['expected_rows'].sum() if frame['expected_rows'].sum() else 0.0):.6f}"
            )
    lines.append("")
    lines.append("## Label Source Summary")
    label_overall = label_summary[label_summary["layer"] == "label_source"] if not label_summary.empty else pd.DataFrame()
    if label_overall.empty:
        lines.append("- no observed utility label rows available yet")
    else:
        for _, row in label_overall.sort_values(["split", "label_source"]).iterrows():
            lines.append(
                f"- {row['split']} {row['label_source']}: rows={int(row['rows'])}, "
                f"U_ELA>0 rows={int(row['u_gt_zero_rows'])}, "
                f"rate={row['u_gt_zero_rate']:.6f}, "
                f"positive utility sum={row['positive_utility_sum']:.6g}"
            )
    lines.append("")
    lines.append("## Selector Quality")
    selector_overall = selector_quality[selector_quality["layer"] == "overall"] if not selector_quality.empty else pd.DataFrame()
    if selector_overall.empty:
        lines.append("- no observed selector quality rows available yet")
    else:
        for _, row in selector_overall.sort_values("split").iterrows():
            lines.append(
                f"- {row['split']}: join={row['selection_reference_join_rate']:.6f}, "
                f"selected=VBS={row['selected_matches_vbs_rate']:.6f}, "
                f"SBS=VBS={row['sbs_matches_vbs_rate']:.6f}, "
                f"mean observed U={row['mean_observed_u_ela_lamT_1']:.6g}"
            )
    lines.append("")
    lines.append("## Proxy/Observed Consistency")
    proxy_overall = proxy_consistency[proxy_consistency["layer"] == "overall"] if not proxy_consistency.empty else pd.DataFrame()
    if proxy_overall.empty:
        lines.append("- proxy/observed consistency is unavailable until at least one observed label shard exists and proxy rows are present")
    else:
        for _, row in proxy_overall.sort_values("split").iterrows():
            lines.append(
                f"- {row['split']}: proxy join={row['proxy_join_rate']:.6f}, "
                f"gain corr={row['observed_proxy_gain_corr']:.6g}, "
                f"same sign={row['same_gain_sign_rate']:.6f}"
            )
    lines.append("")
    lines.append("## Output Tables")
    lines.extend(
        [
            "- `utility_label_shard_coverage.parquet`",
            "- `utility_label_source_summary.parquet`",
            "- `observed_selector_quality_summary.parquet`",
            "- `proxy_observed_consistency_summary.parquet`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def _split_name(config: dict) -> str:
    if "split" in config:
        return str(config["split"])
    return Path(config["output"]).stem.removesuffix("_trajectories")


def _fe_total(config: dict, dimension: int) -> int:
    budgets = config.get("FE_total_by_dimension", {})
    if dimension in budgets:
        return int(budgets[dimension])
    return int(budgets[str(dimension)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 1 utility labels by BBOB split/family/dimension shard.")
    parser.add_argument("--config", type=Path, action="append", default=None)
    parser.add_argument("--behavior-root", type=Path, default=Path("results/phase1_refined_sampling"))
    parser.add_argument("--ela-root", type=Path, default=Path("results/ela"))
    parser.add_argument(
        "--selection-reference",
        type=Path,
        default=Path("results/selection_reference/bbob_train/selection_reference.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/utility_labels/phase1_refined_sampling"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("results/utility_labels/phase1_refined_sampling_quality"),
    )
    parser.add_argument(
        "--selector-proxy",
        type=Path,
        default=Path("results/selection_reference/bbob_train/quality/selection_reference_proxy_rows.parquet"),
    )
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
    if not args.summarize_only:
        generate_utility_label_shards(
            config_paths=config_paths,
            behavior_root=args.behavior_root,
            ela_root=args.ela_root,
            selection_reference_path=args.selection_reference,
            output_root=args.output_root,
            only_functions=args.only_function,
            only_dimensions=args.only_dimension,
            overwrite=args.overwrite,
            workers=args.workers,
        )
    summarize_utility_label_shards(
        config_paths=config_paths,
        output_root=args.output_root,
        selection_reference_path=args.selection_reference,
        selector_proxy_path=args.selector_proxy,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
