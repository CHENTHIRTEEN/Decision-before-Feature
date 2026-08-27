from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks import make_problem
from experiments.phase1_batch_common import (
    as_int_list,
    cv_group_id_name,
    function_id_name,
    landscape_family_name,
    fe_total_for_dimension,
    load_suite_configs,
    runtime_problem_config,
    selected_dimensions,
    selected_functions,
    split_name,
    validate_dynamic_collection_config,
)
from landscape_queries.sampling import make_query_sample_seed, sample_problem
from landscape_queries.specs import SAMPLE_DESIGN_SPECS, get_sample_design_spec


SAMPLE_KEY_COLUMNS = (
    "split",
    "problem_id",
    "function_id",
    "family",
    "cv_group_id",
    "function",
    "instance",
    "dimension",
    "sample_design_id",
)


def default_sample_path(sample_design_id: str, split: str) -> Path:
    return Path("results/landscape_queries/samples") / sample_design_id / split / "samples.parquet"


def generate_query_samples(
    *,
    config: dict,
    sample_design_id: str,
    output_path: Path | None,
    base_seed: int,
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    overwrite: bool,
) -> dict[str, int | str]:
    validate_dynamic_collection_config(config)
    suite = str(config["suite"]).lower()
    if suite not in {"bbob", "cec2017", "cec2022", "mabbob"}:
        raise ValueError("query-sample-batch supports bbob, cec2017, cec2022, and mabbob")
    split = split_name(config)
    design = get_sample_design_spec(sample_design_id)
    output = output_path or default_sample_path(sample_design_id, split)
    if output.exists() and not overwrite:
        raise FileExistsError(f"query sample output already exists; pass --overwrite: {output}")

    rows = []
    for function in selected_functions(config, only_functions):
        for dimension in selected_dimensions(config, only_dimensions):
            fe_total = fe_total_for_dimension(config, dimension)
            expected_fe = design.sample_size(dimension)
            if expected_fe <= 0:
                raise ValueError(f"{sample_design_id} must define a positive sample size per dimension")
            if fe_total < expected_fe:
                raise ValueError(
                    f"{sample_design_id} requires FE_total={fe_total} to be at least FE_query={expected_fe}"
                )
            for instance in as_int_list(config, "instances"):
                problem = None
                try:
                    problem = make_problem(
                        runtime_problem_config(
                            config, function=function, instance=instance, dimension=dimension
                        )
                    )
                    sample = sample_problem(
                        problem=problem,
                        sample_design=design,
                        base_seed=base_seed,
                        function=function,
                        instance=instance,
                        success_gap_target=float(config["success_gap_target"]),
                        failure_loss_cap=float(config["failure_loss_cap"]),
                    )
                    rows.append(
                        {
                            "split": split,
                            "problem_id": problem.problem_id,
                            "function_id": problem.function_id,
                            "family": problem.family,
                            "cv_group_id": problem.cv_group_id,
                            "function": int(function),
                            "instance": int(instance),
                            "dimension": int(dimension),
                            "FE_total": int(fe_total),
                            "sample_design_id": design.sample_design_id,
                            "sampling_protocol": design.protocol,
                            **sample,
                        }
                    )
                except Exception as exc:
                    if suite == "bbob":
                        problem_id = f"bbob_f{int(function):03d}_i{int(instance):02d}_d{int(dimension)}"
                    elif suite == "mabbob":
                        problem_id = f"mabbob_c{int(function):03d}_i{int(instance):02d}_d{int(dimension)}"
                    else:
                        problem_id = f"{suite}_f{int(function):02d}_d{int(dimension)}"
                    failure_type = type(exc).__name__
                    failure_message = str(exc)[:500]
                    rows.append(
                        {
                            "split": split,
                            "problem_id": problem_id,
                            "function_id": function_id_name(suite, function),
                            "family": landscape_family_name(suite, function),
                            "cv_group_id": cv_group_id_name(suite, function),
                            "function": int(function),
                            "instance": int(instance),
                            "dimension": int(dimension),
                            "FE_total": int(fe_total),
                            "sample_design_id": design.sample_design_id,
                            "sampling_protocol": design.protocol,
                            "sample_seed": make_query_sample_seed(
                                base_seed=base_seed,
                                function=function,
                                instance=instance,
                                dimension=dimension,
                                sample_design=design,
                            ),
                            "sample_size": int(expected_fe),
                            "FE_query": 0,
                            "FE_query_planned": int(expected_fe),
                            "runtime_query_sampling": 0.0,
                            "runtime_query_evaluation": 0.0,
                            "runtime_sampling_evaluation": 0.0,
                            "benchmark_reference_value": None,
                            "success_gap_target": float(config["success_gap_target"]),
                            "query_success": False,
                            "query_first_hit_offset": None,
                            "query_best_gap": float(config["failure_loss_cap"]),
                            "lower_bounds": [],
                            "upper_bounds": [],
                            "X": [],
                            "y": [],
                            "sample_status": "failed",
                            "sample_path_completed": False,
                            "sample_planned_FE": int(expected_fe),
                            "sample_effective_FE": 0,
                            "sample_observed_first_hit_FE": None,
                            "sample_target_hit_observed": False,
                            "sample_target_hit_before_failure": False,
                            "sample_endpoint_success": False,
                            "sample_timed_out": False,
                            "sample_failure_type": failure_type,
                            "sample_failure_message": failure_message,
                            "sample_failure": f"{failure_type}: {failure_message}"[:500],
                        }
                    )
                finally:
                    if problem is not None:
                        problem.close()

    if not rows:
        raise ValueError("query sampling produced no rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=sample_schema()), output)
    print(f"wrote {len(rows)} {sample_design_id} problem samples to {output}")
    return {"rows": len(rows), "sample_design_id": sample_design_id, "output": str(output)}


def sample_schema() -> pa.Schema:
    return pa.schema(
        [
            ("split", pa.string()),
            ("problem_id", pa.string()),
            ("function_id", pa.string()),
            ("family", pa.string()),
            ("cv_group_id", pa.string()),
            ("function", pa.int32()),
            ("instance", pa.int32()),
            ("dimension", pa.int32()),
            ("FE_total", pa.int64()),
            ("sample_design_id", pa.string()),
            ("sampling_protocol", pa.string()),
            ("sample_seed", pa.int64()),
            ("sample_size", pa.int64()),
            ("FE_query", pa.int64()),
            ("FE_query_planned", pa.int64()),
            ("runtime_query_sampling", pa.float64()),
            ("runtime_query_evaluation", pa.float64()),
            ("runtime_sampling_evaluation", pa.float64()),
            ("benchmark_reference_value", pa.float64()),
            ("success_gap_target", pa.float64()),
            ("query_success", pa.bool_()),
            ("query_first_hit_offset", pa.int64()),
            ("query_best_gap", pa.float64()),
            ("lower_bounds", pa.list_(pa.float64())),
            ("upper_bounds", pa.list_(pa.float64())),
            ("X", pa.list_(pa.list_(pa.float64()))),
            ("y", pa.list_(pa.float64())),
            ("sample_status", pa.string()),
            ("sample_path_completed", pa.bool_()),
            ("sample_planned_FE", pa.int64()),
            ("sample_effective_FE", pa.int64()),
            ("sample_observed_first_hit_FE", pa.int64()),
            ("sample_target_hit_observed", pa.bool_()),
            ("sample_target_hit_before_failure", pa.bool_()),
            ("sample_endpoint_success", pa.bool_()),
            ("sample_timed_out", pa.bool_()),
            ("sample_failure_type", pa.string()),
            ("sample_failure_message", pa.string()),
            ("sample_failure", pa.string()),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fixed LHS sample boundary for a landscape query.")
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--sample-design-id", choices=sorted(SAMPLE_DESIGN_SPECS), required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output is not None and len(args.config) != 1:
        raise ValueError("--output can be used only with one --config")
    for config_path in args.config:
        for config in load_suite_configs(config_path):
            generate_query_samples(
                config=config,
                sample_design_id=args.sample_design_id,
                output_path=args.output,
                base_seed=args.base_seed,
                only_functions=args.only_function,
                only_dimensions=args.only_dimension,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
