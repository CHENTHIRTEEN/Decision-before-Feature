from __future__ import annotations

import argparse
from pathlib import Path

from behavior.extraction import extract_behavior_file
from behavior.validation import validate_behavior_file
from experiments.phase1_batch_common import load_config, make_shards, require_complete_shard_outputs


def behavior_output_path(trajectory_path: Path) -> Path:
    return trajectory_path.with_name("behavior.parquet")


def extract_behavior_shards(
    *,
    config_paths: list[Path],
    only_functions: list[int] | None,
    only_dimensions: list[int] | None,
    overwrite: bool,
) -> dict[str, int]:
    written_count = 0
    skipped_existing_count = 0
    row_count = 0

    for config_path in config_paths:
        config = load_config(config_path)
        if str(config["suite"]).lower() not in {"bbob", "cec2017", "cec2022", "mabbob"}:
            raise ValueError("behavior-extract-batch supports suites: bbob, cec2017, cec2022, mabbob")

        for shard in make_shards(config, only_functions, only_dimensions):
            require_complete_shard_outputs(shard)
            trajectory_path = shard.output_path
            output_path = behavior_output_path(trajectory_path)

            if output_path.exists() and not overwrite:
                validate_behavior_file(trajectory_path, output_path)
                print(f"skip existing behavior shard {output_path}")
                skipped_existing_count += 1
                continue

            summary = extract_behavior_file(trajectory_path, output_path)
            print(f"wrote {summary['rows']} behavior rows to {summary['output']}")
            written_count += 1
            row_count += int(summary["rows"])

    print(
        "finished "
        f"{written_count} written shards, "
        f"{skipped_existing_count} existing shards skipped"
    )
    return {
        "written_shards": written_count,
        "skipped_existing_shards": skipped_existing_count,
        "rows": row_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract behavior states for Phase 1 BBOB trajectory shards.")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=None,
        help="Phase 1 shard config. Repeat to process train and validation in one run.",
    )
    parser.add_argument("--only-function", type=int, action="append", default=None)
    parser.add_argument("--only-dimension", type=int, action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_paths = args.config or [
        Path("configs/phase1_bbob_train.yaml"),
        Path("configs/phase1_bbob_validation.yaml"),
    ]
    extract_behavior_shards(
        config_paths=config_paths,
        only_functions=args.only_function,
        only_dimensions=args.only_dimension,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
