from __future__ import annotations

import argparse
from pathlib import Path

from behavior.features import extract_behavior_rows, read_trajectory_rows, write_behavior_rows
from behavior.validation import validate_behavior_rows


def extract_behavior_file(input_path: str | Path, output_path: str | Path) -> dict[str, int | str]:
    trajectory_rows = read_trajectory_rows(input_path)
    behavior_rows = extract_behavior_rows(trajectory_rows)
    validate_behavior_rows(trajectory_rows, behavior_rows)
    written = write_behavior_rows(behavior_rows, output_path)
    return {"rows": len(behavior_rows), "output": str(written)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract algorithm-agnostic behavior states from Phase 1 trajectories.")
    parser.add_argument("--input", required=True, help="Phase 1 trajectory Parquet file.")
    parser.add_argument("--output", required=True, help="Behavior state Parquet output file.")
    args = parser.parse_args()
    summary = extract_behavior_file(args.input, args.output)
    print(f"wrote {summary['rows']} behavior rows to {summary['output']}")


if __name__ == "__main__":
    main()
