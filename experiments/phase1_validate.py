from __future__ import annotations

import argparse

from trajectory.validation import validate_trajectory_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase 1 trajectory output.")
    parser.add_argument("path")
    args = parser.parse_args()
    summary = validate_trajectory_file(args.path)
    print(f"validated {summary['rows']} rows across {summary['runs']} runs")


if __name__ == "__main__":
    main()

