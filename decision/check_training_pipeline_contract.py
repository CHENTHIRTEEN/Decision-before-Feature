from __future__ import annotations

import argparse


REPLACEMENT = (
    "The active training contract cannot be checked from one materialized Utility "
    "table. Use the raw fold-specific inputs and measured selected-path timings with "
    "`decision-train-full`; the offline replay runner is still a formal-run blocker."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retired compatibility entry point: the single-table preprocessing check "
            "does not represent the active nested-learning contract."
        ),
        epilog=REPLACEMENT,
    )
    parser.parse_args()
    parser.error(REPLACEMENT)


if __name__ == "__main__":
    main()
