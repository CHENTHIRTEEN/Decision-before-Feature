from __future__ import annotations

import argparse


REPLACEMENT = (
    "The formal Decision workflow no longer accepts one precomputed Utility table. "
    "Use `decision-train-full --replay-plan-only`, execute the emitted fold-specific "
    "selected complete paths with the pending offline replay runner, and then run "
    "`decision-train-full` with the measured complete-path timing inputs."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retired compatibility entry point: single-table Decision materialization "
            "is outside the active fold-specific nested protocol."
        ),
        epilog=REPLACEMENT,
    )
    parser.parse_args()
    parser.error(REPLACEMENT)


if __name__ == "__main__":
    main()
