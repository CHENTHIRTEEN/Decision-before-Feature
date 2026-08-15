"""Retired cost--performance summary entry point.

The previous implementation combined per-state utility labels with policy
predictions drawn from different decision-opportunity sets.  It therefore
could not represent the frozen trajectory-first-trigger estimand and must not
produce formal RQ3 results.  Formal policy comparisons are built from the
scientific endpoints emitted by ``decision.online_controller_evaluate``.
"""

from __future__ import annotations

import argparse
from typing import NoReturn


RETIRED_REASON = (
    "decision.controller_cost_performance_pareto is retired: its per-state "
    "utility join cannot represent the frozen trajectory-first-trigger policy "
    "estimand, matched-rate Random Analysis, FE=0 Traditional AAS, or the "
    "different B3/all_accepted and T0/milestone_only opportunity sets. Build "
    "cost--performance summaries only from Stage-A scientific endpoints and "
    "Stage-B timing replays emitted by decision.online_controller_evaluate."
)


def run_controller_cost_performance_pareto(*args: object, **kwargs: object) -> NoReturn:
    """Reject use of the superseded result producer."""

    del args, kwargs
    raise RuntimeError(RETIRED_REASON)


def main() -> NoReturn:
    parser = argparse.ArgumentParser(
        description="Retired legacy cost--performance result producer.",
        epilog=RETIRED_REASON,
    )
    parser.parse_args()
    parser.error(RETIRED_REASON)


if __name__ == "__main__":
    main()
