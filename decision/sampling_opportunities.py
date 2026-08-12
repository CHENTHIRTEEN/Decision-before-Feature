from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trajectory.sampling import SAMPLING_METADATA_COLUMNS


STATE_KEY_COLUMNS = (
    "split",
    "problem_id",
    "family",
    "dimension",
    "prefix_algorithm",
    "seed",
    "FE",
)


def with_sampling_opportunity_type(frame: pd.DataFrame, *, artifact: str) -> pd.DataFrame:
    required = {*STATE_KEY_COLUMNS, "FE_ratio", *SAMPLING_METADATA_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{artifact} is missing dynamic-sampling columns: {missing}")
    milestone = frame["is_budget_milestone"].to_numpy(dtype=bool)
    event = frame["is_event_sample"].to_numpy(dtype=bool)
    if bool(np.any(~milestone & ~event)):
        raise ValueError(f"{artifact} contains rows that are neither budget milestones nor accepted event samples")
    output = frame.copy()
    output["sampling_opportunity_type"] = np.select(
        [milestone & event, milestone, event],
        ["budget_milestone_with_event", "budget_milestone", "event_only"],
        default="invalid",
    )
    return output


def assert_unique_state_keys(frame: pd.DataFrame, *, artifact: str) -> None:
    duplicated = frame.duplicated(list(STATE_KEY_COLUMNS), keep=False)
    if bool(duplicated.any()):
        example = frame.loc[duplicated, list(STATE_KEY_COLUMNS)].iloc[0].to_dict()
        raise ValueError(f"{artifact} contains duplicate integer-FE state keys: {example}")


def assert_aligned_decision_opportunities(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reference_artifact: str,
    candidate_artifact: str,
) -> None:
    reference = with_sampling_opportunity_type(reference, artifact=reference_artifact)
    candidate = with_sampling_opportunity_type(candidate, artifact=candidate_artifact)
    assert_unique_state_keys(reference, artifact=reference_artifact)
    assert_unique_state_keys(candidate, artifact=candidate_artifact)
    reference = reference.sort_values(list(STATE_KEY_COLUMNS)).reset_index(drop=True)
    candidate = candidate.sort_values(list(STATE_KEY_COLUMNS)).reset_index(drop=True)
    if len(reference) != len(candidate):
        raise ValueError(
            f"{reference_artifact} and {candidate_artifact} must contain the same decision-opportunity row count"
        )
    for column in STATE_KEY_COLUMNS:
        if not np.array_equal(reference[column].to_numpy(), candidate[column].to_numpy()):
            raise ValueError(
                f"{reference_artifact} and {candidate_artifact} use different decision opportunities at {column}"
            )
    for column in ("FE_ratio", *SAMPLING_METADATA_COLUMNS, "sampling_opportunity_type"):
        left = reference[column].tolist()
        right = candidate[column].tolist()
        if not all(_metadata_values_equal(a, b) for a, b in zip(left, right, strict=True)):
            raise ValueError(
                f"{reference_artifact} and {candidate_artifact} disagree on decision-opportunity metadata {column}"
            )


def _metadata_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (list, tuple, np.ndarray)) or isinstance(right, (list, tuple, np.ndarray)):
        if not isinstance(left, (list, tuple, np.ndarray)) or not isinstance(right, (list, tuple, np.ndarray)):
            return False
        return tuple(left) == tuple(right)
    if _is_missing(left) or _is_missing(right):
        return _is_missing(left) and _is_missing(right)
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
    return bool(left == right)


def _is_missing(value: Any) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
