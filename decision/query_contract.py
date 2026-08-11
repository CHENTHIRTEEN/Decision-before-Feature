from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from landscape_queries.specs import get_query_spec


def decision_query_root(query_id: str) -> Path:
    get_query_spec(query_id)
    return Path("results/decision") / query_id


def validate_query_payload(payload: Mapping[str, Any], *, query_id: str, artifact: str) -> None:
    spec = get_query_spec(query_id)
    expected = {
        "query_id": spec.query_id,
        "query_protocol": spec.protocol,
        "sample_design_id": spec.sample_design_id,
    }
    mismatches = {
        field: {"expected": value, "observed": payload.get(field)}
        for field, value in expected.items()
        if payload.get(field) != value
    }
    if mismatches:
        raise ValueError(f"{artifact} does not match the requested landscape-query protocol: {mismatches}")


def validate_query_frame(frame: pd.DataFrame, *, query_id: str, artifact: str) -> None:
    spec = get_query_spec(query_id)
    expected = {
        "query_id": spec.query_id,
        "query_protocol": spec.protocol,
        "sample_design_id": spec.sample_design_id,
    }
    missing = sorted(set(expected).difference(frame.columns))
    if missing:
        raise ValueError(
            f"{artifact} predates the active landscape-query protocol; missing fields: {missing}"
        )
    for field, value in expected.items():
        observed = set(frame[field].astype(str))
        if observed != {str(value)}:
            raise ValueError(
                f"{artifact} contains the wrong {field}: expected {value!r}, observed {sorted(observed)}"
            )
