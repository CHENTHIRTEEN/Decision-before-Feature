from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MATCHED_RANDOM_PROTOCOL = "matched_train_oof_first_trigger"
MATCHED_RANDOM_SOURCE_SPLIT = "bbob_train_oof"
MATCHED_RANDOM_STREAM_CODE = 2026081201
MATCHED_RANDOM_EVENT_CODE = 1
MATCHED_RANDOM_QUANTILE_PROBABILITIES = tuple(
    float(value) for value in np.linspace(0.0, 1.0, 101)
)

_SUITE_CODES = {
    "bbob": 1,
    "cec2017": 2,
    "cec2022": 3,
}
_ALGORITHM_CODES = {
    "de": 101,
    "pso": 202,
    "cmaes": 303,
    "shade": 404,
}
_PROBLEM_PATTERNS = {
    "bbob": re.compile(r"^bbob_f(\d{3})_i(\d+)_d(\d+)$"),
    "cec2017": re.compile(r"^cec2017_f(\d{2})_d(\d+)$"),
    "cec2022": re.compile(r"^cec2022_f(\d{2})_d(\d+)$"),
}


@dataclass
class MatchedRandomCalibration:
    query_id: str
    query_protocol: str
    feature_group: str
    selected_model: str
    threshold_mode: str
    run_call_rate: float
    trigger_quantile_probabilities: tuple[float, ...]
    trigger_fe_quantiles: tuple[float, ...]
    source_split: str
    seed: int
    stream_code: int = MATCHED_RANDOM_STREAM_CODE
    protocol: str = MATCHED_RANDOM_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != MATCHED_RANDOM_PROTOCOL:
            raise ValueError("matched-random calibration uses an unsupported protocol")
        if self.source_split != MATCHED_RANDOM_SOURCE_SPLIT:
            raise ValueError("matched-random calibration must come from BBOB-train OOF trajectories")
        if not self.query_id or not self.query_protocol or not self.feature_group or not self.selected_model:
            raise ValueError("matched-random calibration identity fields must not be empty")
        if not self.threshold_mode:
            raise ValueError("matched-random calibration threshold_mode must not be empty")
        if not np.isfinite(float(self.run_call_rate)) or not 0.0 <= float(self.run_call_rate) <= 1.0:
            raise ValueError("matched-random run_call_rate must lie in [0, 1]")
        probabilities = np.asarray(self.trigger_quantile_probabilities, dtype=float)
        fe_quantiles = np.asarray(self.trigger_fe_quantiles, dtype=float)
        if probabilities.ndim != 1 or fe_quantiles.ndim != 1 or len(probabilities) != len(fe_quantiles):
            raise ValueError("matched-random trigger quantile arrays must be aligned one-dimensional arrays")
        if float(self.run_call_rate) > 0.0 and len(probabilities) < 2:
            raise ValueError("a non-zero matched-random call rate requires a trigger-time distribution")
        if len(probabilities):
            if not np.isfinite(probabilities).all() or not np.isfinite(fe_quantiles).all():
                raise ValueError("matched-random trigger quantiles must be finite")
            if probabilities[0] != 0.0 or probabilities[-1] != 1.0:
                raise ValueError("matched-random trigger quantile probabilities must span [0, 1]")
            if np.any(np.diff(probabilities) <= 0.0):
                raise ValueError("matched-random trigger quantile probabilities must be strictly increasing")
            if np.any(np.diff(fe_quantiles) < 0.0):
                raise ValueError("matched-random trigger FE quantiles must be non-decreasing")
            if np.any((fe_quantiles <= 0.0) | (fe_quantiles >= 1.0)):
                raise ValueError("matched-random trigger FE quantiles must lie in (0, 1)")
        if int(self.seed) < 0 or int(self.stream_code) <= 0:
            raise ValueError("matched-random seed and stream_code must be non-negative integers")

    def payload(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "query_id": self.query_id,
            "query_protocol": self.query_protocol,
            "feature_group": self.feature_group,
            "selected_model": self.selected_model,
            "threshold_mode": self.threshold_mode,
            "run_call_rate": float(self.run_call_rate),
            "trigger_fe_quantiles": {
                "probabilities": list(self.trigger_quantile_probabilities),
                "fe_ratios": list(self.trigger_fe_quantiles),
            },
            "source_split": self.source_split,
            "seed": int(self.seed),
            "stream_code": int(self.stream_code),
        }

    def write(self, path: Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.payload(), indent=2, sort_keys=True), encoding="utf-8")
        return output


def make_matched_random_calibration(
    *,
    query_id: str,
    query_protocol: str,
    feature_group: str,
    selected_model: str,
    threshold_mode: str,
    run_call_rate: float,
    trigger_fe_ratios: np.ndarray,
    seed: int,
    stream_code: int = MATCHED_RANDOM_STREAM_CODE,
) -> MatchedRandomCalibration:
    observed = np.asarray(trigger_fe_ratios, dtype=float).reshape(-1)
    if float(run_call_rate) > 0.0:
        if not len(observed) or not np.isfinite(observed).all():
            raise ValueError("called train-OOF trajectories require finite trigger FE ratios")
        probabilities = np.asarray(MATCHED_RANDOM_QUANTILE_PROBABILITIES, dtype=float)
        fe_quantiles = np.quantile(observed, probabilities, method="linear")
    else:
        probabilities = np.asarray([], dtype=float)
        fe_quantiles = np.asarray([], dtype=float)
    return MatchedRandomCalibration(
        query_id=str(query_id),
        query_protocol=str(query_protocol),
        feature_group=str(feature_group),
        selected_model=str(selected_model),
        threshold_mode=str(threshold_mode),
        run_call_rate=float(run_call_rate),
        trigger_quantile_probabilities=tuple(float(value) for value in probabilities),
        trigger_fe_quantiles=tuple(float(value) for value in fe_quantiles),
        source_split=MATCHED_RANDOM_SOURCE_SPLIT,
        seed=int(seed),
        stream_code=int(stream_code),
    )


def load_matched_random_calibration(
    path: Path,
    *,
    query_id: str,
    query_protocol: str,
    feature_group: str,
    selected_model: str,
    threshold_mode: str,
) -> MatchedRandomCalibration:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"missing train-OOF matched-random calibration: {input_path}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    quantiles = payload.get("trigger_fe_quantiles", {})
    if not isinstance(quantiles, dict):
        raise ValueError("matched-random trigger_fe_quantiles must be a mapping")
    calibration = MatchedRandomCalibration(
        query_id=str(payload.get("query_id", "")),
        query_protocol=str(payload.get("query_protocol", "")),
        feature_group=str(payload.get("feature_group", "")),
        selected_model=str(payload.get("selected_model", "")),
        threshold_mode=str(payload.get("threshold_mode", "")),
        run_call_rate=float(payload.get("run_call_rate", np.nan)),
        trigger_quantile_probabilities=tuple(float(value) for value in quantiles.get("probabilities", [])),
        trigger_fe_quantiles=tuple(float(value) for value in quantiles.get("fe_ratios", [])),
        source_split=str(payload.get("source_split", "")),
        seed=int(payload.get("seed", -1)),
        stream_code=int(payload.get("stream_code", 0)),
        protocol=str(payload.get("protocol", "")),
    )
    expected = {
        "query_id": str(query_id),
        "query_protocol": str(query_protocol),
        "feature_group": str(feature_group),
        "selected_model": str(selected_model),
        "threshold_mode": str(threshold_mode),
    }
    observed_identity = {
        "query_id": calibration.query_id,
        "query_protocol": calibration.query_protocol,
        "feature_group": calibration.feature_group,
        "selected_model": calibration.selected_model,
        "threshold_mode": calibration.threshold_mode,
    }
    if observed_identity != expected:
        raise ValueError(
            "matched-random calibration does not match the deployed controller: "
            f"expected={expected}, observed={observed_identity}"
        )
    return calibration


def matched_random_target(
    calibration: MatchedRandomCalibration,
    *,
    problem_id: str,
    prefix_algorithm: str,
    run_seed: int,
    dimension: int,
    repetition: int,
) -> float | None:
    suite_code, function, unit_number, problem_dimension = _problem_components(problem_id)
    algorithm = str(prefix_algorithm).lower()
    if algorithm not in _ALGORITHM_CODES:
        raise ValueError(f"unsupported matched-random prefix algorithm: {prefix_algorithm}")
    if int(dimension) != problem_dimension:
        raise ValueError("matched-random dimension does not match problem_id")
    if int(run_seed) < 0 or int(repetition) < 0:
        raise ValueError("matched-random run seed and repetition must be non-negative")
    sequence = np.random.SeedSequence(
        [
            int(calibration.seed),
            int(unit_number),
            int(calibration.stream_code),
            0,
            int(_ALGORITHM_CODES[algorithm]),
            MATCHED_RANDOM_EVENT_CODE,
            int(suite_code),
            int(function),
            int(dimension),
            int(run_seed),
            int(repetition),
        ]
    )
    rng = np.random.default_rng(sequence)
    if float(rng.random()) >= float(calibration.run_call_rate):
        return None
    if not calibration.trigger_fe_quantiles:
        raise RuntimeError("matched-random calibration selected a call without trigger quantiles")
    probability = float(rng.random())
    return float(
        np.interp(
            probability,
            np.asarray(calibration.trigger_quantile_probabilities, dtype=float),
            np.asarray(calibration.trigger_fe_quantiles, dtype=float),
        )
    )


def _problem_components(problem_id: str) -> tuple[int, int, int, int]:
    value = str(problem_id)
    for suite, pattern in _PROBLEM_PATTERNS.items():
        match = pattern.match(value)
        if match is None:
            continue
        if suite == "bbob":
            function, instance, dimension = (int(part) for part in match.groups())
            return _SUITE_CODES[suite], function, instance, dimension
        function, dimension = (int(part) for part in match.groups())
        return _SUITE_CODES[suite], function, 1, dimension
    raise ValueError(f"unsupported problem_id for matched-random policy: {problem_id}")
