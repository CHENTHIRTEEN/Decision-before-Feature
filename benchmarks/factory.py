from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from benchmarks.bbob import make_bbob_problem
from benchmarks.cec import make_cec_problem
from benchmarks.core import Problem
from benchmarks.mabbob import make_mabbob_problem


_PROBLEM_ID_PATTERNS = {
    "bbob": re.compile(r"^bbob_f(\d{3})_i(\d+)_d(\d+)$"),
    "cec2017": re.compile(r"^cec2017_f(\d{2})_d(\d+)$"),
    "cec2022": re.compile(r"^cec2022_f(\d{2})_d(\d+)$"),
    "mabbob": re.compile(r"^mabbob_c(\d{3})_i(\d{2})_d(\d+)$"),
}


def make_problem(config: dict) -> Problem:
    suite = str(config["suite"]).lower()
    dimension = int(config["dimension"])

    if suite == "bbob":
        return make_bbob_problem(
            function=int(config["function"]),
            dimension=dimension,
            instance=int(config["instance"]),
            boundary_handling=str(config.get("boundary_handling", "clip")),
        )
    if suite in {"cec2017", "cec2022"}:
        return make_cec_problem(
            year=int(suite.removeprefix("cec")),
            function=int(config["function"]),
            dimension=dimension,
            boundary_handling=str(config.get("boundary_handling", "reflect")),
        )
    if suite == "mabbob":
        manifest_entry = config.get("manifest_entry")
        if manifest_entry is None:
            manifest_entry = _load_mabbob_manifest_entry(config)
        return make_mabbob_problem(
            candidate_id=int(config.get("candidate_id", config.get("function", 1))),
            dimension=dimension,
            instance=int(config.get("instance", 1)),
            boundary_handling=str(config.get("boundary_handling", "clip")),
            manifest_entry=manifest_entry,
        )

    raise ValueError(f"unsupported benchmark suite: {suite}")


@lru_cache(maxsize=None)
def problem_bounds(problem_id: str) -> tuple[np.ndarray, np.ndarray]:
    # MA-BBOB bounds are definition-independent ([-5, 5]^d); returning them
    # directly avoids reconstructing a manifest-driven problem from the ID
    # alone, which cannot resolve the candidate definition.
    config = _problem_config_from_id(problem_id)
    if str(config["suite"]).lower() == "mabbob":
        dimension = int(config["dimension"])
        return (
            np.full(dimension, -5.0, dtype=float),
            np.full(dimension, 5.0, dtype=float),
        )
    problem = make_problem(config)
    try:
        return problem.lower_bounds.copy(), problem.upper_bounds.copy()
    finally:
        problem.close()


def _manifest_path_from_config(config: dict) -> Path | None:
    output = config.get("output")
    if output:
        output_path = Path(str(output))
        candidate = output_path.with_name("mabbob_diversity_manifest.json")
        if candidate.exists():
            return candidate
    manifest_path = config.get("manifest_path")
    if manifest_path:
        path = Path(str(manifest_path))
        if not path.exists():
            # A declared-but-missing manifest must fail loudly: silently
            # falling back to the legacy generator once produced invalid
            # formal shards (candidates 1-200 resolve against legacy
            # definitions without any error).
            raise FileNotFoundError(
                f"config declares manifest_path but it does not exist: {path}"
            )
        return path
    return None


def _load_mabbob_manifest_entry(config: dict) -> dict | None:
    manifest_path = _manifest_path_from_config(config)
    if manifest_path is None:
        raise ValueError(
            "MA-BBOB problems require a pool manifest (manifest_path or a "
            "sibling mabbob_diversity_manifest.json next to the output); "
            "manifest-less MA-BBOB collection is invalid"
        )
    function = int(config.get("candidate_id", config.get("function", 1)))
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for entry in manifest.get("selected", []):
        if int(entry.get("candidate_id", -1)) == function:
            return entry
    raise ValueError(f"candidate {function} not found in manifest {manifest_path}")


def _problem_config_from_id(problem_id: str) -> dict[str, int | str]:
    for suite, pattern in _PROBLEM_ID_PATTERNS.items():
        match = pattern.match(problem_id)
        if match is None:
            continue
        if suite == "bbob":
            function, instance, dimension = (int(value) for value in match.groups())
            return {
                "suite": suite,
                "function": function,
                "instance": instance,
                "dimension": dimension,
            }
        if suite == "mabbob":
            candidate_id, instance, dimension = (int(value) for value in match.groups())
            return {
                "suite": suite,
                "candidate_id": candidate_id,
                "instance": instance,
                "dimension": dimension,
            }
        function, dimension = (int(value) for value in match.groups())
        return {
            "suite": suite,
            "function": function,
            "dimension": dimension,
        }
    raise ValueError(f"unsupported problem_id: {problem_id}")
