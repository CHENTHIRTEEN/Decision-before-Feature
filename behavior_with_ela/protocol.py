from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmarks import make_problem
from benchmarks.core import Problem
from benchmarks.mabbob import make_mabbob_problem
from trajectory.sampling import SAMPLING_PROTOCOL, get_sampling_spec


CORE_PORTFOLIO = ("pso", "shade", "cmaes")
DIMENSION = 10
FE_TOTAL = 10_000
POPULATION_SIZE = 40
BOUNDARY_HANDLING = "reflect"
SUPPORTED_SUITES = ("bbob", "mabbob", "cec2017", "cec2022")


@dataclass
class ReplicationConfig:
    full_coverage: int
    selected_state_repetitions: int
    selected_state_fraction: float
    include_event_states: bool
    selection_seed: int


@dataclass
class LocalLandscapeConfig:
    enabled: bool
    reservoir_size: int
    uniform_fraction: float
    recent_fraction: float
    elite_fraction: float
    elite_sample_fraction: float
    information_window_fe: int
    bootstrap_repetitions: int
    bootstrap_sample_size: int
    bootstrap_confidence: float


@dataclass
class QueryConfig:
    query_id: str
    sample_design_id: str
    minimum_post_query_FE: int


@dataclass
class SoftERTConfig:
    target_slack_fraction: float
    practical_FE_ratio: float
    timeout_penalty_offset_FE: int


@dataclass
class RepeatedDASConfig:
    max_switches: int
    minimum_dwell_FE: int
    hysteresis_probability_margin: float


@dataclass
class SuiteConfig:
    suite: str
    split: str
    functions: tuple[int, ...]
    instances: tuple[int, ...]
    definitions_path: Path | None = None


@dataclass
class ExperimentConfig:
    source_path: Path
    dimension: int
    fe_total: int
    population_size: int
    algorithms: tuple[str, ...]
    seeds: tuple[int, ...]
    sampling_protocol: str
    boundary_handling: str
    log10_gap_floor: float
    log10_gap_cap: float
    failure_loss_cap: float
    success_gap_target: float
    domain_gain_delta: float
    noise_delta_quantile: float
    output_root: Path
    replication: ReplicationConfig
    local_landscape: LocalLandscapeConfig
    query: QueryConfig
    soft_ert: SoftERTConfig
    repeated_das: RepeatedDASConfig
    suites: tuple[SuiteConfig, ...]

    def suite(self, split: str) -> SuiteConfig:
        matches = [item for item in self.suites if item.split == str(split)]
        if len(matches) != 1:
            raise ValueError(f"expected one suite with split={split!r}, found {len(matches)}")
        return matches[0]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("experiment config must contain a YAML mapping")

    replication_raw = raw.get("replication")
    if not isinstance(replication_raw, dict):
        raise ValueError("experiment config must define replication")
    replication = ReplicationConfig(
        full_coverage=int(replication_raw["full_coverage"]),
        selected_state_repetitions=int(replication_raw["selected_state_repetitions"]),
        selected_state_fraction=float(replication_raw["selected_state_fraction"]),
        include_event_states=bool(replication_raw["include_event_states"]),
        selection_seed=int(replication_raw["selection_seed"]),
    )
    local_raw = raw.get("local_landscape")
    if not isinstance(local_raw, dict):
        raise ValueError("experiment config must define local_landscape")
    local_landscape = LocalLandscapeConfig(
        enabled=bool(local_raw["enabled"]),
        reservoir_size=int(local_raw["reservoir_size"]),
        uniform_fraction=float(local_raw["uniform_fraction"]),
        recent_fraction=float(local_raw["recent_fraction"]),
        elite_fraction=float(local_raw["elite_fraction"]),
        elite_sample_fraction=float(local_raw["elite_sample_fraction"]),
        information_window_fe=int(local_raw["information_window_fe"]),
        bootstrap_repetitions=int(local_raw["bootstrap_repetitions"]),
        bootstrap_sample_size=int(local_raw["bootstrap_sample_size"]),
        bootstrap_confidence=float(local_raw["bootstrap_confidence"]),
    )
    query_raw = raw.get("query")
    if not isinstance(query_raw, dict):
        raise ValueError("experiment config must define query")
    query = QueryConfig(
        query_id=str(query_raw["query_id"]),
        sample_design_id=str(query_raw["sample_design_id"]),
        minimum_post_query_FE=int(query_raw["minimum_post_query_FE"]),
    )
    soft_ert_raw = raw.get("soft_ert")
    if not isinstance(soft_ert_raw, dict):
        raise ValueError("experiment config must define soft_ert")
    soft_ert = SoftERTConfig(
        target_slack_fraction=float(soft_ert_raw["target_slack_fraction"]),
        practical_FE_ratio=float(soft_ert_raw["practical_FE_ratio"]),
        timeout_penalty_offset_FE=int(
            soft_ert_raw["timeout_penalty_offset_FE"]
        ),
    )
    repeated_raw = raw.get("repeated_das")
    if not isinstance(repeated_raw, dict):
        raise ValueError("experiment config must define repeated_das")
    repeated_das = RepeatedDASConfig(
        max_switches=int(repeated_raw["max_switches"]),
        minimum_dwell_FE=int(repeated_raw["minimum_dwell_FE"]),
        hysteresis_probability_margin=float(
            repeated_raw["hysteresis_probability_margin"]
        ),
    )

    suite_rows = raw.get("suites")
    if not isinstance(suite_rows, list) or not suite_rows:
        raise ValueError("experiment config must define at least one suite")
    suites: list[SuiteConfig] = []
    for row in suite_rows:
        if not isinstance(row, dict):
            raise ValueError("each suite config must be a mapping")
        definitions = row.get("definitions_path")
        suites.append(
            SuiteConfig(
                suite=str(row["suite"]).lower(),
                split=str(row["split"]),
                functions=_integer_tuple(row.get("functions"), "functions"),
                instances=_integer_tuple(row.get("instances"), "instances"),
                definitions_path=None if definitions is None else Path(str(definitions)),
            )
        )

    config = ExperimentConfig(
        source_path=source,
        dimension=int(raw["dimension"]),
        fe_total=int(raw["FE_total"]),
        population_size=int(raw["population_size"]),
        algorithms=tuple(str(value).lower() for value in raw["algorithms"]),
        seeds=_integer_tuple(raw.get("seeds"), "seeds"),
        sampling_protocol=str(raw["sampling_protocol"]),
        boundary_handling=str(raw["boundary_handling"]),
        log10_gap_floor=float(raw["log10_gap_floor"]),
        log10_gap_cap=float(raw["log10_gap_cap"]),
        failure_loss_cap=float(raw["failure_loss_cap"]),
        success_gap_target=float(raw["success_gap_target"]),
        domain_gain_delta=float(raw["domain_gain_delta"]),
        noise_delta_quantile=float(raw["noise_delta_quantile"]),
        output_root=Path(str(raw["output_root"])),
        replication=replication,
        local_landscape=local_landscape,
        query=query,
        soft_ert=soft_ert,
        repeated_das=repeated_das,
        suites=tuple(suites),
    )
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    if config.dimension != DIMENSION:
        raise ValueError(f"Behavior-with-ELA Phase 1 requires dimension={DIMENSION}")
    if config.fe_total != FE_TOTAL:
        raise ValueError(f"Behavior-with-ELA Phase 1 requires FE_total={FE_TOTAL}")
    if config.population_size != POPULATION_SIZE:
        raise ValueError(
            f"Behavior-with-ELA Phase 1 requires population_size={POPULATION_SIZE}"
        )
    if config.algorithms != CORE_PORTFOLIO:
        raise ValueError(
            "Behavior-with-ELA portfolio must be ordered exactly as pso,shade,cmaes"
        )
    if len(config.seeds) != 10:
        raise ValueError("Behavior-with-ELA Phase 1 requires ten base seeds")
    if config.boundary_handling != BOUNDARY_HANDLING:
        raise ValueError("Behavior-with-ELA experiments require reflect boundary handling")
    get_sampling_spec(config.sampling_protocol)
    if config.sampling_protocol != SAMPLING_PROTOCOL:
        raise ValueError(f"sampling_protocol must be {SAMPLING_PROTOCOL}")
    if not 0.0 < config.log10_gap_floor < config.log10_gap_cap:
        raise ValueError("log10 gap bounds must satisfy 0 < floor < cap")
    if config.failure_loss_cap != config.log10_gap_cap:
        raise ValueError("failure_loss_cap must equal log10_gap_cap")
    if config.success_gap_target <= 0.0:
        raise ValueError("success_gap_target must be positive")
    if config.domain_gain_delta < 0.0:
        raise ValueError("domain_gain_delta must be non-negative")
    if not 0.0 < config.noise_delta_quantile < 1.0:
        raise ValueError("noise_delta_quantile must be in (0, 1)")
    if config.replication.full_coverage != 1:
        raise ValueError("full-coverage action outcomes must use one repetition")
    if config.replication.selected_state_repetitions not in {3, 5}:
        raise ValueError("selected states must use three or five repetitions")
    if not 0.0 <= config.replication.selected_state_fraction <= 1.0:
        raise ValueError("selected_state_fraction must be in [0, 1]")
    local = config.local_landscape
    if not local.enabled:
        raise ValueError("Behavior-with-ELA collection requires local_landscape.enabled=true")
    if local.reservoir_size < config.dimension + 2:
        raise ValueError("local landscape reservoir is too small for a linear meta-model")
    fractions = (
        local.uniform_fraction,
        local.recent_fraction,
        local.elite_fraction,
    )
    if any(value < 0.0 for value in fractions) or not np.isclose(
        sum(fractions), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("local landscape reservoir fractions must be non-negative and sum to one")
    if not 0.0 < local.elite_sample_fraction < 1.0:
        raise ValueError("elite_sample_fraction must be in (0, 1)")
    if not 2 <= local.information_window_fe <= config.fe_total:
        raise ValueError("information_window_fe must lie in [2, FE_total]")
    if local.bootstrap_repetitions < 2:
        raise ValueError("bootstrap_repetitions must be at least two")
    if not config.dimension + 2 <= local.bootstrap_sample_size <= local.reservoir_size:
        raise ValueError(
            "bootstrap_sample_size must lie between dimension + 2 and reservoir_size"
        )
    if not 0.0 < local.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap_confidence must be in (0, 1)")
    if not config.query.query_id or not config.query.sample_design_id:
        raise ValueError("query_id and sample_design_id must be non-empty")
    if config.query.minimum_post_query_FE < config.population_size:
        raise ValueError("minimum_post_query_FE must cover at least one population update")
    from landscape_queries.specs import get_query_spec

    query_spec = get_query_spec(config.query.query_id)
    if query_spec.sample_design_id != config.query.sample_design_id:
        raise ValueError("query_id and sample_design_id are inconsistent")
    query_fe = query_spec.sample_design.sample_size(config.dimension)
    if query_fe + config.query.minimum_post_query_FE > int(
        (1.0 - 0.60) * config.fe_total
    ):
        raise ValueError("latest decision state does not retain the required post-query FE")
    if not 0.0 <= config.soft_ert.target_slack_fraction < 1.0:
        raise ValueError("soft ERT target_slack_fraction must be in [0, 1)")
    if not 0.0 < config.soft_ert.practical_FE_ratio < 1.0:
        raise ValueError("soft ERT practical_FE_ratio must be in (0, 1)")
    if config.soft_ert.timeout_penalty_offset_FE < 1:
        raise ValueError("soft ERT timeout_penalty_offset_FE must be positive")
    if not 1 <= config.repeated_das.max_switches <= 3:
        raise ValueError("repeated DAS max_switches must lie in [1, 3]")
    if not config.population_size <= config.repeated_das.minimum_dwell_FE < config.fe_total:
        raise ValueError("repeated DAS minimum_dwell_FE is outside the valid range")
    if config.repeated_das.minimum_dwell_FE % config.population_size != 0:
        raise ValueError("repeated DAS minimum_dwell_FE must align to population updates")
    if not 0.0 <= config.repeated_das.hysteresis_probability_margin <= 1.0:
        raise ValueError("repeated DAS hysteresis margin must be in [0, 1]")
    if len({suite.split for suite in config.suites}) != len(config.suites):
        raise ValueError("suite split names must be unique")
    for suite in config.suites:
        if suite.suite not in SUPPORTED_SUITES:
            raise ValueError(f"unsupported suite: {suite.suite}")
        if suite.suite == "mabbob":
            if suite.definitions_path is None:
                raise ValueError("selected MA-BBOB requires definitions_path")
            if not suite.definitions_path.exists():
                raise FileNotFoundError(
                    f"missing selected MA-BBOB definitions: {suite.definitions_path}"
                )
        elif suite.definitions_path is not None:
            raise ValueError("definitions_path is only valid for MA-BBOB")


def make_experiment_problem(
    suite: SuiteConfig,
    *,
    function: int,
    instance: int,
    dimension: int,
    boundary_handling: str,
) -> Problem:
    if suite.suite != "mabbob":
        return make_problem(
            {
                "suite": suite.suite,
                "function": int(function),
                "instance": int(instance),
                "dimension": int(dimension),
                "boundary_handling": str(boundary_handling),
            }
        )
    if suite.definitions_path is None:
        raise ValueError("selected MA-BBOB suite is missing definitions_path")
    definition = selected_mabbob_definitions(suite.definitions_path).get(int(function))
    if definition is None:
        raise ValueError(
            f"MA-BBOB candidate {int(function)} is absent from {suite.definitions_path}"
        )
    return make_mabbob_problem(
        candidate_id=int(function),
        dimension=int(dimension),
        instance=int(instance),
        boundary_handling=str(boundary_handling),
        manifest_entry=definition,
    )


def check_problem_availability(
    config: ExperimentConfig,
    tasks: list[tuple[SuiteConfig, int]],
) -> None:
    for suite, function in tasks:
        for instance in suite.instances:
            try:
                problem = make_experiment_problem(
                    suite,
                    function=function,
                    instance=instance,
                    dimension=config.dimension,
                    boundary_handling=config.boundary_handling,
                )
                problem.close()
            except Exception as exc:
                raise ValueError(
                    "configured problem is unavailable: "
                    f"split={suite.split}, suite={suite.suite}, "
                    f"function={int(function)}, instance={int(instance)}, "
                    f"dimension={config.dimension}; {exc}"
                ) from exc


@lru_cache(maxsize=None)
def selected_mabbob_definitions(path: Path) -> dict[int, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rows = raw.get("selected") if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("MA-BBOB definitions file must contain a selected list")
    definitions: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each MA-BBOB definition must be a mapping")
        candidate = int(row["candidate_id"])
        if candidate in definitions:
            raise ValueError(f"duplicate MA-BBOB candidate definition: {candidate}")
        definitions[candidate] = dict(row)
    return definitions


def suite_code(suite: str) -> int:
    return {"bbob": 1, "cec2017": 2, "cec2022": 3, "mabbob": 4}[str(suite)]


def function_label(suite: str, function: int) -> str:
    name = str(suite)
    if name == "bbob":
        return f"bbob_f{int(function):03d}"
    if name == "mabbob":
        return f"mabbob_c{int(function):03d}"
    return f"{name}_f{int(function):02d}"


def _integer_tuple(values: Any, name: str) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(int(value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values")
    return result
