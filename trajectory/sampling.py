from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt
from typing import Any

import pyarrow as pa


SAMPLING_PROTOCOL = "phase1_dynamic_budget_event_v1"
MONITOR_START_RATIO = 0.20
MONITOR_END_RATIO = 0.60
MONITOR_STEP_RATIO = 0.01
EARLY_PHASE_END_RATIO = 0.30
MID_PHASE_END_RATIO = 0.50
SAMPLING_PHASES = ("early", "mid", "late")
BUDGET_MILESTONE_RATIOS = (
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
    0.34,
    0.38,
    0.42,
    0.46,
    0.50,
    0.60,
)
MONITOR_RATIOS = tuple(
    round(MONITOR_START_RATIO + MONITOR_STEP_RATIO * index, 2)
    for index in range(
        int(round((MONITOR_END_RATIO - MONITOR_START_RATIO) / MONITOR_STEP_RATIO)) + 1
    )
)
MIN_SAMPLES_PER_RUN = len(BUDGET_MILESTONE_RATIOS)
EVENT_ONLY_MIN_GAP_RATIO = 0.02
MAX_EVENT_ONLY_PER_PHASE = 2
MAX_SAMPLES_PER_RUN = MIN_SAMPLES_PER_RUN + len(SAMPLING_PHASES) * MAX_EVENT_ONLY_PER_PHASE
EPS = 1e-12

STAGNATION_ONSET_THRESHOLD = 0.05
STAGNATION_ONSET_REARM_THRESHOLD = 0.0
RANK_CHANGE_THRESHOLD = 0.20
RANK_CHANGE_REARM_THRESHOLD = 0.10
ELITE_MIGRATION_THRESHOLD = 0.05
ELITE_MIGRATION_REARM_THRESHOLD = 0.025
DIVERSITY_RECOVERY_THRESHOLD = 0.10
DIVERSITY_RECOVERY_REARM_THRESHOLD = 0.05

FROZEN_EVENT_THRESHOLDS = {
    "stagnation_onset": STAGNATION_ONSET_THRESHOLD,
    "rank_change": RANK_CHANGE_THRESHOLD,
    "elite_migration": ELITE_MIGRATION_THRESHOLD,
    "diversity_recovery": DIVERSITY_RECOVERY_THRESHOLD,
}
FROZEN_EVENT_REARM_THRESHOLDS = {
    "stagnation_onset": STAGNATION_ONSET_REARM_THRESHOLD,
    "rank_change": RANK_CHANGE_REARM_THRESHOLD,
    "elite_migration": ELITE_MIGRATION_REARM_THRESHOLD,
    "diversity_recovery": DIVERSITY_RECOVERY_REARM_THRESHOLD,
}

EVENT_NAMES = (
    "improvement_resume",
    "stagnation_onset",
    "rank_change",
    "elite_migration",
    "diversity_recovery",
)

SAMPLING_METADATA_SCHEMA_FIELDS = (
    ("sampling_protocol", pa.string(), False),
    ("sampling_phase", pa.string(), False),
    ("sampling_triggers", pa.list_(pa.string()), False),
    ("is_budget_milestone", pa.bool_(), False),
    ("budget_milestone_ratio", pa.float64(), True),
    ("is_event_sample", pa.bool_(), False),
    ("monitor_target_ratio", pa.float64(), False),
    ("event_index_in_phase", pa.int64(), True),
    ("event_improvement_resume", pa.bool_(), False),
    ("event_stagnation_onset", pa.bool_(), False),
    ("event_rank_change", pa.bool_(), False),
    ("event_elite_migration", pa.bool_(), False),
    ("event_diversity_recovery", pa.bool_(), False),
    ("event_improvement_resume_metric", pa.float64(), False),
    ("event_stagnation_onset_metric", pa.float64(), False),
    ("event_rank_change_metric", pa.float64(), False),
    ("event_elite_migration_metric", pa.float64(), False),
    ("event_diversity_recovery_metric", pa.float64(), False),
)
SAMPLING_METADATA_COLUMNS = tuple(field[0] for field in SAMPLING_METADATA_SCHEMA_FIELDS)


@dataclass(frozen=True)
class SamplingSpec:
    protocol: str = SAMPLING_PROTOCOL
    budget_milestone_ratios: tuple[float, ...] = BUDGET_MILESTONE_RATIOS
    monitor_ratios: tuple[float, ...] = MONITOR_RATIOS
    min_samples_per_run: int = MIN_SAMPLES_PER_RUN
    max_samples_per_run: int = MAX_SAMPLES_PER_RUN
    event_only_min_gap_ratio: float = EVENT_ONLY_MIN_GAP_RATIO
    max_event_only_per_phase: int = MAX_EVENT_ONLY_PER_PHASE
    stagnation_onset_threshold: float = STAGNATION_ONSET_THRESHOLD
    stagnation_onset_rearm_threshold: float = STAGNATION_ONSET_REARM_THRESHOLD
    rank_change_threshold: float = RANK_CHANGE_THRESHOLD
    rank_change_rearm_threshold: float = RANK_CHANGE_REARM_THRESHOLD
    elite_migration_threshold: float = ELITE_MIGRATION_THRESHOLD
    elite_migration_rearm_threshold: float = ELITE_MIGRATION_REARM_THRESHOLD
    diversity_recovery_threshold: float = DIVERSITY_RECOVERY_THRESHOLD
    diversity_recovery_rearm_threshold: float = DIVERSITY_RECOVERY_REARM_THRESHOLD


@dataclass(frozen=True)
class SamplingDecision:
    should_emit: bool
    sampling_protocol: str
    sampling_phase: str
    sampling_triggers: tuple[str, ...]
    is_budget_milestone: bool
    budget_milestone_ratio: float | None
    is_event_sample: bool
    monitor_target_ratio: float
    event_index_in_phase: int | None
    event_improvement_resume: bool
    event_stagnation_onset: bool
    event_rank_change: bool
    event_elite_migration: bool
    event_diversity_recovery: bool
    event_improvement_resume_metric: float
    event_stagnation_onset_metric: float
    event_rank_change_metric: float
    event_elite_migration_metric: float
    event_diversity_recovery_metric: float

    def metadata(self) -> dict[str, Any]:
        payload = {
            name: getattr(self, name)
            for name in SAMPLING_METADATA_COLUMNS
        }
        payload["sampling_triggers"] = list(self.sampling_triggers)
        return payload


def get_sampling_spec(protocol: str) -> SamplingSpec:
    if str(protocol) != SAMPLING_PROTOCOL:
        raise ValueError(f"unsupported trajectory sampling protocol: {protocol}")
    spec = SamplingSpec()
    _validate_frozen_sampling_spec(spec)
    return spec


def _validate_frozen_sampling_spec(spec: SamplingSpec) -> None:
    if spec.protocol != SAMPLING_PROTOCOL:
        raise RuntimeError("sampling spec protocol must equal the frozen protocol")
    if spec.budget_milestone_ratios != BUDGET_MILESTONE_RATIOS:
        raise RuntimeError("sampling spec milestones must equal the frozen milestones")
    if spec.monitor_ratios != MONITOR_RATIOS:
        raise RuntimeError("sampling spec monitor grid must equal the frozen grid")
    if tuple(sorted(set(spec.monitor_ratios))) != spec.monitor_ratios:
        raise RuntimeError("frozen monitor ratios must be strictly increasing and unique")
    if len(spec.monitor_ratios) != 41 or not isclose(
        spec.monitor_ratios[0], MONITOR_START_RATIO, rel_tol=0.0, abs_tol=EPS
    ) or not isclose(
        spec.monitor_ratios[-1], MONITOR_END_RATIO, rel_tol=0.0, abs_tol=EPS
    ):
        raise RuntimeError("frozen monitor grid must contain 41 points from 0.20 through 0.60")
    if any(
        not isclose(later - earlier, MONITOR_STEP_RATIO, rel_tol=0.0, abs_tol=EPS)
        for earlier, later in zip(spec.monitor_ratios, spec.monitor_ratios[1:])
    ):
        raise RuntimeError("frozen monitor grid must use the 0.01 step")
    if not set(spec.budget_milestone_ratios).issubset(spec.monitor_ratios):
        raise RuntimeError("all frozen milestones must belong to the monitor grid")
    if spec.min_samples_per_run != len(spec.budget_milestone_ratios):
        raise RuntimeError("minimum samples must equal the frozen milestone count")
    if spec.max_samples_per_run != (
        spec.min_samples_per_run + len(SAMPLING_PHASES) * spec.max_event_only_per_phase
    ):
        raise RuntimeError("maximum samples must equal milestones plus phase event-only quotas")
    if spec.event_only_min_gap_ratio != EVENT_ONLY_MIN_GAP_RATIO:
        raise RuntimeError("event-only minimum gap must equal the frozen 0.02 ratio")
    if spec.max_event_only_per_phase != MAX_EVENT_ONLY_PER_PHASE:
        raise RuntimeError("event-only phase quota must equal the frozen value 2")
    for name in FROZEN_EVENT_THRESHOLDS:
        threshold = float(getattr(spec, f"{name}_threshold"))
        rearm = float(getattr(spec, f"{name}_rearm_threshold"))
        if threshold != FROZEN_EVENT_THRESHOLDS[name] or rearm != FROZEN_EVENT_REARM_THRESHOLDS[name]:
            raise RuntimeError(f"sampling spec {name} thresholds must equal the frozen values")
        if not 0.0 <= rearm < threshold:
            raise RuntimeError(f"sampling spec {name} requires 0 <= rearm < trigger threshold")


def sampling_phase(ratio: float) -> str:
    value = float(ratio)
    if MONITOR_START_RATIO - EPS <= value < EARLY_PHASE_END_RATIO - EPS:
        return "early"
    if EARLY_PHASE_END_RATIO - EPS <= value < MID_PHASE_END_RATIO - EPS:
        return "mid"
    if MID_PHASE_END_RATIO - EPS <= value <= MONITOR_END_RATIO + EPS:
        return "late"
    raise ValueError(f"monitor ratio is outside the frozen phase range: {ratio}")


def is_budget_milestone(ratio: float) -> bool:
    return any(isclose(float(ratio), value, rel_tol=0.0, abs_tol=EPS) for value in BUDGET_MILESTONE_RATIOS)


def budget_milestone_metadata(ratio: float) -> dict[str, Any]:
    """Build explicit metadata for a frozen budget-only observation."""
    target = float(ratio)
    if not is_budget_milestone(target):
        raise ValueError(f"ratio is not a frozen budget milestone: {ratio}")
    return SamplingDecision(
        should_emit=True,
        sampling_protocol=SAMPLING_PROTOCOL,
        sampling_phase=sampling_phase(target),
        sampling_triggers=("budget_milestone",),
        is_budget_milestone=True,
        budget_milestone_ratio=target,
        is_event_sample=False,
        monitor_target_ratio=target,
        event_index_in_phase=None,
        event_improvement_resume=False,
        event_stagnation_onset=False,
        event_rank_change=False,
        event_elite_migration=False,
        event_diversity_recovery=False,
        event_improvement_resume_metric=0.0,
        event_stagnation_onset_metric=0.0,
        event_rank_change_metric=0.0,
        event_elite_migration_metric=0.0,
        event_diversity_recovery_metric=0.0,
    ).metadata()


class DynamicSamplingPolicy:
    def __init__(self, protocol: str = SAMPLING_PROTOCOL) -> None:
        self.spec = get_sampling_spec(protocol)
        self._next_monitor_index = 0
        self._previous_best_fitness: float | None = None
        self._last_improvement_fe: int | None = None
        self._previous_improvement_frequency: float | None = None
        self._armed = {name: True for name in EVENT_NAMES}
        self._armed["improvement_resume"] = False
        self._event_only_counts = {phase: 0 for phase in SAMPLING_PHASES}
        self._last_event_only_ratio: float | None = None

    @property
    def complete(self) -> bool:
        return self._next_monitor_index >= len(self.spec.monitor_ratios)

    @property
    def next_monitor_ratio(self) -> float | None:
        if self.complete:
            return None
        return float(self.spec.monitor_ratios[self._next_monitor_index])

    def observe_update(self, *, fe: int, best_fitness: float) -> None:
        current_best = float(best_fitness)
        if self._previous_best_fitness is None:
            self._previous_best_fitness = current_best
            self._last_improvement_fe = int(fe)
            return
        if _strict_improvement(self._previous_best_fitness, current_best):
            self._last_improvement_fe = int(fe)
            self._armed["stagnation_onset"] = True
        self._previous_best_fitness = current_best

    def pending_monitor_ratios(self, actual_fe_ratio: float) -> tuple[float, ...]:
        pending = []
        while not self.complete:
            target = self.spec.monitor_ratios[self._next_monitor_index]
            if float(actual_fe_ratio) + EPS < target:
                break
            pending.append(target)
            self._next_monitor_index += 1
        return tuple(pending)

    def stagnation_span_ratio(self, *, current_fe: int, fe_total: int) -> float:
        if self._last_improvement_fe is None:
            raise ValueError("sampling policy has not observed the initialization population")
        return float(max(int(current_fe) - self._last_improvement_fe, 0) / int(fe_total))

    def decide_pending(
        self,
        *,
        monitor_target_ratios: tuple[float, ...],
        actual_fe_ratio: float,
        event_improvement_resume_metric: float,
        event_stagnation_onset_metric: float,
        event_rank_change_metric: float,
        event_elite_migration_metric: float,
        event_diversity_recovery_metric: float,
    ) -> SamplingDecision:
        if not monitor_target_ratios:
            raise ValueError("at least one pending monitor ratio is required")
        targets = tuple(float(value) for value in monitor_target_ratios)
        if targets != tuple(sorted(set(targets))):
            raise ValueError("pending monitor ratios must be strictly increasing and unique")
        if any(
            not any(isclose(target, value, rel_tol=0.0, abs_tol=EPS) for value in self.spec.monitor_ratios)
            for target in targets
        ):
            raise ValueError("pending monitor ratios must belong to the frozen monitor grid")
        milestones = tuple(target for target in targets if is_budget_milestone(target))
        if len(milestones) > 1:
            raise ValueError(
                "multiple budget milestones aligned to one native update; the frozen "
                "population-size and FE budgets require each update to span at most "
                "0.01, below the minimum 0.02 milestone gap"
            )
        # One completed native update that crossed at least one monitor-grid
        # point constitutes one causal event check. When it crosses a milestone,
        # the milestone owns the merged row; otherwise the latest crossed grid
        # point canonically represents the update.
        target = milestones[0] if milestones else targets[-1]
        phase = sampling_phase(target)
        milestone = bool(milestones)

        improvement_resume = self._improvement_resume_event(float(event_improvement_resume_metric))
        stagnation_onset = self._threshold_event(
            "stagnation_onset",
            float(event_stagnation_onset_metric),
            threshold=self.spec.stagnation_onset_threshold,
            rearm_threshold=self.spec.stagnation_onset_rearm_threshold,
        )
        rank_change = self._threshold_event(
            "rank_change",
            abs(float(event_rank_change_metric)),
            threshold=self.spec.rank_change_threshold,
            rearm_threshold=self.spec.rank_change_rearm_threshold,
        )
        elite_migration = self._threshold_event(
            "elite_migration",
            float(event_elite_migration_metric),
            threshold=self.spec.elite_migration_threshold,
            rearm_threshold=self.spec.elite_migration_rearm_threshold,
        )
        diversity_recovery = self._threshold_event(
            "diversity_recovery",
            float(event_diversity_recovery_metric),
            threshold=self.spec.diversity_recovery_threshold,
            rearm_threshold=self.spec.diversity_recovery_rearm_threshold,
        )
        flags = {
            "improvement_resume": improvement_resume,
            "stagnation_onset": stagnation_onset,
            "rank_change": rank_change,
            "elite_migration": elite_migration,
            "diversity_recovery": diversity_recovery,
        }
        event_names = tuple(name for name in EVENT_NAMES if flags[name])
        event_index = None
        should_emit = milestone
        # A milestone+event row is not event-only and therefore consumes
        # neither phase quota nor the event-only gap anchor.
        if event_names and not milestone:
            gap_ok = (
                self._last_event_only_ratio is None
                or float(actual_fe_ratio) - self._last_event_only_ratio >= self.spec.event_only_min_gap_ratio - EPS
            )
            quota_ok = self._event_only_counts[phase] < self.spec.max_event_only_per_phase
            if gap_ok and quota_ok:
                self._event_only_counts[phase] += 1
                event_index = self._event_only_counts[phase]
                self._last_event_only_ratio = float(actual_fe_ratio)
                should_emit = True

        triggers = (("budget_milestone",) if milestone else ()) + event_names
        return SamplingDecision(
            should_emit=should_emit,
            sampling_protocol=self.spec.protocol,
            sampling_phase=phase,
            sampling_triggers=triggers,
            is_budget_milestone=milestone,
            budget_milestone_ratio=target if milestone else None,
            is_event_sample=bool(event_names),
            monitor_target_ratio=target,
            event_index_in_phase=event_index,
            event_improvement_resume=improvement_resume,
            event_stagnation_onset=stagnation_onset,
            event_rank_change=rank_change,
            event_elite_migration=elite_migration,
            event_diversity_recovery=diversity_recovery,
            event_improvement_resume_metric=float(event_improvement_resume_metric),
            event_stagnation_onset_metric=float(event_stagnation_onset_metric),
            event_rank_change_metric=float(event_rank_change_metric),
            event_elite_migration_metric=float(event_elite_migration_metric),
            event_diversity_recovery_metric=float(event_diversity_recovery_metric),
        )

    def decide(
        self,
        *,
        monitor_target_ratio: float,
        actual_fe_ratio: float,
        event_improvement_resume_metric: float,
        event_stagnation_onset_metric: float,
        event_rank_change_metric: float,
        event_elite_migration_metric: float,
        event_diversity_recovery_metric: float,
    ) -> SamplingDecision:
        """Compatibility wrapper for callers checking exactly one monitor point."""
        return self.decide_pending(
            monitor_target_ratios=(float(monitor_target_ratio),),
            actual_fe_ratio=actual_fe_ratio,
            event_improvement_resume_metric=event_improvement_resume_metric,
            event_stagnation_onset_metric=event_stagnation_onset_metric,
            event_rank_change_metric=event_rank_change_metric,
            event_elite_migration_metric=event_elite_migration_metric,
            event_diversity_recovery_metric=event_diversity_recovery_metric,
        )

    def _improvement_resume_event(self, value: float) -> bool:
        previous = self._previous_improvement_frequency
        if value <= EPS:
            self._armed["improvement_resume"] = True
        fired = bool(
            self._armed["improvement_resume"]
            and previous is not None
            and previous <= EPS
            and value > EPS
        )
        if fired:
            self._armed["improvement_resume"] = False
        self._previous_improvement_frequency = value
        return fired

    def _threshold_event(
        self,
        name: str,
        value: float,
        *,
        threshold: float,
        rearm_threshold: float,
    ) -> bool:
        if name == "stagnation_onset":
            if value <= rearm_threshold + EPS:
                self._armed[name] = True
        elif value < rearm_threshold:
            self._armed[name] = True
        fired = bool(self._armed[name] and value >= threshold - EPS)
        if fired:
            self._armed[name] = False
        return fired


def sampling_metrics(
    *,
    window_statistics: list[dict],
    native_update_history: list[dict],
    dimension: int,
    stagnation_span_ratio: float,
) -> dict[str, float]:
    """Compute the frozen causal event metrics from native-update windows."""
    windows = {str(item["suffix"]): item for item in window_statistics}
    short = windows["w02"]
    medium = windows["w05"]
    short_rows = [
        item for item in native_update_history if int(item["FE"]) >= int(short["anchor_FE"])
    ]
    improvements = sum(
        _strict_improvement(float(left["best_fitness"]), float(right["best_fitness"]))
        for left, right in zip(short_rows, short_rows[1:])
    )
    improvement_frequency = improvements / max(len(short_rows) - 1, 1)
    anchor_diversity = float(medium["anchor_diversity_mean_pairwise"])
    current_diversity = float(native_update_history[-1]["diversity_mean_pairwise"])
    diversity_recovery = max(
        0.0,
        (current_diversity - anchor_diversity) / max(anchor_diversity, EPS),
    )
    return {
        "event_improvement_resume_metric": float(improvement_frequency),
        "event_stagnation_onset_metric": float(stagnation_span_ratio),
        "event_rank_change_metric": float(medium["covariance_effective_rank_change"]),
        "event_elite_migration_metric": float(medium["elite_centroid_shift"]) / sqrt(int(dimension)),
        "event_diversity_recovery_metric": float(diversity_recovery),
    }


def _strict_improvement(previous_best: float, current_best: float) -> bool:
    threshold = EPS * max(1.0, abs(float(previous_best)))
    return float(previous_best) - float(current_best) > threshold
