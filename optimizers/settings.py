from __future__ import annotations

from dataclasses import dataclass

from trajectory.sampling import get_sampling_spec


DEFAULT_CHECKPOINT_RATIOS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)


@dataclass(frozen=True)
class OptimizerSettings:
    population_size: int = 40
    checkpoint_ratios: tuple[float, ...] = DEFAULT_CHECKPOINT_RATIOS
    sampling_protocol: str | None = None
    boundary_handling: str = "clip"

    def validate(self, fe_total: int) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if fe_total < self.population_size:
            raise ValueError("FE_total must be at least population_size")
        if fe_total % self.population_size != 0:
            raise ValueError("FE_total must be a multiple of population_size for complete population records")
        previous = 0.0
        for ratio in self.checkpoint_ratios:
            if ratio <= previous or ratio > 1.0:
                raise ValueError("checkpoint ratios must be strictly increasing and <= 1.0")
            previous = ratio
        if self.boundary_handling not in {"clip", "reflect"}:
            raise ValueError("boundary_handling must be clip or reflect")
        if self.sampling_protocol is not None:
            spec = get_sampling_spec(self.sampling_protocol)
            monitor_gaps = [
                later - earlier
                for earlier, later in zip(spec.monitor_ratios, spec.monitor_ratios[1:])
            ]
            if not monitor_gaps:
                raise ValueError("dynamic sampling protocol must define at least two monitor ratios")
            if self.population_size / fe_total > min(monitor_gaps) + 1e-12:
                raise ValueError(
                    "dynamic sampling requires each complete population update to span no more "
                    "than the minimum monitor-ratio gap"
                )
