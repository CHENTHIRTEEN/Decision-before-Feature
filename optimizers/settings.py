from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CHECKPOINT_RATIOS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)


@dataclass(frozen=True)
class OptimizerSettings:
    population_size: int = 40
    checkpoint_ratios: tuple[float, ...] = DEFAULT_CHECKPOINT_RATIOS

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

