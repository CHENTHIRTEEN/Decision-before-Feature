from __future__ import annotations

import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize

from benchmarks.core import Problem
from optimizers.seeding import derive_seed
from optimizers.settings import OptimizerSettings
from trajectory.recorder import TrajectoryRecorder


class _PymooProblem(PymooProblem):
    def __init__(self, problem: Problem):
        super().__init__(
            n_var=problem.dimension,
            n_obj=1,
            xl=problem.lower_bounds,
            xu=problem.upper_bounds,
        )
        self._source = problem

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        out["F"] = self._source.evaluate(x).reshape(-1, 1)


class _TrajectoryCallback(Callback):
    def __init__(
        self,
        *,
        problem: Problem,
        algorithm_name: str,
        seed: int,
        fe_total: int,
        recorder: TrajectoryRecorder,
    ):
        super().__init__()
        self.problem = problem
        self.algorithm_name = algorithm_name
        self.seed = seed
        self.fe_total = fe_total
        self.recorder = recorder
        self.best_fitness = np.inf

    def notify(self, algorithm) -> None:
        population = np.asarray(algorithm.pop.get("X"), dtype=float)
        fitness = np.asarray(algorithm.pop.get("F"), dtype=float).reshape(-1)
        current_best = float(np.min(fitness))
        self.best_fitness = min(self.best_fitness, current_best)
        self.recorder.observe(
            problem=self.problem,
            algorithm=self.algorithm_name,
            seed=self.seed,
            fe=int(algorithm.evaluator.n_eval),
            fe_total=self.fe_total,
            population=population,
            fitness=fitness,
            best_fitness=self.best_fitness,
        )


def run_pymoo_optimizer(
    *,
    algorithm_name: str,
    problem: Problem,
    seed: int,
    fe_total: int,
    settings: OptimizerSettings,
) -> list:
    settings.validate(fe_total)
    recorder = TrajectoryRecorder(settings.checkpoint_ratios)
    lower = problem.lower_bounds
    upper = problem.upper_bounds

    if algorithm_name == "de":
        algorithm = DE(pop_size=settings.population_size)
        stream_code = 101
    elif algorithm_name == "pso":
        algorithm = PSO(pop_size=settings.population_size)
        stream_code = 202
    else:
        raise ValueError(f"unsupported pymoo optimizer: {algorithm_name}")

    callback = _TrajectoryCallback(
        problem=problem,
        algorithm_name=algorithm_name,
        seed=seed,
        fe_total=fe_total,
        recorder=recorder,
    )
    minimize(
        _PymooProblem(problem),
        algorithm,
        ("n_eval", fe_total),
        seed=derive_seed(seed, stream_code),
        callback=callback,
        verbose=False,
        save_history=False,
    )
    return recorder.records
