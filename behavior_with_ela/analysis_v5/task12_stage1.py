"""Task 12 Stage 1: natural-run strength screening over the candidate pool.

Candidates (frozen before any outcome): pso, lbestpso, de (canonical
DE/rand/1/bin, F=0.5/CR=0.9), shade, lshade, ga, cso. CMA-ES is excluded from
the balanced screening pool and reserved as the dominance control.

Domain: BBOB train (18 x 3 instances) + selected MA-BBOB (24 x 1), seeds 1-5.
Each candidate runs naturally from FE=0 to 10000; log10 gap is recorded at
every 1000-FE mark plus 10000. A deterministic 10% subset of (problem,
candidate) pairs is repeated R=3 with forked RNG streams (offset native
streams) to calibrate per-FE practical noise.
"""
from __future__ import annotations

import argparse
import resource
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from behavior_with_ela.action_dataset import CONTINUATION_REPETITION_STREAM_OFFSET
from behavior_with_ela.protocol import (
    load_experiment_config,
    make_experiment_problem,
    suite_code,
)
from optimizers import (
    OptimizerSettings,
    advance_optimizer_state,
    initialize_optimizer_state,
)
from optimizers.state import NATIVE_STREAMS

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/behavior_with_ela_train.yaml"
OUTPUT = ROOT / "behavior_with_ela/results/portfolio_screening/task12/stage1"
CANDIDATES = ("pso", "lbestpso", "de", "shade", "lshade", "ga", "cso")
FE_MARKS = tuple(range(1000, 10001, 1000))
SEEDS = (1, 2, 3, 4, 5)
REPETITION_FRACTION = 0.10
REPETITION_STREAM = 2026083010


def _log10_loss(gap: float, config) -> float:
    return float(
        np.log10(np.clip(gap, config.log10_gap_floor, config.log10_gap_cap))
    )


def _sampled_for_repetition(*, suite_name: str, function: int, instance: int, seed: int, candidate: str) -> bool:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                REPETITION_STREAM,
                suite_code(suite_name),
                int(function),
                int(instance),
                int(seed),
                NATIVE_STREAMS[candidate],
            ]
        )
    )
    return bool(rng.random() < REPETITION_FRACTION)


def _run_natural(
    *,
    config,
    problem,
    algorithm: str,
    seed: int,
    replicate_id: int,
) -> tuple[dict[int, float], int, float]:
    settings = OptimizerSettings(
        population_size=config.population_size,
        boundary_handling=config.boundary_handling,
    )
    evaluations = 0

    def observe_evaluation(point: np.ndarray, value: float) -> None:
        nonlocal evaluations
        evaluations += 1

    started = perf_counter()
    state = initialize_optimizer_state(
        algorithm=algorithm,
        problem=problem,
        seed=seed,
        settings=settings,
        on_evaluation=observe_evaluation,
    )
    if replicate_id > 0:
        # formal repetition semantics: fork the native RNG stream per replicate
        from optimizers.seeding import make_rng
        from optimizers.state import _restore_rng

        rng = make_rng(
            seed,
            NATIVE_STREAMS[algorithm] + CONTINUATION_REPETITION_STREAM_OFFSET * replicate_id,
            suite_code=problem.suite_code,
            function=problem.function_number,
            instance=problem.instance_number,
            dimension=problem.dimension,
        )
        import copy

        state.rng_state = copy.deepcopy(rng.bit_generator.state)
    marks = list(FE_MARKS)
    losses: dict[int, float] = {}
    while evaluations < config.fe_total:
        advance_optimizer_state(
            state=state,
            problem=problem,
            fe_budget=min(config.population_size, config.fe_total - evaluations),
            on_evaluation=observe_evaluation,
        )
        while marks and evaluations >= marks[0]:
            mark = marks.pop(0)
            gap = float(
                min(
                    max(state.best_fitness - float(problem.reference_value), 0.0),
                    config.failure_loss_cap,
                )
            )
            losses[mark] = _log10_loss(gap, config)
    return losses, evaluations, perf_counter() - started


def _collect_unit(job: dict) -> dict:
    config = load_experiment_config(CONFIG)
    suites = {suite.split: suite for suite in config.suites}
    suite = suites[job["split"]]
    problem = make_experiment_problem(
        suite,
        function=job["function"],
        instance=job["instance"],
        dimension=config.dimension,
        boundary_handling=config.boundary_handling,
    )
    rows = []
    ledgers = []
    for seed in job["seeds"]:
        for candidate in CANDIDATES:
            sampled = _sampled_for_repetition(
                suite_name=suite.suite,
                function=job["function"],
                instance=job["instance"],
                seed=seed,
                candidate=candidate,
            )
            replicates = range(3) if sampled else range(1)
            for replicate_id in replicates:
                losses, used_fe, elapsed = _run_natural(
                    config=config,
                    problem=problem,
                    algorithm=candidate,
                    seed=seed,
                    replicate_id=replicate_id,
                )
                for fe, loss in losses.items():
                    rows.append(
                        {
                            "split": suite.split,
                            "suite": suite.suite,
                            "problem_id": problem.problem_id,
                            "function_id": problem.function_id,
                            "family": problem.family,
                            "cv_group_id": problem.cv_group_id,
                            "instance": int(job["instance"]),
                            "seed": int(seed),
                            "candidate": candidate,
                            "replicate_id": int(replicate_id),
                            "FE": int(fe),
                            "log10_gap": float(loss),
                        }
                    )
                ledgers.append(
                    {
                        "suite": suite.suite,
                        "function": int(job["function"]),
                        "instance": int(job["instance"]),
                        "seed": int(seed),
                        "candidate": candidate,
                        "replicate_id": int(replicate_id),
                        "fe_used": int(used_fe),
                        "wall_seconds": elapsed,
                    }
                )
    problem.close()
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "job": job,
        "rows": rows,
        "ledgers": ledgers,
        "max_rss_mb": rss_mb,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--only-suite", action="append", default=None)
    args = parser.parse_args()
    config = load_experiment_config(CONFIG)
    output = OUTPUT / "shards"
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for suite in config.suites:
        if args.only_suite and suite.split not in set(args.only_suite):
            continue
        for function in suite.functions:
            for instance in suite.instances:
                jobs.append(
                    {
                        "split": suite.split,
                        "suite": suite.suite,
                        "function": int(function),
                        "instance": int(instance),
                        "seeds": tuple(SEEDS[: args.seeds]),
                    }
                )
    print(f"[task12_stage1] units: {len(jobs)}", flush=True)
    total_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_collect_unit, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            job = result["job"]
            shard = output / f"{job['split']}_{job['suite']}_f{job['function']:03d}_i{job['instance']}"
            shard.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result["rows"]).to_parquet(shard / "runs.parquet", index=False)
            ledger = pd.DataFrame(result["ledgers"])
            ledger["max_rss_mb"] = result["max_rss_mb"]
            ledger.to_parquet(shard / "ledger.parquet", index=False)
            total_rows += len(result["rows"])
            if index % 10 == 0 or index == len(jobs):
                print(
                    f"[task12_stage1] {index}/{len(jobs)} done (rows so far {total_rows})",
                    flush=True,
                )
    print(f"[task12_stage1] done: rows={total_rows}", flush=True)


if __name__ == "__main__":
    main()
