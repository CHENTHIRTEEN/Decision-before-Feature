from __future__ import annotations

import numpy as np


def derive_seed(
    seed: int,
    stream_code: int,
    *,
    suite_code: int,
    function: int,
    instance: int,
    dimension: int,
) -> int:
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(stream_code),
            int(suite_code),
            int(function),
            int(instance),
            int(dimension),
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def make_rng(
    seed: int,
    stream_code: int,
    *,
    suite_code: int,
    function: int,
    instance: int,
    dimension: int,
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(stream_code),
                int(suite_code),
                int(function),
                int(instance),
                int(dimension),
            ]
        )
    )


def make_event_rng(
    *,
    seed: int,
    stream_code: int,
    suite_code: int,
    function: int,
    instance: int,
    dimension: int,
    generation: int,
    event: int,
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(stream_code),
            int(suite_code),
            int(function),
            int(instance),
            int(dimension),
            int(generation),
            int(event),
        ]
    )
    return np.random.default_rng(sequence)
