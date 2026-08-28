from __future__ import annotations

import numpy as np


def make_indexed_rng(
    *,
    seed: int,
    unit_number: int,
    stream_code: int,
    generation: int = 0,
    target: int = 0,
    event: int = 0,
) -> np.random.Generator:
    """Create a deterministic RNG from explicit integer experiment indices.

    ``unit_number`` identifies the independently generated unit, while the
    remaining fields identify a stream, generation, target and event within
    that unit.  Keeping all fields in the ``SeedSequence`` makes the RGI
    generator and its algorithm runs reproducible without relying on Python's
    process-dependent hash implementation or a shared global RNG.
    """
    sequence = np.random.SeedSequence(
        [
            int(seed),
            int(unit_number),
            int(stream_code),
            int(generation),
            int(target),
            int(event),
        ]
    )
    return np.random.default_rng(sequence)


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
