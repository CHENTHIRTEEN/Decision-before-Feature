from __future__ import annotations

import numpy as np


def derive_seed(seed: int, stream_code: int) -> int:
    sequence = np.random.SeedSequence([int(seed), int(stream_code)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def make_rng(seed: int, stream_code: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream_code)]))

