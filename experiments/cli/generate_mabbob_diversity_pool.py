from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmarks.mabbob import BBOB_VALIDATION_FUNCTIONS


@dataclass(frozen=True)
class ManifestEntry:
    candidate_id: int
    components: tuple[int, ...]
    weights: tuple[float, ...]
    instances: tuple[int, ...]
    scale_factors: tuple[float, ...]
    bridge_type: str
    xopt_mode: str
    xopt_seed: int
    xopt: tuple[float, ...]
    dimension: int
    strata_tag: str
    profile_tag: str
    variant_tag: str
    is_val_component: bool


DEFAULT_SCALES = (
    11.0, 17.5, 12.3, 12.6, 11.5, 15.3, 12.1, 15.3,
    15.2, 17.4, 13.4, 20.4, 12.9, 10.4, 12.3, 10.3,
    9.8, 10.6, 10.0, 14.7, 10.7, 10.8, 9.0, 12.1,
)

PAIRWISE_PROFILES: dict[str, tuple[float, float]] = {
    "balanced": (0.5, 0.5),
    "dominant_left": (0.8, 0.2),
    "dominant_right": (0.2, 0.8),
}
TRIPLE_PROFILES: dict[str, tuple[float, float, float]] = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "graduated": (0.6, 0.3, 0.1),
    "dominant": (0.7, 0.2, 0.1),
}
DENSE_PROFILES: dict[str, tuple[float, ...]] = {
    "uniform_24": tuple([1.0 / 24.0] * 24),
    "dominant_trace": tuple([0.9] + [0.1 / 23.0] * 23),
    "balanced_dense": tuple([1.0 / 24.0] * 24),
    "geometric_decay": tuple([0.5, 0.3, 0.2] + [1e-4] * 21),
}


def _xopt(candidate_id: int, dimension: int, mode: str) -> tuple[float, ...]:
    rng = np.random.default_rng(10000 + int(candidate_id))
    if mode == "center":
        return tuple(0.0 for _ in range(dimension))
    if mode == "boundary":
        values = rng.uniform(low=-4.9, high=4.9, size=dimension)
        if dimension > 0:
            values[0] = 4.95 if rng.random() < 0.5 else -4.95
        return tuple(float(value) for value in values)
    return tuple(float(value) for value in rng.uniform(low=-5.0, high=5.0, size=dimension))


def _instances(candidate_id: int, components: tuple[int, ...], variant: str) -> tuple[int, ...]:
    if variant == "all_one":
        return tuple(1 for _ in range(24))
    rng = np.random.default_rng(3000 + int(candidate_id))
    values = np.ones(24, dtype=int)
    if variant == "mixed":
        for component in components:
            values[component - 1] = int(rng.integers(1, 11))
    elif variant == "staggered":
        for offset, component in enumerate(components):
            values[component - 1] = 1 + ((candidate_id + offset) % 10)
    else:
        raise ValueError(f"unsupported instance variant: {variant}")
    return tuple(int(value) for value in values)


def _scales(mode: str, components: tuple[int, ...]) -> tuple[float, ...]:
    values = np.asarray(DEFAULT_SCALES, dtype=float).copy()
    if mode == "default":
        return tuple(float(value) for value in values)
    if mode == "dominant_expand":
        values[components[0] - 1] += 3.0
    elif mode == "dominant_contract":
        values[components[0] - 1] = max(1.0, values[components[0] - 1] - 3.0)
    elif mode == "flat":
        values[:] = float(np.mean(values))
    else:
        raise ValueError(f"unsupported scale mode: {mode}")
    return tuple(float(value) for value in values)


def _weights_vector(components: tuple[int, ...], profile: tuple[float, ...]) -> tuple[float, ...]:
    weights = np.zeros(24, dtype=float)
    for component, weight in zip(components, profile, strict=True):
        weights[component - 1] = float(weight)
    return tuple(float(value) for value in weights)


def _make_entry(
    candidate_id: int,
    dimension: int,
    *,
    components: tuple[int, ...],
    weights: tuple[float, ...],
    instances: tuple[int, ...],
    scale_factors: tuple[float, ...],
    bridge_type: str,
    xopt_mode: str,
    strata_tag: str,
    profile_tag: str,
    variant_tag: str,
) -> ManifestEntry:
    return ManifestEntry(
        candidate_id=candidate_id,
        components=components,
        weights=weights,
        instances=instances,
        scale_factors=scale_factors,
        bridge_type=bridge_type,
        xopt_mode=xopt_mode,
        xopt_seed=10000 + candidate_id,
        xopt=_xopt(candidate_id, dimension, xopt_mode),
        dimension=dimension,
        strata_tag=strata_tag,
        profile_tag=profile_tag,
        variant_tag=variant_tag,
        is_val_component=any(component in BBOB_VALIDATION_FUNCTIONS for component in components),
    )


def build_manifest(dimension: int = 10) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    candidate_id = 1

    # A: pure anchors, 10 definitions.
    anchor_specs = [
        (1, "uniform", "mixed", "default", "anchor"),
        (4, "center", "all_one", "default", "anchor"),
        (6, "uniform", "mixed", "default", "anchor"),
        (8, "uniform", "all_one", "default", "anchor"),
        (10, "uniform", "mixed", "dominant_expand", "anchor"),
        (12, "uniform", "all_one", "default", "anchor"),
        (15, "boundary", "mixed", "default", "anchor"),
        (20, "center", "all_one", "dominant_expand", "anchor"),
        (9, "uniform", "mixed", "default", "val_anchor"),
        (24, "boundary", "all_one", "default", "val_anchor"),
    ]
    for function, xopt_mode, instance_variant, scale_mode, variant_tag in anchor_specs:
        components = (function,)
        weights = tuple(1.0 if i == function - 1 else 0.0 for i in range(24))
        entries.append(_make_entry(
            candidate_id,
            dimension,
            components=components,
            weights=weights,
            instances=_instances(candidate_id, components, instance_variant),
            scale_factors=_scales(scale_mode, components),
            bridge_type="anchor",
            xopt_mode=xopt_mode,
            strata_tag="anchor",
            profile_tag=scale_mode,
            variant_tag=variant_tag,
        ))
        candidate_id += 1

    # B: pairwise bridges, 16 definitions.
    pairwise_specs = [
        ((2, 7), "C1×C2", "balanced", "uniform", "mixed", "pairwise_balanced"),
        ((2, 11), "C1×C3", "dominant_left", "boundary", "mixed", "pairwise_extreme"),
        ((2, 16), "C1×C4", "balanced", "uniform", "staggered", "pairwise_balanced"),
        ((2, 21), "C1×C5", "dominant_right", "boundary", "mixed", "pairwise_extreme"),
        ((7, 11), "C2×C3", "balanced", "uniform", "mixed", "pairwise_balanced"),
        ((7, 16), "C2×C4", "dominant_left", "boundary", "mixed", "pairwise_extreme"),
        ((7, 21), "C2×C5", "balanced", "uniform", "staggered", "pairwise_balanced"),
        ((11, 16), "C3×C4", "dominant_right", "boundary", "mixed", "pairwise_extreme"),
        ((11, 21), "C3×C5", "balanced", "uniform", "mixed", "pairwise_balanced"),
        ((16, 21), "C4×C5", "dominant_left", "boundary", "staggered", "pairwise_extreme"),
        ((2, 7), "C1×C2", "dominant_left", "uniform", "mixed", "pairwise_profile"),
        ((2, 11), "C1×C3", "balanced", "boundary", "mixed", "pairwise_profile"),
        ((2, 16), "C1×C4", "dominant_right", "uniform", "staggered", "pairwise_profile"),
        ((2, 21), "C1×C5", "balanced", "boundary", "mixed", "pairwise_profile"),
        ((7, 11), "C2×C3", "dominant_left", "uniform", "mixed", "pairwise_profile"),
        ((7, 21), "C2×C5", "dominant_right", "boundary", "staggered", "pairwise_profile"),
    ]
    for components, strata_tag, profile_tag, xopt_mode, instance_variant, variant_tag in pairwise_specs:
        entries.append(_make_entry(
            candidate_id,
            dimension,
            components=components,
            weights=_weights_vector(components, PAIRWISE_PROFILES[profile_tag]),
            instances=_instances(candidate_id, components, instance_variant),
            scale_factors=_scales("default", components),
            bridge_type="pairwise_bridge",
            xopt_mode=xopt_mode,
            strata_tag=strata_tag,
            profile_tag=profile_tag,
            variant_tag=variant_tag,
        ))
        candidate_id += 1

    # C: sparse 3-way bridges, 10 definitions.
    triple_specs = [
        ((2, 11, 21), "triple_2_11_21", "graduated", "uniform", "mixed", "triple_balanced"),
        ((7, 16, 21), "triple_7_16_21", "balanced", "center", "staggered", "triple_balanced"),
        ((4, 10, 23), "triple_4_10_23", "dominant", "uniform", "mixed", "triple_dominant"),
        ((1, 8, 20), "triple_1_8_20", "graduated", "boundary", "mixed", "triple_balanced"),
        ((6, 12, 18), "triple_6_12_18", "balanced", "uniform", "staggered", "triple_balanced"),
        ((2, 7, 11), "triple_2_7_11", "dominant", "center", "mixed", "triple_dominant"),
        ((2, 11, 21), "triple_2_11_21", "balanced", "boundary", "mixed", "triple_profile"),
        ((7, 16, 21), "triple_7_16_21", "graduated", "uniform", "staggered", "triple_profile"),
        ((4, 10, 23), "triple_4_10_23", "balanced", "center", "mixed", "triple_profile"),
        ((1, 8, 20), "triple_1_8_20", "dominant", "boundary", "mixed", "triple_profile"),
    ]
    for components, strata_tag, profile_tag, xopt_mode, instance_variant, variant_tag in triple_specs:
        entries.append(_make_entry(
            candidate_id,
            dimension,
            components=components,
            weights=_weights_vector(components, TRIPLE_PROFILES[profile_tag if profile_tag in TRIPLE_PROFILES else "balanced"]),
            instances=_instances(candidate_id, components, instance_variant),
            scale_factors=_scales("default", components),
            bridge_type="sparse_3way_bridge",
            xopt_mode=xopt_mode,
            strata_tag=strata_tag,
            profile_tag=profile_tag,
            variant_tag=variant_tag,
        ))
        candidate_id += 1

    # D: dense / composition-like, 6 definitions.
    dense_specs = [
        ((2, 11, 21), "dense_dominant_trace", "dominant_trace", "uniform", "mixed", "dominant_expand", "dense_dominant_trace"),
        ((7, 16, 21), "dense_balanced", "balanced_dense", "center", "mixed", "default", "dense_balanced"),
        ((4, 10, 23), "dense_geometric_decay", "geometric_decay", "boundary", "mixed", "dominant_contract", "dense_geometric_decay"),
        ((1, 8, 20), "dense_uniform_24", "uniform_24", "uniform", "mixed", "flat", "dense_uniform"),
        ((6, 12, 18), "dense_dominant_trace_2", "dominant_trace", "center", "mixed", "default", "dense_dominant_trace"),
        ((2, 7, 11), "dense_balanced_2", "balanced_dense", "boundary", "mixed", "dominant_expand", "dense_balanced"),
    ]
    for components, strata_tag, profile_tag, xopt_mode, instance_variant, scale_mode, variant_tag in dense_specs:
        entries.append(_make_entry(
            candidate_id,
            dimension,
            components=components,
            weights=DENSE_PROFILES[profile_tag],
            instances=_instances(candidate_id, components, instance_variant),
            scale_factors=_scales(scale_mode, components),
            bridge_type="dense_bridge",
            xopt_mode=xopt_mode,
            strata_tag=strata_tag,
            profile_tag=profile_tag,
            variant_tag=variant_tag,
        ))
        candidate_id += 1

    if candidate_id != 43:
        raise RuntimeError(f"expected 42 manifest entries, got {candidate_id - 1}")
    return entries


def _manifest_to_json(entries: list[ManifestEntry], *, dimension: int, version: str, seed: int) -> dict[str, Any]:
    return {
        "manifest_version": version,
        "generation_seed": seed,
        "dimension": dimension,
        "pool_size": len(entries),
        "train_components": [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23],
        "validation_components": [5, 9, 13, 14, 19, 24],
        "selected": [
            {
                "candidate_id": entry.candidate_id,
                "components": list(entry.components),
                "weights": list(entry.weights),
                "instances": list(entry.instances),
                "scale_factors": list(entry.scale_factors),
                "bridge_type": entry.bridge_type,
                "xopt_mode": entry.xopt_mode,
                "xopt_seed": entry.xopt_seed,
                "xopt": list(entry.xopt),
                "dimension": entry.dimension,
                "strata_tag": entry.strata_tag,
                "profile_tag": entry.profile_tag,
                "variant_tag": entry.variant_tag,
                "is_val_component": entry.is_val_component,
            }
            for entry in entries
        ],
    }


def _write_yaml_config(entries: list[ManifestEntry], output_path: Path, *, dimension: int) -> None:
    config = {
        "suite": "mabbob",
        "split": "mabbob_diversity_pilot",
        "function_family_protocol": "mabbob_affine_combination_v1",
        "functions": [entry.candidate_id for entry in entries],
        "instances": [1],
        "dimensions": [dimension],
        "seeds": [1, 2],
        "FE_total_by_dimension": {dimension: dimension * 1000},
        "population_size": 40,
        "efficacy_repetitions": 1,
        "efficacy_aggregation": "median",
        "failure_loss_cap": 1.0e20,
        "log10_gap_floor": 1.0e-12,
        "log10_gap_cap": 1.0e20,
        "success_gap_target": 1.0e-8,
        "action_timeout_seconds": 3600,
        "timing_replay_timeout_seconds": 3600,
        "policy_timeout_seconds": 3600,
        "first_hit_recording": "every_objective_evaluation",
        "timing_repetitions": 3,
        "timing_order_protocol": "cyclic_complete_path_v1",
        "algorithms": ["de", "pso", "cmaes", "shade"],
        "sampling_protocol": "phase1_dynamic_budget_event_v1",
        "boundary_handling": "reflect",
        "output": "results/diversity_pilot_mabbob/trajectories.parquet",
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a structured MA-BBOB diversity pilot manifest.")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mabbob_diversity_pilot"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default="mabbob_diversity_pilot_v1")
    args = parser.parse_args()

    entries = build_manifest(dimension=int(args.dimension))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "mabbob_diversity_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _manifest_to_json(entries, dimension=int(args.dimension), version=str(args.version), seed=int(args.seed)),
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    config_path = output_dir / "phase1_mabbob_diversity_pilot.yaml"
    _write_yaml_config(entries, config_path, dimension=int(args.dimension))

    preview = [
        {
            "candidate_id": entry.candidate_id,
            "components": list(entry.components),
            "bridge_type": entry.bridge_type,
            "strata_tag": entry.strata_tag,
            "profile_tag": entry.profile_tag,
            "variant_tag": entry.variant_tag,
        }
        for entry in entries[:5]
    ]
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "config_path": str(config_path),
                "pool_size": len(entries),
                "preview": preview,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
