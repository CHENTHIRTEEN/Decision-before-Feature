from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from benchmarks.mabbob import BBOB_VALIDATION_FUNCTIONS


@dataclass
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
# Dense weight vectors are scoped: the support stays strictly inside the pool's
# component universe (18 train functions or 6 validation functions) and is
# anchored on the entry's declared components. The earlier fixed 24-slot
# vectors spread weight over all BBOB functions — including the opposite
# split's functions — and anchored dominant/decay profiles on absolute slots
# instead of the entry components.
TRAIN_COMPONENTS = (
    1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 15, 16, 17, 18, 20, 21, 22, 23,
)
VALIDATION_COMPONENTS = tuple(BBOB_VALIDATION_FUNCTIONS)
TRAIN_DENSE_PROFILE_TAGS = ("train_uniform18", "balanced_dense", "dominant_trace", "geometric_decay")
VALIDATION_DENSE_PROFILE_TAGS = ("val_uniform6", "val_dominant_trace", "val_geometric_decay")


def _dense_weights(
    profile_tag: str,
    components: tuple[int, ...],
    scope: tuple[int, ...],
) -> tuple[float, ...]:
    weights = np.zeros(24, dtype=float)
    slots = [component - 1 for component in scope]
    if profile_tag in ("train_uniform18", "val_uniform6"):
        for slot in slots:
            weights[slot] = 1.0 / len(slots)
    elif profile_tag in ("dominant_trace", "val_dominant_trace"):
        dominant = int(components[0])
        weights[dominant - 1] = 0.9
        for slot in slots:
            if slot != dominant - 1:
                weights[slot] = 0.1 / (len(slots) - 1)
    elif profile_tag in ("geometric_decay", "val_geometric_decay"):
        for component, weight in zip(components[:3], (0.5, 0.3, 0.2), strict=True):
            weights[component - 1] = weight
        for slot in slots:
            if weights[slot] == 0.0:
                weights[slot] = 1e-4
    elif profile_tag == "balanced_dense":
        for slot in slots:
            weights[slot] = 1.0 / len(slots)
    else:
        raise ValueError(f"unsupported dense profile: {profile_tag}")
    return tuple(float(value) for value in weights)


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

    # D: dense / composition-like, 6 definitions with train-scoped support.
    dense_specs = [
        ((2, 11, 21), "dense_dominant_trace", "dominant_trace", "uniform", "mixed", "dominant_expand", "dense_dominant_trace"),
        ((7, 16, 21), "dense_balanced", "balanced_dense", "center", "mixed", "default", "dense_balanced"),
        ((4, 10, 23), "dense_geometric_decay", "geometric_decay", "boundary", "mixed", "dominant_contract", "dense_geometric_decay"),
        ((1, 8, 20), "dense_uniform_18", "train_uniform18", "uniform", "mixed", "flat", "dense_uniform"),
        ((6, 12, 18), "dense_dominant_trace_2", "dominant_trace", "center", "mixed", "default", "dense_dominant_trace"),
        ((2, 7, 11), "dense_balanced_2", "balanced_dense", "boundary", "mixed", "dominant_expand", "dense_balanced"),
    ]
    for components, strata_tag, profile_tag, xopt_mode, instance_variant, scale_mode, variant_tag in dense_specs:
        entries.append(_make_entry(
            candidate_id,
            dimension,
            components=components,
            weights=_dense_weights(profile_tag, components, TRAIN_COMPONENTS),
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
    train_scope = set(TRAIN_COMPONENTS)
    for entry in entries:
        # Entries 9/10 are deliberate validation anchors kept in the pool for
        # audit; every other entry must keep its weight support inside the
        # train components.
        if entry.is_val_component:
            continue
        support = {index + 1 for index, value in enumerate(entry.weights) if value > 0.0}
        if not support.issubset(train_scope):
            raise RuntimeError(
                f"train pool entry {entry.candidate_id} has weight support outside the train components"
            )
    return entries


def build_validation_manifest(dimension: int = 10) -> list[ManifestEntry]:
    """Pool whose component support is restricted to the six BBOB validation
    functions (F5/F9/F13/F14/F19/F24); candidate IDs start at 201."""
    entries: list[ManifestEntry] = []
    candidate_id = 201

    # A: pure anchors on each validation function.
    anchor_specs = [
        (5, "uniform", "mixed", "default"),
        (9, "center", "all_one", "default"),
        (13, "uniform", "all_one", "default"),
        (14, "boundary", "mixed", "default"),
        (19, "uniform", "mixed", "dominant_expand"),
        (24, "center", "all_one", "default"),
    ]
    for function, xopt_mode, instance_variant, scale_mode in anchor_specs:
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
            strata_tag="val_anchor",
            profile_tag=scale_mode,
            variant_tag="val_anchor",
        ))
        candidate_id += 1

    # B: cross-category pairwise bridges, 8 definitions.
    pairwise_specs = [
        ((5, 9), "C1×C2", "balanced", "uniform", "mixed"),
        ((5, 13), "C1×C3", "dominant_left", "boundary", "mixed"),
        ((5, 19), "C1×C4", "balanced", "uniform", "staggered"),
        ((5, 24), "C1×C5", "dominant_right", "boundary", "mixed"),
        ((9, 13), "C2×C3", "balanced", "uniform", "mixed"),
        ((9, 24), "C2×C5", "dominant_left", "boundary", "staggered"),
        ((14, 19), "C3×C4", "balanced", "uniform", "mixed"),
        ((13, 24), "C3×C5", "dominant_right", "boundary", "mixed"),
    ]
    for components, strata_tag, profile_tag, xopt_mode, instance_variant in pairwise_specs:
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
            variant_tag="val_pairwise",
        ))
        candidate_id += 1

    # C: sparse 3-way bridges, 5 definitions.
    triple_specs = [
        ((5, 9, 13), "triple_5_9_13", "graduated", "uniform", "mixed"),
        ((5, 19, 24), "triple_5_19_24", "balanced", "center", "staggered"),
        ((9, 13, 19), "triple_9_13_19", "dominant", "uniform", "mixed"),
        ((13, 14, 24), "triple_13_14_24", "balanced", "boundary", "mixed"),
        ((5, 14, 24), "triple_5_14_24", "graduated", "center", "staggered"),
    ]
    for components, strata_tag, profile_tag, xopt_mode, instance_variant in triple_specs:
        entries.append(_make_entry(
            candidate_id,
            dimension,
            components=components,
            weights=_weights_vector(components, TRIPLE_PROFILES[profile_tag]),
            instances=_instances(candidate_id, components, instance_variant),
            scale_factors=_scales("default", components),
            bridge_type="sparse_3way_bridge",
            xopt_mode=xopt_mode,
            strata_tag=strata_tag,
            profile_tag=profile_tag,
            variant_tag="val_triple",
        ))
        candidate_id += 1

    # D: dense bridges with weight support restricted to the validation slots.
    dense_specs = [
        ((5, 9, 13, 14, 19, 24), "val_dense_uniform", "val_uniform6", "uniform", "mixed", "flat"),
        ((19, 5, 9), "val_dense_dominant", "val_dominant_trace", "boundary", "mixed", "dominant_expand"),
        ((5, 19, 24), "val_dense_decay", "val_geometric_decay", "uniform", "mixed", "default"),
        ((9, 13, 19), "val_dense_dominant_2", "val_dominant_trace", "boundary", "staggered", "dominant_contract"),
    ]
    for components, strata_tag, profile_tag, xopt_mode, instance_variant, scale_mode in dense_specs:
        entries.append(_make_entry(
            candidate_id,
            dimension,
            components=components,
            weights=_dense_weights(profile_tag, components, VALIDATION_COMPONENTS),
            instances=_instances(candidate_id, components, instance_variant),
            scale_factors=_scales(scale_mode, components),
            bridge_type="dense_bridge",
            xopt_mode=xopt_mode,
            strata_tag=strata_tag,
            profile_tag=profile_tag,
            variant_tag="val_dense",
        ))
        candidate_id += 1

    if candidate_id != 224:
        raise RuntimeError(f"expected 23 validation manifest entries, got {candidate_id - 201}")
    for entry in entries:
        if not set(entry.components).issubset(set(VALIDATION_COMPONENTS)):
            raise RuntimeError(
                f"validation pool entry {entry.candidate_id} leaves the validation component set"
            )
        if not entry.is_val_component:
            raise RuntimeError(f"validation pool entry {entry.candidate_id} is not flagged val")
    return entries


def _manifest_to_json(
    entries: list[ManifestEntry],
    *,
    dimension: int,
    version: str,
    seed: int,
    component_scope: str = "train",
) -> dict[str, Any]:
    return {
        "manifest_version": version,
        "generation_seed": seed,
        "dimension": dimension,
        "pool_size": len(entries),
        "component_scope": component_scope,
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
    parser = argparse.ArgumentParser(description="Generate a structured MA-BBOB diversity pool manifest.")
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mabbob_diversity_pilot"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--pool",
        choices=("train", "validation"),
        default="train",
        help="train: the 42-entry train-component pool; validation: the 23-entry "
        "validation-component pool with candidate IDs from 201.",
    )
    args = parser.parse_args()

    if args.pool == "train":
        entries = build_manifest(dimension=int(args.dimension))
        version = args.version or "mabbob_diversity_pilot_v1"
        component_scope = "train"
        manifest_filename = "mabbob_diversity_manifest.json"
        pilot_config_name = "phase1_mabbob_diversity_pilot.yaml"
    else:
        entries = build_validation_manifest(dimension=int(args.dimension))
        version = args.version or "mabbob_validation_pool_v1"
        component_scope = "validation"
        manifest_filename = "mabbob_validation_pool_manifest.json"
        pilot_config_name = None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / manifest_filename
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _manifest_to_json(
                entries,
                dimension=int(args.dimension),
                version=version,
                seed=int(args.seed),
                component_scope=component_scope,
            ),
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    config_path = None
    if pilot_config_name is not None:
        config_path = output_dir / pilot_config_name
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
                "pool": component_scope,
                "manifest_path": str(manifest_path),
                "config_path": str(config_path) if config_path is not None else None,
                "pool_size": len(entries),
                "preview": preview,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
