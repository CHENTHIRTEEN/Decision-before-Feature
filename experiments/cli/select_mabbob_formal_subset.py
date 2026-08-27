from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from benchmarks.mabbob import BBOB_TRAIN_FUNCTIONS, BBOB_VALIDATION_FUNCTIONS


ANCHOR_TARGET = 8
PAIRWISE_TARGET = 8
TRIPLE_TARGET = 4
DENSE_TARGET = 4
FORMAL_TARGET = ANCHOR_TARGET + PAIRWISE_TARGET + TRIPLE_TARGET + DENSE_TARGET

VALIDATION_ANCHOR_TARGET = 6
VALIDATION_PAIRWISE_TARGET = 6
VALIDATION_TRIPLE_TARGET = 3
VALIDATION_DENSE_TARGET = 3
VALIDATION_TARGET = (
    VALIDATION_ANCHOR_TARGET + VALIDATION_PAIRWISE_TARGET
    + VALIDATION_TRIPLE_TARGET + VALIDATION_DENSE_TARGET
)


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _score_entry(entry: dict[str, Any], *, prefer_validation: bool) -> tuple[int, int, int, int, int]:
    components = entry.get("components", [])
    arity = len(components)
    has_validation = 1 if entry.get("is_val_component") else 0
    if prefer_validation:
        has_validation = 1 - has_validation
    profile = str(entry.get("profile_tag", ""))
    profile_rank = {
        "anchor": 0,
        "balanced": 0,
        "balanced_dense": 0,
        "graduated": 1,
        "dominant": 2,
        "dominant_left": 2,
        "dominant_right": 2,
        "dominant_trace": 2,
        "val_dominant_trace": 2,
        "geometric_decay": 3,
        "val_geometric_decay": 3,
        "uniform_24": 4,
        "train_uniform18": 4,
        "val_uniform6": 4,
    }.get(profile, 5)
    candidate_id = int(entry["candidate_id"])
    return (has_validation, arity, profile_rank, candidate_id, 0)


def _group_entries(
    entries: list[dict[str, Any]],
    *,
    prefer_validation: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[str(entry.get("bridge_type", "unknown"))].append(entry)
    for items in groups.values():
        items.sort(key=lambda item: _score_entry(item, prefer_validation=prefer_validation))
    return groups


def _select_first(
    groups: dict[str, list[dict[str, Any]]],
    bridge_type: str,
    count: int,
    *,
    validation_only: bool,
) -> list[dict[str, Any]]:
    if validation_only:
        candidates = [entry for entry in groups.get(bridge_type, []) if entry.get("is_val_component")]
        if len(candidates) < count:
            raise ValueError(f"not enough {bridge_type} candidates restricted to validation components")
    else:
        candidates = [entry for entry in groups.get(bridge_type, []) if not entry.get("is_val_component")]
        if len(candidates) < count:
            raise ValueError(f"not enough {bridge_type} candidates without validation leakage")
    return candidates[:count]


def _selected_entry_payload(entry: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = dict(entry)
    payload["selection_reason"] = reason
    return payload


def select_formal_subset(manifest: dict[str, Any], *, split: str = "formal") -> dict[str, Any]:
    if split not in {"formal", "validation"}:
        raise ValueError(f"unsupported split: {split}")
    validation_only = split == "validation"
    selected_entries = list(manifest.get("selected", []))
    groups = _group_entries(selected_entries, prefer_validation=validation_only)

    if validation_only:
        targets = {
            "anchor": VALIDATION_ANCHOR_TARGET,
            "pairwise_bridge": VALIDATION_PAIRWISE_TARGET,
            "sparse_3way_bridge": VALIDATION_TRIPLE_TARGET,
            "dense_bridge": VALIDATION_DENSE_TARGET,
        }
        target_total = VALIDATION_TARGET
    else:
        targets = {
            "anchor": ANCHOR_TARGET,
            "pairwise_bridge": PAIRWISE_TARGET,
            "sparse_3way_bridge": TRIPLE_TARGET,
            "dense_bridge": DENSE_TARGET,
        }
        target_total = FORMAL_TARGET

    buckets = {
        bridge_type: _select_first(
            groups, bridge_type, count, validation_only=validation_only
        )
        for bridge_type, count in targets.items()
    }

    chosen: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for bridge_type, bucket in buckets.items():
        for entry in bucket:
            candidate_id = int(entry["candidate_id"])
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            chosen.append(_selected_entry_payload(entry, bridge_type))

    if len(chosen) != target_total:
        raise RuntimeError(f"expected {target_total} {split} candidates, got {len(chosen)}")

    if validation_only:
        allowed = set(BBOB_VALIDATION_FUNCTIONS)
        violations = [
            int(entry["candidate_id"])
            for entry in chosen
            if not bool(entry.get("is_val_component"))
            or not set(int(c) for c in entry.get("components", [])).issubset(allowed)
        ]
        if violations:
            raise RuntimeError(f"{split} subset contains train-component leakage: {violations}")
    elif any(entry.get("is_val_component") for entry in chosen):
        raise RuntimeError("formal subset contains validation-family leakage")

    manifest_ids = {int(entry["candidate_id"]) for entry in selected_entries}
    chosen_ids = {int(entry["candidate_id"]) for entry in chosen}
    if not chosen_ids.issubset(manifest_ids):
        raise RuntimeError("subset candidate IDs must come from the pool manifest")

    policy: dict[str, Any] = {
        "anchor_target": targets["anchor"],
        "pairwise_target": targets["pairwise_bridge"],
        "triple_target": targets["sparse_3way_bridge"],
        "dense_target": targets["dense_bridge"],
    }
    if validation_only:
        policy["component_scope"] = "validation"
        policy["validation_components"] = list(BBOB_VALIDATION_FUNCTIONS)
        policy["train_component_guard"] = True
    else:
        policy["train_components"] = list(BBOB_TRAIN_FUNCTIONS)
        policy["validation_component_guard"] = True

    return {
        "manifest_version": manifest.get("manifest_version", "unknown"),
        "generation_seed": manifest.get("generation_seed"),
        "dimension": manifest.get("dimension"),
        "split": f"mabbob_{split}",
        "formal_target": target_total,
        "selected_candidate_ids": [int(entry["candidate_id"]) for entry in chosen],
        "selected": chosen,
        "selection_policy": policy,
    }


MABBOB_DIMENSIONS = (10, 20, 40)
MABBOB_SEEDS = (1, 2, 3, 4, 5)


def _write_yaml_config(selection: dict[str, Any], path: Path, *, split: str) -> None:
    selected_ids = selection["selected_candidate_ids"]
    config = {
        "suite": "mabbob",
        "split": f"mabbob_{split}",
        "function_family_protocol": "mabbob_affine_combination_v1",
        "functions": selected_ids,
        "instances": [1],
        "dimensions": list(MABBOB_DIMENSIONS),
        "seeds": list(MABBOB_SEEDS),
        "FE_total_by_dimension": {dimension: dimension * 1000 for dimension in MABBOB_DIMENSIONS},
        "population_size": 40,
        "efficacy_repetitions": 3,
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
        "output": (
            "results/phase1_mabbob_validation/trajectories.parquet"
            if split == "validation"
            else "results/phase1_mabbob/trajectories.parquet"
        ),
    }
    if split == "validation":
        config["manifest_path"] = "results/mabbob_diversity_pilot/mabbob_validation_pool_manifest.json"
        config["selection_manifest_path"] = (
            "results/mabbob_diversity_pilot/mabbob_validation_selection_manifest.json"
        )
    else:
        config["manifest_path"] = "results/mabbob_diversity_pilot/mabbob_diversity_manifest.json"
        config["selection_manifest_path"] = (
            "results/mabbob_diversity_pilot/mabbob_formal_selection_manifest.json"
        )
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a formal MA-BBOB subset from a diversity pool manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mabbob_diversity_pilot"))
    parser.add_argument(
        "--split",
        choices=("formal", "validation"),
        default="formal",
        help="formal: train-side subset excluding validation components; "
        "validation: validation-side subset whose components must all come from "
        "the BBOB validation functions.",
    )
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    selection = select_formal_subset(manifest, split=args.split)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "validation":
        selection_path = output_dir / "mabbob_validation_selection_manifest.json"
        config_filename = "phase1_mabbob_validation.yaml"
    else:
        selection_path = output_dir / "mabbob_formal_selection_manifest.json"
        config_filename = "phase1_mabbob_formal.yaml"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    config_path = None
    if args.write_config:
        config_path = output_dir / config_filename
        _write_yaml_config(selection, config_path, split=args.split)

    print(
        json.dumps(
            {
                "split": args.split,
                "selection_manifest_path": str(selection_path),
                "config_path": str(config_path) if config_path is not None else None,
                "selected_candidate_ids": selection["selected_candidate_ids"],
                "count": len(selection["selected_candidate_ids"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
