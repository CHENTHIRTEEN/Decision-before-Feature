from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from benchmarks.mabbob import BBOB_TRAIN_FUNCTIONS


ANCHOR_TARGET = 8
PAIRWISE_TARGET = 8
TRIPLE_TARGET = 4
DENSE_TARGET = 4
FORMAL_TARGET = ANCHOR_TARGET + PAIRWISE_TARGET + TRIPLE_TARGET + DENSE_TARGET


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _score_entry(entry: dict[str, Any]) -> tuple[int, int, int, int, int]:
    components = entry.get("components", [])
    arity = len(components)
    has_validation = 1 if entry.get("is_val_component") else 0
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
        "geometric_decay": 3,
        "uniform_24": 4,
    }.get(profile, 5)
    candidate_id = int(entry["candidate_id"])
    return (has_validation, arity, profile_rank, candidate_id, 0)


def _group_entries(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[str(entry.get("bridge_type", "unknown"))].append(entry)
    for items in groups.values():
        items.sort(key=lambda item: _score_entry(item))
    return groups


def _select_first(groups: dict[str, list[dict[str, Any]]], bridge_type: str, count: int) -> list[dict[str, Any]]:
    candidates = [entry for entry in groups.get(bridge_type, []) if not entry.get("is_val_component")]
    if len(candidates) < count:
        raise ValueError(f"not enough {bridge_type} candidates without validation leakage")
    return candidates[:count]


def _selected_entry_payload(entry: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = dict(entry)
    payload["selection_reason"] = reason
    return payload


def select_formal_subset(manifest: dict[str, Any]) -> dict[str, Any]:
    selected_entries = list(manifest.get("selected", []))
    groups = _group_entries(selected_entries)

    anchors = _select_first(groups, "anchor", ANCHOR_TARGET)
    pairwise = _select_first(groups, "pairwise_bridge", PAIRWISE_TARGET)
    triple = _select_first(groups, "sparse_3way_bridge", TRIPLE_TARGET)
    dense = _select_first(groups, "dense_bridge", DENSE_TARGET)

    chosen: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for reason, bucket in [
        ("anchor_anchor", anchors),
        ("pairwise_bridge", pairwise),
        ("sparse_3way_bridge", triple),
        ("dense_bridge", dense),
    ]:
        for entry in bucket:
            candidate_id = int(entry["candidate_id"])
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            chosen.append(_selected_entry_payload(entry, reason))

    if len(chosen) != FORMAL_TARGET:
        raise RuntimeError(f"expected {FORMAL_TARGET} formal candidates, got {len(chosen)}")

    if any(entry.get("is_val_component") for entry in chosen):
        raise RuntimeError("formal subset contains validation-family leakage")

    chosen_ids = {int(entry["candidate_id"]) for entry in chosen}
    if not chosen_ids.issubset(set(range(1, 43))):
        raise RuntimeError("formal subset candidate IDs must be within the 42-entry manifest")

    return {
        "manifest_version": manifest.get("manifest_version", "unknown"),
        "generation_seed": manifest.get("generation_seed"),
        "dimension": manifest.get("dimension"),
        "formal_target": FORMAL_TARGET,
        "selected_candidate_ids": [int(entry["candidate_id"]) for entry in chosen],
        "selected": chosen,
        "selection_policy": {
            "anchor_target": ANCHOR_TARGET,
            "pairwise_target": PAIRWISE_TARGET,
            "triple_target": TRIPLE_TARGET,
            "dense_target": DENSE_TARGET,
            "train_components": list(BBOB_TRAIN_FUNCTIONS),
            "validation_component_guard": True,
        },
    }


def _write_yaml_config(selection: dict[str, Any], path: Path) -> None:
    selected_ids = selection["selected_candidate_ids"]
    config = {
        "suite": "mabbob",
        "split": "mabbob_formal",
        "function_family_protocol": "mabbob_affine_combination_v1",
        "functions": selected_ids,
        "instances": [1],
        "dimensions": [int(selection["dimension"])],
        "seeds": [1, 2],
        "FE_total_by_dimension": {int(selection["dimension"]): int(selection["dimension"]) * 1000},
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
        "manifest_path": "results/mabbob_diversity_pilot/mabbob_diversity_manifest.json",
        "selection_manifest_path": "results/mabbob_diversity_pilot/mabbob_formal_selection_manifest.json",
        "output": "results/phase1_mabbob/trajectories.parquet",
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a formal MA-BBOB subset from the diversity manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mabbob_diversity_pilot"))
    parser.add_argument("--write-config", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    selection = select_formal_subset(manifest)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = output_dir / "mabbob_formal_selection_manifest.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    config_path = None
    if args.write_config:
        config_path = output_dir / "phase1_mabbob_formal.yaml"
        _write_yaml_config(selection, config_path)

    print(
        json.dumps(
            {
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
