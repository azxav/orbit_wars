from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .schema import AMOUNT_BIN_NAMES, NOOP_TARGET_SLOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def validate_dataset(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    metadata = json.load(open(out_dir / "metadata.json", "r", encoding="utf-8")) if (out_dir / "metadata.json").exists() else {}

    launch_count = 0
    methods: Counter[str] = Counter()
    launch_valid = 0
    contact = 0
    angular = 0
    launch_path = out_dir / "launch_rows.jsonl"
    if launch_path.exists():
        for row in iter_jsonl(launch_path):
            launch_count += 1
            methods[str(row.get("target_inference_method"))] += 1
            launch_valid += int(bool(row.get("valid_source")))
            contact += int(row.get("target_inference_method") == "first_contact")
            angular += int(row.get("target_inference_method") == "angular_nearest")

    source_count = 0
    amount_bins: Counter[str] = Counter()
    geometry_viable = 0
    ambiguous = 0
    drop_v1 = 0
    positives = 0
    noops = 0
    source_rows_for_checks: list[dict[str, Any]] = []
    source_path = out_dir / "source_turn_rows.jsonl"
    if source_path.exists():
        for row in iter_jsonl(source_path):
            source_rows_for_checks.append(row)
            source_count += 1
            amount_name = row.get("amount_bin_name", AMOUNT_BIN_NAMES[int(row.get("amount_bin_label", 0))] if "amount_bin_label" in row else row.get("amount_bin", 0))
            amount_bins[str(amount_name)] += 1
            geometry_viable += int(bool(row.get("geometry_viable")))
            ambiguous += int(bool(row.get("ambiguous_multi_launch")))
            drop_v1 += int(bool(row.get("drop_for_v1_bc")))
            is_positive = int(row.get("target_slot_label", NOOP_TARGET_SLOT)) != NOOP_TARGET_SLOT
            positives += int(is_positive)
            noops += int(not is_positive)

    arr_info = {}
    dense_arrays = None
    npz_path = out_dir / "dense_bc_arrays.npz"
    if npz_path.exists():
        dense_arrays = np.load(npz_path, allow_pickle=False)
        arr_info = {k: list(v.shape) for k, v in dense_arrays.items() if hasattr(v, "shape")}

    invalid_bc_rows = [
        row
        for row in source_rows_for_checks
        if bool(row.get("drop_for_v1_bc", False))
        or not bool(row.get("geometry_viable", True))
        or bool(row.get("ambiguous_multi_launch", False))
        or str(row.get("target_inference_method", "") or "") == "angular_nearest"
    ]
    num_invalid_bc_rows = len(invalid_bc_rows)

    if dense_arrays is not None and "target_viability_mask" in dense_arrays and "amount_viability_mask" in dense_arrays:
        state_rows = read_jsonl(out_dir / "state_rows.jsonl") if (out_dir / "state_rows.jsonl").exists() else []
        obs_uid_to_dense = {str(row.get("obs_uid")): i for i, row in enumerate(state_rows)}
        target_viability_mask = dense_arrays["target_viability_mask"]
        amount_viability_mask = dense_arrays["amount_viability_mask"]
        bad_targets = 0
        bad_amounts = 0
        for row in source_rows_for_checks:
            dense_idx = obs_uid_to_dense.get(str(row.get("obs_uid")))
            if dense_idx is None:
                continue
            source_slot = int(row.get("source_slot", -1))
            target_label = int(row.get("target_slot_label", NOOP_TARGET_SLOT))
            amount_label = int(row.get("amount_bin_label", 0))
            if source_slot < 0 or source_slot >= target_viability_mask.shape[1]:
                bad_targets += 1
                continue
            if not bool(target_viability_mask[dense_idx, source_slot, target_label]):
                bad_targets += 1
                continue
            if not bool(amount_viability_mask[dense_idx, source_slot, target_label, amount_label]):
                bad_amounts += 1
        if bad_targets:
            raise RuntimeError(f"BC source rows contain {bad_targets} target labels outside target_viability_mask")
        if bad_amounts:
            raise RuntimeError(f"BC source rows contain {bad_amounts} amount labels outside amount_viability_mask")

    denom_launch = max(launch_count, 1)
    denom_source = max(source_count, 1)
    report = {
        "counts": {
            "launch_rows": launch_count,
            "source_turn_rows": source_count,
            "valid_launches": launch_valid,
            "positive_source_turns": positives,
            "noop_source_turns": noops,
            "ambiguous_multi_launch_sources": ambiguous,
            "drop_for_v1_bc_rows": drop_v1,
            "bc_invalid_source_rows": num_invalid_bc_rows,
        },
        "rates": {
            "valid_launch_rate": launch_valid / denom_launch,
            "first_contact_launch_rate": contact / denom_launch,
            "angular_fallback_launch_rate": angular / denom_launch,
            "geometry_viable_source_rate": geometry_viable / denom_source,
            "noop_source_rate": noops / denom_source,
            "drop_for_v1_bc_rate": drop_v1 / denom_source,
            "bc_invalid_source_rate": num_invalid_bc_rows / denom_source,
        },
        "target_inference_methods": dict(methods),
        "amount_bin_distribution_source_turns": dict(amount_bins),
        "dense_array_shapes": arr_info,
        "metadata_stats": metadata.get("stats", {}),
        "decision_checks": {
            "target_head_dim": metadata.get("action_space", {}).get("target_slots"),
            "noop_target_slot": metadata.get("action_space", {}).get("noop_target_slot"),
            "amount_bins": metadata.get("action_space", {}).get("amount_bins"),
            "angle_policy": metadata.get("action_space", {}).get("angle_policy"),
            "primary_train_rows": "source_turn_rows.jsonl",
            "dense_bc_arrays": "dense_bc_arrays.npz",
        },
    }
    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    lines = [
        "# Orbit Wars Dataset Validation Report",
        "",
        "## Counts",
    ]
    for k, v in report["counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Rates"]
    for k, v in report["rates"].items():
        lines.append(f"- `{k}`: {v:.4f}")
    lines += ["", "## Training contract checks"]
    for k, v in report["decision_checks"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Notes", "- Rows with `drop_for_v1_bc=true` should be excluded from the first BC run, but kept for later multi-launch modeling.", "- `target_inference_method=first_contact` means the label came from exact geometry first-hit simulation. `target_inference_method=angular_nearest` is only the fallback for sun, bounds, or no-contact launches."]
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    report = validate_dataset(args.out_dir)
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    print(json.dumps(report["rates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
