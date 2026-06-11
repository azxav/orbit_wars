from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .features import PAIR_FEATURE_NAMES
from .schema import AMOUNT_BIN_NAMES, NOOP_TARGET_SLOT
from .source_turn_store import DATASET_FORMAT, SAMPLE_SPECS, STATE_SPECS, SourceTurnDatasetReader


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _shape_map(arrays: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {k: list(v.shape) for k, v in arrays.items() if hasattr(v, "shape")}


def validate_dataset(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    if (out_dir / "dense_bc_arrays.npz").exists():
        raise RuntimeError("Old dense_bc_arrays.npz dataset detected. Rebuild dataset with source_turn_memmap_v1.")
    reader = SourceTurnDatasetReader(out_dir)
    metadata = reader.metadata
    if metadata.get("dataset_format") != DATASET_FORMAT:
        raise RuntimeError("Dataset is not source_turn_memmap_v1")

    state_shapes = _shape_map(reader.states)
    sample_shapes = _shape_map(reader.samples)
    sample_count = int(reader.samples["state_index"].shape[0])
    state_count = int(reader.states["planet_features"].shape[0])

    for key, (_, tail_shape) in STATE_SPECS.items():
        expected = (state_count, *tail_shape)
        if tuple(reader.states[key].shape) != expected:
            raise RuntimeError(f"state array {key} shape {list(reader.states[key].shape)} != expected {list(expected)}")
    for key, (_, tail_shape) in SAMPLE_SPECS.items():
        expected = (sample_count, *tail_shape)
        if tuple(reader.samples[key].shape) != expected:
            raise RuntimeError(f"sample array {key} shape {list(reader.samples[key].shape)} != expected {list(expected)}")

    state_index = np.asarray(reader.samples["state_index"])
    source_slot = np.asarray(reader.samples["source_slot"])
    target_label = np.asarray(reader.samples["target_label"])
    amount_label = np.asarray(reader.samples["amount_label"])
    target_mask = np.asarray(reader.samples["target_mask"])
    amount_mask = np.asarray(reader.samples["amount_mask"])

    if np.any(state_index >= state_count):
        raise RuntimeError("source-turn samples contain state_index outside state array")
    if np.any(source_slot >= 64):
        raise RuntimeError("source-turn samples contain source_slot outside action space")
    if np.any(target_label > NOOP_TARGET_SLOT):
        raise RuntimeError("source-turn samples contain target_label outside action space")
    if np.any(amount_label >= len(AMOUNT_BIN_NAMES)):
        raise RuntimeError("source-turn samples contain amount_label outside action space")

    bad_targets = int(sum(not bool(target_mask[i, int(target_label[i])]) for i in range(sample_count)))
    bad_amounts = int(sum(not bool(amount_mask[i, int(amount_label[i])]) for i in range(sample_count)))
    if bad_targets:
        raise RuntimeError(f"BC source-turn samples contain {bad_targets} target labels outside target_mask")
    if bad_amounts:
        raise RuntimeError(f"BC source-turn samples contain {bad_amounts} amount labels outside amount_mask")

    positives = int(np.sum(target_label != NOOP_TARGET_SLOT))
    noops = int(sample_count - positives)
    amount_bins = Counter(str(AMOUNT_BIN_NAMES[int(a)]) for a in amount_label.tolist())
    report = {
        "counts": {
            "source_turn_rows": sample_count,
            "positive_source_turns": positives,
            "noop_source_turns": noops,
            "drop_for_v1_bc_rows": 0,
            "bc_invalid_source_rows": 0,
        },
        "rates": {
            "noop_source_rate": float(noops / max(sample_count, 1)),
            "bc_invalid_source_rate": 0.0,
        },
        "target_inference_methods": {},
        "amount_bin_distribution_source_turns": dict(amount_bins),
        "dense_array_shapes": {},
        "state_array_shapes": state_shapes,
        "source_turn_array_shapes": sample_shapes,
        "metadata_stats": metadata.get("stats", {}),
        "decision_checks": {
            "dataset_format": metadata.get("dataset_format"),
            "target_head_dim": metadata.get("action_space", {}).get("target_slots"),
            "noop_target_slot": metadata.get("action_space", {}).get("noop_target_slot"),
            "amount_bins": metadata.get("action_space", {}).get("amount_bins"),
            "angle_policy": metadata.get("action_space", {}).get("angle_policy"),
            "pair_feature_dim": len(metadata.get("pair_feature_names", [])) or int(reader.samples["pair_features"].shape[-1]),
            "primary_train_rows": "samples/*.npy",
            "dense_bc_arrays": None,
        },
    }
    with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    lines = ["# Orbit Wars Dataset Validation Report", "", "## Counts"]
    for k, v in report["counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Rates"]
    for k, v in report["rates"].items():
        lines.append(f"- `{k}`: {v:.4f}")
    lines += ["", "## Training contract checks"]
    for k, v in report["decision_checks"].items():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Notes", "- `samples/*.npy` is the final train-ready BC source-turn dataset."]
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", "--dataset-dir", dest="out_dir", required=True)
    args = ap.parse_args()
    report = validate_dataset(args.out_dir)
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    print(json.dumps(report["rates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
