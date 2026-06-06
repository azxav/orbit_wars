from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def validate_dataset(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    launch_rows = read_jsonl(out_dir / "launch_rows.jsonl") if (out_dir / "launch_rows.jsonl").exists() else []
    source_rows = read_jsonl(out_dir / "source_turn_rows.jsonl") if (out_dir / "source_turn_rows.jsonl").exists() else []
    pair_rows = read_jsonl(out_dir / "pair_rank_rows.jsonl") if (out_dir / "pair_rank_rows.jsonl").exists() else []
    metadata = json.load(open(out_dir / "metadata.json", "r", encoding="utf-8")) if (out_dir / "metadata.json").exists() else {}

    methods = Counter(str(r.get("target_inference_method")) for r in launch_rows)
    amount_bins = Counter(str(r.get("amount_bin_name", AMOUNT_BIN_NAMES[int(r.get("amount_bin_label", 0))] if "amount_bin_label" in r else r.get("amount_bin", 0))) for r in source_rows)
    launch_valid = sum(1 for r in launch_rows if r.get("valid_source"))
    contact = sum(1 for r in launch_rows if r.get("target_inference_method") == "first_contact")
    angular = sum(1 for r in launch_rows if r.get("target_inference_method") == "angular_nearest")
    geometry_viable = sum(1 for r in source_rows if r.get("geometry_viable"))
    ambiguous = sum(1 for r in source_rows if r.get("ambiguous_multi_launch"))
    drop_v1 = sum(1 for r in source_rows if r.get("drop_for_v1_bc"))
    positives = sum(1 for r in source_rows if int(r.get("target_slot_label", NOOP_TARGET_SLOT)) != NOOP_TARGET_SLOT)
    noops = sum(1 for r in source_rows if int(r.get("target_slot_label", NOOP_TARGET_SLOT)) == NOOP_TARGET_SLOT)

    groups = defaultdict(int)
    positives_by_group = defaultdict(int)
    for r in pair_rows:
        gid = str(r.get("group_uid"))
        groups[gid] += 1
        positives_by_group[gid] += int(r.get("label", 0))
    group_positive_hist = Counter(positives_by_group.values())
    bad_pair_groups = [g for g, c in positives_by_group.items() if c != 1]

    arr_info = {}
    npz_path = out_dir / "dense_bc_arrays.npz"
    if npz_path.exists():
        arr = np.load(npz_path, allow_pickle=True)
        arr_info = {k: list(v.shape) for k, v in arr.items() if hasattr(v, "shape")}

    denom_launch = max(len(launch_rows), 1)
    denom_source = max(len(source_rows), 1)
    report = {
        "counts": {
            "launch_rows": len(launch_rows),
            "source_turn_rows": len(source_rows),
            "pair_rank_rows": len(pair_rows),
            "pair_groups": len(groups),
            "valid_launches": launch_valid,
            "positive_source_turns": positives,
            "noop_source_turns": noops,
            "ambiguous_multi_launch_sources": ambiguous,
            "drop_for_v1_bc_rows": drop_v1,
        },
        "rates": {
            "valid_launch_rate": launch_valid / denom_launch,
            "first_contact_launch_rate": contact / denom_launch,
            "angular_fallback_launch_rate": angular / denom_launch,
            "geometry_viable_source_rate": geometry_viable / denom_source,
            "noop_source_rate": noops / denom_source,
            "drop_for_v1_bc_rate": drop_v1 / denom_source,
        },
        "target_inference_methods": dict(methods),
        "amount_bin_distribution_source_turns": dict(amount_bins),
        "pair_group_positive_label_hist": {str(k): int(v) for k, v in group_positive_hist.items()},
        "bad_pair_group_count": len(bad_pair_groups),
        "dense_array_shapes": arr_info,
        "metadata_stats": metadata.get("stats", {}),
        "decision_checks": {
            "target_head_dim": metadata.get("action_space", {}).get("target_slots"),
            "noop_target_slot": metadata.get("action_space", {}).get("noop_target_slot"),
            "amount_bins": metadata.get("action_space", {}).get("amount_bins"),
            "angle_policy": metadata.get("action_space", {}).get("angle_policy"),
            "primary_train_rows": "source_turn_rows.jsonl",
            "baseline_ranker_rows": "pair_rank_rows.jsonl",
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
