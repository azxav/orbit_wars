from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schema import NOOP_TARGET_ID, NOOP_TARGET_SLOT


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def row_file_paths(dataset_root: str | Path, filename: str, out_dir: str | Path | None = None) -> list[Path]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")
    out = Path(out_dir) if out_dir is not None else None
    candidates = sorted(p for p in root.rglob(filename) if p.is_file())
    if out is not None:
        candidates = [
            p for p in candidates
            if not is_under(p, out / "train") and not is_under(p, out / "valid")
        ]
    if not candidates:
        raise FileNotFoundError(f"No {filename} files found under {root}")
    return candidates


def source_turn_sample_weight(row: dict[str, Any]) -> float:
    target = row.get("target_slot_label", row.get("target_planet_id_label"))
    is_noop = target in (NOOP_TARGET_SLOT, NOOP_TARGET_ID, "no_op", "noop", None)
    weight = 0.2 if is_noop else 1.0
    if bool(row.get("winner_action", False)) or float(row.get("final_reward", 0.0) or 0.0) > 0.0:
        weight *= 1.25
    step = int(row.get("step_index", row.get("step", row.get("obs_step", 0))) or 0)
    if step > 430:
        weight *= 0.5
    return float(weight)


def write_split_file(
    *,
    paths: list[Path],
    out_path: Path,
    episode_ids: set[str],
    add_sample_weight: bool,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_uids: set[str] = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for path in paths:
            for row in iter_jsonl(path):
                if str(row.get("episode_id")) not in episode_ids:
                    continue
                uid = row.get("source_turn_uid")
                if uid is not None:
                    uid = str(uid)
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)
                if add_sample_weight:
                    row = dict(row)
                    row["sample_weight"] = source_turn_sample_weight(row)
                f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                count += 1
    return count


def materialize_splits(dataset_root: str | Path, splits_path: str | Path, out_dir: str | Path) -> dict[str, dict[str, int]]:
    splits = load_json(splits_path)
    train_ids = {str(x) for x in splits.get("train", [])}
    valid_ids = {str(x) for x in splits.get("valid", [])}
    overlap = train_ids & valid_ids
    if overlap:
        raise ValueError(f"Episode ids cannot appear in both train and valid: {sorted(overlap)[:5]}")

    source_paths = row_file_paths(dataset_root, "source_turn_rows.jsonl", out_dir=out_dir)
    out = Path(out_dir)
    summary: dict[str, dict[str, int]] = {}
    for split_name, episode_ids in (("train", train_ids), ("valid", valid_ids)):
        summary[split_name] = {
            "source_turn_rows": write_split_file(
                paths=source_paths,
                out_path=out / split_name / "source_turn_rows.jsonl",
                episode_ids=episode_ids,
                add_sample_weight=True,
            ),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary = materialize_splits(args.dataset_root, args.splits, args.out)
    print(json.dumps({"out": args.out, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
