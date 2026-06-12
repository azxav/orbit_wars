from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schema import NOOP_TARGET_ID, NOOP_TARGET_SLOT
from .source_turn_store import DATASET_FORMAT, SourceTurnDatasetReader, SourceTurnDatasetWriter


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
        candidates = [p for p in candidates if not is_under(p, out / "train") and not is_under(p, out / "valid")]
    if not candidates:
        raise FileNotFoundError(f"No {filename} files found under {root}")
    return candidates


def source_turn_sample_weight(row: dict[str, Any]) -> float:
    target = row.get("target_slot_label", row.get("target_planet_id_label"))
    is_noop = target in (NOOP_TARGET_SLOT, NOOP_TARGET_ID, "no_op", "noop", None)
    weight = 1.0
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


def _is_source_turn_memmap_dataset(root: Path) -> bool:
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        return load_json(metadata_path).get("dataset_format") == DATASET_FORMAT
    except Exception:
        return False


def _write_memmap_split(reader: SourceTurnDatasetReader, out_dir: Path, episode_ids: set[str]) -> int:
    state_episode_ids = np.asarray(reader.states["episode_id"])
    state_index = np.asarray(reader.samples["state_index"])
    selected_samples = [i for i in range(int(state_index.shape[0])) if str(state_episode_ids[int(state_index[i])]) in episode_ids]
    selected_state_ids: list[int] = []
    old_to_new: dict[int, int] = {}
    for sample_idx in selected_samples:
        old_state = int(state_index[sample_idx])
        if old_state not in old_to_new:
            old_to_new[old_state] = len(selected_state_ids)
            selected_state_ids.append(old_state)

    writer = SourceTurnDatasetWriter(out_dir)
    for old_state in selected_state_ids:
        writer.append_state(
            planet_features=np.asarray(reader.states["planet_features"][old_state], dtype=np.float32),
            global_features=np.asarray(reader.states["global_features"][old_state], dtype=np.float32),
            target_state_features=np.asarray(reader.states["target_state_features"][old_state], dtype=np.float32),
            episode_id=str(state_episode_ids[old_state]),
        )
    for sample_idx in selected_samples:
        old_state = int(state_index[sample_idx])
        writer.append_sample(
            state_index=old_to_new[old_state],
            source_slot=int(reader.samples["source_slot"][sample_idx]),
            target_label=int(reader.samples["target_label"][sample_idx]),
            amount_label=int(reader.samples["amount_label"][sample_idx]),
            sample_weight=float(reader.samples["sample_weight"][sample_idx]),
            step=int(reader.samples["step"][sample_idx]),
            pair_features=np.asarray(reader.samples["pair_features"][sample_idx], dtype=np.float32),
            target_mask=np.asarray(reader.samples["target_mask"][sample_idx], dtype=bool),
            amount_mask=np.asarray(reader.samples["amount_mask"][sample_idx], dtype=bool),
        )
    metadata = dict(reader.metadata)
    metadata["split_episode_ids"] = sorted(episode_ids)
    metadata["stats"] = {
        "states": len(selected_state_ids),
        "source_turn_rows": len(selected_samples),
        "source_turn_samples": len(selected_samples),
        "positive_source_turns": int(sum(int(reader.samples["target_label"][i]) != NOOP_TARGET_SLOT for i in selected_samples)),
    }
    writer.finalize(extra_metadata=metadata)
    return len(selected_samples)


def _materialize_memmap_splits(dataset_root: Path, splits: dict[str, Any], out_dir: Path) -> dict[str, dict[str, int]]:
    reader = SourceTurnDatasetReader(dataset_root)
    summary: dict[str, dict[str, int]] = {}
    for split_name, raw_ids in (("train", splits.get("train", [])), ("valid", splits.get("valid", []))):
        episode_ids = {str(x) for x in raw_ids}
        count = _write_memmap_split(reader, out_dir / split_name, episode_ids)
        summary[split_name] = {"source_turn_samples": count, "source_turn_rows": count}
    return summary


def materialize_splits(dataset_root: str | Path, splits_path: str | Path, out_dir: str | Path) -> dict[str, dict[str, int]]:
    splits = load_json(splits_path)
    train_ids = {str(x) for x in splits.get("train", [])}
    valid_ids = {str(x) for x in splits.get("valid", [])}
    overlap = train_ids & valid_ids
    if overlap:
        raise ValueError(f"Episode ids cannot appear in both train and valid: {sorted(overlap)[:5]}")

    root = Path(dataset_root)
    out = Path(out_dir)
    if _is_source_turn_memmap_dataset(root):
        return _materialize_memmap_splits(root, splits, out)

    source_paths = row_file_paths(root, "source_turn_rows.jsonl", out_dir=out)
    summary: dict[str, dict[str, int]] = {}
    for split_name, episode_ids in (("train", train_ids), ("valid", valid_ids)):
        count = write_split_file(
            paths=source_paths,
            out_path=out / split_name / "source_turn_rows.jsonl",
            episode_ids=episode_ids,
            add_sample_weight=True,
        )
        summary[split_name] = {"source_turn_rows": count}
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
