from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


EPISODE_ROW_FILES = (
    "source_turn_rows.jsonl",
    "pair_rank_rows.jsonl",
    "state_rows.jsonl",
    "launch_rows.jsonl",
)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def discover_episode_ids(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {root}")

    episode_ids: set[str] = set()
    for filename in EPISODE_ROW_FILES:
        for path in sorted(root.rglob(filename)):
            if "train" in path.parts or "valid" in path.parts:
                continue
            for row in iter_jsonl(path):
                episode_id = row.get("episode_id")
                if episode_id is not None:
                    episode_ids.add(str(episode_id))

    if not episode_ids:
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name.startswith("episode"):
                episode_ids.add(child.name)

    if not episode_ids:
        raise ValueError(f"No episode ids found under {root}")
    return sorted(episode_ids)


def make_episode_splits(episode_ids: list[str], *, valid_frac: float = 0.15, seed: int = 42) -> dict[str, list[str]]:
    unique_ids = sorted({str(e) for e in episode_ids})
    if not unique_ids:
        raise ValueError("No episode ids provided")
    if not 0.0 <= float(valid_frac) < 1.0:
        raise ValueError("valid_frac must be in [0, 1)")

    shuffled = list(unique_ids)
    random.Random(int(seed)).shuffle(shuffled)
    if len(shuffled) == 1 or valid_frac == 0.0:
        valid_count = 0
    else:
        valid_count = max(1, round(len(shuffled) * float(valid_frac)))
        valid_count = min(valid_count, len(shuffled) - 1)

    valid = sorted(shuffled[:valid_count])
    train = sorted(shuffled[valid_count:])
    return {"train": train, "valid": valid}


def write_splits(path: str | Path, splits: dict[str, list[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--valid_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episode_ids = discover_episode_ids(args.dataset_root)
    splits = make_episode_splits(episode_ids, valid_frac=args.valid_frac, seed=args.seed)
    write_splits(args.out, splits)
    print(json.dumps({"out": args.out, "train": len(splits["train"]), "valid": len(splits["valid"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
