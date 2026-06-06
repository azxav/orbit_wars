from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_dense_bc_arrays(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(dataset_dir) / "dense_bc_arrays.npz"
    arr = np.load(path, allow_pickle=True)
    return {k: arr[k] for k in arr.files}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_source_turn_rows(dataset_dir: str | Path, *, drop_v1_bc: bool = True, winner_only: bool = False) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(dataset_dir) / "source_turn_rows.jsonl")
    if drop_v1_bc:
        rows = [r for r in rows if not bool(r.get("drop_for_v1_bc", False))]
    if winner_only:
        rows = [r for r in rows if bool(r.get("winner_action", False))]
    return rows


def load_pair_rank_rows(dataset_dir: str | Path, *, drop_v1_bc: bool = True) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(dataset_dir) / "pair_rank_rows.jsonl")
    if drop_v1_bc:
        rows = [r for r in rows if not bool(r.get("drop_for_v1_bc", False))]
    return rows


def make_pair_rank_numpy(dataset_dir: str | Path, *, drop_v1_bc: bool = True) -> dict[str, Any]:
    """Return pair-ranker arrays plus group ids for LightGBM/MLP sanity baselines."""
    rows = load_pair_rank_rows(dataset_dir, drop_v1_bc=drop_v1_bc)
    x = np.asarray([r["features"] for r in rows], dtype=np.float32)
    y = np.asarray([r["label"] for r in rows], dtype=np.float32)
    weights = np.asarray([r.get("train_weight", 1.0) for r in rows], dtype=np.float32)
    group_uid = np.asarray([r["group_uid"] for r in rows])
    return {"x": x, "y": y, "weights": weights, "group_uid": group_uid, "rows": rows}
