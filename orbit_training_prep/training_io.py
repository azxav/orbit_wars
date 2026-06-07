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
