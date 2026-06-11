from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .source_turn_store import SourceTurnDatasetReader


def load_dense_bc_arrays(dataset_dir: str | Path) -> dict[str, np.ndarray]:
    raise RuntimeError("dense_bc_arrays.npz has been removed. Rebuild/load source_turn_memmap_v1 with SourceTurnDatasetReader.")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_source_turn_rows(dataset_dir: str | Path, *, drop_v1_bc: bool = True, winner_only: bool = False) -> list[dict[str, Any]]:
    del drop_v1_bc
    reader = SourceTurnDatasetReader(dataset_dir)
    state_ids = np.asarray(reader.states.get("episode_id", np.asarray([""] * int(reader.states["planet_features"].shape[0]))))
    rows: list[dict[str, Any]] = []
    for i in range(int(reader.samples["state_index"].shape[0])):
        state_index = int(reader.samples["state_index"][i])
        episode_id = str(state_ids[state_index]) if state_index < int(state_ids.shape[0]) else ""
        target_label = int(reader.samples["target_label"][i])
        row = {
            "source_turn_uid": f"{episode_id}:{i}:s{int(reader.samples['source_slot'][i])}",
            "episode_id": episode_id,
            "state_index": state_index,
            "source_slot": int(reader.samples["source_slot"][i]),
            "target_slot_label": target_label,
            "amount_bin_label": int(reader.samples["amount_label"][i]),
            "sample_weight": np.float32(reader.samples["sample_weight"][i]),
            "step_index": int(reader.samples["step"][i]),
            "winner_action": False,
        }
        if winner_only and not bool(row["winner_action"]):
            continue
        rows.append(row)
    return rows
