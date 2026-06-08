from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from orbit_training_prep.features import (
    GLOBAL_FEATURE_NAMES_V2,
    PAIR_FEATURE_NAMES_V2,
    TARGET_STATE_FEATURE_NAMES_V2,
    pair_features_from_dense_v2,
)
from orbit_training_prep.schema import NOOP_TARGET_SLOT, P_MAX


BAD_FIRST_HITS = {"sun", "bounds", "none"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _find_dense_source(dataset_dir: Path) -> tuple[Path | None, list[dict[str, Any]] | None]:
    local = dataset_dir / "dense_bc_arrays.npz"
    if local.exists():
        state_path = dataset_dir / "state_rows.jsonl"
        return local, _read_jsonl(state_path) if state_path.exists() else None
    candidates = [
        dataset_dir.parent / "dense_bc_arrays.npz",
        dataset_dir.parent.parent / "combined" / "dense_bc_arrays.npz",
    ]
    for dense in candidates:
        state_path = dense.parent / "state_rows.jsonl"
        if dense.exists() and state_path.exists():
            return dense, _read_jsonl(state_path)
    return None, None


def _row_weight(row: dict[str, Any], is_noop: bool) -> float:
    if "sample_weight" in row:
        return float(row["sample_weight"])
    weight = 0.2 if is_noop else 1.0
    if bool(row.get("winner_action", False)) or float(row.get("final_reward", 0.0) or 0.0) > 0.0:
        weight *= 1.25
    step = int(row.get("step_index", row.get("step", row.get("obs_step", 0))) or 0)
    if not is_noop and step <= 100:
        weight *= 3.0
    if step > 430:
        weight *= 0.5
    return float(weight)


class OrbitBCDataset(Dataset):
    """One sample per owned source planet using frozen source-turn labels."""

    noop_target_slot = NOOP_TARGET_SLOT

    def __init__(self, dataset_dir: str | Path, *, feature_version: str = "auto"):
        self.dataset_dir = Path(dataset_dir)
        requested_feature_version = str(feature_version or "auto").lower()
        if requested_feature_version not in {"auto", "v1", "v2"}:
            raise ValueError("feature_version must be one of: auto, v1, v2")
        self.rows = self._load_rows()
        dense_path, state_rows = _find_dense_source(self.dataset_dir)
        self._dense: dict[str, np.ndarray] | None = None
        self._obs_uid_to_dense: dict[str, int] = {}
        self.planet_feature_dim = 13
        self.global_feature_dim = 5
        self.target_state_feature_dim = 0
        self.pair_feature_dim = 0
        self.feature_version = "v1"
        if dense_path is not None:
            loaded = np.load(dense_path, allow_pickle=True)
            self._dense = {k: loaded[k] for k in loaded.files}
            if "planet_features_v2" in self._dense:
                self.feature_version = "v2"
                self.planet_feature_dim = int(self._dense["planet_features_v2"].shape[-1])
                self.global_feature_dim = int(self._dense.get("global_features_v2", np.zeros((1, len(GLOBAL_FEATURE_NAMES_V2)))).shape[-1])
                self.target_state_feature_dim = int(
                    self._dense.get("target_state_features_v2", np.zeros((1, P_MAX, len(TARGET_STATE_FEATURE_NAMES_V2)))).shape[-1]
                )
                self.pair_feature_dim = len(PAIR_FEATURE_NAMES_V2)
            elif "planet_features" in self._dense:
                self.planet_feature_dim = int(self._dense["planet_features"].shape[-1])
            if state_rows:
                self._obs_uid_to_dense = {str(r["obs_uid"]): i for i, r in enumerate(state_rows)}
        if requested_feature_version == "v2" and self.feature_version != "v2":
            dense_label = str(dense_path) if dense_path is not None else "no dense_bc_arrays.npz found"
            raise RuntimeError(
                f"Training requires feature_version='v2', but {self.dataset_dir} resolved {dense_label} with feature_version='{self.feature_version}'. "
                "Rebuild the dataset with the current dataset builder so dense_bc_arrays.npz contains planet_features_v2, "
                "global_features_v2, and target_state_features_v2, or pass --feature_version auto/v1 for a legacy run."
            )
        if requested_feature_version == "v1" and self.feature_version != "v1":
            self.feature_version = "v1"
            if self._dense is not None and "planet_features" in self._dense:
                self.planet_feature_dim = int(self._dense["planet_features"].shape[-1])
            else:
                self.planet_feature_dim = 13
            self.global_feature_dim = 5
            self.target_state_feature_dim = 0
            self.pair_feature_dim = 0

    def _load_rows(self) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.dataset_dir / "source_turn_rows.jsonl")
        out: list[dict[str, Any]] = []
        for row in rows:
            if bool(row.get("drop_for_v1_bc", False)):
                continue
            first_hit = str(row.get("actual_first_hit_type", "") or "")
            if first_hit in BAD_FIRST_HITS:
                continue
            target = int(row.get("target_slot_label", NOOP_TARGET_SLOT))
            amount = int(row.get("amount_bin_label", 0))
            if target == NOOP_TARGET_SLOT and amount != 0:
                row = dict(row)
                row["amount_bin_label"] = 0
            out.append(row)
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def _dense_index(self, row: dict[str, Any]) -> int | None:
        obs_uid = row.get("obs_uid")
        if obs_uid is not None and str(obs_uid) in self._obs_uid_to_dense:
            return self._obs_uid_to_dense[str(obs_uid)]
        return None

    def _planet_features(self, row: dict[str, Any]) -> np.ndarray:
        idx = self._dense_index(row)
        if self._dense is not None and idx is not None:
            if self.feature_version == "v2" and "planet_features_v2" in self._dense:
                return np.asarray(self._dense["planet_features_v2"][idx], dtype=np.float32)
            return np.asarray(self._dense["planet_features"][idx], dtype=np.float32)
        return np.zeros((P_MAX, self.planet_feature_dim), dtype=np.float32)

    def _global_features(self, row: dict[str, Any]) -> np.ndarray:
        idx = self._dense_index(row)
        if self._dense is not None and idx is not None and self.feature_version == "v2" and "global_features_v2" in self._dense:
            return np.asarray(self._dense["global_features_v2"][idx], dtype=np.float32)
        step = int(row.get("step_index", row.get("step", row.get("obs_step", 0))) or 0)
        if self.feature_version == "v2":
            out = np.zeros(len(GLOBAL_FEATURE_NAMES_V2), dtype=np.float32)
            out[0] = step / 500.0
            out[1] = 1.0 - out[0]
            out[2] = float(row.get("player_id", 0)) / 4.0
            out[3] = 0.5
            out[4] = 1.0
            return out
        return np.asarray(
            [
                step / 500.0,
                float(row.get("player_id", 0)) / 4.0,
                float(row.get("source_slot", 0)) / float(P_MAX),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

    def _target_state_features(self, row: dict[str, Any]) -> np.ndarray:
        idx = self._dense_index(row)
        if self._dense is not None and idx is not None and self.feature_version == "v2" and "target_state_features_v2" in self._dense:
            return np.asarray(self._dense["target_state_features_v2"][idx], dtype=np.float32)
        return np.zeros((P_MAX, self.target_state_feature_dim), dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        source_slot = int(row["source_slot"])
        target_label = int(row.get("target_slot_label", NOOP_TARGET_SLOT))
        amount_label = int(row.get("amount_bin_label", 0))
        is_noop = target_label == NOOP_TARGET_SLOT
        if is_noop:
            amount_label = 0
        planet_features = self._planet_features(row)
        target_state_features = self._target_state_features(row)
        pair_features = (
            pair_features_from_dense_v2(planet_features, target_state_features, source_slot)
            if self.feature_version == "v2"
            else np.zeros((P_MAX + 1, 0), dtype=np.float32)
        )
        alive = planet_features[:, 0] > 0.0 if planet_features.size else np.ones(P_MAX, dtype=bool)
        target_mask = np.zeros(P_MAX + 1, dtype=bool)
        target_mask[:P_MAX] = alive
        if 0 <= source_slot < P_MAX:
            target_mask[source_slot] = False
        if 0 <= target_label < P_MAX:
            target_mask[target_label] = True
        target_mask[NOOP_TARGET_SLOT] = True
        amount_mask = np.ones(7, dtype=bool)
        if is_noop:
            amount_mask[:] = False
            amount_mask[0] = True
        else:
            amount_mask[0] = False
        step = int(row.get("step_index", row.get("step", row.get("obs_step", 0))) or 0)
        global_features = self._global_features(row)
        return {
            "planet_features": planet_features,
            "fleet_features": np.zeros((0, 0), dtype=np.float32),
            "global_features": global_features,
            "target_state_features": target_state_features,
            "pair_features": pair_features,
            "source_slot": np.int64(source_slot),
            "target_label": np.int64(target_label),
            "amount_label": np.int64(amount_label),
            "target_mask": target_mask,
            "amount_mask": amount_mask,
            "sample_weight": np.float32(_row_weight(row, is_noop)),
            "is_noop": bool(is_noop),
            "episode_id": str(row.get("episode_id", "")),
            "step": np.int64(step),
            "feature_version": self.feature_version,
        }


def collate_bc_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = {
        "planet_features": torch.float32,
        "global_features": torch.float32,
        "target_state_features": torch.float32,
        "pair_features": torch.float32,
        "source_slot": torch.long,
        "target_label": torch.long,
        "amount_label": torch.long,
        "target_mask": torch.bool,
        "amount_mask": torch.bool,
        "sample_weight": torch.float32,
        "is_noop": torch.bool,
        "step": torch.long,
    }
    batch: dict[str, Any] = {}
    for key, dtype in tensor_keys.items():
        batch[key] = torch.as_tensor(np.stack([s[key] for s in samples]), dtype=dtype)
    batch["episode_id"] = [s["episode_id"] for s in samples]
    batch["feature_version"] = samples[0].get("feature_version", "v1")
    return batch
