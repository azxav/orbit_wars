from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from orbit_training_prep.features import PAIR_FEATURE_NAMES, pair_features_from_dense
from orbit_training_prep.schema import NOOP_TARGET_SLOT, P_MAX


BAD_FIRST_HITS = {"sun", "bounds", "none"}
OLD_FEATURE_KEYS = {
    "planet_features" + "_" + "v2",
    "global_features" + "_" + "v2",
    "target_state_features" + "_" + "v2",
    "planet_feature_names" + "_" + "v2",
    "global_feature_names" + "_" + "v2",
    "target_state_feature_names" + "_" + "v2",
    "pair_feature_names" + "_" + "v2",
    "feature" + "_version",
}
REQUIRED_DENSE_KEYS = {
    "planet_features",
    "global_features",
    "target_state_features",
    "target_labels",
    "amount_labels",
    "source_mask",
}
OLD_DATASET_MESSAGE = "Old feature-versioned dataset detected. Rebuild dataset with the new compact feature contract."


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
        dataset_dir.parent.parent / "dense_bc_arrays.npz",
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
    weight = 0.8 if is_noop else 1.0
    if bool(row.get("winner_action", False)) or float(row.get("final_reward", 0.0) or 0.0) > 0.0:
        weight *= 1.1
    step = int(row.get("step_index", row.get("step", row.get("obs_step", 0))) or 0)
    if not is_noop and step <= 100:
        weight *= 1.1
    if step > 430:
        weight *= 0.8
    return float(weight)


class OrbitBCDataset(Dataset):
    """One sample per owned source planet using compact dense source-turn features."""

    noop_target_slot = NOOP_TARGET_SLOT

    def __init__(self, dataset_dir: str | Path):
        self.dataset_dir = Path(dataset_dir)
        self.rows = self._load_rows()
        dense_path, state_rows = _find_dense_source(self.dataset_dir)
        if dense_path is None:
            raise RuntimeError(f"Missing dense_bc_arrays.npz for {self.dataset_dir}. Rebuild dataset with the new compact feature contract.")
        if not state_rows:
            raise RuntimeError(f"Missing state_rows.jsonl next to {dense_path}. Rebuild dataset with the new compact feature contract.")
        loaded = np.load(dense_path, allow_pickle=False)
        self._dense = {k: loaded[k] for k in loaded.files}
        if OLD_FEATURE_KEYS.intersection(self._dense):
            raise RuntimeError(OLD_DATASET_MESSAGE)
        missing = sorted(REQUIRED_DENSE_KEYS.difference(self._dense))
        if missing:
            raise RuntimeError(f"Incompatible dense_bc_arrays.npz missing compact keys: {', '.join(missing)}")
        self._obs_uid_to_dense = {str(r["obs_uid"]): i for i, r in enumerate(state_rows)}
        self.planet_feature_dim = int(self._dense["planet_features"].shape[-1])
        self.global_feature_dim = int(self._dense["global_features"].shape[-1])
        self.target_state_feature_dim = int(self._dense["target_state_features"].shape[-1])
        self.pair_feature_dim = len(PAIR_FEATURE_NAMES)
        self._has_viability_masks = "target_viability_mask" in self._dense and "amount_viability_mask" in self._dense

    def _load_rows(self) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.dataset_dir / "source_turn_rows.jsonl")
        out: list[dict[str, Any]] = []
        for row in rows:
            if bool(row.get("drop_for_v1_bc", False)):
                continue
            if not bool(row.get("geometry_viable", True)):
                continue
            if bool(row.get("ambiguous_multi_launch", False)):
                continue
            if str(row.get("target_inference_method", "") or "") == "angular_nearest":
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

    def _dense_index(self, row: dict[str, Any]) -> int:
        obs_uid = row.get("obs_uid")
        if obs_uid is not None and str(obs_uid) in self._obs_uid_to_dense:
            return self._obs_uid_to_dense[str(obs_uid)]
        raise RuntimeError(f"Source row obs_uid={obs_uid!r} is missing from compact dense state rows.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        dense_idx = self._dense_index(row)
        source_slot = int(row["source_slot"])
        target_label = int(row.get("target_slot_label", NOOP_TARGET_SLOT))
        amount_label = int(row.get("amount_bin_label", 0))
        is_noop = target_label == NOOP_TARGET_SLOT
        if is_noop:
            amount_label = 0
        planet_features = np.asarray(self._dense["planet_features"][dense_idx], dtype=np.float32)
        global_features = np.asarray(self._dense["global_features"][dense_idx], dtype=np.float32)
        target_state_features = np.asarray(self._dense["target_state_features"][dense_idx], dtype=np.float32)
        pair_features = pair_features_from_dense(planet_features, target_state_features, source_slot)
        if self._has_viability_masks:
            target_mask = np.asarray(self._dense["target_viability_mask"][dense_idx, source_slot], dtype=bool).copy()
            if not bool(target_mask[target_label]):
                raise RuntimeError(
                    f"BC row {row.get('source_turn_uid', idx)!r} target label {target_label} is not geometry-viable for source {source_slot}."
                )
            amount_mask = np.asarray(self._dense["amount_viability_mask"][dense_idx, source_slot, target_label], dtype=bool).copy()
            if not bool(amount_mask[amount_label]):
                raise RuntimeError(
                    f"BC row {row.get('source_turn_uid', idx)!r} amount label {amount_label} is not geometry-viable for source {source_slot}, target {target_label}."
                )
        else:
            alive = planet_features[:, 0] > 0.0
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
    return batch
