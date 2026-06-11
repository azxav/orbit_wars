from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .features import (
    GLOBAL_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    PLANET_FEATURE_NAMES,
    TARGET_STATE_FEATURE_NAMES,
    _targeted_fleets,
    build_feature_state,
    pair_features_from_obs,
)
from .geometry_bridge import make_geometry
from .replay_io import iter_actual_launches, iter_player_steps, load_replay
from .schema import (
    AMOUNT_BIN_NAMES,
    NOOP_TARGET_ID,
    NOOP_TARGET_SLOT,
    P_MAX,
    ActionSpaceSpec,
    owned_source_slots,
)
from .source_turn_store import DATASET_FORMAT, SourceTurnDatasetReader, SourceTurnDatasetWriter
from .target_inference import TargetInferer
from .viability import compute_viability_masks


def resolve_replay_paths(replay_input: str | Path, max_files: int | None = None) -> list[Path]:
    if max_files is not None and int(max_files) < 1:
        raise ValueError("max_files must be at least 1")
    replay_input = Path(replay_input)
    if replay_input.is_dir():
        paths = sorted(p for p in replay_input.glob("*-replay.json") if p.is_file())
        if not paths:
            paths = sorted(
                p for p in replay_input.glob("*.json") if p.is_file() and p.name.lower() != "replay_metadata.json"
            )
        if not paths:
            raise ValueError(f"No replay JSON files found in directory: {replay_input}")
        return paths[: int(max_files)] if max_files is not None else paths
    return [replay_input]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class JsonlWriter:
    def __init__(self, path: Path, enabled: bool = True):
        self.enabled = bool(enabled)
        self.path = path
        self._file = None
        if self.enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _source_row_train_contract(row: dict[str, Any], target_viability_mask: np.ndarray, amount_viability_mask: np.ndarray) -> tuple[bool, dict[str, bool]]:
    target_label = int(row.get("target_slot_label", NOOP_TARGET_SLOT))
    amount_label = int(row.get("amount_bin_label", 0))
    is_noop = target_label == NOOP_TARGET_SLOT
    if is_noop:
        return True, {"ambiguous": False, "geometry_invalid": False, "mask_invalid": False, "angular_fallback": False}
    source_slot = int(row.get("source_slot", -1))
    ambiguous = bool(row.get("ambiguous_multi_launch", False))
    geometry_invalid = not bool(row.get("geometry_viable", False))
    angular_fallback = str(row.get("target_inference_method", "") or "") == "angular_nearest"
    mask_invalid = True
    if (
        0 <= source_slot < int(target_viability_mask.shape[0])
        and 0 <= target_label < int(target_viability_mask.shape[1])
        and 0 <= amount_label < int(amount_viability_mask.shape[2])
    ):
        target_ok = bool(target_viability_mask[source_slot, target_label])
        amount_ok = bool(amount_viability_mask[source_slot, target_label, amount_label])
        mask_invalid = not (target_ok and amount_ok)
    reasons = {
        "ambiguous": ambiguous,
        "geometry_invalid": geometry_invalid,
        "mask_invalid": mask_invalid,
        "angular_fallback": angular_fallback,
    }
    return not any(reasons.values()), reasons


TRAIN_STATS_KEYS = (
    "raw_source_turns",
    "train_source_turns",
    "raw_positive_source_turns",
    "train_positive_source_turns",
    "dropped_source_turns",
    "dropped_ambiguous_sources",
    "dropped_geometry_invalid_sources",
    "dropped_mask_invalid_sources",
    "dropped_angular_fallback_sources",
    "noop_source_turns",
)


def _stats_with_required_train_keys(stats: Counter[str]) -> dict[str, int]:
    out = {str(k): int(v) for k, v in stats.items()}
    for key in TRAIN_STATS_KEYS:
        out.setdefault(key, 0)
    out.setdefault("source_turn_rows", out["train_source_turns"])
    out.setdefault("positive_source_turns", out["train_positive_source_turns"])
    return out


def _sample_weight(row: dict[str, Any]) -> float:
    target = int(row.get("target_slot_label", NOOP_TARGET_SLOT))
    is_noop = target == NOOP_TARGET_SLOT
    weight = float(row.get("train_weight", 1.0))
    if is_noop:
        weight *= 0.2
    if bool(row.get("winner_action", False)) or float(row.get("final_reward", 0.0) or 0.0) > 0.0:
        weight *= 1.1
    step = int(row.get("step_index", row.get("step", row.get("obs_step", 0))) or 0)
    if not is_noop and step <= 100:
        weight *= 1.1
    if step > 430:
        weight *= 0.8
    return float(weight)


class DatasetBuilder:
    def __init__(
        self,
        *,
        horizon: int = 160,
        device: str = "auto",
        batch_size: int | None = 256,
        include_loser_actions: bool = True,
        max_replay_files: int | None = None,
        workers: int = 1,
        write_debug_jsonl: bool = False,
    ):
        self.horizon = int(horizon)
        if batch_size is not None and int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        if int(workers) < 1:
            raise ValueError("workers must be at least 1")
        self.batch_size = None if batch_size is None else int(batch_size)
        self.include_loser_actions = bool(include_loser_actions)
        self.max_replay_files = None if max_replay_files is None else int(max_replay_files)
        self.workers = int(workers)
        self.write_debug_jsonl = bool(write_debug_jsonl)
        self.inferer = TargetInferer(horizon=horizon, device=device, batch_size=batch_size)
        self.device = self.inferer.device

    def _load_replay_or_none(self, replay_path: Path, *, invalid_json_paths: set[str], unreadable_paths: set[str]) -> dict[str, Any] | None:
        try:
            return load_replay(replay_path)
        except json.JSONDecodeError as exc:
            path_str = str(replay_path)
            if path_str not in invalid_json_paths:
                invalid_json_paths.add(path_str)
                print(json.dumps({"warning": "Skipping replay with invalid JSON", "replay_path": path_str, "error": str(exc)}, sort_keys=True), file=sys.stderr)
            return None
        except OSError as exc:
            path_str = str(replay_path)
            if path_str not in unreadable_paths:
                unreadable_paths.add(path_str)
                print(json.dumps({"warning": "Skipping unreadable replay file", "replay_path": path_str, "error": str(exc)}, sort_keys=True), file=sys.stderr)
            return None

    def build_from_replay(self, replay_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
        replay_input = Path(replay_path)
        replay_paths = resolve_replay_paths(replay_input, self.max_replay_files)
        out_dir = Path(out_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = SourceTurnDatasetWriter(out_dir)
        invalid_json_paths: set[str] = set()
        unreadable_paths: set[str] = set()
        valid_replay_paths: list[Path] = []
        stats: Counter[str] = Counter()
        inference_methods: Counter[str] = Counter()
        amount_bins: Counter[int] = Counter()
        debug_dir = out_dir / "debug"

        with (
            JsonlWriter(debug_dir / "launch_rows.jsonl", enabled=self.write_debug_jsonl) as launch_rows,
            JsonlWriter(debug_dir / "source_turn_rows.jsonl", enabled=self.write_debug_jsonl) as source_turn_rows,
            JsonlWriter(debug_dir / "state_rows.jsonl", enabled=self.write_debug_jsonl) as state_rows,
        ):
            for replay_path_obj in replay_paths:
                replay = self._load_replay_or_none(
                    replay_path_obj,
                    invalid_json_paths=invalid_json_paths,
                    unreadable_paths=unreadable_paths,
                )
                if replay is None:
                    continue
                used_replay = False
                for sample in iter_player_steps(replay):
                    obs = sample["obs"]
                    player_id = int(sample["player_id"])
                    final_reward = float(sample["final_reward"])
                    if not self.include_loser_actions and final_reward <= 0:
                        continue
                    action = list(iter_actual_launches(sample["action"]))
                    owned_slots = owned_source_slots(obs, player_id)
                    if not owned_slots and not action:
                        continue
                    used_replay = True
                    obs_uid = f"{sample['episode_id']}:{sample['step_index']}:p{player_id}"
                    feature_state = build_feature_state(obs, player_id, P_MAX)
                    target_viability_mask, amount_viability_mask = compute_viability_masks(
                        obs,
                        player_id,
                        horizon=self.horizon,
                        device=self.device,
                    )
                    pair_geometry = make_geometry(horizon=200, device="cpu")
                    try:
                        obs_no_fleets = dict(obs)
                        obs_no_fleets["fleets"] = []
                        pair_movement = pair_geometry.build_or_update_movement(pair_geometry.obs_to_tensors(obs_no_fleets, player_id=player_id))
                    except Exception:
                        pair_movement = None
                    pair_incoming = _targeted_fleets(obs, player_id, P_MAX)
                    state_index = writer.append_state(
                        planet_features=feature_state.planet_features,
                        global_features=feature_state.global_features,
                        target_state_features=feature_state.target_state_features,
                        episode_id=str(sample["episode_id"]),
                    )
                    state_rows.write({
                        "obs_uid": obs_uid,
                        "episode_id": sample["episode_id"],
                        "step_index": sample["step_index"],
                        "player_id": player_id,
                        "final_reward": final_reward,
                        "num_owned_sources": len(owned_slots),
                        "num_raw_launches": len(action),
                    })
                    stats["states"] += 1
                    stats["owned_source_slots"] += len(owned_slots)
                    stats["raw_launches"] += len(action)
                    if action:
                        stats["raw_launch_batches"] += 1
                        stats["max_raw_launch_batch_size"] = max(int(stats["max_raw_launch_batch_size"]), len(action))

                    inferred_by_source: dict[int, list[dict[str, Any]]] = defaultdict(list)
                    inferred_moves = self.inferer.infer_moves(obs, player_id, action) if action else []
                    for launch_index, inf in enumerate(inferred_moves):
                        row = {
                            "obs_uid": obs_uid,
                            "launch_uid": f"{obs_uid}:l{launch_index}",
                            "episode_id": sample["episode_id"],
                            "step_index": sample["step_index"],
                            "obs_step": sample["obs_step"],
                            "player_id": player_id,
                            "final_reward": final_reward,
                            "winner_action": final_reward > 0,
                            "status": sample["status"],
                            **inf.as_dict(),
                        }
                        launch_rows.write(row)
                        stats["valid_launches"] += int(bool(row["valid_source"]))
                        stats["invalid_launches"] += int(not bool(row["valid_source"]))
                        inference_methods[str(row["target_inference_method"])] += 1
                        amount_bins[int(row["amount_bin"])] += 1
                        if row["valid_source"]:
                            inferred_by_source[int(row["source_slot"])].append(row)

                    for source_slot in owned_slots:
                        launches = inferred_by_source.get(source_slot, [])
                        launches_sorted = sorted(launches, key=lambda r: (-int(r.get("ships", 0)), int(str(r.get("launch_uid", "0")).split("l")[-1])))
                        primary = launches_sorted[0] if launches_sorted else None
                        if primary is None:
                            target_slot = NOOP_TARGET_SLOT
                            target_id = NOOP_TARGET_ID
                            amount_bin = 0
                            ships = 0
                            amount_fraction = 0.0
                            method = "noop"
                            angle_error = 0.0
                            geometry_viable = True
                            contact_target_slot = NOOP_TARGET_SLOT
                            contact_eta = math.inf
                            capture_needed = 0
                        else:
                            target_slot = int(primary["inferred_target_slot"])
                            target_id = int(primary["inferred_target_id"])
                            amount_bin = int(primary["amount_bin"])
                            ships = int(primary["ships"])
                            amount_fraction = float(primary["amount_fraction"])
                            method = str(primary["target_inference_method"])
                            angle_error = float(primary["angle_error"])
                            geometry_viable = bool(primary["geometry_viable"])
                            contact_target_slot = int(primary["contact_target_slot"])
                            contact_eta = float(primary["contact_eta"])
                            capture_needed = int(primary["capture_needed"])
                        multi_launch_count = len(launches_sorted)
                        ambiguous = multi_launch_count > 1
                        row = {
                            "source_turn_uid": f"{obs_uid}:s{source_slot}",
                            "obs_uid": obs_uid,
                            "episode_id": sample["episode_id"],
                            "step_index": sample["step_index"],
                            "obs_step": sample["obs_step"],
                            "player_id": player_id,
                            "final_reward": final_reward,
                            "winner_action": final_reward > 0,
                            "source_slot": int(source_slot),
                            "source_planet_id": int(obs["planets"][source_slot][0]),
                            "target_slot_label": int(target_slot),
                            "target_planet_id_label": int(target_id),
                            "amount_bin_label": int(amount_bin),
                            "amount_bin_name": AMOUNT_BIN_NAMES[int(amount_bin)],
                            "ships_label": int(ships),
                            "amount_fraction": float(amount_fraction),
                            "capture_needed": int(capture_needed),
                            "target_inference_method": method,
                            "contact_target_slot": int(contact_target_slot),
                            "contact_eta": float(contact_eta) if math.isfinite(float(contact_eta)) else None,
                            "angle_error": float(angle_error),
                            "geometry_viable": bool(geometry_viable),
                            "multi_launch_count": int(multi_launch_count),
                            "ambiguous_multi_launch": bool(ambiguous),
                            "train_weight": 1.0 if final_reward > 0 else 0.35,
                            "drop_for_v1_bc": False,
                        }
                        train_valid, drop_reasons = _source_row_train_contract(row, target_viability_mask, amount_viability_mask)
                        is_positive = primary is not None
                        stats["raw_source_turns"] += 1
                        stats["raw_positive_source_turns"] += int(is_positive)
                        stats["noop_source_turns"] += int(not is_positive)
                        stats["ambiguous_multi_launch_sources"] += int(ambiguous)
                        stats["dropped_source_turns"] += int(not train_valid)
                        stats["dropped_ambiguous_sources"] += int(drop_reasons["ambiguous"])
                        stats["dropped_geometry_invalid_sources"] += int(drop_reasons["geometry_invalid"])
                        stats["dropped_mask_invalid_sources"] += int(drop_reasons["mask_invalid"])
                        stats["dropped_angular_fallback_sources"] += int(drop_reasons["angular_fallback"])
                        if not train_valid:
                            continue
                        pair_features = pair_features_from_obs(
                            obs,
                            player_id,
                            int(source_slot),
                            max_planets=P_MAX,
                            target_viability_mask=target_viability_mask[int(source_slot)],
                            amount_viability_mask=amount_viability_mask[int(source_slot)],
                            feature_state=feature_state,
                            geometry=pair_geometry,
                            movement=pair_movement,
                            incoming_by_target=pair_incoming,
                        )
                        target_label = int(target_slot)
                        amount_label = 0 if target_label == NOOP_TARGET_SLOT else int(amount_bin)
                        writer.append_sample(
                            state_index=state_index,
                            source_slot=int(source_slot),
                            target_label=target_label,
                            amount_label=amount_label,
                            sample_weight=_sample_weight(row),
                            step=int(sample["step_index"]),
                            pair_features=pair_features,
                            target_mask=target_viability_mask[int(source_slot)],
                            amount_mask=amount_viability_mask[int(source_slot), target_label],
                        )
                        source_turn_rows.write(row)
                        stats["train_source_turns"] += 1
                        stats["train_positive_source_turns"] += int(is_positive)
                        stats["source_turn_rows"] += 1
                        stats["positive_source_turns"] += int(is_positive)
                if used_replay:
                    valid_replay_paths.append(replay_path_obj)

        metadata = {
            "dataset_format": DATASET_FORMAT,
            "replay_path": str(replay_input),
            "replay_paths": [str(p) for p in valid_replay_paths],
            "replay_paths_requested": [str(p) for p in replay_paths],
            "action_space": ActionSpaceSpec().as_dict(),
            "geometry_horizon": self.horizon,
            "device": self.device,
            "geometry_device": self.device,
            "inference_batch_size": self.batch_size,
            "target_inference_mode": "batched_exact_first_hit_with_angular_fallback",
            "replay_action_alignment": "action at replay index t is paired with observation at replay index t-1",
            "include_loser_actions": self.include_loser_actions,
            "max_replay_files": self.max_replay_files,
            "write_debug_jsonl": self.write_debug_jsonl,
            "planet_feature_names": PLANET_FEATURE_NAMES,
            "global_feature_names": GLOBAL_FEATURE_NAMES,
            "target_state_feature_names": TARGET_STATE_FEATURE_NAMES,
            "pair_feature_names": PAIR_FEATURE_NAMES,
            "stats": _stats_with_required_train_keys(stats),
            "input_replay_files": {
                "requested": len(replay_paths),
                "used": len(valid_replay_paths),
                "skipped_invalid_json": len(invalid_json_paths),
                "skipped_unreadable": len(unreadable_paths),
                "skipped_invalid_json_paths": sorted(invalid_json_paths),
                "skipped_unreadable_paths": sorted(unreadable_paths),
            },
            "target_inference_methods": dict(inference_methods),
            "amount_bin_counts": {AMOUNT_BIN_NAMES[int(k)]: int(v) for k, v in amount_bins.items() if int(k) < len(AMOUNT_BIN_NAMES)},
        }
        return writer.finalize(extra_metadata=metadata)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True, help="Replay JSON file or directory containing replay JSON files.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon", type=int, default=160)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256, help="Maximum number of valid launches to exact-simulate per GPU/CPU batch.")
    ap.add_argument("--winner-only", action="store_true")
    ap.add_argument("--max-files", type=int, default=None, help="Maximum number of replay files to include from a replay directory.")
    ap.add_argument("--workers", type=int, default=6, help="Accepted for compatibility; source-turn builder writes compact memmap output serially.")
    ap.add_argument("--write-debug-jsonl", action="store_true", help="Write debug JSONL files under debug/.")
    args = ap.parse_args()
    builder = DatasetBuilder(
        horizon=args.horizon,
        device=args.device,
        batch_size=args.batch_size,
        include_loser_actions=not args.winner_only,
        max_replay_files=args.max_files,
        workers=args.workers,
        write_debug_jsonl=args.write_debug_jsonl,
    )
    meta = builder.build_from_replay(args.replay, args.out_dir)
    print(json.dumps({"out_dir": args.out_dir, "stats": meta["stats"], "target_inference_methods": meta["target_inference_methods"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
