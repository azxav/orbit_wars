from __future__ import annotations

from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from orbit_bc_training.checkpoints import load_checkpoint
from orbit_bc_training.decode_policy import decode_bc_prediction
from orbit_bc_training.losses import masked_argmax
from orbit_training_prep.features import build_feature_state, pair_features_from_obs
from orbit_training_prep.geometry_bridge import make_geometry
from orbit_training_prep.schema import AMOUNT_BIN_NAMES, P_MAX, NOOP_TARGET_SLOT, capture_needed_ships, decode_amount_bin, owned_source_slots, safe_float
from orbit_training_prep.viability import compute_viability_masks

from .config import DEFAULT_DEVICE, DEFAULT_GEOMETRY_HORIZON


@dataclass(frozen=True)
class MoveValidation:
    ok: bool
    reason: str = ""


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, dict]] = {}
_GEOMETRY_CACHE: dict[tuple[int, str], Any] = {}
_RUNTIME_CONFIG: dict[str, Any] = {}
LAST_DEBUG: dict[str, Any] = {}


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def configure_bc_agent(
    *,
    checkpoint: str | Path,
    device: str = DEFAULT_DEVICE,
    geometry_horizon: int = DEFAULT_GEOMETRY_HORIZON,
    debug: bool = False,
) -> None:
    _RUNTIME_CONFIG.update(
        {
            "bc_checkpoint": str(checkpoint),
            "device": str(device),
            "geometry_horizon": int(geometry_horizon),
            "debug": bool(debug),
        }
    )


def reset_runtime_state() -> None:
    global LAST_DEBUG
    LAST_DEBUG = {}


def get_last_debug() -> dict[str, Any]:
    return dict(LAST_DEBUG)


def _checkpoint_from(config: Any) -> str:
    checkpoint = _config_get(config, "bc_checkpoint") or _RUNTIME_CONFIG.get("bc_checkpoint") or os.environ.get("ORBIT_BC_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("BC checkpoint is not configured. Pass --bc_checkpoint or set ORBIT_BC_CHECKPOINT.")
    return str(checkpoint)


def _load_model_once(checkpoint: str | Path, device: str) -> tuple[Any, dict]:
    key = (str(Path(checkpoint)), str(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_checkpoint(checkpoint, device=device)
    return _MODEL_CACHE[key]


def _geometry_once(horizon: int, device: str = "cpu"):
    key = (int(horizon), str(device))
    if key not in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE[key] = make_geometry(horizon=int(horizon), device=device)
    return _GEOMETRY_CACHE[key]


def _source_viability_slices(
    obs: dict[str, Any],
    player_id: int,
    source_slot: int,
    *,
    target_viability_mask: Any = None,
    amount_viability_mask: Any = None,
    horizon: int = DEFAULT_GEOMETRY_HORIZON,
    geometry: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    if target_viability_mask is None or amount_viability_mask is None:
        target_viability_mask, amount_viability_mask = compute_viability_masks(
            obs,
            int(player_id),
            horizon=int(horizon),
            device="cpu",
            geometry=geometry,
        )
    target_arr = np.asarray(target_viability_mask, dtype=bool)
    amount_arr = np.asarray(amount_viability_mask, dtype=bool)
    if target_arr.ndim == 2:
        target_arr = target_arr[int(source_slot)]
    if amount_arr.ndim == 3:
        amount_arr = amount_arr[int(source_slot)]
    return target_arr, amount_arr


def build_source_batch(
    obs: dict[str, Any],
    player_id: int,
    source_slot: int,
    *,
    device: str = "cpu",
    model_config: Any = None,
    target_viability_mask: Any = None,
    amount_viability_mask: Any = None,
    horizon: int = DEFAULT_GEOMETRY_HORIZON,
    geometry: Any = None,
) -> dict[str, torch.Tensor]:
    del model_config
    feature_state = build_feature_state(obs, int(player_id), P_MAX)
    source_target_mask, source_amount_mask = _source_viability_slices(
        obs,
        int(player_id),
        int(source_slot),
        target_viability_mask=target_viability_mask,
        amount_viability_mask=amount_viability_mask,
        horizon=int(horizon),
        geometry=geometry,
    )
    pair_features = pair_features_from_obs(
        obs,
        int(player_id),
        int(source_slot),
        max_planets=P_MAX,
        target_viability_mask=source_target_mask,
        amount_viability_mask=source_amount_mask,
        feature_state=feature_state,
        geometry=geometry,
    )
    return {
        "planet_features": torch.as_tensor(feature_state.planet_features[None, ...], dtype=torch.float32, device=device),
        "global_features": torch.as_tensor(feature_state.global_features[None, ...], dtype=torch.float32, device=device),
        "target_state_features": torch.as_tensor(feature_state.target_state_features[None, ...], dtype=torch.float32, device=device),
        "pair_features": torch.as_tensor(pair_features[None, ...], dtype=torch.float32, device=device),
        "source_slot": torch.as_tensor([int(source_slot)], dtype=torch.long, device=device),
    }


def target_mask_for_source(
    obs: dict[str, Any],
    source_slot: int,
    *,
    player_id: int | None = None,
    horizon: int = DEFAULT_GEOMETRY_HORIZON,
    geometry: Any = None,
) -> torch.Tensor:
    pid = int(obs.get("player", 0) if player_id is None else player_id)
    target_mask, _ = compute_viability_masks(obs, pid, horizon=int(horizon), device="cpu", geometry=geometry)
    return torch.as_tensor(target_mask[int(source_slot)], dtype=torch.bool)


def masked_target_prediction(obs: dict[str, Any], source_slot: int, target_logits: torch.Tensor, target_mask: torch.Tensor | None = None) -> int:
    mask = target_mask_for_source(obs, source_slot) if target_mask is None else torch.as_tensor(target_mask, dtype=torch.bool)
    mask = mask.to(torch.as_tensor(target_logits).device)
    return int(masked_argmax(torch.as_tensor(target_logits).float().unsqueeze(0), mask.unsqueeze(0))[0].item())


def _masked_amount_prediction(amount_logits: torch.Tensor, amount_mask: torch.Tensor) -> int:
    mask = torch.as_tensor(amount_mask, dtype=torch.bool, device=torch.as_tensor(amount_logits).device)
    return int(masked_argmax(torch.as_tensor(amount_logits).float().unsqueeze(0), mask.unsqueeze(0))[0].item())


def validate_env_move(obs: dict[str, Any], player_id: int, move: list[Any]) -> MoveValidation:
    if not isinstance(move, (list, tuple)) or len(move) != 3:
        return MoveValidation(False, "bad_shape")
    try:
        from_planet_id = int(move[0])
        angle = float(move[1])
        ships = int(move[2])
    except Exception:
        return MoveValidation(False, "bad_types")
    if ships <= 0:
        return MoveValidation(False, "non_positive_ships")
    if not (-7.0 <= angle <= 7.0):
        return MoveValidation(False, "angle_out_of_range")
    for p in obs.get("planets", [])[:P_MAX]:
        if len(p) >= 7 and int(p[0]) == from_planet_id:
            if int(p[1]) != int(player_id):
                return MoveValidation(False, "source_not_owned")
            if ships > int(safe_float(p[5])):
                return MoveValidation(False, "too_many_ships")
            return MoveValidation(True)
    return MoveValidation(False, "source_missing")


def _act_timeout_seconds(config: Any) -> float:
    value = _config_get(config, "actTimeout", _config_get(config, "act_timeout", 1.0))
    try:
        return max(0.05, float(value))
    except Exception:
        return 1.0


def _target_count_key(target: int) -> str:
    return "noop" if int(target) == NOOP_TARGET_SLOT else f"slot_{int(target)}"


def _amount_count_key(amount: int) -> str:
    if 0 <= int(amount) < len(AMOUNT_BIN_NAMES):
        return str(AMOUNT_BIN_NAMES[int(amount)])
    return f"bin_{int(amount)}"


def _increment_count(counts: dict[str, int], key: str, value: int = 1) -> None:
    counts[str(key)] = int(counts.get(str(key), 0)) + int(value)


def _decode_none_reason(obs: dict[str, Any], player_id: int, source_slot: int, target: int, amount_bin: int) -> str:
    if int(target) == NOOP_TARGET_SLOT:
        return "noop"
    planets = obs.get("planets", [])
    if source_slot >= len(planets):
        return "source_missing"
    source = planets[source_slot]
    if len(source) < 7 or int(source[1]) != int(player_id):
        return "source_not_owned"
    target_planet = planets[target] if 0 <= int(target) < len(planets) else None
    ships = decode_amount_bin(int(amount_bin), float(source[5]), capture_needed_ships(source, target_planet, int(player_id)))
    if ships <= 0:
        return "amount_decoded_non_positive"
    return "geometry_no_viable_move"


def agent(obs, config):
    global LAST_DEBUG
    obs = dict(obs or {})
    player_id = int(obs.get("player", _config_get(config, "player", 0)) or 0)
    device = str(_config_get(config, "device", _RUNTIME_CONFIG.get("device", DEFAULT_DEVICE)))
    checkpoint = _checkpoint_from(config)
    horizon = int(_config_get(config, "geometry_horizon", _RUNTIME_CONFIG.get("geometry_horizon", DEFAULT_GEOMETRY_HORIZON)))
    debug_enabled = bool(_config_get(config, "debug", _RUNTIME_CONFIG.get("debug", False)))
    deadline = time.monotonic() + max(0.01, _act_timeout_seconds(config) * 0.85)
    model, _ = _load_model_once(checkpoint, device)
    model_config = getattr(model, "config", None)
    geometry = _geometry_once(horizon, "cpu")
    moves: list[list[Any]] = []
    debug: dict[str, Any] = {
        "step": int(obs.get("step", 0) or 0),
        "player_id": player_id,
        "predictions": [],
        "skipped": [],
        "skip_reasons": {},
        "opening_prediction_counts": {"target": {}, "amount": {}, "target_amount": {}},
        "skipped_invalid_decoded_actions": 0,
        "no_op_source_decisions": 0,
        "predicted_launches": 0,
        "returned_moves": 0,
        "timeout": False,
        "error": None,
    }
    try:
        turn_target_mask_np, turn_amount_mask_np = compute_viability_masks(obs, player_id, horizon=horizon, device="cpu", geometry=geometry)
        for source_slot in owned_source_slots(obs, player_id):
            if time.monotonic() >= deadline:
                debug["timeout"] = True
                break
            source = obs.get("planets", [])[source_slot]
            source_planet_id = int(source[0])
            with torch.no_grad():
                batch = build_source_batch(
                    obs,
                    player_id,
                    source_slot,
                    device=device,
                    model_config=model_config,
                    target_viability_mask=turn_target_mask_np[int(source_slot)],
                    amount_viability_mask=turn_amount_mask_np[int(source_slot)],
                    horizon=horizon,
                    geometry=geometry,
                )
                target_output = model(batch)
            target_logits = target_output["target_logits"][0].detach().cpu()
            target_mask = torch.as_tensor(turn_target_mask_np[int(source_slot)], dtype=torch.bool)
            target_pred = masked_target_prediction(obs, source_slot, target_logits, target_mask)
            amount_mask = torch.as_tensor(turn_amount_mask_np[int(source_slot), int(target_pred)], dtype=torch.bool)
            with torch.no_grad():
                amount_batch = dict(batch)
                amount_batch["target_label"] = torch.as_tensor([int(target_pred)], dtype=torch.long, device=device)
                amount_output = model(amount_batch)
            amount_logits = amount_output["amount_logits"][0].detach().cpu()
            amount_pred = _masked_amount_prediction(amount_logits, amount_mask)
            if target_pred == NOOP_TARGET_SLOT:
                debug["no_op_source_decisions"] += 1
            else:
                debug["predicted_launches"] += 1
            if 0 <= int(debug["step"]) < 100:
                target_key = _target_count_key(target_pred)
                amount_key = _amount_count_key(amount_pred)
                _increment_count(debug["opening_prediction_counts"]["target"], target_key)
                _increment_count(debug["opening_prediction_counts"]["amount"], amount_key)
                _increment_count(debug["opening_prediction_counts"]["target_amount"], f"{target_key}|{amount_key}")
            move = decode_bc_prediction(
                obs,
                player_id,
                source_planet_id,
                target_logits,
                amount_logits,
                geometry,
                target_mask=target_mask,
                amount_mask=amount_mask,
            )
            pred_row = {
                "source_slot": int(source_slot),
                "source_planet_id": source_planet_id,
                "raw_target_argmax": target_pred,
                "amount_argmax": amount_pred,
                "decoded_move": move,
            }
            if move is None:
                reason = _decode_none_reason(obs, player_id, int(source_slot), target_pred, amount_pred)
                if reason != "noop":
                    debug["skipped_invalid_decoded_actions"] += 1
                    _increment_count(debug["skip_reasons"], reason)
                debug["skipped"].append({"source_slot": int(source_slot), "reason": reason})
                debug["predictions"].append(pred_row)
                continue
            validation = validate_env_move(obs, player_id, move)
            if not validation.ok:
                debug["skipped_invalid_decoded_actions"] += 1
                _increment_count(debug["skip_reasons"], validation.reason)
                debug["skipped"].append({"source_slot": int(source_slot), "reason": validation.reason, "move": move})
                debug["predictions"].append(pred_row)
                continue
            moves.append([int(move[0]), float(move[1]), int(move[2])])
            debug["predictions"].append(pred_row)
    except Exception as exc:
        debug["error"] = f"{type(exc).__name__}: {exc}"
        moves = []
    debug["returned_moves"] = len(moves)
    compact_keys = (
        "skipped_invalid_decoded_actions",
        "skip_reasons",
        "opening_prediction_counts",
        "no_op_source_decisions",
        "predicted_launches",
        "returned_moves",
        "timeout",
        "error",
    )
    LAST_DEBUG = debug if debug_enabled else {k: debug[k] for k in compact_keys}
    return moves
