from __future__ import annotations

from dataclasses import dataclass
import os
import time
from pathlib import Path
from typing import Any

import torch

from orbit_bc_training.checkpoints import load_checkpoint
from orbit_bc_training.decode_policy import decode_bc_prediction
from orbit_bc_training.losses import masked_argmax
from orbit_training_prep.features import all_planet_features
from orbit_training_prep.geometry_bridge import make_geometry
from orbit_training_prep.schema import P_MAX, NOOP_TARGET_SLOT, owned_source_slots, safe_float

from .config import DEFAULT_DEVICE, DEFAULT_GEOMETRY_HORIZON


@dataclass(frozen=True)
class MoveValidation:
    ok: bool
    reason: str = ""


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, dict]] = {}
_GEOMETRY = None
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
    global _GEOMETRY
    if _GEOMETRY is None:
        _GEOMETRY = make_geometry(horizon=int(horizon), device=device)
    return _GEOMETRY


def build_source_batch(obs: dict[str, Any], player_id: int, source_slot: int, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    step = int(obs.get("step", 0) or 0)
    episode_steps = max(int(obs.get("episode_steps", 500) or 500), 1)
    planet_features = torch.as_tensor([all_planet_features(obs, int(player_id), P_MAX)], dtype=torch.float32, device=device)
    global_features = torch.as_tensor(
        [[step / float(episode_steps), float(player_id) / 4.0, float(source_slot) / float(P_MAX), 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    return {
        "planet_features": planet_features,
        "global_features": global_features,
        "source_slot": torch.as_tensor([int(source_slot)], dtype=torch.long, device=device),
    }


def target_mask_for_source(obs: dict[str, Any], source_slot: int) -> torch.Tensor:
    mask = torch.zeros(P_MAX + 1, dtype=torch.bool)
    for slot, p in enumerate(obs.get("planets", [])[:P_MAX]):
        if len(p) >= 7 and int(p[0]) >= 0 and int(slot) != int(source_slot):
            mask[slot] = True
    mask[NOOP_TARGET_SLOT] = True
    return mask


def masked_target_prediction(obs: dict[str, Any], source_slot: int, target_logits: torch.Tensor) -> int:
    mask = target_mask_for_source(obs, source_slot).to(torch.as_tensor(target_logits).device)
    return int(masked_argmax(torch.as_tensor(target_logits).float().unsqueeze(0), mask.unsqueeze(0))[0].item())


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
    geometry = _geometry_once(horizon, "cpu")
    moves: list[list[Any]] = []
    debug: dict[str, Any] = {
        "step": int(obs.get("step", 0) or 0),
        "player_id": player_id,
        "predictions": [],
        "skipped": [],
        "skipped_invalid_decoded_actions": 0,
        "no_op_source_decisions": 0,
        "predicted_launches": 0,
        "returned_moves": 0,
        "timeout": False,
        "error": None,
    }
    try:
        for source_slot in owned_source_slots(obs, player_id):
            if time.monotonic() >= deadline:
                debug["timeout"] = True
                break
            source = obs.get("planets", [])[source_slot]
            source_planet_id = int(source[0])
            with torch.no_grad():
                batch = build_source_batch(obs, player_id, source_slot, device=device)
                output = model(batch)
            target_logits = output["target_logits"][0].detach().cpu()
            amount_logits = output["amount_logits"][0].detach().cpu()
            target_pred = masked_target_prediction(obs, source_slot, target_logits)
            amount_pred = int(torch.argmax(amount_logits).item())
            if target_pred == NOOP_TARGET_SLOT:
                debug["no_op_source_decisions"] += 1
            else:
                debug["predicted_launches"] += 1
            move = decode_bc_prediction(obs, player_id, source_planet_id, target_logits, amount_logits, geometry)
            pred_row = {
                "source_slot": int(source_slot),
                "source_planet_id": source_planet_id,
                "raw_target_argmax": target_pred,
                "amount_argmax": amount_pred,
                "decoded_move": move,
            }
            if move is None:
                reason = "noop" if target_pred == NOOP_TARGET_SLOT else "geometry_or_amount_invalid"
                if reason != "noop":
                    debug["skipped_invalid_decoded_actions"] += 1
                debug["skipped"].append({"source_slot": int(source_slot), "reason": reason})
                debug["predictions"].append(pred_row)
                continue
            validation = validate_env_move(obs, player_id, move)
            if not validation.ok:
                debug["skipped_invalid_decoded_actions"] += 1
                debug["skipped"].append({"source_slot": int(source_slot), "reason": validation.reason, "move": move})
                debug["predictions"].append(pred_row)
                continue
            moves.append([int(move[0]), float(move[1]), int(move[2])])
            debug["predictions"].append(pred_row)
    except Exception as exc:
        debug["error"] = f"{type(exc).__name__}: {exc}"
        moves = []
    debug["returned_moves"] = len(moves)
    LAST_DEBUG = debug if debug_enabled else {k: debug[k] for k in ("skipped_invalid_decoded_actions", "no_op_source_decisions", "predicted_launches", "returned_moves", "timeout", "error")}
    return moves
