from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from orbit_bc_training.checkpoints import load_checkpoint
from orbit_bc_training.decode_policy import decode_bc_prediction
from orbit_bc_training.losses import apply_mask, masked_argmax
from orbit_training_prep.canonical import canonicalize_observation, uncanonicalize_move
from orbit_training_prep.features import _targeted_fleets, build_feature_state, pair_features_from_obs
from orbit_training_prep.geometry_bridge import make_geometry
from orbit_training_prep.schema import AMOUNT_BIN_NAMES, P_MAX, NOOP_TARGET_SLOT, capture_needed_ships, decode_amount_bin, owned_source_slots, safe_float
from orbit_training_prep.viability import compute_viability_masks

from .config import DEFAULT_DEVICE, DEFAULT_GEOMETRY_HORIZON


@dataclass(frozen=True)
class MoveValidation:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class HeuristicBCChoice:
    target: int
    amount_bin: int
    move: list[Any] | None
    reason: str = ""
    score: float = float("-inf")
    noop_probability: float = 0.0
    candidates: tuple[dict[str, Any], ...] = ()


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
    feature_state: Any = None,
    movement: Any = None,
    incoming_by_target: Any = None,
) -> dict[str, torch.Tensor]:
    del model_config
    feature_state = feature_state or build_feature_state(obs, int(player_id), P_MAX)
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
        movement=movement,
        incoming_by_target=incoming_by_target,
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


def _slot_for_planet_id(obs: dict[str, Any], planet_id: int) -> int | None:
    for slot, planet in enumerate(obs.get("planets", [])[:P_MAX]):
        if isinstance(planet, (list, tuple)) and len(planet) >= 1 and int(planet[0]) == int(planet_id):
            return int(slot)
    return None


def _masked_target_probabilities(target_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    logits = torch.as_tensor(target_logits).float()
    mask = torch.as_tensor(target_mask, dtype=torch.bool, device=logits.device)
    return apply_mask(logits.unsqueeze(0), mask.unsqueeze(0)).softmax(dim=-1)[0]


def _topk_launch_targets(target_logits: torch.Tensor, target_mask: torch.Tensor, k: int) -> list[int]:
    logits = torch.as_tensor(target_logits).float()
    mask = torch.as_tensor(target_mask, dtype=torch.bool, device=logits.device).clone()
    if mask.numel() <= NOOP_TARGET_SLOT:
        return []
    mask[NOOP_TARGET_SLOT] = False
    ranked = torch.where(mask, logits, torch.full_like(logits, -1.0e9))
    order = torch.argsort(ranked, descending=True, stable=True)
    out: list[int] = []
    for idx in order[: max(1, int(k))].detach().cpu().tolist():
        if 0 <= int(idx) < NOOP_TARGET_SLOT and bool(mask[int(idx)].detach().cpu().item()):
            out.append(int(idx))
    return out


def _amount_mask_for_target(amount_mask_table: Any, target: int) -> torch.Tensor:
    table = torch.as_tensor(amount_mask_table, dtype=torch.bool)
    if table.ndim == 1:
        return table
    return table[int(target)]


def _one_hot_logits_like(logits: torch.Tensor, index: int) -> torch.Tensor:
    out = torch.full_like(torch.as_tensor(logits).float(), -1.0e6)
    if 0 <= int(index) < int(out.numel()):
        out[int(index)] = 0.0
    return out


def _single_target_mask_like(target_mask: torch.Tensor, target: int) -> torch.Tensor:
    out = torch.zeros_like(torch.as_tensor(target_mask, dtype=torch.bool))
    if 0 <= int(target) < int(out.numel()):
        out[int(target)] = True
    return out


def _lite_heuristic_score(
    obs: dict[str, Any],
    player_id: int,
    source_slot: int,
    target_slot: int,
    ships: int,
    angle: float,
    target_probability: float,
) -> float:
    del angle
    planets = obs.get("planets", [])
    if not (0 <= int(source_slot) < len(planets) and 0 <= int(target_slot) < len(planets)):
        return float("-inf")
    source = planets[int(source_slot)]
    target = planets[int(target_slot)]
    if len(source) < 7 or len(target) < 7:
        return float("-inf")
    available = max(1.0, safe_float(source[5]))
    needed = max(1, int(capture_needed_ships(source, target, int(player_id))))
    send = max(1, int(ships))
    sx = safe_float(source[2])
    sy = safe_float(source[3])
    tx = safe_float(target[2])
    ty = safe_float(target[3])
    distance = math.hypot(tx - sx, ty - sy)
    owner = int(target[1])
    production = max(0.0, safe_float(target[6]))
    target_ships = max(0.0, safe_float(target[5]))
    margin = float(send - needed)
    remaining_frac = max(0.0, (available - float(send)) / available)
    oversend = max(0.0, float(send - needed))
    roi = production / max(1.0, float(send))
    if owner == int(player_id):
        owner_value = 0.75
        combat_value = 0.0
        production_weight = 1.25
    elif owner < 0:
        owner_value = 3.0
        combat_value = 0.5
        production_weight = 6.0
    else:
        owner_value = 5.0
        combat_value = min(8.0, target_ships / 4.0)
        production_weight = 6.0
    return (
        owner_value
        + combat_value
        + production * production_weight
        + roi * 12.0
        + margin * 0.15
        + remaining_frac
        + float(target_probability) * 0.25
        - oversend * 0.05
        - distance * 0.01
    )


def choose_heuristic_bc_move(
    obs: dict[str, Any],
    *,
    player_id: int,
    source_planet_id: int,
    target_logits: torch.Tensor,
    amount_logits_for_target: Callable[[int], torch.Tensor],
    geometry: Any,
    target_mask: torch.Tensor,
    amount_mask_table: Any,
    top_k: int = 5,
    noop_threshold: float = 0.85,
) -> HeuristicBCChoice:
    source_slot = _slot_for_planet_id(obs, int(source_planet_id))
    if source_slot is None:
        return HeuristicBCChoice(NOOP_TARGET_SLOT, 0, None, reason="source_missing")
    planets = obs.get("planets", [])
    if source_slot >= len(planets) or len(planets[source_slot]) < 7 or int(planets[source_slot][1]) != int(player_id):
        return HeuristicBCChoice(NOOP_TARGET_SLOT, 0, None, reason="source_not_owned")
    mask = torch.as_tensor(target_mask, dtype=torch.bool, device=torch.as_tensor(target_logits).device)
    probs = _masked_target_probabilities(target_logits, mask)
    noop_prob = float(probs[NOOP_TARGET_SLOT].detach().cpu().item()) if probs.numel() > NOOP_TARGET_SLOT else 0.0
    if noop_prob >= float(noop_threshold):
        return HeuristicBCChoice(NOOP_TARGET_SLOT, 0, None, reason="noop", noop_probability=noop_prob)

    best: HeuristicBCChoice | None = None
    fallback: HeuristicBCChoice | None = None
    candidate_rows: list[dict[str, Any]] = []
    for target in _topk_launch_targets(target_logits, mask, int(top_k)):
        amount_mask = _amount_mask_for_target(amount_mask_table, int(target))
        if not bool(torch.as_tensor(amount_mask, dtype=torch.bool).any().item()):
            row = {"target": int(target), "reason": "amount_mask_empty"}
            candidate_rows.append(row)
            fallback = fallback or HeuristicBCChoice(int(target), 0, None, reason="amount_mask_empty", noop_probability=noop_prob)
            continue
        amount_logits = amount_logits_for_target(int(target))
        amount_bin = _masked_amount_prediction(amount_logits, amount_mask)
        target_planet = planets[int(target)] if 0 <= int(target) < len(planets) else None
        ships = decode_amount_bin(int(amount_bin), float(planets[source_slot][5]), capture_needed_ships(planets[source_slot], target_planet, int(player_id)))
        if ships <= 0:
            row = {"target": int(target), "amount_bin": int(amount_bin), "reason": "amount_decoded_non_positive"}
            candidate_rows.append(row)
            fallback = fallback or HeuristicBCChoice(int(target), int(amount_bin), None, reason="amount_decoded_non_positive", noop_probability=noop_prob)
            continue
        selected_target_logits = _one_hot_logits_like(target_logits, int(target))
        selected_target_mask = _single_target_mask_like(mask, int(target))
        selected_amount_logits = _one_hot_logits_like(torch.as_tensor(amount_logits).float(), int(amount_bin))
        move = decode_bc_prediction(
            obs,
            int(player_id),
            int(source_planet_id),
            selected_target_logits,
            selected_amount_logits,
            geometry,
            target_mask=selected_target_mask,
            amount_mask=amount_mask,
            noop_threshold=1.1,
        )
        if move is None:
            row = {"target": int(target), "amount_bin": int(amount_bin), "ships": int(ships), "reason": "geometry_no_viable_move"}
            candidate_rows.append(row)
            fallback = fallback or HeuristicBCChoice(int(target), int(amount_bin), None, reason="geometry_no_viable_move", noop_probability=noop_prob)
            continue
        angle = float(move[1])
        target_prob = float(probs[int(target)].detach().cpu().item())
        score = _lite_heuristic_score(obs, int(player_id), int(source_slot), int(target), int(ships), angle, target_prob)
        row = {
            "target": int(target),
            "amount_bin": int(amount_bin),
            "ships": int(ships),
            "angle": angle,
            "target_probability": target_prob,
            "score": float(score),
            "reason": "ok",
        }
        candidate_rows.append(row)
        choice = HeuristicBCChoice(int(target), int(amount_bin), move, score=float(score), noop_probability=noop_prob)
        if best is None or choice.score > best.score:
            best = choice

    candidates = tuple(candidate_rows)
    if best is not None:
        return HeuristicBCChoice(best.target, best.amount_bin, best.move, best.reason, best.score, best.noop_probability, candidates)
    if fallback is not None:
        return HeuristicBCChoice(fallback.target, fallback.amount_bin, None, fallback.reason, fallback.score, fallback.noop_probability, candidates)
    return HeuristicBCChoice(NOOP_TARGET_SLOT, 0, None, reason="noop", noop_probability=noop_prob, candidates=candidates)


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


def _planet_id_for_slot(obs: dict[str, Any], slot: int) -> int | None:
    planets = obs.get("planets", [])
    if 0 <= int(slot) < len(planets):
        planet = planets[int(slot)]
        if isinstance(planet, (list, tuple)) and len(planet) >= 1:
            try:
                return int(planet[0])
            except Exception:
                return None
    return None


def _original_slot_for_canonical_slot(transform: Any, slot: int) -> int | None:
    if transform is None:
        return int(slot)
    try:
        return int(transform.canonical_slot_to_original_slot[int(slot)])
    except Exception:
        return None


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
    raw_obs = dict(obs or {})
    raw_player_id = int(raw_obs.get("player", _config_get(config, "player", 0)) or 0)
    canonical_enabled = bool(_config_get(config, "canonicalize_perspective", _RUNTIME_CONFIG.get("canonicalize_perspective", True)))
    canonical_transform = canonicalize_observation(raw_obs, raw_player_id) if canonical_enabled else None
    obs = canonical_transform.obs if canonical_transform is not None else raw_obs
    player_id = 0 if canonical_transform is not None else raw_player_id
    device = str(_config_get(config, "device", _RUNTIME_CONFIG.get("device", DEFAULT_DEVICE)))
    checkpoint = _checkpoint_from(config)
    horizon = int(_config_get(config, "geometry_horizon", _RUNTIME_CONFIG.get("geometry_horizon", DEFAULT_GEOMETRY_HORIZON)))
    debug_enabled = bool(_config_get(config, "debug", _RUNTIME_CONFIG.get("debug", False)))
    candidate_top_k = int(_config_get(config, "candidate_top_k", _RUNTIME_CONFIG.get("candidate_top_k", 5)))
    noop_threshold = float(_config_get(config, "noop_threshold", _RUNTIME_CONFIG.get("noop_threshold", 0.85)))
    deadline = time.monotonic() + max(0.01, _act_timeout_seconds(config) * 0.85)
    model, _ = _load_model_once(checkpoint, device)
    model_config = getattr(model, "config", None)
    geometry = _geometry_once(horizon, "cpu")
    moves: list[list[Any]] = []
    debug: dict[str, Any] = {
        "step": int(obs.get("step", 0) or 0),
        "player_id": player_id,
        "original_player_id": raw_player_id,
        "perspective_canonicalized": canonical_transform is not None,
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
        turn_feature_state = build_feature_state(obs, player_id, P_MAX)
        turn_incoming = _targeted_fleets(obs, player_id, P_MAX)
        try:
            obs_no_fleets = dict(obs)
            obs_no_fleets["fleets"] = []
            turn_movement = geometry.build_or_update_movement(geometry.obs_to_tensors(obs_no_fleets, player_id=player_id))
        except Exception:
            turn_movement = None
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
                    feature_state=turn_feature_state,
                    movement=turn_movement,
                    incoming_by_target=turn_incoming,
                )
                target_output = model(batch)
            target_logits = target_output["target_logits"][0].detach().cpu()
            target_mask = torch.as_tensor(turn_target_mask_np[int(source_slot)], dtype=torch.bool)
            raw_target_argmax = masked_target_prediction(obs, source_slot, target_logits, target_mask)
            amount_logits_cache: dict[int, torch.Tensor] = {}

            def amount_logits_for_target(target: int) -> torch.Tensor:
                target_i = int(target)
                if target_i not in amount_logits_cache:
                    with torch.no_grad():
                        amount_batch = dict(batch)
                        amount_batch["target_label"] = torch.as_tensor([target_i], dtype=torch.long, device=device)
                        amount_output = model(amount_batch)
                    amount_logits_cache[target_i] = amount_output["amount_logits"][0].detach().cpu()
                return amount_logits_cache[target_i]

            choice = choose_heuristic_bc_move(
                obs,
                player_id=player_id,
                source_planet_id=source_planet_id,
                target_logits=target_logits,
                amount_logits_for_target=amount_logits_for_target,
                geometry=geometry,
                target_mask=target_mask,
                amount_mask_table=turn_amount_mask_np[int(source_slot)],
                top_k=candidate_top_k,
                noop_threshold=noop_threshold,
            )
            target_pred = int(choice.target)
            amount_pred = int(choice.amount_bin)
            amount_mask = torch.as_tensor(turn_amount_mask_np[int(source_slot), int(target_pred)], dtype=torch.bool)
            amount_logits = amount_logits_cache.get(int(target_pred), target_output["amount_logits"][0].detach().cpu())
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
            move = choice.move
            canonical_target_planet_id = None if int(target_pred) == NOOP_TARGET_SLOT else _planet_id_for_slot(obs, int(target_pred))
            env_target_slot = None if int(target_pred) == NOOP_TARGET_SLOT else _original_slot_for_canonical_slot(canonical_transform, int(target_pred))
            env_target_planet_id = None if env_target_slot is None else _planet_id_for_slot(raw_obs, int(env_target_slot))
            pred_row = {
                "source_slot": int(source_slot),
                "source_planet_id": source_planet_id,
                "raw_target_argmax": int(raw_target_argmax),
                "selected_target": target_pred,
                "canonical_target_planet_id": canonical_target_planet_id,
                "env_target_slot": env_target_slot,
                "env_target_planet_id": env_target_planet_id,
                "amount_argmax": amount_pred,
                "decoded_move": move,
                "noop_probability": float(choice.noop_probability),
                "candidate_score": float(choice.score),
            }
            if debug_enabled:
                pred_row["candidate_choices"] = list(choice.candidates)
            if move is None:
                reason = choice.reason or _decode_none_reason(obs, player_id, int(source_slot), target_pred, amount_pred)
                if reason != "noop":
                    debug["skipped_invalid_decoded_actions"] += 1
                    _increment_count(debug["skip_reasons"], reason)
                debug["skipped"].append({"source_slot": int(source_slot), "reason": reason})
                debug["predictions"].append(pred_row)
                continue
            env_move = uncanonicalize_move(move, canonical_transform) if canonical_transform is not None else move
            validation = validate_env_move(raw_obs, raw_player_id, env_move)
            if not validation.ok:
                debug["skipped_invalid_decoded_actions"] += 1
                _increment_count(debug["skip_reasons"], validation.reason)
                debug["skipped"].append({"source_slot": int(source_slot), "reason": validation.reason, "move": env_move})
                debug["predictions"].append(pred_row)
                continue
            moves.append([int(env_move[0]), float(env_move[1]), int(env_move[2])])
            pred_row["decoded_move_env"] = [int(env_move[0]), float(env_move[1]), int(env_move[2])]
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
