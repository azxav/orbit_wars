from __future__ import annotations

from typing import Any

import torch

from .geometry_bridge import analytic_first_contact, make_geometry
from .schema import NOOP_TARGET_ID, build_planet_slot_maps, safe_float
from orbit_geometry_skeleton.geometry_skeleton import fleet_speed

LAUNCH_SURFACE_OFFSET = 0.1


def resolve_geometry_device(device: str) -> str:
    requested = str(device).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for target inference, but torch.cuda.is_available() is false.")
    return requested


class ExactTargetSimulator:
    """Cached exact first-contact simulator used by dataset target inference.

    The old path rebuilt GeometrySkeleton and materialized an [launches, horizon, planets]
    collision tensor for every player step. This keeps one geometry instance per worker and
    delegates first contact to the shortlist-based analytic kernel used by the geometry layer.
    """

    def __init__(self, *, horizon: int = 160, device: str = "cpu"):
        self.horizon = int(horizon)
        self.device = resolve_geometry_device(device)
        self._geometry = None

    def _movement_for_obs(self, obs: dict[str, Any], player_id: int):
        if self._geometry is None:
            self._geometry = make_geometry(horizon=self.horizon, device=self.device)
        obs_tensors = self._geometry.obs_to_tensors(obs, player_id=player_id)
        return self._geometry.build_or_update_movement(obs_tensors)

    def first_hit_for_launch(self, obs: dict[str, Any], player_id: int, move: dict[str, Any]) -> dict[str, Any]:
        return self.first_hits_for_launches(obs, player_id, [move])[0]

    def first_hits_for_launches(self, obs: dict[str, Any], player_id: int, moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not moves:
            return []
        id_to_slot, _ = build_planet_slot_maps(obs)
        planets = obs.get("planets", [])
        resolved_slots: list[int] = []
        valid_indices: list[int] = []
        out: list[dict[str, Any] | None] = [None] * len(moves)
        for i, move in enumerate(moves):
            source_id = int(move["source_planet_id"])
            source_slot = id_to_slot.get(source_id, int(move.get("source_slot", -1)))
            if source_slot < 0 or source_slot >= len(planets):
                out[i] = {"hit_type": "invalid", "hit_id": None, "hit_slot": None, "eta": None, "reason": "source_not_found"}
                continue
            resolved_slots.append(int(source_slot))
            valid_indices.append(i)

        if not valid_indices:
            return [hit for hit in out if hit is not None]

        movement = self._movement_for_obs(obs, int(player_id))
        device = movement.device
        dtype = movement.dtype
        horizon = min(int(self.horizon), int(movement.x.shape[0]) - 1)

        valid_moves = [moves[i] for i in valid_indices]
        angle = torch.tensor([safe_float(move.get("raw_angle")) for move in valid_moves], dtype=dtype, device=device)
        ships = torch.tensor([max(1.0, safe_float(move.get("ships"), 1.0)) for move in valid_moves], dtype=dtype, device=device)
        speed = fleet_speed(ships).clamp(min=1e-6)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        source_slots = torch.tensor(resolved_slots, dtype=torch.long, device=device)
        sx, sy = movement.position_at_slots(source_slots, 0)
        sr = movement.radii[source_slots.clamp(0, max(movement.P - 1, 0))]
        launch_x = sx + cos_a * (sr + LAUNCH_SURFACE_OFFSET)
        launch_y = sy + sin_a * (sr + LAUNCH_SURFACE_OFFSET)

        contact_slots, eta = analytic_first_contact(
            launch_x=launch_x,
            launch_y=launch_y,
            cos_a=cos_a,
            sin_a=sin_a,
            speed=speed,
            px=movement.x[: horizon + 1],
            py=movement.y[: horizon + 1],
            p_alive0=movement.alive_at(0),
            radii=movement.radii,
            H=horizon,
        )
        planet_ids = movement.planet_ids.long().detach().cpu().tolist()
        comet_ids = {int(x) for x in obs.get("comet_planet_ids", []) if int(x) >= 0}
        contact_slots_cpu = contact_slots.detach().cpu().tolist()
        eta_cpu = eta.detach().cpu().tolist()

        for batch_index, out_index in enumerate(valid_indices):
            hit_slot = int(contact_slots_cpu[batch_index])
            if 0 <= hit_slot < len(planet_ids):
                pid = int(planet_ids[hit_slot])
                out[out_index] = {
                    "hit_type": "comet" if pid in comet_ids else "planet",
                    "hit_id": pid,
                    "hit_slot": hit_slot,
                    "eta": float(eta_cpu[batch_index]),
                    "reason": "",
                }
            else:
                out[out_index] = {
                    "hit_type": "none",
                    "hit_id": None,
                    "hit_slot": None,
                    "eta": None,
                    "reason": "no_planet_contact_before_environment_or_horizon",
                }
        if any(hit is None for hit in out):
            raise RuntimeError("Internal error: batched first-hit result was not populated for every launch.")
        return [hit for hit in out if hit is not None]


def first_hit_for_launch(obs: dict[str, Any], player_id: int, move: dict[str, Any], *, horizon: int, device: str = "cpu") -> dict[str, Any]:
    return ExactTargetSimulator(horizon=horizon, device=device).first_hit_for_launch(obs, player_id, move)
