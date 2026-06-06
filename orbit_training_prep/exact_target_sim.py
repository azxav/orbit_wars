from __future__ import annotations

from typing import Any

import torch

from .geometry_bridge import make_geometry
from .schema import NOOP_TARGET_ID, build_planet_slot_maps, safe_float
from orbit_geometry_skeleton.geometry_skeleton import fleet_speed

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
LAUNCH_SURFACE_OFFSET = 0.1
BIG = 1_000_000.0


def point_segment_hits_sun(
    old_x: torch.Tensor,
    old_y: torch.Tensor,
    new_x: torch.Tensor,
    new_y: torch.Tensor,
) -> torch.Tensor:
    vx = new_x - old_x
    vy = new_y - old_y
    wx = CENTER - old_x
    wy = CENTER - old_y
    vv = (vx * vx + vy * vy).clamp(min=1e-12)
    t = ((wx * vx + wy * vy) / vv).clamp(0.0, 1.0)
    cx = old_x + t * vx
    cy = old_y + t * vy
    return (cx - CENTER) ** 2 + (cy - CENTER) ** 2 < SUN_RADIUS * SUN_RADIUS


def swept_circle_hit(
    ax: torch.Tensor,
    ay: torch.Tensor,
    bx: torch.Tensor,
    by: torch.Tensor,
    p0x: torch.Tensor,
    p0y: torch.Tensor,
    p1x: torch.Tensor,
    p1y: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = bx - ax - (p1x - p0x)
    dvy = by - ay - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - radius * radius
    near_static = a < 1e-12
    disc = b * b - 4.0 * a * c
    safe_a = torch.where(near_static, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    return torch.where(near_static, c <= 0.0, (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0))


class ExactTargetSimulator:
    def __init__(self, *, horizon: int = 160, device: str = "cpu"):
        self.horizon = int(horizon)
        self.device = device

    def first_hit_for_launch(self, obs: dict[str, Any], player_id: int, move: dict[str, Any]) -> dict[str, Any]:
        id_to_slot, _ = build_planet_slot_maps(obs)
        source_id = int(move["source_planet_id"])
        source_slot = id_to_slot.get(source_id, int(move.get("source_slot", -1)))
        if source_slot < 0 or source_slot >= len(obs.get("planets", [])):
            return {"hit_type": "invalid", "hit_id": None, "hit_slot": None, "eta": None, "reason": "source_not_found"}

        geometry = make_geometry(horizon=self.horizon, device=self.device)
        obs_tensors = geometry.obs_to_tensors(obs, player_id=player_id)
        movement = geometry.build_or_update_movement(obs_tensors)
        device = movement.device
        dtype = movement.dtype

        angle = torch.tensor([safe_float(move.get("raw_angle"))], dtype=dtype, device=device)
        ships = torch.tensor([max(1.0, safe_float(move.get("ships"), 1.0))], dtype=dtype, device=device)
        speed = fleet_speed(ships).clamp(min=1e-6)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        source = obs["planets"][source_slot]
        sx, sy = movement.position_at_slots(torch.tensor([source_slot], dtype=torch.long, device=device), 0)
        sr = torch.tensor([safe_float(source[4])], dtype=dtype, device=device)
        launch_x = sx + cos_a * (sr + LAUNCH_SURFACE_OFFSET)
        launch_y = sy + sin_a * (sr + LAUNCH_SURFACE_OFFSET)

        px = movement.x[: self.horizon + 1]
        py = movement.y[: self.horizon + 1]
        radii = movement.radii
        alive0 = movement.alive_at(0)
        planet_ids = movement.planet_ids.long().tolist()
        comet_ids = {int(x) for x in obs.get("comet_planet_ids", []) if int(x) >= 0}

        k = torch.arange(self.horizon + 1, dtype=dtype, device=device)
        fx = launch_x.view(1, 1) + cos_a.view(1, 1) * speed.view(1, 1) * k.view(1, -1)
        fy = launch_y.view(1, 1) + sin_a.view(1, 1) * speed.view(1, 1) * k.view(1, -1)

        step_axis = torch.arange(1, self.horizon + 1, dtype=dtype, device=device)
        hit = swept_circle_hit(
            fx[:, :-1].unsqueeze(-1),
            fy[:, :-1].unsqueeze(-1),
            fx[:, 1:].unsqueeze(-1),
            fy[:, 1:].unsqueeze(-1),
            px[:-1].unsqueeze(0),
            py[:-1].unsqueeze(0),
            px[1:].unsqueeze(0),
            py[1:].unsqueeze(0),
            radii.view(1, 1, -1),
        )[0]
        hit = hit & alive0.view(1, -1)
        hit_step_by_slot = torch.where(hit, step_axis.view(-1, 1), torch.full_like(hit, BIG, dtype=dtype)).amin(0)
        first_planet_step = float(hit_step_by_slot.amin().item())
        first_planet_slot = int(torch.where(hit_step_by_slot == hit_step_by_slot.amin(), torch.arange(len(hit_step_by_slot), device=device), torch.full_like(hit_step_by_slot.long(), 10_000)).amin().item())

        nfx = fx[0, 1:]
        nfy = fy[0, 1:]
        ofx = fx[0, :-1]
        ofy = fy[0, :-1]
        bounds = (nfx < 0.0) | (nfx > BOARD_SIZE) | (nfy < 0.0) | (nfy > BOARD_SIZE)
        sun = point_segment_hits_sun(ofx, ofy, nfx, nfy)
        env_step_by_kind = {
            "sun": float(torch.where(sun, step_axis, torch.full_like(step_axis, BIG)).amin().item()),
            "bounds": float(torch.where(bounds, step_axis, torch.full_like(step_axis, BIG)).amin().item()),
        }
        first_env_type = min(env_step_by_kind, key=env_step_by_kind.get)
        first_env_step = env_step_by_kind[first_env_type]

        if first_planet_step <= first_env_step and first_planet_step < BIG:
            pid = int(planet_ids[first_planet_slot])
            return {
                "hit_type": "comet" if pid in comet_ids else "planet",
                "hit_id": pid,
                "hit_slot": first_planet_slot,
                "eta": first_planet_step,
                "reason": "",
            }
        if first_env_step < BIG:
            return {"hit_type": first_env_type, "hit_id": None, "hit_slot": None, "eta": first_env_step, "reason": ""}
        return {"hit_type": "none", "hit_id": None, "hit_slot": None, "eta": None, "reason": "no_collision_within_horizon"}


def first_hit_for_launch(obs: dict[str, Any], player_id: int, move: dict[str, Any], *, horizon: int, device: str = "cpu") -> dict[str, Any]:
    return ExactTargetSimulator(horizon=horizon, device=device).first_hit_for_launch(obs, player_id, move)
