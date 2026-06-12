from __future__ import annotations

import torch

from orbit_geometry_skeleton import GeometryConfig, GeometrySkeleton

BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
BIG = 1_000_000.0


def make_geometry(*, horizon: int = 160, device: str = "cpu") -> GeometrySkeleton:
    # Long horizon is important: low-ship fleets can need far more than 20 turns.
    return GeometrySkeleton(GeometryConfig(movement_horizon=int(horizon), track_fleets=False, device=device))


def _point_segment_hits_sun(
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


def _swept_circle_hit(
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


def analytic_first_contact(
    *,
    launch_x: torch.Tensor,
    launch_y: torch.Tensor,
    cos_a: torch.Tensor,
    sin_a: torch.Tensor,
    speed: torch.Tensor,
    px: torch.Tensor,
    py: torch.Tensor,
    p_alive0: torch.Tensor,
    radii: torch.Tensor,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first planet-contact slot and ETA for each launch.

    Environment collisions are respected: if the fleet hits the sun or leaves the board
    before the first planet contact, the returned slot is -1.
    """
    n = int(launch_x.numel())
    device = launch_x.device
    dtype = launch_x.dtype
    h = max(1, min(int(H), int(px.shape[0]) - 1, int(py.shape[0]) - 1))
    if n == 0 or h <= 0 or int(radii.numel()) == 0:
        return (
            torch.full((n,), -1, dtype=torch.long, device=device),
            torch.full((n,), BIG, dtype=dtype, device=device),
        )

    k = torch.arange(h + 1, dtype=dtype, device=device).view(1, -1)
    fx = launch_x.view(n, 1) + cos_a.view(n, 1) * speed.view(n, 1) * k
    fy = launch_y.view(n, 1) + sin_a.view(n, 1) * speed.view(n, 1) * k

    step_axis = torch.arange(1, h + 1, dtype=dtype, device=device)
    big_scalar = torch.tensor(BIG, dtype=dtype, device=device)

    planet_hit = _swept_circle_hit(
        fx[:, :-1].unsqueeze(-1),
        fy[:, :-1].unsqueeze(-1),
        fx[:, 1:].unsqueeze(-1),
        fy[:, 1:].unsqueeze(-1),
        px[:-1].unsqueeze(0),
        py[:-1].unsqueeze(0),
        px[1:].unsqueeze(0),
        py[1:].unsqueeze(0),
        radii.view(1, 1, -1),
    )
    planet_hit = planet_hit & p_alive0.to(device=device, dtype=torch.bool).view(1, 1, -1)
    hit_steps = torch.where(planet_hit, step_axis.view(1, -1, 1), big_scalar).amin(dim=1)
    first_planet_eta = hit_steps.amin(dim=1).values
    slot_axis = torch.arange(hit_steps.shape[1], dtype=torch.long, device=device).view(1, -1)
    first_slots = torch.where(
        hit_steps == first_planet_eta.view(-1, 1),
        slot_axis.expand_as(hit_steps),
        torch.full_like(hit_steps, 10_000, dtype=torch.long),
    ).amin(dim=1)

    next_x = fx[:, 1:]
    next_y = fy[:, 1:]
    prev_x = fx[:, :-1]
    prev_y = fy[:, :-1]
    bounds = (next_x < 0.0) | (next_x > BOARD_SIZE) | (next_y < 0.0) | (next_y > BOARD_SIZE)
    sun = _point_segment_hits_sun(prev_x, prev_y, next_x, next_y)
    sun_eta = torch.where(sun, step_axis.view(1, -1), big_scalar).amin(dim=1)
    bounds_eta = torch.where(bounds, step_axis.view(1, -1), big_scalar).amin(dim=1)
    env_eta = torch.minimum(sun_eta, bounds_eta)

    valid_planet = first_planet_eta <= env_eta
    slots = torch.where(valid_planet & (first_planet_eta < BIG), first_slots, torch.full_like(first_slots, -1))
    eta = torch.where(slots >= 0, first_planet_eta, big_scalar)
    return slots.long(), eta
