from __future__ import annotations

from orbit_geometry_skeleton import GeometryConfig, GeometrySkeleton


def make_geometry(*, horizon: int = 160, device: str = "cpu") -> GeometrySkeleton:
    # Long horizon is important: low-ship fleets can need far more than 20 turns.
    return GeometrySkeleton(GeometryConfig(movement_horizon=int(horizon), track_fleets=False, device=device))
