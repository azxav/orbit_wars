"""
Reusable geometry/mechanics skeleton extracted from the open best heuristic agent.

Purpose:
- Keep deterministic game physics outside BC/RL.
- Let ML decide source/target/amount/priority.
- Let this skeleton compute angle, ETA, feasibility, movement projection, collision, and launch simulation.

This module intentionally exposes only geometry/mechanics APIs, not the heuristic policy planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from . import open_best_heuristic_original as h


@dataclass(frozen=True)
class GeometryConfig:
    """Configuration for deterministic geometry projection."""
    movement_horizon: int = h.DEFAULT_MOVEMENT_HORIZON
    drift_epsilon: float = h.DEFAULT_DRIFT_EPSILON
    track_fleets: bool = True
    player_count: int | None = None
    max_tracked_fleets: int = h.DEFAULT_MAX_TRACKED_FLEETS
    P: int = h.P_MAX
    F: int = h.F_MAX
    device: str = "cpu"


class GeometrySkeleton:
    """
    Deterministic geometry layer for BC/RL.

    ML should output structured decisions, e.g.:
        source_slot, target_slot, ship_count

    This layer converts those decisions into environment actions:
        [from_planet_id, angle, num_ships]

    Main responsibilities:
    - observation tensorization
    - future planet/comet position projection
    - fleet speed / travel time support
    - intercept angle solving
    - first-contact / sun / bounds / collision feasibility
    - planned launch inference
    """

    def __init__(self, config: GeometryConfig | None = None):
        self.config = config or GeometryConfig()
        self._movement: h.PlanetMovement | None = None

    @property
    def movement(self) -> h.PlanetMovement | None:
        return self._movement

    def obs_to_tensors(self, obs: dict[str, Any], *, player_id: int | None = None) -> dict[str, Any]:
        pid = int(obs.get("player", 0) if player_id is None else player_id)
        return h.single_obs_to_tensor(
            obs,
            player_id=pid,
            P=int(self.config.P),
            F=int(self.config.F),
            device=self.config.device,
        )

    def build_or_update_movement(self, obs_tensors: dict[str, Any]) -> h.PlanetMovement:
        expected_cfg = h.MovementConfig(
            movement_horizon=int(self.config.movement_horizon),
            drift_epsilon=float(self.config.drift_epsilon),
            track_fleets=bool(self.config.track_fleets),
            player_count=self.config.player_count,
            max_tracked_fleets=int(self.config.max_tracked_fleets),
        )
        self._movement = h.ensure_planet_movement(
            obs_tensors=obs_tensors,
            expected_cfg=expected_cfg,
            cached_movement=self._movement,
        )
        return self._movement

    def aim_source_to_target(
        self,
        *,
        source_slots: Tensor,
        target_slots: Tensor,
        fleet_sizes: Tensor,
        movement: h.PlanetMovement | None = None,
        active: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Return deterministic angle/ETA/viability for source-target-amount decisions.

        Output keys:
        - angle: angle in radians to emit in Orbit Wars move
        - eta: estimated contact turn if viable, inf otherwise
        - viable: True iff first contact is the intended target before sun/bounds/other planet
        """
        mv = movement or self._movement
        if mv is None:
            raise RuntimeError("Movement is not built. Call build_or_update_movement first.")
        return h.intercept_angle(
            mv,
            source_slots=source_slots,
            target_slots=target_slots,
            fleet_sizes=fleet_sizes,
            active=active,
        )

    def entries_to_planned_launches(
        self,
        *,
        obs_tensors: dict[str, Any],
        source_slots: Tensor,
        target_slots: Tensor,
        ships: Tensor,
        movement: h.PlanetMovement | None = None,
        player_id: int | None = None,
        valid: Tensor | None = None,
    ) -> h.PlannedLaunches:
        """
        Convert ML decisions into simulated launches with target/ETA confirmation.

        This is useful for dataset construction and for checking whether a replay/model
        action really lands on the intended target.
        """
        mv = movement or self._movement
        if mv is None:
            raise RuntimeError("Movement is not built. Call build_or_update_movement first.")
        aim = self.aim_source_to_target(
            source_slots=source_slots,
            target_slots=target_slots,
            fleet_sizes=ships,
            movement=mv,
            active=valid,
        )
        valid_mask = aim["viable"] if valid is None else (valid.to(torch.bool) & aim["viable"])
        entries = h.LaunchEntries(
            source_slots=source_slots.to(torch.long),
            target_slots=target_slots.to(torch.long),
            ships=ships.to(mv.dtype),
            angle=aim["angle"].to(mv.dtype),
            eta=aim["eta"].to(mv.dtype),
            valid=valid_mask.to(torch.bool),
        )
        pid = int(obs_tensors["player"].flatten()[0].item() if player_id is None else player_id)
        return h.infer_planned_launches_from_entries(
            obs_tensors=obs_tensors,
            movement=mv,
            entries=entries,
            player_id=pid,
        )

    def to_env_moves(
        self,
        *,
        obs: dict[str, Any],
        source_slots: Tensor,
        target_slots: Tensor,
        ships: Tensor,
        player_id: int | None = None,
        valid: Tensor | None = None,
    ) -> list[list[Any]]:
        """
        Convert source/target/amount decisions to Orbit Wars move list.

        Returned format:
            [[from_planet_id, angle_in_radians, num_ships], ...]
        """
        obs_tensors = self.obs_to_tensors(obs, player_id=player_id)
        mv = self.build_or_update_movement(obs_tensors)
        launches = self.entries_to_planned_launches(
            obs_tensors=obs_tensors,
            movement=mv,
            source_slots=source_slots,
            target_slots=target_slots,
            ships=ships,
            player_id=player_id,
            valid=valid,
        )
        counts = launches.valid.to(torch.long).sum()
        from_planet_id = mv.planet_ids[launches.source_slots.clamp(0, mv.P - 1)]
        payload = {
            "from_planet_id": from_planet_id[launches.valid],
            "angle": launches.angle[launches.valid],
            "num_ships": launches.ships[launches.valid],
            "counts": counts,
        }
        return h.sparse_action_row_to_moves(payload, obs, player_id=int(obs_tensors["player"].item()))


# Explicitly re-export deterministic primitives for dataset construction.
fleet_speed = h.fleet_speed
parse_obs = h.parse_obs
MovementConfig = h.MovementConfig
PlanetMovement = h.PlanetMovement
DistanceCache = h.DistanceCache
build_distance_cache = h.build_distance_cache
min_distance_to_targets = h.min_distance_to_targets
intercept_angle = h.intercept_angle
analytic_first_contact = h._analytic_first_contact
point_to_segment_distance_sq = h._point_to_segment_distance_sq
swept_pair_hit_mask = h._swept_pair_hit_mask
swept_pair_hit_mask_mv = h._swept_pair_hit_mask_mv
LaunchEntries = h.LaunchEntries
PlannedLaunches = h.PlannedLaunches
infer_planned_launches_from_entries = h.infer_planned_launches_from_entries
single_obs_to_tensor = h.single_obs_to_tensor
capture_floor = h.capture_floor
