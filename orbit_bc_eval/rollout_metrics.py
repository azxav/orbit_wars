from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orbit_training_prep.schema import P_MAX, safe_float

from .config import STEP_BUCKETS


def _bucket_key(lo: int, hi: int) -> str:
    return f"launches_{lo}_{hi}"


@dataclass
class RolloutMetrics:
    game_id: str
    bc_player_id: int
    players: int
    opponent: str
    launches: int = 0
    bucket_counts: dict[str, int] = field(default_factory=lambda: {_bucket_key(lo, hi): 0 for lo, hi in STEP_BUCKETS})
    illegal_actions: int = 0
    skipped_invalid_decoded_actions: int = 0
    timeout_count: int = 0
    error_count: int = 0
    no_op_source_decisions: int = 0
    predicted_launches: int = 0
    returned_move_count: int = 0
    skip_reason_counts: dict[str, int] = field(default_factory=dict)
    opening_prediction_target_counts: dict[str, int] = field(default_factory=dict)
    opening_prediction_amount_counts: dict[str, int] = field(default_factory=dict)
    opening_prediction_target_amount_counts: dict[str, int] = field(default_factory=dict)
    sun_bounds_suspected_waste: int = 0
    owned_planets_samples: list[int] = field(default_factory=list)
    total_ships_samples: list[float] = field(default_factory=list)
    sample_steps: list[int] = field(default_factory=list)
    planets_captured: int = 0
    first_capture_step: int | None = None
    _initial_owned_count: int | None = None
    _initial_ship_count: float | None = None
    _last_owned_ids: set[int] = field(default_factory=set)

    def record_observation(self, obs: dict[str, Any]) -> None:
        step = int(obs.get("step", len(self.sample_steps)) or 0)
        owned_ids: set[int] = set()
        total_ships = 0.0
        for p in obs.get("planets", [])[:P_MAX]:
            if len(p) < 7:
                continue
            if int(p[1]) == int(self.bc_player_id):
                owned_ids.add(int(p[0]))
                total_ships += max(0.0, safe_float(p[5]))
        if self._initial_owned_count is None:
            self._initial_owned_count = len(owned_ids)
            self._initial_ship_count = total_ships
        if self._last_owned_ids:
            captured = len(owned_ids - self._last_owned_ids)
            self.planets_captured += captured
            if captured and self.first_capture_step is None:
                self.first_capture_step = step
        self._last_owned_ids = owned_ids
        self.sample_steps.append(step)
        self.owned_planets_samples.append(len(owned_ids))
        self.total_ships_samples.append(total_ships)

    def record_step(self, *, step: int, actions: list[list[Any]], illegal_actions: int, runtime_debug: dict[str, Any] | None) -> None:
        runtime_debug = runtime_debug or {}
        count = len(actions)
        self.launches += count
        self.returned_move_count += int(runtime_debug.get("returned_moves", count) or 0)
        for lo, hi in STEP_BUCKETS:
            if lo <= int(step) < hi:
                self.bucket_counts[_bucket_key(lo, hi)] += count
                break
        self.illegal_actions += int(illegal_actions)
        self.skipped_invalid_decoded_actions += int(runtime_debug.get("skipped_invalid_decoded_actions", 0) or 0)
        self.no_op_source_decisions += int(runtime_debug.get("no_op_source_decisions", 0) or 0)
        self.predicted_launches += int(runtime_debug.get("predicted_launches", 0) or 0)
        for reason, value in (runtime_debug.get("skip_reasons", {}) or {}).items():
            self.skip_reason_counts[str(reason)] = self.skip_reason_counts.get(str(reason), 0) + int(value or 0)
        opening_counts = runtime_debug.get("opening_prediction_counts", {}) or {}
        for key, attr in (
            ("target", "opening_prediction_target_counts"),
            ("amount", "opening_prediction_amount_counts"),
            ("target_amount", "opening_prediction_target_amount_counts"),
        ):
            target = getattr(self, attr)
            for name, value in (opening_counts.get(key, {}) or {}).items():
                target[str(name)] = target.get(str(name), 0) + int(value or 0)
        self.timeout_count += 1 if runtime_debug.get("timeout") else 0
        self.error_count += 1 if runtime_debug.get("error") else 0

    def finalize(self, *, rewards: list[float] | None, statuses: list[str] | None, final_obs: dict[str, Any] | None) -> dict[str, Any]:
        if final_obs is not None:
            self.record_observation(final_obs)
        rewards = [float(r) for r in (rewards or [])]
        reward = rewards[self.bc_player_id] if self.bc_player_id < len(rewards) else 0.0
        rank = 1 + sum(other > reward for other in rewards)
        win = bool(rewards and rank == 1)
        if statuses:
            self.timeout_count += sum(1 for s in statuses if str(s).upper() == "TIMEOUT")
            self.error_count += sum(1 for s in statuses if str(s).upper() == "ERROR")
        avg_owned = sum(self.owned_planets_samples) / len(self.owned_planets_samples) if self.owned_planets_samples else 0.0
        avg_ships = sum(self.total_ships_samples) / len(self.total_ships_samples) if self.total_ships_samples else 0.0
        opening_owned = [v for step, v in zip(self.sample_steps, self.owned_planets_samples) if step < 100]
        opening_ships = [v for step, v in zip(self.sample_steps, self.total_ships_samples) if step < 100]
        final_owned = self.owned_planets_samples[-1] if self.owned_planets_samples else 0
        final_ships = self.total_ships_samples[-1] if self.total_ships_samples else 0.0
        initial_owned = self._initial_owned_count or 0
        initial_ships = self._initial_ship_count or 0.0
        launch_decisions = self.predicted_launches + self.no_op_source_decisions
        return {
            "game_id": self.game_id,
            "bc_seat": int(self.bc_player_id),
            "players": int(self.players),
            "opponent": self.opponent,
            "reward": reward,
            "rank": int(rank),
            "win": win,
            "final_ship_count": final_ships,
            "final_owned_planets": int(final_owned),
            "owned_planets_auc": float(avg_owned),
            "owned_planets_auc_0_100": float(sum(opening_owned) / len(opening_owned) if opening_owned else 0.0),
            "planet_control_delta": int(final_owned - initial_owned),
            "first_capture_step": int(self.first_capture_step) if self.first_capture_step is not None else -1,
            "capture_efficiency": self.planets_captured / max(1, self.launches),
            "total_ships_auc": float(avg_ships),
            "total_ships_auc_0_100": float(sum(opening_ships) / len(opening_ships) if opening_ships else 0.0),
            "ship_delta_final_vs_initial": float(final_ships - initial_ships),
            "launches": int(self.launches),
            **self.bucket_counts,
            "illegal_actions": int(self.illegal_actions),
            "skipped_invalid_decoded_actions": int(self.skipped_invalid_decoded_actions),
            "skip_reason_counts": dict(sorted(self.skip_reason_counts.items())),
            "opening_prediction_target_counts": dict(sorted(self.opening_prediction_target_counts.items())),
            "opening_prediction_amount_counts": dict(sorted(self.opening_prediction_amount_counts.items())),
            "opening_prediction_target_amount_counts": dict(sorted(self.opening_prediction_target_amount_counts.items())),
            "timeout_count": int(self.timeout_count),
            "error_count": int(self.error_count),
            "no_op_source_decisions": int(self.no_op_source_decisions),
            "predicted_launches": int(self.predicted_launches),
            "predicted_launch_rate": self.predicted_launches / max(1, self.predicted_launches + self.no_op_source_decisions),
            "decode_success_rate": self.returned_move_count / max(1, self.predicted_launches),
            "invalid_decode_rate": self.skipped_invalid_decoded_actions / max(1, self.predicted_launches),
            "actual_launch_rate": self.returned_move_count / max(1, launch_decisions),
            "noop_decision_rate": self.no_op_source_decisions / max(1, launch_decisions),
            "actual_returned_move_count": int(self.returned_move_count),
            "sun_bounds_suspected_waste": int(self.sun_bounds_suspected_waste),
            "planets_captured": int(self.planets_captured),
            "avg_owned_planets": float(avg_owned),
            "avg_total_ships": float(avg_ships),
        }
