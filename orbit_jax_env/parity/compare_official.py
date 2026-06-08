from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import jax.numpy as jnp
import numpy as np

from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
from orbit_jax_env.state import EnvState, state_from_observation
from orbit_jax_env.step import step

from .scripted_actions import (
    launch_outward_to_bounds,
    launch_toward_sun,
    moving_planet_intercept,
    nearest_neutral_capture,
    no_actions,
    opposing_neutral_attack,
    random_scripted_actions,
)
from .parity_cases import PARITY_CASES

POSITION_ABS_TOL = 1.0e-4
FLEET_POSITION_ABS_TOL = 2.0e-4


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "keys"):
        return {k: _plain(value[k]) for k in value.keys()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _state_value(state: Any, name: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(name, default)
    return getattr(state, name, default)


def _scripted(scripted, step_idx: int, players: int, obs: dict[str, Any] | None = None):
    try:
        return scripted(step_idx, players, obs)
    except TypeError:
        return scripted(step_idx, players)


def _official_rollout(*, seed: int, players: int, steps: int, scripted) -> list[dict[str, Any]]:
    try:
        from kaggle_environments import make
    except ModuleNotFoundError as exc:
        raise RuntimeError("kaggle_environments is required for parity comparison") from exc

    def make_agent(player_id: int):
        def agent(obs, _config):
            step_idx = int((obs or {}).get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0))
            return _scripted(scripted, step_idx, players, _plain(obs))[player_id]

        return agent

    env = make("orbit_wars", configuration={"episodeSteps": max(int(steps) + 4, 20), "seed": int(seed)}, debug=False)
    env.run([make_agent(i) for i in range(int(players))])
    observations: list[dict[str, Any]] = []
    for frame in env.steps[: int(steps) + 1]:
        obs = _state_value(frame[0], "observation", {})
        observations.append(_plain(obs))
    return observations


def _empty_jax_actions() -> jnp.ndarray:
    return jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)


def _jax_rollout_from_initial(
    initial_obs: dict[str, Any],
    official_observations: list[dict[str, Any]],
    *,
    players: int,
    steps: int,
    actions_by_step,
) -> list[EnvState]:
    state = state_from_observation(initial_obs, num_players=int(players), episode_steps=max(int(steps) + 4, 20))
    states = [state]
    for step_idx in range(int(steps)):
        actions = _empty_jax_actions()
        scripted_actions = _scripted(actions_by_step, step_idx, players, official_observations[step_idx])
        for player, moves in enumerate(scripted_actions[:MAX_PLAYERS]):
            for action_idx, move in enumerate(moves[:MAX_ACTIONS_PER_PLAYER]):
                actions = actions.at[player, action_idx].set(jnp.asarray(move, dtype=jnp.float32))
        state, _obs, _rewards, _done, _info = step(state, actions)
        states.append(state)
    return states


def _jax_planets(state: EnvState) -> np.ndarray:
    rows = np.stack(
        [
            np.asarray(state.planet_id),
            np.asarray(state.planet_owner),
            np.asarray(state.planet_x),
            np.asarray(state.planet_y),
            np.asarray(state.planet_radius),
            np.asarray(state.planet_ships),
            np.asarray(state.planet_production),
        ],
        axis=-1,
    )
    return rows[np.asarray(state.planet_alive, dtype=bool)]


def _jax_fleets(state: EnvState) -> np.ndarray:
    fleet_ids = np.arange(np.asarray(state.fleet_owner).shape[0])
    rows = np.stack(
        [
            fleet_ids,
            np.asarray(state.fleet_owner),
            np.asarray(state.fleet_x),
            np.asarray(state.fleet_y),
            np.asarray(state.fleet_angle),
            np.asarray(state.fleet_source),
            np.asarray(state.fleet_ships),
        ],
        axis=-1,
    )
    return rows[np.asarray(state.fleet_alive, dtype=bool)]


def _compare_frame(official_obs: dict[str, Any], jax_state: EnvState) -> dict[str, Any]:
    official_planets = np.asarray(official_obs.get("planets", []), dtype=np.float64).reshape((-1, 7))
    jax_planets = _jax_planets(jax_state).astype(np.float64)
    n_planets = min(len(official_planets), len(jax_planets))
    if n_planets:
        pos_err = float(
            np.max(
                np.abs(
                    official_planets[:n_planets, 2:4]
                    - jax_planets[:n_planets, 2:4]
                )
            )
        )
        owner_mismatches = int(np.sum(official_planets[:n_planets, 1].astype(int) != jax_planets[:n_planets, 1].astype(int)))
        ship_mismatches = int(np.sum(np.rint(official_planets[:n_planets, 5]).astype(int) != np.rint(jax_planets[:n_planets, 5]).astype(int)))
    else:
        pos_err = 0.0
        owner_mismatches = 0
        ship_mismatches = 0

    official_fleets = np.asarray(official_obs.get("fleets", []), dtype=np.float64).reshape((-1, 7))
    jax_fleets = _jax_fleets(jax_state).astype(np.float64)
    fleet_count_mismatch = abs(int(len(official_fleets)) - int(len(jax_fleets)))
    fleet_mismatches = fleet_count_mismatch
    fleet_position_error = 0.0
    n_fleets = min(len(official_fleets), len(jax_fleets))
    if n_fleets:
        official_key = np.lexsort(
            (
                np.round(official_fleets[:, 4], 6),
                official_fleets[:, 6],
                official_fleets[:, 5],
                official_fleets[:, 1],
            )
        )
        jax_key = np.lexsort(
            (
                np.round(jax_fleets[:, 4], 6),
                jax_fleets[:, 6],
                jax_fleets[:, 5],
                jax_fleets[:, 1],
            )
        )
        official_sorted = official_fleets[official_key][:n_fleets]
        jax_sorted = jax_fleets[jax_key][:n_fleets]
        fleet_mismatches += int(np.sum(official_sorted[:, 1].astype(int) != jax_sorted[:, 1].astype(int)))
        fleet_mismatches += int(np.sum(official_sorted[:, 5].astype(int) != jax_sorted[:, 5].astype(int)))
        fleet_mismatches += int(np.sum(np.rint(official_sorted[:, 6]).astype(int) != np.rint(jax_sorted[:, 6]).astype(int)))
        fleet_mismatches += int(np.sum(np.abs(official_sorted[:, 4] - jax_sorted[:, 4]) > POSITION_ABS_TOL))
        fleet_position_error = float(np.max(np.abs(official_sorted[:, 2:4] - jax_sorted[:, 2:4])))
    return {
        "max_position_abs_error": pos_err,
        "owner_mismatches": owner_mismatches + abs(int(len(official_planets)) - int(len(jax_planets))),
        "ship_mismatches": ship_mismatches,
        "fleet_count_mismatch": fleet_count_mismatch,
        "fleet_mismatches": fleet_mismatches,
        "fleet_position_abs_error": fleet_position_error,
    }


def _run_imported_comet_movement_case(*, seed: int, players: int) -> dict[str, Any]:
    official = _official_rollout(seed=seed, players=players, steps=51, scripted=no_actions)
    obs_50 = official[50]
    obs_51 = official[51]
    state = state_from_observation(obs_50, num_players=int(players), episode_steps=60)
    next_state, _obs, _rewards, _done, _info = step(state, _empty_jax_actions())
    frame = _compare_frame(obs_51, next_state)
    passed = (
        frame["max_position_abs_error"] <= POSITION_ABS_TOL
        and frame["owner_mismatches"] == 0
        and frame["ship_mismatches"] == 0
        and frame["fleet_count_mismatch"] == 0
        and frame["fleet_mismatches"] == 0
        and frame["fleet_position_abs_error"] <= FLEET_POSITION_ABS_TOL
    )
    return {
        "implemented": True,
        "passed": bool(passed),
        "seed": int(seed),
        "players": int(players),
        "steps": 1,
        "source_step": 50,
        "comet_planets": int(len(obs_50.get("comet_planet_ids", []))),
        "max_position_abs_error": frame["max_position_abs_error"],
        "owner_mismatches": frame["owner_mismatches"],
        "ship_mismatches": frame["ship_mismatches"],
        "fleet_count_mismatch": frame["fleet_count_mismatch"],
        "fleet_mismatches": frame["fleet_mismatches"],
        "fleet_position_abs_error": frame["fleet_position_abs_error"],
        "position_abs_tol": POSITION_ABS_TOL,
        "fleet_position_abs_tol": FLEET_POSITION_ABS_TOL,
    }


def run_parity_case(name: str, *, seed: int | None = None, players: int | None = None, steps: int | None = None) -> dict[str, Any]:
    if name == "case_010_imported_comet_movement":
        if steps not in (None, 1):
            return {"implemented": False, "passed": False, "reason": "case uses a fixed one-step comet movement comparison"}
        return _run_imported_comet_movement_case(
            seed=123 if seed is None else int(seed),
            players=2 if players is None else int(players),
        )

    case_scripts = {
        "case_001_no_actions": no_actions,
        "case_002_simple_capture_static": nearest_neutral_capture,
        "case_003_sun_collision": launch_toward_sun,
        "case_004_bounds_collision": launch_outward_to_bounds,
        "case_005_two_fleets_combat": opposing_neutral_attack,
        "case_006_planet_rotation": no_actions,
        "case_007_moving_planet_collision": moving_planet_intercept,
        "case_008_random_scripted_2p_50_steps": random_scripted_actions,
        "case_009_random_scripted_4p_50_steps": random_scripted_actions,
    }
    if name not in case_scripts:
        return {"implemented": False, "passed": False, "reason": "case not implemented yet"}

    if players is None:
        players = 4 if name == "case_009_random_scripted_4p_50_steps" else 2
    if steps is None:
        steps = {
            "case_001_no_actions": 20,
            "case_002_simple_capture_static": 12,
            "case_003_sun_collision": 30,
            "case_004_bounds_collision": 30,
            "case_005_two_fleets_combat": 30,
            "case_006_planet_rotation": 20,
            "case_007_moving_planet_collision": 30,
            "case_008_random_scripted_2p_50_steps": 49,
            "case_009_random_scripted_4p_50_steps": 49,
        }[name]
    if seed is None:
        seed = {
            "case_001_no_actions": 123,
            "case_002_simple_capture_static": 104,
            "case_003_sun_collision": 123,
            "case_004_bounds_collision": 123,
            "case_005_two_fleets_combat": 140,
            "case_006_planet_rotation": 321,
            "case_007_moving_planet_collision": 222,
            "case_008_random_scripted_2p_50_steps": 808,
            "case_009_random_scripted_4p_50_steps": 909,
        }[name]
    scripted = case_scripts[name]
    official = _official_rollout(seed=seed, players=players, steps=steps, scripted=scripted)
    jax_states = _jax_rollout_from_initial(official[0], official, players=players, steps=steps, actions_by_step=scripted)
    frames = [_compare_frame(o, s) for o, s in zip(official, jax_states)]
    max_position = max((f["max_position_abs_error"] for f in frames), default=0.0)
    owner_mismatches = sum(int(f["owner_mismatches"]) for f in frames)
    ship_mismatches = sum(int(f["ship_mismatches"]) for f in frames)
    fleet_count_mismatch = sum(int(f["fleet_count_mismatch"]) for f in frames)
    fleet_mismatches = sum(int(f["fleet_mismatches"]) for f in frames)
    fleet_position = max((f["fleet_position_abs_error"] for f in frames), default=0.0)
    passed = (
        max_position <= POSITION_ABS_TOL
        and owner_mismatches == 0
        and ship_mismatches == 0
        and fleet_count_mismatch == 0
        and fleet_mismatches == 0
        and fleet_position <= FLEET_POSITION_ABS_TOL
    )
    return {
        "implemented": True,
        "passed": bool(passed),
        "seed": int(seed),
        "players": int(players),
        "steps": int(steps),
        "max_position_abs_error": max_position,
        "owner_mismatches": owner_mismatches,
        "ship_mismatches": ship_mismatches,
        "fleet_count_mismatch": fleet_count_mismatch,
        "fleet_mismatches": fleet_mismatches,
        "fleet_position_abs_error": fleet_position,
        "position_abs_tol": POSITION_ABS_TOL,
        "fleet_position_abs_tol": FLEET_POSITION_ABS_TOL,
    }


def run_parity_report(output: str | Path = "orbit_jax_env/parity_report.json", cases: Iterable[str] | None = None) -> dict:
    selected = list(cases or PARITY_CASES)
    case_results = {name: run_parity_case(name) for name in selected}
    all_passed = all(bool(result.get("passed")) for result in case_results.values())
    any_unimplemented = any(not bool(result.get("implemented")) for result in case_results.values())
    report = {
        "status": "pass" if all_passed else ("partial" if any_unimplemented else "fail"),
        "note": "No-comet parity cases and imported comet movement parity are implemented; JAX-native comet spawn uses deterministic generated paths.",
        "cases": case_results,
    }
    out = Path(output)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="orbit_jax_env/parity_report.json")
    parser.add_argument("--case", action="append", dest="cases", help="Limit report to one or more parity case names.")
    args = parser.parse_args()
    print(json.dumps(run_parity_report(args.output, cases=args.cases), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
