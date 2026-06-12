from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from orbit_jax_env.config import EnvConfig, MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
from orbit_jax_env.jax_policy import greedy_actions
from orbit_jax_env.observation import build_observation
from orbit_jax_env.reset import reset
from orbit_jax_env.simple_heuristic_jax import simple_heuristic_actions
from orbit_jax_env.step import step

from .pfsp import OPP_FROZEN_POLICY, OPP_JAX_PROXY, OPP_NONE, OPP_SIMPLE_HEURISTIC, PFSPEntry, PFSPManifest
from .pfsp_bank import PFSPBank, tree_take
from .train import _policy_greedy_act


def _active_eval_entries(manifest: PFSPManifest) -> list[PFSPEntry]:
    return [entry for entry in manifest.entries if entry.active]


def _rows_for_entry(
    *,
    params: dict[str, Any],
    bc_config: dict[str, Any],
    bank: PFSPBank,
    opponent_kind: jnp.ndarray,
    frozen_slot: jnp.ndarray,
    state,
    source_cap: int,
) -> jnp.ndarray:
    learner_seat = jnp.asarray(0, dtype=jnp.int32)
    learner_rows = _policy_greedy_act(params["bc"], state, learner_seat, bc_config, source_cap)
    simple_rows = simple_heuristic_actions(state)
    proxy_rows = greedy_actions(build_observation(state)["planets"], state.num_players)

    def one_seat(seat):
        frozen_params = tree_take(bank.bc_params, jnp.maximum(frozen_slot, 0))
        frozen_rows = _policy_greedy_act(frozen_params, state, seat, bc_config, source_cap)
        opponent_rows = jax.lax.switch(
            opponent_kind,
            [
                lambda: jnp.zeros((MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32),
                lambda: simple_rows[seat],
                lambda: proxy_rows[seat],
                lambda: frozen_rows,
            ],
        )
        return jnp.where(seat == learner_seat, learner_rows, opponent_rows)

    return jax.vmap(one_seat)(jnp.arange(MAX_PLAYERS, dtype=jnp.int32))


def _entry_kind_slot(entry: PFSPEntry) -> tuple[int, int]:
    if entry.kind == "simple_heuristic_jax":
        return OPP_SIMPLE_HEURISTIC, -1
    if entry.kind == "jax_proxy":
        return OPP_JAX_PROXY, -1
    if entry.kind == "frozen_policy":
        return OPP_FROZEN_POLICY, int(entry.slot if entry.slot is not None else 0)
    return OPP_NONE, -1


def _evaluate_games(
    *,
    params: dict[str, Any],
    bc_config: dict[str, Any],
    bank: PFSPBank,
    config: Any,
    key: Any,
    opponent_kind: int,
    frozen_slot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    games = int(getattr(config, "pfsp_matrix_games", 0))
    players = int(getattr(config, "players", 2))
    env_config = EnvConfig(
        num_players=players,
        episode_steps=int(getattr(config, "episode_steps", 500)),
        enable_comets=bool(getattr(config, "enable_comets", False)),
    )
    source_cap = int(getattr(config, "source_cap", 32))
    episode_steps_limit = int(getattr(config, "episode_steps", 500))
    opponent_kind_jax = jnp.asarray(opponent_kind, dtype=jnp.int32)
    frozen_slot_jax = jnp.asarray(frozen_slot, dtype=jnp.int32)

    def one_game(game_key):
        state = reset(game_key, env_config)

        def body(carry, _step_i):
            state, done, final_reward, final_rank, invalid, submitted, final_step = carry

            def active_step(active_state):
                actions = _rows_for_entry(
                    params=params,
                    bc_config=bc_config,
                    bank=bank,
                    opponent_kind=opponent_kind_jax,
                    frozen_slot=frozen_slot_jax,
                    state=active_state,
                    source_cap=source_cap,
                )
                next_state, _obs, reward_vec, next_done, info = step(active_state, actions)
                return (
                    next_state,
                    next_done,
                    reward_vec[0],
                    info["ranks"][0].astype(jnp.float32),
                    info["invalid_action_count"].astype(jnp.float32),
                    info["submitted_action_count"].astype(jnp.float32),
                    next_state.step.astype(jnp.float32),
                )

            stepped = jax.lax.cond(
                done,
                lambda s: (
                    s,
                    done,
                    final_reward,
                    final_rank,
                    jnp.asarray(0.0, dtype=jnp.float32),
                    jnp.asarray(0.0, dtype=jnp.float32),
                    final_step,
                ),
                active_step,
                state,
            )
            next_state, next_done, reward, rank, invalid_inc, submitted_inc, step_value = stepped
            return (
                next_state,
                next_done,
                reward,
                rank,
                invalid + invalid_inc,
                submitted + submitted_inc,
                step_value,
            ), None

        initial = (
            state,
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(float(players - 1), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        final, _ = jax.lax.scan(body, initial, jnp.arange(episode_steps_limit, dtype=jnp.int32))
        _state, _done, reward, rank, invalid, submitted, final_step = final
        if players == 2:
            result = (rank == 0.0).astype(jnp.float32)
        else:
            result = (float(players - 1) - rank) / float(max(players - 1, 1))
        return reward, rank, result, invalid, submitted, final_step

    if games > 0:
        game_keys = jax.random.split(key, games)
        rewards, ranks, results, invalids, submitted_counts, steps = jax.jit(jax.vmap(one_game))(game_keys)
        return (
            np.asarray(rewards),
            np.asarray(ranks),
            np.asarray(results),
            float(np.sum(np.asarray(invalids))),
            float(np.sum(np.asarray(submitted_counts))),
            np.asarray(steps),
        )
    return (
        np.asarray([], dtype=np.float32),
        np.asarray([], dtype=np.float32),
        np.asarray([], dtype=np.float32),
        0.0,
        0.0,
        np.asarray([], dtype=np.float32),
    )


def _evaluate_entry(
    *,
    params: dict[str, Any],
    bc_config: dict[str, Any],
    bank: PFSPBank,
    entry: PFSPEntry,
    config: Any,
    key: Any,
) -> dict[str, Any]:
    games = int(getattr(config, "pfsp_matrix_games", 0))
    players = int(getattr(config, "players", 2))
    opponent_kind, frozen_slot = _entry_kind_slot(entry)
    rewards_np, ranks_np, results_np, invalid, submitted, steps_np = _evaluate_games(
        params=params,
        bc_config=bc_config,
        bank=bank,
        config=config,
        key=key,
        opponent_kind=opponent_kind,
        frozen_slot=frozen_slot,
    )

    average_result = float(np.mean(results_np)) if results_np.size else 0.0
    return {
        "entry_id": entry.id,
        "kind": entry.kind,
        "games": games,
        "average_result": average_result,
        "average_reward": float(np.mean(rewards_np)) if rewards_np.size else 0.0,
        "average_rank": float(np.mean(ranks_np)) if ranks_np.size else 0.0,
        "win_rate_2p": average_result if players == 2 else 0.0,
        "invalid_action_rate": float(invalid / submitted) if submitted > 0.0 else 0.0,
        "average_episode_step": float(np.mean(steps_np)) if steps_np.size else 0.0,
    }


def evaluate_matrix(
    *,
    params: dict[str, Any],
    bc_config: dict[str, Any],
    bank: PFSPBank,
    manifest: PFSPManifest,
    config: Any,
    key: Any,
    out_dir: Path,
) -> dict[str, Any]:
    league_dir = Path(out_dir) / "league"
    league_dir.mkdir(parents=True, exist_ok=True)
    entries = _active_eval_entries(manifest)
    rows = [
        _evaluate_entry(
            params=params,
            bc_config=bc_config,
            bank=bank,
            entry=entry,
            config=config,
            key=jax.random.fold_in(key, idx),
        )
        for idx, entry in enumerate(entries)
    ]
    matrix_score = (
        sum(float(row["average_result"]) for row in rows) / float(len(rows))
        if rows
        else 0.0
    )
    summary = {
        "players": int(manifest.players),
        "matrix_games": int(getattr(config, "pfsp_matrix_games", 0)),
        "matrix_score": matrix_score,
        "rows": rows,
    }
    with open(league_dir / "eval_matrix.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    lines = [
        "| opponent | kind | games | average_result | average_reward | average_rank | invalid_action_rate | average_episode_step |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['entry_id']} | {row['kind']} | {row['games']} | "
            f"{row['average_result']:.4f} | {row['average_reward']:.4f} | {row['average_rank']:.4f} | "
            f"{row['invalid_action_rate']:.6f} | {row['average_episode_step']:.2f} |"
        )
    with open(league_dir / "eval_matrix.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary
