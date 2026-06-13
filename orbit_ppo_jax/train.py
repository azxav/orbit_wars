from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from orbit_jax_env.config import EnvConfig, MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS, P_MAX
from orbit_jax_env.jax_policy import greedy_actions
from orbit_jax_env.observation import build_observation
from orbit_jax_env.reset import reset
from orbit_jax_env.simple_heuristic_jax import simple_heuristic_actions
from orbit_jax_env.step import step
from orbit_jax_env.state import EnvState

from .actions import action_rows_from_source_choices
from .bc_policy import bc_forward, init_value_head, load_bc_jax_params, value_apply
from .checkpointing import load_jax_checkpoint, load_jax_training_state, save_jax_checkpoint, save_jax_training_state
from .features import NOOP_TARGET_SLOT, build_bc_features_for_seat, build_selected_masks_from_activity
from .pfsp import OPP_FROZEN_POLICY, OPP_JAX_PROXY, OPP_NONE, OPP_SIMPLE_HEURISTIC, add_snapshot_entry, build_initial_manifest, build_match_plan, load_manifest, save_manifest, update_manifest_from_slot_stats
from .pfsp_bank import build_pfsp_bank, tree_stack, tree_take

NEG = -1.0e9


@dataclass
class JaxPPOConfig:
    bc_checkpoint: str
    out_dir: str
    players: int = 4
    envs: int = 8
    rollout_steps: int = 32
    episode_steps: int = 500
    updates: int = 1
    opponent: str = "simple_heuristic_jax"
    eval_heuristic_path: str = "orbit_wars_base.py"
    eval_games: int = 2
    eval_interval_updates: int = 5
    seed: int = 42
    require_cuda: bool = False
    lr: float = 2.0e-5
    clip: float = 0.10
    gamma: float = 0.995
    lam: float = 0.95
    ent: float = 0.01
    vf: float = 0.5
    max_grad_norm: float = 0.5
    freeze_bc_steps: int = 0
    save_interval_updates: int = 5
    enable_comets: bool = False
    initial_state_bank: str | None = None
    state_bank_mode: str = "random"
    source_cap: int = 32
    pfsp_enabled: bool = False
    pfsp_max_policy_slots: int = 32
    pfsp_anchor_fraction: float = 0.25
    pfsp_snapshot_interval_updates: int = 10
    pfsp_warmup_updates: int = 10
    pfsp_min_games_per_entry: int = 16
    pfsp_hard_low: float = 0.20
    pfsp_hard_high: float = 0.55
    pfsp_hard_bonus: float = 0.15
    pfsp_exploration_bonus: float = 0.10
    pfsp_matrix_games: int = 16
    pfsp_eval_interval_updates: int = 10
    pfsp_learner_seat_mode: str = "rotate"
    pfsp_4p_layout: str = "one_pfsp_two_anchors"
    resume: bool = True
    resume_from: str | None = None
    precision: str = "bfloat16"
    matmul_precision: str = "highest"
    remat_policy_eval: bool = True
    recompute_masks: bool = True
    profile_dir: str | None = "traces"
    profile_updates: int = 1
    profile_max_env_steps: int = 1024
    async_rollout_prefetch: bool = True


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _check_runtime(require_cuda: bool) -> dict[str, Any]:
    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    if require_cuda and backend not in {"gpu", "cuda"}:
        raise RuntimeError(f"JAX CUDA backend is required, but backend={backend!r}, devices={devices}")
    return {"jax_backend": backend, "jax_devices": devices}


def _apply_jax_precision_config(config: JaxPPOConfig) -> None:
    if config.matmul_precision == "default":
        return
    jax.config.update("jax_default_matmul_precision", config.matmul_precision)


def _bc_compute_dtype_name(precision: str) -> str:
    name = str(precision).lower()
    if name in {"float32", "bfloat16", "float16"}:
        return name
    raise RuntimeError("--precision must be float32, bfloat16, or float16")


def _compute_dtype(precision: str) -> jnp.dtype:
    name = _bc_compute_dtype_name(precision)
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float16":
        return jnp.float16
    return jnp.float32


def _profile_update_context(config: JaxPPOConfig, out_dir: Path, update_index: int):
    if not config.profile_dir or int(update_index) > int(config.profile_updates):
        return nullcontext()
    profile_limit = int(config.profile_max_env_steps)
    update_env_steps = int(config.envs) * int(config.rollout_steps)
    if profile_limit > 0 and update_env_steps > profile_limit:
        return nullcontext()
    trace_dir = Path(config.profile_dir)
    if not trace_dir.is_absolute():
        trace_dir = out_dir / trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    return jax.profiler.trace(str(trace_dir))


def _safe_target_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    row_any = jnp.any(mask, axis=-1, keepdims=True)
    noop = jnp.arange(mask.shape[-1]) == NOOP_TARGET_SLOT
    safe_mask = jnp.where(row_any, mask, noop[None, :])
    return jnp.where(safe_mask, logits, NEG)


def _safe_amount_logits(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    row_any = jnp.any(mask, axis=-1, keepdims=True)
    none = jnp.arange(mask.shape[-1]) == 0
    safe_mask = jnp.where(row_any, mask, none[None, :])
    return jnp.where(safe_mask, logits, NEG)


def _source_batch(features, target_label: jnp.ndarray | None = None) -> dict[str, jnp.ndarray]:
    p = features.source_slots.shape[0]
    batch = {
        "planet_features": jnp.broadcast_to(features.planet_features[None, :, :], (p, *features.planet_features.shape)),
        "global_features": jnp.broadcast_to(features.global_features[None, :], (p, features.global_features.shape[0])),
        "target_state_features": jnp.broadcast_to(features.target_state_features[None, :, :], (p, *features.target_state_features.shape)),
        "pair_features": features.pair_features,
        "source_slot": features.source_slots.astype(jnp.int32),
    }
    if target_label is not None:
        batch["target_label"] = target_label.astype(jnp.int32)
    return batch


def _explained_variance(values: jnp.ndarray, returns: jnp.ndarray) -> jnp.ndarray:
    var_returns = jnp.var(returns)
    return jnp.where(var_returns < 1.0e-8, 0.0, 1.0 - jnp.var(returns - values) / var_returns)


def _ppo_diagnostic_metrics(
    *,
    old_logprob: jnp.ndarray,
    new_logprob: jnp.ndarray,
    values: jnp.ndarray,
    returns: jnp.ndarray,
    clip_range: float,
) -> dict[str, jnp.ndarray]:
    ratio = jnp.exp(new_logprob - old_logprob)
    return {
        "approx_kl": jnp.mean(old_logprob - new_logprob),
        "clip_frac": jnp.mean((jnp.abs(ratio - 1.0) > float(clip_range)).astype(jnp.float32)),
        "value_explained_variance": _explained_variance(values, returns),
    }


def _policy_eval_from_forward_outputs(params, features, target_out, amount_out, target_idx: jnp.ndarray, amount_idx: jnp.ndarray):
    target_logits = _safe_target_logits(target_out["target_logits"], features.target_mask)
    source_rows = jnp.arange(features.source_slots.shape[0])
    chosen_amount_mask = features.amount_mask[source_rows, jnp.clip(target_idx, 0, P_MAX)]
    amount_logits = _safe_amount_logits(amount_out["amount_logits"], chosen_amount_mask)
    target_lp_all = jax.nn.log_softmax(target_logits)
    amount_lp_all = jax.nn.log_softmax(amount_logits)
    source_active = features.source_mask
    target_lp = target_lp_all[source_rows, target_idx]
    amount_lp = amount_lp_all[source_rows, amount_idx]
    logprob = jnp.sum(jnp.where(source_active, target_lp + jnp.where(target_idx == NOOP_TARGET_SLOT, 0.0, amount_lp), 0.0))
    target_prob = jax.nn.softmax(target_logits)
    amount_prob = jax.nn.softmax(amount_logits)
    target_row_any = jnp.any(features.target_mask, axis=-1, keepdims=True)
    target_noop = jnp.arange(features.target_mask.shape[-1]) == NOOP_TARGET_SLOT
    target_entropy_mask = jnp.where(target_row_any, features.target_mask, target_noop[None, :])
    amount_row_any = jnp.any(chosen_amount_mask, axis=-1, keepdims=True)
    amount_none = jnp.arange(chosen_amount_mask.shape[-1]) == 0
    amount_entropy_mask = jnp.where(amount_row_any, chosen_amount_mask, amount_none[None, :])
    target_ent = -jnp.sum(target_prob * jnp.where(target_entropy_mask, target_lp_all, 0.0), axis=-1)
    amount_ent = -jnp.sum(amount_prob * jnp.where(amount_entropy_mask, amount_lp_all, 0.0), axis=-1)
    entropy = jnp.sum(jnp.where(source_active, target_ent + jnp.where(target_idx == NOOP_TARGET_SLOT, 0.0, amount_ent), 0.0))
    value = value_apply(params["value"], target_out["global_ctx"][0])
    return logprob, value, entropy


def _policy_eval(params, features, config: dict[str, Any], target_idx: jnp.ndarray, amount_idx: jnp.ndarray):
    out = bc_forward(params["bc"], _source_batch(features, target_idx), config)
    return _policy_eval_from_forward_outputs(params, features, out, out, target_idx, amount_idx)


def _policy_sample_act(params, state, seat, key, config: dict[str, Any], source_cap: int):
    features = build_bc_features_for_seat(state, seat, source_cap=source_cap)
    target_out = bc_forward(params["bc"], _source_batch(features), config)
    target_logits = _safe_target_logits(target_out["target_logits"], features.target_mask)
    kt, ka = jax.random.split(key)
    target_idx = jax.random.categorical(kt, target_logits, axis=-1).astype(jnp.int32)
    amount_out = bc_forward(params["bc"], _source_batch(features, target_idx), config)
    source_rows = jnp.arange(features.source_slots.shape[0])
    chosen_amount_mask = features.amount_mask[source_rows, jnp.clip(target_idx, 0, P_MAX)]
    amount_logits = _safe_amount_logits(amount_out["amount_logits"], chosen_amount_mask)
    amount_idx = jax.random.categorical(ka, amount_logits, axis=-1).astype(jnp.int32)
    logprob, value, entropy = _policy_eval_from_forward_outputs(params, features, target_out, amount_out, target_idx, amount_idx)
    rows = action_rows_from_source_choices(state, seat, features.source_slots, target_idx, amount_idx, features.source_mask)
    return rows, logprob, value, entropy, target_idx, amount_idx, features


def _policy_greedy_act(bc_params, state, seat, config: dict[str, Any], source_cap: int):
    features = build_bc_features_for_seat(state, seat, source_cap=source_cap)
    target_out = bc_forward(bc_params, _source_batch(features), config)
    target_logits = _safe_target_logits(target_out["target_logits"], features.target_mask)
    target_idx = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
    amount_out = bc_forward(bc_params, _source_batch(features, target_idx), config)
    source_rows = jnp.arange(features.source_slots.shape[0])
    chosen_amount_mask = features.amount_mask[source_rows, jnp.clip(target_idx, 0, P_MAX)]
    amount_logits = _safe_amount_logits(amount_out["amount_logits"], chosen_amount_mask)
    amount_idx = jnp.argmax(amount_logits, axis=-1).astype(jnp.int32)
    return action_rows_from_source_choices(state, seat, features.source_slots, target_idx, amount_idx, features.source_mask)


def _compute_gae(rewards, values, dones, last_values, gamma: float, lam: float):
    def body(carry, x):
        next_adv, next_value = carry
        reward, value, done = x
        mask = 1.0 - done
        delta = reward + float(gamma) * next_value * mask - value
        adv = delta + float(gamma) * float(lam) * mask * next_adv
        return (adv, value), adv

    _carry, adv_rev = jax.lax.scan(
        body,
        (jnp.zeros_like(last_values), last_values),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    adv = adv_rev[::-1]
    return adv, adv + values


def _value_for_state(params, state, seat, config: dict[str, Any], source_cap: int = 1):
    features = build_bc_features_for_seat(state, seat, source_cap=source_cap)
    out = bc_forward(params["bc"], _source_batch(features), config)
    return value_apply(params["value"], out["global_ctx"][0])


def _learner_terminal_fields(rewards, ranks, learner_seat):
    return rewards[learner_seat], ranks[learner_seat]


def _pfsp_slot_stats(traj, max_slots: int, players: int):
    done = traj["done"] > 0.5
    reward = traj["reward"]
    rank = traj["rank"].astype(jnp.float32)
    if int(players) == 2:
        result = (rank == 0.0).astype(jnp.float32)
    else:
        result = (float(players - 1) - rank) / float(max(players - 1, 1))
    frozen = traj["opponent_kind"] == OPP_FROZEN_POLICY
    simple = traj["opponent_kind"] == OPP_SIMPLE_HEURISTIC
    proxy = traj["opponent_kind"] == OPP_JAX_PROXY
    slot = jnp.clip(traj["opponent_slot"], 0, int(max_slots) - 1)
    terminal_frozen = done[:, :, None] & frozen
    terminal_simple = done[:, :, None] & simple
    terminal_proxy = done[:, :, None] & proxy
    slot_axis = jnp.arange(int(max_slots), dtype=jnp.int32)
    by_slot = terminal_frozen[:, :, :, None] & (slot[:, :, :, None] == slot_axis)
    simple_count = jnp.sum(terminal_simple.astype(jnp.float32))
    proxy_count = jnp.sum(terminal_proxy.astype(jnp.float32))
    return {
        "slot_games": jnp.sum(by_slot.astype(jnp.float32), axis=(0, 1, 2)),
        "slot_score_sum": jnp.sum(by_slot.astype(jnp.float32) * result[:, :, None, None], axis=(0, 1, 2)),
        "slot_reward_sum": jnp.sum(by_slot.astype(jnp.float32) * reward[:, :, None, None], axis=(0, 1, 2)),
        "slot_rank_sum": jnp.sum(by_slot.astype(jnp.float32) * rank[:, :, None, None], axis=(0, 1, 2)),
        "kind_games": jnp.asarray([0.0, simple_count, proxy_count, 0.0], dtype=jnp.float32),
        "kind_score_sum": jnp.asarray(
            [
                0.0,
                jnp.sum(terminal_simple.astype(jnp.float32) * result[:, :, None]),
                jnp.sum(terminal_proxy.astype(jnp.float32) * result[:, :, None]),
                0.0,
            ],
            dtype=jnp.float32,
        ),
        "kind_reward_sum": jnp.asarray(
            [
                0.0,
                jnp.sum(terminal_simple.astype(jnp.float32) * reward[:, :, None]),
                jnp.sum(terminal_proxy.astype(jnp.float32) * reward[:, :, None]),
                0.0,
            ],
            dtype=jnp.float32,
        ),
        "kind_rank_sum": jnp.asarray(
            [
                0.0,
                jnp.sum(terminal_simple.astype(jnp.float32) * rank[:, :, None]),
                jnp.sum(terminal_proxy.astype(jnp.float32) * rank[:, :, None]),
                0.0,
            ],
            dtype=jnp.float32,
        ),
    }


def _where_state(mask: jnp.ndarray, replacement: EnvState, original: EnvState) -> EnvState:
    def choose(a, b):
        shaped = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
        return jnp.where(shaped, a, b)

    return jax.tree_util.tree_map(choose, replacement, original)


def _make_update(config: JaxPPOConfig, bc_config: dict[str, Any], state_bank: EnvState | None = None):
    env_config = EnvConfig(
        num_players=int(config.players),
        episode_steps=int(config.episode_steps),
        enable_comets=bool(config.enable_comets),
    )
    bank_mode = str(config.state_bank_mode)
    trajectory_feature_dtype = _compute_dtype(str(bc_config.get("compute_dtype", config.precision)))

    def store_feature(x: jnp.ndarray) -> jnp.ndarray:
        return x.astype(trajectory_feature_dtype)

    def reset_many(keys, done_mask, next_states, cycle_index):
        if state_bank is None:
            reset_states = jax.vmap(lambda k: reset(k, env_config))(keys)
            next_cycle_index = cycle_index
        elif bank_mode == "random":
            bank_size = state_bank.step.shape[0]
            idx = jax.random.randint(keys[0], (int(config.envs),), 0, bank_size)
            reset_states = jax.tree_util.tree_map(lambda x: x[idx], state_bank)
            next_cycle_index = cycle_index
        else:
            bank_size = state_bank.step.shape[0]
            done_int = done_mask.astype(jnp.int32)
            ranks = jnp.cumsum(done_int) - 1
            idx = jnp.mod(cycle_index + ranks, bank_size)
            reset_states = jax.tree_util.tree_map(lambda x: x[idx], state_bank)
            next_cycle_index = cycle_index + jnp.sum(done_int)
        return _where_state(done_mask, reset_states, next_states), next_cycle_index

    def rollout(params, key, states, cycle_index, frozen_bc_params, learner_seat_plan, opponent_kind_plan, opponent_slot_plan):
        def scan_body(carry, step_key):
            carry_states, carry_cycle_index = carry
            action_key, reset_key = jax.random.split(step_key)
            action_keys = jax.random.split(action_key, int(config.envs))
            reset_keys = jax.random.split(reset_key, int(config.envs))

            def one_env(state, k, learner_seat, opponent_kind, opponent_slot):
                rows0, lp, val, ent, ti, ai, feats = _policy_sample_act(params, state, learner_seat, k, bc_config, int(config.source_cap))
                obs = build_observation(state)
                simple_rows = simple_heuristic_actions(state)
                proxy_rows = greedy_actions(obs["planets"], state.num_players)
                seat_axis = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)
                actions = jnp.zeros((MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
                actions = jnp.where((opponent_kind == OPP_SIMPLE_HEURISTIC)[:, None, None], simple_rows, actions)
                actions = jnp.where((opponent_kind == OPP_JAX_PROXY)[:, None, None], proxy_rows, actions)
                frozen_mask = opponent_kind == OPP_FROZEN_POLICY
                frozen_seat = jnp.argmax(frozen_mask.astype(jnp.int32)).astype(jnp.int32)

                def add_frozen_rows(current_actions):
                    frozen_slot = jnp.maximum(opponent_slot[frozen_seat], 0)
                    frozen_rows = _policy_greedy_act(
                        tree_take(frozen_bc_params, frozen_slot),
                        state,
                        frozen_seat,
                        bc_config,
                        int(config.source_cap),
                    )
                    return jnp.where((seat_axis == frozen_seat)[:, None, None], frozen_rows[None, :, :], current_actions)

                actions = jax.lax.cond(jnp.any(frozen_mask), add_frozen_rows, lambda current_actions: current_actions, actions)
                actions = jnp.where((seat_axis == learner_seat)[:, None, None], rows0[None, :, :], actions)
                next_state, _next_obs, rewards, done, info = step(state, actions)
                learner_reward, learner_rank = _learner_terminal_fields(rewards, info["ranks"], learner_seat)
                store = {
                    "planet_features": store_feature(feats.planet_features),
                    "global_features": store_feature(feats.global_features),
                    "target_state_features": store_feature(feats.target_state_features),
                    "pair_features": store_feature(feats.pair_features),
                    "source_slots": feats.source_slots,
                    "source_mask": feats.source_mask,
                    "active_source_count": feats.active_source_count,
                    "selected_source_count": feats.selected_source_count,
                    "target_idx": ti,
                    "amount_idx": ai,
                    "old_logprob": lp,
                    "value": val,
                    "entropy": ent,
                    "learner_seat": learner_seat,
                    "reward": learner_reward,
                    "done": done.astype(jnp.float32),
                    "submitted_action_count": info["submitted_action_count"].astype(jnp.float32),
                    "valid_action_count": info["valid_action_count"].astype(jnp.float32),
                    "invalid_action_count": info["invalid_action_count"].astype(jnp.float32),
                    "invalid_source_id_count": info["invalid_source_id_count"].astype(jnp.float32),
                    "invalid_inactive_player_id_count": info["invalid_inactive_player_id_count"].astype(jnp.float32),
                    "invalid_source_not_owned_count": info["invalid_source_not_owned_count"].astype(jnp.float32),
                    "invalid_non_positive_ship_amount_count": info["invalid_non_positive_ship_amount_count"].astype(jnp.float32),
                    "invalid_unaffordable_source_total_count": info["invalid_unaffordable_source_total_count"].astype(jnp.float32),
                    "invalid_no_free_fleet_slot_count": info["invalid_no_free_fleet_slot_count"].astype(jnp.float32),
                    "episode_step": next_state.step.astype(jnp.float32),
                    "rank": learner_rank,
                    "opponent_kind": opponent_kind,
                    "opponent_slot": opponent_slot,
                }
                if config.recompute_masks:
                    source_alive = state.planet_alive & (state.planet_owner == learner_seat) & (state.planet_ships >= 1.0)
                    store = {
                        **store,
                        "mask_source_alive": source_alive,
                        "mask_target_alive": state.planet_alive,
                    }
                else:
                    store = {
                        **store,
                        "target_mask": feats.target_mask,
                        "amount_mask": feats.amount_mask,
                    }
                return next_state, store

            stepped_states, stores = jax.vmap(one_env)(carry_states, action_keys, learner_seat_plan, opponent_kind_plan, opponent_slot_plan)
            done_mask = stores["done"].astype(jnp.bool_)
            next_carry_states, next_cycle_index = reset_many(reset_keys, done_mask, stepped_states, carry_cycle_index)
            return (next_carry_states, next_cycle_index), stores

        (final_states, final_cycle_index), traj = jax.lax.scan(
            scan_body,
            (states, cycle_index),
            jax.random.split(key, int(config.rollout_steps)),
        )
        return traj, final_states, final_cycle_index

    def loss_fn(params, traj, last_values):
        rewards = traj["reward"]
        values = traj["value"]
        dones = traj["done"]
        adv, returns = _compute_gae(rewards, values, dones, last_values, float(config.gamma), float(config.lam))
        adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1.0e-8)

        from .features import JaxBCFeatures

        def eval_with_config(p, pf, gf, tsf, pair, tm, am, slots, sm, active_count, selected_count, ti, ai):
            feats = JaxBCFeatures(pf, gf, tsf, pair, tm, am, slots, sm, active_count, selected_count)
            return _policy_eval(p, feats, bc_config, ti, ai)

        policy_eval = jax.checkpoint(eval_with_config) if config.remat_policy_eval else eval_with_config
        if config.recompute_masks:
            def one_eval(pf, gf, tsf, pair, slots, sm, active_count, selected_count, ti, ai, source_alive, target_alive):
                tm, am = build_selected_masks_from_activity(
                    source_alive=source_alive,
                    target_alive=target_alive,
                    source_slots=slots,
                    source_mask=sm,
                )
                return policy_eval(params, pf, gf, tsf, pair, tm, am, slots, sm, active_count, selected_count, ti, ai)

            new_lp, new_v, ent = jax.vmap(jax.vmap(one_eval))(
                traj["planet_features"],
                traj["global_features"],
                traj["target_state_features"],
                traj["pair_features"],
                traj["source_slots"],
                traj["source_mask"],
                traj["active_source_count"],
                traj["selected_source_count"],
                traj["target_idx"],
                traj["amount_idx"],
                traj["mask_source_alive"],
                traj["mask_target_alive"],
            )
        else:
            def one_eval(pf, gf, tsf, pair, tm, am, slots, sm, active_count, selected_count, ti, ai):
                return policy_eval(params, pf, gf, tsf, pair, tm, am, slots, sm, active_count, selected_count, ti, ai)

            new_lp, new_v, ent = jax.vmap(jax.vmap(one_eval))(
                traj["planet_features"],
                traj["global_features"],
                traj["target_state_features"],
                traj["pair_features"],
                traj["target_mask"],
                traj["amount_mask"],
                traj["source_slots"],
                traj["source_mask"],
                traj["active_source_count"],
                traj["selected_source_count"],
                traj["target_idx"],
                traj["amount_idx"],
            )
        ratio = jnp.exp(new_lp - traj["old_logprob"])
        pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1.0 - float(config.clip), 1.0 + float(config.clip)) * adv))
        vloss = jnp.mean((returns - new_v) ** 2)
        entropy = jnp.mean(ent)
        loss = pg + float(config.vf) * vloss - float(config.ent) * entropy
        metrics = {
            "loss": loss,
            "policy_loss": pg,
            "value_loss": vloss,
            "entropy": entropy,
            "mean_reward": jnp.mean(rewards),
            "mean_return": jnp.mean(returns),
            "clip_dev": jnp.mean(jnp.abs(ratio - 1.0)),
            **_ppo_diagnostic_metrics(
                old_logprob=traj["old_logprob"],
                new_logprob=new_lp,
                values=new_v,
                returns=returns,
                clip_range=float(config.clip),
            ),
            "submitted_action_count": jnp.sum(traj["submitted_action_count"]),
            "valid_action_count": jnp.sum(traj["valid_action_count"]),
            "invalid_action_count": jnp.sum(traj["invalid_action_count"]),
            "invalid_source_id_count": jnp.sum(traj["invalid_source_id_count"]),
            "invalid_inactive_player_id_count": jnp.sum(traj["invalid_inactive_player_id_count"]),
            "invalid_source_not_owned_count": jnp.sum(traj["invalid_source_not_owned_count"]),
            "invalid_non_positive_ship_amount_count": jnp.sum(traj["invalid_non_positive_ship_amount_count"]),
            "invalid_unaffordable_source_total_count": jnp.sum(traj["invalid_unaffordable_source_total_count"]),
            "invalid_no_free_fleet_slot_count": jnp.sum(traj["invalid_no_free_fleet_slot_count"]),
            "invalid_action_rate": jnp.where(
                jnp.sum(traj["submitted_action_count"]) > 0.0,
                jnp.sum(traj["invalid_action_count"]) / jnp.sum(traj["submitted_action_count"]),
                0.0,
            ),
            "done_rate": jnp.mean(dones),
            "reset_count": jnp.sum(dones),
            "mean_episode_step": jnp.mean(traj["episode_step"][-1]),
            "max_episode_step": jnp.max(traj["episode_step"][-1]),
            "decisions": jnp.sum(traj["active_source_count"].astype(jnp.float32)),
            "selected_decisions": jnp.sum(traj["selected_source_count"].astype(jnp.float32)),
            "dropped_decisions": jnp.sum(jnp.maximum(traj["active_source_count"] - traj["selected_source_count"], 0).astype(jnp.float32)),
            "source_cap": jnp.asarray(float(config.source_cap), dtype=jnp.float32),
            "recompute_masks": jnp.asarray(float(config.recompute_masks), dtype=jnp.float32),
            "remat_policy_eval": jnp.asarray(float(config.remat_policy_eval), dtype=jnp.float32),
        }
        return loss, metrics

    def train_on_traj(params, opt_state, traj, next_states, next_cycle_index, update_index, learner_seat_plan):
        learner_seats = learner_seat_plan
        last_values = jax.vmap(lambda s, seat: _value_for_state(params, s, seat, bc_config))(next_states, learner_seats)

        def wrapped_loss(p):
            return loss_fn(p, traj, last_values)

        (loss, metrics), grads = jax.value_and_grad(wrapped_loss, has_aux=True)(params)
        grads = jax.lax.cond(
            update_index <= int(config.freeze_bc_steps),
            lambda g: {**g, "bc": jax.tree_util.tree_map(jnp.zeros_like, g["bc"])},
            lambda g: g,
            grads,
        )
        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        metrics = {**metrics, "loss": loss}
        league_stats = _pfsp_slot_stats(traj, int(config.pfsp_max_policy_slots), int(config.players))
        return params2, opt_state2, next_states, next_cycle_index, metrics, league_stats

    def update(params, opt_state, key, states, cycle_index, update_index, frozen_bc_params, learner_seat_plan, opponent_kind_plan, opponent_slot_plan):
        traj, next_states, next_cycle_index = rollout(params, key, states, cycle_index, frozen_bc_params, learner_seat_plan, opponent_kind_plan, opponent_slot_plan)
        return train_on_traj(params, opt_state, traj, next_states, next_cycle_index, update_index, learner_seat_plan)

    optimizer = optax.chain(optax.clip_by_global_norm(float(config.max_grad_norm)), optax.adam(float(config.lr)))
    return jax.jit(update), optimizer, jax.jit(rollout), jax.jit(train_on_traj)


def _default_match_plan_arrays(config: JaxPPOConfig):
    learner_seat = jnp.zeros((int(config.envs),), dtype=jnp.int32)
    opponent_kind = jnp.zeros((int(config.envs), MAX_PLAYERS), dtype=jnp.int32)
    opponent_slot = -jnp.ones((int(config.envs), MAX_PLAYERS), dtype=jnp.int32)
    kind = OPP_JAX_PROXY if config.opponent == "jax_proxy" else OPP_SIMPLE_HEURISTIC
    active_seats = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)[None, :] < int(config.players)
    non_learner = jnp.arange(MAX_PLAYERS, dtype=jnp.int32)[None, :] != 0
    opponent_kind = jnp.where(active_seats & non_learner, kind, opponent_kind)
    opponent_kind = opponent_kind.at[:, 0].set(OPP_NONE)
    return learner_seat, opponent_kind, opponent_slot


def _initial_vector_states(config: JaxPPOConfig, key, env_config: EnvConfig, state_bank: EnvState | None):
    if state_bank is None:
        reset_keys = jax.random.split(key, int(config.envs))
        return jax.vmap(lambda k: reset(k, env_config))(reset_keys), jnp.array(0, dtype=jnp.int32)
    bank_size = state_bank.step.shape[0]
    if config.state_bank_mode == "random":
        idx = jax.random.randint(key, (int(config.envs),), 0, bank_size)
        cycle_index = jnp.array(0, dtype=jnp.int32)
    else:
        idx = jnp.mod(jnp.arange(int(config.envs), dtype=jnp.int32), bank_size)
        cycle_index = jnp.array(int(config.envs), dtype=jnp.int32)
    return jax.tree_util.tree_map(lambda x: x[idx], state_bank), cycle_index


def _resume_checkpoint_dir(config: JaxPPOConfig, out_dir: Path) -> Path | None:
    if config.resume_from:
        path = Path(config.resume_from)
        return path if path.is_dir() else path.parent
    latest = out_dir / "latest"
    if config.resume and (latest / "params.npz").exists():
        return latest
    return None


def _assert_resume_bc_compatible(current_bc_config: dict[str, Any], checkpoint_config: dict[str, Any]) -> None:
    saved_bc_config = checkpoint_config.get("bc_model_config", checkpoint_config.get("model_config", {}))
    if not saved_bc_config:
        return
    required = (
        "planet_feature_dim",
        "global_feature_dim",
        "target_state_feature_dim",
        "pair_feature_dim",
        "max_planets",
        "target_classes",
        "amount_bins",
        "noop_target_slot",
        "hidden_size",
        "num_layers",
        "num_heads",
    )
    for key in required:
        if current_bc_config.get(key) != saved_bc_config.get(key):
            raise RuntimeError(
                f"cannot resume PPO checkpoint with incompatible BC config: "
                f"{key} expected {current_bc_config.get(key)!r}, got {saved_bc_config.get(key)!r}"
            )
    current_dtype = str(current_bc_config.get("compute_dtype", "float32"))
    saved_dtype = str(saved_bc_config.get("compute_dtype", "float32"))
    if current_dtype != saved_dtype:
        raise RuntimeError(
            f"cannot resume PPO checkpoint with incompatible PPO precision: "
            f"expected {current_dtype!r}, got {saved_dtype!r}"
        )


def _resume_env_state_compatible(saved: dict[str, Any], config: JaxPPOConfig) -> bool:
    return (
        int(saved.get("players", -1)) == int(config.players)
        and int(saved.get("envs", -1)) == int(config.envs)
        and int(saved.get("episode_steps", -1)) == int(config.episode_steps)
        and bool(saved.get("enable_comets", False)) == bool(config.enable_comets)
        and saved.get("initial_state_bank") == config.initial_state_bank
        and str(saved.get("state_bank_mode", "")) == str(config.state_bank_mode)
    )


def _league_manifest_path(out_dir: Path, config: JaxPPOConfig) -> Path:
    legacy = out_dir / "league" / "manifest.json"
    if legacy.exists():
        try:
            if int(load_manifest(legacy).players) == int(config.players):
                return legacy
        except Exception:
            pass
        return out_dir / "league" / f"{int(config.players)}p" / "manifest.json"
    mode_specific = out_dir / "league" / f"{int(config.players)}p" / "manifest.json"
    return mode_specific if mode_specific.exists() else legacy


def train(config: JaxPPOConfig) -> dict[str, Any]:
    _apply_jax_precision_config(config)
    runtime = _check_runtime(config.require_cuda)
    if config.opponent not in {"simple_heuristic_jax", "jax_proxy", "pfsp_jax"}:
        raise RuntimeError("orbit_ppo_jax.train supports --opponent simple_heuristic_jax, jax_proxy, or pfsp_jax")
    if config.opponent == "pfsp_jax" and not config.pfsp_enabled:
        raise RuntimeError("--opponent pfsp_jax requires --pfsp_enabled")
    if config.pfsp_learner_seat_mode not in {"fixed0", "rotate", "random"}:
        raise RuntimeError("--pfsp_learner_seat_mode must be fixed0, rotate, or random")
    if config.state_bank_mode not in {"cycle", "random"}:
        raise RuntimeError("--state_bank_mode must be either cycle or random")
    if int(config.source_cap) < 1 or int(config.source_cap) > P_MAX:
        raise RuntimeError(f"--source_cap must be between 1 and {P_MAX}")
    if int(config.profile_updates) < 1:
        raise RuntimeError("--profile_updates must be >= 1")
    if int(config.profile_max_env_steps) < 0:
        raise RuntimeError("--profile_max_env_steps must be >= 0")
    _bc_compute_dtype_name(config.precision)

    state_bank = None
    state_bank_metadata: dict[str, Any] = {}
    state_bank_warning = False
    if config.initial_state_bank:
        from orbit_jax_env.official_state_dataset import apply_runtime_config, has_imported_comet_paths, load_state_bank

        loaded_bank, state_bank_metadata = load_state_bank(config.initial_state_bank)
        if int(state_bank_metadata.get("players", config.players)) != int(config.players):
            raise RuntimeError(
                f"state bank players={state_bank_metadata.get('players')} does not match trainer players={config.players}"
            )
        state_bank_warning = int(state_bank_metadata.get("episode_steps", config.episode_steps)) != int(config.episode_steps)
        state_bank = apply_runtime_config(
            loaded_bank,
            players=int(config.players),
            episode_steps=int(config.episode_steps),
        )
        bank_has_imported_comets = has_imported_comet_paths(state_bank)
    else:
        bank_has_imported_comets = False

    reset_source = "official_state_bank" if state_bank is not None else "jax_reset"
    if bank_has_imported_comets:
        comet_mode = "official_imported"
    elif config.enable_comets:
        comet_mode = "jax_approx"
    else:
        comet_mode = "disabled"
    comet_warning = bool(config.enable_comets and comet_mode == "jax_approx")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_config = {
        **asdict(config),
        **runtime,
        "reset_source": reset_source,
        "state_bank_metadata": state_bank_metadata,
        "state_bank_episode_steps_overridden": state_bank_warning,
        "comet_warning": comet_warning,
        "comet_mode": comet_mode,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(saved_config, f, indent=2, sort_keys=True)

    bc_params, bc_config = load_bc_jax_params(config.bc_checkpoint)
    bc_config = {**bc_config, "compute_dtype": _bc_compute_dtype_name(config.precision)}
    league_manifest = None
    manifest_path = out_dir / "league" / "manifest.json"
    if config.pfsp_enabled:
        manifest_path = _league_manifest_path(out_dir, config)
        if manifest_path.exists():
            league_manifest = load_manifest(manifest_path)
            if int(league_manifest.players) != int(config.players):
                raise RuntimeError(
                    f"PFSP manifest players={league_manifest.players} does not match trainer players={config.players}"
                )
        else:
            league_manifest = build_initial_manifest(
                players=int(config.players),
                max_policy_slots=int(config.pfsp_max_policy_slots),
                bc_checkpoint=config.bc_checkpoint,
            )
            save_manifest(manifest_path, league_manifest)
        frozen_bank = build_pfsp_bank(league_manifest, bc_params, bc_config)
    else:
        frozen_bank = None
    key = jax.random.PRNGKey(int(config.seed))
    key, value_key = jax.random.split(key)
    params = {"bc": bc_params, "value": init_value_head(value_key, int(bc_config["hidden_size"]))}
    update_fn, optimizer, rollout_fn, train_on_traj_fn = _make_update(config, bc_config, state_bank)
    opt_state = optimizer.init(params)
    env_config = EnvConfig(
        num_players=int(config.players),
        episode_steps=int(config.episode_steps),
        enable_comets=bool(config.enable_comets),
    )
    key, reset_key = jax.random.split(key)
    states, state_bank_cycle_index = _initial_vector_states(config, reset_key, env_config, state_bank)
    best_score = -1.0e9
    best_entry_id: str | None = None
    last_metrics: dict[str, Any] = {}
    start_update = 0
    start_env_steps = 0
    resume_path = _resume_checkpoint_dir(config, out_dir)
    resume_env_state = "fresh"
    if resume_path is not None and (resume_path / "params.npz").exists():
        loaded_params, resume_checkpoint_config, resume_metrics = load_jax_checkpoint(resume_path)
        _assert_resume_bc_compatible(bc_config, resume_checkpoint_config)
        params = loaded_params
        opt_state = optimizer.init(params)
        try:
            training_state = load_jax_training_state(
                resume_path,
                opt_state_template=opt_state,
                env_states_template=states,
            )
        except FileNotFoundError:
            start_update = int(resume_metrics.get("update", 0))
            start_env_steps = int(resume_metrics.get("env_steps", 0))
            resume_env_state = "missing_sidecar_reset"
        else:
            opt_state = training_state["opt_state"]
            key = training_state["rng_key"]
            start_update = int(training_state["update_index"])
            start_env_steps = int(training_state["env_steps"])
            best_score = float(training_state["best_score"])
            if _resume_env_state_compatible(training_state, config):
                states = training_state["env_states"]
                state_bank_cycle_index = training_state["state_bank_cycle_index"]
                resume_env_state = "loaded"
            else:
                resume_env_state = "reset_incompatible"

    def match_plan_arrays(update_index: int):
        if config.pfsp_enabled and league_manifest is not None:
            plan = build_match_plan(
                league_manifest,
                rng=np.random.default_rng(int(config.seed) + int(update_index)),
                envs=int(config.envs),
                players=int(config.players),
                learner_seat_mode=str(config.pfsp_learner_seat_mode),
                anchor_fraction=float(config.pfsp_anchor_fraction),
                layout=str(config.pfsp_4p_layout),
                min_games_per_entry=int(config.pfsp_min_games_per_entry),
                hard_low=float(config.pfsp_hard_low),
                hard_high=float(config.pfsp_hard_high),
                hard_bonus=float(config.pfsp_hard_bonus),
                exploration_bonus=float(config.pfsp_exploration_bonus),
            )
            return plan.learner_seat, plan.opponent_kind, plan.opponent_slot
        return _default_match_plan_arrays(config)

    def frozen_policy_params():
        return frozen_bank.bc_params if frozen_bank is not None else tree_stack([bc_params])

    async_prefetch_enabled = bool(config.async_rollout_prefetch and not config.pfsp_enabled)
    prefetched_rollout = None

    for local_update_index in range(1, int(config.updates) + 1):
        update_index = int(start_update) + int(local_update_index)
        t0 = time.time()
        async_prefetch_pending = False
        async_current_policy_lag = 0.0
        async_prefetch_policy_lag = 0.0
        if async_prefetch_enabled:
            pending_prefetch = None
            with _profile_update_context(config, out_dir, update_index):
                if prefetched_rollout is None:
                    key, step_key = jax.random.split(key)
                    learner_seat_plan, opponent_kind_plan, opponent_slot_plan = match_plan_arrays(update_index)
                    traj, rolled_states, rolled_cycle_index = rollout_fn(
                        params,
                        step_key,
                        states,
                        state_bank_cycle_index,
                        frozen_policy_params(),
                        learner_seat_plan,
                        opponent_kind_plan,
                        opponent_slot_plan,
                    )
                else:
                    traj, rolled_states, rolled_cycle_index, learner_seat_plan = prefetched_rollout
                    prefetched_rollout = None
                    async_current_policy_lag = 1.0
                if local_update_index < int(config.updates):
                    next_update_index = int(start_update) + int(local_update_index) + 1
                    key, next_step_key = jax.random.split(key)
                    next_learner_seat_plan, next_opponent_kind_plan, next_opponent_slot_plan = match_plan_arrays(next_update_index)
                    next_traj, next_states, next_cycle_index = rollout_fn(
                        params,
                        next_step_key,
                        rolled_states,
                        rolled_cycle_index,
                        frozen_policy_params(),
                        next_learner_seat_plan,
                        next_opponent_kind_plan,
                        next_opponent_slot_plan,
                    )
                    pending_prefetch = (next_traj, next_states, next_cycle_index, next_learner_seat_plan)
                    async_prefetch_pending = True
                    async_prefetch_policy_lag = 1.0
                params, opt_state, states, state_bank_cycle_index, metrics_jax, league_stats_jax = train_on_traj_fn(
                    params,
                    opt_state,
                    traj,
                    rolled_states,
                    rolled_cycle_index,
                    jnp.asarray(update_index, dtype=jnp.int32),
                    learner_seat_plan,
                )
            prefetched_rollout = pending_prefetch
        else:
            key, step_key = jax.random.split(key)
            with _profile_update_context(config, out_dir, update_index):
                params, opt_state, states, state_bank_cycle_index, metrics_jax, league_stats_jax = update_fn(
                    params,
                    opt_state,
                    step_key,
                    states,
                    state_bank_cycle_index,
                    jnp.asarray(update_index, dtype=jnp.int32),
                    frozen_policy_params(),
                    *match_plan_arrays(update_index),
                )
        jax.block_until_ready(params)
        seconds = time.time() - t0
        update_env_steps = int(config.envs) * int(config.rollout_steps)
        env_steps = int(start_env_steps) + int(local_update_index) * update_env_steps
        metrics = {k: float(v) for k, v in metrics_jax.items()}
        if config.pfsp_enabled and league_manifest is not None:
            league_manifest = update_manifest_from_slot_stats(
                league_manifest,
                slot_games=[int(x) for x in np.asarray(league_stats_jax["slot_games"])],
                slot_score_sum=[float(x) for x in np.asarray(league_stats_jax["slot_score_sum"])],
                slot_reward_sum=[float(x) for x in np.asarray(league_stats_jax["slot_reward_sum"])],
                slot_rank_sum=[float(x) for x in np.asarray(league_stats_jax["slot_rank_sum"])],
                kind_games={idx: int(x) for idx, x in enumerate(np.asarray(league_stats_jax["kind_games"]))},
                kind_score_sum={idx: float(x) for idx, x in enumerate(np.asarray(league_stats_jax["kind_score_sum"]))},
                kind_reward_sum={idx: float(x) for idx, x in enumerate(np.asarray(league_stats_jax["kind_reward_sum"]))},
                kind_rank_sum={idx: float(x) for idx, x in enumerate(np.asarray(league_stats_jax["kind_rank_sum"]))},
                update_index=update_index,
            )
            save_manifest(manifest_path, league_manifest)
        metrics.update(
            {
                "update": update_index,
                "rollout_steps": int(config.rollout_steps),
                "episode_steps": int(config.episode_steps),
                "seconds": seconds,
                "update_env_steps": update_env_steps,
                "env_steps": env_steps,
                "steps_per_second": float(update_env_steps / seconds) if seconds > 0.0 else 0.0,
                "reset_source": reset_source,
                "comet_warning": comet_warning,
                "comet_mode": comet_mode,
                "resume_from": str(resume_path) if resume_path is not None else "",
                "resume_start_update": float(start_update),
                "resume_env_state": resume_env_state,
                "async_rollout_prefetch_requested": float(config.async_rollout_prefetch),
                "async_rollout_prefetch_active": float(async_prefetch_enabled),
                "async_rollout_prefetch_pending": float(async_prefetch_pending),
                "async_rollout_current_policy_lag": float(async_current_policy_lag),
                "async_rollout_prefetch_policy_lag": float(async_prefetch_policy_lag),
                **runtime,
            }
        )
        if not all(jnp.isfinite(jnp.asarray(v)) for v in metrics.values() if isinstance(v, float)):
            raise RuntimeError(f"non-finite JAX PPO metrics at update {update_index}: {metrics}")
        _append_jsonl(out_dir / "metrics.jsonl", metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        checkpoint_config = {**saved_config, "bc_model_config": bc_config}
        save_jax_checkpoint(out_dir / "latest", params, checkpoint_config, metrics)
        if update_index == 1 or update_index % int(config.save_interval_updates) == 0:
            save_jax_checkpoint(out_dir / "checkpoints" / f"update_{update_index:05d}", params, checkpoint_config, metrics)
        if config.pfsp_enabled and league_manifest is not None:
            should_add = (
                update_index >= int(config.pfsp_warmup_updates)
                and update_index % int(config.pfsp_snapshot_interval_updates) == 0
                and float(metrics["invalid_action_rate"]) <= 0.001
                and float(metrics["done_rate"]) > 0.0
            )
            if should_add:
                entry_id = f"update_{update_index:05d}"
                snapshot_dir = manifest_path.parent / "snapshots" / entry_id
                save_jax_checkpoint(snapshot_dir, params, checkpoint_config, metrics)
                league_manifest = add_snapshot_entry(
                    league_manifest,
                    entry_id=entry_id,
                    path=str(snapshot_dir),
                    update_index=update_index,
                    protected_entry_ids={best_entry_id} if best_entry_id is not None else set(),
                )
                save_manifest(manifest_path, league_manifest)
                frozen_bank = build_pfsp_bank(league_manifest, bc_params, bc_config)

        score = float(metrics["mean_reward"])
        if (
            config.pfsp_enabled
            and league_manifest is not None
            and frozen_bank is not None
            and int(config.pfsp_matrix_games) > 0
            and (update_index == 1 or update_index % int(config.pfsp_eval_interval_updates) == 0)
        ):
            from .pfsp_eval import evaluate_matrix

            matrix_summary = evaluate_matrix(
                params=params,
                bc_config=bc_config,
                bank=frozen_bank,
                manifest=league_manifest,
                config=config,
                key=key,
                out_dir=out_dir,
            )
            score = float(matrix_summary.get("matrix_score", score))
            metrics.update({f"pfsp_matrix_{k}": v for k, v in matrix_summary.items() if isinstance(v, (int, float))})
        if int(config.eval_games) > 0 and (update_index == 1 or update_index % int(config.eval_interval_updates) == 0):
            from .eval_vs_heuristic import evaluate

            eval_summary = evaluate(
                out_dir / "latest",
                config.eval_heuristic_path,
                games=int(config.eval_games),
                players=int(config.players),
                out_dir=out_dir / "eval",
                episode_steps=int(config.episode_steps),
            )
            with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
                json.dump(eval_summary, f, indent=2, sort_keys=True)
            score = float(eval_summary.get("average_final_reward", score))
            metrics.update({f"eval_{k}": v for k, v in eval_summary.items() if isinstance(v, (int, float))})
        if score > best_score:
            best_score = score
            if config.pfsp_enabled:
                best_entry_id = f"update_{update_index:05d}"
            save_jax_checkpoint(out_dir / "best", params, checkpoint_config, metrics)
        save_jax_training_state(
            out_dir / "latest",
            opt_state=opt_state,
            rng_key=key,
            env_states=states,
            state_bank_cycle_index=state_bank_cycle_index,
            update_index=update_index,
            env_steps=env_steps,
            best_score=best_score,
            players=int(config.players),
            envs=int(config.envs),
            episode_steps=int(config.episode_steps),
            enable_comets=bool(config.enable_comets),
            initial_state_bank=config.initial_state_bank,
            state_bank_mode=str(config.state_bank_mode),
        )
        last_metrics = metrics
        if metrics.get("invalid_action_count", 0.0) > float(config.envs * config.rollout_steps * P_MAX):
            break

    return {"out_dir": str(out_dir), "best_score": best_score, "last_metrics": last_metrics, **runtime}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train JAX PPO from an Orbit Wars BC checkpoint.")
    ap.add_argument("--bc_checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--players", type=int, default=4, choices=[2, 4])
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None, help="Legacy alias for --rollout_steps.")
    ap.add_argument("--rollout_steps", type=int, default=None)
    ap.add_argument("--episode_steps", type=int, default=500)
    ap.add_argument("--updates", type=int, default=1)
    ap.add_argument("--opponent", default="simple_heuristic_jax", choices=["simple_heuristic_jax", "jax_proxy", "pfsp_jax"])
    ap.add_argument("--eval_heuristic_path", default="orbit_wars_base.py")
    ap.add_argument("--eval_games", type=int, default=2)
    ap.add_argument("--eval_interval_updates", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--clip", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=0.5)
    ap.add_argument("--freeze_bc_steps", type=int, default=0)
    ap.add_argument("--save_interval_updates", type=int, default=5)
    ap.add_argument("--enable_comets", action="store_true")
    ap.add_argument("--initial_state_bank", default=None)
    ap.add_argument("--state_bank_mode", default="random", choices=["cycle", "random"])
    ap.add_argument("--source_cap", type=int, default=32)
    ap.add_argument("--pfsp_enabled", action="store_true")
    ap.add_argument("--pfsp_max_policy_slots", type=int, default=32)
    ap.add_argument("--pfsp_anchor_fraction", type=float, default=0.25)
    ap.add_argument("--pfsp_snapshot_interval_updates", type=int, default=10)
    ap.add_argument("--pfsp_warmup_updates", type=int, default=10)
    ap.add_argument("--pfsp_min_games_per_entry", type=int, default=16)
    ap.add_argument("--pfsp_hard_low", type=float, default=0.20)
    ap.add_argument("--pfsp_hard_high", type=float, default=0.55)
    ap.add_argument("--pfsp_hard_bonus", type=float, default=0.15)
    ap.add_argument("--pfsp_exploration_bonus", type=float, default=0.10)
    ap.add_argument("--pfsp_matrix_games", type=int, default=16)
    ap.add_argument("--pfsp_eval_interval_updates", type=int, default=10)
    ap.add_argument("--pfsp_learner_seat_mode", default="rotate", choices=["fixed0", "rotate", "random"])
    ap.add_argument("--pfsp_4p_layout", default="one_pfsp_two_anchors", choices=["one_pfsp_two_anchors"])
    ap.add_argument("--no_resume", dest="resume", action="store_false", help="Start fresh even when out_dir/latest exists.")
    ap.add_argument("--resume_from", default=None, help="Checkpoint directory to resume from; defaults to out_dir/latest when present.")
    ap.add_argument("--precision", default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Compute dtype for BC policy forward passes.")
    ap.add_argument(
        "--matmul_precision",
        default="highest",
        choices=["default", "highest", "high", "float32", "tensorfloat32", "bfloat16"],
        help="Value for jax_default_matmul_precision when not default.",
    )
    ap.add_argument("--remat_policy_eval", action="store_true", help="Use jax.checkpoint on PPO loss policy evaluation.")
    ap.add_argument("--no_remat_policy_eval", dest="remat_policy_eval", action="store_false", help="Disable PPO loss policy-evaluation rematerialization.")
    ap.add_argument("--recompute_masks", action="store_true", help="Recompute action masks in the PPO loss from compact state fields instead of storing full masks.")
    ap.add_argument("--no_recompute_masks", dest="recompute_masks", action="store_false", help="Store full trajectory masks instead of recomputing them during PPO loss.")
    ap.add_argument("--profile_dir", default="traces", help="Directory for jax.profiler.trace output; relative paths are under out_dir.")
    ap.add_argument("--no_profile", dest="profile_dir", action="store_const", const=None, help="Disable default jax.profiler.trace output.")
    ap.add_argument("--profile_updates", type=int, default=1, help="Number of initial updates to trace when --profile_dir is set.")
    ap.add_argument("--profile_max_env_steps", type=int, default=1024, help="Skip automatic profiling when envs * rollout_steps exceeds this limit; set 0 to force tracing.")
    ap.add_argument("--async_rollout_prefetch", action="store_true", help="Use split rollout/train JITs and queue the next non-PFSP rollout before the current train step.")
    ap.add_argument("--no_async_rollout_prefetch", dest="async_rollout_prefetch", action="store_false", help="Disable split rollout/train prefetching.")
    ap.set_defaults(resume=True)
    ap.set_defaults(remat_policy_eval=True, recompute_masks=True, async_rollout_prefetch=True)
    return ap


def config_from_args(args: argparse.Namespace) -> JaxPPOConfig:
    values = vars(args).copy()
    legacy_steps = values.pop("steps", None)
    rollout_steps = values.get("rollout_steps")
    values["rollout_steps"] = int(rollout_steps if rollout_steps is not None else (legacy_steps if legacy_steps is not None else 32))
    return JaxPPOConfig(**values)


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    config = config_from_args(build_arg_parser().parse_args(argv))
    summary = train(config)
    if argv is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return None
    return summary


if __name__ == "__main__":
    main()
