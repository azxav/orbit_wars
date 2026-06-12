from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import optax

from orbit_jax_env.config import EnvConfig, MAX_PLAYERS, P_MAX
from orbit_jax_env.jax_policy import greedy_actions
from orbit_jax_env.observation import build_observation
from orbit_jax_env.reset import reset
from orbit_jax_env.step import step
from orbit_jax_env.state import EnvState

from .actions import action_rows_from_choices
from .bc_policy import bc_forward, init_value_head, load_bc_jax_params, value_apply
from .checkpointing import save_jax_checkpoint
from .features import NOOP_TARGET_SLOT, build_bc_features_for_seat

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
    opponent: str = "jax_proxy"
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
    p = features.planet_features.shape[0]
    batch = {
        "planet_features": jnp.broadcast_to(features.planet_features[None, :, :], (p, *features.planet_features.shape)),
        "global_features": jnp.broadcast_to(features.global_features[None, :], (p, features.global_features.shape[0])),
        "target_state_features": jnp.broadcast_to(features.target_state_features[None, :, :], (p, *features.target_state_features.shape)),
        "pair_features": features.pair_features,
        "source_slot": jnp.arange(p, dtype=jnp.int32),
    }
    if target_label is not None:
        batch["target_label"] = target_label.astype(jnp.int32)
    return batch


def _policy_eval(params, features, config: dict[str, Any], target_idx: jnp.ndarray, amount_idx: jnp.ndarray):
    out = bc_forward(params["bc"], _source_batch(features, target_idx), config)
    target_logits = _safe_target_logits(out["target_logits"], features.target_mask)
    chosen_amount_mask = features.amount_mask[jnp.arange(P_MAX), jnp.clip(target_idx, 0, P_MAX)]
    amount_logits = _safe_amount_logits(out["amount_logits"], chosen_amount_mask)
    target_lp_all = jax.nn.log_softmax(target_logits)
    amount_lp_all = jax.nn.log_softmax(amount_logits)
    source_active = features.target_mask[:, NOOP_TARGET_SLOT]
    target_lp = target_lp_all[jnp.arange(P_MAX), target_idx]
    amount_lp = amount_lp_all[jnp.arange(P_MAX), amount_idx]
    logprob = jnp.sum(jnp.where(source_active, target_lp + jnp.where(target_idx == NOOP_TARGET_SLOT, 0.0, amount_lp), 0.0))
    target_prob = jax.nn.softmax(target_logits)
    amount_prob = jax.nn.softmax(amount_logits)
    target_ent = -jnp.sum(target_prob * target_lp_all, axis=-1)
    amount_ent = -jnp.sum(amount_prob * amount_lp_all, axis=-1)
    entropy = jnp.sum(jnp.where(source_active, target_ent + jnp.where(target_idx == NOOP_TARGET_SLOT, 0.0, amount_ent), 0.0))
    value = value_apply(params["value"], out["global_ctx"][0])
    return logprob, value, entropy


def _learner_act(params, state, key, config: dict[str, Any]):
    features = build_bc_features_for_seat(state, 0)
    target_out = bc_forward(params["bc"], _source_batch(features), config)
    target_logits = _safe_target_logits(target_out["target_logits"], features.target_mask)
    kt, ka = jax.random.split(key)
    target_idx = jax.random.categorical(kt, target_logits, axis=-1).astype(jnp.int32)
    amount_out = bc_forward(params["bc"], _source_batch(features, target_idx), config)
    chosen_amount_mask = features.amount_mask[jnp.arange(P_MAX), jnp.clip(target_idx, 0, P_MAX)]
    amount_logits = _safe_amount_logits(amount_out["amount_logits"], chosen_amount_mask)
    amount_idx = jax.random.categorical(ka, amount_logits, axis=-1).astype(jnp.int32)
    logprob, value, entropy = _policy_eval(params, features, config, target_idx, amount_idx)
    rows = action_rows_from_choices(state, 0, target_idx, amount_idx)
    return rows, logprob, value, entropy, target_idx, amount_idx, features


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


def _value_for_state(params, state, config: dict[str, Any]):
    features = build_bc_features_for_seat(state, 0)
    out = bc_forward(params["bc"], _source_batch(features), config)
    return value_apply(params["value"], out["global_ctx"][0])


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

    def rollout(params, key, states, cycle_index):
        def scan_body(carry, step_key):
            carry_states, carry_cycle_index = carry
            action_key, reset_key = jax.random.split(step_key)
            action_keys = jax.random.split(action_key, int(config.envs))
            reset_keys = jax.random.split(reset_key, int(config.envs))

            def one_env(state, k):
                rows0, lp, val, ent, ti, ai, feats = _learner_act(params, state, k, bc_config)
                obs = build_observation(state)
                proxy = greedy_actions(obs["planets"], state.num_players)
                actions = proxy.at[0].set(rows0)
                next_state, _next_obs, rewards, done, info = step(state, actions)
                source_active = feats.target_mask[:, NOOP_TARGET_SLOT]
                store = {
                    "planet_features": feats.planet_features,
                    "global_features": feats.global_features,
                    "target_state_features": feats.target_state_features,
                    "pair_features": feats.pair_features,
                    "target_mask": feats.target_mask,
                    "amount_mask": feats.amount_mask,
                    "target_idx": ti,
                    "amount_idx": ai,
                    "old_logprob": lp,
                    "value": val,
                    "entropy": ent,
                    "reward": rewards[0],
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
                    "rank": info["ranks"][0],
                }
                return next_state, store

            stepped_states, stores = jax.vmap(one_env)(carry_states, action_keys)
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

        def one_eval(pf, gf, tsf, pair, tm, am, ti, ai):
            from .features import JaxBCFeatures

            feats = JaxBCFeatures(pf, gf, tsf, pair, tm, am)
            return _policy_eval(params, feats, bc_config, ti, ai)

        new_lp, new_v, ent = jax.vmap(jax.vmap(one_eval))(
            traj["planet_features"],
            traj["global_features"],
            traj["target_state_features"],
            traj["pair_features"],
            traj["target_mask"],
            traj["amount_mask"],
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
            "decisions": jnp.sum(traj["target_mask"][:, :, :, NOOP_TARGET_SLOT].astype(jnp.float32)),
        }
        return loss, metrics

    def update(params, opt_state, key, states, cycle_index, update_index):
        traj, next_states, next_cycle_index = rollout(params, key, states, cycle_index)
        last_values = jax.vmap(lambda s: _value_for_state(params, s, bc_config))(next_states)

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
        return params2, opt_state2, next_states, next_cycle_index, metrics

    optimizer = optax.chain(optax.clip_by_global_norm(float(config.max_grad_norm)), optax.adam(float(config.lr)))
    return jax.jit(update), optimizer


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


def train(config: JaxPPOConfig) -> dict[str, Any]:
    runtime = _check_runtime(config.require_cuda)
    if config.opponent != "jax_proxy":
        raise RuntimeError("orbit_ppo_jax.train currently supports --opponent jax_proxy")
    if config.state_bank_mode not in {"cycle", "random"}:
        raise RuntimeError("--state_bank_mode must be either cycle or random")

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
    key = jax.random.PRNGKey(int(config.seed))
    key, value_key = jax.random.split(key)
    params = {"bc": bc_params, "value": init_value_head(value_key, int(bc_config["hidden_size"]))}
    update_fn, optimizer = _make_update(config, bc_config, state_bank)
    opt_state = optimizer.init(params)
    env_config = EnvConfig(
        num_players=int(config.players),
        episode_steps=int(config.episode_steps),
        enable_comets=bool(config.enable_comets),
    )
    key, reset_key = jax.random.split(key)
    states, state_bank_cycle_index = _initial_vector_states(config, reset_key, env_config, state_bank)
    best_score = -1.0e9
    last_metrics: dict[str, Any] = {}

    for update_index in range(1, int(config.updates) + 1):
        key, step_key = jax.random.split(key)
        t0 = time.time()
        params, opt_state, states, state_bank_cycle_index, metrics_jax = update_fn(
            params,
            opt_state,
            step_key,
            states,
            state_bank_cycle_index,
            jnp.asarray(update_index, dtype=jnp.int32),
        )
        jax.block_until_ready(params)
        seconds = time.time() - t0
        update_env_steps = int(config.envs) * int(config.rollout_steps)
        env_steps = update_index * update_env_steps
        metrics = {k: float(v) for k, v in metrics_jax.items()}
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

        score = float(metrics["mean_reward"])
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
            save_jax_checkpoint(out_dir / "best", params, checkpoint_config, metrics)
        if metrics.get("invalid_action_count", 0.0) > float(config.envs * config.rollout_steps * P_MAX):
            break
        last_metrics = metrics

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
    ap.add_argument("--opponent", default="jax_proxy", choices=["jax_proxy"])
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
