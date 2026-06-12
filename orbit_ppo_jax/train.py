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
    steps: int = 32
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


def _compute_gae(rewards, values, dones, gamma: float, lam: float):
    def body(carry, x):
        next_adv, next_value = carry
        reward, value, done = x
        mask = 1.0 - done
        delta = reward + float(gamma) * next_value * mask - value
        adv = delta + float(gamma) * float(lam) * mask * next_adv
        return (adv, value), adv

    _carry, adv_rev = jax.lax.scan(
        body,
        (jnp.zeros_like(values[0]), jnp.zeros_like(values[0])),
        (rewards[::-1], values[::-1], dones[::-1]),
    )
    adv = adv_rev[::-1]
    return adv, adv + values


def _make_update(config: JaxPPOConfig, bc_config: dict[str, Any]):
    env_config = EnvConfig(num_players=int(config.players), episode_steps=int(config.steps))
    seats = jnp.arange(MAX_PLAYERS)

    def rollout(params, key):
        reset_keys = jax.random.split(key, int(config.envs))
        states = jax.vmap(lambda k: reset(k, env_config))(reset_keys)

        def scan_body(carry, step_key):
            action_keys = jax.random.split(step_key, int(config.envs))

            def one_env(state, k):
                rows0, lp, val, ent, ti, ai, feats = _learner_act(params, state, k, bc_config)
                obs = build_observation(state)
                proxy = greedy_actions(obs["planets"], state.num_players)
                actions = proxy.at[0].set(rows0)
                next_state, _next_obs, rewards, done, info = step(state, actions)
                source_active = feats.target_mask[:, NOOP_TARGET_SLOT]
                invalid = jnp.asarray(0.0, dtype=jnp.float32)
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
                    "invalid": invalid,
                    "rank": info["ranks"][0],
                }
                return next_state, store

            next_states, stores = jax.vmap(one_env)(carry, action_keys)
            return next_states, stores

        _states, traj = jax.lax.scan(scan_body, states, jax.random.split(key, int(config.steps)))
        return traj

    def loss_fn(params, traj):
        rewards = traj["reward"]
        values = traj["value"]
        dones = traj["done"]
        adv, returns = _compute_gae(rewards, values, dones, float(config.gamma), float(config.lam))
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
            "invalid_action_count": jnp.sum(traj["invalid"]),
            "decisions": jnp.sum(traj["target_mask"][:, :, :, NOOP_TARGET_SLOT].astype(jnp.float32)),
        }
        return loss, metrics

    def update(params, opt_state, key, update_index):
        traj = rollout(params, key)

        def wrapped_loss(p):
            return loss_fn(p, traj)

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
        return params2, opt_state2, metrics

    optimizer = optax.chain(optax.clip_by_global_norm(float(config.max_grad_norm)), optax.adam(float(config.lr)))
    return jax.jit(update), optimizer


def train(config: JaxPPOConfig) -> dict[str, Any]:
    runtime = _check_runtime(config.require_cuda)
    if config.opponent != "jax_proxy":
        raise RuntimeError("orbit_ppo_jax.train currently supports --opponent jax_proxy")
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(config), **runtime}, f, indent=2, sort_keys=True)

    bc_params, bc_config = load_bc_jax_params(config.bc_checkpoint)
    key = jax.random.PRNGKey(int(config.seed))
    key, value_key = jax.random.split(key)
    params = {"bc": bc_params, "value": init_value_head(value_key, int(bc_config["hidden_size"]))}
    update_fn, optimizer = _make_update(config, bc_config)
    opt_state = optimizer.init(params)
    best_score = -1.0e9
    last_metrics: dict[str, Any] = {}

    for update_index in range(1, int(config.updates) + 1):
        key, step_key = jax.random.split(key)
        t0 = time.time()
        params, opt_state, metrics_jax = update_fn(params, opt_state, step_key, jnp.asarray(update_index, dtype=jnp.int32))
        jax.block_until_ready(params)
        metrics = {k: float(v) for k, v in metrics_jax.items()}
        metrics.update({"update": update_index, "seconds": time.time() - t0, **runtime})
        if not all(jnp.isfinite(jnp.asarray(v)) for v in metrics.values() if isinstance(v, float)):
            raise RuntimeError(f"non-finite JAX PPO metrics at update {update_index}: {metrics}")
        _append_jsonl(out_dir / "metrics.jsonl", metrics)
        save_jax_checkpoint(out_dir / "latest", params, {**asdict(config), "bc_model_config": bc_config, **runtime}, metrics)
        if update_index == 1 or update_index % int(config.save_interval_updates) == 0:
            save_jax_checkpoint(out_dir / "checkpoints" / f"update_{update_index:05d}", params, {**asdict(config), "bc_model_config": bc_config, **runtime}, metrics)

        score = float(metrics["mean_reward"])
        if int(config.eval_games) > 0 and (update_index == 1 or update_index % int(config.eval_interval_updates) == 0):
            from .eval_vs_heuristic import evaluate

            eval_summary = evaluate(out_dir / "latest", config.eval_heuristic_path, games=int(config.eval_games), players=int(config.players), out_dir=out_dir / "eval")
            with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
                json.dump(eval_summary, f, indent=2, sort_keys=True)
            score = float(eval_summary.get("average_final_reward", score))
            metrics.update({f"eval_{k}": v for k, v in eval_summary.items() if isinstance(v, (int, float))})
        if score > best_score:
            best_score = score
            save_jax_checkpoint(out_dir / "best", params, {**asdict(config), "bc_model_config": bc_config, **runtime}, metrics)
        if metrics.get("invalid_action_count", 0.0) > float(config.envs * config.steps * P_MAX):
            break
        last_metrics = metrics

    return {"out_dir": str(out_dir), "best_score": best_score, "last_metrics": last_metrics, **runtime}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train JAX PPO from an Orbit Wars BC checkpoint.")
    ap.add_argument("--bc_checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--players", type=int, default=4, choices=[2, 4])
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
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
    return ap


def config_from_args(args: argparse.Namespace) -> JaxPPOConfig:
    return JaxPPOConfig(**vars(args))


def main(argv: list[str] | None = None) -> dict[str, Any] | None:
    config = config_from_args(build_arg_parser().parse_args(argv))
    summary = train(config)
    if argv is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return None
    return summary


if __name__ == "__main__":
    main()
