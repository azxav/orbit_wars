from __future__ import annotations

import jax
import jax.numpy as jnp

from .actions import empty_actions
from .config import EnvConfig
from .observation import build_observation
from .reset import reset
from .step import step


def jax_rollout(policy_apply, params, keys, config: EnvConfig, steps: int = 500):
    states = jax.vmap(lambda k: reset(k, config))(keys)

    def body(carry, _):
        obs = jax.vmap(build_observation)(carry)
        policy_out = policy_apply(params, obs)
        actions = policy_out.get("actions", jnp.zeros((keys.shape[0], *empty_actions().shape), dtype=jnp.float32))
        next_states, next_obs, rewards, dones, info = jax.vmap(step)(carry, actions)
        transition = {
            "obs": obs,
            "next_obs": next_obs,
            "rewards": rewards,
            "dones": dones,
            "logprobs": policy_out.get("logprobs", jnp.zeros((keys.shape[0],), dtype=jnp.float32)),
            "values": policy_out.get("values", jnp.zeros((keys.shape[0],), dtype=jnp.float32)),
            "info": info,
        }
        return next_states, transition

    return jax.lax.scan(body, states, None, length=int(steps))
