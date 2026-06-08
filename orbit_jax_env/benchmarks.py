from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from .config import EnvConfig, MAX_ACTIONS_PER_PLAYER, MAX_PLAYERS
from .reset import reset
from .step import step


def run_benchmark(num_envs: int, steps: int, players: int) -> dict[str, float | int | str]:
    cfg = EnvConfig(num_players=int(players), enable_comets=False)
    keys = jax.random.split(jax.random.PRNGKey(0), int(num_envs))
    states = jax.vmap(lambda k: reset(k, cfg))(keys)
    actions = jnp.zeros((int(num_envs), MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)

    def body(carry, _):
        new_carry, *_ = jax.vmap(step)(carry, actions)
        return new_carry, None

    fn = jax.jit(lambda s: jax.lax.scan(body, s, None, length=int(steps))[0])
    t0 = time.perf_counter()
    compiled_state = fn(states)
    jax.block_until_ready(compiled_state.step)
    t1 = time.perf_counter()
    compiled_state = fn(states)
    jax.block_until_ready(compiled_state.step)
    t2 = time.perf_counter()
    run_time = max(t2 - t1, 1.0e-9)
    total_steps = int(num_envs) * int(steps)
    return {
        "compile_time": t1 - t0,
        "steps_per_second": int(steps) / run_time,
        "envs_per_second": total_steps / run_time,
        "device": str(jax.devices()[0]),
        "num_envs": int(num_envs),
        "steps": int(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--players", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.num_envs, args.steps, args.players), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
