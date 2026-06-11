"""GPU self-play smoke: run jax_rollout with the greedy JAX policy, measure
games/sec and confirm games terminate with winners."""
import time, jax, jax.numpy as jnp
from .config import EnvConfig
from .rollout import jax_rollout
from .jax_policy import greedy_policy_apply

def main(num_envs=1024, steps=500, players=4):
    cfg = EnvConfig(num_players=players, episode_steps=steps)
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    t0 = time.time()
    final_states, traj = jax_rollout(greedy_policy_apply, {}, keys, cfg, steps=steps)
    traj["rewards"].block_until_ready()
    dt = time.time() - t0
    # rewards [steps, B, players]; dones [steps, B]
    rew = traj["rewards"]
    dones = traj["dones"]
    done_any = int(jnp.sum(jnp.any(dones, axis=0)))
    gps = num_envs / dt
    print(f"device={jax.default_backend()} num_envs={num_envs} steps={steps} players={players}")
    print(f"wall={dt:.1f}s  games/sec={gps:.1f}  envs_reaching_done={done_any}/{num_envs}")
    print(f"rewards shape={rew.shape}  mean_final={[round(float(x),3) for x in jnp.mean(rew[-1],axis=0)]}")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv)>1 else 1024
    main(num_envs=n)
