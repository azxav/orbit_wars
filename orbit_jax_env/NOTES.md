# Orbit JAX Environment Notes

## Official Step Order

The implementation blueprint is the installed Kaggle file:
`.venv/Lib/site-packages/kaggle_environments/envs/orbit_wars/orbit_wars.py`.

The v1 JAX step follows the official no-comet turn order:

1. Launch valid fleets from player actions.
2. Apply production to owned planets.
3. Compute each planet's end-of-tick position from its initial orbit.
4. Move fleets using the official ship-speed curve.
5. During fleet movement, test swept fleet-vs-planet collision before bounds and sun removal.
6. Remove fleets that hit planets, leave bounds, or cross the sun.
7. Apply planet rotation.
8. Resolve grouped combat per planet.
9. Compute terminal status, raw ship scores, ranks, and training rewards.

Comet movement/expiration is implemented for imported official paths and for reset-generated approximate JAX schedules. Official Python-random comet path parity is not implemented.

## Commands

Run targeted tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_orbit_jax_env.py -q
```

Write the current parity report:

```powershell
.\.venv\Scripts\python.exe -m orbit_jax_env.parity.compare_official --output orbit_jax_env/parity_report.json
```

Implemented v1 parity cases:

- `case_001_no_actions`
- `case_002_simple_capture_static`
- `case_003_sun_collision`
- `case_004_bounds_collision`
- `case_005_two_fleets_combat`
- `case_006_planet_rotation`
- `case_007_moving_planet_collision`
- `case_008_random_scripted_2p_50_steps`
- `case_009_random_scripted_4p_50_steps`
- `case_010_imported_comet_movement`

Run a smoke benchmark:

```powershell
.\.venv\Scripts\python.exe -m orbit_jax_env.benchmarks --num_envs 8 --steps 10 --players 4
```

Run the requested benchmark shape:

```powershell
.\.venv\Scripts\python.exe -m orbit_jax_env.benchmarks --num_envs 1024 --steps 500 --players 4
```

Latest CPU result in this workspace:

```json
{
  "compile_time": 571.55580989999,
  "device": "cpu:0",
  "envs_per_second": 3119.133498914733,
  "num_envs": 1024,
  "steps": 500,
  "steps_per_second": 3.046028807533919
}
```

## PPO Rollout Adapter

`orbit_jax_env.rollout.jax_rollout(policy_apply, params, keys, config, steps)`:

- resets a batch with `jax.vmap(reset)`;
- runs `jax.lax.scan`;
- calls `policy_apply(params, obs)` each step;
- expects `policy_apply` to return raw `actions` shaped `[B, MAX_PLAYERS, MAX_ACTIONS_PER_PLAYER, 3]`;
- stores policy observations in `traj["obs"]` and post-step observations in `traj["next_obs"]`;
- stores `rewards`, `dones`, `logprobs`, `values`, and `info`.

## Comet Status

- Official comet planets and path arrays can be imported from Kaggle observations via `state_from_observation`.
- Imported active comets advance along fixed-size JAX path arrays and expire when the path ends.
- `reset(..., EnvConfig(enable_comets=True))` builds a deterministic JAX-native comet schedule.
- `step` spawns four symmetric comet planets at official spawn turns using the reset-generated schedule.
- JAX-native comet paths are approximate and deterministic; they are not byte-identical to Kaggle's Python `random.Random` comet path generation.

## Unsupported Or Approximate Parts

- JAX-native comet spawning is implemented with approximate deterministic paths, not official Python-random path parity.
- `reset` is deterministic JAX-native and structurally similar, but not byte-for-byte identical to Kaggle's Python `random.Random` map generator.
- The official parity harness compares exact official-seeded observations for the implemented no-comet cases plus imported active comet movement. Fleet positions use a slightly looser tolerance (`2e-4`) than planet positions (`1e-4`) to allow accumulated float32 movement drift.
- Batched launches are validated in aggregate per source planet for JAX speed. Official sequential multi-action edge cases from the same source may differ.
