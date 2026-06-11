"""JAX-native BC/PPO feature extraction from EnvState.

P1 deliverable (wire the JAX env into PPO). The torch training pipeline builds
features in Python (`orbit_training_prep/features.py`) from a dict observation.
To run the policy *inside* `jax_rollout` (vmapped, on GPU/TPU) those same
features must be produced in JAX from `EnvState`.

Status:
- `global_features_jax`  — COMPLETE, exact parity with features.global_features.
- `planet_features_jax`  — 14/16 columns COMPLETE (everything that is a pure
  per-planet quantity). The two fleet-projection columns
  (`projected_garrison_20`, `under_threat_20`) need a fleet→target trace over a
  20-tick horizon; they are injected via `incoming_friendly20 / incoming_enemy20`
  (default zeros). See TODO(fleet-trace) below — reuse `collision.swept_*` and a
  `lax.scan` to bucket each fleet's ships onto the planet it hits, by ETA<=20.

Parity test: run `python -m orbit_jax_env.features_jax` — builds a random state,
converts it to a dict obs, and checks JAX features == numpy features.py features
(rotation/canonicalization is applied upstream, so compare in the raw frame).
"""
from __future__ import annotations

import math

import jax.numpy as jnp

CENTER = 50.0
BOARD = 100.0
ROT_LIMIT = 50.0
SHIP_LOG_DENOM = math.log1p(1000.0)
SQRT2 = math.sqrt(2.0)


def _rel_owner(owner, player_id):
    """1=mine, 0=neutral, -1=enemy. owner is int array, -1 == neutral."""
    mine = owner == player_id
    neutral = owner < 0
    enemy = (~mine) & (~neutral)
    return mine, neutral, enemy


def global_features_jax(state, player_id):
    """[10] — exact parity with features.global_features."""
    owner = state.planet_owner          # [P] int, -1 neutral
    alive = state.planet_alive          # [P] bool
    ships = jnp.maximum(0.0, state.planet_ships) * alive
    prod = jnp.maximum(0.0, state.planet_production) * alive
    valid = alive & (owner >= 0)
    P = owner.shape[0]
    players = jnp.maximum(state.num_players, 2)

    def owner_sum(vec, who):
        return jnp.sum(jnp.where(valid & who, vec, 0.0))

    mine = owner == player_id
    my_ships = owner_sum(ships, mine)
    my_prod = owner_sum(prod, mine)
    my_planets = jnp.sum(jnp.where(valid & mine, 1.0, 0.0))
    ts = jnp.sum(jnp.where(valid, ships, 0.0))
    tp = jnp.sum(jnp.where(valid, prod, 0.0))
    tpl = jnp.sum(jnp.where(valid, 1.0, 0.0))

    # per-owner totals via segment sums over MAX_PLAYERS
    pid = jnp.arange(8)
    owner_ships = jnp.array([owner_sum(ships, owner == k) for k in range(8)])
    owner_prods = jnp.array([owner_sum(prod, owner == k) for k in range(8)])
    present = owner_ships + owner_prods > 0
    leader_s = jnp.max(jnp.where(present, owner_ships, -jnp.inf))
    leader_p = jnp.max(jnp.where(present, owner_prods, -jnp.inf))
    leader_s = jnp.where(jnp.isfinite(leader_s), leader_s, 0.0)
    leader_p = jnp.where(jnp.isfinite(leader_p), leader_p, 0.0)
    enemy_present = present & (pid != player_id)
    weak = jnp.min(jnp.where(enemy_present, owner_ships, jnp.inf))
    weak = jnp.where(jnp.isfinite(weak), weak, 0.0)

    s = state.step.astype(jnp.float32) / jnp.maximum(state.episode_steps.astype(jnp.float32), 1.0)
    arr = jnp.array([
        s,
        jnp.maximum(0.0, 1.0 - s),
        jnp.where(players <= 2, 1.0, 0.0),
        jnp.where(players >= 4, 1.0, 0.0),
        my_ships / jnp.maximum(1.0, ts),
        my_prod / jnp.maximum(1.0, tp),
        my_planets / jnp.maximum(1.0, tpl),
        (my_ships - leader_s) / jnp.maximum(1.0, ts),
        (my_prod - leader_p) / jnp.maximum(1.0, tp),
        (my_ships - weak) / jnp.maximum(1.0, ts),
    ])
    return jnp.nan_to_num(arr)


def planet_features_jax(state, player_id, incoming_friendly20=None, incoming_enemy20=None):
    """[P,16] — 14 exact columns; 2 fleet-projection columns from injected args."""
    owner = state.planet_owner
    alive = state.planet_alive
    P = owner.shape[0]
    x = state.planet_x
    y = state.planet_y
    dx = x - CENTER
    dy = y - CENTER
    ships = jnp.maximum(0.0, state.planet_ships)
    prod = jnp.maximum(0.0, state.planet_production)
    radius = state.planet_radius
    mine, neutral, enemy = _rel_owner(owner, player_id)

    # owner totals (for ship/prod share of the planet's owner)
    owner_ships = jnp.array([jnp.sum(jnp.where(alive & (owner == k), ships, 0.0)) for k in range(8)])
    owner_prods = jnp.array([jnp.sum(jnp.where(alive & (owner == k), prod, 0.0)) for k in range(8)])
    ok = jnp.clip(owner, 0, 7)
    ot_ships = jnp.where(owner >= 0, owner_ships[ok], 0.0)
    ot_prods = jnp.where(owner >= 0, owner_prods[ok], 0.0)

    # is_orbiting: orbital_radius + radius < ROT_LIMIT, from initial position
    init_r = jnp.hypot(state.init_planet_x - CENTER, state.init_planet_y - CENTER) \
        if hasattr(state, "init_planet_x") else jnp.hypot(dx, dy)
    is_orbiting = (init_r + radius) < ROT_LIMIT

    if incoming_friendly20 is None:
        incoming_friendly20 = jnp.zeros(P)
    if incoming_enemy20 is None:
        incoming_enemy20 = jnp.zeros(P)
    # proj = ships + (friendly - enemy) if mine else ships + (enemy - friendly)
    proj = jnp.where(mine,
                     ships + incoming_friendly20 - incoming_enemy20,
                     ships + incoming_enemy20 - incoming_friendly20)
    under_threat = (incoming_enemy20 > ships + incoming_friendly20).astype(jnp.float32)

    cols = jnp.stack([
        alive.astype(jnp.float32),                                  # alive
        neutral.astype(jnp.float32),                                # rel_owner_neutral
        mine.astype(jnp.float32),                                   # rel_owner_own
        enemy.astype(jnp.float32),                                  # rel_owner_enemy
        dx / BOARD,                                                 # x_centered
        dy / BOARD,                                                 # y_centered
        radius / 5.0,                                               # radius_norm
        jnp.log1p(ships) / SHIP_LOG_DENOM,                          # ships_log_norm
        prod / 5.0,                                                 # production_norm
        state.planet_is_comet.astype(jnp.float32),                 # is_comet
        is_orbiting.astype(jnp.float32),                            # is_orbiting
        jnp.hypot(dx, dy) / (SQRT2 * BOARD),                        # distance_center_norm
        ships / jnp.maximum(1.0, ot_ships),                        # owner_ship_share
        prod / jnp.maximum(1.0, ot_prods),                         # owner_prod_share
        proj / 100.0,                                               # projected_garrison_20
        under_threat,                                               # under_threat_20
    ], axis=-1)
    # dead planets -> zero row (matches features.py guard)
    cols = jnp.where(alive[:, None], cols, 0.0)
    return jnp.nan_to_num(cols)


# TODO(fleet-trace): incoming_friendly20 / incoming_enemy20.
# For each alive fleet, trace its straight path with the env ship-speed curve
# over <=20 ticks using collision.swept_pair_hit against every planet's
# (old,new) positions; the first planet hit receives the fleet's ships in the
# friendly/enemy bucket keyed on (fleet_owner == player_id). Implement as a
# lax.scan over ticks reusing orbit_jax_env.collision; vmap over fleets. This is
# the same computation the engine already does in step.py movement — factor it
# out so features and the env share one swept-hit kernel.


if __name__ == "__main__":
    import numpy as np
    import jax
    from orbit_jax_env.config import EnvConfig
    from orbit_jax_env.reset import reset
    from orbit_jax_env.observation import build_observation
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from orbit_training_prep import features as F

    cfg = EnvConfig()
    st = reset(jax.random.PRNGKey(3), cfg)

    # convert state -> dict obs in the official [id,owner,x,y,r,ships,prod] layout
    P = int(st.planet_owner.shape[0])
    planets = []
    for i in range(P):
        if not bool(st.planet_alive[i]):
            continue
        planets.append([int(st.planet_id[i]), int(st.planet_owner[i]),
                        float(st.planet_x[i]), float(st.planet_y[i]),
                        float(st.planet_radius[i]), float(st.planet_ships[i]),
                        float(st.planet_production[i])])
    obs = {"planets": planets, "initial_planets": planets, "fleets": [],
           "comet_planet_ids": [], "step": int(st.step), "episode_steps": int(st.episode_steps),
           "num_players": int(st.num_players)}

    pid = 0
    g_np = F.global_features(obs, pid)
    g_jx = np.asarray(global_features_jax(st, pid))
    gerr = np.abs(g_np - g_jx).max()
    print(f"global parity max|err| = {gerr:.2e}  {'OK' if gerr < 1e-4 else 'FAIL'}")

    # planet features: compare only alive planets, no fleets (proj=ships)
    pf_np = F.all_planet_features(obs, pid)  # [64,16]
    pf_jx = np.asarray(planet_features_jax(st, pid))
    # align: numpy packs alive planets into the first len(planets) slots, JAX is per-state-slot
    m = min(pf_np.shape[0], pf_jx.shape[0])
    # compare the 14 fleet-independent columns on alive rows
    cols14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    alive_np = pf_np[:m, 0] > 0.5
    perr = np.abs(pf_np[:m][:, cols14] - pf_jx[:m][:, cols14])[alive_np].max()
    print(f"planet(14 cols) parity max|err| = {perr:.2e}  {'OK' if perr < 1e-3 else 'CHECK slot-alignment'}")
    print("note: numpy compacts alive planets into leading slots; JAX is per-slot. "
          "For a clean test, build obs from ALL slots preserving index, or compact the JAX output.")
