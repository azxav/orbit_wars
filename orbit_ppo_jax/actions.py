from __future__ import annotations

import jax.numpy as jnp

from orbit_jax_env.config import MAX_ACTIONS_PER_PLAYER, P_MAX

NOOP_TARGET_SLOT = P_MAX


def decode_amount_bin_jax(amount_bin, available, capture_needed):
    amount_bin, available, capture_needed = jnp.broadcast_arrays(
        jnp.asarray(amount_bin, dtype=jnp.int32),
        jnp.asarray(available, dtype=jnp.float32),
        jnp.asarray(capture_needed, dtype=jnp.float32),
    )
    available_i = jnp.maximum(0, jnp.floor(available).astype(jnp.int32))
    capture_i = jnp.where(available_i > 0, jnp.minimum(available_i, jnp.maximum(1, jnp.rint(capture_needed).astype(jnp.int32))), 0)
    values = jnp.stack(
        [
            jnp.zeros_like(available_i),
            jnp.minimum(1, available_i),
            capture_i,
            jnp.where(available_i > 0, jnp.maximum(1, jnp.rint(0.25 * available).astype(jnp.int32)), 0),
            jnp.where(available_i > 0, jnp.maximum(1, jnp.rint(0.50 * available).astype(jnp.int32)), 0),
            jnp.where(available_i > 0, jnp.maximum(1, jnp.rint(0.75 * available).astype(jnp.int32)), 0),
            available_i,
        ],
        axis=-1,
    )
    return jnp.take_along_axis(values, amount_bin[..., None], axis=-1)[..., 0]


def action_rows_from_choices(state, seat: int | jnp.ndarray, target_idx: jnp.ndarray, amount_idx: jnp.ndarray) -> jnp.ndarray:
    source_axis = jnp.arange(P_MAX, dtype=jnp.int32)
    source_mask = jnp.ones((P_MAX,), dtype=jnp.bool_)
    return action_rows_from_source_choices(state, seat, source_axis, target_idx, amount_idx, source_mask)


def action_rows_from_source_choices(
    state,
    seat: int | jnp.ndarray,
    source_slots: jnp.ndarray,
    target_idx: jnp.ndarray,
    amount_idx: jnp.ndarray,
    source_mask: jnp.ndarray,
) -> jnp.ndarray:
    seat = jnp.asarray(seat, dtype=jnp.int32)
    source_slots = jnp.clip(source_slots.astype(jnp.int32), 0, P_MAX - 1)
    target_clamped = jnp.clip(target_idx.astype(jnp.int32), 0, P_MAX - 1)
    is_noop = target_idx.astype(jnp.int32) == NOOP_TARGET_SLOT
    source_owned = state.planet_alive[source_slots] & (state.planet_owner[source_slots] == seat) & (state.planet_ships[source_slots] >= 1.0)
    target_alive = state.planet_alive[target_clamped]
    target_owner = state.planet_owner[target_clamped]
    target_ships = jnp.maximum(0.0, state.planet_ships[target_clamped])
    available = jnp.maximum(0.0, state.planet_ships[source_slots])
    capture_needed = jnp.where(target_owner == seat, 1.0, target_ships + 1.0)
    send = decode_amount_bin_jax(amount_idx.astype(jnp.int32), available, capture_needed).astype(jnp.float32)
    valid = source_mask & source_owned & target_alive & (~is_noop) & (target_clamped != source_slots) & (send > 0.0)
    angle = jnp.arctan2(state.planet_y[target_clamped] - state.planet_y[source_slots], state.planet_x[target_clamped] - state.planet_x[source_slots])
    compact_rows = jnp.stack(
        [
            jnp.where(valid, state.planet_id[source_slots].astype(jnp.float32), 0.0),
            jnp.where(valid, angle, 0.0),
            jnp.where(valid, send, 0.0),
        ],
        axis=-1,
    )
    rows = jnp.zeros((MAX_ACTIONS_PER_PLAYER, 3), dtype=jnp.float32)
    scatter_slots = jnp.clip(source_slots, 0, MAX_ACTIONS_PER_PLAYER - 1)
    in_action_table = source_slots < MAX_ACTIONS_PER_PLAYER
    scatter_rows = jnp.where(in_action_table[:, None], compact_rows, 0.0)
    return rows.at[scatter_slots].add(scatter_rows)
