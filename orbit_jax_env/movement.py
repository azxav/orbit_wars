from __future__ import annotations

import jax.numpy as jnp

from .config import CENTER, LOG_1000, ROTATION_RADIUS_LIMIT


def fleet_speed(ships: jnp.ndarray, max_speed: jnp.ndarray | float = 6.0) -> jnp.ndarray:
    safe = jnp.maximum(ships, 1.0)
    ratio = jnp.minimum(jnp.log(safe) / LOG_1000, 1.0)
    speed = 1.0 + (max_speed - 1.0) * ratio**1.5
    return jnp.minimum(speed, max_speed)


def rotated_planet_positions(
    initial_x: jnp.ndarray,
    initial_y: jnp.ndarray,
    current_x: jnp.ndarray,
    current_y: jnp.ndarray,
    radius: jnp.ndarray,
    alive: jnp.ndarray,
    is_comet: jnp.ndarray,
    angular_velocity: jnp.ndarray,
    step_value: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dx = initial_x - CENTER
    dy = initial_y - CENTER
    orbit_radius = jnp.sqrt(dx * dx + dy * dy)
    orbiting = alive & (~is_comet) & ((orbit_radius + radius) < ROTATION_RADIUS_LIMIT)
    angle0 = jnp.arctan2(dy, dx)
    angle = angle0 + angular_velocity * step_value.astype(jnp.float32)
    nx = CENTER + orbit_radius * jnp.cos(angle)
    ny = CENTER + orbit_radius * jnp.sin(angle)
    return jnp.where(orbiting, nx, current_x), jnp.where(orbiting, ny, current_y)
