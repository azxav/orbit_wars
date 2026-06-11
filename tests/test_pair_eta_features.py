from __future__ import annotations

import math

import pytest

from orbit_training_prep.features import PAIR_FEATURE_NAMES, _incoming_by_slot, pair_features_from_obs
from orbit_training_prep.schema import NOOP_TARGET_SLOT


def _obs() -> dict:
    return {
        "player": 0,
        "step": 0,
        "episode_steps": 500,
        "players": 2,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 80.0, 1.0],
            [1, 0, 80.0, 10.0, 1.0, 80.0, 1.0],
            [2, 1, 50.0, 10.0, 1.0, 20.0, 1.0],
        ],
        "initial_planets": [
            [0, 0, 10.0, 10.0, 1.0, 80.0, 1.0],
            [1, 0, 80.0, 10.0, 1.0, 80.0, 1.0],
            [2, 1, 50.0, 10.0, 1.0, 20.0, 1.0],
        ],
        "fleets": [],
    }


def _feature(row, name: str) -> float:
    return float(row[PAIR_FEATURE_NAMES.index(name)])


def test_enemy_fleet_before_candidate_eta_affects_pair_features() -> None:
    obs = _obs()
    obs["fleets"] = [{"owner": 1, "target_planet_id": 2, "eta": 5, "ships": 30}]

    row = pair_features_from_obs(obs, player_id=0, source_slot=0, max_planets=3)[2]

    assert _feature(row, "enemy_ships_before_our_arrival") == pytest.approx(0.3)
    assert "enemy_arrives_before_us_flag" not in PAIR_FEATURE_NAMES


def test_enemy_fleet_after_candidate_eta_does_not_affect_before_arrival_features() -> None:
    obs = _obs()
    obs["fleets"] = [{"owner": 1, "target_planet_id": 2, "eta": 80, "ships": 30}]

    row = pair_features_from_obs(obs, player_id=0, source_slot=0, max_planets=3)[2]

    assert _feature(row, "enemy_ships_before_our_arrival") == 0.0
    assert "friendly_support_before_us_flag" not in PAIR_FEATURE_NAMES


def test_same_target_from_different_sources_has_different_eta_aware_features() -> None:
    obs = _obs()
    obs["fleets"] = [{"owner": 1, "target_planet_id": 2, "eta": 13, "ships": 30}]

    far_row = pair_features_from_obs(obs, player_id=0, source_slot=0, max_planets=3)[2]
    near_row = pair_features_from_obs(obs, player_id=0, source_slot=1, max_planets=3)[2]

    assert _feature(far_row, "our_eta_norm") > _feature(near_row, "our_eta_norm")
    assert _feature(far_row, "enemy_ships_before_our_arrival") == pytest.approx(0.3)
    assert _feature(near_row, "enemy_ships_before_our_arrival") == 0.0


def test_native_six_field_fleet_before_candidate_eta_affects_pair_features() -> None:
    obs = _obs()
    # Native format: [id, owner, x, y, angle, ships]. Starts left of target 2 and flies right.
    obs["fleets"] = [[99, 1, 35.0, 10.0, 0.0, 1000.0]]

    row = pair_features_from_obs(obs, player_id=0, source_slot=0, max_planets=3)[2]

    assert _feature(row, "enemy_ships_before_our_arrival") == pytest.approx(10.0)
    assert "enemy_arrives_before_us_flag" not in PAIR_FEATURE_NAMES


def test_short_term_incoming_buckets_still_use_5_and_10_step_horizons() -> None:
    obs = _obs()
    obs["fleets"] = [
        {"owner": 1, "target_planet_id": 2, "eta": 5, "ships": 7},
        {"owner": 1, "target_planet_id": 2, "eta": 10, "ships": 11},
        {"owner": 0, "target_planet_id": 2, "eta": 10, "ships": 13},
    ]

    incoming = _incoming_by_slot(obs, player_id=0, max_planets=3)

    assert incoming[2]["enemy_5"] == 7.0
    assert incoming[2]["enemy_10"] == 18.0
    assert incoming[2]["friendly_5"] == 0.0
    assert incoming[2]["friendly_10"] == 13.0


def test_pair_feature_contract_is_compact_and_noop_row_is_zero() -> None:
    assert PAIR_FEATURE_NAMES == [
        "capture_ratio",
        "surplus_after_capture",
        "roi_prod_per_ship",
        "distance",
        "angle_sin",
        "angle_cos",
        "geom_viable_amount_frac",
        "safe_sendable_ships",
        "post_send_frac_capture",
        "our_eta_norm",
        "enemy_ships_before_our_arrival",
        "friendly_ships_before_our_arrival",
        "projected_garrison_at_our_arrival",
        "projected_owner_at_our_arrival",
        "target_capture_margin_at_arrival",
    ]
    features = pair_features_from_obs(_obs(), player_id=0, source_slot=0, max_planets=64)
    assert features.shape == (65, 15)
    assert features[NOOP_TARGET_SLOT].tolist() == [0.0] * 15


def test_pair_features_from_dense_uses_compact_contract() -> None:
    import numpy as np

    from orbit_training_prep.features import (
        PAIR_FEATURE_NAMES,
        PLANET_FEATURE_NAMES,
        TARGET_STATE_FEATURE_NAMES,
        pair_features_from_dense,
    )

    planet_features = np.zeros((64, len(PLANET_FEATURE_NAMES)), dtype=np.float32)
    ni = {name: i for i, name in enumerate(PLANET_FEATURE_NAMES)}
    planet_features[:, ni["alive"]] = 1.0
    planet_features[0, ni["rel_owner_own"]] = 1.0
    planet_features[0, ni["ships_log_norm"]] = 0.5
    planet_features[1, ni["rel_owner_enemy"]] = 1.0
    planet_features[1, ni["ships_log_norm"]] = 0.1
    target_state_features = np.zeros((64, len(TARGET_STATE_FEATURE_NAMES)), dtype=np.float32)
    target_mask = np.zeros((65,), dtype=bool)
    target_mask[1] = True
    amount_mask = np.zeros((65, 7), dtype=bool)
    amount_mask[1, 3] = True

    rows = pair_features_from_dense(
        planet_features,
        target_state_features,
        0,
        target_viability_mask=target_mask,
        amount_viability_mask=amount_mask,
    )

    assert rows.shape == (65, len(PAIR_FEATURE_NAMES))
    assert np.isfinite(rows).all()
