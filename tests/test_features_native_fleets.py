from __future__ import annotations

from orbit_training_prep.features import TARGET_STATE_FEATURE_NAMES, _incoming_by_slot, target_state_features


def test_native_six_field_fleet_derives_target_and_eta_for_incoming_features() -> None:
    obs = {
        "step": 0,
        "episode_steps": 500,
        "players": 2,
        "planets": [
            [0, 1, 20.0, 50.0, 5.0, 50.0, 1.0],
            [1, 1, 80.0, 50.0, 5.0, 20.0, 1.0],
        ],
        # Native observation format: [id, owner, x, y, angle, ships].
        # This fleet starts outside the sun and travels right into planet id=1 within 5 steps.
        "fleets": [[99, 0, 60.0, 50.0, 0.0, 1000.0]],
    }

    incoming = _incoming_by_slot(obs, player_id=1, max_planets=2)
    assert incoming[1]["enemy_5"] == 1000.0
    assert incoming[1]["enemy_10"] == 1000.0
    assert incoming[1]["enemy_20"] == 1000.0

    target_features = target_state_features(obs, player_id=1, max_planets=2)
    hostile_idx = TARGET_STATE_FEATURE_NAMES.index("hostile_arrivals_before_10")
    projected_idx = TARGET_STATE_FEATURE_NAMES.index("projected_garrison_20")

    assert target_features[1, hostile_idx] == 10.0
    assert target_features[1, projected_idx] < 0.0
