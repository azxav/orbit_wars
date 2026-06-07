from __future__ import annotations


def test_ppo_recording_agent_forwards_skip_reasons_and_opening_counts() -> None:
    from orbit_ppo_training.policy import PolicyTurn
    from orbit_ppo_training.rollout_worker import PPORecordingAgent

    class FakePolicy:
        def act_observation(self, obs, player_id, *, deterministic, device, geometry):
            return PolicyTurn(
                moves=[],
                records=[],
                illegal_action_count=0,
                skipped_invalid_action_count=1,
                predicted_launches=1,
                no_op_source_decisions=0,
                entropy=0.0,
                value=0.0,
                skip_reasons={"amount_decoded_non_positive": 1},
                opening_prediction_counts={
                    "target": {"slot_1": 1},
                    "amount": {"none": 1},
                    "target_amount": {"slot_1|none": 1},
                },
            )

    agent = PPORecordingAgent(FakePolicy(), player_id=0, deterministic=True, device="cpu", geometry=object())

    assert agent({"player": 0, "step": 25, "planets": []}, {}) == []

    debug = agent.step_debug[0]["runtime_debug"]
    assert debug["skipped_invalid_decoded_actions"] == 1
    assert debug["skip_reasons"] == {"amount_decoded_non_positive": 1}
    assert debug["opening_prediction_counts"]["target"] == {"slot_1": 1}
    assert debug["opening_prediction_counts"]["amount"] == {"none": 1}
    assert debug["opening_prediction_counts"]["target_amount"] == {"slot_1|none": 1}


def test_ppo_early_launch_rate_uses_opening_0_100_bucket() -> None:
    from orbit_ppo_training.rollout_worker import _early_launch_rate

    row = {"launches_0_100": 5, "launches_100_250": 20}

    assert _early_launch_rate(row) == 0.05


def test_ppo_summary_aggregates_debug_counts_for_training_metrics() -> None:
    from orbit_ppo_training.metrics import summarize_rollout

    summary = summarize_rollout(
        [
            {
                "skip_reason_counts": {"geometry_no_viable_move": 2},
                "opening_prediction_target_counts": {"slot_1": 3},
                "opening_prediction_amount_counts": {"capture_plus_one": 3},
                "opening_prediction_target_amount_counts": {"slot_1|capture_plus_one": 3},
            },
            {
                "skip_reason_counts": {"amount_decoded_non_positive": 1},
                "opening_prediction_target_counts": {"noop": 1},
                "opening_prediction_amount_counts": {"none": 1},
                "opening_prediction_target_amount_counts": {"noop|none": 1},
            },
        ]
    )

    assert summary["skip_reason_counts"] == {"amount_decoded_non_positive": 1, "geometry_no_viable_move": 2}
    assert summary["opening_prediction_target_counts"] == {"noop": 1, "slot_1": 3}
    assert summary["opening_prediction_amount_counts"] == {"capture_plus_one": 3, "none": 1}
    assert summary["opening_prediction_target_amount_counts"] == {"noop|none": 1, "slot_1|capture_plus_one": 3}
