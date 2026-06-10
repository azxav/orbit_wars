from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


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


def test_ppo_act_observation_uses_geometry_target_mask(monkeypatch) -> None:
    import orbit_ppo_training.policy as policy_mod
    from orbit_ppo_training.policy import PPOPolicy

    class DummyPolicy(nn.Module):
        act_observation = PPOPolicy.act_observation

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(amount_bins=7)

        def forward(self, batch):
            target_logits = torch.full((1, 65), -10.0)
            target_logits[0, 1] = 100.0
            target_logits[0, 2] = 10.0
            amount_logits = torch.full((1, 7), -10.0)
            amount_logits[0, 4] = 10.0
            return {"target_logits": target_logits, "amount_logits": amount_logits, "value": torch.tensor([0.0])}

    obs = {"player": 0, "step": 0, "planets": [[101, 0, 0, 0, 1, 20, 1], [202, 1, 10, 0, 1, 5, 1], [303, 1, 20, 0, 1, 5, 1]]}
    selected: dict[str, int] = {}
    target_mask = torch.zeros(65, dtype=torch.bool)
    target_mask[2] = True
    target_mask_table = torch.zeros((64, 65), dtype=torch.bool).numpy()
    target_mask_table[0, 2] = True
    amount_mask_table = torch.zeros((64, 65, 7), dtype=torch.bool).numpy()
    amount_mask_table[0, 2, 4] = True

    monkeypatch.setattr(
        policy_mod,
        "build_source_batch",
        lambda obs, player_id, source_slot, device="cpu": {
            "planet_features": torch.zeros((1, 64, 16)),
            "global_features": torch.zeros((1, 10)),
            "target_state_features": torch.zeros((1, 64, 9)),
            "pair_features": torch.zeros((1, 65, 22)),
            "source_slot": torch.tensor([source_slot]),
        },
    )
    monkeypatch.setattr(policy_mod, "compute_viability_masks", lambda obs, player_id, **kwargs: (target_mask_table, amount_mask_table))

    def fake_decode(obs, player_id, source_planet_id, target_logits, amount_logits, geometry, **kwargs):
        selected["target"] = int(torch.argmax(target_logits).item())
        return [source_planet_id, 0.0, 10]

    monkeypatch.setattr(policy_mod, "decode_bc_prediction", fake_decode)

    turn = DummyPolicy().act_observation(obs, 0, deterministic=True, device="cpu", geometry=object())

    assert selected["target"] == 2
    assert turn.records[0].target_action == 2
    assert turn.records[0].target_mask.tolist() == target_mask.tolist()
