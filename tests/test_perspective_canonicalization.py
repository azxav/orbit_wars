from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from orbit_training_prep.schema import NOOP_TARGET_SLOT, wrap_angle


def _p1_replay(path: Path) -> None:
    replay = {
        "info": {"EpisodeId": "p1-canon"},
        "configuration": {"episodeSteps": 3},
        "rewards": [0.0, 1.0],
        "steps": [
            [
                {"observation": {}, "status": "ACTIVE", "reward": 0},
                {
                    "observation": {
                        "step": 0,
                        "player": 1,
                        "episode_steps": 3,
                        "players": 2,
                        "planets": [
                            [101, 0, 10.0, 50.0, 1.0, 5.0, 1.0],
                            [202, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
                            [303, -1, 50.0, 70.0, 1.0, 4.0, 1.0],
                        ],
                        "initial_planets": [
                            [101, 0, 10.0, 50.0, 1.0, 5.0, 1.0],
                            [202, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
                            [303, -1, 50.0, 70.0, 1.0, 4.0, 1.0],
                        ],
                        "fleets": [],
                        "comets": [],
                    },
                    "status": "ACTIVE",
                    "reward": 0,
                },
            ],
            [
                {"observation": {}, "action": [], "status": "ACTIVE", "reward": 0},
                {"observation": {}, "action": [[202, math.pi, 10]], "status": "ACTIVE", "reward": 0},
            ],
            [
                {"observation": {}, "action": [], "status": "DONE", "reward": 0},
                {"observation": {}, "action": [], "status": "DONE", "reward": 1},
            ],
        ],
    }
    path.write_text(json.dumps(replay), encoding="utf-8")


def test_canonicalize_p1_observation_rotates_owner_coords_slots_and_action_to_p0_frame() -> None:
    from orbit_training_prep.canonical import canonicalize_observation, canonicalize_action

    obs = {
        "player": 1,
        "players": 2,
        "step": 0,
        "planets": [
            [101, 0, 10.0, 50.0, 1.0, 5.0, 1.0],
            [202, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
            [303, -1, 50.0, 70.0, 1.0, 4.0, 1.0],
        ],
        "initial_planets": [],
        "fleets": [],
    }

    result = canonicalize_observation(obs, 1)
    canon = result.obs

    assert canon["player"] == 0
    assert canon["canonicalized_player_id"] == 1
    assert canon["perspective_canonicalized"] is True
    assert result.rotation_radians == -math.pi

    # Planet 202 was P1-owned at raw x=90; in canonical P0 frame it becomes owner 0 at x=10.
    slot_202 = result.id_to_canonical_slot[202]
    p202 = canon["planets"][slot_202]
    assert int(p202[1]) == 0
    assert p202[2] == np.float64(10.0) or abs(float(p202[2]) - 10.0) < 1e-6
    assert abs(float(p202[3]) - 50.0) < 1e-6

    # Raw action from x=90 to the left is pi; after rotation into P0 frame it points right (0).
    transformed = list(canonicalize_action([[202, math.pi, 10]], result))
    assert transformed[0][0] == 202
    assert abs(wrap_angle(float(transformed[0][1]))) < 1e-6
    assert transformed[0][2] == 10


def test_dataset_builder_canonicalizes_p1_replay_rows_and_records_metadata(tmp_path: Path) -> None:
    from orbit_training_prep.dataset_builder import DatasetBuilder
    from orbit_training_prep.source_turn_store import SourceTurnDatasetReader

    replay_path = tmp_path / "p1_replay.json"
    _p1_replay(replay_path)

    out_dir = tmp_path / "dataset"
    metadata = DatasetBuilder(horizon=12, device="cpu", backend="lite", write_debug_jsonl=True).build_from_replay(replay_path, out_dir)
    reader = SourceTurnDatasetReader(out_dir)

    assert metadata["perspective_canonicalization"]["enabled"] is True
    assert metadata["perspective_canonicalization"]["frame"] == "p0"
    assert metadata["stats"]["canonicalized_player_steps"] == 1
    assert int(reader.samples["target_label"][0]) != NOOP_TARGET_SLOT
    assert bool(reader.samples["target_mask"][0, int(reader.samples["target_label"][0])])

    debug_rows = [json.loads(line) for line in (out_dir / "debug" / "source_turn_rows.jsonl").read_text().splitlines()]
    launch_rows = [row for row in debug_rows if int(row["target_slot_label"]) != NOOP_TARGET_SLOT]
    assert launch_rows
    assert launch_rows[0]["player_id"] == 0
    assert launch_rows[0]["original_player_id"] == 1
    assert launch_rows[0]["source_planet_id"] == 202


def test_runtime_canonical_move_is_rotated_back_to_original_frame() -> None:
    from orbit_training_prep.canonical import canonicalize_observation, uncanonicalize_move

    obs = {
        "player": 1,
        "players": 2,
        "planets": [
            [101, 0, 10.0, 50.0, 1.0, 5.0, 1.0],
            [202, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
        ],
    }
    transform = canonicalize_observation(obs, 1)

    move = uncanonicalize_move([202, 0.0, 10], transform)

    assert move[0] == 202
    assert abs(wrap_angle(float(move[1]) - math.pi)) < 1e-6
    assert move[2] == 10


def _p1_runtime_obs() -> dict:
    return {
        "player": 1,
        "players": 2,
        "step": 0,
        "episode_steps": 500,
        "planets": [
            [101, 0, 10.0, 50.0, 1.0, 5.0, 1.0],
            [202, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
        ],
        "initial_planets": [
            [101, 0, 10.0, 50.0, 1.0, 5.0, 1.0],
            [202, 1, 90.0, 50.0, 1.0, 30.0, 1.0],
        ],
        "fleets": [],
        "remainingOverageTime": 60.0,
    }


def test_bc_agent_runtime_canonicalizes_p1_and_rotates_decoded_move_back(monkeypatch) -> None:
    import torch

    from orbit_bc_eval import bc_agent_runtime
    from orbit_training_prep.schema import NOOP_TARGET_SLOT, wrap_angle

    class FakeModel:
        def __call__(self, batch):
            target_logits = torch.full((1, NOOP_TARGET_SLOT + 1), -10.0)
            target_logits[0, 1] = 10.0
            amount_logits = torch.full((1, 7), -10.0)
            amount_logits[0, 4] = 10.0
            return {"target_logits": target_logits, "amount_logits": amount_logits}

    monkeypatch.setattr(bc_agent_runtime, "_load_model_once", lambda checkpoint, device: (FakeModel(), {}))
    monkeypatch.setattr(bc_agent_runtime, "_geometry_once", lambda horizon, device="cpu": object())
    target_mask = torch.zeros((64, 65), dtype=torch.bool)
    amount_mask = torch.zeros((64, 65, 7), dtype=torch.bool)
    target_mask[:, NOOP_TARGET_SLOT] = True
    amount_mask[:, NOOP_TARGET_SLOT, 0] = True
    target_mask[0, 1] = True
    amount_mask[0, 1, 4] = True

    def fake_masks(obs, player_id, **kwargs):
        assert player_id == 0
        assert obs["player"] == 0
        assert obs["perspective_canonicalized"] is True
        return target_mask.numpy(), amount_mask.numpy()

    def fake_decode(obs, player_id, source_planet_id, target_logits, amount_logits, geometry, **kwargs):
        assert player_id == 0
        assert source_planet_id == 202
        return [source_planet_id, 0.0, 10]

    monkeypatch.setattr(bc_agent_runtime, "compute_viability_masks", fake_masks)
    monkeypatch.setattr(bc_agent_runtime, "decode_bc_prediction", fake_decode)

    moves = bc_agent_runtime.agent(_p1_runtime_obs(), {"bc_checkpoint": "fake.pt", "device": "cpu", "geometry_horizon": 1})

    assert len(moves) == 1
    assert moves[0][0] == 202
    assert abs(wrap_angle(float(moves[0][1]) - math.pi)) < 1e-6
    assert moves[0][2] == 10


def test_ppo_policy_canonicalizes_p1_and_records_canonical_decision(monkeypatch) -> None:
    import torch
    from types import SimpleNamespace

    import orbit_ppo_training.policy as policy_mod
    from orbit_ppo_training.policy import PPOPolicy
    from orbit_training_prep.features import PAIR_FEATURE_NAMES
    from orbit_training_prep.schema import NOOP_TARGET_SLOT, wrap_angle

    class DummyPolicy(torch.nn.Module):
        act_observation = PPOPolicy.act_observation

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(amount_bins=7)

        def eval(self):
            return self

        def forward(self, batch):
            target_logits = torch.full((1, 65), -10.0)
            target_logits[0, 1] = 100.0
            amount_logits = torch.full((1, 7), -10.0)
            amount_logits[0, 4] = 10.0
            return {"target_logits": target_logits, "amount_logits": amount_logits, "value": torch.tensor([0.0])}

    target_mask_table = torch.zeros((64, 65), dtype=torch.bool).numpy()
    amount_mask_table = torch.zeros((64, 65, 7), dtype=torch.bool).numpy()
    target_mask_table[:, NOOP_TARGET_SLOT] = True
    amount_mask_table[:, NOOP_TARGET_SLOT, 0] = True
    target_mask_table[0, 1] = True
    amount_mask_table[0, 1, 4] = True

    def fake_masks(obs, player_id, **kwargs):
        assert player_id == 0
        assert obs["perspective_canonicalized"] is True
        return target_mask_table, amount_mask_table

    def fake_build_source_batch(obs, player_id, source_slot, device="cpu", **kwargs):
        assert player_id == 0
        assert source_slot == 0
        assert obs["planets"][source_slot][0] == 202
        return {
            "planet_features": torch.zeros((1, 64, 16), device=device),
            "global_features": torch.zeros((1, 10), device=device),
            "target_state_features": torch.zeros((1, 64, 9), device=device),
            "pair_features": torch.zeros((1, 65, len(PAIR_FEATURE_NAMES)), device=device),
            "source_slot": torch.tensor([source_slot], device=device),
        }

    def fake_decode(obs, player_id, source_planet_id, target_logits, amount_logits, geometry, **kwargs):
        assert player_id == 0
        assert source_planet_id == 202
        return [source_planet_id, 0.0, 10]

    monkeypatch.setattr(policy_mod, "compute_viability_masks", fake_masks)
    monkeypatch.setattr(policy_mod, "build_source_batch", fake_build_source_batch)
    monkeypatch.setattr(policy_mod, "decode_bc_prediction", fake_decode)

    turn = DummyPolicy().act_observation(_p1_runtime_obs(), 1, deterministic=True, device="cpu", geometry=object())

    assert len(turn.moves) == 1
    assert abs(wrap_angle(float(turn.moves[0][1]) - math.pi)) < 1e-6
    assert turn.records[0].source_slot == 0
    assert turn.records[0].target_action == 1
