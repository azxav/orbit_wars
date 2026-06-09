from __future__ import annotations

import json
from pathlib import Path


def _obs(step: int = 0) -> dict:
    return {
        "player": 0,
        "step": step,
        "episode_steps": 500,
        "planets": [
            [101, 0, 10.0, 10.0, 1.0, 20.0, 1.0],
            [202, -1, 20.0, 10.0, 1.0, 4.0, 1.0],
            [303, 1, 50.0, 50.0, 1.0, 30.0, 1.0],
        ],
        "initial_planets": [
            [101, 0, 10.0, 10.0, 1.0, 20.0, 1.0],
            [202, -1, 20.0, 10.0, 1.0, 4.0, 1.0],
            [303, 1, 50.0, 50.0, 1.0, 30.0, 1.0],
        ],
        "fleets": [],
        "remainingOverageTime": 60.0,
    }


def _capture_obs(step: int) -> dict:
    obs = _obs(step=step)
    obs["planets"] = [
        [101, 0, 10.0, 10.0, 1.0, 25.0, 1.0],
        [202, 0, 20.0, 10.0, 1.0, 6.0, 1.0],
        [303, 1, 50.0, 50.0, 1.0, 30.0, 1.0],
    ]
    return obs


def test_runtime_batch_uses_training_feature_contract() -> None:
    from orbit_bc_eval.bc_agent_runtime import build_source_batch
    from orbit_training_prep.features import (
        GLOBAL_FEATURE_NAMES,
        PAIR_FEATURE_NAMES,
        PLANET_FEATURE_NAMES,
        TARGET_STATE_FEATURE_NAMES,
    )

    batch = build_source_batch(_obs(step=25), player_id=0, source_slot=0)

    assert tuple(batch["planet_features"].shape) == (1, 64, len(PLANET_FEATURE_NAMES))
    assert tuple(batch["global_features"].shape) == (1, len(GLOBAL_FEATURE_NAMES))
    assert tuple(batch["target_state_features"].shape) == (1, 64, len(TARGET_STATE_FEATURE_NAMES))
    assert tuple(batch["pair_features"].shape) == (1, 65, len(PAIR_FEATURE_NAMES))
    assert int(batch["source_slot"][0]) == 0
    assert abs(float(batch["global_features"][0, 0]) - 0.05) < 1e-6


def test_runtime_legal_filter_rejects_bad_moves() -> None:
    from orbit_bc_eval.bc_agent_runtime import validate_env_move

    obs = _obs()

    assert validate_env_move(obs, 0, [101, 1.0, 5]).ok
    assert not validate_env_move(obs, 0, [202, 1.0, 5]).ok
    assert not validate_env_move(obs, 0, [101, 1.0, 0]).ok
    assert not validate_env_move(obs, 0, [101, 1.0, 21]).ok


def test_runtime_masked_target_prediction_excludes_source_slot() -> None:
    import torch

    from orbit_bc_eval.bc_agent_runtime import masked_target_prediction
    from orbit_training_prep.schema import NOOP_TARGET_SLOT

    logits = torch.full((65,), -10.0)
    logits[0] = 100.0
    logits[NOOP_TARGET_SLOT] = 5.0

    assert masked_target_prediction(_obs(), 0, logits) == NOOP_TARGET_SLOT


def test_runtime_compact_debug_counts_opening_predictions_and_skip_reasons(monkeypatch) -> None:
    import torch

    from orbit_bc_eval import bc_agent_runtime
    from orbit_training_prep.schema import AMOUNT_BIN_NONE, NOOP_TARGET_SLOT

    class FakeModel:
        def __call__(self, batch):
            target_logits = torch.full((1, NOOP_TARGET_SLOT + 1), -10.0)
            target_logits[0, 1] = 10.0
            amount_logits = torch.full((1, 7), -10.0)
            amount_logits[0, AMOUNT_BIN_NONE] = 10.0
            return {"target_logits": target_logits, "amount_logits": amount_logits}

    monkeypatch.setattr(bc_agent_runtime, "_load_model_once", lambda checkpoint, device: (FakeModel(), {}))
    monkeypatch.setattr(bc_agent_runtime, "_geometry_once", lambda horizon, device="cpu": object())

    moves = bc_agent_runtime.agent(
        _obs(step=25),
        {"bc_checkpoint": "fake.pt", "device": "cpu", "geometry_horizon": 1, "debug": False},
    )
    debug = bc_agent_runtime.get_last_debug()

    assert moves == []
    assert debug["predicted_launches"] == 1
    assert debug["skipped_invalid_decoded_actions"] == 1
    assert debug["skip_reasons"] == {"amount_decoded_non_positive": 1}
    assert debug["opening_prediction_counts"]["target"] == {"slot_1": 1}
    assert debug["opening_prediction_counts"]["amount"] == {"none": 1}
    assert debug["opening_prediction_counts"]["target_amount"] == {"slot_1|none": 1}


def test_simple_expand_agent_targets_nearest_capturable_neutral() -> None:
    from orbit_bc_eval.base_agents import simple_expand_agent

    class FakeGeometry:
        def to_env_moves(self, **kwargs):
            assert kwargs["source_slots"].tolist() == [0]
            assert kwargs["target_slots"].tolist() == [1]
            assert kwargs["ships"].tolist() == [5]
            return [[101, 0.0, 5]]

    assert simple_expand_agent(_obs(), {}, geometry=FakeGeometry()) == [[101, 0.0, 5]]


def test_rollout_metrics_tracks_activity_legality_and_buckets() -> None:
    from orbit_bc_eval.rollout_metrics import RolloutMetrics

    metrics = RolloutMetrics(game_id="g1", bc_player_id=0, players=2, opponent="random")
    metrics.record_observation(_obs(step=0))
    metrics.record_step(
        step=10,
        actions=[[101, 0.0, 5]],
        illegal_actions=0,
        runtime_debug={
            "skipped_invalid_decoded_actions": 1,
            "no_op_source_decisions": 2,
            "predicted_launches": 1,
            "skip_reasons": {"geometry_no_viable_move": 1},
            "opening_prediction_counts": {
                "target": {"slot_1": 1},
                "amount": {"capture_plus_one": 1},
                "target_amount": {"slot_1|capture_plus_one": 1},
            },
        },
    )
    metrics.record_step(step=260, actions=[], illegal_actions=1, runtime_debug={})
    row = metrics.finalize(rewards=[1.0, -1.0], statuses=["DONE", "DONE"], final_obs=_obs(step=499))

    assert row["launches"] == 1
    assert row["launches_0_100"] == 1
    assert row["launches_250_430"] == 0
    assert row["illegal_actions"] == 1
    assert row["skipped_invalid_decoded_actions"] == 1
    assert row["skip_reason_counts"] == {"geometry_no_viable_move": 1}
    assert row["opening_prediction_target_counts"] == {"slot_1": 1}
    assert row["opening_prediction_amount_counts"] == {"capture_plus_one": 1}
    assert row["opening_prediction_target_amount_counts"] == {"slot_1|capture_plus_one": 1}
    assert row["no_op_source_decisions"] == 2
    assert row["predicted_launches"] == 1
    assert row["win"] is True
    assert row["rank"] == 1


def test_rollout_metrics_derives_gameplay_auc_and_action_rates() -> None:
    from orbit_bc_eval.rollout_metrics import RolloutMetrics

    metrics = RolloutMetrics(game_id="g1", bc_player_id=0, players=2, opponent="random")
    metrics.record_observation(_obs(step=0))
    metrics.record_step(
        step=10,
        actions=[[101, 0.0, 5], [101, 1.0, 4]],
        illegal_actions=0,
        runtime_debug={
            "predicted_launches": 4,
            "skipped_invalid_decoded_actions": 1,
            "no_op_source_decisions": 6,
            "returned_moves": 2,
        },
    )
    metrics.record_observation(_capture_obs(step=50))
    row = metrics.finalize(rewards=[1.0, -1.0], statuses=["DONE", "DONE"], final_obs=_capture_obs(step=499))

    assert row["final_owned_planets"] == 2
    assert row["owned_planets_auc"] == 5 / 3
    assert row["owned_planets_auc_0_100"] == 1.5
    assert row["planet_control_delta"] == 1
    assert row["first_capture_step"] == 50
    assert row["capture_efficiency"] == 0.5
    assert row["total_ships_auc"] == 82 / 3
    assert row["total_ships_auc_0_100"] == 25.5
    assert row["ship_delta_final_vs_initial"] == 11.0
    assert row["decode_success_rate"] == 0.5
    assert row["invalid_decode_rate"] == 0.25
    assert row["actual_launch_rate"] == 0.2
    assert row["noop_decision_rate"] == 0.6


def test_eval_report_writes_summary_jsonl_and_csv(tmp_path: Path) -> None:
    from orbit_bc_eval.eval_report import write_eval_report

    rows = [
        {
            "game_id": "g1",
            "bc_seat": 0,
            "players": 2,
            "opponent": "passive",
            "reward": 1.0,
            "rank": 1,
            "win": True,
            "launches": 2,
            "launches_0_100": 1,
            "timeout_count": 0,
            "illegal_actions": 0,
            "avg_owned_planets": 2.5,
            "owned_planets_auc": 2.5,
            "owned_planets_auc_0_100": 2.0,
            "final_owned_planets": 3,
            "planet_control_delta": 1,
            "first_capture_step": 40,
            "capture_efficiency": 0.5,
            "avg_total_ships": 30.0,
            "total_ships_auc": 30.0,
            "total_ships_auc_0_100": 25.0,
            "ship_delta_final_vs_initial": 5.0,
            "no_op_source_decisions": 3,
            "actual_returned_move_count": 2,
            "decode_success_rate": 0.5,
            "invalid_decode_rate": 0.25,
            "actual_launch_rate": 0.4,
            "noop_decision_rate": 0.6,
            "predicted_launch_rate": 1.0,
            "predicted_launches": 4,
            "skipped_invalid_decoded_actions": 1,
            "skip_reason_counts": {"geometry_no_viable_move": 1},
            "opening_prediction_target_counts": {"slot_1": 2},
            "opening_prediction_amount_counts": {"capture_plus_one": 2},
            "opening_prediction_target_amount_counts": {"slot_1|capture_plus_one": 2},
        }
    ]

    summary = write_eval_report(rows, out_dir=tmp_path, opponent="passive", players=2, bc_seats=[0])

    assert summary["num_games"] == 1
    assert summary["winrate"] == 1.0
    assert summary["total_launch_decisions"] == 5.0
    assert summary["total_launches"] == 2
    assert summary["early_launches_0_100"] == 1
    assert summary["total_predicted_launches"] == 4
    assert summary["total_no_op_source_decisions"] == 3
    assert summary["total_actual_returned_move_count"] == 2
    assert summary["total_skipped_invalid_decoded_actions"] == 1
    assert summary["average_final_owned_planets"] == 3.0
    assert summary["average_owned_planets_auc"] == 2.5
    assert summary["average_total_ships_auc"] == 30.0
    assert summary["average_decode_success_rate"] == 0.5
    assert "gameplay_score" in summary
    assert summary["skip_reason_counts"] == {"geometry_no_viable_move": 1}
    assert summary["opening_prediction_target_counts"] == {"slot_1": 2}
    assert summary["opening_prediction_amount_counts"] == {"capture_plus_one": 2}
    assert summary["opening_prediction_target_amount_counts"] == {"slot_1|capture_plus_one": 2}
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "games.jsonl").read_text(encoding="utf-8").strip().startswith("{")
    assert "game_id" in (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["opponent"] == "passive"


def test_local_match_defaults_run_full_bc_v1_against_root_checkpoint() -> None:
    from orbit_bc_eval.run_local_matches import build_arg_parser

    args = build_arg_parser().parse_args([])

    assert args.opponent == "bc_checkpoint"
    assert args.players == "2"
    assert Path(args.bc_checkpoint).as_posix().endswith("bc_checkpoints/full_bc_v1/best/checkpoint.pt")
    assert Path(args.opponent_bc_checkpoint).name == "checkpoint.pt"
    assert args.out_dir == "bc_eval_runs/full_bc_v1_best_vs_checkpoint"


def test_bc_checkpoint_agent_wrapper_passes_its_checkpoint_config(monkeypatch) -> None:
    from orbit_bc_eval import run_local_matches

    calls = []

    def fake_agent(obs, config):
        calls.append(dict(config))
        return [[101, 0.0, 1]]

    monkeypatch.setattr(run_local_matches.bc_agent_runtime, "agent", fake_agent)
    wrapper = run_local_matches._make_bc_agent_for_checkpoint(
        checkpoint="alt/checkpoint.pt",
        device="cpu",
        geometry_horizon=80,
        debug=True,
        player_id=1,
    )

    assert wrapper({"player": 1, "planets": []}, {}) == [[101, 0.0, 1]]
    assert calls[0]["bc_checkpoint"] == "alt/checkpoint.pt"
    assert calls[0]["device"] == "cpu"
    assert calls[0]["geometry_horizon"] == 80
    assert calls[0]["debug"] is True


def test_compare_metric_relevance_pairs_runs_and_marks_keep_drop(tmp_path: Path, capsys) -> None:
    from orbit_bc_eval.compare_metric_relevance import main

    old = tmp_path / "old" / "games.jsonl"
    new = tmp_path / "new" / "games.jsonl"
    old.parent.mkdir()
    new.parent.mkdir()
    old.write_text(
        "\n".join(
            [
                json.dumps({"game_id": "game_00000", "players": 2, "seed": 1, "bc_seat": 0, "opponent": "h", "reward": 0.0, "rank": 2, "owned_planets_auc": 1.0, "invalid_decode_rate": 0.5}),
                json.dumps({"game_id": "game_00001", "players": 2, "seed": 2, "bc_seat": 1, "opponent": "h", "reward": 0.0, "rank": 2, "owned_planets_auc": 1.0, "invalid_decode_rate": 0.2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    new.write_text(
        "\n".join(
            [
                json.dumps({"game_id": "game_00000", "players": 2, "seed": 1, "bc_seat": 0, "opponent": "h", "reward": 1.0, "rank": 1, "owned_planets_auc": 3.0, "invalid_decode_rate": 0.1}),
                json.dumps({"game_id": "game_00001", "players": 2, "seed": 2, "bc_seat": 1, "opponent": "h", "reward": -1.0, "rank": 2, "owned_planets_auc": 0.5, "invalid_decode_rate": 0.4}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    main([str(old.parent), str(new.parent)])

    out = capsys.readouterr().out
    assert "metric,spearman_corr_with_reward,spearman_corr_with_rank,direction,keep_drop" in out
    assert "owned_planets_auc,1.0,-1.0,higher,keep" in out
    assert "invalid_decode_rate,-1.0,1.0,lower,keep" in out
