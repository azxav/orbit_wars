from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def test_dynamic_seat_features_are_jittable() -> None:
    from orbit_jax_env.state import manual_state
    from orbit_ppo_jax.features import build_bc_features_for_seat

    state = manual_state(
        planet_rows=[
            [10, 0, 20.0, 50.0, 2.0, 20.0, 3.0],
            [11, 1, 80.0, 50.0, 2.0, 18.0, 2.0],
            [12, -1, 50.0, 80.0, 2.0, 5.0, 1.0],
        ],
        num_players=2,
    )

    @jax.jit
    def build(seat):
        feats = build_bc_features_for_seat(state, seat, source_cap=4)
        return feats.global_features, feats.source_mask, feats.active_source_count

    gf0, mask0, count0 = build(jnp.asarray(0, dtype=jnp.int32))
    gf1, mask1, count1 = build(jnp.asarray(1, dtype=jnp.int32))

    assert gf0.shape == (10,)
    assert gf1.shape == (10,)
    assert int(count0) == 1
    assert int(count1) == 1
    assert np.asarray(mask0).tolist()[0] is True
    assert np.asarray(mask1).tolist()[0] is True


def test_dynamic_seat_action_rows_are_jittable() -> None:
    from orbit_jax_env.state import manual_state
    from orbit_ppo_jax.actions import action_rows_from_source_choices

    state = manual_state(
        planet_rows=[
            [10, 0, 20.0, 50.0, 2.0, 20.0, 3.0],
            [11, 1, 80.0, 50.0, 2.0, 18.0, 2.0],
            [12, -1, 50.0, 80.0, 2.0, 5.0, 1.0],
        ],
        num_players=2,
    )

    @jax.jit
    def rows(seat):
        return action_rows_from_source_choices(
            state,
            seat,
            jnp.asarray([seat], dtype=jnp.int32),
            jnp.asarray([2], dtype=jnp.int32),
            jnp.asarray([2], dtype=jnp.int32),
            jnp.asarray([True]),
        )

    rows0 = np.asarray(rows(jnp.asarray(0, dtype=jnp.int32)))
    rows1 = np.asarray(rows(jnp.asarray(1, dtype=jnp.int32)))

    assert rows0[0, 0] == 10.0
    assert rows1[1, 0] == 11.0


def test_learner_terminal_fields_use_rotated_seat() -> None:
    from orbit_ppo_jax.train import _learner_terminal_fields

    reward, rank = _learner_terminal_fields(
        jnp.asarray([10.0, -3.0, 2.0, 4.0], dtype=jnp.float32),
        jnp.asarray([0, 3, 1, 2], dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )

    assert float(reward) == -3.0
    assert int(rank) == 3


def test_pfsp_manifest_roundtrip(tmp_path: Path) -> None:
    from orbit_ppo_jax.pfsp import PFSPEntry, PFSPEntryStats, PFSPManifest, load_manifest, save_manifest

    manifest = PFSPManifest(
        version=1,
        players=4,
        max_policy_slots=4,
        entries=[
            PFSPEntry("anchor_simple_heuristic_jax", "simple_heuristic_jax", None, True, True, None, 0),
            PFSPEntry("initial_bc", "frozen_policy", 0, True, True, "bc/checkpoint.pt", 0),
        ],
        stats={"initial_bc": PFSPEntryStats()},
    )
    save_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_manifest(tmp_path / "manifest.json")

    assert loaded.entries[0].id == "anchor_simple_heuristic_jax"
    assert loaded.entries[1].slot == 0
    assert loaded.stats["initial_bc"].games == 0


def test_pfsp_sampling_prefers_mid_score_entries() -> None:
    from orbit_ppo_jax.pfsp import PFSPEntryStats, pfsp_weight

    easy = pfsp_weight(
        PFSPEntryStats(games=64, score_sum=60.0),
        hard_low=0.20,
        hard_high=0.55,
        hard_bonus=0.15,
        exploration_bonus=0.10,
    )
    mid = pfsp_weight(
        PFSPEntryStats(games=64, score_sum=24.0),
        hard_low=0.20,
        hard_high=0.55,
        hard_bonus=0.15,
        exploration_bonus=0.10,
    )
    unknown = pfsp_weight(
        PFSPEntryStats(games=0, score_sum=0.0),
        hard_low=0.20,
        hard_high=0.55,
        hard_bonus=0.15,
        exploration_bonus=0.10,
    )

    assert mid > easy
    assert unknown > 0.0


def test_manifest_updates_anchor_stats_from_kind_totals() -> None:
    from orbit_ppo_jax.pfsp import OPP_SIMPLE_HEURISTIC, build_initial_manifest, update_manifest_from_slot_stats

    manifest = build_initial_manifest(players=2, max_policy_slots=4, bc_checkpoint="bc.pt")
    updated = update_manifest_from_slot_stats(
        manifest,
        slot_games=[0, 0, 0, 0],
        slot_score_sum=[0.0, 0.0, 0.0, 0.0],
        slot_reward_sum=[0.0, 0.0, 0.0, 0.0],
        slot_rank_sum=[0.0, 0.0, 0.0, 0.0],
        kind_games={OPP_SIMPLE_HEURISTIC: 2},
        kind_score_sum={OPP_SIMPLE_HEURISTIC: 1.0},
        kind_reward_sum={OPP_SIMPLE_HEURISTIC: 3.0},
        kind_rank_sum={OPP_SIMPLE_HEURISTIC: 1.0},
        update_index=7,
    )

    stats = updated.stats["anchor_simple_heuristic_jax"]
    assert stats.games == 2
    assert stats.score_sum == 1.0
    assert stats.last_played_update == 7


def test_manifest_slot_stats_skip_inactive_reused_entries() -> None:
    from orbit_ppo_jax.pfsp import PFSPEntry, PFSPEntryStats, PFSPManifest, update_manifest_from_slot_stats

    manifest = PFSPManifest(
        version=1,
        players=2,
        max_policy_slots=3,
        entries=[
            PFSPEntry("initial_bc", "frozen_policy", 0, True, True, "bc", 0),
            PFSPEntry("old", "frozen_policy", 1, False, False, "old", 1),
            PFSPEntry("new", "frozen_policy", 1, False, True, "new", 2),
        ],
        stats={
            "initial_bc": PFSPEntryStats(),
            "old": PFSPEntryStats(),
            "new": PFSPEntryStats(),
        },
    )

    updated = update_manifest_from_slot_stats(
        manifest,
        slot_games=[0, 1, 0],
        slot_score_sum=[0.0, 1.0, 0.0],
        slot_reward_sum=[0.0, 2.0, 0.0],
        slot_rank_sum=[0.0, 0.0, 0.0],
        update_index=3,
    )

    assert updated.stats["old"].games == 0
    assert updated.stats["new"].games == 1


def test_match_plan_rotates_learner_seat() -> None:
    from orbit_ppo_jax.pfsp import OPP_NONE, build_initial_manifest, build_match_plan

    manifest = build_initial_manifest(players=4, max_policy_slots=4, bc_checkpoint="bc.pt")
    plan = build_match_plan(
        manifest,
        rng=np.random.default_rng(0),
        envs=8,
        players=4,
        learner_seat_mode="rotate",
        anchor_fraction=0.25,
        layout="one_pfsp_two_anchors",
    )

    np.testing.assert_array_equal(np.asarray(plan.learner_seat), np.asarray([0, 1, 2, 3, 0, 1, 2, 3]))
    for env_i, seat in enumerate(np.asarray(plan.learner_seat)):
        assert int(np.asarray(plan.opponent_kind)[env_i, seat]) == OPP_NONE


def test_match_plan_uses_frozen_policy_when_anchor_fraction_is_zero() -> None:
    from orbit_ppo_jax.pfsp import OPP_FROZEN_POLICY, build_initial_manifest, build_match_plan

    manifest = build_initial_manifest(players=2, max_policy_slots=4, bc_checkpoint="bc.pt")
    plan = build_match_plan(
        manifest,
        rng=np.random.default_rng(0),
        envs=4,
        players=2,
        learner_seat_mode="fixed0",
        anchor_fraction=0.0,
        layout="one_pfsp_two_anchors",
    )

    assert np.asarray(plan.opponent_kind)[:, 1].tolist() == [OPP_FROZEN_POLICY] * 4
    assert np.asarray(plan.opponent_slot)[:, 1].tolist() == [0] * 4


def test_match_plan_weights_frozen_entries_by_pfsp_score() -> None:
    from orbit_ppo_jax.pfsp import PFSPEntry, PFSPEntryStats, PFSPManifest, build_match_plan

    manifest = PFSPManifest(
        version=1,
        players=2,
        max_policy_slots=4,
        entries=[
            PFSPEntry("anchor_simple_heuristic_jax", "simple_heuristic_jax", None, True, True, None, 0),
            PFSPEntry("easy", "frozen_policy", 0, False, True, "easy", 1),
            PFSPEntry("mid", "frozen_policy", 1, False, True, "mid", 2),
        ],
        stats={
            "easy": PFSPEntryStats(games=64, score_sum=60.0),
            "mid": PFSPEntryStats(games=64, score_sum=24.0),
        },
    )

    plan = build_match_plan(
        manifest,
        rng=np.random.default_rng(7),
        envs=200,
        players=2,
        learner_seat_mode="fixed0",
        anchor_fraction=0.0,
        layout="one_pfsp_two_anchors",
    )
    slots = np.asarray(plan.opponent_slot)[:, 1]

    assert int(np.sum(slots == 1)) > int(np.sum(slots == 0))


def test_match_plan_accepts_explicit_pfsp_weight_parameters() -> None:
    from orbit_jax_env.config import MAX_PLAYERS
    from orbit_ppo_jax.pfsp import PFSPEntry, PFSPEntryStats, PFSPManifest, build_match_plan

    manifest = PFSPManifest(
        version=1,
        players=2,
        max_policy_slots=4,
        entries=[
            PFSPEntry("easy", "frozen_policy", 0, False, True, "easy", 1),
            PFSPEntry("mid", "frozen_policy", 1, False, True, "mid", 2),
        ],
        stats={
            "easy": PFSPEntryStats(games=64, score_sum=60.0),
            "mid": PFSPEntryStats(games=64, score_sum=24.0),
        },
    )

    plan = build_match_plan(
        manifest,
        rng=np.random.default_rng(7),
        envs=20,
        players=2,
        learner_seat_mode="fixed0",
        anchor_fraction=0.0,
        layout="one_pfsp_two_anchors",
        hard_low=0.20,
        hard_high=0.55,
        hard_bonus=0.15,
        exploration_bonus=0.10,
    )

    assert np.asarray(plan.opponent_slot).shape == (20, MAX_PLAYERS)


def test_match_plan_prioritizes_entries_below_min_games() -> None:
    from orbit_ppo_jax.pfsp import PFSPEntry, PFSPEntryStats, PFSPManifest, build_match_plan

    manifest = PFSPManifest(
        version=1,
        players=2,
        max_policy_slots=4,
        entries=[
            PFSPEntry("covered", "frozen_policy", 0, False, True, "covered", 1),
            PFSPEntry("new", "frozen_policy", 1, False, True, "new", 2),
        ],
        stats={
            "covered": PFSPEntryStats(games=64, score_sum=32.0),
            "new": PFSPEntryStats(games=0, score_sum=0.0),
        },
    )

    plan = build_match_plan(
        manifest,
        rng=np.random.default_rng(7),
        envs=20,
        players=2,
        learner_seat_mode="fixed0",
        anchor_fraction=0.0,
        layout="one_pfsp_two_anchors",
        min_games_per_entry=16,
    )

    assert np.asarray(plan.opponent_slot)[:, 1].tolist() == [1] * 20


def test_match_plan_4p_does_not_fill_all_opponents_with_same_frozen_policy() -> None:
    from orbit_ppo_jax.pfsp import OPP_FROZEN_POLICY, OPP_SIMPLE_HEURISTIC, build_initial_manifest, build_match_plan

    manifest = build_initial_manifest(players=4, max_policy_slots=4, bc_checkpoint="bc.pt")
    plan = build_match_plan(
        manifest,
        rng=np.random.default_rng(0),
        envs=1,
        players=4,
        learner_seat_mode="fixed0",
        anchor_fraction=0.0,
        layout="one_pfsp_two_anchors",
    )
    kinds = np.asarray(plan.opponent_kind)[0, 1:4].tolist()

    assert kinds.count(OPP_FROZEN_POLICY) == 1
    assert OPP_SIMPLE_HEURISTIC in kinds


def test_match_plan_4p_keeps_one_frozen_policy_with_multiple_frozen_entries() -> None:
    from orbit_ppo_jax.pfsp import OPP_FROZEN_POLICY, PFSPEntry, PFSPEntryStats, PFSPManifest, build_match_plan

    manifest = PFSPManifest(
        version=1,
        players=4,
        max_policy_slots=4,
        entries=[
            PFSPEntry("anchor_simple_heuristic_jax", "simple_heuristic_jax", None, True, True, None, 0),
            PFSPEntry("anchor_jax_proxy", "jax_proxy", None, True, True, None, 0),
            PFSPEntry("initial_bc", "frozen_policy", 0, True, True, "bc", 0),
            PFSPEntry("snapshot", "frozen_policy", 1, False, True, "snap", 1),
        ],
        stats={
            "initial_bc": PFSPEntryStats(),
            "snapshot": PFSPEntryStats(),
        },
    )

    plan = build_match_plan(
        manifest,
        rng=np.random.default_rng(0),
        envs=8,
        players=4,
        learner_seat_mode="rotate",
        anchor_fraction=0.0,
        layout="one_pfsp_two_anchors",
    )

    frozen_counts = np.sum(np.asarray(plan.opponent_kind) == OPP_FROZEN_POLICY, axis=1)
    np.testing.assert_array_equal(frozen_counts, np.ones((8,), dtype=np.int64))


def test_pfsp_bank_stacks_checkpoint_params_fixed_shape(tmp_path: Path) -> None:
    from orbit_ppo_jax.bc_policy import load_bc_jax_params
    from orbit_ppo_jax.pfsp import build_initial_manifest
    from orbit_ppo_jax.pfsp_bank import build_pfsp_bank, tree_take
    from tests.test_ppo_jax_readiness import _tiny_bc_checkpoint

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    manifest = build_initial_manifest(players=2, max_policy_slots=4, bc_checkpoint=str(ckpt))
    bc_params, bc_config = load_bc_jax_params(ckpt)
    bank = build_pfsp_bank(manifest, bc_params, bc_config)

    assert bank.active_mask.shape == (4,)
    assert bool(np.asarray(bank.active_mask)[0]) is True
    first = tree_take(bank.bc_params, jnp.asarray(0, dtype=jnp.int32))
    np.testing.assert_allclose(
        np.asarray(first["planet_encoder"]["weight"]),
        np.asarray(bc_params["planet_encoder"]["weight"]),
    )


def test_pfsp_pruning_keeps_anchors_best_latest() -> None:
    from orbit_ppo_jax.pfsp import PFSPEntry, prune_entries

    entries = [
        PFSPEntry("anchor_simple_heuristic_jax", "simple_heuristic_jax", None, True, True, None, 0),
        PFSPEntry("initial_bc", "frozen_policy", 0, True, True, "bc.pt", 0),
        PFSPEntry("update_00010", "frozen_policy", 1, False, True, "u10", 10),
        PFSPEntry("update_00020", "frozen_policy", 2, False, True, "u20", 20),
        PFSPEntry("update_00030", "frozen_policy", 3, False, True, "u30", 30),
    ]
    kept = prune_entries(entries, max_policy_slots=2, best_entry_id="update_00020", latest_entry_id="update_00030")
    kept_ids = {entry.id for entry in kept}

    assert "anchor_simple_heuristic_jax" in kept_ids
    assert "initial_bc" in kept_ids
    assert "update_00020" in kept_ids
    assert "update_00030" in kept_ids


def test_add_snapshot_entry_reuses_oldest_non_anchor_slot_when_bank_full() -> None:
    from orbit_ppo_jax.pfsp import PFSPEntry, PFSPEntryStats, PFSPManifest, add_snapshot_entry

    manifest = PFSPManifest(
        version=1,
        players=2,
        max_policy_slots=3,
        entries=[
            PFSPEntry("initial_bc", "frozen_policy", 0, True, True, "bc", 0),
            PFSPEntry("update_00001", "frozen_policy", 1, False, True, "u1", 1),
            PFSPEntry("update_00002", "frozen_policy", 2, False, True, "u2", 2),
        ],
        stats={
            "initial_bc": PFSPEntryStats(),
            "update_00001": PFSPEntryStats(),
            "update_00002": PFSPEntryStats(),
        },
    )

    updated = add_snapshot_entry(manifest, entry_id="update_00003", path="u3", update_index=3)
    active = {entry.id: entry for entry in updated.entries if entry.active}

    assert active["initial_bc"].slot == 0
    assert "update_00001" not in active
    assert active["update_00003"].slot == 1


def test_add_snapshot_entry_preserves_protected_entries_when_bank_full() -> None:
    from orbit_ppo_jax.pfsp import PFSPEntry, PFSPEntryStats, PFSPManifest, add_snapshot_entry

    manifest = PFSPManifest(
        version=1,
        players=2,
        max_policy_slots=3,
        entries=[
            PFSPEntry("initial_bc", "frozen_policy", 0, True, True, "bc", 0),
            PFSPEntry("best", "frozen_policy", 1, False, True, "best", 1),
            PFSPEntry("old", "frozen_policy", 2, False, True, "old", 2),
        ],
        stats={
            "initial_bc": PFSPEntryStats(),
            "best": PFSPEntryStats(),
            "old": PFSPEntryStats(),
        },
    )

    updated = add_snapshot_entry(
        manifest,
        entry_id="new",
        path="new",
        update_index=3,
        protected_entry_ids={"best"},
    )
    active = {entry.id: entry for entry in updated.entries if entry.active}

    assert active["best"].slot == 1
    assert "old" not in active
    assert active["new"].slot == 2


def test_pfsp_cli_args_are_accepted() -> None:
    from orbit_ppo_jax.train import build_arg_parser, config_from_args

    args = build_arg_parser().parse_args(
        [
            "--bc_checkpoint",
            "bc.pt",
            "--out_dir",
            "out",
            "--opponent",
            "pfsp_jax",
            "--pfsp_enabled",
            "--pfsp_max_policy_slots",
            "8",
            "--pfsp_anchor_fraction",
            "0.5",
            "--pfsp_snapshot_interval_updates",
            "3",
            "--pfsp_warmup_updates",
            "2",
            "--pfsp_learner_seat_mode",
            "rotate",
        ]
    )
    config = config_from_args(args)

    assert config.opponent == "pfsp_jax"
    assert config.pfsp_enabled is True
    assert config.pfsp_max_policy_slots == 8
    assert config.pfsp_anchor_fraction == 0.5


def test_tiny_train_pfsp_writes_manifest_metrics_and_latest_checkpoint(tmp_path: Path) -> None:
    from orbit_ppo_jax.train import main
    from tests.test_ppo_jax_readiness import _tiny_bc_checkpoint

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    out_dir = tmp_path / "ppo_pfsp"
    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--rollout_steps",
            "1",
            "--episode_steps",
            "20",
            "--updates",
            "1",
            "--eval_games",
            "0",
            "--opponent",
            "pfsp_jax",
            "--pfsp_enabled",
            "--pfsp_max_policy_slots",
            "4",
            "--pfsp_warmup_updates",
            "99",
            "--pfsp_matrix_games",
            "0",
        ]
    )

    assert (out_dir / "league" / "manifest.json").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "latest" / "params.npz").exists()


def test_tiny_train_pfsp_updates_initial_bc_stats_on_terminal_episode(tmp_path: Path) -> None:
    from orbit_ppo_jax.pfsp import load_manifest
    from orbit_ppo_jax.train import main
    from tests.test_ppo_jax_readiness import _tiny_bc_checkpoint

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    out_dir = tmp_path / "ppo_pfsp_stats"
    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--rollout_steps",
            "1",
            "--episode_steps",
            "1",
            "--updates",
            "1",
            "--eval_games",
            "0",
            "--opponent",
            "pfsp_jax",
            "--pfsp_enabled",
            "--pfsp_max_policy_slots",
            "4",
            "--pfsp_anchor_fraction",
            "0.0",
            "--pfsp_warmup_updates",
            "99",
            "--pfsp_learner_seat_mode",
            "fixed0",
            "--pfsp_matrix_games",
            "0",
        ]
    )

    manifest = load_manifest(out_dir / "league" / "manifest.json")
    stats = manifest.stats["initial_bc"]
    assert stats.games == 1
    assert stats.last_played_update == 1


def test_tiny_train_pfsp_updates_anchor_stats_on_terminal_episode(tmp_path: Path) -> None:
    from orbit_ppo_jax.pfsp import load_manifest
    from orbit_ppo_jax.train import main
    from tests.test_ppo_jax_readiness import _tiny_bc_checkpoint

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    out_dir = tmp_path / "ppo_pfsp_anchor_stats"
    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--rollout_steps",
            "1",
            "--episode_steps",
            "1",
            "--updates",
            "1",
            "--eval_games",
            "0",
            "--opponent",
            "pfsp_jax",
            "--pfsp_enabled",
            "--pfsp_max_policy_slots",
            "4",
            "--pfsp_anchor_fraction",
            "1.0",
            "--pfsp_warmup_updates",
            "99",
            "--pfsp_learner_seat_mode",
            "fixed0",
            "--pfsp_matrix_games",
            "0",
        ]
    )

    manifest = load_manifest(out_dir / "league" / "manifest.json")
    stats = manifest.stats["anchor_simple_heuristic_jax"]
    assert stats.games == 1
    assert stats.last_played_update == 1


def test_tiny_train_pfsp_promotes_snapshot_when_gate_passes(tmp_path: Path) -> None:
    from orbit_ppo_jax.pfsp import load_manifest
    from orbit_ppo_jax.train import main
    from tests.test_ppo_jax_readiness import _tiny_bc_checkpoint

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    out_dir = tmp_path / "ppo_pfsp_promote"
    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--rollout_steps",
            "1",
            "--episode_steps",
            "1",
            "--updates",
            "1",
            "--eval_games",
            "0",
            "--opponent",
            "pfsp_jax",
            "--pfsp_enabled",
            "--pfsp_max_policy_slots",
            "4",
            "--pfsp_anchor_fraction",
            "0.0",
            "--pfsp_warmup_updates",
            "1",
            "--pfsp_snapshot_interval_updates",
            "1",
            "--pfsp_matrix_games",
            "0",
        ]
    )

    snapshot_dir = out_dir / "league" / "snapshots" / "update_00001"
    assert (snapshot_dir / "params.npz").exists()
    manifest = load_manifest(out_dir / "league" / "manifest.json")
    promoted = [entry for entry in manifest.entries if entry.id == "update_00001"]
    assert len(promoted) == 1
    assert promoted[0].kind == "frozen_policy"
    assert promoted[0].slot == 1


def test_tiny_pfsp_eval_matrix_writes_outputs(tmp_path: Path) -> None:
    from orbit_ppo_jax.train import main
    from tests.test_ppo_jax_readiness import _tiny_bc_checkpoint

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    out_dir = tmp_path / "ppo_pfsp_matrix"
    main(
        [
            "--bc_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--players",
            "2",
            "--envs",
            "1",
            "--rollout_steps",
            "1",
            "--episode_steps",
            "1",
            "--updates",
            "1",
            "--eval_games",
            "0",
            "--opponent",
            "pfsp_jax",
            "--pfsp_enabled",
            "--pfsp_max_policy_slots",
            "4",
            "--pfsp_anchor_fraction",
            "0.0",
            "--pfsp_warmup_updates",
            "99",
            "--pfsp_matrix_games",
            "1",
            "--pfsp_eval_interval_updates",
            "1",
        ]
    )

    assert (out_dir / "league" / "eval_matrix.json").exists()
    assert (out_dir / "league" / "eval_matrix.md").exists()
    assert (out_dir / "best" / "params.npz").exists()


def test_pfsp_eval_matrix_runs_jax_games_without_manifest_stats(tmp_path: Path) -> None:
    from orbit_ppo_jax.bc_policy import init_value_head, load_bc_jax_params
    from orbit_ppo_jax.pfsp import build_initial_manifest
    from orbit_ppo_jax.pfsp_bank import build_pfsp_bank
    from orbit_ppo_jax.pfsp_eval import evaluate_matrix
    from orbit_ppo_jax.train import JaxPPOConfig
    from tests.test_ppo_jax_readiness import _tiny_bc_checkpoint

    ckpt = _tiny_bc_checkpoint(tmp_path / "bc")
    bc_params, bc_config = load_bc_jax_params(ckpt)
    params = {"bc": bc_params, "value": init_value_head(jax.random.PRNGKey(0), int(bc_config["hidden_size"]))}
    manifest = build_initial_manifest(players=2, max_policy_slots=4, bc_checkpoint=str(ckpt))
    bank = build_pfsp_bank(manifest, bc_params, bc_config)
    config = JaxPPOConfig(
        bc_checkpoint=str(ckpt),
        out_dir=str(tmp_path / "out"),
        players=2,
        episode_steps=1,
        source_cap=4,
        pfsp_matrix_games=2,
    )

    summary = evaluate_matrix(
        params=params,
        bc_config=bc_config,
        bank=bank,
        manifest=manifest,
        config=config,
        key=jax.random.PRNGKey(1),
        out_dir=tmp_path / "out",
    )

    rows = {row["entry_id"]: row for row in summary["rows"]}
    assert rows["initial_bc"]["games"] == 2
    assert "average_episode_step" in rows["initial_bc"]
