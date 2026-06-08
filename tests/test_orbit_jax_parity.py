from __future__ import annotations


def test_no_action_parity_case_passes_and_writes_report(tmp_path) -> None:
    from orbit_jax_env.parity.compare_official import run_parity_case, run_parity_report

    result = run_parity_case("case_001_no_actions", seed=123, players=2, steps=3)
    assert result["implemented"] is True
    assert result["passed"] is True
    assert result["max_position_abs_error"] <= 1.0e-4
    assert result["owner_mismatches"] == 0
    assert result["ship_mismatches"] == 0

    report = run_parity_report(tmp_path / "parity_report.json", cases=["case_001_no_actions"])
    assert report["status"] == "pass"
    assert report["cases"]["case_001_no_actions"]["passed"] is True


def test_simple_capture_parity_case_passes() -> None:
    from orbit_jax_env.parity.compare_official import run_parity_case

    result = run_parity_case("case_002_simple_capture_static", seed=104, players=2, steps=12)
    assert result["implemented"] is True
    assert result["passed"] is True
    assert result["owner_mismatches"] == 0
    assert result["ship_mismatches"] == 0
    assert result["fleet_count_mismatch"] == 0


def test_sun_bounds_and_rotation_parity_cases_pass() -> None:
    from orbit_jax_env.parity.compare_official import run_parity_case

    for case_name in (
        "case_003_sun_collision",
        "case_004_bounds_collision",
        "case_006_planet_rotation",
    ):
        result = run_parity_case(case_name, players=2)
        assert result["implemented"] is True
        assert result["passed"] is True
        assert result["owner_mismatches"] == 0
        assert result["ship_mismatches"] == 0
        assert result["fleet_count_mismatch"] == 0
        assert result["fleet_mismatches"] == 0


def test_two_fleets_combat_parity_case_passes() -> None:
    from orbit_jax_env.parity.compare_official import run_parity_case

    result = run_parity_case("case_005_two_fleets_combat", players=2)
    assert result["implemented"] is True
    assert result["passed"] is True
    assert result["owner_mismatches"] == 0
    assert result["ship_mismatches"] == 0
    assert result["fleet_count_mismatch"] == 0
    assert result["fleet_mismatches"] == 0


def test_moving_planet_and_random_scripted_parity_cases_pass() -> None:
    from orbit_jax_env.parity.compare_official import run_parity_case

    for case_name, players in (
        ("case_007_moving_planet_collision", 2),
        ("case_008_random_scripted_2p_50_steps", 2),
        ("case_009_random_scripted_4p_50_steps", 4),
    ):
        result = run_parity_case(case_name, players=players)
        assert result["implemented"] is True
        assert result["passed"] is True
        assert result["owner_mismatches"] == 0
        assert result["ship_mismatches"] == 0
        assert result["fleet_count_mismatch"] == 0
        assert result["fleet_mismatches"] == 0


def test_full_parity_report_uses_four_players_for_4p_case(tmp_path) -> None:
    from orbit_jax_env.parity.compare_official import run_parity_report

    report = run_parity_report(tmp_path / "parity_report.json", cases=["case_009_random_scripted_4p_50_steps"])
    case = report["cases"]["case_009_random_scripted_4p_50_steps"]
    assert case["implemented"] is True
    assert case["passed"] is True
    assert case["players"] == 4
