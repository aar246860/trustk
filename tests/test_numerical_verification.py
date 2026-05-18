import json

from trustk.experiments.run_numerical_verification import run_numerical_verification


def test_numerical_verification_reports_grid_time_and_boundary_checks(tmp_path):
    report_path = tmp_path / "numerical_verification.json"
    report = run_numerical_verification(report_path=report_path)
    loaded = json.loads(report_path.read_text(encoding="utf-8"))

    assert loaded["checks"]["baseline_theis_error"]["pass"]
    assert loaded["checks"]["grid_refinement"]["pass"]
    assert loaded["checks"]["time_step_refinement"]["pass"]
    assert loaded["checks"]["outer_boundary_sensitivity"]["pass"]
    assert report["baseline"]["relative_l2_error"] < 0.06
    assert len(loaded["grid_refinement"]) == 3
    assert len(loaded["time_step_refinement"]) == 3
