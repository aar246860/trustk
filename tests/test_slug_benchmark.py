import json

from trustk.experiments.run_benchmark_slug import run_slug_benchmark


def test_slug_benchmark_writes_report_and_figure(tmp_path):
    report_path = tmp_path / "slug_benchmark.json"
    figure_prefix = tmp_path / "fig_slug_benchmark"

    report = run_slug_benchmark(report_path=report_path, figure_prefix=figure_prefix)
    loaded = json.loads(report_path.read_text(encoding="utf-8"))

    assert loaded["relative_l2_error"] < 0.10
    assert loaded["monotonic_recovery_pass"]
    assert report["relative_l2_error"] == loaded["relative_l2_error"]
    assert figure_prefix.with_suffix(".pdf").exists()
    assert figure_prefix.with_suffix(".png").exists()
