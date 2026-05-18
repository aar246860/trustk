import json
from pathlib import Path

from trustk.experiments.run_benchmark_theis import run_theis_benchmark


def test_theis_benchmark_writes_report_and_figure(tmp_path):
    report_path = tmp_path / "theis_benchmark.json"
    figure_prefix = tmp_path / "fig_theis_benchmark"

    report = run_theis_benchmark(
        report_path=report_path,
        figure_prefix=figure_prefix,
        n_r=120,
        n_theta=12,
    )

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["relative_l2_error"] < 0.22
    assert loaded["max_absolute_error_m"] < 0.08
    assert report["relative_l2_error"] == loaded["relative_l2_error"]
    assert figure_prefix.with_suffix(".pdf").exists()
    assert figure_prefix.with_suffix(".png").exists()
