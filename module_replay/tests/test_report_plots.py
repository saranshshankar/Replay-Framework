"""Tests for the offline matplotlib report plots (01-16 Task 2 / A2).

The plots layer is PRESENTATION over OUR metrics.json / MetricResult data model
and OUR read-once BagReader — it must stay offline pure-Python (matplotlib Agg,
NO rclpy, NO torch, NO GPU) and be FAIL-SAFE: a single plot raising must never
propagate (a plot failure can never crash a run or flip a verdict).
"""
import sys

import pytest

from replay.metrics.base import MetricResult
from replay.metrics.bag_reader import BagReader


# Topics the synthetic_bag fixture actually writes (rgb8, no depth_raw_sim, no diagnostics).
_SYNTH_TOPICS = [
    "/perception_node/camera_0/image_raw",
    "/perception_node/camera_0/image_raw_sim",
]


def _pipeline_mr():
    """A pipeline_throughput_hz MetricResult shaped like pipeline.py emits."""
    return MetricResult(
        name="pipeline_throughput_hz",
        module="perception",
        value={
            "pipeline_throughput_hz": 10.0,
            "mean_hz": 10.0,
            "per_topic": {
                "/perception_node/camera_0/image_raw_sim": {
                    "num_messages": 10, "mean_hz": 10.0, "mean_interval_ms": 100.0,
                },
                "/perception_node/camera_1/semantic_raw_sim": {
                    "num_messages": 10, "mean_hz": 9.5, "mean_interval_ms": 105.3,
                },
            },
        },
        passed=True,
        is_regression=False,
    )


def _latency_mr(p95=45.0):
    """A latency_p95_ms MetricResult with AGGREGATE scalars only (latency.py shape)."""
    return MetricResult(
        name="latency_p95_ms",
        module="perception",
        value={
            "latency_p95_ms": p95,
            "p50_ms": p95 - 10.0,
            "p95_ms": p95,
            "p99_ms": p95 + 5.0,
            "mean_ms": p95 - 8.0,
            "max_ms": p95 + 8.0,
            "num_windows": 12,
            "skipped": False,
        },
        passed=True,
        is_regression=False,
    )


def _skipped_latency_mr():
    """The no-diagnostics path: skipped=True, latency_p95_ms None."""
    return MetricResult(
        name="latency_p95_ms",
        module="perception",
        value={
            "latency_p95_ms": None,
            "skipped": True,
            "reason": "no diagnostics topic configured/present",
            "num_windows": 0,
        },
        passed=None,
        is_regression=False,
    )


def test_generate_report_plots_writes_pngs(synthetic_bag, tmp_path):
    """For the synthetic bag, generate_report_plots returns a name->Path dict and
    writes at least the pipeline + latency PNGs (non-empty); depth topics being
    empty must NOT raise."""
    from replay.metrics.report.plots import generate_report_plots

    reader = BagReader(synthetic_bag, _SYNTH_TOPICS)
    out = tmp_path / "plots_out"
    out.mkdir()
    plots = generate_report_plots([_pipeline_mr(), _latency_mr()], reader, out)

    assert isinstance(plots, dict)
    assert "pipeline" in plots and "latency" in plots
    for name in ("pipeline", "latency"):
        p = plots[name]
        assert p.exists(), f"{name} png not written"
        assert p.stat().st_size > 0, f"{name} png is empty"


def test_skipped_latency_row_renders_no_plot(synthetic_bag, tmp_path):
    """When the latency row is skipped (latency_p95_ms None), no latency png is
    emitted (no misleading empty/zero plot) and the function does not raise."""
    from replay.metrics.report.plots import generate_report_plots

    reader = BagReader(synthetic_bag, _SYNTH_TOPICS)
    out = tmp_path / "plots_out"
    out.mkdir()
    plots = generate_report_plots([_pipeline_mr(), _skipped_latency_mr()], reader, out)

    assert "latency" not in plots
    assert not (out / "latency_time_series.png").exists()


def test_plots_use_agg_no_gui():
    """Importing plots.py forces the Agg backend (no display required)."""
    import matplotlib

    import replay.metrics.report.plots  # noqa: F401  (import sets Agg)
    assert matplotlib.get_backend().lower() == "agg"


def test_plots_no_rclpy_no_torch():
    """Importing the plots module must not drag in rclpy or torch (offline pure-Python)."""
    # Drop any prior import so this assertion is about plots.py's own import graph.
    sys.modules.pop("replay.metrics.report.plots", None)
    import replay.metrics.report.plots  # noqa: F401
    assert "rclpy" not in sys.modules
    assert "torch" not in sys.modules


def test_generate_report_plots_degrades_on_exception(synthetic_bag, tmp_path):
    """A malformed metric value that would make one plot raise must NOT propagate —
    generate_report_plots still returns a dict (possibly missing that plot)."""
    from replay.metrics.report.plots import generate_report_plots

    reader = BagReader(synthetic_bag, _SYNTH_TOPICS)
    out = tmp_path / "plots_out"
    out.mkdir()
    # A pipeline MetricResult whose per_topic is the WRONG type (a string), so the
    # pipeline plot generator raises internally — it must be swallowed.
    bad_pipeline = MetricResult(
        name="pipeline_throughput_hz", module="perception",
        value={"pipeline_throughput_hz": 10.0, "per_topic": "not-a-dict"},
        passed=True, is_regression=False,
    )
    plots = generate_report_plots([bad_pipeline, _latency_mr()], reader, out)
    assert isinstance(plots, dict)
    # The good latency plot still rendered; the bad pipeline plot was omitted.
    assert "pipeline" not in plots
    assert "latency" in plots
