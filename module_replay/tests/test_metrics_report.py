"""Tests for the criteria evaluator + report generator (MTRC-03 / MTRC-05).

Covers the AND-gate quality verdict, the validity-tier short-circuit (B9
INVALID RUN distinct from FAIL), tolerance-band evaluation, the no-silent-pass
rule for uncomputed thresholds, and the metrics.json / report.html artifacts.
"""
import json

from replay.metrics.base import MetricResult
from replay.module_config import ThresholdSpec
from replay.metrics.report.generator import evaluate_threshold, generate_report


def _mr(name, val):
    return MetricResult(
        name=name, module="perception", value={name: val}, passed=False, is_regression=False
    )


def test_and_gate_pass_fail(tmp_path):
    """MTRC-03: all quality metrics must pass for overall pass."""
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tolerance_band=5.0, tier="quality")}
    rc = generate_report("perception", "t", [_mr("latency_p95_ms", 40.0)], tmp_path, th)
    assert rc == 0
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["pass"] is True


def test_exit_code_nonzero_on_breach(tmp_path):
    """MTRC-05: a breach returns exit 1 and pass=false."""
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tolerance_band=1.0, tier="quality")}
    rc = generate_report("perception", "t", [_mr("latency_p95_ms", 60.0)], tmp_path, th)
    assert rc == 1
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["pass"] is False


def test_tolerance_band_applied(tmp_path):
    th = ThresholdSpec(max=50.0, tolerance_band=5.0, tier="quality")
    assert evaluate_threshold(54.0, th) is True   # within max+tolerance
    assert evaluate_threshold(56.0, th) is False


def test_validity_breach_forces_fail(tmp_path):
    th = {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "replay_drop_rate": ThresholdSpec(max=0.02, tier="validity"),
        "latency_p95_ms": ThresholdSpec(max=50.0, tier="quality"),
    }
    rc = generate_report(
        "perception", "t", [_mr("latency_p95_ms", 10.0)], tmp_path, th,
        faithfulness={"max_gap_ms": 500.0, "breach_count": 3, "drop_rate": 0.1},
    )
    assert rc == 2   # quality fine, but validity breached -> INVALID RUN (B9), not FAIL
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["verdict"] == "INVALID" and doc["pass"] is False


def test_uncomputed_threshold_skipped(tmp_path):
    """A configured threshold with no computed metric is a visible 'skipped' row, never a failure."""
    th = {
        "latency_p95_ms": ThresholdSpec(max=50.0, tier="quality"),
        "mask_iou_vs_golden": ThresholdSpec(min=0.98, tier="quality"),
    }
    rc = generate_report("perception", "t", [_mr("latency_p95_ms", 10.0)], tmp_path, th)
    assert rc == 0
    doc = json.loads((tmp_path / "metrics.json").read_text())
    skipped = [m for m in doc["metrics"] if m["name"] == "mask_iou_vs_golden"]
    assert skipped and skipped[0]["passed"] is None and "skipped" in skipped[0]["note"]


def test_report_html_written(tmp_path):
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tier="quality")}
    generate_report("perception", "t", [_mr("latency_p95_ms", 10.0)], tmp_path, th)
    assert (tmp_path / "report.html").read_text().strip() != ""
