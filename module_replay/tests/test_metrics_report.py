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


def test_validity_missing_key_fails_closed(tmp_path):
    """WR-03: a validity threshold whose faithfulness key is ABSENT must fail
    closed (exit 2 / INVALID), never silently pass. Today it wrongly returns 0."""
    th = {"replay_jitter_ms": ThresholdSpec(max=50.0, tier="validity")}
    rc = generate_report(
        "perception", "t", [], tmp_path, th,
        faithfulness={"max_gap_ms": 100.0, "breach_count": 0, "drop_rate": 0.0},
    )
    assert rc == 2
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["verdict"] == "INVALID" and doc["pass"] is False


def test_validity_missing_key_is_visible(tmp_path):
    """WR-03: the unevaluable validity threshold is surfaced VISIBLY (not silent) —
    the faithfulness block reads 'fail' AND a metrics row records it with passed:None
    and a note naming the missing faithfulness field."""
    th = {"replay_jitter_ms": ThresholdSpec(max=50.0, tier="validity")}
    generate_report(
        "perception", "t", [], tmp_path, th,
        faithfulness={"max_gap_ms": 100.0, "breach_count": 0, "drop_rate": 0.0},
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["replay_faithfulness"]["verdict"] == "fail"
    rows = [m for m in doc["metrics"] if m["name"] == "replay_jitter_ms"]
    assert rows and rows[0]["passed"] is None
    assert "jitter_ms" in rows[0]["note"]


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


def test_breach_count_gate_invalidates(tmp_path):
    """01-12 residual: a per-topic stall (breach_count>=1) on ANY topic must invalidate via
    the replay_breach_count validity gate (max 0) — the rate-aware signal 01-12 built is now
    a gated field, even when the headline max_gap_ms is clean."""
    th = {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "replay_breach_count": ThresholdSpec(max=0, tier="validity"),
    }
    rc = generate_report(
        "perception", "t", [], tmp_path, th,
        faithfulness={"max_gap_ms": 100.0, "breach_count": 1, "drop_rate": 0.0},
    )
    assert rc == 2
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["verdict"] == "INVALID"


def test_report_html_written(tmp_path):
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tier="quality")}
    generate_report("perception", "t", [_mr("latency_p95_ms", 10.0)], tmp_path, th)
    assert (tmp_path / "report.html").read_text().strip() != ""


# ── 01-16 Task 1: rich report data model (summary block) + run_artifacts ──────


def _pass_mr(name, val):
    return MetricResult(
        name=name, module="perception", value={name: val}, passed=True, is_regression=False
    )


def test_generate_report_accepts_run_artifacts_optional(tmp_path):
    """A1/A3: generate_report gains an optional run_artifacts arg WITHOUT changing
    the B9 exit code, and the existing call shape (no run_artifacts) still works."""
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tolerance_band=5.0, tier="quality")}
    artifacts = {"bag": "/x/replay_output", "logs": "/x/logs", "report": "report.html"}
    rc_with = generate_report(
        "perception", "t", [_pass_mr("latency_p95_ms", 40.0)], tmp_path / "a", th,
        run_artifacts=artifacts,
    )
    rc_without = generate_report(
        "perception", "t", [_pass_mr("latency_p95_ms", 40.0)], tmp_path / "b", th,
    )
    assert rc_with == rc_without == 0
    assert (tmp_path / "a" / "report.html").read_text().strip() != ""
    assert (tmp_path / "b" / "report.html").read_text().strip() != ""


def test_metrics_json_schema_unchanged(tmp_path):
    """The CONTRACT: metrics.json keeps EXACTLY its existing top-level keys plus
    the new additive 'summary'/'run_artifacts'; pre-existing key TYPES are intact
    (pass:bool, verdict in {PASS,FAIL,INVALID}, metrics:list of row dicts)."""
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tolerance_band=5.0, tier="quality")}
    generate_report("perception", "t", [_pass_mr("latency_p95_ms", 40.0)], tmp_path, th)
    doc = json.loads((tmp_path / "metrics.json").read_text())
    # Every pre-existing contract key still present:
    for k in ("module", "run_id", "pass", "replay_faithfulness", "metrics", "verdict", "details"):
        assert k in doc, f"contract key '{k}' missing"
    # Only the additive keys joined the top level:
    assert set(doc) - {
        "module", "run_id", "pass", "replay_faithfulness", "metrics", "verdict", "details",
    } <= {"summary", "run_artifacts", "overlap_pairs", "plots"}
    # Types intact — the CI gate reads doc["pass"]/doc["verdict"]/the rows:
    assert isinstance(doc["pass"], bool) and doc["pass"] is True
    assert doc["verdict"] in {"PASS", "FAIL", "INVALID"}
    assert isinstance(doc["metrics"], list)
    row = next(m for m in doc["metrics"] if m["name"] == "latency_p95_ms")
    assert row["value"] == 40.0 and row["passed"] is True and row["tier"] == "quality"


def test_summary_block_built_from_rows(tmp_path):
    """doc['summary'] is a list of cards each with label/value/status in
    {PASS,BREACH,FAIL,NONE}, derived from the row's passed + tier; a validity-tier
    breach yields BREACH, a failed quality row yields FAIL, passed:None -> NONE."""
    th = {
        "latency_p95_ms": ThresholdSpec(max=50.0, tier="quality"),         # will FAIL
        "depth_validity": ThresholdSpec(min=0.5, tier="quality"),          # will PASS
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "replay_breach_count": ThresholdSpec(max=0, tier="validity"),      # will BREACH
        "segmentation_coverage": ThresholdSpec(tier="quality"),            # no bound -> NONE
    }
    results = [
        _mr("latency_p95_ms", 60.0),       # passed False, quality -> FAIL
        _pass_mr("depth_validity", 0.9),   # passed True -> PASS
        MetricResult(name="segmentation_coverage", module="perception",
                     value={"segmentation_coverage": 0.4}, passed=True, is_regression=False),
    ]
    generate_report(
        "perception", "t", results, tmp_path, th,
        faithfulness={"max_gap_ms": 100.0, "breach_count": 1, "drop_rate": 0.0},
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    summary = doc["summary"]
    assert isinstance(summary, list) and summary
    for card in summary:
        assert {"label", "value", "status"} <= set(card)
        assert card["status"] in {"PASS", "BREACH", "FAIL", "NONE"}
    statuses = {c["label"]: c["status"] for c in summary}
    # A failed quality row -> FAIL; a passed row -> PASS.
    assert any(s == "FAIL" for s in statuses.values())
    assert any(s == "PASS" for s in statuses.values())
    # The breach_count validity card (max 0, value 1) reads BREACH.
    assert any(c["status"] == "BREACH" for c in summary)


def test_report_html_renders_summary_and_debug(tmp_path):
    """The rendered report.html carries the summary card markup (metric-grid) AND a
    Debug section containing each run_artifacts value; autoescape stays ON so a
    '<script>'-bearing metric name is escaped (T-07-01 preserved)."""
    th = {"<script>alert(1)</script>": ThresholdSpec(max=50.0, tier="quality")}
    artifacts = {
        "bag": "/runs/42/replay_output",
        "logs": "/runs/42/logs",
        "report": "report.html",
    }
    generate_report(
        "perception", "t",
        [_mr("<script>alert(1)</script>", 10.0)],
        tmp_path, th, run_artifacts=artifacts,
    )
    html = (tmp_path / "report.html").read_text()
    # Summary card grid present.
    assert "metric-grid" in html
    # Debug section renders each run_artifacts value.
    assert "/runs/42/replay_output" in html
    assert "/runs/42/logs" in html
    # Autoescape ON — the raw <script> tag never appears unescaped.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# ── 01-16 Task 3: wire plots into generator + render in template ──────────────


def _overlap_mr(pairs):
    return MetricResult(
        name="cross_camera_overlap_iou",
        module="perception",
        value={
            "cross_camera_overlap_iou": 0.78,
            "num_pairs": len(pairs),
            "feature_matcher": "akaze",
            "pairs": pairs,
        },
        passed=True,
        is_regression=False,
    )


def test_generate_report_invokes_plots_when_reader_given(tmp_path, mocker):
    """generate_report given an optional reader calls generate_report_plots and
    references the returned plot paths as <img src> in report.html."""
    fake = {
        "latency": tmp_path / "latency_time_series.png",
        "pipeline": tmp_path / "pipeline_breakdown.png",
    }
    spy = mocker.patch(
        "replay.metrics.report.generator.generate_report_plots", return_value=fake
    )
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tolerance_band=5.0, tier="quality")}
    rc = generate_report(
        "perception", "t", [_pass_mr("latency_p95_ms", 40.0)], tmp_path, th,
        reader=object(),  # any non-None reader triggers the plot path
    )
    assert rc == 0
    spy.assert_called_once()
    html = (tmp_path / "report.html").read_text()
    assert "latency_time_series.png" in html
    assert "pipeline_breakdown.png" in html


def test_generate_report_no_reader_no_plots_still_renders(tmp_path, mocker):
    """WITHOUT a reader, generate_report renders the summary + tables, emits NO
    plot imgs, never calls the plot generator, and returns the same exit code."""
    spy = mocker.patch("replay.metrics.report.generator.generate_report_plots")
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tolerance_band=5.0, tier="quality")}
    rc = generate_report("perception", "t", [_pass_mr("latency_p95_ms", 40.0)], tmp_path, th)
    assert rc == 0
    spy.assert_not_called()
    html = (tmp_path / "report.html").read_text()
    assert "metric-grid" in html
    assert ".png" not in html  # no plot images when no reader


def test_plot_failure_does_not_change_verdict(tmp_path, mocker):
    """A plot generator raising must NOT crash a run or flip the verdict —
    report.html is still written and the B9 exit code is unchanged (graceful degrade)."""
    mocker.patch(
        "replay.metrics.report.generator.generate_report_plots",
        side_effect=RuntimeError("matplotlib blew up"),
    )
    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tolerance_band=5.0, tier="quality")}
    rc = generate_report(
        "perception", "t", [_pass_mr("latency_p95_ms", 40.0)], tmp_path, th,
        reader=object(),
    )
    assert rc == 0  # verdict unchanged despite the plot exception
    assert (tmp_path / "report.html").read_text().strip() != ""
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["pass"] is True and doc["verdict"] == "PASS"


def test_template_renders_cross_camera_pair_table(tmp_path):
    """A cross_camera_overlap_iou row's value['pairs'] (LIVE cam0_cam1 key shape)
    renders a per-pair table — one row per pair, with the pair key + its
    mean/min agreement values present in report.html."""
    pairs = {
        "cam0_cam1": {
            "description": "Front-Top<->Front-Bottom", "num_frames": 827,
            "calibration_inliers": 136, "mean_agreement": 0.775,
            "min_agreement": 0.565, "overlap_pixels": 1234,
        },
        "cam0_cam5": {
            "description": "Front-Top<->Left", "num_frames": 640,
            "calibration_inliers": 98, "mean_agreement": 0.812,
            "min_agreement": 0.610, "overlap_pixels": 980,
        },
    }
    th = {"cross_camera_overlap_iou": ThresholdSpec(min=0.75, tier="quality")}
    generate_report("perception", "t", [_overlap_mr(pairs)], tmp_path, th)
    html = (tmp_path / "report.html").read_text()
    # Both pair keys appear (one row each).
    assert "cam0_cam1" in html
    assert "cam0_cam5" in html
    # Their agreement values are rendered.
    assert "0.775" in html and "0.565" in html
    assert "0.812" in html and "0.61" in html  # 0.610 may render as 0.61
