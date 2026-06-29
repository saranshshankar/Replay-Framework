"""Tests for the additive incident_verdict in generate_report (01.1-04, Task 2).

Design requirements:
- generate_report(..., incident_spec=None) is the golden path: doc has no incident_verdict
  (or it's None) AND doc["pass"]/exit-code are byte-for-byte unchanged.
- incident_spec given + validity_pass + reproduced=False + no new_signature + quality_pass
  -> incident_verdict["verdict"] == "fixed".
- incident_spec given + reproduced=True -> verdict == "reproduced" (NOT fixed).
- incident_spec given + not validity_pass (INVALID run) -> verdict FORCED "inconclusive",
  NEVER "fixed" (T-0104-02: never-fixed-on-INVALID).
- incident_spec given + other_conditions with one newly-breaching member (incident's own
  condition NOT breaching) -> verdict == "new_signature" (FR-6(b)).
- doc["pass"] and the 0/1/2 exit code are UNCHANGED by incident_spec (T-0104-03).
"""
from __future__ import annotations

import json

import pytest

from replay.metrics.base import MetricResult
from replay.module_config import ThresholdSpec
from replay.metrics.report.generator import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mr(name, val):
    return MetricResult(
        name=name, module="perception", value={name: val}, passed=False, is_regression=False
    )


def _seg_result(coverage: float):
    return MetricResult(
        name="segmentation_coverage",
        module="perception",
        value={"segmentation_coverage": coverage, "temporal_consistency_mean": 0.9,
               "mean_class_coverage": 0.5},
        passed=True,
        is_regression=False,
    )


def _quality_thresholds():
    return {
        "segmentation_coverage": ThresholdSpec(min=0.05, tier="quality"),
    }


def _validity_thresholds():
    return {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "segmentation_coverage": ThresholdSpec(min=0.05, tier="quality"),
    }


def _good_faithfulness():
    return {"max_gap_ms": 50.0, "drop_rate": 0.001, "breach_count": 0}


def _breached_faithfulness():
    return {"max_gap_ms": 500.0, "drop_rate": 0.5, "breach_count": 5}


def _seg_coverage_condition(threshold=0.05):
    """A condition that breaches when segmentation_coverage < threshold."""
    return {
        "metric": "segmentation_coverage",
        "field": "segmentation_coverage",
        "op": "lt",
        "threshold": threshold,
    }


def _lat_condition(threshold=200.0):
    """A condition that breaches when latency_p95_ms > threshold."""
    return {
        "metric": "latency_p95_ms",
        "field": "latency_p95_ms",
        "op": "gt",
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Test 1: incident_spec=None (golden path) — no incident_verdict, pass/rc unchanged
# ---------------------------------------------------------------------------

def test_no_incident_spec_golden_path_pass(tmp_path):
    """incident_spec=None: doc has no incident_verdict or it's None; pass=True; rc=0."""
    th = _quality_thresholds()
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=None,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 0
    assert doc["pass"] is True
    # incident_verdict must be absent or explicitly None
    assert doc.get("incident_verdict") is None


def test_no_incident_spec_golden_path_fail(tmp_path):
    """incident_spec=None on a FAIL run: doc has no incident_verdict; pass=False; rc=1."""
    th = {"segmentation_coverage": ThresholdSpec(min=0.5, tier="quality")}
    rc = generate_report(
        "perception", "t", [_seg_result(0.1)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=None,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 1
    assert doc["pass"] is False
    assert doc.get("incident_verdict") is None


def test_no_incident_spec_doc_pass_present(tmp_path):
    """doc['pass'] key is always present regardless of incident_spec (T-0104-03)."""
    th = _quality_thresholds()
    generate_report("perception", "t", [_seg_result(0.3)], tmp_path, th)
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert "pass" in doc


# ---------------------------------------------------------------------------
# Test 2: incident verdict = "fixed" (VALID + no-repro + no-new-sig + quality pass)
# ---------------------------------------------------------------------------

def test_incident_verdict_fixed_on_valid_pass_no_repro(tmp_path):
    """VALID run + condition no longer breaching + quality_pass -> verdict='fixed'."""
    th = _validity_thresholds()
    incident_spec = {
        "verifier_type": "metric_condition",
        "incident_id": "INC-001",
        "condition": _seg_coverage_condition(threshold=0.05),  # breaches if coverage < 0.05
    }
    # coverage=0.3 -> condition does NOT breach -> not reproduced
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 0
    assert doc["pass"] is True
    assert doc["incident_verdict"] is not None
    assert doc["incident_verdict"]["verdict"] == "fixed"
    assert doc["incident_verdict"]["incident_id"] == "INC-001"


def test_incident_verdict_fixed_pass_unchanged(tmp_path):
    """When verdict='fixed', doc['pass'] is still True (T-0104-03: pass is untouched)."""
    th = _validity_thresholds()
    incident_spec = {
        "verifier_type": "metric_condition",
        "condition": _seg_coverage_condition(threshold=0.05),
    }
    generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["pass"] is True
    assert doc["verdict"] == "PASS"
    # Confirm exit code is still 0, not affected by incident_verdict
    # (we test rc above; here confirm the doc verdict is correct)
    assert doc["incident_verdict"]["verdict"] == "fixed"


# ---------------------------------------------------------------------------
# Test 3: incident verdict = "reproduced" (condition still breaches)
# ---------------------------------------------------------------------------

def test_incident_verdict_reproduced_when_condition_breaches(tmp_path):
    """Condition still breaches -> verdict='reproduced' (incident NOT fixed)."""
    th = _validity_thresholds()
    incident_spec = {
        "verifier_type": "metric_condition",
        "incident_id": "INC-002",
        "condition": _seg_coverage_condition(threshold=0.05),  # breaches if coverage < 0.05
    }
    # coverage=0.0 -> condition breaches -> reproduced=True
    generate_report(
        "perception", "t", [_seg_result(0.0)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["incident_verdict"]["verdict"] == "reproduced"
    assert doc["incident_verdict"]["reproduced"] is True


# ---------------------------------------------------------------------------
# Test 4: INVALID run forces inconclusive (never-fixed-on-INVALID, T-0104-02)
# ---------------------------------------------------------------------------

def test_incident_verdict_inconclusive_on_invalid_run(tmp_path):
    """Not validity_pass -> incident_verdict forced 'inconclusive', NEVER 'fixed'."""
    # validity threshold present; faithfulness max_gap_ms breaches validity
    th = {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "segmentation_coverage": ThresholdSpec(min=0.05, tier="quality"),
    }
    incident_spec = {
        "verifier_type": "metric_condition",
        "condition": _seg_coverage_condition(threshold=0.05),
    }
    # coverage=0.3 (condition would NOT breach), but faithfulness is invalid
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_breached_faithfulness(),  # max_gap_ms=500 > threshold 200 -> INVALID
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 2   # INVALID RUN (validity breached)
    assert doc["pass"] is False
    assert doc["verdict"] == "INVALID"
    assert doc["incident_verdict"]["verdict"] == "inconclusive"
    assert doc["incident_verdict"]["verdict"] != "fixed"


def test_incident_verdict_inconclusive_via_missing_faithfulness_key(tmp_path):
    """WR-03 path: a validity threshold with no matching faithfulness field forces INVALID;
    incident_verdict is 'inconclusive', never 'fixed' (reuse validity short-circuit)."""
    th = {"replay_jitter_ms": ThresholdSpec(max=50.0, tier="validity")}
    incident_spec = {
        "verifier_type": "metric_condition",
        "condition": _seg_coverage_condition(threshold=0.05),
    }
    # faithfulness has no 'jitter_ms' key -> WR-03 fails closed -> validity_pass=False
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness={"max_gap_ms": 100.0, "breach_count": 0, "drop_rate": 0.0},
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 2
    assert doc["incident_verdict"]["verdict"] == "inconclusive"
    assert doc["incident_verdict"]["verdict"] != "fixed"


# ---------------------------------------------------------------------------
# Test 5: new_signature verdict (FR-6(b): a DIFFERENT registered condition breaches)
# ---------------------------------------------------------------------------

def test_incident_verdict_new_signature_when_other_condition_breaches(tmp_path):
    """other_conditions list contains one newly-breaching member; incident's own
    condition does NOT breach -> verdict='new_signature' (FR-6(b))."""
    th = _validity_thresholds()

    # Latency result that would breach a "latency > 200ms" condition
    lat_result = MetricResult(
        name="latency_p95_ms", module="perception",
        value={"latency_p95_ms": 350.0}, passed=True, is_regression=False,
    )

    incident_spec = {
        "verifier_type": "metric_condition",
        "incident_id": "INC-003",
        # Own condition: segmentation_coverage < 0.05 — does NOT breach (coverage=0.3)
        "condition": _seg_coverage_condition(threshold=0.05),
        # other_conditions: a latency spike condition — DOES breach (latency=350 > 200)
        "other_conditions": [
            {
                "verifier_type": "metric_condition",
                "condition": _lat_condition(threshold=200.0),
            }
        ],
    }

    rc = generate_report(
        "perception", "t", [_seg_result(0.3), lat_result], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 0  # The golden gate is unaffected — latency is not in the quality threshold
    assert doc["pass"] is True
    assert doc["incident_verdict"]["verdict"] == "new_signature"


def test_incident_verdict_fixed_when_other_conditions_empty(tmp_path):
    """With an empty other_conditions list, no new-signature trip -> verdict='fixed'."""
    th = _validity_thresholds()
    incident_spec = {
        "verifier_type": "metric_condition",
        "condition": _seg_coverage_condition(threshold=0.05),
        "other_conditions": [],  # explicitly empty
    }
    generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["incident_verdict"]["verdict"] == "fixed"


# ---------------------------------------------------------------------------
# Test: exit code is UNCHANGED by incident_spec (T-0104-03)
# ---------------------------------------------------------------------------

def test_exit_code_unchanged_by_incident_spec_pass(tmp_path):
    """incident_spec does NOT change the B9 exit code (0 = PASS)."""
    th = _validity_thresholds()
    incident_spec = {
        "verifier_type": "metric_condition",
        "condition": _seg_coverage_condition(threshold=0.05),
    }
    rc_without = generate_report("perception", "t0", [_seg_result(0.3)], tmp_path / "a",
                                 th, faithfulness=_good_faithfulness())
    rc_with = generate_report("perception", "t1", [_seg_result(0.3)], tmp_path / "b",
                               th, faithfulness=_good_faithfulness(), incident_spec=incident_spec)
    assert rc_without == rc_with == 0


def test_exit_code_unchanged_by_incident_spec_fail(tmp_path):
    """incident_spec does NOT change the B9 exit code (1 = FAIL)."""
    th = {
        "segmentation_coverage": ThresholdSpec(min=0.9, tier="quality"),
    }
    faithfulness = _good_faithfulness()
    incident_spec = {
        "verifier_type": "metric_condition",
        "condition": _seg_coverage_condition(threshold=0.05),
    }
    # coverage=0.3, threshold min=0.9 -> quality FAIL (exit 1)
    rc_without = generate_report("perception", "t0", [_seg_result(0.3)], tmp_path / "a",
                                 th, faithfulness=faithfulness)
    rc_with = generate_report("perception", "t1", [_seg_result(0.3)], tmp_path / "b",
                               th, faithfulness=faithfulness, incident_spec=incident_spec)
    assert rc_without == rc_with == 1


# ---------------------------------------------------------------------------
# Test: "regressed" verdict when quality_pass=False but validity is ok and not reproduced
# ---------------------------------------------------------------------------

def test_incident_verdict_regressed_when_quality_fails_but_not_reproduced(tmp_path):
    """Quality fails (regression) + validity ok + condition not reproduced -> 'regressed'."""
    th = {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "segmentation_coverage": ThresholdSpec(min=0.9, tier="quality"),  # will fail
    }
    incident_spec = {
        "verifier_type": "metric_condition",
        # The incident's own condition does NOT breach (coverage < 0.05 is False at 0.3)
        "condition": _seg_coverage_condition(threshold=0.05),
    }
    # coverage=0.3: quality threshold min=0.9 fails -> FAIL, but condition not breaching
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 1   # quality FAIL
    assert doc["pass"] is False
    assert doc["incident_verdict"]["verdict"] == "regressed"


# ---------------------------------------------------------------------------
# Test: "inconclusive" verdict when condition is uncomputable
# ---------------------------------------------------------------------------

def test_incident_verdict_inconclusive_when_uncomputable(tmp_path):
    """When the incident's metric was not computed -> uncomputable -> 'inconclusive'."""
    th = _validity_thresholds()
    incident_spec = {
        "verifier_type": "metric_condition",
        # mask_iou_vs_golden is not in the metric_results
        "condition": {"metric": "mask_iou_vs_golden", "field": "mask_iou_vs_golden",
                      "op": "lt", "threshold": 0.5},
    }
    generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["incident_verdict"]["verdict"] == "inconclusive"


# ---------------------------------------------------------------------------
# Test: doc["pass"] / doc["verdict"] are ALWAYS present and untouched (T-0104-03)
# ---------------------------------------------------------------------------

def test_doc_pass_always_present_with_incident_spec(tmp_path):
    """doc['pass'] key is present and unmodified when incident_spec is given (T-0104-03)."""
    th = _validity_thresholds()
    incident_spec = {
        "verifier_type": "metric_condition",
        "condition": _seg_coverage_condition(threshold=0.05),
    }
    generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert "pass" in doc
    assert doc["pass"] is True   # overall == validity_pass and quality_pass


def test_doc_verdict_key_present_with_incident_spec(tmp_path):
    """doc['verdict'] (PASS/FAIL/INVALID) is present alongside incident_verdict."""
    th = _validity_thresholds()
    incident_spec = {"verifier_type": "metric_condition", "condition": _seg_coverage_condition()}
    generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(),
        incident_spec=incident_spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["verdict"] in ("PASS", "FAIL", "INVALID")
    assert "incident_verdict" in doc
