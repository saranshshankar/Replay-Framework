"""Tests for the additive incident_verdict in generate_report (01.1 — config-as-checkset, D-21).

Model (D-21):
- incident_spec carries a CHECK SET: {"checks": {key: detector_dict}, "incident_id", "mode"}.
  The gate runs EVERY detector; "fixed" == VALID AND no detector trips.
- generate_report(..., incident_spec=None) is the golden path: doc has no incident_verdict
  (or it's None) AND doc["pass"]/exit-code are byte-for-byte unchanged.
- Any detector tripping -> verdict "not_fixed" (with a `tripped` list).
- not validity_pass (INVALID) -> verdict FORCED "inconclusive", NEVER "fixed".
- A detector whose metric is uncomputable (and none tripped) -> "inconclusive".
- THE KNOB: golden-quality thresholds are NOT part of the incident verdict — a VALID run
  with no detector tripping is "fixed" even if golden quality FAILs (rc=1).
- doc["pass"] and the 0/1/2 exit code are UNCHANGED by incident_spec (T-0104-03).
"""
from __future__ import annotations

import json

from replay.metrics.base import MetricResult
from replay.module_config import ThresholdSpec
from replay.metrics.report.generator import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg_result(coverage: float):
    return MetricResult(
        name="segmentation_coverage",
        module="perception",
        value={"segmentation_coverage": coverage, "temporal_consistency_mean": 0.9,
               "mean_class_coverage": 0.5},
        passed=True,
        is_regression=False,
    )


def _lat_result(latency_ms: float):
    return MetricResult(
        name="latency_p95_ms", module="perception",
        value={"latency_p95_ms": latency_ms}, passed=True, is_regression=False,
    )


def _quality_thresholds():
    return {"segmentation_coverage": ThresholdSpec(min=0.05, tier="quality")}


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
    """A condition that breaches (known failure present) when segmentation_coverage < threshold."""
    return {"metric": "segmentation_coverage", "field": "segmentation_coverage",
            "op": "lt", "threshold": threshold}


def _lat_condition(threshold=200.0):
    """A condition that breaches when latency_p95_ms > threshold."""
    return {"metric": "latency_p95_ms", "field": "latency_p95_ms",
            "op": "gt", "threshold": threshold}


def _check(condition):
    """Wrap a raw condition as a metric_condition detector (the incident_detectors shape)."""
    return {"verifier_type": "metric_condition", "condition": condition}


def _spec(checks: dict, incident_id="INC-001", mode="all"):
    return {"incident_id": incident_id, "mode": mode,
            "checks": checks, "verifier_type": "metric_condition"}


# ---------------------------------------------------------------------------
# Test 1: incident_spec=None (golden path) — no incident_verdict, pass/rc unchanged
# ---------------------------------------------------------------------------

def test_no_incident_spec_golden_path_pass(tmp_path):
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, _quality_thresholds(),
        faithfulness=_good_faithfulness(), incident_spec=None,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 0
    assert doc["pass"] is True
    assert doc.get("incident_verdict") is None


def test_no_incident_spec_golden_path_fail(tmp_path):
    th = {"segmentation_coverage": ThresholdSpec(min=0.5, tier="quality")}
    rc = generate_report(
        "perception", "t", [_seg_result(0.1)], tmp_path, th,
        faithfulness=_good_faithfulness(), incident_spec=None,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 1
    assert doc["pass"] is False
    assert doc.get("incident_verdict") is None


def test_no_incident_spec_doc_pass_present(tmp_path):
    generate_report("perception", "t", [_seg_result(0.3)], tmp_path, _quality_thresholds())
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert "pass" in doc


# ---------------------------------------------------------------------------
# Test 2: "fixed" — VALID + no detector trips
# ---------------------------------------------------------------------------

def test_incident_verdict_fixed_on_valid_no_trip(tmp_path):
    spec = _spec({"seg_collapse": _check(_seg_coverage_condition(0.05))})  # breaches if coverage<0.05
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, _validity_thresholds(),
        faithfulness=_good_faithfulness(), incident_spec=spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 0
    assert doc["pass"] is True
    iv = doc["incident_verdict"]
    assert iv["verdict"] == "fixed"
    assert iv["incident_id"] == "INC-001"
    assert iv["tripped"] == []


def test_incident_verdict_fixed_multi_check_none_trip(tmp_path):
    """ALL detectors run (CI mode); none trips -> fixed."""
    spec = _spec({
        "seg_collapse": _check(_seg_coverage_condition(0.05)),
        "latency_collapse": _check(_lat_condition(200.0)),
    })
    generate_report(
        "perception", "t", [_seg_result(0.3), _lat_result(40.0)], tmp_path,
        _validity_thresholds(), faithfulness=_good_faithfulness(), incident_spec=spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert doc["incident_verdict"]["verdict"] == "fixed"
    assert doc["incident_verdict"]["tripped"] == []


# ---------------------------------------------------------------------------
# Test 3: "not_fixed" — at least one detector trips (with the tripped list)
# ---------------------------------------------------------------------------

def test_incident_verdict_not_fixed_when_detector_trips(tmp_path):
    spec = _spec({"seg_collapse": _check(_seg_coverage_condition(0.05))})
    generate_report(
        "perception", "t", [_seg_result(0.0)], tmp_path, _validity_thresholds(),
        faithfulness=_good_faithfulness(), incident_spec=spec,
    )
    iv = json.loads((tmp_path / "metrics.json").read_text())["incident_verdict"]
    assert iv["verdict"] == "not_fixed"
    assert iv["tripped"] == ["seg_collapse"]


def test_incident_verdict_not_fixed_when_one_of_many_trips(tmp_path):
    """CI runs ALL detectors; one (latency) trips -> not_fixed, listing only the tripped one."""
    spec = _spec({
        "seg_collapse": _check(_seg_coverage_condition(0.05)),     # coverage 0.3 -> no trip
        "latency_collapse": _check(_lat_condition(200.0)),         # latency 350 -> trips
    })
    generate_report(
        "perception", "t", [_seg_result(0.3), _lat_result(350.0)], tmp_path,
        _validity_thresholds(), faithfulness=_good_faithfulness(), incident_spec=spec,
    )
    iv = json.loads((tmp_path / "metrics.json").read_text())["incident_verdict"]
    assert iv["verdict"] == "not_fixed"
    assert iv["tripped"] == ["latency_collapse"]


# ---------------------------------------------------------------------------
# Test 4: INVALID run forces inconclusive (never-fixed-on-INVALID)
# ---------------------------------------------------------------------------

def test_incident_verdict_inconclusive_on_invalid_run(tmp_path):
    th = {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "segmentation_coverage": ThresholdSpec(min=0.05, tier="quality"),
    }
    spec = _spec({"seg_collapse": _check(_seg_coverage_condition(0.05))})
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_breached_faithfulness(),   # max_gap_ms 500 > 200 -> INVALID
        incident_spec=spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 2
    assert doc["verdict"] == "INVALID"
    assert doc["incident_verdict"]["verdict"] == "inconclusive"


def test_incident_verdict_inconclusive_via_missing_faithfulness_key(tmp_path):
    th = {"replay_jitter_ms": ThresholdSpec(max=50.0, tier="validity")}
    spec = _spec({"seg_collapse": _check(_seg_coverage_condition(0.05))})
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness={"max_gap_ms": 100.0, "breach_count": 0, "drop_rate": 0.0},
        incident_spec=spec,
    )
    iv = json.loads((tmp_path / "metrics.json").read_text())["incident_verdict"]
    assert rc == 2
    assert iv["verdict"] == "inconclusive"


# ---------------------------------------------------------------------------
# Test 5: THE KNOB — golden quality is NOT part of the incident verdict
# ---------------------------------------------------------------------------

def test_incident_verdict_fixed_even_when_golden_quality_fails(tmp_path):
    """A VALID incident run where no detector trips is 'fixed' EVEN IF golden quality FAILs
    (rc=1). An incident bag is a degraded scenario — the bar is 'catastrophic signatures gone',
    not full golden quality (D-21 knob)."""
    th = {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "segmentation_coverage": ThresholdSpec(min=0.9, tier="quality"),  # 0.3 -> quality FAIL
    }
    # detector trips only if coverage < 0.05; coverage 0.3 does NOT trip the catastrophic check
    spec = _spec({"seg_collapse": _check(_seg_coverage_condition(0.05))})
    rc = generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, th,
        faithfulness=_good_faithfulness(), incident_spec=spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert rc == 1               # golden quality FAIL still drives the exit code...
    assert doc["pass"] is False  # ...and doc["pass"] (unchanged contract)
    assert doc["incident_verdict"]["verdict"] == "fixed"   # ...but the incident is fixed (the knob)


# ---------------------------------------------------------------------------
# Test 6: uncomputable detector -> inconclusive (never a silent pass)
# ---------------------------------------------------------------------------

def test_incident_verdict_inconclusive_when_uncomputable(tmp_path):
    spec = _spec({"mask_iou": _check(
        {"metric": "mask_iou_vs_golden", "field": "mask_iou_vs_golden", "op": "lt", "threshold": 0.5}
    )})
    generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, _validity_thresholds(),
        faithfulness=_good_faithfulness(), incident_spec=spec,
    )
    iv = json.loads((tmp_path / "metrics.json").read_text())["incident_verdict"]
    assert iv["verdict"] == "inconclusive"
    assert iv["uncomputable"] == ["mask_iou"]


# ---------------------------------------------------------------------------
# Test 7: exit code + doc["pass"]/["verdict"] UNCHANGED by incident_spec (T-0104-03)
# ---------------------------------------------------------------------------

def test_exit_code_unchanged_by_incident_spec_pass(tmp_path):
    spec = _spec({"seg_collapse": _check(_seg_coverage_condition(0.05))})
    rc_without = generate_report("perception", "t0", [_seg_result(0.3)], tmp_path / "a",
                                 _validity_thresholds(), faithfulness=_good_faithfulness())
    rc_with = generate_report("perception", "t1", [_seg_result(0.3)], tmp_path / "b",
                              _validity_thresholds(), faithfulness=_good_faithfulness(),
                              incident_spec=spec)
    assert rc_without == rc_with == 0


def test_exit_code_unchanged_by_incident_spec_fail(tmp_path):
    th = {"segmentation_coverage": ThresholdSpec(min=0.9, tier="quality")}
    spec = _spec({"seg_collapse": _check(_seg_coverage_condition(0.05))})
    rc_without = generate_report("perception", "t0", [_seg_result(0.3)], tmp_path / "a",
                                 th, faithfulness=_good_faithfulness())
    rc_with = generate_report("perception", "t1", [_seg_result(0.3)], tmp_path / "b",
                              th, faithfulness=_good_faithfulness(), incident_spec=spec)
    assert rc_without == rc_with == 1


def test_doc_pass_and_verdict_present_with_incident_spec(tmp_path):
    spec = _spec({"seg_collapse": _check(_seg_coverage_condition(0.05))})
    generate_report(
        "perception", "t", [_seg_result(0.3)], tmp_path, _validity_thresholds(),
        faithfulness=_good_faithfulness(), incident_spec=spec,
    )
    doc = json.loads((tmp_path / "metrics.json").read_text())
    assert "pass" in doc and doc["pass"] is True
    assert doc["verdict"] in ("PASS", "FAIL", "INVALID")
    assert "incident_verdict" in doc
