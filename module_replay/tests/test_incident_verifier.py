"""Tests for incident_verifier — the pluggable metric-condition seam (D-13/D-14).

Design requirements (01.1-04):
- evaluate_condition: evaluates a metric+field+op+threshold condition against a
  hand-built value dict. Pure, offline, no bag, no replay, no rclpy/torch.
- evaluate_incident: dispatches by verifier_type; metric_condition is active for
  perception; error_code verifier with a missing/None code returns NOT-blocking.
- UNCOMPUTABLE sentinel: a missing metric or missing field is NEVER a silent pass.
- An unknown verifier_type raises ValueError (fail loud at config time).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Import the module under test (will fail until Task 1 creates the file)
# ---------------------------------------------------------------------------

from replay.metrics.incident_verifier import (
    UNCOMPUTABLE,
    evaluate_condition,
    evaluate_incident,
    metric_values_by_name,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _seg_result(coverage: float):
    """Build a MetricResult-like object for segmentation_coverage."""
    from replay.metrics.base import MetricResult
    return MetricResult(
        name="segmentation_coverage",
        module="perception",
        value={
            "segmentation_coverage": coverage,
            "temporal_consistency_mean": 0.9,
            "mean_class_coverage": 0.5,
        },
        passed=True,
        is_regression=False,
    )


def _lat_result(latency_ms: float):
    from replay.metrics.base import MetricResult
    return MetricResult(
        name="latency_p95_ms",
        module="perception",
        value={"latency_p95_ms": latency_ms},
        passed=True,
        is_regression=False,
    )


# ---------------------------------------------------------------------------
# metric_values_by_name
# ---------------------------------------------------------------------------

def test_metric_values_by_name_maps_name_to_value():
    """metric_values_by_name returns {metric_name -> value_dict} for each result."""
    results = [_seg_result(0.3), _lat_result(42.0)]
    mapping = metric_values_by_name(results)
    assert "segmentation_coverage" in mapping
    assert "latency_p95_ms" in mapping
    assert mapping["segmentation_coverage"]["segmentation_coverage"] == 0.3
    assert mapping["latency_p95_ms"]["latency_p95_ms"] == 42.0


def test_metric_values_by_name_empty_returns_empty():
    assert metric_values_by_name([]) == {}


# ---------------------------------------------------------------------------
# evaluate_condition — basic ops
# ---------------------------------------------------------------------------

def test_condition_lt_breaches_when_value_below_threshold():
    """segmentation_coverage < 0.05 breaches on value 0.0 (all-background collapse)."""
    cond = {"metric": "segmentation_coverage", "field": "segmentation_coverage",
            "op": "lt", "threshold": 0.05}
    values = {"segmentation_coverage": {"segmentation_coverage": 0.0}}
    assert evaluate_condition(cond, values) is True  # breaches


def test_condition_lt_does_not_breach_on_healthy_value():
    """segmentation_coverage < 0.05 does NOT breach on 0.3."""
    cond = {"metric": "segmentation_coverage", "field": "segmentation_coverage",
            "op": "lt", "threshold": 0.05}
    values = {"segmentation_coverage": {"segmentation_coverage": 0.3}}
    assert evaluate_condition(cond, values) is False  # does not breach


def test_condition_le_breaches_at_exact_threshold():
    cond = {"metric": "latency_p95_ms", "field": "latency_p95_ms",
            "op": "le", "threshold": 100.0}
    values = {"latency_p95_ms": {"latency_p95_ms": 100.0}}
    assert evaluate_condition(cond, values) is True


def test_condition_gt_breaches_when_value_above_threshold():
    cond = {"metric": "latency_p95_ms", "field": "latency_p95_ms",
            "op": "gt", "threshold": 200.0}
    values = {"latency_p95_ms": {"latency_p95_ms": 350.0}}
    assert evaluate_condition(cond, values) is True


def test_condition_ge_breaches_at_exact_threshold():
    cond = {"metric": "depth_validity", "field": "depth_validity",
            "op": "ge", "threshold": 0.9}
    values = {"depth_validity": {"depth_validity": 0.9}}
    assert evaluate_condition(cond, values) is True


def test_condition_ge_does_not_breach_below_threshold():
    cond = {"metric": "depth_validity", "field": "depth_validity",
            "op": "ge", "threshold": 0.9}
    values = {"depth_validity": {"depth_validity": 0.5}}
    assert evaluate_condition(cond, values) is False


def test_condition_eq_breaches_on_exact_match():
    cond = {"metric": "segmentation_coverage", "field": "segmentation_coverage",
            "op": "eq", "threshold": 0.0}
    values = {"segmentation_coverage": {"segmentation_coverage": 0.0}}
    assert evaluate_condition(cond, values) is True


def test_condition_eq_does_not_breach_on_different_value():
    cond = {"metric": "segmentation_coverage", "field": "segmentation_coverage",
            "op": "eq", "threshold": 0.0}
    values = {"segmentation_coverage": {"segmentation_coverage": 0.3}}
    assert evaluate_condition(cond, values) is False


# ---------------------------------------------------------------------------
# evaluate_condition — field defaults to metric name
# ---------------------------------------------------------------------------

def test_condition_field_defaults_to_metric_name():
    """When 'field' is omitted from condition, it defaults to condition['metric']."""
    cond = {"metric": "segmentation_coverage", "op": "lt", "threshold": 0.05}
    values = {"segmentation_coverage": {"segmentation_coverage": 0.0}}
    # Should still resolve correctly using metric name as field
    assert evaluate_condition(cond, values) is True


# ---------------------------------------------------------------------------
# evaluate_condition — UNCOMPUTABLE sentinel
# ---------------------------------------------------------------------------

def test_condition_returns_uncomputable_when_metric_absent():
    """A metric not present in values -> UNCOMPUTABLE (never a silent pass)."""
    cond = {"metric": "mask_iou_vs_golden", "field": "mask_iou_vs_golden",
            "op": "lt", "threshold": 0.5}
    values = {"segmentation_coverage": {"segmentation_coverage": 0.3}}
    result = evaluate_condition(cond, values)
    assert result is UNCOMPUTABLE


def test_condition_returns_uncomputable_when_field_absent():
    """A field missing from the metric's value dict -> UNCOMPUTABLE."""
    cond = {"metric": "segmentation_coverage", "field": "nonexistent_field",
            "op": "lt", "threshold": 0.5}
    values = {"segmentation_coverage": {"segmentation_coverage": 0.3}}
    result = evaluate_condition(cond, values)
    assert result is UNCOMPUTABLE


def test_condition_returns_uncomputable_when_field_is_none():
    """A field present but with None value -> UNCOMPUTABLE."""
    cond = {"metric": "segmentation_coverage", "field": "segmentation_coverage",
            "op": "lt", "threshold": 0.05}
    values = {"segmentation_coverage": {"segmentation_coverage": None}}
    result = evaluate_condition(cond, values)
    assert result is UNCOMPUTABLE


# ---------------------------------------------------------------------------
# evaluate_incident — metric_condition verifier type
# ---------------------------------------------------------------------------

def test_evaluate_incident_metric_condition_breaches():
    """metric_condition: reproduced=True when the condition breaches."""
    spec = {
        "verifier_type": "metric_condition",
        "condition": {"metric": "segmentation_coverage", "field": "segmentation_coverage",
                      "op": "lt", "threshold": 0.05},
    }
    results = [_seg_result(0.0)]
    out = evaluate_incident(spec, results)
    assert out["reproduced"] is True
    assert out.get("uncomputable") is not True


def test_evaluate_incident_metric_condition_not_breaching():
    """metric_condition: reproduced=False when the condition does NOT breach."""
    spec = {
        "verifier_type": "metric_condition",
        "condition": {"metric": "segmentation_coverage", "field": "segmentation_coverage",
                      "op": "lt", "threshold": 0.05},
    }
    results = [_seg_result(0.3)]
    out = evaluate_incident(spec, results)
    assert out["reproduced"] is False
    assert out.get("uncomputable") is not True


def test_evaluate_incident_metric_condition_uncomputable():
    """metric_condition: uncomputable=True when the metric was not computed."""
    spec = {
        "verifier_type": "metric_condition",
        "condition": {"metric": "mask_iou_vs_golden", "field": "mask_iou_vs_golden",
                      "op": "lt", "threshold": 0.5},
    }
    # No mask_iou_vs_golden in results
    results = [_seg_result(0.3)]
    out = evaluate_incident(spec, results)
    assert out.get("uncomputable") is True
    # reproduced must be None or falsy — never a silent pass treated as "fixed"
    assert out.get("reproduced") is None or out.get("reproduced") is False


def test_evaluate_incident_defaults_verifier_type_to_metric_condition():
    """When verifier_type is absent, defaults to metric_condition."""
    spec = {
        # no verifier_type key
        "condition": {"metric": "segmentation_coverage", "field": "segmentation_coverage",
                      "op": "lt", "threshold": 0.05},
    }
    results = [_seg_result(0.0)]
    out = evaluate_incident(spec, results)
    assert out["reproduced"] is True


# ---------------------------------------------------------------------------
# evaluate_incident — error_code verifier type (D-14: never blocks)
# ---------------------------------------------------------------------------

def test_evaluate_incident_error_code_missing_code_not_blocking():
    """D-14: error_code verifier with a missing/None code returns reproduced=False
    AND blocking=False. A missing code NEVER blocks."""
    spec = {
        "verifier_type": "error_code",
        # No 'code' key — simulates a missing/absent error code
    }
    out = evaluate_incident(spec, [])
    assert out["reproduced"] is False
    assert out["blocking"] is False


def test_evaluate_incident_error_code_none_code_not_blocking():
    """error_code verifier with code=None is still not blocking."""
    spec = {
        "verifier_type": "error_code",
        "code": None,
    }
    out = evaluate_incident(spec, [])
    assert out["reproduced"] is False
    assert out["blocking"] is False


def test_evaluate_incident_error_code_has_informative_note():
    """error_code verifier returns a 'note' explaining it never blocks (D-14)."""
    spec = {"verifier_type": "error_code"}
    out = evaluate_incident(spec, [])
    assert "note" in out
    assert "never blocks" in out["note"].lower() or "d-14" in out["note"].lower() or "missing" in out["note"].lower()


# ---------------------------------------------------------------------------
# evaluate_incident — unknown verifier_type raises ValueError
# ---------------------------------------------------------------------------

def test_evaluate_incident_unknown_verifier_type_raises():
    """An unknown verifier_type raises ValueError (fail loud at config time)."""
    spec = {"verifier_type": "magic_unknown_type"}
    with pytest.raises(ValueError, match="magic_unknown_type"):
        evaluate_incident(spec, [])


# ---------------------------------------------------------------------------
# Offline-pure invariant: no rclpy or torch imports in the module
# ---------------------------------------------------------------------------

def test_incident_verifier_is_offline_pure():
    """The incident_verifier module must not import rclpy or torch (offline-pure)."""
    import importlib
    import sys

    # Ensure the module is imported (it should be by now)
    import replay.metrics.incident_verifier as iv_mod
    source_file = iv_mod.__file__
    assert source_file is not None

    text = open(source_file).read()
    assert "import rclpy" not in text, "incident_verifier must not import rclpy"
    assert "import torch" not in text, "incident_verifier must not import torch"


# ---------------------------------------------------------------------------
# No @register_metric decorator (explicitly-invoked, not registered)
# ---------------------------------------------------------------------------

def test_incident_verifier_not_registered():
    """incident_verifier must NOT use @register_metric — it is explicitly invoked,
    following the replay_faithfulness.py precedent."""
    import replay.metrics.incident_verifier as iv_mod
    source_file = iv_mod.__file__
    text = open(source_file).read()
    assert "@register_metric" not in text, (
        "incident_verifier must not be a @register_metric plugin — "
        "it is explicitly invoked (faithfulness precedent)"
    )
