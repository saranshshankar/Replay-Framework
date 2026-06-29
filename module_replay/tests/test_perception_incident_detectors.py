"""Tests for perception.yaml seed incident_detectors block (plan 01.1-05).

Design requirements:
- Five provisional seed metric-conditions are present in perception.yaml covering
  the four named failure modes: TimeSync collapse, all-background segmentation,
  depth-all-zero, FPS collapse, latency collapse.
- load_module_config("perception", configs/modules) parses them into
  ModuleSpec.incident_detectors without coercion.
- Every entry's condition.metric is a registered perception metric name.
- Every entry's condition.op is in the whitelist {lt, le, gt, ge, eq}.
- Each seed condition is evaluable end-to-end by incident_verifier.evaluate_condition:
  breaches on a collapse value, does NOT breach on a healthy value.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from replay.module_config import load_module_config
from replay.metrics.incident_verifier import evaluate_condition, UNCOMPUTABLE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "modules"

# Registered perception metric names (from the plugin pack, plan 01-05 / interfaces block)
REGISTERED_PERCEPTION_METRICS = {
    "segmentation_coverage",
    "depth_validity",
    "pipeline_throughput_hz",
    "latency_p95_ms",
    "cross_camera_overlap_iou",
    "mask_iou_vs_golden",
}

# Operator whitelist (T-0104-01)
VALID_OPS = {"lt", "le", "gt", "ge", "eq"}

# Required seed condition keys
REQUIRED_SEED_KEYS = {
    "seg_all_background",
    "depth_all_zero",
    "fps_collapse",
    "latency_collapse",
    "timesync_collapse",
}


# ---------------------------------------------------------------------------
# Parse + structural tests
# ---------------------------------------------------------------------------

def test_incident_detectors_block_present():
    """perception.yaml carries an incident_detectors block."""
    spec = load_module_config("perception", CONFIGS_DIR)
    assert isinstance(spec.incident_detectors, dict)
    assert len(spec.incident_detectors) > 0, "incident_detectors must not be empty"


def test_five_seed_conditions_present():
    """Exactly five seed conditions are present (all four named failure modes)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    assert len(spec.incident_detectors) == 5, (
        f"Expected 5 seed conditions, got {len(spec.incident_detectors)}: "
        f"{list(spec.incident_detectors.keys())}"
    )


def test_all_required_seed_keys_present():
    """All five named seed keys are present in incident_detectors."""
    spec = load_module_config("perception", CONFIGS_DIR)
    keys = set(spec.incident_detectors.keys())
    assert keys == REQUIRED_SEED_KEYS, (
        f"Expected keys {REQUIRED_SEED_KEYS}, got {keys}"
    )


def test_every_entry_has_verifier_type_metric_condition():
    """Every seed entry carries verifier_type='metric_condition'."""
    spec = load_module_config("perception", CONFIGS_DIR)
    for key, entry in spec.incident_detectors.items():
        assert entry.get("verifier_type") == "metric_condition", (
            f"{key}: expected verifier_type='metric_condition', got {entry.get('verifier_type')!r}"
        )


def test_every_entry_has_condition_block():
    """Every seed entry has a 'condition' sub-dict."""
    spec = load_module_config("perception", CONFIGS_DIR)
    for key, entry in spec.incident_detectors.items():
        cond = entry.get("condition")
        assert isinstance(cond, dict), f"{key}: 'condition' must be a dict, got {type(cond)}"


def test_every_condition_metric_is_registered():
    """Every seed condition.metric is one of the registered perception metric names."""
    spec = load_module_config("perception", CONFIGS_DIR)
    for key, entry in spec.incident_detectors.items():
        cond = entry.get("condition", {})
        metric = cond.get("metric")
        assert metric in REGISTERED_PERCEPTION_METRICS, (
            f"{key}: condition.metric={metric!r} is not a registered perception metric; "
            f"must be one of {REGISTERED_PERCEPTION_METRICS}"
        )


def test_every_condition_op_is_valid():
    """Every seed condition.op is in the whitelist {lt, le, gt, ge, eq}."""
    spec = load_module_config("perception", CONFIGS_DIR)
    for key, entry in spec.incident_detectors.items():
        cond = entry.get("condition", {})
        op = cond.get("op")
        assert op in VALID_OPS, (
            f"{key}: condition.op={op!r} is not in the valid ops {VALID_OPS}"
        )


def test_every_entry_has_provisional_true():
    """Every seed entry carries provisional=True (pending RQ-9 / Aniket sign-off)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    for key, entry in spec.incident_detectors.items():
        assert entry.get("provisional") is True, (
            f"{key}: expected provisional=true, got {entry.get('provisional')!r}"
        )


def test_every_entry_has_title():
    """Every seed entry has a human-readable title."""
    spec = load_module_config("perception", CONFIGS_DIR)
    for key, entry in spec.incident_detectors.items():
        title = entry.get("title")
        assert isinstance(title, str) and len(title) > 0, (
            f"{key}: missing or empty 'title'"
        )


# ---------------------------------------------------------------------------
# End-to-end evaluability: each seed condition is evaluable by the plan-04 verifier
# ---------------------------------------------------------------------------
#
# Each test pair proves: (a) a collapse value BREACHES the condition (reproduced=True),
# (b) a healthy value does NOT breach (reproduced=False). This proves the seed entry
# is evaluable end-to-end against incident_verifier.evaluate_condition.
#
# Collapse values are hand-chosen to trigger the condition; healthy values are
# realistic "good run" values from the real e2e bag.

def _build_values(metric: str, field: str, value: float) -> dict:
    """Build a minimal values dict for evaluate_condition."""
    return {metric: {field: value}}


def test_seg_all_background_breaches_on_collapse():
    """seg_all_background: coverage 0.0 breaches (all-background = reproduced)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["seg_all_background"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 0.0)
    result = evaluate_condition(cond, values)
    assert result is True, "seg_all_background should breach on 0.0 coverage"


def test_seg_all_background_does_not_breach_on_healthy():
    """seg_all_background: coverage 0.45 does NOT breach (healthy run)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["seg_all_background"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 0.45)
    result = evaluate_condition(cond, values)
    assert result is False, "seg_all_background should NOT breach on 0.45 coverage"


def test_depth_all_zero_breaches_on_collapse():
    """depth_all_zero: depth_validity 0.0 breaches (all-zero = reproduced)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["depth_all_zero"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 0.0)
    result = evaluate_condition(cond, values)
    assert result is True, "depth_all_zero should breach on 0.0 validity"


def test_depth_all_zero_does_not_breach_on_healthy():
    """depth_all_zero: depth_validity 0.85 does NOT breach (healthy run)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["depth_all_zero"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 0.85)
    result = evaluate_condition(cond, values)
    assert result is False, "depth_all_zero should NOT breach on 0.85 validity"


def test_fps_collapse_breaches_on_low_fps():
    """fps_collapse: 0.5 Hz breaches (FPS floor = reproduced)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["fps_collapse"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 0.5)
    result = evaluate_condition(cond, values)
    assert result is True, "fps_collapse should breach on 0.5 Hz throughput"


def test_fps_collapse_does_not_breach_on_healthy():
    """fps_collapse: 5.2 Hz does NOT breach (healthy EoMT run)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["fps_collapse"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 5.2)
    result = evaluate_condition(cond, values)
    assert result is False, "fps_collapse should NOT breach on 5.2 Hz throughput"


def test_latency_collapse_breaches_on_high_latency():
    """latency_collapse: 350.0 ms breaches (latency > budget = reproduced)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["latency_collapse"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 350.0)
    result = evaluate_condition(cond, values)
    assert result is True, "latency_collapse should breach on 350 ms"


def test_latency_collapse_does_not_breach_on_healthy():
    """latency_collapse: 38.0 ms does NOT breach (healthy latency)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["latency_collapse"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 38.0)
    result = evaluate_condition(cond, values)
    assert result is False, "latency_collapse should NOT breach on 38 ms"


def test_timesync_collapse_breaches_on_low_consistency():
    """timesync_collapse: temporal_consistency_mean 0.05 breaches (all-or-nothing sync = reproduced)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["timesync_collapse"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 0.05)
    result = evaluate_condition(cond, values)
    assert result is True, "timesync_collapse should breach on 0.05 temporal consistency"


def test_timesync_collapse_does_not_breach_on_healthy():
    """timesync_collapse: temporal_consistency_mean 0.85 does NOT breach (healthy sync)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    cond = spec.incident_detectors["timesync_collapse"]["condition"]
    values = _build_values(cond["metric"], cond["field"], 0.85)
    result = evaluate_condition(cond, values)
    assert result is False, "timesync_collapse should NOT breach on 0.85 temporal consistency"


# ---------------------------------------------------------------------------
# UNCOMPUTABLE: missing metric is never a silent pass
# ---------------------------------------------------------------------------

def test_evaluate_condition_returns_uncomputable_for_missing_metric():
    """A missing metric in the values dict returns UNCOMPUTABLE (never a silent pass)."""
    spec = load_module_config("perception", CONFIGS_DIR)
    # Use seg_all_background's condition but pass an empty values dict
    cond = spec.incident_detectors["seg_all_background"]["condition"]
    result = evaluate_condition(cond, {})
    assert result is UNCOMPUTABLE, "Missing metric must return UNCOMPUTABLE"


# ---------------------------------------------------------------------------
# Existing tests still pass: perception.yaml change is additive only
# ---------------------------------------------------------------------------

def test_existing_perception_yaml_keys_unaffected():
    """Additive-only check: core ModuleSpec fields are unchanged after adding incident_detectors."""
    spec = load_module_config("perception", CONFIGS_DIR)
    # Core topology fields
    assert spec.name == "perception"
    assert spec.container == "planner"
    assert spec.colcon_package == "realtime_perception"
    assert len(spec.input_topics) > 0
    assert len(spec.output_topics) > 0
    # Thresholds block still parsed (not destroyed by additive change)
    assert "latency_p95_ms" in spec.thresholds
    assert "segmentation_coverage" in spec.thresholds
    # Metric-cfg fields still parsed
    assert spec.diagnostics_topic == "/perception_node/diagnostics"
    assert spec.latency_stage == "inference_seg_extract_segmentation"
    assert "image_raw_sim" in spec.expected_hz
