"""Pluggable incident-verification seam (D-13 / D-14).

This module evaluates an incident's verification condition against the EXISTING
perception metric outputs (segmentation_coverage, depth_validity, pipeline_throughput_hz,
latency_p95_ms, cross_camera_overlap_iou, mask_iou_vs_golden) — NOT against error-code
observation.

Two verifier types are supported:

- ``metric_condition`` (ACTIVE for perception): evaluates a {metric, field, op, threshold}
  condition over a hand-built ``{metric_name -> value_dict}`` mapping derived from the
  MetricResult list produced by _run_metrics_pipeline. Reproduced = the condition breaches.

- ``error_code`` (optional future plug-in, e.g. nav/control): a missing or absent error
  code NEVER blocks the gate (D-14). This verifier always returns reproduced=False and
  blocking=False — it is a non-blocking informational path only.

This module is EXPLICITLY INVOKED (called by generate_report when incident_spec is given),
NOT registered as a @register_metric quality plugin — following the replay_faithfulness.py
precedent (replay_faithfulness.py:19-27). It must never appear as a quality row in the
metrics output and must never auto-run on every golden-gate pass.

Offline-pure invariant: no rclpy, no torch, no ROS runtime imports anywhere in this file.
"""
from __future__ import annotations

from operator import eq, ge, gt, le, lt
from typing import Any

# ---------------------------------------------------------------------------
# Operator whitelist (T-0104-01: no eval, no arbitrary ops — fixed lookup only)
# ---------------------------------------------------------------------------

_OPS: dict[str, Any] = {
    "lt": lt,
    "le": le,
    "gt": gt,
    "ge": ge,
    "eq": eq,
}

# ---------------------------------------------------------------------------
# UNCOMPUTABLE sentinel
# ---------------------------------------------------------------------------

#: Returned by evaluate_condition when the metric or field was not computed this
#: run. The verdict layer treats this as ``uncomputable=True``, mapping to
#: ``incident_verdict="inconclusive"`` — never a silent pass.
UNCOMPUTABLE = object()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def metric_values_by_name(metric_results) -> dict[str, dict]:
    """Map MetricResult.name -> MetricResult.value for the computed metrics.

    This is a pure projection so it is unit-testable with hand-built MetricResult
    lists. The result is consumed by evaluate_condition via evaluate_incident.

    Args:
        metric_results: iterable of MetricResult (or objects with .name / .value).

    Returns:
        dict mapping metric name -> the metric's value dict.
    """
    return {r.name: r.value for r in metric_results}


def evaluate_condition(condition: dict, values: dict[str, dict]):
    """Evaluate a single metric-condition against the computed values dict.

    A condition dict has the shape::

        {"metric": str, "field": str (optional), "op": str, "threshold": float}

    - ``metric``: the metric name to look up in ``values``.
    - ``field``: the field within that metric's value dict to compare.
      Defaults to ``metric`` when omitted (e.g. ``segmentation_coverage``
      both in metric name and in the value dict).
    - ``op``: one of ``"lt" | "le" | "gt" | "ge" | "eq"`` (T-0104-01 whitelist).
    - ``threshold``: the numeric comparison value.

    Returns:
        - ``True``  — the condition BREACHES (the incident reproduces).
        - ``False`` — the condition does NOT breach (the incident did not reproduce).
        - ``UNCOMPUTABLE`` sentinel — the metric was not computed this run, or the
          field is absent/None in the value dict. **Never a silent pass.**

    Raises:
        KeyError: if ``op`` is not in the whitelist (_OPS). Fails loud at config
            time rather than silently evaluating incorrectly.
    """
    metric = condition["metric"]
    field = condition.get("field", metric)
    op = condition["op"]
    threshold = condition["threshold"]

    # Missing metric -> UNCOMPUTABLE (never a silent pass)
    if metric not in values:
        return UNCOMPUTABLE

    metric_value = values[metric]

    # Missing field or None value -> UNCOMPUTABLE
    if field not in metric_value or metric_value[field] is None:
        return UNCOMPUTABLE

    return _OPS[op](float(metric_value[field]), float(threshold))


def evaluate_incident(incident_spec: dict, metric_results) -> dict:
    """Dispatch incident verification by verifier_type.

    For ``verifier_type="metric_condition"`` (active for perception):
        Evaluates ``incident_spec["condition"]`` over the computed metrics.
        Returns ``{"reproduced": bool|None, "uncomputable": bool, "condition": dict}``.

    For ``verifier_type="error_code"`` (optional future plug-in, D-14):
        A missing/absent error code NEVER blocks the gate. Always returns
        ``{"reproduced": False, "blocking": False, "note": "..."}``.

    Args:
        incident_spec: the incident descriptor dict with at minimum a
            ``"verifier_type"`` key (defaults to ``"metric_condition"`` when absent).
            For metric_condition, must also contain a ``"condition"`` dict.
        metric_results: iterable of MetricResult from _run_metrics_pipeline.
            May be an empty list (no plugins ran).

    Returns:
        A dict with the verification outcome. Key presence depends on verifier_type:
        - metric_condition: ``reproduced`` (bool|None), ``uncomputable`` (bool), ``condition`` (dict).
        - error_code: ``reproduced`` (False), ``blocking`` (False), ``note`` (str).

    Raises:
        ValueError: for an unknown verifier_type. Fail loud at config time,
            not silently with an ambiguous outcome.
    """
    verifier_type = incident_spec.get("verifier_type", "metric_condition")

    if verifier_type == "error_code":
        # D-14: a missing/absent error code NEVER blocks the gate. This verifier
        # is a non-blocking informational path. The optional error_code label is
        # never authoritative (T-0104-04).
        return {
            "reproduced": False,
            "blocking": False,
            "note": (
                "error-code verifier: a missing/absent code never blocks (D-14); "
                "this verifier type is informational only"
            ),
        }

    if verifier_type == "metric_condition":
        condition = incident_spec.get("condition", {})
        values = metric_values_by_name(metric_results)
        result = evaluate_condition(condition, values)

        if result is UNCOMPUTABLE:
            return {
                "reproduced": None,
                "uncomputable": True,
                "condition": condition,
            }

        return {
            "reproduced": bool(result),
            "uncomputable": False,
            "condition": condition,
        }

    raise ValueError(
        f"Unknown verifier_type {verifier_type!r}. "
        "Supported values: 'metric_condition' (active), 'error_code' (optional future plug-in)."
    )
