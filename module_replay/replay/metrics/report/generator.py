"""Criteria evaluator + report generator: the PRIMARY gate (MTRC-03) and its
machine-readable output (MTRC-05).

Takes the metric dicts produced by the perception plugins plus the (explicitly
invoked) replay-faithfulness validity dict, evaluates each against its
``ThresholdSpec`` (tolerance-based, never bit-exact — RESEARCH Pitfall 4),
AND-gates the quality-tier criteria into one pass/fail verdict, short-circuits
on a validity-tier breach, writes ``metrics.json`` + ``report.html``, and
returns the exit code the CI gate reads.

EXIT-CODE CONTRACT (SYSTEM-DESIGN-HLD-LLD B9):
    0 = PASS
    1 = quality criterion breached (a real regression -> FAIL)
    2 = INVALID RUN (a validity tier breach -> infra noise, not a code regression)

A configured threshold whose metric was NOT computed this run (e.g.
``mask_iou_vs_golden`` before any golden fixture exists) is recorded as a
visible "skipped" row with a warning note — it never fails the gate, and a
metric with no configured threshold is recorded with ``passed: None`` so a gate
with no teeth can never silently pass (RESEARCH Pitfall 7 / threat T-07-02).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from replay.metrics.base import MetricResult


def evaluate_threshold(value: float, threshold) -> bool:
    """Tolerance-based pass check (RESEARCH Pitfall 4).

    ``threshold`` is a ``ThresholdSpec``. A ``None`` max/min means that side is
    unconstrained. Returns True if the value sits within the bounds widened by
    the tolerance band:
      - max threshold: passed if value <= max + tolerance_band
      - min threshold: passed if value >= min - tolerance_band
    """
    tb = getattr(threshold, "tolerance_band", 0.0) or 0.0
    if threshold.max is not None and value > threshold.max + tb:
        return False
    if threshold.min is not None and value < threshold.min - tb:
        return False
    return True


def generate_report(
    module: str,
    run_id: str,
    metric_results: list[MetricResult],
    output_dir: Path,
    thresholds: dict,
    faithfulness: Optional[dict] = None,
) -> int:
    """Evaluate metrics vs thresholds, AND-gate quality, write artifacts, return the B9 exit code.

    See module docstring for the exit-code contract.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validity tier (faithfulness) short-circuit ──────────────────────────
    # Evaluate EVERY tier=="validity" threshold against the faithfulness dict.
    # Key map: a "replay_"-prefixed threshold name maps onto the faithfulness
    # field with that prefix stripped (replay_max_gap_ms -> max_gap_ms,
    # replay_drop_rate -> drop_rate); a non-prefixed name is matched verbatim.
    validity_pass = True
    faith_block = None
    if faithfulness is not None:
        for tname, vt in thresholds.items():
            if getattr(vt, "tier", "quality") != "validity":
                continue
            fkey = tname[len("replay_"):] if tname.startswith("replay_") else tname
            if fkey in faithfulness:
                validity_pass = validity_pass and evaluate_threshold(
                    float(faithfulness[fkey]), vt
                )
        faith_block = {
            **faithfulness,
            "verdict": "pass" if validity_pass else "fail",
            "tier": "validity",
        }

    # ── Quality tier (AND-gate) ─────────────────────────────────────────────
    rows = []
    quality_pass = True
    for r in metric_results:
        spec = thresholds.get(r.name)
        # The plugin's headline scalar lives in value[r.name] when present.
        scalar = r.value.get(r.name) if isinstance(r.value, dict) and r.name in r.value else None
        if spec is None or scalar is None:
            # Pitfall 7 / T-07-02: no threshold (or no scalar) -> record but do
            # NOT silently pass a gate with no teeth.
            rows.append(
                {
                    "name": r.name,
                    "value": scalar,
                    "passed": None,
                    "provisional": getattr(spec, "provisional", True) if spec else True,
                    "tier": getattr(spec, "tier", "quality") if spec else "quality",
                    "note": "no threshold configured",
                }
            )
            continue
        passed = evaluate_threshold(float(scalar), spec)
        quality_pass = quality_pass and passed
        rows.append(
            {
                "name": r.name,
                "value": scalar,
                "threshold_max": spec.max,
                "threshold_min": spec.min,
                "tolerance_band": spec.tolerance_band,
                "passed": passed,
                "provisional": spec.provisional,
                "tier": spec.tier,
            }
        )

    # Configured-but-uncomputed quality thresholds (e.g. mask_iou_vs_golden
    # before any golden fixture exists) are emitted as explicit "skipped" rows —
    # visible, never a failure.
    computed = {r.name for r in metric_results}
    for tname, spec in thresholds.items():
        if getattr(spec, "tier", "quality") == "quality" and tname not in computed:
            rows.append(
                {
                    "name": tname,
                    "value": None,
                    "passed": None,
                    "tier": "quality",
                    "provisional": spec.provisional,
                    "note": "skipped — metric not computed this run",
                }
            )

    overall = validity_pass and quality_pass
    verdict = "INVALID" if not validity_pass else ("PASS" if quality_pass else "FAIL")
    failed_quality = [m["name"] for m in rows if m.get("passed") is False]
    if not validity_pass:
        details = "validity tier breached — INVALID RUN (infra, not a code regression)"
    elif failed_quality:
        details = f"{len(failed_quality)} quality criterion failed: {', '.join(failed_quality)}"
    else:
        details = "all configured criteria passed"

    doc = {
        "module": module,
        "run_id": run_id,
        "pass": overall,
        "replay_faithfulness": faith_block,
        "metrics": rows,
        "verdict": verdict,
        "details": details,
    }
    # metrics.json is the machine-read gate signal (CI reads doc["pass"] / verdict).
    # json.dumps cannot be HTML-injected, so the values pass through untransformed.
    (output_dir / "metrics.json").write_text(json.dumps(doc, indent=2))

    # report.html — rendered with autoescape ON (threat T-07-01) so a crafted
    # topic/metric name string can never inject markup into the artifact.
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent)),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("template.html").render(doc=doc)
    (output_dir / "report.html").write_text(html)

    # B9: INVALID RUN (2) beats FAIL (1) — infra noise must never read as a code regression.
    return 2 if not validity_pass else (0 if quality_pass else 1)
