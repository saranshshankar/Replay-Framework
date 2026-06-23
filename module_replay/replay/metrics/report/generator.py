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
import logging
from pathlib import Path
from typing import Optional

from replay.metrics.base import MetricResult

logger = logging.getLogger(__name__)

# Bound at module level so report tests can ``mocker.patch`` it on this module and
# so the name always exists even when matplotlib is absent (the cheap CI
# metrics-gate path never imports it). It is only CALLED when a reader is
# explicitly passed to generate_report, always inside a try/except that degrades
# to no plots — a plot failure (or a missing matplotlib) can never crash a run or
# flip the B9 verdict (T-16-02).
try:
    from replay.metrics.report.plots import generate_report_plots
except Exception:  # pragma: no cover - matplotlib optional on the metrics-only path
    generate_report_plots = None  # type: ignore[assignment]


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


# Human-readable card labels for the summary grid. Falls back to the raw row
# name (autoescaped at render) for any metric not listed here.
_SUMMARY_LABELS = {
    "latency_p95_ms": "p95 Inference Latency",
    "pipeline_throughput_hz": "Pipeline Throughput",
    "depth_validity": "Depth Validity",
    "segmentation_coverage": "Segmentation Coverage",
    "cross_camera_overlap_iou": "Cross-Camera Overlap",
    "mask_iou_vs_golden": "Mask IoU vs Golden",
    "replay_max_gap_ms": "Max Replay Gap",
    "replay_drop_rate": "Replay Drop Rate",
    "replay_breach_count": "Replay Breaches",
    "max_gap_ms": "Max Replay Gap (ms)",
    "drop_rate": "Replay Drop Rate",
    "breach_count": "Replay Breaches",
}


def _card_status(passed: Optional[bool], tier: str) -> str:
    """Map a row's (passed, tier) to a summary-card badge status.

    Mirrors the badge semantics the template renders:
      passed is True                 -> "PASS"
      passed is False & validity     -> "BREACH"  (B9 INVALID-flavoured)
      passed is False & quality      -> "FAIL"
      passed is None                 -> "NONE"    (skipped / no-threshold)
    """
    if passed is True:
        return "PASS"
    if passed is False:
        return "BREACH" if tier == "validity" else "FAIL"
    return "NONE"


def _build_summary(
    rows: list[dict],
    faithfulness: Optional[dict],
    breached_faith_fields: Optional[set] = None,
) -> list[dict]:
    """Derive the summary-card list from the evaluated metric rows + faithfulness.

    Building this in the generator (not the template) keeps the template
    logic-light and the card semantics unit-testable without rendering HTML.
    Each card = {"label", "value", "status"} where status is from ``_card_status``.
    Faithfulness headline cards (max_gap_ms, drop_rate, breach_count) are appended
    when a faithfulness dict is present; a card reads BREACH when its matching
    validity threshold breached this run (``breached_faith_fields`` carries the
    faithfulness field names whose validity gate failed — note an EVALUATED
    validity breach produces no metrics row, so the rows alone cannot reveal it).
    """
    breached_faith_fields = breached_faith_fields or set()
    summary: list[dict] = []
    for r in rows:
        summary.append(
            {
                "label": _SUMMARY_LABELS.get(r["name"], r["name"]),
                "value": r.get("value"),
                "status": _card_status(r.get("passed"), r.get("tier", "quality")),
            }
        )
    if faithfulness is not None:
        # Headline faithfulness cards (always shown when faithfulness ran).
        for fkey in ("max_gap_ms", "drop_rate", "breach_count"):
            if fkey not in faithfulness:
                continue
            summary.append(
                {
                    "label": _SUMMARY_LABELS.get(fkey, fkey),
                    "value": faithfulness[fkey],
                    "status": "BREACH" if fkey in breached_faith_fields else "PASS",
                }
            )
    return summary


def generate_report(
    module: str,
    run_id: str,
    metric_results: list[MetricResult],
    output_dir: Path,
    thresholds: dict,
    faithfulness: Optional[dict] = None,
    run_artifacts: Optional[dict] = None,
    reader=None,
    visualizations: Optional[list] = None,
) -> int:
    """Evaluate metrics vs thresholds, AND-gate quality, write artifacts, return the B9 exit code.

    See module docstring for the exit-code contract.

    ``run_artifacts`` (A3) is an optional ``{"bag": ..., "logs": ..., "report": ...}``
    pointer map rendered into the report's Debug section. It is ADDITIVE: the CI
    gate reads ``metrics.json`` (``doc["pass"]``/``doc["verdict"]``/the rows), never
    ``run_artifacts``, so it cannot affect the verdict or the exit code.

    ``reader`` (A2) is an optional ``BagReader``: when given, the report's static
    matplotlib plots are generated from the metric rows + bag. Plot generation runs
    in a try/except that degrades to no plots on ANY failure — it never alters the
    verdict or the exit code. The CLI (_run_metrics_pipeline) ALWAYS forwards a
    reader, so plots are DEFAULT on every run (incl. the CI metrics job) — the agreed
    scope. ``reader`` is None only for the pure-evaluator unit tests (and a future
    --no-plots / pure-gate mode), where no plots are generated.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Validity tier (faithfulness) short-circuit ──────────────────────────
    # Evaluate EVERY tier=="validity" threshold against the faithfulness dict.
    # Key map: a "replay_"-prefixed threshold name maps onto the faithfulness
    # field with that prefix stripped (replay_max_gap_ms -> max_gap_ms,
    # replay_drop_rate -> drop_rate); a non-prefixed name is matched verbatim.
    # ``rows`` is the report's visible record; the validity else-branch below
    # appends to it, so it must be initialized BEFORE the validity loop (the
    # quality loop then continues appending to the same list).
    rows = []
    validity_pass = True
    faith_block = None
    # Faithfulness fields whose validity threshold breached this run — used ONLY to
    # badge the additive headline summary cards (BREACH vs PASS). Purely
    # presentational; does not touch validity_pass or the exit code.
    breached_faith_fields: set[str] = set()
    if faithfulness is not None:
        for tname, vt in thresholds.items():
            if getattr(vt, "tier", "quality") != "validity":
                continue
            fkey = tname[len("replay_"):] if tname.startswith("replay_") else tname
            if fkey in faithfulness:
                field_ok = evaluate_threshold(float(faithfulness[fkey]), vt)
                if not field_ok:
                    breached_faith_fields.add(fkey)
                validity_pass = validity_pass and field_ok
            else:
                # WR-03 / T-07-02: a validity threshold that cannot be evaluated
                # must NOT pass silently. Validity is the stronger gate (exit 2) —
                # fail closed and surface it visibly, mirroring the quality tier's
                # no-silent-pass guard.
                validity_pass = False
                rows.append(
                    {
                        "name": tname,
                        "value": None,
                        "passed": None,
                        "tier": "validity",
                        "provisional": getattr(vt, "provisional", True),
                        "note": (
                            f"validity threshold has no matching faithfulness "
                            f"field '{fkey}' — failing closed"
                        ),
                    }
                )
        faith_block = {
            **faithfulness,
            "verdict": "pass" if validity_pass else "fail",
            "tier": "validity",
        }

    # ── Quality tier (AND-gate) ─────────────────────────────────────────────
    # ``rows`` was initialized before the validity loop (it may already hold a
    # failed-closed validity row); the quality loop continues appending to it.
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

    # ── Additive presentation layer (A1/A3) ────────────────────────────────
    # These keys are NEW and never rename/remove a contract key. The CI gate
    # reads doc["pass"]/doc["verdict"]/the rows; it ignores everything below.
    summary = _build_summary(rows, faithfulness, breached_faith_fields)

    # A1: surface the cross-camera overlap plugin's per-pair dict (the richer
    # value['pairs']) onto the doc so the template can render the per-camera-pair
    # table while staying logic-light. The metric row only carries the headline
    # scalar; the per-pair detail comes through here. Additive — not in the gate.
    overlap_pairs = None
    for r in metric_results:
        if r.name == "cross_camera_overlap_iou" and isinstance(r.value, dict):
            overlap_pairs = r.value.get("pairs")
            break

    doc = {
        "module": module,
        "run_id": run_id,
        "pass": overall,
        "replay_faithfulness": faith_block,
        "metrics": rows,
        "verdict": verdict,
        "details": details,
        # additive:
        "summary": summary,
        "run_artifacts": run_artifacts,
        "overlap_pairs": overlap_pairs,
        # Tier-3 viz (additive, never gated): report-relative mp4 links when the
        # --run-viz / viz path produced them, else None -> the template shows a hint.
        "visualizations": visualizations,
    }
    # metrics.json is the machine-read gate signal (CI reads doc["pass"] / verdict).
    # json.dumps cannot be HTML-injected, so the values pass through untransformed.
    (output_dir / "metrics.json").write_text(json.dumps(doc, indent=2))

    # ── A2: static plots (only when a reader is explicitly passed) ──────────
    # Generated inside a try/except that degrades to {} on ANY failure (incl. a
    # missing matplotlib) — plot generation must NEVER crash a run or flip the
    # verdict (T-16-02). Paths are made report-relative for the <img src> refs
    # (adapts the PoC _make_relative).
    plots: dict[str, str] = {}
    if reader is not None and generate_report_plots is not None:
        try:
            raw_plots = generate_report_plots(metric_results, reader, output_dir)
            for name, p in (raw_plots or {}).items():
                p = Path(p)
                try:
                    plots[name] = str(p.relative_to(output_dir))
                except ValueError:
                    plots[name] = str(p)
        except Exception:
            logger.warning("plot generation failed; rendering report without plots", exc_info=True)
            plots = {}

    # report.html — rendered with autoescape ON (threat T-07-01) so a crafted
    # topic/metric name string can never inject markup into the artifact.
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent)),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("template.html").render(doc=doc, plots=plots)
    (output_dir / "report.html").write_text(html)

    # B9: INVALID RUN (2) beats FAIL (1) — infra noise must never read as a code regression.
    return 2 if not validity_pass else (0 if quality_pass else 1)
