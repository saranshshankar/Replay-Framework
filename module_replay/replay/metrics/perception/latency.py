"""Perception inference-latency metric (MTRC-02) — seg_argmax compute time.

Ported from the PoC ``perception_metrics/metrics/latency_metrics.py`` (10xCode
branch ``aniket/feat/module_wise_replay_poc``, which uses ``rosbags`` — A2:
offline pure-Python, no ROS runtime client library). The PoC's real latency
signal is the node's SELF-REPORTED compute time published on
``/perception_node/diagnostics`` (``diagnostic_msgs/DiagnosticArray``): for each
diagnostics window the ``seg_argmax`` status carries an ``avg_compute_ms``
KeyValue, and the aggregate is the mean/p50/p95/p99/max over those windows.

Why NOT an input->output stamp delta: the perception output carries the input
frame's CAPTURE stamp unchanged (JOIN RULE / timestamp-sanctity), so an
input->output stamp delta is ~0 and measures nothing — it is the wrong signal,
not a small one. The earlier zip(input_topics, output_topics) pairing was also
positional (it paired lidar->camera-semantic and dropped most outputs, WR-01).
Both are removed; latency now measures the node's reported seg_argmax compute
time only.

GATE SEAM: the report generator enforces the 50ms quality threshold via
``value[r.name]`` (generator.py:91), i.e. it reads ``value["latency_p95_ms"]``.
``compute`` therefore emits a TOP-LEVEL scalar key ``latency_p95_ms`` (== the
metric ``name``) equal to the seg_argmax p95 — without it the gate has no teeth.

The percentile keys use ``numpy.percentile`` exactly as the PoC's seg_argmax
aggregate did.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from replay.metrics.base import BaseMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric

# The diagnostics stage whose avg_compute_ms is the perception inference latency
# (PoC config). seg_argmax is the dominant GPU stage the 50ms gate targets.
SEG_ARGMAX_STAGE = "seg_argmax"
COMPUTE_KEY = "avg_compute_ms"


def _extract_stage_avg_ms(
    diagnostics: list[tuple[int, Any]], stage_name: str, key: str
) -> list[float]:
    """Collect ``float(kv.value)`` for ``kv.key == key`` from the named status.

    Mirrors the PoC ``_extract_stage_values``: per diagnostics msg, find the
    status whose ``.name == stage_name`` and read its KeyValues. Non-parseable
    values are skipped (graceful) rather than raising.
    """
    out: list[float] = []
    for _ts, msg in diagnostics:
        for status in getattr(msg, "status", []) or []:
            if getattr(status, "name", None) != stage_name:
                continue
            for kv in getattr(status, "values", []) or []:
                if getattr(kv, "key", None) == key:
                    try:
                        out.append(float(kv.value))
                    except (ValueError, TypeError):
                        pass
            break  # one seg_argmax status per msg
    return out


@register_metric("perception")
class LatencyMetric(BaseMetric):
    """Perception inference latency (p95 ms) from seg_argmax diagnostics windows."""

    name = "latency_p95_ms"
    requires_baseline = False

    def compute(self, reader: BagReader, config: dict) -> dict:
        diag_topic = config.get("diagnostics_topic")
        diagnostics = reader.get_messages(diag_topic) if diag_topic else []

        if not diag_topic or not diagnostics:
            # VISIBLE skip — NOT a silent 0.0 pass. With no scalar latency_p95_ms
            # key, generator.py:91 records a passed:None skipped row, so the 50ms
            # gate is never (wrongly) satisfied by a fabricated zero. The old
            # input->output stamp-delta fallback is intentionally gone.
            return {
                "latency_p95_ms": None,
                "skipped": True,
                "reason": (
                    "no diagnostics topic configured/present — latency reads "
                    "seg_argmax compute time from /perception_node/diagnostics; "
                    "no stamp-delta fallback (timestamp-sanctity makes it ~0)"
                ),
                "diagnostics_topic": diag_topic,
                "num_windows": 0,
            }

        avg_values = _extract_stage_avg_ms(diagnostics, SEG_ARGMAX_STAGE, COMPUTE_KEY)
        if not avg_values:
            # Diagnostics present but no seg_argmax/avg_compute_ms — still a visible
            # skip, never a silent pass.
            return {
                "latency_p95_ms": None,
                "skipped": True,
                "reason": (
                    f"no '{SEG_ARGMAX_STAGE}' status with '{COMPUTE_KEY}' in "
                    f"{len(diagnostics)} diagnostics msgs"
                ),
                "diagnostics_topic": diag_topic,
                "num_windows": 0,
            }

        arr = np.array(avg_values, dtype=np.float64)
        p95 = round(float(np.percentile(arr, 95)), 3)
        return {
            # TOP-LEVEL scalar == self.name so generator value[r.name] enforces the gate.
            "latency_p95_ms": p95,
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": p95,
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
            "num_windows": int(arr.size),
            "skipped": False,
        }
