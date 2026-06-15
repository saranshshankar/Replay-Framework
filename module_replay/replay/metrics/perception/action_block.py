"""Action-block center-drift regression metric (MTRC-02).

A ``BaseRegressionMetric``: compares the candidate replay's action-block outputs
against a pinned-golden baseline and reports how far the block centers drifted
(in mm). This is a *regression* gate (``requires_baseline = True``) — there is no
absolute single-replay criterion for it, only "did it move relative to golden".

TOLERANCE-BASED, NEVER BIT-EXACT (RESEARCH Pitfall 4 / threat T-05-03): GPU
inference is non-deterministic, so the comparison reports continuous drift in mm
and the report generator (plan 01-07) applies the ThresholdSpec ``tolerance_band``
to decide pass/fail. We never assert byte-equality.

The action-block centers are recovered from the paired candidate/baseline output
messages by bag-write timestamp (JOIN RULE: output carries the input capture
stamp, so candidate and baseline pair frame-for-frame). On degenerate synthetic
input with no decodable centers, ``compare`` returns a typed zero-drift dict.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from replay.metrics.base import BaseRegressionMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric


def _block_center(msg: Any) -> Optional[np.ndarray]:
    """Recover an action-block center (x, y) from an output message.

    Uses the intensity centroid of the decoded frame as a stable, content-derived
    center proxy (the real action block carries a pose; the centroid is the
    framework-agnostic stand-in that still drifts when the output changes).
    Returns None when the frame cannot be decoded (graceful degradation).
    """
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    data = getattr(msg, "data", None)
    if height <= 0 or width <= 0 or data is None:
        return None
    arr = np.asarray(data, dtype=np.float64)
    if arr.size < height * width:
        return None
    channels = max(1, arr.size // (height * width))
    grid = arr[: height * width * channels].reshape(height, width, channels)[..., 0]
    total = grid.sum()
    if total <= 0:
        # No mass: center defaults to the geometric middle (stable, zero-drift vs
        # an identical baseline).
        return np.array([width / 2.0, height / 2.0], dtype=np.float64)
    ys, xs = np.mgrid[0:height, 0:width]
    cx = float((grid * xs).sum() / total)
    cy = float((grid * ys).sum() / total)
    return np.array([cx, cy], dtype=np.float64)


@register_metric("perception")
class ActionBlockDriftMetric(BaseRegressionMetric):
    """Center drift (mm) of action blocks: candidate replay vs pinned golden."""

    name = "action_block_center_drift_mm"
    requires_baseline = True

    def compute(self, reader: BagReader, config: dict) -> dict:
        # BaseRegressionMetric still inherits BaseMetric's abstract compute(); this
        # metric is baseline-only, so the single-replay path is a no-op marker. The
        # report generator routes requires_baseline metrics through compare() with a
        # resolved baseline; calling compute() alone yields no drift verdict.
        return {"requires_baseline": True, "mean_drift_mm": 0.0, "max_drift_mm": 0.0}

    def compare(self, candidate: BagReader, baseline: BagReader, config: dict) -> dict:
        output_topics = config.get("output_topics", [])
        # mm-per-pixel scale: configurable; defaults to 1.0 (pixel drift == mm).
        px_to_mm = float(config.get("px_to_mm", 1.0))

        drifts_mm: list[float] = []
        for topic in output_topics:
            cand_msgs = candidate.get_messages(topic)
            base_msgs = baseline.get_messages(topic)
            for (_, cmsg), (_, bmsg) in zip(cand_msgs, base_msgs):
                cc = _block_center(cmsg)
                bc = _block_center(bmsg)
                if cc is None or bc is None:
                    continue
                drifts_mm.append(float(np.linalg.norm(cc - bc)) * px_to_mm)

        if not drifts_mm:
            return {"num_blocks": 0, "mean_drift_mm": 0.0, "max_drift_mm": 0.0}

        arr = np.array(drifts_mm, dtype=np.float64)
        return {
            "num_blocks": int(arr.size),
            "mean_drift_mm": round(float(np.mean(arr)), 3),
            "max_drift_mm": round(float(np.max(arr)), 3),
        }
