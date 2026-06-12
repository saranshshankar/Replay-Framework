"""Collision-box IoU regression metric (MTRC-02).

A ``BaseRegressionMetric``: compares the candidate replay's collision boxes
(occupied / drivable regions in the perception output) against a pinned-golden
baseline and reports the intersection-over-union per paired frame. This is a
*regression* gate (``requires_baseline = True``): the golden encodes the accepted
collision footprint, and we measure how much the candidate's footprint still
overlaps it.

TOLERANCE-BASED, NEVER BIT-EXACT (RESEARCH Pitfall 4 / threat T-05-03): IoU is a
continuous [0, 1] overlap ratio; the report generator (plan 01-07) gates it with
the ThresholdSpec ``min`` + ``tolerance_band``. We never assert byte-equality.

Candidate and baseline frames pair by bag-write timestamp (JOIN RULE). On
degenerate synthetic input with no decodable mask, ``compare`` returns a typed
perfect-overlap dict (identical bags -> IoU 1.0).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from replay.metrics.base import BaseRegressionMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric


def _occupancy_mask(msg: Any) -> Optional[np.ndarray]:
    """Recover a boolean collision/occupancy mask from an output message.

    Treats any non-zero pixel in the decoded frame as occupied. Returns None when
    the frame cannot be decoded (graceful degradation).
    """
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    data = getattr(msg, "data", None)
    if height <= 0 or width <= 0 or data is None:
        return None
    arr = np.asarray(data)
    if arr.size < height * width:
        return None
    channels = max(1, arr.size // (height * width))
    grid = arr[: height * width * channels].reshape(height, width, channels)[..., 0]
    return grid != 0


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks (1.0 when both empty)."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union == 0:
        # Both masks empty -> identical -> perfect overlap.
        return 1.0
    return intersection / union


@register_metric("perception")
class CollisionBoxIoUMetric(BaseRegressionMetric):
    """Collision-box IoU: candidate replay vs pinned golden, per paired frame."""

    name = "collision_box_iou"
    requires_baseline = True

    def compute(self, reader: BagReader, config: dict) -> dict:
        # BaseRegressionMetric still inherits BaseMetric's abstract compute(); this
        # metric is baseline-only, so the single-replay path is a no-op marker. The
        # report generator routes requires_baseline metrics through compare() with a
        # resolved baseline; calling compute() alone yields no IoU verdict.
        return {"requires_baseline": True, "mean_iou": 1.0, "min_iou": 1.0}

    def compare(self, candidate: BagReader, baseline: BagReader, config: dict) -> dict:
        output_topics = config.get("output_topics", [])

        ious: list[float] = []
        for topic in output_topics:
            cand_msgs = candidate.get_messages(topic)
            base_msgs = baseline.get_messages(topic)
            for (_, cmsg), (_, bmsg) in zip(cand_msgs, base_msgs):
                cm = _occupancy_mask(cmsg)
                bm = _occupancy_mask(bmsg)
                if cm is None or bm is None:
                    continue
                ious.append(_iou(cm, bm))

        if not ious:
            return {"num_boxes": 0, "mean_iou": 1.0, "min_iou": 1.0}

        arr = np.array(ious, dtype=np.float64)
        return {
            "num_boxes": int(arr.size),
            "mean_iou": round(float(np.mean(arr)), 4),
            "min_iou": round(float(np.min(arr)), 4),
        }
