"""Perception semantic-segmentation coverage metric (MTRC-02).

Ported from the PoC ``perception_metrics/metrics/segmentation_metrics.py``
(temporal consistency + class distribution on semantic masks). Re-implemented
against the framework's read-once ``BagReader``: decodes the output topic frames
to numpy arrays and computes per-frame class-coverage plus frame-to-frame
temporal agreement, exactly the PoC's two quantities, but topic-config driven
rather than hard-wired to ``semantic_raw``.

GRACEFUL DEGRADATION (per plan): on the tiny 2x2 synthetic frames (or any
output bag without a decodable semantic mask) ``compute`` returns a well-formed
zero-shaped dict instead of raising — the unit test gates the SHAPE +
JSON-serializability; real coverage is validated in the MOD-01 manual e2e.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from replay.metrics.base import BaseMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric

# Perception's semantic head emits 6 classes (PoC config.yaml semantic_classes).
NUM_CLASSES = 6


def _decode_mask(msg: Any) -> Optional[np.ndarray]:
    """Decode a sensor_msgs/Image-like message to a 2D class-id array.

    Returns None for encodings / shapes we cannot interpret as a single-channel
    mask (graceful degradation) rather than raising.
    """
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    data = getattr(msg, "data", None)
    if height <= 0 or width <= 0 or data is None:
        return None
    arr = np.asarray(data)
    if arr.size < height * width:
        return None
    # Take the first plane's worth of bytes as the class-id grid. For multi-channel
    # encodings (e.g. rgb8) this collapses to the red channel stride; for mono/8UC1
    # it is the mask directly. Either way it is a valid 2D grid for coverage math.
    if arr.size == height * width:
        return arr.reshape(height, width).astype(np.int16)
    channels = arr.size // (height * width)
    if channels >= 1:
        return arr[: height * width * channels].reshape(height, width, channels)[..., 0].astype(np.int16)
    return None


@register_metric("perception")
class SegmentationMetric(BaseMetric):
    """Semantic-mask class coverage + temporal consistency over output frames."""

    name = "segmentation_coverage"
    requires_baseline = False

    def compute(self, reader: BagReader, config: dict) -> dict:
        output_topics = config.get("output_topics", [])

        coverage_per_class = {f"class_{c}": [] for c in range(NUM_CLASSES)}
        agreements: list[float] = []
        num_frames = 0
        prev: Optional[np.ndarray] = None

        for topic in output_topics:
            for _ts, msg in reader.get_messages(topic):
                mask = _decode_mask(msg)
                if mask is None:
                    continue
                num_frames += 1
                total = mask.size
                for c in range(NUM_CLASSES):
                    coverage_per_class[f"class_{c}"].append(
                        float(np.sum(mask == c)) / total
                    )
                if prev is not None and prev.shape == mask.shape:
                    agreements.append(float(np.mean(mask == prev)))
                prev = mask

        mean_coverage = {
            cls: round(float(np.mean(vals)), 4) if vals else 0.0
            for cls, vals in coverage_per_class.items()
        }
        return {
            "num_frames": num_frames,
            "mean_class_coverage": mean_coverage,
            "temporal_consistency_mean": round(float(np.mean(agreements)), 4) if agreements else 0.0,
            "temporal_consistency_min": round(float(np.min(agreements)), 4) if agreements else 0.0,
            "num_consistency_pairs": len(agreements),
        }
