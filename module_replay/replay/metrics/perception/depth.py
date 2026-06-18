"""Perception depth-validity metric (MTRC-02).

Ported from the PoC ``perception_metrics/metrics/depth_metrics.py`` (fraction of
valid/finite depth pixels + depth stats), re-implemented against the framework's
read-once ``BagReader``.

VERIFIED PERCEPTION FACT (KT/playbooks/01-perception.md, 2026-06-11): in *replay*
the depth output is ``/perception_node/camera_N/depth_raw_sim`` (32FC4,
lidar-interpolated — there is no monocular depth model in sim). The PoC's
``/depth_undistort/camera_N/depth`` topics DO NOT EXIST in replay output bags, so
this metric reads whatever depth topics are passed in ``config["depth_topics"]``
(falling back to the output topics) and its values are only loosely comparable to
live depth — gate replay-vs-replay, not replay-vs-live.

GRACEFUL DEGRADATION (per plan): on degenerate synthetic frames (no decodable
depth) ``compute`` returns a typed zero-shaped dict instead of raising.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from replay.metrics.base import BaseMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric


def _is_depth_encoding(encoding: str) -> bool:
    """True for float/uint depth encodings (32FC*, 16UC*, mono16) — NOT rgb/rgba.

    The gap-5 fix: when ``depth_topics`` is not configured, the metric must
    self-filter by encoding so it never decodes an rgb8/rgba8 colour frame as
    float32 depth (which produced garbage depth stats over the wrong topics).
    """
    enc = str(encoding or "")
    return enc.startswith("32FC") or enc.startswith("16UC") or enc == "mono16"


def _decode_depth(msg: Any) -> Optional[np.ndarray]:
    """Decode a depth Image-like message to a flat float array of depth values.

    Handles the replay depth encoding (32FC*) and the legacy 16UC1, returning the
    first channel as a 2D grid. Returns None when it cannot interpret the buffer
    (graceful degradation) rather than raising.
    """
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    data = getattr(msg, "data", None)
    encoding = str(getattr(msg, "encoding", "") or "")
    if height <= 0 or width <= 0 or data is None:
        return None
    raw = np.asarray(data)
    if raw.size == 0:
        return None
    try:
        if encoding.startswith("32FC"):
            vals = raw.view(np.float32) if raw.dtype == np.uint8 else raw.astype(np.float32)
        elif encoding.startswith("16UC") or encoding == "mono16":
            vals = raw.view(np.uint16) if raw.dtype == np.uint8 else raw.astype(np.uint16)
        else:
            vals = raw.astype(np.float32)
    except (ValueError, TypeError):
        return None
    if vals.size < height * width:
        return None
    channels = max(1, vals.size // (height * width))
    grid = vals[: height * width * channels].reshape(height, width, channels)
    return grid[..., 0].astype(np.float64)


@register_metric("perception")
class DepthMetric(BaseMetric):
    """Fraction of valid (finite, > 0) depth pixels + depth value stats."""

    name = "depth_validity"
    requires_baseline = False

    def compute(self, reader: BagReader, config: dict) -> dict:
        # Gap 5: scope to the configured depth topics; if none are configured fall
        # back to output topics whose name carries the depth_raw_sim substring, and
        # only if THAT is also empty do we scan all output topics — but in that last
        # case self-filter each frame by depth encoding (32FC*/16UC*) so an rgb8 /
        # rgba8 colour frame is never decoded as float32 depth.
        depth_topics = config.get("depth_topics")
        output_topics = config.get("output_topics", [])
        if depth_topics:
            topics = depth_topics
            enforce_encoding = False
        else:
            named = [t for t in output_topics if "depth_raw_sim" in t]
            topics = named or output_topics
            # When we had to fall back to all output topics, the topic name no
            # longer guarantees depth — filter by encoding inside the loop.
            enforce_encoding = not named

        valid_fractions: list[float] = []
        means: list[float] = []
        num_frames = 0

        for topic in topics:
            for _ts, msg in reader.get_messages(topic):
                if enforce_encoding and not _is_depth_encoding(getattr(msg, "encoding", "")):
                    continue
                depth = _decode_depth(msg)
                if depth is None:
                    continue
                num_frames += 1
                finite = np.isfinite(depth)
                valid_mask = finite & (depth > 0)
                valid_fractions.append(float(np.sum(valid_mask)) / depth.size)
                if np.any(valid_mask):
                    means.append(float(np.mean(depth[valid_mask])))

        mean_valid_fraction = round(float(np.mean(valid_fractions)), 4) if valid_fractions else 0.0
        return {
            # BUG2: headline scalar must live under a key == self.name so the
            # generator's value[r.name] lookup (generator.py:224) finds it. Keep
            # mean_valid_fraction (the existing detail key); depth_validity is the
            # same value, additive.
            "depth_validity": mean_valid_fraction,
            "num_frames": num_frames,
            "mean_valid_fraction": mean_valid_fraction,
            "min_valid_fraction": round(float(np.min(valid_fractions)), 4) if valid_fractions else 0.0,
            "mean_depth": round(float(np.mean(means)), 3) if means else 0.0,
        }
