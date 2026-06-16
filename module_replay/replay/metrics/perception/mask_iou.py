"""Per-frame semantic-mask IoU vs a pinned golden — perception's C1 regression
metric (MTRC-02, contract §5).

A ``BaseRegressionMetric``: this is the PRIMARY regression signal the perception
contract specifies (``mask_iou_vs_golden``, min **0.98**). It compares the
candidate replay's ``semantic_raw_sim`` masks against a pinned-golden replay run,
frame-for-frame, and reports the per-class semantic IoU averaged over the matched
frames. It is the ONE regression metric perception carries — the prior
action-block / collision-box metrics targeted out-of-scope 3D/service outputs and
were dropped (01-15 / UAT decision_locked).

WHY 0.98 / TOLERANCE-BASED, NEVER BIT-EXACT (contract §5; RESEARCH Pitfall 4 /
threat T-05-03): FP16 TensorRT inference with kernel autotuning is nondeterministic,
so two replays of the SAME code differ at object boundaries. Boundary-pixel flips
are noise; whole-region changes are real regressions. The 0.98 floor (applied by
``generator.py`` via the top-level ``mask_iou_vs_golden`` scalar) absorbs the noise
while still catching a region change. We never assert byte-equality.

SEMANTIC DECODE (contract §2): ``semantic_raw_sim`` frames are ``rgba8`` with the
class id in the **R channel** (``_decode_classid_plane``); grayscaling all channels
would mangle the labels. The per-frame IoU is the mean over the classes present in
either frame of ``intersection / union`` of the boolean ``(mask == c)`` regions —
the standard semantic mean-IoU — so a shifted region (not just a boundary flip)
drives the score down.

JOIN RULE (the BagReader/PoC searchsorted join): candidate and baseline frames pair
by ``header.stamp`` within a 100 ms tolerance (outputs carry the input capture
stamp, so a golden replay of the same bag pairs frame-for-frame). The bag-write
timestamp is the fallback when a frame carries no header stamp.

FAIL-CLOSED (UAT gap 2): if ZERO comparable frames are produced (an empty,
missing, or mis-aligned baseline), the top-level scalar is ``None`` — NOT a false
1.0. A ``None`` scalar makes ``generator.py`` emit a visible no-scalar row (never a
silent pass), so a broken baseline can never green-light the gate.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from replay.metrics.base import BaseRegressionMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric

STAMP_TOLERANCE_NS = 100_000_000  # 100 ms — the PoC searchsorted join tolerance


def _stamp_ns(msg: Any) -> Optional[int]:
    """Extract header.stamp (sec/nanosec) as an absolute nanosecond timestamp."""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


def _decode_classid_plane(msg: Any) -> Optional[np.ndarray]:
    """Decode a semantic Image's R channel as the class-id plane (int16).

    ``semantic_raw_sim`` is ``rgba8`` with the class id in the R channel (contract
    §2). A single-channel frame is treated as the class-id plane directly. Returns
    None when the frame cannot be decoded (graceful degradation, never raises).
    """
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    data = getattr(msg, "data", None)
    if height <= 0 or width <= 0 or data is None:
        return None
    arr = np.asarray(data, dtype=np.uint8)
    if arr.size < height * width:
        return None
    if arr.size == height * width:
        return arr.reshape(height, width).astype(np.int16)
    channels = arr.size // (height * width)
    img = arr[: height * width * channels].reshape(height, width, channels)
    return img[..., 0].astype(np.int16)  # R channel = class id (rgba8 / rgb8)


def _join_by_stamp(
    msgs_a: list[tuple[int, Any]],
    msgs_b: list[tuple[int, Any]],
    tolerance_ns: int = STAMP_TOLERANCE_NS,
) -> list[tuple[Any, Any]]:
    """Pair messages from two topics by header.stamp (the PoC searchsorted join).

    Falls back to the bag-write timestamp when a message carries no header stamp.
    Returns a list of (msg_a, msg_b) pairs within ``tolerance_ns``.
    """
    keyed_a = [(_stamp_ns(m) if _stamp_ns(m) is not None else t, m) for t, m in msgs_a]
    keyed_b = [(_stamp_ns(m) if _stamp_ns(m) is not None else t, m) for t, m in msgs_b]
    if not keyed_a or not keyed_b:
        return []
    keyed_b.sort(key=lambda kv: kv[0])
    ts_b = np.array([k for k, _ in keyed_b], dtype=np.int64)
    pairs: list[tuple[Any, Any]] = []
    for ts_a, msg_a in keyed_a:
        idx = int(np.searchsorted(ts_b, ts_a))
        best_ci, best_diff = None, tolerance_ns + 1
        for ci in (idx - 1, idx):
            if 0 <= ci < len(ts_b):
                diff = abs(int(ts_b[ci]) - int(ts_a))
                if diff < best_diff:
                    best_diff, best_ci = diff, ci
        if best_ci is not None and best_diff <= tolerance_ns:
            pairs.append((msg_a, keyed_b[best_ci][1]))
    return pairs


def _frame_mean_iou(cand: np.ndarray, base: np.ndarray) -> Optional[float]:
    """Mean semantic IoU over the classes present in either frame.

    For each class id ``c`` appearing in candidate OR baseline, IoU =
    ``|cand==c AND base==c| / |cand==c OR base==c|``; the frame score is the mean
    over those classes (standard semantic mean-IoU). Returns None on a shape
    mismatch (cannot compare) so the caller can fail closed rather than fabricate.
    """
    if cand.shape != base.shape:
        return None
    classes = np.unique(np.concatenate([cand.reshape(-1), base.reshape(-1)]))
    ious: list[float] = []
    for c in classes:
        cand_c = cand == c
        base_c = base == c
        union = int(np.logical_or(cand_c, base_c).sum())
        if union == 0:
            continue  # class present in neither -> not a contributing region
        inter = int(np.logical_and(cand_c, base_c).sum())
        ious.append(inter / union)
    if not ious:
        return None
    return float(np.mean(ious))


@register_metric("perception")
class MaskIoUVsGoldenMetric(BaseRegressionMetric):
    """Per-frame semantic-mask IoU, candidate replay vs pinned golden (min 0.98)."""

    name = "mask_iou_vs_golden"
    requires_baseline = True

    def compute(self, reader: BagReader, config: dict) -> dict:
        # BaseRegressionMetric inherits BaseMetric's abstract compute(); this metric
        # is baseline-only, so the single-replay path is a no-op marker. The report
        # generator / cli routes requires_baseline metrics through compare() with a
        # resolved baseline; compute() alone yields no IoU verdict.
        return {"requires_baseline": True, "mask_iou_vs_golden": None}

    def compare(self, candidate: BagReader, baseline: BagReader, config: dict) -> dict:
        # Scan the semantic-mask output topics (the contract's C1 comparison
        # surface). When the config names no semantic topic (e.g. the degenerate
        # synthetic fixture) fall back to the candidate's actual topics.
        output_topics = config.get("output_topics", [])
        semantic_topics = [t for t in output_topics if "semantic_raw_sim" in t]
        topics = semantic_topics or [
            t for t in candidate.topics() if "semantic_raw_sim" in t
        ]

        per_topic: dict[str, dict] = {}
        frame_ious: list[float] = []
        for topic in topics:
            pairs = _join_by_stamp(
                candidate.get_messages(topic),
                baseline.get_messages(topic),
            )
            topic_ious: list[float] = []
            for cmsg, bmsg in pairs:
                cand = _decode_classid_plane(cmsg)
                base = _decode_classid_plane(bmsg)
                if cand is None or base is None:
                    continue
                iou = _frame_mean_iou(cand, base)
                if iou is None:
                    continue
                topic_ious.append(iou)
            if topic_ious:
                frame_ious.extend(topic_ious)
                per_topic[topic] = {
                    "num_frames": len(topic_ious),
                    "mean_iou": round(float(np.mean(topic_ious)), 4),
                    "min_iou": round(float(np.min(topic_ious)), 4),
                }

        if not frame_ious:
            # FAIL-CLOSED: zero comparable frames (empty / missing / mis-aligned
            # baseline) must NOT report a passing scalar. A None scalar makes
            # generator.py record a visible no-scalar row, never a silent pass.
            return {
                "mask_iou_vs_golden": None,
                "num_frames": 0,
                "min_iou": None,
                "per_topic": per_topic,
                "note": (
                    "no comparable semantic frames between candidate and baseline "
                    "— failing closed (not a passing scalar)"
                ),
            }

        arr = np.array(frame_ious, dtype=np.float64)
        return {
            # Headline scalar (== self.name) in [0,1] = mean per-frame IoU, so
            # generator.py enforces the 0.98 min threshold when a baseline IS present.
            "mask_iou_vs_golden": round(float(np.mean(arr)), 4),
            "num_frames": int(arr.size),
            "min_iou": round(float(np.min(arr)), 4),
            "per_topic": per_topic,
        }
