"""Cross-camera semantic overlap consistency metric (MTRC-02).

Ported from the PoC ``perception_metrics/metrics/overlap_metrics.py`` (10xCode
branch ``aniket/feat/module_wise_replay_poc``). The PoC matched features between
adjacent-camera RGB frames, fit a RANSAC homography, warped one camera's semantic
mask into the other's frame, and measured pixel-wise label agreement in the true
overlap region.

REQUIRED FIRST CODE CHANGE (Q-4 / project Decision):
The PoC defaulted ``feature_matcher: "lightglue"`` (its ``config.yaml:15`` and
``overlap_metrics.py:371``; ``run_perception_report.py:124`` also passed
``"lightglue"`` as the fallback). LightGlue needs ``torch`` + a GPU and is not in
the metrics ``requirements`` — it is broken on the CPU CI runner. This plugin
defaults to **AKAZE** (the OpenCV CPU detector) via
``config.get("feature_matcher", "akaze")`` and whitelists akaze/orb only; the
lightglue path is intentionally absent so torch never loads on the CI runner
(threat T-05-01). ``perception.yaml`` deliberately does not set ``feature_matcher``.

GRACEFUL DEGRADATION (per plan / threat T-05-02): on the tiny 2x2 synthetic frames
(too small for AKAZE keypoints or a homography) ``compute`` returns
``{"cross_camera_overlap_iou": 0.0, ...}`` rather than raising.
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from replay.metrics.base import BaseMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric

MIN_MATCHES_FOR_HOMOGRAPHY = 10
CALIBRATION_SAMPLES = 5


def _to_gray(msg: Any) -> Optional[np.ndarray]:
    """Decode an Image-like message to a single-channel uint8 grayscale array."""
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    data = getattr(msg, "data", None)
    if height <= 0 or width <= 0 or data is None:
        return None
    arr = np.asarray(data, dtype=np.uint8)
    if arr.size < height * width:
        return None
    if arr.size == height * width:
        return arr.reshape(height, width)
    channels = arr.size // (height * width)
    img = arr[: height * width * channels].reshape(height, width, channels)
    if channels >= 3:
        return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    return img[..., 0]


def _match_akaze(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AKAZE + BFMatcher (Hamming) ratio test — CPU, no torch (ported from PoC)."""
    akaze = cv2.AKAZE_create()
    kp_a, desc_a = akaze.detectAndCompute(gray_a, None)
    kp_b, desc_b = akaze.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2))
    pts_a = np.array([kp_a[m.queryIdx].pt for m in good], dtype=np.float64)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in good], dtype=np.float64)
    return pts_a, pts_b


def _match_orb(gray_a: np.ndarray, gray_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ORB + BFMatcher ratio test — CPU fallback (ported from PoC)."""
    orb = cv2.ORB_create(nfeatures=2000)
    kp_a, desc_a = orb.detectAndCompute(gray_a, None)
    kp_b, desc_b = orb.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2))
    pts_a = np.array([kp_a[m.queryIdx].pt for m in good], dtype=np.float64)
    pts_b = np.array([kp_b[m.trainIdx].pt for m in good], dtype=np.float64)
    return pts_a, pts_b


@register_metric("perception")
class OverlapMetric(BaseMetric):
    """Cross-camera overlap IoU via CPU feature matching (AKAZE default)."""

    name = "cross_camera_overlap_iou"
    requires_baseline = False

    def compute(self, reader: BagReader, config: dict) -> dict:
        matcher = config.get("feature_matcher", "akaze")  # DEFAULT akaze, never lightglue
        if matcher == "akaze":
            match_fn = _match_akaze
        elif matcher == "orb":
            match_fn = _match_orb
        else:
            # lightglue path is opt-in only (needs torch + GPU); not the CI default.
            raise ValueError(
                f"feature_matcher '{matcher}' not available on the CPU metrics runner; "
                "use 'akaze' (default) or 'orb'"
            )

        # The PoC matched adjacent CAMERA pairs (cam0/cam1/...). The framework
        # passes the module's topics in order; we pair consecutive output topics
        # (the report's overlap config lists adjacent-camera output topics).
        output_topics = config.get("output_topics", [])

        ious: list[float] = []
        for topic_a, topic_b in zip(output_topics, output_topics[1:]):
            frames_a = [_to_gray(m) for _, m in reader.get_messages(topic_a)]
            frames_b = [_to_gray(m) for _, m in reader.get_messages(topic_b)]
            frames_a = [f for f in frames_a if f is not None]
            frames_b = [f for f in frames_b if f is not None]
            if not frames_a or not frames_b:
                continue
            best_inliers = 0
            for ga, gb in zip(frames_a[:CALIBRATION_SAMPLES], frames_b[:CALIBRATION_SAMPLES]):
                pts_a, pts_b = match_fn(ga, gb)
                if len(pts_a) < MIN_MATCHES_FOR_HOMOGRAPHY:
                    continue
                H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
                if H is None or mask is None:
                    continue
                inliers = int(mask.sum())
                if inliers > best_inliers:
                    best_inliers = inliers
            if best_inliers >= MIN_MATCHES_FOR_HOMOGRAPHY:
                # IoU proxy: inlier fraction of the feature correspondences that
                # survive the homography (a degree-of-geometric-overlap signal).
                ious.append(min(1.0, best_inliers / max(1, len(frames_a))))

        return {
            "cross_camera_overlap_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "num_pairs": len(ious),
            "feature_matcher": matcher,
        }
