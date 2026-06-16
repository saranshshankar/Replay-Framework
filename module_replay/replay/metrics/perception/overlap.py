"""Cross-camera semantic overlap consistency metric (MTRC-02).

Ported from the PoC ``perception_metrics/metrics/overlap_metrics.py`` (10xCode
branch ``aniket/feat/module_wise_replay_poc``). ``cross_camera_overlap_iou``
measures pixel-wise SEMANTIC-LABEL AGREEMENT in the true warped overlap region on
a **[0, 1] scale**, so the 0.75 quality gate is meaningful (UAT gap 3: the prior
AKAZE port had gutted this into an unnormalized RANSAC-inlier proxy of ~0.01-0.25).

Two-phase calibrate-then-agree pipeline (the PoC's), per explicit adjacency pair
(``OVERLAP_PAIRS``; NOT a flat consecutive-output-topic zip):

  Phase 1 (calibrate): join the pair's ``image_raw_sim`` RGB frames by
  ``header.stamp``, sample up to ``CALIBRATION_SAMPLES`` frame pairs, AKAZE/ORB
  feature-match each, and keep the best-inlier RANSAC homography H (RGB_a -> RGB_b).
  Scale H to the semantic resolution (``_scale_homography_to_semantic``) and
  compute the overlap mask (``_compute_overlap_mask``). Skip a pair whose
  calibration fails or whose overlap < ``MIN_OVERLAP_PIXELS``.

  Phase 2 (evaluate): join the pair's ``semantic_raw_sim`` frames by
  ``header.stamp``; for each pair, warp the class-id plane (the **R channel** of
  the rgba8 frame, decoded via ``_decode_classid_plane`` — NOT a grayscale of all
  channels) of camera A into B and measure the [0, 1] label agreement in the
  overlap mask (``_compute_semantic_agreement``).

The headline scalar ``cross_camera_overlap_iou`` (== ``self.name``) is the mean of
the per-pair mean agreements, so ``generator.py`` enforces the 0.75 min threshold.

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
MIN_OVERLAP_PIXELS = 50
STAMP_TOLERANCE_NS = 100_000_000  # 100 ms — the PoC searchsorted join tolerance

# Explicit per-camera adjacency (the PoC's OVERLAP_PAIRS) — NOT a flat consecutive
# zip of output topics. (cam_a, cam_b, description); only pairs whose BOTH cameras
# are present in the module's output topics are evaluated.
OVERLAP_PAIRS = [
    (2, 3, "Front-Top <-> Front-Bottom"),
    (2, 4, "Front-Top <-> Left"),
    (2, 5, "Front-Top <-> Right"),
    (3, 4, "Front-Bottom <-> Left"),
    (3, 5, "Front-Bottom <-> Right"),
    (0, 4, "Rear <-> Left"),
    (0, 5, "Rear <-> Right"),
]


def _to_gray(msg: Any) -> Optional[np.ndarray]:
    """Decode an Image-like message to a single-channel uint8 grayscale array.

    Used ONLY for the RGB homography input (feature matching). For the semantic
    class-id plane use ``_decode_classid_plane`` instead — grayscaling an rgba8
    semantic frame destroys the class ids (UAT gap 3).
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
        return arr.reshape(height, width)
    channels = arr.size // (height * width)
    img = arr[: height * width * channels].reshape(height, width, channels)
    if channels >= 3:
        return cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    return img[..., 0]


def _decode_classid_plane(msg: Any) -> Optional[np.ndarray]:
    """Decode a semantic Image's R channel as the class-id plane (int16).

    The replay's ``semantic_raw_sim`` frames are ``rgba8`` with the class id in
    the **R channel** (perception contract §2). Decoding via a grayscale of all
    channels (as the gutted port did) mangles the labels — this returns the raw
    R-channel ids so pixel-wise label agreement is meaningful. A single-channel
    frame is treated as the class-id plane directly.
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


def _join_by_stamp(
    msgs_a: list[tuple[int, Any]],
    msgs_b: list[tuple[int, Any]],
    tolerance_ns: int = STAMP_TOLERANCE_NS,
) -> list[tuple[Any, Any]]:
    """Pair messages from two topics by header.stamp (the PoC searchsorted join).

    Falls back to the bag-write timestamp when a message carries no header stamp,
    so the join still works on the degenerate synthetic fixture. Returns a list of
    (msg_a, msg_b) pairs within ``tolerance_ns``.
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


def _compute_homography(
    pts_a: np.ndarray, pts_b: np.ndarray
) -> tuple[Optional[np.ndarray], int]:
    """Homography from matched points via RANSAC (ported from the PoC).

    Returns (H, num_inliers); H maps points from image A to image B. Returns
    (None, 0) when there are too few matches or RANSAC fails.
    """
    if len(pts_a) < MIN_MATCHES_FOR_HOMOGRAPHY:
        return None, 0
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return None, 0
    num_inliers = int(mask.sum())
    if num_inliers < MIN_MATCHES_FOR_HOMOGRAPHY:
        return None, 0
    return H, num_inliers


def _scale_homography_to_semantic(
    H: np.ndarray,
    rgb_wh: tuple[int, int],
    sem_wh: tuple[int, int],
) -> np.ndarray:
    """Scale a homography from RGB resolution to semantic resolution.

    Ported from the PoC but parameterized by the ACTUAL decoded resolutions
    (the PoC hardcoded 640x480 RGB / 112x112 sem; our V2 sim semantic is 448x448
    per contract §2, so the scale must come from the real frame dimensions).

    ``H`` maps RGB_a -> RGB_b. ``H_sem = S @ H @ S_inv`` maps SEM_a -> SEM_b with
    ``S = diag(sx, sy, 1)``, ``sx = sem_w/rgb_w``, ``sy = sem_h/rgb_h``.
    """
    rgb_w, rgb_h = rgb_wh
    sem_w, sem_h = sem_wh
    sx = sem_w / rgb_w
    sy = sem_h / rgb_h
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    S_inv = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float64)
    return S @ H @ S_inv


def _compute_overlap_mask(
    H_sem: np.ndarray,
    sem_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Overlap masks for both cameras at semantic resolution (ported from PoC).

    Args:
        H_sem: 3x3 homography mapping semantic_a -> semantic_b.
        sem_wh: (width, height) of the semantic frame.

    Returns (mask_a, mask_b) boolean arrays of shape (height, width):
        mask_b: B-pixels that RECEIVE a mapped A-pixel (warp ones_a -> B).
        mask_a: A-pixels that map to valid B-pixels (warp ones_b -> A via H_inv).
    """
    sem_w, sem_h = sem_wh
    ones = np.ones((sem_h, sem_w), dtype=np.float32)
    warped_a_in_b = cv2.warpPerspective(
        ones, H_sem, (sem_w, sem_h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    mask_b = warped_a_in_b > 0.5

    try:
        H_inv = np.linalg.inv(H_sem)
    except np.linalg.LinAlgError:
        # A singular H_sem has no inverse; A-side overlap is then undefined.
        return np.zeros((sem_h, sem_w), dtype=bool), mask_b
    warped_b_in_a = cv2.warpPerspective(
        ones, H_inv, (sem_w, sem_h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    mask_a = warped_b_in_a > 0.5
    return mask_a, mask_b


def _compute_semantic_agreement(
    sem_a: np.ndarray,
    sem_b: np.ndarray,
    H_sem: np.ndarray,
    mask_b: np.ndarray,
) -> Optional[float]:
    """Pixel-wise semantic-label agreement in the overlap region (ported from PoC).

    Warps the class-id plane ``sem_a`` into B's frame with ``H_sem`` (INTER_NEAREST,
    borderValue=-1 marks pixels with no source), then returns the fraction of
    matching class ids over ``mask_b & (warped >= 0)``.

    Returns a value in [0, 1], or ``None`` when the realised overlap is below
    ``MIN_OVERLAP_PIXELS`` (so an empty/degenerate overlap never yields a number).
    """
    sem_h, sem_w = mask_b.shape
    warped_a = cv2.warpPerspective(
        sem_a.astype(np.float32),
        H_sem,
        (sem_w, sem_h),
        flags=cv2.INTER_NEAREST,
        borderValue=-1,
    ).astype(np.int16)
    overlap = mask_b & (warped_a >= 0)
    n_pixels = int(overlap.sum())
    if n_pixels < MIN_OVERLAP_PIXELS:
        return None
    matching = int(np.sum(warped_a[overlap] == sem_b.astype(np.int16)[overlap]))
    return float(matching) / n_pixels


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


def _camera_topic_map(output_topics: list[str], suffix: str) -> dict[int, str]:
    """Map camera index -> output topic by the substring ``camera_{N}/{suffix}``."""
    mapping: dict[int, str] = {}
    for topic in output_topics:
        for cam in range(8):  # the rig has cams 0..5; scan a small safe range
            if f"camera_{cam}/{suffix}" in topic:
                mapping[cam] = topic
                break
    return mapping


@register_metric("perception")
class OverlapMetric(BaseMetric):
    """Cross-camera semantic-label agreement [0,1] via CPU feature matching (AKAZE default).

    Calibrate-then-agree, per explicit adjacency pair (see module docstring). The
    headline ``cross_camera_overlap_iou`` is the mean of per-pair mean agreements,
    so ``generator.py`` enforces the 0.75 quality gate (SC2). Degrades gracefully
    (returns ``0.0``, never raises) when no pair calibrates — e.g. the tiny 2x2
    synthetic frames that yield no AKAZE keypoints.
    """

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

        output_topics = config.get("output_topics", [])
        rgb_topic = _camera_topic_map(output_topics, "image_raw_sim")
        sem_topic = _camera_topic_map(output_topics, "semantic_raw_sim")

        # Only adjacency pairs whose BOTH cameras have an RGB topic are evaluable.
        # When the config carries no camera-shaped topics at all (e.g. the tiny
        # synthetic_bag with one bare image_raw_sim), fall back to pairing the
        # first two output topics so graceful-degradation still exercises the path.
        pairs: list[tuple[int | str, int | str, str]] = [
            (a, b, desc) for a, b, desc in OVERLAP_PAIRS if a in rgb_topic and b in rgb_topic
        ]
        if not pairs and len(output_topics) >= 2:
            pairs = [("_0", "_1", "fallback")]
            rgb_topic = {"_0": output_topics[0], "_1": output_topics[1]}
            sem_topic = {}

        per_pair: dict[str, dict] = {}
        agreements_mean: list[float] = []

        for cam_a, cam_b, description in pairs:
            # ── Phase 1: calibrate a homography on the RGB pair ──────────────
            rgb_pairs = _join_by_stamp(
                reader.get_messages(rgb_topic[cam_a]),
                reader.get_messages(rgb_topic[cam_b]),
            )
            if not rgb_pairs:
                continue
            step = max(1, len(rgb_pairs) // CALIBRATION_SAMPLES)
            samples = rgb_pairs[:: step][:CALIBRATION_SAMPLES]

            best_H, best_inliers = None, 0
            rgb_wh: tuple[int, int] | None = None
            for msg_a, msg_b in samples:
                gray_a, gray_b = _to_gray(msg_a), _to_gray(msg_b)
                if gray_a is None or gray_b is None:
                    continue
                pts_a, pts_b = match_fn(gray_a, gray_b)
                H, inliers = _compute_homography(pts_a, pts_b)
                if H is not None and inliers > best_inliers:
                    best_H, best_inliers = H, inliers
                    rgb_wh = (gray_a.shape[1], gray_a.shape[0])  # (w, h)
            if best_H is None or rgb_wh is None:
                continue

            # ── Phase 2: per-stamp semantic-label agreement in the overlap ───
            sem_a_topic = sem_topic.get(cam_a)
            sem_b_topic = sem_topic.get(cam_b)
            if not sem_a_topic or not sem_b_topic:
                continue
            sem_pairs = _join_by_stamp(
                reader.get_messages(sem_a_topic),
                reader.get_messages(sem_b_topic),
            )
            if not sem_pairs:
                continue

            H_sem = None
            mask_b = None
            agree_vals: list[float] = []
            for msg_a, msg_b in sem_pairs:
                sem_a = _decode_classid_plane(msg_a)
                sem_b = _decode_classid_plane(msg_b)
                if sem_a is None or sem_b is None or sem_a.shape != sem_b.shape:
                    continue
                if H_sem is None:
                    sem_wh = (sem_a.shape[1], sem_a.shape[0])  # (w, h)
                    H_sem = _scale_homography_to_semantic(best_H, rgb_wh, sem_wh)
                    _, mask_b = _compute_overlap_mask(H_sem, sem_wh)
                    if int(mask_b.sum()) < MIN_OVERLAP_PIXELS:
                        break  # overlap too small -> skip this pair (PoC guard)
                agreement = _compute_semantic_agreement(sem_a, sem_b, H_sem, mask_b)
                if agreement is not None:
                    agree_vals.append(agreement)

            if not agree_vals:
                continue
            mean_agreement = float(np.mean(agree_vals))
            agreements_mean.append(mean_agreement)
            per_pair[f"cam{cam_a}_cam{cam_b}"] = {
                "description": description,
                "num_frames": len(agree_vals),
                "calibration_inliers": best_inliers,
                "mean_agreement": round(mean_agreement, 4),
                "min_agreement": round(float(np.min(agree_vals)), 4),
                "overlap_pixels": int(mask_b.sum()) if mask_b is not None else 0,
            }

        return {
            # headline scalar (== self.name) in [0,1] = mean of per-pair means, so
            # generator.py enforces the 0.75 min quality gate (SC2).
            "cross_camera_overlap_iou": (
                round(float(np.mean(agreements_mean)), 4) if agreements_mean else 0.0
            ),
            "num_pairs": len(agreements_mean),
            "feature_matcher": matcher,
            "pairs": per_pair,
        }
