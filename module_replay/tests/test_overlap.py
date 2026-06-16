"""OverlapMetric tests (MTRC-02 / MTRC-03 / MOD-01) — gap-closure plan 01-14.

This file OWNS the cross_camera_overlap_iou tests. It exists separately from
``test_metrics_perception.py`` (whose ``test_overlap_defaults_to_akaze`` is left
untouched by this plan — 01-15 reconciles that shared file) so this same-wave
plan does not collide with 01-13's edits there.

The metric was GUTTED in the original AKAZE port (UAT gap 3, blocker): it reported
unnormalized RANSAC inliers / frame_count (~0.01-0.25) instead of the PoC's
pixel-wise semantic-label agreement in the true warped overlap region. These tests
pin the RESTORED behaviour:

  * the geometry/agreement helpers compute on a [0,1] scale (Task 1), and
  * ``OverlapMetric.compute`` returns a top-level scalar ``cross_camera_overlap_iou``
    in [0,1] via the calibrate-then-agree pipeline so the 0.75 gate is meaningful
    again (Task 2).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from replay.metrics.perception.overlap import (
    _compute_overlap_mask,
    _compute_semantic_agreement,
    _scale_homography_to_semantic,
)


# ── Task 1: ported geometry + agreement helpers ────────────────────────────


def test_scale_homography_identity_equal_res():
    """Identity H at equal RGB/SEM resolution scales to identity (S @ I @ S_inv = I)."""
    H = np.eye(3, dtype=np.float64)
    H_sem = _scale_homography_to_semantic(H, rgb_wh=(640, 480), sem_wh=(640, 480))
    assert H_sem.shape == (3, 3)
    np.testing.assert_allclose(H_sem, np.eye(3), atol=1e-9)


def test_scale_homography_translation_differing_res():
    """A pure RGB-pixel translation scales by sx/sy into semantic-pixel units.

    H_sem = S @ H @ S_inv with S = diag(sx, sy, 1); a translation (tx, ty) in RGB
    pixels becomes (tx*sx, ty*sy) in semantic pixels. With sem 112 / rgb 640|480
    (sx = 112/640, sy = 112/480), a +40px x-translation -> +7px, +48px y -> +11.2px.
    """
    tx, ty = 40.0, 48.0
    H = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    H_sem = _scale_homography_to_semantic(H, rgb_wh=(640, 480), sem_wh=(112, 112))
    assert H_sem.shape == (3, 3)
    sx, sy = 112 / 640, 112 / 480
    # rotation/scale block unchanged (identity), translation scaled per-axis.
    np.testing.assert_allclose(H_sem[:2, :2], np.eye(2), atol=1e-9)
    assert H_sem[0, 2] == pytest.approx(tx * sx)
    assert H_sem[1, 2] == pytest.approx(ty * sy)


def test_overlap_mask_identity_is_full():
    """Identity H_sem -> both overlap masks are fully True (every pixel overlaps)."""
    sem_wh = (112, 112)
    mask_a, mask_b = _compute_overlap_mask(np.eye(3, dtype=np.float64), sem_wh)
    assert mask_a.dtype == bool and mask_b.dtype == bool
    assert mask_a.shape == (112, 112) and mask_b.shape == (112, 112)
    assert mask_a.all() and mask_b.all()


def test_overlap_mask_translation_is_partial():
    """A translation that pushes content off-frame leaves mask_b strictly smaller."""
    sem_wh = (112, 112)
    H_sem = np.array([[1, 0, 60], [0, 1, 60], [0, 0, 1]], dtype=np.float64)
    _, mask_b = _compute_overlap_mask(H_sem, sem_wh)
    assert mask_b.dtype == bool
    assert 0 < int(mask_b.sum()) < mask_b.size  # partial overlap, not full, not empty


def test_semantic_agreement_identical_is_one():
    """Identical class-id masks, identity H_sem, full overlap -> agreement == 1.0."""
    sem = (np.arange(112 * 112, dtype=np.int16) % 6).reshape(112, 112)
    mask_b = np.ones((112, 112), dtype=bool)
    agree = _compute_semantic_agreement(sem.copy(), sem.copy(), np.eye(3), mask_b)
    assert agree is not None
    assert 0.0 <= agree <= 1.0
    assert agree == pytest.approx(1.0)


def test_semantic_agreement_half_differ_is_half():
    """When half the overlap pixels disagree, agreement is ~0.5 (in [0,1])."""
    sem_a = np.zeros((112, 112), dtype=np.int16)
    sem_b = np.zeros((112, 112), dtype=np.int16)
    sem_b[:, 56:] = 1  # right half differs -> exactly half disagree
    mask_b = np.ones((112, 112), dtype=bool)
    agree = _compute_semantic_agreement(sem_a, sem_b, np.eye(3), mask_b)
    assert agree is not None
    assert agree == pytest.approx(0.5, abs=0.02)


def test_semantic_agreement_empty_overlap_is_none():
    """An empty overlap region (mask all-False) returns None, never raises/NaN."""
    sem = np.zeros((112, 112), dtype=np.int16)
    mask_b = np.zeros((112, 112), dtype=bool)
    assert _compute_semantic_agreement(sem, sem, np.eye(3), mask_b) is None


# ── Task 2: OverlapMetric.compute calibrate-then-agree pipeline ─────────────

# Adjacency pair (2, 3) is in the PoC OVERLAP_PAIRS; build a bag with those two
# cameras' image_raw_sim (textured RGB so AKAZE matches) + semantic_raw_sim
# (rgba8, R = class id) so the metric can calibrate a homography and then measure
# pixel-wise label agreement in the true overlap region.
_CAM_A, _CAM_B = 2, 3
RGB_A = f"/perception_node/camera_{_CAM_A}/image_raw_sim"
RGB_B = f"/perception_node/camera_{_CAM_B}/image_raw_sim"
SEM_A = f"/perception_node/camera_{_CAM_A}/semantic_raw_sim"
SEM_B = f"/perception_node/camera_{_CAM_B}/semantic_raw_sim"
OVERLAP_TOPICS = [RGB_A, RGB_B, SEM_A, SEM_B]


def _textured_rgb(side: int = 160, seed: int = 7) -> np.ndarray:
    """A feature-rich HxWx3 uint8 RGB frame (AKAZE finds plenty of stable corners).

    Pure i.i.d. noise yields too few repeatable AKAZE keypoints (< MIN_MATCHES);
    scattered rectangles + circles + light noise give strong, repeatable corners so
    matching the SAME scene against itself fits a near-identity homography with many
    inliers — the calibration the metric needs.
    """
    import cv2

    rng = np.random.default_rng(seed)
    img = np.zeros((side, side, 3), dtype=np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, side - 20, 2)
        w, h = rng.integers(6, 20, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      rng.integers(60, 256, 3).tolist(), -1)
    for _ in range(40):
        c = rng.integers(8, side - 8, 2)
        cv2.circle(img, (int(c[0]), int(c[1])), int(rng.integers(3, 9)),
                   rng.integers(60, 256, 3).tolist(), -1)
    return cv2.add(img, rng.integers(0, 40, (side, side, 3)).astype(np.uint8))


def _classid_rgba(classid_plane: np.ndarray) -> np.ndarray:
    """Pack a HxW int class-id plane into an rgba8 HxWx4 uint8 frame (R = class id)."""
    h, w = classid_plane.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = classid_plane.astype(np.uint8)  # R channel carries the class id
    rgba[..., 3] = 255
    return rgba


def _write_overlap_bag(
    bag_dir: Path,
    sem_a_plane: np.ndarray,
    sem_b_plane: np.ndarray,
    rgb_side: int = 160,
    n_frames: int = 6,
) -> Path:
    """Write a rosbag2 with two adjacent cameras' RGB (identical textured scene)
    + their semantic class-id planes, joined by header.stamp.

    Both cameras observe the SAME textured RGB scene so AKAZE/RANSAC fits a near-
    identity homography (full overlap). The semantic agreement is then driven
    entirely by how similar ``sem_a_plane`` and ``sem_b_plane`` are.
    """
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    rgb = _textured_rgb(rgb_side, seed=7)
    rgba_a = _classid_rgba(sem_a_plane)
    rgba_b = _classid_rgba(sem_b_plane)
    sem_h, sem_w = sem_a_plane.shape

    def _img(arr, encoding, ts):
        sec, nanosec = ts // 1_000_000_000, ts % 1_000_000_000
        hdr = Header(stamp=Time(sec=int(sec), nanosec=int(nanosec)), frame_id="cam")
        h, w = arr.shape[:2]
        ch = arr.shape[2] if arr.ndim == 3 else 1
        return Image(
            header=hdr, height=h, width=w, encoding=encoding, is_bigendian=0,
            step=w * ch, data=np.ascontiguousarray(arr).reshape(-1).astype(np.uint8),
        )

    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        conns = {
            RGB_A: writer.add_connection(RGB_A, Image.__msgtype__, typestore=typestore),
            RGB_B: writer.add_connection(RGB_B, Image.__msgtype__, typestore=typestore),
            SEM_A: writer.add_connection(SEM_A, Image.__msgtype__, typestore=typestore),
            SEM_B: writer.add_connection(SEM_B, Image.__msgtype__, typestore=typestore),
        }
        for i in range(n_frames):
            ts = i * 100_000_000  # 10 Hz; identical stamps across topics -> join
            writer.write(conns[RGB_A], ts, typestore.serialize_cdr(
                _img(rgb, "rgb8", ts), Image.__msgtype__))
            writer.write(conns[RGB_B], ts, typestore.serialize_cdr(
                _img(rgb, "rgb8", ts), Image.__msgtype__))
            writer.write(conns[SEM_A], ts, typestore.serialize_cdr(
                _img(rgba_a, "rgba8", ts), Image.__msgtype__))
            writer.write(conns[SEM_B], ts, typestore.serialize_cdr(
                _img(rgba_b, "rgba8", ts), Image.__msgtype__))
    return bag_dir


def _overlap_cfg() -> dict:
    return {"output_topics": OVERLAP_TOPICS}  # no feature_matcher -> akaze default


def test_compute_returns_scale_zero_to_one_for_consistent_masks(tmp_path):
    """Spatially-consistent pair -> cross_camera_overlap_iou is a [0,1] scalar
    reflecting HIGH semantic-label agreement (NOT an unnormalized inlier proxy)."""
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.perception.overlap import OverlapMetric

    sem = (np.indices((96, 96)).sum(0) % 6).astype(np.int16)  # structured class ids
    bag = _write_overlap_bag(tmp_path / "consistent", sem, sem.copy())
    reader = BagReader(bag, OVERLAP_TOPICS)
    out = OverlapMetric().compute(reader, _overlap_cfg())

    assert "cross_camera_overlap_iou" in out
    val = out["cross_camera_overlap_iou"]
    assert isinstance(val, float) and 0.0 <= val <= 1.0
    assert val > 0.75  # identical masks in the overlap region -> high agreement
    json.dumps(out)


def test_gate_enforces_high_passes_low_fails(tmp_path):
    """SC2: via generate_report with the 0.75 min threshold, a high-agreement run
    passes (exit 0) and a mismatched-class-id run fails (exit 1)."""
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.base import MetricResult
    from replay.metrics.perception.overlap import OverlapMetric
    from replay.metrics.report.generator import generate_report
    from replay.module_config import ThresholdSpec

    th = {"cross_camera_overlap_iou": ThresholdSpec(min=0.75, tolerance_band=0.03, tier="quality")}

    def _run(sem_a, sem_b, name):
        bag = _write_overlap_bag(tmp_path / name, sem_a, sem_b)
        reader = BagReader(bag, OVERLAP_TOPICS)
        val = OverlapMetric().compute(reader, _overlap_cfg())
        mr = MetricResult(
            name="cross_camera_overlap_iou", module="perception",
            value=val, passed=False, is_regression=False,
        )
        return generate_report("perception", name, [mr], tmp_path / f"{name}_rep", th)

    sem = (np.indices((96, 96)).sum(0) % 6).astype(np.int16)
    # High agreement: identical masks -> PASS (0).
    assert _run(sem, sem.copy(), "high") == 0
    # Low agreement: B's overlap class ids are all shifted -> mismatched -> FAIL (1).
    mismatched = ((sem + 1) % 6).astype(np.int16)
    assert _run(sem, mismatched, "low") == 1


def test_compute_degrades_gracefully_on_tiny_bag(synthetic_bag):
    """Graceful degradation (akaze default, no torch): the tiny 2x2 synthetic_bag
    has no AKAZE keypoints, so compute returns a typed JSON-serializable result with
    cross_camera_overlap_iou present (0.0) — never raises. (Parity with the old
    test_overlap_defaults_to_akaze intent, in this plan's owned file.)"""
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.perception.overlap import OverlapMetric

    IN = "/perception_node/camera_0/image_raw"
    OUT = "/perception_node/camera_0/image_raw_sim"
    reader = BagReader(synthetic_bag, [IN, OUT])
    # No feature_matcher key -> must default to akaze and run without torch.
    out = OverlapMetric().compute(reader, {"input_topics": [IN], "output_topics": [OUT]})
    assert "cross_camera_overlap_iou" in out
    assert out["cross_camera_overlap_iou"] == 0.0
    assert out.get("feature_matcher") == "akaze"
    json.dumps(out)
