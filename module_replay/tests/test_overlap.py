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
