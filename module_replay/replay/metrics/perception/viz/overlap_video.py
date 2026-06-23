"""overlap_video — cross-camera overlap consistency video (V2 revamp of Aniket's PoC).

Faithful port of ``perception_metrics/visualizations/overlap_video.py`` from
``origin/aniket/feat/module_wise_replay_poc``, rewired to V2:

  * frames come from the framework ``BagReader`` (``get_messages`` + the
    ``overlap.py`` decode helpers), not the PoC ``BagReader.iter_camera_*`` API;
  * the per-pair homography / overlap masks are RECOMPUTED here from a sampled
    frame pair using ``overlap.py``'s geometry helpers (the metric does not expose
    its calibration), parameterized by the ACTUAL decoded resolutions;
  * cameras come from ``_camera_topic_map(config['output_topics'], ...)``, not PoC
    constants; the class id is the rgba8 R channel (``_decode_classid_plane``);
  * encoded with ``imageio[ffmpeg]`` (reliable mp4) instead of ``cv2.VideoWriter``.

Per camera pair, a 2x2 frame: top = both RGB side-by-side with AKAZE match lines;
bottom = each camera's semantic-overlap region alpha-blended on its RGB; header =
the per-frame semantic-agreement % + a PASS/FAIL badge; a class-color legend.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from replay.metrics.base import BaseVisualization
from replay.metrics.registry import register_viz
from replay.metrics.perception.overlap import (
    CALIBRATION_SAMPLES,
    MIN_OVERLAP_PIXELS,
    OVERLAP_PAIRS,
    STAMP_TOLERANCE_NS,
    _camera_topic_map,
    _compute_homography,
    _compute_overlap_mask,
    _compute_semantic_agreement,
    _decode_classid_plane,
    _join_by_stamp,
    _match_akaze,
    _match_orb,
    _scale_homography_to_semantic,
    _stamp_ns,
    _to_gray,
)

DEFAULT_COLOR_MAP = {
    0: (128, 128, 128),
    1: (0, 200, 0),
    2: (255, 0, 0),
    3: (0, 80, 255),
    4: (255, 220, 0),
    5: (180, 0, 255),
}
HEADER_H = 35
OVERLAY_ALPHA = 0.45
AGREEMENT_PASS = 0.75


def _pad_to_even(img: np.ndarray) -> np.ndarray:
    """libx264/yuv420p requires even width & height — pad a trailing row/col if odd."""
    h, w = img.shape[:2]
    if h % 2 or w % 2:
        img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    return img


def _ns_to_datetime_str(timestamp_ns: int) -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    dt = datetime.fromtimestamp(timestamp_ns / 1e9, tz=ist)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " IST"


def _decode_rgb(msg: Any) -> Optional[np.ndarray]:
    """Decode an Image-like message to an HxWx3 uint8 RGB array (display only)."""
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    data = getattr(msg, "data", None)
    if height <= 0 or width <= 0 or data is None:
        return None
    arr = np.asarray(data, dtype=np.uint8)
    if arr.size < height * width:
        return None
    if arr.size == height * width:  # mono -> replicate to 3 channels
        g = arr.reshape(height, width)
        return np.stack([g, g, g], axis=-1)
    channels = arr.size // (height * width)
    img = arr[: height * width * channels].reshape(height, width, channels)
    return np.ascontiguousarray(img[..., :3])


def _nearest_by_stamp(
    msgs: list[tuple[int, Any]], target_ts: int, tol: int = STAMP_TOLERANCE_NS
) -> Optional[Any]:
    best, best_diff = None, tol + 1
    for t, m in msgs:
        s = _stamp_ns(m)
        s = s if s is not None else t
        d = abs(int(s) - int(target_ts))
        if d < best_diff:
            best_diff, best = d, m
    return best if best_diff <= tol else None


def _apply_semantic_color(semantic: np.ndarray, color_map: dict) -> np.ndarray:
    h, w = semantic.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colored[semantic == class_id] = color
    return colored


def _draw_legend(img: np.ndarray, color_map: dict, y: int, x_start: int = 8) -> None:
    x = x_start
    box_w, box_h = 12, 12
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.35
    for class_id in sorted(color_map.keys()):
        cv2.rectangle(img, (x, y), (x + box_w, y + box_h), color_map[class_id], -1)
        cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (255, 255, 255), 1)
        label = str(class_id)
        cv2.putText(img, label, (x + box_w + 2, y + box_h - 1), font, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
        text_w = cv2.getTextSize(label, font, fs, 1)[0][0]
        x += box_w + text_w + 10


def _draw_feature_matches(rgb_a, rgb_b, pts_a, pts_b, panel_h, panel_w) -> np.ndarray:
    """Both RGB side-by-side with up to 50 colored match lines (ported)."""
    src_h_a, src_w_a = rgb_a.shape[:2]
    src_h_b, src_w_b = rgb_b.shape[:2]
    img_a = cv2.resize(rgb_a, (panel_w, panel_h))
    img_b = cv2.resize(rgb_b, (panel_w, panel_h))
    combined = np.concatenate([img_a, img_b], axis=1)

    if len(pts_a) and len(pts_b):
        scaled_a = pts_a * np.array([panel_w / src_w_a, panel_h / src_h_a])
        scaled_b = pts_b * np.array([panel_w / src_w_b, panel_h / src_h_b])
        scaled_b[:, 0] += panel_w
        max_lines, n = 50, len(scaled_a)
        step = max(1, n // max_lines)
        for i, idx in enumerate(list(range(0, n, step))[:max_lines]):
            hue = int(180 * i / max(max_lines, 1))
            color_rgb = cv2.cvtColor(np.array([[[hue, 255, 200]]], dtype=np.uint8),
                                     cv2.COLOR_HSV2RGB)[0, 0]
            color = tuple(int(c) for c in color_rgb)
            pa, pb = tuple(scaled_a[idx].astype(int)), tuple(scaled_b[idx].astype(int))
            cv2.line(combined, pa, pb, color, 1, cv2.LINE_AA)
            cv2.circle(combined, pa, 3, color, -1)
            cv2.circle(combined, pb, 3, color, -1)
    return combined


def _semantic_overlay_on_rgb(rgb, semantic, mask, color_map, panel_h, panel_w) -> np.ndarray:
    """RGB with the semantic overlap region alpha-blended + boundary drawn (ported)."""
    h, w = rgb.shape[:2]
    sem_up = cv2.resize(semantic.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_up = cv2.resize(mask.astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
    sem_colored = _apply_semantic_color(sem_up, color_map)
    result = rgb.copy()
    if mask_up.any():
        result[mask_up] = (
            (1 - OVERLAY_ALPHA) * rgb[mask_up].astype(np.float32)
            + OVERLAY_ALPHA * sem_colored[mask_up].astype(np.float32)
        ).astype(np.uint8)
        contours, _ = cv2.findContours(mask_up.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 0), 1)
    return cv2.resize(result, (panel_w, panel_h))


@register_viz("perception")
class OverlapVideo(BaseVisualization):
    name = "overlap_video"

    def render(self, reader: Any, config: dict, output_dir: Path) -> list[Path]:
        import imageio.v2 as imageio

        output_topics = list(config.get("output_topics", []))
        rgb_map = _camera_topic_map(output_topics, "image_raw_sim")
        sem_map = _camera_topic_map(output_topics, "semantic_raw_sim")
        color_map = config.get("semantic_color_map") or DEFAULT_COLOR_MAP
        fps = float(config.get("viz_fps", 10.0))
        matcher = config.get("feature_matcher", "akaze")
        match_fn = _match_orb if matcher == "orb" else _match_akaze

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        for cam_a, cam_b, description in OVERLAP_PAIRS:
            if not all(c in rgb_map and c in sem_map for c in (cam_a, cam_b)):
                continue

            # ── calibrate H on the RGB pair (mirror OverlapMetric.compute) ──
            rgb_pairs = _join_by_stamp(reader.get_messages(rgb_map[cam_a]),
                                       reader.get_messages(rgb_map[cam_b]))
            if not rgb_pairs:
                continue
            step = max(1, len(rgb_pairs) // CALIBRATION_SAMPLES)
            best_H, best_inliers, rgb_wh = None, 0, None
            best_pts: tuple[np.ndarray, np.ndarray] = (np.empty((0, 2)), np.empty((0, 2)))
            for msg_a, msg_b in rgb_pairs[::step][:CALIBRATION_SAMPLES]:
                gray_a, gray_b = _to_gray(msg_a), _to_gray(msg_b)
                if gray_a is None or gray_b is None:
                    continue
                pts_a, pts_b = match_fn(gray_a, gray_b)
                H, inliers = _compute_homography(pts_a, pts_b)
                if H is not None and inliers > best_inliers:
                    best_H, best_inliers = H, inliers
                    best_pts = (pts_a, pts_b)
                    rgb_wh = (gray_a.shape[1], gray_a.shape[0])
            if best_H is None or rgb_wh is None:
                continue

            # ── per-stamp semantic frames -> overlay video ──────────────────
            sem_pairs = _join_by_stamp(reader.get_messages(sem_map[cam_a]),
                                       reader.get_messages(sem_map[cam_b]))
            if not sem_pairs:
                continue
            rgb_a_msgs = reader.get_messages(rgb_map[cam_a])
            rgb_b_msgs = reader.get_messages(rgb_map[cam_b])

            out_path = output_dir / f"overlap_cam{cam_a}_cam{cam_b}.mp4"
            writer = None
            H_sem = mask_a = mask_b = None
            try:
                for msg_sa, msg_sb in sem_pairs:
                    sem_a = _decode_classid_plane(msg_sa)
                    sem_b = _decode_classid_plane(msg_sb)
                    if sem_a is None or sem_b is None or sem_a.shape != sem_b.shape:
                        continue
                    if H_sem is None:
                        sem_wh = (sem_a.shape[1], sem_a.shape[0])
                        H_sem = _scale_homography_to_semantic(best_H, rgb_wh, sem_wh)
                        mask_a, mask_b = _compute_overlap_mask(H_sem, sem_wh)
                        if int(mask_b.sum()) < MIN_OVERLAP_PIXELS:
                            break
                    agreement = _compute_semantic_agreement(sem_a, sem_b, H_sem, mask_b)
                    ts_a, ts_b = _stamp_ns(msg_sa), _stamp_ns(msg_sb)
                    rgb_a = _decode_rgb(_nearest_by_stamp(rgb_a_msgs, ts_a or 0))
                    rgb_b = _decode_rgb(_nearest_by_stamp(rgb_b_msgs, ts_b or 0))
                    if rgb_a is None or rgb_b is None:
                        continue

                    frame = self._compose(
                        rgb_a, rgb_b, sem_a, sem_b, mask_a, mask_b, best_pts,
                        agreement, color_map, cam_a, cam_b, description,
                        best_inliers, ts_a or 0,
                    )
                    if writer is None:
                        writer = imageio.get_writer(str(out_path), fps=fps,
                                                    macro_block_size=1)
                    writer.append_data(frame)
            finally:
                if writer is not None:
                    writer.close()

            if out_path.exists() and out_path.stat().st_size > 0:
                paths.append(out_path)
        return paths

    @staticmethod
    def _compose(rgb_a, rgb_b, sem_a, sem_b, mask_a, mask_b, best_pts, agreement,
                 color_map, cam_a, cam_b, description, inliers, ts_ns) -> np.ndarray:
        panel_h, panel_w = rgb_a.shape[0] // 2 or 1, rgb_a.shape[1] // 2 or 1
        frame_w, frame_h = panel_w * 2, HEADER_H + panel_h * 2
        frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        badge = (0, 200, 0) if (agreement is not None and agreement >= AGREEMENT_PASS) else (0, 0, 255)
        cv2.putText(frame, f"{description} | {_ns_to_datetime_str(ts_ns)}", (8, 24),
                    font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        agree_text = "Agreement: n/a" if agreement is None else f"Agreement: {agreement * 100:.1f}%"
        tw = cv2.getTextSize(agree_text, font, 0.5, 1)[0][0]
        cv2.putText(frame, agree_text, (frame_w - tw - 30, 24), font, 0.5, badge, 1, cv2.LINE_AA)
        cv2.circle(frame, (frame_w - 12, 20), 7, badge, -1)

        top = _draw_feature_matches(rgb_a, rgb_b, best_pts[0], best_pts[1], panel_h, panel_w)
        frame[HEADER_H:HEADER_H + panel_h, :] = top
        cv2.putText(frame, f"Cam {cam_a}", (5, HEADER_H + 15), font, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Cam {cam_b}", (panel_w + 5, HEADER_H + 15), font, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{inliers} matches", (panel_w - 90, HEADER_H + 15), font, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        row2 = HEADER_H + panel_h
        frame[row2:row2 + panel_h, :panel_w] = _semantic_overlay_on_rgb(rgb_a, sem_a, mask_a, color_map, panel_h, panel_w)
        frame[row2:row2 + panel_h, panel_w:] = _semantic_overlay_on_rgb(rgb_b, sem_b, mask_b, color_map, panel_h, panel_w)
        cv2.putText(frame, "Semantic Overlap", (5, row2 + 15), font, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
        _draw_legend(frame, color_map, y=frame_h - 16, x_start=8)
        return _pad_to_even(frame)
