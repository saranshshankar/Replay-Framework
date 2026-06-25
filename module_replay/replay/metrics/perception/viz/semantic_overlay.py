"""semantic_overlay — per-camera 2x2 combined grid (V2 revamp of Aniket's PoC).

Faithful port of ``perception_metrics/visualizations/semantic_overlay.py``
(``generate_combined_videos``) from ``origin/aniket/feat/module_wise_replay_poc``,
rewired to V2: frames come from the framework ``BagReader`` (``get_messages`` +
the ``overlap.py`` / ``depth.py`` decode helpers) instead of the PoC
``BagReader.iter_*`` API, cameras come from ``_camera_topic_map``, and it is encoded
with ``imageio[ffmpeg]``.

Per camera, a 2x2 grid:  RGB | semantic-colored | depth heatmap | temporal-diff
(semantic vs the previous frame + a consistency %). A camera with no ``depth_raw_sim``
shows an "N/A" depth quadrant rather than being skipped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from replay.metrics.base import BaseVisualization
from replay.metrics.registry import register_viz
from replay.metrics.perception.depth import _decode_depth
from replay.metrics.perception.overlap import (
    _camera_topic_map,
    _decode_classid_plane,
    _join_by_stamp,
    _stamp_ns,
)
from replay.metrics.perception.viz.overlap_video import (
    DEFAULT_COLOR_MAP,
    _apply_semantic_color,
    _decode_rgb,
    _draw_legend,
    _nearest_by_stamp,
    _ns_to_datetime_str,
    _pad_to_even,
)

DEPTH_HOLD_TOLERANCE_NS = 2_000_000_000  # depth is slow (~0.5-10 Hz) — hold last


def _add_label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def _depth_range(depth_msgs: list[tuple[int, Any]]) -> tuple[float, float]:
    """Robust (2nd, 98th)-percentile depth range over valid (>0) pixels."""
    valid = []
    for _, m in depth_msgs:
        d = _decode_depth(m)
        if d is None:
            continue
        v = d[d > 0]
        if v.size:
            valid.append(v)
    if not valid:
        return 0.0, 1.0
    allv = np.concatenate(valid)
    return float(np.percentile(allv, 2)), float(np.percentile(allv, 98))


def _depth_to_colormap(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    norm = np.clip((depth.astype(np.float32) - vmin) / (vmax - vmin + 1e-6) * 255,
                   0, 255).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colored[depth <= 0] = 0
    return colored


@register_viz("perception")
class SemanticOverlay(BaseVisualization):
    name = "semantic_overlay"

    def render(self, reader: Any, config: dict, output_dir: Path) -> list[Path]:
        import imageio.v2 as imageio

        output_topics = list(config.get("output_topics", []))
        rgb_map = _camera_topic_map(output_topics, "image_raw_sim")
        sem_map = _camera_topic_map(output_topics, "semantic_raw_sim")
        depth_map = _camera_topic_map(output_topics, "depth_raw_sim")
        color_map = config.get("semantic_color_map") or DEFAULT_COLOR_MAP
        fps = float(config.get("viz_fps", 10.0))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        for cam in sorted(set(rgb_map) & set(sem_map)):
            pairs = _join_by_stamp(reader.get_messages(rgb_map[cam]),
                                   reader.get_messages(sem_map[cam]))
            if not pairs:
                continue
            depth_msgs = reader.get_messages(depth_map[cam]) if cam in depth_map else []
            vmin, vmax = _depth_range(depth_msgs)

            out_path = output_dir / f"combined_cam{cam}.mp4"
            writer = None
            prev_sem: Optional[np.ndarray] = None
            try:
                for rgb_msg, sem_msg in pairs:
                    rgb = _decode_rgb(rgb_msg)
                    sem = _decode_classid_plane(sem_msg)
                    if rgb is None or sem is None:
                        continue
                    ts = _stamp_ns(sem_msg) or 0
                    depth = None
                    if depth_msgs:
                        dmsg = _nearest_by_stamp(depth_msgs, ts, tol=DEPTH_HOLD_TOLERANCE_NS)
                        depth = _decode_depth(dmsg) if dmsg is not None else None
                    frame = self._compose(rgb, sem, depth, vmin, vmax, prev_sem,
                                          color_map, cam, ts)
                    prev_sem = sem.copy()
                    if writer is None:
                        writer = imageio.get_writer(str(out_path), fps=fps,
                                                    macro_block_size=16)  # /16 auto-pad for player compatibility (QuickTime/Safari/PowerPoint)
                    writer.append_data(frame)
            finally:
                if writer is not None:
                    writer.close()

            if out_path.exists() and out_path.stat().st_size > 0:
                paths.append(out_path)
        return paths

    @staticmethod
    def _compose(rgb, sem, depth, vmin, vmax, prev_sem, color_map, cam, ts_ns) -> np.ndarray:
        h, w = rgb.shape[:2]
        cam_label = f"cam_{cam}"

        q_rgb = rgb.copy()
        _add_label(q_rgb, f"RGB - {cam_label}")

        sem_up = cv2.resize(sem.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        q_sem = _apply_semantic_color(sem_up, color_map)
        _add_label(q_sem, "Semantic")
        _draw_legend(q_sem, color_map, y=30, x_start=8)

        if depth is not None:
            q_depth = _depth_to_colormap(depth, vmin, vmax)
            if q_depth.shape[:2] != (h, w):
                q_depth = cv2.resize(q_depth, (w, h), interpolation=cv2.INTER_NEAREST)
            _add_label(q_depth, "Depth")
        else:
            q_depth = np.zeros((h, w, 3), dtype=np.uint8)
            _add_label(q_depth, "Depth (N/A)")

        q_diff = np.zeros((h, w, 3), dtype=np.uint8)
        if prev_sem is not None and prev_sem.shape == sem.shape:
            consistency = float(np.mean(sem == prev_sem))
            diff_up = cv2.resize((sem != prev_sem).astype(np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST)
            q_diff[diff_up > 0] = (255, 255, 255)
            _add_label(q_diff, "Temporal Diff")
            cv2.putText(q_diff, f"Consistency: {consistency:.2%}", (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            _add_label(q_diff, "Temporal Diff (first frame)")

        combined = np.vstack([np.hstack([q_rgb, q_sem]), np.hstack([q_depth, q_diff])])
        cv2.putText(combined, _ns_to_datetime_str(ts_ns),
                    (combined.shape[1] // 2 - 130, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return _pad_to_even(combined)
