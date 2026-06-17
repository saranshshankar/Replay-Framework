"""Offline matplotlib plot generators for the rich report (MTRC-05 / A2).

Adapted FROM the PoC ``perception_metrics/visualizations/*`` presentation style
but DRIVEN BY our flat metrics.json / ``MetricResult`` value dicts and our
read-once ``BagReader`` — the PoC's nested per-window/per-camera shapes and its
``iter_depth`` / ``DEPTH_CAMERA_IDS`` / ``get_bag_metadata`` reader API DO NOT
EXIST here and are not used.

INVARIANTS (PATTERNS § Key invariants #3 + threat T-16-02):
- Offline pure-Python: matplotlib **Agg** backend only; this module imports no
  ROS runtime client library and no deep-learning framework, and uses no GPU. The
  backend is forced at import time, before pyplot.
- FAIL-SAFE: every plot generator is wrapped in its own ``try/except`` and always
  closes its figure; a single plot failing is OMITTED from the returned dict and
  never raises out. A plot failure must NEVER crash a run or flip the B9 verdict —
  ``generate_report`` calls this purely for presentation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# Force the headless backend BEFORE importing pyplot (no display, no GUI toolkit).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np  # noqa: E402

from replay.metrics.base import MetricResult  # noqa: E402

logger = logging.getLogger(__name__)

# Perception's replay depth output topics: 6 cameras, 32FC* lidar-interpolated
# depth (KT/playbooks/01-perception.md). Decoded exactly as depth.py does.
DEPTH_TOPICS = [f"/perception_node/camera_{i}/depth_raw_sim" for i in range(6)]

# Default p95 latency gate (ms) — drawn as a threshold line on the latency plot.
DEFAULT_LATENCY_THRESHOLD_MS = 50.0

# Sample every Nth depth frame for the heatmap montage (bounds figure count).
DEPTH_SAMPLE_RATE = 5


def _find(metric_results: list[MetricResult], name: str) -> Optional[MetricResult]:
    for r in metric_results:
        if r.name == name:
            return r
    return None


def _plot_latency(value: dict, output_dir: Path) -> Optional[Path]:
    """Aggregate-latency bar chart (p50/p95/p99/max) with a threshold line.

    OUR latency plugin emits AGGREGATE SCALARS ONLY (no per-window series), so a
    PoC-style time series is impossible; we render the aggregate bars instead.
    Returns None (renders NOTHING) when the latency row is skipped / has no p95 —
    a misleading empty/zero plot is worse than no plot.
    """
    if value.get("skipped") or value.get("latency_p95_ms") is None:
        return None
    labels = ["p50", "p95", "p99", "max"]
    keys = ["p50_ms", "p95_ms", "p99_ms", "max_ms"]
    vals = [float(value.get(k, 0.0) or 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#1a237e", "#5c6bc0", "#FF5722"]
    ax.bar(labels, vals, color=colors, edgecolor="white")
    threshold = DEFAULT_LATENCY_THRESHOLD_MS
    ax.axhline(
        y=threshold, color="#E91E63", linestyle=":", lw=2,
        label=f"Threshold ({threshold:.0f}ms)",
    )
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(
        f"Inference Latency (seg_argmax) — {int(value.get('num_windows', 0))} windows",
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "latency_time_series.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pipeline(value: dict, output_dir: Path) -> Optional[Path]:
    """Horizontal bar of per-output-topic effective Hz (our pipeline value shape).

    The PoC charted ``image_pipeline.stages`` avg-ms; OUR pipeline plugin instead
    emits ``per_topic[topic].mean_hz`` (the topic-level cadence the gate reads), so
    we render per-topic Hz. Returns None when there is no per-topic data to plot.
    """
    per_topic = value.get("per_topic") or {}
    # Surface only topics with a measurable rate (>=2 messages -> mean_hz>0).
    items = [
        (topic, float(d.get("mean_hz", 0.0) or 0.0))
        for topic, d in per_topic.items()
        if isinstance(d, dict) and d.get("num_messages", 0) >= 2
    ]
    if not items:
        return None
    items.sort(key=lambda kv: kv[1])
    # Shorten topic names to the trailing two path segments for the y labels.
    names = ["/".join(t.strip("/").split("/")[-2:]) for t, _ in items]
    hz = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(items) + 2)))
    colors = plt.cm.Set2(np.linspace(0, 1, len(items)))
    ax.barh(range(len(items)), hz, color=colors, edgecolor="white")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(names, fontsize=9)
    for i, v in enumerate(hz):
        ax.text(v, i, f" {v:.2f} Hz", va="center", fontsize=9, fontweight="bold")
    mean_hz = float(value.get("mean_hz", 0.0) or 0.0)
    ax.set_xlabel("Effective rate (Hz)", fontsize=12)
    ax.set_title(f"Pipeline Throughput per Topic (headline {mean_hz:.2f} Hz)", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "pipeline_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _decode_depth_grid(msg) -> Optional[np.ndarray]:
    """Decode a 32FC* depth Image msg to a 2D float grid (mirrors depth.py)."""
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    data = getattr(msg, "data", None)
    encoding = str(getattr(msg, "encoding", "") or "")
    if height <= 0 or width <= 0 or data is None or not encoding.startswith("32FC"):
        return None
    raw = np.asarray(data)
    if raw.size == 0:
        return None
    vals = raw.view(np.float32) if raw.dtype == np.uint8 else raw.astype(np.float32)
    if vals.size < height * width:
        return None
    channels = max(1, vals.size // (height * width))
    grid = vals[: height * width * channels].reshape(height, width, channels)
    return grid[..., 0].astype(np.float64)


def _plot_depth_heatmaps(reader, output_dir: Path) -> list[Path]:
    """Per-frame depth heatmaps for the 6 depth_raw_sim cameras (turbo colormap).

    REWIRED to OUR BagReader: iterates ``reader.get_messages(topic)`` for each
    depth topic and decodes 32FC* exactly like depth.py. The PoC's iter_depth /
    DEPTH_CAMERA_IDS / get_bag_metadata are NOT used. Returns [] (no files) when
    no depth frames decode — common for non-perception bags or the tiny synthetic
    fixture — without raising.
    """
    saved: list[Path] = []
    cmap = plt.get_cmap("turbo")
    depth_dir = output_dir / "depth_heatmaps"
    for cam_id, topic in enumerate(DEPTH_TOPICS):
        msgs = reader.get_messages(topic)
        if not msgs:
            continue
        frames = []
        for _ts, msg in msgs:
            grid = _decode_depth_grid(msg)
            if grid is not None:
                frames.append(grid)
        if not frames:
            continue
        # Consistent colour range across this camera's sampled frames.
        valid = np.concatenate([g[np.isfinite(g) & (g > 0)].ravel() for g in frames]) \
            if any(np.any(np.isfinite(g) & (g > 0)) for g in frames) else np.array([])
        if valid.size == 0:
            continue
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        depth_dir.mkdir(parents=True, exist_ok=True)
        for idx, grid in enumerate(frames):
            if idx % DEPTH_SAMPLE_RATE != 0:
                continue
            g = grid.astype(np.float32)
            g[~(np.isfinite(g) & (g > 0))] = np.nan
            norm = np.clip((g - vmin) / (vmax - vmin + 1e-6), 0, 1)
            colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
            colored[np.isnan(g)] = 0
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(colored)
            ax.set_title(f"Depth — Camera {cam_id} (frame {idx})", fontsize=12)
            ax.axis("off")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Depth", fontsize=10)
            path = depth_dir / f"depth_cam{cam_id}_frame_{idx:03d}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    return saved


def generate_report_plots(
    metric_results: list[MetricResult],
    reader,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate the report's static plots, returning a ``name -> Path`` dict.

    Each generator runs in its own ``try/except``: a failure logs + skips and is
    OMITTED from the dict; the function never raises (threat T-16-02). Depth
    heatmaps are returned under the ``"depth"`` key as the FIRST heatmap path (the
    template loops the depth_heatmaps/ dir); the per-frame list is on disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    # ── Latency (aggregate bar) ─────────────────────────────────────────────
    lat = _find(metric_results, "latency_p95_ms")
    if lat is not None and isinstance(lat.value, dict):
        try:
            p = _plot_latency(lat.value, output_dir)
            if p is not None:
                plots["latency"] = p
        except Exception:  # never propagate — degrade to no latency plot
            logger.warning("latency plot failed; omitting", exc_info=True)

    # ── Pipeline throughput (per-topic horizontal bar) ──────────────────────
    pipe = _find(metric_results, "pipeline_throughput_hz")
    if pipe is not None and isinstance(pipe.value, dict):
        try:
            p = _plot_pipeline(pipe.value, output_dir)
            if p is not None:
                plots["pipeline"] = p
        except Exception:
            logger.warning("pipeline plot failed; omitting", exc_info=True)

    # ── Depth heatmaps (rewired to OUR BagReader) ───────────────────────────
    try:
        depth_paths = _plot_depth_heatmaps(reader, output_dir)
        if depth_paths:
            plots["depth"] = depth_paths[0]
    except Exception:
        logger.warning("depth heatmaps failed; omitting", exc_info=True)

    return plots
