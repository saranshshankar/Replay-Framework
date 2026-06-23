"""Tier-3 visualization tests (TIER3-VIZ-DESIGN.md).

Covers the viz foundation (the shared ``render_visualizations`` runner, the
``BaseVisualization`` plugin seam, dependency isolation from the cheap metrics
gate) and the two perception viz plugins (``overlap_video`` / ``semantic_overlay``),
which are faithful V2 revamps of Aniket's PoC viz.

INVARIANT (TIER3-VIZ-DESIGN §2/§9): the metrics gate must stay cheap — it never
imports the viz package or the mp4 encoder, and viz never affects the verdict.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from replay.metrics.base import BaseVisualization
from replay.metrics.registry import register_viz, get_viz_plugins


# ── shared synthetic-frame helpers (mirrors test_overlap.py) ─────────────────


def _textured_rgb(side: int = 160, seed: int = 7) -> np.ndarray:
    """A feature-rich HxWx3 uint8 RGB frame (AKAZE finds stable corners)."""
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
    rgba[..., 0] = classid_plane.astype(np.uint8)
    rgba[..., 3] = 255
    return rgba


def _img_msg(typestore, arr: np.ndarray, encoding: str, ts: int):
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    sec, nanosec = ts // 1_000_000_000, ts % 1_000_000_000
    hdr = Header(stamp=Time(sec=int(sec), nanosec=int(nanosec)), frame_id="cam")
    h, w = arr.shape[:2]
    ch = arr.shape[2] if arr.ndim == 3 else 1
    return Image(
        header=hdr, height=h, width=w, encoding=encoding, is_bigendian=0,
        step=w * ch, data=np.ascontiguousarray(arr).reshape(-1).astype(np.uint8),
    )


def _write_pair_bag(bag_dir: Path, cams=(2, 3), n_frames: int = 6,
                    sem_side: int = 96, rgb_side: int = 160,
                    sem_a=None, sem_b=None) -> tuple[Path, list[str]]:
    """Write a rosbag2 with two adjacent cameras' image_raw_sim (identical textured
    RGB so AKAZE matches) + semantic_raw_sim (rgba8, R = class id), stamp-aligned."""
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    a, b = cams
    rgb_a = f"/perception_node/camera_{a}/image_raw_sim"
    rgb_b = f"/perception_node/camera_{b}/image_raw_sim"
    semt_a = f"/perception_node/camera_{a}/semantic_raw_sim"
    semt_b = f"/perception_node/camera_{b}/semantic_raw_sim"
    topics = [rgb_a, rgb_b, semt_a, semt_b]

    rgb = _textured_rgb(rgb_side, seed=7)
    if sem_a is None:
        sem_a = (np.indices((sem_side, sem_side)).sum(0) % 6).astype(np.int16)
    if sem_b is None:
        sem_b = sem_a.copy()
    rgba_a, rgba_b = _classid_rgba(sem_a), _classid_rgba(sem_b)

    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        conns = {t: writer.add_connection(t, Image.__msgtype__, typestore=typestore)
                 for t in topics}
        for i in range(n_frames):
            ts = i * 100_000_000
            for t, arr, enc in (
                (rgb_a, rgb, "rgb8"), (rgb_b, rgb, "rgb8"),
                (semt_a, rgba_a, "rgba8"), (semt_b, rgba_b, "rgba8"),
            ):
                writer.write(conns[t], ts,
                             typestore.serialize_cdr(_img_msg(typestore, arr, enc, ts),
                                                     Image.__msgtype__))
    return bag_dir, topics


# ── foundation: render_visualizations runner + plugin seam + isolation ───────


class _DummyViz(BaseVisualization):
    """A registered viz that writes a sentinel file and returns its path."""

    def render(self, reader, config, output_dir: Path) -> list[Path]:
        p = Path(output_dir) / "dummy.txt"
        p.write_text("rendered")
        return [p]


class _BoomViz(BaseVisualization):
    def render(self, reader, config, output_dir: Path) -> list[Path]:
        raise RuntimeError("plugin blew up")


def test_render_visualizations_runs_registered_plugins(tmp_path):
    """render_visualizations invokes each registered viz plugin and returns its
    output paths under <output>/viz/."""
    from replay.metrics.viz_runner import render_visualizations

    register_viz("vizmod_ok")(_DummyViz)
    out = render_visualizations("vizmod_ok", reader=None, config={}, output_dir=tmp_path)

    assert len(out) == 1
    assert out[0].exists() and out[0].read_text() == "rendered"
    assert out[0].parent == tmp_path / "viz"


def test_render_visualizations_isolates_plugin_failure(tmp_path):
    """One plugin raising never aborts the others or the caller (per-plugin try/except)."""
    from replay.metrics.viz_runner import render_visualizations

    register_viz("vizmod_mixed")(_BoomViz)
    register_viz("vizmod_mixed")(_DummyViz)
    out = render_visualizations("vizmod_mixed", reader=None, config={}, output_dir=tmp_path)

    # the good plugin still produced its file; no exception propagated
    assert any(p.name == "dummy.txt" and p.exists() for p in out)


def test_render_visualizations_degrades_without_encoder(tmp_path, monkeypatch):
    """When the [viz] encoder is absent, render_visualizations returns [] with a
    clear install hint and never invokes a plugin (no traceback)."""
    from replay.metrics import viz_runner

    register_viz("vizmod_noenc")(_DummyViz)
    # Simulate `import imageio` failing (the encoder isn't installed).
    monkeypatch.setitem(sys.modules, "imageio", None)

    out = viz_runner.render_visualizations(
        "vizmod_noenc", reader=None, config={}, output_dir=tmp_path)

    assert out == []
    assert not (tmp_path / "viz" / "dummy.txt").exists()


def test_metrics_path_does_not_import_viz_or_encoder():
    """The cheap metrics pack must not pull the viz package or the mp4 encoder
    (bifurcation: `pip install ...[metrics]` stays lean)."""
    code = (
        "import sys; import replay.metrics.perception; "
        "import replay.cli; "
        "bad = [m for m in sys.modules "
        "      if m == 'imageio' or m.startswith('replay.metrics.perception.viz')]; "
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "CLEAN" in r.stdout, r.stdout


# ── overlap_video plugin (V2 revamp of Aniket's generate_overlap_videos) ─────


def _read_mp4_first_frame(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    rdr = imageio.get_reader(str(path))
    try:
        return rdr.get_data(0)
    finally:
        rdr.close()


def test_overlap_video_produces_readable_mp4(tmp_path):
    """A calibratable cam pair (2,3) yields a non-empty, ffmpeg-decodable mp4."""
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.perception.viz.overlap_video import OverlapVideo

    bag, topics = _write_pair_bag(tmp_path / "bag", cams=(2, 3), n_frames=6)
    reader = BagReader(bag, topics)
    out = OverlapVideo().render(reader, {"output_topics": topics}, tmp_path)

    assert len(out) >= 1
    mp4 = out[0]
    assert mp4.suffix == ".mp4" and mp4.exists() and mp4.stat().st_size > 0
    frame = _read_mp4_first_frame(mp4)
    assert frame.ndim == 3 and frame.shape[2] == 3  # decodable RGB video


def test_overlap_video_no_configured_pairs_returns_empty(synthetic_bag):
    """A bag with no complete adjacency pair (only cam 0's image_raw_sim, no
    semantic) renders nothing and never raises."""
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.perception.viz.overlap_video import OverlapVideo

    out_topic = "/perception_node/camera_0/image_raw_sim"
    reader = BagReader(synthetic_bag, [out_topic])
    out = OverlapVideo().render(
        reader, {"output_topics": [out_topic]}, synthetic_bag.parent / "viz_none")
    assert out == []


# ── semantic_overlay plugin (V2 revamp of Aniket's generate_combined_videos) ──


def _write_combined_bag(bag_dir: Path, cam: int = 0, n_frames: int = 4,
                        with_depth: bool = True) -> tuple[Path, list[str]]:
    """Write one camera's image_raw_sim (rgb8) + semantic_raw_sim (rgba8) +
    optionally depth_raw_sim (32FC1), stamp-aligned, for the combined grid."""
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    rgb_t = f"/perception_node/camera_{cam}/image_raw_sim"
    sem_t = f"/perception_node/camera_{cam}/semantic_raw_sim"
    depth_t = f"/perception_node/camera_{cam}/depth_raw_sim"
    topics = [rgb_t, sem_t] + ([depth_t] if with_depth else [])

    rgb = _textured_rgb(120, seed=3)
    sem_plane = (np.indices((90, 90)).sum(0) % 6).astype(np.int16)
    rgba = _classid_rgba(sem_plane)
    depth = (np.linspace(0.5, 4.0, 64 * 64, dtype=np.float32)).reshape(64, 64)

    def _depth_msg(ts):
        sec, nanosec = ts // 1_000_000_000, ts % 1_000_000_000
        hdr = Header(stamp=Time(sec=int(sec), nanosec=int(nanosec)), frame_id="cam")
        h, w = depth.shape
        return Image(header=hdr, height=h, width=w, encoding="32FC1", is_bigendian=0,
                     step=w * 4, data=np.frombuffer(depth.tobytes(), dtype=np.uint8))

    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        conns = {t: writer.add_connection(t, Image.__msgtype__, typestore=typestore)
                 for t in topics}
        for i in range(n_frames):
            ts = i * 100_000_000
            writer.write(conns[rgb_t], ts, typestore.serialize_cdr(
                _img_msg(typestore, rgb, "rgb8", ts), Image.__msgtype__))
            # vary the semantic plane slightly so the temporal-diff quadrant differs
            shifted = ((sem_plane + i) % 6).astype(np.int16)
            writer.write(conns[sem_t], ts, typestore.serialize_cdr(
                _img_msg(typestore, _classid_rgba(shifted), "rgba8", ts), Image.__msgtype__))
            if with_depth:
                writer.write(conns[depth_t], ts, typestore.serialize_cdr(
                    _depth_msg(ts), Image.__msgtype__))
    return bag_dir, topics


def test_semantic_overlay_produces_readable_mp4(tmp_path):
    """A camera with rgb + semantic + depth yields a decodable per-camera grid mp4."""
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.perception.viz.semantic_overlay import SemanticOverlay

    bag, topics = _write_combined_bag(tmp_path / "bag", cam=0, n_frames=4, with_depth=True)
    reader = BagReader(bag, topics)
    out = SemanticOverlay().render(reader, {"output_topics": topics}, tmp_path)

    assert len(out) >= 1
    mp4 = out[0]
    assert mp4.suffix == ".mp4" and mp4.exists() and mp4.stat().st_size > 0
    frame = _read_mp4_first_frame(mp4)
    assert frame.ndim == 3 and frame.shape[2] == 3


def test_semantic_overlay_without_depth_still_renders(tmp_path):
    """A camera with no depth topic still renders (depth quadrant shows N/A)."""
    from replay.metrics.bag_reader import BagReader
    from replay.metrics.perception.viz.semantic_overlay import SemanticOverlay

    bag, topics = _write_combined_bag(tmp_path / "bag", cam=0, n_frames=3, with_depth=False)
    reader = BagReader(bag, topics)
    out = SemanticOverlay().render(reader, {"output_topics": topics}, tmp_path)

    assert len(out) >= 1 and out[0].exists() and out[0].stat().st_size > 0


# ── report Visualizations block + CLI entry points (§8, §2) ──────────────────


def test_report_links_visualizations_when_present(tmp_path):
    """generate_report renders a Visualizations block linking the mp4s when given."""
    from replay.metrics.report.generator import generate_report

    generate_report("perception", "run1", [], tmp_path, {},
                    visualizations=["../viz/overlap_cam2_cam3.mp4", "../viz/combined_cam2.mp4"])
    html = (tmp_path / "report.html").read_text()
    assert "overlap_cam2_cam3.mp4" in html
    assert "../viz/combined_cam2.mp4" in html


def test_report_shows_hint_when_no_visualizations(tmp_path):
    """With no viz, the report shows the `replay-module viz` hint, not links."""
    from replay.metrics.report.generator import generate_report

    generate_report("perception", "run1", [], tmp_path, {})
    html = (tmp_path / "report.html").read_text()
    assert "replay-module viz" in html


def test_viz_subcommand_generates_videos(tmp_path):
    """`replay-module viz` renders mp4s offline from an existing output bag."""
    from click.testing import CliRunner
    from replay.cli import main

    bag, _ = _write_pair_bag(tmp_path / "bag", cams=(2, 3), n_frames=6)
    out = tmp_path / "out"
    res = CliRunner().invoke(
        main, ["viz", "--module", "perception", "--bag", str(bag), "--output", str(out)])
    assert res.exit_code == 0, res.output
    assert list((out / "viz").glob("*.mp4")), res.output


def test_metrics_run_viz_renders_and_links(tmp_path):
    """`metrics --run-viz` renders viz AND the report links them (whatever the verdict)."""
    from click.testing import CliRunner
    from replay.cli import main

    bag, _ = _write_pair_bag(tmp_path / "bag", cams=(2, 3), n_frames=6)
    out = tmp_path / "out"
    CliRunner().invoke(
        main, ["metrics", "--module", "perception", "--bag", str(bag),
               "--output", str(out), "--run-viz"])
    assert list((out / "viz").glob("*.mp4"))
    assert "../viz/" in (out / "reports" / "report.html").read_text()


# ── on-demand cloud viz workflow template (§7) ───────────────────────────────

VIZ_WORKFLOW = Path(__file__).resolve().parents[1] / "ci" / "10xcode" / "replay-perception-viz.yml"


def test_viz_workflow_parses_and_is_decoupled_cpu_only():
    """The cloud viz template is on-demand, CPU-only, and fully decoupled from the
    blocking GPU gate (TIER3-VIZ-DESIGN §7): no RunsOn GPU, no engine, no S3/OIDC,
    not a pull_request trigger; it just installs [viz] and renders the recorded bag."""
    raw = VIZ_WORKFLOW.read_text()
    wf = yaml.safe_load(raw)
    assert isinstance(wf, dict)

    # on-demand only — never part of the gate
    on = wf.get("on") or wf.get(True)
    assert "workflow_dispatch" in on
    assert "pull_request" not in on

    # CPU only: GitHub-hosted ubuntu, no RunsOn GPU label, no GPU profile
    for job in wf["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest"
    assert "runs-on=" not in raw and "replay-gpu" not in raw

    # decoupled: no S3/OIDC, no engine, no GHCR image pull
    assert "id-token" not in (wf.get("permissions") or {})
    assert "configure-aws-credentials" not in raw
    assert "PERCEPTION_ENGINE_S3" not in raw

    # it installs the [viz] extra and runs the viz subcommand
    assert "[viz]" in raw and "replay-module viz" in raw

    # every action pinned to a concrete ref; the artifact is ephemeral
    pin = re.compile(r"@(v?\d|[0-9a-f]{40})")
    for job in wf["jobs"].values():
        for step in job.get("steps") or []:
            uses = step.get("uses")
            if uses:
                assert pin.search(uses), f"action {uses!r} must pin a concrete ref"
            if (uses or "").startswith("actions/upload-artifact"):
                assert (step.get("with") or {}).get("retention-days") == 5
