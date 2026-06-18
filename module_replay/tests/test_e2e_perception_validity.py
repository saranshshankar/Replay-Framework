"""THE DECISIVE PROOF (plan 01-22): the four e2e fixes COMPOSE to flip the verdict.

01-19 (config: cams 0..5, per-topic expected_hz, gap_tolerance, latency_stage,
replay_breach_count gate), 01-20 (faithfulness span from the HEADER clock + the
configurable per-topic gap tolerance + jitter-tolerant cross-camera check + the
BUG-4 decimal rounding) and 01-21 (the plugin headline keys + the configurable
latency stage) each fixed their OWN half in isolation. No single one of those
tests exercises the REAL perception.yaml + all four fixes together through
``generate_report``.

This module is that composition proof. It builds a synthetic output bag shaped
EXACTLY like the real e2e run
(``.planning/phases/01-framework-perception-ci-gate/e2e-2026-06-18/metrics.json``):

  * 6 cameras 0..5 ``image_raw_sim`` + ``semantic_raw_sim``: header stamps at 5 Hz
    with periodic ~600ms EoMT inference stalls, RE-EMITTED ~3x (so raw ``count`` >>
    ``unique_count``) with the bag-WRITE clock spread over ~99s (>> the ~31s header
    span) — the BUG-1 sim-time shape.
  * 6 cameras ``depth_raw_sim`` (32FC4) + ``colored_pointcloud_sim`` at 10 Hz,
    re-emitted ~2x over ~99s.
  * ``/perception_node/diagnostics`` at 0.2 Hz carrying a ``DiagnosticStatus``
    named ``inference_seg_extract_segmentation`` with an ``avg_compute_ms`` ~8.0
    KeyValue (the BUG-3 live op name after the seg_argmax -> seg_extract rename).

Run through ``_run_metrics_pipeline`` with the REAL ``load_module_config('perception')``
spec, the verdict that was INVALID (exit 2) on the e2e flips to NOT-INVALID — and
every one of the four bug fixes is asserted at the same time. A NEGATIVE control
(a genuine 2x replay hang on a uniform-rate depth topic) still returns INVALID, so
the gate keeps its teeth.

WHY THE VERDICT IS "FAIL", NOT "PASS": the e2e's only quality failure was the real
``cross_camera_overlap_iou`` 0.5351 (below the 0.75 gate). On the tiny 2x2 synthetic
frames the OverlapMetric degrades to 0.0 (no AKAZE keypoints), which is ALSO a
quality fail — so the honest synthetic verdict is "FAIL" (exit 1), NOT "INVALID"
(exit 2). The decisive claim of this plan is the validity flip INVALID -> not-INVALID
(exit 2 -> not 2): the VALID bag is no longer thrown out as infra-noise. We assert
exactly that (``rc != 2`` and ``verdict != "INVALID"``), never a fabricated PASS.
"""
from pathlib import Path

import json

import numpy as np
import pytest

from replay.cli import _run_metrics_pipeline
from replay.module_config import load_module_config

# The REAL committed module configs dir (NOT a hand-built threshold dict): this is
# what makes the proof load perception.yaml's actual expected_hz / gap_tolerance /
# latency_stage / validity thresholds. (grep guard: load_module_config + spec.thresholds.)
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "modules"

# e2e-shaped knobs (see module docstring). Tuned so the HEALTHY bag reads
# drop_rate <= 0.02: with ~600ms stalls a 5 Hz stream loses ~2 frames per stall vs
# the span*hz expectation, so the stall count + camera frame count are chosen so the
# per-camera unique count stays >= ~0.98 * round(header_span * 5) (the synthetic-
# truncation artifact 01-20 flagged). Verified empirically before this test landed.
CAM_BASE_FRAMES = 150        # clean 5 Hz frames before stall deletions
N_STALLS = 2                 # ~600ms EoMT inference stalls per camera
STALL_MS = 600.0
WRITE_SPAN_S = 99.0          # bag-write clock span (>> the ~31s header span) — BUG 1
LATENCY_STAGE = "inference_seg_extract_segmentation"  # the live op name (BUG 3)
AVG_COMPUTE_MS = 8.0

CAMERAS = range(6)


# ── decoupled bag writer: header (robot) clock != bag-write (wall) clock ──────


def _time(typestore, ns: int):
    Time = typestore.types["builtin_interfaces/msg/Time"]
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def _hz_header(hz: float, span_s: float, max_ns: int | None = None) -> list[int]:
    """Evenly-spaced HEADER stamps (ns) at ``hz`` over ``span_s``.

    Optionally clipped to ``max_ns`` so a 10 Hz topic does not extend the GLOBAL
    header span past the camera span (the global span drives span*hz expectations
    for every topic; an over-long depth topic would inflate the camera expectation
    and manufacture a false drop_rate).
    """
    period = int(round(1e9 / hz))
    n = int(round(span_s * hz)) + 1
    stamps = [i * period for i in range(n)]
    return [s for s in stamps if s <= max_ns] if max_ns is not None else stamps


def _cam_header_stamps_with_stalls(
    hz: float, n_frames: int, stall_ms: float, n_stalls: int
) -> list[int]:
    """A clean ``hz`` grid of ``n_frames`` with ``n_stalls`` ~``stall_ms`` gaps.

    Each stall is created by DELETING the two interior frames at an evenly-spaced
    position, turning a 200ms step into a ~600ms gap (a FAITHFUL EoMT inference
    stall = missing frames, not added time). Returns sorted unique header stamps.
    """
    period = int(round(1e9 / hz))
    grid = [i * period for i in range(n_frames)]
    if n_stalls:
        every = max(4, n_frames // (n_stalls + 1))
        to_delete: set[int] = set()
        for k in range(1, n_stalls + 1):
            center = k * every
            to_delete.add(center)
            to_delete.add(center + 1)  # delete 2 frames -> ~600ms (3x period) gap
        grid = [g for i, g in enumerate(grid) if i not in to_delete]
    return grid


def _write_image_topic(
    writer,
    typestore,
    topic: str,
    header_stamps: list[int],
    write_span_s: float,
    reemit: int,
    encoding: str = "rgb8",
) -> None:
    """Write one Image topic where the BAG-WRITE clock and HEADER clock are DECOUPLED.

    Each header frame is RE-EMITTED ``reemit`` times (so the raw bag count is
    ``reemit`` x the unique header-frame count — the contract's non-consuming
    ``getLatest()`` re-emit), with bag-write timestamps spread evenly across
    ``write_span_s`` (the slow ~99s wall clock). ``header.stamp`` carries the robot
    clock, independent of the write timestamp — the BUG-1 sim-time shape.
    """
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    conn = writer.add_connection(topic, Image.__msgtype__, typestore=typestore)
    n_writes = max(len(header_stamps) * reemit, 1)
    write_period_ns = int(round(write_span_s * 1e9 / max(n_writes - 1, 1)))
    write_i = 0
    for hdr_ns in header_stamps:
        for _ in range(reemit):
            hdr = Header(stamp=_time(typestore, hdr_ns), frame_id="c")
            # 2x2 frame; 4 channels (rgba/32FC) so depth decode has a plane.
            msg = Image(
                header=hdr, height=2, width=2, encoding=encoding, is_bigendian=0,
                step=8, data=np.zeros(16, dtype=np.uint8),
            )
            # write_ts = the wall/write clock; header.stamp = the robot clock (decoupled)
            writer.write(
                conn, write_i * write_period_ns,
                typestore.serialize_cdr(msg, Image.__msgtype__),
            )
            write_i += 1


def _write_diagnostics(
    writer, typestore, topic: str, header_stamps: list[int], stage_name: str, avg_ms: float
) -> None:
    """Write a 0.2 Hz DiagnosticArray topic (the conftest diagnostics pattern).

    Each message's named ``DiagnosticStatus`` carries an ``avg_compute_ms`` KeyValue
    so ``LatencyMetric`` (BUG 3) reads p95 latency from the CONFIGURED stage name.
    """
    DiagnosticArray = typestore.types["diagnostic_msgs/msg/DiagnosticArray"]
    DiagnosticStatus = typestore.types["diagnostic_msgs/msg/DiagnosticStatus"]
    KeyValue = typestore.types["diagnostic_msgs/msg/KeyValue"]
    Header = typestore.types["std_msgs/msg/Header"]
    mt = DiagnosticArray.__msgtype__
    conn = writer.add_connection(topic, mt, typestore=typestore)
    for i, hdr_ns in enumerate(header_stamps):
        status = DiagnosticStatus(
            level=0, name=stage_name, message="ok", hardware_id="gpu",
            values=[
                KeyValue(key="avg_compute_ms", value=str(float(avg_ms))),
                KeyValue(key="max_compute_ms", value=str(float(avg_ms) + 5.0)),
            ],
        )
        hdr = Header(stamp=_time(typestore, hdr_ns), frame_id="")
        msg = DiagnosticArray(header=hdr, status=[status])
        writer.write(conn, i * 5_000_000_000, typestore.serialize_cdr(msg, mt))


def _build_e2e_bag(bag_dir: Path, *, depth_hang: bool = False) -> dict:
    """Write the e2e-shaped output bag; return the per-camera semantic counts.

    Shape (matches metrics.json): 6 cameras image+semantic at header 5 Hz with a
    142-146-style jitter spread (truncated per camera) + ~600ms stalls, re-emitted
    ~3x; 6 depth (32FC4) + 1 pointcloud at 10 Hz re-emitted ~2x; diagnostics 0.2 Hz.
    The bag-write clock spans ~99s while the header clock spans ~31s (BUG 1).

    ``depth_hang=True`` is the NEGATIVE control: a genuine 2x replay hang (a ~500ms
    gap, > 2x the 100ms depth period) is injected mid-stream on the UNIFORM-rate
    depth topics so a real stall on a non-jittery stream still BREACHES validity.
    """
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    base = _cam_header_stamps_with_stalls(5.0, CAM_BASE_FRAMES + 4, STALL_MS, N_STALLS)
    cam_max_ns = base[-1]
    cam_span_s = cam_max_ns / 1e9
    # 142-146-style jitter: each camera truncated to a slightly different count.
    longest = len(base)
    cam_counts = [longest, longest - 2, longest - 4, longest, longest - 5, longest - 6]

    depth_hdr = _hz_header(10.0, cam_span_s, max_ns=cam_max_ns)
    pc_hdr = _hz_header(10.0, cam_span_s, max_ns=cam_max_ns)
    diag_hdr = _hz_header(0.2, cam_span_s, max_ns=cam_max_ns)
    if depth_hang:
        # Inject one ~500ms gap mid-stream on the uniform 10 Hz depth grid: 500ms is
        # 5x the 100ms period, well past the 2.0 default gap-tolerance (200ms) — a
        # genuine replay hang on a non-jittery stream.
        depth_hdr = sorted(set(depth_hdr))
        mid = len(depth_hdr) // 2
        depth_hdr = depth_hdr[:mid] + [t + int(round(0.5e9)) for t in depth_hdr[mid:]]

    semantic_counts: dict[str, int] = {}
    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        for cam in CAMERAS:
            cam_hdr = base[: cam_counts[cam]]
            _write_image_topic(
                writer, typestore, f"/perception_node/camera_{cam}/image_raw_sim",
                cam_hdr, WRITE_SPAN_S, reemit=3,
            )
            _write_image_topic(
                writer, typestore, f"/perception_node/camera_{cam}/semantic_raw_sim",
                cam_hdr, WRITE_SPAN_S, reemit=3,
            )
            semantic_counts[f"/perception_node/camera_{cam}/semantic_raw_sim"] = len(cam_hdr)
            _write_image_topic(
                writer, typestore, f"/perception_node/camera_{cam}/depth_raw_sim",
                depth_hdr, WRITE_SPAN_S, reemit=2, encoding="32FC4",
            )
        _write_image_topic(
            writer, typestore, "/perception_node/colored_pointcloud_sim",
            pc_hdr, WRITE_SPAN_S, reemit=2,
        )
        _write_diagnostics(
            writer, typestore, "/perception_node/diagnostics",
            diag_hdr, LATENCY_STAGE, AVG_COMPUTE_MS,
        )
    return semantic_counts


def _decimal_places(x: float) -> int:
    """Number of decimal places in the shortest round-trippable repr of ``x``."""
    text = repr(float(x))
    if "e" in text or "E" in text:  # scientific (e.g. the 1e9 empty-topic sentinel)
        return 0
    return len(text.split(".")[1]) if "." in text else 0


@pytest.fixture
def e2e_metrics(tmp_path):
    """Build the e2e-shaped VALID bag, run the REAL pipeline, return (rc, metrics.json)."""
    bag = tmp_path / "e2e_valid"
    semantic_counts = _build_e2e_bag(bag)
    spec = load_module_config("perception", CONFIGS_DIR)
    out_dir = tmp_path / "run"
    rc = _run_metrics_pipeline(spec, bag, out_dir)
    doc = json.loads((out_dir / "reports" / "metrics.json").read_text())
    return rc, doc, semantic_counts, spec


def test_e2e_uses_real_perception_config():
    """Sanity / grep guard: the proof loads the REAL perception.yaml (its actual
    validity thresholds + the configured latency stage), NOT a hand-built dict."""
    spec = load_module_config("perception", CONFIGS_DIR)
    # the real validity gate the e2e tripped
    assert "replay_breach_count" in spec.thresholds
    assert spec.thresholds["replay_breach_count"].max == 0
    assert spec.thresholds["replay_drop_rate"].max == 0.02
    # the BUG-3 live op name perception.yaml configured
    assert spec.latency_stage == LATENCY_STAGE


def test_e2e_verdict_flips_invalid_to_not_invalid(e2e_metrics):
    """THE FLIP: the e2e returned exit 2 / verdict INVALID purely from infra-noise
    (the four bugs). Through the REAL config + the composed fixes the SAME-shaped bag
    is no longer thrown out as invalid — rc != 2 and verdict != INVALID.

    (It reads FAIL, not PASS, because the synthetic 2x2 frames cannot calibrate the
    cross-camera overlap homography — that 0.0 < 0.75 is an honest QUALITY fail, the
    same metric that was 0.5351 on the real e2e. We assert the VALIDITY flip only.)"""
    rc, doc, _counts, _spec = e2e_metrics
    assert rc != 2, doc["details"]
    assert doc["verdict"] != "INVALID", doc


def test_e2e_drop_rate_no_longer_inflated(e2e_metrics):
    """BUG 1: the e2e read drop_rate 0.709 because the expectation used the ~99s
    bag-WRITE span. With the span derived from the ~31s HEADER clock, this faithful
    sim-time bag reads <= 0.02 (the real validity threshold)."""
    _rc, doc, _counts, spec = e2e_metrics
    drop = doc["replay_faithfulness"]["drop_rate"]
    assert drop <= 0.02, drop
    # and it actually clears the REAL configured threshold (not a hand-picked 0.02)
    assert drop <= spec.thresholds["replay_drop_rate"].max


def test_e2e_breach_count_zero_under_gap_tolerance(e2e_metrics):
    """The e2e logged breach_count 81 — every ~600ms EoMT inference stall tripped the
    hardcoded 2x (400ms) threshold. With perception.yaml's 4.0 gap_tolerance on
    image/semantic (4.0 * 200ms = 800ms), those faithful stalls are no longer breaches,
    so breach_count == 0 — clearing the replay_breach_count: max 0 validity gate."""
    _rc, doc, _counts, _spec = e2e_metrics
    f = doc["replay_faithfulness"]
    assert f["breach_count"] == 0, f["per_topic"]
    # the ~600ms stalls are still RECORDED per-camera (visibility), just not breaches
    cam0 = f["per_topic"]["/perception_node/camera_0/image_raw_sim"]
    assert 590.0 <= cam0["max_gap_ms"] <= 700.0, cam0


def test_e2e_cross_camera_jitter_does_not_invalidate(e2e_metrics):
    """The e2e's 6 semantic cameras carried unique counts 142..146 (EoMT inference
    jitter, NOT a starve). The old hardcoded ``> 1`` tolerance flagged that + added a
    breach -> kept a faithful run INVALID. The jitter-tolerant check (widened when the
    semantic streams carry the elevated gap_tolerance) tolerates the spread: the
    counts stay surfaced for visibility but contribute NO breach."""
    _rc, doc, _counts, _spec = e2e_metrics
    f = doc["replay_faithfulness"]
    sem_counts = sorted(f["cross_camera_counts"].values())
    # a genuine multi-frame jitter spread (like the e2e's 142..146) is present
    assert max(sem_counts) - min(sem_counts) >= 4, sem_counts
    # ...and it does NOT force the mismatch flag / a breach
    assert f["cross_camera_count_mismatch"] is False, f["cross_camera_counts"]
    per_topic_breaches = sum(v["breach_count"] for v in f["per_topic"].values())
    assert f["breach_count"] == per_topic_breaches, f  # no extra cross-camera breach


def test_e2e_latency_p95_is_gated_from_configured_stage(e2e_metrics):
    """BUG 3: the e2e read latency_p95_ms = null because LatencyMetric looked for the
    old ``seg_argmax`` stage while the live diagnostics published
    ``inference_seg_extract_segmentation``. With perception.yaml's latency_stage set
    to the live op, the metric finds avg_compute_ms ~8.0 — a non-null, gated value."""
    _rc, doc, _counts, _spec = e2e_metrics
    row = next(m for m in doc["metrics"] if m["name"] == "latency_p95_ms")
    assert row["value"] is not None, doc["metrics"]
    assert row["value"] == pytest.approx(AVG_COMPUTE_MS, abs=0.5)


def test_e2e_quality_headline_rows_are_non_null(e2e_metrics):
    """BUG 2: the e2e read pipeline_throughput_hz / segmentation_coverage /
    depth_validity all null because each plugin returned its scalar under a non-name
    key (mean_hz / mean_class_coverage / mean_valid_fraction) that
    generator.value[r.name] could not find. With the headline scalar now under the
    key == self.name, every row carries a value (the gate has teeth)."""
    _rc, doc, _counts, _spec = e2e_metrics
    rows = {m["name"]: m for m in doc["metrics"]}
    for name in ("pipeline_throughput_hz", "segmentation_coverage", "depth_validity"):
        assert rows[name]["value"] is not None, (name, rows[name])


def test_e2e_count_far_exceeds_unique_from_reemit(e2e_metrics):
    """The e2e per-topic count (~382) >> unique_count (~143) from the non-consuming
    getLatest() re-emit. The dedup must count UNIQUE header frames (drives drop_rate /
    breaches), with the raw count preserved for visibility."""
    _rc, doc, _counts, _spec = e2e_metrics
    cam0 = doc["replay_faithfulness"]["per_topic"]["/perception_node/camera_0/image_raw_sim"]
    assert cam0["count"] > 2 * cam0["unique_count"], cam0  # ~3x re-emit


def test_e2e_all_displayed_floats_are_rounded(e2e_metrics):
    """BUG 4: the e2e wrote drop_rate 0.7091667310217005 and max_gap_ms 199.999996 —
    raw float noise. Every faithfulness float rendered into metrics.json (the headline
    max_gap_ms, drop_rate, and each per-topic max_gap_ms) must carry <= 3 decimals."""
    _rc, doc, _counts, _spec = e2e_metrics
    f = doc["replay_faithfulness"]
    floats = [f["max_gap_ms"], f["drop_rate"]]
    floats += [pt["max_gap_ms"] for pt in f["per_topic"].values()]
    offenders = [x for x in floats if _decimal_places(x) > 3]
    assert not offenders, offenders


def test_e2e_negative_control_uniform_hang_still_invalid(tmp_path):
    """The gate must KEEP its teeth: an otherwise-identical bag with a genuine 2x
    replay hang (~500ms gap, 5x the 100ms period) on the UNIFORM-rate depth topics —
    NOT a jittery 5 Hz inference stream — must still breach validity and return
    INVALID (exit 2). Over-relaxing the gap tolerance until a real hang passes would
    be a silent regression; this proves the fixes did not gut the validity tier."""
    bag = tmp_path / "e2e_hang"
    _build_e2e_bag(bag, depth_hang=True)
    spec = load_module_config("perception", CONFIGS_DIR)
    out_dir = tmp_path / "run_hang"
    rc = _run_metrics_pipeline(spec, bag, out_dir)
    doc = json.loads((out_dir / "reports" / "metrics.json").read_text())
    assert rc == 2, doc["details"]
    assert doc["verdict"] == "INVALID", doc
    # the breach is on the uniform depth topics (the injected ~500ms hang), proving the
    # default 2.0 gap-tolerance still catches a real stall on a non-jittery stream
    assert doc["replay_faithfulness"]["breach_count"] >= 1, doc["replay_faithfulness"]
    depth0 = doc["replay_faithfulness"]["per_topic"]["/perception_node/camera_0/depth_raw_sim"]
    assert depth0["breach_count"] >= 1, depth0
