"""Perception plugin-pack tests (MTRC-02, MOD-01).

Runs the ported perception metrics against the shared ``synthetic_bag`` fixture
(2x2 rgb8 Image messages on ``.../image_raw`` + ``.../image_raw_sim``). The
intrinsic metrics assert dict SHAPE and JSON-serializability — real per-pixel
values are validated in the MOD-01 manual e2e against a full perception output
bag, since the tiny synthetic frames are degenerate for segmentation/depth/overlap.
"""
import json

import numpy as np

from replay.metrics.bag_reader import BagReader

IN = "/perception_node/camera_0/image_raw"
OUT = "/perception_node/camera_0/image_raw_sim"
DIAG = "/perception_node/diagnostics"
CFG = {"input_topics": [IN], "output_topics": [OUT], "feature_matcher": "akaze"}


def test_latency_metric_skips_visibly_without_diagnostics(synthetic_bag):
    """Gap 4/5: the synthetic image-only bag has NO diagnostics topic, so latency
    must return a VISIBLE skipped marker (no false 0.0 pass), not a silent zero.

    The old input->output stamp-delta path is dropped — latency now parses the
    node's self-reported seg_argmax compute time from /perception_node/diagnostics.
    """
    from replay.metrics.perception.latency import LatencyMetric

    reader = BagReader(synthetic_bag, [IN, OUT])
    out = LatencyMetric().compute(reader, CFG)  # CFG has no diagnostics_topic
    json.dumps(out)  # must be JSON-serializable
    # Visible skip: either a {skipped: True} marker, or NO scalar latency_p95_ms key.
    # It must NOT silently report a numeric scalar (which generator.py would gate as a pass).
    assert out.get("skipped") is True
    assert out.get("latency_p95_ms") is None
    # Must NOT have fabricated a 0.0 pass on the dead stamp-delta path.
    assert out.get("p95_ms") in (None, 0.0) or "p95_ms" not in out


def test_latency_metric_parses_seg_argmax_scalar_key(diagnostics_bag):
    """Gap 5 (scalar-key gate fix): LatencyMetric parses seg_argmax avg_compute_ms
    from the diagnostics topic and emits a TOP-LEVEL scalar key 'latency_p95_ms'
    (== the metric name) — the exact key generator.py:91 (value[r.name]) reads to
    enforce the 50ms gate."""
    from replay.metrics.perception.latency import LatencyMetric

    avg_values = [40.0, 42.0, 45.0, 60.0]
    bag = diagnostics_bag(avg_values)
    reader = BagReader(bag, [DIAG])
    cfg = {"diagnostics_topic": DIAG}
    out = LatencyMetric().compute(reader, cfg)
    json.dumps(out)

    # The scalar gate key MUST exist at the top level and equal the p95 of avg_compute_ms.
    assert "latency_p95_ms" in out
    assert isinstance(out["latency_p95_ms"], float)
    expected_p95 = round(float(np.percentile(np.array(avg_values), 95)), 3)
    assert out["latency_p95_ms"] == expected_p95
    # PoC seg_argmax aggregate shape is also reported.
    assert out["num_windows"] == 4
    assert out["p50_ms"] == round(float(np.percentile(np.array(avg_values), 50)), 3)
    assert out["p99_ms"] == round(float(np.percentile(np.array(avg_values), 99)), 3)
    assert out.get("skipped") is not True


def test_latency_gate_enforces_end_to_end(diagnostics_bag, tmp_path):
    """Gap 4 (BLOCKER): wired through generate_report, the 50ms gate now ENFORCES.
    A run whose seg_argmax p95 > 50 returns exit 1; p95 <= 50 returns 0. Proves the
    dead gate (plugin emitted p95_ms, generator read latency_p95_ms) is now live."""
    from replay.metrics.perception.latency import LatencyMetric
    from replay.metrics.base import MetricResult
    from replay.module_config import ThresholdSpec
    from replay.metrics.report.generator import generate_report

    th = {"latency_p95_ms": ThresholdSpec(max=50.0, tolerance_band=0.0, tier="quality")}

    def _run(avg_values, name):
        bag = diagnostics_bag(avg_values, name=name)
        reader = BagReader(bag, [DIAG])
        val = LatencyMetric().compute(reader, {"diagnostics_topic": DIAG})
        mr = MetricResult(
            name="latency_p95_ms", module="perception", value=val,
            passed=False, is_regression=False,
        )
        out_dir = tmp_path / name
        return generate_report("perception", "t", [mr], out_dir, th)

    # p95 = 60 (all windows at 60ms) -> breach -> exit 1 (FAIL)
    assert _run([55.0, 58.0, 60.0, 60.0], "fail_run") == 1
    # p95 = 45 (all windows <= 45ms) -> pass -> exit 0
    assert _run([40.0, 42.0, 44.0, 45.0], "pass_run") == 0


def test_intrinsic_metrics_json_serializable(synthetic_bag):
    """MTRC-02: pipeline/segmentation/depth return JSON-serializable dicts."""
    from replay.metrics.perception.pipeline import PipelineMetric
    from replay.metrics.perception.segmentation import SegmentationMetric
    from replay.metrics.perception.depth import DepthMetric

    reader = BagReader(synthetic_bag, [IN, OUT])
    for cls in (PipelineMetric, SegmentationMetric, DepthMetric):
        json.dumps(cls().compute(reader, CFG))  # no exception


# ── BUG 2: headline scalar keyed by the metric name (generator value[r.name]) ──
# The report generator reads each plugin's headline scalar as ``value[r.name]``
# (generator.py:224). latency_p95_ms already follows that contract; pipeline /
# segmentation / depth did NOT, so all three rendered null/— in the e2e
# metrics.json. These tests pin the EXACT key the generator reads, additive to
# the existing detail keys downstream code + plots rely on.


def test_pipeline_emits_throughput_scalar_key(tmp_path):
    """PipelineMetric emits a top-level scalar 'pipeline_throughput_hz' == mean_hz,
    keeping the existing per_topic + mean_hz keys (generator value[r.name] seam)."""
    from replay.metrics.perception.pipeline import PipelineMetric

    bag = tmp_path / "pipethroughput"
    _write_image_bag(bag, [(RGB0, "rgb8", 2, 2, 3, 10, 100_000_000)])  # 10 Hz
    reader = BagReader(bag, [RGB0])
    out = PipelineMetric().compute(reader, {"output_topics": [RGB0]})
    json.dumps(out)
    # The headline scalar lives under the metric NAME so generator value[r.name] finds it.
    assert "pipeline_throughput_hz" in out
    assert isinstance(out["pipeline_throughput_hz"], float)
    assert out["pipeline_throughput_hz"] == out["mean_hz"]
    # Existing detail keys preserved (per-topic + the original mean_hz aggregate).
    assert "per_topic" in out
    assert "mean_hz" in out


def test_segmentation_emits_coverage_scalar_key(tmp_path):
    """SegmentationMetric emits a top-level scalar 'segmentation_coverage' in [0,1],
    keeping the per-class mean_class_coverage dict + temporal-consistency keys."""
    from replay.metrics.perception.segmentation import SegmentationMetric

    bag = tmp_path / "segcoverage"
    # A 2-frame rgba8 semantic stream (one topic -> one consistency pair).
    _write_image_bag(bag, [(SEM0, "rgba8", 4, 4, 4, 2, 100_000_000)])
    reader = BagReader(bag, [SEM0])
    out = SegmentationMetric().compute(reader, {"output_topics": [SEM0]})
    json.dumps(out)
    assert "segmentation_coverage" in out
    assert isinstance(out["segmentation_coverage"], float)
    assert 0.0 <= out["segmentation_coverage"] <= 1.0
    # The per-class coverage DICT (the headline scalar is derived from it) stays.
    assert "mean_class_coverage" in out
    assert isinstance(out["mean_class_coverage"], dict)
    assert "temporal_consistency_mean" in out


def test_depth_emits_validity_scalar_key(tmp_path):
    """DepthMetric emits a top-level scalar 'depth_validity' == mean_valid_fraction,
    keeping the existing detail keys."""
    from replay.metrics.perception.depth import DepthMetric

    bag = tmp_path / "depthvalidity"
    _write_image_bag(bag, [(DEPTH0, "32FC4", 4, 4, 4, 4, 100_000_000)])
    reader = BagReader(bag, [DEPTH0])
    out = DepthMetric().compute(reader, {"output_topics": [DEPTH0], "depth_topics": [DEPTH0]})
    json.dumps(out)
    assert "depth_validity" in out
    assert isinstance(out["depth_validity"], float)
    assert out["depth_validity"] == out["mean_valid_fraction"]
    assert "mean_valid_fraction" in out


def test_three_metrics_gate_through_generator(tmp_path):
    """End-to-end: wired through generate_report with min thresholds, pipeline /
    segmentation / depth each produce a row with a NON-NULL value and a
    True/False ``passed`` (not None) — i.e. the gate now SEES the scalar the
    plugin emits under its own name (the e2e null/— bug is gone)."""
    from replay.metrics.perception.pipeline import PipelineMetric
    from replay.metrics.perception.segmentation import SegmentationMetric
    from replay.metrics.perception.depth import DepthMetric
    from replay.metrics.base import MetricResult
    from replay.module_config import ThresholdSpec
    from replay.metrics.report.generator import generate_report

    bag = tmp_path / "gatebag"
    _write_image_bag(bag, [
        (RGB0, "rgb8", 2, 2, 3, 10, 100_000_000),
        (SEM0, "rgba8", 4, 4, 4, 2, 100_000_000),
        (DEPTH0, "32FC4", 4, 4, 4, 4, 100_000_000),
    ])
    reader = BagReader(bag, [RGB0, SEM0, DEPTH0])
    cfg = {"output_topics": [RGB0, SEM0, DEPTH0], "depth_topics": [DEPTH0]}

    pipe = PipelineMetric().compute(reader, cfg)
    seg = SegmentationMetric().compute(reader, cfg)
    dep = DepthMetric().compute(reader, cfg)

    results = [
        MetricResult(name="pipeline_throughput_hz", module="perception", value=pipe,
                     passed=False, is_regression=False),
        MetricResult(name="segmentation_coverage", module="perception", value=seg,
                     passed=False, is_regression=False),
        MetricResult(name="depth_validity", module="perception", value=dep,
                     passed=False, is_regression=False),
    ]
    th = {
        "pipeline_throughput_hz": ThresholdSpec(min=1.0, tolerance_band=0.0, tier="quality"),
        "segmentation_coverage": ThresholdSpec(min=0.0, tolerance_band=0.0, tier="quality"),
        "depth_validity": ThresholdSpec(min=0.0, tolerance_band=0.0, tier="quality"),
    }
    out_dir = tmp_path / "report"
    generate_report("perception", "t", results, out_dir, th)
    doc = json.loads((out_dir / "metrics.json").read_text())
    by_name = {row["name"]: row for row in doc["metrics"]}
    for name in ("pipeline_throughput_hz", "segmentation_coverage", "depth_validity"):
        assert by_name[name]["value"] is not None, name  # gate SEES the scalar
        assert by_name[name]["passed"] in (True, False), name  # evaluated, not skipped


def _write_image_bag(bag_dir, topic_specs):
    """Write a rosbag2 dir of Image messages from ``topic_specs``.

    Each spec: (topic, encoding, height, width, channels, count, period_ns).
    Returns the list of topics written. Used by the topic-scoping RED tests so a
    single bag can carry e.g. a 32FC4 depth topic + an rgb8 topic + a 0.2 Hz
    diagnostics-rate camera topic.
    """
    import numpy as _np
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    topics = []
    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        for topic, encoding, h, w, ch, count, period in topic_specs:
            conn = writer.add_connection(topic, Image.__msgtype__, typestore=typestore)
            topics.append(topic)
            bytes_per = 4 if encoding.startswith("32FC") else (2 if encoding.startswith("16UC") else 1)
            nbytes = h * w * ch * bytes_per
            for i in range(count):
                ts = i * period
                hdr = Header(
                    stamp=Time(sec=int(ts // 1_000_000_000), nanosec=int(ts % 1_000_000_000)),
                    frame_id="cam",
                )
                if encoding.startswith("32FC"):
                    # Valid finite positive depths so depth_validity is meaningful.
                    payload = _np.full(h * w * ch, 2.5, dtype=_np.float32).view(_np.uint8)
                else:
                    payload = _np.zeros(nbytes, dtype=_np.uint8)
                msg = Image(
                    header=hdr, height=h, width=w, encoding=encoding,
                    is_bigendian=0, step=w * ch * bytes_per, data=payload,
                )
                writer.write(conn, ts, typestore.serialize_cdr(msg, Image.__msgtype__))
    return topics


DEPTH0 = "/perception_node/camera_0/depth_raw_sim"
RGB0 = "/perception_node/camera_0/image_raw_sim"
SEM0 = "/perception_node/camera_0/semantic_raw_sim"
SEM1 = "/perception_node/camera_1/semantic_raw_sim"


def test_depth_scoped_to_depth_topics(tmp_path):
    """Gap 5: DepthMetric scans ONLY cfg['depth_topics'] — never rgb/semantic frames.

    A bag with 4 depth (32FC4) frames + 6 rgb8 frames: num_frames must equal the
    depth frame count (4), not depth+rgb (10), when depth_topics names only depth."""
    from replay.metrics.perception.depth import DepthMetric

    bag = tmp_path / "depthscope"
    _write_image_bag(bag, [
        (DEPTH0, "32FC4", 4, 4, 4, 4, 100_000_000),
        (RGB0, "rgb8", 4, 4, 3, 6, 100_000_000),
    ])
    reader = BagReader(bag, [DEPTH0, RGB0])
    cfg = {"output_topics": [DEPTH0, RGB0], "depth_topics": [DEPTH0]}
    out = DepthMetric().compute(reader, cfg)
    json.dumps(out)
    assert out["num_frames"] == 4  # only depth frames, not depth+rgb


def test_depth_self_filters_by_encoding_when_no_depth_topics(tmp_path):
    """Gap 5 fallback: with depth_topics empty, depth self-filters by 32FC encoding
    rather than decoding every rgb8 frame as float32 depth."""
    from replay.metrics.perception.depth import DepthMetric

    bag = tmp_path / "depthencfilter"
    _write_image_bag(bag, [
        (DEPTH0, "32FC4", 4, 4, 4, 3, 100_000_000),
        (RGB0, "rgb8", 4, 4, 3, 5, 100_000_000),
    ])
    reader = BagReader(bag, [DEPTH0, RGB0])
    # No depth_topics AND no depth_raw_sim in output -> must self-filter by encoding.
    cfg = {"output_topics": [DEPTH0, RGB0], "depth_topics": []}
    out = DepthMetric().compute(reader, cfg)
    json.dumps(out)
    assert out["num_frames"] == 3  # only the 32FC4 frames decoded as depth


def test_pipeline_mean_hz_excludes_diagnostics(tmp_path):
    """Gap 8: pipeline mean_hz must NOT average the 0.2 Hz diagnostics topic in with
    the 10 Hz camera topics. Per-topic detail may list it; the headline mean excludes it."""
    from replay.metrics.perception.pipeline import PipelineMetric

    DIAG_TOPIC = "/perception_node/diagnostics"
    bag = tmp_path / "pipescope"
    _write_image_bag(bag, [
        (RGB0, "rgb8", 2, 2, 3, 10, 100_000_000),       # 10 Hz
        (DIAG_TOPIC, "rgb8", 2, 2, 3, 4, 5_000_000_000),  # 0.2 Hz (stand-in)
    ])
    reader = BagReader(bag, [RGB0, DIAG_TOPIC])
    cfg = {"output_topics": [RGB0, DIAG_TOPIC], "diagnostics_topic": DIAG_TOPIC}
    out = PipelineMetric().compute(reader, cfg)
    json.dumps(out)
    # The 10 Hz camera mean must not be dragged toward ~5 Hz by the 0.2 Hz topic.
    assert out["mean_hz"] > 5.0
    assert abs(out["mean_hz"] - 10.0) < 0.5


def test_segmentation_no_cross_topic_prev_bleed(tmp_path):
    """Gap 8: SegmentationMetric resets prev at each topic boundary, so scanning two
    different semantic topics never produces a cross-topic consistency pair."""
    from replay.metrics.perception.segmentation import SegmentationMetric

    bag = tmp_path / "segscope"
    # Two semantic topics, ONE frame each: within-topic there are zero consistency
    # pairs; a cross-topic bleed would wrongly pair the two single frames.
    _write_image_bag(bag, [
        (SEM0, "rgba8", 4, 4, 4, 1, 100_000_000),
        (SEM1, "rgba8", 4, 4, 4, 1, 100_000_000),
    ])
    reader = BagReader(bag, [SEM0, SEM1])
    cfg = {"output_topics": [SEM0, SEM1]}
    out = SegmentationMetric().compute(reader, cfg)
    json.dumps(out)
    assert out["num_frames"] == 2
    # No within-topic pair (1 frame each) and NO cross-topic pair (prev reset).
    assert out["num_consistency_pairs"] == 0


def test_overlap_defaults_to_akaze(synthetic_bag):
    """Decision/MTRC-02: overlap uses AKAZE, not lightglue/torch."""
    from replay.metrics.perception.overlap import OverlapMetric

    reader = BagReader(synthetic_bag, [IN, OUT])
    # No feature_matcher key -> must default to akaze and run without torch.
    out = OverlapMetric().compute(reader, {"input_topics": [IN], "output_topics": [OUT]})
    assert "cross_camera_overlap_iou" in out  # ran without torch -> akaze default
    json.dumps(out)


# ── mask_iou_vs_golden: perception's ONE regression metric (01-15 / UAT gap 2) ──
# semantic_raw_sim is rgba8 with the class id in the R channel (contract §2). The
# helper below writes a baseline + candidate semantic bag where the candidate's
# class regions can be shifted, so a per-class IoU drop is provable. We keep the
# bags tiny but >= MIN region size so the IoU math is non-degenerate.
SEM_TOPIC = "/perception_node/camera_0/semantic_raw_sim"


def _write_semantic_bag(bag_dir, frames, topic=SEM_TOPIC):
    """Write a rosbag2 dir of rgba8 semantic Image frames.

    ``frames`` is a list of (H, W) int class-id arrays; the R channel carries the
    class id (G/B/A zeroed) — exactly the contract §2 semantic_raw_sim layout. The
    header stamp is the frame index * 100ms so candidate and baseline join by stamp.
    """
    import numpy as _np
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        conn = writer.add_connection(topic, Image.__msgtype__, typestore=typestore)
        for i, classid in enumerate(frames):
            classid = _np.asarray(classid, dtype=_np.uint8)
            h, w = classid.shape
            rgba = _np.zeros((h, w, 4), dtype=_np.uint8)
            rgba[..., 0] = classid  # R channel = class id
            ts = i * 100_000_000
            hdr = Header(
                stamp=Time(sec=int(ts // 1_000_000_000), nanosec=int(ts % 1_000_000_000)),
                frame_id="cam0",
            )
            msg = Image(
                header=hdr, height=h, width=w, encoding="rgba8",
                is_bigendian=0, step=w * 4, data=rgba.reshape(-1),
            )
            writer.write(conn, ts, typestore.serialize_cdr(msg, Image.__msgtype__))
    return topic


def _mask_iou_cfg():
    return {"output_topics": [SEM_TOPIC]}


def test_mask_iou_identical_is_one(tmp_path):
    """Identical candidate==baseline semantic frames -> mask_iou_vs_golden == 1.0,
    a top-level scalar in [0,1]."""
    from replay.metrics.perception.mask_iou import MaskIoUVsGoldenMetric

    import numpy as _np
    frame = _np.zeros((8, 8), dtype=_np.uint8)
    frame[:4, :] = 1  # class 1 occupies the top half; class 0 the bottom
    _write_semantic_bag(tmp_path / "cand", [frame, frame])
    _write_semantic_bag(tmp_path / "base", [frame, frame])

    cand = BagReader(tmp_path / "cand", [SEM_TOPIC])
    base = BagReader(tmp_path / "base", [SEM_TOPIC])
    out = MaskIoUVsGoldenMetric().compare(cand, base, _mask_iou_cfg())
    json.dumps(out)
    assert "mask_iou_vs_golden" in out
    assert 0.0 <= out["mask_iou_vs_golden"] <= 1.0
    assert out["mask_iou_vs_golden"] == 1.0
    assert out["num_frames"] > 0  # frames actually matched (not a vacuous 1.0)


def test_mask_iou_shifted_region_drops_below_one(tmp_path):
    """A candidate where one class region is shifted -> per-class mean IoU < 1.0
    but > 0 (boundary changes are real, not noise)."""
    from replay.metrics.perception.mask_iou import MaskIoUVsGoldenMetric

    import numpy as _np
    base_frame = _np.zeros((8, 8), dtype=_np.uint8)
    base_frame[:4, :] = 1                  # class 1 = top 4 rows
    cand_frame = _np.zeros((8, 8), dtype=_np.uint8)
    cand_frame[:6, :] = 1                  # class 1 region grown to top 6 rows (shifted boundary)

    _write_semantic_bag(tmp_path / "cand", [cand_frame])
    _write_semantic_bag(tmp_path / "base", [base_frame])
    cand = BagReader(tmp_path / "cand", [SEM_TOPIC])
    base = BagReader(tmp_path / "base", [SEM_TOPIC])
    out = MaskIoUVsGoldenMetric().compare(cand, base, _mask_iou_cfg())
    json.dumps(out)
    assert 0.0 < out["mask_iou_vs_golden"] < 1.0


def test_mask_iou_gate_enforces_end_to_end(tmp_path):
    """Wired through generate_report with min 0.98: a matching golden (IoU ~0.99)
    PASSES (exit 0), a regressed golden (IoU ~0.90) FAILS the gate (exit 1)."""
    from replay.metrics.perception.mask_iou import MaskIoUVsGoldenMetric
    from replay.metrics.base import MetricResult
    from replay.module_config import ThresholdSpec
    from replay.metrics.report.generator import generate_report

    import numpy as _np
    th = {"mask_iou_vs_golden": ThresholdSpec(min=0.98, tolerance_band=0.0, tier="quality")}

    def _run(cand_frame, base_frame, name):
        _write_semantic_bag(tmp_path / f"{name}_c", [cand_frame])
        _write_semantic_bag(tmp_path / f"{name}_b", [base_frame])
        cand = BagReader(tmp_path / f"{name}_c", [SEM_TOPIC])
        base = BagReader(tmp_path / f"{name}_b", [SEM_TOPIC])
        val = MaskIoUVsGoldenMetric().compare(cand, base, _mask_iou_cfg())
        mr = MetricResult(name="mask_iou_vs_golden", module="perception",
                          value=val, passed=False, is_regression=True)
        return generate_report("perception", "t", [mr], tmp_path / name, th), val["mask_iou_vs_golden"]

    # A 100x100 frame: flipping ~100 of 10000 pixels in one class region -> IoU ~0.99 -> PASS.
    base100 = _np.zeros((100, 100), dtype=_np.uint8)
    base100[:50, :] = 1
    pass_cand = base100.copy()
    pass_cand[50, :] = 1  # one extra row in class 1 -> tiny boundary flip -> IoU ~0.99
    rc_pass, iou_pass = _run(pass_cand, base100, "pass_run")
    assert iou_pass >= 0.98, iou_pass
    assert rc_pass == 0

    # Shift the boundary 10 rows -> region change -> IoU well below 0.98 -> FAIL exit 1.
    fail_cand = _np.zeros((100, 100), dtype=_np.uint8)
    fail_cand[:60, :] = 1
    rc_fail, iou_fail = _run(fail_cand, base100, "fail_run")
    assert iou_fail < 0.98, iou_fail
    assert rc_fail == 1


def test_mask_iou_empty_comparison_is_not_a_passing_scalar(tmp_path):
    """FAIL-CLOSED: zero comparable frames (mis-aligned/empty baseline) must NOT
    yield a passing 1.0 scalar that green-lights the gate."""
    from replay.metrics.perception.mask_iou import MaskIoUVsGoldenMetric

    import numpy as _np
    frame = _np.zeros((8, 8), dtype=_np.uint8)
    frame[:4, :] = 1
    # Candidate has frames; baseline bag has the topic but ZERO messages -> no joins.
    _write_semantic_bag(tmp_path / "cand", [frame])
    _write_semantic_bag(tmp_path / "base", [])  # empty -> no comparable frames
    cand = BagReader(tmp_path / "cand", [SEM_TOPIC])
    base = BagReader(tmp_path / "base", [SEM_TOPIC])
    out = MaskIoUVsGoldenMetric().compare(cand, base, _mask_iou_cfg())
    json.dumps(out)
    assert out["num_frames"] == 0
    # The scalar must be None (or otherwise non-passing), never a false 1.0.
    assert out["mask_iou_vs_golden"] is None


def test_action_block_and_collision_box_deleted():
    """01-15 / UAT decision_locked: action_block + collision_box are removed from
    perception entirely (out-of-scope 3D/service outputs)."""
    import pytest

    with pytest.raises(ModuleNotFoundError):
        import replay.metrics.perception.action_block  # noqa: F401
    with pytest.raises(ModuleNotFoundError):
        import replay.metrics.perception.collision_box  # noqa: F401


def test_all_perception_plugins_registered():
    """MOD-01: importing the pack registers exactly 6 perception plugins (5 intrinsic
    + mask_iou_vs_golden). action_block + collision_box were dropped (01-15)."""
    import replay.metrics.perception  # noqa: F401  (triggers registration)
    from replay.metrics.registry import get_metric_plugins

    plugins = get_metric_plugins("perception")
    assert len(plugins) == 6
    names = {p.name for p in plugins}
    assert "mask_iou_vs_golden" in names
    assert "action_block_center_drift_mm" not in names
    assert "collision_box_iou" not in names
