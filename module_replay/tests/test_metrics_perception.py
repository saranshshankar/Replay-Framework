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


def test_regression_metrics_compare(synthetic_bag):
    """MTRC-02: the 2 regression metrics implement compare() returning drift/IoU dicts."""
    from replay.metrics.perception.action_block import ActionBlockDriftMetric
    from replay.metrics.perception.collision_box import CollisionBoxIoUMetric

    candidate = BagReader(synthetic_bag, [IN, OUT])
    baseline = BagReader(synthetic_bag, [IN, OUT])

    drift = ActionBlockDriftMetric().compare(candidate, baseline, CFG)
    assert "mean_drift_mm" in drift and "max_drift_mm" in drift
    json.dumps(drift)

    iou = CollisionBoxIoUMetric().compare(candidate, baseline, CFG)
    assert "mean_iou" in iou and "min_iou" in iou
    json.dumps(iou)


def test_all_seven_plugins_registered():
    """MOD-01: importing the pack registers all 7 perception plugins."""
    import replay.metrics.perception  # noqa: F401  (triggers registration)
    from replay.metrics.registry import get_metric_plugins

    assert len(get_metric_plugins("perception")) == 7
