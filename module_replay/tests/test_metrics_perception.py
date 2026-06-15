"""Perception plugin-pack tests (MTRC-02, MOD-01).

Runs the ported perception metrics against the shared ``synthetic_bag`` fixture
(2x2 rgb8 Image messages on ``.../image_raw`` + ``.../image_raw_sim``). The
intrinsic metrics assert dict SHAPE and JSON-serializability — real per-pixel
values are validated in the MOD-01 manual e2e against a full perception output
bag, since the tiny synthetic frames are degenerate for segmentation/depth/overlap.
"""
import json

from replay.metrics.bag_reader import BagReader

IN = "/perception_node/camera_0/image_raw"
OUT = "/perception_node/camera_0/image_raw_sim"
CFG = {"input_topics": [IN], "output_topics": [OUT], "feature_matcher": "akaze"}


def test_latency_metric_shape(synthetic_bag):
    """MTRC-02: LatencyMetric.compute returns a dict with p95_ms (float)."""
    from replay.metrics.perception.latency import LatencyMetric

    reader = BagReader(synthetic_bag, [IN, OUT])
    out = LatencyMetric().compute(reader, CFG)
    assert "p95_ms" in out and isinstance(out["p95_ms"], float)
    json.dumps(out)  # must be JSON-serializable


def test_intrinsic_metrics_json_serializable(synthetic_bag):
    """MTRC-02: pipeline/segmentation/depth return JSON-serializable dicts."""
    from replay.metrics.perception.pipeline import PipelineMetric
    from replay.metrics.perception.segmentation import SegmentationMetric
    from replay.metrics.perception.depth import DepthMetric

    reader = BagReader(synthetic_bag, [IN, OUT])
    for cls in (PipelineMetric, SegmentationMetric, DepthMetric):
        json.dumps(cls().compute(reader, CFG))  # no exception


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
