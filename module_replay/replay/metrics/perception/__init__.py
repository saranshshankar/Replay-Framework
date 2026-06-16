"""Perception plugin pack (MOD-01: perception is the framework-builder module).

Importing this package self-registers every perception metric under the
``"perception"`` module name via the register-metric decorator on each plugin
class. The report generator (plan 01-07) then discovers them with
``get_metric_plugins("perception")`` and keys verdicts by each plugin's ``name``
against ``configs/modules/perception.yaml`` thresholds.

INVARIANT (CLAUDE.md / PATTERNS § Key invariants #3): nothing here imports the
ROS runtime client library. Bag access is rosbags-only via ``BagReader``; image
math is OpenCV/numpy on already-deserialized messages.

The pack is 6 plugins: 5 intrinsic ``BaseMetric`` subclasses (latency, pipeline,
segmentation, depth, overlap) + 1 regression ``BaseRegressionMetric`` —
``mask_iou_vs_golden`` (per-frame semantic-mask IoU vs a pinned golden, the
contract's PRIMARY C1 regression signal). Each import line triggers registration
as a side effect.

The earlier ``action_block_center_drift_mm`` and ``collision_box_iou`` regression
metrics were DROPPED (01-15 / UAT decision_locked 2026-06-13): they were 2D
pixel proxies of out-of-scope 3D / service outputs (action blocks = PoseStamped
meters, collision boxes = 3D OBB MarkerArray from the action_block_segmentation
service) on topics not even in ``perception.yaml``. They belong to a future
action-block module that replays that service node, not to perception's
feed-forward replay.
"""
from .latency import LatencyMetric
from .pipeline import PipelineMetric
from .segmentation import SegmentationMetric
from .depth import DepthMetric
from .overlap import OverlapMetric
from .mask_iou import MaskIoUVsGoldenMetric

__all__ = [
    "LatencyMetric",
    "PipelineMetric",
    "SegmentationMetric",
    "DepthMetric",
    "OverlapMetric",
    "MaskIoUVsGoldenMetric",
]
