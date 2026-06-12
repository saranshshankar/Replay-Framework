"""Perception plugin pack (MOD-01: perception is the framework-builder module).

Importing this package self-registers every perception metric under the
``"perception"`` module name via the register-metric decorator on each plugin
class. The report generator (plan 01-07) then discovers them with
``get_metric_plugins("perception")`` and keys verdicts by each plugin's ``name``
against ``configs/modules/perception.yaml`` thresholds.

INVARIANT (CLAUDE.md / PATTERNS § Key invariants #3): nothing here imports the
ROS runtime client library. Bag access is rosbags-only via ``BagReader``; image
math is OpenCV/numpy on already-deserialized messages.

The 5 intrinsic metrics (latency, pipeline, segmentation, depth, overlap) are
``BaseMetric`` subclasses; the 2 regression metrics (action-block drift,
collision-box IoU) are ``BaseRegressionMetric`` subclasses. Each import line
triggers registration as a side effect.
"""
from .latency import LatencyMetric
from .pipeline import PipelineMetric
from .segmentation import SegmentationMetric
from .depth import DepthMetric

__all__ = [
    "LatencyMetric",
    "PipelineMetric",
    "SegmentationMetric",
    "DepthMetric",
]
