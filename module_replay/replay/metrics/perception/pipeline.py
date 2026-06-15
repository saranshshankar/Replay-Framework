"""Perception pipeline throughput metric (MTRC-02).

Ported from the PoC ``perception_metrics/metrics/pipeline_metrics.py``. The PoC
broke the pipeline down by diagnostics stage (``/perception_node/diagnostics``,
which the replay output catalog does not always carry); here we re-implement the
throughput proxy the PoC also computed — the effective publish rate per output
topic from ``np.diff`` of bag-write timestamps — against the framework's
read-once ``BagReader``. This is the topic-level cadence the report generator
gates on, and it works on any output bag without the diagnostics topic.
"""
from __future__ import annotations

import numpy as np

from replay.metrics.base import BaseMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric


@register_metric("perception")
class PipelineMetric(BaseMetric):
    """Per-output-topic effective message rate (Hz) from inter-message intervals."""

    name = "pipeline_throughput_hz"
    requires_baseline = False

    def compute(self, reader: BagReader, config: dict) -> dict:
        output_topics = config.get("output_topics", [])

        per_topic: dict[str, dict] = {}
        for topic in output_topics:
            msgs = reader.get_messages(topic)
            if len(msgs) < 2:
                # Graceful degradation: a single (or zero) message gives no
                # interval; report a typed zero entry rather than raising.
                per_topic[topic] = {
                    "num_messages": len(msgs),
                    "mean_hz": 0.0,
                    "mean_interval_ms": 0.0,
                }
                continue
            ts = np.array([t for t, _ in msgs], dtype=np.int64)
            intervals_ms = np.diff(ts) / 1e6
            mean_interval_ms = float(np.mean(intervals_ms))
            mean_hz = round(1000.0 / mean_interval_ms, 3) if mean_interval_ms > 0 else 0.0
            per_topic[topic] = {
                "num_messages": len(msgs),
                "mean_hz": mean_hz,
                "mean_interval_ms": round(mean_interval_ms, 3),
            }

        hz_values = [v["mean_hz"] for v in per_topic.values() if v["num_messages"] >= 2]
        return {
            "per_topic": per_topic,
            "mean_hz": round(float(np.mean(hz_values)), 3) if hz_values else 0.0,
        }
