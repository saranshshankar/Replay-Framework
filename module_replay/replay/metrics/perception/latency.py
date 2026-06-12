"""Perception input -> output publish latency metric (MTRC-02).

Ported from the PoC ``perception_metrics/metrics/latency_metrics.py`` (10xCode
branch ``aniket/feat/module_wise_replay_poc``, which uses ``rosbags`` — A2:
offline pure-Python, no ROS runtime client library). The PoC measured per-topic
inter-message *interval* stats from its own
multi-topic BagReader; here we re-implement against the framework's read-once
``BagReader`` (``get_messages`` / ``iter_paired``) and report the input->output
publish *delta* per matched pair, which is the per-frame replay latency the
perception output bag exposes (output carries the input frame's capture stamp —
JOIN RULE — so pairing is by bag-write timestamp within tolerance).

The reported keys (p50/p95/p99) use ``numpy.percentile`` exactly as the PoC's
``_compute_interval_stats`` did.
"""
from __future__ import annotations

import numpy as np

from replay.metrics.base import BaseMetric
from replay.metrics.bag_reader import BagReader
from replay.metrics.registry import register_metric


@register_metric("perception")
class LatencyMetric(BaseMetric):
    """Input->output publish latency (percentile ms) across paired frames."""

    name = "latency_p95_ms"
    requires_baseline = False

    def compute(self, reader: BagReader, config: dict) -> dict:
        input_topics = config.get("input_topics", [])
        output_topics = config.get("output_topics", [])

        deltas_ms: list[float] = []
        for in_topic, out_topic in zip(input_topics, output_topics):
            msgs_out = reader.get_messages(out_topic)
            if not msgs_out:
                continue
            ts_out = np.array([t for t, _ in msgs_out], dtype=np.int64)
            for ts_in, _msg_in, _msg_out in reader.iter_paired(in_topic, out_topic):
                # iter_paired hands us the matched output message but not its
                # bag-write timestamp; recover it as the nearest output ts to
                # this input ts (same searchsorted contract iter_paired uses).
                idx = int(np.searchsorted(ts_out, ts_in))
                best = None
                for ci in (idx - 1, idx):
                    if 0 <= ci < len(ts_out):
                        if best is None or abs(int(ts_out[ci]) - ts_in) < abs(int(ts_out[best]) - ts_in):
                            best = ci
                if best is None:
                    continue
                deltas_ms.append((int(ts_out[best]) - ts_in) / 1e6)

        if not deltas_ms:
            # Graceful degradation: no matched pairs (e.g. a degenerate bag with
            # no output topic). Return a well-formed zero result rather than
            # raising, so the report generator still gets a typed dict.
            return {
                "num_pairs": 0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "mean_ms": 0.0,
                "max_ms": 0.0,
            }

        arr = np.array(deltas_ms, dtype=np.float64)
        return {
            "num_pairs": int(arr.size),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
        }
