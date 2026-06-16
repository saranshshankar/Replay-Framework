"""Replay-faithfulness validity-tier metric (RPLY-02 / RPLY-03).

Faithfulness distinguishes infra-noise (a stalled/dropping replay) from a real
quality regression. It is the VALIDITY tier: if ``max_gap_ms`` (or the drop rate)
exceeds the module's validity threshold, the replay is invalid and the run's
quality metrics are not trusted -- the gate fails on validity, not quality.

This metric is offline, pure-Python numpy compute over already-loaded timestamps
(no I/O, no deserialization here -- the BagReader did that). It imports rosbags
only transitively (via BagReader); nothing here touches the ROS runtime client.
"""
from __future__ import annotations

import numpy as np

from replay.metrics.base import BaseMetric


# NOTE: deliberately NOT @register_metric("perception") -- the CLI/report layer invokes this
# class explicitly as the validity tier (plan 01-07). Registering it would (a) make the
# perception pack count 8 and break 01-05's seven-plugin test in full-suite runs, and
# (b) run faithfulness a second time as a quality-row metric.
class ReplayFaithfulnessMetric(BaseMetric):
    name = "replay_faithfulness"
    requires_baseline = False
    tier = "validity"   # validity-tier: a breach invalidates the run

    @staticmethod
    def _resolve_default_hz(expected_hz) -> float:
        """Coerce cfg['expected_hz'] (a per-topic dict OR a bare scalar) to one float.

        A dict resolves to its 'default' entry (fallback 10.0); a scalar passes
        through. Retained as the back-compat seam: ``_expected_hz_for`` reuses the
        same dict|scalar tolerance, and callers passing a bare float still work.
        """
        if isinstance(expected_hz, dict):
            return float(expected_hz.get("default", 10.0))
        return float(expected_hz)

    @staticmethod
    def _expected_hz_for(topic: str, expected_hz) -> float:
        """Return the expected publish rate (Hz) for ``topic``.

        ``expected_hz`` is the 01-10 substring->Hz map (e.g.
        ``{"default": 10.0, "diagnostics": 0.2}``) OR a bare scalar (back-compat).
        For the dict form, the first key (other than "default") that is a SUBSTRING
        of the topic wins — so "diagnostics" matches "/perception_node/diagnostics"
        at 0.2 Hz, while every camera falls through to "default" (10 Hz). This is the
        gap-1 fix: the 0.2 Hz diagnostics topic is no longer scanned at a flat 10 Hz.
        """
        if not isinstance(expected_hz, dict):
            return float(expected_hz)
        for key, hz in expected_hz.items():
            if key == "default":
                continue
            if key in topic:
                return float(hz)
        return float(expected_hz.get("default", 10.0))

    def compute(self, reader, config: dict) -> dict:
        topics = config.get("output_topics") or config.get("input_topics") or []
        # 01-10 threads a per-topic map (e.g. {"default": 10.0, "diagnostics": 0.2})
        # into cfg["expected_hz"]; older callers/tests pass a bare scalar. Both shapes
        # flow through _expected_hz_for, which resolves the rate PER TOPIC (substring
        # match, "default" fallback). This is the UAT-gap-1 fix: diagnostics at 0.2 Hz
        # gets a 10000 ms breach threshold and a span*0.2 expected count, not a flat 10 Hz.
        expected_hz_map = config.get("expected_hz", 10.0)
        # Denominator guard (C1: one starved camera zeroes ALL outputs -- that collapse must
        # BREACH validity, never pass vacuously): expectations use the OVERALL bag span, so
        # an empty/near-empty topic contributes its full expected count as drops.
        all_ts = [t for topic in topics for t, _ in reader.get_messages(topic)]
        span_s = ((max(all_ts) - min(all_ts)) / 1e9) if len(all_ts) >= 2 else 0.0
        if span_s == 0.0:
            # Nothing (or a single message) in the whole output bag: maximally invalid.
            return {"max_gap_ms": 1e9, "breach_count": len(topics), "drop_rate": 1.0,
                    "per_topic": {t: {"max_gap_ms": 1e9, "breach_count": 1, "count": 0} for t in topics},
                    "empty_topics": list(topics), "tier": "validity"}
        per_topic, empty_topics = {}, []
        worst_gap, total_breaches = 0.0, 0
        total_actual, total_expected = 0, 0
        for topic in topics:
            # PER-TOPIC rate: cameras 10 Hz (default), diagnostics 0.2 Hz, etc.
            hz = self._expected_hz_for(topic, expected_hz_map)
            expected_period_ms = 1000.0 / hz
            expected_per_topic = max(int(round(span_s * hz)), 1)
            total_expected += expected_per_topic
            ts = np.array([t for t, _ in reader.get_messages(topic)], dtype=float)
            if ts.size < 2:
                # empty/singleton topic over a non-trivial span = a validity breach, not a free pass
                per_topic[topic] = {"max_gap_ms": 1e9, "breach_count": 1, "count": int(ts.size)}
                empty_topics.append(topic)
                worst_gap = max(worst_gap, 1e9)
                total_breaches += 1
                total_actual += int(ts.size)
                continue
            intervals_ms = np.diff(np.sort(ts)) / 1e6
            max_gap = float(np.max(intervals_ms))
            # breach threshold is PER-TOPIC: 2 * this topic's nominal period
            # (diagnostics at 0.2 Hz => 10000 ms, not the cameras' 200 ms).
            breaches = int(np.sum(intervals_ms > 2 * expected_period_ms))
            per_topic[topic] = {"max_gap_ms": max_gap, "breach_count": breaches, "count": int(ts.size)}
            worst_gap = max(worst_gap, max_gap)
            total_breaches += breaches
            total_actual += int(ts.size)
        drop_rate = ((total_expected - total_actual) / total_expected) if total_expected else 0.0
        return {
            "max_gap_ms": worst_gap,
            "breach_count": total_breaches,
            "drop_rate": float(min(max(drop_rate, 0.0), 1.0)),
            "per_topic": per_topic,
            "empty_topics": empty_topics,
            "tier": "validity",
        }
