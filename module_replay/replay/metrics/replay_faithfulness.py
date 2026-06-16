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
    def _unique_stamps_ns(messages) -> tuple[np.ndarray, int]:
        """Return (sorted unique stamps in ns, raw message count) for a topic.

        Dedup is by the message HEADER stamp when the decoded message exposes
        ``header.stamp`` (sec/nanosec -> ns) -- the contract's stamp-level dedup
        (``getLatest()`` is non-consuming, ``topic_reader.hpp:84-88``, and can re-emit
        the same stamp, inflating a raw message count). Falls back to the bag-write
        timestamp when no header stamp is present. The raw count is preserved
        alongside ``unique_count`` so a large raw/unique gap stays visible.
        """
        raw_count = len(messages)
        stamps: list[int] = []
        for write_ts, msg in messages:
            hdr = getattr(msg, "header", None)
            stamp = getattr(hdr, "stamp", None) if hdr is not None else None
            if stamp is not None and hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
                stamps.append(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
            else:
                stamps.append(int(write_ts))  # fallback: bag-write timestamp
        if not stamps:
            return np.empty(0, dtype=float), raw_count
        return np.unique(np.array(stamps, dtype=float)), raw_count

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
        """Validity-tier faithfulness with per-topic rate + the contract's structural checks.

        Structural checks (KT/PERCEPTION-REPLAY-CONTRACT.md §5 "Also validated structurally"):
          * UNIQUE-STAMP DEDUP -- gap/count math is over de-duplicated header stamps so a
            non-consuming ``getLatest()`` re-emit cannot inflate counts. ``unique_count`` and
            the raw ``count`` are both recorded per topic.
          * EQUAL CROSS-CAMERA COUNTS -- the 6 semantic cameras (``semantic_raw_sim``) must
            carry equal unique counts (TimeSync emits complete 6-sets only, so inequality is a
            red flag). DECISION: a mismatch is surfaced as ``cross_camera_count_mismatch=True``
            with the per-camera ``cross_camera_counts`` AND contributes a breach -- a partial
            starve (each camera still has >=2 frames) would otherwise NOT trip the C1
            empty-topic rule, so without a breach contribution it could pass silently. This
            mirrors the WR-03 fail-closed / no-silent-pass philosophy (01-11).
        """
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
                    "per_topic": {t: {"max_gap_ms": 1e9, "breach_count": 1, "count": 0, "unique_count": 0}
                                  for t in topics},
                    "empty_topics": list(topics), "cross_camera_count_mismatch": False,
                    "cross_camera_counts": {}, "tier": "validity"}
        per_topic, empty_topics = {}, []
        worst_gap, total_breaches = 0.0, 0
        total_actual, total_expected = 0, 0
        # The headline max_gap_ms is gated by the flat replay_max_gap_ms threshold (200 ms =
        # 2x the 10 Hz period), which is only meaningful for UNIFORM-rate topics. Slow
        # housekeeping topics (e.g. /diagnostics at 0.2 Hz, a legitimate ~5000 ms interval)
        # are EXCLUDED from worst_gap so a healthy mixed-rate run does not false-INVALID
        # (UAT gap-1 residual). They are still gated PER TOPIC via breach_count below
        # (the replay_breach_count validity threshold), so a genuine slow-topic stall is caught.
        default_hz = self._resolve_default_hz(expected_hz_map)
        for topic in topics:
            # PER-TOPIC rate: cameras 10 Hz (default), diagnostics 0.2 Hz, etc.
            hz = self._expected_hz_for(topic, expected_hz_map)
            is_uniform = hz >= default_hz   # slow topics excluded from the headline max_gap_ms
            expected_period_ms = 1000.0 / hz
            expected_per_topic = max(int(round(span_s * hz)), 1)
            total_expected += expected_per_topic
            # UNIQUE-STAMP DEDUP before the gap/drop math (re-emit must not inflate counts).
            ts, raw_count = self._unique_stamps_ns(reader.get_messages(topic))
            if ts.size < 2:
                # empty/singleton topic over a non-trivial span = a validity breach, not a free pass
                per_topic[topic] = {"max_gap_ms": 1e9, "breach_count": 1,
                                    "count": raw_count, "unique_count": int(ts.size)}
                empty_topics.append(topic)
                if is_uniform:
                    worst_gap = max(worst_gap, 1e9)
                total_breaches += 1
                total_actual += int(ts.size)
                continue
            intervals_ms = np.diff(ts) / 1e6   # ts is already sorted-unique
            max_gap = float(np.max(intervals_ms))
            # breach threshold is PER-TOPIC: 2 * this topic's nominal period
            # (diagnostics at 0.2 Hz => 10000 ms, not the cameras' 200 ms).
            breaches = int(np.sum(intervals_ms > 2 * expected_period_ms))
            per_topic[topic] = {"max_gap_ms": max_gap, "breach_count": breaches,
                                "count": raw_count, "unique_count": int(ts.size)}
            if is_uniform:
                worst_gap = max(worst_gap, max_gap)
            total_breaches += breaches
            total_actual += int(ts.size)   # drop_rate counts UNIQUE frames, not re-emits
        # EQUAL CROSS-CAMERA COUNTS: the 6 semantic streams must agree (within 1 frame).
        cross_camera_counts = {
            t: per_topic[t]["unique_count"] for t in topics if "semantic_raw_sim" in t
        }
        cross_camera_count_mismatch = False
        if len(cross_camera_counts) >= 2:
            counts = list(cross_camera_counts.values())
            # tolerance: all within 1 frame of each other (BestEffort startup jitter)
            if (max(counts) - min(counts)) > 1:
                cross_camera_count_mismatch = True
                total_breaches += 1   # fail-closed: a partial starve must not pass silently
        drop_rate = ((total_expected - total_actual) / total_expected) if total_expected else 0.0
        return {
            "max_gap_ms": worst_gap,
            "breach_count": total_breaches,
            "drop_rate": float(min(max(drop_rate, 0.0), 1.0)),
            "per_topic": per_topic,
            "empty_topics": empty_topics,
            "cross_camera_count_mismatch": cross_camera_count_mismatch,
            "cross_camera_counts": cross_camera_counts,
            "tier": "validity",
        }
