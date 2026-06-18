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
    def _header_stamps_ns(messages) -> list[int]:
        """Extract per-message HEADER stamps (sec/nanosec -> ns) for a topic.

        Uses the decoded message's ``header.stamp`` (the robot clock) when present,
        falling back to the bag-write timestamp when no header stamp is exposed. This
        is the SINGLE source of the header-clock basis: ``_unique_stamps_ns`` dedups
        it for the per-topic counts, and ``compute`` takes the global min/max for the
        span (BUG 1 fix -- span and counts must share the header clock, not the
        ~4x-larger bag-write clock that ``ros2 bag record`` stamps).
        """
        stamps: list[int] = []
        for write_ts, msg in messages:
            hdr = getattr(msg, "header", None)
            stamp = getattr(hdr, "stamp", None) if hdr is not None else None
            if stamp is not None and hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
                stamps.append(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))
            else:
                stamps.append(int(write_ts))  # fallback: bag-write timestamp
        return stamps

    @classmethod
    def _unique_stamps_ns(cls, messages) -> tuple[np.ndarray, int]:
        """Return (sorted unique stamps in ns, raw message count) for a topic.

        Dedup is by the message HEADER stamp (via ``_header_stamps_ns``) -- the
        contract's stamp-level dedup (``getLatest()`` is non-consuming,
        ``topic_reader.hpp:84-88``, and can re-emit the same stamp, inflating a raw
        message count). The raw count is preserved alongside ``unique_count`` so a
        large raw/unique gap stays visible.
        """
        raw_count = len(messages)
        stamps = cls._header_stamps_ns(messages)
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

    @staticmethod
    def _gap_factor_for(topic: str, gap_tolerance) -> float:
        """Return the per-topic gap-tolerance FACTOR for ``topic`` (default 2.0).

        A breach is ``interval > factor * expected_period``. ``gap_tolerance`` is the
        01-19 substring->factor map (e.g. ``{"default": 2.0, "image_raw_sim": 4.0}``);
        the first non-"default" key that is a SUBSTRING of the topic wins, else the
        "default" entry (fallback 2.0). Mirrors ``_expected_hz_for`` exactly so the two
        maps share one matching shape. image/semantic carry 4.0 (their FAITHFUL ~600ms
        EoMT inference stalls are ~3.2x the 200ms period); a uniform stream stays 2.0,
        so a genuine replay hang still breaches. A non-dict / absent map -> 2.0
        (back-compat: every pre-01-20 test runs at exactly the old hardcoded 2x).
        """
        if not isinstance(gap_tolerance, dict):
            return 2.0
        for key, factor in gap_tolerance.items():
            if key == "default":
                continue
            if key in topic:
                return float(factor)
        return float(gap_tolerance.get("default", 2.0))

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
        # 01-19/01-20 per-topic gap-tolerance map (substring->factor, e.g.
        # {"default": 2.0, "image_raw_sim": 4.0}). The per-topic breach is
        # ``interval > factor * expected_period``; image/semantic carry 4.0 so their
        # FAITHFUL ~600ms EoMT inference stalls don't trip while a uniform-stream hang
        # still does. Absent map -> _gap_factor_for returns 2.0 (the old hardcoded 2x).
        gap_tolerance = config.get("gap_tolerance", {})
        # Denominator guard (C1: one starved camera zeroes ALL outputs -- that collapse must
        # BREACH validity, never pass vacuously): expectations use the OVERALL bag span, so
        # an empty/near-empty topic contributes its full expected count as drops.
        # BUG 1: the span MUST come from HEADER stamps (the robot clock), the SAME basis as
        # the per-topic unique counts below. ``ros2 bag record`` stamps the wall-clock write
        # time (a ~99s span in the e2e) while the recorded header carries the robot clock
        # (~27s). A write-clock span makes expected = span * hz ~3.6x too large, so a faithful
        # sim-time replay reads drop_rate ~0.7 (e2e: 0.709). Header stamps keep both sides aligned.
        #
        # BUG 1 residual: the span basis EXCLUDES the diagnostics housekeeping topic. Diagnostics
        # publishes on a WALL-clock timer (report_interval_sec) so its header stamps cover a
        # LONGER window (~100s: 20 msgs @ 0.2Hz) than the perception camera/depth output (~27s).
        # Including it makes the global span ~100s and inflates every output topic's
        # expected = span * hz, so a faithful run false-INVALIDs (drop_rate ~0.7). The span must be
        # the window the perception OUTPUT covers; diagnostics is still drop/gap-checked per-topic
        # below. Falls back to all topics if excluding diagnostics leaves nothing (degenerate bag).
        diag_topic = config.get("diagnostics_topic")
        span_topics = [t for t in topics if t != diag_topic] or topics
        all_header_ns = [s for topic in span_topics for s in self._header_stamps_ns(reader.get_messages(topic))]
        span_s = ((max(all_header_ns) - min(all_header_ns)) / 1e9) if len(all_header_ns) >= 2 else 0.0
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
            # PER-TOPIC gap-tolerance factor (image/semantic 4.0, else default 2.0).
            gap_factor = self._gap_factor_for(topic, gap_tolerance)
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
            # breach threshold is PER-TOPIC: gap_factor * this topic's nominal period
            # (diagnostics at 0.2 Hz => 2x 5000 = 10000 ms; image/semantic at 5 Hz with a
            # 4.0 factor => 800 ms, so their ~600ms EoMT inference stalls are not breaches).
            breaches = int(np.sum(intervals_ms > gap_factor * expected_period_ms))
            per_topic[topic] = {"max_gap_ms": max_gap, "breach_count": breaches,
                                "count": raw_count, "unique_count": int(ts.size)}
            if is_uniform:
                worst_gap = max(worst_gap, max_gap)
            total_breaches += breaches
            total_actual += int(ts.size)   # drop_rate counts UNIQUE frames, not re-emits
        # EQUAL CROSS-CAMERA COUNTS: the 6 semantic streams must agree. TimeSync emits
        # complete 6-sets, so a GROSS mismatch is a partial-starve red flag. But the e2e's
        # 6 cameras carried unique counts 142,143,144,145,146,146 (spread 4) from EoMT
        # inference jitter, NOT a starve — the old hardcoded ``> 1`` tolerance flagged that
        # + added a breach -> a FAITHFUL run stuck INVALID. Option (a) (01-20): when the
        # semantic streams carry an ELEVATED gap_tolerance (> 2.0, i.e. the jittery
        # inference streams), widen the count tolerance proportionally —
        # ``max(round(0.1 * max_count), 6)`` — so a <= ~5-frame jitter passes while a gross
        # ~125-frame starve (10 vs ~135) STILL flags + breaches (fail-closed survives).
        # Streams WITHOUT an elevated factor keep the strict ``> 1`` tolerance.
        cross_camera_counts = {
            t: per_topic[t]["unique_count"] for t in topics if "semantic_raw_sim" in t
        }
        cross_camera_count_mismatch = False
        if len(cross_camera_counts) >= 2:
            counts = list(cross_camera_counts.values())
            sem_topics = [t for t in topics if "semantic_raw_sim" in t]
            sem_gap_factor = self._gap_factor_for(sem_topics[0], gap_tolerance)
            if sem_gap_factor > 2.0:
                # jitter band proportional to the stream depth, floored at 6 frames
                cross_camera_tol = max(int(round(0.1 * max(counts))), 6)
            else:
                cross_camera_tol = 1   # strict default (BestEffort startup jitter only)
            if (max(counts) - min(counts)) > cross_camera_tol:
                cross_camera_count_mismatch = True
                total_breaches += 1   # fail-closed: a partial starve must not pass silently
        drop_rate = ((total_expected - total_actual) / total_expected) if total_expected else 0.0
        # BUG 4: round for display/storage ONLY, at the assembled output — the threshold
        # comparisons (breach counts, the min/max clamp) already ran on raw precision.
        # drop_rate -> 3 decimals; the top-level max_gap_ms and each per-topic max_gap_ms
        # -> 1 decimal. The 1e9 empty-topic sentinels round harmlessly to 1e9.
        per_topic_rounded = {
            t: {**pt, "max_gap_ms": round(pt["max_gap_ms"], 1)} for t, pt in per_topic.items()
        }
        return {
            "max_gap_ms": round(worst_gap, 1),
            "breach_count": total_breaches,
            "drop_rate": round(float(min(max(drop_rate, 0.0), 1.0)), 3),
            "per_topic": per_topic_rounded,
            "empty_topics": empty_topics,
            "cross_camera_count_mismatch": cross_camera_count_mismatch,
            "cross_camera_counts": cross_camera_counts,
            "tier": "validity",
        }
