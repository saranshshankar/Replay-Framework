from pathlib import Path

import numpy as np
import pytest

from replay.metrics.bag_reader import BagReader
from replay.metrics.replay_faithfulness import ReplayFaithfulnessMetric

IN = "/perception_node/camera_0/image_raw"


def _write_bag(bag_dir: Path, topic_stamps: dict[str, list[int]]) -> Path:
    """Write a rosbag2 dir with the given per-topic message write-timestamps (ns).

    Each message also carries a ``header.stamp`` equal to its write timestamp so
    stamp-based dedup can be exercised. ``topic_stamps`` maps a topic to the list
    of timestamps (one message per entry; duplicate stamps allowed).
    """
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        for topic, stamps in topic_stamps.items():
            conn = writer.add_connection(topic, Image.__msgtype__, typestore=typestore)
            for ts in stamps:
                sec = ts // 1_000_000_000
                nanosec = ts % 1_000_000_000
                hdr = Header(stamp=Time(sec=int(sec), nanosec=int(nanosec)), frame_id="c")
                msg = Image(
                    header=hdr, height=2, width=2, encoding="rgb8", is_bigendian=0,
                    step=6, data=np.zeros(12, dtype=np.uint8),
                )
                writer.write(conn, ts, typestore.serialize_cdr(msg, Image.__msgtype__))
    return bag_dir


def _hz_stamps(hz: float, span_s: float, start_ns: int = 0) -> list[int]:
    """Generate evenly-spaced write timestamps (ns) for a topic at ``hz`` over ``span_s``."""
    period_ns = int(round(1e9 / hz))
    n = int(round(span_s * hz)) + 1
    return [start_ns + i * period_ns for i in range(n)]


def test_faithfulness_keys_and_tier(synthetic_bag):
    """RPLY-02: faithfulness reports max_gap_ms / breach_count / drop_rate; validity tier."""
    reader = BagReader(synthetic_bag, [IN])
    out = ReplayFaithfulnessMetric().compute(reader, {"output_topics": [IN], "expected_hz": 10.0})
    assert {"max_gap_ms", "breach_count", "drop_rate"} <= set(out)
    assert out["tier"] == "validity"


def test_faithfulness_no_breach_on_clean_bag(synthetic_bag):
    """RPLY-03 foundation: a uniform 10Hz bag has 0 breaches and max_gap ~100ms."""
    reader = BagReader(synthetic_bag, [IN])
    out = ReplayFaithfulnessMetric().compute(reader, {"output_topics": [IN], "expected_hz": 10.0})
    assert out["breach_count"] == 0
    assert 90.0 <= out["max_gap_ms"] <= 110.0


def test_faithfulness_empty_topic_breaches(synthetic_bag):
    """C1 anti-vacuous rule: a configured topic with no messages must BREACH validity,
    not pass with zero gaps (the all-camera-collapse failure mode)."""
    missing = "/perception_node/camera_1/semantic_raw_sim"   # not in the synthetic bag
    reader = BagReader(synthetic_bag, [IN, missing])
    out = ReplayFaithfulnessMetric().compute(reader, {"output_topics": [IN, missing], "expected_hz": 10.0})
    assert missing in out["empty_topics"]
    assert out["breach_count"] >= 1
    assert out["drop_rate"] > 0.02   # breaches the replay_drop_rate validity threshold


def test_faithfulness_accepts_expected_hz_dict(synthetic_bag):
    """01-10 foundation: cfg['expected_hz'] is now a per-topic dict (from
    _build_metrics_cfg). compute() must accept the dict shape without crashing,
    resolving the 'default' rate. (True per-topic-rate logic lands in 01-12.)"""
    reader = BagReader(synthetic_bag, [IN])
    out = ReplayFaithfulnessMetric().compute(
        reader, {"output_topics": [IN], "expected_hz": {"default": 10.0, "diagnostics": 0.2}}
    )
    # Same result as the scalar-10.0 path: the 'default' entry resolves to 10 Hz.
    assert out["breach_count"] == 0
    assert 90.0 <= out["max_gap_ms"] <= 110.0


def test_resolve_default_hz_handles_both_shapes():
    """The coercion helper accepts a dict (-> 'default') or a bare scalar."""
    assert ReplayFaithfulnessMetric._resolve_default_hz({"default": 10.0, "diagnostics": 0.2}) == 10.0
    assert ReplayFaithfulnessMetric._resolve_default_hz({}) == 10.0  # empty dict -> 10.0 fallback
    assert ReplayFaithfulnessMetric._resolve_default_hz(7.5) == 7.5


# --- 01-12 Task 1: per-topic expected_hz (UAT gap 1 / Test 3) ---

CAM = "/perception_node/camera_0/image_raw_sim"
DIAG = "/perception_node/diagnostics"


def test_faithfulness_diagnostics_02hz_no_false_breach(tmp_path):
    """UAT gap 1 (blocker): a healthy bag with a 10 Hz camera AND a 0.2 Hz diagnostics
    topic must NOT breach. The diagnostics' ~5000ms inter-message interval is within its
    OWN 0.2 Hz threshold (2 * 1000/0.2 = 10000ms), not the camera's 200ms. Today a flat
    expected_hz=10 makes diagnostics 25x-breach on every run."""
    bag = _write_bag(
        tmp_path / "mixed",
        {CAM: _hz_stamps(10.0, span_s=10.0), DIAG: _hz_stamps(0.2, span_s=10.0)},
    )
    reader = BagReader(bag, [CAM, DIAG])
    out = ReplayFaithfulnessMetric().compute(
        reader,
        {"output_topics": [CAM, DIAG], "expected_hz": {"default": 10.0, "diagnostics": 0.2}},
    )
    assert out["breach_count"] == 0, out["per_topic"]
    # diagnostics' ~5000ms gap is recorded but is NOT a breach against its 0.2 Hz rate
    assert out["per_topic"][DIAG]["breach_count"] == 0
    assert 4900.0 <= out["per_topic"][DIAG]["max_gap_ms"] <= 5100.0


def test_faithfulness_drop_rate_per_topic_expectation(tmp_path):
    """drop_rate is computed against each topic's OWN expected count (diagnostics =
    span*0.2, cameras = span*10), so a healthy mixed-rate bag stays <= 0.02. A flat 10 Hz
    expectation for diagnostics would demand ~100 msgs where 3 are correct -> huge false drop."""
    bag = _write_bag(
        tmp_path / "mixed_drop",
        {CAM: _hz_stamps(10.0, span_s=10.0), DIAG: _hz_stamps(0.2, span_s=10.0)},
    )
    reader = BagReader(bag, [CAM, DIAG])
    out = ReplayFaithfulnessMetric().compute(
        reader,
        {"output_topics": [CAM, DIAG], "expected_hz": {"default": 10.0, "diagnostics": 0.2}},
    )
    assert out["drop_rate"] <= 0.02, out


def test_faithfulness_per_topic_rate_scalar_backcompat(tmp_path):
    """A bare-float expected_hz still works (older callers): uniform 10 Hz, 0 breaches."""
    bag = _write_bag(tmp_path / "scalar", {CAM: _hz_stamps(10.0, span_s=5.0)})
    reader = BagReader(bag, [CAM])
    out = ReplayFaithfulnessMetric().compute(reader, {"output_topics": [CAM], "expected_hz": 10.0})
    assert out["breach_count"] == 0
    assert 90.0 <= out["max_gap_ms"] <= 110.0
