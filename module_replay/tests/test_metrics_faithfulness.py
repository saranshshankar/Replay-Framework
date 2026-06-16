from replay.metrics.bag_reader import BagReader
from replay.metrics.replay_faithfulness import ReplayFaithfulnessMetric

IN = "/perception_node/camera_0/image_raw"


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
