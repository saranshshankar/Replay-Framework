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
