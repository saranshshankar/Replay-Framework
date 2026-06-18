from pathlib import Path

import numpy as np
import pytest

from replay.metrics.bag_reader import BagReader
from replay.metrics.replay_faithfulness import ReplayFaithfulnessMetric
from replay.metrics.report.generator import generate_report
from replay.module_config import ThresholdSpec, load_module_config

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


def _write_simtime_bag(bag_dir: Path, topic_specs: list[tuple[str, float, float, float]]) -> Path:
    """Write a bag where the BAG-WRITE clock and the HEADER (robot) clock are DECOUPLED.

    This reproduces the e2e's defining shape (BUG 1): ``ros2 bag record`` stamps each
    message with the wall-clock write time (a slow ~99s span), while every message's
    ``header.stamp`` carries the original recorded robot time (a faster ~24-27s span).
    The faithfulness span MUST be derived from the HEADER stamps (the same basis as the
    per-topic unique counts) -- using the write clock inflates ``expected = span * hz``.

    Each ``topic_specs`` entry is ``(topic, header_hz, write_hz, span_header_s)``: header
    stamps are spaced ``1/header_hz`` over ``span_header_s``; the SAME number of messages
    are written with bag-write timestamps spaced ``1/write_hz`` (a slower clock, so the
    write span is larger). ``header.stamp`` is set to the header ts, decoupled from write.
    """
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        for topic, header_hz, write_hz, span_header_s in topic_specs:
            header_stamps = _hz_stamps(header_hz, span_s=span_header_s)
            n = len(header_stamps)
            write_period_ns = int(round(1e9 / write_hz))
            write_stamps = [i * write_period_ns for i in range(n)]
            conn = writer.add_connection(topic, Image.__msgtype__, typestore=typestore)
            for hdr_ts, write_ts in zip(header_stamps, write_stamps):
                sec = hdr_ts // 1_000_000_000
                nanosec = hdr_ts % 1_000_000_000
                hdr = Header(stamp=Time(sec=int(sec), nanosec=int(nanosec)), frame_id="c")
                msg = Image(
                    header=hdr, height=2, width=2, encoding="rgb8", is_bigendian=0,
                    step=6, data=np.zeros(12, dtype=np.uint8),
                )
                # write_ts is the bag-write clock; header.stamp is the robot clock (decoupled)
                writer.write(conn, write_ts, typestore.serialize_cdr(msg, Image.__msgtype__))
    return bag_dir


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


# --- 01-20 Task 1: BUG 1 — span_s from HEADER stamps (sim-time drop_rate inflation) ---

DEPTH = "/perception_node/camera_0/depth_raw_sim"


def test_faithfulness_simtime_droprate_uses_header_span(tmp_path):
    """BUG 1: ``ros2 bag record`` stamps the bag-WRITE clock (~99s span) while every
    message's header carries the robot clock (~27s span). drop_rate = (expected-actual)
    / expected with expected = span * hz; if span comes from the WRITE clock,
    expected is ~3.6x inflated and a FAITHFUL run reads drop_rate ~0.7 (the e2e: 0.709).

    The fix derives span from HEADER stamps (the same basis as the per-topic unique
    counts), so a sim-time bag whose header span << write span reads drop_rate ~0.
    """
    header_span_s = 27.0
    write_span_s = 99.0
    # one Image per header frame; write clock spread over ~99s, header clock over ~27s.
    write_hz = (round(header_span_s * 5.0) + 1) / write_span_s   # same msg count, slower write clock
    bag = _write_simtime_bag(
        tmp_path / "simtime",
        [
            (CAM, 5.0, write_hz, header_span_s),     # image_raw_sim at header 5 Hz
            (DEPTH, 10.0, write_hz * 2, header_span_s),  # depth at header 10 Hz
        ],
    )
    reader = BagReader(bag, [CAM, DEPTH])
    out = ReplayFaithfulnessMetric().compute(
        reader,
        {"output_topics": [CAM, DEPTH],
         "expected_hz": {"default": 10.0, "image_raw_sim": 5, "semantic_raw_sim": 5}},
    )
    # BUG-1 proof: with a header-stamp span the expectation matches the header counts,
    # so a faithful sim-time bag is ~0 drop. A write-clock span would read ~0.7.
    assert out["drop_rate"] <= 0.05, out
    # Documentation of the inflation a write-clock span WOULD have produced (~3.6x):
    inflation = write_span_s / header_span_s
    assert inflation > 3.0  # the e2e write/header span ratio that drove drop_rate ~0.7


# --- 01-20 Task 2: configurable per-topic gap tolerance (locked validity policy) ---

UNIFORM = "/perception_node/colored_pointcloud_sim"  # default-rate stream (no hz/gap override)


def _stamps_with_gaps(base_hz: float, n_normal: int, gap_ms: float, n_gaps: int,
                      start_ns: int = 0) -> list[int]:
    """Evenly-spaced stamps at ``base_hz`` with ``n_gaps`` oversized ``gap_ms`` jumps inserted.

    Produces ``n_normal`` nominal-period intervals plus ``n_gaps`` deliberate stalls of
    ``gap_ms`` (e.g. the EoMT inference's ~600ms stalls on a 200ms-period 5 Hz stream).
    """
    period_ns = int(round(1e9 / base_hz))
    gap_ns = int(round(gap_ms * 1e6))
    stamps = [start_ns]
    for i in range(n_normal + n_gaps):
        step = gap_ns if i < n_gaps else period_ns   # front-load the big gaps
        stamps.append(stamps[-1] + step)
    return stamps


def test_gap_tolerance_factor_allows_inference_stalls(tmp_path):
    """image_raw_sim's FAITHFUL ~600ms EoMT inference stalls (~3.2x the 200ms period at
    5 Hz) must NOT breach when gap_tolerance is 4.0 (4.0 * 200ms = 800ms threshold). The
    e2e tripped these at the hardcoded 2x (400ms) -> breach_count 81. Configurable factor."""
    # 5 Hz (200ms) stream with three 600ms inference stalls
    bag = _write_bag(
        tmp_path / "stalls",
        {CAM: _stamps_with_gaps(5.0, n_normal=40, gap_ms=600.0, n_gaps=3)},
    )
    reader = BagReader(bag, [CAM])
    out = ReplayFaithfulnessMetric().compute(
        reader,
        {"output_topics": [CAM],
         "expected_hz": {"default": 10.0, "image_raw_sim": 5},
         "gap_tolerance": {"default": 2.0, "image_raw_sim": 4.0}},
    )
    assert out["per_topic"][CAM]["breach_count"] == 0, out["per_topic"][CAM]
    assert out["breach_count"] == 0, out  # no breach contribution from the 600ms stalls
    # the 600ms gap is still RECORDED (visibility), just not a breach against the 4.0 factor
    assert 590.0 <= out["per_topic"][CAM]["max_gap_ms"] <= 610.0


def test_gap_tolerance_default_two_still_breaches(tmp_path):
    """A genuine hang on a UNIFORM default-rate topic still breaches at factor 2.0. The
    elevated 4.0 factor is configured per-stream (image/semantic), NOT a blanket relax: a
    600ms gap on a 10 Hz (100ms-period) default topic is 6x period > 2.0*100ms = 200ms."""
    bag = _write_bag(
        tmp_path / "hang",
        {UNIFORM: _stamps_with_gaps(10.0, n_normal=50, gap_ms=600.0, n_gaps=1)},
    )
    reader = BagReader(bag, [UNIFORM])
    out = ReplayFaithfulnessMetric().compute(
        reader,
        {"output_topics": [UNIFORM],
         "expected_hz": {"default": 10.0, "image_raw_sim": 5},
         "gap_tolerance": {"default": 2.0, "image_raw_sim": 4.0}},
    )
    assert out["per_topic"][UNIFORM]["breach_count"] >= 1, out["per_topic"][UNIFORM]
    assert out["breach_count"] >= 1, out


def test_gap_factor_for_substring_and_default():
    """The new _gap_factor_for helper mirrors _expected_hz_for: first non-'default' key
    that is a substring of the topic wins; else 'default'; a non-dict map -> 2.0."""
    gt = {"default": 2.0, "image_raw_sim": 4.0, "semantic_raw_sim": 4.0}
    assert ReplayFaithfulnessMetric._gap_factor_for(
        "/perception_node/camera_0/image_raw_sim", gt) == 4.0
    assert ReplayFaithfulnessMetric._gap_factor_for(
        "/perception_node/camera_0/depth_raw_sim", gt) == 2.0   # falls through to default
    assert ReplayFaithfulnessMetric._gap_factor_for(
        "/perception_node/camera_0/depth_raw_sim", {}) == 2.0   # empty map -> 2.0 fallback
    assert ReplayFaithfulnessMetric._gap_factor_for("/anything", None) == 2.0  # non-dict -> 2.0


# --- 01-12 Task 2: structural validity checks (contract §5 / verification finding #8) ---

SEM = [f"/perception_node/camera_{i}/semantic_raw_sim" for i in range(6)]


def test_faithfulness_unique_stamp_dedup(tmp_path):
    """Contract §5 / finding #8: getLatest() is non-consuming and can re-emit a stamp,
    so counting must be over UNIQUE stamps. A topic with 10 messages but only 5 distinct
    stamps must report unique_count == 5 (not 10)."""
    period = 100_000_000  # 100 ms
    # 5 distinct stamps, each emitted twice (10 messages, 5 unique) -- simulates re-emit
    stamps = [period * (i // 2) for i in range(10)]
    assert len(stamps) == 10 and len(set(stamps)) == 5
    bag = _write_bag(tmp_path / "dup", {CAM: stamps})
    reader = BagReader(bag, [CAM])
    out = ReplayFaithfulnessMetric().compute(reader, {"output_topics": [CAM], "expected_hz": 10.0})
    assert out["per_topic"][CAM]["unique_count"] == 5, out["per_topic"][CAM]
    assert out["per_topic"][CAM]["count"] == 10  # raw count preserved for visibility


def test_faithfulness_cross_camera_count_mismatch_flagged(tmp_path):
    """Contract §5: TimeSync emits complete 6-sets only, so unequal semantic-camera counts
    are a red flag. compute() must surface cross_camera_count_mismatch=True when one camera
    has materially fewer frames than its peers."""
    topic_stamps = {t: _hz_stamps(10.0, span_s=5.0) for t in SEM}
    # starve camera 3: keep only the first 10 of ~51 frames
    topic_stamps[SEM[3]] = topic_stamps[SEM[3]][:10]
    bag = _write_bag(tmp_path / "mismatch", topic_stamps)
    reader = BagReader(bag, SEM)
    out = ReplayFaithfulnessMetric().compute(reader, {"output_topics": SEM, "expected_hz": 10.0})
    assert out["cross_camera_count_mismatch"] is True, out
    # the offending per-camera counts must be visible in the output
    assert "cross_camera_counts" in out


def test_faithfulness_cross_camera_equal_counts_not_flagged(tmp_path):
    """Equal counts across the 6 semantic cameras must NOT flag the mismatch."""
    topic_stamps = {t: _hz_stamps(10.0, span_s=5.0) for t in SEM}
    bag = _write_bag(tmp_path / "equal", topic_stamps)
    reader = BagReader(bag, SEM)
    out = ReplayFaithfulnessMetric().compute(reader, {"output_topics": SEM, "expected_hz": 10.0})
    assert out["cross_camera_count_mismatch"] is False, out
    assert out["breach_count"] == 0


# --- 01-20 Task 3: jitter-tolerant cross-camera check + BUG 4 decimal rounding ---


def test_cross_camera_inference_jitter_not_invalid(tmp_path):
    """The e2e's 6 semantic cameras carried unique counts 142,143,144,145,146,146 — a
    spread of 4 from EoMT inference jitter (NOT a starve; every camera has ~145 frames).
    The hardcoded ``> 1`` tolerance flagged this + added a breach -> kept a FAITHFUL run
    INVALID. With the streams carrying an elevated gap_tolerance (4.0), the cross-camera
    tolerance widens proportionally so a <= ~5-frame jitter does NOT breach. Option (a):
    the flag/counts stay surfaced for visibility; only the BREACH contribution is removed.
    """
    # mirror the e2e: header 5 Hz over ~29s -> ~146 frames; truncate to spread 142-146
    counts = [144, 143, 146, 146, 145, 142]
    topic_stamps = {SEM[i]: _hz_stamps(5.0, span_s=29.0)[:counts[i]] for i in range(6)}
    bag = _write_bag(tmp_path / "jitter", topic_stamps)
    reader = BagReader(bag, SEM)
    base = ReplayFaithfulnessMetric().compute(
        reader,
        {"output_topics": SEM,
         "expected_hz": {"default": 10.0, "semantic_raw_sim": 5},
         "gap_tolerance": {"default": 2.0, "semantic_raw_sim": 4.0}},
    )
    # counts are still surfaced (visibility preserved)
    assert set(base["cross_camera_counts"].values()) == set(counts), base["cross_camera_counts"]
    # but the 142-146 jitter contributes NO cross-camera breach (would have been +1 before)
    assert base["cross_camera_count_mismatch"] is False, base
    # prove the breach contribution is 0: per-topic breaches sum == total breach_count
    per_topic_breaches = sum(v["breach_count"] for v in base["per_topic"].values())
    assert base["breach_count"] == per_topic_breaches, base


def test_cross_camera_gross_starve_still_flags(tmp_path):
    """A GROSS partial-starve (one camera at ~10 frames vs peers at ~135) must STILL flag
    AND breach even under the widened jitter tolerance — the fail-closed partial-starve
    guard survives. Over-widening past the gross-starve spread would let a real starve
    pass silently; the tolerance must absorb a ~5-frame jitter but never a ~125-frame gap."""
    counts = [135, 135, 134, 10, 135, 135]   # camera 3 grossly starved
    topic_stamps = {SEM[i]: _hz_stamps(5.0, span_s=27.0)[:counts[i]] for i in range(6)}
    bag = _write_bag(tmp_path / "gross", topic_stamps)
    reader = BagReader(bag, SEM)
    out = ReplayFaithfulnessMetric().compute(
        reader,
        {"output_topics": SEM,
         "expected_hz": {"default": 10.0, "semantic_raw_sim": 5},
         "gap_tolerance": {"default": 2.0, "semantic_raw_sim": 4.0}},
    )
    assert out["cross_camera_count_mismatch"] is True, out
    # the cross-camera mismatch adds its fail-closed breach on top of any per-topic breaches
    per_topic_breaches = sum(v["breach_count"] for v in out["per_topic"].values())
    assert out["breach_count"] > per_topic_breaches, out


def test_faithfulness_decimals_rounded(tmp_path):
    """BUG 4 (display/storage): the e2e wrote drop_rate 0.7091667310217005 and max_gap_ms
    199.999996 — unreadable float noise. drop_rate rounds to 3 decimals; the top-level
    max_gap_ms and every per-topic max_gap_ms round to 1 decimal. Rounding is at the RETURN
    dict only — the threshold comparisons already ran on raw precision."""
    # a bag that yields non-round values: 5 Hz with a 599.999986-style gap
    bag = _write_bag(
        tmp_path / "round",
        {CAM: _stamps_with_gaps(5.0, n_normal=20, gap_ms=600.0, n_gaps=2),
         UNIFORM: _hz_stamps(10.0, span_s=4.0)},
    )
    reader = BagReader(bag, [CAM, UNIFORM])
    out = ReplayFaithfulnessMetric().compute(
        reader,
        {"output_topics": [CAM, UNIFORM],
         "expected_hz": {"default": 10.0, "image_raw_sim": 5}},
    )
    assert round(out["drop_rate"], 3) == out["drop_rate"], out["drop_rate"]
    assert round(out["max_gap_ms"], 1) == out["max_gap_ms"], out["max_gap_ms"]
    for topic, pt in out["per_topic"].items():
        assert round(pt["max_gap_ms"], 1) == pt["max_gap_ms"], (topic, pt["max_gap_ms"])


# --- 01-12 residual (UAT gap 1, validity-VERDICT half): the headline max_gap_ms gate ---
# 01-12 fixed per-topic breach_count, but the TOP-LEVEL max_gap_ms (the field
# replay_max_gap_ms gates on) still carried diagnostics' legit 5000ms gap -> healthy
# runs false-INVALIDed (exit 2) through the real generator. These tests pin the verdict path.


def test_faithfulness_headline_max_gap_excludes_slow_topics(tmp_path):
    """The TOP-LEVEL max_gap_ms (gated by replay_max_gap_ms at 200ms) must reflect
    uniform-rate topics only — NOT the 0.2Hz diagnostics topic's legitimate ~5000ms
    interval. The diagnostics gap stays recorded per-topic (visibility preserved)."""
    bag = _write_bag(
        tmp_path / "headline",
        {CAM: _hz_stamps(10.0, span_s=10.0), DIAG: _hz_stamps(0.2, span_s=10.0)},
    )
    reader = BagReader(bag, [CAM, DIAG])
    out = ReplayFaithfulnessMetric().compute(
        reader, {"output_topics": [CAM, DIAG], "expected_hz": {"default": 10.0, "diagnostics": 0.2}},
    )
    assert out["max_gap_ms"] <= 200.0, out["max_gap_ms"]
    assert 4900.0 <= out["per_topic"][DIAG]["max_gap_ms"] <= 5100.0


def test_healthy_mixed_rate_run_validates_through_generator(tmp_path):
    """The assertion the original gap-1 test was MISSING: a healthy mixed-rate bag's
    faithfulness result, run through the REAL generator with the real validity thresholds,
    must be VALID (exit 0) — not INVALID. Previously the headline max_gap_ms carried
    diagnostics' 5000ms gap and tripped replay_max_gap_ms (200) -> exit 2."""
    bag = _write_bag(
        tmp_path / "verdict",
        {CAM: _hz_stamps(10.0, span_s=10.0), DIAG: _hz_stamps(0.2, span_s=10.0)},
    )
    reader = BagReader(bag, [CAM, DIAG])
    faith = ReplayFaithfulnessMetric().compute(
        reader, {"output_topics": [CAM, DIAG], "expected_hz": {"default": 10.0, "diagnostics": 0.2}},
    )
    th = {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "replay_drop_rate": ThresholdSpec(max=0.02, tier="validity"),
        "replay_breach_count": ThresholdSpec(max=0, tier="validity"),
    }
    rc = generate_report("perception", "t", [], tmp_path, th, faithfulness=faith)
    assert rc == 0, faith


def test_perception_config_gates_breach_count():
    """The rate-aware per-topic breach signal (01-12) must actually be GATED. perception.yaml
    must carry a replay_breach_count validity threshold (max 0) so a per-topic stall on ANY
    topic (incl diagnostics, now excluded from the headline max_gap_ms) still invalidates."""
    configs = Path(__file__).resolve().parent.parent / "configs" / "modules"
    spec = load_module_config("perception", configs)
    assert "replay_breach_count" in spec.thresholds, list(spec.thresholds)
    bc = spec.thresholds["replay_breach_count"]
    assert bc.tier == "validity"
    assert bc.max == 0
