from pathlib import Path

import pytest

from replay.module_config import (
    ModuleSpec,
    ThresholdSpec,
    load_checkout_paths,
    load_module_config,
    missing_preflight_assets,
)


# Minimal perception.yaml carrying the two-tier thresholds + qos_override block,
# used by the threshold-parsing tests. Mirrors the real perception.yaml shape.
_PERCEPTION_WITH_THRESHOLDS = """
name: perception
container: planner
colcon_package: realtime_perception
input_topics:
  - /perception_node/camera_0/image_raw
output_topics:
  - /perception_node/camera_0/image_raw_sim
launch:
  command: "ros2 launch realtime_perception perception_node.launch.py use_replay:=true"
qos_override: configs/qos/perception.yaml
thresholds:
  validity:
    replay_max_gap_ms:
      max: 200
      provisional: true
  quality:
    latency_p95_ms:
      max: 50.0
      tolerance_band: 5.0
      provisional: true
mocks: []
"""

# Minimal perception.yaml with NO thresholds block (back-compat case).
_PERCEPTION_NO_THRESHOLDS = """
name: perception
container: planner
colcon_package: realtime_perception
input_topics:
  - /perception_node/camera_0/image_raw
output_topics:
  - /perception_node/camera_0/image_raw_sim
launch:
  command: "ros2 launch realtime_perception perception_node.launch.py use_replay:=true"
"""

# Minimal perception.yaml carrying the new metric-config block (01-10): per-topic
# expected_hz, depth_topics, diagnostics_topic. These thread into the metric cfg so
# faithfulness/depth/latency stop falling back to broken flat-10Hz/all-topics defaults.
_PERCEPTION_WITH_METRIC_CFG = """
name: perception
container: planner
colcon_package: realtime_perception
input_topics:
  - /perception_node/camera_0/image_raw
output_topics:
  - /perception_node/camera_0/image_raw_sim
  - /perception_node/camera_0/depth_raw_sim
  - /perception_node/diagnostics
launch:
  command: "ros2 launch realtime_perception perception_node.launch.py use_replay:=true"
expected_hz:
  default: 10.0
  diagnostics: 0.2
depth_topics:
  - /perception_node/camera_0/depth_raw_sim
diagnostics_topic: /perception_node/diagnostics
gap_tolerance:
  default: 2.0
  image_raw_sim: 4.0
  semantic_raw_sim: 4.0
latency_stage: inference_seg_extract_segmentation
"""


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics:
  - /lidar_front/points
  - /camera_0/image_raw/compressed
output_topics:
  - /perception/rgb
launch:
  command: "ros2 launch realtime_perception perception.launch.py mode:=sim"
"""
    )
    return d


def test_loads_perception_spec(configs_dir: Path):
    spec = load_module_config("perception", configs_dir)
    assert isinstance(spec, ModuleSpec)
    assert spec.name == "perception"
    assert spec.container == "planner"
    assert spec.colcon_package == "realtime_perception"
    assert "/lidar_front/points" in spec.input_topics
    assert spec.output_topics == ["/perception/rgb"]
    assert spec.launch_command.startswith("ros2 launch")


def test_unknown_module_raises(configs_dir: Path):
    with pytest.raises(FileNotFoundError):
        load_module_config("teleportation", configs_dir)


def test_rejects_invalid_container(tmp_path: Path):
    d = tmp_path / "modules"
    d.mkdir()
    (d / "bad.yaml").write_text(
        """
name: bad
container: spaceship
colcon_package: x
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    with pytest.raises(ValueError, match="container"):
        load_module_config("bad", d)


def test_load_checkout_paths_returns_module_list(tmp_path: Path):
    d = tmp_path / "modules"
    d.mkdir()
    (d / "checkout_paths.yaml").write_text(
        """
perception:
  - perception/
  - common_interfaces/
navigation:
  - navigation/
"""
    )
    assert load_checkout_paths("perception", d) == ["perception/", "common_interfaces/"]
    assert load_checkout_paths("navigation", d) == ["navigation/"]


def test_load_checkout_paths_missing_module_raises(tmp_path: Path):
    d = tmp_path / "modules"
    d.mkdir()
    (d / "checkout_paths.yaml").write_text("perception: [x]\n")
    with pytest.raises(KeyError, match="manipulation"):
        load_checkout_paths("manipulation", d)


def test_load_checkout_paths_missing_file_raises(tmp_path: Path):
    d = tmp_path / "modules"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        load_checkout_paths("perception", d)


def test_thresholds_loaded(tmp_path: Path):
    """MTRC-06: validity + quality thresholds flatten into a typed ThresholdSpec dict."""
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(_PERCEPTION_WITH_THRESHOLDS)
    spec = load_module_config("perception", d)
    assert isinstance(spec.thresholds["latency_p95_ms"], ThresholdSpec)
    assert spec.thresholds["latency_p95_ms"].max == 50.0
    assert spec.thresholds["latency_p95_ms"].provisional is True
    assert spec.thresholds["latency_p95_ms"].tier == "quality"
    assert spec.thresholds["replay_max_gap_ms"].tier == "validity"
    assert spec.thresholds["replay_max_gap_ms"].max == 200
    # qos_override + defaulted mocks/launch_args also land on the spec.
    assert spec.qos_override_path == Path("configs/qos/perception.yaml")
    assert spec.mocks == []
    assert isinstance(spec.launch_args, dict)


def test_config_without_thresholds_still_loads(tmp_path: Path):
    """Back-compat: a minimal config with no thresholds block loads with thresholds == {}."""
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(_PERCEPTION_NO_THRESHOLDS)
    spec = load_module_config("perception", d)
    assert spec.thresholds == {}
    assert spec.qos_override_path is None
    assert spec.mocks == []
    assert spec.launch_args == {}
    assert spec.preflight_assets == []


def test_expected_hz_parses_into_dict(tmp_path: Path):
    """01-10 / UAT gap 1: the `expected_hz:` block parses into a dict[str, float].

    Faithfulness reads this per-topic map so the 0.2 Hz diagnostics topic is no
    longer scanned against a flat 10 Hz (which breached the 200ms gate every run).
    """
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(_PERCEPTION_WITH_METRIC_CFG)
    spec = load_module_config("perception", d)
    assert spec.expected_hz == {"default": 10.0, "diagnostics": 0.2}
    assert isinstance(spec.expected_hz["default"], float)


def test_depth_topics_parses_into_list(tmp_path: Path):
    """01-10 / UAT gap 5: `depth_topics:` parses into a list DepthMetric scans."""
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(_PERCEPTION_WITH_METRIC_CFG)
    spec = load_module_config("perception", d)
    assert spec.depth_topics == ["/perception_node/camera_0/depth_raw_sim"]


def test_diagnostics_topic_parses_into_str(tmp_path: Path):
    """01-10: `diagnostics_topic:` parses into a str LatencyMetric will read."""
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(_PERCEPTION_WITH_METRIC_CFG)
    spec = load_module_config("perception", d)
    assert spec.diagnostics_topic == "/perception_node/diagnostics"


def test_gap_tolerance_parses_into_dict(tmp_path: Path):
    """01-19 / e2e gap-closure: the `gap_tolerance:` block parses into a
    dict[str, float] (substring->breach-gap factor), mirroring expected_hz.

    Faithfulness (01-20) reads this per-topic map so the FAITHFUL ~600ms EoMT
    inference stall on image/semantic (~3.2x the 200ms period at 5Hz) no longer
    false-breaches against the default 2x tolerance, while a genuine 2x replay
    hang on a uniform topic still breaches.
    """
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(_PERCEPTION_WITH_METRIC_CFG)
    spec = load_module_config("perception", d)
    assert spec.gap_tolerance == {
        "default": 2.0,
        "image_raw_sim": 4.0,
        "semantic_raw_sim": 4.0,
    }
    assert isinstance(spec.gap_tolerance["default"], float)


def test_latency_stage_parses_into_str(tmp_path: Path):
    """01-19 / e2e gap-closure: `latency_stage:` parses into a str LatencyMetric
    (01-21) parses avg_compute_ms under (the live `inference_seg_extract_segmentation`
    op name, post seg_argmax->seg_extract rename, e2e module.log)."""
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(_PERCEPTION_WITH_METRIC_CFG)
    spec = load_module_config("perception", d)
    assert spec.latency_stage == "inference_seg_extract_segmentation"


def test_metric_cfg_fields_default_when_absent(tmp_path: Path):
    """Back-compat: a config with no metric-cfg block defaults expected_hz={},
    depth_topics=[], diagnostics_topic=None, gap_tolerance={}, latency_stage=None."""
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(_PERCEPTION_NO_THRESHOLDS)
    spec = load_module_config("perception", d)
    assert spec.expected_hz == {}
    assert spec.depth_topics == []
    assert spec.diagnostics_topic is None
    assert spec.gap_tolerance == {}
    assert spec.latency_stage is None


def test_six_field_modulespec_defaults_metric_cfg():
    """The 6-field conftest fixture (and runner callers) construct ModuleSpec with
    only the required fields; the new metric-cfg fields MUST all be defaulted."""
    spec = ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=[],
        output_topics=[],
        launch_command="x",
    )
    assert spec.expected_hz == {}
    assert spec.depth_topics == []
    assert spec.diagnostics_topic is None
    assert spec.gap_tolerance == {}
    assert spec.latency_stage is None


def test_build_metrics_cfg_threads_gap_tolerance_and_latency_stage():
    """01-19: _build_metrics_cfg threads gap_tolerance + latency_stage into the cfg
    every plugin/faithfulness receives, alongside the pre-existing keys.

    01-20 (faithfulness) reads gap_tolerance for the per-topic breach threshold;
    01-21 (LatencyMetric) reads latency_stage for the diagnostics op name."""
    from replay.cli import _build_metrics_cfg

    spec = ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=["/perception_node/camera_0/image_raw"],
        output_topics=["/perception_node/camera_0/image_raw_sim"],
        launch_command="x",
        gap_tolerance={"default": 2.0, "image_raw_sim": 4.0},
        latency_stage="inference_seg_extract_segmentation",
    )
    cfg = _build_metrics_cfg(spec)
    assert cfg["gap_tolerance"] == {"default": 2.0, "image_raw_sim": 4.0}
    assert cfg["latency_stage"] == "inference_seg_extract_segmentation"
    # Pre-existing keys remain present (additive change, nothing dropped).
    assert cfg["input_topics"] == spec.input_topics
    assert cfg["output_topics"] == spec.output_topics
    assert "expected_hz" in cfg
    assert "depth_topics" in cfg
    assert "diagnostics_topic" in cfg


def test_missing_preflight_assets_reports_only_missing(tmp_path: Path):
    """B5 phase-0 fail-fast: the checker returns exactly the asset paths that don't exist."""
    present = tmp_path / "present.trt"
    present.write_text("x")
    absent = tmp_path / "absent.trt"
    spec = ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=[],
        output_topics=[],
        launch_command="x",
        preflight_assets=[str(present), str(absent)],
    )
    missing = missing_preflight_assets(spec)
    assert missing == [str(absent)]


def test_missing_preflight_assets_empty_when_no_block(tmp_path: Path):
    """Back-compat: a spec with no preflight_assets returns []."""
    spec = ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=[],
        output_topics=[],
        launch_command="x",
    )
    assert missing_preflight_assets(spec) == []


# ---------------------------------------------------------------------------
# 01.1-01: incident_detectors + error_code_topic (Task 1)
# ---------------------------------------------------------------------------

def test_incident_fields_default_when_absent(configs_dir: Path):
    """01.1-01 D-14: load_module_config on a YAML with NO incident block yields
    incident_detectors=={} and error_code_topic is None (existing caller unaffected)."""
    spec = load_module_config("perception", configs_dir)
    assert spec.incident_detectors == {}
    assert spec.error_code_topic is None


def test_incident_detectors_round_trips_verbatim(tmp_path: Path):
    """01.1-01 D-13: an `incident_detectors:` mapping is returned VERBATIM (no
    coercion) — the verifier in plan 04 owns interpretation, mirroring the
    expected_hz/gap_tolerance precedent."""
    d = tmp_path / "modules"
    d.mkdir()
    (d / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
incident_detectors:
  INC-001:
    metric: latency_p95_ms
    field: latency_p95_ms
    op: "ge"
    threshold: 50.0
  INC-002:
    metric: segmentation_coverage
    field: segmentation_coverage
    op: "le"
    threshold: 0.3
error_code_topic: /perception_node/error_code
"""
    )
    spec = load_module_config("perception", d)
    assert spec.incident_detectors == {
        "INC-001": {
            "metric": "latency_p95_ms",
            "field": "latency_p95_ms",
            "op": "ge",
            "threshold": 50.0,
        },
        "INC-002": {
            "metric": "segmentation_coverage",
            "field": "segmentation_coverage",
            "op": "le",
            "threshold": 0.3,
        },
    }
    assert spec.error_code_topic == "/perception_node/error_code"


def test_six_field_modulespec_incident_fields_default():
    """01.1-01 design-fidelity: the 6-field conftest fixture constructs ModuleSpec
    with only the required fields; the new incident fields MUST be defaulted so
    existing 6-field callers keep constructing without TypeError."""
    spec = ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=[],
        output_topics=[],
        launch_command="x",
    )
    assert spec.incident_detectors == {}
    assert spec.error_code_topic is None
