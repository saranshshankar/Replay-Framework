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
