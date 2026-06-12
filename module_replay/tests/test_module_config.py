from pathlib import Path

import pytest

from replay.module_config import ModuleSpec, load_checkout_paths, load_module_config


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
