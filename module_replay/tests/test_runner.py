from pathlib import Path

import pytest

from replay.data_manager import DataRef
from replay.module_config import ModuleSpec
from replay.runner import RunResult, run_replay


@pytest.fixture
def perception_spec() -> ModuleSpec:
    return ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=["/lidar_front/points", "/camera_0/image_raw/compressed"],
        output_topics=["/perception/rgb", "/perception/depth"],
        launch_command="ros2 launch realtime_perception perception.launch.py mode:=sim",
    )


def test_run_replay_issues_single_shell_script(tmp_path: Path, perception_spec, mocker):
    # Place the bag outside HOST_BAG_LIBRARY so the runner takes the copy path.
    mocker.patch("replay.runner.paths.HOST_BAG_LIBRARY", tmp_path / "elsewhere")
    bag_dir = tmp_path / "bag_in"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("version: 5")
    data_ref = DataRef(local_path=bag_dir, source="local")
    output_dir = tmp_path / "out"

    exec_mock = mocker.patch("replay.runner.exec_in_container")
    # Runner also moves the output bag back; mock to avoid touching the host fs.
    mocker.patch("replay.runner.shutil.move")
    copytree_mock = mocker.patch("replay.runner.shutil.copytree")

    result = run_replay(
        module=perception_spec,
        data=data_ref,
        output_dir=output_dir,
    )

    assert isinstance(result, RunResult)
    assert result.output_bag_path == output_dir / "replay_output"

    # Exactly one docker exec — everything runs in one shell.
    assert exec_mock.call_count == 1
    container_arg, script = exec_mock.call_args.args
    assert container_arg == "v2-planner-docker-x86"

    # Recorder starts first with both output topics.
    assert "ros2 bag record" in script
    assert "/perception/rgb" in script
    assert "/perception/depth" in script

    # Module launched.
    assert perception_spec.launch_command in script

    # ros2 bag play with --topics flag (NOT ros2 bag filter — that doesn't exist
    # in Humble; we filter via `ros2 bag play --topics` instead).
    assert "ros2 bag filter" not in script
    assert "ros2 bag play" in script
    assert "--topics" in script
    assert "/lidar_front/points" in script
    assert "/camera_0/image_raw/compressed" in script

    # Graceful shutdown is driven by a trap that fires on EXIT/INT/TERM
    # and signals child process groups (so descendants like perception_node
    # spawned by `ros2 launch` also receive the signal).
    assert "trap cleanup EXIT INT TERM" in script
    assert "kill -INT" in script
    assert "kill -KILL" in script  # escalation if SIGINT is ignored
    assert "setsid ros2 bag record" in script
    assert "setsid ros2 bag play" in script
    assert "wait $PLAY_PID" in script

    # Bag was outside the library, so the runner copied it.
    copytree_mock.assert_called_once()


def test_run_replay_record_before_play_before_module(tmp_path: Path, perception_spec, mocker):
    """Three ordering invariants:
    - Recorder starts before bag play (so it doesn't miss module output).
    - Bag play starts before module launch (so /tf_static, published with
      transient_local QoS, is on the bus BEFORE the module's on_configure
      callback runs and tries to read static TFs)."""
    mocker.patch("replay.runner.paths.HOST_BAG_LIBRARY", tmp_path / "elsewhere")
    data_ref = DataRef(local_path=tmp_path / "bag_in", source="local")
    (tmp_path / "bag_in").mkdir()
    (tmp_path / "bag_in" / "metadata.yaml").write_text("version: 5")

    exec_mock = mocker.patch("replay.runner.exec_in_container")
    mocker.patch("replay.runner.shutil.move")
    mocker.patch("replay.runner.shutil.copytree")

    run_replay(module=perception_spec, data=data_ref, output_dir=tmp_path / "out")

    _, script = exec_mock.call_args.args
    record_pos = script.index("ros2 bag record")
    play_pos = script.index("ros2 bag play")
    module_pos = script.index(perception_spec.launch_command)
    assert record_pos < play_pos < module_pos


def test_run_replay_uses_library_mount_skips_copy(tmp_path: Path, perception_spec, mocker):
    """Bags under the bag library are read directly via the bind mount —
    no `shutil.copytree` of multi-GB data."""
    library_root = tmp_path / "data"
    library_root.mkdir()
    bag_dir = library_root / "rosbag2_x"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("version: 5")

    mocker.patch("replay.runner.paths.HOST_BAG_LIBRARY", library_root)
    mocker.patch("replay.runner.paths.CONTAINER_BAG_LIBRARY", Path("/root/data"))

    exec_mock = mocker.patch("replay.runner.exec_in_container")
    mocker.patch("replay.runner.shutil.move")
    copytree_mock = mocker.patch("replay.runner.shutil.copytree")

    data_ref = DataRef(local_path=bag_dir, source="local")
    run_replay(module=perception_spec, data=data_ref, output_dir=tmp_path / "out")

    copytree_mock.assert_not_called()
    _, script = exec_mock.call_args.args
    # The play command must reference the container-translated path.
    assert "ros2 bag play /root/data/rosbag2_x" in script


def test_run_replay_missing_input_bag_raises(tmp_path: Path, perception_spec):
    data_ref = DataRef(local_path=tmp_path / "does_not_exist", source="local")
    with pytest.raises(FileNotFoundError):
        run_replay(module=perception_spec, data=data_ref, output_dir=tmp_path / "out")
