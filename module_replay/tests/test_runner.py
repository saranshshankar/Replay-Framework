from pathlib import Path

import pytest

from replay.data_manager import DataRef
from replay.runner import RunResult, run_replay

# The shared ``perception_spec`` fixture now lives in tests/conftest.py and
# resolves automatically. It was removed from this module to avoid duplication
# (plan 01-01 Task 2).


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


def _staged_bag(tmp_path: Path) -> DataRef:
    """A bag outside HOST_BAG_LIBRARY so the runner takes the copy path."""
    bag = tmp_path / "bag_in"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    return DataRef(local_path=bag, source="local")


def _run_and_capture_script(tmp_path: Path, perception_spec, mocker, exit_code: int = 0) -> tuple:
    """Run run_replay with all side-effects mocked; return (result, generated_script)."""
    mocker.patch("replay.runner.paths.HOST_BAG_LIBRARY", tmp_path / "elsewhere")
    data_ref = _staged_bag(tmp_path)
    exec_mock = mocker.patch("replay.runner.exec_in_container", return_value=exit_code)
    mocker.patch("replay.runner.shutil.move")
    mocker.patch("replay.runner.shutil.copytree")
    mocker.patch("replay.runner.shutil.copy")  # QoS file staging — no real IO in tests
    result = run_replay(module=perception_spec, data=data_ref, output_dir=tmp_path / "out")
    _, script = exec_mock.call_args.args
    return result, script


def test_run_replay_exit_code_propagated(tmp_path: Path, perception_spec, mocker):
    """FRWK-03: runner returns the container exit code, not hardcoded 0."""
    result, _ = _run_and_capture_script(tmp_path, perception_spec, mocker, exit_code=1)
    assert result.exit_code == 1


def test_run_replay_exit_code_zero_when_clean(tmp_path: Path, perception_spec, mocker):
    """FRWK-03: a clean replay (container exit 0) still reports 0."""
    result, _ = _run_and_capture_script(tmp_path, perception_spec, mocker, exit_code=0)
    assert result.exit_code == 0


def test_build_replay_script_has_queue_size(tmp_path: Path, perception_spec, mocker):
    """RPLY-01: --read-ahead-queue-size 5000 present (the documented stall fix)."""
    _, script = _run_and_capture_script(tmp_path, perception_spec, mocker)
    assert "--read-ahead-queue-size 5000" in script


def test_build_replay_script_has_qos_override_flag(tmp_path: Path, perception_spec, mocker):
    """RPLY-01: QoS override flag present so /tf_static is latched transient_local."""
    _, script = _run_and_capture_script(tmp_path, perception_spec, mocker)
    assert "--qos-profile-overrides-path" in script


def test_run_replay_prefers_spec_qos_override_path(tmp_path: Path, mocker):
    """Deferred 01-02/01-03 follow-up: when ModuleSpec carries an absolute
    qos_override_path, the runner uses IT (not the filesystem-derived lookup)."""
    import dataclasses

    from replay.module_config import ModuleSpec

    custom_qos = tmp_path / "custom_qos.yaml"
    custom_qos.write_text("/tf_static:\n  durability: transient_local\n")
    spec = ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=["/in"],
        output_topics=["/out"],
        launch_command="ros2 launch x y.launch.py",
        qos_override_path=custom_qos,
    )
    mocker.patch("replay.runner.paths.HOST_BAG_LIBRARY", tmp_path / "elsewhere")
    data_ref = _staged_bag(tmp_path)
    exec_mock = mocker.patch("replay.runner.exec_in_container", return_value=0)
    mocker.patch("replay.runner.shutil.move")
    mocker.patch("replay.runner.shutil.copytree")
    copy_mock = mocker.patch("replay.runner.shutil.copy")
    run_replay(module=spec, data=data_ref, output_dir=tmp_path / "out")
    _, script = exec_mock.call_args.args
    # The script runs in the container: it must reference the CONTAINER-staged
    # QoS path, never the host path. (Regression guard — the previous assertion
    # checked for the host path, which is exactly the bug that broke bag play.)
    from replay import paths

    container_qos = str(paths.CONTAINER_REPLAY_WORKDIR / "qos_override.yaml")
    assert container_qos in script
    assert str(custom_qos) not in script
    copy_mock.assert_called_once_with(
        custom_qos, paths.HOST_REPLAY_WORKDIR / "qos_override.yaml"
    )
    assert dataclasses.is_dataclass(spec)  # spec is the typed source of the path


def test_build_replay_script_no_fixed_sleep(tmp_path: Path, perception_spec, mocker):
    """RPLY-04: no 'sleep 3'; readiness loop present; no --pause."""
    _, script = _run_and_capture_script(tmp_path, perception_spec, mocker)
    assert "sleep 3" not in script
    assert "ros2 topic list" in script   # post-launch readiness loop
    assert "--pause" not in script        # paused player would drop latched /tf_static


def test_build_replay_script_records_mcap(tmp_path: Path, perception_spec, mocker):
    """RESEARCH Standard Stack: newly recorded output bags use MCAP, not sqlite3 .db3."""
    _, script = _run_and_capture_script(tmp_path, perception_spec, mocker)
    assert "--storage mcap" in script


def test_build_replay_script_preserves_cleanup_invariant(tmp_path: Path, perception_spec, mocker):
    """The setsid + trap + kill escalation block must survive the hardening edits."""
    _, script = _run_and_capture_script(tmp_path, perception_spec, mocker)
    assert "trap cleanup EXIT INT TERM" in script
    assert "kill -INT" in script
    assert "kill -KILL" in script
    assert "setsid ros2 bag record" in script
    assert "setsid ros2 bag play" in script
    assert "wait $PLAY_PID" in script
    # The in-container script still WRITES the logs to {log_dir}/recorder.log and
    # {log_dir}/module.log — 01-17 only COPIES them out on the host afterwards, it
    # must NOT change where the script writes them.
    assert "recorder.log" in script
    assert "module.log" in script


# ── 01-17: logs persisted to <output>/logs/ (host-side copy after exec) ──────


def _patch_runner_io(tmp_path: Path, mocker, exit_code: int = 0):
    """Stage a bag outside the library and patch the runner's host-side IO.

    Returns the patched ``shutil.copy`` mock so log-copy targets can be asserted.
    Unlike ``_run_and_capture_script`` this returns the copy mock (the log-copy
    seam) rather than the generated script.
    """
    mocker.patch("replay.runner.paths.HOST_BAG_LIBRARY", tmp_path / "elsewhere")
    data_ref = _staged_bag(tmp_path)
    mocker.patch("replay.runner.exec_in_container", return_value=exit_code)
    mocker.patch("replay.runner.shutil.move")
    mocker.patch("replay.runner.shutil.copytree")
    copy_mock = mocker.patch("replay.runner.shutil.copy")
    return data_ref, copy_mock


def test_run_replay_copies_logs_to_output(tmp_path: Path, perception_spec, mocker):
    """GAP c: after a run, RunResult.logs_dir == <output>/logs and BOTH
    recorder.log + module.log are copied from HOST_REPLAY_WORKDIR into it."""
    from replay import paths

    work_host = tmp_path / "work"
    mocker.patch("replay.runner.paths.HOST_REPLAY_WORKDIR", work_host)
    data_ref, copy_mock = _patch_runner_io(tmp_path, mocker, exit_code=0)

    output_dir = tmp_path / "out"
    result = run_replay(module=perception_spec, data=data_ref, output_dir=output_dir)

    logs_dir = output_dir / "logs"
    assert result.logs_dir == logs_dir
    # Both logs copied FROM the host workdir TO <output>/logs/.
    copy_targets = {Path(call.args[1]) for call in copy_mock.call_args_list}
    copy_sources = {Path(call.args[0]) for call in copy_mock.call_args_list}
    assert logs_dir / "recorder.log" in copy_targets
    assert logs_dir / "module.log" in copy_targets
    assert work_host / "recorder.log" in copy_sources
    assert work_host / "module.log" in copy_sources


def test_run_replay_copies_logs_even_on_failure(tmp_path: Path, perception_spec, mocker):
    """The logs are the failure evidence: on a non-zero container exit (137) the
    exit code is still propagated AND logs_dir is set AND the copy still happened."""
    work_host = tmp_path / "work"
    mocker.patch("replay.runner.paths.HOST_REPLAY_WORKDIR", work_host)
    data_ref, copy_mock = _patch_runner_io(tmp_path, mocker, exit_code=137)

    output_dir = tmp_path / "out"
    result = run_replay(module=perception_spec, data=data_ref, output_dir=output_dir)

    assert result.exit_code == 137
    assert result.logs_dir == output_dir / "logs"
    targets = {Path(call.args[1]).name for call in copy_mock.call_args_list}
    assert {"recorder.log", "module.log"} <= targets


def test_run_replay_log_copy_is_failsafe(tmp_path: Path, perception_spec, mocker):
    """A missing/unreadable log must NOT crash the run or change the exit code:
    if the log copy raises FileNotFoundError, run_replay still returns a RunResult
    with the original exit code."""
    work_host = tmp_path / "work"
    mocker.patch("replay.runner.paths.HOST_REPLAY_WORKDIR", work_host)
    data_ref = _staged_bag(tmp_path)
    mocker.patch("replay.runner.exec_in_container", return_value=137)
    mocker.patch("replay.runner.shutil.move")
    mocker.patch("replay.runner.shutil.copytree")
    # The log copy raises (source log never produced). QoS staging also routes
    # through shutil.copy, so we raise only for the log filenames and pass through
    # otherwise — the run must survive a log that does not exist.
    def _copy(src, dst, *a, **k):
        if Path(src).name in {"recorder.log", "module.log"}:
            raise FileNotFoundError(src)
        return dst
    mocker.patch("replay.runner.shutil.copy", side_effect=_copy)

    output_dir = tmp_path / "out"
    result = run_replay(module=perception_spec, data=data_ref, output_dir=output_dir)

    # No raise; original exit code preserved; logs_dir still reported.
    assert result.exit_code == 137
    assert result.logs_dir == output_dir / "logs"
