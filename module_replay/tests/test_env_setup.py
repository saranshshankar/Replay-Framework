from pathlib import Path

import pytest

from replay.env_setup import (
    CONTROLLER_REPLAY_OVERRIDE,
    PLANNER_REPLAY_OVERRIDE,
    TENXCODE_PLANNER_OVERRIDE,
    setup_environment,
)
from replay.module_config import ModuleSpec
from replay.version_manager import SubmoduleOverride, VersionSpec


@pytest.fixture
def perception_spec() -> ModuleSpec:
    return ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=[],
        output_topics=[],
        launch_command="ros2 launch x y.launch.py",
    )


@pytest.fixture
def manipulation_spec() -> ModuleSpec:
    return ModuleSpec(
        name="manipulation",
        container="controller",
        colcon_package="manipulation_manager",
        input_topics=[],
        output_topics=[],
        launch_command="ros2 launch x y.launch.py",
    )


def test_setup_planner_path(perception_spec, mocker):
    up = mocker.patch("replay.env_setup.compose_up")
    checkout = mocker.patch("replay.env_setup.checkout_branch")
    submodule_update = mocker.patch("replay.env_setup.submodule_update_recursive")
    submodule_checkout = mocker.patch("replay.env_setup.checkout_submodule_branch")
    exec_in = mocker.patch("replay.env_setup.exec_in_container")

    spec = VersionSpec(tenxcode_branch="dev", submodule_overrides=[])

    setup_environment(spec, perception_spec)

    from replay import paths
    expected_overrides = [TENXCODE_PLANNER_OVERRIDE, PLANNER_REPLAY_OVERRIDE]
    up.assert_called_once_with(
        paths.PLANNER_COMPOSE,
        overrides=expected_overrides,
        pull_policy="missing",
    )
    checkout.assert_called_once_with(paths.HOST_TENXCODE, "dev")
    submodule_update.assert_called_once_with(paths.HOST_TENXCODE)
    submodule_checkout.assert_not_called()
    assert exec_in.call_count == 1
    container_arg, cmd = exec_in.call_args.args
    assert container_arg == paths.PLANNER_CONTAINER
    assert "source /opt/ros/humble/setup.bash" in cmd
    assert "colcon build --packages-up-to realtime_perception" in cmd


def test_setup_controller_path(manipulation_spec, mocker):
    mocker.patch("replay.env_setup.compose_up")
    mocker.patch("replay.env_setup.checkout_branch")
    mocker.patch("replay.env_setup.submodule_update_recursive")
    exec_in = mocker.patch("replay.env_setup.exec_in_container")

    spec = VersionSpec(tenxcode_branch="dev", submodule_overrides=[])
    setup_environment(spec, manipulation_spec)

    from replay import paths
    assert exec_in.call_count == 1
    container_arg, cmd = exec_in.call_args.args
    assert container_arg == paths.CONTROLLER_CONTAINER
    assert "source /opt/ros/humble/setup.bash" in cmd
    assert "colcon build --packages-up-to manipulation_manager" in cmd


def test_setup_applies_submodule_overrides(perception_spec, mocker):
    mocker.patch("replay.env_setup.compose_up")
    mocker.patch("replay.env_setup.checkout_branch")
    mocker.patch("replay.env_setup.submodule_update_recursive")
    mocker.patch("replay.env_setup.exec_in_container")
    submodule_checkout = mocker.patch("replay.env_setup.checkout_submodule_branch")

    override = SubmoduleOverride(name="common_interfaces", branch="master")
    spec = VersionSpec(tenxcode_branch="feature/foo", submodule_overrides=[override])

    setup_environment(spec, perception_spec)

    from replay import paths
    submodule_checkout.assert_called_once_with(paths.HOST_TENXCODE, override)


def test_setup_checkout_runs_before_compose_up(perception_spec, mocker):
    """The 10xCode branch must be checked out before `docker compose up -d`,
    because `up` may build the image on demand and the build reads the
    Dockerfile + COPY context from the 10xCode tree."""
    calls: list[str] = []
    mocker.patch(
        "replay.env_setup.checkout_branch",
        side_effect=lambda *a, **kw: calls.append("checkout_branch"),
    )
    mocker.patch(
        "replay.env_setup.submodule_update_recursive",
        side_effect=lambda *a, **kw: calls.append("submodule_update"),
    )
    mocker.patch(
        "replay.env_setup.compose_up",
        side_effect=lambda *a, **kw: calls.append("compose_up"),
    )
    mocker.patch(
        "replay.env_setup.exec_in_container",
        side_effect=lambda *a, **kw: calls.append("exec_in_container"),
    )

    spec = VersionSpec(tenxcode_branch="dev", submodule_overrides=[])
    setup_environment(spec, perception_spec)

    assert calls.index("checkout_branch") < calls.index("compose_up")
    assert calls.index("submodule_update") < calls.index("compose_up")
    # colcon build (exec_in_container) still runs last.
    assert calls[-1] == "exec_in_container"


def test_setup_default_build_jobs_caps_parallelism(perception_spec, mocker):
    mocker.patch("replay.env_setup.compose_up")
    mocker.patch("replay.env_setup.checkout_branch")
    mocker.patch("replay.env_setup.submodule_update_recursive")
    exec_in = mocker.patch("replay.env_setup.exec_in_container")

    spec = VersionSpec(tenxcode_branch="dev", submodule_overrides=[])
    setup_environment(spec, perception_spec)

    _, cmd = exec_in.call_args.args
    assert "MAKEFLAGS='-j2'" in cmd
    assert "--parallel-workers 2" in cmd
    assert "colcon build --packages-up-to realtime_perception" in cmd


def test_setup_custom_build_jobs(perception_spec, mocker):
    mocker.patch("replay.env_setup.compose_up")
    mocker.patch("replay.env_setup.checkout_branch")
    mocker.patch("replay.env_setup.submodule_update_recursive")
    exec_in = mocker.patch("replay.env_setup.exec_in_container")

    spec = VersionSpec(tenxcode_branch="dev", submodule_overrides=[])
    setup_environment(spec, perception_spec, build_jobs=1)

    _, cmd = exec_in.call_args.args
    assert "MAKEFLAGS='-j1'" in cmd
    assert "--parallel-workers 1" in cmd


def test_setup_partial_checkout_uses_paths_and_skips_recursive(perception_spec, mocker):
    mocker.patch("replay.env_setup.compose_up")
    mocker.patch("replay.env_setup.exec_in_container")
    full_checkout = mocker.patch("replay.env_setup.checkout_branch")
    recursive_submodule = mocker.patch("replay.env_setup.submodule_update_recursive")
    partial_checkout = mocker.patch("replay.env_setup.checkout_paths_from_branch")
    submodule_single = mocker.patch("replay.env_setup.submodule_init")

    spec = VersionSpec(tenxcode_branch="dev", submodule_overrides=[])
    setup_environment(
        spec,
        perception_spec,
        checkout_paths=["perception/", "common_interfaces/"],
    )

    from replay import paths
    partial_checkout.assert_called_once_with(
        paths.HOST_TENXCODE, "dev", ["perception/", "common_interfaces/"]
    )
    # Only submodules listed in the partial paths get their contents fetched.
    submodule_single.assert_called_once_with(paths.HOST_TENXCODE, "common_interfaces")
    # Full-tree git operations are not run.
    full_checkout.assert_not_called()
    recursive_submodule.assert_not_called()


def test_replay_override_files_have_no_hardcoded_host_paths():
    """FRWK-02: the .replay_work bind mount must not assume any developer's
    home directory — the host side comes from the REPLAY_WORK_DIR env var."""
    for override in (PLANNER_REPLAY_OVERRIDE, CONTROLLER_REPLAY_OVERRIDE):
        text = override.read_text()
        assert "/home/" not in text, f"{override.name} hardcodes a host home path"
        assert "${REPLAY_WORK_DIR" in text, f"{override.name} missing REPLAY_WORK_DIR mount"


def test_setup_exports_replay_work_dir_for_compose(perception_spec, mocker, monkeypatch):
    """setup_environment must export REPLAY_WORK_DIR (derived portably from
    paths.HOST_REPLAY_WORKDIR) so docker compose interpolates the work mount."""
    import os

    mocker.patch("replay.env_setup.compose_up")
    mocker.patch("replay.env_setup.checkout_branch")
    mocker.patch("replay.env_setup.submodule_update_recursive")
    mocker.patch("replay.env_setup.exec_in_container")
    monkeypatch.delenv("REPLAY_WORK_DIR", raising=False)

    spec = VersionSpec(tenxcode_branch="dev", submodule_overrides=[])
    setup_environment(spec, perception_spec)

    from replay import paths
    assert os.environ["REPLAY_WORK_DIR"] == str(paths.HOST_REPLAY_WORKDIR)
