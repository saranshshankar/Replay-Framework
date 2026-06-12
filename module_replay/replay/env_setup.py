"""Environment setup: docker compose + git checkout + colcon build."""
from __future__ import annotations

import os

from replay import paths
from replay.docker_utils import (
    compose_up,
    exec_in_container,
)
from replay.git_utils import (
    checkout_branch,
    checkout_paths_from_branch,
    checkout_submodule_branch,
    submodule_init,
    submodule_update_recursive,
)
from replay.module_config import ModuleSpec
from replay.version_manager import VersionSpec

# Submodule directories, relative to 10xCode root. If a partial checkout lists
# one of these, we also run `git submodule update --init <path>` so the
# submodule's contents are fetched (the bare gitlink isn't enough to build).
_SUBMODULE_DIRS = {"common_interfaces", "sim-testing-envs", "version_system", "shared_db"}


REPLAY_OVERRIDE_DIR = paths.HOST_REPLAY_ROOT / "replay" / "compose"
PLANNER_REPLAY_OVERRIDE = REPLAY_OVERRIDE_DIR / "planner.replay.override.yml"
CONTROLLER_REPLAY_OVERRIDE = REPLAY_OVERRIDE_DIR / "controller.replay.override.yml"

# docker compose auto-loads `docker-compose.override.yml` only when no `-f` is
# passed. Once we pass explicit `-f`s, we must re-include 10xCode's own override
# or the 10xCode source bind-mount into the container is lost.
TENXCODE_PLANNER_OVERRIDE = (
    paths.HOST_TENXCODE / ".devcontainer/v2/planner/docker-compose.override.yml"
)
TENXCODE_CONTROLLER_OVERRIDE = (
    paths.HOST_TENXCODE / ".devcontainer/v2/controller/docker-compose.override.yml"
)


def _compose_container_and_overrides(module: ModuleSpec):
    if module.container == "planner":
        return (
            paths.PLANNER_COMPOSE,
            paths.PLANNER_CONTAINER,
            [TENXCODE_PLANNER_OVERRIDE, PLANNER_REPLAY_OVERRIDE],
        )
    if module.container == "controller":
        return (
            paths.CONTROLLER_COMPOSE,
            paths.CONTROLLER_CONTAINER,
            [TENXCODE_CONTROLLER_OVERRIDE, CONTROLLER_REPLAY_OVERRIDE],
        )
    raise ValueError(f"Unknown container type: {module.container}")


def _apply_full_checkout(version: VersionSpec) -> None:
    checkout_branch(paths.HOST_TENXCODE, version.tenxcode_branch)
    submodule_update_recursive(paths.HOST_TENXCODE)
    for sub_override in version.submodule_overrides:
        checkout_submodule_branch(paths.HOST_TENXCODE, sub_override)


def _apply_partial_checkout(version: VersionSpec, checkout_paths: list[str]) -> None:
    checkout_paths_from_branch(
        paths.HOST_TENXCODE, version.tenxcode_branch, checkout_paths
    )
    for p in checkout_paths:
        name = p.rstrip("/")
        if name in _SUBMODULE_DIRS:
            submodule_init(paths.HOST_TENXCODE, name)
    for sub_override in version.submodule_overrides:
        checkout_submodule_branch(paths.HOST_TENXCODE, sub_override)


DEFAULT_BUILD_JOBS = 2


def setup_environment(
    version: VersionSpec,
    module: ModuleSpec,
    *,
    checkout_paths: list[str] | None = None,
    build_jobs: int = DEFAULT_BUILD_JOBS,
) -> None:
    """Prepare the Docker container + 10xCode checkout + colcon build.

    Flow: git checkout (full or partial) -> `docker compose up -d --pull
    missing` -> colcon build inside container. The `--pull missing` policy
    pulls the image only when no local image matches the compose file's
    tag, so first runs pull automatically and subsequent runs touch the
    registry zero times. `up -d` also builds the image on demand if pull
    fails or the compose file declares a `build:` section.

    If `checkout_paths` is given, only those 10xCode-relative paths are
    updated to match the branch (partial checkout). Submodules in the list
    are initialised after. HEAD is not moved and files outside the list are
    left as-is. When `checkout_paths` is None the full branch is checked out
    and all submodules are updated recursively.

    `build_jobs` caps colcon's `--parallel-workers` and the inner make
    `-j` so the laptop stays responsive during heavy compiles. Default 2
    means at most 2 packages compile concurrently with at most 2 compiler
    subprocesses each. Drop to 1 for the gentlest possible build.
    """
    compose_file, container, overrides = _compose_container_and_overrides(module)

    # Check out the requested 10xCode branch + submodules FIRST. `docker compose
    # up -d` may build the image (when none is local), and the build reads the
    # Dockerfile + COPY context from the 10xCode tree, so the tree has to match
    # the Dockerfile's expectations before we run up.
    if checkout_paths is None:
        _apply_full_checkout(version)
    else:
        _apply_partial_checkout(version, checkout_paths)

    # The replay override files mount ${REPLAY_WORK_DIR} as the host side of
    # the work dir — derive it portably so the mount works on any machine
    # (FRWK-02). setdefault keeps an explicit user export authoritative.
    os.environ.setdefault("REPLAY_WORK_DIR", str(paths.HOST_REPLAY_WORKDIR))

    compose_up(compose_file, overrides=overrides, pull_policy="missing")

    # Source ROS Humble (and any existing overlay from a prior build) before
    # running colcon. The container's ~/.bashrc sources both, but it bails
    # early for non-interactive shells, so `bash -lc` never picks them up.
    build_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "{ [ -f /root/ros2_ws/install/setup.bash ] && "
        "source /root/ros2_ws/install/setup.bash; } ; "
        f"cd /root/ros2_ws && "
        f"MAKEFLAGS='-j{build_jobs}' "
        f"colcon build --packages-up-to {module.colcon_package} "
        f"--parallel-workers {build_jobs}"
    )
    exec_in_container(container, build_cmd)
