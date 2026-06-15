"""Host and container path constants - single source of truth."""
import os
from pathlib import Path

# The 10xCode checkout location varies per developer. Override via the
# TENXCODE_ROOT environment variable; default assumes the layout described
# in the README's Setup section.
HOST_TENXCODE = Path(
    os.environ.get("TENXCODE_ROOT", "~/workspace/src/10xCode")
).expanduser().resolve()

CONTAINER_TENXCODE = Path("/root/ros2_ws/src/10xCode")

PLANNER_COMPOSE = HOST_TENXCODE / ".devcontainer/v2/planner/docker-compose.yml"
CONTROLLER_COMPOSE = HOST_TENXCODE / ".devcontainer/v2/controller/docker-compose.yml"

PLANNER_CONTAINER = "v2-planner-docker-x86"
CONTROLLER_CONTAINER = "v2-controller-docker-x86"

# HOST_REPLAY_ROOT is derived from where this package lives — module_replay/replay/paths.py
# lives inside the module_replay repo, so two parents up is the repo root.
HOST_REPLAY_ROOT = Path(__file__).resolve().parent.parent
HOST_REPLAY_WORKDIR = HOST_REPLAY_ROOT / ".replay_work"

CONTAINER_REPLAY_WORKDIR = Path("/root/ros2_ws/replay_work")

# Bag library: a host directory that holds rosbag2 directories. Bind-mounted
# read-only into the container so the runner can play bags directly without
# copying them into .replay_work. Override via REPLAY_BAG_LIBRARY env var.
HOST_BAG_LIBRARY = Path(
    os.environ.get("REPLAY_BAG_LIBRARY", "~/data")
).expanduser().resolve()
CONTAINER_BAG_LIBRARY = Path("/root/data")
