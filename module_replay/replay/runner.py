"""Replay runner: play input bag through module in container, record output."""
from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from replay import paths
from replay.data_manager import DataRef
from replay.docker_utils import exec_in_container
from replay.module_config import ModuleSpec

# Seconds to wait after starting recorder + module so ROS 2 discovery settles
# before we start playing the bag. Without this, the first messages of the bag
# are published before the recorder has finished subscribing and are lost.
DISCOVERY_WAIT_SECS = 3
# Bigger read-ahead queue keeps `ros2 bag play` ahead of consumers when the
# bag is dense (many high-rate camera + lidar topics). 1000 is comfortably
# above the default and small enough not to balloon container memory.
BAG_PLAY_READ_AHEAD_QUEUE = 1000


@dataclass(frozen=True)
class RunResult:
    output_bag_path: Path
    exit_code: int


def _container_for(module: ModuleSpec) -> str:
    if module.container == "planner":
        return paths.PLANNER_CONTAINER
    if module.container == "controller":
        return paths.CONTROLLER_CONTAINER
    raise ValueError(f"Unknown container: {module.container}")


def _translate_bag_path(host_bag_path: Path) -> Optional[Path]:
    """Map a host bag path to its container path if it sits under the bag library.

    Returns None when the bag lives outside `paths.HOST_BAG_LIBRARY`, in
    which case the caller falls back to copying the bag into the writable
    workdir.
    """
    resolved = host_bag_path.resolve()
    if not resolved.is_relative_to(paths.HOST_BAG_LIBRARY):
        return None
    relative = resolved.relative_to(paths.HOST_BAG_LIBRARY)
    return paths.CONTAINER_BAG_LIBRARY / relative


def _build_replay_script(
    in_bag: Path,
    output_bag: Path,
    input_topics: list[str],
    output_topics: list[str],
    launch_command: str,
    log_dir: Path,
) -> str:
    """Return a single bash script that records, launches, plays, and cleans up.

    The script runs inside one shell so the backgrounded recorder and module
    are direct children of that shell — `kill` + `wait` target them reliably,
    giving rosbag2 a chance to flush its final chunk before the shell exits.
    """
    in_topics = " ".join(shlex.quote(t) for t in input_topics)
    out_topics = " ".join(shlex.quote(t) for t in output_topics)
    # Each child runs under `setsid` so it gets its own process group; the
    # cleanup function then signals the entire group with `kill -<SIG> -PGID`
    # so descendants (e.g. perception_node spawned by `ros2 launch`) also
    # receive the signal. `set -e` is intentionally NOT set — cleanup must
    # run regardless of whether play exited normally or via Ctrl-C/SIGTERM.
    # The trap fires on EXIT, INT, and TERM, escalating to SIGKILL after a
    # grace period if any process refuses SIGINT.
    # The output bag is written by root (the container default user). Match
    # the bind-mount root's host UID/GID at the end so the host can move the
    # files without sudo.
    return f"""source /opt/ros/humble/setup.bash
[ -f /root/ros2_ws/install/setup.bash ] && source /root/ros2_ws/install/setup.bash

rm -rf {output_bag}

cleanup() {{
  echo "[runner] cleanup: signalling child process groups..." >&2
  for pid in "$REC_PID" "$MOD_PID" "$PLAY_PID"; do
    [ -n "$pid" ] && kill -INT -"$pid" 2>/dev/null
  done
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    alive=0
    for pid in "$REC_PID" "$MOD_PID" "$PLAY_PID"; do
      [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    [ "$alive" = "0" ] && break
  done
  for pid in "$REC_PID" "$MOD_PID" "$PLAY_PID"; do
    [ -n "$pid" ] && kill -KILL -"$pid" 2>/dev/null
  done

  HOST_UID=$(stat -c '%u' {log_dir})
  HOST_GID=$(stat -c '%g' {log_dir})
  [ -d {output_bag} ] && chown -R "$HOST_UID":"$HOST_GID" {output_bag} 2>/dev/null
}}
trap cleanup EXIT INT TERM

# Recorder first so it doesn't miss any module output.
setsid ros2 bag record -o {output_bag} {out_topics} > {log_dir}/recorder.log 2>&1 &
REC_PID=$!

# Bag play second so /tf_static (transient_local) is on the bus BEFORE the
# module's on_configure runs. Perception reads static TFs during configure;
# if the module starts before play, those latched messages aren't available.
# Trade-off: we lose the first few seconds of bag content while the module
# is still coming up. Acceptable.
setsid ros2 bag play {in_bag} --topics {in_topics} --clock --read-ahead-queue-size {BAG_PLAY_READ_AHEAD_QUEUE} &
PLAY_PID=$!

sleep {DISCOVERY_WAIT_SECS}

# Module last. By now the bag is publishing /tf_static, so the configure
# callback will see the latched messages.
setsid {launch_command} > {log_dir}/module.log 2>&1 &
MOD_PID=$!

# Wait for play to finish (normal exit) or for a signal (trap fires).
wait $PLAY_PID
"""


def run_replay(
    *,
    module: ModuleSpec,
    data: DataRef,
    output_dir: Path,
) -> RunResult:
    if not data.local_path.exists():
        raise FileNotFoundError(f"Input bag not found: {data.local_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    container = _container_for(module)

    work_host = paths.HOST_REPLAY_WORKDIR
    work_host.mkdir(parents=True, exist_ok=True)
    work_container = paths.CONTAINER_REPLAY_WORKDIR

    # If the bag lives under the bag library, the container already sees it
    # via the read-only bind mount — no copy needed. Otherwise stage it into
    # the writable workdir so the container can reach it.
    library_in_bag = _translate_bag_path(data.local_path)
    if library_in_bag is not None:
        in_bag = library_in_bag
        print(f"Using bag-library bind mount: {data.local_path} -> {in_bag}")
    else:
        input_stage = work_host / "input_bag"
        if input_stage.exists():
            shutil.rmtree(input_stage)
        shutil.copytree(data.local_path, input_stage)
        in_bag = work_container / "input_bag"

    output_bag = work_container / "replay_output"

    script = _build_replay_script(
        in_bag=in_bag,
        output_bag=output_bag,
        input_topics=module.input_topics,
        output_topics=module.output_topics,
        launch_command=module.launch_command,
        log_dir=work_container,
    )
    exec_in_container(container, script)

    # Output bag is already on the host via bind mount; move it to output_dir.
    host_output = output_dir / "replay_output"
    if host_output.exists():
        shutil.rmtree(host_output)
    shutil.move(str(work_host / "replay_output"), str(host_output))

    return RunResult(output_bag_path=host_output, exit_code=0)
