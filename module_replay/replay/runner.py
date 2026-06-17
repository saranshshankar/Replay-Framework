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

# Read-ahead buffer for `ros2 bag play`. 5000 is the documented fix for the
# observed ~3.9 s publish-interval stall gaps (RESEARCH § Pitfall 3): with the
# default-sized queue, a dense bag (many high-rate camera + lidar topics) lets
# the player fall behind its consumers and emit large publish gaps. 5000 keeps
# the player comfortably ahead while staying within container memory.
#
# There is intentionally NO fixed discovery sleep: the old `sleep 3` was a
# non-deterministic guess. It is replaced by a POST-LAUNCH readiness loop in
# `_build_replay_script` that polls for the module's first output topic and
# fails fast (exit 1) if the module never comes up — so a crashed launch is
# visible to the CI gate instead of being masked.
BAG_PLAY_READ_AHEAD_QUEUE = 5000


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
    qos_override_path: Optional[Path] = None,
) -> str:
    """Return a single bash script that records, launches, plays, and cleans up.

    The script runs inside one shell so the backgrounded recorder and module
    are direct children of that shell — `kill` + `wait` target them reliably,
    giving rosbag2 a chance to flush its final chunk before the shell exits.
    """
    in_topics = " ".join(shlex.quote(t) for t in input_topics)
    out_topics = " ".join(shlex.quote(t) for t in output_topics)
    # QoS override flag (RPLY-01): when a per-module qos yaml exists, pass it to
    # `ros2 bag play` so /tf_static is republished with transient_local durability
    # — perception needs the latched static TFs during its on_configure window.
    # The path is package-relative and `.exists()`-guarded by the caller (never
    # user-supplied), so no shell-quoting threat (T-02-01).
    qos_flag = (
        f" --qos-profile-overrides-path {qos_override_path}"
        if qos_override_path is not None
        else ""
    )
    # The readiness loop polls for the module's first output topic — it can only
    # appear once the module is up, so it is the "module came up" gate that
    # replaces the old fixed `sleep 3`.
    first_output_topic = output_topics[0] if output_topics else ""
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

# Recorder first so it doesn't miss any module output. The MCAP storage flag
# writes the newly recorded output bag as MCAP (SIGKILL-resilient + indexed
# seeking) instead of the legacy sqlite3/.db3 default (CLAUDE.md: sqlite3
# CHALLENGED; RESEARCH § Standard Stack). The offline faithfulness/metrics
# reader (rosbags BagReader, plan 01-06) reads MCAP natively, so this is
# transparent downstream.
setsid ros2 bag record --storage mcap -o {output_bag} {out_topics} > {log_dir}/recorder.log 2>&1 &
REC_PID=$!

# Bag play second so the bag's latched /tf_static is on the bus during the
# module's on_configure (the bag is the ONLY source of the *_corrected frames —
# KT/playbooks/01-perception.md Step 4); the long-running player keeps it
# latched. The player stays UNPAUSED-before-module: a paused player publishes
# nothing, including /tf_static. The pause+preamble zero-loss variant is
# deferred to the 8-phase runner (KT/SYSTEM-DESIGN-HLD-LLD.md B5/DD10).
setsid ros2 bag play {in_bag} --topics {in_topics} --clock \\
  --read-ahead-queue-size {BAG_PLAY_READ_AHEAD_QUEUE}{qos_flag} &
PLAY_PID=$!

# Module launch. By now the bag is publishing /tf_static, so the configure
# callback will see the latched messages.
setsid {launch_command} > {log_dir}/module.log 2>&1 &
MOD_PID=$!

# Readiness: poll for the module's first output topic instead of a fixed sleep.
# This can only succeed once the module is up, so it must run AFTER the launch.
# Fails fast (exit 1) if the module never comes up - a crashed launch is then
# visible to the CI gate instead of being masked.
WAIT_MAX=30
for i in $(seq 1 $WAIT_MAX); do
  if ros2 topic list 2>/dev/null | grep -q "{first_output_topic}"; then break; fi
  [ "$i" -eq "$WAIT_MAX" ] && echo "[runner] module readiness timeout" >&2 && exit 1
  sleep 1
done

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

    # Resolve the per-module QoS override. Prefer the typed `module.qos_override_path`
    # (added to ModuleSpec in plan 01-03 — the deferred 01-02/01-03 follow-up); a
    # relative value is resolved against the package root, mirroring how cli.py
    # resolves configs. Fall back to the filesystem-derived lookup for specs that
    # do not carry the field (e.g. the 6-field test fixtures). Guarded by .exists()
    # so modules without a qos yaml simply run without the flag.
    package_root = Path(__file__).resolve().parent.parent
    if module.qos_override_path is not None:
        spec_qos = Path(module.qos_override_path)
        qos_host = spec_qos if spec_qos.is_absolute() else package_root / spec_qos
    else:
        qos_host = package_root / "configs" / "qos" / f"{module.name}.yaml"
    # The replay script runs INSIDE the container, where a host path to the QoS
    # file does not exist — passing it would make `ros2 bag play` die ("file not
    # found"). The framework's configs/ dir is not bind-mounted, so stage the
    # file into the writable workdir (which IS mounted) and pass the CONTAINER
    # path, exactly as the input bag is staged above.
    if qos_host.exists():
        shutil.copy(qos_host, work_host / "qos_override.yaml")
        qos_override_path = work_container / "qos_override.yaml"
    else:
        qos_override_path = None

    script = _build_replay_script(
        in_bag=in_bag,
        output_bag=output_bag,
        input_topics=module.input_topics,
        output_topics=module.output_topics,
        launch_command=module.launch_command,
        log_dir=work_container,
        qos_override_path=qos_override_path,
    )
    exit_code = exec_in_container(container, script)

    # Output bag is already on the host via bind mount; move it to output_dir.
    host_output = output_dir / "replay_output"
    if host_output.exists():
        shutil.rmtree(host_output)
    shutil.move(str(work_host / "replay_output"), str(host_output))

    return RunResult(output_bag_path=host_output, exit_code=exit_code)
