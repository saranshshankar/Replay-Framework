# module_replay
This repo consists of changes to enable module-wise replay and relevant metrics and visualization

## Replay Platform - Local Usage

A module-wise replay platform that reruns a selected 10xCode module (perception /
navigation / manipulation) against recorded ROS bag data.

### Setup

The replay platform needs to know where your 10xCode checkout lives on disk.
The default assumes `~/workspace/src/10xCode`. If your checkout is elsewhere,
set the `TENXCODE_ROOT` environment variable.

Add this to `~/.bashrc` (or `~/.zshrc`, depending on your shell):

```bash
# Only needed if your 10xCode checkout is NOT at ~/workspace/src/10xCode
export TENXCODE_ROOT=~/path/to/your/10xCode
```

Then reload your shell config or open a new terminal:

```bash
source ~/.bashrc
```

Verify:

```bash
echo $TENXCODE_ROOT
```

Notes:
- Not setting `TENXCODE_ROOT` is fine — the default (`~/workspace/src/10xCode`)
  will be used.
- Paths with `~` are expanded automatically.
- The `module_replay` repo's own location is auto-detected (derived from where
  the Python package is installed from), so no env var is needed for it.

#### Bag library location

The replay platform bind-mounts a single host directory holding your rosbag
recordings into the container at `/root/data` (read-only). When you pass a
bag whose path is under this directory, the runner reads it directly via
the mount — no copy. For a 13 GB bag this turns multi-minute staging into
zero seconds.

The default is `~/data`. Override via `REPLAY_BAG_LIBRARY` if your bags
live elsewhere:

```bash
# Only needed if your bags are NOT under ~/data
export REPLAY_BAG_LIBRARY=~/path/to/bags
```

Notes:
- After changing this, recreate the container so the mount picks up the
  new value: `replay-module setup-env --module perception` (compose detects
  the env-var change and recreates).
- Bags outside the library still work — the runner falls back to copying
  them into `.replay_work/`. Slower for large bags, fine for tiny test bags.
- The mount is read-only, so the container can't accidentally modify your
  source bags.

### Install

Run from the `module_replay` repo root:

```bash
cd /path/to/module_replay
pip install -e '.[dev]'
```

- `.` refers to the package in the current directory (defined in `pyproject.toml`).
- `[dev]` pulls in the optional `dev` dependency group (`pytest`, `pytest-mock`,
  `moto[s3]`) — needed to run the test suite. Omit it (`pip install -e .`) if
  you only want the runtime deps.
- `-e` is an editable install, so local code edits are picked up without
  reinstalling.
- The quotes around `'.[dev]'` stop the shell from expanding the brackets.

Note: `dev` here is a Python packaging convention, unrelated to the `dev`
branch of 10xCode.

### CLI overview

```bash
# Validate a module + version spec (no side-effects).
# Prints the resolved module parameters from configs/modules/<module>.yaml
# (container, colcon package, input/output topics, launch command) and the
# resolved 10xCode branch + any submodule overrides. Use this to confirm the
# config the other subcommands will act on, without touching Docker or git.
replay-module validate --module perception

# Example output:
#   Module: perception
#     Container: planner
#     Colcon package: realtime_perception
#     Input topics: ['/perception_node/camera_0/image_raw', ..., '/tf', '/tf_static']
#     Output topics: ['/perception/rgb', '/perception/depth', '/perception/semantic']
#   10xCode branch: dev
#   Submodule overrides: (none - pinned from 10xCode branch)

# Resolve a bag (local path -> prints absolute path).
# --local-bag MUST point at the rosbag2 directory itself (the one containing
# metadata.yaml + *.db3), NOT at a parent folder that holds multiple bags and
# NOT at an individual .db3 file. A chunked single bag (one metadata.yaml with
# several .db3 files for the same recording) is fine — that's one logical bag.
# If the path you pass holds multiple separate rosbag2 directories, pick one.
replay-module fetch-data --local-bag /path/to/rosbag2_2026_04_09-18_44_09 --output /tmp/out

# Resolve a bag from S3 (requires REPLAY_S3_BUCKET env var)
export REPLAY_S3_BUCKET=my-bucket
export AWS_PROFILE=10x-dev
replay-module fetch-data --task-id task_20260409_184409 --robot-id robot_03 --output /tmp/out

# Set up environment for a module.
# Runs: git checkout + submodule update on 10xCode, then a single
# `docker compose up -d --pull missing`, then colcon build of the module
# inside the container. The `--pull missing` policy pulls the image only
# when no local image matches the compose file's tag — first runs pull
# automatically, subsequent runs reuse the local image with no registry
# round-trip and no flag toggling. The image is also built on demand if
# the compose file declares a `build:` section and no image is available.
#
# By default the colcon build runs with `--parallel-workers 2` and
# `MAKEFLAGS=-j2` so the laptop stays responsive (≤4 compiler processes
# at once). Override with `--build-jobs N`: pick higher (faster, hungrier)
# or 1 (slowest, gentlest) if 2 still stalls.
replay-module setup-env --module perception --version-yaml configs/versions/default.yaml

# Partial checkout: update only the module's own paths on the 10xCode tree
# instead of switching the whole tree to the target branch. Useful on slow
# networks or when you want to avoid disturbing files outside this module.
# Paths per module are in configs/modules/checkout_paths.yaml. Submodules
# listed there (e.g. common_interfaces) are also initialised.
replay-module setup-env --module perception --partial-checkout

# Gentlest possible build — drop to a single compiler at a time:
replay-module setup-env --module perception --partial-checkout --build-jobs 1

# Run replay given a module + resolved bag
replay-module run --module perception --bag /tmp/out/rosbag2_x --output /tmp/out

# Do everything end-to-end
replay-module all --module perception --local-bag /path/to/bag --output /tmp/out
```

### Version YAML format

```yaml
tenxcode:
  branch: dev
submodules:
  common_interfaces:
    branch: master
  # Omitted submodules use the commit pinned by the 10xCode branch.
```

### Module config

Per-module input/output topics and colcon package names live in
`configs/modules/{perception,navigation,manipulation}.yaml`. Edit these to match
the actual topic names in your 10xCode branch.

### S3 data layout

```
s3://$REPLAY_S3_BUCKET/data/{robot_id}/{YYYY-MM-DD}/recordings/{mode}/{task_id}_{ts}/
  ├── metadata.json
  ├── bag/
  │   ├── metadata.yaml
  │   └── rosbag2_*.db3
  └── ...
```

The `fetch-data` subcommand scans under `data/{robot_id}/` for a directory whose
name starts with `{task_id}_`.
