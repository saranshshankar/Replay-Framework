# Module-wise Replay Framework — Code Walkthrough

> This document is **only about the code**: what the framework is, how it's
> structured, how data flows through one run, what each part does and *why* it's
> written that way, and the changes currently in flight. (CI wiring, infra setup,
> and the local test plan live elsewhere — they're deliberately not here.)
>
> If you're reading this to present or to understand the system for the first time,
> read **Part 0** end to end. Parts 1–3 walk the actual code. Part 4 covers what's
> being added right now.

---

## Part 0 — The framework in plain language

### What problem it solves

When someone changes a 10xCode module — say perception — the only way to know whether
they broke it has been to flash the build onto a robot and watch. That's slow, manual,
and not something you can gate a pull request on.

This framework replaces that with something a computer can do automatically and
repeatably. The raw material is a **rosbag**: a recording of every ROS message the
robot exchanged during a run. If you come from a backend background, picture a
`tcpdump` of the robot's internal message bus, or the request/response capture in a
`.har` file — it's the same idea, just for ROS topics instead of HTTP.

The framework takes that recording and does four things:

1. Plays **only one module's recorded inputs** back onto the bus,
2. Runs a **freshly built copy of that one module** against those inputs,
3. **Records what the module now produces**, and
4. **Scores** that fresh output against per-module quality thresholds.

The entire result collapses into a single process **exit code**, so a CI job can pass
or block a PR purely on that number. The mental model: *replay recorded production
traffic against a new build of one microservice and diff the responses* — except the
"microservice" is a perception node and the "traffic" is camera and lidar frames.

### The one architectural idea: two layers, split by a recorded bag

Almost every design decision in the codebase follows from one split:

```
   ONLINE layer  (expensive)                     OFFLINE layer  (cheap)
   GPU + ROS 2 + Docker                           plain Python, no ROS, no GPU
   ─────────────────────────────                  ──────────────────────────────
   Run the real module against the     ── a new ──►   Read that bag back with the
   recorded inputs and record its         MCAP         `rosbags` library, compute
   outputs into a brand-new bag.          bag          metrics, decide pass/fail.

   runner.py, docker_utils.py,                       everything under
   env_setup.py, isolation/                          replay/metrics/
```

The boundary between the two layers is a **file** — the recorded output bag. The
online layer's only job is to *produce* that bag. The offline layer's only job is to
*read and grade* it. Nothing in the offline layer ever touches ROS or a GPU; it just
opens a file.

Why this matters so much:

- **Cost.** The GPU replay is the expensive part. Because grading is a separate step
  that only reads a file, you run the costly replay once and grade on the cheapest
  machine available.
- **Testability.** Every metric can be unit-tested on a laptop with a tiny synthetic
  bag — no robot, no GPU, no container needed.
- **Clarity.** The two halves can fail for completely different reasons, and the exit
  code tells them apart (more on that below).

Keep this split in your head and the rest of the code reads naturally.

### How one run flows, end to end

This is the single command a developer (or CI) runs, and what happens inside it:

```
  replay-module all --module perception --local-bag <bag> --output <dir> --run-metrics
        │
        1  RESOLVE    Read perception.yaml → a ModuleSpec object (which topics are
        │             inputs, which are outputs, the thresholds, the assets it needs).
        │             Also pick which 10xCode branch to test.
        │
        2  SETUP      git checkout that branch  →  docker compose up  →  colcon build
        │             (build the one module under test, inside the container).
        │
        3  PREFLIGHT  Check the on-disk assets the module needs actually exist.
        │             If one is missing → stop now with EXIT 3, naming the path.
        │
        4  REPLAY     Start a recorder → play ONLY perception's input topics →
        │             launch ONLY the perception node → wait for it to finish →
        │             a new OUTPUT BAG (MCAP) has been written.
        │
        5  METRICS    Read that output bag. Run validity checks ("was the replay
        │             healthy?") and quality metrics ("did the code regress?").
        │
        6  VERDICT    Write metrics.json + report.html, and return EXIT 0 / 1 / 2.
```

The word "module-wise" is earned in **step 4**. The bag may contain all 64 of the
robot's recorded topics, but the runner plays only perception's ~10 input topics onto
the bus, launches only the perception node, and records only perception's outputs.
Feed the *same* recording with `--module navigation` instead and you've isolated a
different suspect — one recording, sliced per module. That's the core capability:
take a field incident's bag and replay it through each module in turn to find which
one actually misbehaved.

### How you run it

```bash
replay-module all \
  --module perception \
  --local-bag ~/data/rosbag2_2026_06_03-18_52_40 \
  --version-yaml configs/versions/default.yaml \
  --output /tmp/perception_run \
  --run-metrics
echo "exit=$?"        # 0 PASS · 1 quality FAIL · 2 INVALID RUN · 3 setup error
```

What each flag means:

- `--module perception` — which module to isolate and test.
- `--local-bag <dir>` — the recording to replay (or `--task-id` + `--robot-id` to pull
  one from S3 instead).
- `--version-yaml` — pins the exact 10xCode branch under test, so a run is a controlled,
  reproducible experiment.
- `--output` — where the output bag, the report, and the logs are written.
- `--run-metrics` — also run the scoring step. Drop it and you just get the output bag
  to eyeball by hand.

`all` is the full pipeline. There are smaller subcommands too — `run` does only the
replay, `metrics` does only the offline scoring on an existing bag — but `all` is the
one to understand first, because reading it top to bottom *is* reading the framework.

### The preflight checks — what they are, and why they exist

A rosbag captures *messages*. But a module needs more than messages to run: it needs
model weights, camera calibration, and the robot's own description — none of which are
ever in a bag. These live as files on disk, and if any is missing the module fails in
confusing, deep ways (a crash 30 seconds into a GPU run, or a node that silently
produces nothing).

The **preflight gate** exists to turn that confusion into a clear, instant error. Before
the framework spends a single GPU-second, it checks that each required asset exists, and
if one is missing it stops immediately with **exit 3, naming the exact missing path**.
The three assets it checks for perception:

| Asset | What it actually is | Why the run dies without it |
|-------|---------------------|------------------------------|
| **TensorRT engine** (`flicker_drywall.trt` — the V2 EoMT segmentation model) | The GPU-compiled neural network that produces the segmentation masks. It is compiled for one specific GPU model + TensorRT version. | The perception node loads this engine when it activates. If it's missing (or built for the wrong GPU), activation fails and the node produces nothing at all. |
| **Fisheye LUT calibration tree** (6 camera folders, 3 files each) | Lookup tables that map each fisheye camera's distorted pixels to straight 3D rays. The pipeline uses them to project lidar points onto the images and to build the colored pointcloud and interpolated depth. | The operators that build those outputs load the LUTs at startup and **throw an exception** if any file is missing — so the node never finishes activating. |
| **`version_system/config/current` (param + URDF)** | The robot's physical description (URDF). The launch file reads it *while it's still parsing*, to set up a sibling filter node. This folder is empty in a fresh checkout until it's populated. | If it's empty, the launch file crashes during parsing — perception never even starts. |

The one-line summary to give a manager: **preflight converts "mysterious deep crash"
into "you're missing file X — stage it and rerun."** It's the difference between a
five-minute confused debugging session and a one-line fix.

### What you get out of a run

| Output | Where | What it is |
|--------|-------|------------|
| **Output bag** | `<output>/replay_output` (MCAP) | The fresh `_sim` topics the module produced: `semantic_raw_sim` ×6 (the segmentation masks — the primary signal), `image_raw_sim` ×6, `depth_raw_sim` ×6, `colored_pointcloud_sim`, and `diagnostics`. |
| **`metrics.json`** | `<output>/reports/` | The machine-readable verdict. Top-level `pass` (bool) and `verdict` (PASS/FAIL/INVALID), a `replay_faithfulness` block (`max_gap_ms`, `drop_rate`, `breach_count`), and a `metrics[]` array with one row per criterion (its value vs its threshold). |
| **`report.html`** | `<output>/reports/` | The human-readable view of the same data (being upgraded — see Part 4). |
| **Logs** | `<output>/logs/` | `recorder.log` and `module.log` from the run (being persisted per-run — see Part 4). |
| **Exit code** | the process itself | The whole contract in one number. |

### The most important concept: validity vs. quality

This is the idea worth making sure your manager walks away with, because it's what makes
the gate trustworthy enough to block people's PRs.

Metrics are split into two tiers:

- **Validity** asks: *was the replay itself healthy?* Did all the frames flow, were
  there no big gaps, did no camera starve? This has nothing to do with whether the code
  is good — it's about whether the *experiment* was sound.
- **Quality** asks: *did the code regress?* Given a healthy replay, are the segmentation
  masks, latency, and overlap within their thresholds?

The verdict respects that order:

- A **validity** breach → **exit 2 (INVALID RUN)**. The experiment was broken (a starved
  camera, dropped frames, a bad bag), so the quality numbers can't be trusted and aren't
  even reported as a failure. Crucially, this is **not** blamed on the developer's code.
- A **quality** breach on an otherwise-valid run → **exit 1 (FAIL)**. This is a real
  regression.
- Everything clean → **exit 0 (PASS)**.
- Anything operational (missing asset, container crash, the module never coming up) →
  **exit 3**, kept entirely separate from the quality verdict.

Without this split you get the classic flaky-gate problem: infrastructure hiccups show
up as "your code failed," people stop trusting the gate, and it gets ignored. The
validity tier is what prevents that.

> **A concrete example, from the first real run.** We replayed a sparse manipulation bag
> (5 cameras, ~0.6 Hz, no lidar) through the real V2 EoMT engine. The pipeline ran end to
> end, produced an output bag (3115 messages), and the EoMT model loaded and did genuine
> inference. But the gate returned **INVALID RUN** (`drop_rate 0.986`, `max_gap_ms 1e9`),
> because the bag was simply too degraded to draw any conclusion from. That's the validity
> tier doing exactly its job: rather than emit a misleading "pass," it refused to grade a
> broken experiment.

---

## Part 1 — The code map

Before walking the execution path, here's where everything lives. The package is
`module_replay/`, installed as a CLI called `replay-module`.

```
module_replay/
├── replay/
│   ├── cli.py              # the command-line entry point (the `all` pipeline lives here)
│   ├── module_config.py    # ModuleSpec: parses a module's YAML contract; the preflight check
│   ├── version_manager.py  # VersionSpec: which 10xCode branch + submodule pins to test
│   ├── data_manager.py     # DataRef: resolve a bag from a local path or from S3
│   ├── env_setup.py        # checkout + docker compose up + colcon build
│   ├── docker_utils.py     # thin wrappers around `docker` / `docker compose`
│   ├── git_utils.py        # thin wrappers around `git`
│   ├── paths.py            # the single source of truth for host/container paths
│   ├── runner.py           # THE REPLAY CORE — builds and runs the replay shell script
│   ├── executor.py         # the "run a command in a container" seam (local now, cloud later)
│   ├── isolation/          # pure functions: turn a ModuleSpec into replay arguments
│   │   ├── topic_filter.py #   → the `--topics` allow-list for `ros2 bag play`
│   │   ├── launch_args.py  #   → `key:=value` launch arguments
│   │   └── mock_nodes.py   #   → stand-in nodes for modules that need them (none for perception)
│   └── metrics/            # THE OFFLINE LAYER — pure Python, never imports ROS
│       ├── base.py         #   the Metric interface every plugin implements
│       ├── registry.py     #   @register_metric — plugins self-register on import
│       ├── bag_reader.py   #   reads the output bag with `rosbags`
│       ├── replay_faithfulness.py  # the validity-tier metric
│       ├── baseline.py     #   pinned-golden comparison (for the "vs golden" metrics)
│       ├── perception/     #   the 7 perception quality metrics
│       └── report/         #   generator.py → metrics.json + report.html, and the exit code
└── configs/
    ├── modules/perception.yaml   # the contract: topics, thresholds, assets, launch command
    ├── versions/default.yaml     # which 10xCode branch to test (dev)
    ├── qos/perception.yaml       # a QoS override for replaying /tf_static
    └── baselines/perception/golden.yaml   # the pinned-golden reference manifest
```

Two things to notice from the map alone. First, the online code (`runner`, `docker_*`,
`env_setup`, `isolation`) and the offline code (`metrics/`) are cleanly separated — that's
the two-layer split made physical. Second, almost everything module-specific is **data**,
not code: `configs/modules/perception.yaml` plus the `metrics/perception/` plugin pack.
Adding a new module later means adding a YAML file and a plugin pack — not editing the
runner or the CLI. That reusability is the framework's main bet.

---

## Part 2 — Walking the code, the path of one run

The clearest way to understand the code is to follow a single `replay-module all` run
through it, in order.

### 2.1 The entry point — `cli.py`

`cli.py` is a `click` command group. The command that matters is `all` (function
`all_cmd`); the rest (`validate`, `fetch-data`, `setup-env`, `run`, `metrics`) are
individual stages you can run on their own, which is handy for debugging but not
essential to understanding the whole.

`all_cmd` reads almost like a checklist, and that's deliberate — it's the spine, and
each line delegates to one focused module:

1. **Load the specs.** It builds a `ModuleSpec` (from `perception.yaml`) and a
   `VersionSpec` (which 10xCode branch). If you don't pass `--version-yaml`, it quietly
   falls back to `configs/versions/default.yaml`.
2. **Resolve the bag.** Either `--local-bag` (a path) or `--task-id` + `--robot-id` (an
   S3 lookup). Exactly one is required, or it errors clearly.
3. **Set up the environment** (`setup_environment`) — checkout, compose up, build.
4. **Run the preflight gate.** If any asset is missing, it prints the missing paths and
   exits 3 *before* the container does any real work.
5. **Run the replay** (`run_replay`). If the replay process returns a non-zero exit
   code, the CLI maps that to its own exit 3 — a setup/replay failure, kept distinct
   from a metrics verdict.
6. **If `--run-metrics`,** run the offline scoring and exit with its verdict code.

The function worth singling out is the small helper that bridges the two layers,
`_run_metrics_pipeline`. It is shared by both `all --run-metrics` and the standalone
`metrics` command — which is precisely *why* the offline half can run on its own, on a
cheap machine, against a bag that some other run produced. Inside it does three things
that are each worth understanding:

- It does `import replay.metrics.perception`. That import looks like a no-op but it's
  load-bearing: importing the package runs the `@register_metric` decorators on each
  plugin, which is how all seven perception metrics register themselves (more in 2.6).
  This is the same pattern as Flask blueprints or pytest plugins — discovery by import.
- It opens the output bag **once** with a `BagReader` and reuses it for every metric, so
  the bag is parsed a single time, not once per metric.
- It runs the faithfulness (validity) metric explicitly, then loops over the registered
  quality plugins, then calls the report generator — which writes the files and returns
  the exit code.

### 2.2 The contract — `module_config.py`

This is the most important file in the framework, because it's where one YAML file
becomes the single source of truth for everything about a module.

`ModuleSpec` is a frozen dataclass describing a module: its name, its container, the
colcon package to build, its input and output topics, its launch command — plus the
fields that let one file carry the whole contract: the QoS override path, the thresholds,
any mock nodes, launch arguments, and the preflight asset list. The newer fields all
have defaults, so older code and test fixtures that only know the original six fields
keep working unchanged — a small but deliberate bit of backward-compatibility hygiene.

`ThresholdSpec` is the typed shape of a single threshold: a `max`, a `min`, a
`tolerance_band`, a `provisional` flag, and a `tier` (`"validity"` or `"quality"`). That
`tier` field is the seed of the whole validity-vs-quality split — it's set here, in
config, and the gate reads it later to decide whether a breach means INVALID or FAIL.

`load_module_config` reads `configs/modules/<module>.yaml`, validates that the container
is one it knows, and **flattens** the two-tier `thresholds:` block in the YAML (which is
nested under `validity:` and `quality:`) into one flat dictionary keyed by metric name,
stamping each threshold with its tier as it goes. So the YAML stays readable and grouped,
while the code gets a simple flat lookup.

`missing_preflight_assets` is the function behind the exit-3 gate. It takes the spec's
`preflight_assets` list, expands `~` and `$ENV` variables in each path, and returns the
ones that don't exist on the host. The CLI calls this and stops if the list is non-empty.
That's the entire mechanism — simple, but it's what turns a deep activation crash into a
named missing file.

### 2.3 Setting up the environment — `env_setup.py`

`setup_environment` is a strict three-step sequence, and the order is the point:

1. **Check out the 10xCode branch first** (full or partial). This has to happen before
   anything else, because the next step may *build the container image*, and the image
   build reads files from the 10xCode tree — so the tree has to be on the right branch
   first.
2. **`docker compose up`** with a "pull only if missing" policy: the first run pulls the
   image, later runs reuse the local one with no registry round-trip. It layers two
   compose override files — 10xCode's own, plus the framework's replay override that adds
   the work directory and bag-library mounts.
3. **`colcon build`** the one module, inside the container.

One detail in that build command is worth calling out because it caused a real bug and
its fix is subtle: the build uses **`--symlink-install`**. Here's why it matters. The
module's launch file resolves its config (`perception_sim.yaml`) from the *install*
directory, not from the source tree. Without `--symlink-install`, colcon *copies* config
files into the install directory at build time — so if you edit a config and the copy
step doesn't re-run, the module silently loads the stale copy. With `--symlink-install`,
the install-directory config is a live symlink back to the source, so an edit always
takes effect. (It's also 10xCode's own build convention.) The lesson worth internalizing:
"I edited the config but nothing changed" is very often an install-space-vs-source-tree
problem, not a logic bug.

The helpers this leans on are deliberately thin: `git_utils.py` is a set of small
wrappers over `git -C <repo> …`, and `docker_utils.py` is the same over `docker`. One
function in `docker_utils.py` is foundational, though: `exec_in_container` runs a command
in the container and **returns its exit code without ever raising**. That non-raising
behaviour is what the entire exit-code contract is built on — the runner needs the real
container exit code, not a Python exception, to decide what happened.

### 2.4 The replay core — `runner.py`

This is the heart of the online layer. `run_replay` does the orchestration around one
replay, and `_build_replay_script` writes the actual shell script that runs inside the
container. Understanding the script is understanding the replay.

`run_replay` first decides how to get the bag into the container. If the bag already
lives under the bag-library directory (which is bind-mounted read-only into the
container), it's used in place — zero copy, which turns multi-minute staging of a 13 GB
bag into nothing. Otherwise the bag is copied into the work directory. Then it resolves
the QoS override file. This is where a recent bug lived and is worth understanding:

> The replay script runs **inside the container**, but the QoS file lives in the
> framework's `configs/` directory on the **host**, which isn't mounted into the
> container. The original code passed the host path straight into the script, so
> `ros2 bag play` inside the container couldn't find the file and died. The fix is to
> **stage the QoS file into the bind-mounted work directory and pass the container path**
> — exactly the same trick already used for the bag. The general principle: anything the
> in-container script references must be a *container* path, and the way you get a host
> file there is to stage it into a mounted directory.

Now the script itself. `_build_replay_script` produces one bash script that runs
everything in a *single shell*, so all the child processes are direct children of that
shell and can be cleaned up together. In order, it:

```
source ROS + the workspace overlay
set a cleanup trap on EXIT/INT/TERM   (SIGINT each process group, escalate to
                                       SIGKILL after a grace period, then chown the
                                       output back to the host user)
start the recorder   (ros2 bag record --storage mcap, the OUTPUT topics)
start the player     (ros2 bag play, ONLY the input topics, --clock,
                      --read-ahead-queue-size 5000, the QoS override)
launch the module    (the module's launch command)
poll until the module's first output topic appears (up to 30s, else exit 1)
wait for the player to finish
```

Every line of that ordering is load-bearing, and each was a deliberate fix:

- **Recorder first, then player, then module.** The recorder must already be listening
  before the module emits its first message, or that message is lost. And the player must
  be running before the module configures itself, because some topics (like the static
  transform tree `/tf_static`) are "latched" — published once and retained for late
  subscribers, like a retained MQTT message — and the module reads them in a brief window
  at startup.
- **`--read-ahead-queue-size 5000`** is the documented fix for multi-second stalls that
  the player otherwise hits on dense bags.
- **`--storage mcap`** writes the output bag in MCAP, which survives a hard kill and is
  read natively by the offline reader.
- **The readiness poll replaces a fixed `sleep`.** The old code slept a few seconds and
  hoped the module was up; if the module crashed on launch, the sleep hid it. Polling for
  the module's first output topic means a crash surfaces as a visible timeout (exit 1)
  instead of being masked.
- **`setsid` + the process-group kill + the trap** guarantee the recorder flushes its
  final chunk and no orphaned `perception_node` survives — whether the run ends cleanly
  or you Ctrl-C it.

A small but important safety note: every topic name spliced into the script is shell-
quoted, and the QoS path is package-relative and existence-checked, never user-supplied —
so there's no command-injection surface in the generated script.

### 2.5 The isolation seam and the executor seam

Two small pieces exist mainly to keep `runner.py` clean and to make future modules drop
in without rewrites.

The **`isolation/`** package is three pure functions that turn a `ModuleSpec` into
strings the runner splices into the script: one builds the `--topics` allow-list from the
input topics, one builds the `key:=value` launch arguments, and one builds a bash
fragment that starts any mock nodes the module needs. For perception that last one
returns an empty string — perception needs no mocks — but it's the seam a later module
with upstream service dependencies will fill, without the runner changing. Keeping these
as pure functions also means they're trivially unit-testable.

**`executor.py`** is the "run a command in a container and return its exit code" contract,
expressed as an abstract base class with a `LocalExecutor` implementation today. It's the
seam for running replays on cloud infrastructure later: a future cloud executor drops in
here without the runner being rewritten. In Phase 1 the runner still calls the container
directly; routing it through this abstraction is the additive step left for the cloud
phase.

### 2.6 The offline metrics layer — `metrics/`

This is the second half of the two-layer split, and its single hard rule is: **nothing
under `metrics/` may import ROS** (`rclpy`). Bag access is through the `rosbags` library
only. That's what lets the whole layer run on a plain laptop.

A few pieces work together:

- **The plugin contract (`base.py`).** Every metric is a class implementing
  `compute(reader, config) -> dict`. Regression metrics (the "vs golden" ones) extend a
  variant that also implements `compare(candidate, baseline, config)` and flag
  themselves with `requires_baseline = True`, so the pipeline knows to skip them when no
  baseline is supplied.
- **The registry (`registry.py`).** The `@register_metric("perception")` decorator adds a
  plugin class to a per-module list at import time, and `get_metric_plugins("perception")`
  returns them. This is the pluggable architecture in one line: a new module is a new
  plugin pack that self-registers on import, with zero changes to the runner or CLI.
- **The reader (`bag_reader.py`).** `BagReader` opens the output bag once, deserializes
  only the topics asked for, and serves them. It also has an `iter_paired` helper that
  time-aligns two topics by timestamp — that's how latency and faithfulness match an
  output frame back to the input frame it came from.
- **The validity metric (`replay_faithfulness.py`).** This computes the max gap between
  messages, the drop rate, and a breach count over the output topics. It is **deliberately
  not in the registry** — the pipeline invokes it explicitly, so it gates as the validity
  tier and is never double-counted as a quality metric. It's also written so an empty or
  starved topic *breaches* validity rather than passing vacuously: a camera that produced
  nothing counts as a maximal gap, not as "no data, no problem." That single choice is
  what made the smoke test correctly return INVALID instead of a false pass.
- **The quality plugins (`metrics/perception/`).** Seven of them — latency, pipeline
  throughput, segmentation coverage, depth validity, cross-camera overlap, action-block
  drift, and collision-box IoU. Their internals are domain logic and out of scope here;
  what matters at the framework level is that they all implement the same `compute`
  contract and self-register.
- **The gate (`report/generator.py`).** This is where the verdict is decided.
  `generate_report` first evaluates every validity-tier threshold against the faithfulness
  numbers; then, only if the run is valid, it evaluates each quality metric against its
  threshold (with the tolerance band — nothing is ever required to be bit-exact). It
  writes `metrics.json` (the machine signal) and an auto-escaped `report.html` (the human
  view), and returns the exit code. One nice property: a threshold with no matching metric,
  or a metric with no threshold, is recorded as a visible "skipped" row rather than
  silently ignored — a gate with no teeth can never quietly pass.

### 2.7 The exit-code contract

Everything above funnels into one number. This is the contract the code emits:

| Exit | Meaning | Where it's decided |
|------|---------|--------------------|
| **0** | PASS — the run was valid and every quality criterion is within tolerance. | `generator.py` |
| **1** | FAIL — the run was valid, but a quality threshold was breached (a real regression). | `generator.py` → `cli.py` |
| **2** | INVALID RUN — replay faithfulness was breached; the replay can't be trusted, so this is *not* a code regression. | `generator.py` → `cli.py` |
| **3** | Setup/replay error — a missing asset, a container crash, or the module never coming up. | `cli.py` |

Codes 1 and 2 are reserved for the metrics verdict; everything operational maps to 3. A
consumer never has to parse a file — the process return code already encodes the outcome,
and the validity-vs-quality distinction (2 vs 1) is the part that keeps the gate honest.

---

## Part 3 — How an input bag flows, and module-wise debugging


It's worth being precise about how a bag moves through the system, and about the one
genuinely powerful workflow this enables.

There are actually **two** bags in play, and conflating them causes confusion:

- The **input (candidate) bag** is what you replay *through* the module this run. It's
  always a CLI argument, never hardcoded.
- The **baseline (golden) bag** is the pinned reference the "vs golden" regression
  metrics compare *against*. It's set in `configs/baselines/<module>/golden.yaml` and
  stays fixed regardless of which input you feed.

The input bag's journey through the code: it's resolved into a `DataRef` (a validated
local path, or one pulled from S3); staged for the container (used in place if it's
already under the bag library, otherwise copied in); and then — the module-wise step —
the replay script plays *only* `module.input_topics`, launches *only* that module, and
records *only* `module.output_topics`. So whatever bag you feed, the replay is sliced to
exactly one module.

That slicing is what makes **incident debugging** work. Suppose you have a bag from a
field incident and want to know whether perception was the culprit:

```bash
replay-module all --module perception \
  --local-bag /path/to/incident_rosbag \
  --version-yaml configs/versions/incident-build.yaml \
  --output /tmp/incident --run-metrics
```

This reproduces perception in isolation against the exact recorded inputs from that
incident, on the exact 10xCode version that was running at the time. Nothing else in the
robot's graph runs. To check a *different* suspect from the *same* incident bag, you just
change `--module` — the bag is module-agnostic and the framework slices it per module.
That is the whole debugging story: one recording, replayed through each module in turn to
localize the fault.

The exit code even classifies the failure for you: a **2** means the replay itself didn't
reproduce faithfully (an infra or bag problem — stalls, drops, missing topics), a **1**
means it reproduced fine and the metrics show a genuine regression, and a **3** means a
setup/asset/launch problem. And if you only want to eyeball the output without grading,
`replay-module run` does the replay alone — no metrics, no preflight gate.

The one requirement for a faithful incident replay: the incident bag must actually
contain the module's input topics (the names must match `configs/modules/<module>.yaml`),
and the matching on-disk assets must be staged. A renamed or missing input topic shows up
honestly as a readiness timeout (exit 3) or a faithfulness breach (exit 2), never as a
silent wrong answer.

---

## Part 4 — What's changing right now (in flight)

Several recent improvements are worth calling out because they're visible in the next demo.
The report/logging changes (01-16, 01-17) and the V2 model swap below have **landed**; the
one genuinely-in-flight item is the CI enablement (last paragraph). None of them change the
core contract above — the `metrics.json` shape and the exit-code meanings are stable.

**Plan 01-16 — a richer, debug-first `report.html`.** Today the HTML report is a bare
table. This change turns it into a report a developer can actually debug from: a summary
section of metric cards with PASS / BREACH / FAIL badges, a per-camera-pair table for the
cross-camera-overlap metric, static plots (latency, pipeline breakdown, depth heatmaps)
generated offline with matplotlib, and a "Debug" section pointing at the run's bag, logs,
and report. The important constraint: this is a **pure presentation layer over the same
data** — the `metrics.json` schema and the exit-code verdict are untouched, and if
matplotlib is missing or a plot fails, the report still renders (graceful degradation).

**Plan 01-17 — stop orphaning the logs.** Today `recorder.log` and `module.log` are
written into a gitignored work directory, never moved to `--output`, and overwritten by
the next run. So a developer who hits a red check has nothing to debug with — which is
exactly the wrong time to lose the logs. This change copies both logs to `<output>/logs/`
on **every** run, success or failure (especially failure, since that's when the logs *are*
the evidence), prints the path, and threads it into the report's Debug section. It does
all this without disturbing the runner's cleanup/trap/kill ordering described in 2.4.

**The V2 model swap (from this round of testing).** The replay now runs the current V2
**EoMT** segmentation model (`flicker_drywall.trt`) instead of the older SegFormer model
the sim config used to pin. Getting there surfaced two real fixes worth knowing: the sim
config's post-processing had to point at the `seg_extract` operator (which matches the
EoMT model's output shape) rather than the legacy argmax operator, and the framework's
preflight asset path is moving to the new engine. The model loaded and ran genuine
inference in the smoke test, which is how we know the end-to-end path is sound.

**The `--symlink-install` and QoS-staging fixes (from this round of testing).** Both are
described in 2.3 and 2.4 above — they're recent and they close real correctness gaps (the
stale-config trap and the in-container QoS-path failure). They're mentioned here too so
the "what changed" list is complete.

**CI enablement (Path A) — genuinely in flight.** The PR gate is moving out of this
framework repo and into **10xCode's CI**, with this framework imported as a library. The
deployable workflow templates now live in `module_replay/ci/10xcode/` (replacing the old
in-repo `replay-gate.yml`/`replay-nightly.yml`): an ephemeral RunsOn L4 GPU runs the replay,
the image is pulled from GHCR, and runtime assets come from S3 at job start. See
`docs/CI-ENABLEMENT-GUIDE.md` (the deploy runbook) and `docs/PERCEPTION-REPLAY-SPEC.md` §7.
This is wiring only — it does not touch the metric code or the exit-code contract.

---

*This walkthrough covers the framework code only. Pair it with `docs/isolation-map.md`
(per-module isolation cost) and the per-module replay contract docs for the perception-
specific details.*
