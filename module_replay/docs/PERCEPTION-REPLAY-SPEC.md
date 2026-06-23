# Perception Module-Wise Replay — Technical Spec

**Audience:** module + platform leads · **Status:** local e2e working on a real sim bag; CI enablement pending platform infra · **Date:** 2026-06-18

---

## 1. What this is

A framework that takes a **recorded perception input bag**, replays it through the **real `perception_node`** (unmodified, in its production container), records the node's outputs to a new bag, and computes an **offline, CPU-only pass/fail verdict** over those outputs. The goal is a trustworthy regression gate: a code change that degrades perception output is caught by a red CI check; a clean change passes.

Two things run in two very different places:

| Stage | Where | Cost |
|---|---|---|
| **Replay** (play bag → perception → record output bag) | GPU box / self-hosted GPU CI runner, inside the planner container | Heavy (GPU, TensorRT) |
| **Metrics** (read output bag → verdict) | Any machine / cheap CPU CI runner | Light (pure-Python, no ROS, no GPU) |

This split is deliberate: the expensive GPU replay produces an artifact (the output bag); the cheap metrics step is what actually gates CI and can be re-run by any developer locally.

---

## 2. How the local e2e runs perception

One command runs the whole chain:

```
replay-module all --module perception \
  --version-yaml <version.yaml> \
  --local-bag <input_bag>  |  --task-id <id> \
  --output <dir> \
  --run-metrics [--baseline <golden_bag>]
```

Sequence:

1. **Fetch** the input bag (local path or pulled from S3 by task id).
2. **Set up** the container: check out the target 10xCode branch, `docker compose up` the planner container with replay mounts, build.
3. **Preflight**: fail fast if required assets are missing (TensorRT engine, fisheye calibration LUTs, version-system params) — before burning a GPU replay.
4. **Replay** inside the container, as one orchestrated shell:
   - start a `ros2 bag record` (MCAP) on the perception output topics;
   - `ros2 bag play <input> --clock` with `use_sim_time` (sim-time playback) and a large read-ahead buffer; a QoS override republishes `/tf_static` as transient-local so the node sees it during configuration;
   - launch `perception_node` with `use_replay:=true` (selects the sim config, subscribes to bag topics instead of hardware);
   - a **readiness loop** waits for the node's first output (no fixed sleep); cleanup is trap-based so the recorder/module/player are always torn down.
5. **Metrics**: the offline pipeline reads the output bag and writes `report.html` + `metrics.json`, and exits with the verdict code.
6. **Artifacts**: the output bag, `recorder.log` + `module.log`, and the report all land under `<output>/` so a failed run is debuggable.

The metrics step can also be run standalone on an already-recorded output bag (no GPU/Docker):

```
replay-module metrics --module perception --bag <output_bag> --output <dir> [--baseline <golden_bag>]
```

---

## 3. Input contract — what the replay consumes

**Input bag topics** (the recorded raw sensor streams played into the node):

| Topic | Type | Notes |
|---|---|---|
| `/perception_node/camera_{0,1,3,4,5,6}/image_raw` | `sensor_msgs/Image` | The 6 fisheye cameras. **All six must be present and mutually stamped within ~55 ms** — the node's time-sync is all-or-nothing: one silent camera ⇒ zero output on all six. |
| `/perception_node/lidar_{107,108}/points` | `sensor_msgs/PointCloud2` | Feed the lidar-interpolated depth + colored pointcloud. A missing lidar degrades (not kills) those outputs. |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | `/tf_static` is the **only** source of the `*_corrected` extrinsic frames the node needs at configure time. Must be present and flowing before the node starts. |

> Camera numbering: the recorded bag's physical cameras are `0,1,3,4,5,6`; the node renumbers them to a contiguous `0..5` internally, so **output topics are `camera_0..5`**.

**Provisioned assets** (host-side, not in the bag — required or the node fails to activate):

- the TensorRT segmentation engine (`.trt`, sim variant, FP16, batch-6);
- the fisheye LUT calibration tree (3 files per camera);
- a populated version-system param tree;
- (the static `*_corrected` TF frames, supplied via the bag's `/tf_static`).

---

## 4. Output contract — what the recorded output bag has

Perception republishes its outputs with a **`_sim` suffix** (diagnostics is the exception — no suffix). The framework records exactly these:

| Topic | Type | Content |
|---|---|---|
| `/perception_node/camera_0..5/image_raw_sim` | `Image` | RGB passthrough of each camera frame |
| `/perception_node/camera_0..5/semantic_raw_sim` | `Image` (rgba8, 448×448) | **Primary output.** Segmentation mask: R channel = class id, G = confidence |
| `/perception_node/camera_0..5/depth_raw_sim` | `Image` (32FC4, 448×448) | Lidar-interpolated metric depth (ch0 = depth in metres) |
| `/perception_node/colored_pointcloud_sim` | `PointCloud2` | TF-dependent colored cloud |
| `/perception_node/diagnostics` | `DiagnosticArray` | Per-stage compute timings; published on a ~5 s wall timer (~0.2 Hz) |

Two properties the metrics layer relies on:

- **Timestamp sanctity:** every output message's `header.stamp` equals its source frame's capture stamp from the input bag. Input↔output frames join by exact stamp; no nearest-neighbour windowing.
- **Non-consuming reads:** the node can re-emit the same frame (same stamp) when a tick outpaces a topic, so the raw message count overcounts. **Metrics de-duplicate by `header.stamp`** and count unique frames.

Topics that are dead in sim (per-lidar `points_sim`, `pointcloud_merged_sim`, IMU `_sim`) are intentionally **not** recorded, so count-based checks don't false-fail.

---

## 5. How metrics are computed from the output bag

The metrics layer is a pure-Python plugin set over the recorded output bag. It produces a two-tier verdict.

### Tier 1 — Validity (is the replay itself trustworthy?)

A run is only judged on quality if the replay faithfully reproduced the recording. Validity is computed per output topic, over **unique header stamps**, against each topic's **expected rate**:

| Signal | How it's computed | Gate |
|---|---|---|
| `replay_max_gap_ms` | Worst inter-frame gap among the **uniform-rate** (~10 Hz) topics. Slow housekeeping topics (diagnostics at 0.2 Hz) are excluded from this headline. | ≤ 200 ms |
| `replay_drop_rate` | `(expected − actual) / expected`, where `expected = span × per-topic rate`. The span is the **header-clock** window of the perception outputs (the diagnostics topic, which runs on a longer wall-timer window, is excluded so it can't inflate the span). | ≤ 0.05 |
| `replay_breach_count` | Count of per-topic gaps exceeding `tolerance_factor × expected_period`. The factor is per-topic: 2× for uniform streams; **4× for the inference-limited semantic/RGB streams** so their genuine ~600 ms EoMT inference stalls aren't counted as replay failures, while a real replay hang still breaches. | 0 |

A validity breach ⇒ verdict **INVALID RUN** — the quality metrics are not trusted. (This is distinct from a quality failure: an INVALID run means "we can't judge this," not "perception is broken.")

Key sim-time correctness point: because the recorder stamps wall-clock write times while headers carry robot/sim time, the rate math is done strictly on **header stamps** — otherwise a faithful sim-time replay reads a spurious ~4× drop rate.

### Tier 2 — Quality (is perception's output good?)

Computed only when validity passes:

| Metric | How it's computed | Gate |
|---|---|---|
| `latency_p95_ms` | p95 of the node's self-reported segmentation-inference compute time, read from the `diagnostics` stream | ≤ 50 ms |
| `cross_camera_overlap_iou` | For each adjacent camera pair: estimate a homography from RGB feature matches (AKAZE, CPU), warp one camera's **class-id mask** into the other, and measure **pixel-wise semantic agreement** in the overlapping region (0–1). Reported as the mean over pairs. | ≥ 0.75 |
| `mask_iou_vs_golden` | Per-frame semantic-mask IoU vs a **pinned golden** output bag. Requires `--baseline`; without one it's a visible "skipped" row (warning-only), never a silent pass. | ≥ 0.98 |
| `pipeline_throughput_hz`, `segmentation_coverage`, `depth_validity` | Effective per-topic publish rate; fraction of non-background class pixels; fraction of valid depth pixels. | Informational (no gate) |

### Verdict & artifacts

- **AND-gate:** all gated quality metrics must pass.
- **Exit codes:** `0` PASS · `1` quality FAIL · `2` INVALID RUN (validity) · `3` setup/replay error.
- **`metrics.json`** — the machine signal CI reads (`pass`, `verdict`, per-metric rows).
- **`report.html`** — human-facing: summary cards with PASS/BREACH/FAIL badges, the validity breakdown, the per-camera-pair overlap table, latency/throughput plots, and a Debug section linking the run's bag + logs.

> All thresholds above are marked **provisional** — they are best-guesses pending module-owner (perception lead) sign-off against empirical multi-run data.

---

## 6. Status — what's been validated

A real 6-camera sim bag has been run end-to-end on the GPU box: perception activated on all six cameras, the TensorRT engine loaded and inference ran, and the full output (RGB + semantic + depth + colored pointcloud + diagnostics) was recorded and scored. The metrics layer's correctness was verified against that real output (input/output contracts, the sim-time rate math, the inference-limited stall handling, and the diagnostics-stage latency read were all corrected to match the real run). The cheap metrics path reproduces the verdict offline.

Net: **local e2e is working and produces a trustworthy validity verdict.** Remaining work is CI infra (below) and threshold calibration with the perception lead.

---

## 7. CI enablement — how this becomes a gate

The gate runs **inside 10xCode's CI** (Path A): perception changes live in 10xCode PRs, and the GPU runner (RunsOn) + the container image (GHCR) are both org-scoped to OriginAutonomy, so running in-org needs zero cross-org plumbing. This framework is imported as a **library** — 10xCode checks it out, `pip install`s it, and runs `replay-module` against the PR's perception code. The three deployable workflow files live in `module_replay/ci/10xcode/`; `docs/CI-ENABLEMENT-GUIDE.md` is the click-by-click deploy runbook.

Three workflows are implemented:

- **PR gate** — a 3-job pipeline: **replay** (RunsOn ephemeral L4 GPU; pulls the planner image from GHCR; replay only) → **metrics** (free `ubuntu-latest`, no GPU; reads `metrics.json["pass"]`) → **gate** (the single required check; red on a non-zero verdict). Change detection is the workflow's `paths:` filter (CI-05), not a separate job. The output bag, logs, and report are uploaded as ephemeral artifacts (5-day retention) with a debug link surfaced on the run.
- **Nightly** — replays one fixed perception bag, plus a determinism check (two replays must agree within each metric's tolerance band).
- **Lint + unit tests** — on every push/PR (cheap, no GPU; stays in this framework repo).

Security posture: PR-triggered (never `pull_request_target`); the image pulled from GHCR with the org's existing `DOCKER_DEPLOYMENT_PAT` (no AWS for the image); OIDC only for the S3-read of runtime assets (no long-lived keys); pinned action versions; least-privilege permissions.

### What the platform team must provide (the inputs)

The gate logic is implemented and unit-tested; it is **blocked only on infra/assets**:

| Input | Purpose |
|---|---|
| **RunsOn on OriginAutonomy** (ephemeral GPU EC2; `replay-gpu` profile = NVIDIA L4 / sm_89) | boots the machine that runs the replay job (TensorRT inference) |
| **`DOCKER_DEPLOYMENT_PAT`** (the org's existing GHCR credential — likely already set) | pulls the planner image from GHCR, 10xCode's primary registry; ECR is only a robot mirror |
| **OIDC S3-read role** (exposed as a CI variable) | downloads the runtime assets (engine, LUTs, bag) from S3 at job start — no static keys, no ECR |
| **x86 / sm_89 TensorRT engine** (built once from the ONNX — guide Part 5a) | the S3 `flicker_drywall.trt` is an ARM/Orin build and will **not** deserialize on the x86 L4 |
| **Curated perception input bag** + its assets (fisheye LUT tree, a `/tf_static` carrying the `*_corrected` frames) | the fixture the gate replays |
| **Pinned golden output bag** | enables the `mask_iou_vs_golden` regression metric (warning-only until provided) |
| **S3 key schema** for fixtures/outputs | how the CI fetches inputs and stores artifacts |

Once these land, branch protection makes the **gate** job the single required check, and a perception-affecting PR is gated end-to-end.

### Open items

- Thresholds are provisional — calibrate with the perception lead against multi-run data (the inference-limited streams in particular: rate, stall tolerance, drop rate).
- `mask_iou_vs_golden` is dormant until a golden bag is pinned.
- Heavy debugging visualizations (overlap/semantic comparison videos) are deferred to a local-only tool; CI stays light (report + plots only).
