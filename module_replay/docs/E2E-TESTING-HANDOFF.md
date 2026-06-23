# E2E Testing Handoff — Perception Replay on the x86 GPU Machine

> **Audience:** a fresh Claude Code agent running on the **x86 GPU box**, helping the user run the
> `replay-module` workflow **live** (planner container up, GPU replay through the real ROS 2 node
> graph) and validate Success Criteria SC1–SC4 for Phase 1.
>
> **Why this doc exists:** all the work below was built and validated **offline** on a dev machine
> (162→166 unit tests green, pushed to `origin/main`). The four things that *cannot* be tested
> without a GPU + curated bag — live faithful replay (SC1), the CI red/green gate (SC2), the nightly
> smoke (SC3), and FP16 determinism (SC4) — are your job. This doc is **self-contained**: the dev
> machine's `.planning/` and `KT/` directories are gitignored local symlinks and are **NOT** in this
> repo, so everything you need is here or in committed `module_replay/` code.

---

## 0. Your mission (TL;DR)

Run, on a real x86 GPU machine with a curated perception bag:

```bash
replay-module all --module perception --local-bag <curated_bag> --output /tmp/run1 --run-metrics
```

This should: check out the 10xCode `dev` branch → `docker compose up` the **planner** container
(`v2-planner-docker-x86`) → `colcon build realtime_perception` → play the input bag through the live
perception node graph → record the output topics to an MCAP bag → run the offline metrics → write
`report.html` + `metrics.json` → exit `0` (PASS) / `1` (quality FAIL) / `2` (INVALID run) / `3`
(setup or replay error).

Then validate SC1–SC4 (Section 5). Report results so the dev-side agent can close
`01-HUMAN-UAT.md`.

**You will need the user to provide:** the curated perception input bag, and confirmation that the
perception assets (TensorRT engine, camera-intrinsics LUTs, `config/current` param tree) are present
on the box (Section 3). Ask for these first — nothing runs without them.

---

## 1. What this project is

**Module-wise Replay Framework** — replays a single 10xCode module (perception is the pilot) from a
recorded rosbag through a live ROS 2 node graph in a container, then computes **offline** regression
metrics on the output bag and emits a trustworthy pass/fail verdict for CI.

- **Replay** = GPU/ROS, runs the real node graph in a Docker container (the only heavy step).
- **Metrics** = pure-Python offline analysis of the output bag (`rosbags` lib, no rclpy, no GPU).
- **Verdict** = exit code a CI gate reads.

Repo: `https://github.com/saranshshankar/Replay-Framework.git`, branch `main`. All code lives under
`module_replay/`. The console entry point is **`replay-module`** (→ `replay.cli:main`).

---

## 2. Current state — what's done vs what you're testing

**Done & pushed (offline-validated):**
- Full Phase 1 framework: replay layer (`replay/runner.py`, `docker_utils.py`, `env_setup.py`),
  offline metrics layer (`replay/metrics/`), the `replay-module` CLI, and 3 CI workflows
  (`.github/workflows/`).
- A **metrics-correctness rebuild** just landed (an audit found the original metrics untrustworthy;
  8 gaps were fixed across plans 01-10..01-15 + a residual). **166 unit tests pass, 1 env-gated skip.**
- The perception plugin pack is now **6 metrics**: 5 intrinsic (`latency`, `pipeline`,
  `segmentation`, `depth`, `overlap`) + 1 regression (`mask_iou_vs_golden`). Two earlier metrics
  (`action_block`, `collision_box`) were deleted as out-of-scope.

**NOT yet tested — your job (these need the GPU box):**
- **SC1** — live faithful replay produces a faithful output bag + `report.html` + `metrics.json`,
  exit nonzero on a criterion violation.
- **SC2** — a breaching PR is blocked red by the CI gate; a clean PR passes.
- **SC3** — the nightly smoke job replays a fixed bag and exits green.
- **SC4** — two identical replays produce metrics within the tolerance band (FP16 determinism).

The offline metrics path (plugins → verdict → exit code) is unit-tested and trustworthy; what's
unproven is **the live GPU replay feeding it real data**.

---

## 3. Prerequisites on the GPU box (check these FIRST)

Ask the user to confirm each. If any is missing, stop and report — don't guess.

| # | Requirement | How to check |
|---|-------------|--------------|
| 1 | x86 machine with an NVIDIA GPU + CUDA; `nvidia-smi` works | `nvidia-smi` |
| 2 | Docker + `docker compose` + nvidia-container-toolkit | `docker compose version`; `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi` |
| 3 | **10xCode checkout** present; path exported as `TENXCODE_ROOT` (default `~/workspace/src/10xCode`). Its working tree should be clean — the framework will `git checkout dev` there. | `ls $TENXCODE_ROOT/.devcontainer/v2/planner/docker-compose.yml` |
| 4 | The planner compose file resolves and the image is pullable (ECR). | `docker compose -f $TENXCODE_ROOT/.devcontainer/v2/planner/docker-compose.yml config` |
| 5 | **Perception assets** (the preflight gate checks these host paths, exit 3 if missing): TensorRT engine `~/.ros/perception/models/L2_L4_segformer_B1_x86_b6_fp16.trt`; 6 fisheye LUT dirs under `~/.ros/perception/calibration/camera_intrinsics/<cam>/luts`; `$TENXCODE_ROOT/version_system/config/current/param`. | `replay-module validate --module perception` then check `preflight_assets` paths exist |
| 6 | **A curated perception input bag** (a rosbag2 directory recorded from the robot). This is the platform-team item "Q-K" — the user must supply it. | `ls <bag>/metadata.yaml` |
| 7 | Python 3.10+ for the venv | `python3 --version` |

> **TensorRT caveat:** engines are GPU-arch + TRT-version specific. If the `.trt` engine was built
> for a different GPU, perception will fail to load it. The engine must match this box's GPU, or be
> rebuilt/cached on it.

---

## 4. Setup on the GPU box

```bash
# 1. Clone the framework repo (this handoff comes with it)
git clone https://github.com/saranshshankar/Replay-Framework.git
cd Replay-Framework

# 2. Python venv with the metrics extra (offline deps: rosbags, numpy, opencv, jinja2, boto3, ...)
python3 -m venv module_replay/.venv
module_replay/.venv/bin/pip install -e "module_replay[metrics]"
module_replay/.venv/bin/pip install -e "module_replay[dev]"   # if you want to run the unit suite

# 3. Make the CLI available (either use the venv's console script or python -m)
module_replay/.venv/bin/replay-module --help

# 4. Confirm the offline suite is green on this box (sanity)
module_replay/.venv/bin/python -m pytest module_replay/tests/ -q   # expect 166 passed, 1 skipped

# 5. Point at the 10xCode checkout (skip if already the default ~/workspace/src/10xCode)
export TENXCODE_ROOT=/path/to/10xCode
```

> **Note on `.planning/` and `KT/`:** on the dev machine these are gitignored symlinks into a separate
> reference workspace. They are **not** in this clone and you don't need them. The only committed
> design doc you may want is `module_replay/docs/isolation-map.md` (per-module isolation cost map).
>
> **Note on `REPLAY_WORK_DIR`:** `setup_environment` exports it automatically from the repo location
> (the `.replay_work` host bind-mount source). You normally don't set it. If you run a stage that
> bypasses `setup_environment`, export it to `<repo>/module_replay/.replay_work`.
>
> **Note on the bag library mount:** the planner/controller compose override mounts
> `${REPLAY_BAG_LIBRARY:-~/data}` read-only into the container. Set `REPLAY_BAG_LIBRARY` to the dir
> holding your curated bag if it isn't under `~/data`.

---

## 5. The tests — exact commands + what success looks like

### SC1 — live faithful replay (the big one)

```bash
module_replay/.venv/bin/replay-module all \
  --module perception \
  --local-bag /path/to/curated_perception_bag \
  --output /tmp/run1 \
  --run-metrics
```

What it does, in order (see `replay/cli.py::all_cmd` → `env_setup.setup_environment` → `runner.run_replay`):
1. `_load_specs` — loads `configs/modules/perception.yaml` + version (defaults to 10xCode `dev` via
   `configs/versions/default.yaml`; override with `--version-yaml <file>`).
2. `setup_environment` — `git checkout dev` in `$TENXCODE_ROOT`, `docker compose up -d` the
   **planner** container (`v2-planner-docker-x86`), `colcon build --packages-up-to realtime_perception`.
3. **Preflight gate** — `missing_preflight_assets`: if the TRT engine / LUTs / param tree are
   missing, it prints the missing path and exits **3** (a named setup error, not a metrics verdict).
4. `run_replay` — starts the MCAP recorder, runs `ros2 bag play` of the input bag into the container
   (read-ahead queue 5000; `/tf_static` QoS override; a topic-readiness wait, not a fixed sleep),
   launches `ros2 launch realtime_perception perception_node.launch.py use_replay:=true`, records the
   ~20 perception output topics, then cleans up (setsid/trap/kill escalation/chown). **Exits 3 on any
   replay/container failure** (this is a critical fix — it used to always report success).
5. `--run-metrics` — runs the 6 perception plugins + faithfulness on the output bag → `generate_report`
   → writes `/tmp/run1/.../report.html` + `metrics.json` → `sys.exit(verdict)`.

**Success (SC1):**
- An output MCAP bag exists with perception's outputs (6×`*_sim` image streams + semantics + depth + `colored_pointcloud_sim` + `/perception_node/diagnostics`).
- `report.html` and `metrics.json` exist.
- On a healthy bag: `metrics.json` → `"verdict": "PASS"`, process exit **0**.
- Faithfulness is sane: `max_gap_ms` ≈ 100 ms (NOT ~5000 — see Section 7), `drop_rate` ≈ 0,
  `breach_count` = 0.
- To prove the gate bites (the "exit nonzero on violation" half of SC1): tighten a threshold in
  `configs/modules/perception.yaml` (e.g. `latency_p95_ms.max: 1`) and re-run the **cheap** metrics
  step (no replay needed) on the same output bag:
  ```bash
  module_replay/.venv/bin/replay-module metrics --module perception \
    --bag /tmp/run1/<output_bag> --output /tmp/verdict_test
  echo "exit=$?"   # expect 1 (quality FAIL); revert the threshold afterwards
  ```

### SC4 — determinism (do this right after SC1; reuses the same setup)

Run the replay **twice** on the same input bag, compute metrics for each, and diff the metric values:

```bash
module_replay/.venv/bin/replay-module all --module perception --local-bag <bag> --output /tmp/run_a --run-metrics
module_replay/.venv/bin/replay-module all --module perception --local-bag <bag> --output /tmp/run_b --run-metrics
# compare /tmp/run_a/.../metrics.json vs /tmp/run_b/.../metrics.json
```

**Success (SC4):** each metric's value differs by less than its `tolerance_band` in
`perception.yaml`. FP16 inference is **not** bit-exact, which is exactly why the thresholds carry
tolerance bands and why `mask_iou_vs_golden`'s floor is 0.98, not 1.0. If two runs diverge by more
than the band, that's a real finding — report it.

### SC2 / SC3 — CI gate red/green + nightly (Path A: GitHub Actions inside 10xCode, RunsOn GPU)

The gate runs **inside 10xCode's CI** with this framework imported as a library; the deployable
templates live in `module_replay/ci/10xcode/` and `docs/CI-ENABLEMENT-GUIDE.md` is the deploy runbook.
These need the platform-team items (Q-F/Q-J/Q-K):
- **RunsOn on OriginAutonomy** (ephemeral L4 GPU; the `replay-gpu` profile in `runs-on.yml`) — not a
  babysat self-hosted box.
- **`DOCKER_DEPLOYMENT_PAT`** for the GHCR image pull (the org's existing credential; no AWS) + an
  **S3-read OIDC role** (`REPLAY_ROLE_ARN`) for the runtime assets (engine/LUTs/bag). ECR is not used.
- the curated perception CI bag + the S3 schema.
- Templates (deploy into 10xCode `.github/`): `replay-perception-gate.yml` (PR gate: `paths:`-filter
  detection → RunsOn GPU replay → cheap-runner metrics → summary gate) and `replay-perception-nightly.yml`
  (cron smoke + second-replay determinism). `lint-and-test.yml` (unit suite) stays in this framework repo.

**SC2 success:** open a PR that makes perception breach a criterion (e.g. forces latency > 50 ms or
overlap < 0.75) → the `gate` job goes **red** and blocks; a clean PR's `gate` job is **green**.
Make the `gate` job a required check in branch protection.

**SC3 success:** `replay-perception-nightly.yml` (scheduled or `workflow_dispatch`) replays the one fixed
perception bag, evaluates it, and exits green.

> If RunsOn / the S3-read role / CI bag aren't provisioned yet, SC2/SC3 are blocked on the
> platform team — validate SC1 + SC4 locally first and report SC2/SC3 as infra-pending.

---

## 6. How the verdict works (so you can read results correctly)

Two **independent** ways a metric passes/fails, plus three tiers:

- **Validity tier** (faithfulness, always on): is the replay itself trustworthy? Gates
  `replay_max_gap_ms` (≤200 ms, computed over uniform ~10 Hz topics), `replay_drop_rate` (≤0.05),
  `replay_breach_count` (=0, the rate-aware per-topic stall count). **Breach → exit 2 INVALID** —
  quality metrics are not even trusted.
- **Quality tier** (intrinsic, no baseline needed): is the output good in absolute terms?
  `latency_p95_ms` ≤ 50 ms (parsed from `/perception_node/diagnostics` `seg_argmax` compute time),
  `cross_camera_overlap_iou` ≥ 0.75 (pixel-wise semantic-label agreement in the warped overlap
  region). AND-gated. **Breach → exit 1 FAIL.** (`segmentation`, `depth`, `pipeline` are
  informational — recorded, not gated.)
- **Regression tier** (needs a baseline): is the output worse than a known-good golden?
  `mask_iou_vs_golden` ≥ 0.98 (per-frame semantic-mask IoU vs the golden). **Only runs when you pass
  `--baseline`.**

**Exit codes:** `0` PASS · `1` quality/regression FAIL · `2` INVALID run (validity) · `3` setup or
replay error.

**`--baseline` is OPTIONAL** (`default=None`). Without it, validity + quality gates still fully run;
the regression metric shows a visible "skipped" warning row and does **not** block. With it:
```bash
replay-module metrics --module perception --bag <run> --output <dir> --baseline <golden_bag>
```
A local rosbag2 path is used directly as the golden; any other value resolves the pinned golden via
`BaselineManager` (`configs/baselines/perception/golden.yaml` → S3). The golden bags (Q-K) don't
exist yet, so today the gate runs **without** `--baseline` and the regression metric is warning-only.
You can smoke-test the regression path by passing a **copy of the run's own output bag** as
`--baseline` → IoU ≈ 1.0 → pass.

---

## 7. Perception-specific facts you must know (verified against 10xCode `dev@001223f9b`)

- **Output catalog (~20 topics):** 6×`camera_N/image_raw_sim` (RGB passthrough), 6×`semantic_raw_sim`
  (rgba8: **R channel = class id**, G = confidence), 6×`depth_raw_sim` (**32FC4, ch0 = depth in
  meters**, lidar-interpolated), `colored_pointcloud_sim`, and `/perception_node/diagnostics`.
- **`/perception_node/diagnostics` runs at 0.2 Hz** (`report_interval_sec: 5.0` → ~5000 ms interval).
  This is *expected and healthy* — the faithfulness metric uses a per-topic expected rate
  (diagnostics 0.2 Hz, cameras 10 Hz) and **excludes** diagnostics from the headline `max_gap_ms`, so
  a healthy run shows `max_gap_ms` ≈ 100 ms, not 5000. If you see `max_gap_ms` ≈ 5000 and exit 2 on a
  healthy bag, something regressed in the per-topic rate handling — report it.
- **Timestamp-sanctity:** perception outputs carry the *input frame's* capture stamp, so an
  input→output stamp delta is ≈0. That's why `latency_p95_ms` reads the node's self-reported
  `seg_argmax` compute time from `/diagnostics`, NOT a stamp delta.
- **Camera time-sync is all-or-nothing:** TimeSync emits all 6 cameras or none. A starved/unequal
  camera set is a real red flag (faithfulness flags `cross_camera_count_mismatch`).
- **`/imu/filtered` has no producer in V2; `points_sim`/`pointcloud_merged_sim`/`imu_sim` are dead in
  sim** — don't expect them. `depth_raw_sim` *does* publish (lidar-interpolated).
- Launch: `ros2 launch realtime_perception perception_node.launch.py use_replay:=true`. Container:
  `v2-planner-docker-x86`. colcon package: `realtime_perception`.

---

## 8. Known risks / gotchas for the live run

- **Preflight exit 3 ≠ metrics failure.** If you get exit 3 with a path printed, an asset is missing
  (Section 3, item 5) — it's a setup problem, not a perception regression.
- **First run pulls + builds** (image pull + colcon). Budget time; use `--build-jobs 1` if the box is
  memory-constrained, `--partial-checkout` to check out only perception's 10xCode paths.
- **The framework drives 10xCode git + colcon.** It will `git checkout dev` in `$TENXCODE_ROOT` and
  build in the container. Make sure that checkout has no uncommitted work you care about.
- **Don't trust a green that's actually empty.** If the output bag has zero frames on some topics,
  faithfulness should mark the run INVALID (anti-vacuous rule). If it passes with empty topics, that's
  a bug — report it.
- **The `metrics` subcommand is the cheap path** (no Docker/GPU) — use it to iterate on verdict logic
  against an already-recorded output bag without re-replaying.

---

## 9. What was already validated offline (context, not to redo)

The metrics correctness rebuild fixed these (all unit-tested, `origin/main`):
1. faithfulness no longer false-INVALIDs healthy mixed-rate runs (diagnostics rate handling)
2. `mask_iou_vs_golden` implemented + `--baseline` wired end-to-end, fail-closed (regression detection)
3. `overlap` restored to real [0,1] semantic-label agreement (0.75 gate meaningful)
4. `latency_p95_ms` reads `/diagnostics` seg_argmax AND the 50 ms gate actually enforces (was dead)
5. `depth` scoped to depth topics; 6. WR-03 validity fail-closed; 7. `--run-viz` is a documented
   deferral; 8. pipeline/segmentation topic-scoping.

`--run-viz` is a **no-op deferral** in Phase 1 (visualizations not ported) — don't expect plots.

---

## 10. What to report back

For each of SC1–SC4: command run, exit code, the key `metrics.json` fields
(`verdict`, `max_gap_ms`, `breach_count`, `drop_rate`, `latency_p95_ms`, `cross_camera_overlap_iou`),
and any anomaly. The dev-side agent will use this to close the items in `01-HUMAN-UAT.md` and finalize
Phase 1. If you hit a real bug in the live path (not an env/asset issue), capture the exact command +
output + the relevant `metrics.json` so it can be turned into a gap-closure plan.

---

*Generated 2026-06-16 from the dev-machine workspace. Repo state at handoff: `main` @ `df1016e`
(gap-closure batch 01-10..01-15 + gap-1 residual), 166 unit tests passing.*
