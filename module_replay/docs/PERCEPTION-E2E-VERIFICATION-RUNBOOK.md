# Perception E2E Verification & CI-Enablement Runbook

> **What this is:** the single, current, hand-holding guide to prove the Module-wise Replay
> Framework's **perception** module works end-to-end — first **locally on the x86 GPU sim box**,
> then **live inside 10xCode CI** — and to sign perception off as *done and dusted* before Phase 2.
>
> **Branch under test:** `perception-ci` (adds the 01.1 **incident loop** on top of `main`, which
> you already ran on the box on 2026-06-18).
>
> **Who runs it:** Saransh, on the RTX 4080 sim box, with Claude assisting step-by-step.
>
> **Supersedes (for anything about the incident loop / current workflow set):**
> `CI-ENABLEMENT-GUIDE.md`, `E2E-TESTING-HANDOFF.md`, `PERCEPTION-REPLAY-SPEC.md §7`,
> `CODE-WALKTHROUGH.md Part 4`. Those four predate 01.1 (last reconciled 2026-06-23) and are
> correct for the *golden* flow but **blind to the incident branch, the 5-file workflow set,
> `INCIDENT_DB_URL`/RDS, the incident S3 bucket, and the current engine filename**. See
> [Appendix C](#appendix-c--what-changed-vs-the-old-docs).

**Legend used throughout:**
`⚙️ PROVISION` = you set up infra/config · `▶ RUN` = a command to execute · `✅ EXPECT` = the
pass condition · `⚠️ GOTCHA` = a known trap · `📸 CAPTURE` = save this as evidence.

---

## 0. The two known blockers — clear these FIRST

Everything downstream depends on these. Neither is a framework bug; both are 10xCode-side config.

### Blocker 1 — `perception_sim.yaml` still names the old model (node won't activate)

`configs/modules/perception.yaml` preflight wants `~/.ros/perception/models/flicker_drywall.trt`,
but **10xCode `dev`'s `perception_sim.yaml` still points at the old `L2_L4_segformer`** (not in the
model bucket). If you run with the default version spec (`configs/versions/default.yaml → branch: dev`),
the perception node **fails to activate** with a missing-model / port-wiring error.

**Fix (same as your 2026-06-18 run):** use a 10xCode branch whose `perception_sim.yaml` carries the
three config fixes (F1/F2/F3 from the e2e README) and pin the framework to it via `--version-yaml`
so `git checkout` is a no-op that preserves those edits:

- **F1** — 6 cameras, **contiguous internal ids 0–5**, each fisheye mapped to the bag's actual index
  (rear→cam0, top→cam1, ft→**cam3**, fb→cam4, l→cam5, r→**cam6**); `camera_ids: [0..5]`.
- **F2** — engine path → `flicker_drywall.trt`; add the missing `seg_extract` block.
- **F3** — (framework side, already committed) `input_topics: camera_{0,1,3,4,5,6}`; preflight asset
  → `flicker_drywall.trt`.

> ⚠️ **GOTCHA — B1 (non-contiguous camera ids):** a source list like `[0,1,3,4,5]` makes the node
> ACTIVATE-FAIL (`Interface port 'rgb_output_5' not found in upstream Subgraph 'camera'`). The
> camera subgraph names ports by position but the consumer references by id. Renumber to `0..5`.

### Blocker 2 — the engine must be the **x86 / sm_89 / TRT-10.3** build

`flicker_drywall.trt` is GPU-arch + TRT-version specific. On the box you already have a working
x86 engine at `~/.ros/perception/models/flicker_drywall.trt` (the one your 2026-06-18 run built by
constant-folding the RoPE `If` nodes out of the ONNX, then `trtexec` — F6). Two demo engines live
on-box: **`flicker_drywall.trt` (good)** and `eomt_semantic.trt` (bad) — make sure the good one is
staged.

> ⚠️ **GOTCHA:** the ARM/Orin engine that used to sit in `s3://10xai-team-models/segmentation/semantic14/`
> will **not** deserialize on the L4/x86 CI runner. **CI (Stage B) now pulls the x86/sm_89 build Stage A
> proved**, from the consolidated bucket: `s3://replay-framework-assets/perception_assets/flicker_drywall.trt`
> (staged 2026-07-01). Stage A's local run was its first real x86 load-test proof.

---

## Stage A — Local verification on the x86 GPU sim box

Goal: prove the framework runs the full perception pipeline faithfully on a real 6-camera bag, that
the **B2/B3/B4 metric fixes** (which landed after 2026-06-18) now produce a **trustworthy verdict**,
and that the **net-new incident loop** works — all without touching 10xCode CI.

### A0 · Prerequisites (mostly already on the box)

| ✅ | Item | Check |
|----|------|-------|
| ☐ | GPU visible | `▶ nvidia-smi` → RTX 4080 shown |
| ☐ | Docker + nvidia-container-toolkit | `▶ docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi` |
| ☐ | 10xCode checkout | `▶ echo $TENXCODE_ROOT` (default `~/workspace/src/10xCode`); planner compose resolves |
| ☐ | Engine (x86) | `▶ ls -la ~/.ros/perception/models/flicker_drywall.trt` (~54 MiB) |
| ☐ | Camera LUTs (6 fisheye dirs) | `▶ ls ~/.ros/perception/calibration/camera_intrinsics/` → `fisheye_{rear,top,ft,fb,l,r}` |
| ☐ | Param tree | `▶ ls $TENXCODE_ROOT/version_system/config/current/param` |
| ☐ | Input bag | `▶ ls $REPLAY_BAG_LIBRARY/rosbag_20260618_125310` (your valid 6-cam bag; `metadata.yaml` present) |
| ☐ | Python 3.10+ | `▶ python3 --version` |

`⚙️ PROVISION` env vars (add to your shell / a `.env`):

```bash
export TENXCODE_ROOT=$HOME/workspace/src/10xCode      # wherever the box has 10xCode
export REPLAY_BAG_LIBRARY=$HOME/data                  # holds rosbag_20260618_125310
# REPLAY_WORK_DIR defaults to <repo>/.replay_work; REPLAY_ROS_DOMAIN_ID defaults to 42
```

### A1 · Install the framework + run the offline unit suite

```bash
cd /home/wisewizard/Origin/RosbagReplay/Replay-Framework/module_replay
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[metrics,viz,dev]"
python -m pytest -q
```

`✅ EXPECT:` the full suite green — **357 passed, 1 skipped, 0 failed** (observed on `perception-ci`,
2026-07-01; the 1 skip is `test_integration_smoke`, which needs a real bag via
`REPLAY_INTEGRATION_BAG_PATH`). This alone exercises the incident verifier, the PR-checklist parser,
the CI workflow-invariant tests, and every offline metric — before you spend a second of GPU time.
> ✔ **Already run 2026-07-01 on the dev box → PASS.** Re-run on the sim box to confirm the install there.

> ⚠️ The old `E2E-TESTING-HANDOFF.md` says "166 passed" — that's stale (many 01.1 plans landed since).

### A2 · Pin a version spec whose `perception_sim.yaml` is correct (Blocker 1)

Create `configs/versions/perception-ci-e2e.yaml` pointing at your fixed 10xCode branch:

```yaml
# The framework checks out this 10xCode branch; because we pin the branch you're already on,
# `git checkout` is a no-op that PRESERVES the F1/F2 perception_sim.yaml edits.
tenxcode:
  branch: saransh-10x/feat/replay-test   # ⚙️ branch with perception_sim.yaml → flicker_drywall.trt + seg_extract + contiguous 6-cam
```

`▶ RUN` a config sanity check (pure read, no Docker/GPU):

```bash
replay-module validate --module perception --version-yaml configs/versions/perception-ci-e2e.yaml
```

`✅ EXPECT:` resolved specs print — container `planner`, colcon package `realtime_perception`,
input topics `camera_{0,1,3,4,5,6}`, tenxcode branch = your branch.

### A3 · SC1 — full golden replay produces a trustworthy verdict

This is the decisive local run. It checks out + builds perception in the container, replays the bag
through the live EoMT pipeline on GPU, records the `_sim` output, computes metrics, and gates.

> ✔ **RAN 2026-07-01 on simp-c-3 (RTX 4080) → clean VALID + PASS (exit 0).** 33 GB output bag, 6 cams
> all producing RGB+semantic+depth `_sim`. B2/B3/B4 confirmed healed: `drop_rate 0.0`, `max_gap 200/breach 0`,
> `latency_p95 8.263 ms` (gated PASS), `pipeline 9.994`, `segmentation 0.096`, `overlap 0.6081` (PASS@0.55),
> `depth_validity 0.0` (ungated, `passed=None`), `mask_iou_vs_golden` skipped. Full record:
> `.planning/phases/01.2-…/01.2-STAGE-A-RESULTS.md`. Re-run on the sim box to reconfirm on any code change.

```bash
replay-module all --module perception \
  --version-yaml configs/versions/perception-ci-e2e.yaml \
  --local-bag $REPLAY_BAG_LIBRARY/rosbag_20260618_125310 \
  --output /tmp/run1 --run-metrics
echo "exit=$?"
```

`✅ EXPECT` (the pipeline **runs fully** and the gate returns an *itemized, trustworthy* verdict):

- Output at `/tmp/run1/`: `replay_output/replay_output_0.mcap` (large, ~tens of GB) + `metadata.yaml`;
  `reports/metrics.json` + `report.html` + `pipeline_breakdown.png`; `logs/module.log` + `recorder.log`.
- 6 internal cameras produce **RGB + semantic + depth**, plus `colored_pointcloud_sim`.
- **The three fixes that were broken on 2026-06-18 are now healed** — verify in `metrics.json`:

| Field | 2026-06-18 (main, broken) | Now (perception-ci, fixed) | Why |
|-------|---------------------------|-----------------------------|-----|
| `replay_faithfulness.drop_rate` | `0.709` (inflated) | **≤ 0.05** (near-zero) | B2: span from header clock, not write clock |
| `latency_p95_ms` value | `null` (dead gate) | **~8 ms**, gated | B3: reads `inference_seg_extract_segmentation` |
| `pipeline_throughput_hz` / `segmentation_coverage` / `depth_validity` headline | `—` | **real scalars** | B4: headline key == metric name |
| `cross_camera_overlap_iou` | `0.5351` FAIL vs 0.75 | now gated at **min 0.55** (observed ~0.61 → PASS) | threshold recalibrated to first real run |

`▶ RUN` quick verdict read:

```bash
python -c "import json; d=json.load(open('/tmp/run1/reports/metrics.json')); \
print('verdict=',d['verdict'],'pass=',d['pass']); \
print('drop_rate=',d['replay_faithfulness']['drop_rate'],'breach=',d['replay_faithfulness']['breach_count'])"
```

`📸 CAPTURE:` copy `/tmp/run1/reports/` (metrics.json + report.html) into the phase evidence dir
`.planning/phases/01.2-perception-e2e-verification-ci-enablement/e2e-local/` for the sign-off record.

> ⚠️ **GOTCHA — a `FAIL` (exit 1) would still satisfy SC1.** SC1 proves the verdict is *trustworthy
> and itemized*, not that it's green. `depth_raw_sim` is genuinely all-zero (a real perception issue,
> deliberately **not** gated in the quality tier) — the run stays VALID regardless. What must NOT
> happen: an INVALID (exit 2) driven by the old drop-rate inflation. *(2026-07-01 the real run came
> back a clean VALID+PASS — stronger than this floor.)* But see the **transient-INVALID finding** in
> Stage B: an occasional exit-2 from a transient depth/pointcloud stall is the validity tier working
> as designed, not a regression — CI must retry on it.

### A4 — Prove the quality gate BITES (cheap offline path, no GPU)

Re-use the recorded bag; tighten one threshold and confirm the gate flips to FAIL. This exercises the
**cheap `metrics` path** the CI metrics job uses.

```bash
cp configs/modules/perception.yaml /tmp/perception.yaml.bak
# temporarily make the latency gate impossible:
sed -i 's/      max: 50.0/      max: 1.0/' configs/modules/perception.yaml
replay-module metrics --module perception --bag /tmp/run1/replay_output --output /tmp/verdict_test
echo "exit=$?"          # ✅ EXPECT 1 (quality FAIL)
cp /tmp/perception.yaml.bak configs/modules/perception.yaml   # restore — do not commit the tweak
```

`✅ EXPECT:` exit **1**; `report.html` shows `latency_p95_ms` as **FAIL**. Restore the config.

### A5 · SC4 — determinism (two runs agree within tolerance)

```bash
replay-module all --module perception --version-yaml configs/versions/perception-ci-e2e.yaml \
  --local-bag $REPLAY_BAG_LIBRARY/rosbag_20260618_125310 --output /tmp/run_a --run-metrics
replay-module all --module perception --version-yaml configs/versions/perception-ci-e2e.yaml \
  --local-bag $REPLAY_BAG_LIBRARY/rosbag_20260618_125310 --output /tmp/run_b --run-metrics
```

`▶ RUN` the delta check (each metric within its `tolerance_band`):

```bash
python - <<'PY'
import json
a=json.load(open('/tmp/run_a/reports/metrics.json')); b=json.load(open('/tmp/run_b/reports/metrics.json'))
va={m['name']:m['value'] for m in a['metrics'] if isinstance(m['value'],(int,float))}
vb={m['name']:m['value'] for m in b['metrics'] if isinstance(m['value'],(int,float))}
for k in va:
    if k in vb: print(f"{k:28} a={va[k]!s:>10} b={vb[k]!s:>10} Δ={abs(va[k]-vb[k]):.4g}")
PY
```

`✅ EXPECT:` deltas small (within each metric's `tolerance_band`: latency ±5 ms, overlap ±0.05, etc.).

### A6 · Tier-3 visualization (CPU-only, opt-in)

```bash
replay-module viz --module perception --bag /tmp/run1/replay_output --output /tmp/viz
ls -la /tmp/viz/viz/*.mp4
```

`✅ EXPECT:` decodable `.mp4`s render (overlap-video + the 2×2 semantic/depth/temporal grids). Viz
never affects the verdict (exit always 0).
> ✔ **RAN 2026-07-01 → 7 mp4s** (`overlap_cam2_cam3` + `combined_cam0..5`), ffmpeg-decode verified.

### A7 · SC-INCIDENT — the net-new incident loop (the point of this branch)

The incident subcommand replays a bag and writes an additive
`metrics.json["incident_verdict"]` = `fixed` | `not_fixed` | `inconclusive`. The 5 perception
detectors (`configs/modules/perception.yaml → incident_detectors`) are:

> ✔ **RAN 2026-07-01 → all three verdict states proven live:** A7a `--incident-key seg_all_background`
> → `fixed` (tripped `[]`); A7b `--incident-id` all-5 → `not_fixed` (tripped `["depth_all_zero"]` —
> the real all-zero-depth signature, exercising the AND-gate); A7c forced `fps_collapse` → `inconclusive`
> on a transient INVALID replay (proving the "never-confirm-on-INVALID" guard), then `not_fixed` on the
> clean re-run. This is exactly the JSON the CI `mark`/`sweep` jobs read.

| Detector | Condition | Fires when |
|----------|-----------|-----------|
| `seg_all_background` | `segmentation_coverage < 0.05` | segmenter outputs ~no foreground |
| `depth_all_zero` | `depth_validity < 0.05` | depth plane invalid/empty |
| `fps_collapse` | `pipeline_throughput_hz < 3.0` | pipeline throughput floor |
| `latency_collapse` | `latency_p95_ms > 100.0` | inference latency blows the budget |
| `timesync_collapse` | `temporal_consistency_mean < 0.2` | cross-camera time-sync falls apart |

**A7a — verify ONE detector against a healthy bag (local-debug mode, `--incident-key`):**

```bash
replay-module incident --module perception \
  --version-yaml configs/versions/perception-ci-e2e.yaml \
  --incident-bag $REPLAY_BAG_LIBRARY/rosbag_20260618_125310 \
  --incident-key seg_all_background --output /tmp/inc_seg
python -c "import json;print(json.load(open('/tmp/inc_seg/reports/metrics.json'))['incident_verdict'])"
```

`✅ EXPECT:` `verdict: "fixed"`, `tripped: []` (a healthy bag does not reproduce the seg-collapse).

**A7b — verify ALL detectors (CI mode, `--incident-id`):**

```bash
replay-module incident --module perception \
  --version-yaml configs/versions/perception-ci-e2e.yaml \
  --incident-bag $REPLAY_BAG_LIBRARY/rosbag_20260618_125310 \
  --incident-id INC-LOCAL-TEST --output /tmp/inc_all
python -c "import json;d=json.load(open('/tmp/inc_all/reports/metrics.json'))['incident_verdict'];print(d)"
```

`✅ EXPECT:` `mode: "all"`, verdict + `tripped` list.
> ⚠️ **GOTCHA:** `depth_all_zero` will legitimately **trip** if depth is genuinely all-zero on this
> bag (a real perception issue, not a framework fault). If so, `verdict = "not_fixed"` with
> `tripped: ["depth_all_zero"]`. That is *correct behavior* — the detector caught a real signature.
> To see a clean `"fixed"`, use a bag/version where depth is valid, or temporarily narrow the run to
> a non-depth detector via `--incident-key`.

**A7c — prove `not_fixed` teeth (force a reproduction):** the honest way is a genuinely broken
version, but the fast local proof is to make a detector trip on a known-healthy metric — e.g.
temporarily set `fps_collapse` threshold to `threshold: 999` so it fires, run A7b, confirm
`verdict: "not_fixed"`, `tripped: ["fps_collapse"]`, then revert. This proves the AND-gate that the
CI `mark` job depends on.

`📸 CAPTURE:` save `/tmp/inc_all/reports/metrics.json` — its `incident_verdict` block is the exact
JSON the CI `mark`/`sweep` jobs read.

### A8 · (Optional) golden regression path — `mask_iou_vs_golden`

The pinned golden is `PROVISIONAL` (`configs/baselines/perception/golden.yaml` → `REPLAY_BASELINES_BUCKET`),
so by default the regression row is a visible **"skipped"** warning (never a silent pass/fail). To
exercise it locally, pass a **local** golden output bag directly:

```bash
replay-module metrics --module perception --bag /tmp/run1/replay_output \
  --baseline /path/to/golden_output_bag --output /tmp/regr
```

`✅ EXPECT:` `mask_iou_vs_golden` computes a per-frame semantic mean-IoU (min 0.98). Defer to Phase 2
if no curated golden exists yet — this is warning-only by design until goldens land.

### Stage A exit criteria

- [ ] A1 unit suite green (~363 passed / 1 skipped)
- [ ] A3 SC1 — full replay produces 6-cam output + a **VALID**, itemized verdict; B2/B3/B4 confirmed healed
- [ ] A4 quality gate bites (exit 1 on a tightened threshold)
- [ ] A5 SC4 — two runs agree within tolerance
- [ ] A6 Tier-3 viz renders
- [ ] A7 incident loop writes a correct `incident_verdict` (fixed / not_fixed proven both directions)
- [ ] evidence captured in `.planning/phases/01.2-…/e2e-local/`

---

## Stage B — Deploy into 10xCode CI and verify live

The perception gate runs **inside 10xCode's CI** (Path A): the framework is `pip install`-ed as a
library, replay runs on a **RunsOn ephemeral L4 GPU** (sm_89, same arch as the 4080), the image comes
from **GHCR**, and runtime assets come from **S3**. Five files ship in `module_replay/ci/10xcode/`.

**How the gate routes every PR** (from `replay-perception-gate.yml`):

```
 pull_request touching perception/v2/realtime_perception/**   (or workflow_dispatch)
        │
        ▼
 ┌─ detect ── ubuntu-latest, no secrets/GPU ──────────────────────────────────┐
 │  • dorny/paths-filter → perception = true/false                            │
 │  • read PR comments → parse_ticked_incident_ids → incident_ids (may be "") │
 └────────────────────────────────────────────────────────────────────────────┘
        │ perception==false → replay SKIPS → gate NEUTRAL-PASS
        │ perception==true (and NOT a fork):
        ▼
 ┌─ replay ── RunsOn replay-gpu (GPU) ─ fork-guarded ─────────────────────────┐
 │  checkout 10xCode PR + framework · GHCR login · pip install[metrics]       │
 │  · OIDC → stage engine+LUTs+bag into ~/.ros · nvidia-smi · headless setup  │
 │  · pin /tmp/version.yaml to the PR SHA                                      │
 │    ├─ incident_ids == ""  → GOLDEN:  replay-module all  --local-bag $BAG    │
 │    └─ incident_ids != ""  → INCIDENT: psql SELECT s3_bag_uri per id →       │
 │                             replay-module incident --incident-id <id> …     │
 │  uploads: output-bag-<run_id>, logs-<run_id>, incident-out-<run_id>         │
 └────────────────────────────────────────────────────────────────────────────┘
        │                                   │
   GOLDEN branch                       INCIDENT branch
        ▼                                   ▼
 ┌─ metrics ─ ubuntu-latest ─┐     ┌─ mark ─ ubuntu-latest ─────────────────────┐
 │ download bag →            │     │ download incident-out → read each           │
 │ replay-module metrics →   │     │ incident_verdict.verdict → UPDATE RDS        │
 │ exit 0 iff doc['pass']    │     │ status='fixed' (only if 'fixed'); FAIL if    │
 └───────────────────────────┘     │ ANY tagged incident is not 'fixed' (D-21)   │
        │                          └──────────────────────────────────────────────┘
        └──────────────┬───────────────────┘
                       ▼
 ┌─ gate ── always() — THE SINGLE REQUIRED CHECK (job context name: `gate`) ──┐
 │  replay skipped → pass · golden → require metrics · incident → require mark │
 └────────────────────────────────────────────────────────────────────────────┘
```

> The **required status check is the job named `gate`** — that one name is all branch protection ever
> needs, for every future module.

Work top-to-bottom: **B0 provisioning → B1 RDS → B2 deploy → B3 dry-run → B4 SC2 → B5 incident loop →
B6 branch protection.** B7 is a troubleshooting matrix.

> ⚠️ **Carry this Stage-A finding into Stage B — transient-INVALID gate stability.** On the sim box,
> 1 of 4 GPU replays came back **INVALID (exit 2)** from a transient ~500 ms gap on all 6 `depth_raw_sim`
> + `colored_pointcloud_sim` (breach_count 7, max_gap 500 > 250); the re-run was clean; camera
> RGB/semantic were never affected. This is the **validity tier working as designed** — exit 2 means
> "infra-noisy replay, don't trust it as a regression signal" — but a blocking PR gate / incident `mark`
> that lands on INVALID would non-deterministically block a clean PR.
>
> ✔ **IMPLEMENTED 2026-07-01 (`replay-perception-gate.yml`):** the **golden** replay retries on INVALID
> (`REPLAY_RETRIES=2` → up to 3 attempts) and only hands a **VALID** bag to the downstream `metrics`
> job — so the golden gate's red/green now reflects **quality only** (a real regression), never a
> transient stall. A quality FAIL (metrics rc=1) is a real regression and is kept immediately (never
> retried). The **incident** branch does the same per bag: it retries while the verdict is
> `inconclusive` (the INVALID-equivalent) so the `mark` AND-gate sees a definitive `fixed`/`not_fixed`.
> If all attempts are INVALID (a *persistent*, not transient, breach) the job fails as **infra** (clear
> `::error::` for a human) — never a silent pass. Thresholds stay strict (owner: depth not produced
> yet; recalibration is Aniket's call). Suite still green (357 passed). See `01.2-STAGE-A-RESULTS.md`
> Finding 3. **In B3, confirm the retry fires** by watching a dispatch where one attempt logs the
> `INVALID … retrying` warning and a later attempt goes VALID.

---

### B0 · Provisioning — do every sub-step before deploying the workflows

Each sub-step ends with **✅ worked when…** so you can checkpoint. Run the `gh`/`aws` commands from a
shell authenticated to the org (`gh auth status` shows `OriginAutonomy` access; `aws sts
get-caller-identity` shows account `390403890757`).

#### B0.1 — Confirm RunsOn is installed on the org, then define the `replay-gpu` profile

RunsOn boots the ephemeral GPU EC2. It's installed **once per org** (a GitHub App on `OriginAutonomy`
+ a CloudFormation stack in the AWS account). 10xSim already uses it (`sim-gpu`), so it almost
certainly exists.

- **Confirm:** open any recent 10xSim Actions run that used `runs-on=…/runner=sim-gpu` → it got a
  machine (not stuck "Queued").
- **If missing:** follow runs-on.com's install (CloudFormation one-click in the org AWS account +
  install the GitHub App). One-time platform task.

Then add the `replay-gpu` profile by **merging** `runs-on.merge.yml` into 10xCode's `.github/runs-on.yml`
(this is part of B2, previewed here so you know what it is):

```yaml
# .github/runs-on.yml  — ADD this block under the existing `runners:` key; do NOT overwrite the file
runners:
  # ... 10xCode's existing sim-gpu etc. stay here ...
  replay-gpu:
    family: ["g6"]              # NVIDIA L4 — Ada sm_89, same arch as the RTX 4080 (engine loads as-is)
    image: ubuntu24-gpu-x64     # NVIDIA drivers preinstalled
    cpu: [16]                   # colcon build of perception is CPU-bound
    spot: false                 # on-demand: a blocking gate must not lose its machine
    volume: 250gb               # ~20 GB image + ~25 GB output bag + build tree
    extras: s3-cache+ecr-cache  # pip cache + Docker-layer cache (the ~20 GB image isn't re-pulled)
```

`✅ worked when:` a 10xSim `sim-gpu` run provisioned a machine, and you have the `replay-gpu` block
ready to merge.

#### B0.2 — Create the two GitHub PATs

1. **`FRAMEWORK_READ_TOKEN`** (read-only clone of the framework, cross-org until it moves in-org):
   GitHub → your **Settings → Developer settings → Fine-grained tokens → Generate** → Resource owner
   = you/`saranshshankar`, Repository access = **only `Replay-Framework`**, Permissions →
   **Contents: Read-only**. Copy the token.
2. **`DOCKER_DEPLOYMENT_PAT`** (pull the planner image from GHCR): the org almost certainly already
   has this (it's what `v2-docker-ci.yml` uses to push). It needs **`read:packages`** on
   `ghcr.io/originautonomy/*`. If reusing the org one, just confirm it's readable from 10xCode Actions.

`✅ worked when:` you hold both token strings (you'll set them as secrets in B0.6).

#### B0.3 — Create the S3-read OIDC role → `REPLAY_ROLE_ARN`

The privileged `replay` job needs short-lived S3 read creds (no long-lived keys). This is the only AWS
role in the system, and it touches **S3 only** (the image is on GHCR).

**(a)** Ensure the GitHub OIDC identity provider exists in the account (create once if not):

```bash
aws iam list-open-id-connect-providers   # look for token.actions.githubusercontent.com
# if absent:
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

**(b)** Trust policy — scope it to the 10xCode repo (`trust.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::390403890757:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
      "StringLike":   { "token.actions.githubusercontent.com:sub": "repo:OriginAutonomy/10xCode:*" }
    }
  }]
}
```

**(c)** Permissions policy — S3 read on the asset + incident buckets (`perms.json`). Replace `<INCIDENT_BUCKET>`
with your incident-bag bucket name:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::replay-framework-assets", "arn:aws:s3:::replay-framework-assets/*",
      "arn:aws:s3:::<INCIDENT_BUCKET>",       "arn:aws:s3:::<INCIDENT_BUCKET>/*"
    ]
  }]
}
```

**(d)** Create the role + attach the policy, and capture the ARN:

```bash
aws iam create-role --role-name replay-perception-ci \
  --assume-role-policy-document file://trust.json
aws iam put-role-policy --role-name replay-perception-ci \
  --policy-name s3-read --policy-document file://perms.json
aws iam get-role --role-name replay-perception-ci --query 'Role.Arn' --output text   # → REPLAY_ROLE_ARN
```

> ✔ **Region (reconciled 2026-07-01):** all perception assets now live in the single
> `replay-framework-assets` bucket in **`ap-south-1`**, and all three workflows (gate/sweep/nightly)
> assume `ap-south-1` with no per-command `--region` overrides — so there is no cross-region access left.
> If you stage assets in a different-region bucket later, either set that job's `aws-region` to match or
> add `--region <bucket-region>` on the specific `aws s3` step.

`✅ worked when:` `aws iam get-role` prints an ARN, and a quick `aws s3 ls s3://replay-framework-assets/perception_assets/calibration/camera_intrinsics/`
under that role lists the LUT dirs.

#### B0.4 — Confirm/point the x86 engine → `PERCEPTION_ENGINE_S3`

The exact sm_89 engine Stage A3 loaded on the box is now staged in the consolidated bucket (2026-07-01):
`s3://replay-framework-assets/perception_assets/flicker_drywall.trt` (~57 MB) — point `PERCEPTION_ENGINE_S3`
at it. **A3 is its real load-test:** it deserialized on the RTX 4080 (sm_89, same arch as the L4 runner),
so no rebuild is needed. (Only if a future L4's TRT minor differs would you rebuild from the ONNX per
`CI-ENABLEMENT-GUIDE.md` Part 5a: `polygraphy surgeon sanitize --fold-constants` → `trtexec --fp16
--saveEngine=…` dynamic batch 1–6, then re-upload.) `eomt_semantic.trt` sits beside it as the known-bad
counter-example — do NOT point the var at that one.

`✅ worked when:` `aws s3 ls <PERCEPTION_ENGINE_S3>` shows the `.trt`, and A3 proved it deserializes.

#### B0.5 — Curate the golden input bag → `PERCEPTION_BAG_S3` + `PERCEPTION_BAG_PATH`

The 6-cam `rosbag_20260618_125310` (the bag Stage A3/A7 used) is already staged (2026-07-01) at
`s3://replay-framework-assets/rosbags/rosbag_20260618_125310/` — set `PERCEPTION_BAG_S3` to it and
`PERCEPTION_BAG_PATH` to the runner-local sync target (e.g. `/mnt/efs/bags/perception_ci`). LUTs also
need no upload — they're in the same bucket at
`s3://replay-framework-assets/perception_assets/calibration/camera_intrinsics/` (the workflow syncs them
into `~/.ros/perception/calibration/...`).

`✅ worked when:` `aws s3 ls <PERCEPTION_BAG_S3>` shows a rosbag2 (`metadata.yaml` + `*.mcap`).

#### B0.6 — Set every secret + variable on 10xCode

Via `gh` (or the UI: **Settings → Secrets and variables → Actions**):

```bash
R=OriginAutonomy/10xCode
# secrets
gh secret   set FRAMEWORK_READ_TOKEN --repo $R --body "<the fine-grained PAT>"
gh secret   set DOCKER_DEPLOYMENT_PAT --repo $R --body "<org GHCR PAT>"   # skip if already org-wide
gh secret   set INCIDENT_DB_URL       --repo $R --body "<set in B1>"       # set after RDS exists
# variables
gh variable set REPLAY_ROLE_ARN       --repo $R --body "arn:aws:iam::390403890757:role/replay-perception-ci"
gh variable set PERCEPTION_ENGINE_S3  --repo $R --body "s3://replay-framework-assets/perception_assets/flicker_drywall.trt"
gh variable set PERCEPTION_BAG_S3     --repo $R --body "s3://replay-framework-assets/rosbags/rosbag_20260618_125310/"
gh variable set PERCEPTION_BAG_PATH   --repo $R --body "/mnt/efs/bags/perception_ci"
```

`✅ worked when:` `gh secret list --repo $R` and `gh variable list --repo $R` show all seven names.

---

### B1 · Create the RDS incidents table + roles → set `INCIDENT_DB_URL`

> ✔ **DONE 2026-07-01** on RDS `database-1.cviscog0koyp.ap-south-1.rds.amazonaws.com` (Postgres 17.9).
> Created a **dedicated** database `module_replay` (the 20 pre-existing DBs were left untouched — verified
> before/after), applied the DDL there, and created two **least-privilege** login roles scoped to *only*
> `module_replay_incidents` (non-superuser, no CREATEDB/CREATEROLE — the CI credential can never reach the
> other databases). Verified: 21 columns, 3 indexes, `replay_gate` = SELECT + UPDATE on
> `{status,fixed,fixed_by_pr,fixed_by_sha,fixed_by_run,fixed_at}`, `replay_worker` = INSERT + UPDATE(s3_bag_uri).
> The `INCIDENT_DB_URL` (gate role) was generated in-session. The role names + passwords + full
> connection URLs are recorded in the **git-ignored** file
> `.planning/phases/01.2-…/CREDENTIALS.local.md` (roles `replay_gate` / `replay_worker`) — kept out of
> this tracked doc so they never enter git history. **Set `INCIDENT_DB_URL` as the GitHub secret** from
> that file (command below).

The exact steps that were run (recorded for reproducibility / other modules):

```bash
HOST=database-1.cviscog0koyp.ap-south-1.rds.amazonaws.com
MASTER="postgresql://postgres:<master-pw>@${HOST}/postgres"          # admin, default DB
# 1. dedicated DB (additive — never touches the other databases)
psql "$MASTER" -c "CREATE DATABASE module_replay;"
NEWDB="postgresql://postgres:<master-pw>@${HOST}/module_replay?sslmode=require"
# 2. least-privilege roles (cluster-global logins, no superuser)
psql "$NEWDB" -c "CREATE ROLE replay_gate   LOGIN PASSWORD '<gate-pw>';"
psql "$NEWDB" -c "CREATE ROLE replay_worker LOGIN PASSWORD '<worker-pw>';"
# 3. the idempotent DDL
psql "$NEWDB" -f module_replay/ci/infra/incidents_table.sql
# 4. narrow grants (ONLY module_replay / ONLY the incidents table)
psql "$NEWDB" \
  -c "GRANT CONNECT ON DATABASE module_replay TO replay_gate, replay_worker;" \
  -c "GRANT USAGE ON SCHEMA public TO replay_gate, replay_worker;" \
  -c "GRANT SELECT, UPDATE (status, fixed, fixed_by_pr, fixed_by_sha, fixed_at, fixed_by_run) ON module_replay_incidents TO replay_gate;" \
  -c "GRANT INSERT, UPDATE (s3_bag_uri) ON module_replay_incidents TO replay_worker;"
```

Set the secret (use the real gate password from the session output):

```bash
gh secret set INCIDENT_DB_URL --repo OriginAutonomy/10xCode \
  --body "postgresql://replay_gate:<gate-pw>@database-1.cviscog0koyp.ap-south-1.rds.amazonaws.com/module_replay?sslmode=require"
```

> **Security posture / honest caveat:** `replay_gate` has **zero object privileges** outside
> `module_replay_incidents` and is non-superuser, so it cannot read or modify any of the other 20
> databases. It *can technically* `CONNECT` to them (Postgres grants `CONNECT` to `PUBLIC` by default) —
> revoking that would mean altering the pre-existing databases' config, which we deliberately did **not**
> do (owner: don't touch existing DBs). Object-level least privilege is the guarantee that matters here.
>
> The gate/sweep also **degrade gracefully** if `INCIDENT_DB_URL` is unset — they still verify verdicts and
> AND-gate; only the RDS UPDATE is skipped. So the golden path can go live before the secret is set.

---

### B2 · Deploy the workflows + merge the RunsOn fragment (one PR into 10xCode)

On a branch in 10xCode:

```bash
# from a 10xCode checkout, with the framework repo available at ../Replay-Framework
cp ../Replay-Framework/module_replay/ci/10xcode/replay-perception-gate.yml    .github/workflows/
cp ../Replay-Framework/module_replay/ci/10xcode/replay-perception-sweep.yml   .github/workflows/
cp ../Replay-Framework/module_replay/ci/10xcode/replay-perception-nightly.yml .github/workflows/
cp ../Replay-Framework/module_replay/ci/10xcode/replay-perception-viz.yml     .github/workflows/
# MERGE the replay-gpu block into the existing runs-on.yml (do NOT overwrite — see B0.1)
$EDITOR .github/runs-on.yml
```

Then open a PR titled e.g. `ci: add perception replay gate + incident loop`.

Two things to confirm in that PR:
- **Path filter matches reality:** the gate triggers on `perception/v2/realtime_perception/**` (gate
  line 22). If perception lives elsewhere in 10xCode, edit the `paths:` and the `dorny/paths-filter`
  `filters:` block to match.
- **`perception_sim.yaml` model currency** (Blocker 1): the branch/`dev` must point at
  `flicker_drywall.trt` for the node to activate. The gate pins `/tmp/version.yaml` to the PR SHA, so
  the PR's own `perception_sim.yaml` is what's used.

`✅ worked when:` the PR shows the 4 workflow files + the merged `runs-on.yml`, and 10xCode's existing
CI stays green.

---

### B3 · Dry-run each workflow via `workflow_dispatch` (before branch protection)

From 10xCode → **Actions**, "Run workflow" on each:

```bash
gh workflow run replay-perception-gate.yml    --repo OriginAutonomy/10xCode   # golden path (no incidents)
gh workflow run replay-perception-nightly.yml --repo OriginAutonomy/10xCode
gh workflow run replay-perception-viz.yml     --repo OriginAutonomy/10xCode -f run_id=<a gate run id>
gh run watch --repo OriginAutonomy/10xCode $(gh run list --repo OriginAutonomy/10xCode -w replay-perception-gate.yml -L1 --json databaseId -q '.[0].databaseId')
```

`✅ EXPECT:`
- **gate**: `replay` provisions a GPU (you'll see `nvidia-smi` in the log) → produces `replay_output`
  → uploads `output-bag-<run_id>` → `metrics` downloads it, computes, exits on `doc['pass']` →
  `gate` green (or a trustworthy INVALID/FAIL with `report-<run_id>` attached). The run **Summary**
  lists the artifact names.
- **nightly**: smoke replay green + the determinism step passes (or names the metric that drifted
  beyond its `tolerance_band`).
- **viz**: `viz-<run_id>` mp4 artifacts.

If anything fails here, jump to **B7** — do not proceed to branch protection until the golden dry-run
is green.

---

### B4 · SC2 — live red / green / neutral on real PRs

1. **Clean PR** touching `perception/v2/realtime_perception/**` (a harmless change) →
   `✅ EXPECT` **gate green** (detect→replay→metrics→gate, golden path).
2. **Bad PR** that regresses perception (e.g. force the segmenter to emit ~no foreground, or tighten
   nothing but break an output) → `✅ EXPECT` **gate red** — `metrics` exits 1 on `doc['pass']==false`;
   `report-<run_id>` shows which quality metric failed; the `$GITHUB_STEP_SUMMARY` links the artifacts.
3. **Untouched PR** (no perception files) → `✅ EXPECT` **neutral pass** — `detect.perception==false`,
   `replay` skips, `gate` exits 0.

Read the result from the `gate` check on the PR, or:

```bash
gh pr checks <PR#> --repo OriginAutonomy/10xCode   # the line named `gate` is the one that matters
```

`📸 CAPTURE:` the run URLs for the green, red, and neutral PRs (SC2 evidence).

---

### B5 · SC-INCIDENT (live) — the full incident loop, end to end

This proves detect→incident-replay→`mark`→RDS, in both the `fixed` and `not_fixed` directions, plus
the sweep. You'll drive the incident bag + the checklist comment manually (the auto-posting record-side
workflow `incident-checklist.yml` is part of the 10xCode record-side PRs, not this deployable set).

**B5.1 — Seed one open incident in RDS**, pointing at a bag that reproduces a known failure (e.g. an
all-background segmentation bag → trips `seg_all_background`):

```sql
INSERT INTO module_replay_incidents
  (incident_id, module_name, status, s3_bag_uri, error_code, title, ts)
VALUES
  ('INC-20260701-1200-segcollapse', 'perception', 'open',
   's3://<INCIDENT_BUCKET>/incidents/perception/INC-20260701-1200-segcollapse/',
   'SEG-COLLAPSE', 'Segmentation collapsed to all-background', now());
```

> ⚠️ The `incident_id` must match `[A-Za-z0-9_-]+` — the checklist parser drops anything else at the
> source (CR-01 SQL-injection guard). And the bag at `s3_bag_uri` must be a rosbag2 dir the OIDC role
> can read.

**B5.2 — Post the PR checklist comment** (the exact marker the gate greps for — the first line is
mandatory), and tick the incident:

```bash
cat > /tmp/checklist.md <<'MD'
<!-- module-replay-incident-checklist -->
## Open incidents for `perception`

- [x] INC-20260701-1200-segcollapse — Segmentation collapsed to all-background (SEG-COLLAPSE)
MD
gh pr comment <PR#> --repo OriginAutonomy/10xCode --body-file /tmp/checklist.md
```

**B5.3 — The PR that *fixes* the incident.** With that ticked comment present, push the fix to the PR
(touching `perception/v2/realtime_perception/**` so `detect` fires). What happens:

- `detect` → `incident_ids = "INC-20260701-1200-segcollapse"` (non-empty → **incident branch**).
- `replay` → `psql SELECT s3_bag_uri` for that id → `aws s3 cp` the bag → `replay-module incident
  --incident-bag <staged> --incident-id INC-… --version-yaml /tmp/version.yaml` → writes
  `incident-out-<run_id>/INC-…/reports/metrics.json` with `incident_verdict`.
- `mark` → reads `incident_verdict.verdict`; if **`fixed`** → `UPDATE … SET status='fixed', fixed=true,
  fixed_by_pr=<PR#>, … WHERE incident_id=… AND status<>'fixed'`; if **`not_fixed`/`inconclusive`** →
  the job **fails** (gate red).

`✅ EXPECT (fix works):` gate **green**; then confirm the RDS row flipped:

```bash
psql "$INCIDENT_DB_URL" -c \
  "SELECT incident_id,status,fixed,fixed_by_pr,fixed_by_sha FROM module_replay_incidents WHERE incident_id='INC-20260701-1200-segcollapse';"
# → status=fixed, fixed=t, fixed_by_pr=<PR#>
```

**B5.4 — Prove the `not_fixed` teeth.** Repeat B5.1–B5.3 but point `s3_bag_uri` at a bag that *still*
reproduces the failure (or open the fixing PR against code that doesn't actually fix it).
`✅ EXPECT:` `mark` fails → **gate red**; the RDS row stays `status='open'`; the run's
`incident-out-<run_id>` shows `verdict: not_fixed`, `tripped: ["seg_all_background"]`.

**B5.5 — Sweep (fixed-stays-fixed).** With at least one open incident whose bag reproduces:

```bash
gh workflow run replay-perception-sweep.yml --repo OriginAutonomy/10xCode -f module=perception
```

`✅ EXPECT:` the sweep `SELECT DISTINCT ON (error_code)` picks the open incidents, replays each, and on
a `not_fixed` verdict runs `UPDATE … SET status='open', fixed=false` and **fails the job** (blocking a
QA cut). Green only when nothing reproduces.

`📸 CAPTURE:` the RDS row before/after, and the green (fixed) + red (not_fixed) + sweep run URLs.

---

### B6 · Branch protection — make `gate` required

GitHub → 10xCode → **Settings → Branches → (protected branch) → Require status checks to pass** →
search and select **`gate`**. Or via API:

```bash
gh api -X PATCH repos/OriginAutonomy/10xCode/branches/<branch>/protection/required_status_checks \
  -f 'contexts[]=gate'
```

`✅ worked when:` a red `gate` blocks the merge button on a regressing perception PR; a green `gate`
(or a neutral-skip) lets it merge. Adding a future module never changes this — each ships its own
`gate` job under the same context name.

---

### B7 · Troubleshooting matrix (first-run failures)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `replay` stuck **"Queued"** | RunsOn not picking up the `replay-gpu` label | Re-check B0.1; ensure `runs-on.yml` is at `.github/runs-on.yml` on the branch being run; confirm the org RunsOn stack covers 10xCode |
| `replay` fails at **preflight, exit 3** | a staged asset path is missing | The log **names the exact path**. Check the OIDC role can read it (B0.3), the engine/LUT/bag S3 URIs (B0.4/B0.5), and the `~/.ros` remap |
| **Engine won't deserialize** on the runner | ARM engine, or TRT-minor mismatch vs the L4 | Ensure `PERCEPTION_ENGINE_S3` is the **x86/sm_89** build; rebuild on an L4 if the TRT minor differs (Part 5a) |
| Node **ACTIVATE-FAIL** / missing model | `perception_sim.yaml` names the old `L2_L4_segformer` (Blocker 1) or non-contiguous cam ids (B1) | Repoint the PR's `perception_sim.yaml` at `flicker_drywall.trt`; renumber cameras to `0..5` |
| **GHCR login/pull fails** | `DOCKER_DEPLOYMENT_PAT` not visible to 10xCode or lacks `read:packages`, or the package hasn't granted 10xCode read | Fix the secret/scope; grant the planner package read access to 10xCode |
| **OIDC `AssumeRoleWithWebIdentity` denied** | trust `sub` doesn't match, or provider missing | Confirm the trust `sub` is `repo:OriginAutonomy/10xCode:*` and the OIDC provider exists (B0.3a) |
| **psql** connection refused / auth | RDS security group, sslmode, or `gate_user` grants | Test `psql "$INCIDENT_DB_URL" -c 'select 1'` from a runner-like network; add `?sslmode=require`; re-check B1 grants |
| gate is **green but should've caught a regression** | the regression isn't in a **gated** metric (e.g. `depth_validity` is ungated by design) | Confirm which metric moved; only gated thresholds fail the golden gate — see Appendix B |
| gate **red** / incident **`inconclusive`** from a **transient** ~500 ms depth/pointcloud stall (INVALID, exit 2) | validity tier caught infra noise, not a regression (Stage-A Finding 3); ~1-in-4 on the box | **Already handled** — the replay job retries on INVALID (`REPLAY_RETRIES=2`) so a VALID bag reaches metrics/mark. If it still goes red, all 3 attempts were INVALID → a *persistent* infra/runner problem (check `nvidia-smi`, runner disk/IO, the `::error::` line), not the PR. Do NOT relax gap tolerance (owner: keep strict) |
| privileged jobs **skipped on a PR** | the PR is from a **fork** (fork guard `head.repo==repo`) | Push the branch internally to 10xCode; forks intentionally can't get GPU/secrets |
| gate **runs on an unrelated PR** / doesn't run on a perception PR | `paths:` filter mismatch | Align the `paths:` + `dorny/paths-filter` block with where perception lives (B2) |

---

### Stage B exit criteria

- [ ] B0 provisioning complete (RunsOn `replay-gpu`, both PATs, OIDC role, engine, bag, all 7 vars/secrets)
- [ ] B1 RDS table + `gate_user`/`worker_user` grants applied; `INCIDENT_DB_URL` set + tested
- [ ] B2 4 workflows deployed; RunsOn fragment **merged** (not overwritten); path filter confirmed
- [ ] B3 gate/nightly/viz dry-run green via dispatch
- [ ] B4 SC2 — clean PR green, bad PR red, untouched PR neutral-skip (URLs captured)
- [ ] B5 incident loop: `fixed` marks RDS + gate green; `not_fixed` blocks + stays open; sweep reopens
- [ ] B6 `gate` is the single required check; red blocks merge

---

## Definition of Done — "perception done and dusted"

Perception is signed off when **all** hold and evidence is captured:

| # | Criterion | Where proven | Status (2026-07-01) |
|---|-----------|--------------|---------------------|
| SC1 | Faithful isolated replay → trustworthy itemized verdict (B2/B3/B4 healed) | A3 | ✅ clean VALID+PASS |
| SC-Q | Quality gate demonstrably bites | A4 (manual) / offline suite | ⏭ A4 skipped; covered by the A1 suite (`test_metrics_report`, exit-code paths, `test_e2e_negative_control`) |
| SC4 | Two replays agree within tolerance | A5 / B3 nightly | ⏭ A5 skipped → **prove via the nightly determinism step in B3** (only data so far: 3 VALID/1 transient-INVALID) |
| VIZ | Tier-3 debug videos render | A6 / B3 viz | ✅ 7 mp4s |
| SC-INC | Incident loop verdict correct both directions; live mark + sweep | A7 / B5 | ✅ all 3 states local (A7); ◻ live mark+sweep (B5) |
| SC2 | Live CI red on bad PR, green on clean, neutral-skip on untouched | B4 | ◻ Stage B |
| SC3 | Nightly smoke green | B3 | ◻ Stage B |
| DOCS | Stale docs reconciled ([Appendix C](#appendix-c--what-changed-vs-the-old-docs)) | — | ◻ open |
| UAT | Owner sign-off recorded in `01-HUMAN-UAT.md` (closes the SC2–SC4 items) | — | ◻ open |

When these are checked, run `/gsd-verify-work` for phase 01.2 and mark the phase complete, then
transition to **Phase 2: Sensor Fusion + Localization**.

---

## Appendix A — CLI quick reference

| Command | Path | Docker/GPU | Purpose |
|---------|------|-----------|---------|
| `validate` | offline | no | resolve + print module/version spec |
| `fetch-data` | offline | no | resolve a bag (local or S3) |
| `setup-env` | full | Docker | checkout + `docker compose up` + colcon build |
| `run` | full | Docker+GPU | replay a bag → record `_sim` output |
| `all` | full | Docker+GPU | fetch → setup → **preflight** → replay → (`--run-metrics`) gate → (`--run-viz`) |
| `metrics` | **cheap** | no | offline metrics + report + gate on an existing output bag |
| `incident` | full | Docker+GPU | replay incident bag → additive `incident_verdict` (`--incident-id` all / `--incident-key` one) |
| `viz` | cheap | no | render Tier-3 mp4s from an output bag |

**Exit-code contract (B9):** `0` PASS · `1` quality FAIL · `2` INVALID RUN (validity breach — infra
noise, beats 1) · `3` setup/replay error (missing preflight asset, container/ROS crash).

## Appendix B — Perception config at a glance (`configs/modules/perception.yaml`)

- **Input:** `camera_{0,1,3,4,5,6}/image_raw`, `lidar_{107,108}/points`, `/tf`,`/tf_static`.
- **Output:** `camera_{0..5}/{image,semantic,depth}_raw_sim`, `colored_pointcloud_sim`, `diagnostics`.
- **Rates:** default 10 Hz; `image_raw_sim`/`semantic_raw_sim` 5 Hz (EoMT ~5 FPS); `diagnostics` 0.2 Hz.
- **Validity gates:** `replay_max_gap_ms ≤ 250`, `replay_drop_rate ≤ 0.05`, `replay_breach_count = 0`.
- **Quality gates:** `latency_p95_ms ≤ 50`, `cross_camera_overlap_iou ≥ 0.55`,
  `pipeline_throughput_hz ≥ 3.0`, `segmentation_coverage ≥ 0.05`, `mask_iou_vs_golden ≥ 0.98`
  (regression, needs `--baseline`). `depth_validity` is intentionally **not** gated (real all-zero
  issue to fix first) — but the `depth_all_zero` incident detector still watches it.
- **Metric pack = 6** (action_block + collision_box were dropped as out-of-scope 3D proxies).
- **Preflight assets:** engine + 6 LUT dirs + `$TENXCODE_ROOT/version_system/config/current/param`.

## Appendix C — What changed vs. the old docs

The four docs in `module_replay/docs/` predate the 01.1 incident loop. This runbook is authoritative;
they need these fixes (tracked as a phase-01.2 doc-reconciliation task):

1. **Workflow set:** it's **5 files** (`gate`, `sweep`, `nightly`, `viz` + `runs-on.merge.yml`), not
   "three files" and not `runs-on.yml`. Sweep + viz are entirely undocumented.
2. **Incident branch:** the gate branches on `detect.outputs.incident_ids`; the `mark` job, the
   incident AND-gate, RDS marking, and sweep reopen-on-reproduce are absent from all four docs.
3. **New provisioning:** `INCIDENT_DB_URL` secret, the RDS `module_replay_incidents` table + grants,
   and the incident S3 bucket (added to `REPLAY_ROLE_ARN`) are missing from every checklist.
4. **Engine filename:** `E2E-TESTING-HANDOFF.md §3` names `L2_L4_segformer_B1_x86_b6_fp16.trt`;
   the real preflight asset is `flicker_drywall.trt` (EoMT swap).
5. **Metric count:** `CODE-WALKTHROUGH.md` self-contradicts (7 vs 6); it's **6**.
6. **Test count:** "166 passed" is stale (~363 now).
7. **AWS regions & buckets (reconciled 2026-07-01):** all perception CI assets (engine, LUTs, golden
   bag) are consolidated into `s3://replay-framework-assets` (**`ap-south-1`**); gate/sweep/nightly all
   assume `ap-south-1` with no `--region` overrides. The old `eu-north-1`/`us-east-1`/`rosbags-10x`
   references are superseded.

---

*Runbook created 2026-07-01 for phase 01.2. Ground-truth sources: `module_replay/replay/cli.py`,
`module_replay/ci/10xcode/*.yml`, `configs/modules/perception.yaml`, and the real e2e run at
`.planning/phases/01-framework-perception-ci-gate/e2e-2026-06-18/`.*
