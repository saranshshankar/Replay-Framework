# CI Enablement Guide — Perception PR Gate (Module-wise Replay Framework)

**Audience:** the platform owner standing this up (you) — written assuming **no prior CI
knowledge**. Part 9 is a click-by-click deploy walkthrough; Parts 1–8 are the reference.
**Goal:** a perception PR in **10xCode** gets automatically replay-tested and blocked if it
regresses — green when clean.
**Scope:** infra + CI wiring only. The framework's metric code is already bug-fixed and
verified locally (the 2026-06-18 e2e + follow-up fixes; a valid sim bag flips the verdict
correctly).

**Verified against:** 10xCode `dev@aae1d5827`, OriginAutonomy/10xSim PR #453 (the RunsOn GPU
pattern we copy), Replay-Framework `main`, and RunsOn docs (runs-on.com). Citations inline.

---

## Part 0 — The whole thing in one picture

We chose **Path A: the gate runs inside 10xCode's CI**, importing this framework as a library.
Why: perception changes live in 10xCode PRs, so detection and gating belong there; and the GPU
runner (RunsOn) + the container image (GHCR) are both **org-scoped to OriginAutonomy** — running
in-org means they "just work" with zero cross-org plumbing.

```
 A 10xCode dev opens a PR that edits perception/v2/realtime_perception/**
        │  (TRIGGER)
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Job: replay      machine: a FRESH ephemeral GPU EC2 (RunsOn boots it)      │
 │   1. check out the PR's 10xCode code                                       │
 │   2. check out THIS framework + `pip install` it  (imported as a library)  │
 │   3. log in to GHCR (org PAT) → pull the v2-planner-docker-x86 image       │
 │   4. `replay-module all` : build the PR's perception, play the bag in,     │
 │      record perception's outputs to a NEW bag                              │
 │   5. upload that output bag as an artifact   ── then RunsOn destroys the VM │
 └──────────────────────────────────────────────────────────────────────────┘
        │ output bag (artifact)
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Job: metrics     machine: a free GitHub `ubuntu-latest` (no GPU)           │
 │   download the bag → `replay-module metrics` → metrics.json + report.html  │
 │   exit 0=PASS / 1=FAIL / 2=INVALID                                         │
 └──────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Job: gate    the single ✅/❌ GitHub shows on the PR (branch protection)    │
 └──────────────────────────────────────────────────────────────────────────┘
```

### Decisions (and why)
| Decision | Choice | Why |
|---|---|---|
| **Where the gate runs** | **Inside 10xCode CI** (Path A); framework imported as a library | Perception PRs are in 10xCode; RunsOn + GHCR are org-scoped. |
| **GPU machine** | **RunsOn ephemeral EC2, `g6` / NVIDIA L4** | Org standard (10xSim#453). L4 is Ada **sm_89 — same arch as the dev-box RTX 4080**, so the prebuilt engine loads without a rebuild. |
| **Metrics machine** | GitHub-hosted `ubuntu-latest` | Pure-Python, no GPU/ROS/secrets — effectively free. |
| **Container registry** | **GHCR** (`ghcr.io/originautonomy`), via the org's existing `DOCKER_DEPLOYMENT_PAT` | GHCR is 10xCode's primary registry (`v2-docker-ci.yml`); ECR is only a robot mirror. In-org (Path A) the org PAT pulls it with **no AWS role**. |
| **Input persistence** | **Download assets from S3 at job start (NO custom AMI)** | Engine (~54 MiB) + LUTs are small runtime assets, NOT baked into the image; downloading per-run takes seconds and beats AMI maintenance. AMI is optional, only worth it for a very large fixed bag. Part 10. |
| **Bags (now)** | One fixed bag staged on EFS/S3 | S3 schema (Q-F) + fixtures (Q-K) resolved in Part 6; not a blocker for first green. |

---

## Part 1 — Vocabulary (skip if you know CI)

- **Workflow** — a YAML file in `.github/workflows/` saying "when X happens, run these steps."
- **Runner** — the actual computer that executes the steps. Two kinds:
  - **GitHub-hosted** (`runs-on: ubuntu-latest`): a fresh throwaway machine GitHub rents you. No GPU.
  - **Self-hosted**: your own machine. Needed for a GPU. Can be *persistent* (one box you babysit)
    or *ephemeral* (a fresh box per job, then destroyed).
- **`runs-on:`** — the line in a job that picks the machine, by label.
- **RunsOn** (runs-on.com) — a service the org installs once that makes *ephemeral self-hosted*
  runners automatic: it boots an EC2 machine in **your own AWS account** for each job and tears
  it down after. Configured by a file `.github/runs-on.yml` (machine profiles) + a magic
  `runs-on:` label. **Three different things are spelled "runs-on" — keep them straight:**
  the YAML key `runs-on:`, the service *RunsOn*, and its config file `runs-on.yml`.
- **OIDC** — a way for a workflow to get short-lived AWS credentials without storing long-lived
  keys. The workflow proves "I'm GitHub Actions for repo X" and AWS hands back temporary creds.
- **Artifact** — a file one job uploads and another downloads (how our two jobs pass the bag).
- **Vars / Secrets** — small config (vars) and credentials (secrets) you set once in the repo's
  GitHub settings; the workflow reads them as `${{ vars.X }}` / `${{ secrets.X }}`.

---

## Part 2 — Current state (what exists, what you add)

**This framework repo** now ships:
- `.github/workflows/lint-and-test.yml` — the framework's *own* unit suite (runs on framework PRs).
- `module_replay/ci/10xcode/` — **the three files you deploy into 10xCode** (Part 7):
  `runs-on.yml`, `replay-perception-gate.yml`, `replay-perception-nightly.yml`.
- the `replay-module` CLI (`replay` / `metrics` / `all` subcommands) and `configs/modules/perception.yaml`.

> The old framework-repo `replay-gate.yml`/`replay-nightly.yml` were **removed** — under Path A
> the gate runs in 10xCode, not here. Their content lives on as the `ci/10xcode/` templates.

**10xCode** already has: a self-hosted GPU CI pool (gate-4/gate-5 sim run on `self-hosted`
GPU); `v2-docker-ci.yml` that builds `ghcr.io/originautonomy/v2-planner-docker-x86` and
**mirrors it to ECR** (`v2-docker-ci.yml:549-642`, *"robots pull ECR"*); and the
`repository_dispatch` cross-repo pattern. You add the three files from `ci/10xcode/`.

**10xSim PR #453** is our template: it boots an ephemeral `g6e.4xlarge` GPU via RunsOn
(`.github/runs-on.yml` profile `sim-gpu`), pulls the sim image from ECR via OIDC, and runs a
benchmark. We copy that shape for perception replay.

---

## Part 3 — The runner: RunsOn

A job that says
```yaml
runs-on: runs-on=${{ github.run_id }}/runner=replay-gpu
```
tells RunsOn: *"boot the machine described by the `replay-gpu` profile, run this job on it,
destroy it after."* The profile lives in `.github/runs-on.yml`:
```yaml
runners:
  replay-gpu:
    family: ["g6"]              # NVIDIA L4 (Ada sm_89 — matches the RTX 4080 engine)
    image: ubuntu24-gpu-x64     # base image with NVIDIA drivers preinstalled
    cpu: [16]                   # the colcon build of perception is CPU-bound
    spot: false                 # on-demand: a blocking PR gate shouldn't lose its machine
    volume: 250gb               # ~20 GB image + ~25 GB output bag + build tree
    extras: s3-cache+ecr-cache  # pip cache + Docker-layer cache (Part 10)
```
Config keys (RunsOn docs, runs-on.com/docs/runners/labels): `family` (instance type/family),
`cpu`/`ram`, `image` (a base or **custom** image name), `ami` (a specific AMI id), `volume`
(`size:type:throughput:iops`), `spot`, `extras` (`s3-cache`, `ecr-cache`, `efs`, …).

**One-time org prerequisite (platform team):** RunsOn is installed per-organization — a GitHub
App on `OriginAutonomy` + a CloudFormation stack in the AWS account that grants it permission to
create/destroy EC2. 10xSim#453 implies this exists; confirm it covers 10xCode. After that, any
in-org repo just adds a `runs-on.yml`.

---

## Part 4 — Registry: pull the image from GHCR

10xCode publishes every image to **GHCR** (`v2-docker-ci.yml`: `IMAGE_PREFIX:
ghcr.io/originautonomy`, verified at dev HEAD); the ECR copy is only a post-release mirror for
robots (*"robots pull ECR"*). So CI pulls from GHCR — the primary, always-current source. The
framework never `docker pull`s directly — `replay-module setup-env` runs `docker compose up -d
--pull missing`, and the compose file references the image as
`${REGISTRY}/v2-planner-docker-x86:latest` (`v2-manifest.yml:21-33`). So the registry is just the
`REGISTRY` env var + a docker login. The gate template:
1. `docker/login-action@v3` to `ghcr.io` with `${{ secrets.DOCKER_DEPLOYMENT_PAT }}` — the org's
   **existing** GHCR credential (the same one `v2-docker-ci.yml` uses to push); no new secret.
2. sets `REGISTRY=ghcr.io/originautonomy` (hardcoded in the template).

Because the gate runs **in-org** (Path A), this needs **no AWS role at all** for the image —
that's the payoff of running inside 10xCode. (AWS OIDC re-enters only if test bags move to S3,
Part 6.) An earlier draft of this guide recommended ECR; that was correct only while the gate
lived in a personal repo. In-org + GHCR-primary makes GHCR strictly simpler.

---

## Part 5 — The image is the toolchain, not the code

Important mental model: `v2-planner-docker-x86` is the **toolchain base** (ROS 2 + CUDA 12.6 +
TensorRT 10.3 + a baseline build). The **code under test is the PR's code**, checked out by the
workflow and **incrementally colcon-built inside the running container** at replay time
(README:137-149; e2e §3). The gate pins the framework's version-spec to the PR's commit SHA, so
the framework's internal `git checkout` is a no-op over the already-checked-out PR tree (e2e fix
F5). That's why a stale image mirror doesn't matter — the toolchain rarely changes; the code is
always fresh.

## Part 5a — Building the x86 engine (one-time) — REQUIRED

A TensorRT engine is locked to the **GPU arch + TRT version it was built on**. The only EoMT
engine in S3 (`10xai-team-models/segmentation/semantic14/flicker_drywall.trt`) was built for the
robot's **ARM/Orin** GPU — it will **not deserialize on the x86 L4** runner (this is the error
you saw). No x86 build exists in S3 (checked `10xai-team-models`, `cv-models-production`,
`image-segmentation-models-10x`, `ai-models-rishi-10x` on 2026-06-19). So build one **once** from
the arch-independent ONNX and reuse it — don't rebuild per run.

You already produced a working x86/sm_89 engine on the RTX 4080 dev box during the e2e (fix F6).
To reproduce on an **x86 + sm_89** machine (the L4 runner, or the 4080 dev box — both Ada sm_89),
inside the `v2-planner-docker-x86` container (TRT 10.3):
```bash
# 1. fetch the ONNX (arch-independent)
aws s3 cp --region eu-north-1 \
  s3://10xai-team-models/segmentation/eomt_448_flicker_ep9.onnx ./eomt_flicker.onnx
# 2. e2e fix F6: TRT 10.3 rejects the RoPE `If` nodes — constant-fold them out first
polygraphy surgeon sanitize eomt_flicker.onnx --fold-constants -o eomt_flicker_folded.onnx
# 3. build the FP16 engine for THIS GPU (dynamic batch 1–6, matching perception_sim.yaml)
trtexec --onnx=eomt_flicker_folded.onnx --fp16 --saveEngine=flicker_drywall_x86_sm89.trt \
  --minShapes=...:1x... --optShapes=...:6x... --maxShapes=...:6x...   # use the model's input name/dims
# 4. upload to S3 with an ARCH-EXPLICIT name (so this confusion can't recur), then set
#    PERCEPTION_ENGINE_S3 to it. (cv-models-production already names engines by GPU, e.g. *_l4_*.)
aws s3 cp flicker_drywall_x86_sm89.trt s3://<your-bucket>/perception/engines/
```
Ideally build **on an L4** (the runner GPU) so the engine matches exactly; a 4080-built sm_89
engine usually loads on the L4 too (same compute capability + TRT version), but validate it
(Part 8 / a 1-frame dry-run). If the model changes, rebuild + re-upload — a data change, not infra.

**Done (2026-06-22):** an x86 engine is uploaded at
`s3://10xai-team-models/segmentation/semantic14/flicker_drywall_x86.trt` (57 MB) — set
`PERCEPTION_ENGINE_S3` to it. It still must be **load-tested on the x86 GPU** (it was uploaded,
not yet proven to deserialize) — the Part 8a local dry-run is exactly that proof.

---

## Part 6 — AWS resources, IAM, S3 (closes Q-F / Q-K)

### OIDC role `REPLAY_ROLE_ARN` (S3 read only)
The gate downloads the engine/LUTs/bag from S3 at job start, so it needs one minimal role. (The
**image** stays on GHCR — this role does **not** touch ECR.)
- Trust: `token.actions.githubusercontent.com`, `sub` scoped to `repo:OriginAutonomy/10xCode:*`.
- Permissions: `s3:GetObject` + `s3:ListBucket` on **both** existing buckets (validated 2026-06-19):
  `10xai-team-models` (eu-north-1, engine) and `rosbags-10x` (us-east-1, LUTs + bag). Read only, no
  write, no ECR. (Account `390403890757`.) Cross-region read is fine; the engine is only 50 MB.

### S3 bucket (versioned) — Q-F + Q-K resolved
```
s3://origin-replay-bags/
  bags/<module>/<task_id>/rosbag2/          # Q-F: a bag = (module, task_id); task_id = a repo var
      metadata.yaml + *.mcap
  fixtures/<module>/<name>/                 # Q-K: a curated bundle =
      bag/ rosbag2/...                       #   input bag (6 healthy cams + /tf_static *_corrected)
      golden/ metrics.json + replay_output/  #   + pinned golden
      README.md                              #   + provenance (10xCode sha, engine, date)
  goldens/<module>/<engine_version>/golden.yaml   # golden is engine-version-specific
```

### Repo vars / secrets to set on **10xCode**
| Name | Kind | Value | Purpose |
|---|---|---|---|
| `DOCKER_DEPLOYMENT_PAT` | secret | the org's existing GHCR PAT (already used by `v2-docker-ci.yml`) | pull the planner image from GHCR — **likely already set org-wide** |
| `PERCEPTION_ENGINE_S3` | var | S3 URI of the **x86/sm_89** engine you build in Part 5a | the engine the gate stages (the S3 ARM one won't work) |
| `PERCEPTION_BAG_S3` | var | S3 URI of the curated input bag (e.g. `s3://rosbags-10x/<task>/...`) | the bag the gate replays (TBD — pick/curate it) |
| `PERCEPTION_BAG_PATH` | var | local download target, e.g. `/tmp/perception_ci` | where the bag is synced to on the runner |
| `PERCEPTION_TEST_BAG_TASK_ID` | var | the curated bag's task_id | identifies the bag (CI-04) |
| `FRAMEWORK_READ_TOKEN` | secret | read-only PAT on `saranshshankar/Replay-Framework` | check out the framework (cross-org until the repo moves in-org) |
| `REPLAY_ROLE_ARN` | var | OIDC role ARN (S3-read only) | download the engine/LUTs/bag from S3 |

---

## Part 7 — The files you deploy into 10xCode

All four are in this repo under `module_replay/ci/10xcode/` and are copy-paste ready:

| Source (here) | Destination (10xCode) |
|---|---|
| `module_replay/ci/10xcode/runs-on.yml` | `.github/runs-on.yml` |
| `module_replay/ci/10xcode/replay-perception-gate.yml` | `.github/workflows/replay-perception-gate.yml` |
| `module_replay/ci/10xcode/replay-perception-nightly.yml` | `.github/workflows/replay-perception-nightly.yml` |
| `module_replay/ci/10xcode/replay-perception-viz.yml` | `.github/workflows/replay-perception-viz.yml` |

The gate workflow's logic = the Part 0 picture: detect (via the `paths:` filter) → replay
(RunsOn GPU, replay only) → metrics (cheap, gates on `metrics.json["pass"]`) → gate.

---

## Part 7a — Generate visualizations on demand (Tier-3 viz)

The gate stays cheap and image-free. When a developer wants to *see* what a run did —
cross-camera overlap, semantic masks, depth, temporal consistency — they trigger
**`replay-perception-viz`** manually for that gate run. It is fully decoupled from the
blocking gate (TIER3-VIZ-DESIGN §7):

- **CPU-only, GitHub-hosted `ubuntu-latest`** — NO GPU, NO RunsOn, NO TensorRT engine, NO
  S3/OIDC. It only reads the gate run's already-recorded `output-bag-<run_id>` artifact.
- It `pip install`s the framework with the **`[viz]`** extra (the mp4 encoder), runs
  `replay-module viz`, and uploads the mp4s as `viz-<run_id>` (5-day retention).
- It is `workflow_dispatch` only — never a `pull_request` trigger — so it never runs in,
  slows, or adds dependencies to the gate.

**Developer loop:** gate goes red on a PR → open **Actions → replay-perception-viz → Run
workflow**, paste the gate run's id → download the `viz-<run_id>` artifact → inspect
`overlap_cam*.mp4` + `combined_cam*.mp4`. Locally the same videos come from
`replay-module viz --module perception --bag <output_bag> --output <dir>` (or `--run-viz`
on `all` / `metrics`); both need the `[viz]` extra (`pip install module_replay[viz]`).

---

## Part 8a — Local dry-run (test the gate's logic before any CI)

Do this **on the x86 GPU box** (the RTX 4080 dev box — both it and the L4 are Ada sm_89). It runs
the *exact command sequence* the CI gate runs, sourcing assets from S3 the same way — so it proves
the new x86 engine deserializes, the LUT remap is right, and the verdict/exit code are correct,
**without** waiting on any GitHub/RunsOn wiring.
```bash
# 1. Stage assets into ~/.ros (bind-mounted into the container) — exactly what the gate does
mkdir -p ~/.ros/perception/models ~/.ros/perception/calibration/camera_intrinsics
aws s3 cp s3://10xai-team-models/segmentation/semantic14/flicker_drywall_x86.trt \
  ~/.ros/perception/models/flicker_drywall.trt                      # NOTE the local name (sim loads this)
aws s3 sync s3://rosbags-10x/planner_v2_ros/.ros/perception/camera_intrinsics/ \
  ~/.ros/perception/calibration/camera_intrinsics/                  # remap → +calibration/ segment
# 2. Run the gate's command (replay + metrics + verdict) against a bag + a 10xCode checkout whose
#    perception_sim.yaml points at flicker_drywall.trt (the e2e branch already does)
cd <framework>/module_replay
replay-module all --module perception --version-yaml <ver.yaml> --local-bag <bag> \
  --output /tmp/dryrun --run-metrics
python -c "import json,sys; d=json.load(open('/tmp/dryrun/reports/metrics.json')); \
  print('verdict pass=' , d['pass']); sys.exit(0 if d['pass'] else 1)"
```
**This covers:** asset staging, the x86 engine load, replay, metrics, verdict + exit code.
**It does NOT cover:** RunsOn provisioning, GHCR pull on a fresh runner, GitHub triggers/secrets/
branch-protection, artifact hand-off between jobs — those only validate in a real Actions run
(Part 9 Step 6). If your "laptop" has **no GPU**, only the `metrics` half runs there; the replay
half needs the GPU box.

## Part 8 — Validation (SC1–SC4)

| Criterion | How to confirm |
|---|---|
| **SC1** faithful replay → bag + report + metrics + correct exit code | Run the gate via `workflow_dispatch`. Confirm the replay job emits `replay_output` (6 non-empty `semantic_raw_sim`), the metrics job writes `metrics.json` + `report.html`, exit code matches verdict. |
| **SC2** bad PR red, clean PR green | Open a PR degrading perception → gate red (exit 1). Open a clean PR → green. |
| **SC3** nightly smoke green | Let `replay-perception-nightly.yml` fire (or dispatch); fixed-bag replay exits green. |
| **SC4** two replays agree within tolerance | The nightly's second-replay step fails if any metric's `|Δ|` exceeds its `tolerance_band`. Widen bands in `perception.yaml` if a metric is legitimately noisy (a threshold call for Aniket, not code). |

---

## Part 9 — Deploy it yourself, step by step

This is the full runbook, assuming you've never set up CI. Do the steps in order. Each ends with
**"you'll know it worked when…"** so you can checkpoint.

### Step 1 — Confirm RunsOn is installed on the org *(platform/IT)*
RunsOn must be installed once on `OriginAutonomy`: the GitHub App + its CloudFormation stack in
AWS. 10xSim#453 uses it, so it likely exists. To confirm: open any 10xSim Actions run that used
`runs-on=…/runner=sim-gpu` and check it got a machine.
- **Worked when:** a 10xSim GPU run shows a runner was provisioned (not stuck "Queued").
- If it's *not* installed: follow runs-on.com's install (a CloudFormation one-click in the org's
  AWS account + installing the GitHub App). This is a one-time platform task.

### Step 2 — Credentials: GHCR for the image, a minimal S3-read role for the assets
- **Image:** pulled from GHCR with the org's existing `DOCKER_DEPLOYMENT_PAT` — **no AWS**.
- **Assets** (engine/LUTs/bag): downloaded from S3, so create **one S3-read-only** OIDC role
  (Part 6) and set it as `REPLAY_ROLE_ARN`. That role touches S3 only — not ECR.
- **Worked when:** `DOCKER_DEPLOYMENT_PAT` is visible to 10xCode Actions and the S3-read role exists.

### Step 3 — The assets are ALREADY in S3 (validated 2026-06-19) — just grant read
You don't need to upload the engine or LUTs — they already live in existing buckets, and the
templates point at them. The workflow downloads them into the host `~/.ros` at job start (they're
not in the Docker image, Part 5/10). Validated layout:

| Asset | Location | Region | Note |
|---|---|---|---|
| **Engine** (EoMT) | ⚠️ the S3 `flicker_drywall.trt` is an **ARM/Orin** build — unusable on x86. Build an **x86/sm_89** engine once (Part 5a) → set `PERCEPTION_ENGINE_S3` | — | source ONNX: `s3://10xai-team-models/segmentation/eomt_448_flicker_ep9.onnx` (eu-north-1) |
| **LUTs** (6 cams) | `s3://rosbags-10x/planner_v2_ros/.ros/perception/camera_intrinsics/<cam>/luts/` | **us-east-1** | arch-independent data; exact sim folder names + the 3 required files ✓ |
| **Input bag** | *not in either prefix* — set `PERCEPTION_BAG_S3` to the curated bag | (likely us-east-1) | TBD: pick/curate the gate bag |

Two gotchas the templates already handle (don't re-introduce them): the buckets are in **two
regions** (the engine `cp` uses `--region eu-north-1`), and the LUT bucket has **no
`calibration/` segment** so the sync **remaps** into `~/.ros/perception/calibration/...` (sim
requires that segment). `config/current` is populated by `apply_changes.py` on the checked-out
tree, not from S3. **No AMI, no upload.**
- **Worked when:** the role can `aws s3 ls` both buckets and the gate's staging step succeeds.

### Step 4 — Set the repo vars & secrets on 10xCode
GitHub → 10xCode repo → **Settings → Secrets and variables → Actions**. Add the five entries in
Part 6's table (Variables tab for vars, Secrets tab for `FRAMEWORK_READ_TOKEN`). For the PAT:
GitHub → your **Settings → Developer settings → Fine-grained tokens** → a token with
**read-only Contents** on `saranshshankar/Replay-Framework`.
- **Worked when:** all five names show up under the repo's Actions vars/secrets.

### Step 5 — Add the three workflow files to 10xCode
Copy the three files (Part 7 table) into 10xCode at the destination paths, on a branch, and open
a PR *to 10xCode* titled e.g. "ci: add perception replay gate." Point the `replay-gpu` profile's
`image:`/`ami:` at your Step-3 AMI if you baked assets in.
- **Worked when:** the PR shows the three new files and 10xCode's existing CI is still green.

### Step 6 — Dry-run via `workflow_dispatch`
Merge Step 5 (or before merging, since `workflow_dispatch` is enabled, trigger it): 10xCode →
**Actions → replay-perception-gate → Run workflow**. Watch the three jobs.
- **Worked when:** `replay` provisions a GPU machine (you'll see `nvidia-smi` output), produces
  `replay_output`, uploads it; `metrics` downloads it and prints a verdict; `gate` is green (or a
  trustworthy INVALID/FAIL with the report attached). Open the run's **Summary** for artifact links.
- **If `replay` is stuck "Queued":** RunsOn isn't picking up the label → re-check Step 1 and that
  `runs-on.yml` is at `.github/runs-on.yml` on the branch being run.
- **If `replay` fails at preflight (exit 3):** an asset is missing — the log names the exact path.
  Fix Step 3. *(This is the framework protecting you — see Part 10.)*
- **If GHCR login/pull fails:** `DOCKER_DEPLOYMENT_PAT` isn't available to the repo or lacks
  `read:packages`, or the planner package hasn't granted 10xCode read access.

### Step 7 — Turn it into a real gate (SC2)
Open a normal perception PR. The gate runs automatically (the `paths:` filter caught it). Then
GitHub → 10xCode → **Settings → Branches → branch protection → Require status checks** → select
the **gate** check. Now a red gate blocks merge.
- **Worked when:** a clean perception PR goes green and is mergeable; a PR that breaks perception
  goes red and the merge button is blocked.

### Step 8 — (Fast-follow, optional) the nightly (SC3/SC4)
**Not required for the e2e bring-up** — the gate already proves SC1/SC2 on every perception PR,
and on an active repo it provides continuous smoke coverage. Deploy the nightly later. Its one
*unique* value is the **determinism check (SC4)** — replay twice, compare deltas vs
`tolerance_band` — which is the evidence behind your threshold bands; the scheduled smoke (SC3)
mostly overlaps the gate (it only adds value when no perception PR is open, catching toolchain/
asset drift). Recommendation: run the determinism check **once manually** now to characterise
variance, then keep the nightly on a **weekly** cron (or `workflow_dispatch`) rather than every
night to save GPU cost.
- **Worked when:** the smoke replay is green and the determinism step passes (or names the metric
  that drifted beyond its band).

---

## Part 10 — Ensuring the inputs persist (the research)

**The core problem:** RunsOn runners are **ephemeral** — every job gets a brand-new EC2 machine
that is destroyed at the end (runs-on.com: "fresh, ephemeral EC2 runners for every job"). So
**nothing on the machine survives between runs by default.**

**The key realization: match the tool to the asset's *size*, not its importance.** The inputs
differ by four orders of magnitude, and that — not how "critical" they are — decides the right
mechanism:

| Input | Size | Mechanism | Why |
|---|---|---|---|
| **Engine + LUTs** | **~54 MiB + a few MB** | **Download from S3 at job start** (`aws s3 sync` → `~/.ros`) | Tiny → downloading takes **seconds**. A custom AMI buys nothing here and adds maintenance. This is the recommended approach (Part 9 Step 3). |
| **Test bag** | multi-GB | **Download from S3**, or **`extras: efs`** (shared mount, stored once) | Big but not huge. S3 download is simplest; EFS avoids the per-run copy if it ever matters. |
| **Toolchain image** | **~20 GB** | **`extras: ecr-cache`** (RunsOn local Docker layer cache — name is historical, works for GHCR) | Huge → you *don't* re-pull it; the layer cache means only changed layers download. It changes rarely. |
| **pip / Actions cache** | small | **`extras: s3-cache`** (RunsOn magic cache) | Speeds `pip install` each run. |
| **Output bag (job → job)** | multi-GB | **GitHub Actions artifact** | The one input meant to be transient — replay uploads, metrics downloads. |

**Source of truth vs. cache:** **S3 (versioned) is the durable source of truth** for the engine,
LUTs, bag, and goldens (the Q-K fixtures bundle, Part 6). The ephemeral runner downloads copies
each run. A runner being destroyed loses nothing — everything durable lives in S3, outside the
machine.

### Why NOT a custom AMI (the cost you asked about)
A custom AMI *can* bake assets onto the boot disk so they need no download. But it is the wrong
tool for small assets, because it carries real **operational** cost (the dollar cost — a few
$/month of snapshot storage — is the minor part):
- **It's a release process you now own.** Every time the engine changes (new model), the LUTs
  change (recalibration), or you want NVIDIA-driver/security patches from a newer base, you
  re-run Packer, publish a new AMI, and bump the `image:`/`ami:` in `runs-on.yml`. That's a
  build-and-release cycle (~10–30 min each) for a file you could have just re-uploaded to S3.
- **It couples your model version to your machine image.** A model bump becomes an *infra* change
  (rebuild AMI) instead of a *data* change (`aws s3 cp`). Wrong layer.
- **It drifts.** A frozen snapshot silently falls behind the base image's patches until you rebuild.
- **The benefit is only "zero per-run download"** — which matters solely for assets large enough
  that downloading them is a real bottleneck. The engine is 54 MiB. Downloading it costs seconds.

**So: skip the AMI.** Use the base `ubuntu24-gpu-x64` image + download the engine/LUTs/bag from S3
in a workflow step (the templates already do this). An AMI only earns its keep if you later find
the *bag* download dominates wall-clock and EFS/cache don't help — a narrow case, decided by data,
not assumed up front.

### The correctness guarantee (why a provisioning slip can't corrupt a verdict)
However the assets get there, before any replay the framework runs a **preflight** that checks
every required path exists and exits **3 (setup error)** with the *missing path named* if not.
So a bad `s3 sync`, a wrong `REPLAY_ROLE_ARN`, or a mis-typed bag path fails **loudly as
"infra/setup"** — visibly distinct from a real regression (exit 1) and from an invalid replay
(exit 2), and it **never silently passes**. This is the answer to "why preflight assets": it is
not how assets are *provided* (that's the download step) — it is the *check* that turns a
confusing deep crash inside the container (or, worse, a silent zero-output run that looks like a
pass) into one clear line naming the missing file. Provisioning and the preflight check are
**orthogonal**: the preflight is valuable no matter whether assets come from S3, EFS, or an AMI.

---

## Part 11 — Checklist
- [ ] RunsOn confirmed installed on OriginAutonomy, covers 10xCode (Part 9 Step 1)
- [ ] `DOCKER_DEPLOYMENT_PAT` available to 10xCode (image); minimal S3-read OIDC role `REPLAY_ROLE_ARN` created (assets) (Step 2)
- [x] x86 engine uploaded → `s3://10xai-team-models/segmentation/semantic14/flicker_drywall_x86.trt` (2026-06-22). Set `PERCEPTION_ENGINE_S3` to it; **load-test it on the GPU (Part 8a dry-run)** before trusting it
- [ ] LUTs already in S3 (validated, arch-independent); role granted read on the engine bucket + `rosbags-10x`; curate the input bag → `PERCEPTION_BAG_S3` — no AMI, no upload for LUTs
- [ ] Confirm the engine deserializes on the L4 (1-frame dry-run); confirm 10xCode `perception_sim.yaml` points at `flicker_drywall.trt` (model-currency)
- [ ] 10xCode repo vars/secrets set: `DOCKER_DEPLOYMENT_PAT`, `REPLAY_ROLE_ARN`, `PERCEPTION_ENGINE_S3`, `PERCEPTION_BAG_S3`, `PERCEPTION_BAG_PATH`, `PERCEPTION_TEST_BAG_TASK_ID`, `FRAMEWORK_READ_TOKEN` (Step 4)
- [ ] Three files added to 10xCode `.github/` (Step 5)
- [ ] `workflow_dispatch` dry-run green/trustworthy (Step 6)
- [ ] Branch protection requires the **gate** check (Step 7)
- [ ] Nightly verified (Step 8)
- [ ] **Long-term:** move `Replay-Framework` into the OriginAutonomy org → drops `FRAMEWORK_READ_TOKEN` (built-in token works in-org)

---

*Grounding: 10xCode `dev@aae1d5827` (`v2-docker-ci.yml`, `v2-manifest.yml`, planner compose);
OriginAutonomy/10xSim#453 (`.github/runs-on.yml` + `sim-benchmark.yml` — the RunsOn template);
Replay-Framework `main` (`module_replay/ci/10xcode/*`, `configs/modules/perception.yaml`,
README, the `e2e-2026-06-18` evidence); RunsOn docs (runs-on.com/docs/runners/labels,
/guides/building-custom-ami-with-packer); perception behavior per `KT/PERCEPTION-REPLAY-CONTRACT.md`.*
