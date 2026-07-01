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

> ⚠️ **GOTCHA:** the ARM/Orin engine in S3 (`s3://10xai-team-models/segmentation/semantic14/flicker_drywall.trt`)
> will **not** deserialize on the L4/x86 CI runner. The x86 build lives at
> `…/semantic14/flicker_drywall_x86.trt` (uploaded 2026-06-22) — that one is for **CI** (Stage B),
> and Stage A's local run is its first real x86 load-test proof.

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

> ⚠️ **GOTCHA — expected verdict may still be `FAIL` (exit 1), and that's OK for SC1.** SC1 proves
> the verdict is *trustworthy and itemized*, not that it's green. If `depth_raw_sim` is genuinely
> all-zero again (a real perception issue, deliberately **not** gated in the quality tier), the run
> can still be VALID. What must NOT happen: an INVALID (exit 2) driven by the old drop-rate inflation.
> A VALID verdict (green or an honest quality FAIL) = SC1 satisfied.

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

### A7 · SC-INCIDENT — the net-new incident loop (the point of this branch)

The incident subcommand replays a bag and writes an additive
`metrics.json["incident_verdict"]` = `fixed` | `not_fixed` | `inconclusive`. The 5 perception
detectors (`configs/modules/perception.yaml → incident_detectors`) are:

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

### B0 · Provisioning inventory (`⚙️ you own all of this`)

**GitHub secrets (on `OriginAutonomy/10xCode`):**

| Secret | Purpose | Used by |
|--------|---------|---------|
| `FRAMEWORK_READ_TOKEN` | read-only PAT to clone the framework repo (cross-org) | all workflows |
| `DOCKER_DEPLOYMENT_PAT` | pull the planner image from `ghcr.io/originautonomy/*` (org PAT, likely already exists) | gate, sweep, nightly |
| `INCIDENT_DB_URL` | Postgres conn string (**gate role**: SELECT + UPDATE of status/fixed fields) | gate (resolve + mark), sweep |

**GitHub repo variables:**

| Var | Value | Used by |
|-----|-------|---------|
| `REPLAY_ROLE_ARN` | OIDC role ARN, S3-read on engine + LUTs + golden + **incident bucket** | gate, sweep, nightly |
| `PERCEPTION_ENGINE_S3` | `s3://10xai-team-models/segmentation/semantic14/flicker_drywall_x86.trt` (**x86 build!**) | gate, sweep, nightly |
| `PERCEPTION_BAG_S3` | S3 URI of the curated golden input bag | gate (golden), nightly |
| `PERCEPTION_BAG_PATH` | runner-local mount for the bag, e.g. `/mnt/efs/bags/perception_ci` | gate (golden), nightly |

> ⚠️ **GOTCHA — regions are inconsistent in the workflows today:** gate + sweep set
> `aws-region: ap-south-1`; nightly sets `us-east-1`; the engine `aws s3 cp` auto-resolves region.
> Confirm your buckets are reachable from those regions (or normalize them) — see
> [Appendix C](#appendix-c--what-changed-vs-the-old-docs). The old docs said `eu-north-1` for the
> engine bucket — reconcile before first run.

**AWS / infra:**
- `⚙️` **RunsOn** installed on the org (GitHub App + CloudFormation stack); the **`replay-gpu`** runner
  profile defined by merging `runs-on.merge.yml` (g6/L4, sm_89, on-demand, 250 GB, `s3-cache+ecr-cache`).
- `⚙️` **OIDC role** `REPLAY_ROLE_ARN`: trust `token.actions.githubusercontent.com` scoped to
  `repo:OriginAutonomy/10xCode:*`; `s3:GetObject`+`s3:ListBucket` on the engine bucket, `rosbags-10x`
  (LUTs/golden), and the incident bag bucket. Read-only. No long-lived keys.
- `⚙️` **RDS Postgres** with the incidents table (next step).
- `⚙️` **x86 engine in S3** — confirm `flicker_drywall_x86.trt` is the sm_89/TRT-10.3 build (Stage A3
  is your proof it loads on x86; if the CI runner TRT minor differs, rebuild per
  `CI-ENABLEMENT-GUIDE.md` Part 5a).

### B1 · Create the RDS incidents table (once, by the DB owner)

```bash
psql "$INCIDENT_DB_URL_ADMIN" -f module_replay/ci/infra/incidents_table.sql
```

`✅ EXPECT:` table `module_replay_incidents` + two indexes (`module_name,status` and `error_code`),
idempotent (`IF NOT EXISTS`). Then grant least-privilege roles:

```sql
-- gate role (the connection string in INCIDENT_DB_URL):
GRANT SELECT, UPDATE (status, fixed, fixed_by_pr, fixed_by_sha, fixed_at, fixed_by_run)
  ON module_replay_incidents TO <gate_role>;
-- sync-worker role (record side, populates s3_bag_uri):
GRANT INSERT, UPDATE (s3_bag_uri) ON module_replay_incidents TO <worker_role>;
```

> The gate **degrades gracefully** if `INCIDENT_DB_URL` is unset (it verifies verdicts but skips the
> RDS mark), so you can bring CI up golden-path-first and add the DB later.

### B2 · Deploy the workflows + merge the RunsOn fragment

Copy into `OriginAutonomy/10xCode/.github/workflows/`:

- `replay-perception-gate.yml` — the PR gate (golden + incident branches; single required check)
- `replay-perception-sweep.yml` — weekly QA-cut sweep (reopen-on-reproduce)
- `replay-perception-nightly.yml` — daily smoke + two-run determinism
- `replay-perception-viz.yml` — on-demand Tier-3 viz (CPU, `workflow_dispatch` with a `run_id`)

**MERGE** (do not overwrite) the `replay-gpu:` block from `runs-on.merge.yml` into 10xCode's existing
`.github/runs-on.yml` under `runners:` — this preserves 10xCode's live `sim-gpu` profiles.

> ⚠️ The gate triggers on `pull_request` (never `pull_request_target`) filtered to
> `perception/v2/realtime_perception/**`; a privileged (GPU/secrets) job runs **only if**
> `head.repo.full_name == github.repository` (fork guard). Confirm the path filter matches where
> perception actually lives in 10xCode.

### B3 · Dry-run each workflow via `workflow_dispatch`

Before wiring branch protection, run each manually from the Actions tab:

- **gate** (`workflow_dispatch`): with no incident tags → exercises detect → **golden** replay →
  metrics → gate. `✅ EXPECT` green; artifacts uploaded (output bag, logs, report).
- **nightly** (`workflow_dispatch`): `✅ EXPECT` smoke green + determinism deltas within tolerance.
- **viz** (`workflow_dispatch` with the gate's `run_id`): `✅ EXPECT` mp4 artifacts.

### B4 · SC2 — live red/green on real PRs

1. **Clean PR** touching `perception/v2/realtime_perception/**` with a harmless change →
   `✅ EXPECT` **gate green** (golden path: detect → replay → metrics → gate).
2. **Bad PR** that regresses perception (e.g. break the segmenter output) →
   `✅ EXPECT` **gate red** (metrics exit 1), with `report.html` in the run artifacts + the
   `$GITHUB_STEP_SUMMARY` dev link.
3. **Untouched PR** (no perception files) → `✅ EXPECT` gate **neutral-skips** (green, replay skipped).

`📸 CAPTURE:` links to the green run, the red run, and the neutral-skip run.

### B5 · SC-INCIDENT (live) — the full incident loop

1. `⚙️` Seed one incident row in RDS with a real `s3_bag_uri` pointing at a bag that reproduces a
   known perception failure (`status='open'`, `module_name='perception'`).
2. `⚙️` Post the PR **incident checklist** comment (rendered by `ci/incident_checklist.py`, carrying
   the `<!-- module-replay-incident-checklist -->` marker) and **tick** the incident's box.
3. Open a PR that *fixes* that incident.
   `✅ EXPECT:` detect picks up the ticked `incident_id` → the **incident branch** stages the bag from
   S3, replays it with `replay-module incident --incident-id …`, the **`mark`** job reads
   `incident_verdict.verdict`, and — only if `fixed` — UPDATEs the row to `status='fixed'` and the
   gate passes. A `not_fixed` verdict → mark fails → gate red (incident still reproduces).
4. **Sweep** (`workflow_dispatch`): `✅ EXPECT` it replays all open incident bags (dedup by
   `error_code`) and **reopens** any that reproduce, failing the QA-cut.

### B6 · Branch protection

Make **`gate`** the single required status check on the protected branch(es). Adding future modules
never changes this (each module ships its own `gate` job that consolidates to one check).

### Stage B exit criteria

- [ ] B0 provisioning complete; B1 RDS table + grants applied
- [ ] B2 workflows deployed; RunsOn fragment merged (not overwritten)
- [ ] B3 all four workflows dry-run green via dispatch
- [ ] B4 SC2 — clean PR green, bad PR red, untouched PR neutral-skip
- [ ] B5 incident loop marks `fixed` live; sweep reopens a reproduced incident
- [ ] B6 `gate` is the required check

---

## Definition of Done — "perception done and dusted"

Perception is signed off when **all** hold and evidence is captured:

| # | Criterion | Where proven |
|---|-----------|--------------|
| SC1 | Faithful isolated replay → trustworthy itemized verdict (B2/B3/B4 healed) | A3 |
| SC-Q | Quality gate demonstrably bites | A4 |
| SC4 | Two replays agree within tolerance | A5 / B3 nightly |
| VIZ | Tier-3 debug videos render | A6 / B3 viz |
| SC-INC | Incident loop verdict correct both directions; live mark + sweep | A7 / B5 |
| SC2 | Live CI red on bad PR, green on clean, neutral-skip on untouched | B4 |
| SC3 | Nightly smoke green | B3 |
| DOCS | Stale docs reconciled ([Appendix C](#appendix-c--what-changed-vs-the-old-docs)) | — |
| UAT | Owner sign-off recorded in `01-HUMAN-UAT.md` (closes the SC2–SC4 items) | — |

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
7. **AWS regions:** docs say `eu-north-1` engine / `us-east-1` LUTs; live workflows use `ap-south-1`
   (gate/sweep) and `us-east-1` (nightly). Reconcile.

---

*Runbook created 2026-07-01 for phase 01.2. Ground-truth sources: `module_replay/replay/cli.py`,
`module_replay/ci/10xcode/*.yml`, `configs/modules/perception.yaml`, and the real e2e run at
`.planning/phases/01-framework-perception-ci-gate/e2e-2026-06-18/`.*
