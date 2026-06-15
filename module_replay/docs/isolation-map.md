# Module Isolation Map

**Isolation Cost** grades how hard it is to replay one robot software module's recorded
inputs through a chosen code version, in isolation, with no physical robot. Cost is driven
by what each module needs *beyond* the bag topics: mocks/stubs, a trigger to fire it, warm-up
state to seed, runtime assets to provision, and any TF surgery. A `MED` is mostly provisioning;
a `HIGH` adds live lifecycle orchestration, stubs, and warm-up gating.

These grades are **code-verified against 10xCode `dev@001223f9b` (June 2026)** — distilled from
the completed module-isolation spike (7 parallel code-grounded deep-dives + claim-by-claim
audits), not re-derived here. This page is the architecture-level summary; **step-level build
detail (with `path:line` citations) lives in the per-module playbooks** referenced in Evidence.

## Isolation Cost — per-module grading

Values copied from `SYSTEM-DESIGN-HLD-LLD.md` § C7 (sim-enablement matrix) and the C1–C6
contracts. Effort/warm-up column from `MODULE-ISOLATION-SPIKE.md` § 5.

| Module | Cost | Method (seam) | Trigger + source | Mocks / Stubs | State seeding | Assets | Effort / warm-up |
|---|---|---|---|---|---|---|---|
| **Perception** | **MED** (provisioning only) | `use_replay:=true` → `perception_sim.yaml` `mode:"sim"` → `RosInterfaceBridge` | none — sensor-driven, auto-activate | none | none (stateless) | `.trt` engine, fisheye LUT ×18 (3×6 cams), `perception_sim.yaml`, `config/current/` URDF | LOW — replay all topics; latched `/tf_static` preamble (500 ms configure window) |
| **nvblox planes** (sensor-fusion) | **LOW** (offline oracle) / **HIGH** (runtime routes) | offline `map_pipeline_standalone` first (file-in/file-out RANSAC, ROS-free); else `local_mapper` live flow / `global_mapper` corrected-zone flow | `run_plane_pipeline` → `extract_planes` (ordered — registry-empty guard makes order mandatory); prod timing ≥15 s after frames | none (it is the producer) | local = recent-window map (decaying, trigger-time matters); global = provisioned snapshot | camera-intrinsic LUTs, `semantic_mask_config.json`; corrected `<zone>_3/<zone>.nvblx` (**global flow only**; local builds live) | HIGH (runtime: provision/build map) / LOW–MED (offline route) — TF frame-root seed `odom→lsm_cache` (local) or `map→map_3d` (global) |
| **Localization** | **MED–HIGH** | `localisation_v2.launch.py` + `enable_sim_odom:=false`; lifecycle drive + seeding | lifecycle drive (manual configure→activate ×3 of map_server/amcl/localisation_score — **no lifecycle manager covers them**) + `load_map`; or Route B `SET_LOCALIZATION_MODE` goal | `health_monitor_interrupt` (hygiene) | `/initialpose` := first bag `/amcl_pose` (B18); EKF relative self-warm (or EKF-bypass) | zone `_4` 2D map (`<zone>_4/<zone>.yaml`+`.pgm`), `config/current/`; default `map.yaml` must exist pre-configure | HIGH — seed `/initialpose`; `/imu/filtered` has **NO v2 producer** (IMU-less replay = prod parity, do not reconstruct); `enable_odom` is a **dead variable** |
| **Navigation** | **HIGH** | manager tree (`navigation_manager_v2`), TF Option A (replay all) / Option B (strip + live loc) | `NAV_GO_TO` reconstructed from recorded `/bpo/current_goal` (B16 builder), after costmap warm-up gate | amcl lifecycle stub (Option A); stub the 9 lifecycle nodes the manager would activate (serve `get_state`+`change_state`, answer <4 s) | warm-up window 10–20 s + readiness gate — **plan-fatal if skipped** (SmacPlannerLattice `allow_unknown:false`) | zone map, lattice JSON, `constraint_navigation.xml`, plugin registry, `config/current/` | HIGH — costmap fill / warm-up gate; one-producer-per-frame TF (Option A/B exclusivity) |
| **Manipulation** | **MED** | plain launch + `use_sim_time` **injection** (launch wires none); controller-boundary cut | `GOAL_WORKFLOW` **fabricated** (`goal=1, tool_enum, level` from task metadata; B16 builder) | `ExtractPlanes` response (synthetic block / recorded-LSM fixture); `OriginRobotController` action mock = **sink + capture point** (expect ONE stitched DIFF_IK goal) | ≥1 `/joint_states` + 1 collision cloud before goal | URDF + `joint_limits` (via `manipulation_manager_v2_config.yaml`; `params.yaml` deleted), `config/current/` | LOW–MED — `robot_state_publisher` helper on bag `/joint_states` (strip bag arm TF); collision cloud needs param override to `colored_pointcloud` |
| **Control** | **LOW–MED** (logic regression) | `test_suite_cs mode:=mock` — mock-HW `GenericSystem` + GoalDumper CSV goal | GoalDumper CSV → `goal_dump_to_trajectory.py` → `trajectory_sender.py` re-send (t=0 normalized; B16); `manager_cs` self-configures + activates | mock-HW `mock_components/GenericSystem` (`calculate_dynamics:true`) | goal-boundary start only (integrator anchors/FSM carry pre-goal state) | URDF (`config/current`), controller YAML (`origin_controllers.yaml` patched for mock — 500 Hz) | HIGH (goal-boundary only) — **SafetyMonitor GPIO seeding `safety_mode=1, robot_mode=7, program_running=1` is load-bearing**; **bag-only replay is architecturally impossible** |

## Map provisioning (resolved)

The old Phase-2 `.nvblx` provisioning blocker is **RESOLVED BY DESIGN** (see § C2/C3):

- **Production plane flow (`local_mapper`):** rebuilds its decaying map live from bag frames —
  **no pre-seed** (trigger timing matters; it is a recent-window map).
- **Zone / global flow (`global_mapper`):** loads a provisioned `<zone>_3/<zone>.nvblx` via the
  plain `load_map` service (**no backend** required).
- **Localization:** needs the 2D `<zone>_4/<zone>.yaml` + `.pgm` map as a **per-bag
  fixture-bundle asset** (loaded via map_server; never replay bag `/map` while map_server runs).

## Record-side asks

In-house, separately scoped; each converts a replay risk into a provisioning step (§ C8).

1. **Archive control GoalDumper CSVs with the bag** — written per accepted goal with no
   retention/rotation; they are control's canonical replay input. *(owner: Saransh / controls)*
2. **Record `navigation_manager_v2_action` feedback/status** (2 catalog lines) — nav action
   telemetry is entirely absent today; with `/bpo/current_goal` this completes nav-goal
   reconstruction. *(owner: platform / Navigation)*
3. **Shadow-record `ExtractPlanes` responses** — highest-value shadow; removes the
   synthetic-LSM gap in manipulation replay. *(owner: platform / robotics)*
4. **Shadow/record task episode metadata** — `START_TASK`/`SET_AUTO_MODE` params
   (task_id/tool/level/zone) as a latched topic, so goal fabrication needs no out-of-band
   lookup. *(owner: platform)*
5. **Archive the BT's `SaveNvblx` snapshot** (`local_lsm.nvblx`, already produced per task)
   alongside the bag — the local-flow map seed. *(owner: platform / Navigation)*
6. **Recorder start-ordering guard** — topics absent at START are silently skipped; gate
   recording start on graph readiness, or log skipped topics loudly into bag metadata.
   *(owner: Saransh — recorder)*
7. **Wire the V2 recording trigger** — nothing in the V2 stack calls `/rosbag_record` today
   (V1-only client; V2 `record_bag` param is dead); document/own the external trigger, or
   restore an in-BT RecordBag node. *(owner: Saransh — recorder)*
8. **`PreserveDebugBag`** — the srv exists only on `saransh-10x/feat/debug-recorder-msgs`;
   land it with the debug/buffer recorder to make incident windows replayable.
   *(owner: Saransh — debug recorder)*

## Evidence

Per-module build manual + claim-by-claim verification report (both carry the `path:line`
citations against `dev@001223f9b` — not duplicated here).

- **Perception** — `KT/playbooks/01-perception.md` + `.planning/research/verification/01-perception.md`
- **nvblox / sensor-fusion** — `KT/playbooks/02-sensor-fusion-nvblox.md` + `.planning/research/verification/02-nvblox.md`
- **Localization** — `KT/playbooks/03-localization.md` + `.planning/research/verification/03-localization.md`
- **Navigation** — `KT/playbooks/04-navigation.md` + `.planning/research/verification/04-navigation.md`
- **Manipulation** — `KT/playbooks/05-manipulation.md` + `.planning/research/verification/05-manipulation.md`
- **Control** — `KT/playbooks/06-control.md` + `.planning/research/verification/06-control.md`
- **Cross-cutting** (goal injection, TF provenance, state seeding, mocks/lifecycle, record-side) —
  `KT/SYSTEM-DESIGN-HLD-LLD.md` Part C (C1–C8) + `.planning/research/verification/07-cross-cutting.md`

Grades and contracts distilled from `KT/SYSTEM-DESIGN-HLD-LLD.md` § C7 (matrix) / § C1–C6 /
§ C8, and `KT/MODULE-ISOLATION-SPIKE.md` § 5 + § 0-bis (post-verification addendum).
