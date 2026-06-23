from pathlib import Path

from click.testing import CliRunner

from replay.cli import main
from replay.module_config import ModuleSpec


def test_cli_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["validate", "fetch-data", "setup-env", "run", "all"]:
        assert cmd in result.output


def test_validate_prints_resolved_spec(tmp_path: Path, mocker):
    # Point CLI at a temporary modules dir containing perception.yaml
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: ["/a"]
output_topics: ["/b"]
launch:
  command: "ros2 launch x y.launch.py"
"""
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["validate", "--module", "perception", "--configs-dir", str(configs.parent)],
    )
    assert result.exit_code == 0, result.output
    assert "perception" in result.output
    assert "realtime_perception" in result.output
    assert "dev" in result.output  # default tenxcode branch


def test_fetch_data_local_bag(tmp_path: Path, mocker):
    bag = tmp_path / "rosbag2_x"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["fetch-data", "--local-bag", str(bag), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert str(bag) in result.output


def test_fetch_data_requires_task_id_or_local(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, ["fetch-data", "--output", str(tmp_path)])
    assert result.exit_code != 0
    assert "task-id" in result.output.lower() or "local-bag" in result.output.lower()


def test_setup_env_calls_setup_environment(tmp_path: Path, mocker):
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs.parent)],
    )
    assert result.exit_code == 0, result.output
    setup.assert_called_once()


def test_setup_env_auto_loads_default_version_yaml(tmp_path: Path, mocker):
    """If --version-yaml is not passed, `configs/versions/default.yaml` is used
    automatically when it exists."""
    configs = tmp_path / "configs"
    (configs / "modules").mkdir(parents=True)
    (configs / "modules" / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    (configs / "versions").mkdir()
    (configs / "versions" / "default.yaml").write_text(
        "tenxcode:\n  branch: feature/from-default-yaml\n"
    )

    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs)],
    )
    assert result.exit_code == 0, result.output
    version_spec = setup.call_args.args[0]
    assert version_spec.tenxcode_branch == "feature/from-default-yaml"


def test_setup_env_partial_checkout_loads_paths(tmp_path: Path, mocker):
    configs = tmp_path / "configs"
    (configs / "modules").mkdir(parents=True)
    (configs / "modules" / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    (configs / "modules" / "checkout_paths.yaml").write_text(
        "perception:\n  - perception/\n  - common_interfaces/\n"
    )

    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "setup-env", "--module", "perception",
            "--configs-dir", str(configs),
            "--partial-checkout",
        ],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert kwargs["checkout_paths"] == ["perception/", "common_interfaces/"]


def test_setup_env_no_partial_checkout_means_none(tmp_path: Path, mocker):
    configs = tmp_path / "configs"
    (configs / "modules").mkdir(parents=True)
    (configs / "modules" / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs)],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert kwargs["checkout_paths"] is None


def test_setup_env_passes_only_checkout_paths_kwarg(tmp_path: Path, mocker):
    """After dropping the skip flags, the only kwarg setup-env forwards is
    `checkout_paths` (default None)."""
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs.parent)],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert "skip_pull" not in kwargs
    assert "skip_build" not in kwargs
    assert kwargs["checkout_paths"] is None


def test_setup_env_default_build_jobs_is_two(tmp_path: Path, mocker):
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs.parent)],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert kwargs["build_jobs"] == 2


def test_setup_env_build_jobs_flag_forwarded(tmp_path: Path, mocker):
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "setup-env", "--module", "perception",
            "--configs-dir", str(configs.parent),
            "--build-jobs", "4",
        ],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert kwargs["build_jobs"] == 4


def test_run_calls_run_replay(tmp_path: Path, mocker):
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")

    run_mock = mocker.patch("replay.cli.run_replay")
    run_mock.return_value = mocker.MagicMock(output_bag_path=tmp_path / "out" / "replay_output", exit_code=0)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--module", "perception",
            "--bag", str(bag),
            "--output", str(tmp_path / "out"),
            "--configs-dir", str(configs.parent),
        ],
    )
    assert result.exit_code == 0, result.output
    run_mock.assert_called_once()


def test_all_cmd_propagates_nonzero_exit(tmp_path: Path, mocker):
    """FRWK-03 / B9: a non-zero replay exit maps to process exit 3.

    Codes 1 and 2 are reserved for the metrics verdict; any replay/setup
    failure maps to 3 with the container code echoed for diagnosis.
    """
    from replay.cli import main
    from replay.runner import RunResult

    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    mocker.patch("replay.cli.setup_environment")
    # The real perception.yaml lists robot-only preflight assets (.trt/LUTs/param
    # tree) that don't exist on dev/CI machines; without this mock the CLI would
    # exit 3 at the preflight gate before reaching run_replay, masking the intent
    # of this test (a non-zero REPLAY exit -> 3).
    mocker.patch("replay.cli.missing_preflight_assets", return_value=[])
    mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=bag, exit_code=137),
    )
    r = CliRunner().invoke(
        main,
        [
            "all",
            "--module", "perception",
            "--local-bag", str(bag),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert r.exit_code == 3, r.output   # B9: replay failures map to 3


def test_all_cmd_preflight_gate_fails_named_path(tmp_path: Path, mocker):
    """B5/C1 fail-fast: a missing preflight asset exits 3 with the path named in stderr."""
    from replay.runner import RunResult

    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    mocker.patch("replay.cli.setup_environment")
    missing_path = "/nonexistent/perception/models/segformer.trt"
    mocker.patch("replay.cli.missing_preflight_assets", return_value=[missing_path])
    run_mock = mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=bag, exit_code=0),
    )
    r = CliRunner().invoke(
        main,
        [
            "all",
            "--module", "perception",
            "--local-bag", str(bag),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert r.exit_code == 3, r.output
    assert missing_path in r.output
    run_mock.assert_not_called()   # gate fires BEFORE replay


def test_all_cmd_empty_preflight_proceeds_to_replay(tmp_path: Path, mocker):
    """An empty preflight_assets list lets replay proceed (mocked)."""
    from replay.runner import RunResult

    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    mocker.patch("replay.cli.setup_environment")
    mocker.patch("replay.cli.missing_preflight_assets", return_value=[])
    run_mock = mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=bag, exit_code=0),
    )
    r = CliRunner().invoke(
        main,
        [
            "all",
            "--module", "perception",
            "--local-bag", str(bag),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert r.exit_code == 0, r.output
    run_mock.assert_called_once()


def test_all_run_metrics_produces_report(tmp_path: Path, synthetic_bag, mocker):
    """--run-metrics runs the registered perception plugins + faithfulness,
    generates the report, and echoes the verdict (MTRC-03/MTRC-05).

    Uses the real default configs dir (perception.yaml thresholds from 01-03 +
    the plugin pack from 01-05). missing_preflight_assets is mocked to [] so the
    preflight gate doesn't exit 3 on the dev/CI machine's missing robot assets.
    """
    from replay.runner import RunResult

    mocker.patch("replay.cli.setup_environment")
    mocker.patch("replay.cli.missing_preflight_assets", return_value=[])
    mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=synthetic_bag, exit_code=0),
    )
    out = tmp_path / "out"
    r = CliRunner().invoke(
        main,
        [
            "all",
            "--module", "perception",
            "--local-bag", str(synthetic_bag),
            "--output", str(out),
            "--run-metrics",
        ],
    )
    assert "Verdict:" in r.output, r.output
    assert (out / "reports" / "metrics.json").exists()


def test_metrics_subcommand_offline_produces_report(tmp_path: Path, synthetic_bag):
    """Standalone `metrics` subcommand (B9/CI-01): offline, no preflight, no replay.

    Computes metrics on an EXISTING output bag and writes reports/metrics.json.
    """
    import json

    out = tmp_path / "out"
    r = CliRunner().invoke(
        main,
        [
            "metrics",
            "--module", "perception",
            "--bag", str(synthetic_bag),
            "--output", str(out),
        ],
    )
    assert "Verdict:" in r.output, r.output
    metrics_json = out / "reports" / "metrics.json"
    assert metrics_json.exists()
    doc = json.loads(metrics_json.read_text())
    assert doc["module"] == "perception"
    assert "pass" in doc and "verdict" in doc


def test_metrics_pipeline_passes_reader_so_plots_render(tmp_path: Path, synthetic_bag, mocker):
    """Tier-2 plots are DEFAULT on every --run-metrics / metrics run (the agreed scope:
    'we generate the report anyway'). _run_metrics_pipeline MUST forward a non-None reader
    to generate_report — 01-16 gates plot rendering on reader != None. Regression guard for
    the CLI-wiring gap where the reader was built (cli.py:102) but never passed, so a real
    run produced a report.html with NO plots even though the plots module existed."""
    spy = mocker.patch(
        "replay.metrics.report.generator.generate_report", return_value=0
    )
    CliRunner().invoke(
        main,
        ["metrics", "--module", "perception", "--bag", str(synthetic_bag),
         "--output", str(tmp_path / "out")],
    )
    assert spy.called, "generate_report was not called"
    assert spy.call_args.kwargs.get("reader") is not None, (
        "reader not forwarded to generate_report → Tier-2 plots never render on a real run"
    )


def test_build_metrics_cfg_threads_new_keys():
    """01-10 / UAT gaps 1+5: _build_metrics_cfg threads the per-topic expected_hz
    map, depth_topics, and diagnostics_topic from the ModuleSpec into the cfg dict.

    Downstream plans read these keys: 01-12 (faithfulness) reads expected_hz, 01-13
    DepthMetric reads depth_topics, 01-13 LatencyMetric reads diagnostics_topic.
    """
    from replay.cli import _build_metrics_cfg

    spec = ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=["/perception_node/camera_0/image_raw"],
        output_topics=["/perception_node/camera_0/depth_raw_sim", "/perception_node/diagnostics"],
        launch_command="x",
        expected_hz={"default": 10.0, "diagnostics": 0.2},
        depth_topics=["/perception_node/camera_0/depth_raw_sim"],
        diagnostics_topic="/perception_node/diagnostics",
        gap_tolerance={"default": 2.0, "image_raw_sim": 4.0},
        latency_stage="inference_seg_extract_segmentation",
    )
    cfg = _build_metrics_cfg(spec)
    assert cfg["input_topics"] == spec.input_topics
    assert cfg["output_topics"] == spec.output_topics
    assert cfg["expected_hz"] == {"default": 10.0, "diagnostics": 0.2}
    assert cfg["depth_topics"] == ["/perception_node/camera_0/depth_raw_sim"]
    assert cfg["diagnostics_topic"] == "/perception_node/diagnostics"
    # 01-19 gap-closure: the two new keys thread through alongside the existing ones.
    assert cfg["gap_tolerance"] == {"default": 2.0, "image_raw_sim": 4.0}
    assert cfg["latency_stage"] == "inference_seg_extract_segmentation"


def _semantic_bag(bag_dir: Path, frames):
    """Write a tiny rgba8 semantic_raw_sim bag (R channel = class id) for the
    baseline-wiring CLI tests. ``frames`` is a list of (H, W) class-id arrays."""
    import numpy as np
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    topic = "/perception_node/camera_0/semantic_raw_sim"
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        conn = writer.add_connection(topic, Image.__msgtype__, typestore=typestore)
        for i, classid in enumerate(frames):
            classid = np.asarray(classid, dtype=np.uint8)
            h, w = classid.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[..., 0] = classid
            ts = i * 100_000_000
            hdr = Header(
                stamp=Time(sec=int(ts // 1_000_000_000), nanosec=int(ts % 1_000_000_000)),
                frame_id="cam0",
            )
            msg = Image(header=hdr, height=h, width=w, encoding="rgba8",
                        is_bigendian=0, step=w * 4, data=rgba.reshape(-1))
            writer.write(conn, ts, typestore.serialize_cdr(msg, Image.__msgtype__))
    return bag_dir


def _baseline_configs(tmp_path: Path) -> Path:
    """A configs dir whose perception.yaml carries ONLY the semantic topic + the
    mask_iou_vs_golden min-0.98 threshold, so the baseline-wiring tests gate purely
    on mask_iou (no diagnostics/depth machinery needed)."""
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics:
  - /perception_node/camera_0/semantic_raw_sim
launch:
  command: "x"
thresholds:
  quality:
    mask_iou_vs_golden:
      min: 0.98
      provisional: true
"""
    )
    return configs.parent


def test_metrics_baseline_invokes_compare(tmp_path: Path):
    """01-15 / UAT gap 2: `metrics --baseline <bag>` resolves the baseline, builds a
    baseline BagReader, and invokes MaskIoUVsGoldenMetric.compare — metrics.json has
    a numeric mask_iou_vs_golden row (NOT a 'skipped' row)."""
    import json
    import numpy as np

    frame = np.zeros((16, 16), dtype=np.uint8)
    frame[:8, :] = 1
    cand = _semantic_bag(tmp_path / "cand", [frame, frame])
    base = _semantic_bag(tmp_path / "base", [frame, frame])
    configs = _baseline_configs(tmp_path)

    out = tmp_path / "out"
    r = CliRunner().invoke(
        main,
        ["metrics", "--module", "perception", "--bag", str(cand),
         "--baseline", str(base), "--output", str(out), "--configs-dir", str(configs)],
    )
    assert r.exit_code == 0, r.output
    doc = json.loads((out / "reports" / "metrics.json").read_text())
    rows = {m["name"]: m for m in doc["metrics"]}
    assert "mask_iou_vs_golden" in rows
    row = rows["mask_iou_vs_golden"]
    # A real comparison ran: numeric value, evaluated against the threshold (passed
    # is a bool), NOT a skipped/None row.
    assert isinstance(row["value"], (int, float)) and row["value"] is not None, row
    assert row["passed"] is True
    assert row["value"] == 1.0  # identical candidate==baseline


def test_metrics_without_baseline_is_visible_skip_not_silent_pass(tmp_path: Path):
    """01-15: WITHOUT --baseline, mask_iou_vs_golden is a VISIBLE skipped row
    (passed: None) — never a silent pass, never a false fail. The run can still PASS
    on its intrinsic gates (here there are none, so verdict is PASS)."""
    import json
    import numpy as np

    frame = np.zeros((16, 16), dtype=np.uint8)
    frame[:8, :] = 1
    cand = _semantic_bag(tmp_path / "cand", [frame, frame])
    configs = _baseline_configs(tmp_path)

    out = tmp_path / "out"
    r = CliRunner().invoke(
        main,
        ["metrics", "--module", "perception", "--bag", str(cand),
         "--output", str(out), "--configs-dir", str(configs)],
    )
    # No baseline -> regression metric must NOT silently fail the gate. Verdict PASS.
    assert r.exit_code == 0, r.output
    doc = json.loads((out / "reports" / "metrics.json").read_text())
    rows = {m["name"]: m for m in doc["metrics"]}
    assert "mask_iou_vs_golden" in rows
    row = rows["mask_iou_vs_golden"]
    assert row["passed"] is None          # visible skip, neither pass nor fail
    assert row["value"] is None
    # Must NOT have a numeric value that would have gated as a pass.
    assert doc["pass"] is True


def test_metrics_baseline_fails_closed_on_empty_comparison(tmp_path: Path):
    """01-15 / FAIL-CLOSED: --baseline with ZERO comparable frames must NOT green-light
    the regression metric. The mask_iou_vs_golden row must not be a passing scalar."""
    import json
    import numpy as np

    frame = np.zeros((16, 16), dtype=np.uint8)
    frame[:8, :] = 1
    cand = _semantic_bag(tmp_path / "cand", [frame])
    base = _semantic_bag(tmp_path / "base", [])  # empty baseline -> no comparable frames
    configs = _baseline_configs(tmp_path)

    out = tmp_path / "out"
    r = CliRunner().invoke(
        main,
        ["metrics", "--module", "perception", "--bag", str(cand),
         "--baseline", str(base), "--output", str(out), "--configs-dir", str(configs)],
    )
    doc = json.loads((out / "reports" / "metrics.json").read_text())
    rows = {m["name"]: m for m in doc["metrics"]}
    assert "mask_iou_vs_golden" in rows
    row = rows["mask_iou_vs_golden"]
    # NOT a false 1.0 / passing scalar. A None scalar (visible no-scalar row) is the
    # fail-closed signal: passed must not be True, and there is no green scalar.
    assert row["passed"] is not True, row
    assert row["value"] is None, row


# ── 01-17: cli surfaces the persisted logs path (success AND exit-3 failure) ──


def test_all_cmd_echoes_logs_on_success(tmp_path: Path, mocker):
    """01-17 / GAP c: on a clean run, `all` echoes the <output>/logs/ path so the
    developer knows where the run logs landed."""
    from replay.runner import RunResult

    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    logs_dir = tmp_path / "out" / "logs"
    mocker.patch("replay.cli.setup_environment")
    mocker.patch("replay.cli.missing_preflight_assets", return_value=[])
    mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=bag, exit_code=0, logs_dir=logs_dir),
    )
    r = CliRunner().invoke(
        main,
        ["all", "--module", "perception", "--local-bag", str(bag),
         "--output", str(tmp_path / "out")],
    )
    assert r.exit_code == 0, r.output
    assert "logs" in r.output.lower()
    assert str(logs_dir) in r.output


def test_all_cmd_echoes_logs_on_exit3_failure(tmp_path: Path, mocker):
    """01-17 HEADLINE ASK (B1): on the exit-3 replay-failure branch the logs path
    is printed — the failure logs are EXACTLY what the developer pulls to debug a
    red run. Mirrors test_all_cmd_propagates_nonzero_exit, plus a logs_dir."""
    from replay.runner import RunResult

    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    logs_dir = tmp_path / "out" / "logs"
    mocker.patch("replay.cli.setup_environment")
    mocker.patch("replay.cli.missing_preflight_assets", return_value=[])
    mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=bag, exit_code=137, logs_dir=logs_dir),
    )
    r = CliRunner().invoke(
        main,
        ["all", "--module", "perception", "--local-bag", str(bag),
         "--output", str(tmp_path / "out")],
    )
    assert r.exit_code == 3, r.output      # B9: replay failures still map to 3
    assert str(logs_dir) in r.output       # ...and the logs path is surfaced


def test_run_cmd_echoes_logs(tmp_path: Path, mocker):
    """01-17: the `run` subcommand echoes the logs dir alongside Output bag/Exit code."""
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    logs_dir = tmp_path / "out" / "logs"
    run_mock = mocker.patch("replay.cli.run_replay")
    run_mock.return_value = mocker.MagicMock(
        output_bag_path=tmp_path / "out" / "replay_output",
        exit_code=0,
        logs_dir=logs_dir,
    )
    r = CliRunner().invoke(
        main,
        ["run", "--module", "perception", "--bag", str(bag),
         "--output", str(tmp_path / "out"), "--configs-dir", str(configs.parent)],
    )
    assert r.exit_code == 0, r.output
    assert str(logs_dir) in r.output


def test_run_metrics_pipeline_passes_run_artifacts(tmp_path: Path, synthetic_bag, mocker):
    """01-17 wires the run into 01-16's report: --run-metrics threads a run_artifacts
    dict carrying the bag + logs paths into generate_report (Debug section)."""
    from replay.runner import RunResult

    mocker.patch("replay.cli.setup_environment")
    mocker.patch("replay.cli.missing_preflight_assets", return_value=[])
    logs_dir = tmp_path / "out" / "logs"
    mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=synthetic_bag, exit_code=0, logs_dir=logs_dir),
    )
    gen_mock = mocker.patch(
        "replay.metrics.report.generator.generate_report", return_value=0
    )
    out = tmp_path / "out"
    r = CliRunner().invoke(
        main,
        ["all", "--module", "perception", "--local-bag", str(synthetic_bag),
         "--output", str(out), "--run-metrics"],
    )
    assert r.exit_code == 0, r.output
    gen_mock.assert_called_once()
    run_artifacts = gen_mock.call_args.kwargs.get("run_artifacts")
    assert run_artifacts is not None, gen_mock.call_args
    assert "logs" in run_artifacts
    assert str(logs_dir) in run_artifacts["logs"]
    assert "bag" in run_artifacts


def test_all_run_viz_invokes_viz_pipeline(tmp_path: Path, mocker):
    """Tier-3 (TIER3-VIZ-DESIGN): --run-viz on `all` (without --run-metrics) now
    RENDERS the debug videos via the viz pipeline — no longer the old scope-honest
    'deferred' no-op echo (01-10 / UAT gap 7 is closed)."""
    from replay.runner import RunResult

    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    mocker.patch("replay.cli.setup_environment")
    mocker.patch("replay.cli.missing_preflight_assets", return_value=[])
    mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=bag, exit_code=0),
    )
    # The viz render itself is unit-tested in test_viz.py; here we assert the `all`
    # command WIRES --run-viz to the viz pipeline (no real bag decode needed).
    spy = mocker.patch("replay.cli._run_viz_pipeline", return_value=[bag / "viz" / "x.mp4"])
    r = CliRunner().invoke(
        main,
        [
            "all",
            "--module", "perception",
            "--local-bag", str(bag),
            "--output", str(tmp_path / "out"),
            "--run-viz",
        ],
    )
    assert r.exit_code == 0, r.output
    spy.assert_called_once()
    assert "deferred" not in r.output
