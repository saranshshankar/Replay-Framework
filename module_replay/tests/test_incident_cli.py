"""Tests for the `incident` CLI subcommand (01.1, D-21/D-22).

Design requirements:
- The bag is provided LOCALLY via --incident-bag (a rosbag2 dir). The framework does
  NOT resolve S3 paths: in CI the gate stages the bag from the RDS s3_bag_uri (data_sync
  writes it after upload) and passes the local path (D-22). --incident-id alone raises a
  UsageError directing to --incident-bag.
- --incident-id (no key) = verify against ALL the module's detectors (CI default);
  --incident-key = verify ONE detector (local debug). module threaded as the click.Choice.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_incident_bag_dir(tmp_path: Path, name: str = "incident_bag") -> Path:
    """Create a minimal rosbag2 dir (metadata.yaml present) for resolve_local_bag."""
    bag_dir = tmp_path / name
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("version: 9\n")
    (bag_dir / "rosbag2_0.db3").write_bytes(b"fake")
    return bag_dir


# ---------------------------------------------------------------------------
# CLI tests: `replay-module incident` subcommand
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_incident_bag(tmp_path: Path) -> Path:
    """Minimal rosbag2 dir (metadata.yaml present) for the local --incident-bag path."""
    return _make_incident_bag_dir(tmp_path)


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    """Minimal configs/modules/perception.yaml for the CLI tests."""
    d = tmp_path / "configs" / "modules"
    d.mkdir(parents=True)
    (d / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: [/cam0/image_raw]
output_topics: [/cam0/image_raw_sim]
launch:
  command: "ros2 launch realtime_perception perception.launch.py use_replay:=true"
"""
    )
    return tmp_path / "configs"


def test_incident_help_shows_both_flags():
    """The `incident` subcommand help text lists --incident-bag and --incident-id."""
    from replay.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["incident", "--help"])
    assert result.exit_code == 0
    assert "--incident-bag" in result.output
    assert "--incident-id" in result.output


def test_incident_cmd_local_bag_path(
    tmp_path: Path, synthetic_incident_bag: Path, configs_dir: Path
):
    """--incident-bag resolves via resolve_local_bag -> DataRef(source='local');
    module 'perception' is threaded to load_module_config AND to run_replay."""
    from replay.cli import main

    output_dir = tmp_path / "output"
    runner = CliRunner()

    captured = {}

    def fake_run_replay(module, data, output_dir):
        captured["module_name"] = module.name
        captured["data_local_path"] = data.local_path
        captured["data_source"] = data.source
        result = MagicMock()
        result.exit_code = 0
        result.output_bag_path = output_dir / "bag"
        result.logs_dir = None
        return result

    def fake_metrics_pipeline(module_spec, bag_path, output_dir, **kwargs):
        return 0

    with patch("replay.cli.run_replay", side_effect=fake_run_replay), \
         patch("replay.cli._run_metrics_pipeline", side_effect=fake_metrics_pipeline):
        result = runner.invoke(
            main,
            [
                "incident",
                "--module", "perception",
                "--incident-bag", str(synthetic_incident_bag),
                "--output", str(output_dir),
                "--configs-dir", str(configs_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["module_name"] == "perception"
    assert captured["data_source"] == "local"
    assert Path(captured["data_local_path"]) == synthetic_incident_bag


def test_incident_cmd_no_flags_raises_usage_error(tmp_path: Path, configs_dir: Path):
    """Providing neither --incident-bag nor --incident-id raises a UsageError."""
    from replay.cli import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "incident",
            "--module", "perception",
            "--output", str(tmp_path / "output"),
            "--configs-dir", str(configs_dir),
        ],
    )
    assert result.exit_code != 0
    assert "incident-bag" in result.output.lower() or "incident-id" in result.output.lower()


def test_incident_cmd_incident_id_without_bag_raises_usage_error(
    tmp_path: Path, configs_dir: Path
):
    """--incident-id WITHOUT --incident-bag raises a UsageError: the gate stages the bag
    from the RDS s3_bag_uri; the framework does not resolve S3 paths (D-22)."""
    from replay.cli import main

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "incident",
            "--module", "perception",
            "--incident-id", "INC-001",
            "--output", str(tmp_path / "output"),
            "--configs-dir", str(configs_dir),
        ],
    )
    assert result.exit_code != 0
    assert "incident-bag" in result.output.lower()


def test_incident_cmd_replay_failure_exits_3(
    tmp_path: Path, synthetic_incident_bag: Path, configs_dir: Path
):
    """When run_replay returns exit_code != 0, the incident subcommand exits 3 (B9
    setup-error contract, mirroring all_cmd:410-424)."""
    from replay.cli import main

    output_dir = tmp_path / "output"
    runner = CliRunner()

    def fake_run_replay(module, data, output_dir):
        result = MagicMock()
        result.exit_code = 1
        result.output_bag_path = output_dir / "bag"
        result.logs_dir = None
        return result

    with patch("replay.cli.run_replay", side_effect=fake_run_replay):
        result = runner.invoke(
            main,
            [
                "incident",
                "--module", "perception",
                "--incident-bag", str(synthetic_incident_bag),
                "--output", str(output_dir),
                "--configs-dir", str(configs_dir),
            ],
        )

    assert result.exit_code == 3


# ---------------------------------------------------------------------------
# New tests for Task 3 (01.1-04): incident_spec threading + other_conditions
# auto-assembly (FR-6(b))
# ---------------------------------------------------------------------------

@pytest.fixture
def configs_dir_with_detectors(tmp_path: Path) -> Path:
    """configs_dir with a perception.yaml that has two incident_detectors entries."""
    d = tmp_path / "configs" / "modules"
    d.mkdir(parents=True)
    (d / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: [/cam0/image_raw]
output_topics: [/cam0/image_raw_sim]
launch:
  command: "ros2 launch realtime_perception perception.launch.py use_replay:=true"
incident_detectors:
  all_black_frame:
    verifier_type: metric_condition
    title: All-black frame collapse
    provisional: true
    condition:
      metric: segmentation_coverage
      field: segmentation_coverage
      op: lt
      threshold: 0.05
  latency_spike:
    verifier_type: metric_condition
    title: Latency spike
    provisional: true
    condition:
      metric: latency_p95_ms
      field: latency_p95_ms
      op: gt
      threshold: 200.0
"""
    )
    return tmp_path / "configs"


def test_incident_key_targeted_threads_single_check(
    tmp_path: Path, configs_dir_with_detectors: Path
):
    """LOCAL DEBUG (D-21): --incident-key threads a 'targeted' spec whose checks are
    exactly that ONE detector; incident_id falls back to the key when --incident-id absent."""
    incident_bag = _make_incident_bag_dir(tmp_path)
    output_dir = tmp_path / "output"
    captured = {}

    def fake_run_replay(module, data, output_dir):
        result = MagicMock()
        result.exit_code = 0
        result.output_bag_path = output_dir / "bag"
        result.logs_dir = None
        return result

    def fake_pipeline(module_spec, bag_path, output_dir, **kwargs):
        captured["incident_spec"] = kwargs.get("incident_spec")
        return 0

    from replay.cli import main
    runner = CliRunner()
    with patch("replay.cli.run_replay", side_effect=fake_run_replay), \
         patch("replay.cli._run_metrics_pipeline", side_effect=fake_pipeline):
        result = runner.invoke(
            main,
            [
                "incident",
                "--module", "perception",
                "--incident-bag", str(incident_bag),
                "--incident-key", "all_black_frame",
                "--output", str(output_dir),
                "--configs-dir", str(configs_dir_with_detectors),
            ],
        )

    assert result.exit_code == 0, result.output
    spec = captured["incident_spec"]
    assert spec is not None
    assert spec["mode"] == "targeted"
    assert set(spec["checks"].keys()) == {"all_black_frame"}
    assert spec["checks"]["all_black_frame"]["condition"]["metric"] == "segmentation_coverage"
    assert spec["incident_id"] == "all_black_frame"


def test_incident_id_no_key_verifies_all_detectors(
    tmp_path: Path, configs_dir_with_detectors: Path
):
    """CI DEFAULT (D-21): --incident-id with no --incident-key threads an 'all' spec whose
    checks are EVERY registered detector; the incident_id is stamped onto the spec."""
    incident_bag = _make_incident_bag_dir(tmp_path)
    output_dir = tmp_path / "output"
    captured = {}

    def fake_run_replay(module, data, output_dir):
        result = MagicMock()
        result.exit_code = 0
        result.output_bag_path = output_dir / "bag"
        result.logs_dir = None
        return result

    def fake_pipeline(module_spec, bag_path, output_dir, **kwargs):
        captured["incident_spec"] = kwargs.get("incident_spec")
        return 0

    from replay.cli import main
    runner = CliRunner()
    with patch("replay.cli.run_replay", side_effect=fake_run_replay), \
         patch("replay.cli._run_metrics_pipeline", side_effect=fake_pipeline):
        result = runner.invoke(
            main,
            [
                "incident",
                "--module", "perception",
                "--incident-bag", str(incident_bag),
                "--incident-id", "INC-2026-001",
                "--output", str(output_dir),
                "--configs-dir", str(configs_dir_with_detectors),
            ],
        )

    assert result.exit_code == 0, result.output
    spec = captured["incident_spec"]
    assert spec is not None
    assert spec["mode"] == "all"
    assert set(spec["checks"].keys()) == {"all_black_frame", "latency_spike"}
    assert spec["incident_id"] == "INC-2026-001"


def test_incident_key_not_found_runs_without_incident_spec(
    tmp_path: Path, configs_dir_with_detectors: Path
):
    """An unknown --incident-key runs with incident_spec=None (plain golden verdict)."""
    incident_bag = _make_incident_bag_dir(tmp_path)
    output_dir = tmp_path / "output"
    captured = {"incident_spec": "sentinel"}  # will be overwritten

    def fake_run_replay(module, data, output_dir):
        result = MagicMock()
        result.exit_code = 0
        result.output_bag_path = output_dir / "bag"
        result.logs_dir = None
        return result

    def fake_pipeline(module_spec, bag_path, output_dir, **kwargs):
        captured["incident_spec"] = kwargs.get("incident_spec")
        return 0

    from replay.cli import main
    runner = CliRunner()
    with patch("replay.cli.run_replay", side_effect=fake_run_replay), \
         patch("replay.cli._run_metrics_pipeline", side_effect=fake_pipeline):
        result = runner.invoke(
            main,
            [
                "incident",
                "--module", "perception",
                "--incident-bag", str(incident_bag),
                "--incident-key", "nonexistent_key",
                "--output", str(output_dir),
                "--configs-dir", str(configs_dir_with_detectors),
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["incident_spec"] is None


def test_no_incident_key_runs_without_incident_spec(
    tmp_path: Path, configs_dir_with_detectors: Path
):
    """No --incident-key: incident_spec=None (backward-compatible golden-style run)."""
    incident_bag = _make_incident_bag_dir(tmp_path)
    output_dir = tmp_path / "output"
    captured = {}

    def fake_run_replay(module, data, output_dir):
        result = MagicMock()
        result.exit_code = 0
        result.output_bag_path = output_dir / "bag"
        result.logs_dir = None
        return result

    def fake_pipeline(module_spec, bag_path, output_dir, **kwargs):
        captured["incident_spec"] = kwargs.get("incident_spec")
        return 0

    from replay.cli import main
    runner = CliRunner()
    with patch("replay.cli.run_replay", side_effect=fake_run_replay), \
         patch("replay.cli._run_metrics_pipeline", side_effect=fake_pipeline):
        result = runner.invoke(
            main,
            [
                "incident",
                "--module", "perception",
                "--incident-bag", str(incident_bag),
                "--output", str(output_dir),
                "--configs-dir", str(configs_dir_with_detectors),
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured.get("incident_spec") is None


def test_incident_help_shows_incident_key_flag():
    """The `incident` subcommand help text includes --incident-key."""
    from replay.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["incident", "--help"])
    assert result.exit_code == 0
    assert "--incident-key" in result.output


def test_other_detector_tripping_is_not_fixed_end_to_end(tmp_path: Path):
    """End-to-end (D-21): the gate runs ALL detectors. The incident's 'own' signature
    (all_black_frame) does NOT trip, but a DIFFERENT known failure (latency_spike) DOES —
    so the verdict is 'not_fixed' (a known catastrophic failure is still present), with that
    detector named in `tripped`. Drives generate_report with a hand-built check set."""
    import json
    from replay.metrics.base import MetricResult
    from replay.module_config import ThresholdSpec
    from replay.metrics.report.generator import generate_report

    # incident_spec as the CLI assembles it in CI mode: ALL detectors in `checks`.
    incident_spec = {
        "incident_id": "INC-E2E",
        "mode": "all",
        "verifier_type": "metric_condition",
        "checks": {
            "all_black_frame": {
                "verifier_type": "metric_condition",
                "condition": {"metric": "segmentation_coverage", "field": "segmentation_coverage",
                              "op": "lt", "threshold": 0.05},
            },
            "latency_spike": {
                "verifier_type": "metric_condition",
                "condition": {"metric": "latency_p95_ms", "field": "latency_p95_ms",
                              "op": "gt", "threshold": 200.0},
            },
        },
    }

    # coverage=0.3 (all_black_frame does NOT trip); latency=350 (latency_spike DOES trip)
    seg_result = MetricResult(
        name="segmentation_coverage", module="perception",
        value={"segmentation_coverage": 0.3, "temporal_consistency_mean": 0.9,
               "mean_class_coverage": 0.5},
        passed=True, is_regression=False,
    )
    lat_result = MetricResult(
        name="latency_p95_ms", module="perception",
        value={"latency_p95_ms": 350.0},
        passed=True, is_regression=False,
    )

    th = {
        "replay_max_gap_ms": ThresholdSpec(max=200.0, tier="validity"),
        "segmentation_coverage": ThresholdSpec(min=0.05, tier="quality"),
    }
    output_dir = tmp_path / "report"

    rc = generate_report(
        "perception", "t-e2e", [seg_result, lat_result], output_dir, th,
        faithfulness={"max_gap_ms": 50.0, "drop_rate": 0.001, "breach_count": 0},
        incident_spec=incident_spec,
    )

    doc = json.loads((output_dir / "metrics.json").read_text())
    assert rc == 0   # golden gate unaffected (latency not in quality thresholds)
    assert doc["pass"] is True
    iv = doc["incident_verdict"]
    assert iv["verdict"] == "not_fixed"
    assert iv["tripped"] == ["latency_spike"]


def test_graceful_degrade_pops_incident_spec():
    """The try/except TypeError graceful-degrade block in _run_metrics_pipeline
    contains report_kwargs.pop("incident_spec", None) — confirmed by source inspection.
    This is the same pattern as run_artifacts/visualizations pops."""
    import inspect
    from replay.cli import _run_metrics_pipeline

    source = inspect.getsource(_run_metrics_pipeline)
    assert 'report_kwargs.pop("incident_spec"' in source, (
        "graceful-degrade except block must pop 'incident_spec' from report_kwargs "
        "(mirrors run_artifacts/visualizations pattern)"
    )
