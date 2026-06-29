"""Tests for resolve_incident_bag (data_manager) and the `incident` CLI subcommand.

Design requirements (01.1-01, LLD B4 / HLD A5):
- resolve_incident_bag builds the canonical `incidents/<module>/<incident_id>/` key
  (NOT data/{robot_id}/) and accepts an injectable s3_client for testability — no
  live S3, no boto3 network.
- The `incident` subcommand resolves both a local fixture bag (--incident-bag) and
  an S3 incident-id (--incident-id, canonical layout) to a DataRef, threads module
  as the existing click.Choice string, and reuses run_replay + the metrics pipeline.
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


def _make_mock_s3_client(
    bucket: str,
    incident_prefix: str,
    dest_dir: Path,
    files: dict[str, bytes],
) -> MagicMock:
    """Build a fake s3_client whose paginator yields the given key→content pairs and
    whose download_file writes the content to dest_dir.

    `files` maps S3 key (relative to the bucket root) to file bytes content.
    The mock implements:
      - client.get_paginator("list_objects_v2") -> paginator
      - paginator.paginate(Bucket=..., Prefix=...) -> [{"Contents": [...]}]
      - client.download_file(bucket, key, local_path) -> writes content to local_path
    """
    client = MagicMock()

    # Build the Contents list for the paginator
    contents = [{"Key": key} for key in files]
    page = {"Contents": contents}
    paginator = MagicMock()
    paginator.paginate.return_value = [page]
    client.get_paginator.return_value = paginator

    def _download_file(bkt, key, local_path_str):
        content = files[key]
        p = Path(local_path_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    client.download_file.side_effect = _download_file
    return client


# ---------------------------------------------------------------------------
# Unit tests: resolve_incident_bag (data_manager — no CLI, no live S3)
# ---------------------------------------------------------------------------

def test_resolve_incident_bag_builds_canonical_layout(tmp_path: Path):
    """HLD A5: resolve_incident_bag uses `incidents/<module>/<incident_id>/` as the
    S3 key prefix — NOT resolve_s3_bag's `data/{robot_id}/` layout."""
    from replay.data_manager import resolve_incident_bag

    incident_id = "INC-001"
    module = "perception"
    bucket = "incident-bags"
    incident_prefix = f"incidents/{module}/{incident_id}/"

    # Files to "download" from S3 — place them relative to the incident_prefix
    files = {
        f"{incident_prefix}metadata.yaml": b"version: 9\n",
        f"{incident_prefix}bag/rosbag2_0.db3": b"fake",
    }
    s3_client = _make_mock_s3_client(bucket, incident_prefix, tmp_path, files)

    ref = resolve_incident_bag(
        incident_id=incident_id,
        module=module,
        bucket=bucket,
        dest_dir=tmp_path,
        s3_client=s3_client,
    )

    assert ref.source == "s3"
    # The canonical incidents/ layout key must appear in the s3_uri
    assert f"incidents/{module}/{incident_id}/" in ref.s3_uri
    # local_path must exist (download_file wrote the files)
    assert ref.local_path.exists()
    # paginator was called with the canonical prefix
    call_kwargs = s3_client.get_paginator.return_value.paginate.call_args
    assert call_kwargs.kwargs["Prefix"] == incident_prefix or incident_prefix in str(call_kwargs)


def test_resolve_incident_bag_threads_module(tmp_path: Path):
    """The canonical key contains the `module` param — it is NOT hardcoded."""
    from replay.data_manager import resolve_incident_bag

    for module in ("navigation", "manipulation"):
        incident_id = "INC-TEST"
        bucket = "incident-bags"
        incident_prefix = f"incidents/{module}/{incident_id}/"
        files = {f"{incident_prefix}metadata.yaml": b"version: 9\n"}
        s3_client = _make_mock_s3_client(bucket, incident_prefix, tmp_path, files)
        ref = resolve_incident_bag(
            incident_id=incident_id,
            module=module,
            bucket=bucket,
            dest_dir=tmp_path,
            s3_client=s3_client,
        )
        assert f"incidents/{module}/{incident_id}/" in ref.s3_uri


def test_resolve_incident_bag_raises_on_empty_prefix(tmp_path: Path):
    """resolve_incident_bag raises FileNotFoundError when the incident prefix lists
    no objects — mirrors resolve_s3_bag's no-match raise (T-0101-03)."""
    from replay.data_manager import resolve_incident_bag

    client = MagicMock()
    paginator = MagicMock()
    # Empty Contents — no objects under the incident prefix
    paginator.paginate.return_value = [{"Contents": []}]
    client.get_paginator.return_value = paginator

    with pytest.raises(FileNotFoundError, match="No incident bag found"):
        resolve_incident_bag(
            incident_id="INC-MISSING",
            module="perception",
            bucket="incident-bags",
            dest_dir=tmp_path,
            s3_client=client,
        )


def test_resolve_incident_bag_raises_on_page_with_no_contents(tmp_path: Path):
    """resolve_incident_bag raises FileNotFoundError when paginator yields a page
    with no Contents key (AWS returns an empty page on empty prefixes)."""
    from replay.data_manager import resolve_incident_bag

    client = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [{}]  # no "Contents" key at all
    client.get_paginator.return_value = paginator

    with pytest.raises(FileNotFoundError):
        resolve_incident_bag(
            incident_id="INC-MISSING",
            module="perception",
            bucket="incident-bags",
            dest_dir=tmp_path,
            s3_client=client,
        )


def test_resolve_incident_bag_downloads_files_to_dest(tmp_path: Path):
    """Files listed under the incident prefix are downloaded to dest_dir."""
    from replay.data_manager import resolve_incident_bag

    incident_id = "INC-002"
    module = "perception"
    bucket = "incident-bags"
    incident_prefix = f"incidents/{module}/{incident_id}/"
    files = {
        f"{incident_prefix}metadata.yaml": b"version: 9\n",
        f"{incident_prefix}bag/rosbag2_0.db3": b"bagdata",
    }
    s3_client = _make_mock_s3_client(bucket, incident_prefix, tmp_path, files)
    ref = resolve_incident_bag(
        incident_id=incident_id,
        module=module,
        bucket=bucket,
        dest_dir=tmp_path,
        s3_client=s3_client,
    )
    # download_file was called for each file in the files dict
    assert s3_client.download_file.call_count == len(files)
    assert ref.local_path.exists()


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


def test_incident_cmd_incident_id_path_threads_module(
    tmp_path: Path, configs_dir: Path
):
    """--incident-id + --s3-bucket resolves via resolve_incident_bag; `module` param is
    threaded into the call (not hardcoded). A monkeypatched resolve_incident_bag returns
    a fake DataRef so no live S3/boto3 network is hit."""
    from replay.data_manager import DataRef
    from replay.cli import main

    incident_id = "INC-001"
    bucket = "incident-bags"
    output_dir = tmp_path / "output"

    fake_bag_dir = tmp_path / "fake_bag"
    fake_bag_dir.mkdir()
    (fake_bag_dir / "metadata.yaml").write_text("version: 9\n")

    fake_ref = DataRef(local_path=fake_bag_dir, source="s3", s3_uri=f"s3://{bucket}/incidents/perception/{incident_id}/")
    captured = {}

    def fake_resolve_incident_bag(incident_id, module, bucket, dest_dir, *, s3_client=None):
        captured["incident_id"] = incident_id
        captured["module"] = module
        captured["bucket"] = bucket
        # Confirm canonical layout key would be built (module threaded)
        assert module == "perception"
        return fake_ref

    def fake_run_replay(module, data, output_dir):
        captured["run_replay_module"] = module.name
        captured["data_source"] = data.source
        result = MagicMock()
        result.exit_code = 0
        result.output_bag_path = output_dir / "bag"
        result.logs_dir = None
        return result

    def fake_metrics_pipeline(module_spec, bag_path, output_dir, **kwargs):
        return 0

    runner = CliRunner()
    with patch("replay.cli.resolve_incident_bag", side_effect=fake_resolve_incident_bag), \
         patch("replay.cli.run_replay", side_effect=fake_run_replay), \
         patch("replay.cli._run_metrics_pipeline", side_effect=fake_metrics_pipeline):
        result = runner.invoke(
            main,
            [
                "incident",
                "--module", "perception",
                "--incident-id", incident_id,
                "--s3-bucket", bucket,
                "--output", str(output_dir),
                "--configs-dir", str(configs_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["module"] == "perception"
    assert captured["incident_id"] == incident_id
    assert captured["data_source"] == "s3"
    assert captured["run_replay_module"] == "perception"


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


def test_incident_cmd_incident_id_without_bucket_raises_usage_error(
    tmp_path: Path, configs_dir: Path
):
    """--incident-id without --s3-bucket (and no INCIDENT_BAG_BUCKET env) raises a
    UsageError mentioning --s3-bucket."""
    from replay.cli import main
    import os

    runner = CliRunner()
    # Ensure the env var is not set
    env = {k: v for k, v in os.environ.items() if k != "INCIDENT_BAG_BUCKET"}
    result = runner.invoke(
        main,
        [
            "incident",
            "--module", "perception",
            "--incident-id", "INC-001",
            "--output", str(tmp_path / "output"),
            "--configs-dir", str(configs_dir),
        ],
        env=env,
    )
    assert result.exit_code != 0
    assert "s3-bucket" in result.output.lower() or "bucket" in result.output.lower()


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
