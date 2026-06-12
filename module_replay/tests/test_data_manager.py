from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from replay.data_manager import DataRef, resolve_local_bag, resolve_s3_bag


def _make_fake_bag(tmp_path: Path) -> Path:
    bag_dir = tmp_path / "rosbag2_2026_04_09-18_44_09"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("version: 5\n")
    (bag_dir / "rosbag2_2026_04_09-18_44_09_0.db3").write_bytes(b"fake")
    return bag_dir


def test_local_bag_happy_path(tmp_path: Path):
    bag_dir = _make_fake_bag(tmp_path)
    ref = resolve_local_bag(bag_dir)
    assert isinstance(ref, DataRef)
    assert ref.local_path == bag_dir
    assert ref.source == "local"


def test_local_bag_missing_dir_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_local_bag(tmp_path / "nope")


def test_local_bag_missing_metadata_raises(tmp_path: Path):
    bag_dir = tmp_path / "empty"
    bag_dir.mkdir()
    with pytest.raises(ValueError, match="metadata.yaml"):
        resolve_local_bag(bag_dir)


@mock_aws
def test_resolve_s3_bag_downloads_bag_directory(tmp_path: Path):
    bucket = "replay-test-bucket"
    robot = "robot_03"
    task = "task_20260409_184409"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket)

    prefix = f"data/{robot}/2026-04-09/recordings/task/{task}_184409"
    s3.put_object(Bucket=bucket, Key=f"{prefix}/metadata.json", Body=b"{}")
    s3.put_object(Bucket=bucket, Key=f"{prefix}/bag/metadata.yaml", Body=b"version: 5")
    s3.put_object(Bucket=bucket, Key=f"{prefix}/bag/rosbag2_x_0.db3", Body=b"fake")

    ref = resolve_s3_bag(
        task_id=task,
        robot_id=robot,
        bucket=bucket,
        dest_dir=tmp_path,
        s3_client=s3,
    )

    assert ref.source == "s3"
    assert ref.s3_uri == f"s3://{bucket}/{prefix}/bag/"
    assert (ref.local_path / "metadata.yaml").read_bytes() == b"version: 5"
    assert (ref.local_path / "rosbag2_x_0.db3").read_bytes() == b"fake"


@mock_aws
def test_resolve_s3_bag_no_match_raises(tmp_path: Path):
    bucket = "replay-test-bucket"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket)

    with pytest.raises(FileNotFoundError, match="No recording found"):
        resolve_s3_bag(
            task_id="nonexistent",
            robot_id="robot_03",
            bucket=bucket,
            dest_dir=tmp_path,
            s3_client=s3,
        )


@mock_aws
def test_resolve_s3_bag_multiple_matches_raises(tmp_path: Path):
    bucket = "replay-test-bucket"
    robot = "robot_03"
    task = "task_123"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket)

    for date in ("2026-04-09", "2026-04-10"):
        prefix = f"data/{robot}/{date}/recordings/task/{task}_0000"
        s3.put_object(Bucket=bucket, Key=f"{prefix}/bag/metadata.yaml", Body=b"x")

    with pytest.raises(ValueError, match="Multiple recordings"):
        resolve_s3_bag(
            task_id=task,
            robot_id=robot,
            bucket=bucket,
            dest_dir=tmp_path,
            s3_client=s3,
        )
