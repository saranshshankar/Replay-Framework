import boto3
import pytest
from pathlib import Path
from moto import mock_aws

from replay.metrics.baseline import BaselineManager
from replay.metrics.base import BaselineRef


def _write_manifest(configs_dir: Path):
    d = configs_dir / "baselines" / "perception"
    d.mkdir(parents=True)
    (d / "golden.yaml").write_text(
        "s3_bucket: test-baselines\ns3_key: baselines/perception/v1/\nbaseline_sha: abc123\n")


@mock_aws
def test_pinned_golden_resolve(tmp_path):
    """MTRC-04: resolve('perception','pinned_golden') returns a BaselineRef from S3."""
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-baselines")
    s3.put_object(Bucket="test-baselines", Key="baselines/perception/v1/metrics.json", Body=b"{}")
    _write_manifest(tmp_path)
    ref = BaselineManager(configs_dir=tmp_path, s3_client=s3, cache_dir=tmp_path / "cache").resolve("perception")
    assert isinstance(ref, BaselineRef) and ref.strategy == "pinned_golden"
    assert ref.bag_path.exists()


def _write_golden_bag(bag_dir: Path, start_ns: int) -> int:
    """Write a tiny readable rosbag2 dir whose first message is at ``start_ns``.

    Returns the bag's metadata start_time (what aligned_start_ts must derive to).
    """
    import numpy as np
    from rosbags.rosbag2 import Reader, Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    topic = "/perception_node/camera_0/semantic_raw_sim"
    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        conn = writer.add_connection(topic, Image.__msgtype__, typestore=typestore)
        for i in range(3):
            ts = start_ns + i * 100_000_000
            hdr = Header(
                stamp=Time(sec=int(ts // 1_000_000_000), nanosec=int(ts % 1_000_000_000)),
                frame_id="cam0",
            )
            msg = Image(header=hdr, height=2, width=2, encoding="rgba8",
                        is_bigendian=0, step=8, data=np.zeros(16, dtype=np.uint8))
            writer.write(conn, ts, typestore.serialize_cdr(msg, Image.__msgtype__))
    with Reader(bag_dir) as reader:
        return int(reader.start_time)


@mock_aws
def test_aligned_start_ts_is_derived_not_hardcoded_zero(tmp_path):
    """01-15 / UAT gap 2: aligned_start_ts must be DERIVED from the resolved golden
    bag (its metadata start_time), NOT the hardcoded literal 0."""
    # Build a real readable golden bag with a non-zero start, upload its files to
    # the mocked S3 under the manifest key, then resolve and assert derivation.
    local_golden = tmp_path / "golden_src"
    start_ns = 1_700_000_000_000_000_000  # a realistic, decidedly-non-zero ns stamp
    expected_start = _write_golden_bag(local_golden, start_ns)
    assert expected_start != 0

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-baselines")
    for f in local_golden.iterdir():
        s3.upload_file(str(f), "test-baselines", f"baselines/perception/v1/{f.name}")
    _write_manifest(tmp_path)

    ref = BaselineManager(
        configs_dir=tmp_path, s3_client=s3, cache_dir=tmp_path / "cache"
    ).resolve("perception")
    assert isinstance(ref, BaselineRef)
    assert ref.bag_path.exists()
    # The derived value: equals the golden bag's metadata start_time, and is NOT 0.
    assert ref.aligned_start_ts == expected_start
    assert ref.aligned_start_ts != 0


def test_unknown_strategy_raises(tmp_path):
    with pytest.raises(ValueError, match="strategy"):
        BaselineManager(configs_dir=tmp_path).resolve("perception", "bogus")


def test_rerun_dev_never_hits_s3(tmp_path):
    """Gray Area 3: rerun_dev must not be a CI/S3 path."""
    with pytest.raises(NotImplementedError):
        BaselineManager(configs_dir=tmp_path).resolve("perception", "rerun_dev")
