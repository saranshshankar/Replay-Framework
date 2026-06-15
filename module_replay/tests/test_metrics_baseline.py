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


def test_unknown_strategy_raises(tmp_path):
    with pytest.raises(ValueError, match="strategy"):
        BaselineManager(configs_dir=tmp_path).resolve("perception", "bogus")


def test_rerun_dev_never_hits_s3(tmp_path):
    """Gray Area 3: rerun_dev must not be a CI/S3 path."""
    with pytest.raises(NotImplementedError):
        BaselineManager(configs_dir=tmp_path).resolve("perception", "rerun_dev")
