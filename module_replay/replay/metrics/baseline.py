"""BaselineManager: resolve a module's baseline for the metrics gate (MTRC-04).

The CI gate compares a candidate replay against a PINNED GOLDEN -- a stored,
version-controlled artifact (the bucket/key come from a committed manifest, not
from PR-controlled input). It must NEVER auto-promote or re-run a live ``dev``
build: ``rerun_dev`` is a local-only convenience that raises on the CI path.

Trust boundaries (plan 01-06 threat model):
- S3 -> BaselineManager: the golden artifact is downloaded read-only.
- golden.yaml -> BaselineManager: the manifest supplies bucket/key; parsed
  safely (T-06-01); the manifest is reviewed/committed, so a malicious
  PR cannot redirect the fetch. Download targets are built under a fixed cache
  dir from the trusted manifest prefix (T-06-04, mirrors data_manager).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from replay.metrics.base import BaselineRef

VALID_STRATEGIES = {"pinned_golden", "rerun_dev"}


class BaselineManager:
    def __init__(self, configs_dir: Path, s3_client=None, cache_dir: Optional[Path] = None):
        self._configs_dir = Path(configs_dir)
        self._s3_client = s3_client
        self._cache_dir = cache_dir

    def _load_golden_manifest(self, module: str) -> dict:
        path = self._configs_dir / "baselines" / module / "golden.yaml"
        if not path.exists():
            raise FileNotFoundError(f"No golden manifest for module '{module}' at {path}")
        return yaml.safe_load(path.read_text()) or {}

    def resolve(self, module: str, strategy: str = "pinned_golden") -> BaselineRef:
        if strategy not in VALID_STRATEGIES:
            raise ValueError(f"Unknown baseline strategy '{strategy}'; must be one of {sorted(VALID_STRATEGIES)}")
        if strategy == "pinned_golden":
            manifest = self._load_golden_manifest(module)
            bag = self._fetch_from_s3(manifest)   # mirror data_manager.resolve_s3_bag
            return BaselineRef(bag_path=bag, strategy="pinned_golden",
                               aligned_start_ts=0, run_id=str(manifest.get("baseline_sha", "PROVISIONAL")))
        # rerun_dev: local interactive only -- never used by CI
        raise NotImplementedError("rerun_dev is a local-only convenience; not available from CI")

    def _fetch_from_s3(self, manifest: dict) -> Path:
        import boto3

        client = self._s3_client or boto3.client("s3")
        bucket = manifest["s3_bucket"]
        prefix = manifest["s3_key"]
        dest = (self._cache_dir or Path(".replay_work/baselines")) / prefix
        dest.mkdir(parents=True, exist_ok=True)
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):].lstrip("/")
                if not rel:  # skip the prefix "directory" marker
                    continue
                local = dest / rel
                local.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(local))
        return dest
