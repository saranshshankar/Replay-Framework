"""Data manager: resolves a local or S3-hosted rosbag to a local path."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class DataRef:
    """A resolved, local rosbag directory ready to be played."""
    local_path: Path
    source: Literal["local", "s3"]
    s3_uri: Optional[str] = None


def resolve_local_bag(bag_dir: Path) -> DataRef:
    if not bag_dir.exists() or not bag_dir.is_dir():
        raise FileNotFoundError(f"Local bag directory not found: {bag_dir}")

    metadata = bag_dir / "metadata.yaml"
    if not metadata.exists():
        raise ValueError(
            f"Directory {bag_dir} does not contain metadata.yaml — expected a "
            f"rosbag2 directory (metadata.yaml + *.db3 files at its top level). "
            f"If this path holds multiple rosbag2 directories, point --local-bag "
            f"at a specific one, not the parent."
        )

    return DataRef(local_path=bag_dir, source="local")


def resolve_incident_bag(
    incident_id: str,
    module: str,
    bucket: str,
    dest_dir: Path,
    *,
    s3_client=None,
) -> DataRef:
    """Download the incident bag at the canonical incidents/<module>/<incident_id>/ key (HLD A5).

    The S3 key layout is `incidents/<module>/<incident_id>/` — NOT resolve_s3_bag's
    `data/{robot_id}/` layout. The injectable ``s3_client=None`` (default
    ``boto3.client("s3")``) is the testability seam: tests pass a FAKE client whose
    paginator/download_file write files into dest_dir — no live S3, no boto3 network
    in unit tests (T-0101-02).

    Raises FileNotFoundError when the incident prefix lists no objects (mirrors
    resolve_s3_bag's no-match raise — T-0101-03 no silent partial-download surface).
    """
    import boto3

    client = s3_client or boto3.client("s3")
    incident_prefix = f"incidents/{module}/{incident_id}/"  # HLD A5 — NOT data/{robot_id}/

    local_bag_dir = dest_dir / incident_id / "bag"
    local_bag_dir.mkdir(parents=True, exist_ok=True)

    found = False
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=incident_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(incident_prefix):]
            if not relative:
                continue
            local_file = local_bag_dir / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_file))
            found = True

    if not found:
        raise FileNotFoundError(
            f"No incident bag found at s3://{bucket}/{incident_prefix}"
        )

    return DataRef(
        local_path=local_bag_dir,
        source="s3",
        s3_uri=f"s3://{bucket}/{incident_prefix}",
    )


def resolve_s3_bag(
    task_id: str,
    robot_id: str,
    bucket: str,
    dest_dir: Path,
    *,
    s3_client=None,
) -> DataRef:
    """Find the S3 prefix matching `{task_id}_*` under `data/{robot_id}/` and download bag/."""
    import boto3

    client = s3_client or boto3.client("s3")
    robot_prefix = f"data/{robot_id}/"

    matches: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=robot_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Expected: data/{robot}/{date}/recordings/{mode}/{task_id}_{ts}/...
            # Indices:    0     1       2          3        4         5
            parts = key.split("/")
            if len(parts) < 7:
                continue
            if parts[3] != "recordings":
                continue
            recording_dirname = parts[5]
            if recording_dirname.startswith(f"{task_id}_") or recording_dirname == task_id:
                session_prefix = "/".join(parts[:6]) + "/"
                if session_prefix not in matches:
                    matches.append(session_prefix)

    if not matches:
        raise FileNotFoundError(
            f"No recording found for task_id={task_id} under s3://{bucket}/{robot_prefix}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple recordings match task_id={task_id}: {matches}. "
            f"Disambiguate by supplying --date or cleaning up S3."
        )

    session_prefix = matches[0]
    bag_prefix = session_prefix + "bag/"

    local_bag_dir = dest_dir / Path(session_prefix.rstrip("/")).name / "bag"
    local_bag_dir.mkdir(parents=True, exist_ok=True)

    for page in paginator.paginate(Bucket=bucket, Prefix=bag_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(bag_prefix):]
            if not relative:
                continue
            local_file = local_bag_dir / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_file))

    return DataRef(
        local_path=local_bag_dir,
        source="s3",
        s3_uri=f"s3://{bucket}/{bag_prefix}",
    )
