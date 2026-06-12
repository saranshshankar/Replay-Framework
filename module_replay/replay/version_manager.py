"""Parse the version-spec YAML into a resolved VersionSpec."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

KNOWN_SUBMODULES = {
    "common_interfaces",
    "sim-testing-envs",
    "version_system",
    "shared_db",
}
DEFAULT_TENXCODE_BRANCH = "dev"


@dataclass(frozen=True)
class SubmoduleOverride:
    name: str
    branch: str


@dataclass(frozen=True)
class VersionSpec:
    tenxcode_branch: str = DEFAULT_TENXCODE_BRANCH
    submodule_overrides: List[SubmoduleOverride] = field(default_factory=list)


def load_version_spec(yaml_path: Optional[Path]) -> VersionSpec:
    if yaml_path is None:
        return VersionSpec()

    if not yaml_path.exists():
        raise FileNotFoundError(f"Version YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text()) or {}

    tenx = data.get("tenxcode") or {}
    branch = tenx.get("branch") or DEFAULT_TENXCODE_BRANCH

    overrides: List[SubmoduleOverride] = []
    submodules = data.get("submodules") or {}
    for name, cfg in submodules.items():
        if name not in KNOWN_SUBMODULES:
            raise ValueError(
                f"Unknown submodule '{name}'; known: {sorted(KNOWN_SUBMODULES)}"
            )
        overrides.append(SubmoduleOverride(name=name, branch=cfg["branch"]))

    return VersionSpec(tenxcode_branch=branch, submodule_overrides=overrides)
