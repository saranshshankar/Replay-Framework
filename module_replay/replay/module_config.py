"""Load per-module topic/package config from YAML."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

VALID_CONTAINERS = {"planner", "controller"}


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    container: str
    colcon_package: str
    input_topics: List[str]
    output_topics: List[str]
    launch_command: str


def load_module_config(module_name: str, configs_dir: Path) -> ModuleSpec:
    path = configs_dir / f"{module_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No module config found at {path}")

    data = yaml.safe_load(path.read_text())

    container = data["container"]
    if container not in VALID_CONTAINERS:
        raise ValueError(
            f"Invalid container '{container}' for module '{module_name}'; "
            f"must be one of {sorted(VALID_CONTAINERS)}"
        )

    return ModuleSpec(
        name=data["name"],
        container=container,
        colcon_package=data["colcon_package"],
        input_topics=list(data["input_topics"]),
        output_topics=list(data["output_topics"]),
        launch_command=data["launch"]["command"],
    )


def load_checkout_paths(module_name: str, configs_dir: Path) -> List[str]:
    """Load the list of 10xCode-relative paths to check out for a module.

    Reads `configs_dir / checkout_paths.yaml` — a mapping of module name to
    a list of path strings. Raises if the file or module entry is missing.
    """
    path = configs_dir / "checkout_paths.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Checkout paths config not found at {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if module_name not in data:
        raise KeyError(
            f"No checkout paths configured for module '{module_name}' in {path}"
        )
    return list(data[module_name])
