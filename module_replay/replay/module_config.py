"""Load per-module topic/package config from YAML."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

VALID_CONTAINERS = {"planner", "controller"}


@dataclass(frozen=True)
class ThresholdSpec:
    """A typed per-metric goal threshold parsed from a module's `thresholds:` block.

    `tier` distinguishes validity gates (replay faithfulness — must hold for the
    run to count) from quality gates (the module's output standards). `provisional`
    flags a threshold that is a best-guess pending module-owner sign-off.
    """

    max: Optional[float] = None
    min: Optional[float] = None
    tolerance_band: float = 0.0
    provisional: bool = True
    tier: str = "quality"  # "validity" | "quality"


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    container: str
    colcon_package: str
    input_topics: List[str]
    output_topics: List[str]
    launch_command: str
    # New fields are all defaulted so the 6-field perception_spec conftest fixture
    # and all existing callers keep constructing ModuleSpec unchanged.
    qos_override_path: Optional[Path] = None      # relative to configs/qos/<module>.yaml
    thresholds: dict = field(default_factory=dict)   # name -> ThresholdSpec
    mocks: list = field(default_factory=list)        # Phase 3+ mock node specs
    launch_args: dict = field(default_factory=dict)  # extra "key:=value" launch pairs
    preflight_assets: list = field(default_factory=list)  # host paths checked fail-fast
    # Metric-config block (01-10): threaded into the metric cfg by cli so plugins
    # stop falling back to broken defaults (flat-10Hz / all-output-topics).
    expected_hz: dict = field(default_factory=dict)  # topic-substring -> expected publish Hz
    depth_topics: list = field(default_factory=list)  # depth output topics for DepthMetric
    diagnostics_topic: Optional[str] = None           # perception diagnostics topic for LatencyMetric


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

    qos_raw = data.get("qos_override")
    qos_override_path = Path(qos_raw) if qos_raw else None

    # Flatten the two-tier (validity + quality) thresholds block into a single
    # name -> ThresholdSpec dict, carrying the tier on each spec.
    thresholds_raw = data.get("thresholds") or {}
    thresholds: dict[str, ThresholdSpec] = {}
    for tier_name, tier_dict in thresholds_raw.items():
        for metric_name, spec in (tier_dict or {}).items():
            thresholds[metric_name] = ThresholdSpec(
                max=spec.get("max"),
                min=spec.get("min"),
                tolerance_band=spec.get("tolerance_band", 0.0),
                provisional=spec.get("provisional", True),
                tier=tier_name,
            )

    launch_block = data["launch"]
    launch_args = launch_block.get("args") or {}
    mocks = data.get("mocks") or []
    preflight_assets = list(data.get("preflight_assets") or [])

    # Metric-config block (01-10): per-topic expected_hz map, depth topic list, and
    # the diagnostics topic. All optional (defaulted) so 6-field callers are unaffected.
    expected_hz = data.get("expected_hz") or {}
    depth_topics = list(data.get("depth_topics") or [])
    diagnostics_topic = data.get("diagnostics_topic")

    return ModuleSpec(
        name=data["name"],
        container=container,
        colcon_package=data["colcon_package"],
        input_topics=list(data["input_topics"]),
        output_topics=list(data["output_topics"]),
        launch_command=launch_block["command"],
        qos_override_path=qos_override_path,
        thresholds=thresholds,
        mocks=mocks,
        launch_args=launch_args,
        preflight_assets=preflight_assets,
        expected_hz=expected_hz,
        depth_topics=depth_topics,
        diagnostics_topic=diagnostics_topic,
    )


def missing_preflight_assets(spec: ModuleSpec) -> List[str]:
    """Expand ~ and $ENV in each preflight asset path; return the paths that don't exist.

    B5 phase-0 fail-fast: a missing TensorRT engine / camera-intrinsics LUT /
    `config/current` param tree must die with the path named, not as a deep
    `on_activate` mystery inside the container (KT/playbooks/01-perception.md Step 8).
    CLI enforcement (echo missing + sys.exit(3) before run_replay) is wired in plan 01-07.
    """
    missing: List[str] = []
    for raw in spec.preflight_assets:
        p = Path(os.path.expandvars(os.path.expanduser(str(raw))))
        if not p.exists():
            missing.append(str(p))
    return missing


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
