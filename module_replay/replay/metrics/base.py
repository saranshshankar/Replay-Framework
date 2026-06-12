"""Shared metrics contracts: ABCs + frozen dataclasses.

These are the offline, pure-Python interfaces that every module plugin pack
(plan 01-05 perception, 01-06 faithfulness, future modules) implements, plus
the shared result/baseline dataclasses the report generator (plan 01-07)
consumes.

INVARIANT (PATTERNS § Key invariants #3): nothing under replay/metrics/ may
import the ROS runtime client library. Bag access is rosbags-only.

The ``"BagReader"`` type hints below are forward references kept as strings on
purpose: base.py must NOT import bag_reader, to avoid an import cycle (bag_reader
itself has no reason to import base, but keeping these as strings makes the
contract direction explicit and the module import-light).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineRef:
    bag_path: Path
    strategy: str            # "pinned_golden" | "rerun_dev"
    aligned_start_ts: int    # nanoseconds
    run_id: str


@dataclass(frozen=True)
class MetricResult:
    name: str
    module: str
    value: dict              # JSON-serializable
    passed: bool
    is_regression: bool


class BaseMetric(ABC):
    name: str
    requires_baseline: bool = False

    @abstractmethod
    def compute(self, reader: "BagReader", config: dict) -> dict: ...


class BaseRegressionMetric(BaseMetric):
    requires_baseline: bool = True

    @abstractmethod
    def compare(self, candidate: "BagReader", baseline: "BagReader", config: dict) -> dict: ...


class BaseVisualization(ABC):
    @abstractmethod
    def render(self, reader: "BagReader", metrics: dict, output_dir: Path) -> list[Path]: ...
