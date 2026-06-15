"""Derive the `ros2 launch` key:=value argument string from a ModuleSpec (RPLY-05)."""
from __future__ import annotations

from replay.module_config import ModuleSpec


def build_launch_args_str(spec: ModuleSpec) -> str:
    """Return 'key:=value ...' rendered from spec.launch_args (empty string when none)."""
    return " ".join(f"{k}:={v}" for k, v in spec.launch_args.items())
