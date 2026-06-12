"""Derive the `ros2 bag play --topics` filter from a ModuleSpec (RPLY-05)."""
from __future__ import annotations

import shlex

from replay.module_config import ModuleSpec


def build_topics_arg(spec: ModuleSpec) -> str:
    """Return the space-joined, shell-quoted --topics argument from spec.input_topics.

    Topic names flow into a bash command string, so every topic is shell-quoted
    (T-03-02) — mirroring the same invariant the runner already enforces when it
    builds the replay script directly.
    """
    return " ".join(shlex.quote(t) for t in spec.input_topics)
