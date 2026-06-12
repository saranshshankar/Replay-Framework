"""Execution backend abstraction (FRWK-05).

`ExecutorBackend` is the seam that keeps an AWS/remote executor additive: the
runner depends on this ABC, not on a concrete docker-exec call, so a future
``AwsBatchExecutor`` (or similar) drops in without rewriting the runner. The
only backend implemented in Phase 1 is `LocalExecutor`, which wraps the same
``docker exec -i <container> bash -lc <cmd>`` invariant as
``docker_utils.exec_in_container`` and returns the container's exit code
without raising on non-zero (the caller owns the exit-code contract, B9).
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ExecSpec:
    """A single execution request: the shell command to run inside ``container``.

    ``log_dir`` is an optional host directory for a backend to stream logs to;
    `LocalExecutor` does not use it yet (logs go to the inherited stdout/stderr),
    but it is part of the contract so a remote backend can honour it.
    """

    container: str
    shell_cmd: str
    log_dir: Optional[Path] = None


class ExecutorBackend(ABC):
    @abstractmethod
    def run(self, spec: ExecSpec) -> int:
        """Execute ``spec.shell_cmd``; return the exit code. Never raises on non-zero."""
        ...


class LocalExecutor(ExecutorBackend):
    def run(self, spec: ExecSpec) -> int:
        # Mirrors docker_utils.exec_in_container: -i keeps stdin attached so a
        # host SIGINT propagates into the container shell; subprocess.run (not
        # check=True) so a non-zero container exit is returned, not raised.
        result = subprocess.run(
            ["docker", "exec", "-i", spec.container, "bash", "-lc", spec.shell_cmd],
            text=True,
        )
        return result.returncode
