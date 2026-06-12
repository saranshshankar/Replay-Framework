"""Thin subprocess wrappers around docker compose + docker exec."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional


def run_cmd(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command as a list of args. Raises CalledProcessError on non-zero by default."""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def _compose_files_args(compose_file: Path, overrides: Optional[List[Path]] = None) -> List[str]:
    args = ["-f", str(compose_file)]
    for o in overrides or []:
        args += ["-f", str(o)]
    return args


def compose_pull(compose_file: Path, overrides: Optional[List[Path]] = None) -> None:
    run_cmd(["docker", "compose", *_compose_files_args(compose_file, overrides), "pull"])


def compose_build(compose_file: Path, overrides: Optional[List[Path]] = None) -> None:
    run_cmd(["docker", "compose", *_compose_files_args(compose_file, overrides), "build"])


def compose_up(
    compose_file: Path,
    overrides: Optional[List[Path]] = None,
    *,
    pull_policy: Optional[str] = None,
) -> None:
    """Run `docker compose ... up -d`.

    `pull_policy` maps to `docker compose up`'s `--pull` flag. Pass
    "missing" to pull only when no local image exists, "always" to force a
    fresh pull, or "never" to refuse to pull. When None, the flag is
    omitted and docker compose's default applies.
    """
    extra: List[str] = []
    if pull_policy is not None:
        extra = ["--pull", pull_policy]
    run_cmd(
        ["docker", "compose", *_compose_files_args(compose_file, overrides), "up", "-d", *extra]
    )


def exec_in_container(container: str, shell_cmd: str) -> None:
    """Execute `shell_cmd` inside the named running container using bash -lc.

    The `-i` flag keeps stdin attached so SIGINT from the host terminal
    propagates into the container's shell — needed for the runner's
    trap-based cleanup to fire on Ctrl-C.
    """
    run_cmd(["docker", "exec", "-i", container, "bash", "-lc", shell_cmd])
