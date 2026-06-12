"""Git helpers for the replay platform."""
from __future__ import annotations

from pathlib import Path

from replay.docker_utils import run_cmd
from replay.version_manager import SubmoduleOverride


def checkout_branch(repo: Path, branch: str) -> None:
    """Fetch + check out a branch in the given repo.

    The fetch is best-effort (``check=False``): if ``origin`` is missing or the
    branch has no remote counterpart, the local ``checkout`` will still run and
    surface a clear error if the branch does not exist locally either.
    """
    run_cmd(["git", "-C", str(repo), "fetch", "origin", branch], check=False)
    run_cmd(["git", "-C", str(repo), "checkout", branch])


def submodule_update_recursive(repo: Path) -> None:
    """Pin all submodules to the commits recorded in the current superproject commit."""
    run_cmd(["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"])


def checkout_submodule_branch(repo: Path, override: SubmoduleOverride) -> None:
    """Check out a specific branch in a submodule, overriding the pinned commit."""
    submodule_path = repo / override.name
    run_cmd(["git", "-C", str(submodule_path), "fetch", "origin", override.branch])
    run_cmd(["git", "-C", str(submodule_path), "checkout", override.branch])


def checkout_paths_from_branch(repo: Path, branch: str, paths: list[str]) -> None:
    """Update only the given paths to match ``branch`` (partial checkout).

    Uses ``git checkout <branch> -- <paths>`` so files outside the list are
    left untouched. HEAD does not move. Caller is responsible for picking a
    path list that covers everything the downstream build needs.
    """
    if not paths:
        raise ValueError("checkout_paths_from_branch requires at least one path")
    run_cmd(["git", "-C", str(repo), "fetch", "origin", branch], check=False)
    run_cmd(["git", "-C", str(repo), "checkout", branch, "--", *paths])


def submodule_init(repo: Path, submodule_path: str) -> None:
    """Fetch a single submodule's contents (``git submodule update --init <path>``).

    Needed after a partial checkout that updates the gitlink for a submodule —
    the gitlink alone doesn't populate the submodule's working tree.
    """
    run_cmd(
        ["git", "-C", str(repo), "submodule", "update", "--init", submodule_path]
    )
