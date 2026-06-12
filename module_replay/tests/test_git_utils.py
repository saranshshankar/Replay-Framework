from pathlib import Path

import pytest

from replay.git_utils import (
    checkout_branch,
    checkout_paths_from_branch,
    checkout_submodule_branch,
    submodule_init,
    submodule_update_recursive,
)
from replay.version_manager import SubmoduleOverride


def _init_repo_with_branches(path: Path) -> None:
    import subprocess as sp

    sp.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    (path / "a.txt").write_text("a")
    sp.run(["git", "-C", str(path), "add", "."], check=True)
    sp.run(
        ["git", "-C", str(path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-qm", "init"],
        check=True,
    )
    sp.run(["git", "-C", str(path), "branch", "dev"], check=True)
    sp.run(["git", "-C", str(path), "branch", "feature/foo"], check=True)


def test_checkout_branch_switches_branch(tmp_path: Path):
    repo = tmp_path / "r"
    _init_repo_with_branches(repo)

    checkout_branch(repo, "feature/foo")

    import subprocess as sp
    result = sp.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "feature/foo"


def test_checkout_branch_missing_raises(tmp_path: Path):
    repo = tmp_path / "r"
    _init_repo_with_branches(repo)

    with pytest.raises(Exception):
        checkout_branch(repo, "no-such-branch")


def test_submodule_update_recursive_calls_git(tmp_path: Path, mocker):
    repo = tmp_path / "r"
    _init_repo_with_branches(repo)
    mock_run = mocker.patch("replay.git_utils.run_cmd")

    submodule_update_recursive(repo)

    mock_run.assert_called_once_with(
        ["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"]
    )


def test_checkout_submodule_branch(tmp_path: Path, mocker):
    repo = tmp_path / "r"
    _init_repo_with_branches(repo)
    mock_run = mocker.patch("replay.git_utils.run_cmd")

    override = SubmoduleOverride(name="common_interfaces", branch="master")
    checkout_submodule_branch(repo, override)

    submodule_path = repo / "common_interfaces"
    calls = [c.args[0] for c in mock_run.call_args_list]
    assert ["git", "-C", str(submodule_path), "fetch", "origin", "master"] in calls
    assert ["git", "-C", str(submodule_path), "checkout", "master"] in calls


def test_checkout_paths_from_branch_passes_paths(tmp_path: Path, mocker):
    repo = tmp_path / "r"
    mock_run = mocker.patch("replay.git_utils.run_cmd")

    checkout_paths_from_branch(repo, "dev", ["perception/", "common_interfaces/"])

    calls = [c.args[0] for c in mock_run.call_args_list]
    assert ["git", "-C", str(repo), "fetch", "origin", "dev"] in calls
    assert [
        "git", "-C", str(repo), "checkout", "dev",
        "--", "perception/", "common_interfaces/",
    ] in calls


def test_checkout_paths_from_branch_empty_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        checkout_paths_from_branch(tmp_path, "dev", [])


def test_submodule_init_invokes_git(tmp_path: Path, mocker):
    mock_run = mocker.patch("replay.git_utils.run_cmd")
    submodule_init(tmp_path, "common_interfaces")
    mock_run.assert_called_once_with(
        ["git", "-C", str(tmp_path), "submodule", "update", "--init", "common_interfaces"]
    )
