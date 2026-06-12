from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from replay.docker_utils import (
    compose_build,
    compose_pull,
    compose_up,
    exec_in_container,
    run_cmd,
)


def test_run_cmd_invokes_subprocess():
    with patch("replay.docker_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        result = run_cmd(["echo", "hi"])
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] == ["echo", "hi"]
        assert kwargs["check"] is True
        assert result.returncode == 0


def test_run_cmd_nonzero_raises_by_default():
    import subprocess as sp

    with patch("replay.docker_utils.subprocess.run") as mock_run:
        mock_run.side_effect = sp.CalledProcessError(1, ["false"])
        with pytest.raises(sp.CalledProcessError):
            run_cmd(["false"])


def test_compose_pull_builds_correct_args():
    with patch("replay.docker_utils.run_cmd") as mock_run:
        compose_pull(Path("/tmp/compose.yml"))
        mock_run.assert_called_once_with(
            ["docker", "compose", "-f", "/tmp/compose.yml", "pull"]
        )


def test_compose_build_builds_correct_args():
    with patch("replay.docker_utils.run_cmd") as mock_run:
        compose_build(Path("/tmp/compose.yml"))
        mock_run.assert_called_once_with(
            ["docker", "compose", "-f", "/tmp/compose.yml", "build"]
        )


def test_compose_up_detached():
    with patch("replay.docker_utils.run_cmd") as mock_run:
        compose_up(Path("/tmp/compose.yml"))
        mock_run.assert_called_once_with(
            ["docker", "compose", "-f", "/tmp/compose.yml", "up", "-d"]
        )


def test_exec_in_container_joins_command():
    with patch("replay.docker_utils.run_cmd") as mock_run:
        exec_in_container("v2-planner-docker-x86", "echo hello world")
        mock_run.assert_called_once_with(
            [
                "docker",
                "exec",
                "-i",
                "v2-planner-docker-x86",
                "bash",
                "-lc",
                "echo hello world",
            ]
        )


def test_compose_up_with_override():
    with patch("replay.docker_utils.run_cmd") as mock_run:
        compose_up(
            Path("/tmp/compose.yml"),
            overrides=[Path("/tmp/override.yml")],
        )
        mock_run.assert_called_once_with([
            "docker", "compose",
            "-f", "/tmp/compose.yml",
            "-f", "/tmp/override.yml",
            "up", "-d",
        ])


def test_compose_up_with_pull_policy():
    with patch("replay.docker_utils.run_cmd") as mock_run:
        compose_up(Path("/tmp/compose.yml"), pull_policy="missing")
        mock_run.assert_called_once_with([
            "docker", "compose",
            "-f", "/tmp/compose.yml",
            "up", "-d",
            "--pull", "missing",
        ])
