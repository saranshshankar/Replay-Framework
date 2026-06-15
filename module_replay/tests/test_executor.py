import pytest

from replay.executor import ExecSpec, ExecutorBackend, LocalExecutor


def test_local_executor_returns_exit_code(mocker):
    """FRWK-05: LocalExecutor returns the container's exit code (never raises)."""
    m = mocker.patch("replay.executor.subprocess.run")
    m.return_value.returncode = 42
    assert LocalExecutor().run(ExecSpec(container="c", shell_cmd="exit 42")) == 42


def test_local_executor_builds_docker_exec_argv(mocker):
    """The argv mirrors exec_in_container: docker exec -i <c> bash -lc <cmd>."""
    m = mocker.patch("replay.executor.subprocess.run")
    m.return_value.returncode = 0
    LocalExecutor().run(ExecSpec(container="planner-x86", shell_cmd="echo hi"))
    args, _ = m.call_args
    assert args[0] == ["docker", "exec", "-i", "planner-x86", "bash", "-lc", "echo hi"]


def test_executor_backend_is_abstract():
    with pytest.raises(TypeError):
        ExecutorBackend()  # cannot instantiate abstract base
