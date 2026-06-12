import importlib
from pathlib import Path

from replay import paths


def test_host_tenxcode_path_is_absolute():
    assert paths.HOST_TENXCODE.is_absolute()


def test_host_tenxcode_honours_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("TENXCODE_ROOT", str(tmp_path))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.HOST_TENXCODE == tmp_path.resolve()
    finally:
        monkeypatch.delenv("TENXCODE_ROOT", raising=False)
        importlib.reload(paths)


def test_host_tenxcode_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("TENXCODE_ROOT", raising=False)
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.HOST_TENXCODE == Path("~/workspace/src/10xCode").expanduser().resolve()
    finally:
        importlib.reload(paths)


def test_container_tenxcode_path():
    assert paths.CONTAINER_TENXCODE == Path("/root/ros2_ws/src/10xCode")


def test_compose_files_are_under_tenxcode():
    assert paths.PLANNER_COMPOSE.is_relative_to(paths.HOST_TENXCODE)
    assert paths.CONTROLLER_COMPOSE.is_relative_to(paths.HOST_TENXCODE)


def test_container_names():
    assert paths.PLANNER_CONTAINER == "v2-planner-docker-x86"
    assert paths.CONTROLLER_CONTAINER == "v2-controller-docker-x86"


def test_host_replay_root_is_the_repo_root():
    # paths.py lives at module_replay/replay/paths.py — two parents up is the repo root.
    assert paths.HOST_REPLAY_ROOT == Path(paths.__file__).resolve().parent.parent
    assert (paths.HOST_REPLAY_ROOT / "pyproject.toml").exists()


def test_host_bag_library_honours_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("REPLAY_BAG_LIBRARY", str(tmp_path))
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.HOST_BAG_LIBRARY == tmp_path.resolve()
    finally:
        monkeypatch.delenv("REPLAY_BAG_LIBRARY", raising=False)
        importlib.reload(paths)


def test_host_bag_library_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("REPLAY_BAG_LIBRARY", raising=False)
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.HOST_BAG_LIBRARY == Path("~/data").expanduser().resolve()
    finally:
        importlib.reload(paths)


def test_container_bag_library_path():
    assert paths.CONTAINER_BAG_LIBRARY == Path("/root/data")
