from pathlib import Path

import pytest

from replay.version_manager import (
    SubmoduleOverride,
    VersionSpec,
    load_version_spec,
)


def test_no_yaml_path_returns_defaults():
    spec = load_version_spec(None)
    assert spec == VersionSpec(tenxcode_branch="dev", submodule_overrides=[])


def test_empty_tenxcode_branch_defaults_to_dev(tmp_path: Path):
    path = tmp_path / "v.yaml"
    path.write_text("tenxcode: {}\n")
    spec = load_version_spec(path)
    assert spec.tenxcode_branch == "dev"


def test_explicit_tenxcode_branch(tmp_path: Path):
    path = tmp_path / "v.yaml"
    path.write_text("tenxcode:\n  branch: feature/foo\n")
    spec = load_version_spec(path)
    assert spec.tenxcode_branch == "feature/foo"


def test_submodule_override(tmp_path: Path):
    path = tmp_path / "v.yaml"
    path.write_text(
        """
tenxcode:
  branch: dev
submodules:
  common_interfaces:
    branch: master
  shared_db:
    branch: feature/db-fix
"""
    )
    spec = load_version_spec(path)
    assert spec.tenxcode_branch == "dev"
    assert SubmoduleOverride("common_interfaces", "master") in spec.submodule_overrides
    assert SubmoduleOverride("shared_db", "feature/db-fix") in spec.submodule_overrides
    assert len(spec.submodule_overrides) == 2


def test_unknown_submodule_raises(tmp_path: Path):
    path = tmp_path / "v.yaml"
    path.write_text(
        """
submodules:
  not_a_real_submodule:
    branch: master
"""
    )
    with pytest.raises(ValueError, match="Unknown submodule"):
        load_version_spec(path)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_version_spec(tmp_path / "nope.yaml")
