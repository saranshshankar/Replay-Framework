import os
from pathlib import Path

from click.testing import CliRunner

from replay.cli import main

# Machine-independent bag path: read from the environment so the test is portable
# (FRWK-02/FRWK-04). Set REPLAY_INTEGRATION_BAG_PATH to a local rosbag2 directory
# to exercise the end-to-end CLI path; the test skips cleanly when it is unset.
_bag_env = os.environ.get("REPLAY_INTEGRATION_BAG_PATH")
SAMPLE_BAG = Path(_bag_env) if _bag_env else None


def test_all_end_to_end_with_sample_bag(tmp_path: Path, mocker):
    if SAMPLE_BAG is None or not SAMPLE_BAG.exists():
        import pytest
        pytest.skip(
            "Set REPLAY_INTEGRATION_BAG_PATH to a real rosbag2 dir to run this test"
        )

    mocker.patch("replay.env_setup.compose_up")
    mocker.patch("replay.env_setup.checkout_branch")
    mocker.patch("replay.env_setup.submodule_update_recursive")
    mocker.patch("replay.env_setup.exec_in_container")
    mocker.patch("replay.runner.exec_in_container")

    # Short-circuit the runner's host-side filesystem operations.
    mocker.patch("replay.runner.shutil.copytree")
    mocker.patch("replay.runner.shutil.move")

    configs = Path(__file__).resolve().parent.parent / "configs"
    out = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "all",
            "--module", "perception",
            "--local-bag", str(SAMPLE_BAG),
            "--output", str(out),
            "--configs-dir", str(configs),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Output bag" in result.output
