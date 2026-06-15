from pathlib import Path

from click.testing import CliRunner

from replay.cli import main

SAMPLE_BAG = Path("/home/bharat/data/local_fusion/local_fusion_data/rosbag2_2026_04_09-18_44_09")


def test_all_end_to_end_with_sample_bag(tmp_path: Path, mocker):
    if not SAMPLE_BAG.exists():
        import pytest
        pytest.skip(f"Sample bag not present at {SAMPLE_BAG}")

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
