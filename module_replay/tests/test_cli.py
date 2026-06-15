from pathlib import Path

from click.testing import CliRunner

from replay.cli import main


def test_cli_help_lists_subcommands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["validate", "fetch-data", "setup-env", "run", "all"]:
        assert cmd in result.output


def test_validate_prints_resolved_spec(tmp_path: Path, mocker):
    # Point CLI at a temporary modules dir containing perception.yaml
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: ["/a"]
output_topics: ["/b"]
launch:
  command: "ros2 launch x y.launch.py"
"""
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["validate", "--module", "perception", "--configs-dir", str(configs.parent)],
    )
    assert result.exit_code == 0, result.output
    assert "perception" in result.output
    assert "realtime_perception" in result.output
    assert "dev" in result.output  # default tenxcode branch


def test_fetch_data_local_bag(tmp_path: Path, mocker):
    bag = tmp_path / "rosbag2_x"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["fetch-data", "--local-bag", str(bag), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert str(bag) in result.output


def test_fetch_data_requires_task_id_or_local(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(main, ["fetch-data", "--output", str(tmp_path)])
    assert result.exit_code != 0
    assert "task-id" in result.output.lower() or "local-bag" in result.output.lower()


def test_setup_env_calls_setup_environment(tmp_path: Path, mocker):
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs.parent)],
    )
    assert result.exit_code == 0, result.output
    setup.assert_called_once()


def test_setup_env_auto_loads_default_version_yaml(tmp_path: Path, mocker):
    """If --version-yaml is not passed, `configs/versions/default.yaml` is used
    automatically when it exists."""
    configs = tmp_path / "configs"
    (configs / "modules").mkdir(parents=True)
    (configs / "modules" / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    (configs / "versions").mkdir()
    (configs / "versions" / "default.yaml").write_text(
        "tenxcode:\n  branch: feature/from-default-yaml\n"
    )

    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs)],
    )
    assert result.exit_code == 0, result.output
    version_spec = setup.call_args.args[0]
    assert version_spec.tenxcode_branch == "feature/from-default-yaml"


def test_setup_env_partial_checkout_loads_paths(tmp_path: Path, mocker):
    configs = tmp_path / "configs"
    (configs / "modules").mkdir(parents=True)
    (configs / "modules" / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    (configs / "modules" / "checkout_paths.yaml").write_text(
        "perception:\n  - perception/\n  - common_interfaces/\n"
    )

    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "setup-env", "--module", "perception",
            "--configs-dir", str(configs),
            "--partial-checkout",
        ],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert kwargs["checkout_paths"] == ["perception/", "common_interfaces/"]


def test_setup_env_no_partial_checkout_means_none(tmp_path: Path, mocker):
    configs = tmp_path / "configs"
    (configs / "modules").mkdir(parents=True)
    (configs / "modules" / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs)],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert kwargs["checkout_paths"] is None


def test_setup_env_passes_only_checkout_paths_kwarg(tmp_path: Path, mocker):
    """After dropping the skip flags, the only kwarg setup-env forwards is
    `checkout_paths` (default None)."""
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs.parent)],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert "skip_pull" not in kwargs
    assert "skip_build" not in kwargs
    assert kwargs["checkout_paths"] is None


def test_setup_env_default_build_jobs_is_two(tmp_path: Path, mocker):
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["setup-env", "--module", "perception", "--configs-dir", str(configs.parent)],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert kwargs["build_jobs"] == 2


def test_setup_env_build_jobs_flag_forwarded(tmp_path: Path, mocker):
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    setup = mocker.patch("replay.cli.setup_environment")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "setup-env", "--module", "perception",
            "--configs-dir", str(configs.parent),
            "--build-jobs", "4",
        ],
    )
    assert result.exit_code == 0, result.output
    _, kwargs = setup.call_args
    assert kwargs["build_jobs"] == 4


def test_run_calls_run_replay(tmp_path: Path, mocker):
    configs = tmp_path / "configs" / "modules"
    configs.mkdir(parents=True)
    (configs / "perception.yaml").write_text(
        """
name: perception
container: planner
colcon_package: realtime_perception
input_topics: []
output_topics: []
launch:
  command: "x"
"""
    )
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")

    run_mock = mocker.patch("replay.cli.run_replay")
    run_mock.return_value = mocker.MagicMock(output_bag_path=tmp_path / "out" / "replay_output", exit_code=0)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--module", "perception",
            "--bag", str(bag),
            "--output", str(tmp_path / "out"),
            "--configs-dir", str(configs.parent),
        ],
    )
    assert result.exit_code == 0, result.output
    run_mock.assert_called_once()


def test_all_cmd_propagates_nonzero_exit(tmp_path: Path, mocker):
    """FRWK-03 / B9: a non-zero replay exit maps to process exit 3.

    Codes 1 and 2 are reserved for the metrics verdict; any replay/setup
    failure maps to 3 with the container code echoed for diagnosis.
    """
    from replay.cli import main
    from replay.runner import RunResult

    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("version: 5")
    mocker.patch("replay.cli.setup_environment")
    mocker.patch(
        "replay.cli.run_replay",
        return_value=RunResult(output_bag_path=bag, exit_code=137),
    )
    r = CliRunner().invoke(
        main,
        [
            "all",
            "--module", "perception",
            "--local-bag", str(bag),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert r.exit_code == 3, r.output   # B9: replay failures map to 3
