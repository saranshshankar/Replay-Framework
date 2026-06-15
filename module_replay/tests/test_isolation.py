"""Tests for the pure isolation-derivation package (RPLY-05).

These functions take a ModuleSpec and derive replay-isolation strings (the
--topics filter, the launch key:=value args, the mock-node bash fragment) with
no I/O, so plain function calls assert on the returned strings — no mocking
needed, mirroring the string-assertion style in test_runner.py.
"""
from replay.isolation.launch_args import build_launch_args_str
from replay.isolation.mock_nodes import build_mock_fragment
from replay.isolation.topic_filter import build_topics_arg


def test_topic_filter_includes_all_input_topics(perception_spec):
    """RPLY-05: build_topics_arg returns every ModuleSpec.input_topics entry."""
    result = build_topics_arg(perception_spec)
    for topic in perception_spec.input_topics:
        assert topic in result


def test_topic_filter_shell_quotes_topics():
    """T-03-02: topic names flow into a shell arg, so each is shlex-quoted."""
    from replay.module_config import ModuleSpec

    spec = ModuleSpec(
        name="x",
        container="planner",
        colcon_package="p",
        input_topics=["/safe", "/has space"],
        output_topics=[],
        launch_command="ros2 launch p x.launch.py",
    )
    result = build_topics_arg(spec)
    # shlex.quote wraps a topic containing whitespace in single quotes.
    assert "'/has space'" in result


def test_launch_args_empty_when_none(perception_spec):
    assert build_launch_args_str(perception_spec) == ""


def test_launch_args_renders_key_value_pairs():
    from replay.module_config import ModuleSpec

    spec = ModuleSpec(
        name="x",
        container="planner",
        colcon_package="p",
        input_topics=[],
        output_topics=[],
        launch_command="ros2 launch p x.launch.py",
        launch_args={"use_replay": "true", "mode": "sim"},
    )
    result = build_launch_args_str(spec)
    assert "use_replay:=true" in result
    assert "mode:=sim" in result


def test_mock_fragment_empty_for_perception(perception_spec):
    """Perception has no upstream service deps -> empty fragment."""
    assert build_mock_fragment(perception_spec) == ""


def test_mock_fragment_renders_ros2_run_for_each_mock():
    from replay.module_config import ModuleSpec

    spec = ModuleSpec(
        name="x",
        container="planner",
        colcon_package="p",
        input_topics=[],
        output_topics=[],
        launch_command="ros2 launch p x.launch.py",
        mocks=[{"pkg": "mock_pkg", "node": "mock_node"}],
    )
    fragment = build_mock_fragment(spec)
    assert "ros2 run mock_pkg mock_node" in fragment
    assert "setsid" in fragment
