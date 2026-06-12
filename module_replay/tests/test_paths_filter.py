from pathlib import Path

import yaml

FILTER = Path(__file__).resolve().parent.parent / "configs" / "ci" / "paths-filter.yml"


def test_paths_filter_parses_and_has_framework_and_perception():
    """CI-05: filter config is valid YAML with a framework fan-out and a perception filter."""
    data = yaml.safe_load(FILTER.read_text())
    assert "framework" in data and "perception" in data
    # framework must include the core files that should trigger ALL module gates
    fw = "\n".join(data["framework"])
    assert "runner.py" in fw and "metrics/base.py" in fw
    # perception must include its plugin pack
    assert any("metrics/perception" in g for g in data["perception"])
