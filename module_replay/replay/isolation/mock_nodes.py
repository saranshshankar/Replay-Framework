"""Generate the mock-node bash launch fragment from a ModuleSpec (RPLY-05)."""
from __future__ import annotations

from replay.module_config import ModuleSpec


def build_mock_fragment(spec: ModuleSpec) -> str:
    """Return a bash fragment launching mock ROS service/action servers for spec.mocks.

    Returns '' when spec has no mocks (perception in Phase 1). Each mock is a
    dict ``{pkg, node}`` launched under ``setsid`` so it is a direct child of
    the replay shell and the runner's trap-based cleanup can kill it reliably
    (same backgrounding discipline as the recorder/player in runner.py). Phase
    3+ populates spec.mocks for modules with upstream service deps.
    """
    if not spec.mocks:
        return ""
    parts: list[str] = []
    for mock in spec.mocks:
        parts.append(f"setsid ros2 run {mock['pkg']} {mock['node']} &")
    return "\n".join(parts)
