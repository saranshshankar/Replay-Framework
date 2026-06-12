"""Tests for the metrics registry (FRWK-06: pluggable architecture).

@register_metric / @register_viz populate per-module registries that
get_metric_plugins / get_viz_plugins read back. Mirrors the valid/invalid
style of test_module_config.
"""
from __future__ import annotations


def test_register_metric_populates_registry():
    """FRWK-06: @register_metric makes the class discoverable by module name."""
    from replay.metrics.registry import register_metric, get_metric_plugins

    @register_metric("unit_test_module")
    class FakeMetric:
        pass

    assert FakeMetric in get_metric_plugins("unit_test_module")


def test_get_metric_plugins_unknown_returns_empty():
    from replay.metrics.registry import get_metric_plugins

    assert get_metric_plugins("does_not_exist_xyz") == []


def test_register_viz_populates_viz_registry():
    """FRWK-06: @register_viz makes the viz class discoverable by module name."""
    from replay.metrics.registry import register_viz, get_viz_plugins

    @register_viz("unit_test_module")
    class FakeViz:
        pass

    assert FakeViz in get_viz_plugins("unit_test_module")


def test_get_viz_plugins_unknown_returns_empty():
    from replay.metrics.registry import get_viz_plugins

    assert get_viz_plugins("does_not_exist_xyz") == []
