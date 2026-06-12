"""Module -> plugin registry (FRWK-06: pluggable architecture).

Plugin classes self-register at import time via the ``@register_metric`` /
``@register_viz`` decorators; the runner discovers them with
``get_metric_plugins`` / ``get_viz_plugins``. Pure in-process Python, no I/O.
"""
from __future__ import annotations

_REGISTRY: dict[str, list[type]] = {}
_VIZ_REGISTRY: dict[str, list[type]] = {}


def register_metric(module: str):
    def decorator(cls):
        _REGISTRY.setdefault(module, []).append(cls)
        return cls
    return decorator


def register_viz(module: str):
    def decorator(cls):
        _VIZ_REGISTRY.setdefault(module, []).append(cls)
        return cls
    return decorator


def get_metric_plugins(module: str) -> list:
    return _REGISTRY.get(module, [])


def get_viz_plugins(module: str) -> list:
    return _VIZ_REGISTRY.get(module, [])
