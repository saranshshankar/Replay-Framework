"""Shared Tier-3 visualization runner (TIER3-VIZ-DESIGN.md §4).

Iterates a module's registered ``BaseVisualization`` plugins and renders each into
``<output_dir>/viz/``, isolating per-plugin failures so one bad plugin never aborts
the others or the caller.

LAZY / BIFURCATION (§2/§9): this module and the per-module ``viz`` package are
imported ONLY on the viz path (``--run-viz`` / the ``viz`` subcommand), never by the
cheap metrics pipeline. The metrics gate must stay lean and free of the mp4 encoder;
a missing ``[viz]`` extra can therefore never break a metrics run.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from replay.metrics.registry import get_viz_plugins

logger = logging.getLogger(__name__)

INSTALL_HINT = (
    "Visualization requires the [viz] extra (mp4 encoder). "
    "Install it with:  pip install module_replay[viz]"
)


def _viz_deps_available() -> bool:
    """True when the mp4 encoder (imageio[ffmpeg], the [viz] extra) is importable."""
    try:
        import imageio  # noqa: F401
    except Exception:
        return False
    return True


def _import_viz_package(module: str) -> None:
    """Best-effort import of ``replay.metrics.<module>.viz`` so its plugins
    self-register. A missing package is fine (the module may ship no viz, or the
    plugins may already be imported); a real error inside the package is logged,
    never raised."""
    try:
        importlib.import_module(f"replay.metrics.{module}.viz")
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("viz package for %s failed to import: %s", module, exc)


def render_visualizations(
    module: str,
    reader: Any,
    config: dict,
    output_dir: Path,
) -> list[Path]:
    """Render every registered viz plugin for ``module`` into ``<output_dir>/viz/``.

    Returns the list of produced file paths (empty if the encoder is absent or no
    plugin is registered). Never raises on a plugin failure — each is isolated.
    """
    if not _viz_deps_available():
        logger.warning(INSTALL_HINT)
        print(INSTALL_HINT)
        return []

    _import_viz_package(module)
    plugins = get_viz_plugins(module)
    if not plugins:
        return []

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    produced: list[Path] = []
    for cls in plugins:
        try:
            paths = cls().render(reader, config, viz_dir) or []
            produced.extend(Path(p) for p in paths)
        except Exception as exc:
            logger.warning("viz plugin %s failed: %s", cls.__name__, exc)
            print(f"viz plugin {cls.__name__} failed: {exc}")
    return produced
