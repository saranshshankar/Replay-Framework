"""Perception Tier-3 visualization plugins (TIER3-VIZ-DESIGN.md).

Importing this package self-registers each perception viz under ``"perception"``
via ``@register_viz``. It is imported LAZILY by ``replay.metrics.viz_runner`` only
on the viz path — never by the cheap metrics pipeline (bifurcation §2/§9), which is
why these modules (and the mp4 encoder they pull) stay out of a ``[metrics]`` install.

Both plugins are faithful V2 revamps of Aniket's PoC viz
(``origin/aniket/feat/module_wise_replay_poc:.../visualizations/``), rewired off the
PoC ``BagReader``/constants onto the framework ``BagReader`` + ``overlap.py`` geometry.
"""
from .overlap_video import OverlapVideo

__all__ = ["OverlapVideo"]
