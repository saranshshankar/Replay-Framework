"""Tests for the read-once BagReader (MTRC-01).

Runs against the shared ``synthetic_bag`` fixture (plan 01-01 conftest), a real
rosbags-readable bag with one input topic (``.../image_raw``) and one output
topic (``.../image_raw_sim``). Mirrors the happy/error-path style of
test_data_manager.
"""
from __future__ import annotations

import pytest

from replay.metrics.bag_reader import BagReader

IN = "/perception_node/camera_0/image_raw"
OUT = "/perception_node/camera_0/image_raw_sim"


def test_reads_once(synthetic_bag):
    """MTRC-01: one pass; get_messages returns cached data; second call is the same object."""
    reader = BagReader(synthetic_bag, [IN])
    msgs = reader.get_messages(IN)
    assert len(msgs) > 0
    assert reader.get_messages(IN) is msgs    # cached, not re-read


def test_missing_bag_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        BagReader(tmp_path / "nope", [IN])


def test_iter_paired_aligns(synthetic_bag):
    reader = BagReader(synthetic_bag, [IN, OUT])
    pairs = list(reader.iter_paired(IN, OUT))
    assert len(pairs) > 0
