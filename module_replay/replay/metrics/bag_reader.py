"""Read-once, topic-config-driven bag reader (MTRC-01).

Opens an output bag exactly once via ``rosbags`` (pure offline Python, no ROS
runtime), deserializes only the topics in ``topic_list`` into a per-topic cache,
and serves them on demand without re-reading. ``iter_paired`` aligns two topics
by timestamp for input/output (latency, faithfulness) metrics.

INVARIANT (PATTERNS § Key invariants #3): this module imports rosbags only,
never the ROS runtime client library.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np


class BagReader:
    def __init__(self, bag_path: Path, topic_list: list[str]):
        self._bag_path = Path(bag_path)
        if not self._bag_path.exists():
            raise FileNotFoundError(f"Bag not found: {self._bag_path}")
        self._cache: dict[str, list[tuple[int, Any]]] = {}
        self._metadata: dict[str, Any] = {}
        self._read_once(topic_list)

    def _read_once(self, topic_list: list[str]) -> None:
        # rosbags 0.11.3: read connections, filter to wanted topics, deserialize
        # CDR via the same typestore the synthetic_bag writer used
        # (Stores.ROS2_HUMBLE -> typestore.serialize_cdr). reader.deserialize is
        # not used so we keep one explicit typestore on both write and read sides.
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore

        typestore = get_typestore(Stores.ROS2_HUMBLE)
        with Reader(self._bag_path) as reader:
            wanted = [c for c in reader.connections if c.topic in topic_list]
            for connection, timestamp, rawdata in reader.messages(connections=wanted):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                self._cache.setdefault(connection.topic, []).append((timestamp, msg))

    def get_messages(self, topic: str) -> list[tuple[int, Any]]:
        return self._cache.get(topic, [])

    def topics(self) -> list[str]:
        return list(self._cache.keys())

    def iter_paired(
        self,
        topic_a: str,
        topic_b: str,
        tolerance_ns: int = 100_000_000,
    ) -> Iterator[tuple[int, Any, Any]]:
        msgs_a = self._cache.get(topic_a, [])
        msgs_b = self._cache.get(topic_b, [])
        if not msgs_a or not msgs_b:
            return
        ts_b = np.array([t for t, _ in msgs_b])
        for ts_a, msg_a in msgs_a:
            idx = int(np.searchsorted(ts_b, ts_a))
            for candidate_idx in (idx - 1, idx):
                if 0 <= candidate_idx < len(ts_b) and abs(int(ts_b[candidate_idx]) - ts_a) <= tolerance_ns:
                    yield (ts_a, msg_a, msgs_b[candidate_idx][1])
                    break
