"""Shared pytest fixtures for the module_replay test suite.

Provides:
- ``synthetic_bag``: a real, rosbags-readable rosbag2 directory with typed
  messages on perception-shaped input/output topics. Downstream metrics tests
  (BagReader, perception plugins, faithfulness, report) read this with
  ``rosbags.rosbag2.Reader``.
- ``perception_spec``: a ``ModuleSpec`` for the perception module, shared across
  runner/metrics tests. Uses only the 6 existing ModuleSpec fields so it stays
  valid after plan 01-03 extends ModuleSpec with defaulted fields.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from replay.module_config import ModuleSpec


@pytest.fixture
def synthetic_bag(tmp_path: Path) -> Path:
    """Minimal rosbag2 directory with typed messages on a few perception-shaped topics.

    Readable by ``rosbags.rosbag2.Reader`` so BagReader / perception plugin tests
    can run against real CDR-encoded messages. Writes ~10 messages at ~10 Hz on
    one input topic (``.../image_raw``) and one output topic (``.../image_raw_sim``)
    so interval/latency/faithfulness metrics have data to compute on.
    """
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    bag_dir = tmp_path / "synthetic_bag"
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    Image = typestore.types["sensor_msgs/msg/Image"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]

    in_topic = "/perception_node/camera_0/image_raw"
    out_topic = "/perception_node/camera_0/image_raw_sim"

    # rosbags 0.11.3 Writer requires a keyword-only ``version``; VERSION_LATEST (9)
    # with the default SQLITE3 storage plugin produces a real readable .db3 bag.
    with Writer(bag_dir, version=Writer.VERSION_LATEST) as writer:
        in_conn = writer.add_connection(in_topic, Image.__msgtype__, typestore=typestore)
        out_conn = writer.add_connection(out_topic, Image.__msgtype__, typestore=typestore)
        # ~10 messages on each topic at ~10 Hz (100 ms spacing in ns). The output
        # topic is offset by 30 ms to give latency metrics a measurable lag.
        for i in range(10):
            ts = i * 100_000_000  # 100 ms in ns
            sec = ts // 1_000_000_000
            nanosec = ts % 1_000_000_000
            hdr = Header(
                stamp=Time(sec=int(sec), nanosec=int(nanosec)),
                frame_id="cam0",
            )
            msg = Image(
                header=hdr,
                height=2,
                width=2,
                encoding="rgb8",
                is_bigendian=0,
                step=6,
                # rosbags serializes uint8[] fields from a numpy array, not bytes.
                data=np.zeros(12, dtype=np.uint8),
            )
            writer.write(in_conn, ts, typestore.serialize_cdr(msg, Image.__msgtype__))
            writer.write(
                out_conn,
                ts + 30_000_000,
                typestore.serialize_cdr(msg, Image.__msgtype__),
            )

    return bag_dir


@pytest.fixture
def perception_spec() -> ModuleSpec:
    return ModuleSpec(
        name="perception",
        container="planner",
        colcon_package="realtime_perception",
        input_topics=["/lidar_front/points", "/camera_0/image_raw/compressed"],
        output_topics=["/perception/rgb", "/perception/depth"],
        launch_command="ros2 launch realtime_perception perception.launch.py mode:=sim",
    )
