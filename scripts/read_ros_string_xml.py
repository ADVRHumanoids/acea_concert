#!/usr/bin/env python3
"""Print one XML document received from a ``std_msgs/String`` topic.

This avoids passing robot descriptions through ``ros2 topic echo``.  The CLI
serializes the message as YAML, while XBot expects the first byte of the
captured command output to belong to the XML document.
"""

import argparse
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


class XmlReader(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("read_ros_string_xml")
        self.document: str | None = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.subscription = self.create_subscription(String, topic, self._on_xml, qos)

    def _on_xml(self, message: String) -> None:
        document = message.data.lstrip("\ufeff \t\r\n")
        if document:
            self.document = document


def _receive_document(args: argparse.Namespace) -> str | None:
    rclpy.init()
    node: XmlReader | None = None
    try:
        node = XmlReader(args.topic)
        deadline = time.monotonic() + max(0.0, args.timeout)
        while rclpy.ok() and node.document is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.document
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    args = _parse_args()

    # Fast DDS may write transport diagnostics directly to file descriptor 1.
    # XBot captures this program's stdout as XML, so hide every ROS lifecycle
    # message and restore stdout only after the node has shut down.
    stdout_fd = sys.stdout.fileno()
    sys.stdout.flush()
    saved_stdout_fd = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, stdout_fd)
    os.close(devnull_fd)
    try:
        document = _receive_document(args)
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)

    if document is None:
        print(
            f"read_ros_string_xml: timed out waiting for {args.topic}",
            file=sys.stderr,
        )
        return 1

    sys.stdout.write(document)
    if not document.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
