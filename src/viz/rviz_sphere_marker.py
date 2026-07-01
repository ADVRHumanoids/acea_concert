#!/usr/bin/env python3

import sys
import time

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker


class SphereMarkerPublisher(Node):

    def __init__(self, name, frame, x, y, z, radius, marker_id=0,
                 topic=None, namespace=None, color_r=0.0, color_g=1.0,
                 color_b=0.2, color_a=0.35):
        super().__init__(f"{name}_marker_publisher")
        self.name = name
        self.frame = frame
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.marker_id = marker_id
        self.namespace = namespace or name
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.color_a = color_a
        self.pub = self.create_publisher(Marker, topic or f"/{name}", 10)
        time.sleep(0.2)

    def publish_once(self):
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.ns = self.namespace
        marker.id = self.marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.x
        marker.pose.position.y = self.y
        marker.pose.position.z = self.z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2.0 * self.radius
        marker.scale.y = 2.0 * self.radius
        marker.scale.z = 2.0 * self.radius
        marker.color.r = self.color_r
        marker.color.g = self.color_g
        marker.color.b = self.color_b
        marker.color.a = self.color_a
        marker.lifetime.sec = 0
        self.pub.publish(marker)

    def publish_forever(self, rate=0.02):
        while rclpy.ok():
            self.publish_once()
            time.sleep(rate)


def main():
    name = sys.argv[1]
    frame = sys.argv[2]
    x, y, z = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    radius = float(sys.argv[6])
    marker_id = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    topic = sys.argv[8] if len(sys.argv) > 8 else None
    namespace = sys.argv[9] if len(sys.argv) > 9 else None
    color_r = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
    color_g = float(sys.argv[11]) if len(sys.argv) > 11 else 1.0
    color_b = float(sys.argv[12]) if len(sys.argv) > 12 else 0.2
    color_a = float(sys.argv[13]) if len(sys.argv) > 13 else 0.35

    rclpy.init()
    node = SphereMarkerPublisher(
        name, frame, x, y, z, radius, marker_id, topic, namespace,
        color_r, color_g, color_b, color_a)
    node.publish_forever()


if __name__ == "__main__":
    main()
