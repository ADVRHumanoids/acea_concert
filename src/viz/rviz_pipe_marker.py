#!/usr/bin/env python3
"""
Standalone RViz marker publisher for the weld pipe cylinder.
Run as a subprocess to avoid rclpy conflicts.

Usage: python3 rviz_pipe_marker.py <name> <cx> <cy> <cz> <radius> <length>
       <qx> <qy> <qz> <qw> [marker_id] [topic] [namespace] [r g b a]
"""
import sys
import rclpy
from visualization_msgs.msg import Marker
from rclpy.node import Node
import time


class PipeMarkerPublisher(Node):
    def __init__(self, name, cx, cy, cz, radius, length, qx, qy, qz, qw,
                 frame='world', color_r=1.0, color_g=0.5, color_b=0.0,
                 color_a=0.9, marker_id=0, topic=None, namespace=None):
        super().__init__(f'{name}_marker_publisher')

        self.name = name
        self.namespace = namespace or name
        self.marker_id = marker_id
        self.frame = frame
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.radius = radius
        self.length = length
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.color_a = color_a
        
        self.pub = self.create_publisher(Marker, topic or f'/{name}', 10)
        time.sleep(0.2)  # Give RViz time to process the delete

    def publish_once(self):
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.ns = self.namespace
        marker.id = self.marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = self.cx
        marker.pose.position.y = self.cy
        marker.pose.position.z = self.cz

        marker.pose.orientation.x = self.qx
        marker.pose.orientation.y = self.qy
        marker.pose.orientation.z = self.qz
        marker.pose.orientation.w = self.qw

        marker.scale.x = self.radius * 2.0
        marker.scale.y = self.radius * 2.0
        marker.scale.z = self.length

        marker.color.r = self.color_r
        marker.color.g = self.color_g
        marker.color.b = self.color_b
        marker.color.a = self.color_a

        marker.lifetime.sec = 0

        marker.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(marker)

    def publish_forever(self, rate=0.5):
        while rclpy.ok():
            self.publish_once()
            time.sleep(rate)

def main():
    name= sys.argv[1]
    cx, cy, cz = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    radius = float(sys.argv[5])
    length = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
    qx = float(sys.argv[7])
    qy = float(sys.argv[8])
    qz = float(sys.argv[9])
    qw = float(sys.argv[10])
    marker_id = int(sys.argv[11]) if len(sys.argv) > 11 else 0
    topic = sys.argv[12] if len(sys.argv) > 12 else None
    namespace = sys.argv[13] if len(sys.argv) > 13 else None
    color_r = float(sys.argv[14]) if len(sys.argv) > 14 else 1.0
    color_g = float(sys.argv[15]) if len(sys.argv) > 15 else 0.5
    color_b = float(sys.argv[16]) if len(sys.argv) > 16 else 0.0
    color_a = float(sys.argv[17]) if len(sys.argv) > 17 else 0.9

    rclpy.init()
    node = PipeMarkerPublisher(
        name, cx, cy, cz, radius, length, qx, qy, qz, qw,
        color_r=color_r, color_g=color_g, color_b=color_b, color_a=color_a,
        marker_id=marker_id, topic=topic, namespace=namespace)
    node.publish_forever()

if __name__ == '__main__':
    main()
