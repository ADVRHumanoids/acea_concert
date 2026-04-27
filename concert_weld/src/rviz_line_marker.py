#!/usr/bin/env python3
"""
Publish a line strip marker in RViz for a given set of 3D points.
Usage: python3 rviz_line_marker.py x1 y1 z1 x2 y2 z2 ... xn yn zn
"""
import sys
import rclpy
from visualization_msgs.msg import Marker
import time
from rclpy.node import Node
from geometry_msgs.msg import Point


class LineMarkerPublisher(Node):
    def __init__(self, name, points, frame='world', color_r=1.0, color_g=0.0, color_b=0.0, color_a=1.0):
        super().__init__(f'{name}_marker_publisher')
        self.name = name
        self.frame = frame
        self.points = points
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.color_a = color_a

        self.pub = self.create_publisher(Marker, name, 10)
        time.sleep(0.2)

    def publish_once(self):
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.ns = self.name
        marker.id = 100
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # Line width
        marker.color.r = self.color_r
        marker.color.g = self.color_g
        marker.color.b = self.color_b
        marker.color.a = self.color_a
        marker.lifetime.sec = 0
        marker.points = [Point(x=pt[0], y=pt[1], z=pt[2]) for pt in self.points]
        marker.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(marker)

    def publish_forever(self, rate=0.5):
        while rclpy.ok():
            self.publish_once()
            time.sleep(rate)

def main():
    # Require node name as first argument, no print statements
    if len(sys.argv) < 8:  # name + at least 2 points (6 floats)
        return
    name = sys.argv[1]
    point_args = sys.argv[2:]

    try:
        args = list(map(float, point_args))
    except Exception:
        return
    if len(args) % 3 != 0 or len(args) < 6:
        return
    points = [args[i:i+3] for i in range(0, len(args), 3)]

    rclpy.init()
    node = LineMarkerPublisher(name, points)
    node.publish_forever()

if __name__ == '__main__':
    main()
