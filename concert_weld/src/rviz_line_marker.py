#!/usr/bin/env python3
"""
Publish a line strip marker in RViz for a given set of 3D points.
Usage: python3 rviz_line_marker.py x1 y1 z1 x2 y2 z2 ... xn yn zn
"""
import sys
import rclpy
from visualization_msgs.msg import Marker
import time


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
    node = rclpy.create_node('line_marker_publisher')
    pub = node.create_publisher(Marker, name, 10)
    time.sleep(0.2)

    marker = Marker()
    marker.header.frame_id = 'world'
    marker.ns = name
    marker.id = 100
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.01  # Line width
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.lifetime.sec = 0

    marker.points = []
    from geometry_msgs.msg import Point
    for pt in points:
        p = Point()
        p.x, p.y, p.z = pt
        marker.points.append(p)

    while rclpy.ok():
        marker.header.stamp = node.get_clock().now().to_msg()
        pub.publish(marker)
        time.sleep(0.5)

if __name__ == '__main__':
    main()
