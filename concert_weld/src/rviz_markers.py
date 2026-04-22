#!/usr/bin/env python3
"""
Standalone RViz marker publisher for the weld pipe cylinder.
Run as a subprocess to avoid rclpy conflicts.

Usage: python3 rviz_markers.py <center_x> <center_y> <center_z> <radius> [length]
"""
import sys
import rclpy
from visualization_msgs.msg import Marker
import time


def main():
    cx, cy, cz = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    radius = float(sys.argv[4])
    length = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    # Orientation as quaternion (qx, qy, qz, qw)
    qx = float(sys.argv[6])
    qy = float(sys.argv[7])
    qz = float(sys.argv[8])
    qw = float(sys.argv[9])


    rclpy.init()
    node = rclpy.create_node('pipe_marker_publisher')
    pub = node.create_publisher(Marker, '/visualization_marker', 10)
    time.sleep(0.2)  # Give RViz time to process the delete

    marker = Marker()
    marker.header.frame_id = 'world'
    marker.ns = 'weld_pipe'
    marker.id = 0
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD

    marker.pose.position.x = cx
    marker.pose.position.y = cy
    marker.pose.position.z = cz

    marker.pose.orientation.x = qx
    marker.pose.orientation.y = qy
    marker.pose.orientation.z = qz
    marker.pose.orientation.w = qw

    marker.scale.x = radius * 2.0
    marker.scale.y = radius * 2.0
    marker.scale.z = length

    marker.color.r = 1.0
    marker.color.g = 0.5
    marker.color.b = 0.0
    marker.color.a = 0.9

    marker.lifetime.sec = 0

    # Keep publishing so RViz always sees it
    while rclpy.ok():
        marker.header.stamp = node.get_clock().now().to_msg()
        pub.publish(marker)
        time.sleep(0.5)


if __name__ == '__main__':
    main()
