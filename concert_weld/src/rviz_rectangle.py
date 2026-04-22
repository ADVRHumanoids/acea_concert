#!/usr/bin/env python3
"""
Publish a transparent rectangle marker in RViz at a random XY pose.
"""
import sys
import rclpy
from visualization_msgs.msg import Marker
import numpy as np
import time

def main():
    # Accept cx, cy, size_x, size_y as arguments if provided, else random
    name = sys.argv[1]
    cx = float(sys.argv[2])
    cy = float(sys.argv[3])
    size_x = float(sys.argv[4])
    size_y = float(sys.argv[5])
    frame = sys.argv[6] if len(sys.argv) > 6 else 'world'
    color_r = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    color_g = float(sys.argv[8]) if len(sys.argv) > 8 else 0.5
    color_b = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
    color_a = float(sys.argv[10]) if len(sys.argv) > 10 else 0.3

    cz = 0.1  # Slightly above ground

    rclpy.init()
    node = rclpy.create_node(f'{name}_marker_publisher')
    pub = node.create_publisher(Marker, f'/{name}', 10)
    time.sleep(0.2)

    marker = Marker()
    marker.header.frame_id = frame
    marker.ns = name
    marker.id = 1
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x = cx
    marker.pose.position.y = cy
    marker.pose.position.z = cz
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = size_x
    marker.scale.y = size_y
    marker.scale.z = 0.01  # Thin

    marker.color.r = color_r
    marker.color.g = color_g
    marker.color.b = color_b
    marker.color.a = color_a  # Transparent

    marker.lifetime.sec = 0

    while rclpy.ok():
        marker.header.stamp = node.get_clock().now().to_msg()
        pub.publish(marker)
        time.sleep(0.5)

if __name__ == '__main__':
    main()
