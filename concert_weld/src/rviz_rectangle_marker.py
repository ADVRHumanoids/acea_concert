#!/usr/bin/env python3
"""
Publish a transparent rectangle marker in RViz at a random XY pose.
"""
import sys
import time
from visualization_msgs.msg import Marker
from rclpy.node import Node
import rclpy


class RectangleMarkerPublisher(Node):
    def __init__(self, name, cx, cy, size_x, size_y, frame='world', color_r=0.0, color_g=0.5, color_b=1.0, color_a=0.3):
        super().__init__(f'{name}_marker_publisher')
        self.name = name
        self.cx = cx
        self.cy = cy
        self.size_x = size_x
        self.size_y = size_y
        self.frame = frame
        self.color_r = color_r
        self.color_g = color_g
        self.color_b = color_b
        self.color_a = color_a
        self.cz = 0.1  # Slightly above ground
        self.pub = self.create_publisher(Marker, f'/{name}', 10)

    def publish_once(self):
        marker = Marker()
        marker.header.frame_id = self.frame
        marker.ns = self.name
        marker.id = 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = self.cx
        marker.pose.position.y = self.cy
        marker.pose.position.z = self.cz
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.size_x
        marker.scale.y = self.size_y
        marker.scale.z = 0.01
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

    rclpy.init()
    rect_pub = RectangleMarkerPublisher(name, cx, cy, size_x, size_y, frame, color_r, color_g, color_b, color_a)
    rect_pub.publish_forever()

if __name__ == '__main__':
    main()
