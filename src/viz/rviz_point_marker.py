from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class PersistentPointSpawner(Node):
    def __init__(self, name, frame='world', color_r=1.0, color_g=0.0, color_b=0.0, color_a=1.0):
        super().__init__(f'{name}_marker_publisher')

        self.publisher = self.create_publisher(Marker, f'{name}', 10)

        # Create ONE persistent marker
        self.marker = Marker()
        self.marker.header.frame_id = frame

        self.marker.ns = "points"
        self.marker.id = 0
        self.marker.type = Marker.POINTS
        self.marker.action = Marker.ADD

        # Point size
        self.marker.scale.x = 0.02
        self.marker.scale.y = 0.02

        # Default color (red)
        self.marker.color.r = color_r
        self.marker.color.g = color_g
        self.marker.color.b = color_b
        self.marker.color.a = color_a

        # Store points
        self.marker.points = []

    def add_point(self, x, y, z):
        p = Point()
        p.x = x
        p.y = y
        p.z = z

        self.marker.points.append(p)

        # Update timestamp every publish
        self.marker.header.stamp = self.get_clock().now().to_msg()

        self.publisher.publish(self.marker)
        print(f'Published point: x={x}, y={y}, z={z}')
        print(f'Total points published so far: {len(self.marker.points)}')
        print('---')