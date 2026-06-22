#!/usr/bin/env python3

import argparse
import sys
from time import perf_counter, sleep

import rclpy
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import Twist, TwistStamped


class OmnisteeringCommand(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__('omnisteering_command')
        self._args = args
        self._start_time = perf_counter()
        self._stopping_ticks = 0

        msg_type = TwistStamped if args.stamped else Twist
        self._pub = self.create_publisher(msg_type, args.topic, 10)
        self._timer = self.create_timer(1.0 / args.rate, self._tick)

        msg_name = 'TwistStamped' if args.stamped else 'Twist'
        self.get_logger().info(
            f'Publishing {msg_name} cmd_vel to {args.topic} at {args.rate:.1f} Hz for '
            f'{args.duration:.2f}s: vy={args.vy:.3f} m/s, '
            f'wz={args.wz:.3f} rad/s'
        )

    def _tick(self) -> None:
        if perf_counter() - self._start_time >= self._args.duration:
            self._publish_zero()
            self._stopping_ticks += 1
            if self._stopping_ticks >= self._args.stop_ticks:
                self.get_logger().info('Duration elapsed; sent stop command.')
                rclpy.shutdown()
            return

        twist = Twist()
        twist.linear.y = self._args.vy
        twist.angular.z = self._args.wz
        self._publish_twist(twist)

    def _publish_twist(self, twist: Twist) -> None:
        if self._args.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._args.frame_id
            msg.twist = twist
        else:
            msg = twist

        self._pub.publish(msg)

    def _publish_zero(self) -> None:
        self._publish_twist(Twist())

    def publish_zero_burst(self) -> None:
        for _ in range(self._args.stop_ticks):
            self._publish_zero()
            rclpy.spin_once(self, timeout_sec=0.0)
            sleep(1.0 / self._args.rate)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Publish a constant lateral velocity and yaw velocity to omnisteering.'
    )
    parser.add_argument('--topic', default='/omnisteering/cmd_vel')
    parser.add_argument('--rate', type=float, default=50.0)
    parser.add_argument('--duration', type=float, required=True,
                        help='Seconds to publish the command.')
    parser.add_argument('--vy', type=float, default=0.0,
                        help='Linear y velocity amplitude [m/s].')
    parser.add_argument('--wz', type=float, default=0.0,
                        help='Yaw velocity amplitude [rad/s].')
    parser.add_argument('--stamped', action='store_true',
                        help='Publish geometry_msgs/TwistStamped instead of Twist.')
    parser.add_argument('--frame-id', default='base_link',
                        help='Frame id used only with --stamped.')
    parser.add_argument('--stop-ticks', type=int, default=5,
                        help='Number of zero commands to publish on exit.')

    args = parser.parse_args(remove_ros_args(args=argv)[1:])
    if args.rate <= 0.0:
        parser.error('--rate must be positive')
    if args.duration <= 0.0:
        parser.error('--duration must be positive')
    if args.stop_ticks < 1:
        parser.error('--stop-ticks must be at least 1')
    if args.vy == 0.0 and args.wz == 0.0:
        parser.error('at least one of --vy or --wz must be non-zero')
    return args


def main(argv=None) -> None:
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(raw_argv)

    rclpy.init(args=raw_argv)
    node = OmnisteeringCommand(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted; sending stop command.')
    finally:
        if rclpy.ok():
            node.publish_zero_burst()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
