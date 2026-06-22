#!/usr/bin/env python3
"""Drive the omnisteering base to the optimized relative weld pose."""

import argparse
import math
import sys
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R


DEFAULT_MAT_FILE = Path(
    "/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat")


def _yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > max_norm and norm > 1e-9:
        return vector * (max_norm / norm)
    return vector


def _yaw_from_vector_xy(vector: np.ndarray) -> float:
    return math.atan2(float(vector[1]), float(vector[0]))


def _target_gap_pose_base(mat_file: Path) -> tuple[np.ndarray, float]:
    mat_data = loadmat(str(mat_file))
    robot_pose = np.asarray(
        mat_data["initial_robot_pose"], dtype=float).reshape(-1)
    gap_xyz_base = np.asarray(
        mat_data["pos_center_pipe_base"], dtype=float).reshape(-1)

    world_R_base = R.from_quat(robot_pose[3:7]).as_matrix()
    world_R_gap = np.column_stack([
        np.asarray(mat_data["pipe_x_axis_world"], dtype=float).reshape(3),
        np.asarray(mat_data["pipe_y_axis_world"], dtype=float).reshape(3),
        np.asarray(mat_data["pipe_z_axis_world"], dtype=float).reshape(3),
    ])
    base_R_gap = world_R_base.T @ world_R_gap
    return gap_xyz_base[:2], _yaw_from_vector_xy(base_R_gap[:2, 0])


class DriveBaseToWeldPose(Node):

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("drive_base_to_weld_pose")
        self._args = args
        self._measured_gap_pose: tuple[np.ndarray, float] | None = None
        self._done = False
        self.exit_code = 0
        self._start_time = perf_counter()

        self._target_gap_xy, self._target_gap_yaw = _target_gap_pose_base(
            args.mat_file)

        self._pub = self.create_publisher(Twist, args.topic, 10)
        self._gap_robot_sub = self.create_subscription(
            PoseStamped,
            args.gap_pose_robot_topic,
            self._on_gap_pose_robot,
            10,
        )
        self._timer = self.create_timer(1.0 / args.rate, self._tick)

        self.get_logger().info(
            "Driving base until /gap/pose_robot matches the optimized "
            "relative gap pose: "
            f"gap_xy_base=[{self._target_gap_xy[0]:+.3f}, "
            f"{self._target_gap_xy[1]:+.3f}], "
            f"gap_yaw_base={self._target_gap_yaw:+.3f} rad"
        )

    def _on_gap_pose_robot(self, msg: PoseStamped) -> None:
        quat = msg.pose.orientation
        gap_xy = np.array(
            [msg.pose.position.x, msg.pose.position.y], dtype=float)
        gap_yaw = _yaw_from_quat_xyzw(quat.x, quat.y, quat.z, quat.w)
        self._measured_gap_pose = (gap_xy, gap_yaw)

    def _tick(self) -> None:
        elapsed = perf_counter() - self._start_time
        if elapsed > self._args.timeout:
            self.get_logger().error(
                f"Timed out after {self._args.timeout:.1f}s before reaching "
                "the optimized weld pose"
            )
            self.exit_code = 1
            self._finish()
            return

        if self._measured_gap_pose is None:
            self.get_logger().warn(
                f"Waiting for {self._args.gap_pose_robot_topic}",
                throttle_duration_sec=2.0,
            )
            return

        gap_xy, gap_yaw = self._measured_gap_pose
        xy_error_base = gap_xy - self._target_gap_xy
        yaw_error = _wrap_to_pi(gap_yaw - self._target_gap_yaw)

        if (np.linalg.norm(xy_error_base) <= self._args.tolerance_xy
                and abs(yaw_error) <= self._args.tolerance_yaw):
            self.get_logger().info("Reached optimized relative weld pose.")
            self._finish()
            return

        linear_cmd = _clip_norm(
            self._args.kp_xy * xy_error_base,
            self._args.max_linear_speed,
        )
        yaw_cmd = float(np.clip(
            self._args.kp_yaw * yaw_error,
            -self._args.max_yaw_speed,
            self._args.max_yaw_speed,
        ))

        twist = Twist()
        twist.linear.x = float(linear_cmd[0])
        twist.linear.y = float(linear_cmd[1])
        twist.angular.z = yaw_cmd
        self._pub.publish(twist)

    def _publish_zero(self) -> None:
        self._pub.publish(Twist())

    def _publish_zero_burst(self) -> None:
        for _ in range(self._args.stop_ticks):
            self._publish_zero()
            sleep(1.0 / self._args.rate)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._publish_zero_burst()
        rclpy.shutdown()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the mobile base until the measured gap pose in base_link "
            "matches the optimized relative weld pose."
        )
    )
    parser.add_argument("--mat-file", type=Path, default=DEFAULT_MAT_FILE)
    parser.add_argument("--gap-pose-robot-topic", default="/gap/pose_robot",
                        help="Measured gap pose in base_link.")
    parser.add_argument("--topic", default="/omnisteering/cmd_vel")
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--kp-xy", type=float, default=0.8)
    parser.add_argument("--kp-yaw", type=float, default=1.5)
    parser.add_argument("--max-linear-speed", type=float, default=0.2)
    parser.add_argument("--max-yaw-speed", type=float, default=0.1)
    parser.add_argument("--tolerance-xy", type=float, default=0.01)
    parser.add_argument("--tolerance-yaw", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--stop-ticks", type=int, default=10)

    args = parser.parse_args(remove_ros_args(args=argv)[1:])
    if not args.mat_file.exists():
        parser.error(f"--mat-file does not exist: {args.mat_file}")
    if args.rate <= 0.0:
        parser.error("--rate must be positive")
    if args.max_linear_speed <= 0.0:
        parser.error("--max-linear-speed must be positive")
    if args.max_yaw_speed <= 0.0:
        parser.error("--max-yaw-speed must be positive")
    if args.tolerance_xy < 0.0:
        parser.error("--tolerance-xy must be non-negative")
    if args.tolerance_yaw < 0.0:
        parser.error("--tolerance-yaw must be non-negative")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.stop_ticks < 1:
        parser.error("--stop-ticks must be at least 1")
    return args


def main(argv=None) -> None:
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(raw_argv)
    rclpy.init(args=raw_argv)
    node = DriveBaseToWeldPose(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted; sending stop command.")
        node.exit_code = 130
    finally:
        if rclpy.ok():
            node._publish_zero_burst()
        exit_code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
