import threading
from time import monotonic

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation as R

from utils.diagnostic import DiagnosticPlotter
from utils.gap_pose_filter import GapPoseLowPass
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String


def fetch_robot_description(
        node_name: str, timeout_sec: float = 15.0) -> tuple[str, str]:
    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node(node_name)

    robot_description = None
    robot_description_semantic = None

    def robot_description_callback(msg):
        nonlocal robot_description
        robot_description = msg.data

    def robot_description_semantic_callback(msg):
        nonlocal robot_description_semantic
        robot_description_semantic = msg.data

    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )

    node.create_subscription(
        String,
        "/xbotcore/robot_description",
        robot_description_callback,
        qos,
    )

    node.create_subscription(
        String,
        "/xbotcore/robot_description_semantic",
        robot_description_semantic_callback,
        qos,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = monotonic() + timeout_sec

    try:
        while rclpy.ok():
            if (robot_description is not None and
                    robot_description_semantic is not None):
                return robot_description, robot_description_semantic

            if monotonic() >= deadline:
                raise RuntimeError(
                    "Failed to receive robot description topics. "
                    "Check QoS compatibility and that /xbotcore is running."
                )
            executor.spin_once(timeout_sec=0.1)

        raise RuntimeError(
            "ROS shut down while reading robot description topics."
        )

    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()


class ControllerRosInterface:
    """ROS interface for the gap controller.

    Owns ROS parameters, gap subscriptions, PlotJuggler-friendly pose
    publishers, and diagnostics. The control loop should stay in controller.py.
    """

    def __init__(self, gain_defaults: dict[str, float],
                 gap_pose_filter_tau_s: float = 0.0,
                 gap_pose_filter_history_size: int = 1,
                 gap_pose_filter_max_position_jump_m: float = 0.0,
                 gap_pose_filter_max_angle_jump_deg: float = 0.0,
                 node_name: str = 'ee_gap_controller'):
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node(node_name)
        self._gain_defaults = {
            name: float(value) for name, value in gain_defaults.items()
        }
        self._gap_pose_filter = GapPoseLowPass(
            tau_s=gap_pose_filter_tau_s,
            history_size=gap_pose_filter_history_size,
            max_position_jump_m=gap_pose_filter_max_position_jump_m,
            max_angle_jump_deg=gap_pose_filter_max_angle_jump_deg,
        )

        self._pub_des = self.node.create_publisher(
            PoseStamped, '/ee/desired', 10)
        self._pub_sent = self.node.create_publisher(
            PoseStamped, '/ee/sent', 10)
        self._pub_cur = self.node.create_publisher(
            PoseStamped, '/ee/current', 10)
        self._pub_ik = self.node.create_publisher(
            PoseStamped, '/ee/ik', 10)
        self._pub_js = self.node.create_publisher(
            JointState, '/controller/joints', 10)

        self.diagnostic_plotter = DiagnosticPlotter(self.node)

        for param_name, default_value in self._gain_defaults.items():
            self.node.declare_parameter(param_name, default_value)

        self._gap_origin_base: np.ndarray | None = None
        self._gap_x_axis_base: np.ndarray | None = None
        self._gap_y_axis_base: np.ndarray | None = None
        self._gap_z_axis_base: np.ndarray | None = None
        self._last_gap_pose_time: float | None = None

        self.node.create_subscription(
            PoseStamped, '/gap/pose_robot', self._on_gap_pose_robot, 10)
        if self._gap_pose_filter.enabled:
            self.node.get_logger().info(
                "Filtering /gap/pose_robot with "
                f"tau={gap_pose_filter_tau_s:.3f}s, "
                f"history={gap_pose_filter_history_size}, "
                f"max_pos_jump={gap_pose_filter_max_position_jump_m:.4f}m, "
                f"max_angle_jump={gap_pose_filter_max_angle_jump_deg:.2f}deg")

        self._ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True)
        self._ros_thread.start()

    @property
    def gap_origin_base(self) -> np.ndarray | None:
        if self._gap_origin_base is None:
            return None
        return self._gap_origin_base.copy()

    @property
    def gap_y_axis_base(self) -> np.ndarray | None:
        if self._gap_y_axis_base is None:
            return None
        return self._gap_y_axis_base.copy()

    @property
    def gap_axes_base(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if (self._gap_x_axis_base is None
                or self._gap_y_axis_base is None
                or self._gap_z_axis_base is None):
            return None
        return (
            self._gap_x_axis_base.copy(),
            self._gap_y_axis_base.copy(),
            self._gap_z_axis_base.copy(),
        )

    def gap_pose_age_s(self) -> float | None:
        if self._last_gap_pose_time is None:
            return None
        return monotonic() - self._last_gap_pose_time

    def gap_pose_is_fresh(self, timeout_s: float) -> bool:
        age = self.gap_pose_age_s()
        return age is not None and age <= timeout_s

    def controller_gains(self) -> dict[str, float]:
        return {
            name: self._float_param(name, default)
            for name, default in self._gain_defaults.items()
        }

    def robot_description(self, node_name: str = 'controller_urdf_reader'):
        return fetch_robot_description(node_name)

    def publish_controller_state(self, ee_pose_des, ee_pose_sent, ee_pose_ik,
                                 ee_pose_cur, frame_id: str,
                                 gap_y_axis_base: np.ndarray,
                                 metrics: dict[str, float]):
        self._publish_pose(self._pub_des, ee_pose_des, frame_id)
        self._publish_pose(self._pub_sent, ee_pose_sent, frame_id)
        self._publish_pose(self._pub_ik, ee_pose_ik, frame_id)
        self._publish_pose(self._pub_cur, ee_pose_cur, frame_id)
        self.diagnostic_plotter.publish_controller_status(
            gap_y_axis_base, metrics)

    def _float_param(self, name: str, default: float) -> float:
        value = self.node.get_parameter(name).value
        try:
            value = float(value)
        except (TypeError, ValueError):
            self.node.get_logger().warn(
                f"Invalid value for parameter '{name}', using {default}",
                throttle_duration_sec=2.0,
            )
            return float(default)

        if not np.isfinite(value):
            self.node.get_logger().warn(
                f"Non-finite value for parameter '{name}', using {default}",
                throttle_duration_sec=2.0,
            )
            return float(default)
        return value

    def _on_gap_pose_robot(self, msg: PoseStamped):
        quat = msg.pose.orientation
        q = np.array([quat.x, quat.y, quat.z, quat.w], dtype=float)
        norm = np.linalg.norm(q)
        if norm <= 1e-9:
            return

        gap_origin_base = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)
        base_R_gap = R.from_quat(q / norm).as_matrix()
        now_s = monotonic()
        gap_origin_base, base_R_gap = self._gap_pose_filter.update(
            gap_origin_base, base_R_gap, now_s)

        self._gap_origin_base = gap_origin_base
        self._gap_x_axis_base = self._unit_axis(base_R_gap[:, 0])
        self._gap_y_axis_base = self._unit_axis(base_R_gap[:, 1])
        self._gap_z_axis_base = self._unit_axis(base_R_gap[:, 2])
        self._last_gap_pose_time = now_s

    def _unit_axis(self, axis: np.ndarray) -> np.ndarray | None:
        norm = np.linalg.norm(axis)
        if norm > 1e-9:
            return axis / norm
        return None

    def _publish_pose(self, pub, affine, frame_id: str):
        msg = PoseStamped()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        xyz = affine.translation
        q = affine.quaternion
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        pub.publish(msg)
