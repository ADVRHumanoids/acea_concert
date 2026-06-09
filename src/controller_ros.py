import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from utils.diagnostic import DiagnosticPlotter
from rcl_interfaces.srv import GetParameters


def fetch_robot_description(node_name: str):
    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node(node_name)
    client = node.create_client(
        GetParameters, '/robot_description_publisher/get_parameters')
    if not client.wait_for_service(timeout_sec=15.0):
        raise RuntimeError(
            "[controller] /robot_description_publisher not available. "
            "Is the simulation running?")

    req = GetParameters.Request()
    req.names = ['robot_description', 'robot_description_semantic']
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
    node.destroy_node()

    if future.result() is None:
        raise RuntimeError(
            "[controller] Failed to read robot_description parameters.")

    vals = future.result().values
    return vals[0].string_value, vals[1].string_value


class ControllerRosInterface:
    """ROS interface for the gap controller.

    Owns ROS parameters, gap subscriptions, PlotJuggler-friendly pose
    publishers, and diagnostics. The control loop should stay in controller.py.
    """

    def __init__(self, gain_defaults: dict[str, float],
                 node_name: str = 'ee_gap_controller'):
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node(node_name)
        self._gain_defaults = {
            name: float(value) for name, value in gain_defaults.items()
        }

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

        self._gap_pos_robot: np.ndarray | None = None
        self._gap_y_axis_robot: np.ndarray | None = None

        # FOR THE TIME BEING, GROUND TRUTH
        self.node.create_subscription(
            PoseStamped, '/gap/pose_robot', self._on_gap_pose_robot, 10)
        self.node.create_subscription(
            Vector3Stamped, '/gap/y_axis_robot', self._on_gap_y_axis_robot, 10)

        self._ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True)
        self._ros_thread.start()

    @property
    def gap_pos_robot(self) -> np.ndarray | None:
        if self._gap_pos_robot is None:
            return None
        return self._gap_pos_robot.copy()

    @property
    def gap_y_axis_robot(self) -> np.ndarray | None:
        if self._gap_y_axis_robot is None:
            return None
        return self._gap_y_axis_robot.copy()

    def controller_gains(self) -> dict[str, float]:
        return {
            name: self._float_param(name, default)
            for name, default in self._gain_defaults.items()
        }

    def robot_description(self, node_name: str = 'controller_urdf_reader'):
        return fetch_robot_description(node_name)

    def publish_controller_state(self, ee_pose_des, ee_pose_sent, ee_pose_ik,
                                 ee_pose_cur, frame_id: str,
                                 gap_y_axis_robot: np.ndarray,
                                 metrics: dict[str, float]):
        self._publish_pose(self._pub_des, ee_pose_des, frame_id)
        self._publish_pose(self._pub_sent, ee_pose_sent, frame_id)
        self._publish_pose(self._pub_ik, ee_pose_ik, frame_id)
        self._publish_pose(self._pub_cur, ee_pose_cur, frame_id)
        self.diagnostic_plotter.publish_controller_status(
            gap_y_axis_robot, metrics)

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
        self._gap_pos_robot = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=float)

    def _on_gap_y_axis_robot(self, msg: Vector3Stamped):
        axis = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        norm = np.linalg.norm(axis)
        if norm > 1e-9:
            self._gap_y_axis_robot = axis / norm

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
