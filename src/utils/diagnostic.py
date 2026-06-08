import numpy as np
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node

CONTROLLER_DIAGNOSTICS_TOPIC = '/ee/controller/diagnostics'


class DiagnosticPlotter:

    def __init__(self, node: Node):
        self.node = node
        self._pub_ctrl_status = self.node.create_publisher(
            DiagnosticArray, CONTROLLER_DIAGNOSTICS_TOPIC, 10)

    def _kv(self, key: str, value: float) -> KeyValue:
        value = float(value)
        if not np.isfinite(value):
            value = 0.0
        return KeyValue(key=key, value=f"{value:.9g}")

    def _gap_orientation_from_base_y(self, gap_y_axis_robot):
        gap_y_axis_robot = np.asarray(gap_y_axis_robot, dtype=float)
        xy_norm = np.linalg.norm(gap_y_axis_robot[:2])
        if xy_norm < 1e-9:
            return 0.0
        gap_y_axis_robot = gap_y_axis_robot / np.linalg.norm(gap_y_axis_robot)
        return np.arctan2(-gap_y_axis_robot[0], gap_y_axis_robot[1])

    def publish_controller_status(self, gap_y_axis_robot, metrics):
        gap_orientation_angle = self._gap_orientation_from_base_y(
            gap_y_axis_robot)

        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = 'ee_gap_controller'
        status.message = 'gap-frame tracking'
        status.hardware_id = 'concert'
        status.values = [
            self._kv('gap/orientation_from_base_y_deg',
                     np.degrees(gap_orientation_angle)),
        ]
        status.values.extend(
            self._kv(key, value) for key, value in metrics.items()
        )

        msg = DiagnosticArray()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.status = [status]

        self._pub_ctrl_status.publish(msg)
