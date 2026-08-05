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

    def _gap_orientation_from_base_y(self, gap_y_axis_base):
        gap_y_axis_base = np.asarray(gap_y_axis_base, dtype=float)
        xy_norm = np.linalg.norm(gap_y_axis_base[:2])
        if xy_norm < 1e-9:
            return 0.0
        gap_y_axis_base = gap_y_axis_base / np.linalg.norm(gap_y_axis_base)
        return np.arctan2(-gap_y_axis_base[0], gap_y_axis_base[1])

    def publish_controller_status(self, gap_y_axis_base, metrics):
        gap_orientation_angle = self._gap_orientation_from_base_y(
            gap_y_axis_base)

        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = 'ee_gap_controller'
        status.message = 'gap-frame tracking'
        status.hardware_id = 'concert'
        status.values = [
            self._kv('gap/orientation yaw from base y [deg]',
                     np.degrees(gap_orientation_angle)),
        ]
        status.values.extend(
            self._kv(key, value) for key, value in metrics.items()
        )

        msg = DiagnosticArray()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.status = [status]

        self._pub_ctrl_status.publish(msg)
