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

    def publish_controller_status(
        self,
        vy_cmd,
        gap_normal_coord,
        ee_measured_normal_coord,
        gap_y_axis_robot,
        linear_correction_angle,
        sent_normal_coord,
        translation_tracking_normal,
        linear_tracking_angle,
    ):
        measured_gap_error = gap_normal_coord - ee_measured_normal_coord

        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = 'ee_gap_controller'
        status.message = 'gap-normal tracking'
        status.hardware_id = 'concert'
        status.values = [
            self._kv('plane/gap_target_m', gap_normal_coord),
            self._kv('plane/ee_actual_m', ee_measured_normal_coord),
            self._kv('plane/ee_sent_m', sent_normal_coord),
            self._kv('error/gap_to_ee_m', measured_gap_error),
            self._kv('tracking/position_error_m', translation_tracking_normal),
            self._kv('tracking/orientation_error_deg',
                     np.degrees(linear_tracking_angle)),
            self._kv('command/velocity_mps', vy_cmd),
            self._kv('command/orientation_correction_deg',
                     np.degrees(linear_correction_angle)),
        ]

        msg = DiagnosticArray()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.status = [status]

        self._pub_ctrl_status.publish(msg)
