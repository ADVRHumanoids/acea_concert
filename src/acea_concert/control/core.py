"""ROS-free feedback law for following a weld path in the gap frame."""

import numpy as np


class GapFeedbackController:
    """Stateful PD correction for the gap normal and tangent axes."""

    def __init__(self, dt, max_normal_velocity, max_tangent_velocity):
        self.dt = float(dt)
        self.max_normal_velocity = float(max_normal_velocity)
        self.max_tangent_velocity = float(max_tangent_velocity)
        self._previous_normal_error = None
        self._previous_tangent_error = None

    def compute(
        self,
        *,
        postural_position,
        current_position,
        weld_position_gap,
        weld_rotation_gap,
        gap_origin_base,
        gap_rotation_base,
        gains,
        tangent_correction,
    ):
        """Return corrected position, rotation, and diagnostic metrics."""
        postural_position = np.asarray(postural_position, dtype=float)
        current_position = np.asarray(current_position, dtype=float)
        gap_origin_base = np.asarray(gap_origin_base, dtype=float)
        gap_rotation_base = np.asarray(gap_rotation_base, dtype=float)

        tangent_axis = gap_rotation_base[:, 0]
        normal_axis = gap_rotation_base[:, 1]
        weld_target_base = (
            gap_origin_base
            + gap_rotation_base @ np.asarray(weld_position_gap, dtype=float)
        )

        normal_error = float(
            np.dot(weld_target_base - current_position, normal_axis))
        tangent_error = float(
            np.dot(weld_target_base - current_position, tangent_axis))

        normal_error_rate = self._error_rate(
            normal_error, self._previous_normal_error)
        self._previous_normal_error = normal_error
        normal_velocity = float(np.clip(
            gains['kp_normal'] * normal_error
            + gains['kd_normal'] * normal_error_rate,
            -self.max_normal_velocity,
            self.max_normal_velocity,
        ))
        normal_delta = (
            np.dot(current_position, normal_axis)
            + normal_velocity * self.dt
            - np.dot(postural_position, normal_axis)
        )

        tangent_velocity = 0.0
        tangent_delta = 0.0
        if tangent_correction:
            tangent_error_rate = self._error_rate(
                tangent_error, self._previous_tangent_error)
            self._previous_tangent_error = tangent_error
            tangent_velocity = float(np.clip(
                gains['kp_tangent_x'] * tangent_error
                + gains['kd_tangent_x'] * tangent_error_rate,
                -self.max_tangent_velocity,
                self.max_tangent_velocity,
            ))
            tangent_delta = (
                np.dot(current_position, tangent_axis)
                + tangent_velocity * self.dt
                - np.dot(postural_position, tangent_axis)
            )

        corrected_position = (
            postural_position
            + normal_delta * normal_axis
            + tangent_delta * tangent_axis
        )
        corrected_rotation = (
            gap_rotation_base @ np.asarray(weld_rotation_gap, dtype=float)
        )
        metrics = {
            'normal/error_m': normal_error,
            'normal/correction_m': float(normal_delta),
            'normal/correction_velocity_mps': normal_velocity,
            'tangent/error_m': tangent_error,
            'tangent/correction_m': float(tangent_delta),
            'tangent/correction_velocity_mps': tangent_velocity,
        }
        return corrected_position, corrected_rotation, metrics

    def _error_rate(self, error, previous_error):
        if previous_error is None:
            return 0.0
        return (error - previous_error) / self.dt


def rotation_error_angle(target_rotation, current_rotation):
    """Return the angle between two rotation matrices in radians."""
    delta = (
        np.asarray(target_rotation, dtype=float)
        @ np.asarray(current_rotation, dtype=float).T
    )
    cos_angle = (np.trace(delta) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
