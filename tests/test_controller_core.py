"""Small ROS-free check for the weld feedback controller."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from acea_concert.control.core import (
    GapFeedbackController,
    rotation_error_angle,
)


def main():
    zero_controller = GapFeedbackController(0.1, 0.5, 0.25)
    zero_position, _, zero_metrics = zero_controller.compute(
        postural_position=np.zeros(3),
        current_position=np.zeros(3),
        weld_position_gap=np.zeros(3),
        weld_rotation_gap=np.eye(3),
        gap_origin_base=np.zeros(3),
        gap_rotation_base=np.eye(3),
        gains={
            'kp_normal': 2.0,
            'kd_normal': 0.0,
            'kp_tangent_x': 2.0,
            'kd_tangent_x': 0.0,
        },
        tangent_correction=True,
    )
    assert np.allclose(zero_position, np.zeros(3))
    assert zero_metrics['normal/error_m'] == 0.0
    assert zero_metrics['tangent/error_m'] == 0.0

    controller = GapFeedbackController(
        dt=0.1,
        max_normal_velocity=0.5,
        max_tangent_velocity=0.25,
    )
    position, rotation, metrics = controller.compute(
        postural_position=np.zeros(3),
        current_position=np.zeros(3),
        weld_position_gap=np.array([1.0, 1.0, 0.0]),
        weld_rotation_gap=np.eye(3),
        gap_origin_base=np.zeros(3),
        gap_rotation_base=np.eye(3),
        gains={
            'kp_normal': 2.0,
            'kd_normal': 0.0,
            'kp_tangent_x': 2.0,
            'kd_tangent_x': 0.0,
        },
        tangent_correction=False,
    )

    assert np.allclose(position, [0.0, 0.05, 0.0])
    assert np.allclose(rotation, np.eye(3))
    assert metrics['normal/correction_velocity_mps'] == 0.5
    assert metrics['tangent/correction_m'] == 0.0
    assert rotation_error_angle(np.eye(3), np.eye(3)) == 0.0


if __name__ == "__main__":
    main()
