#!/usr/bin/env python3
"""Run the canonical CONCERT generator with an opt-in wrist RGB-D camera.

The historical ``concert_with_torch.py`` remains untouched and is still used
by ``weld_sim.launch.py``.  This adapter is selected only by the perception
wrapper.  It patches the generator entrypoint in-process, adds ``camera_F`` to
the completed robot tree, then delegates all URDF/SRDF generation to the
canonical implementation.
"""

from __future__ import annotations

import math
import runpy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from modular.URDF_writer import UrdfWriter


CANONICAL_GENERATOR = Path(__file__).resolve().with_name("concert_with_torch.py")
CAMERA_NAME = "camera_F"

# Camera pose from the welding-tool-holder CAD sheet, matrix 3, turned 180 deg
# about Z: the sheet's TOOL_BASE is our mounting flange seen the other way round.
# The real robot's URDF (recorded in acea_real_1) shows why - it carries the tool
# camera at (+0.1357, 0, 0.2427) from end_effector_E, on the positive X side,
# where the sheet puts it at negative X. Parent is the flange, not ee_E.
_TOOL_BASE_TO_REALSENSE_CENTRE_XYZ_M = (0.292055, 0.0, 0.243060)
_CAMERA_BODY_PITCH_RAD = math.radians(-20.0)
_TOOL_BASE_YAW_RAD = math.radians(180.0)
CAMERA_RPY = (0.0, _CAMERA_BODY_PITCH_RAD, _TOOL_BASE_YAW_RAD)

# The RealSense xacro anchors at its bottom screw, 17.5/12.5 mm off the optical
# centre, so take that back out through the full rotation - the pitch alone
# leaves the optical centre 22 mm out once the yaw is there.
_BOTTOM_SCREW_TO_OPTICAL_CENTRE_XYZ_M = (0.0, 0.0175, 0.0125)


def _rotate_by_camera_rpy(vector):
    """Rotate a camera-frame vector into the parent frame using CAMERA_RPY."""
    roll, pitch, yaw = CAMERA_RPY
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in rotation)


_BOTTOM_SCREW_TO_OPTICAL_IN_TOOL_BASE_M = _rotate_by_camera_rpy(
    _BOTTOM_SCREW_TO_OPTICAL_CENTRE_XYZ_M
)
CAMERA_XYZ = tuple(
    centre - offset
    for centre, offset in zip(
        _TOOL_BASE_TO_REALSENSE_CENTRE_XYZ_M,
        _BOTTOM_SCREW_TO_OPTICAL_IN_TOOL_BASE_M,
    )
)
USE_PRISMATIC = "--use-prismatic-joint" in sys.argv[1:]


def _add_wrist_camera(writer: UrdfWriter) -> None:
    ET.SubElement(
        writer.root,
        "xacro:include",
        filename="${MODULAR_PATH}/modular_data/urdf/concert.sensors.urdf.xacro",
    )
    camera = ET.SubElement(
        writer.root,
        "xacro:add_rgbd_camera",
        name=CAMERA_NAME,
        parent_name="end_effector_F" if USE_PRISMATIC else "end_effector_E",
        publish_tf="true",
        gazebo_urdf="true",
    )
    ET.SubElement(
        camera,
        "origin",
        xyz=" ".join(str(value) for value in CAMERA_XYZ),
        rpy=" ".join(str(value) for value in CAMERA_RPY),
    )
    writer.add_sensor_name("camera", CAMERA_NAME)


_parse_generator_args = UrdfWriter.parse_generator_cli_args
_write_file_to_stdout = UrdfWriter.write_file_to_stdout


def _parse_generator_args_with_local_flags(argv=None, known_only=False):
    """Let the canonical script consume its own prismatic-joint flag later."""
    parsed, unknown = _parse_generator_args(argv=argv, known_only=True)
    unexpected = [token for token in unknown if token != "--use-prismatic-joint"]
    if unexpected:
        raise SystemExit(
            "Unsupported generator arguments: " + " ".join(unexpected)
        )
    if known_only:
        return parsed, unknown
    return parsed


def _write_with_wrist_camera(
    writer,
    homing_map,
    robot_name="modularbot",
    args=None,
):
    _add_wrist_camera(writer)
    return _write_file_to_stdout(
        writer,
        homing_map,
        robot_name=robot_name,
        args=args,
    )


UrdfWriter.parse_generator_cli_args = staticmethod(
    _parse_generator_args_with_local_flags
)
UrdfWriter.write_file_to_stdout = _write_with_wrist_camera

runpy.run_path(str(CANONICAL_GENERATOR), run_name="__main__")
