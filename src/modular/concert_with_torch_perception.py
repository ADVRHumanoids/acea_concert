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

# Camera calibration from:
# docs/Useful Transformation matrix for the welding tool holder.pdf
#
# The PDF gives TOOL_BASE -> TOOL_TIP as +271.500 mm along Z, with no
# rotation, and TOOL_BASE -> REALSENSE_CENTRE as:
#
#   R = [[ 0,       0.342020,  0.939693],
#        [-1,       0,         0       ],
#        [ 0,      -0.939693,  0.342020]]
#   t = [-292.055, 0, 243.060] mm
#
# ``ee_E``/``ee_F`` is the functional TOOL_TIP frame.  The resulting optical
# centre relative to that frame is therefore [-292.055, 0, -28.440] mm.  The
# generic RealSense xacro does not attach its origin at the optical centre: it
# attaches ``camera_F_bottom_screw_frame`` and then offsets ``camera_F_link``
# by [0, 17.5, 12.5] mm.  Its final optical-frame rotation is also fixed at
# RPY [-90, 0, -90] deg.  Solving those two fixed xacro transforms gives the
# mount pose below: when the URDF chain is composed, camera_F_depth_optical_frame
# exactly recovers the PDF REALSENSE_CENTRE pose.
_TOOL_BASE_TO_TOOL_TIP_Z_M = 0.271500
_TOOL_BASE_TO_REALSENSE_CENTRE_XYZ_M = (-0.292055, 0.0, 0.243060)
_TOOL_TIP_TO_REALSENSE_CENTRE_XYZ_M = (
    _TOOL_BASE_TO_REALSENSE_CENTRE_XYZ_M[0],
    _TOOL_BASE_TO_REALSENSE_CENTRE_XYZ_M[1],
    _TOOL_BASE_TO_REALSENSE_CENTRE_XYZ_M[2] - _TOOL_BASE_TO_TOOL_TIP_Z_M,
)
_CAMERA_BODY_PITCH_RAD = math.radians(-20.0)
_BOTTOM_SCREW_TO_OPTICAL_CENTRE_XYZ_M = (0.0, 0.0175, 0.0125)
_PITCH_COS = math.cos(_CAMERA_BODY_PITCH_RAD)
_PITCH_SIN = math.sin(_CAMERA_BODY_PITCH_RAD)
_BOTTOM_SCREW_TO_OPTICAL_IN_TOOL_TIP_M = (
    _PITCH_SIN * _BOTTOM_SCREW_TO_OPTICAL_CENTRE_XYZ_M[2],
    _BOTTOM_SCREW_TO_OPTICAL_CENTRE_XYZ_M[1],
    _PITCH_COS * _BOTTOM_SCREW_TO_OPTICAL_CENTRE_XYZ_M[2],
)
CAMERA_XYZ = tuple(
    centre - offset
    for centre, offset in zip(
        _TOOL_TIP_TO_REALSENSE_CENTRE_XYZ_M,
        _BOTTOM_SCREW_TO_OPTICAL_IN_TOOL_TIP_M,
    )
)
CAMERA_RPY = (0.0, _CAMERA_BODY_PITCH_RAD, 0.0)
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
        parent_name="ee_F" if USE_PRISMATIC else "ee_E",
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
