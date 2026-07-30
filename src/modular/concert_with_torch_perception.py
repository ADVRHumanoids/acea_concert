#!/usr/bin/env python3
"""Run the canonical CONCERT generator with an opt-in wrist RGB-D camera.

The historical ``concert_with_torch.py`` remains untouched and is still used
by ``weld_sim.launch.py``.  This adapter is selected only by the perception
wrapper.  It patches the generator entrypoint in-process, adds ``camera_F`` to
the completed robot tree, then delegates all URDF/SRDF generation to the
canonical implementation.
"""

from __future__ import annotations

import runpy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from modular.URDF_writer import UrdfWriter


CANONICAL_GENERATOR = Path(__file__).resolve().with_name("concert_with_torch.py")
CAMERA_NAME = "camera_F"
CAMERA_XYZ = (0.05, 0.0, -0.20)
CAMERA_RPY = (3.141593, -1.4, 0.0)
USE_PRISMATIC = "--use-prismatic-joint" in sys.argv[1:]


def _add_wrist_camera(writer: UrdfWriter) -> None:
    ET.SubElement(
        writer.root,
        "xacro:include",
        filename="${MODULAR_PATH}/modular_data/urdf/concert.sensors.urdf.xacro",
    )
    camera = ET.SubElement(
        writer.root,
        "xacro:add_realsense_d_camera",
        name=CAMERA_NAME,
        parent_name="ee_F" if USE_PRISMATIC else "ee_E",
        add_gazebo_sensor="true",
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
