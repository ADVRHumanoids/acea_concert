#!/usr/bin/env python3
"""V10 simulation with configurable, less detector-specific seam rendering.

This launch wraps ``weld_sim_v10_validation.launch.py`` and leaves that
historical baseline untouched. The detector remains geometry-first and uses
the simulation color-OFF preset; the orange pipe therefore changes image
formation without becoming an acquisition label.

Junction modes:
  physical_gap      two pipe halves with no additional visual geometry
                    (degenerate rendering control: no backdrop, no ambient
                    occlusion, ~0.1/255 contrast — not a realism test)
  inner_wall        physically plausible pipe interior: the inner wall at
                    pipe_radius - wall_thickness in dark neutral steel,
                    visible only through the physical gap (the repaired
                    realism test for the empty gap)
  dark_recess       a recessed dark cylindrical surface behind the gap
  collar            a slightly raised, dark-orange collar (default)
  tone_band         a flush dark-orange material transition
  legacy_black_band reproduce the old black surface marker
"""

import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_V10_LAUNCH_PATH = Path(__file__).resolve().with_name(
    "weld_sim_v10_validation.launch.py"
)


def _load_v10_launch_module():
    spec = importlib.util.spec_from_file_location(
        "acea_concert_weld_sim_v10_validation",
        _V10_LAUNCH_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load v10 validation launch: {_V10_LAUNCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_V10 = _load_v10_launch_module()
_V10.DEFAULT_SPAWN_GAP_VISUAL_MARKER = "false"
_BASE = _V10._BASE
# Keep the historical V10 launcher/generator untouched. The realistic V14
# wrapper uses the currently available official weld-torch resource through a
# dedicated generator copy.
_BASE.MODULAR_DESCRIPTION = str(
    _BASE.PATH_TO_ACEA_CONCERT
    / "src"
    / "modular"
    / "concert_with_torch_v14_sim.py"
)


_JUNCTION_SDF = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="junction_visual_link">
      <visual name="junction_visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{ambient}</ambient>
          <diffuse>{diffuse}</diffuse>
          <specular>{specular}</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


_BLACK = {
    "ambient": "0.0 0.0 0.0 1",
    "diffuse": "0.0 0.0 0.0 1",
    "specular": "0.0 0.0 0.0 1",
}


def _positive_launch_float(context, name):
    value = float(LaunchConfiguration(name).perform(context))
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, received {value}")
    return value


def _junction_pose(context, data):
    if _BASE._bool_launch_config(context, "optimized_robot_pose"):
        return _BASE._optimized_gap_pose_from_mat(data)
    return (
        _BASE._float_launch_config(context, "pipe_offset_x"),
        _BASE._float_launch_config(context, "pipe_offset_y"),
        _BASE._float_launch_config(context, "pipe_y_axis_yaw"),
    )


def _junction_material(context):
    return {
        "ambient": _BASE._rgba_launch_config(context, "junction_ambient_rgba"),
        "diffuse": _BASE._rgba_launch_config(context, "junction_diffuse_rgba"),
        "specular": _BASE._rgba_launch_config(context, "junction_specular_rgba"),
    }


def _spawn_realistic_junction(context, *args, **kwargs):
    mode = LaunchConfiguration("junction_visual_mode").perform(context).strip().lower()
    valid_modes = {
        "physical_gap",
        "inner_wall",
        "dark_recess",
        "collar",
        "tone_band",
        "legacy_black_band",
    }
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown junction_visual_mode={mode!r}; expected one of: "
            + ", ".join(sorted(valid_modes))
        )
    if mode == "physical_gap":
        print("[weld_sim_v10_realistic] junction mode=physical_gap (no visual insert)")
        return []

    mat_file, data = _BASE._load_mat_file(context)
    pipe_center = _BASE._mat_vector(data, "pos_center_pipe")
    pipe_radius = _BASE._mat_scalar(data, "radius_pipe", 0.1)
    physical_gap = _BASE._mat_scalar(data, "pipe_gap", 0.01)
    band_width = float(LaunchConfiguration("junction_band_width_m").perform(context))
    if band_width <= 0.0:
        band_width = physical_gap

    material = _junction_material(context)
    radius = pipe_radius
    length = max(0.002, band_width)

    if mode == "inner_wall":
        # Every real pipe has an interior. Through a 5 mm gap between two butt
        # sections the camera sees the inner wall: a surface one wall
        # thickness below the outer radius, in shadowed dark steel. This is
        # geometry that physically exists, not a seam marker: the cylinder
        # spans far beyond the gap and is occluded by the pipe halves
        # everywhere except through the gap itself. The material is a dark
        # neutral gray (a shadowed metal interior), deliberately NOT black and
        # NOT warm, so simulation detection stays color-independent.
        radius = max(
            0.001,
            pipe_radius - _positive_launch_float(context, "junction_wall_thickness_m"),
        )
        length = max(0.20, band_width)
        material = {
            "ambient": "0.02 0.02 0.02 1",
            "diffuse": "0.06 0.06 0.06 1",
            "specular": "0.04 0.04 0.04 1",
        }
    elif mode == "dark_recess":
        radius = max(
            0.001,
            pipe_radius - _positive_launch_float(context, "junction_recess_depth_m"),
        )
        material = _BLACK
    elif mode == "collar":
        radius = pipe_radius + _positive_launch_float(
            context, "junction_collar_height_m"
        )
        length = max(
            physical_gap,
            _positive_launch_float(context, "junction_collar_width_m"),
        )
    elif mode == "tone_band":
        # Keep the band nearly flush so the cue is mostly photometric rather
        # than a large, detector-friendly depth discontinuity.
        radius = pipe_radius + 0.0002
    elif mode == "legacy_black_band":
        radius = pipe_radius + 0.001
        material = _BLACK

    spawn_x, spawn_y, yaw = _junction_pose(context, data)
    sdf = _JUNCTION_SDF.format(
        model_name=f"weld_junction_v10_{mode}",
        radius=radius,
        length=length,
        **material,
    )
    print(
        "[weld_sim_v10_realistic] "
        f"mode={mode}, radius={radius:.4f} m, width={length:.4f} m, mat={mat_file}"
    )
    return [
        TimerAction(
            period=2.15,
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    name=f"spawn_weld_junction_v10_{mode}",
                    arguments=[
                        "-string", sdf,
                        "-x", f"{spawn_x:.6f}",
                        "-y", f"{spawn_y:.6f}",
                        "-z", f"{float(pipe_center[2]):.6f}",
                        "-R", "1.5708",
                        "-P", "0.0",
                        "-Y", f"{yaw:.6f}",
                    ],
                    output="screen",
                ),
            ],
        )
    ]


def generate_launch_description():
    baseline = _V10.generate_launch_description()
    return LaunchDescription([
        DeclareLaunchArgument(
            "pipe_visual_preset",
            default_value="painted_orange",
            description=(
                "Pipe material preset. painted_orange is appearance-only; "
                "the simulation detector keeps its color prior disabled."
            ),
        ),
        DeclareLaunchArgument(
            "junction_visual_mode",
            default_value="collar",
            description=(
                "physical_gap, inner_wall, dark_recess, collar, tone_band, "
                "or legacy_black_band."
            ),
        ),
        DeclareLaunchArgument(
            "junction_wall_thickness_m",
            default_value="0.008",
            description=(
                "inner_wall mode: pipe wall thickness [m]; the interior "
                "surface renders at pipe_radius minus this value."
            ),
        ),
        DeclareLaunchArgument(
            "junction_band_width_m",
            default_value="-1.0",
            description="Band/recess width [m]; <=0 follows pipe_gap from MAT.",
        ),
        DeclareLaunchArgument(
            "junction_collar_width_m",
            default_value="0.018",
            description="Axial width of the raised collar [m].",
        ),
        DeclareLaunchArgument(
            "junction_collar_height_m",
            default_value="0.002",
            description="Collar radial height above the pipe [m].",
        ),
        DeclareLaunchArgument(
            "junction_recess_depth_m",
            default_value="0.006",
            description="Dark-recess radial depth below the pipe surface [m].",
        ),
        DeclareLaunchArgument(
            "junction_ambient_rgba",
            default_value="0.08 0.015 0.003 1",
            description="Ambient RGBA of collar/tone-band modes.",
        ),
        DeclareLaunchArgument(
            "junction_diffuse_rgba",
            default_value="0.24 0.05 0.01 1",
            description="Diffuse RGBA of collar/tone-band modes.",
        ),
        DeclareLaunchArgument(
            "junction_specular_rgba",
            default_value="0.18 0.18 0.18 1",
            description="Specular RGBA of collar/tone-band modes.",
        ),
        *baseline.entities,
        OpaqueFunction(function=_spawn_realistic_junction),
    ])
