#!/usr/bin/env python3
"""Welding simulation with the wrist ``camera_F`` enabled and bridged.

The baseline ``weld_sim.launch.py`` remains the owner of Gazebo, XBot2, pipe
placement and MAT handling.  This wrapper swaps only the modular-description
generator for the camera-enabled validation copy and adds the three GZ-to-ROS
camera bridges required by the v10 detector.
"""

import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_BASE_LAUNCH_PATH = Path(__file__).resolve().with_name("weld_sim.launch.py")


def _load_base_launch_module():
    spec = importlib.util.spec_from_file_location(
        "acea_concert_weld_sim_base",
        _BASE_LAUNCH_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load baseline launch: {_BASE_LAUNCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BASE = _load_base_launch_module()
_BASE.MODULAR_DESCRIPTION = str(
    _BASE.PATH_TO_ACEA_CONCERT
    / "src"
    / "modular"
    / "concert_with_torch_v10_sim.py"
)

# Wrappers may change this before calling generate_launch_description() while
# the historical v10 validation keeps its original black-marker default.
DEFAULT_SPAWN_GAP_VISUAL_MARKER = "true"


_GAP_MARKER_SDF = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="weld_gap_v10_visual_marker">
    <static>true</static>
    <link name="gap_marker_link">
      <visual name="gap_marker_visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.0 0.0 0.0 1</ambient>
          <diffuse>0.0 0.0 0.0 1</diffuse>
          <specular>0.0 0.0 0.0 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def _spawn_gap_visual_marker(context, *args, **kwargs):
    """Render the seam appearance expected by the RGB junction frontend.

    The marker is visual-only and centred at the MAT-file gap pose. It supplies
    no pose or label to perception; Gazebo still generates the RGB-D pixels
    consumed by the normal detector pipeline.
    """
    if not _BASE._bool_launch_config(context, "spawn_gap_visual_marker"):
        return []

    _mat_file, data = _BASE._load_mat_file(context)
    pipe_center = _BASE._mat_vector(data, "pos_center_pipe")
    radius = _BASE._mat_scalar(data, "radius_pipe", 0.1)
    gap = float(LaunchConfiguration("gap_visual_marker_width_m").perform(context))
    if gap <= 0.0:
        gap = _BASE._mat_scalar(data, "pipe_gap", 0.01)

    if _BASE._bool_launch_config(context, "optimized_robot_pose"):
        spawn_x, spawn_y, yaw = _BASE._optimized_gap_pose_from_mat(data)
    else:
        spawn_x = _BASE._float_launch_config(context, "pipe_offset_x")
        spawn_y = _BASE._float_launch_config(context, "pipe_offset_y")
        yaw = _BASE._float_launch_config(context, "pipe_y_axis_yaw")

    marker_sdf = _GAP_MARKER_SDF.format(
        radius=radius + 0.001,
        length=max(0.002, gap),
    )
    return [
        TimerAction(
            period=2.1,
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    name="spawn_weld_gap_v10_visual_marker",
                    arguments=[
                        "-string", marker_sdf,
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
        ),
    ]


def generate_launch_description():
    base_description = _BASE.generate_launch_description()
    start_xbot_after_clock = str(
        _BASE.PATH_TO_ACEA_CONCERT / "scripts" / "start_xbot2_after_clock.sh"
    )
    default_xbot2_config = str(
        Path(get_package_share_directory("acea_concert"))
        / "config"
        / "xbot2_v10_sim.yaml"
    )

    # concert_gazebo starts Gazebo and xbot2-core concurrently.  With a cold
    # start both processes can create different instances of the shared
    # /gz_to_xbot2_time channel, leaving XBot frozen at its first tick.  Keep
    # the baseline launch in charge of Gazebo and descriptions, but start XBot
    # only after Gazebo has emitted a real clock sample.
    actions = [
        DeclareLaunchArgument("xbot2", default_value="true"),
        DeclareLaunchArgument(
            "start_delayed_xbot2",
            default_value=LaunchConfiguration("xbot2"),
            description="Start XBot2 after the Gazebo clock is active.",
        ),
        DeclareLaunchArgument(
            "delayed_xbot2_config",
            default_value=default_xbot2_config,
            description="XBot2 configuration used by the delayed sim process.",
        ),
        SetLaunchConfiguration("xbot2", "false"),
        *base_description.entities,
        DeclareLaunchArgument(
            "start_arm_camera_bridge",
            default_value="true",
            description=(
                "Bridge camera_F RGB, aligned depth and camera_info from "
                "Gazebo to ROS 2."
            ),
        ),
        DeclareLaunchArgument(
            "spawn_gap_visual_marker",
            default_value=DEFAULT_SPAWN_GAP_VISUAL_MARKER,
            description=(
                "Spawn a visual-only black seam at the MAT-file gap centre "
                "for the RGB junction frontend."
            ),
        ),
        DeclareLaunchArgument(
            "gap_visual_marker_width_m",
            default_value="-1.0",
            description="Visual seam width [m]; <=0 uses pipe_gap from the MAT file.",
        ),
        Node(
            condition=IfCondition(LaunchConfiguration("start_arm_camera_bridge")),
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="camera_f_v10_validation_bridge",
            arguments=[
                "/camera_F/image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/camera_F/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
                "/camera_F/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            ],
            remappings=[
                ("/camera_F/image", "/camera_F/color/image_raw"),
            ],
            output="screen",
        ),
        OpaqueFunction(function=_spawn_gap_visual_marker),
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("start_delayed_xbot2")),
            cmd=[
                "bash", start_xbot_after_clock,
                "--timeout", "120",
                "--settle", "1.0",
                "--config", LaunchConfiguration("delayed_xbot2_config"),
            ],
            output="screen",
        ),
    ]
    return LaunchDescription(actions)
