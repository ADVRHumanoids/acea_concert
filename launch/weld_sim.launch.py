#!/usr/bin/env python3
"""
Launch the CONCERT welding robot (concert_with_torch) in Gazebo.

Usage:
    ros2 launch acea_concert weld_sim.launch.py
    ros2 launch acea_concert weld_sim.launch.py gui:=false
    ros2 launch acea_concert weld_sim.launch.py xbot2:=false
    ros2 launch acea_concert weld_sim.launch.py rviz:=true
"""

import os
import math
from pathlib import Path
import sys

import scipy

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PATH_TO_CONCERT_WS = Path("/home/user/concert_ws")
PATH_TO_ACEA_CONCERT = PATH_TO_CONCERT_WS / "src" / "acea_concert"
MODULAR_DESCRIPTION = str(PATH_TO_ACEA_CONCERT / "src" / "modular" / "concert_with_torch.py")
GZ_RESOURCE_PATH = str(PATH_TO_CONCERT_WS / "install" / "share")
MAT_FILE = PATH_TO_ACEA_CONCERT / "mat_files" / "weld_concert.mat"

if not MAT_FILE.exists():
  print(f"File not found: {MAT_FILE}")
  sys.exit(1)

# ── Pipe geometry & placement from weld_opt MAT file ────────────────────────
matdata = scipy.io.loadmat(str(MAT_FILE))
init_pos_robot = matdata['initial_robot_pose'][0]
pipe_center = matdata['pos_center_pipe'].reshape(3)

PIPE_RADIUS = float(matdata['radius_pipe'].reshape(-1)[0])
PIPE_TOTAL_LENGTH = float(matdata['length_pipe'].reshape(-1)[0])
PIPE_GAP = float(matdata['pipe_gap'].reshape(-1)[0])
PIPE_HALF_LENGTH = (PIPE_TOTAL_LENGTH - PIPE_GAP) / 2.0
PIPE_X = float(pipe_center[0])
PIPE_Y = float(pipe_center[1])
PIPE_Z = float(pipe_center[2])

# Derived: centre of each half = half_length/2 + half_gap.
_pipe_y = PIPE_HALF_LENGTH / 2.0 + PIPE_GAP / 2.0
# ─────────────────────────────────────────────────────────────────────────────
PIPE_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{{name}}">
    <static>true</static>
    <link name="pipe_link">
      <visual name="pipe_visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.4 0.4 0.4 1</ambient>
          <diffuse>0.6 0.6 0.6 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>
      <collision name="pipe_collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{length}</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>""".format(radius=PIPE_RADIUS, length=PIPE_HALF_LENGTH)

PIPE_SDF_LEFT  = PIPE_SDF_TEMPLATE.format(name="weld_pipe_left")
PIPE_SDF_RIGHT = PIPE_SDF_TEMPLATE.format(name="weld_pipe_right")


def _float_launch_config(context, name):
    return float(LaunchConfiguration(name).perform(context))


def _spawn_pipe_actions(context, *args, **kwargs):
    pipe_offset_x = _float_launch_config(context, "pipe_offset_x")
    pipe_offset_y = _float_launch_config(context, "pipe_offset_y")
    pipe_y_axis_yaw = _float_launch_config(context, "pipe_y_axis_yaw")

    # The sim robot is spawned at world XY = 0. Express the optimized pipe
    # position in this nominal robot-start frame.
    nominal_robot_x = PIPE_X - pipe_offset_x
    nominal_robot_y = PIPE_Y - pipe_offset_y
    pipe_spawn_x = PIPE_X - nominal_robot_x
    pipe_spawn_y = PIPE_Y - nominal_robot_y
    pipe_y_axis = (
        -math.sin(pipe_y_axis_yaw),
        math.cos(pipe_y_axis_yaw),
    )
    pipe_offset_x = _pipe_y * pipe_y_axis[0]
    pipe_offset_y = _pipe_y * pipe_y_axis[1]

    return [
        # Spawn two pipe halves with a small gap. With the default nominal frame
        # the robot starts centered 2 m from the pipe. pipe_y_axis_yaw rotates
        # the whole pipe/gap frame around vertical Z.
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    name="spawn_weld_pipe_left",
                    arguments=[
                        "-string", PIPE_SDF_LEFT,
                        "-x", f"{pipe_spawn_x + pipe_offset_x:.6f}",
                        "-y", f"{pipe_spawn_y + pipe_offset_y:.6f}",
                        "-z", str(PIPE_Z),
                        "-R", "1.5708",
                        "-P", "0.0",
                        "-Y", f"{pipe_y_axis_yaw:.6f}",
                    ],
                    output="screen",
                ),
            ],
        ),
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    name="spawn_weld_pipe_right",
                    arguments=[
                        "-string", PIPE_SDF_RIGHT,
                        "-x", f"{pipe_spawn_x - pipe_offset_x:.6f}",
                        "-y", f"{pipe_spawn_y - pipe_offset_y:.6f}",
                        "-z", str(PIPE_Z),
                        "-R", "1.5708",
                        "-P", "0.0",
                        "-Y", f"{pipe_y_axis_yaw:.6f}",
                    ],
                    output="screen",
                ),
            ],
        ),
    ]


def generate_launch_description():

    # Ensure Gazebo can find the meshes
    gz_resource_env = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=GZ_RESOURCE_PATH + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
    )

    return LaunchDescription([
        gz_resource_env,

        DeclareLaunchArgument("gui",       default_value="true",  description="Launch Gazebo GUI"),
        DeclareLaunchArgument("xbot2",     default_value="true",  description="Launch XBot2"),
        DeclareLaunchArgument("rviz",      default_value="false", description="Launch RViz"),
        DeclareLaunchArgument("realsense", default_value="true", description="Include RealSense"),
        DeclareLaunchArgument("velodyne",  default_value="false", description="Include Velodyne"),
        DeclareLaunchArgument("pipe_offset_x", default_value="2.0",
                              description="Pipe center X in the robot nominal start frame [m]"),
        DeclareLaunchArgument("pipe_offset_y", default_value="0.0",
                              description="Pipe center Y in the robot nominal start frame [m]"),
        DeclareLaunchArgument("pipe_y_axis_yaw", default_value="0.0",
                              description="Yaw of the pipe/gap Y axis from nominal +Y around world Z [rad]"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("concert_gazebo"), "launch", "modular.launch.py"
            )),
            launch_arguments={
                "modular_description": MODULAR_DESCRIPTION,
                "xbot2_gui":           "false",
                "gui":                 LaunchConfiguration("gui"),
                "xbot2":               LaunchConfiguration("xbot2"),
                "rviz":                LaunchConfiguration("rviz"),
                "realsense":           LaunchConfiguration("realsense"),
                "velodyne":            LaunchConfiguration("velodyne"),
            }.items(),
        ),

        OpaqueFunction(function=_spawn_pipe_actions),
    ])
