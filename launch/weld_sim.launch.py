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
from pathlib import Path
import sys

import scipy

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
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
        DeclareLaunchArgument("realsense", default_value="false", description="Include RealSense"),
        DeclareLaunchArgument("velodyne",  default_value="false", description="Include Velodyne"),

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

        # Spawn two pipe halves with a small gap
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    name="spawn_weld_pipe_left",
                    arguments=[
                        "-string", PIPE_SDF_LEFT,
                        "-x", str(PIPE_X - init_pos_robot[0]),
                        "-y", str(PIPE_Y + _pipe_y - init_pos_robot[1]),
                        "-z", str(PIPE_Z),
                        "-R", "1.5708",
                        "-P", "0.0",
                        "-Y", "0.0",
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
                        "-x", str(PIPE_X - init_pos_robot[0]),
                        "-y", str(PIPE_Y - _pipe_y - init_pos_robot[1]),
                        "-z", str(PIPE_Z),
                        "-R", "1.5708",
                        "-P", "0.0",
                        "-Y", "0.0",
                    ],
                    output="screen",
                ),
            ],
        ),
    ])
