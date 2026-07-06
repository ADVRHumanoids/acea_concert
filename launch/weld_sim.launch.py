#!/usr/bin/env python3
"""
Launch the CONCERT welding robot (concert_with_torch) in Gazebo.

Usage:
    ros2 launch acea_concert weld_sim.launch.py
    ros2 launch acea_concert weld_sim.launch.py gui:=false
    ros2 launch acea_concert weld_sim.launch.py xbot2:=false
    ros2 launch acea_concert weld_sim.launch.py rviz:=true
    ros2 launch acea_concert weld_sim.launch.py mat_file:=mat_files/weld_concert.mat optimized_robot_pose:=true
"""

import os
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R

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
DEFAULT_MAT_FILE = PATH_TO_ACEA_CONCERT / "mat_files" / "weld_concert.mat"


def _mat_vector(matdata, name, default=None):
    if name in matdata:
        return np.asarray(matdata[name], dtype=float).reshape(-1)
    if default is not None:
        return np.asarray(default, dtype=float)
    raise KeyError(f"MAT file is missing required field: {name}")


def _resolve_mat_file(raw_path: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if not path.is_absolute():
        path = PATH_TO_ACEA_CONCERT / path
    return path


def _load_mat_file(context):
    mat_file = _resolve_mat_file(LaunchConfiguration("mat_file").perform(context))
    if not mat_file.exists():
        raise FileNotFoundError(f"File not found: {mat_file}")
    return mat_file, scipy.io.loadmat(str(mat_file))


def _pipe_geometry_from_mat(matdata):
    pipe_center = _mat_vector(matdata, "pos_center_pipe")
    pipe_radius = float(_mat_vector(matdata, "radius_pipe")[0])
    pipe_total_length = float(_mat_vector(matdata, "length_pipe")[0])
    pipe_gap = float(_mat_vector(matdata, "pipe_gap")[0])
    pipe_half_length = (pipe_total_length - pipe_gap) / 2.0

    pipe_sdf_template = """<?xml version="1.0" ?>
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
</sdf>""".format(radius=pipe_radius, length=pipe_half_length)

    return {
        "x": float(pipe_center[0]),
        "y": float(pipe_center[1]),
        "z": float(pipe_center[2]),
        "half_center_offset": pipe_half_length / 2.0 + pipe_gap / 2.0,
        "sdf_left": pipe_sdf_template.format(name="weld_pipe_left"),
        "sdf_right": pipe_sdf_template.format(name="weld_pipe_right"),
    }


def _optimized_gap_pose_from_mat(matdata):
    robot_pose = _mat_vector(matdata, "initial_robot_pose")
    if robot_pose.size < 7:
        raise ValueError("initial_robot_pose must contain xyz + quaternion xyzw")

    world_R_base = R.from_quat(robot_pose[3:7]).as_matrix()
    if "pos_center_pipe_base" in matdata:
        gap_base = _mat_vector(matdata, "pos_center_pipe_base")
    else:
        pipe_center = _mat_vector(matdata, "pos_center_pipe")
        gap_base = world_R_base.T @ (pipe_center - robot_pose[:3])

    world_R_gap = np.column_stack([
        _mat_vector(matdata, "pipe_x_axis_world", [1.0, 0.0, 0.0]),
        _mat_vector(matdata, "pipe_y_axis_world", [0.0, 1.0, 0.0]),
        _mat_vector(matdata, "pipe_z_axis_world", [0.0, 0.0, 1.0]),
    ])
    base_R_gap = world_R_base.T @ world_R_gap
    pipe_y_axis_yaw = math.atan2(
        float(base_R_gap[1, 0]),
        float(base_R_gap[0, 0]),
    )
    return float(gap_base[0]), float(gap_base[1]), pipe_y_axis_yaw


def _float_launch_config(context, name):
    return float(LaunchConfiguration(name).perform(context))


def _bool_launch_config(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in (
        "1", "true", "yes", "on")


def _spawn_pipe_actions(context, *args, **kwargs):
    mat_file, matdata = _load_mat_file(context)
    pipe = _pipe_geometry_from_mat(matdata)

    if _bool_launch_config(context, "optimized_robot_pose"):
        pipe_spawn_x, pipe_spawn_y, pipe_y_axis_yaw = (
            _optimized_gap_pose_from_mat(matdata))
        print(
            "[weld_sim] Starting at optimized relative gap pose from "
            f"{mat_file}: x={pipe_spawn_x:.3f}, y={pipe_spawn_y:.3f}, "
            f"yaw={pipe_y_axis_yaw:.3f}"
        )
    else:
        pipe_offset_x = _float_launch_config(context, "pipe_offset_x")
        pipe_offset_y = _float_launch_config(context, "pipe_offset_y")
        pipe_y_axis_yaw = _float_launch_config(context, "pipe_y_axis_yaw")

        # The sim robot is spawned at world XY = 0. Express the optimized pipe
        # position in this nominal robot-start frame.
        nominal_robot_x = pipe["x"] - pipe_offset_x
        nominal_robot_y = pipe["y"] - pipe_offset_y
        pipe_spawn_x = pipe["x"] - nominal_robot_x
        pipe_spawn_y = pipe["y"] - nominal_robot_y

    pipe_y_axis = (
        -math.sin(pipe_y_axis_yaw),
        math.cos(pipe_y_axis_yaw),
    )
    half_offset_x = pipe["half_center_offset"] * pipe_y_axis[0]
    half_offset_y = pipe["half_center_offset"] * pipe_y_axis[1]

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
                        "-string", pipe["sdf_left"],
                        "-x", f"{pipe_spawn_x + half_offset_x:.6f}",
                        "-y", f"{pipe_spawn_y + half_offset_y:.6f}",
                        "-z", f"{pipe['z']:.6f}",
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
                        "-string", pipe["sdf_right"],
                        "-x", f"{pipe_spawn_x - half_offset_x:.6f}",
                        "-y", f"{pipe_spawn_y - half_offset_y:.6f}",
                        "-z", f"{pipe['z']:.6f}",
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
        DeclareLaunchArgument("mat_file", default_value=str(DEFAULT_MAT_FILE),
                              description="Optimization MAT file used for pipe geometry and optimized_robot_pose"),
        DeclareLaunchArgument("optimized_robot_pose", default_value="false",
                              description="Place the gap at the optimized pose relative to the robot from mat_file"),
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
