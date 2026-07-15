#!/usr/bin/env python3
"""
Launch the CONCERT welding robot (concert_with_torch) in Gazebo.
 
Usage:
    ros2 launch acea_concert weld_sim.launch.py
    ros2 launch acea_concert weld_sim.launch.py gui:=false
    ros2 launch acea_concert weld_sim.launch.py xbot2:=false
    ros2 launch acea_concert weld_sim.launch.py rviz:=true
    ros2 launch acea_concert weld_sim.launch.py mat_file:=mat_files/weld_concert.mat optimized_robot_pose:=true
    ros2 launch acea_concert weld_sim.launch.py use_prismatic_joint:=true
"""
 
import os
import math
from pathlib import Path
 
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
 
 
PATH_TO_CONCERT_WS = Path("/home/user/concert_ws")
PATH_TO_ACEA_CONCERT = PATH_TO_CONCERT_WS / "src" / "acea_concert"
PATH_TO_MODULAR_PYTHON = PATH_TO_CONCERT_WS / "src" / "modular" / "src"
MODULAR_DESCRIPTION = str(PATH_TO_ACEA_CONCERT / "src" / "modular" / "concert_with_torch.py")
GZ_RESOURCE_PATH = str(PATH_TO_CONCERT_WS / "install" / "share")
DEFAULT_MAT_FILE = PATH_TO_ACEA_CONCERT / "mat_files" / "weld_concert.mat"
 
 
def _mat_vector(data, name, default=None):
    if name in data:
        return np.asarray(data[name], dtype=float).reshape(-1)
    if default is not None:
        return np.asarray(default, dtype=float)
    raise KeyError(f"MAT file is missing required field: {name}")


def _mat_scalar(data, name, default):
    if name in data:
        return float(np.asarray(data[name], dtype=float).reshape(-1)[0])
    return float(default)


def _resolve_mat_file(raw_path: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if not path.is_absolute():
        path = PATH_TO_ACEA_CONCERT / path
    return path


def _load_mat_file(context):
    mat_file = _resolve_mat_file(LaunchConfiguration("mat_file").perform(context))
    if not mat_file.exists():
        raise FileNotFoundError(f"File not found: {mat_file}")
    return mat_file, loadmat(str(mat_file))


def _optimized_gap_pose_from_mat(data):
    robot_pose = _mat_vector(data, "initial_robot_pose")
    if robot_pose.size < 7:
        raise ValueError("initial_robot_pose must contain xyz + quaternion xyzw")

    world_R_base = R.from_quat(robot_pose[3:7]).as_matrix()
    if "pos_center_pipe_base" in data:
        gap_base = _mat_vector(data, "pos_center_pipe_base")
    else:
        pipe_center_mat = _mat_vector(data, "pos_center_pipe")
        gap_base = world_R_base.T @ (pipe_center_mat - robot_pose[:3])

    world_R_gap = np.column_stack([
        _mat_vector(data, "pipe_x_axis_world", [1.0, 0.0, 0.0]),
        _mat_vector(data, "pipe_y_axis_world", [0.0, 1.0, 0.0]),
        _mat_vector(data, "pipe_z_axis_world", [0.0, 0.0, 1.0]),
    ])
    base_R_gap = world_R_base.T @ world_R_gap
    pipe_y_axis_yaw = math.atan2(
        float(base_R_gap[1, 0]),
        float(base_R_gap[0, 0]),
    )
    return float(gap_base[0]), float(gap_base[1]), pipe_y_axis_yaw


PIPE_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
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
</sdf>"""


def _pipe_geometry_from_mat(data):
    pipe_center = _mat_vector(data, "pos_center_pipe")
    pipe_radius = _mat_scalar(data, "radius_pipe", 0.1)
    pipe_total_length = _mat_scalar(data, "length_pipe", 5.0)
    pipe_gap = _mat_scalar(data, "pipe_gap", 0.01)
    pipe_half_length = max(0.01, (pipe_total_length - pipe_gap) / 2.0)
    return {
        "z": float(pipe_center[2]),
        "half_center_offset": pipe_half_length / 2.0 + pipe_gap / 2.0,
        "sdf_left": PIPE_SDF_TEMPLATE.format(
            name="weld_pipe_left",
            radius=pipe_radius,
            length=pipe_half_length,
        ),
        "sdf_right": PIPE_SDF_TEMPLATE.format(
            name="weld_pipe_right",
            radius=pipe_radius,
            length=pipe_half_length,
        ),
    }
 
 
def _float_launch_config(context, name):
    return float(LaunchConfiguration(name).perform(context))
 
 
def _bool_launch_config(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ("1", "true", "yes", "on")
 
 
def _spawn_pipe_actions(context, *args, **kwargs):
    mat_file, selected_matdata = _load_mat_file(context)
    pipe = _pipe_geometry_from_mat(selected_matdata)

    if _bool_launch_config(context, "optimized_robot_pose"):
        pipe_spawn_x, pipe_spawn_y, pipe_y_axis_yaw = (
            _optimized_gap_pose_from_mat(selected_matdata))
        print(
            "[weld_sim] Starting at optimized relative gap pose from "
            f"{mat_file}: x={pipe_spawn_x:.3f}, y={pipe_spawn_y:.3f}, "
            f"yaw={pipe_y_axis_yaw:.3f}"
        )
    else:
        pipe_spawn_x = _float_launch_config(context, "pipe_offset_x")
        pipe_spawn_y = _float_launch_config(context, "pipe_offset_y")
        pipe_y_axis_yaw = _float_launch_config(context, "pipe_y_axis_yaw")

    pipe_y_axis = (
        -math.sin(pipe_y_axis_yaw),
        math.cos(pipe_y_axis_yaw),
    )
    pipe_offset_x = pipe["half_center_offset"] * pipe_y_axis[0]
    pipe_offset_y = pipe["half_center_offset"] * pipe_y_axis[1]
 
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
                        "-x", f"{pipe_spawn_x + pipe_offset_x:.6f}",
                        "-y", f"{pipe_spawn_y + pipe_offset_y:.6f}",
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
                        "-x", f"{pipe_spawn_x - pipe_offset_x:.6f}",
                        "-y", f"{pipe_spawn_y - pipe_offset_y:.6f}",
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
    modular_python_env = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=str(PATH_TO_MODULAR_PYTHON) + ":" + os.environ.get("PYTHONPATH", ""),
    )
    # concert_gazebo's gui:=false path currently does not reliably add `-s`
    # before invoking gz sim. Passing the world file with `-s` keeps this
    # package runnable in headless Docker without modifying concert_gazebo.
    gazebo_world_file = os.path.join(
        get_package_share_directory("concert_gazebo"),
        "world",
        "empty_world.sdf",
    )
    gazebo_world_file_with_headless_flag = PythonExpression([
        "'",
        gazebo_world_file,
        " -s' if '",
        LaunchConfiguration("gui"),
        "' == 'false' else '",
        gazebo_world_file,
        "'",
    ])
    modular_description = PythonExpression([
        "'",
        MODULAR_DESCRIPTION,
        " --use-prismatic-joint' if '",
        LaunchConfiguration("use_prismatic_joint"),
        "' == 'true' else '",
        MODULAR_DESCRIPTION,
        "'",
    ])
    robot_description_tf = Command([
        'python3 ', modular_description,
        ' -o urdf -a gazebo_urdf:=false floating_base:=true',
        ' realsense:=', LaunchConfiguration('realsense'),
        ' velodyne:=', LaunchConfiguration('velodyne'),
        ' ultrasound:=false',
        ' use_gpu_ray:=false',
        ' -r modularbot_tf'
    ], on_stderr='ignore')
    robot_state_publisher_node = Node(
        condition=IfCondition(LaunchConfiguration("publish_robot_state_tf")),
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description_tf},
            {"use_sim_time": True},
        ],
    )
    front_camera_bridge_condition = IfCondition(PythonExpression([
        "'",
        LaunchConfiguration("realsense"),
        "' == 'true' and '",
        LaunchConfiguration("start_front_camera_bridges"),
        "' == 'true'",
    ]))
    front_camera_bridge_node = Node(
        condition=front_camera_bridge_condition,
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="d435i_front_camera_bridge",
        arguments=[
            "/D435i_camera_front/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/D435i_camera_front/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/D435i_camera_front/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        remappings=[
            ("/D435i_camera_front/image", "/D435i_camera_front/color/image_raw"),
        ],
        output="screen",
    )
 
    return LaunchDescription([
        gz_resource_env,
        modular_python_env,
 
        DeclareLaunchArgument("gui",       default_value="true",  description="Launch Gazebo GUI"),
        DeclareLaunchArgument("xbot2",     default_value="true",  description="Launch XBot2"),
        DeclareLaunchArgument("rviz",      default_value="false", description="Launch RViz"),
        DeclareLaunchArgument("realsense", default_value="true", description="Include RealSense"),
        DeclareLaunchArgument("velodyne",  default_value="false", description="Include Velodyne"),
        DeclareLaunchArgument("mat_file", default_value=str(DEFAULT_MAT_FILE),
                              description="Optimization MAT file used for optimized_robot_pose"),
        DeclareLaunchArgument("optimized_robot_pose", default_value="false",
                              description="Place the gap at the optimized pose relative to the robot from mat_file"),
        DeclareLaunchArgument("use_prismatic_joint", default_value="false",
                              description="Use the prismatic cart block instead of the first yaw joint in concert_with_torch.py"),
        DeclareLaunchArgument("start_front_camera_bridges", default_value="true",
                              description="Start explicit ros_gz_bridge GZ->ROS bridges for the front D435i RGB/depth/camera_info topics"),
        DeclareLaunchArgument("publish_robot_state_tf", default_value="false",
                              description="Fallback-only: publish URDF fixed transforms if the main robot launch does not already provide base_link -> D435i camera frames"),
        DeclareLaunchArgument("pipe_offset_x", default_value="2.0",
                              description="Pipe center X in the robot nominal start frame [m]"),
        DeclareLaunchArgument("pipe_offset_y", default_value="0.0",
                              description="Pipe center Y in the robot nominal start frame [m]"),
        DeclareLaunchArgument("pipe_y_axis_yaw", default_value="0.0",
                              description="Yaw of the pipe/gap Y axis from nominal +Y around world Z [rad]"),

        robot_state_publisher_node,
        front_camera_bridge_node,
 
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("concert_gazebo"), "launch", "modular.launch.py"
            )),
            launch_arguments={
                "modular_description": modular_description,
                "xbot2_gui":           "false",
                "gui":                 LaunchConfiguration("gui"),
                "xbot2":               LaunchConfiguration("xbot2"),
                "rviz":                LaunchConfiguration("rviz"),
                "realsense":           LaunchConfiguration("realsense"),
                "velodyne":            LaunchConfiguration("velodyne"),
                "world_file":          gazebo_world_file_with_headless_flag,
            }.items(),
        ),
 
        OpaqueFunction(function=_spawn_pipe_actions),
    ])
