#!/usr/bin/env python3
"""
Launch the CONCERT welding robot (concert_with_torch) in Gazebo.
 
Usage:
    ros2 launch acea_concert weld_sim.launch.py
    ros2 launch acea_concert weld_sim.launch.py gui:=false
    ros2 launch acea_concert weld_sim.launch.py xbot2:=false
    ros2 launch acea_concert weld_sim.launch.py rviz:=true
    ros2 launch acea_concert weld_sim.launch.py use_prismatic_joint:=true
"""
 
import os
import math
from pathlib import Path
import sys
 
import scipy
 
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
MAT_FILE = PATH_TO_ACEA_CONCERT / "mat_files" / "weld_concert.mat"
 
if not MAT_FILE.exists():
  print(f"File not found: {MAT_FILE}")
  sys.exit(1)
 
# ── Pipe geometry & placement from weld_opt MAT file ────────────────────────
matdata = scipy.io.loadmat(str(MAT_FILE))
init_pos_robot = matdata['initial_robot_pose'][0]
pipe_center = matdata['pos_center_pipe'].reshape(3)
 
def _mat_scalar(name: str, default: float) -> float:
    if name not in matdata:
        print(
            f"[weld_sim] MAT file has no '{name}', using default {default}. "
            "Regenerate mat_files/weld_concert.mat with src/weld_opt.py to "
            "store this value explicitly."
        )
        return float(default)
    return float(matdata[name].reshape(-1)[0])
 
 
PIPE_RADIUS = _mat_scalar('radius_pipe', 0.1)
PIPE_TOTAL_LENGTH = _mat_scalar('length_pipe', 5.0)
# The current MAT file has no pipe_gap. Default to a 1 cm visible junction for
# Gazebo debugging; override with pipe_gap_m:=0.003 for the 3 mm target case.
PIPE_GAP = _mat_scalar('pipe_gap', 0.01)
PIPE_X = float(pipe_center[0])
PIPE_Y = float(pipe_center[1])
PIPE_Z = float(pipe_center[2])
 
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
GAP_MARKER_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="weld_gap_black_filler">
    <static>true</static>
    <link name="gap_black_filler_link">
      <visual name="gap_black_filler_visual">
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
BLACK_BACKDROP_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="weld_gap_black_backdrop">
    <static>true</static>
    <link name="black_backdrop_link">
      <visual name="black_backdrop_visual">
        <geometry>
          <box>
            <size>{thickness_x} {length_y} {height_z}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.0 0.0 0.0 1</ambient>
          <diffuse>0.0 0.0 0.0 1</diffuse>
          <specular>0.0 0.0 0.0 1</specular>
          <emissive>0.0 0.0 0.0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
# ─────────────────────────────────────────────────────────────────────────────
 
 
def _float_launch_config(context, name):
    return float(LaunchConfiguration(name).perform(context))
 
 
def _bool_launch_config(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ("1", "true", "yes", "on")
 
 
def _spawn_pipe_actions(context, *args, **kwargs):
    pipe_offset_x = _float_launch_config(context, "pipe_offset_x")
    pipe_offset_y = _float_launch_config(context, "pipe_offset_y")
    pipe_y_axis_yaw = _float_launch_config(context, "pipe_y_axis_yaw")
    pipe_radius = _float_launch_config(context, "pipe_radius_m")
    pipe_total_length = _float_launch_config(context, "pipe_total_length_m")
    pipe_gap = _float_launch_config(context, "pipe_gap_m")
    pipe_z = _float_launch_config(context, "pipe_z_m")
    gap_marker_width = _float_launch_config(context, "gap_visual_marker_width_m")
    if gap_marker_width <= 0.0:
        gap_marker_width = pipe_gap
    spawn_gap_marker = _bool_launch_config(context, "spawn_gap_visual_marker")
    spawn_black_backdrop = _bool_launch_config(context, "spawn_gap_black_backdrop")
    backdrop_offset = _float_launch_config(context, "gap_black_backdrop_offset_m")
    backdrop_length = _float_launch_config(context, "gap_black_backdrop_length_m")
    backdrop_height = _float_launch_config(context, "gap_black_backdrop_height_m")
    pipe_half_length = max(0.01, (pipe_total_length - pipe_gap) / 2.0)
    pipe_half_center_offset = pipe_half_length / 2.0 + pipe_gap / 2.0
    pipe_sdf_left = PIPE_SDF_TEMPLATE.format(
        name="weld_pipe_left",
        radius=pipe_radius,
        length=pipe_half_length,
    )
    pipe_sdf_right = PIPE_SDF_TEMPLATE.format(
        name="weld_pipe_right",
        radius=pipe_radius,
        length=pipe_half_length,
    )
    gap_marker_sdf = GAP_MARKER_SDF_TEMPLATE.format(
        radius=pipe_radius + 0.001,
        length=max(0.002, gap_marker_width),
    )
    black_backdrop_sdf = BLACK_BACKDROP_SDF_TEMPLATE.format(
        thickness_x=0.025,
        length_y=max(pipe_total_length + 1.0, backdrop_length),
        height_z=max(3.0 * pipe_radius, backdrop_height),
    )
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
    pipe_back_axis = (
        math.cos(pipe_y_axis_yaw),
        math.sin(pipe_y_axis_yaw),
    )
    pipe_offset_x = pipe_half_center_offset * pipe_y_axis[0]
    pipe_offset_y = pipe_half_center_offset * pipe_y_axis[1]
 
    actions = [
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
                        "-string", pipe_sdf_left,
                        "-x", f"{pipe_spawn_x + pipe_offset_x:.6f}",
                        "-y", f"{pipe_spawn_y + pipe_offset_y:.6f}",
                        "-z", f"{pipe_z:.6f}",
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
                        "-string", pipe_sdf_right,
                        "-x", f"{pipe_spawn_x - pipe_offset_x:.6f}",
                        "-y", f"{pipe_spawn_y - pipe_offset_y:.6f}",
                        "-z", f"{pipe_z:.6f}",
                        "-R", "1.5708",
                        "-P", "0.0",
                        "-Y", f"{pipe_y_axis_yaw:.6f}",
                    ],
                    output="screen",
                ),
            ],
        ),
    ]
    if spawn_black_backdrop:
        actions.append(
            TimerAction(
                period=2.05,
                actions=[
                    Node(
                        package="ros_gz_sim",
                        executable="create",
                        name="spawn_weld_gap_black_backdrop",
                        arguments=[
                            "-string", black_backdrop_sdf,
                            "-x", f"{pipe_spawn_x + (pipe_radius + backdrop_offset) * pipe_back_axis[0]:.6f}",
                            "-y", f"{pipe_spawn_y + (pipe_radius + backdrop_offset) * pipe_back_axis[1]:.6f}",
                            "-z", f"{pipe_z:.6f}",
                            "-R", "0.0",
                            "-P", "0.0",
                            "-Y", f"{pipe_y_axis_yaw:.6f}",
                        ],
                        output="screen",
                    ),
                ],
            )
        )
    if spawn_gap_marker:
        actions.append(
            TimerAction(
                period=2.1,
                actions=[
                    Node(
                        package="ros_gz_sim",
                        executable="create",
                        name="spawn_weld_gap_visual_marker",
                        arguments=[
                            "-string", gap_marker_sdf,
                            "-x", f"{pipe_spawn_x:.6f}",
                            "-y", f"{pipe_spawn_y:.6f}",
                            "-z", f"{pipe_z:.6f}",
                            "-R", "1.5708",
                            "-P", "0.0",
                            "-Y", f"{pipe_y_axis_yaw:.6f}",
                        ],
                        output="screen",
                    ),
                ],
            )
        )
    return actions
 
 
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
        get_package_share_directory("acea_concert"),
        "world",
        "empty_world_no_ros2_camera_system.sdf",
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
    prismatic_joint_arg = PythonExpression([
        "' --use-prismatic-joint' if '",
        LaunchConfiguration("use_prismatic_joint"),
        "' == 'true' else ''",
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
        'python3', ' ', MODULAR_DESCRIPTION, prismatic_joint_arg,
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
        DeclareLaunchArgument("use_prismatic_joint", default_value="false",
                              description="Use the prismatic cart block instead of the first yaw joint in concert_with_torch.py"),
        DeclareLaunchArgument("start_front_camera_bridges", default_value="true",
                              description="Start explicit ros_gz_bridge GZ->ROS bridges for the front D435i RGB/depth/camera_info topics"),
        DeclareLaunchArgument("publish_robot_state_tf", default_value="false",
                              description="Fallback-only: publish URDF fixed transforms if the main robot launch does not already provide base_link -> D435i camera frames"),
        DeclareLaunchArgument("pipe_radius_m", default_value=str(PIPE_RADIUS),
                              description="Pipe radius used for the spawned debug pipe [m]"),
        DeclareLaunchArgument("pipe_total_length_m", default_value=str(PIPE_TOTAL_LENGTH),
                              description="Total pipe length before splitting around the gap [m]"),
        DeclareLaunchArgument("pipe_gap_m", default_value=str(PIPE_GAP),
                              description="Visible gap between the two spawned pipe halves [m]"),
        DeclareLaunchArgument("pipe_z_m", default_value=str(PIPE_Z),
                              description="Pipe/gap center height used for the spawned debug pipe [m]"),
        DeclareLaunchArgument("spawn_gap_visual_marker", default_value="true",
                              description="Spawn a short black cylindrical mini-pipe inside the junction for RGB detector smoke tests"),
        DeclareLaunchArgument("gap_visual_marker_width_m", default_value="-1.0",
                              description="Length of the black junction filler along the pipe axis [m]; <=0 follows pipe_gap_m"),
        DeclareLaunchArgument("spawn_gap_black_backdrop", default_value="false",
                              description="Spawn a large black panel behind the pipe so the real gap appears dark in RGB"),
        DeclareLaunchArgument("gap_black_backdrop_offset_m", default_value="0.35",
                              description="Distance from the back pipe surface to the black backdrop [m]"),
        DeclareLaunchArgument("gap_black_backdrop_length_m", default_value="7.0",
                              description="Minimum black backdrop length along the pipe axis [m]"),
        DeclareLaunchArgument("gap_black_backdrop_height_m", default_value="2.0",
                              description="Minimum black backdrop height [m]"),
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
