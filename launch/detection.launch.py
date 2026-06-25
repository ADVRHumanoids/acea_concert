"""ACEA perception detection module.

Brings up the two nodes that turn a depth camera into the gap pose the rest of
acea_concert already consumes:

  1. acea_pipe_junction_node : RGB + depth + camera_info -> weld-seam gap geometry
                               (camera frame) + junction detection
  2. gap_pose_robot_node     : gap geometry -> /gap/pose_robot (PoseStamped,
                               base_link), a drop-in replacement for the Gazebo
                               ground-truth gap_pose_publisher.py

Usage:
    ros2 launch acea_concert detection.launch.py
    ros2 launch acea_concert detection.launch.py use_yolo_seg_frontend:=true

It does NOT start the simulator or the camera bridge; point the detector's
rgb/depth/camera_info topics at whatever publishes them (see config/detector.yaml).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("acea_concert")

    return LaunchDescription([
        DeclareLaunchArgument(
            "detector_config",
            default_value=PathJoinSubstitution([pkg, "config", "detector.yaml"]),
            description="Parameter YAML for the pipe-junction detector.",
        ),
        DeclareLaunchArgument(
            "gap_pose_config",
            default_value=PathJoinSubstitution([pkg, "config", "gap_pose_robot.yaml"]),
            description="Parameter YAML for the gap_pose_robot publisher.",
        ),
        DeclareLaunchArgument(
            "use_yolo_seg_frontend",
            default_value="false",
            description="Enable the YOLO-seg detector frontend (needs the YOLO venv).",
        ),

        Node(
            package="acea_concert",
            executable="acea_pipe_junction_node.py",
            name="acea_pipe_junction_node",
            output="screen",
            parameters=[
                LaunchConfiguration("detector_config"),
                {"use_yolo_seg_frontend": LaunchConfiguration("use_yolo_seg_frontend")},
            ],
        ),
        Node(
            package="acea_concert",
            executable="gap_pose_robot_node.py",
            name="gap_pose_robot_node",
            output="screen",
            parameters=[LaunchConfiguration("gap_pose_config")],
        ),
    ])
