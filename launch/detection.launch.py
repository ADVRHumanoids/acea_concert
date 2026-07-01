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
            "junction_acceptance_mode",
            default_value="variant_a_rgb",
            description="Detector acceptance mode: variant_a_rgb or rgb_temporal.",
        ),
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="/D435i_camera_front/color/image_raw",
            description="RGB image topic override.",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/D435i_camera_front/depth_image",
            description="Depth image topic override.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/D435i_camera_front/camera_info",
            description="Camera info topic override. If not published, nominal intrinsics fallback is used.",
        ),
        DeclareLaunchArgument(
            "sync_slop_s",
            default_value="1.0",
            description="Maximum RGB/depth receive-time mismatch accepted by the detector [s].",
        ),
        DeclareLaunchArgument(
            "use_receive_time_for_sync",
            default_value="true",
            description="Use node receive time instead of camera header stamps for RGB/depth sync.",
        ),
        DeclareLaunchArgument(
            "stream_stale_s",
            default_value="2.0",
            description="Mark a camera stream stale when no fresh message was received for this many seconds.",
        ),
        DeclareLaunchArgument(
            "stale_subscription_reset_s",
            default_value="5.0",
            description="Minimum interval before recreating stale camera subscriptions after a bridge restart.",
        ),
        DeclareLaunchArgument(
            "camera_qos_reliability",
            default_value="best_effort",
            description="Camera subscription QoS reliability: best_effort or reliable.",
        ),

        Node(
            package="acea_concert",
            executable="acea_pipe_junction_node.py",
            name="acea_pipe_junction_node",
            output="screen",
            parameters=[
                LaunchConfiguration("detector_config"),
                {"junction_acceptance_mode": LaunchConfiguration("junction_acceptance_mode")},
                {"rgb_topic": LaunchConfiguration("rgb_topic")},
                {"depth_topic": LaunchConfiguration("depth_topic")},
                {"camera_info_topic": LaunchConfiguration("camera_info_topic")},
                {"sync_slop_s": LaunchConfiguration("sync_slop_s")},
                {"use_receive_time_for_sync": LaunchConfiguration("use_receive_time_for_sync")},
                {"stream_stale_s": LaunchConfiguration("stream_stale_s")},
                {"stale_subscription_reset_s": LaunchConfiguration("stale_subscription_reset_s")},
                {"camera_qos_reliability": LaunchConfiguration("camera_qos_reliability")},
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
