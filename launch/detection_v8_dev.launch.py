"""ACEA pipe-junction detection — plain v8 dev baseline."""

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
        DeclareLaunchArgument("junction_acceptance_mode", default_value="variant_a_rgb"),
        DeclareLaunchArgument("rgb_topic", default_value="/camera/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera/camera/aligned_depth_to_color/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera/color/camera_info"),
        DeclareLaunchArgument("sync_slop_s", default_value="1.0"),
        DeclareLaunchArgument("use_receive_time_for_sync", default_value="true"),
        DeclareLaunchArgument("stream_stale_s", default_value="2.0"),
        DeclareLaunchArgument("stale_subscription_reset_s", default_value="5.0"),
        DeclareLaunchArgument("camera_qos_reliability", default_value="best_effort"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),

        Node(
            package="acea_concert",
            executable="acea_pipe_junction_node_v8_dev.py",
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
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
        Node(
            package="acea_concert",
            executable="gap_pose_robot_node.py",
            name="gap_pose_robot_node",
            output="screen",
            parameters=[
                LaunchConfiguration("gap_pose_config"),
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
        ),
    ])
