#!/usr/bin/env python3
"""Bring the ARM-mounted camera (camera_F on ee_F) online and point the
pipe-junction detector at it -- WITHOUT modifying weld_sim.launch.py or
detection.launch.py (non-invasive wrapper).

Background
----------
weld_sim.launch.py bridges only the BODY camera (/D435i_camera_front) and the
detector (detection.launch.py) defaults to it. The colleague's arm camera
`camera_F` (parent ee_F, defined in src/modular/concert_with_torch.py) needs two
things the running stack does not provide:
  (a) a GZ->ROS bridge for /camera_F/{image,depth_image,camera_info}
  (b) the detector repointed at those topics
This launch does both, so the camera that scans the junction is the one on the
arm (it moves WITH ee_F), not the body camera.

Prerequisite (user, on GPU): (re)start weld_sim so `camera_F` is actually in the
robot description and rendered by Gazebo, then verify:
    gz topic -l | grep camera_F          # expect /camera_F/image,/depth_image,/camera_info
    ros2 topic echo --once --field data /robot_description | grep -c camera_F   # > 0
If camera_F does not appear, the description was not regenerated with it
(see the 'future' module note) -- fix that first; the bridge alone cannot
invent the sensor.

Usage (run AFTER weld_sim is up, INSTEAD of the plain detection.launch.py):
    ros2 launch acea_concert arm_camera_detection.launch.py
    # if you bridge the camera elsewhere already:
    ros2 launch acea_concert arm_camera_detection.launch.py start_bridge:=false
    # different camera name / gz topic prefix:
    ros2 launch acea_concert arm_camera_detection.launch.py camera_name:=camera_F
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _bridge_and_detector(context, *args, **kwargs):
    cam = LaunchConfiguration("camera_name").perform(context)
    start_bridge = LaunchConfiguration("start_bridge").perform(context).lower() in (
        "1", "true", "yes", "on")
    detector_start_delay_s = float(LaunchConfiguration("detector_start_delay_s").perform(context))
    camera_qos_reliability = LaunchConfiguration("camera_qos_reliability").perform(context)
    rgb = f"/{cam}/color/image_raw"
    depth = f"/{cam}/depth_image"
    info = f"/{cam}/camera_info"

    actions = []
    if start_bridge:
        # Mirror of weld_sim's d435i_front_camera_bridge, but for the arm camera.
        # '[gz.msgs.*' = GZ -> ROS direction. The /<cam>/image gz topic is remapped
        # to /<cam>/color/image_raw so the detector's rgb_topic convention matches.
        actions.append(Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name=f"{cam}_arm_camera_bridge",
            arguments=[
                f"/{cam}/image@sensor_msgs/msg/Image[gz.msgs.Image",
                f"/{cam}/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
                f"/{cam}/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            ],
            remappings=[(f"/{cam}/image", rgb)],
            output="screen",
        ))

    pkg_share = get_package_share_directory("acea_concert")
    detector_config = os.path.join(pkg_share, "config", "detector.yaml")
    gap_pose_config = os.path.join(pkg_share, "config", "gap_pose_robot.yaml")

    detector_action = Node(
        package="acea_concert",
        executable="acea_pipe_junction_node.py",
        name="acea_pipe_junction_node",
        output="screen",
        parameters=[
            detector_config,
            {"use_yolo_seg_frontend": False},
            {"junction_acceptance_mode": "variant_a_rgb"},
            {"use_depth_gap_gate": True},
            {"rgb_topic": rgb},
            {"depth_topic": depth},
            {"camera_info_topic": info},
            {"camera_qos_reliability": camera_qos_reliability},
            {"sync_slop_s": 1.0},
            {"use_receive_time_for_sync": True},
            {"stream_stale_s": 2.0},
            {"stale_subscription_reset_s": 5.0},
        ],
    )
    if start_bridge and detector_start_delay_s > 0.0:
        actions.append(TimerAction(period=detector_start_delay_s, actions=[detector_action]))
    else:
        actions.append(detector_action)
    actions.append(Node(
        package="acea_concert",
        executable="gap_pose_robot_node.py",
        name="gap_pose_robot_node",
        output="screen",
        parameters=[gap_pose_config],
    ))
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "camera_name", default_value="camera_F",
            description="Arm camera name = gz topic prefix /<camera_name>/{image,depth_image,camera_info}."),
        DeclareLaunchArgument(
            "start_bridge", default_value="true",
            description="Start the GZ->ROS bridge for the arm camera (set false if it is bridged elsewhere)."),
        DeclareLaunchArgument(
            "detector_start_delay_s", default_value="3.0",
            description="Delay detector startup after the camera_F bridge so subscriptions see fresh RGB-D."),
        DeclareLaunchArgument(
            "camera_qos_reliability", default_value="reliable",
            description="camera_F bridge offers reliable Image QoS; use reliable subscriptions by default."),
        OpaqueFunction(function=_bridge_and_detector),
    ])
