#!/usr/bin/env python3
"""VC1 detector + strict sensor-stamped gap-pose bridge -> /gap/pose_robot.

  ros2 launch acea_concert detection_vc1.launch.py             # robot (camera_E)
  ros2 launch acea_concert detection_vc1.launch.py sim:=true   # Gazebo (camera_F)

For a real bag replay add use_sim_time:=true and play the bag with --clock.
The only workpiece prior is pipe_diameter_m (default 0.20 m, DN200).
In simulation the detector also ignores objects in front of the pipe (the
welding torch on the camera arm).
"""
import json

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PRESETS = {
    "real": {"rgb": "/camera_E/color/image_raw",
             "depth": "/camera_E/aligned_depth_to_color/image_raw",
             "info": "/camera_E/color/camera_info",
             "qos": "best_effort", "use_sim_time": "false", "core": {}},
    "sim": {"rgb": "/camera_F/color/image_raw",
            "depth": "/camera_F/depth_image",
            "info": "/camera_F/camera_info",
            "qos": "reliable", "use_sim_time": "true", "core": {"occluder_mask": True}},
}


def _truthy(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _nodes(context):
    preset = LaunchConfiguration("camera_preset").perform(context).strip().lower()
    if not preset:
        preset = "sim" if _truthy(LaunchConfiguration("sim").perform(context)) else "real"
    if preset not in PRESETS:
        raise RuntimeError(f"camera_preset must be one of {sorted(PRESETS)}")
    cfg = PRESETS[preset]
    pick = lambda name, key: LaunchConfiguration(name).perform(context) or cfg[key]
    use_sim_time = _truthy(pick("use_sim_time", "use_sim_time"))
    detector = Node(
        package="acea_concert", executable="acea_junction_vc1_node.py",
        name="acea_pipe_junction_node", output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "rgb_topic": pick("rgb_topic", "rgb"),
            "depth_topic": pick("depth_topic", "depth"),
            "camera_info_topic": pick("camera_info_topic", "info"),
            "camera_qos_reliability": pick("camera_qos_reliability", "qos"),
            "pipe_diameter_m": float(LaunchConfiguration("pipe_diameter_m").perform(context)),
            "publish_rgb_overlay": _truthy(LaunchConfiguration("publish_rgb_overlay").perform(context)),
            "stateless": _truthy(LaunchConfiguration("stateless").perform(context)),
            # a string parameter: launch_ros would otherwise read the JSON as a YAML dict
            "core_params_json": ParameterValue(json.dumps(cfg["core"]), value_type=str),
        }],
    )
    bridge = Node(
        package="acea_concert", executable="gap_pose_robot_node_v14_dev.py",
        name="gap_pose_robot_node", output="screen",
        parameters=[LaunchConfiguration("gap_pose_config"), {
            "use_sim_time": use_sim_time,
            "use_tf": True, "tf_timeout_s": 0.02, "use_sensor_stamp_tf": True,
            "dedicated_tf_listener": True, "tf_bounded_latest_fallback": False,
            "preserve_input_stamp": True,            # /gap/pose_robot carries the CAMERA stamp
            "static_fallback_enabled": False, "allow_identity_static_fallback": False,
            "hold_last_pose": False,
            "axis_sign": -1.0, "radial_sign": 1.0,   # same camera sign convention as V17
            "radial_orientation_mode": "horizontal_from_axis",
            "use_reference_sign_canonicalization": False,
            "use_temporal_sign_continuity": False,
            "require_pose_valid": True, "require_metric_3d": True, "reject_assumed_depth": True,
        }],
    )
    out = [detector]
    if _truthy(LaunchConfiguration("start_gap_pose_bridge").perform(context)):
        out.append(bridge)
    return out


def generate_launch_description() -> LaunchDescription:
    from launch.substitutions import PathJoinSubstitution
    from launch_ros.substitutions import FindPackageShare
    share = FindPackageShare("acea_concert")
    return LaunchDescription([
        DeclareLaunchArgument("sim", default_value="false", description="Gazebo camera_F instead of camera_E"),
        DeclareLaunchArgument("camera_preset", default_value="", description="real | sim (default from sim)"),
        DeclareLaunchArgument("use_sim_time", default_value=""),
        DeclareLaunchArgument("rgb_topic", default_value=""),
        DeclareLaunchArgument("depth_topic", default_value=""),
        DeclareLaunchArgument("camera_info_topic", default_value=""),
        DeclareLaunchArgument("camera_qos_reliability", default_value=""),
        DeclareLaunchArgument("pipe_diameter_m", default_value="0.20"),
        # full-resolution debug image per frame (2.7 MB at 1280x720): enable only when viewing it
        DeclareLaunchArgument("publish_rgb_overlay", default_value="false"),
        DeclareLaunchArgument("stateless", default_value="false"),
        DeclareLaunchArgument("start_gap_pose_bridge", default_value="true"),
        DeclareLaunchArgument("gap_pose_config",
                              default_value=PathJoinSubstitution([share, "config", "gap_pose_robot.yaml"])),
        OpaqueFunction(function=_nodes),
    ])
