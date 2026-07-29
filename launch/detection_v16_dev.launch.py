"""ACEA pipe-junction detection - V16 color-OFF runtime candidate.

Runs acea_pipe_junction_node_v16_dev.py and its frozen V14 pose
bridge, leaving V15 and all earlier detectors untouched.

Topic preset:
  use_sim_time:=true  -> /camera_F/{color/image_raw,depth_image,camera_info}
  use_sim_time:=false -> /camera/camera/{color/image_raw,aligned_depth_to_color/image_raw,color/camera_info}

Set rgb_topic/depth_topic/camera_info_topic explicitly to override either preset.
This launch does not start bridges. In simulation, start/bridge the camera with
the existing simulation stack, e.g. arm_camera_detection.launch.py or an
equivalent camera_F bridge.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from scipy.io import loadmat


DEFAULT_SOURCE_MAT_FILE = Path("/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat")


def _truthy(text: str) -> bool:
    return str(text).lower() in ("1", "true", "yes", "on")


def _resolve_topic(value: str, sim_value: str, real_value: str, use_sim_time: bool, camera_preset: str) -> str:
    if value and value.lower() != "auto":
        return value
    preset = camera_preset.lower()
    if preset == "sim" or (preset == "auto" and use_sim_time):
        return sim_value
    return real_value


def _mat_scalar(mat_file: str, name: str) -> float:
    matdata = loadmat(mat_file)
    if name not in matdata:
        raise RuntimeError(f"MAT file '{mat_file}' has no '{name}' field")
    return float(matdata[name].reshape(-1)[0])


def _auto_mat_file_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_mat_file = os.environ.get("ACEA_CONCERT_MAT_FILE")
    if env_mat_file:
        candidates.append(Path(env_mat_file))

    # First prefer the source workspace path used by weld_sim.launch.py, because
    # src/weld_opt.py regenerates this file without requiring a colcon rebuild.
    candidates.append(DEFAULT_SOURCE_MAT_FILE)

    # If this launch is run from source, this resolves to the live source tree.
    # If run from an installed package, it becomes the installed fallback copy.
    candidates.append(Path(__file__).resolve().parents[1] / "mat_files" / "weld_concert.mat")
    return candidates


def _resolve_mat_file(context) -> str:
    value = LaunchConfiguration("mat_file").perform(context).strip()
    if value and value.lower() != "auto":
        return value

    for candidate in _auto_mat_file_candidates():
        if candidate.exists():
            return str(candidate)
    searched = ", ".join(str(candidate) for candidate in _auto_mat_file_candidates())
    raise RuntimeError(
        "mat_file:=auto could not find weld_concert.mat. Generate it first with "
        "`ros2 run acea_concert weld_opt.py` or `python3 src/weld_opt.py`, then "
        "launch detection again. Searched: "
        f"{searched}"
    )


def _resolve_pipe_radius(context) -> float | None:
    value = LaunchConfiguration("pipe_radius_m").perform(context).strip()
    if not value or value.lower() in ("none", "off", "false", "yaml"):
        return None
    if value.lower() != "auto":
        return float(value)

    mat_file = _resolve_mat_file(context)
    radius = _mat_scalar(mat_file, "radius_pipe")
    print(f"[detection_v16_dev] pipe_radius_m={radius:.6g} from {mat_file}")
    return radius


def _resolve_detector_extra_config(context, use_sim_time: bool, camera_preset: str) -> str:
    value = LaunchConfiguration("detector_extra_config").perform(context).strip()
    if value and value.lower() != "auto":
        return value
    preset = camera_preset.lower()
    use_sim_preset = preset == "sim" or (preset == "auto" and use_sim_time)
    config_name = "detector_v14_station_sim.yaml" if use_sim_preset else "detector_v8_complete_real.yaml"
    return str(Path(get_package_share_directory("acea_concert")) / "config" / config_name)


def _resolve_gap_radial_mode(value: str, use_sim_time: bool, camera_preset: str) -> str:
    if value and value.lower() != "auto":
        return value
    preset = camera_preset.lower()
    use_sim_preset = preset == "sim" or (preset == "auto" and use_sim_time)
    return "horizontal_from_axis" if use_sim_preset else "measured"


def _resolve_sync_settings(
    sync_slop_value: str,
    receive_time_value: str,
    use_sim_time: bool,
    camera_preset: str,
) -> tuple[float, bool]:
    """Choose sensor-time synchronization for simulation only.

    A loaded live run paired RGB and depth frames 315 ms apart when arrival
    time was used, even though 95% of the recorded camera headers were exactly
    equal and the worst nearest-header delta was 33 ms.  Geometry from such a
    pair is not a current RGB-D measurement.  Real-camera defaults stay
    unchanged because their bridge/header behavior is covered separately.
    """
    preset = camera_preset.lower()
    use_sim_preset = preset == "sim" or (preset == "auto" and use_sim_time)
    if receive_time_value.strip().lower() == "auto":
        use_receive_time = not use_sim_preset
    else:
        use_receive_time = _truthy(receive_time_value)
    if sync_slop_value.strip().lower() == "auto":
        sync_slop_s = 0.040 if use_sim_preset else 1.0
    else:
        sync_slop_s = float(sync_slop_value)
    return sync_slop_s, use_receive_time


def _launch_nodes(context, *args, **kwargs):
    pkg = FindPackageShare("acea_concert")
    use_sim_time_text = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time = _truthy(use_sim_time_text)
    camera_preset = LaunchConfiguration("camera_preset").perform(context)
    sim_camera_name = LaunchConfiguration("sim_camera_name").perform(context)

    rgb = _resolve_topic(
        LaunchConfiguration("rgb_topic").perform(context),
        f"/{sim_camera_name}/color/image_raw",
        "/camera/camera/color/image_raw",
        use_sim_time,
        camera_preset,
    )
    depth = _resolve_topic(
        LaunchConfiguration("depth_topic").perform(context),
        f"/{sim_camera_name}/depth_image",
        "/camera/camera/aligned_depth_to_color/image_raw",
        use_sim_time,
        camera_preset,
    )
    info = _resolve_topic(
        LaunchConfiguration("camera_info_topic").perform(context),
        f"/{sim_camera_name}/camera_info",
        "/camera/camera/color/camera_info",
        use_sim_time,
        camera_preset,
    )
    qos_arg = LaunchConfiguration("camera_qos_reliability").perform(context)
    if qos_arg.lower() == "auto":
        qos_arg = "reliable" if use_sim_time or camera_preset.lower() == "sim" else "best_effort"

    detector_extra_config = _resolve_detector_extra_config(context, use_sim_time, camera_preset)
    use_sim_preset = camera_preset.lower() == "sim" or (
        camera_preset.lower() == "auto" and use_sim_time
    )
    detector_config = LaunchConfiguration("detector_config")
    detector_v10_config = LaunchConfiguration("detector_v10_config")
    detector_v15_config = LaunchConfiguration("detector_v15_config")
    detector_v15_sim_config = LaunchConfiguration("detector_v15_sim_config")
    detector_v16_config = LaunchConfiguration("detector_v16_config")
    detector_v16_sim_config = str(
        Path(get_package_share_directory("acea_concert")) / "config" / "detector_v16_sim.yaml"
    )
    if use_sim_preset:
        # Simulation-specific normal/HOLD thresholds must win over the generic
        # v10 defaults. Keep the validated real-camera precedence unchanged
        # until it is covered by a separate real-bag A/B.
        detector_yaml_parameters = [
            detector_config,
            detector_v10_config,
            detector_extra_config,
            detector_v15_config,
            detector_v15_sim_config,
            detector_v16_sim_config,
        ]
    else:
        detector_yaml_parameters = [
            detector_config,
            detector_extra_config,
            detector_v10_config,
            detector_v15_config,
            detector_v16_config,
        ]
    print(
        "[detection_v16_dev] detector YAML precedence: "
        + " -> ".join(str(value) for value in detector_yaml_parameters)
    )
    gap_radial_mode = _resolve_gap_radial_mode(
        LaunchConfiguration("gap_radial_orientation_mode").perform(context),
        use_sim_time,
        camera_preset,
    )
    sync_slop_s, use_receive_time_for_sync = _resolve_sync_settings(
        LaunchConfiguration("sync_slop_s").perform(context),
        LaunchConfiguration("use_receive_time_for_sync").perform(context),
        use_sim_time,
        camera_preset,
    )
    print(
        "[detection_v16_dev] camera sync: "
        f"source={'receive' if use_receive_time_for_sync else 'header'}, "
        f"slop={sync_slop_s:.3f}s"
    )
    pipe_radius = _resolve_pipe_radius(context)
    pipe_radius_params = [] if pipe_radius is None else [{"pipe_radius_m": pipe_radius}]

    detector = Node(
        package="acea_concert",
        executable="acea_pipe_junction_node_v16_dev.py",
        name="acea_pipe_junction_node",
        output="screen",
        parameters=detector_yaml_parameters + [
            {"junction_acceptance_mode": LaunchConfiguration("junction_acceptance_mode")},
            {"rgb_topic": rgb},
            {"depth_topic": depth},
            {"camera_info_topic": info},
            {"sync_slop_s": sync_slop_s},
            {"use_receive_time_for_sync": use_receive_time_for_sync},
            {"stream_stale_s": LaunchConfiguration("stream_stale_s")},
            {"stale_subscription_reset_s": LaunchConfiguration("stale_subscription_reset_s")},
            {"camera_qos_reliability": qos_arg},
            {"use_sim_time": use_sim_time},
        ] + pipe_radius_params,
    )
    return [
        detector,
        Node(
        package="acea_concert",
        executable="gap_pose_robot_node_v14_dev.py",
        name="gap_pose_robot_node",
        output="screen",
        parameters=[
            LaunchConfiguration("gap_pose_config"),
            {"use_sim_time": use_sim_time},
            {"radial_orientation_mode": gap_radial_mode},
            {"use_sensor_stamp_tf": use_sim_preset},
            {"preserve_input_stamp": use_sim_preset},
            {"dedicated_tf_listener": use_sim_preset},
            {"tf_bounded_latest_fallback": use_sim_preset},
            {"tf_fallback_max_age_s": 0.04},
        ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    pkg = FindPackageShare("acea_concert")

    return LaunchDescription([
        DeclareLaunchArgument(
            "detector_config",
            default_value=PathJoinSubstitution([pkg, "config", "detector.yaml"]),
            description="Base parameter YAML for the pipe-junction detector.",
        ),
        DeclareLaunchArgument(
            "detector_extra_config",
            default_value="auto",
            description=(
                "auto selects detector_v14_station_sim.yaml when use_sim_time/camera_preset "
                "selects sim, otherwise the unchanged detector_v8_complete_real.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "detector_v10_config",
            default_value=PathJoinSubstitution([pkg, "config", "detector_v10_dev.yaml"]),
            description=(
                "Generic v10 thresholds inherited by V15. The V14 simulation "
                "geometry preset and then the V15 continuity preset are applied "
                "after this file."
            ),
        ),
        DeclareLaunchArgument(
            "detector_v15_config",
            default_value=PathJoinSubstitution(
                [pkg, "config", "detector_v15_coloroff_dev.yaml"]
            ),
            description=(
                "V15 color-OFF continuity and fail-closed border lifecycle. "
                "Applied in both camera presets; the simulation-only overlay "
                "follows it only in simulation."
            ),
        ),
        DeclareLaunchArgument(
            "detector_v15_sim_config",
            default_value=PathJoinSubstitution(
                [pkg, "config", "detector_v15_coloroff_sim.yaml"]
            ),
            description=(
                "V15 simulation-only synchronized global-candidate recovery. "
                "Loaded last only by the simulation camera preset."
            ),
        ),
        DeclareLaunchArgument(
            "detector_v16_config",
            default_value=PathJoinSubstitution(
                [pkg, "config", "detector_v16_dev.yaml"]
            ),
            description=(
                "V16 runtime-only overrides, applied last in both real and "
                "simulation presets. V15 measurement and safety presets stay "
                "unchanged underneath it."
            ),
        ),
        DeclareLaunchArgument(
            "gap_pose_config",
            default_value=PathJoinSubstitution([pkg, "config", "gap_pose_robot.yaml"]),
            description="Parameter YAML for the gap_pose_robot publisher.",
        ),
        DeclareLaunchArgument(
            "gap_radial_orientation_mode",
            default_value="auto",
            description=(
                "Gap-frame roll convention. auto uses horizontal_from_axis in sim and "
                "the measured camera-facing radial on the real camera."
            ),
        ),
        DeclareLaunchArgument(
            "mat_file",
            default_value="auto",
            description="MAT file used to derive geometry parameters. auto prefers the regenerated source mat_files/weld_concert.mat.",
        ),
        DeclareLaunchArgument(
            "pipe_radius_m",
            default_value="auto",
            description="Detector pipe radius [m]. auto reads radius_pipe from mat_file; yaml leaves detector_config value unchanged.",
        ),
        DeclareLaunchArgument(
            "junction_acceptance_mode",
            default_value="variant_a_rgb",
            description="Detector acceptance mode: variant_a_rgb or rgb_temporal.",
        ),
        DeclareLaunchArgument(
            "camera_preset",
            default_value="auto",
            description="auto uses sim topics when use_sim_time=true and real RealSense topics otherwise; set sim or real to force.",
        ),
        DeclareLaunchArgument(
            "sim_camera_name",
            default_value="camera_F",
            description="Simulation camera topic prefix, matching arm_camera_detection.launch.py.",
        ),
        DeclareLaunchArgument(
            "rgb_topic",
            default_value="auto",
            description="RGB image topic override; auto follows camera_preset/use_sim_time.",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="auto",
            description="Depth image topic override; auto follows camera_preset/use_sim_time.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="auto",
            description="Camera info topic override; auto follows camera_preset/use_sim_time.",
        ),
        DeclareLaunchArgument(
            "sync_slop_s",
            default_value="auto",
            description=(
                "Maximum RGB/depth mismatch [s]. auto uses 0.040 on simulation "
                "sensor timestamps and preserves 1.0 on real receive time."
            ),
        ),
        DeclareLaunchArgument(
            "use_receive_time_for_sync",
            default_value="auto",
            description=(
                "auto uses camera header stamps in simulation and preserves "
                "receive-time synchronization for the real-camera preset."
            ),
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
            default_value="auto",
            description="Camera subscription QoS reliability: auto, best_effort, or reliable.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use Gazebo /clock. Set true in simulation; leave false for real cameras without /clock.",
        ),

        OpaqueFunction(function=_launch_nodes),
    ])
