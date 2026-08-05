#!/usr/bin/env python3
"""Plan a compact, collision-checked path to the weld trajectory start.

Usage while weld_sim.launch.py is running:
    ros2 run acea_concert plan_homing_from_mat.py \
        --initial-pose-from-gazebo \
        --output /tmp/weld_concert_compact.mat --duration 8
"""

import argparse
import select
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from time import monotonic, sleep

import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
import numpy as np
from scipy.io import loadmat, savemat

from acea_concert.homing.trajectory import (
    DEFAULT_ACCEL_WEIGHT,
    DEFAULT_CARTESIAN_MOTION_WEIGHT,
    DEFAULT_COMPACT_WEIGHT,
    DEFAULT_INFLATION_RADIUS,
    DEFAULT_INITIAL_GUESS_MODE,
    DEFAULT_IPOPT_PRINT_LEVEL,
    DEFAULT_MAX_JOINT_ACCEL_STEP,
    DEFAULT_MAX_JOINT_STEP,
    DEFAULT_MOTION_WEIGHT,
    DEFAULT_TOOL_APPROACH_NODES,
    _pipe_halves,
    model_homing_geometry,
    plan_homing_trajectory,
)


PATH_TO_ACEA_CONCERT = Path("/home/user/concert_ws/src/acea_concert")
DEFAULT_MAT_FILE = PATH_TO_ACEA_CONCERT / "mat_files" / "weld_concert.mat"

# Planner tuning: edit these values here, not on the command line.
PIPE_CLEARANCE = 0.0  # Extra exact collision clearance around each pipe [m].
PIPE_INFLATION_RADIUS = DEFAULT_INFLATION_RADIUS  # Guide-frame margin [m].
TOOL_APPROACH_NODES = DEFAULT_TOOL_APPROACH_NODES  # Relax tool clearance near goal.
JOINT_MOTION_WEIGHT = DEFAULT_MOTION_WEIGHT  # Penalize joint travel.
JOINT_ACCEL_WEIGHT = DEFAULT_ACCEL_WEIGHT  # Penalize joint acceleration.
COMPACT_WEIGHT = DEFAULT_COMPACT_WEIGHT  # Pull links toward the shoulder.
CARTESIAN_MOTION_WEIGHT = DEFAULT_CARTESIAN_MOTION_WEIGHT  # Penalize link travel.
MAX_JOINT_STEP = DEFAULT_MAX_JOINT_STEP  # Per-node joint change [rad].
MAX_JOINT_ACCEL_STEP = DEFAULT_MAX_JOINT_ACCEL_STEP  # Joint second difference.
IPOPT_PRINT_LEVEL = DEFAULT_IPOPT_PRINT_LEVEL  # 0 is quiet; 5 shows iterations.
INITIAL_GUESS_MODE = DEFAULT_INITIAL_GUESS_MODE  # linear or goal-after-start.


def _scalar(data, key):
    return float(np.asarray(data[key]).reshape(-1)[0])


def _vector(data, key, size):
    value = np.asarray(data[key], dtype=float).reshape(-1)
    if value.size != size:
        raise ValueError(f"{key} must have {size} values, got {value.size}")
    return value


def _make_robot_description():
    modular = PATH_TO_ACEA_CONCERT / "src" / "modular" / "concert_with_torch.py"
    urdf = subprocess.check_output(
        ["python3", str(modular), "-o", "urdf"], text=True)
    srdf = subprocess.check_output(
        ["python3", str(modular), "-o", "srdf"], text=True)

    urdf_path = Path("/tmp/concert_weld_homing.urdf")
    srdf_path = Path("/tmp/concert_weld_homing.srdf")
    urdf_path.write_text(urdf)
    srdf_path.write_text(srdf)
    subprocess.run([
        "moveit_compute_default_collisions",
        "--urdf_path",
        str(urdf_path),
        "--srdf_path",
        str(srdf_path),
    ], check=True)
    return urdf, srdf_path.read_text()


def _homing_start_q(kin_dyn, q_goal, srdf):
    q_start = np.asarray(q_goal, dtype=float).reshape(-1).copy()
    root = ET.fromstring(srdf)
    home_state = next(
        (
            state for state in root.findall("group_state")
            if state.attrib.get("name") == "home"
        ),
        None,
    )
    if home_state is None:
        raise ValueError("Current SRDF has no 'home' group state")

    applied = []
    for joint in home_state.findall("joint"):
        name = joint.attrib.get("name")
        if name is None or kin_dyn.joint_nq(name) != 1:
            continue
        idx = kin_dyn.joint_iq(name)
        if idx >= q_start.size:
            continue
        q_start[idx] = float(joint.attrib["value"])
        applied.append(name)
    if not applied:
        raise ValueError("No SRDF home joints exist in the current model")
    return q_start


def _gazebo_scene_pose(timeout_s):
    from gap_pose_publisher import (
        GAP_MODEL_LEFT,
        GAP_MODEL_RIGHT,
        GZ_POSE_TOPIC,
        ROBOT_BASE_LINK,
        ROBOT_MODEL,
        _parse_pose_v,
    )

    def scene_from_poses(poses):
        robot_name = next(
            (
                name for name in (ROBOT_BASE_LINK, ROBOT_MODEL)
                if name in poses
            ),
            None,
        )
        if (
            robot_name is None
            or GAP_MODEL_LEFT not in poses
            or GAP_MODEL_RIGHT not in poses
        ):
            return None

        robot_xyz, robot_quat = poses[robot_name]
        left_xyz, left_quat = poses[GAP_MODEL_LEFT]
        right_xyz, _ = poses[GAP_MODEL_RIGHT]
        return (
            np.concatenate([robot_xyz, robot_quat]),
            0.5 * (left_xyz + right_xyz),
            left_quat,
        )

    deadline = monotonic() + timeout_s
    proc = subprocess.Popen(
        ["gz", "topic", "-e", "-t", GZ_POSE_TOPIC],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    buffer = []
    try:
        while monotonic() < deadline:
            remaining = deadline - monotonic()
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                break

            line = proc.stdout.readline()
            if not line:
                break

            if line.startswith("header {") and len(buffer) > 1:
                poses = _parse_pose_v("".join(buffer))
                scene = scene_from_poses(poses)
                if scene is not None:
                    return scene
                buffer = [line]
            else:
                buffer.append(line)

        poses = _parse_pose_v("".join(buffer))
        scene = scene_from_poses(poses)
        if scene is not None:
            return scene
    finally:
        proc.terminate()

    raise RuntimeError(
        "Could not read robot and both pipe poses from Gazebo within "
        f"{timeout_s:.1f}s"
    )


def _xbot_joint_positions(timeout_s):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from xbot_msgs.msg import JointState as XbotJointState

    if not rclpy.ok():
        rclpy.init(args=None)

    node = rclpy.create_node("homing_xbot_state_reader")
    joint_positions = {}

    def on_joint_state(msg):
        joint_positions.update(zip(
            [str(name).strip() for name in msg.name],
            [float(value) for value in msg.link_position],
        ))

    sub = node.create_subscription(
        XbotJointState,
        "/xbotcore/joint_states",
        on_joint_state,
        qos_profile_sensor_data,
    )

    deadline = monotonic() + timeout_s
    try:
        while monotonic() < deadline and not joint_positions:
            rclpy.spin_once(
                node,
                timeout_sec=max(0.0, min(0.1, deadline - monotonic())),
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not joint_positions:
        raise RuntimeError(
            "Could not read /xbotcore/joint_states within "
            f"{timeout_s:.1f}s"
        )
    return joint_positions


def _apply_xbot_joint_positions(q, kin_dyn, joint_positions):
    q = np.asarray(q, dtype=float).reshape(-1).copy()
    used = []
    for name in kin_dyn.joint_names():
        name = str(name).strip()
        if name not in joint_positions or kin_dyn.joint_nq(name) != 1:
            continue

        idx = kin_dyn.joint_iq(name)
        if idx < q.size:
            q[idx] = joint_positions[name]
            used.append(name)

    if not used:
        raise RuntimeError("No model joints matched /xbotcore/joint_states")
    return q, used


def _mat_payload(data):
    return {
        key: value
        for key, value in data.items()
        if not key.startswith("__")
    }


def _show_homing_rviz(urdf, kin_dyn, homing, pipe_center, pipe_radius,
                      pipe_length, pipe_gap, pipe_orientation, clearance,
                      inflation_radius):
    pipe_markers = []
    pipe_marker = PATH_TO_ACEA_CONCERT / "src" / "viz" / "rviz_pipe_marker.py"
    sphere_marker = (
        PATH_TO_ACEA_CONCERT / "src" / "viz" / "rviz_sphere_marker.py")
    marker_layers = (
        ("pipe", pipe_radius, 0, (1.0, 0.5, 0.0, 0.9)),
        (
            "inflation",
            pipe_radius + clearance + inflation_radius,
            10,
            (0.0, 0.7, 1.0, 0.18),
        ),
    )
    pipe_halves = _pipe_halves(pipe_center, pipe_length, pipe_gap,
                               pipe_orientation)
    for layer, radius, marker_id_offset, color in marker_layers:
        for idx, (center, length) in enumerate(pipe_halves):
            marker_args = [
                "python3",
                str(pipe_marker),
                f"homing_{layer}_{idx}",
                str(center[0]),
                str(center[1]),
                str(center[2]),
                str(radius),
                str(length),
                str(pipe_orientation[0]),
                str(pipe_orientation[1]),
                str(pipe_orientation[2]),
                str(pipe_orientation[3]),
                str(marker_id_offset + idx),
                "/weld_pipe",
                "weld_pipe",
            ]
            marker_args.extend(str(value) for value in color)
            pipe_markers.append(subprocess.Popen(marker_args))
    _, tool_spheres = model_homing_geometry(urdf)
    for idx, (frame, offset, radius) in enumerate(tool_spheres):
        pipe_markers.append(subprocess.Popen([
            "python3",
            str(sphere_marker),
            f"homing_tool_sphere_{idx}",
            frame,
            str(offset[0]),
            str(offset[1]),
            str(offset[2]),
            str(radius),
            str(100 + idx),
            "/weld_pipe",
            "tool_spheres",
            "0.0",
            "1.0",
            "0.2",
            "0.35",
        ]))

    rsp_urdf_path = Path("/tmp/concert_homing_rviz.urdf")
    rsp_urdf_path.write_text(urdf)
    rsp = subprocess.Popen([
        "ros2",
        "run",
        "robot_state_publisher",
        "robot_state_publisher",
        str(rsp_urdf_path),
        "--ros-args",
        "-r",
        "/joint_states:=/homing_joint_states",
        "-p",
        "use_sim_time:=false",
    ])
    sleep(0.5)
    if rsp.poll() is not None:
        raise RuntimeError("robot_state_publisher exited before RViz replay")
    try:
        q_cycle = np.concatenate([homing.q, np.flip(homing.q, axis=1)], axis=1)
        _replay_homing_rviz(kin_dyn, q_cycle, homing.dt)
    except KeyboardInterrupt:
        print("[plan_homing_from_mat] RViz replay stopped; continuing.")
    finally:
        rsp.terminate()
        for proc in pipe_markers:
            proc.terminate()


def _replay_homing_rviz(kin_dyn, q_cycle, dt):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import JointState
    from tf2_ros import TransformBroadcaster

    if not rclpy.ok():
        rclpy.init(args=None)

    node = rclpy.create_node("homing_rviz_replayer")
    joint_pub = node.create_publisher(JointState, "/homing_joint_states", 10)
    tf_pub = TransformBroadcaster(node)

    joints_1dof = [
        name for name in kin_dyn.joint_names()
        if kin_dyn.joint_nq(name) == 1
    ]
    iq_1dof = [kin_dyn.joint_iq(name) for name in joints_1dof]
    floating_joints = [
        (
            kin_dyn.joint_iq(name),
            kin_dyn.parentLink(name).lstrip("/"),
            kin_dyn.childLink(name).lstrip("/"),
        )
        for name in kin_dyn.joint_names()
        if kin_dyn.joint_nq(name) == 7
    ]

    try:
        while rclpy.ok():
            for qk in q_cycle.T:
                t0 = monotonic()
                qk = np.asarray(qk, dtype=float).reshape(-1)
                stamp = node.get_clock().now().to_msg()

                joint_msg = JointState()
                joint_msg.header.stamp = stamp
                joint_msg.name = joints_1dof
                joint_msg.position = qk[iq_1dof].tolist()
                joint_pub.publish(joint_msg)

                for iq, parent, child in floating_joints:
                    quat = qk[iq + 3:iq + 7].copy()
                    norm = np.linalg.norm(quat)
                    if norm > 1e-9:
                        quat /= norm
                    else:
                        quat = np.array([0.0, 0.0, 0.0, 1.0])

                    msg = TransformStamped()
                    msg.header.stamp = stamp
                    msg.header.frame_id = parent or "world"
                    msg.child_frame_id = child
                    msg.transform.translation.x = float(qk[iq])
                    msg.transform.translation.y = float(qk[iq + 1])
                    msg.transform.translation.z = float(qk[iq + 2])
                    msg.transform.rotation.x = float(quat[0])
                    msg.transform.rotation.y = float(quat[1])
                    msg.transform.rotation.z = float(quat[2])
                    msg.transform.rotation.w = float(quat[3])
                    tf_pub.sendTransform(msg)

                rclpy.spin_once(node, timeout_sec=0.0)
                sleep_time = dt - (monotonic() - t0)
                if sleep_time > 0.0:
                    sleep(sleep_time)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _ask_show_rejected_candidate(enabled, urdf, kin_dyn, candidate, node,
                                 pairs, pipe_center, pipe_radius, pipe_length,
                                 pipe_gap, pipe_orientation, clearance,
                                 inflation_radius):
    if not enabled:
        return

    print(
        "[plan_homing_from_mat] Rejected candidate collides at "
        f"node {node}: {pairs}"
    )
    try:
        answer = input(
            "[plan_homing_from_mat] Show rejected candidate in RViz? [y/N] "
        )
    except EOFError:
        return
    if answer.strip().lower().startswith("y"):
        _show_homing_rviz(
            urdf, kin_dyn, candidate, pipe_center, pipe_radius, pipe_length,
            pipe_gap, pipe_orientation, clearance, inflation_radius)


def _retry_settings(planner_nodes, inflation_radius, retry):
    if not retry:
        return [(planner_nodes, inflation_radius)]

    node_values = sorted({
        planner_nodes,
        max(planner_nodes, 60),
        max(planner_nodes, 90),
        max(planner_nodes, 120),
        max(planner_nodes, 180),
        max(planner_nodes, 240),
    })
    return [
        (nodes, inflation_radius)
        for nodes in node_values
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat-file", type=Path, default=DEFAULT_MAT_FILE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--initial-pose-from-gazebo",
        action="store_true",
        help=(
            "Use current Gazebo robot base pose and /xbotcore/joint_states "
            "for q_homing_start."
        ),
    )
    parser.add_argument("--gazebo-pose-timeout", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--planner-nodes", type=int, default=30)
    parser.add_argument(
        "--rviz",
        action="store_true",
        help="Replay the computed homing trajectory in RViz after saving it.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Try more nodes until one validated path is found.",
    )
    parser.add_argument(
        "--ask-rviz-failures",
        action="store_true",
        help="Ask whether to replay each rejected Horizon candidate in RViz.",
    )
    args = parser.parse_args()

    data = loadmat(args.mat_file)
    q_goal = np.asarray(data["q"], dtype=float)[:, 0]
    pipe_center = _vector(data, "pos_center_pipe", 3)
    pipe_orientation = _vector(data, "orientation_pipe", 4)
    if "weld_standoff_from_pipe" in data:
        standoff = _scalar(data, "weld_standoff_from_pipe")
        if PIPE_CLEARANCE >= standoff:
            print(
                "[plan_homing_from_mat] Warning: PIPE_CLEARANCE is >= "
                "weld_standoff_from_pipe; the weld-start pose may be "
                "intentionally closer than this."
            )

    urdf, srdf = _make_robot_description()
    kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)
    q_start = _homing_start_q(kin_dyn, q_goal, srdf)
    if args.initial_pose_from_gazebo:
        gazebo_robot_pose, pipe_center, pipe_orientation = (
            _gazebo_scene_pose(args.gazebo_pose_timeout))
        q_start[:7] = gazebo_robot_pose
        q_start, xbot_joints = _apply_xbot_joint_positions(
            q_start,
            kin_dyn,
            _xbot_joint_positions(args.gazebo_pose_timeout),
        )
        print(
            "[plan_homing_from_mat] Using Gazebo scene for q_homing_start: "
            f"{q_start[:7].tolist()}"
        )
        print(
            "[plan_homing_from_mat] Using Gazebo pipe pose: "
            f"center={pipe_center.tolist()}, "
            f"orientation={pipe_orientation.tolist()}"
        )
        print(
            "[plan_homing_from_mat] Using XBot joint state for "
            f"{len(xbot_joints)} q_homing_start joints."
        )
    if q_start.size != q_goal.size:
        raise ValueError(
            f"q0 has {q_start.size} DoFs, MAT q has {q_goal.size}")
    q_goal_homing = q_goal.copy()
    if q_goal_homing.size >= 7:
        q_goal_homing[:7] = q_start[:7]

    pipe_radius = _scalar(data, "radius_pipe")
    pipe_length = _scalar(data, "length_pipe")
    pipe_gap = _scalar(data, "pipe_gap")
    print(
        "[plan_homing_from_mat] Loaded MAT: "
        f"q_dofs={q_goal.size}, pipe_center={pipe_center.tolist()}, "
        f"pipe_radius={pipe_radius:.4f}, pipe_length={pipe_length:.4f}, "
        f"pipe_gap={pipe_gap:.4f}"
    )
    print(
        "[plan_homing_from_mat] Homing setup: "
        f"duration={args.duration:.3f}s, dt={args.dt:.4f}s, "
        f"retry={'on' if args.retry else 'off'}"
    )
    last_error = None
    for attempt, (planner_nodes, inflation_radius) in enumerate(
            _retry_settings(
                args.planner_nodes,
                PIPE_INFLATION_RADIUS,
                args.retry,
            ),
            start=1):
        print(
            f"[plan_homing_from_mat] Attempt {attempt}: "
            f"planner_nodes={planner_nodes}, "
            f"inflation_radius={inflation_radius:.4f}, "
            f"tool_approach_nodes={TOOL_APPROACH_NODES}, "
            f"ipopt_print_level={IPOPT_PRINT_LEVEL}, "
            f"initial_guess={INITIAL_GUESS_MODE}"
        )
        try:
            homing = plan_homing_trajectory(
                urdf=urdf,
                srdf=srdf,
                kin_dyn=kin_dyn,
                q_start=q_start,
                q_goal=q_goal_homing,
                pipe_center=pipe_center,
                pipe_radius=pipe_radius,
                pipe_length=pipe_length,
                pipe_gap=pipe_gap,
                pipe_orientation=pipe_orientation,
                duration=args.duration,
                dt=args.dt,
                planner_nodes=planner_nodes,
                clearance=PIPE_CLEARANCE,
                inflation_radius=inflation_radius,
                tool_approach_nodes=TOOL_APPROACH_NODES,
                motion_weight=JOINT_MOTION_WEIGHT,
                accel_weight=JOINT_ACCEL_WEIGHT,
                compact_weight=COMPACT_WEIGHT,
                cartesian_motion_weight=CARTESIAN_MOTION_WEIGHT,
                max_joint_step=MAX_JOINT_STEP,
                max_joint_accel_step=MAX_JOINT_ACCEL_STEP,
                ipopt_print_level=IPOPT_PRINT_LEVEL,
                initial_guess_mode=INITIAL_GUESS_MODE,
                candidate_callback=lambda candidate, node, pairs: (
                    _ask_show_rejected_candidate(
                        args.ask_rviz_failures,
                        urdf,
                        kin_dyn,
                        candidate,
                        node,
                        pairs,
                        pipe_center,
                        pipe_radius,
                        pipe_length,
                        pipe_gap,
                        pipe_orientation,
                        PIPE_CLEARANCE,
                        inflation_radius,
                    )
                ),
            )
            break
        except RuntimeError as exc:
            if "endpoint collides" in str(exc):
                raise
            last_error = exc
            print(f"[plan_homing_from_mat] Attempt {attempt} failed: {exc}")
    else:
        raise RuntimeError(
            f"No homing trajectory found after retry sweep. Last error: {last_error}"
        ) from last_error

    output = args.output or args.mat_file
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _mat_payload(data)
    payload.pop("q_homing_frame_radius", None)
    payload.pop("q_homing_tool_radius", None)
    payload.pop("q_homing_collision_substeps", None)
    payload.pop("q_homing_direct_weight", None)
    payload.pop("q_homing_max_cartesian_step", None)
    payload.update({
        "q_homing": homing.q,
        "q_homing_start": q_start.reshape(-1, 1),
        "q_homing_goal": q_goal_homing.reshape(-1, 1),
        "q_homing_dt": homing.dt,
        "q_homing_duration": args.duration,
        "q_homing_method": homing.method,
        "q_homing_pipe_center_world": pipe_center,
        "q_homing_pipe_orientation_world": pipe_orientation,
        "q_homing_pipe_clearance": PIPE_CLEARANCE,
        "q_homing_inflation_radius": inflation_radius,
        "q_homing_tool_approach_nodes": TOOL_APPROACH_NODES,
        "q_homing_planner_nodes": planner_nodes,
        "q_homing_motion_weight": JOINT_MOTION_WEIGHT,
        "q_homing_accel_weight": JOINT_ACCEL_WEIGHT,
        "q_homing_compact_weight": COMPACT_WEIGHT,
        "q_homing_cartesian_motion_weight": CARTESIAN_MOTION_WEIGHT,
        "q_homing_max_joint_step": MAX_JOINT_STEP,
        "q_homing_max_joint_accel_step": MAX_JOINT_ACCEL_STEP,
        "q_homing_ipopt_print_level": IPOPT_PRINT_LEVEL,
        "q_homing_initial_guess": INITIAL_GUESS_MODE,
        "q_homing_tool_spheres": np.asarray(
            [
                (*offset, radius)
                for _, offset, radius in model_homing_geometry(urdf)[1]
            ],
            dtype=float,
        ),
    })
    savemat(output, payload)
    print(
        f"[plan_homing_from_mat] Saved {homing.method} homing trajectory "
        f"with {homing.q.shape[1]} nodes to {output}"
    )
    if args.rviz:
        _show_homing_rviz(
            urdf, kin_dyn, homing, pipe_center, pipe_radius, pipe_length,
            pipe_gap, pipe_orientation, PIPE_CLEARANCE, inflation_radius)


if __name__ == "__main__":
    main()
