#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
import numpy as np
from scipy.io import loadmat, savemat

from homing_trajectory import (
    DEFAULT_ACCEL_WEIGHT,
    DEFAULT_INFLATION_RADIUS,
    DEFAULT_INITIAL_GUESS_MODE,
    DEFAULT_IPOPT_PRINT_LEVEL,
    DEFAULT_MAX_JOINT_ACCEL_STEP,
    DEFAULT_MAX_JOINT_STEP,
    DEFAULT_MOTION_WEIGHT,
    DEFAULT_TOOL_APPROACH_NODES,
    TOOL_SPHERES,
    _pipe_halves,
    plan_homing_trajectory,
)


PATH_TO_ACEA_CONCERT = Path("/home/user/concert_ws/src/acea_concert")
DEFAULT_MAT_FILE = PATH_TO_ACEA_CONCERT / "mat_files" / "weld_concert.mat"
VIRTUAL_JOINTS = {"universe", "reference"}
ARM_HOME_Q = {
    "J1_E": -np.pi / 2,
    "J2_E": 0.0,
    "J1_F": 0.0,
    "J2_F": 0.0,
    "J3_F": 0.0,
    "J4_F": 0.0,
    "J5_F": 0.0,
    "J6_F": 0.0,
}


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


def _homing_start_q(kin_dyn, q_goal):
    q_start = np.asarray(q_goal, dtype=float).reshape(-1).copy()
    actuated_names = [
        str(name).strip() for name in kin_dyn.joint_names()
        if str(name).strip() not in VIRTUAL_JOINTS
    ]
    missing = []
    for name, value in ARM_HOME_Q.items():
        if name not in actuated_names:
            missing.append(name)
            continue
        q_start[7 + actuated_names.index(name)] = value
    if missing:
        raise ValueError(f"Arm home joints not found in model: {missing}")
    return q_start


def _mat_payload(data):
    return {
        key: value
        for key, value in data.items()
        if not key.startswith("__")
    }


def _show_homing_rviz(urdf, kin_dyn, homing, pipe_center, pipe_radius,
                      pipe_length, pipe_gap, pipe_orientation, clearance,
                      inflation_radius):
    from horizon.ros import replay_trajectory

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
    for idx, (frame, offset, radius) in enumerate(TOOL_SPHERES):
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

    rsp = subprocess.Popen([
        "ros2",
        "run",
        "robot_state_publisher",
        "robot_state_publisher",
        "--ros-args",
        "-p",
        f"robot_description:={urdf}",
    ])
    try:
        q_cycle = np.concatenate([homing.q, np.flip(homing.q, axis=1)], axis=1)
        repl = replay_trajectory.replay_trajectory(
            homing.dt,
            kin_dyn.joint_names(),
            q_cycle,
            kindyn=kin_dyn,
            future_trajectory_markers={"ee_F": "world"},
        )
        repl.replay()
    except KeyboardInterrupt:
        print("[plan_homing_from_mat] RViz replay stopped; continuing.")
    finally:
        rsp.terminate()
        for proc in pipe_markers:
            proc.terminate()


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
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--planner-nodes", type=int, default=30)
    parser.add_argument("--clearance", type=float, default=0.0)
    parser.add_argument(
        "--inflation-radius",
        type=float,
        default=DEFAULT_INFLATION_RADIUS,
        help="Conservative pipe inflation for arm guide constraints.",
    )
    parser.add_argument(
        "--tool-approach-nodes",
        type=int,
        default=DEFAULT_TOOL_APPROACH_NODES,
        help="Last Horizon nodes where tool-sphere pipe clearance is relaxed.",
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=DEFAULT_MOTION_WEIGHT,
        help="Weight for minimizing arm joint motion between nodes.",
    )
    parser.add_argument(
        "--accel-weight",
        type=float,
        default=DEFAULT_ACCEL_WEIGHT,
        help="Weight for smoothing arm joint acceleration.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=DEFAULT_MAX_JOINT_STEP,
        help="Maximum arm joint change between Horizon nodes; <=0 disables.",
    )
    parser.add_argument(
        "--max-joint-accel-step",
        type=float,
        default=DEFAULT_MAX_JOINT_ACCEL_STEP,
        help="Maximum arm joint second difference between Horizon nodes; <=0 disables.",
    )
    parser.add_argument(
        "--ipopt-print-level",
        type=int,
        default=DEFAULT_IPOPT_PRINT_LEVEL,
        help="IPOPT verbosity; use 5 to show iteration steps.",
    )
    parser.add_argument(
        "--initial-guess",
        choices=("linear", "goal-after-start"),
        default=DEFAULT_INITIAL_GUESS_MODE,
        help="Initial guess for Horizon q nodes.",
    )
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
        if args.clearance >= standoff:
            print(
                "[plan_homing_from_mat] Warning: --clearance is >= "
                "weld_standoff_from_pipe; the weld-start pose may be "
                "intentionally closer than this."
            )

    urdf, srdf = _make_robot_description()
    kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)
    q_start = _homing_start_q(kin_dyn, q_goal)
    if q_start.size != q_goal.size:
        raise ValueError(
            f"q0 has {q_start.size} DoFs, MAT q has {q_goal.size}")

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
                args.inflation_radius,
                args.retry,
            ),
            start=1):
        print(
            f"[plan_homing_from_mat] Attempt {attempt}: "
            f"planner_nodes={planner_nodes}, "
            f"inflation_radius={inflation_radius:.4f}, "
            f"tool_approach_nodes={args.tool_approach_nodes}, "
            f"ipopt_print_level={args.ipopt_print_level}, "
            f"initial_guess={args.initial_guess}"
        )
        try:
            homing = plan_homing_trajectory(
                urdf=urdf,
                srdf=srdf,
                kin_dyn=kin_dyn,
                q_start=q_start,
                q_goal=q_goal,
                pipe_center=pipe_center,
                pipe_radius=pipe_radius,
                pipe_length=pipe_length,
                pipe_gap=pipe_gap,
                pipe_orientation=pipe_orientation,
                duration=args.duration,
                dt=args.dt,
                planner_nodes=planner_nodes,
                clearance=args.clearance,
                inflation_radius=inflation_radius,
                tool_approach_nodes=args.tool_approach_nodes,
                motion_weight=args.motion_weight,
                accel_weight=args.accel_weight,
                max_joint_step=args.max_joint_step,
                max_joint_accel_step=args.max_joint_accel_step,
                ipopt_print_level=args.ipopt_print_level,
                initial_guess_mode=args.initial_guess,
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
                        args.clearance,
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
        "q_homing_goal": q_goal.reshape(-1, 1),
        "q_homing_dt": homing.dt,
        "q_homing_duration": args.duration,
        "q_homing_method": homing.method,
        "q_homing_pipe_clearance": args.clearance,
        "q_homing_inflation_radius": inflation_radius,
        "q_homing_tool_approach_nodes": args.tool_approach_nodes,
        "q_homing_planner_nodes": planner_nodes,
        "q_homing_motion_weight": args.motion_weight,
        "q_homing_accel_weight": args.accel_weight,
        "q_homing_max_joint_step": args.max_joint_step,
        "q_homing_max_joint_accel_step": args.max_joint_accel_step,
        "q_homing_ipopt_print_level": args.ipopt_print_level,
        "q_homing_initial_guess": args.initial_guess,
        "q_homing_tool_spheres": np.asarray(
            [(*offset, radius) for _, offset, radius in TOOL_SPHERES],
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
            pipe_gap, pipe_orientation, args.clearance, inflation_radius)


if __name__ == "__main__":
    main()
