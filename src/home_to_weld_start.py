#!/usr/bin/env python3
"""Home the weld joints to trajectory node 0 before running controller.py.

Usage:
    ros2 run acea_concert home_to_weld_start.py
    ros2 run acea_concert home_to_weld_start.py \
        --homing-trajectory /tmp/weld_concert_compact.mat
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from time import sleep

import numpy as np
import rclpy
from rclpy.utilities import remove_ros_args
from scipy.io import loadmat
from acea_concert.control.ros import fetch_robot_description
from xbot2_interface import pyxbot2_interface as xbi


DEFAULT_MAT_FILE = Path(
    "/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat")

BASE_AND_WHEEL_JOINTS = (
    "J1_A", "J1_B", "J1_C", "J1_D",
    "J_wheel_A", "J_wheel_B", "J_wheel_C", "J_wheel_D",
)

VIRTUAL_JOINTS = {"universe", "reference"}
PLANNED_START_TOLERANCE = 0.05
FINAL_HOLD_DURATION = 1.0


def _mat_string(value):
    if isinstance(value, np.ndarray):
        flattened = value.reshape(-1)
        if flattened.size == 1:
            return _mat_string(flattened[0])
        if value.dtype.kind in ("U", "S"):
            return "".join(str(item) for item in flattened).strip()
    return str(value).strip()


def _mat_joint_names(mat_data):
    return [_mat_string(name) for name in mat_data["joint_names"].flatten()]


def ee_link_from_urdf(urdf: str):
    links = {
        link.attrib["name"]
        for link in ET.fromstring(urdf).findall("link")
        if "name" in link.attrib
    }
    for ee_link in ("ee_F", "ee_E"):
        if ee_link in links:
            return ee_link
    raise RuntimeError("Neither ee_F nor ee_E exists in the URDF.")


def load_home_map(mat_file: Path):
    mat_data = loadmat(str(mat_file))
    q_traj = mat_data["q"]
    all_jnames = _mat_joint_names(mat_data)
    jnames = [name for name in all_jnames if name not in VIRTUAL_JOINTS]

    q_act = q_traj[7:, :]
    q_home_act = q_act[:, 0]
    return {
        name: float(q_home_act[i])
        for i, name in enumerate(jnames)
        if name not in BASE_AND_WHEEL_JOINTS
    }


def load_planned_homing(mat_file: Path):
    mat_data = loadmat(str(mat_file))
    if "q_homing" not in mat_data:
        return None

    path = np.asarray(mat_data["q_homing"], dtype=float)
    all_jnames = _mat_joint_names(mat_data)
    jnames = [name for name in all_jnames if name not in VIRTUAL_JOINTS]
    if path.ndim != 2 or path.shape[0] != 7 + len(jnames):
        raise ValueError(
            "q_homing rows do not match floating base + MAT joint_names")

    dt = float(np.asarray(mat_data["q_homing_dt"]).reshape(-1)[0])
    if dt <= 0.0:
        raise ValueError("q_homing_dt must be positive")

    joint_path = [
        {
            name: float(path[7 + index, node])
            for index, name in enumerate(jnames)
            if name not in BASE_AND_WHEEL_JOINTS
        }
        for node in range(path.shape[1])
    ]
    return joint_path, dt


def build_robot_interface(urdf: str, srdf: str):
    cfg = xbi.ConfigOptions()
    cfg.set_urdf(urdf)
    cfg.set_srdf(srdf)
    cfg.set_string_parameter("model_type", "pin")
    cfg.set_bool_parameter("is_model_floating_base", True)
    cfg.set_string_parameter("framework", "ros2")
    return xbi.RobotInterface2(cfg)


def command_weld_position_mode(robot, commanded_joints):
    robot.setControlMode(xbi.ControlMode.NONE)
    weld_control_mode = {
        name: xbi.ControlMode.POSITION
        for name in commanded_joints
    }
    if not robot.setControlMode(weld_control_mode):
        raise RuntimeError(
            f"Failed to set weld-only control mode: {commanded_joints}")


def ramp_to_home(robot, home_joints: dict[str, float],
                 duration: float, dt: float):
    robot.sense()
    q_start_map = robot.qToMap(robot.getPositionReferenceFeedback())
    steps = max(1, int(duration / dt))
    interp_map = dict(home_joints)

    for step in range(steps):
        alpha = (step + 1) / steps
        interp_map = {
            name: (1.0 - alpha) * q_start_map[name] + alpha * target
            for name, target in home_joints.items()
        }
        command_map = dict(q_start_map)
        command_map.update(interp_map)
        robot.setPositionReference(robot.mapToQ(command_map))
        robot.move()
        sleep(dt)

    robot.sense()
    return interp_map


def replay_homing(robot, joint_path, dt, start_tolerance):
    robot.sense()
    reference_map = robot.qToMap(robot.getPositionReferenceFeedback())
    first = joint_path[0]
    start_error = max(
        abs(reference_map[name] - target)
        for name, target in first.items()
    )
    if start_error > start_tolerance:
        raise RuntimeError(
            "Robot does not match q_homing_start: "
            f"maximum joint error {start_error:.4f} rad exceeds "
            f"{start_tolerance:.4f} rad. Replan from the current state.")

    for target in joint_path[1:]:
        command_map = dict(reference_map)
        command_map.update(target)
        robot.setPositionReference(robot.mapToQ(command_map))
        robot.move()
        sleep(dt)

    robot.sense()
    return joint_path[-1]


def hold_position(robot, target, duration, dt):
    if duration <= 0.0:
        return
    command_map = robot.qToMap(robot.getPositionReferenceFeedback())
    command_map.update(target)
    command = robot.mapToQ(command_map)
    for _ in range(max(1, int(duration / dt))):
        robot.setPositionReference(command)
        robot.move()
        sleep(dt)
    robot.sense()


def print_tracking_error(robot, target_map: dict[str, float],
                         urdf: str, srdf: str):
    motor_map = robot.qToMap(robot.getJointPosition())
    print("[home_to_weld_start] Joint error after homing:")
    for joint, target in target_map.items():
        current = motor_map[joint]
        print(f"    {joint:20s}: {target - current:+.6f}")

    desired_map = dict(motor_map)
    desired_map.update(target_map)

    model = xbi.ModelInterface2(urdf, srdf, "pin")
    ee_link = ee_link_from_urdf(urdf)
    model.setJointPosition(desired_map)
    model.update()
    ee_pose_des = model.getPose(ee_link, "world")

    model.setJointPosition(motor_map)
    model.update()
    ee_pose_cur = model.getPose(ee_link, "world")

    ee_err = ee_pose_des.translation - ee_pose_cur.translation
    print(
        f"[home_to_weld_start] EE error: err={ee_err}, "
        f"|err|={np.linalg.norm(ee_err)}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Home weld joints to node 0 of the optimized trajectory.")
    parser.add_argument(
        "--homing-trajectory",
        type=Path,
        help=(
            "Planned homing MAT file to replay. Without it, interpolate "
            "directly to the default welding trajectory start."
        ),
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.01)

    args = parser.parse_args(remove_ros_args(args=argv)[1:])
    if (
        args.homing_trajectory is not None
        and not args.homing_trajectory.exists()
    ):
        parser.error(
            "--homing-trajectory does not exist: "
            f"{args.homing_trajectory}")
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    return args


def main(argv=None) -> None:
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(raw_argv)
    trajectory_file = args.homing_trajectory or DEFAULT_MAT_FILE

    rclpy.init(args=raw_argv)
    try:
        print(
            "[home_to_weld_start] Loading home target from "
            f"{trajectory_file}")
        home_map = load_home_map(trajectory_file)
        if not home_map:
            raise RuntimeError("No weld joints found in MAT trajectory.")
        planned = (
            load_planned_homing(trajectory_file)
            if args.homing_trajectory is not None
            else None
        )
        if args.homing_trajectory is not None and planned is None:
            raise RuntimeError(
                "The homing trajectory has no q_homing. "
                "Run plan_homing_from_mat.py first.")

        print("[home_to_weld_start] Waiting for robot description...")
        urdf, srdf = fetch_robot_description(
            "home_to_weld_start_urdf_reader")

        print("[home_to_weld_start] Connecting to RobotInterface2...")
        robot = build_robot_interface(urdf, srdf)
        print("[home_to_weld_start] RobotInterface2 connected.")
        robot.sense()
        robot_q_map = robot.qToMap(robot.getPositionReferenceFeedback())
        robot_joint_names = set(robot_q_map.keys())
        commanded_joints = [
            name for name in home_map.keys()
            if name in robot_joint_names
        ]
        if not commanded_joints:
            raise RuntimeError("No weld joints are available on RobotInterface2.")

        missing = sorted(set(home_map.keys()) - set(commanded_joints))
        if missing:
            print(f"[home_to_weld_start] Skipping unavailable joints: {missing}")

        home_joints = {
            name: home_map[name]
            for name in commanded_joints
        }
        command_weld_position_mode(robot, commanded_joints)
        if planned is None:
            print(
                f"[home_to_weld_start] Direct homing {commanded_joints} "
                f"over {args.duration:.2f}s")
            final_target = ramp_to_home(
                robot, home_joints, args.duration, args.dt)
            command_dt = args.dt
        else:
            joint_path, planned_dt = planned
            joint_path = [
                {
                    name: target[name]
                    for name in commanded_joints
                }
                for target in joint_path
            ]
            print(
                "[home_to_weld_start] Replaying planned compact homing: "
                f"{len(joint_path)} nodes, dt={planned_dt:.4f}s")
            final_target = replay_homing(
                robot, joint_path, planned_dt, PLANNED_START_TOLERANCE)
            command_dt = planned_dt
        hold_position(
            robot, final_target, FINAL_HOLD_DURATION, command_dt)
        print_tracking_error(robot, final_target, urdf, srdf)
        print("[home_to_weld_start] Homing complete.")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
