#!/usr/bin/env python3
"""Home the weld joints to trajectory node 0 before running controller.py."""

import argparse
import sys
from pathlib import Path
from time import sleep

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.utilities import remove_ros_args
from scipy.io import loadmat

from xbot2_interface import pyxbot2_interface as xbi


DEFAULT_MAT_FILE = Path(
    "/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat")

WELD_JOINTS = (
    "J1_E", "J2_E",
    "J1_F", "J2_F", "J3_F", "J4_F", "J5_F", "J6_F",
)

VIRTUAL_JOINTS = {"universe", "reference"}


def fetch_robot_description(node_name: str):
    if not rclpy.ok():
        rclpy.init()

    node = rclpy.create_node(node_name)
    client = node.create_client(
        GetParameters, "/robot_description_publisher/get_parameters")
    if not client.wait_for_service(timeout_sec=15.0):
        node.destroy_node()
        raise RuntimeError(
            "/robot_description_publisher not available. Is the simulation running?")

    req = GetParameters.Request()
    req.names = ["robot_description", "robot_description_semantic"]
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
    node.destroy_node()

    if future.result() is None:
        raise RuntimeError("Failed to read robot_description parameters.")

    vals = future.result().values
    return vals[0].string_value, vals[1].string_value


def load_home_trajectory(mat_file: Path):
    mat_data = loadmat(str(mat_file))
    has_homing = "q_homing" in mat_data
    q_traj = mat_data["q_homing"] if has_homing else mat_data["q"][:, :1]
    all_jnames = [str(n).strip() for n in mat_data["joint_names"].flatten()]
    jnames = [name for name in all_jnames if name not in VIRTUAL_JOINTS]

    q_act = q_traj[7:, :]
    dt_key = "q_homing_dt" if has_homing else "dt"
    dt = float(np.asarray(mat_data.get(dt_key, [[0.01]])).reshape(-1)[0])
    return {
        name: q_act[i, :].astype(float)
        for i, name in enumerate(jnames)
        if name in WELD_JOINTS
    }, dt, "q_homing" if has_homing else "q[:, 0]"


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


def follow_home_trajectory(robot, home_trajectory: dict[str, np.ndarray],
                           dt: float):
    robot.sense()
    command_map = robot.qToMap(robot.getPositionReferenceFeedback())
    nodes = len(next(iter(home_trajectory.values())))

    for node in range(nodes):
        for name, trajectory in home_trajectory.items():
            command_map[name] = float(trajectory[node])
        robot.setPositionReference(robot.mapToQ(command_map))
        robot.move()
        sleep(dt)

    robot.sense()
    return {
        name: float(trajectory[-1])
        for name, trajectory in home_trajectory.items()
    }


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
    model.setJointPosition(desired_map)
    model.update()
    ee_pose_des = model.getPose("ee_F", "world")

    model.setJointPosition(motor_map)
    model.update()
    ee_pose_cur = model.getPose("ee_F", "world")

    ee_err = ee_pose_des.translation - ee_pose_cur.translation
    print(
        f"[home_to_weld_start] EE error: err={ee_err}, "
        f"|err|={np.linalg.norm(ee_err)}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Home weld joints to node 0 of the optimized trajectory.")
    parser.add_argument("--mat-file", type=Path, default=DEFAULT_MAT_FILE)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--start-tolerance", type=float, default=0.05)

    args = parser.parse_args(remove_ros_args(args=argv)[1:])
    if not args.mat_file.exists():
        parser.error(f"--mat-file does not exist: {args.mat_file}")
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    return args


def main(argv=None) -> None:
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(raw_argv)

    rclpy.init(args=raw_argv)
    try:
        print(f"[home_to_weld_start] Loading home target from {args.mat_file}")
        home_traj, home_dt, home_source = load_home_trajectory(args.mat_file)
        if not home_traj:
            raise RuntimeError("No weld joints found in MAT trajectory.")

        print("[home_to_weld_start] Waiting for robot_description...")
        urdf, srdf = fetch_robot_description("home_to_weld_start_urdf_reader")

        print("[home_to_weld_start] Connecting to RobotInterface2...")
        robot = build_robot_interface(urdf, srdf)
        robot.sense()

        robot_q_map = robot.qToMap(robot.getPositionReferenceFeedback())
        robot_joint_names = set(robot_q_map.keys())
        commanded_joints = [
            name for name in home_traj.keys()
            if name in robot_joint_names
        ]
        if not commanded_joints:
            raise RuntimeError("No weld joints are available on RobotInterface2.")

        missing = sorted(set(home_traj.keys()) - set(commanded_joints))
        if missing:
            print(f"[home_to_weld_start] Skipping unavailable joints: {missing}")

        home_trajectory = {
            name: home_traj[name]
            for name in commanded_joints
        }
        print(
            f"[home_to_weld_start] Homing {commanded_joints} "
            f"from {home_source}")

        command_weld_position_mode(robot, commanded_joints)
        if home_source == "q_homing":
            start_error = {
                name: abs(robot_q_map[name] - float(trajectory[0]))
                for name, trajectory in home_trajectory.items()
            }
            worst_joint, worst_error = max(
                start_error.items(), key=lambda item: item[1])
            if worst_error > args.start_tolerance:
                raise RuntimeError(
                    "Current arm pose is not the planned q_homing start. "
                    f"Worst joint: {worst_joint} error={worst_error:.4f}. "
                    "Do not execute this homing trajectory from here.")
            final_target = follow_home_trajectory(
                robot, home_trajectory, home_dt)
        else:
            home_joints = {
                name: float(trajectory[-1])
                for name, trajectory in home_trajectory.items()
            }
            final_target = ramp_to_home(
                robot, home_joints, args.duration, args.dt)
        print_tracking_error(robot, final_target, urdf, srdf)
        print("[home_to_weld_start] Homing complete.")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
