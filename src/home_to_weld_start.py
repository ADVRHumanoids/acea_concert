#!/usr/bin/env python3
"""Home the weld joints to trajectory node 0 before running controller.py."""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from time import sleep

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.utilities import remove_ros_args
from scipy.io import loadmat
from controller_ros import fetch_robot_description
from xbot2_interface import pyxbot2_interface as xbi


DEFAULT_MAT_FILE = Path(
    "/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat")

BASE_AND_WHEEL_JOINTS = (
    "J1_A", "J1_B", "J1_C", "J1_D",
    "J_wheel_A", "J_wheel_B", "J_wheel_C", "J_wheel_D",
)

VIRTUAL_JOINTS = {"universe", "reference"}


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


# def fetch_robot_description(node_name: str):
#     if not rclpy.ok():
#         rclpy.init()

#     node = rclpy.create_node(node_name)

#     robot_description = None
#     robot_description_semantic = None

#     def robot_description_callback(msg):
#         nonlocal robot_description
#         robot_description = msg.data

#     def robot_description_semantic_callback(msg):
#         nonlocal robot_description_semantic
#         robot_description_semantic = msg.data

#     qos = QoSProfile(
#         depth=1,
#         durability=DurabilityPolicy.TRANSIENT_LOCAL,
#         reliability=ReliabilityPolicy.RELIABLE,
#     )

#     node.create_subscription(
#         String,
#         "/xbotcore/robot_description",
#         robot_description_callback,
#         qos,
#     )

#     node.create_subscription(
#         String,
#         "/xbotcore/robot_description_semantic",
#         robot_description_semantic_callback,
#         qos,
#     )

#     timeout = 15.0
#     start_time = node.get_clock().now()

#     while rclpy.ok():
#         rclpy.spin_once(node, timeout_sec=0.1)

#         elapsed = (
#             node.get_clock().now() - start_time
#         ).nanoseconds / 1e9

#         if robot_description is not None and robot_description_semantic is not None:
#             break

#         if elapsed > timeout:
#             node.destroy_node()
#             raise RuntimeError(
#                 "Failed to receive robot description topics. "
#                 "Check QoS compatibility and that /xbotcore is running."
#             )

#     node.destroy_node()

#     return robot_description, robot_description_semantic

# def fetch_robot_description(node_name: str):
#     if not rclpy.ok():
#         rclpy.init()

#     node = rclpy.create_node(node_name)
#     client = node.create_client(
#         GetParameters, "/robot_description_publisher/get_parameters")
#     if not client.wait_for_service(timeout_sec=15.0):
#         node.destroy_node()
#         raise RuntimeError(
#             "/robot_description_publisher not available. Is the simulation running?")

#     req = GetParameters.Request()
#     req.names = ["robot_description", "robot_description_semantic"]
#     future = client.call_async(req)
#     rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
#     node.destroy_node()

#     if future.result() is None:
#         raise RuntimeError("Failed to read robot_description parameters.")

#     vals = future.result().values
#     return vals[0].string_value, vals[1].string_value


def load_home_map(mat_file: Path):
    mat_data = loadmat(str(mat_file))
    q_traj = mat_data["q"]
    all_jnames = [str(n).strip() for n in mat_data["joint_names"].flatten()]
    jnames = [name for name in all_jnames if name not in VIRTUAL_JOINTS]

    q_act = q_traj[7:, :]
    q_home_act = q_act[:, 0]
    return {
        name: float(q_home_act[i])
        for i, name in enumerate(jnames)
        if name not in BASE_AND_WHEEL_JOINTS
    }


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
    parser.add_argument("--mat-file", type=Path, default=DEFAULT_MAT_FILE)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.01)

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
        home_map = load_home_map(args.mat_file)
        if not home_map:
            raise RuntimeError("No weld joints found in MAT trajectory.")

        print("[home_to_weld_start] Waiting for robot_description...")
        urdf, srdf = fetch_robot_description("home_to_weld_start_urdf_reader")

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
        print(
            f"[home_to_weld_start] Homing {commanded_joints} "
            f"over {args.duration:.2f}s")

        command_weld_position_mode(robot, commanded_joints)
        final_target = ramp_to_home(robot, home_joints, args.duration, args.dt)
        print_tracking_error(robot, final_target, urdf, srdf)
        print("[home_to_weld_start] Homing complete.")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
