#!/usr/bin/env python3
"""Move camera_F with a small repeatable wrist sweep around the current pose.

Use this only after the simulator is running, /xbotcore/ros_ctrl is Running, and
camera_F is already looking at the pipe junction. The script commands a bounded
sinusoidal offset on selected F-arm joints and optionally returns to the start.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.utilities import remove_ros_args
from std_srvs.srv import SetBool
from xbot2_interface import pyxbot2_interface as xbi


DEFAULT_MAT_FILE = Path("/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat")
DEFAULT_JOINTS = ("J1_E", "J2_E", "J1_F", "J2_F", "J3_F", "J4_F", "J5_F", "J6_F")


def _deg(value: float) -> float:
    return math.radians(float(value))


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _envelope(t: float, duration: float, ramp_s: float) -> float:
    if ramp_s <= 1e-9:
        return 1.0
    return min(_smoothstep(t / ramp_s), _smoothstep((duration - t) / ramp_s))


def fetch_robot_description(node_name: str) -> tuple[str, str]:
    node = rclpy.create_node(node_name)
    try:
        client = node.create_client(GetParameters, "/robot_description_publisher/get_parameters")
        if not client.wait_for_service(timeout_sec=15.0):
            raise RuntimeError("/robot_description_publisher not available")
        req = GetParameters.Request()
        req.names = ["robot_description", "robot_description_semantic"]
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
        if future.result() is None:
            raise RuntimeError("failed to read robot_description parameters")
        vals = future.result().values
        return vals[0].string_value, vals[1].string_value
    finally:
        node.destroy_node()


def enable_ros_ctrl() -> None:
    node = rclpy.create_node("pipe_junction_camera_sweep_ros_ctrl")
    try:
        client = node.create_client(SetBool, "/xbotcore/ros_ctrl/switch")
        if not client.wait_for_service(timeout_sec=3.0):
            print("[camera_sweep] warning: /xbotcore/ros_ctrl/switch unavailable")
            return
        req = SetBool.Request()
        req.data = True
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            print("[camera_sweep] warning: ros_ctrl switch did not reply")
            return
        print(f"[camera_sweep] ros_ctrl switch: success={result.success} message={result.message!r}")
    finally:
        node.destroy_node()


def build_robot_interface(urdf: str, srdf: str):
    cfg = xbi.ConfigOptions()
    cfg.set_urdf(urdf)
    cfg.set_srdf(srdf)
    cfg.set_string_parameter("model_type", "pin")
    cfg.set_bool_parameter("is_model_floating_base", True)
    cfg.set_string_parameter("framework", "ros2")
    return xbi.RobotInterface2(cfg)


def command_position_mode(robot: Any, joints: list[str]) -> None:
    robot.setControlMode(xbi.ControlMode.NONE)
    mode = {name: xbi.ControlMode.POSITION for name in joints}
    if not robot.setControlMode(mode):
        raise RuntimeError(f"failed to set position mode for {joints}")


def _offsets(args: argparse.Namespace, t: float) -> dict[str, float]:
    phase = 2.0 * math.pi * float(args.cycles) * t / max(float(args.duration), 1e-9)
    return {
        "J1_F": _deg(args.j1_f_deg) * math.sin(phase),
        "J2_F": _deg(args.j2_f_deg) * math.sin(phase + 0.5 * math.pi),
        "J3_F": _deg(args.j3_f_deg) * math.sin(phase + math.pi),
        "J4_F": _deg(args.j4_f_deg) * math.sin(phase),
        "J5_F": _deg(args.j5_f_deg) * math.sin(phase + 0.5 * math.pi),
        "J6_F": _deg(args.j6_f_deg) * math.sin(phase + math.pi),
    }


def _ramp_to(robot: Any, start_map: dict[str, float], target_map: dict[str, float], duration: float, dt: float) -> None:
    steps = max(1, int(duration / dt))
    for step in range(steps):
        alpha = _smoothstep((step + 1) / steps)
        cmd = dict(start_map)
        for name, target in target_map.items():
            cmd[name] = (1.0 - alpha) * start_map[name] + alpha * target
        robot.setPositionReference(robot.mapToQ(cmd))
        robot.move()
        time.sleep(dt)


def run(args: argparse.Namespace) -> None:
    if args.enable_ros_ctrl:
        enable_ros_ctrl()

    print("[camera_sweep] waiting for robot description")
    urdf, srdf = fetch_robot_description("pipe_junction_camera_sweep_urdf_reader")
    print("[camera_sweep] connecting to RobotInterface2")
    robot = build_robot_interface(urdf, srdf)
    robot.sense()

    q_current = robot.qToMap(robot.getJointPosition())
    q_ref = robot.qToMap(robot.getPositionReferenceFeedback())
    base = dict(q_ref)
    for name in DEFAULT_JOINTS:
        if name in q_current:
            base[name] = float(q_current[name])

    commanded = [name for name in DEFAULT_JOINTS if name in base]
    command_position_mode(robot, commanded)
    print(f"[camera_sweep] commanding joints: {commanded}")
    print(
        "[camera_sweep] amplitudes deg: "
        f"J1_F={args.j1_f_deg}, J2_F={args.j2_f_deg}, J3_F={args.j3_f_deg}, "
        f"J4_F={args.j4_f_deg}, J5_F={args.j5_f_deg}, J6_F={args.j6_f_deg}"
    )

    dt = float(args.dt)
    duration = float(args.duration)
    steps = max(1, int(duration / dt))
    t0 = time.monotonic()
    for step in range(steps):
        t = min(duration, step * dt)
        env = _envelope(t, duration, float(args.ramp_s))
        cmd = dict(base)
        offsets = _offsets(args, t)
        for name, offset in offsets.items():
            if name in cmd:
                cmd[name] = float(base[name] + env * offset)
        robot.setPositionReference(robot.mapToQ(cmd))
        robot.move()
        if step % max(1, int(1.0 / dt)) == 0:
            print(
                f"\r[camera_sweep] t={t:5.1f}/{duration:.1f}s "
                f"env={env:.2f} J4={math.degrees(cmd.get('J4_F', 0.0) - base.get('J4_F', 0.0)):+.1f}deg "
                f"J5={math.degrees(cmd.get('J5_F', 0.0) - base.get('J5_F', 0.0)):+.1f}deg "
                f"J6={math.degrees(cmd.get('J6_F', 0.0) - base.get('J6_F', 0.0)):+.1f}deg",
                end="",
                flush=True,
            )
        sleep_until = t0 + (step + 1) * dt
        time.sleep(max(0.0, sleep_until - time.monotonic()))
    print()

    if args.return_home:
        print(f"[camera_sweep] returning to start over {args.return_s:.1f}s")
        robot.sense()
        now_map = robot.qToMap(robot.getPositionReferenceFeedback())
        _ramp_to(robot, now_map, {name: base[name] for name in commanded}, float(args.return_s), dt)
    print("[camera_sweep] done")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--cycles", type=float, default=1.0)
    parser.add_argument("--ramp-s", type=float, default=3.0)
    parser.add_argument("--return-home", action="store_true", default=True)
    parser.add_argument("--no-return-home", dest="return_home", action="store_false")
    parser.add_argument("--return-s", type=float, default=4.0)
    parser.add_argument("--enable-ros-ctrl", action="store_true", default=True)
    parser.add_argument("--no-enable-ros-ctrl", dest="enable_ros_ctrl", action="store_false")
    parser.add_argument("--j1-f-deg", type=float, default=0.0,
                        help="Optional shoulder/arm offset. Keep 0 for first tests.")
    parser.add_argument("--j2-f-deg", type=float, default=0.0,
                        help="Optional shoulder/arm offset. Keep 0 for first tests.")
    parser.add_argument("--j3-f-deg", type=float, default=0.0,
                        help="Optional elbow offset. Keep 0 for first tests.")
    parser.add_argument("--j4-f-deg", type=float, default=5.0)
    parser.add_argument("--j5-f-deg", type=float, default=3.0)
    parser.add_argument("--j6-f-deg", type=float, default=8.0)
    args = parser.parse_args(remove_ros_args(args=argv)[1:])
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.dt <= 0.0:
        parser.error("--dt must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv if argv is None else argv
    rclpy.init(args=raw_argv)
    try:
        run(parse_args(raw_argv))
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
