#!/usr/bin/env python3
"""
CartesIO-based end-effector controller for the CONCERT welding robot.

Architecture:
  - RobotInterface2 senses the robot state and sends joint position references.
  - A standalone CartesianInterface (OpenSot) runs the IK locally each tick.
  - URDF/SRDF are read from XBot's transient-local description topics.

Usage (simulation must already be running and homing already completed):
    ros2 launch acea_concert weld_sim.launch.py
    ros2 run acea_concert home_to_weld_start.py
    python3 controller.py
    python3 controller.py --open-loop
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from time import sleep, perf_counter

import numpy as np
from rclpy.utilities import remove_ros_args
from scipy.io import loadmat

from acea_concert.control.core import (
    GapFeedbackController,
    rotation_error_angle,
)
from acea_concert.control.ros import ControllerRosInterface
from acea_concert.control.trajectory import (
    CyclicPosturalTrajectory,
    CyclicQuaternionTrajectory,
    CyclicVectorTrajectory,
    ee_pose_from_postural,
)

from xbot2_interface import pyxbot2_interface as xbi
from cartesian_interface import pyci

# ── Parameters ───────────────────────────────────────────────────────────────
DT          = 0.01     # [s]   controller dt (100 Hz)

BASE_AND_WHEEL_JOINTS = (
    'J1_A', 'J1_B', 'J1_C', 'J1_D',
    'J_wheel_A', 'J_wheel_B', 'J_wheel_C', 'J_wheel_D',
)

# ── Trajectory (world X back-and-forth) ──────────────────────────────────────

# ── PD gains (gap frame: x=tangent along pipe, y=gap normal) ────────────────
KP_XYZ = [30.0, 30.0, 1.0]    # proportional [1/s]
KD_XYZ = [2.0,  2.0, 0.05]  # derivative   [s]
MAX_Y_VEL = 10.0          # [m/s] cap for the gap-normal correction velocity
MAX_X_VEL = 10.0          # [m/s] cap for the gap-tangent correction velocity

GAIN_PARAM_DEFAULTS = {
    'kp_tangent_x': KP_XYZ[0],
    'kd_tangent_x': KD_XYZ[0],
    'kp_normal': KP_XYZ[1],
    'kd_normal': KD_XYZ[1],
}

# ── Trajectory slowdown factor ───────────────────────────────────────────────
TRAJ_SLOWDOWN = 12.0  # 1.0 = normal speed, 2.0 = half speed, etc.

GAP_POSE_TIMEOUT_S = 0.25

# Path to the CartesIO problem description YAML
CARTESIO_YAML = Path('/home/user/concert_ws/src/acea_concert/config/cartesio_stack.yaml')

# ── Mat file trajectory ───────────────────────────────────────────────────────
MAT_FILE = Path('/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat')


def ee_link_from_urdf(urdf: str):
    links = {
        link.attrib['name']
        for link in ET.fromstring(urdf).findall('link')
        if 'name' in link.attrib
    }
    for ee_link in ('ee_F', 'ee_E'):
        if ee_link in links:
            return ee_link
    raise RuntimeError('Neither ee_F nor ee_E exists in the URDF.')


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the weld end-effector controller.")
    parser.add_argument(
        "--open-loop",
        "--replay-open-loop",
        action="store_true",
        help=(
            "Replay the optimized trajectory without requiring "
            "/gap/pose_robot or applying gap-frame corrections."
        ),
    )
    parser.add_argument(
        "--stop-on-gap-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pause before sending another command if /gap/pose_robot becomes "
            "stale. The first gap pose is always required."
        ),
    )
    parser.add_argument(
        "--tangent-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the corrective action along the gap tangent.",
    )
    parser.add_argument(
        "--gap-pose-timeout",
        type=float,
        default=GAP_POSE_TIMEOUT_S,
        help="Maximum accepted age for /gap/pose_robot in seconds.",
    )
    parser.add_argument(
        "--gap-filter-tau",
        "--gap-filter-time-constant",
        type=float,
        default=0.0,
        help=(
            "Low-pass filter time constant for /gap/pose_robot [s]. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--gap-filter-history-size",
        type=int,
        default=1,
        help=(
            "Number of accepted /gap/pose_robot samples used for the median "
            "history estimate. Use 1 to disable history."
        ),
    )
    parser.add_argument(
        "--gap-filter-max-position-jump",
        type=float,
        default=0.0,
        help=(
            "Reject camera pose samples farther than this from the current "
            "filtered gap position [m]. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--gap-filter-max-angle-jump",
        type=float,
        default=0.0,
        help=(
            "Reject camera pose samples whose orientation differs by more "
            "than this from the current filtered gap orientation [deg]. "
            "Use 0 to disable."
        ),
    )
    args = parser.parse_args(remove_ros_args(args=argv)[1:])
    if args.gap_pose_timeout <= 0.0:
        parser.error("--gap-pose-timeout must be positive")
    if args.gap_filter_tau < 0.0:
        parser.error("--gap-filter-tau must be >= 0")
    if args.gap_filter_history_size < 1:
        parser.error("--gap-filter-history-size must be >= 1")
    if args.gap_filter_max_position_jump < 0.0:
        parser.error("--gap-filter-max-position-jump must be >= 0")
    if args.gap_filter_max_angle_jump < 0.0:
        parser.error("--gap-filter-max-angle-jump must be >= 0")
    return args


args = _parse_args(sys.argv)

# ── Load weld_opt trajectory from mat file ────────────────────────────────────
print(f"[controller] Loading trajectory from {MAT_FILE} …")
mat_data = loadmat(str(MAT_FILE))

# q: (nq x N_nodes) — full model joint vector (floating base + actuated)
# joint_names: casadi_kin_dyn list
q_traj = mat_data['q'] # shape (nq, N)

all_jnames = [str(n).strip() for n in mat_data['joint_names'].flatten()]

trj_dt  = float(np.atleast_1d(mat_data['dt']).flat[0])
N       = q_traj.shape[1]
trj_dur = (N - 1) * trj_dt

# The first 7 rows are floating-base (pos xyz + quat xyzw); rest are actuated.
n_base  = 7
q_act   = q_traj[n_base:, :]  # (n_actuated x N)
weld_pos_traj_gap = np.asarray(
    mat_data['desired_traj_weld_pos_gap'], dtype=float)
weld_quat_traj_gap = np.asarray(
    mat_data['desired_traj_weld_quat_gap'], dtype=float)

# casadi_kin_dyn prepends virtual/fixed joints ('universe', 'reference', …)
# that carry no DOF — strip them so len(_jnames) == _n_act
VIRTUAL = {'universe', 'reference'}
jnames  = [n for n in all_jnames if n not in VIRTUAL]

weld_jnames = [name for name in jnames if name not in BASE_AND_WHEEL_JOINTS]

print(f"[controller] Trajectory: {N} nodes, dt={trj_dt:.4f}s, duration={trj_dur:.2f}s")
print(f"[controller] Actuated joint names ({len(jnames)}): {jnames}")
print(f"[controller] Weld joints from trajectory ({len(weld_jnames)}): {weld_jnames}")

postural_trajectory = CyclicPosturalTrajectory(q_act, jnames, trj_dt)
weld_gap_trajectory = CyclicVectorTrajectory(weld_pos_traj_gap, trj_dt) # desired EE/weld position expressed in the gap frame
weld_gap_orientation = CyclicQuaternionTrajectory(weld_quat_traj_gap, trj_dt) # desired EE orientation expressed in the gap frame
print("[controller] Trajectory interpolator ready.")

# ── Read URDF/SRDF ───────────────────────────────────────────────────────────
print("[controller] Waiting for robot description …")
controller_ros = ControllerRosInterface(
    GAIN_PARAM_DEFAULTS,
    gap_pose_filter_tau_s=args.gap_filter_tau,
    gap_pose_filter_history_size=args.gap_filter_history_size,
    gap_pose_filter_max_position_jump_m=args.gap_filter_max_position_jump,
    gap_pose_filter_max_angle_jump_deg=args.gap_filter_max_angle_jump,
)

urdf, srdf = controller_ros.robot_description()
print("[controller] URDF and SRDF received.")
ee_link = ee_link_from_urdf(urdf)

# ── Build ConfigOptions for RobotInterface2 ───────────────────────────────────
cfg = xbi.ConfigOptions()
cfg.set_urdf(urdf)
cfg.set_srdf(srdf)
cfg.set_string_parameter('model_type', 'pin')
cfg.set_bool_parameter('is_model_floating_base', True)
cfg.set_string_parameter('framework', 'ros2')

# ── Create RobotInterface2 and sense initial state ────────────────────────────
print("[controller] Connecting to RobotInterface2 …")
robot = xbi.RobotInterface2(cfg)
robot.sense()

robot_q_map = robot.qToMap(robot.getPositionReferenceFeedback())
robot_joint_names = set(robot_q_map.keys())
print(f"[controller] Robot joint names ({len(robot_joint_names)}): {sorted(robot_joint_names)}")
print(f"[controller] Robot joint positions: {robot_q_map}")

commanded_jnames = [name for name in weld_jnames if name in robot_joint_names]
if not commanded_jnames:
    raise RuntimeError("No weld joints are available on RobotInterface2")

robot.setControlMode(xbi.ControlMode.NONE)
weld_control_mode = {name: xbi.ControlMode.POSITION for name in commanded_jnames}
if not robot.setControlMode(weld_control_mode):
    raise RuntimeError(f"Failed to set weld-only control mode: {commanded_jnames}")
print(f"[controller] Commanding weld joints only: {commanded_jnames}")

# ── Build ModelInterface2 for the standalone solver ───────────────────────────
model = xbi.ModelInterface2(urdf, srdf, 'pin')
model_shadow = xbi.ModelInterface2(urdf, srdf, 'pin')

# Sync model to robot: actuated joints
model.setJointPosition(robot_q_map)
model.update()

# The weld joints should already be at trajectory node-0. Re-sync from the
# current robot state before building CartesianInterface.
robot_q_map = robot.qToMap(robot.getPositionReferenceFeedback())
model.setJointPosition(robot_q_map)
model.update()

# ── Create standalone CartesianInterface ─────────────────────────────────────
print("[controller] Building standalone CartesianInterface …")
ci = pyci.CartesianInterface.MakeInstance(
    'OpenSot',
    CARTESIO_YAML.read_text().replace('distal_link: ee_F', f'distal_link: {ee_link}'),
    model,
    DT,
)

print(f"[controller] All tasks: {ci.getTaskList()}")
postural_task = ci.getTask('Postural')

ee_task = ci.getTask(ee_link)
ee_distal = ee_task.getDistalLink()
ee_base   = ee_task.getBaseLink()
print(f"[controller] Task '{ee_link}': {ee_distal} → {ee_base}")

# ── Feedback controller ───────────────────────────────────────────────────────
feedback_controller = GapFeedbackController(DT, MAX_Y_VEL, MAX_X_VEL)
if args.open_loop:
    print("[controller] Open-loop replay: /gap/pose_robot is not required.")

# ── Control loop ──────────────────────────────────────────────────────────────
t = 0.0
gap_pose_paused = False
input("[controller] Press Enter to start the control loop.")
while True:
    t0 = perf_counter()

    gap_pose_age_s = controller_ros.gap_pose_age_s()
    gap_pose_fresh = controller_ros.gap_pose_is_fresh(args.gap_pose_timeout)
    should_pause_for_gap = (
        not args.open_loop
        and (
            gap_pose_age_s is None
            or (args.stop_on_gap_loss and not gap_pose_fresh)
        )
    )
    if should_pause_for_gap:
        if not gap_pose_paused:
            age_text = (
                "never received"
                if gap_pose_age_s is None
                else f"stale for {gap_pose_age_s:.3f}s"
            )
            print(f"[controller] Lost /gap/pose_robot ({age_text}); pausing.")
            gap_pose_paused = True
        elapsed = perf_counter() - t0
        sleep(max(0.0, DT - elapsed))
        continue

    if gap_pose_paused:
        print("[controller] /gap/pose_robot available; resuming.")
        gap_pose_paused = False

    # Slow down the trajectory by scaling time
    t_traj = t / TRAJ_SLOWDOWN

    # ── Update postural reference from mat trajectory (slowdown applied) ─
    postural_map = {
        k: v
        for k, v in postural_trajectory.postural_map(t_traj).items()
        if k in commanded_jnames
    }
    postural_task.setReferencePosture(postural_map)

    # ── Sense actual robot state for feedback ──────────────────────────────
    robot.sense()
    q_map_robot = robot.qToMap(robot.getJointPosition())

    # ── model shadow updated with the robot state for ee_cur ────────────────
    model_shadow.setJointPosition(q_map_robot)
    model_shadow.update()
    ee_pose_cur = model_shadow.getPose(ee_distal, ee_base)
    ee_pos_cur = ee_pose_cur.translation

    ee_pose_des = ee_pose_from_postural(
        model_shadow, postural_map, ee_distal, ee_base)
    ee_pos_des = ee_pose_des.translation.copy()

    # ── Pipe-relative weld target in base_link frame ────────────────────────
    ee_pose_des_mod = ee_pose_des.copy()
    if args.open_loop:
        gap_y_axis_base = np.array([0.0, 1.0, 0.0])
        metrics = {
            'normal/error_m': 0.0,
            'normal/correction_m': 0.0,
            'normal/correction_velocity_mps': 0.0,
            'tangent/error_m': 0.0,
            'tangent/correction_m': 0.0,
            'tangent/correction_velocity_mps': 0.0,
        }
    else:
        gap_origin_base = controller_ros.gap_origin_base
        gap_axes_base = controller_ros.gap_axes_base
        gap_x_axis_base, gap_y_axis_base, gap_z_axis_base = gap_axes_base
        base_R_gap = np.column_stack(
            [gap_x_axis_base, gap_y_axis_base, gap_z_axis_base])
        gains = controller_ros.controller_gains()
        corrected_position, corrected_rotation, metrics = (
            feedback_controller.compute(
                postural_position=ee_pos_des,
                current_position=ee_pos_cur,
                weld_position_gap=weld_gap_trajectory.value(t_traj),
                weld_rotation_gap=weld_gap_orientation.matrix(t_traj),
                gap_origin_base=gap_origin_base,
                gap_rotation_base=base_R_gap,
                gains=gains,
                tangent_correction=args.tangent_correction,
            )
        )
        ee_pose_des_mod.translation = corrected_position
        ee_pose_des_mod.linear = corrected_rotation

    ee_task.setPoseReference(ee_pose_des_mod)

    # ── IK step — writes model.v ─────────────────────────────────────────
    ci.update(t, DT)

    # ── Integrate model state ─────────────────────────────────────────────
    model.setJointPosition(model.sum(model.q, model.v * DT))
    model.update()

    # ── Send to robot ─────────────────────────────────────────────────────
    solver_map = model.qToMap(model.getJointPosition())
    command_map = dict(q_map_robot)
    command_map.update({
        name: solver_map[name]
        for name in commanded_jnames
        if name in solver_map
    })
    robot.setPositionReference(robot.mapToQ(command_map))
    robot.move()

    # ── IK output EE pose ────────────────────────────────────────────────
    ee_pose_ik = model.getPose(ee_distal, ee_base)

    robot.sense()

    q_map_robot = robot.qToMap(robot.getJointPosition())
    model_shadow.setJointPosition(q_map_robot)
    model_shadow.update()
    ee_pose_cur = model_shadow.getPose(ee_distal, ee_base)
    linear_tracking_angle = rotation_error_angle(
        ee_pose_des_mod.linear,
        ee_pose_cur.linear,
    )
    metrics['orientation/error_deg'] = np.degrees(linear_tracking_angle)

    controller_ros.publish_controller_state(
        ee_pose_des,
        ee_pose_des_mod,
        ee_pose_ik,
        ee_pose_cur,
        ee_base,
        gap_y_axis_base,
        metrics,
    )

    t += DT

    # ── Pace the loop ─────────────────────────────────────────────────────
    elapsed = perf_counter() - t0
    sleep(max(0.0, DT - elapsed))
