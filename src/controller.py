"""
CartesIO-based end-effector controller for the CONCERT welding robot.

Architecture:
  - RobotInterface2 senses the robot state and sends joint position references.
  - A standalone CartesianInterface (OpenSot) runs the IK locally each tick.
  - URDF/SRDF are read from robot_description_publisher ROS parameters.

Usage (simulation must already be running):
    ros2 launch acea_concert weld_sim.launch.py   # terminal 1
    python3 controller.py                          # terminal 2
"""

import math
import threading
from pathlib import Path
from time import sleep, perf_counter

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import rclpy

from utils.ros_utils import fetch_robot_description
from xbot2_interface import pyxbot2_interface as xbi
from xbot2_interface.pyaffine3 import Affine3
from cartesian_interface import pyci

# ── Parameters ───────────────────────────────────────────────────────────────
TASK_NAME   = 'ee_F'   # CartesIO task name for the welding end-effector
DT          = 0.01     # [s]   controller dt (100 Hz)

# ── Homing ───────────────────────────────────────────────────────────────────
HOMING_DURATION = 5.0  # [s]  time to move from current pose to trajectory node-0

# ── Trajectory (world X back-and-forth) ──────────────────────────────────────
TRJ_HALF    = 0.40     # [m]   half-stroke
TRJ_PERIOD  = 10.0     # [s]   period of one full cycle

# ── PD gains (world frame, per axis [x, y, z]) ───────────────────────────────
KP_XYZ = [1.0, 2.0, 1.0]    # proportional [1/s]
KD_XYZ = [0.05, 0.1, 0.05]  # derivative   [s]

# ── Trajectory slowdown factor ───────────────────────────────────────────────
TRAJ_SLOWDOWN = 12.0  # 1.0 = normal speed, 2.0 = half speed, etc.
# ── Gap Y target ─────────────────────────────────────────────────────────────
# Set to None to use the robot's initial EE Y as the gap level,
# or set explicitly, e.g. Y_GAP = 0.35 (world-frame metres).
Y_GAP: float | None = None

# Path to the CartesIO problem description YAML
CARTESIO_YAML = Path('/home/user/concert_ws/src/acea_concert/config/cartesio_stack.yaml')

# ── Mat file trajectory ───────────────────────────────────────────────────────
MAT_FILE = Path('/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat')

# --- Pipe and gap parameters (match weld_sim.launch.py) ---
GAP_Y = 0.0  # The gap plane is at y=0 in simulation

# ── Load weld_opt trajectory from mat file ────────────────────────────────────
print(f"[controller] Loading trajectory from {MAT_FILE} …")
mat_data = loadmat(str(MAT_FILE))

# q: (nq x N_nodes) — full model joint vector (floating base + actuated)
# joint_names: casadi_kin_dyn list — starts with virtual joints ('universe',
# 'reference', …) that have no rows in q; real actuated joints follow.
q_traj = mat_data['q']                          # shape (nq, N)

all_jnames = [str(n).strip() for n in mat_data['joint_names'].flatten()]

trj_dt  = float(np.atleast_1d(mat_data['dt']).flat[0])
N       = q_traj.shape[1]
trj_dur = (N - 1) * trj_dt

# The first 7 rows are floating-base (pos xyz + quat xyzw); rest are actuated.
n_base  = 7
q_act   = q_traj[n_base:, :]              # (n_actuated x N)
n_act   = q_act.shape[0]

# casadi_kin_dyn prepends virtual/fixed joints ('universe', 'reference', …)
# that carry no DOF — strip them so len(_jnames) == _n_act
VIRTUAL = {'universe', 'reference'}
jnames  = [n for n in all_jnames if n not in VIRTUAL]

print(f"[controller] Trajectory: {N} nodes, dt={trj_dt:.4f}s, duration={trj_dur:.2f}s")
print(f"[controller] Actuated joint names ({len(jnames)}): {jnames}")

# Build a per-joint interpolator (cyclic: forward then backward)
q_cycle = np.concatenate([q_act, q_act[:, ::-1]], axis=1)   # forward + backward
cycle_dur = 2 * trj_dur
t_nodes   = np.linspace(0.0, cycle_dur, q_cycle.shape[1])
trj_interp = interp1d(t_nodes, q_cycle, axis=1, kind='linear', fill_value='extrapolate')

def get_postural_map(t_global: float) -> dict:
    """Return {joint_name: angle} for all actuated joints at time t (cyclic)."""
    t_mod = t_global % cycle_dur
    q_now = trj_interp(t_mod)
    return {name: float(q_now[i]) for i, name in enumerate(jnames)}

def get_ee_pose_from_postural(model_target, postural_map: dict) -> Affine3:
    """
    Compute the EE pose (Affine3) for a given postural joint map, using a temporary model.
    """
    model_target.setJointPosition(postural_map)
    model_target.update()
    return model_target.getPose(ee_distal, ee_base)

print("[controller] Trajectory interpolator ready.")

# ── Read URDF/SRDF from robot_description_publisher ───────────────────────────
print("[controller] Waiting for robot_description ROS parameters …")
rclpy.init()

urdf, srdf = fetch_robot_description('controller_urdf_reader')
print("[controller] URDF and SRDF received.")

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


# ── Build ModelInterface2 for the standalone solver ───────────────────────────
model = xbi.ModelInterface2(urdf, srdf, 'pin')
model_shadow = xbi.ModelInterface2(urdf, srdf, 'pin')

# Sync model to robot: actuated joints
model.setJointPosition(robot_q_map)
model.update()

# ── Homing phase: move robot to trajectory node-0 ────────────────────────────
# q_home_act: actuated joint targets from node-0 of the mat trajectory
q_home_act = q_act[:, 0]  # (n_actuated,)
q_home_map  = {name: float(q_home_act[i]) for i, name in enumerate(jnames)}

# Current actuated joint positions (from robot sense)
q_cur_map = robot.qToMap(robot.getPositionReferenceFeedback())

# Only home the joints that exist both in the trajectory and on the robot
home_joints = {k: v for k, v in q_home_map.items() if k in q_cur_map}
print(f"[controller] Homing to node-0 over {HOMING_DURATION}s …")

homing_steps = int(HOMING_DURATION / DT)
for step in range(homing_steps):
    alpha = (step + 1) / homing_steps          # 0 → 1 linear ramp
    interp_map = {
        k: (1.0 - alpha) * q_cur_map[k] + alpha * home_joints[k]
        for k in home_joints
    }
    robot.setPositionReference(robot.mapToQ(interp_map))
    robot.move()
    sleep(DT)

i = 0
for i in range(10):  # extra sense cycles at the end for better convergence
    robot.sense()
    i+=1

motor_pos = robot.getJointPosition() # no getMotorPosition()
motor_map = robot.qToMap(motor_pos)
for name, j in motor_map.items():
    print(f"{name}: {j}")
# Joint error (for actuated joints in home_joints)
joint_err = {k: interp_map[k] - motor_map[k] for k in home_joints}

# Pretty print joint errors
print("[JOINT ERROR]")
for joint, err in joint_err.items():
    print(f"    {joint:20s}: {err:+.6f}")

# EE error
model.setJointPosition(interp_map)
model.update()
ee_pos_des = model.getPose('ee_F', 'world').translation

model.setJointPosition(motor_map)
model.update()
ee_pos_cur = model.getPose('ee_F', 'world').translation

ee_err = ee_pos_des - ee_pos_cur
print(f"[EE ERROR] err={ee_err}, |err|={np.linalg.norm(ee_err)}")

print("[controller] Homing complete.")


# Re-sync model to the post-homing robot state for building the CartesianInterface
robot_q_map = robot.qToMap(robot.getPositionReferenceFeedback())
model.setJointPosition(robot_q_map)
model.update()

# ── Create standalone CartesianInterface ─────────────────────────────────────
print("[controller] Building standalone CartesianInterface …")
ci = pyci.CartesianInterface.MakeInstance(
    'OpenSot',
    CARTESIO_YAML.read_text(),
    model,
    DT,
)

print(f"[controller] All tasks: {ci.getTaskList()}")
postural_task = ci.getTask('Postural')

ee_task = ci.getTask(TASK_NAME)
ee_distal = ee_task.getDistalLink()
ee_base   = ee_task.getBaseLink()
print(f"[controller] Task '{TASK_NAME}': {ee_base} → {ee_distal}")

# ── Get XZ plane for the welding ─────────────────────────────────────────────
T_start = model.getPose(ee_distal, ee_base)

# Resolve gap Y: use startup Y if not set explicitly
gap_y = GAP_Y
print(f"[controller] Gap Y (from pipe center): {gap_y:.4f} m")

# ── PD state ─────────────────────────────────────────────────────────────────
prev_pos_err = np.zeros(3)   # world-frame position error at previous tick

# print("[controller] Starting PD control loop …")

# ── Control loop ──────────────────────────────────────────────────────────────
t = 0.0
input("[controller] Press Enter to start the control loop.")
while True:
    t0 = perf_counter()

    # Slow down the trajectory by scaling time
    t_traj = t / TRAJ_SLOWDOWN

    # ── Update postural reference from mat trajectory (slowdown applied) ─
    postural_map = get_postural_map(t_traj)
    postural_task.setReferencePosture(postural_map)

    # ── Compute desired EE pose from postural (XZ and orientation from postural, Y clamped)
    ee_pose_des = get_ee_pose_from_postural(model_shadow, postural_map)
    # ee_pose_des.translation[1] = gap_y  # Clamp Y to the gap

     # ── PD law (world frame) ──────────────────────────────────────────────    
    ee_pose_cur = model.getPose(ee_distal, ee_base)
    y_cur = ee_pose_cur.translation[1]


    robot.sense()
    q_map_robot = robot.qToMap(robot.getJointPosition())
    model_shadow.setJointPosition(q_map_robot)
    model_shadow.update()
    ee_pose_cur = model_shadow.getPose(ee_distal, ee_base)
    ee_pos_cur = ee_pose_cur.translation

    # ── (Optional) Estimate y_dot_cur (numerical derivative)
    # if 'y_prev' not in locals():
        # y_prev = y_cur
        # y_dot_cur = 0.0
    # else:
        # y_dot_cur = (y_cur - y_prev) / DT
        # y_prev = y_cur

    # ── PD control for Y
    # pos_err  = prev_pos_err
    # vel_err  = (pos_err - prev_pos_err) / DT   # numerical derivative of error
    # prev_pos_err = pos_err.copy()

    # KP = np.array(KP_XYZ)
    # KD = np.array(KD_XYZ)

    # Cartesian velocity command [vx, vy, vz] in world frame
    # v_cmd = np.array([0.0, 0.0, 0.0]) \
            # + KP * pos_err \
            # + KD * vel_err

    # ── Send pose and velocity reference
    ee_task.setPoseReference(ee_pose_des)
    # ee_task.setVelocityReference(v_cmd)

    # ── IK step — writes model.v ─────────────────────────────────────────
    ci.update(t, DT)

    # ── Integrate model state ─────────────────────────────────────────────
    model.setJointPosition(model.sum(model.q, model.v * DT))
    model.update()

    # ── Send to robot ─────────────────────────────────────────────────────
    robot.setPositionReference(model.getJointPosition())
    robot.move()

    # --- Print error between actual and desired EE pose (position only) ---
    ee_pos_des = ee_pose_des.translation
    ee_pos_cur = ee_pose_cur.translation
    ee_err = ee_pos_des - ee_pos_cur
    print(f"[EE DES] [{ee_pos_des[0]:.4f}, {ee_pos_des[1]:.4f}, {ee_pos_des[2]:.4f}] m")
    print(f"[EE CUR] [{ee_pos_cur[0]:.4f}, {ee_pos_cur[1]:.4f}, {ee_pos_cur[2]:.4f}] m")
    print(f"[EE ERR] [{ee_err[0]:+.4f}, {ee_err[1]:+.4f}, {ee_err[2]:+.4f}] m")

    t += DT

    # ── 13) Pace the loop ─────────────────────────────────────────────────────
    elapsed = perf_counter() - t0
    sleep(max(0.0, DT - elapsed))
