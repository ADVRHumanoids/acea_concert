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
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped, QuaternionStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header, Float64MultiArray

from utils.ros_utils import fetch_robot_description
from xbot2_interface import pyxbot2_interface as xbi
from xbot2_interface.pyaffine3 import Affine3
from cartesian_interface import pyci

import scipy, os
# ── Parameters ───────────────────────────────────────────────────────────────
TASK_NAME   = 'ee_F'   # CartesIO task name for the welding end-effector
DT          = 0.01     # [s]   controller dt (100 Hz)

# ── Homing ───────────────────────────────────────────────────────────────────
HOMING_DURATION = 5.0  # [s]  time to move from current pose to trajectory node-0

# ── Trajectory (world X back-and-forth) ──────────────────────────────────────
TRJ_HALF    = 0.40     # [m]   half-stroke
TRJ_PERIOD  = 10.0     # [s]   period of one full cycle

# ── PD gains (world frame, per axis [x, y, z]) ───────────────────────────────
KP_XYZ = [1.0, 30.0, 1.0]    # proportional [1/s]
KD_XYZ = [0.05,  2.0, 0.05]  # derivative   [s]
KI_XYZ = [1.0, 0.1, 1.0]  # integral   [s]
MAX_Y_VEL = 10.0          # [m/s] cap for the gap correction velocity

# ── Trajectory slowdown factor ───────────────────────────────────────────────
TRAJ_SLOWDOWN = 12.0  # 1.0 = normal speed, 2.0 = half speed, etc.

# ── Gap Y target ─────────────────────────────────────────────────────────────
# GAP_Y is computed from the mat-file initial robot Y so it matches the pipe
# gap placement used by weld_sim.launch.py.

# Path to the CartesIO problem description YAML
CARTESIO_YAML = Path('/home/user/concert_ws/src/acea_concert/config/cartesio_stack.yaml')

# ── Mat file trajectory ───────────────────────────────────────────────────────
MAT_FILE = Path('/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat')

# --- Pipe and gap parameters (match weld_sim.launch.py) ---

# ── Load weld_opt trajectory from mat file ────────────────────────────────────
print(f"[controller] Loading trajectory from {MAT_FILE} …")
mat_data = loadmat(str(MAT_FILE))

init_pos_robot = mat_data['initial_robot_pose'][0]
GAP_Y = - init_pos_robot[1]
print(f"[controller] Gap Y (from pipe center): {GAP_Y:.4f} m")

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

# ── Publishers for PlotJuggler ─────────────────────────────────────────────
_plot_node = rclpy.create_node('controller_plot')
_pub_des  = _plot_node.create_publisher(PoseStamped, '/ee/desired', 10)
_pub_sent = _plot_node.create_publisher(PoseStamped, '/ee/sent',    10)
_pub_cur  = _plot_node.create_publisher(PoseStamped, '/ee/current', 10)
_pub_ik   = _plot_node.create_publisher(PoseStamped,  '/ee/ik',            10)
_pub_js   = _plot_node.create_publisher(JointState,        '/controller/joints', 10)
_pub_ctrl = _plot_node.create_publisher(Float64MultiArray, '/ee/controller',     10)

def _publish_controller(y_err, y_err_dot, y_err_integral, vy_cmd, y_des, y_cur):
    """Publish P-controller signals as Float64MultiArray.
    Layout: [y_err, y_err_dot, y_err_integral, vy_cmd, y_des, y_cur]
    """
    msg = Float64MultiArray()
    msg.data = [float(y_err), float(y_err_dot), float(y_err_integral),
                float(vy_cmd), float(y_des), float(y_cur)]
    _pub_ctrl.publish(msg)

def _publish_joint_state(pub, joint_map):
    """Publish a dict {name: position} as JointState."""
    msg = JointState()
    msg.header = Header()
    msg.header.stamp = _plot_node.get_clock().now().to_msg()
    msg.name     = list(joint_map.keys())
    msg.position = [float(v) for v in joint_map.values()]
    pub.publish(msg)

def _publish_pose(pub, affine):
    """Publish an Affine3 as PoseStamped (position + quaternion)."""
    msg = PoseStamped()
    msg.header = Header()
    msg.header.stamp = _plot_node.get_clock().now().to_msg()
    msg.header.frame_id = 'world'
    xyz = affine.translation
    q   = affine.quaternion  # [x, y, z, w]
    msg.pose.position.x = float(xyz[0])
    msg.pose.position.y = float(xyz[1])
    msg.pose.position.z = float(xyz[2])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    pub.publish(msg)
# ──────────────────────────────────────────────────────────────────────────────

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

robot.sense()

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
ee_pose_des = model.getPose('ee_F', 'world')

model.setJointPosition(motor_map)
model.update()
ee_pose_cur = model.getPose('ee_F', 'world')

ee_err = ee_pose_des.translation - ee_pose_cur.translation
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
print(f"[controller] Task '{TASK_NAME}': {ee_distal} → {ee_base}")

# ── initial pose  ────────────────────────────────────────────────────────────

initial_ee_pose = model.getPose(ee_distal, ee_base).copy()

# ── PD state ─────────────────────────────────────────────────────────────────
prev_y_err = None            # world-frame Y error at previous tick
y_err_integral = 0.0
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

    # ── Sense actual robot state for feedback ──────────────────────────────
    robot.sense()
    q_map_robot = robot.qToMap(robot.getJointPosition())
    model_shadow.setJointPosition(q_map_robot)
    model_shadow.update()
    ee_pose_cur = model_shadow.getPose(ee_distal, ee_base)
    ee_pos_cur = ee_pose_cur.translation 

    ee_pose_des = get_ee_pose_from_postural(model_shadow, postural_map)
    ee_pos_des = ee_pose_des.translation

    ## test: add a sinusoidal offset in world Y to the EE reference, to simulate a gap correction
    # world_y_in_ee = np.array([0.0, 1.0, 0.0]) #  initial_ee_pose.linear.T @
    # Y_TEST_AMP = 0.1
    # y_test_offset = Y_TEST_AMP * math.sin(2 * math.pi * t / TRJ_PERIOD) * world_y_in_ee
    # ee_pose_des.translation += y_test_offset

    # ── Y P correction ─
    y_err = (ee_pos_des - ee_pos_cur)[1]
    y_err_dot = 0.0 if prev_y_err is None else (y_err - prev_y_err) / DT
    y_err_integral += y_err * DT
    prev_y_err = y_err

    vy_cmd = KP_XYZ[1] * y_err  + KD_XYZ[1] * y_err_dot #+ KI_XYZ[1] * y_err_integral
    vy_cmd = float(np.clip(vy_cmd, -MAX_Y_VEL, MAX_Y_VEL))
    # Closed-loop tracking: move from current Y toward desired Y
    ee_pose_des_mod = ee_pose_des.copy()
    ee_pose_des_mod.translation[1] = ee_pos_des[1] + vy_cmd * DT

    ee_task.setPoseReference(ee_pose_des)

    # ── IK step — writes model.v ─────────────────────────────────────────
    ci.update(t, DT)

    # ── Integrate model state ─────────────────────────────────────────────
    model.setJointPosition(model.sum(model.q, model.v * DT))
    model.update()

    # ── Send to robot ─────────────────────────────────────────────────────
    robot.setPositionReference(model.getJointPosition())
    robot.move()

    # ── Publish model joint trajectory ───────────────────────────────────
    # _publish_joint_state(_pub_js, model.qToMap(model.getJointPosition()))

    # ── IK output EE pose ────────────────────────────────────────────────
    ee_pose_ik = model.getPose(ee_distal, ee_base)

    robot.sense()
    q_map_robot = robot.qToMap(robot.getJointPosition())
    model_shadow.setJointPosition(q_map_robot)
    model_shadow.update()
    ee_pose_cur = model_shadow.getPose(ee_distal, ee_base)

    # ── Publish to PlotJuggler topics ─────────────────────────────────────
    _publish_pose(_pub_des,  ee_pose_des)
    _publish_pose(_pub_ik,   ee_pose_ik)
    _publish_pose(_pub_cur,  ee_pose_cur)
    _publish_controller(y_err, y_err_dot, y_err_integral, vy_cmd,
                        ee_pos_des[1], ee_pos_cur[1])

    t += DT

    # ── Pace the loop ─────────────────────────────────────────────────────
    elapsed = perf_counter() - t0
    sleep(max(0.0, DT - elapsed))
