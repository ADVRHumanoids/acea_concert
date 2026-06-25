"""
CartesIO-based end-effector controller for the CONCERT welding robot.

Architecture:
  - RobotInterface2 senses the robot state and sends joint position references.
  - A standalone CartesianInterface (OpenSot) runs the IK locally each tick.
  - URDF/SRDF are read from robot_description_publisher ROS parameters.

Usage (simulation must already be running and homing already completed):
    ros2 launch acea_concert weld_sim.launch.py
    ros2 run acea_concert home_to_weld_start.py
    python3 controller.py
"""

from pathlib import Path
from time import sleep, perf_counter

import numpy as np
from scipy.io import loadmat

from controller_ros import ControllerRosInterface
from utils.controller_geometry import (
    rotation_correction,
)
from utils.controller_trajectory import (
    CyclicPosturalTrajectory,
    CyclicQuaternionTrajectory,
    CyclicVectorTrajectory,
    ee_pose_from_postural,
)

from xbot2_interface import pyxbot2_interface as xbi
from cartesian_interface import pyci

# ── Parameters ───────────────────────────────────────────────────────────────
TASK_NAME   = 'ee_F'   # CartesIO task name for the welding end-effector
DT          = 0.01     # [s]   controller dt (100 Hz)

# The MAT trajectory contains the whole robot, but this controller should only
# command the welding axes. A-D wheel modules are owned by omnisteering.
WELD_JOINTS = (
    'J1_E', 'J2_E',
    'J1_F', 'J2_F', 'J3_F', 'J4_F', 'J5_F', 'J6_F',
)

# ── Trajectory (world X back-and-forth) ──────────────────────────────────────
TRJ_HALF    = 0.40     # [m]   half-stroke
TRJ_PERIOD  = 10.0     # [s]   period of one full cycle

# ── PD gains (gap frame: x=tangent along pipe, y=gap normal) ────────────────
KP_XYZ = [30.0, 30.0, 1.0]    # proportional [1/s]
KD_XYZ = [2.0,  2.0, 0.05]  # derivative   [s]
KI_XYZ = [1.0, 0.1, 1.0]  # integral   [s]
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

# Path to the CartesIO problem description YAML
CARTESIO_YAML = Path('/home/user/concert_ws/src/acea_concert/config/cartesio_stack.yaml')

# ── Mat file trajectory ───────────────────────────────────────────────────────
MAT_FILE = Path('/home/user/concert_ws/src/acea_concert/mat_files/weld_concert.mat')

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
n_act   = q_act.shape[0]
weld_pos_traj_gap = np.asarray(
    mat_data['desired_traj_weld_pos_gap'], dtype=float)
weld_quat_traj_gap = np.asarray(
    mat_data['desired_traj_weld_quat_gap'], dtype=float)

# casadi_kin_dyn prepends virtual/fixed joints ('universe', 'reference', …)
# that carry no DOF — strip them so len(_jnames) == _n_act
VIRTUAL = {'universe', 'reference'}
jnames  = [n for n in all_jnames if n not in VIRTUAL]

weld_jnames = [name for name in jnames if name in WELD_JOINTS]

print(f"[controller] Trajectory: {N} nodes, dt={trj_dt:.4f}s, duration={trj_dur:.2f}s")
print(f"[controller] Actuated joint names ({len(jnames)}): {jnames}")
print(f"[controller] Weld joints from trajectory ({len(weld_jnames)}): {weld_jnames}")

postural_trajectory = CyclicPosturalTrajectory(q_act, jnames, trj_dt)
weld_gap_trajectory = CyclicVectorTrajectory(weld_pos_traj_gap, trj_dt) # desired EE/weld position expressed in the gap frame
weld_gap_orientation = CyclicQuaternionTrajectory(weld_quat_traj_gap, trj_dt) # desired EE orientation expressed in the gap frame
print("[controller] Trajectory interpolator ready.")

# ── Read URDF/SRDF from robot_description_publisher ───────────────────────────
print("[controller] Waiting for robot_description ROS parameters …")
controller_ros = ControllerRosInterface(GAIN_PARAM_DEFAULTS)

urdf, srdf = controller_ros.robot_description()
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
prev_y_err = None            # gap-normal error at previous tick
prev_x_err = None            # gap-tangent error at previous tick
y_err_integral = 0.0
x_err_integral = 0.0
# print("[controller] Starting PD control loop …")

# ── Control loop ──────────────────────────────────────────────────────────────
t = 0.0
input("[controller] Press Enter to start the control loop.")
while True:
    t0 = perf_counter()

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

    # Keep wheel/steering joints at the measured state before each IK step.
    model.setJointPosition(q_map_robot)
    model.update()

    # ── model shadow updated with the robot state for ee_cur ────────────────
    model_shadow.setJointPosition(q_map_robot)
    model_shadow.update()
    ee_pose_cur = model_shadow.getPose(ee_distal, ee_base)
    ee_pos_cur = ee_pose_cur.translation 

    ee_pose_des = ee_pose_from_postural(model_shadow, postural_map, ee_distal, ee_base)
    ee_pos_des = ee_pose_des.translation.copy()

    # NOW: read the measured gap pose and full gap frame from ROS topics.
    gap_origin_base = controller_ros.gap_origin_base
    gap_axes_base = controller_ros.gap_axes_base
    has_gap_axes = gap_axes_base is not None

    if has_gap_axes:
        gap_x_axis_base, gap_y_axis_base, gap_z_axis_base = gap_axes_base
    else:
        gap_x_axis_base = np.array([1.0, 0.0, 0.0])
        gap_y_axis_base = np.array([0.0, 1.0, 0.0])
        gap_z_axis_base = np.array([0.0, 0.0, 1.0])

    base_R_gap = np.column_stack([gap_x_axis_base, gap_y_axis_base, gap_z_axis_base])

    # ── Pipe-relative weld target in base_link frame ────────────────────────
    weld_pos_des_gap = weld_gap_trajectory.value(t_traj)
    if gap_origin_base is not None:
        weld_target_pos_base = gap_origin_base + base_R_gap @ weld_pos_des_gap
    else:
        weld_target_pos_base = ee_pos_des

    # Gap-frame setpoints: normal is across the pipes, tangent is along the
    # planned weld direction.
    gap_normal_coord = float(np.dot(weld_target_pos_base, gap_y_axis_base))
    ee_normal_coord = float(np.dot(ee_pos_cur, gap_y_axis_base))

    gap_tangent_target_coord = float(np.dot(weld_target_pos_base, gap_x_axis_base))
    ee_tangent_coord = float(np.dot(ee_pos_cur, gap_x_axis_base))

    ## add a sinusoidal offset in world Y to the EE reference, to simulate a gap correction
    # world_y_in_ee = np.array([0.0, 1.0, 0.0]) #  initial_ee_pose.linear.T @
    # Y_TEST_AMP = 0.1
    # y_test_offset = Y_TEST_AMP * math.sin(2 * math.pi * t / TRJ_PERIOD) * world_y_in_ee
    # ee_pose_des.translation += y_test_offset

    gains = controller_ros.controller_gains()

    # ── PD correction toward the y-gap plane ─────────────────────────────────
    y_err = gap_normal_coord - ee_normal_coord # gap_normal_coord is the desired position of the EE along the gap normal, ee_normal_coord is the current position of the EE along the gap normal, so their difference is how much we need to correct to get to the desired position
    y_err_dot = 0.0 if prev_y_err is None else (y_err - prev_y_err) / DT
    y_err_integral += y_err * DT
    prev_y_err = y_err

    vy_cmd = (
        gains['kp_normal'] * y_err
        + gains['kd_normal'] * y_err_dot
        # + KI_XYZ[1] * y_err_integral
    )
    vy_cmd = float(np.clip(vy_cmd, -MAX_Y_VEL, MAX_Y_VEL))

    x_err = gap_tangent_target_coord - ee_tangent_coord
    x_err_dot = 0.0 if prev_x_err is None else (x_err - prev_x_err) / DT
    x_err_integral += x_err * DT
    prev_x_err = x_err

    vx_cmd = (
        gains['kp_tangent_x'] * x_err
        + gains['kd_tangent_x'] * x_err_dot
        # + KI_XYZ[0] * x_err_integral
    )
    vx_cmd = float(np.clip(vx_cmd, -MAX_X_VEL, MAX_X_VEL))

    # Apply both corrections in the gap frame: normal keeps the tool centered
    # between the pipes, tangent keeps it on the moving/rotated gap line.
    ee_pose_des_mod = ee_pose_des.copy()
    commanded_normal_coord = ee_normal_coord + vy_cmd * DT
    postural_normal_coord = float(np.dot(ee_pos_des, gap_y_axis_base))
    normal_delta = commanded_normal_coord - postural_normal_coord
    
    commanded_tangent_coord = ee_tangent_coord + vx_cmd * DT
    postural_tangent_coord = float(np.dot(ee_pos_des, gap_x_axis_base))
    tangent_delta = commanded_tangent_coord - postural_tangent_coord
    ee_pose_des_mod.translation = (
        ee_pos_des
        + normal_delta * gap_y_axis_base
        + tangent_delta * gap_x_axis_base
    )
    
    # Apply the optimized tool orientation relative to the current gap frame.
    if has_gap_axes:
        gap_R_ee_des = weld_gap_orientation.matrix(t_traj)
        ee_pose_des_mod.linear = base_R_gap @ gap_R_ee_des

    linear_correction_angle, _ = rotation_correction(
        ee_pose_des_mod.linear,
        ee_pose_des.linear,
    )

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
    ee_measured_normal_coord = float(
        np.dot(ee_pose_cur.translation, gap_y_axis_base))
    ee_measured_tangent_coord = float(
        np.dot(ee_pose_cur.translation, gap_x_axis_base))
    linear_tracking_angle, _ = rotation_correction(
        ee_pose_des_mod.linear,
        ee_pose_cur.linear,
    )
    gap_origin_normal_coord = (
        float(np.dot(gap_origin_base, gap_y_axis_base))
        if gap_origin_base is not None else 0.0
    )

    controller_ros.publish_controller_state(
        ee_pose_des,
        ee_pose_des_mod,
        ee_pose_ik,
        ee_pose_cur,
        ee_base,
        gap_y_axis_base,
        {
            'gap/origin position along normal [m]': gap_origin_normal_coord,
            'normal tracking/target position [m]': gap_normal_coord,
            'normal tracking/ee measured position [m]': ee_measured_normal_coord,
            'normal tracking/error target minus ee [m]': y_err,
            'normal tracking/command velocity [m/s]': vy_cmd,
            'tangent tracking/target position [m]': gap_tangent_target_coord,
            'tangent tracking/ee measured position [m]': ee_measured_tangent_coord,
            'tangent tracking/error target minus ee [m]': x_err,
            'tangent tracking/command velocity [m/s]': vx_cmd,
            'orientation/commanded change from nominal [deg]': np.degrees(linear_correction_angle),
            'orientation/tracking error commanded vs ee [deg]': np.degrees(linear_tracking_angle),
        },
    )

    t += DT

    # ── Pace the loop ─────────────────────────────────────────────────────
    elapsed = perf_counter() - t0
    sleep(max(0.0, DT - elapsed))
