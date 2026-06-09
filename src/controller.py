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

from pathlib import Path
from time import sleep, perf_counter

import numpy as np
from scipy.io import loadmat

from controller_ros import ControllerRosInterface
from utils.controller_geometry import (
    gap_tangent_axis,
    rotation_correction,
    rotation_with_y_axis,
)
from utils.controller_trajectory import (
    CyclicPosturalTrajectory,
    ee_pose_from_postural,
)

from xbot2_interface import pyxbot2_interface as xbi
from cartesian_interface import pyci

# ── Parameters ───────────────────────────────────────────────────────────────
TASK_NAME   = 'ee_F'   # CartesIO task name for the welding end-effector
DT          = 0.01     # [s]   controller dt (100 Hz)

# ── Homing ───────────────────────────────────────────────────────────────────
HOMING_DURATION = 5.0  # [s]  time to move from current pose to trajectory node-0

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

init_pos_robot = mat_data['initial_robot_pose'][0]
GAP_Y = - init_pos_robot[1]
print(f"[controller] Gap Y (from pipe center): {GAP_Y:.4f} m")

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

# casadi_kin_dyn prepends virtual/fixed joints ('universe', 'reference', …)
# that carry no DOF — strip them so len(_jnames) == _n_act
VIRTUAL = {'universe', 'reference'}
jnames  = [n for n in all_jnames if n not in VIRTUAL]

print(f"[controller] Trajectory: {N} nodes, dt={trj_dt:.4f}s, duration={trj_dur:.2f}s")
print(f"[controller] Actuated joint names ({len(jnames)}): {jnames}")

postural_trajectory = CyclicPosturalTrajectory(q_act, jnames, trj_dt)
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
prev_y_err = None            # gap-normal error at previous tick
prev_x_err = None            # gap-tangent error at previous tick
y_err_integral = 0.0
x_err_integral = 0.0
gap_reference_pos_robot = None
gap_reference_x_axis_robot = None
# print("[controller] Starting PD control loop …")

# ── Control loop ──────────────────────────────────────────────────────────────
t = 0.0
input("[controller] Press Enter to start the control loop.")
while True:
    t0 = perf_counter()

    # Slow down the trajectory by scaling time
    t_traj = t / TRAJ_SLOWDOWN

    # ── Update postural reference from mat trajectory (slowdown applied) ─
    postural_map = postural_trajectory.postural_map(t_traj)
    postural_task.setReferencePosture(postural_map)

    # ── Sense actual robot state for feedback ──────────────────────────────
    robot.sense()
    q_map_robot = robot.qToMap(robot.getJointPosition())

    # ── model shadow updated with the robot state for ee_cur ────────────────
    model_shadow.setJointPosition(q_map_robot)
    model_shadow.update()
    ee_pose_cur = model_shadow.getPose(ee_distal, ee_base)
    ee_pos_cur = ee_pose_cur.translation 

    ee_pose_des = ee_pose_from_postural(model_shadow, postural_map, ee_distal, ee_base)
    ee_pos_des = ee_pose_des.translation.copy()

    # NOW: read the gap position and orientation from ROS topics
    gap_pos_robot = controller_ros.gap_pos_robot     # gap position in robot frame, from ROS topic
    gap_y_axis_msg = controller_ros.gap_y_axis_robot # gap y-axis in robot frame, from ROS topic
    has_gap_axis = gap_y_axis_msg is not None

    gap_y_axis_robot = (gap_y_axis_msg if has_gap_axis else np.array([0.0, 1.0, 0.0]))
    gap_x_axis_robot = gap_tangent_axis(gap_y_axis_robot) # tangent along the pipe/gap line, in robot frame

    if gap_pos_robot is not None and gap_reference_pos_robot is None:
        gap_reference_pos_robot = gap_pos_robot.copy()
        gap_reference_x_axis_robot = gap_x_axis_robot.copy()

    # ── Gap-frame setpoints in base_link frame ──────────────────────────────
    # Normal coordinate: distance to the y-gap plane.
    # Tangent coordinate: position along the pipe/gap line.
    gap_point_robot = gap_pos_robot if gap_pos_robot is not None else ee_pos_des # point used for the gap plane can be either measured gap position (the planned EE position if the gap position is not available)
    gap_normal_coord = float(np.dot(gap_point_robot, gap_y_axis_robot)) # distance from gap plane along gap normal of planned EE position
    ee_normal_coord = float(np.dot(ee_pos_cur, gap_y_axis_robot)) # distance from gap plane along gap normal for current EE position

    # Tangent coordinate: position along the pipe/gap line.
    # If a reference is available, we compute the planned offset along the tangent from the reference position, 
    # and apply it to the current measured gap position to get the target tangent coordinate. 
    # This allows to keep the same planned tangent offset even if the gap moves.

    if gap_reference_pos_robot is not None:
        planned_x_offset = float(
            np.dot(ee_pos_des - gap_reference_pos_robot,
                   gap_reference_x_axis_robot)
        )
        gap_tangent_target_coord = float(
            np.dot(gap_point_robot, gap_x_axis_robot) + planned_x_offset
        )
    else:
        gap_tangent_target_coord = float(np.dot(ee_pos_des, gap_x_axis_robot))

    ee_tangent_coord = float(np.dot(ee_pos_cur, gap_x_axis_robot))

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
    postural_normal_coord = float(np.dot(ee_pos_des, gap_y_axis_robot))
    normal_delta = commanded_normal_coord - postural_normal_coord
    commanded_tangent_coord = ee_tangent_coord + vx_cmd * DT
    postural_tangent_coord = float(np.dot(ee_pos_des, gap_x_axis_robot))
    tangent_delta = commanded_tangent_coord - postural_tangent_coord
    ee_pose_des_mod.translation = (
        ee_pos_des
        + normal_delta * gap_y_axis_robot
        + tangent_delta * gap_x_axis_robot
    )
    # Align the EE local Y axis to the measured y-gap direction when available.
    # The sign is selected inside the helper to stay close to the postural pose.
    if has_gap_axis:
        ee_pose_des_mod.linear = rotation_with_y_axis(
            ee_pose_des.linear,
            gap_y_axis_robot,
        )

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
    robot.setPositionReference(model.getJointPosition())
    robot.move()

    # ── IK output EE pose ────────────────────────────────────────────────
    ee_pose_ik = model.getPose(ee_distal, ee_base)

    robot.sense()




























    q_map_robot = robot.qToMap(robot.getJointPosition())
    model_shadow.setJointPosition(q_map_robot)
    model_shadow.update()
    ee_pose_cur = model_shadow.getPose(ee_distal, ee_base)
    ee_measured_normal_coord = float(
        np.dot(ee_pose_cur.translation, gap_y_axis_robot))
    ee_measured_tangent_coord = float(
        np.dot(ee_pose_cur.translation, gap_x_axis_robot))
    translation_tracking_error = ee_pose_des_mod.translation - ee_pose_cur.translation
    translation_tracking_normal = float(
        np.dot(translation_tracking_error, gap_y_axis_robot))
    translation_tracking_tangent = float(
        np.dot(translation_tracking_error, gap_x_axis_robot))
    linear_tracking_angle, _ = rotation_correction(
        ee_pose_des_mod.linear,
        ee_pose_cur.linear,
    )

    controller_ros.publish_controller_state(
        ee_pose_des,
        ee_pose_des_mod,
        ee_pose_ik,
        ee_pose_cur,
        ee_base,
        gap_y_axis_robot,
        {
            'gap/normal_target_m': gap_normal_coord,
            'gap/normal_actual_m': ee_measured_normal_coord, 
            'gap/tangent_x_target_m': gap_tangent_target_coord,
            'gap/tangent_x_actual_m': ee_measured_tangent_coord,
            'error/normal_m': y_err,
            'error/tangent_x_m': x_err,
            'tracking/normal_m': translation_tracking_normal,
            'tracking/tangent_x_m': translation_tracking_tangent,
            'command/normal_velocity_mps': vy_cmd,
            'command/tangent_x_velocity_mps': vx_cmd,
            'gain/kp_normal': gains['kp_normal'],
            'gain/kd_normal': gains['kd_normal'],
            'gain/kp_tangent_x': gains['kp_tangent_x'],
            'gain/kd_tangent_x': gains['kd_tangent_x'],
            'command/orientation_correction_deg': np.degrees(linear_correction_angle),
            'tracking/orientation_error_deg': np.degrees(linear_tracking_angle),
        },
    )

    t += DT

    # ── Pace the loop ─────────────────────────────────────────────────────
    elapsed = perf_counter() - t0
    sleep(max(0.0, DT - elapsed))
