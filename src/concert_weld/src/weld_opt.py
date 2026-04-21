from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils import utils
from pathlib import Path
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
from scipy.spatial.transform import Rotation
# from xbot_interface import config_options as co
# from xbot_interface import xbot_interface as xbot
from geometry_msgs.msg import Vector3
from scipy.spatial.transform import Rotation as R
from horizon.ros import replay_trajectory

import casadi as cs
import numpy as np

import subprocess


'''
Initialize Horizon problem
'''
ns = 30
T = 1.5
dt = T / ns

prb = Problem(ns, receding=True, casadi_type=cs.SX)
prb.setDt(dt)

PATH_TO_CONCERT_WS = Path("/home/user/concert_ws")
modular_prismatic = PATH_TO_CONCERT_WS/"src"/"concert_description"/"concert_examples"/"src"/"concert_prismatic.py"
horizon_config = PATH_TO_CONCERT_WS/"src"/"concert_weld"/"config"/"weld.yaml"

urdf = subprocess.check_output(["python3", str(modular_prismatic), "-o", "urdf"], text=True)

# Launch robot_state_publisher in background with the URDF
rsp_process = subprocess.Popen(
    ["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args", "-p", f"robot_description:={urdf}"],
)

kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)
print(f"joint names: {kin_dyn.joint_names()}")

# Apply circular trajectory for ee_F
from circular_trajectory import generate_circular_trajectory

length_pipe = 5.0
center_pipe = [1.5, 0.0, 0.25]
radius_pipe = 0.5
angle_weld_start = 1/3 *np.pi
angle_weld_end = 1/3 *np.pi # np.pi/3 #2 * np.pi

# Generate trajectory and get initial desired pose
position, orientation = generate_circular_trajectory(
    ns,
    center=center_pipe,
    radius=radius_pipe,
    angle_start=angle_weld_start,
    angle_end=angle_weld_end,
)

# Set base_init so base is under the first trajectory point
base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Y (no lateral offset)

# Keep q_init as before, but you can tune these if needed
q_init = {
    'J1_A': 0.0,
    'J_wheel_A': 0.0,
    'J1_B': 0.0,
    'J_wheel_B': 0.0,
    'J1_C': 0.0,
    'J_wheel_C': 0.0,
    'J1_D': 0.0,
    'J_wheel_D': 0.0,
    'J1_E': -np.pi/2,
    'J2_E': 0.0,
    'J1_F': 0.0,
    'J2_F': 0.0,
    'J3_F': 0.0,
}

model = FullModelInverseDynamics(problem=prb,
                                 kd=kin_dyn,
                                 q_init=q_init,
                                 base_init=base_init
                                 )

ti = TaskInterface(prb=prb, model=model)
ti.setTaskFromYaml(horizon_config)

# Print initial FK vs desired for user awareness (after model and ti are defined)
fk_ee_pos = kin_dyn.fk('ee_F')(q=model.q0)['ee_pos'][:, 0]
fk_ee_rot = R.from_matrix((kin_dyn.fk('ee_F')(q=model.q0)['ee_rot'].full())).as_quat()
desired_pos = position[:, 0]
desired_rot = orientation[:, 0]


print(f"[INFO] Desired initial ee_F pos: {desired_pos}, rot (quat): {desired_rot}")
print(f"[INFO] Actual initial ee_F pos: {fk_ee_pos}, rot (quat): {fk_ee_rot}")

# Kill any existing rviz_markers.py processes before starting a new one
import subprocess
subprocess.run("pkill -f rviz_markers.py", shell=True)

# Draw pipe cylinder in RViz (as background subprocess)
rviz_marker_proc = subprocess.Popen([
    "python3", str(PATH_TO_CONCERT_WS / "src" / "concert_weld" / "src" / "rviz_markers.py"),
    str(center_pipe[0]), str(center_pipe[1]), str(center_pipe[2]),
    str(radius_pipe), str(length_pipe)
])

position_aug = np.full((7, ns + 1), 0.0)
position_aug[:3, :] = position

orientation_aug = np.full((7, ns + 1), 0.0)
orientation_aug[3:, :] = orientation

pos_task_name = 'ee_pos'
ori_task_name = 'ee_ori'

ee_pos_task = ti.getTask(pos_task_name)
ee_ori_task = ti.getTask(ori_task_name)

ee_pos_task.setRef(position_aug)
ee_ori_task.setRef(orientation_aug)

print(f"circular trajectory applied: center={center_pipe}, radius={radius_pipe}")
print(f"angle range: [{angle_weld_start}, {angle_weld_end}] rad, steps: {ns + 1}")

# Set base pose in XZ plane and pitch-only orientation (pitch=0)
tmp_q0 = ti.model.q0.copy()
tmp_q0[1] = 0.0  # Y 
tmp_q0[2] = 0.0  # Z (set to desired height if needed)
pitch_angle = 0.0  # Set to desired pitch in radians
base_quat = R.from_euler('y', pitch_angle).as_quat()  # [x, y, z, w]
tmp_q0[3:7] = base_quat  # Set quaternion part

ti.model.q0 = tmp_q0

ti.model.q[1].setBounds(tmp_q0[1], tmp_q0[1])
ti.model.q[2].setBounds(tmp_q0[2], tmp_q0[2])
ti.model.q[3:7].setBounds(tmp_q0[3:7], tmp_q0[3:7])

base_pos_x = prb.createSingleVariable('base_pos_x', 1)

prb.createConstraint('base_pos_x_constraint', model.q[0] - base_pos_x)

ti.model.q.setInitialGuess(ti.model.q0)
ti.model.v.setInitialGuess(ti.model.v0)

ti.model.q[3:].setBounds(kin_dyn.q_min()[3:], kin_dyn.q_max()[3:])

# prb.createResidual('max_q', 1e1 * utils.utils.barrier(kin_dyn.q_max()[7:] - model.q[7:]))
# prb.createResidual('min_q', 1e1 * utils.utils.barrier1(kin_dyn.q_min()[7:] - model.q[7:]))

# vel_lims = model.kd.velocityLimits()
# prb.createResidual('max_vel', 1e2 * utils.utils.barrier(vel_lims[7:] - model.v[7:]))
# prb.createResidual('min_vel', 1e1 * utils.utils.barrier1(-1 * vel_lims[7:] - model.v[7:]))

ti.finalize()

ti.bootstrap()
solution = ti.solution


base_pos_x_opt = solution['base_pos_x']
print(f"Optimized base x position: {base_pos_x_opt}")


contact_list_repl = list(model.cmap.keys())
repl = replay_trajectory.replay_trajectory(dt, model.kd.joint_names(), solution['q'],
                                           {k: None for k in model.fmap.keys()},
                                           model.kd_frame, model.kd,
                                           trajectory_markers=contact_list_repl,
                                           future_trajectory_markers={'ee_F': 'world'})

repl.replay()