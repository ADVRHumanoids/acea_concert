from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils import mat_storer
from horizon.utils import kin_dyn as kin_dyn_utils

import os, sys

from pathlib import Path
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn

from geometry_msgs.msg import Vector3
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat
from horizon.ros import replay_trajectory

from collision_checker import CollisionChecker
import casadi as cs
import numpy as np

import subprocess


'''
Initialize Horizon problem
'''
ns = 30
T = 2.0
dt = T / ns

prb = Problem(ns, receding=True, casadi_type=cs.SX)
prb.setDt(dt)

# ========================================================
# ========================================================

PATH_TO_CONCERT_WS = Path("/home/user/concert_ws")
PATH_TO_CONCERT_WELD = PATH_TO_CONCERT_WS/"src"/"concert_weld"
modular_prismatic = PATH_TO_CONCERT_WS/"src"/"concert_description"/"concert_examples"/"src"/"concert_prismatic.py"
horizon_config = PATH_TO_CONCERT_WS/"src"/"concert_weld"/"config"/"weld.yaml"

urdf = subprocess.check_output(["python3", str(modular_prismatic), "-o", "urdf"], text=True)
srdf = subprocess.check_output(["python3", str(modular_prismatic), "-o", "srdf"], text=True)

with open('/tmp/concert_weld.urdf', 'w') as f:
    f.write(urdf)

with open('/tmp/concert_weld.srdf', 'w') as f:
    f.write(srdf)

subprocess.run(f'moveit_compute_default_collisions --urdf_path /tmp/concert_weld.urdf --srdf_path /tmp/concert_weld.srdf', shell=True, check=True)

with open('/tmp/concert_weld.srdf', 'r') as f:
    srdf = f.read()

# ========================================================
# ========================================================

# Launch robot_state_publisher in background with the URDF
rsp_process = subprocess.Popen(
    ["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args", "-p", f"robot_description:={urdf}"],
)

kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)
# print(f"joint names: {kin_dyn.joint_names()}")

# Apply circular trajectory for ee_F
from circular_trajectory import generate_circular_trajectory
footprint_robot_x = 1.2
footprint_robot_y = 0.7

length_pipe = 5.0
pos_center_pipe = [1.5, 0.0, 0.75]
orientation_pipe = [0.7071068, 0.0, 0.0, 0.7071068]
radius_pipe = 0.5
# angle_weld_start = 1/2 *np.pi
# angle_weld_end = np.pi # 3/2 * np.pi #2 * np.pi

angle_weld_start = 1/2 * np.pi
angle_weld_end = 3/2 * np.pi # 3/2 * np.pi #2 * np.pi

# angle_weld_end = np.pi # np.pi/3 #2 * np.pi

margin_x = 0. # Some margin around the pipe
bound_initial_pos_x_low = -0.5
bound_initial_pos_x_high = pos_center_pipe[0] - radius_pipe - footprint_robot_x/2 - margin_x 

bound_initial_pos_y_low = -1.5 + footprint_robot_y/2
bound_initial_pos_y_high = 1.5 - footprint_robot_y/2

# Draw rectangle in RViz covering all possible random XY positions
center_x = (bound_initial_pos_x_low + bound_initial_pos_x_high) / 2.0
center_y = (bound_initial_pos_y_low + bound_initial_pos_y_high) / 2.0
size_x = abs(bound_initial_pos_x_high - bound_initial_pos_x_low)
size_y = abs(bound_initial_pos_y_high - bound_initial_pos_y_low)

# Generate trajectory and get initial desired pose
position, orientation = generate_circular_trajectory(
    ns,
    center=pos_center_pipe,
    radius=radius_pipe,
    angle_start=angle_weld_start,
    angle_end=angle_weld_end,
)

coll_checker = CollisionChecker(urdf, srdf)
coll_checker.add_pipe('weld_pipe', radius_pipe, length_pipe, pos_center_pipe, orientation_pipe)

# ========================================================

# Kill any existing rviz_markers.py processes before starting a new one
import subprocess
subprocess.run("pkill -f rviz_markers.py", shell=True)
subprocess.run("pkill -f rviz_rectangle.py", shell=True)
subprocess.run("pkill -f rviz_line_marker.py", shell=True)

# Draw pipe cylinder in RViz (as background subprocess)
rviz_marker_pipe = subprocess.Popen([
    "python3", str(PATH_TO_CONCERT_WS / "src" / "concert_weld" / "src" / "rviz_markers.py"),
    str(pos_center_pipe[0]), str(pos_center_pipe[1]), str(pos_center_pipe[2]),
    str(radius_pipe), str(length_pipe),
    str(orientation_pipe[0]), str(orientation_pipe[1]), str(orientation_pipe[2]), str(orientation_pipe[3])
])

# Draw pipe cylinder in RViz (as background subprocess)
rviz_marker_footprint = subprocess.Popen([
    "python3", str(PATH_TO_CONCERT_WS / "src" / "concert_weld" / "src" / "rviz_rectangle.py"),
    'footprint_robot',
    str(0.0), str(0.0),  # center_x, center_y
    str(footprint_robot_x), str(footprint_robot_y),
    'base_link',
    '0.0', '1.0', '0.0', '0.3'  # Green with some transparency
])

# Draw rectangle in RViz at the random pose
subprocess.Popen([
    "python3", str(PATH_TO_CONCERT_WS / "src" / "concert_weld" / "src" / "rviz_rectangle.py"),
    'initial_zone',
    str(center_x), str(center_y),
    str(size_x), str(size_y),
    'world'
])

# Draw a line along the circular trajectory in RViz
line_points = []
for k in range(position.shape[1]):
    line_points.extend([position[0, k], position[1, k], position[2, k]])

subprocess.Popen([
    "python3", str(PATH_TO_CONCERT_WS / "src" / "concert_weld" / "src" / "rviz_line_marker.py"),
    'weld_trajectory',
    *[str(x) for x in line_points]
])

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

# for cost_name, cost in prb.getCosts().items():
#     print(f"Cost '{cost_name}' nodes: {cost.getNodes()}")

# exit()
# Print initial FK vs desired for user awareness (after model and ti are defined)
fk_ee_pos = kin_dyn.fk('ee_F')(q=model.q0)['ee_pos'][:, 0]
fk_ee_rot = R.from_matrix((kin_dyn.fk('ee_F')(q=model.q0)['ee_rot'].full())).as_quat()

print(f"[INFO] Initial ee_F pos: {fk_ee_pos}, rot (quat): {fk_ee_rot}")

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

print(f"circular trajectory applied: center={pos_center_pipe}, radius={radius_pipe}")
print(f"angle range: [{angle_weld_start}, {angle_weld_end}] rad, steps: {ns + 1}")

# Set base pose in XZ plane and pitch-only orientation (pitch=0)
tmp_q0 = ti.model.q0.copy()
tmp_q0[2] = 0.0  # Z (set to desired height if needed)
pitch_angle = 0.0  # Set to desired pitch in radians
base_quat = R.from_euler('y', pitch_angle).as_quat()  # [x, y, z, w]
tmp_q0[3:7] = base_quat  # Set quaternion part

ti.model.q0 = tmp_q0

# Set bounds to fix the base pose (except for XY which will be randomized)
ti.model.q[2].setBounds(tmp_q0[2], tmp_q0[2])
ti.model.q[3:7].setBounds(tmp_q0[3:7], tmp_q0[3:7])

# Create optimization variable for base XY
base_pos_xy = prb.createSingleVariable('base_pos_xy', 2)
prb.createConstraint('base_pos_xy_constraint', model.q[:2] - base_pos_xy)

# ti.model.q[0].setBounds(bound_initial_pos_x_low, bound_initial_pos_x_high) 
# ti.model.q[1].setBounds(bound_initial_pos_y_low, bound_initial_pos_y_high)

# joint limits
ti.model.q[7:].setBounds(kin_dyn.q_min()[7:], kin_dyn.q_max()[7:])

ti.model.v.setInitialGuess(ti.model.v0)

# start with zero velocity
# ti.model.v.setBounds(np.zeros_like(ti.model.v0), np.zeros_like(ti.model.v0), nodes=0)
# ti.model.v.setBounds(np.zeros_like(ti.model.v0), np.zeros_like(ti.model.v0), nodes=ns)

# Constrain linear axis (q[8]) to only move in one direction
cnsrt_vel_linear_guide = prb.createConstraint('linear_axis_positive_velocity', -model.v[15])
cnsrt_vel_linear_guide.setBounds(0, np.inf)

# constant_vel_linear_guide = prb.createIntermediateConstraint('linear_axis_positive_velocity', model.a[14])
# constant_vel_linear_guide.setBounds(0., 0.)

# constant_vel_yaw_guide = prb.createIntermediateConstraint('yaw_axis_constant_velocity', model.a[15])
# constant_vel_yaw_guide.setBounds(0., 0.)

# id_fn = kin_dyn_utils.InverseDynamics(kin_dyn) # force_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL
# tau_weight_min = 1e1
# tau = id_fn.call(model.q, 0., 0.)
# prb.createResidual('tau_cost', tau_weight_min * tau)

ti.finalize()

is_colliding = True
solution_found = False

# =================================================================================
# matfile = os.path.join(os.path.dirname(__file__), '../mat_files/weld_concert_very_good.mat')
# if not os.path.exists(matfile):
#     print(f"File not found: {matfile}")
#     sys.exit(1)

# data = loadmat(matfile)
# q_init = data.get('q')  # Use the first column of q as the initial guess
# =================================================================================

while is_colliding == True or solution_found == False:

    random_pose_x = np.random.uniform(low=bound_initial_pos_x_low, high=bound_initial_pos_x_high, size=1)
    random_pose_y = np.random.uniform(low=bound_initial_pos_y_low, high=bound_initial_pos_y_high, size=1)

    

    initial_guess_q = ti.model.q0.copy()
    initial_guess_q[0] = random_pose_x  # Randomize base X position in initial guess
    initial_guess_q[1] = random_pose_y  # Randomize base Y position in initial guess

    ti.model.q.setInitialGuess(initial_guess_q)
    ti.model.q[0].setBounds(random_pose_x, random_pose_x)  # Update bounds for base X
    ti.model.q[1].setBounds(random_pose_y, random_pose_y)  # Update bounds for base Y

    solution_found = ti.bootstrap()

    solution = ti.solution

    is_colliding = False
    for node in range(ns + 1):
        # print(f"Node {node}:")
        is_colliding_node, pairs = coll_checker.compute_collisions(solution['q'][:, node])
        # print("Colliding pairs:", pairs)
        if is_colliding_node:
            is_colliding = True
        
        # print("-----")


id_fn = kin_dyn_utils.InverseDynamics(kin_dyn) # force_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL
# Compute tau for all nodes
tau = np.zeros_like(solution['a'])
for node in range(ns - 1):
    # tau_node = id_fn.call(solution['q'][:, node], solution['v'][:, node], solution['a'][:, node])
    tau_node = id_fn.call(solution['q'][:, node], 0., 0.)
    tau_node = np.asarray(tau_node).flatten()
    tau[:, node] = tau_node


name_file = "weld_concert"
if not os.path.exists(f"{PATH_TO_CONCERT_WELD}/mat_files"):
    os.mkdir(f"{PATH_TO_CONCERT_WELD}/mat_files")
ms = mat_storer.matStorer(f"{PATH_TO_CONCERT_WELD}/mat_files/" + name_file + '.mat')
info_dict = dict(
    n_nodes=prb.getNNodes(),
    dt=prb.getDt(),
    pos_center_pipe=pos_center_pipe,
    orientation_pipe=orientation_pipe,
    radius_pipe=radius_pipe,
    angle_weld_start=angle_weld_start,
    angle_weld_end=angle_weld_end,
    tau=tau,
    joint_names=model.kd.joint_names(),
    initial_robot_pose=solution['q'][:, 0], 
)

ms.store({**solution, **info_dict})

q_forward = solution['q']
q_backward = np.flip(q_forward, axis=1)
q_cycle = np.concatenate([q_forward, q_backward], axis=1)

contact_list_repl = list(model.cmap.keys())
repl = replay_trajectory.replay_trajectory(dt, model.kd.joint_names(), q_cycle,
                                        {k: None for k in model.fmap.keys()},
                                        model.kd_frame, model.kd,
                                        trajectory_markers=contact_list_repl,
                                        future_trajectory_markers={'ee_F': 'world'})

repl.replay()