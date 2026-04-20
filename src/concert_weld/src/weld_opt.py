from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils import utils
from pathlib import Path
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
# from xbot_interface import config_options as co
# from xbot_interface import xbot_interface as xbot
from geometry_msgs.msg import Vector3

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
modular_prismatic = PATH_TO_CONCERT_WS / "src" / "concert_weld" / "concert_prismatic.py"

urdf = subprocess.check_output(["python3", str(modular_prismatic), "-o", "urdf"], text=True)
print(urdf)
# kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)
# '''
# Build ModelInterface and RobotStatePublisher
# '''
# cfg = co.ConfigOptions()
# cfg.set_urdf(urdf)
# cfg.set_srdf(srdf)
# cfg.generate_jidmap()
# cfg.set_string_parameter('model_type', 'RBDL')
# cfg.set_string_parameter('framework', 'ROS')
# cfg.set_bool_parameter('is_model_floating_base', True)

# robot = None

# if xbot_mode:
#     print('Getting robot...\n')
#     robot = xbot.RobotInterface(cfg)
#     robot.sense()
#     q_init = robot.getPositionReference()
#     q_init = robot.eigenToMap(q_init)
#     print('done\n')
# else:
#     print('XBot-RobotInterface not created.\n Using initial q default values.\n')
#     q_init = {'J1_A': 0.0,
#               'J_wheel_A': 0.0,
#               'J1_B': 0.0,
#               'J_wheel_B': 0.0,
#               'J1_C': 0.0,
#               'J_wheel_C': 0.0,
#               'J1_D': 0.0,
#               'J_wheel_D': 0.0,
#               'J1_E': 0.0,
#               'J2_E': -0.5,
#               'J3_E': 0.0,
#               'J4_E': 0.5,
#               'J5_E': 0.0,
#               'J6_E': -0.5
#               }


# base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


# wheel_radius = 0.16
# FK = kin_dyn.fk('J_wheel_A')
# init_pos_wheel = FK(q=kin_dyn.mapToQ(q_init))['ee_pos']
# base_init[2] = -init_pos_wheel[2] + wheel_radius

# model = FullModelInverseDynamics(problem=prb,
#                                  kd=kin_dyn,
#                                  q_init=q_init,
#                                  base_init=base_init
#                                  )


# ti = TaskInterface(prb=prb, model=model)
# ti.setTaskFromYaml(rospkg.RosPack().get_path('concert_horizon') + '/config/concert_config.yaml')

# ee_ref = ti.getTask('ee_force').getValues()[:, 0]
# ee_pos_0 = kin_dyn.fk('ee_E')(q=model.q0)['ee_pos'][:, 0]
# ee_rot_0 = scipy_rot.from_matrix((kin_dyn.fk('ee_E')(q=model.q0)['ee_rot'].full())).as_quat()

# print(f"initial ee reference: {ee_ref}")
# print(f"initial ee pos: {ee_pos_0}, {ee_rot_0}")

# ti.model.q.setBounds(ti.model.q0, ti.model.q0, nodes=0)
# ti.model.v.setBounds(ti.model.v0, ti.model.v0, nodes=0)
# ti.model.a.setBounds(np.zeros([model.a.shape[0], 1]), np.zeros([model.a.shape[0], 1]), nodes=0)
# ti.model.q.setInitialGuess(ti.model.q0)
# ti.model.v.setInitialGuess(ti.model.v0)


# prb.createResidual('max_q', 1e1 * utils.utils.barrier(kin_dyn.q_max()[7:] - model.q[7:]))
# prb.createResidual('min_q', 1e1 * utils.utils.barrier1(kin_dyn.q_min()[7:] - model.q[7:]))

# vel_lims = model.kd.velocityLimits()
# prb.createResidual('max_vel', 1e2 * utils.utils.barrier(vel_lims[7:] - model.v[7:]))
# prb.createResidual('min_vel', 1e1 * utils.utils.barrier1(-1 * vel_lims[7:] - model.v[7:]))


# ============== REQUIRED ONLY FOR OMNISTEERING ==============
# model.v[2].setBounds(0, 0) # the robot cannot fly
# model.v[3:5].setBounds([0, 0], [0,0]) # the robot cannot pitch and roll

# wheel_vel_max_index = [elem for elem in kin_dyn.joint_names() if elem not in ['universe', 'reference']].index('J_wheel_A')

# vel_lin_max_padding = 0.
# vel_ang_max_padding = 0.
# base_vel_lin_max = 0.3 #vel_lims[6 + wheel_vel_max_index] * wheel_radius
# base_vel_ang_max = 0.3 #(vel_lims[6 + wheel_vel_max_index] * wheel_radius) / 0.6 --> radius of robot more or less

# base_vel_lin_max_padded = base_vel_lin_max - vel_lin_max_padding
# base_vel_ang_max_padded = base_vel_ang_max - vel_ang_max_padding

# print(f"base_vel_lin_max: {base_vel_lin_max_padded} ")
# print(f"base_vel_ang_max: {base_vel_ang_max_padded} ")

# prb.createResidual('max_base_linear_vel', 1e2 * utils.utils.barrier(base_vel_lin_max_padded - model.v[:2]))
# prb.createResidual('min_base_linear_vel', 1e2 * utils.utils.barrier1(- base_vel_lin_max_padded - model.v[:2]))

# base_vel_max
# prb.createResidual("max_base_angular_vel", 1e2 * utils.utils.barrier(base_vel_ang_max_padded - model.v[5]))
# prb.createResidual("min_base_angular_vel", 1e2 * utils.utils.barrier1(- base_vel_ang_max_padded - model.v[5]))


# ti.finalize()

# ti.bootstrap()
# ti.load_initial_guess()
# solution = ti.solution

# rate = rospy.Rate(1 / dt)

# contact_list_repl = list(model.cmap.keys())

# '''
# Build ModelInterface and RobotStatePublisher
# '''
# cfg = co.ConfigOptions()
# cfg.set_urdf(urdf)
# cfg.set_srdf(srdf)
# cfg.generate_jidmap()
# cfg.set_string_parameter('model_type', 'RBDL')
# cfg.set_string_parameter('framework', 'ROS')
# cfg.set_bool_parameter('is_model_floating_base', True)

# robot = None

# if xbot_mode:
#     print('Getting robot...\n')
#     robot = xbot.RobotInterface(cfg)
#     robot.sense()
#     q_init = robot.getPositionReference()
#     q_init = robot.eigenToMap(q_init)
#     print('done\n')
# else:
#     print('XBot-RobotInterface not created.\n Using initial q default values.\n')
#     q_init = {'J1_A': 0.0,
#               'J_wheel_A': 0.0,
#               'J1_B': 0.0,
#               'J_wheel_B': 0.0,
#               'J1_C': 0.0,
#               'J_wheel_C': 0.0,
#               'J1_D': 0.0,
#               'J_wheel_D': 0.0,
#               'J1_E': 0.0,
#               'J2_E': -0.5,
#               'J3_E': 0.0,
#               'J4_E': 0.5,
#               'J5_E': 0.0,
#               'J6_E': -0.5
#               }


# base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


# wheel_radius = 0.16
# FK = kin_dyn.fk('J_wheel_A')
# init_pos_wheel = FK(q=kin_dyn.mapToQ(q_init))['ee_pos']
# base_init[2] = -init_pos_wheel[2] + wheel_radius

# model = FullModelInverseDynamics(problem=prb,
#                                  kd=kin_dyn,
#                                  q_init=q_init,
#                                  base_init=base_init
#                                  )
# repl = replay_trajectory.replay_trajectory(dt, model.kd.joint_names(), np.array([]),
#                                            {k: None for k in model.fmap.keys()},
#                                            model.kd_frame, model.kd,
#                                            trajectory_markers=contact_list_repl)
#                                            # future_trajectory_markers={'base_link': 'world', 'J_wheel_D': 'world'})