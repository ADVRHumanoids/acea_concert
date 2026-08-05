#!/usr/bin/python3
import sys, os
from geometry_msgs.msg import Vector3
import numpy as np
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
import scipy
import time
import argparse
import subprocess
from pathlib import Path
from horizon.ros import replay_trajectory

PATH_TO_CONCERT_WS = Path("/home/user/concert_ws")
PATH_TO_ACEA_CONCERT = PATH_TO_CONCERT_WS/"src"/"acea_concert"
# modular_prismatic = PATH_TO_CONCERT_WS/"src"/"concert_description"/"concert_examples"/"src"/"concert_prismatic.py"
modular_prismatic = PATH_TO_ACEA_CONCERT/"src"/"modular"/"concert_with_torch.py"
PATH_TO_HORIZON_CONFIG = PATH_TO_ACEA_CONCERT/"config"/"weld.yaml"

class TrajectoryReplayer:
    def __init__(self, data, fixed_joint_map=None):

        self.urdf = subprocess.check_output(["python3", str(modular_prismatic), "-o", "urdf"], text=True)

        self.solution_augmented = None
        self.fixed_joint_map = fixed_joint_map
        self.kin_dyn = casadi_kin_dyn.CasadiKinDyn(self.urdf)
        self.data = data

        self.joint_names = [elem for elem in self.kin_dyn.joint_names() if elem not in ['universe', 'reference']]

        self.pos_center_pipe = self.data['pos_center_pipe'][0]
        self.radius_pipe = self.data['radius_pipe'][0][0]
        self.length_pipe = self.data['length_pipe'][0][0]
        self.orientation_pipe = self.data['orientation_pipe'][0]
        self.footprint_robot_x = self.data['footprint_robot_x'][0][0]
        self.footprint_robot_y = self.data['footprint_robot_y'][0][0]
        self.initial_zone_center_x = self.data['initial_zone_center_x'][0][0]
        self.initial_zone_center_y = self.data['initial_zone_center_y'][0][0]
        self.initial_zone_size_x = self.data['initial_zone_size_x'][0][0]
        self.initial_zone_size_y = self.data['initial_zone_size_y'][0][0]
        self.desired_traj_weld_pos = self.data['desired_traj_weld_pos']
        self.angle_weld_start=self.data['angle_weld_start'][0][0]
        self.angle_weld_end=self.data['angle_weld_end'][0][0]

        self.dim_q = self.data['q'].shape[0]
        self.dim_v = self.data['v'].shape[0]
        self.dim_a = self.data['a'].shape[0]
        self.dim_tau = self.data['tau'].shape[0]

        self.nodes_q = self.data['q'].shape[1]
        self.nodes_v = self.data['v'].shape[1]
        self.nodes_a = self.data['a'].shape[1]
        self.nodes_tau = self.data['tau'].shape[1]

        self.dt = self.data['dt'][0][0]
        self.ns = self.data['n_nodes'][0][0]

        print(f'n nodes: {self.ns}')
        print(f'dt: {self.dt}')
        print(f'dim q: {self.dim_q}x{self.nodes_q}') # n nodes
        print(f'dim v: {self.dim_v}x{self.nodes_v}') # n nodes
        print(f'dim a: {self.dim_a}x{self.nodes_a}') # n nodes - 1
        print(f'dim tau: {self.dim_tau}x{self.nodes_tau}') # n nodes - 1

    def init_robot_state_publisher(self):

        # Launch robot_state_publisher in background with the URDF
        subprocess.Popen(
            ["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args", "-p", f"robot_description:={self.urdf}"],
        )

# ========================================================

    def init_scene(self):

        from viz.init_scene import InitScene

        self.init_robot_state_publisher()

        init_scene = InitScene(
            path_ws=PATH_TO_ACEA_CONCERT/"src"/"viz",
            pos_center_pipe=self.pos_center_pipe,
            radius_pipe=self.radius_pipe,
            length_pipe=self.length_pipe,
            orientation_pipe=self.orientation_pipe,
            footprint_robot_x=self.footprint_robot_x,
            footprint_robot_y=self.footprint_robot_y,
            center_x=self.initial_zone_center_x,
            center_y=self.initial_zone_center_y,
            size_x=self.initial_zone_size_x,
            size_y=self.initial_zone_size_y,
            position=self.desired_traj_weld_pos
        )
        init_scene.kill_existing_markers()
        init_scene.launch_scene()

    def replay(self, speed=1.0):
        assert speed > 0, "speed must be positive"
        
        if self.solution_augmented is not None:
            q_forward = self.solution_augmented['q']
        else:
            replay_dt = 0.01  # Fixed 100 Hz for smooth playback.
            duration = (self.nodes_q - 1) * self.dt / speed
            q_forward = self.resample_q(round(duration / replay_dt) + 1)

            q_backward = np.flip(q_forward, axis=1)
            q_cycle = np.concatenate([q_forward, q_backward], axis=1)

            print(f"Replaying trajectory with {q_cycle.shape[1]} steps (forward and backward)")
            repl = replay_trajectory.replay_trajectory(replay_dt,
                                                       self.kin_dyn.joint_names(),
                                                       q_cycle,
                                                       kindyn=self.kin_dyn
                                                       )
            repl.replay()

    def add_wiggling_ee_y(self, nodes, t_max, wiggle_amplitude=0.01, wiggle_frequency=2*np.pi):

        from horizon.problem import Problem
        import casadi as cs
        from horizon.rhc.model_description import FullModelInverseDynamics
        from horizon.rhc.taskInterface import TaskInterface
        from acea_concert.optimization.circular_trajectory import (
            generate_circular_trajectory,
        )

        ns = nodes
        T = t_max
        dt = T / ns

        prb = Problem(ns, receding=True, casadi_type=cs.SX)
        prb.setDt(dt)


        circular_pos, circular_ori = generate_circular_trajectory(
        ns,
        center=self.pos_center_pipe,
        radius=self.radius_pipe,
        angle_start=self.angle_weld_start,
        angle_end=self.angle_weld_end,
        upside_down=False,
        wiggle_amplitude=wiggle_amplitude,
        wiggle_frequency=wiggle_frequency
        )

        self.desired_traj_weld_pos = circular_pos

        self.init_scene()  # initialize the scene to update the desired trajectory visualization

        base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # Y (no lateral offset)

        # Keep q_init as before, but you can tune these if needed
        
        exit  # Ensure joint names are loaded
        q_init = dict(zip(self.joint_names, self.data['q'][7:, 0]))

        model = FullModelInverseDynamics(problem=prb,
                                        kd=self.kin_dyn,
                                        q_init=q_init,
                                        base_init=base_init
                                        )

        ti = TaskInterface(prb=prb, model=model)
        ti.setTaskFromYaml(PATH_TO_HORIZON_CONFIG)

        position_aug = np.full((7, ns + 1), 0.0)
        position_aug[:3, :] = circular_pos

        orientation_aug = np.full((7, ns + 1), 0.0)
        orientation_aug[3:, :] = circular_ori

        pos_task_name = 'ee_pos'
        ori_task_name = 'ee_ori'

        ee_pos_task = ti.getTask(pos_task_name)
        ee_ori_task = ti.getTask(ori_task_name)

        ee_pos_task.setRef(position_aug)
        ee_ori_task.setRef(orientation_aug)

        ti.finalize()

        ti.model.q.setInitialGuess(self.resample_q(ns + 1))  # Use the original solution as the initial guess
        ti.model.q.setBounds(self.data['q'][:, 0], self.data['q'][:, 0], nodes=0)  # Update bounds for base X
        ti.model.q[:7].setBounds(self.data['q'][:7, 0], self.data['q'][:7, 0])  # Update bounds for base X

        solution_found = ti.bootstrap()

        self.solution_augmented = ti.solution



    def resample_q(self, n_nodes):
        """
        Resample the trajectory q to have n_nodes using linear interpolation.
        Overwrites self.data['q'] and updates self.nodes_q and self.ns.
        """
        q = self.data['q']  # shape: (dim_q, old_nodes)
        old_nodes = q.shape[1]
        old_x = np.linspace(0, 1, old_nodes)
        new_x = np.linspace(0, 1, n_nodes)
        from scipy.interpolate import interp1d
        interp_func = interp1d(old_x, q, kind='linear', axis=1)
        q_resampled = interp_func(new_x)
        self.data['q'] = q_resampled
        self.nodes_q = n_nodes
        self.ns = n_nodes
        print(f"Resampled q to {n_nodes} nodes.")

        return q_resampled



if __name__ == '__main__':

    
    matfile = os.path.join(os.path.dirname(__file__), '../mat_files/weld_concert.mat')
    if not os.path.exists(matfile):
        print(f"File not found: {matfile}")
        sys.exit(1)

    data = scipy.io.loadmat(matfile)
    xbot_horizon_replayer = TrajectoryReplayer(data)
    xbot_horizon_replayer.init_robot_state_publisher()
    xbot_horizon_replayer.init_scene()
    # xbot_horizon_replayer.add_wiggling_ee_y(nodes=200,
                                            # t_max=2,
                                            # wiggle_amplitude=0.01, 
                                            # wiggle_frequency=200*np.pi)  # Resample to 100 nodes for smoother replay

    xbot_horizon_replayer.replay(speed=1.0)
