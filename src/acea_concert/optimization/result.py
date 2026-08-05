"""Post-process and store a solved weld trajectory."""

from dataclasses import dataclass

from horizon.utils import kin_dyn as kin_dyn_utils
from horizon.utils import mat_storer
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class WeldResult:
    data: dict
    pipe_center: np.ndarray
    trajectory: np.ndarray
    pipe_z: float


def build_weld_result(
    *,
    solution,
    problem,
    kin_dyn,
    nominal_pipe_center,
    circle_offset,
    circular_orientation,
    optimize_pipe_height,
    metadata,
):
    pipe_z = (
        float(np.asarray(solution['pipe_z']).reshape(-1)[0])
        if optimize_pipe_height else float(nominal_pipe_center[2])
    )
    pipe_center = np.asarray(nominal_pipe_center, dtype=float).copy()
    pipe_center[2] = pipe_z
    trajectory = circle_offset + pipe_center.reshape(3, 1)

    inverse_dynamics = kin_dyn_utils.InverseDynamics(kin_dyn)
    tau = np.zeros_like(solution['a'])
    zero_velocity = np.zeros(solution['v'].shape[0])
    zero_acceleration = np.zeros(solution['a'].shape[0])
    for node in range(tau.shape[1]):
        tau_node = inverse_dynamics.call(
            solution['q'][:, node],
            zero_velocity,
            zero_acceleration,
        )
        tau[:, node] = np.asarray(tau_node).flatten()

    desired_position_base = np.zeros_like(trajectory)
    for node in range(trajectory.shape[1]):
        base_position_world = solution['q'][:3, node]
        base_quaternion_world = solution['q'][3:7, node]
        desired_position_base[:, node] = R.from_quat(
            base_quaternion_world
        ).inv().apply(trajectory[:, node] - base_position_world)

    pipe_x_axis_world = np.array([1.0, 0.0, 0.0])
    pipe_y_axis_world = np.array([0.0, 1.0, 0.0])
    pipe_z_axis_world = np.array([0.0, 0.0, 1.0])
    pipe_rotation_world = np.column_stack([
        pipe_x_axis_world,
        pipe_y_axis_world,
        pipe_z_axis_world,
    ])

    desired_position_gap = pipe_rotation_world.T @ (
        trajectory - pipe_center.reshape(3, 1)
    )
    desired_quaternion_gap = np.zeros_like(circular_orientation)
    for node in range(trajectory.shape[1]):
        world_rotation_ee = R.from_quat(
            circular_orientation[:, node]).as_matrix()
        gap_rotation_ee = pipe_rotation_world.T @ world_rotation_ee
        desired_quaternion_gap[:, node] = R.from_matrix(
            gap_rotation_ee).as_quat()

    pipe_center_base = R.from_quat(solution['q'][3:7, 0]).inv().apply(
        pipe_center - solution['q'][:3, 0])

    info = dict(
        n_nodes=problem.getNNodes(),
        dt=problem.getDt(),
        pos_center_pipe_nominal=np.asarray(nominal_pipe_center),
        pipe_z_nominal=float(nominal_pipe_center[2]),
        pos_center_pipe=pipe_center,
        pipe_z=pipe_z,
        optimize_pipe_height=optimize_pipe_height,
        tau=tau,
        joint_names=kin_dyn.joint_names(),
        desired_traj_weld_pos=trajectory,
        desired_traj_weld_pos_base=desired_position_base,
        desired_traj_weld_pos_gap=desired_position_gap,
        desired_traj_weld_quat_gap=desired_quaternion_gap,
        pos_center_pipe_base=pipe_center_base,
        pipe_x_axis_world=pipe_x_axis_world,
        pipe_y_axis_world=pipe_y_axis_world,
        pipe_z_axis_world=pipe_z_axis_world,
        initial_robot_pose=solution['q'][:, 0],
        **metadata,
    )
    return WeldResult(
        data={**solution, **info},
        pipe_center=pipe_center,
        trajectory=trajectory,
        pipe_z=pipe_z,
    )


def store_weld_result(path, result):
    mat_storer.matStorer(str(path)).store(result.data)
