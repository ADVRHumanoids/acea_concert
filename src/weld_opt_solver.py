"""Horizon torque setup and collision-aware weld solution search."""

from dataclasses import dataclass

import casadi as cs
import casadi_kin_dyn.py3casadi_kin_dyn as casadi_kin_dyn
import numpy as np
from horizon.problem import Problem
from horizon.rhc.model_description import FullModelInverseDynamics
from horizon.rhc.taskInterface import TaskInterface
from horizon.utils import kin_dyn as kin_dyn_utils
from scipy.spatial.transform import Rotation as R
from scipy.stats import qmc

from weld_opt_attempt_log import WeldOptAttemptLog


VIRTUAL_JOINT_NAMES = {'universe', 'reference'}
FLOATING_BASE_VELOCITY_DOF = 6


@dataclass(frozen=True)
class WeldProblem:
    problem: object
    kin_dyn: object
    model: object
    task_interface: object
    critical_torque_indices: list[int]
    missing_critical_torque_joints: list[str]
    inverse_dynamics: object


def torque_indices_for_joints(joint_names, selected_joint_names):
    actuated_names = [
        str(name).strip()
        for name in joint_names
        if str(name).strip() not in VIRTUAL_JOINT_NAMES
    ]
    torque_indices = []
    missing_names = []
    for name in selected_joint_names:
        if name not in actuated_names:
            missing_names.append(name)
            continue
        torque_indices.append(
            FLOATING_BASE_VELOCITY_DOF + actuated_names.index(name))
    return torque_indices, missing_names


def make_sobol_base_points(sample_count, bounds, seed):
    if sample_count is None:
        return None
    x_low, x_high, y_low, y_high = bounds
    sample_power = (sample_count - 1).bit_length()
    unit_points = qmc.Sobol(
        d=2,
        scramble=True,
        seed=seed,
    ).random_base2(sample_power)[:sample_count]
    return qmc.scale(unit_points, [x_low, y_low], [x_high, y_high])


def add_quasi_static_torque_cost(
    problem,
    model,
    kin_dyn,
    enabled,
    joint_names,
    weight,
):
    if not enabled:
        return [], [], None

    torque_indices, missing_names = torque_indices_for_joints(
        model.kd.joint_names(), joint_names)
    if missing_names:
        print(
            '[weld_opt] Warning: torque-cost joints not found: '
            f'{missing_names}'
        )
    if not torque_indices:
        print('[weld_opt] Warning: torque minimization enabled but no joints matched.')
        return torque_indices, missing_names, None

    inverse_dynamics = kin_dyn_utils.InverseDynamics(kin_dyn)
    tau_sym = inverse_dynamics.call(
        model.q,
        cs.SX.zeros(model.v.shape[0], 1),
        cs.SX.zeros(model.a.shape[0], 1),
    )
    critical_tau = cs.vertcat(*[
        tau_sym[idx] for idx in torque_indices
    ])
    problem.createIntermediateResidual(
        'min_critical_joint_torque',
        weight * critical_tau,
    )
    print(
        '[weld_opt] Minimizing quasi-static torques for joints '
        f'{joint_names}\n with tau rows {torque_indices}\n'
        f'and weight {weight}'
    )
    return torque_indices, missing_names, inverse_dynamics


def build_weld_problem(
    *,
    urdf,
    task_config,
    ee_link,
    n_intervals,
    duration,
    circular_orientation,
    circle_offset,
    nominal_pipe_center,
    optimize_pipe_height,
    pipe_z_bounds,
    minimize_critical_joint_torques,
    critical_torque_joints,
    torque_cost_weight,
):
    problem = Problem(n_intervals, receding=True, casadi_type=cs.SX)
    problem.setDt(duration / n_intervals)
    kin_dyn = casadi_kin_dyn.CasadiKinDyn(urdf)

    base_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
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
    projected_base_z = kin_dyn.fk('base_link_projected')(
        q=kin_dyn.mapToQ(q_init))['ee_pos'][2, 0]
    base_init[2] = -projected_base_z

    model = FullModelInverseDynamics(
        problem=problem,
        kd=kin_dyn,
        q_init=q_init,
        base_init=base_init,
    )
    task_interface = TaskInterface(prb=problem, model=model)
    task_interface.setTaskFromYaml(task_config)

    initial_ee_pos = kin_dyn.fk(ee_link)(q=model.q0)['ee_pos'][:, 0]
    initial_ee_quat = R.from_matrix(
        kin_dyn.fk(ee_link)(q=model.q0)['ee_rot'].full()).as_quat()
    print(
        f"[INFO] Initial {ee_link} pos: {initial_ee_pos}, "
        f"rot (quat): {initial_ee_quat}"
    )

    orientation_reference = np.zeros((7, n_intervals + 1))
    orientation_reference[3:, :] = circular_orientation
    task_interface.getTask('ee_ori').setRef(orientation_reference)

    q0 = task_interface.model.q0.copy()
    q0[2] = base_init[2]
    q0[3:7] = R.from_euler('y', 0.0).as_quat()
    task_interface.model.q0 = q0
    task_interface.model.q[2].setBounds(q0[2], q0[2])
    task_interface.model.q[3:7].setBounds(q0[3:7], q0[3:7])

    base_pos_xy = problem.createSingleVariable('base_pos_xy', 2)
    problem.createConstraint(
        'base_pos_xy_constraint', model.q[:2] - base_pos_xy)

    if optimize_pipe_height:
        pipe_z = problem.createSingleVariable('pipe_z', 1)
        pipe_z.setBounds(pipe_z_bounds[0], pipe_z_bounds[1])
        pipe_z.setInitialGuess(nominal_pipe_center[2])
    else:
        pipe_z = float(nominal_pipe_center[2])

    fk_ee_pos = kin_dyn.fk(ee_link)(q=model.q)['ee_pos']
    pipe_x = float(nominal_pipe_center[0])
    pipe_y = float(nominal_pipe_center[1])
    for node in range(n_intervals + 1):
        ee_pos_ref = cs.vertcat(
            pipe_x + circle_offset[0, node],
            pipe_y + circle_offset[1, node],
            pipe_z + circle_offset[2, node],
        )
        problem.createConstraint(
            f'ee_pos_pipe_height_{node}',
            fk_ee_pos - ee_pos_ref,
            nodes=node,
        )

    task_interface.model.q[7:].setBounds(
        kin_dyn.q_min()[7:], kin_dyn.q_max()[7:])
    task_interface.model.v.setInitialGuess(task_interface.model.v0)

    linear_axis_velocity = problem.createConstraint(
        'linear_axis_positive_velocity', -model.v[15])
    linear_axis_velocity.setBounds(0, np.inf)

    (
        critical_torque_indices,
        missing_critical_torque_joints,
        inverse_dynamics,
    ) = add_quasi_static_torque_cost(
        problem,
        model,
        kin_dyn,
        minimize_critical_joint_torques,
        critical_torque_joints,
        torque_cost_weight,
    )
    task_interface.finalize()
    return WeldProblem(
        problem=problem,
        kin_dyn=kin_dyn,
        model=model,
        task_interface=task_interface,
        critical_torque_indices=critical_torque_indices,
        missing_critical_torque_joints=missing_critical_torque_joints,
        inverse_dynamics=inverse_dynamics,
    )


def _peak_critical_quasi_static_torque(
    candidate_solution,
    inverse_dynamics,
    torque_indices,
):
    zero_velocity = np.zeros(candidate_solution['v'].shape[0])
    zero_acceleration = np.zeros(candidate_solution['a'].shape[0])
    peak = 0.0
    for node in range(candidate_solution['a'].shape[1]):
        tau_node = np.asarray(inverse_dynamics.call(
            candidate_solution['q'][:, node],
            zero_velocity,
            zero_acceleration,
        )).flatten()
        peak = max(
            peak,
            float(np.max(np.abs(tau_node[torque_indices]))),
        )
    return peak


def solve_weld_problem(
    *,
    task_interface,
    n_intervals,
    base_bounds,
    base_search_points,
    max_random_attempts,
    nominal_pipe_center,
    optimize_pipe_height,
    make_collision_checker,
    inverse_dynamics,
    critical_torque_indices,
    on_base_candidate=None,
):
    """Solve candidates and return the first valid or lowest-torque solution."""
    if base_search_points is not None and inverse_dynamics is None:
        raise RuntimeError('--base-search requires at least one critical torque joint.')

    x_low, x_high, y_low, y_high = base_bounds
    best_solution = None
    best_peak_torque = np.inf
    attempt_count = 0
    attempt_log = WeldOptAttemptLog()
    print(f"[weld_opt] Attempt log: {attempt_log.path}")

    while True:
        attempt_count += 1
        if (base_search_points is not None
                and attempt_count > len(base_search_points)):
            break
        if (base_search_points is None
                and max_random_attempts > 0
                and attempt_count > max_random_attempts):
            attempt_log.write(
                attempt_count,
                "max_attempts_exceeded",
                max_attempts=max_random_attempts,
            )
            raise RuntimeError(
                'No collision-free solution found after '
                f'{max_random_attempts} attempts.'
            )

        if base_search_points is None:
            base_x = np.random.uniform(x_low, x_high, size=1)
            base_y = np.random.uniform(y_low, y_high, size=1)
        else:
            base_x = base_search_points[attempt_count - 1, 0:1]
            base_y = base_search_points[attempt_count - 1, 1:2]

        if on_base_candidate is not None:
            on_base_candidate(base_x[0], base_y[0])
        print(f'Publishing point: x={base_x[0]}, y={base_y[0]}')
        attempt_log.write(attempt_count, "start")

        initial_guess_q = task_interface.model.q0.copy()
        initial_guess_q[0] = base_x
        initial_guess_q[1] = base_y
        task_interface.model.q.setInitialGuess(initial_guess_q)
        task_interface.model.q[0].setBounds(base_x, base_x)
        task_interface.model.q[1].setBounds(base_y, base_y)

        if not task_interface.bootstrap():
            attempt_log.write(attempt_count, "bootstrap_failed")
            continue

        candidate_solution = task_interface.solution
        pipe_z = (
            float(np.asarray(candidate_solution['pipe_z']).reshape(-1)[0])
            if optimize_pipe_height else float(nominal_pipe_center[2])
        )
        pipe_center = np.asarray(nominal_pipe_center, dtype=float).copy()
        pipe_center[2] = pipe_z
        collision_checker = make_collision_checker(pipe_center)

        collision = None
        for node in range(n_intervals + 1):
            is_colliding, pairs = collision_checker.compute_collisions(
                candidate_solution['q'][:, node])
            if is_colliding:
                collision = (node, pairs)
                break
        if collision is not None:
            node, pairs = collision
            print(
                f"[weld_opt] Rejecting attempt {attempt_count}: "
                f"collision at node {node}: {pairs}"
            )
            attempt_log.write(
                attempt_count,
                "collision",
                node=node,
                pairs=[str(pair) for pair in pairs],
            )
            continue

        attempt_log.write(attempt_count, "accepted")
        if base_search_points is None:
            return candidate_solution

        peak_torque = _peak_critical_quasi_static_torque(
            candidate_solution,
            inverse_dynamics,
            critical_torque_indices,
        )
        print(
            f"[weld_opt] Sobol base {attempt_count}/{len(base_search_points)}: "
            f"peak critical torque={peak_torque:.3f} Nm"
        )
        if peak_torque < best_peak_torque:
            best_peak_torque = peak_torque
            best_solution = {
                name: np.array(value, copy=True)
                for name, value in candidate_solution.items()
            }

    if best_solution is None:
        raise RuntimeError(
            f'No collision-free solution found for the '
            f'{len(base_search_points)} Sobol base positions.'
        )
    print(
        f"[weld_opt] Selected lowest collision-free peak torque: "
        f"{best_peak_torque:.3f} Nm"
    )
    return best_solution
