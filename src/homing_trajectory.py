from dataclasses import dataclass

import casadi as cs
import numpy as np
from horizon.problem import Problem
from horizon.solvers import Solver
from scipy.spatial.transform import Rotation as R
from xbot2_interface import pyxbot2_collision as xbc
from xbot2_interface import pyxbot2_interface as xbi
from xbot2_interface.pyaffine3 import Affine3


ARM_GUIDE_FRAMES = (
    "J1_E_stator",
    "L_1_E",
    "J2_E_stator",
    "L_2_E",
    "J1_F_stator",
    "L_1_F",
    "J2_F_stator",
    "L_2_F",
    "L_2_link_1_F",
    "J3_F_stator",
    "L_3_F",
    "J4_F_stator",
    "L_4_F",
    "L_4_link_1_F",
    "J5_F_stator",
    "L_5_F",
    "J6_F_stator",
    "L_6_F",
)
TOOL_SPHERES = (
    ("end_effector_F", (0.0, 0.0, 0.0), 0.15),
    ("ee_F", (0.0, 0.0, 0.0), 0.1),
    ("L_6_F", (0.1, 0.0, 0.0), 0.1),
)
DEFAULT_INFLATION_RADIUS = 0.05
DEFAULT_TOOL_APPROACH_NODES = 5
DEFAULT_MOTION_WEIGHT = 10.0
DEFAULT_ACCEL_WEIGHT = 1.0
DEFAULT_MAX_JOINT_STEP = 1.0
DEFAULT_MAX_JOINT_ACCEL_STEP = 0.5
DEFAULT_IPOPT_PRINT_LEVEL = 5
DEFAULT_INITIAL_GUESS_MODE = "linear"


@dataclass
class HomingTrajectory:
    q: np.ndarray
    dt: float
    method: str


def _pipe_axis(orientation):
    axis = R.from_quat(orientation).apply([0.0, 0.0, 1.0])
    return axis / np.linalg.norm(axis)


def _pipe_halves(center, length, gap, orientation):
    half_length = 0.5 * (length - gap)
    if half_length <= 0.0:
        raise ValueError("pipe_gap must be smaller than length_pipe")

    offset = _pipe_axis(orientation) * (0.5 * half_length + 0.5 * gap)
    return (
        (np.asarray(center, dtype=float) + offset, half_length),
        (np.asarray(center, dtype=float) - offset, half_length),
    )


def _interpolate(q_start, q_goal, steps):
    alpha = np.linspace(0.0, 1.0, steps + 1)
    return (1.0 - alpha) * q_start[:, None] + alpha * q_goal[:, None]


def _initial_guess(q_start, q_goal, steps, mode):
    if mode == "linear":
        return _interpolate(q_start, q_goal, steps)
    if mode == "goal-after-start":
        guess = np.repeat(q_goal[:, None], steps + 1, axis=1)
        guess[:, 0] = q_start
        return guess
    raise ValueError(f"Unknown initial guess mode: {mode}")


def _resample(path_q, steps):
    old_x = np.linspace(0.0, 1.0, path_q.shape[1])
    new_x = np.linspace(0.0, 1.0, steps + 1)
    return np.vstack([
        np.interp(new_x, old_x, path_q[row])
        for row in range(path_q.shape[0])
    ])


class _PipeChecker:

    def __init__(self, urdf, srdf, center, radius, length, gap, orientation,
                 clearance):
        self.model = xbi.ModelInterface2(urdf, srdf, "pin")
        self.collision_model = xbc.CollisionModel(self.model)
        self.pipe_halves = _pipe_halves(center, length, gap, orientation)

        for idx, (pipe_center, pipe_length) in enumerate(self.pipe_halves):
            name = f"homing_pipe_{idx}"
            cyl = xbc.shape.Cylinder()
            cyl.radius = radius + clearance
            cyl.length = pipe_length
            self.collision_model.addCollisionShape(
                name,
                "world",
                cyl,
                Affine3(pos=pipe_center, rot=orientation),
                ["world"],
            )

    def collision(self, q):
        self.model.q = q
        self.model.update()
        self.collision_model.update()
        pairs = self.collision_model.getCollisionPairs(include_env=True)
        is_colliding, pair_ids = self.collision_model.checkCollision(
            include_env=True)
        if is_colliding:
            return True, [pairs[i] for i in pair_ids]

        return False, []


def _first_collision(path_q, checker):
    for node in range(path_q.shape[1]):
        is_colliding, pairs = checker.collision(path_q[:, node])
        if is_colliding:
            return node, pairs
    return None


def _pipe_clearance_constraints(prb, q, kin_dyn, center, radius, orientation,
                                clearance, inflation_radius,
                                tool_approach_nodes, nodes):
    axis = cs.DM(_pipe_axis(orientation))
    pipe_center = cs.DM(center)
    arm_nodes = range(1, nodes)
    tool_stop = max(1, nodes - max(0, int(tool_approach_nodes)))
    tool_nodes = range(1, tool_stop)

    def pipe_wall(point, point_radius):
        min_radius_sq = (radius + clearance + point_radius) ** 2
        rel = point - pipe_center
        radial = rel - cs.dot(rel, axis) * axis
        return cs.dot(radial, radial) - min_radius_sq

    def add_constraint(name, point, point_radius, constraint_nodes):
        if not constraint_nodes:
            return
        constraint = prb.createConstraint(name, pipe_wall(point, point_radius),
                                          nodes=constraint_nodes)
        constraint.setBounds(0.0, np.inf)

    def add_frame_constraint(frame, constraint_nodes):
        frame_pos = kin_dyn.fk(frame)(q=q)["ee_pos"]
        add_constraint(
            f"homing_{frame}_pipe_clearance",
            frame_pos,
            inflation_radius,
            constraint_nodes,
        )

    for frame in ARM_GUIDE_FRAMES:
        add_frame_constraint(frame, arm_nodes)
    for idx, (frame, offset, sphere_radius) in enumerate(TOOL_SPHERES):
        fk = kin_dyn.fk(frame)(q=q)
        sphere_center = fk["ee_pos"] + cs.mtimes(fk["ee_rot"], cs.DM(offset))
        add_constraint(
            f"homing_tool_sphere_{idx}_pipe_clearance",
            sphere_center,
            sphere_radius,
            tool_nodes,
        )


def _plan_horizon(kin_dyn, q_start, q_goal, center, radius, orientation,
                  clearance, inflation_radius, motion_weight,
                  accel_weight, max_joint_step, max_joint_accel_step,
                  tool_approach_nodes, ipopt_print_level,
                  initial_guess_mode, nodes):
    print("[homing_trajectory] Building Horizon problem...", flush=True)
    prb = Problem(nodes, casadi_type=cs.SX)
    prb.setDt(1.0 / nodes)
    q = prb.createVariable("q_homing", q_start.size)
    actuated_weight = cs.diag(cs.DM(
        [0.0] * 7 + [motion_weight] * (q_start.size - 7)))
    accel_track_weight = cs.diag(cs.DM(
        [0.0] * 7 + [accel_weight] * (q_start.size - 7)))
    arm_selector = cs.diag(cs.DM([0.0] * 7 + [1.0] * (q_start.size - 7)))

    q_min = np.asarray(kin_dyn.q_min(), dtype=float).reshape(-1)
    q_max = np.asarray(kin_dyn.q_max(), dtype=float).reshape(-1)
    if q_min.size == q_start.size:
        q[7:].setBounds(q_min[7:], q_max[7:])

    for idx in range(q_start.size):
        if abs(q_start[idx] - q_goal[idx]) < 1e-9:
            q[idx].setBounds(q_start[idx], q_start[idx])

    q.setBounds(q_start, q_start, nodes=0)
    q.setBounds(q_goal, q_goal, nodes=nodes)
    q.setInitialGuess(_initial_guess(
        q_start, q_goal, nodes, initial_guess_mode))
    has_arm = q_start.size > 7
    if has_arm and motion_weight > 0.0:
        prb.createResidual(
            "min_homing_arm_step",
            cs.mtimes(actuated_weight, q.getVarOffset(1) - q),
            nodes=range(nodes),
        )
    if has_arm and accel_weight > 0.0:
        prb.createResidual(
            "min_homing_arm_accel",
            cs.mtimes(
                accel_track_weight,
                q.getVarOffset(2) - 2.0 * q.getVarOffset(1) + q,
            ),
            nodes=range(nodes - 1),
        )
    if has_arm and max_joint_step > 0.0:
        max_step = np.full(q_start.size, max_joint_step)
        step_constraint = prb.createConstraint(
            "max_homing_arm_step",
            cs.mtimes(arm_selector, q.getVarOffset(1) - q),
            nodes=range(nodes),
        )
        step_constraint.setBounds(-max_step, max_step)
    if has_arm and max_joint_accel_step > 0.0:
        max_accel = np.full(q_start.size, max_joint_accel_step)
        accel_constraint = prb.createConstraint(
            "max_homing_arm_accel",
            cs.mtimes(
                arm_selector,
                q.getVarOffset(2) - 2.0 * q.getVarOffset(1) + q,
            ),
            nodes=range(nodes - 1),
        )
        accel_constraint.setBounds(-max_accel, max_accel)
    print(
        "[homing_trajectory] Adding pipe clearance constraints: "
        f"arm_frames={len(ARM_GUIDE_FRAMES)}, "
        f"tool_spheres={len(TOOL_SPHERES)}",
        flush=True,
    )
    _pipe_clearance_constraints(
        prb, q, kin_dyn, center, radius, orientation, clearance,
        inflation_radius, tool_approach_nodes, nodes)

    print("[homing_trajectory] Creating IPOPT solver...", flush=True)
    solver = Solver.make_solver("ipopt", prb, {
        "ipopt.print_level": ipopt_print_level,
        "ipopt.max_iter": 500,
        "ipopt.tol": 1e-4,
        "print_time": int(ipopt_print_level > 0),
    })
    print("[homing_trajectory] Starting IPOPT solve.", flush=True)
    if not solver.solve():
        raise RuntimeError("Horizon failed to find a homing trajectory")

    return solver.getSolutionDict()["q_homing"]


def plan_homing_trajectory(urdf, srdf, kin_dyn, q_start, q_goal, pipe_center,
                           pipe_radius, pipe_length, pipe_gap,
                           pipe_orientation, duration=5.0, dt=0.02,
                           planner_nodes=30, clearance=0.0,
                           inflation_radius=DEFAULT_INFLATION_RADIUS,
                           tool_approach_nodes=DEFAULT_TOOL_APPROACH_NODES,
                           motion_weight=DEFAULT_MOTION_WEIGHT,
                           accel_weight=DEFAULT_ACCEL_WEIGHT,
                           max_joint_step=DEFAULT_MAX_JOINT_STEP,
                           max_joint_accel_step=DEFAULT_MAX_JOINT_ACCEL_STEP,
                           ipopt_print_level=DEFAULT_IPOPT_PRINT_LEVEL,
                           initial_guess_mode=DEFAULT_INITIAL_GUESS_MODE,
                           candidate_callback=None):
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
    pipe_center = np.asarray(pipe_center, dtype=float).reshape(3)
    pipe_orientation = np.asarray(pipe_orientation, dtype=float).reshape(4)
    steps = max(1, int(round(duration / dt)))
    checker = _PipeChecker(
        urdf, srdf, pipe_center, pipe_radius, pipe_length, pipe_gap,
        pipe_orientation, clearance)

    direct = _interpolate(q_start, q_goal, steps)
    print(
        "[homing_trajectory] Checking direct path: "
        f"{steps + 1} samples, dt={duration / steps:.4f}s"
    )
    collision = _first_collision(direct, checker)
    if collision is None:
        print("[homing_trajectory] Direct path accepted.")
        return HomingTrajectory(q=direct, dt=duration / steps, method="direct")

    node, pairs = collision
    print(
        "[homing_trajectory] Direct path collides at "
        f"sample {node}/{direct.shape[1] - 1}: {pairs}"
    )
    if node in (0, direct.shape[1] - 1):
        raise RuntimeError(
            f"Homing endpoint collides with the pipe at node {node}: {pairs}")

    print(
        "[homing_trajectory] Solving Horizon homing: "
        f"nodes={planner_nodes}, inflation_radius={inflation_radius:.4f}, "
        f"tool_approach_nodes={tool_approach_nodes}, "
        f"max_joint_step={max_joint_step:.4f}, "
        f"max_joint_accel_step={max_joint_accel_step:.4f}, "
        f"ipopt_print_level={ipopt_print_level}, "
        f"initial_guess={initial_guess_mode}"
    )
    planned = _plan_horizon(
        kin_dyn, q_start, q_goal, pipe_center, pipe_radius, pipe_orientation,
        clearance, inflation_radius, motion_weight, accel_weight,
        max_joint_step, max_joint_accel_step, tool_approach_nodes,
        ipopt_print_level, initial_guess_mode, planner_nodes)
    dense = _resample(planned, steps)
    print(
        "[homing_trajectory] Horizon solved; validating dense replay: "
        f"{dense.shape[1]} samples"
    )
    collision = _first_collision(dense, checker)
    if collision is not None:
        node, pairs = collision
        print(
            "[homing_trajectory] Horizon candidate rejected at "
            f"sample {node}/{dense.shape[1] - 1}: {pairs}"
        )
        if candidate_callback is not None:
            candidate_callback(
                HomingTrajectory(
                    q=dense,
                    dt=duration / steps,
                    method="horizon_rejected",
                ),
                node,
                pairs,
            )
        raise RuntimeError(
            f"Horizon homing trajectory still collides at node {node}: {pairs}")

    print("[homing_trajectory] Horizon path accepted.")
    return HomingTrajectory(q=dense, dt=duration / steps, method="horizon")
