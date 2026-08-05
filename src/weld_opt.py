import argparse
from dataclasses import replace
import os

import sys

from pathlib import Path

from horizon.ros import replay_trajectory

from circular_trajectory import generate_circular_trajectory

from collision_checker import CollisionChecker
from weld_robot_config import weld_robot_config
from weld_opt_result import build_weld_result, store_weld_result
from weld_opt_solver import (
    build_weld_problem,
    make_sobol_base_points,
    solve_weld_problem,
)
from weld_opt_runtime import (
    weld_opt_runtime_from_env,
    weld_output_path,
    weld_scenario_from_env,
)
import numpy as np

import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-prismatic-joint",
        action="store_true",
        help="Generate the prismatic-cart robot model for optimization.",
    )
    parser.add_argument(
        "--upside-down",
        dest="upside_down",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Weld with the torch upside down. Overrides the "
            "WELD_OPT_WELD_UPSIDE_DOWN environment variable / batch scenario "
            "setting when passed explicitly (use --no-upside-down to force "
            "it off)."
        ),
    )
    parser.add_argument(
        "--base-search",
        nargs="?",
        type=int,
        const=8,
        default=None,
        metavar="SAMPLES",
        help=(
            "Evaluate Sobol-sampled base XY positions and keep the "
            "collision-free solution with the lowest peak critical-joint "
            "torque. Defaults to 8 samples when no value is given."
        ),
    )
    parsed = parser.parse_args()
    if parsed.base_search is not None and parsed.base_search < 1:
        parser.error("--base-search must use at least one sample")
    return parsed


args = parse_args()
runtime = weld_opt_runtime_from_env()
scenario = weld_scenario_from_env()
if args.upside_down is not None:
    scenario = replace(scenario, upside_down=args.upside_down)
if runtime.seed is not None:
    np.random.seed(runtime.seed)


'''
Initialize Horizon problem
'''
ns = 30
T = 2.0
dt = T / ns

# ========================================================
# ========================================================

PATH_TO_CONCERT_WS = Path("/home/user/concert_ws")
PATH_TO_ACEA_CONCERT = PATH_TO_CONCERT_WS/"src"/"acea_concert"
# modular_prismatic = PATH_TO_CONCERT_WS/"src"/"concert_description"/"concert_examples"/"src"/"concert_prismatic.py"
modular_prismatic = PATH_TO_ACEA_CONCERT/"src"/"modular"/"concert_with_torch.py"
horizon_config = PATH_TO_ACEA_CONCERT/"config"/"weld.yaml"
robot_config = weld_robot_config(args.use_prismatic_joint)
ee_link = robot_config.ee_link
urdf, srdf = robot_config.robot_description(modular_prismatic)
task_config = robot_config.write_task_yaml(
    horizon_config,
    Path("/tmp/concert_weld.yaml"),
)

with open('/tmp/concert_weld.urdf', 'w') as f:
    f.write(urdf)

with open('/tmp/concert_weld.srdf', 'w') as f:
    f.write(srdf)

subprocess.run(f'moveit_compute_default_collisions --urdf_path /tmp/concert_weld.urdf --srdf_path /tmp/concert_weld.srdf', shell=True, check=True)

with open('/tmp/concert_weld.srdf', 'r') as f:
    srdf = f.read()

# ========================================================

# Launch robot_state_publisher in background with the URDF when something will
# publish or replay a ROS visualization.
rsp_process = None
if not (runtime.skip_rviz_scene and runtime.skip_replay):
    rsp_process = subprocess.Popen(
        ["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args", "-p", f"robot_description:={urdf}"],
    )

# ========================================================

footprint_robot_x = 1.2
footprint_robot_y = 0.7

length_pipe = 5.0
pipe_gap = 0.005
pos_center_pipe = [1.5, 0.0, 0.587 + 0.45 + 0.3 + 0.05]  # nominal pipe center position in world frame
orientation_pipe = [0.7071068, 0.0, 0.0, 0.7071068]
radius_pipe = 0.1 # 0.15, 0.25, 0.35 available

OPTIMIZE_PIPE_HEIGHT = False
margin_around_pipe_height = 1.0 # how much the pipe height can be optimized around the nominal height (in both directions)
pipe_z_bounds = (pos_center_pipe[2] - margin_around_pipe_height, pos_center_pipe[2] + margin_around_pipe_height)
MINIMIZE_CRITICAL_JOINT_TORQUES = True
TORQUE_COST_WEIGHT = 2e-1
# CRITICAL_TORQUE_JOINTS = ('J1_E', 'J2_E', 'J1_F', 'J2_F', 'J3_F', 'J4_F', 'J5_F', 'J6_F')
CRITICAL_TORQUE_JOINTS = ('J2_E', 'J4_E')

weld_standoff_from_pipe = 0.02  # EE path distance from pipe surface [m]
weld_trajectory_radius = radius_pipe + weld_standoff_from_pipe
if weld_trajectory_radius <= 0.0:
    raise ValueError(
        'weld_trajectory_radius must be positive. Check radius_pipe and '
        'weld_standoff_from_pipe.'
    )

trajectory_scenario_name = scenario.name
angle_weld_start = scenario.angle_start
angle_weld_end = scenario.angle_end

# Manual default: edit this directly to weld upside down when running
# weld_opt.py by hand. Automatically overridden by run_weld_opt_batch.py
# through the WELD_OPT_WELD_UPSIDE_DOWN env var, or by passing
# --upside-down / --no-upside-down on the command line.
weld_upside_down = False
if args.upside_down is not None:
    weld_upside_down = args.upside_down
elif 'WELD_OPT_WELD_UPSIDE_DOWN' in os.environ:
    weld_upside_down = scenario.upside_down

margin_x = 0.5 # Some margin around the pipe w.r.t. the initial robot position (the robot cannot start too close to the pipe)
bound_initial_pos_x_low = -0.5 # 0 is good!
bound_initial_pos_x_high = pos_center_pipe[0] - radius_pipe - footprint_robot_x/2 - margin_x 

bound_initial_pos_y_low = -1.5 + footprint_robot_y/2
bound_initial_pos_y_high = 1.5 - footprint_robot_y/2

base_bounds = (
    bound_initial_pos_x_low,
    bound_initial_pos_x_high,
    bound_initial_pos_y_low,
    bound_initial_pos_y_high,
)
base_search_points = make_sobol_base_points(
    args.base_search,
    base_bounds,
    runtime.seed,
)
if base_search_points is not None:
    print(f"[weld_opt] Searching {args.base_search} Sobol base positions.")

# Draw rectangle in RViz covering all possible random XY positions
center_x = (bound_initial_pos_x_low + bound_initial_pos_x_high) / 2.0
center_y = (bound_initial_pos_y_low + bound_initial_pos_y_high) / 2.0
size_x = abs(bound_initial_pos_x_high - bound_initial_pos_x_low)
size_y = abs(bound_initial_pos_y_high - bound_initial_pos_y_low)

# Generate trajectory and get initial desired pose
circular_pos, circular_ori = generate_circular_trajectory(
    ns,
    center=pos_center_pipe,
    radius=weld_trajectory_radius,
    angle_start=angle_weld_start,
    angle_end=angle_weld_end,
    upside_down=weld_upside_down
)
pipe_center_nominal = np.asarray(pos_center_pipe, dtype=float).reshape(3, 1)
circle_offset = circular_pos - pipe_center_nominal

# =================================================================================
# Initialize collision checker
def make_collision_checker(pipe_center):
    coll_checker = CollisionChecker(urdf, srdf)
    coll_checker.add_pipe(
        'weld_pipe', radius_pipe, length_pipe, pipe_center, orientation_pipe)
    return coll_checker
# =================================================================================

# Initialize RViz scene with pipe, footprint, and trajectory markers
from viz.init_scene import InitScene

def launch_rviz_scene(pipe_center, trajectory):
    if runtime.skip_rviz_scene:
        return

    init_scene = InitScene(
        path_ws=PATH_TO_ACEA_CONCERT/"src"/"viz",
        pos_center_pipe=pipe_center,
        radius_pipe=radius_pipe,
        length_pipe=length_pipe,
        orientation_pipe=orientation_pipe,
        footprint_robot_x=footprint_robot_x,
        footprint_robot_y=footprint_robot_y,
        center_x=center_x,
        center_y=center_y,
        size_x=size_x,
        size_y=size_y,
        position=trajectory,
    )
    init_scene.kill_existing_markers()
    init_scene.launch_scene()


launch_rviz_scene(pos_center_pipe, circular_pos)

print(
    f"circular trajectory applied: center={pos_center_pipe}, "
    f"pipe_radius={radius_pipe}, "
    f"standoff={weld_standoff_from_pipe}, "
    f"trajectory_radius={weld_trajectory_radius}"
)
print(f"angle range: [{angle_weld_start}, {angle_weld_end}] rad, steps: {ns + 1}")

problem = build_weld_problem(
    urdf=urdf,
    task_config=task_config,
    ee_link=ee_link,
    n_intervals=ns,
    duration=T,
    circular_orientation=circular_ori,
    circle_offset=circle_offset,
    nominal_pipe_center=pos_center_pipe,
    optimize_pipe_height=OPTIMIZE_PIPE_HEIGHT,
    pipe_z_bounds=pipe_z_bounds,
    minimize_critical_joint_torques=MINIMIZE_CRITICAL_JOINT_TORQUES,
    critical_torque_joints=CRITICAL_TORQUE_JOINTS,
    torque_cost_weight=TORQUE_COST_WEIGHT,
)
prb = problem.problem
kin_dyn = problem.kin_dyn
model = problem.model
ti = problem.task_interface
critical_torque_indices = problem.critical_torque_indices
missing_critical_torque_joints = problem.missing_critical_torque_joints
id_fn_cost = problem.inverse_dynamics

# =================================================================================
# matfile = os.path.join(os.path.dirname(__file__), '../mat_files/weld_concert_very_good.mat')
# if not os.path.exists(matfile):
#     print(f"File not found: {matfile}")
#     sys.exit(1)

# data = loadmat(matfile)
# q_init = data.get('q')  # Use the first column of q as the initial guess
# =================================================================================
point_pub = None
on_base_candidate = None
if not runtime.skip_rviz_scene:
    import rclpy
    from viz.rviz_point_marker import PersistentPointSpawner

    rclpy.init()

    point_pub = PersistentPointSpawner('initial_points',
                                       'world',
                                       1.0, 0.0, 0.0, 1.0) # color RGBA for the points

    def on_base_candidate(base_x, base_y):
        point_pub.add_point(base_x, base_y, 0.1)
        rclpy.spin_once(point_pub, timeout_sec=0.0)

solution = solve_weld_problem(
    task_interface=ti,
    n_intervals=ns,
    base_bounds=base_bounds,
    base_search_points=base_search_points,
    max_random_attempts=runtime.max_random_initial_pose_attempts,
    nominal_pipe_center=pos_center_pipe,
    optimize_pipe_height=OPTIMIZE_PIPE_HEIGHT,
    make_collision_checker=make_collision_checker,
    inverse_dynamics=id_fn_cost,
    critical_torque_indices=critical_torque_indices,
    on_base_candidate=on_base_candidate,
)

result = build_weld_result(
    solution=solution,
    problem=prb,
    kin_dyn=kin_dyn,
    nominal_pipe_center=pos_center_pipe,
    circle_offset=circle_offset,
    circular_orientation=circular_ori,
    optimize_pipe_height=OPTIMIZE_PIPE_HEIGHT,
    metadata=dict(
        pipe_z_bounds=np.asarray(pipe_z_bounds),
        minimize_critical_joint_torques=MINIMIZE_CRITICAL_JOINT_TORQUES,
        critical_torque_joint_names=np.asarray(CRITICAL_TORQUE_JOINTS),
        critical_torque_indices=np.asarray(critical_torque_indices, dtype=int),
        missing_critical_torque_joints=np.asarray(
            missing_critical_torque_joints),
        torque_cost_weight=TORQUE_COST_WEIGHT,
        orientation_pipe=orientation_pipe,
        radius_pipe=radius_pipe,
        weld_standoff_from_pipe=weld_standoff_from_pipe,
        weld_trajectory_radius=weld_trajectory_radius,
        length_pipe=length_pipe,
        pipe_gap=pipe_gap,
        initial_zone_center_x=center_x,
        initial_zone_center_y=center_y,
        initial_zone_size_x=size_x,
        initial_zone_size_y=size_y,
        footprint_robot_x=footprint_robot_x,
        footprint_robot_y=footprint_robot_y,
        trajectory_scenario_name=trajectory_scenario_name,
        angle_weld_start=angle_weld_start,
        angle_weld_end=angle_weld_end,
        angle_weld_span=angle_weld_end - angle_weld_start,
        weld_upside_down=weld_upside_down,
    ),
)
if OPTIMIZE_PIPE_HEIGHT:
    print(f"[weld_opt] Optimized pipe height: z={result.pipe_z:.4f} m")
else:
    print(f"[weld_opt] Fixed pipe height: z={result.pipe_z:.4f} m")
launch_rviz_scene(result.pipe_center, result.trajectory)

mat_file_path = weld_output_path(PATH_TO_ACEA_CONCERT)
store_weld_result(mat_file_path, result)
print(f"[weld_opt] Saved optimization result to: {mat_file_path}")

print(f"[INFO] Final initial robot pose: {solution['q'][:3, 0]}")

if runtime.skip_replay:
    if rsp_process is not None:
        rsp_process.terminate()
    sys.exit(0)


q_forward = solution['q']
q_backward = np.flip(q_forward, axis=1)
q_cycle = np.concatenate([q_forward, q_backward], axis=1)

contact_list_repl = list(model.cmap.keys())
repl = replay_trajectory.replay_trajectory(dt, model.kd.joint_names(), q_cycle,
                                        {k: None for k in model.fmap.keys()},
                                        model.kd_frame, model.kd,
                                        trajectory_markers=contact_list_repl,
                                        future_trajectory_markers={ee_link: 'world'})

repl.replay()

if rsp_process is not None:
    rsp_process.terminate()
