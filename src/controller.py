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

import math
import threading
from pathlib import Path
from time import sleep, perf_counter

import numpy as np
import rclpy
from rcl_interfaces.srv import GetParameters
from xbot2_interface import pyxbot2_interface as xbi
from xbot2_interface.pyaffine3 import Affine3
from cartesian_interface import pyci

# ── Parameters ───────────────────────────────────────────────────────────────
TASK_NAME   = 'ee_F'   # CartesIO task name for the welding end-effector
TRJ_HALF    = 0.5     # [m]   half-length of the back-and-forth stroke
TRJ_PERIOD  = 10.0     # [s]   time for one full back-and-forth cycle
DT          = 0.01     # [s]   controller dt (100 Hz)
WARMUP_TICKS = 200     # ticks to let solver converge before starting (with integration)
# Hard clamp on joint velocity (rad/s or m/s) to prevent snapping
MAX_JOINT_VEL = 0.5    # [rad/s or m/s]

# Path to the CartesIO problem description YAML
CARTESIO_YAML = Path('/home/user/concert_ws/src/acea_concert/config/cartesio_stack.yaml')

# ── Read URDF/SRDF from robot_description_publisher ───────────────────────────
print("[controller] Waiting for robot_description ROS parameters …")
rclpy.init()

_spin_node     = rclpy.create_node('controller_spin')
_spin_executor = rclpy.executors.SingleThreadedExecutor()
_spin_executor.add_node(_spin_node)
_spin_thread   = threading.Thread(target=_spin_executor.spin, daemon=True)
_spin_thread.start()

def _fetch_robot_description():
    node   = rclpy.create_node('controller_urdf_reader')
    client = node.create_client(GetParameters, '/robot_description_publisher/get_parameters')
    if not client.wait_for_service(timeout_sec=15.0):
        raise RuntimeError("[controller] /robot_description_publisher not available. Is the simulation running?")
    req       = GetParameters.Request()
    req.names = ['robot_description', 'robot_description_semantic']
    future    = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
    node.destroy_node()
    if future.result() is None:
        raise RuntimeError("[controller] Failed to read robot_description parameters.")
    vals = future.result().values
    return vals[0].string_value, vals[1].string_value

urdf, srdf = _fetch_robot_description()
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

# Sync model to robot: actuated joints
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

ee_task = ci.getTask(TASK_NAME)
if ee_task is None:
    raise RuntimeError(f"[controller] Task '{TASK_NAME}' not found. Check cartesio_stack.yaml.")

print(f"[controller] All tasks: {ci.getTaskList()}")

ee_distal = ee_task.getDistalLink()
ee_base   = ee_task.getBaseLink()
print(f"[controller] Task '{TASK_NAME}': {ee_base} → {ee_distal}")

# ── Latch starting EE pose as trajectory centre ───────────────────────────────
T_start = model.getPose(ee_distal, ee_base)
centre  = T_start.translation.copy()
print(f"[controller] T_start (world frame): {centre}")
print(f"[controller] T_start rotation:\n{T_start.linear}")
omega = 2.0 * math.pi / TRJ_PERIOD


print("[controller] Starting control loop …")

# ── Control loop ──────────────────────────────────────────────────────────────
t = 0.0

while True:
    t0 = perf_counter()

    # ── 1) Set EE reference — back-and-forth along world X ──────────────────
    # T_start.translation is in the EE frame; offsets must also be in EE frame.
    # Rotate the world-X unit vector into the EE frame: R^T * [1,0,0]
    world_x_in_ee = T_start.linear.T @ np.array([1.0, 0.0, 0.0])
    offset = TRJ_HALF * math.sin(omega * t) * world_x_in_ee
    target = Affine3(pos=centre + offset, rot=T_start.quaternion)
    ee_task.setPoseReference(target)

    # ── 2) IK step — writes model.v ─────────────────────────────────────────
    ci.update(t, DT)


    # ── 3) Integrate model state ───────────────────────
    model.setJointPosition(model.sum(model.q, model.v * DT))
    model.update()

    # ── 4) Send to robot (actuated joints only) ──────────────────────────────
    robot.setPositionReference(model.getJointPosition())
    # _send_model_q_to_robot()
    robot.move()

    t += DT

    # ── 5) Pace the loop ─────────────────────────────────────────────────────
    elapsed = perf_counter() - t0
    sleep(max(0.0, DT - elapsed))
