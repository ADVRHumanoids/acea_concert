"""Offline closed-loop alignment model: derive the control signs and PROVE
convergence from large misalignment, using the *existing* scan-control law.

This does NOT replace the robot's motion stack. Omnisteering/XBot2 and the
``acea_pipe_junction_scan_control_node`` FSM + P-control already exist. This
module only:

  1. models the geometry  base pose -> camera -> pipe-pose-in-camera-frame
     (``forward_observation``), faithfully to the estimator's conventions;
  2. replicates the EXACT scan-control formulas (``control_law``), so the proof
     is about the real controller, not a different one;
  3. DERIVES the correct ``yaw_axis_sign`` / ``stand_off_axis_sign`` from the
     camera mount + the standard ROS twist convention (``derive_signs``), instead
     of finding them by trial as the project did before;
  4. simulates the closed loop (``simulate_closed_loop``) to show the robot
     converges even from a large initial yaw/stand-off error.

The one assumption that still needs a single live yes/no: that Omnisteering
follows REP-103 (``+angular.z`` = counter-clockwise about world up). Everything
else is derived.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation


# --- Geometry from the Isaac scene metadata (data/.../frame_000060) ---------
@dataclass
class Mount:
    """Camera mounting on base_link (Isaac ``offset_cfg``)."""
    pos_base_cam_m: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.5]))
    quat_base_cam_wxyz: np.ndarray = field(default_factory=lambda: np.array([0.5, -0.5, 0.5, -0.5]))
    base_height_m: float = 0.747  # robot_base.root_position_world z

    def r_base_cam(self) -> Rotation:
        w, x, y, z = self.quat_base_cam_wxyz
        return Rotation.from_quat([x, y, z, w])


@dataclass
class PipeScene:
    """Pipe fixed in the world (Isaac: axis along +Y)."""
    axis_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    point_world: np.ndarray = field(default_factory=lambda: np.array([2.0, 0.0, 1.0]))
    radius_m: float = 0.45


@dataclass
class BasePose:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # heading about world +Z (rad)


def _canonical(axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return -axis if axis[0] < 0.0 else axis


def forward_observation(base: BasePose, scene: PipeScene, mount: Mount) -> dict[str, float]:
    """Pipe pose in the camera optical frame for a given base pose.

    Faithful to ``pipe_pose.fit_pipe_pose``: yaw_error = atan2(axis_z, axis_x)
    with axis canonicalised to x>0; surface_distance = |perp camera->axis| - r.
    """
    r_wb = Rotation.from_euler("z", base.theta)
    r_wc = r_wb * mount.r_base_cam()                      # camera->world rotation
    cam_world = (np.array([base.x, base.y, mount.base_height_m])
                 + r_wb.apply(mount.pos_base_cam_m))

    axis_cam = _canonical(r_wc.inv().apply(scene.axis_world))
    p0_cam = r_wc.inv().apply(scene.point_world - cam_world)
    foot = p0_cam - float(p0_cam @ axis_cam) * axis_cam   # nearest axis point to camera
    axis_distance = float(np.linalg.norm(foot))
    return {
        "yaw_error_deg": math.degrees(math.atan2(float(axis_cam[2]), float(axis_cam[0]))),
        "axis_distance_m": axis_distance,
        "surface_distance_m": max(0.0, axis_distance - scene.radius_m),
        "lateral_offset_m": float(foot[0]),
        "vertical_offset_m": float(foot[1]),
    }


# --- Control law: EXACTLY the existing scan-control formulas -----------------
# acea_pipe_junction_scan_control_node.py:
#   angular_z = yaw_axis_sign * -yaw_kp * radians(yaw_error)          (line 320)
#   linear_x  = stand_off_axis_sign * stand_off_kp * stand_off_error  (line 311)
DEFAULT_CONTROL_PARAMS: dict[str, float] = {
    "yaw_kp": 0.40,
    "yaw_align_deadband_deg": 8.0,
    "max_angular_speed_radps": 0.10,
    "yaw_axis_sign": 1.0,            # derived by derive_signs()
    "stand_off_kp": 0.20,
    "target_stand_off_m": 1.30,
    "stand_off_deadband_m": 0.15,
    "max_lateral_speed_mps": 0.03,
    "stand_off_axis_sign": 1.0,      # derived by derive_signs()
}


def _clamp(value: float, limit: float) -> float:
    return max(-abs(limit), min(abs(limit), value))


def control_law(obs: dict[str, float], p: dict[str, float]) -> dict[str, float]:
    """Base-frame Twist from a pose observation (yaw + stand-off alignment)."""
    yaw = obs["yaw_error_deg"]
    if abs(yaw) <= float(p["yaw_align_deadband_deg"]):
        wz = 0.0
    else:
        wz = _clamp(float(p["yaw_axis_sign"]) * -float(p["yaw_kp"]) * math.radians(yaw),
                    float(p["max_angular_speed_radps"]))

    err = obs["surface_distance_m"] - float(p["target_stand_off_m"])
    if abs(err) <= float(p["stand_off_deadband_m"]):
        vx = 0.0
    else:
        vx = _clamp(float(p["stand_off_axis_sign"]) * float(p["stand_off_kp"]) * err,
                    float(p["max_lateral_speed_mps"]))
    return {"linear_x": vx, "linear_y": 0.0, "angular_z": wz}


def derive_signs(base: BasePose, scene: PipeScene, mount: Mount,
                 eps_deg: float = 1.0, eps_m: float = 0.02) -> dict[str, float]:
    """Derive yaw_axis_sign / stand_off_axis_sign from the geometry.

    For the existing formulas, negative feedback requires (see module docstring
    derivation):  yaw_axis_sign = sign(d yaw_error / d theta),
                  stand_off_axis_sign = -sign(d surface_distance / d forward).
    """
    th = base.theta
    yp = forward_observation(BasePose(base.x, base.y, th + math.radians(eps_deg)), scene, mount)["yaw_error_deg"]
    ym = forward_observation(BasePose(base.x, base.y, th - math.radians(eps_deg)), scene, mount)["yaw_error_deg"]
    g_yaw = (yp - ym) / (2.0 * eps_deg)  # deg yaw_error per deg base yaw

    fwd = np.array([math.cos(th), math.sin(th)])  # base +x in world
    sp = forward_observation(BasePose(base.x + eps_m * fwd[0], base.y + eps_m * fwd[1], th), scene, mount)["surface_distance_m"]
    sm = forward_observation(BasePose(base.x - eps_m * fwd[0], base.y - eps_m * fwd[1], th), scene, mount)["surface_distance_m"]
    h_off = (sp - sm) / (2.0 * eps_m)

    return {
        "g_yaw": g_yaw,
        "h_stand_off": h_off,
        "yaw_axis_sign": float(np.sign(g_yaw)) or 1.0,
        "stand_off_axis_sign": float(-np.sign(h_off)) or 1.0,
    }


def simulate_closed_loop(base0: BasePose, scene: PipeScene, mount: Mount,
                         p: dict[str, float], steps: int = 900, dt: float = 0.1
                         ) -> list[dict[str, float]]:
    """Run the alignment loop. Returns a per-step trajectory."""
    bx, by, th = base0.x, base0.y, base0.theta
    traj: list[dict[str, float]] = []
    for i in range(steps):
        obs = forward_observation(BasePose(bx, by, th), scene, mount)
        u = control_law(obs, p)
        traj.append({"t": i * dt, "yaw_error_deg": obs["yaw_error_deg"],
                     "surface_distance_m": obs["surface_distance_m"],
                     "angular_z": u["angular_z"], "linear_x": u["linear_x"]})
        # Integrate the holonomic (Omnisteering) base in the world frame.
        th += u["angular_z"] * dt
        bx += (u["linear_x"] * math.cos(th) - u["linear_y"] * math.sin(th)) * dt
        by += (u["linear_x"] * math.sin(th) + u["linear_y"] * math.cos(th)) * dt
    return traj
