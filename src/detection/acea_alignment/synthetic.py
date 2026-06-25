"""Synthetic RGB-D cylinder renderer with exact ground-truth pose.

Why this exists
---------------
The logged depth sequences on this checkout are truncated/empty, and even when
they are intact they have no per-frame analytic ground truth for the pipe axis.
To validate the *geometry and the sign conventions* of the deterministic
estimator and (later) the controller, we render a cylinder analytically: we know
its axis, radius, stand-off, lateral offset and yaw exactly, so we can assert the
estimator recovers them and that the closed loop drives the error to zero.

The renderer matches the camera optical frame used everywhere else:
    +x right, +y down, +z forward, depth = z component of the surface point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def default_intrinsics() -> np.ndarray:
    """The Isaac RGB-D intrinsics used in this project (640x480, fx=fy=733)."""
    return np.array([
        [732.999268, 0.0, 320.0],
        [0.0, 732.999268, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def yaw_axis(yaw_deg: float) -> np.ndarray:
    """Pipe axis direction for a given yaw, in the camera frame.

    Base axis is +x (pipe runs left->right across the image). A positive yaw is
    a rotation about the camera +y (down) axis. With axis(yaw) = (cos, 0, -sin),
    ``atan2(axis_z, axis_x) = -yaw``; i.e. the estimator's ``yaw_error_deg`` has
    the opposite sign to this rotation parameter. The eval asserts exactly this
    relationship so the sign is pinned, not assumed.
    """
    psi = np.radians(yaw_deg)
    return np.array([np.cos(psi), 0.0, -np.sin(psi)], dtype=np.float64)


@dataclass
class SyntheticScene:
    depth: np.ndarray
    k: np.ndarray
    gt_axis_camera: np.ndarray   # unit, canonicalized to x>0
    gt_axis_point_camera: np.ndarray  # nearest axis point to camera origin
    gt_radius_m: float
    gt_yaw_error_deg: float
    gt_axis_distance_m: float
    gt_stand_off_m: float          # median surface z (matches estimator convention)
    gt_surface_distance_m: float   # perpendicular camera->surface (axis_distance - r)
    gt_lateral_offset_m: float
    gt_vertical_offset_m: float


def render_cylinder(
    *,
    axis_distance_m: float = 1.58,
    radius_m: float = 0.45,
    yaw_deg: float = 0.0,
    lateral_offset_m: float = 0.0,
    vertical_offset_m: float = 0.0,
    length_m: float = 2.0,
    k: np.ndarray | None = None,
    width: int = 640,
    height: int = 480,
    background_m: float = 6.0,
    noise_m: float = 0.0,
    seed: int = 0,
) -> SyntheticScene:
    """Render a depth image of one cylinder with known ground-truth pose.

    The cylinder axis passes through the point
    ``(lateral_offset_m, vertical_offset_m, axis_distance_m)`` (the nearest axis
    point to the camera) with direction ``yaw_axis(yaw_deg)``.
    """
    if k is None:
        k = default_intrinsics()
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    axis = yaw_axis(yaw_deg)
    if axis[0] < 0.0:
        axis = -axis
    # Nearest axis point to the camera; offset it slightly off the perpendicular
    # plane so the axial extent is exercised, then recompute the true nearest
    # point for ground truth.
    axis_point0 = np.array([lateral_offset_m, vertical_offset_m, axis_distance_m])

    # Per-pixel ray directions (optical frame), z-component = 1 so depth = z.
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    dir_x = (uu - cx) / fx
    dir_y = (vv - cy) / fy
    dir_z = np.ones_like(dir_x)
    rays = np.stack([dir_x, dir_y, dir_z], axis=-1)  # (H, W, 3), not normalized

    # Infinite-cylinder intersection: point O + t*r, axis line through C dir A.
    # Solve |(O + t r - C) - ((..)·A)A|^2 = radius^2 for the smallest t>0.
    o_minus_c = -axis_point0  # camera origin is 0
    r_perp = rays - (rays @ axis)[..., None] * axis           # ray component perp to axis
    w_perp = o_minus_c - (o_minus_c @ axis) * axis            # (3,)
    a_q = np.einsum("hwc,hwc->hw", r_perp, r_perp)
    b_q = 2.0 * np.einsum("hwc,c->hw", r_perp, w_perp)
    c_q = float(np.dot(w_perp, w_perp)) - radius_m * radius_m
    disc = b_q * b_q - 4.0 * a_q * c_q

    depth = np.full((height, width), background_m, dtype=np.float32)
    hit = disc > 0.0
    sqrt_disc = np.zeros_like(disc)
    sqrt_disc[hit] = np.sqrt(disc[hit])
    with np.errstate(invalid="ignore", divide="ignore"):
        t0 = (-b_q - sqrt_disc) / (2.0 * a_q)  # near intersection
    valid_hit = hit & (t0 > 0.0)

    hit_pts = rays * t0[..., None]
    # Clip to a finite cylinder length about axis_point0 along the axis.
    axial = (hit_pts - axis_point0) @ axis
    within_len = np.abs(axial) <= (length_m * 0.5)
    surface = valid_hit & within_len & np.isfinite(t0)

    clean_surface_z = hit_pts[..., 2][surface].astype(np.float64)
    depth[surface] = clean_surface_z.astype(np.float32)

    # Ground-truth stand-off in the estimator's convention (median surface z),
    # computed from the clean render before any noise is added.
    gt_stand_off = float(np.median(clean_surface_z)) if clean_surface_z.size else float("nan")

    if noise_m > 0.0:
        rng = np.random.default_rng(seed)
        depth[surface] += rng.normal(0.0, noise_m, size=int(surface.sum())).astype(np.float32)

    # Ground truth nearest axis point to camera origin = component of (-C) removed
    # along the axis: foot = C - (C·A)A.
    foot = axis_point0 - float(axis_point0 @ axis) * axis
    gt_axis_distance = float(np.linalg.norm(foot))
    gt_surface_distance = gt_axis_distance - radius_m  # perpendicular to near surface
    gt_yaw = float(np.degrees(np.arctan2(axis[2], axis[0])))

    return SyntheticScene(
        depth=depth,
        k=k,
        gt_axis_camera=axis,
        gt_axis_point_camera=foot,
        gt_radius_m=radius_m,
        gt_yaw_error_deg=gt_yaw,
        gt_axis_distance_m=gt_axis_distance,
        gt_stand_off_m=gt_stand_off,
        gt_surface_distance_m=gt_surface_distance,
        gt_lateral_offset_m=float(foot[0]),
        gt_vertical_offset_m=float(foot[1]),
    )
