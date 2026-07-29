"""V16 robust pipe-pose estimation with an explicit normal-axis switch.

CAMERA FRAME CONVENTION (the single source of truth for every sign below)
-------------------------------------------------------------------------
This module works in the **camera optical frame**, exactly the frame produced
by ``backproject`` and by the Isaac RGB-D camera (``convention: ros`` optical):

    +x : image right
    +y : image down
    +z : forward, into the scene (this is the depth value)

Consequences, all *derived*, none guessed:

    stand_off    = forward distance to the pipe surface  -> median z of surface pts
    surface_dist = perpendicular camera-origin -> surface = axis_distance - radius
    lateral off. = horizontal axis offset from optical axis -> x of nearest axis pt
    vertical off.= vertical   axis offset from optical axis -> y of nearest axis pt
    yaw_error    = rotation of the pipe axis about the vertical (+y) axis,
                   measured in the horizontal x-z plane -> atan2(axis_z, axis_x)

These conventions match both legacy estimators (``acea_pipe_junction_node.py``
and ``acea_pipe_pose_pcl_node.cpp``), so a pose published from here is a drop-in
for ``/acea/pipe_junction/pipe_pose``.

THE FIT (robust first; deterministic as a bonus)
------------------------------------------------
Robustness to *large* initial misalignment is the priority here, not strict
determinism. (Determinism is the constraint on the junction *detector*, not on
alignment: alignment must be strong even when the robot starts badly misaligned
to the pipe.) The axis is estimated two independent ways and the better one is
kept, judged by the cylinder-fit residual:

    * PCA of the 3D points  -- noise-robust, but collapses when the pipe is
                               close / foreshortened / only partly seen (the
                               longest point spread stops being the axis).
    * surface-normal SVD    -- the axis is perpendicular to every surface normal,
                               so it is robust to orientation and foreshortening;
                               needs Gaussian depth smoothing to survive noise.

    foreground depth -> 3D points
      -> candidate axes: {PCA, normal-SVD on smoothed depth}
      -> keep the axis with the lowest robust circle-fit residual
      -> hold that axis fixed, refine radius / centre / inliers (Kasa + MAD gate)
      -> radius R, axis, nearest axis point, residual, inlier fraction

This recovers yaw correctly through at least +-60 deg, fixes the close-range
PCA collapse, and tolerates depth noise. Every step is closed-form /
eigendecomposition so it also stays deterministic; if tougher real-world clutter
needs it, a (randomised) RANSAC cylinder refinement can be layered on top without
changing this interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Mirrors the defaults already used in acea_pipe_junction_node.py so behaviour
# does not silently change when this module replaces the inline logic.
DEFAULT_PIPE_POSE_PARAMS: dict[str, float | int] = {
    "min_depth_m": 0.05,
    "max_depth_m": 20.0,
    "min_pipe_pixels": 1000,
    "max_fit_points": 60000,
    "sample_stride": 4,
    "consensus_iterations": 3,
    "radius_tolerance_m": 0.08,
    "min_inliers": 200,
    "min_inlier_fraction": 0.35,
    # Surface-normal axis estimation (robust to large yaw / close / foreshortened).
    "normal_smooth_sigma_px": 2.0,
    "normal_jump_threshold_m": 0.05,
    # Acquisition keeps the robust normal-SVD candidate. A caller that has
    # already isolated a clean lock-seeded cylinder surface may disable this
    # redundant full-frame pass and use PCA for that refit only.
    "use_normal_axis": 1,
    # Validity gate on the fitted cylinder (large industrial pipe, ~0.45 m here).
    "radius_min_m": 0.20,
    "radius_max_m": 0.90,
    "max_residual_m": 0.05,
}


@dataclass
class PipePose:
    """A robust pipe-pose estimate, in the camera optical frame."""

    valid: bool
    reason: str
    pose_source: str = "python_robust_cylinder"
    axis_method: str = "pca"  # which axis estimate won: pca | normal_svd
    # Geometry (camera optical frame, metres / unit vector).
    axis_camera_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    axis_point_camera_xyz_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius_m: float = 0.0
    axis_distance_m: float = 0.0      # perpendicular camera-origin -> axis line
    surface_distance_m: float = 0.0   # axis_distance - radius
    stand_off_m: float = 0.0          # median forward z of surface points
    lateral_offset_m: float = 0.0     # x of nearest axis point
    vertical_offset_m: float = 0.0    # y of nearest axis point
    yaw_error_deg: float = 0.0
    # Quality.
    inlier_count: int = 0
    inlier_fraction: float = 0.0
    residual_m: float = 0.0
    elongation: float = 0.0
    foreground_pixels: int = 0

    def to_pipe_pose_json(self, *, frame_id: str = "rgbd_camera",
                          stamp: float | None = None) -> dict[str, Any]:
        """Serialise to the exact contract the scan-control node consumes.

        Field names match ``acea_pipe_pose_pcl_node.cpp`` so this is a drop-in
        replacement publisher for ``/acea/pipe_junction/pipe_pose``.
        """
        def vec(a: np.ndarray) -> list[float] | None:
            arr = np.asarray(a, dtype=float)
            if not np.all(np.isfinite(arr)):
                return None
            return [round(float(v), 6) for v in arr]

        return {
            "valid": bool(self.valid),
            "reason": self.reason,
            "pose_source": self.pose_source,
            "axis_method": self.axis_method,
            "frame_id": frame_id,
            "camera_intrinsics_source": "camera_info",
            "pipe_axis_camera_xyz": vec(self.axis_camera_xyz),
            "cylinder_axis_point_camera_xyz_m": vec(self.axis_point_camera_xyz_m),
            "pipe_radius_m": _round(self.radius_m),
            "pipe_axis_distance_m": _round(self.axis_distance_m),
            "pipe_surface_distance_m": _round(self.surface_distance_m),
            "stand_off_m": _round(self.stand_off_m),
            "lateral_offset_m": _round(self.lateral_offset_m),
            "vertical_offset_m": _round(self.vertical_offset_m),
            "yaw_error_deg": _round(self.yaw_error_deg),
            "inlier_count": int(self.inlier_count),
            "inlier_fraction": _round(self.inlier_fraction),
            "residual_m": _round(self.residual_m),
            "elongation": _round(self.elongation),
            "stamp": None if stamp is None else float(stamp),
        }


def _round(value: float, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vec / norm


def _canonical_axis(direction: np.ndarray) -> np.ndarray:
    """Force axis_x > 0 so yaw=atan2(z,x) is well conditioned for this task."""
    direction = _normalize(direction)
    if direction[0] < 0.0:
        direction = -direction
    return direction


def _pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid, eigenvalues_desc, dominant_direction)."""
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 3:
        raise ValueError("Need at least three points for PCA")
    centroid = points.mean(axis=0)
    cov = np.cov((points - centroid).T)
    if not np.all(np.isfinite(cov)):
        raise ValueError("PCA covariance contains non-finite values")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return centroid, eigenvalues, _normalize(eigenvectors[:, 0])


def _plane_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``axis``."""
    axis = _normalize(axis)
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(seed - np.dot(seed, axis) * axis)
    v = np.cross(axis, u)
    return u, v


def _fit_circle_kasa(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) circle fit. Returns (center_a, center_b, radius).

    Closed-form linear least squares, so it is deterministic. Good enough as the
    inlier-selection / radius estimate for a well-sampled cylinder cross-section.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    amat = np.stack([2.0 * a, 2.0 * b, np.ones_like(a)], axis=1)
    rhs = a * a + b * b
    sol, *_ = np.linalg.lstsq(amat, rhs, rcond=None)
    ca, cb, d = float(sol[0]), float(sol[1]), float(sol[2])
    radius_sq = d + ca * ca + cb * cb
    radius = math.sqrt(radius_sq) if radius_sq > 0.0 else 0.0
    return ca, cb, radius


def _organized_points(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Back-project the whole depth image to an organized (H, W, 3) point grid."""
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur in pure NumPy (no SciPy runtime dependency).

    Shift-and-accumulate separable convolution with edge padding. Deterministic
    and fast enough for online use; used only to denoise depth before normals.
    """
    img = np.asarray(image, dtype=np.float64)
    if sigma <= 0.0:
        return img
    radius = max(1, int(round(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    h, w = img.shape
    padded = np.pad(img, radius, mode="edge")
    horizontal = np.zeros_like(img)
    for i, weight in enumerate(kernel):
        horizontal += weight * padded[radius:radius + h, i:i + w]
    padded2 = np.pad(horizontal, radius, mode="edge")
    out = np.zeros_like(img)
    for i, weight in enumerate(kernel):
        out += weight * padded2[i:i + h, radius:radius + w]
    return out


def _axis_circle_residual(axis: np.ndarray, points: np.ndarray,
                          radius_tolerance_m: float) -> float:
    """Robust circle-fit residual for a candidate axis (lower = better cylinder)."""
    u, v = _plane_basis(axis)
    a = points @ u
    b = points @ v
    ca, cb, radius = _fit_circle_kasa(a, b)
    if radius <= 0.0:
        return float("inf")
    plane_radius = np.sqrt((a - ca) ** 2 + (b - cb) ** 2)
    deviation = plane_radius - radius
    med = float(np.median(deviation))
    mad = float(np.median(np.abs(deviation - med)))
    tol = max(radius_tolerance_m, 3.0 * 1.4826 * mad)
    inliers = np.abs(deviation - med) <= tol
    if not inliers.any():
        return float("inf")
    return float(np.median(np.abs(deviation[inliers])))


def _estimate_normal_axis(depth: np.ndarray, k: np.ndarray, mask: np.ndarray,
                          params: dict[str, Any]) -> np.ndarray | None:
    """Cylinder axis from surface normals: robust to orientation / foreshortening
    where PCA-on-points collapses. The axis is perpendicular to every surface
    normal, i.e. the smallest eigenvector of sum(n n^T).

    Depth is Gaussian-smoothed first so finite-difference normals are not
    dominated by depth noise; normals across depth discontinuities are rejected.
    Returns None if too few reliable normals are available.
    """
    sigma = float(params["normal_smooth_sigma_px"])
    jump = float(params["normal_jump_threshold_m"])
    smooth = _gaussian_blur(depth, sigma)
    pts = _organized_points(smooth, k)
    normals = np.cross(pts[1:-1, 2:, :] - pts[1:-1, :-2, :],
                       pts[2:, 1:-1, :] - pts[:-2, 1:-1, :])
    interior = mask[1:-1, 1:-1]
    cont = ((np.abs(depth[1:-1, 2:] - depth[1:-1, :-2]) < jump)
            & (np.abs(depth[2:, 1:-1] - depth[:-2, 1:-1]) < jump))
    norm_len = np.linalg.norm(normals, axis=-1)
    good = interior & cont & (norm_len > 1e-9) & np.isfinite(norm_len)
    if int(good.sum()) < int(params["min_inliers"]):
        return None
    unit = normals[good] / norm_len[good][:, None]
    scatter = unit.T @ unit
    if not np.all(np.isfinite(scatter)):
        return None
    _eigvals, eigvecs = np.linalg.eigh(scatter)
    return _canonical_axis(eigvecs[:, 0])  # smallest eigenvector = axis direction


def foreground_mask(depth: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Deterministic near-foreground (pipe) mask via a 2-means depth split.

    Identical idea to ``_depth_threshold_kmeans`` in the legacy node: split the
    valid depth values into a near and a far cluster and keep the near one.
    """
    min_depth = float(params["min_depth_m"])
    max_depth = float(params["max_depth_m"])
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    stride = max(1, int(params["sample_stride"]))
    sample = depth[::stride, ::stride]
    sample_valid = valid[::stride, ::stride]
    values = sample[sample_valid].astype(np.float64)
    if values.size < 32:
        return np.zeros_like(valid)

    centers = np.percentile(values, [10.0, 90.0]).astype(np.float64)
    for _ in range(32):
        threshold = 0.5 * (centers[0] + centers[1])
        low = values[values <= threshold]
        high = values[values > threshold]
        if low.size == 0 or high.size == 0:
            break
        new_centers = np.array([low.mean(), high.mean()])
        if np.linalg.norm(new_centers - centers) < 1e-9:
            centers = new_centers
            break
        centers = new_centers
    centers = np.sort(centers)
    threshold = float(0.5 * (centers[0] + centers[1]))
    return valid & (depth <= threshold)


def backproject(depth: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                k: np.ndarray) -> np.ndarray:
    """Pixel (xs, ys) + depth -> camera-frame points (N, 3) in the optical frame."""
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if not np.isfinite(fx) or not np.isfinite(fy) or abs(fx) < 1e-9 or abs(fy) < 1e-9:
        raise ValueError(f"Invalid camera intrinsics: fx={fx}, fy={fy}")
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _invalid(reason: str) -> PipePose:
    return PipePose(valid=False, reason=reason)


def fit_pipe_pose(depth: np.ndarray, k: np.ndarray,
                  mask: np.ndarray | None = None,
                  params: dict[str, Any] | None = None) -> PipePose:
    """Robustly estimate the pipe pose from one depth frame.

    Parameters
    ----------
    depth : (H, W) float array of metric depth (NaN / <=0 = invalid).
    k     : (3, 3) camera intrinsics.
    mask  : optional (H, W) bool pipe mask. If None a near-foreground mask is
            computed deterministically.
    """
    p = dict(DEFAULT_PIPE_POSE_PARAMS)
    if params:
        p.update(params)

    depth = np.asarray(depth, dtype=np.float64)
    if mask is None:
        mask = foreground_mask(depth, p)
    mask = np.asarray(mask, dtype=bool)

    ys, xs = np.nonzero(mask)
    foreground_pixels = int(xs.size)
    if foreground_pixels < int(p["min_pipe_pixels"]):
        out = _invalid(f"too_few_pipe_pixels:{foreground_pixels}")
        out.foreground_pixels = foreground_pixels
        return out

    # Deterministic subsample if huge (linspace indices -> reproducible).
    max_pts = int(p["max_fit_points"])
    if foreground_pixels > max_pts:
        idx = np.linspace(0, foreground_pixels - 1, max_pts).astype(np.int64)
        ys, xs = ys[idx], xs[idx]

    points = backproject(depth, ys, xs, k)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < int(p["min_inliers"]):
        out = _invalid(f"too_few_finite_points:{points.shape[0]}")
        out.foreground_pixels = foreground_pixels
        return out

    n_total = points.shape[0]
    radius_tol = float(p["radius_tolerance_m"])

    # --- Robust axis estimation: keep the better of PCA-on-points and a
    # surface-normal axis, judged by cylinder-fit residual. PCA is noise-robust
    # but collapses on close/foreshortened views; the normal axis is robust to
    # orientation/foreshortening. Selecting by residual is strong across large
    # initial misalignment, close range, and depth noise simultaneously. ---
    try:
        _centroid, eigvals, axis_pca = _pca(points)
    except ValueError as exc:
        out = _invalid(f"pca_failed:{exc}")
        out.foreground_pixels = foreground_pixels
        return out
    axis_pca = _canonical_axis(axis_pca)

    candidates: list[tuple[str, np.ndarray]] = [("pca", axis_pca)]
    if bool(p.get("use_normal_axis", 1)):
        axis_override = p.get("normal_axis_override")
        if axis_override is None:
            axis_nrm = _estimate_normal_axis(depth, k, mask, p)
        else:
            try:
                axis_nrm = _canonical_axis(
                    np.asarray(axis_override, dtype=np.float64).reshape(3)
                )
            except (TypeError, ValueError):
                axis_nrm = None
        if axis_nrm is not None:
            candidates.append(("normal_svd", axis_nrm))

    axis = axis_pca
    axis_method = "pca"
    best_res = float("inf")
    for name, cand in candidates:
        res = _axis_circle_residual(cand, points, radius_tol)
        if res < best_res:
            best_res, axis, axis_method = res, cand, name

    # --- Refine radius / centre / inliers with the chosen axis held fixed.
    # Refitting the axis by PCA here would re-introduce the close-range collapse,
    # so we deliberately do not. ---
    inliers = np.ones(n_total, dtype=bool)
    ca = cb = radius = 0.0
    residual = 0.0
    for _ in range(max(1, int(p["consensus_iterations"]))):
        u, v = _plane_basis(axis)
        a = points @ u
        b = points @ v
        ca, cb, radius = _fit_circle_kasa(a[inliers], b[inliers])
        if radius <= 0.0:
            break
        plane_radius = np.sqrt((a - ca) ** 2 + (b - cb) ** 2)
        deviation = plane_radius - radius
        med = float(np.median(deviation[inliers]))
        mad = float(np.median(np.abs(deviation[inliers] - med)))
        tol = max(radius_tol, 3.0 * 1.4826 * mad)
        new_inliers = np.abs(deviation - med) <= tol
        if int(new_inliers.sum()) < int(p["min_inliers"]):
            break
        inliers = new_inliers
        residual = float(np.median(np.abs(deviation[inliers])))

    u, v = _plane_basis(axis)
    a = points @ u
    b = points @ v
    ca, cb, radius = _fit_circle_kasa(a[inliers], b[inliers])
    plane_radius = np.sqrt((a - ca) ** 2 + (b - cb) ** 2)
    residual = float(np.median(np.abs(plane_radius[inliers] - radius)))

    inlier_count = int(inliers.sum())
    inlier_fraction = float(inlier_count) / float(n_total)
    elongation = float(eigvals[0] / eigvals[1]) if eigvals[1] > 1e-12 else float("inf")

    # Nearest axis point to the camera origin is exactly (ca*u + cb*v): the axial
    # component drops out because u, v are perpendicular to the axis. Hence
    # axis_distance is the true perpendicular camera-origin -> axis distance.
    axis_point = ca * u + cb * v
    axis_distance = float(np.linalg.norm(axis_point))
    surface_distance = max(0.0, axis_distance - radius)
    stand_off = float(np.median(points[inliers, 2]))
    yaw_error_deg = math.degrees(math.atan2(float(axis[2]), float(axis[0])))

    pose = PipePose(
        valid=True,
        reason="ok",
        axis_method=axis_method,
        axis_camera_xyz=axis,
        axis_point_camera_xyz_m=axis_point,
        radius_m=radius,
        axis_distance_m=axis_distance,
        surface_distance_m=surface_distance,
        stand_off_m=stand_off,
        lateral_offset_m=float(axis_point[0]),
        vertical_offset_m=float(axis_point[1]),
        yaw_error_deg=yaw_error_deg,
        inlier_count=inlier_count,
        inlier_fraction=inlier_fraction,
        residual_m=residual,
        elongation=elongation,
        foreground_pixels=foreground_pixels,
    )

    # Validity gate: reject implausible fits rather than feed them to control.
    # NOTE: no elongation gate -- a foreshortened large-yaw / close view is
    # exactly when we still need a usable pose, and the normal-SVD axis handles it.
    reasons: list[str] = []
    if inlier_count < int(p["min_inliers"]):
        reasons.append(f"low_inliers:{inlier_count}")
    if inlier_fraction < float(p["min_inlier_fraction"]):
        reasons.append(f"low_inlier_fraction:{inlier_fraction:.3f}")
    if not (float(p["radius_min_m"]) <= radius <= float(p["radius_max_m"])):
        reasons.append(f"radius_out_of_range:{radius:.3f}")
    if residual > float(p["max_residual_m"]):
        reasons.append(f"residual_too_high:{residual:.4f}")
    if reasons:
        pose.valid = False
        pose.reason = ";".join(reasons)
    return pose
