"""VC1 pipe-junction detector (Claude version 1).

Single self-contained module: RGB + aligned metric depth + intrinsics + known
pipe radius -> circumferential junction pose in the camera frame.

Contract (same as the ACEA perception contract):
  * RGB finds the junction line; depth measures the pipe (known radius) and
    lifts the RGB line to 3-D.  Depth is never required on the seam itself.
  * The only workpiece prior is the pipe radius.  No ML, colour class, fixture
    pose, ground truth or scene-specific region is used.
  * Every published pose is measured in the current frame.  Temporal state is
    used only to (a) warm-start the pipe fit and (b) propose the tracked line
    as a candidate / break near-ties between candidates.  A temporal
    prediction is never published, and no feature is remembered as vetoed.

Stages:
  1. depth -> block-averaged points + validity-aware normals
  2. known-radius cylinder: axis votes p - R n, deterministic RANSAC line,
     robust surface refinement (warm start from the previous frame if any)
  3. unwrap the visible cylinder surface into (azimuth, axial) image
  4. dark-valley response; best near-vertical path per candidate station
  5. physical verification (continuity, diffuse-row support, pipe on both
     sides / image border, pipe-end and specular-interruption vetoes)
  6. sub-pixel line + ring fit to the observed RGB line
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import cv2
import numpy as np

DEFAULTS: dict[str, Any] = {
    # ---- depth / normals
    "min_depth_m": 0.12,
    "max_depth_m": 3.0,
    "grid_target_width": 160,          # block-averaged depth grid width
    "block_max_std_m": 0.008,          # reject blocks straddling a depth edge
    "normal_step": 1,                  # +-cells for central differences
    "normal_max_jump_m": 0.03,
    "normal_mode": "pca",              # plane fit on valid cells only ("diff" = central differences, older)
    "pca_window_radius": 2,
    "pca_min_points": 6,
    "pca_max_plane_residual_m": 0.003,
    # ---- cylinder
    "ransac_hypotheses": 256,
    "ransac_hypotheses_warm": 128,
    "max_pipe_models": 2,
    "global_every_n": 5,               # full global search every 5 frames while tracking (accuracy-neutral on all 3 bags)
    "warm_keep_support": 0.9,
    "candidate_min_rel_count": 0.25,
    "second_round": True,
    "ransac_score_points": 1024,
    "ransac_finalists": 8,
    "ransac_finalist_min_rel_score": 0.25,  # distinct finalists with fewer votes than this x best are not refit
    "ransac_min_pair_span_m": 0.04,
    "ransac_max_normal_coherence": 0.995,   # |mean normal| of a hypothesis' votes: 1 = plane, ~0.97 = 45 deg arc
    "ransac_seed": 20260927,
    "vote_tolerance_m": 0.012,
    "normal_axis_max_dot": 0.35,       # normals must be ~perpendicular to the axis
    "refine_iterations": 8,
    "refine_huber_m": 0.004,
    "surface_inlier_m": 0.008,
    "tight_inlier_m": 0.003,
    "precise_inlier_m": 0.001,
    "min_azimuth_precise_deg": 25.0,
    "min_azimuth_warm_deg": 25.0,       # narrower warm arcs gave imprecise axes (real_1 550-560)
    "warm_small_move_deg": 2.0,
    "warm_small_move_m": 0.005,
    "precise_residual_m": 0.0006,
    "min_model_consistency": 0.75,
    "min_model_consistency_precise": 0.60,   # ... down to this when the fit is precise:
    "min_precise_fraction": 0.55,            #     share of inliers within precise_inlier_m
    "min_model_consistency_occluded": 0.50,  # ... and down to this for a precise model
    "occluded_min_azimuth_deg": 90.0,        #     seen over a wide arc (occluder in front)
    "radius_ratio_range": (0.8, 1.25),   # free-fit radius / configured radius (RealSense bias ~+2..7%)
    "radius_check_min_azimuth_deg": 70.0,
    "min_surface_inliers": 150,
    "min_azimuth_coverage_deg": 45.0,
    "min_axial_extent_m": 0.10,
    "max_residual_median_m": 0.0045,
    "warm_start_max_axis_change_deg": 12.0,
    "warm_start_max_offset_m": 0.06,
    # ---- unwrap
    "unwrap_max_rows": 240,
    "unwrap_row_px": 2.0,               # target pixel spacing between rows at the arc centre
    "unwrap_col_px": 1.0,               # target pixel spacing between axial columns (finest)
    "unwrap_max_cols": 1800,
    "unwrap_col_min_m": 0.0006,
    "unwrap_limb_margin": 0.96,         # use |theta| <= margin * theta_limit
    # ---- intensity image the valley is measured on
    # "pipe": per-channel weights from the pipe's own measured colour (samples the
    # depth confirms on the cylinder): w_c ~ albedo_c^power * (1 - clipped_c).
    # A shading/cavity darkening scales all channels with the albedo, so this is
    # the matched intensity for ANY pipe colour, invariant to channel order
    # (orange, violet, grey metal give the same image).  "luma" = Rec.601.
    "intensity_mode": "pipe",
    "intensity_albedo_power": 1.0,
    "intensity_clip_level": 250,
    "intensity_clip_power": 1.0,
    # ---- valley response
    "valley_halfwidths_px": (2, 3, 5, 8),
    "valley_flank_px": 3,
    "row_smooth": 3,
    # ---- candidate search / path
    "candidates": 6,
    "candidate_separation_px": 14,
    "path_halfband_px": 6,
    "centroid_halfwin_px": 7,
    "path_max_step": 1,
    # ---- verification
    "valley_min_rel": 0.06,             # (flank - centre)/flank
    "valley_min_abs": 6.0,              # grey levels
    "valley_noise_k": 3.0,
    "min_line_pixels": 28,              # distinct image pixels on the proven run
    "min_run_fraction": 0.55,           # proven rows / rows of the run
    "max_hole_px": 4.0,
    "weak_evidence_factor": 0.5,
    "min_diffuse_fraction": 0.45,       # diffuse (non-specular) rows with valley
    "min_diffuse_rows": 6,
    # grey pipe / monochrome image: when the body has (almost) no chroma the
    # "highlight is whiter" test cannot work; a highlight row is then one much
    # brighter than the ring's smooth diffuse shading at the same azimuth
    # an object in front of the pipe (torch, cable): no junction evidence on it
    "occluder_mask": False,
    "occluder_front_m": 0.030,          # depth this much in front of the surface (2x the flank tolerance)
    # pipe end in view (reported for the mission, never a junction): beyond the measured
    # axial range the model surface still projects into the image, but the depth shows
    # the scene BEHIND it
    "pipe_end_gap_m": 0.015,            # skipped right after the last measured station (rim)
    "pipe_end_probe_m": 0.05,           # length examined beyond it
    "pipe_end_min_inside_frac": 0.5,    # of the probe samples inside the image
    "pipe_end_min_valid_frac": 0.5,     # of those, with depth
    "pipe_end_min_behind_frac": 0.8,    # of those with depth, behind the surface
    "achromatic_specular": True,
    "achromatic_max_chroma": 0.08,      # median body chroma below this -> achromatic
    "achromatic_partial_coverage": 0.7, # only a line over part of the ring can be a reflection gap
    "achromatic_edge_rows": 4,          # end rows of the line compared with ...
    "achromatic_side_rows": 8,          # ... the ring rows just outside it (after 2 rows)
    "achromatic_step_ratio": 1.3,       # end rows > ratio * outside + abs: a band with sharp edges
    "achromatic_step_abs": 15.0,
    "achromatic_cavity_ratio": 0.85,    # floor below ratio * outside = a cavity (gap), not a reflection gap
    "specular_rel": 1.25,               # flank > rel * arc diffuse level -> specular row
    "flank_offset_m": 0.018,
    "flank_depth_tol_m": 0.015,
    "min_flank_rows": 8,
    "border_min_px": 10.0,
    "flank_outside_frac": 0.3,
    "end_walk_m": 0.030,
    "end_resume_m": 0.010,
    "end_min_row_frac": 0.5,
    "pipe_end_max_match": 0.35,         # inside-image flank matched below this -> pipe end
    "step_inner_m": 0.006,              # radial-step side windows: 6..30 mm from the line
    "step_outer_m": 0.030,
    "step_min_samples": 20,
    "ambiguity_ratio": 1.35,
    # ---- ring fit
    "ring_fit_iterations": 6,
    "ring_tilt_prior_deg": 0.3,
    "ring_tilt_ref_azimuth_deg": 90.0,   # prior grows as ref/azimuth on narrower arcs ...
    "ring_tilt_max_scale": 5.0,          # ... up to 1.5 deg
    "ring_max_tilt_deg": 5.0,
    # ---- tracking
    "track_max_jump_px": 60.0,
    "track_tie_px": 20.0,
    "track_tie_ratio": 1.5,
    "track_weak_factor": 0.75,          # evidence relaxation near a live track
    "track_max_misses": 2,              # the track survives this many rejected frames
    "reset_gap_s": 2.0,                 # stamp gap (or time going back) that resets all search state
}


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors (np.cross has a large per-call overhead)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _perp_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(_cross3(d, ref))
    return e1, _cross3(d, e1)


@dataclass
class Cylinder:
    valid: bool
    reason: str
    axis: np.ndarray | None = None      # unit direction (camera frame)
    point: np.ndarray | None = None     # a point on the axis, closest to the camera origin
    radius: float = 0.0
    inliers: int = 0
    residual_median_m: float = float("nan")
    azimuth_deg: float = 0.0
    axial_extent_m: float = 0.0
    axial_range: tuple[float, float] = (0.0, 0.0)
    warm: bool = False
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    accepted: bool
    reason: str
    cylinder: Cylinder | None = None
    center: np.ndarray | None = None       # ring centre on the axis (camera frame)
    axis: np.ndarray | None = None         # ring normal (pipe axis at the junction)
    surface: np.ndarray | None = None      # front-visible surface point of the ring
    station_m: float | None = None
    line_uv: np.ndarray | None = None      # observed RGB line samples (sub-pixel)
    stats: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# 1. depth -> points / normals
# ----------------------------------------------------------------------------

def _smallest_eig_sym3(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Smallest eigenvalue/eigenvector of many symmetric 3x3 matrices (closed form,
    trigonometric method; eigenvector from the best cross product of (C - l I) rows)."""
    a, b, c = C[:, 0, 0], C[:, 1, 1], C[:, 2, 2]
    d, e, f = C[:, 0, 1], C[:, 0, 2], C[:, 1, 2]
    p1 = d * d + e * e + f * f
    q = (a + b + c) / 3.0
    p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2.0 * p1
    pp = np.sqrt(np.maximum(p2 / 6.0, 1e-30))
    ba, bb, bc = (a - q) / pp, (b - q) / pp, (c - q) / pp
    bd, be, bf = d / pp, e / pp, f / pp
    detB = ba * (bb * bc - bf * bf) - bd * (bd * bc - bf * be) + be * (bd * bf - bb * be)
    r = np.clip(detB / 2.0, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    lam = q + 2.0 * pp * np.cos(phi + 2.0 * np.pi / 3.0)          # smallest
    r0 = np.stack([a - lam, d, e], -1)
    r1 = np.stack([d, b - lam, f], -1)
    r2 = np.stack([e, f, c - lam], -1)
    c01, c02, c12 = np.cross(r0, r1), np.cross(r0, r2), np.cross(r1, r2)
    n01, n02, n12 = (np.einsum("ij,ij->i", v, v) for v in (c01, c02, c12))
    vec = np.where((n01 >= n02)[:, None] & (n01 >= n12)[:, None], c01, np.where((n02 >= n12)[:, None], c02, c12))
    nrm = np.sqrt(np.maximum(np.maximum(np.maximum(n01, n02), n12), 1e-30))
    return lam, vec / nrm[:, None]


def _pca_normals(P: np.ndarray, p: dict) -> np.ndarray:
    """Normals from a local plane fit over the VALID cells of a (2r+1)^2 window.

    Holes never contaminate a normal (no filling, no blur over invalid cells);
    windows straddling a depth jump are rejected by their out-of-plane residual."""
    r = int(p["pca_window_radius"])
    ks = (2 * r + 1, 2 * r + 1)
    V = np.isfinite(P[..., 2])
    X = np.where(V[..., None], P, 0.0)
    box = lambda a: cv2.boxFilter(a, -1, ks, normalize=False, borderType=cv2.BORDER_CONSTANT)
    n = box(V.astype(np.float64))
    sx, sy, sz = box(X[..., 0]), box(X[..., 1]), box(X[..., 2])
    sxx, syy, szz = box(X[..., 0] ** 2), box(X[..., 1] ** 2), box(X[..., 2] ** 2)
    sxy, sxz, syz = box(X[..., 0] * X[..., 1]), box(X[..., 0] * X[..., 2]), box(X[..., 1] * X[..., 2])
    ok = V & (n >= p["pca_min_points"])
    out = np.full_like(P, np.nan)
    if not ok.any():
        return out
    idx = np.nonzero(ok)
    nn = n[idx]
    mx, my, mz = sx[idx] / nn, sy[idx] / nn, sz[idx] / nn
    C = np.empty((len(nn), 3, 3))
    C[:, 0, 0] = sxx[idx] / nn - mx * mx
    C[:, 1, 1] = syy[idx] / nn - my * my
    C[:, 2, 2] = szz[idx] / nn - mz * mz
    C[:, 0, 1] = C[:, 1, 0] = sxy[idx] / nn - mx * my
    C[:, 0, 2] = C[:, 2, 0] = sxz[idx] / nn - mx * mz
    C[:, 1, 2] = C[:, 2, 1] = syz[idx] / nn - my * mz
    lam_min, normal = _smallest_eig_sym3(C)
    planar = np.sqrt(np.maximum(lam_min, 0)) < p["pca_max_plane_residual_m"]
    centre = P[idx]
    flip = np.einsum("ij,ij->i", normal, centre) > 0
    normal[flip] *= -1.0
    normal[~planar] = np.nan
    out[idx] = normal
    return out


def depth_grid(depth_m: np.ndarray, k: np.ndarray, p: dict) -> dict:
    h, w = depth_m.shape
    g = max(1, int(round(w / p["grid_target_width"])))
    hh, ww = h // g, w // g
    d = depth_m[: hh * g, : ww * g].astype(np.float32, copy=False)
    valid = ((d > p["min_depth_m"]) & (d < p["max_depth_m"])).astype(np.float32)
    dz = np.where(valid > 0, d, np.float32(0.0))              # NaN/inf no-returns must not poison a block
    cnt = cv2.resize(valid, (ww, hh), interpolation=cv2.INTER_AREA)
    s1 = cv2.resize(dz, (ww, hh), interpolation=cv2.INTER_AREA)
    s2 = cv2.resize(dz * dz, (ww, hh), interpolation=cv2.INTER_AREA)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = s2 / cnt - mean * mean
    ok = (cnt >= 0.5) & (np.sqrt(np.maximum(var, 0)) < p["block_max_std_m"] + 0.004 * mean)
    z = np.where(ok, mean, np.nan).astype(np.float64)
    vv, uu = np.mgrid[0:hh, 0:ww]
    uc = uu * g + (g - 1) / 2.0
    vc = vv * g + (g - 1) / 2.0
    x = (uc - k[0, 2]) * z / k[0, 0]
    y = (vc - k[1, 2]) * z / k[1, 1]
    P = np.stack([x, y, z], axis=-1)
    if p.get("normal_mode", "diff") == "pca":
        n = _pca_normals(P, p)
        return {"P": P, "N": n, "g": g, "uc": uc, "vc": vc, "valid": np.isfinite(z), "depth": depth_m, "k": k}
    # normals from central differences (+-step cells)
    s = int(p["normal_step"])
    n = np.full_like(P, np.nan)
    a = P[s:-s, 2 * s:] - P[s:-s, : -2 * s]
    b = P[2 * s:, s:-s] - P[: -2 * s, s:-s]
    c = np.cross(a, b)
    jump = p["normal_max_jump_m"]
    good = (np.abs(a[..., 2]) < jump) & (np.abs(b[..., 2]) < jump)
    nn = np.linalg.norm(c, axis=-1)
    good &= nn > 1e-12
    c = c / np.maximum(nn[..., None], 1e-12)
    centre = P[s:-s, s:-s]
    flip = np.sum(c * centre, axis=-1) > 0          # face the camera
    c[flip] *= -1.0
    c[~good] = np.nan
    n[s:-s, s:-s] = c
    return {"P": P, "N": n, "g": g, "uc": uc, "vc": vc, "valid": np.isfinite(z), "depth": depth_m, "k": k}


# ----------------------------------------------------------------------------
# 2. known-radius cylinder
# ----------------------------------------------------------------------------

def _axis_residuals(P: np.ndarray, d: np.ndarray, c: np.ndarray, R: float) -> tuple[np.ndarray, np.ndarray]:
    q = P - c
    t = q @ d
    rad = q - t[:, None] * d
    return np.linalg.norm(rad, axis=1) - R, t


def _residuals_jacobian(P: np.ndarray, d: np.ndarray, c: np.ndarray, R: float,
                        e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radial residuals r = |rho| - R and their analytic Jacobian w.r.t. the axis
    direction (d -> d + a1 e1 + a2 e2) and point (c -> c + b1 e1 + b2 e2):
    dr/db = -u.e,  dr/da = -t u.e   (u = rho/|rho|, t = (P - c).d, e _|_ d)."""
    q = P - c
    t = q @ d
    rho = q - t[:, None] * d
    dist = np.sqrt(np.einsum("ij,ij->i", rho, rho))
    u = rho / np.maximum(dist, 1e-12)[:, None]
    u1, u2 = u @ e1, u @ e2
    return dist - R, np.column_stack([-t * u1, -t * u2, -u1, -u2])


def _refine_cylinder(P: np.ndarray, d: np.ndarray, c: np.ndarray, R: float, p: dict,
                     iterations: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Robust Gauss-Newton on 4-DOF line with fixed radius (Huber IRLS)."""
    huber = p["refine_huber_m"]
    for _ in range(iterations or p["refine_iterations"]):
        e1, e2 = _perp_basis(d)
        r, J = _residuals_jacobian(P, d, c, R, e1, e2)
        a = np.abs(r)
        w = np.where(a <= huber, 1.0, huber / np.maximum(a, 1e-12))
        JW = J * w[:, None]
        H = JW.T @ J + 1e-9 * np.eye(4)
        gvec = JW.T @ r
        try:
            delta = -np.linalg.solve(H, gvec)
        except np.linalg.LinAlgError:
            break
        d = _unit(d + delta[0] * e1 + delta[1] * e2)
        c = c + delta[2] * e1 + delta[3] * e2
        if np.linalg.norm(delta) < 1e-7:
            break
    # re-anchor the axis point to the one closest to the camera origin
    c = c - (c @ d) * d
    return d, c


def free_radius(P: np.ndarray, N: np.ndarray, d: np.ndarray, c: np.ndarray, R: float,
                max_points: int = 800) -> float:
    """Radius re-estimated with the radius FREE (5-DOF Gauss-Newton) on the pipe
    points near the configured model.  Used to refuse a wrong configured
    diameter: a too-small model sits inside the real pipe and the free radius
    drifts toward the true one."""
    sel, _, _ = _select_surface(P, N, d, c, R, 0.025, 0.5)
    idx = np.flatnonzero(sel)
    if len(idx) < 50:
        return float("nan")
    if len(idx) > max_points:
        idx = idx[:: int(math.ceil(len(idx) / max_points))]
    Q = P[idx]
    r = float(R)
    for _ in range(8):
        e1, e2 = _perp_basis(d)
        res, J4 = _residuals_jacobian(Q, d, c, r, e1, e2)
        J = np.column_stack([J4, -np.ones(len(Q))])
        a = np.abs(res)
        w = np.where(a <= 0.004, 1.0, 0.004 / np.maximum(a, 1e-12))
        JW = J * w[:, None]
        try:
            delta = -np.linalg.solve(JW.T @ J + 1e-9 * np.eye(5), JW.T @ res)
        except np.linalg.LinAlgError:
            break
        d = _unit(d + delta[0] * e1 + delta[1] * e2)
        c = c + delta[2] * e1 + delta[3] * e2
        r = r + delta[4]
        if not (0.2 * R < r < 5 * R):
            return float("nan")
        if np.linalg.norm(delta) < 1e-7:
            break
    return float(r)


def _select_surface(P: np.ndarray, N: np.ndarray, d: np.ndarray, c: np.ndarray, R: float,
                    tol_r: float, min_align: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points on the camera-facing side of the cylinder whose normal is radial."""
    q = P - c
    t = q @ d
    rad = q - t[:, None] * d
    dist = np.linalg.norm(rad, axis=1)
    rho = rad / np.maximum(dist[:, None], 1e-12)
    r = dist - R
    align = np.einsum("ij,ij->i", N, rho)
    facing = np.einsum("ij,ij->i", rho, -P) > 0.0
    return (np.abs(r) < tol_r) & (align > min_align) & facing, r, t


def _largest_component(mask_flat: np.ndarray, shape: tuple[int, int], keep_idx: np.ndarray) -> np.ndarray:
    """Restrict a selection (indices into the valid grid) to its largest 8-connected blob."""
    img = np.zeros(shape[0] * shape[1], np.uint8)
    img[keep_idx[mask_flat]] = 1
    img = img.reshape(shape)
    n, lab = cv2.connectedComponents(img, connectivity=8)
    if n <= 2:
        return mask_flat
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    best = int(np.argmax(counts))
    lab_flat = lab.ravel()[keep_idx]
    return mask_flat & (lab_flat == best)


def _small_move(d0, c0, d, c, p) -> bool:
    """The refit stayed on the previous (tracked) pipe."""
    ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(d0, d))))))
    q = c - c0
    off = float(np.linalg.norm(q - np.dot(q, d0) * d0))
    return ang < p["warm_small_move_deg"] and off < p["warm_small_move_m"]


def model_consistency(cyl: Cylinder, depth_m: np.ndarray, k: np.ndarray, p: dict) -> dict:
    """Does the measured depth agree with the model wherever its visible surface
    projects inside the image (over the supported axial range)?  A real pipe
    has little in front of it; a cylinder fitted into cloth folds or background
    structures has as much scene in front of its surface as on it."""
    q = dict(p)
    q.update(unwrap_max_rows=48, unwrap_col_px=6.0, unwrap_col_min_m=0.004)
    ug = unwrap_grid(cyl, k, depth_m.shape, q)
    if ug is None:
        return {"consistency": 0.0}
    lo, hi = cyl.axial_range
    cols = (ug["stations"] >= lo) & (ug["stations"] <= hi)
    meas = cv2.remap(depth_m, ug["u"], ug["v"], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ins = ug["inside"][:, cols]
    me = meas[:, cols]
    z = ug["z"][:, cols]
    valid = ins & (me > 0)
    tol = p["flank_depth_tol_m"]
    match = int(np.count_nonzero(valid & (np.abs(me - z) < tol)))
    behind = int(np.count_nonzero(valid & (me > z + tol)))
    front = int(np.count_nonzero(valid & (me < z - tol)))
    return {"consistency": match / max(1, match + behind + front), "match": match, "behind": behind, "front": front}


def fit_cylinder(grid: dict, R: float, p: dict, prior: Cylinder | None = None) -> Cylinder:
    return fit_cylinders(grid, R, p, prior)[0]


def fit_cylinders(grid: dict, R: float, p: dict, prior: Cylinder | None = None,
                  global_search: bool = True) -> list[Cylinder]:
    """Distinct known-radius pipe models, best first (valid, tight support)."""
    Pg = grid["P"].reshape(-1, 3)
    Ng = grid["N"].reshape(-1, 3)
    ok = np.isfinite(Pg[:, 2]) & np.isfinite(Ng[:, 0])
    keep_idx = np.flatnonzero(ok)
    P, N = Pg[ok], Ng[ok]
    shape = grid["P"].shape[:2]
    if len(P) < p["min_surface_inliers"]:
        return [Cylinder(False, "too_few_depth_points")]

    def polish(d, c):
        for tol, align, iters in ((0.020, 0.6, 4), (0.012, 0.75, 4), (0.008, 0.8, 3)):
            sel, _, _ = _select_surface(P, N, d, c, R, tol, align)
            sel = _largest_component(sel, shape, keep_idx)
            if sel.sum() < 30:
                return d, c, sel
            sub = np.flatnonzero(sel)
            if len(sub) > 3000:
                sub = sub[:: int(math.ceil(len(sub) / 3000))]
            d, c = _refine_cylinder(P[sub], d, c, R, p, iterations=iters)
        sel, _, _ = _select_surface(P, N, d, c, R, p["surface_inlier_m"], 0.8)
        sel = _largest_component(sel, shape, keep_idx)
        return d, c, sel

    def finish(d, c, warm):
        d0, c0 = d, c
        d, c, sel = polish(d, c)
        r, t = _axis_residuals(P, d, c, R)
        cyl = Cylinder(True, "ok", d, c, R, int(sel.sum()), warm=warm, stats={"points": int(len(P))})
        if sel.sum() >= 5:
            q = P[sel] - c
            rad = q - (q @ d)[:, None] * d
            e1, e2 = _perp_basis(d)
            ang = np.sort(np.degrees(np.arctan2(rad @ e2, rad @ e1)))
            gaps = np.diff(np.r_[ang, ang[0] + 360.0])
            cyl.azimuth_deg = 360.0 - float(gaps.max())
            lo, hi = np.percentile(t[sel], [1, 99])
            cyl.axial_range = (float(lo), float(hi))
            cyl.axial_extent_m = float(hi - lo)
            cyl.residual_median_m = float(np.median(np.abs(r[sel])))
            cyl.stats["tight"] = int(np.count_nonzero(np.abs(r[sel]) < p["tight_inlier_m"]))
            cyl.stats["precise"] = int(np.count_nonzero(np.abs(r[sel]) < p["precise_inlier_m"]))
        cyl.stats["inlier_index"] = keep_idx[sel]
        reason = None
        if cyl.inliers < p["min_surface_inliers"]:
            reason = "few_inliers"
        elif cyl.azimuth_deg < p["min_azimuth_coverage_deg"] and not (
                cyl.azimuth_deg >= (p["min_azimuth_warm_deg"] if warm and _small_move(d0, c0, d, c, p)
                                    else p["min_azimuth_precise_deg"])
                and cyl.residual_median_m <= p["precise_residual_m"]
                and cyl.stats.get("tight", 0) >= 0.95 * max(cyl.inliers, 1)
                and cyl.axial_extent_m >= 0.15):
            reason = "low_azimuth"
        elif cyl.axial_extent_m < p["min_axial_extent_m"]:
            reason = "short_axial_extent"
        elif not (cyl.residual_median_m <= p["max_residual_median_m"]):
            reason = "high_residual"
        elif np.linalg.norm(c) <= R * 1.05:
            reason = "camera_inside"
        if reason is None and "depth" in grid:
            mc = model_consistency(cyl, grid["depth"], grid["k"], p)
            cyl.stats.update(mc)
            cons = mc["consistency"]
            # A PRECISE model may have some scene in front of it (an occluder in
            # front of a real pipe).  Precision is the share of inliers within
            # 1 mm: unlike the median residual it is insensitive to a few % error
            # of the configured radius.  Heavier occlusion is accepted only on a
            # wide arc: a narrow strip with degraded depth (camera below the
            # sensor's minimum range) gives precise but biased models.
            precise_frac = cyl.stats.get("precise", 0) / max(cyl.inliers, 1)
            cyl.stats["precise_fraction"] = precise_frac
            precise = precise_frac >= p["min_precise_fraction"]
            if not (cons >= p["min_model_consistency"]
                    or (precise and cons >= p["min_model_consistency_precise"])
                    or (precise and cons >= p["min_model_consistency_occluded"]
                        and cyl.azimuth_deg >= p["occluded_min_azimuth_deg"])):
                reason = "model_contradicted_by_scene"
        if reason:
            cyl.valid, cyl.reason = False, reason
        return cyl

    # warm start from the previous pipe; accepted only if it keeps most of the
    # previous support, otherwise a fresh global fit decides.
    warm_cyl = None
    n_hyp = int(p["ransac_hypotheses"])
    if prior is not None and prior.valid:
        cyl = finish(prior.axis, prior.point, True)
        change = math.degrees(math.acos(min(1.0, abs(float(cyl.axis @ prior.axis)))))
        if cyl.valid and change < p["warm_start_max_axis_change_deg"]:
            warm_cyl = cyl
            # A stable warm model (support kept) may skip the global challenge
            # on this frame; the tracker forces it periodically and on doubt.
            if not global_search and cyl.inliers >= p["warm_keep_support"] * prior.inliers:
                return [cyl]
            # otherwise the warm model is challenged by a reduced global search
            n_hyp = int(p["ransac_hypotheses_warm"])

    # votes: axis points
    V = P - R * N
    rng = np.random.default_rng(p["ransac_seed"])
    m = len(V)
    score_idx = np.arange(m) if m <= p["ransac_score_points"] else rng.choice(m, p["ransac_score_points"], replace=False)
    Vs, Ns = V[score_idx], N[score_idx]
    tol = p["vote_tolerance_m"]
    maxdot = p["normal_axis_max_dot"]
    ia = rng.integers(0, m, size=n_hyp * 4)
    ib = rng.integers(0, m, size=n_hyp * 4)
    dvec = V[ib] - V[ia]
    span = np.linalg.norm(dvec, axis=1)
    okp = span >= p["ransac_min_pair_span_m"]
    dirs = dvec / np.maximum(span[:, None], 1e-12)
    okp &= (np.abs(np.einsum("ij,ij->i", N[ia], dirs)) < maxdot) & (np.abs(np.einsum("ij,ij->i", N[ib], dirs)) < maxdot)
    sel_h = np.flatnonzero(okp)[:n_hyp]
    if len(sel_h) == 0:
        return [warm_cyl] if warm_cyl is not None else [Cylinder(False, "no_hypothesis")]
    A = V[ia[sel_h]]
    D = dirs[sel_h]
    # score all hypotheses at once: votes within tol of the line, normals ~perpendicular
    Q = Vs[None, :, :] - A[:, None, :]                                  # (H, S, 3)
    tt = np.einsum("hsk,hk->hs", Q, D)
    dist2 = np.einsum("hsk,hsk->hs", Q, Q) - tt * tt
    inl = (dist2 < tol * tol) & (np.abs(D @ Ns.T) < maxdot)
    scores = inl.sum(axis=1)
    # A pipe's normals rotate about its axis; the votes of a PLANE also line up
    # along any in-plane line, but all with one normal.  Such hypotheses would
    # crowd the finalists (walls, floor, cloth) and hide a smaller real pipe.
    nsum = inl.astype(np.float64) @ Ns
    nsum -= np.einsum("hk,hk->h", nsum, D)[:, None] * D                 # component about the axis
    coherence = np.linalg.norm(nsum, axis=1) / np.maximum(scores, 1)
    scores = np.where(coherence < p["ransac_max_normal_coherence"], scores, 0)
    # Finalists are DISTINCT axis lines (non-maximum suppression): the dominant
    # pipe yields many near-identical top hypotheses that would otherwise fill
    # every slot and hide a second pipe.  A warm model counts as already taken.
    taken = [(warm_cyl.axis, warm_cyl.point)] if warm_cyl is not None else []
    n_final = 3 if warm_cyl is not None else int(p["ransac_finalists"])
    order = []
    min_score = p["ransac_finalist_min_rel_score"] * float(scores.max()) if len(scores) else 0.0
    for j in np.argsort(-scores, kind="stable"):
        if scores[j] <= 0 or scores[j] < min_score or len(order) >= n_final:
            break
        dup = False
        for ax, pt in taken:
            q = A[j] - pt
            if abs(D[j] @ ax) > 0.99 and np.linalg.norm(q - (q @ ax) * ax) < 0.02 + tol:
                dup = True
                break
        if not dup:
            order.append(int(j))
            taken.append((D[j], A[j]))
    results = []
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    for j in order:
        d = D[j]
        q = V - A[j]
        tt = q @ d
        dist2 = np.einsum("ij,ij->i", q, q) - tt * tt
        good = (dist2 < tol * tol) & (np.abs(N @ d) < maxdot)
        if good.sum() < 10:
            continue
        cen = V[good].mean(axis=0)
        _, _, vt = np.linalg.svd(V[good] - cen, full_matrices=False)
        d2 = vt[0] if vt[0] @ d > 0 else -vt[0]
        if any(abs(d2 @ sd) > 0.996 and np.linalg.norm((cen - sc) - ((cen - sc) @ sd) * sd) < 0.02
               for sd, sc in seen):
            continue
        seen.append((d2, cen))
        results.append(finish(d2, cen, False))
        if len(results) >= 3:
            break
    if warm_cyl is not None:
        results.append(warm_cyl)
    if not results:
        return [Cylinder(False, "no_vote_consensus")]
    key = lambda cy: (cy.valid, cy.stats.get("precise", 0) * (1.1 if cy.warm else 1.0))
    results.sort(key=key, reverse=True)
    # sequential round: a small precise pipe can hide behind a larger soft
    # structure (cloth folds).  Search again on the unexplained points.
    best = results[0]
    if p["second_round"] and best.axis is not None and (
            best.residual_median_m > p["precise_residual_m"] or best.inliers < 0.5 * len(P)):
        r_b, _ = _axis_residuals(P, best.axis, best.point, R)
        rest = np.abs(r_b) > 0.02
        if rest.sum() >= p["min_surface_inliers"]:
            sub = {"P": np.full_like(grid["P"], np.nan), "N": np.full_like(grid["N"], np.nan),
                   "depth": grid.get("depth"), "k": grid.get("k")}
            if sub["depth"] is None:
                sub.pop("depth"); sub.pop("k")
            flatP = sub["P"].reshape(-1, 3); flatN = sub["N"].reshape(-1, 3)
            flatP[keep_idx[rest]] = P[rest]; flatN[keep_idx[rest]] = N[rest]
            q = dict(p); q["second_round"] = False; q["ransac_hypotheses"] = p["ransac_hypotheses_warm"]
            more = fit_cylinders(sub, R, q, None)
            for cy in more:
                if cy.valid and cy.axis is not None:
                    # re-express its support on the full point set
                    cy2 = finish(cy.axis, cy.point, False)
                    results.append(cy2)
            results.sort(key=key, reverse=True)
    distinct: list[Cylinder] = []
    for cy in results:
        if cy.axis is None:
            continue
        dup = False
        for dd in distinct:
            q = cy.point - dd.point
            if abs(cy.axis @ dd.axis) > 0.995 and np.linalg.norm(q - (q @ dd.axis) * dd.axis) < 0.02:
                dup = True
                break
        if not dup:
            distinct.append(cy)
    return distinct if distinct else [results[0]]


# ----------------------------------------------------------------------------
# 3. unwrap the visible cylinder surface
# ----------------------------------------------------------------------------

def _project(X: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = X[..., 2]
    zz = np.where(z > 1e-6, z, np.nan)
    u = k[0, 0] * X[..., 0] / zz + k[0, 2]
    v = k[1, 1] * X[..., 1] / zz + k[1, 2]
    return u, v, z


def unwrap_grid(cyl: Cylinder, k: np.ndarray, shape: tuple[int, int], p: dict) -> dict | None:
    d, c, R = cyl.axis, cyl.point, cyl.radius
    h, w = shape
    D = float(np.linalg.norm(c))
    if D <= R * 1.02:
        return None
    er = -c / D
    et = np.cross(d, er)
    tlim = math.acos(R / D) * p["unwrap_limb_margin"]
    # axial range where the surface projects inside the image
    s_probe = np.linspace(-2.0, 2.0, 801)
    th_probe = np.linspace(-tlim, tlim, 9)
    Xp = (c[None, None, :] + s_probe[None, :, None] * d[None, None, :]
          + R * (np.cos(th_probe)[:, None, None] * er + np.sin(th_probe)[:, None, None] * et))
    up, vp, zp = _project(Xp, k)
    ins = (zp > 0.05) & (up >= 0) & (up <= w - 1) & (vp >= 0) & (vp <= h - 1)
    cols_in = ins.any(axis=0)
    if not cols_in.any():
        return None
    idx = np.flatnonzero(cols_in)
    s0 = s_probe[max(idx[0] - 1, 0)]
    s1 = s_probe[min(idx[-1] + 1, len(s_probe) - 1)]
    # axial step: finest projected scale along theta=0 over the range
    s_line = np.linspace(s0, s1, 200)
    Xc = c + s_line[:, None] * d + R * er
    uc, vc, zc = _project(Xc, k)
    duv = np.hypot(np.diff(uc), np.diff(vc)) / np.diff(s_line)
    px_per_m = float(np.nanmax(duv)) if np.isfinite(duv).any() else 0.0
    if px_per_m < 50:
        return None
    step = max(p["unwrap_col_px"] / px_per_m, p["unwrap_col_min_m"])
    ncol = int(math.ceil((s1 - s0) / step)) + 1
    if ncol > p["unwrap_max_cols"]:
        ncol = int(p["unwrap_max_cols"])
        step = (s1 - s0) / (ncol - 1)
    stations = s0 + step * np.arange(ncol)
    # rows: angular step from the arc-centre scale R*f/(z - R), i.e. somewhat
    # denser than row_px there (capped by unwrap_max_rows)
    zmid = float(np.nanmedian(zc)) if np.isfinite(zc).any() else D
    px_per_rad = R * k[0, 0] / max(zmid - R, 0.05)
    nrow = int(np.clip(2 * tlim * px_per_rad / p["unwrap_row_px"], 16, p["unwrap_max_rows"]))
    thetas = np.linspace(-tlim, tlim, nrow)
    nrm = np.cos(thetas)[:, None] * er + np.sin(thetas)[:, None] * et          # (nrow,3)
    A = (c[None, :] + R * nrm).astype(np.float32)                              # (nrow,3)
    st = stations.astype(np.float32)
    df = d.astype(np.float32)
    z = A[:, 2:3] + st[None, :] * df[2]
    X0 = A[:, 0:1] + st[None, :] * df[0]
    Y0 = A[:, 1:2] + st[None, :] * df[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        zi = np.where(z > 0.05, 1.0 / z, np.nan).astype(np.float32)
    u = np.float32(k[0, 0]) * X0 * zi + np.float32(k[0, 2])
    v = np.float32(k[1, 1]) * Y0 * zi + np.float32(k[1, 2])
    inside = (z > 0.05) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    u = np.where(inside, u, -1).astype(np.float32)
    v = np.where(inside, v, -1).astype(np.float32)
    return {"shape": (h, w), "stations": stations, "thetas": thetas, "u": u, "v": v, "z": z, "inside": inside,
            "normals": nrm, "er": er, "et": et, "step_m": step, "px_per_m": px_per_m,
            "px_per_rad": px_per_rad}


def surface_match(grid: dict, depth_m: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(match, contradiction, measured depth): measured depth agrees with /
    contradicts the cylinder surface at each unwrapped sample.  Missing depth
    is neither."""
    meas = cv2.remap(depth_m, grid["u"], grid["v"], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = grid["inside"] & (meas > 0)
    err = np.abs(meas - grid["z"])
    return valid & (err < tol), valid & (err >= 2.0 * tol), meas


def pipe_ends(ug: dict, meas: np.ndarray, cyl: Cylinder, p: dict) -> list[dict]:
    """Ends of the measured pipe that are in view.  Beyond each end of the
    measured axial range, where the model surface would still project inside
    the image, the depth must show the scene BEHIND the surface (background, or
    the inside of an open pipe).  The image border (no samples), missing depth
    and something in front (an occluder) are not an end."""
    st = ug["stations"]
    behind_tol = 2.0 * p["flank_depth_tol_m"]
    ends = []
    for side, edge in zip((-1, 1), cyl.axial_range):
        a = edge + side * p["pipe_end_gap_m"]
        b = a + side * p["pipe_end_probe_m"]
        cols = (st >= min(a, b)) & (st <= max(a, b))
        ins = ug["inside"][:, cols]
        if ins.size == 0 or ins.mean() < p["pipe_end_min_inside_frac"]:
            continue
        me, z = meas[:, cols], ug["z"][:, cols]
        valid = ins & (me > 0)
        nv = int(valid.sum())
        if nv == 0 or nv < p["pipe_end_min_valid_frac"] * int(ins.sum()):
            continue
        if np.count_nonzero(valid & (me > z + behind_tol)) >= p["pipe_end_min_behind_frac"] * nv:
            pt = cyl.point + float(edge) * cyl.axis
            ends.append({"side": side, "station_m": float(edge), "point_camera_xyz_m": [float(x) for x in pt]})
    return ends


# ----------------------------------------------------------------------------
# 4. dark valley response and candidate paths
# ----------------------------------------------------------------------------

def _shift_cols(a: np.ndarray, off: int, fill=np.nan) -> np.ndarray:
    """out[:, j] = a[:, j + off] (NaN/`fill` outside)."""
    out = np.full_like(a, fill)
    n = a.shape[1]
    if off == 0:
        return a.copy()
    if abs(off) >= n:
        return out
    if off > 0:
        out[:, : n - off] = a[:, off:]
    else:
        out[:, -off:] = a[:, : n + off]
    return out


def valley_response(U: np.ndarray, valid: np.ndarray, p: dict) -> dict:
    """Two-sided (and weaker one-sided) dark-line response in the unwrapped image.
    Computed on column-shifted VIEWS (no copies); NaN marks samples outside the image."""
    Uf = np.where(valid, U, 0.0).astype(np.float32)
    Vf = valid.astype(np.float32)
    kr = int(p["row_smooth"])
    if kr > 1:
        Uf = cv2.blur(Uf, (1, kr), borderType=cv2.BORDER_CONSTANT)
        Vf = cv2.blur(Vf, (1, kr), borderType=cv2.BORDER_CONSTANT)
    ok = Vf > 0.99
    centre = np.where(ok, Uf / np.maximum(Vf, 1e-6), np.nan).astype(np.float32)
    cfill = np.nan_to_num(centre, nan=0.0)
    fw = int(p["valley_flank_px"])
    B = cv2.blur(cfill, (fw, 1), borderType=cv2.BORDER_CONSTANT)
    C = cv2.blur(ok.astype(np.float32), (fw, 1), borderType=cv2.BORDER_CONSTANT)
    Bm = np.where(C > 0.99, B / np.maximum(C, 1e-6), np.nan).astype(np.float32)
    n = U.shape[1]
    best = np.zeros(U.shape, np.float32)
    best_flank = np.full(U.shape, np.nan, np.float32)
    best_off = np.zeros(U.shape, np.int16)             # flank distance of the winning scale
    one_sided = np.zeros(U.shape, bool)
    half_fw = (fw + 1) // 2
    nanL = np.isnan(Bm)
    for hw in p["valley_halfwidths_px"]:
        o = hw + half_fw
        if 2 * o >= n:
            continue
        L, R = Bm[:, : n - 2 * o], Bm[:, 2 * o:]          # flanks of column j = o .. n-o-1
        cen = centre[:, o: n - o]
        flank = np.fmin(L, R)                              # NaN only if both flanks missing
        resp = flank - cen
        single = nanL[:, : n - 2 * o] ^ nanL[:, 2 * o:]
        np.multiply(resp, 0.7, out=resp, where=single)
        np.nan_to_num(resp, copy=False, nan=0.0)
        tgt = best[:, o: n - o]
        better = resp > tgt
        np.copyto(tgt, resp, where=better)
        np.copyto(best_flank[:, o: n - o], flank, where=better)
        np.copyto(best_off[:, o: n - o], o, where=better)
        np.copyto(one_sided[:, o: n - o], single, where=better)
        # columns within o of the grid ends: one-sided only (flank at j + o / j - o)
        for cols, fl in ((slice(0, o), Bm[:, o: 2 * o]),
                         (slice(n - o, n), Bm[:, n - 2 * o: n - o])):
            c2 = centre[:, cols]
            r2 = np.nan_to_num(0.7 * (fl - c2), nan=0.0)
            t2 = best[:, cols]
            b2 = r2 > t2
            np.copyto(t2, r2, where=b2)
            np.copyto(best_flank[:, cols], fl, where=b2)
            np.copyto(best_off[:, cols], o, where=b2)
            np.copyto(one_sided[:, cols], True, where=b2)
    best_rel = best / np.maximum(np.nan_to_num(best_flank, nan=1.0), 1.0)
    best[~np.isfinite(centre)] = 0.0
    return {"resp": best, "rel": best_rel, "flank": best_flank, "off": best_off, "one_sided": one_sided,
            "centre": centre}


def _noise_level(U: np.ndarray) -> float:
    """Robust grey-level noise from second differences along the axial direction."""
    Us = U[::2]
    d2 = Us[:, 2:] - 2 * Us[:, 1:-1] + Us[:, :-2]
    d2 = d2[np.isfinite(d2)]
    if d2.size > 20000:
        d2 = d2[:: d2.size // 20000]
    if d2.size < 100:
        return 2.0
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / math.sqrt(6.0))


def trace_paths(score: np.ndarray, cols: list[int], half: int) -> np.ndarray:
    """Max-sum near-vertical paths (|step| <= 1 column per row) for several
    candidate columns at once.  Returns (ncand, nrow) absolute columns."""
    nrow, ncol = score.shape
    nc = len(cols)
    wdt = 2 * half + 1
    base = np.array(cols, np.int64)[:, None] + np.arange(-half, half + 1)[None, :]
    inb = (base >= 0) & (base < ncol)
    basec = np.clip(base, 0, ncol - 1)
    S = score[:, basec]                                   # (nrow, nc, wdt)
    S = np.where(inb[None], S, -1e9).astype(np.float64)
    acc = S[0].copy()
    back = np.zeros((nrow, nc, wdt), np.int8)
    neg = np.full((nc, 1), -1e18)
    for r in range(1, nrow):
        left = np.concatenate([neg, acc[:, :-1]], axis=1)
        right = np.concatenate([acc[:, 1:], neg], axis=1)
        best = np.maximum(acc, np.maximum(left, right))
        b = np.where(best == acc, 0, np.where(best == left, -1, 1)).astype(np.int8)
        acc = S[r] + best
        back[r] = b
    idx = np.argmax(acc, axis=1)
    paths = np.empty((nc, nrow), np.int64)
    paths[:, -1] = idx
    ar = np.arange(nc)
    for r in range(nrow - 1, 0, -1):
        idx = idx + back[r, ar, idx]
        paths[:, r - 1] = idx
    return base[ar[:, None], paths]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ----------------------------------------------------------------------------
# 6. ring fit to the observed RGB line
# ----------------------------------------------------------------------------

def ray_cylinder(uv: np.ndarray, k: np.ndarray, cyl: Cylinder) -> tuple[np.ndarray, np.ndarray]:
    """Front intersection of pixel rays with the cylinder.  Returns X (n,3), ok."""
    d, c, R = cyl.axis, cyl.point, cyl.radius
    rays = np.column_stack([(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    # |(t r - c) - ((t r - c).d) d|^2 = R^2
    rp = rays - (rays @ d)[:, None] * d
    cp = -c + (c @ d) * d          # component of (0 - c) perpendicular to d  (= -c since c.d = 0)
    A = np.einsum("ij,ij->i", rp, rp)
    B = 2 * rp @ cp
    C = cp @ cp - R * R
    disc = B * B - 4 * A * C
    ok = (disc >= 0) & (A > 1e-12)
    t = (-B - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(A, 1e-12))
    ok &= t > 0
    return rays * t[:, None], ok


def fit_ring(line_uv: np.ndarray, k: np.ndarray, cyl: Cylinder, p: dict, weights: np.ndarray | None = None) -> dict:
    """Plane of the junction ring from observed line pixels lifted onto the cylinder.

    t_i = s - a cos(phi_i) - b sin(phi_i),  a = alpha R, b = beta R (small tilts)."""
    X, ok = ray_cylinder(line_uv, k, cyl)
    if ok.sum() < 5:
        return {"ok": False}
    X = X[ok]
    d, c, R = cyl.axis, cyl.point, cyl.radius
    D = float(np.linalg.norm(c)); er = -c / D; et = np.cross(d, er)
    q = X - c
    t = q @ d
    rad = q - t[:, None] * d
    phi = np.arctan2(rad @ et, rad @ er)
    w0 = np.ones(len(t)) if weights is None else weights[ok]
    # MAP estimate: residual noise estimated robustly from the data, Gaussian
    # prior on the ring tilt (a = alpha R, b = beta R).
    # The depth axis is less certain when the IMAGE itself shows only a narrow
    # arc of the pipe (strip cut by the frame border): allow more tilt there.
    # A narrow depth arc caused by missing depth on a pipe that fills the image
    # keeps the strict prior (extra freedom would only fit depth noise).
    az = max(float(cyl.stats.get("image_azimuth_deg", 180.0)), 1.0)
    tilt_deg = p["ring_tilt_prior_deg"] * float(np.clip(p["ring_tilt_ref_azimuth_deg"] / az, 1.0, p["ring_tilt_max_scale"]))
    sig_a = R * math.radians(tilt_deg)
    A = np.column_stack([np.ones_like(t), -np.cos(phi), -np.sin(phi)])
    wts = w0.copy()
    sol = np.array([np.median(t), 0.0, 0.0])
    sig_t = 5e-4
    for _ in range(p["ring_fit_iterations"]):
        W = wts[:, None]
        H = (A.T @ (A * W)) / sig_t ** 2
        H[1, 1] += 1.0 / sig_a ** 2
        H[2, 2] += 1.0 / sig_a ** 2
        sol = np.linalg.solve(H + 1e-12 * np.eye(3), (A.T @ (wts * t)) / sig_t ** 2)
        res = t - A @ sol
        sig_t = max(1.4826 * float(np.median(np.abs(res))), 1.5e-4)
        wts = w0 * np.clip(2.0 * sig_t / np.maximum(np.abs(res), 1e-12), 0, 1)
    s, a, b = sol
    alpha, beta = a / R, b / R
    tilt = math.degrees(math.hypot(alpha, beta))
    maxt = math.radians(p["ring_max_tilt_deg"])
    if math.hypot(alpha, beta) > maxt:
        sc = maxt / math.hypot(alpha, beta)
        alpha, beta = alpha * sc, beta * sc
    n = _unit(d + alpha * er + beta * et)
    center = c + s * d
    return {"ok": True, "station": float(s), "normal": n, "center": center, "tilt_deg": tilt,
            "phi": phi, "t": t, "X": X, "resid_m": t - A @ sol}


def ring_pixels(center: np.ndarray, normal: np.ndarray, R: float, k: np.ndarray, n: int = 361) -> np.ndarray:
    radial = -center + (center @ normal) * normal
    dist = float(np.linalg.norm(radial))
    if dist <= R:
        return np.empty((0, 2))
    radial /= dist
    tr = np.cross(normal, radial)
    lim = math.acos(R / dist)
    th = np.linspace(-lim, lim, n)
    X = center + R * (np.cos(th)[:, None] * radial + np.sin(th)[:, None] * tr)
    X = X[X[:, 2] > 1e-6]
    uvw = X @ k.T
    return uvw[:, :2] / uvw[:, 2:3]


def point_curve_distance(pts: np.ndarray, curve: np.ndarray) -> np.ndarray:
    if len(curve) < 2 or len(pts) == 0:
        return np.full(len(pts), np.inf)
    a, dlt = curve[:-1], np.diff(curve, axis=0)
    den = np.maximum(np.sum(dlt * dlt, axis=1), 1e-18)
    diff = pts[:, None, :] - a[None, :, :]
    tt = np.clip(np.sum(diff * dlt[None], axis=2) / den, 0, 1)
    res = diff - tt[:, :, None] * dlt[None]
    return np.sqrt(np.min(np.sum(res * res, axis=2), axis=1))


# ----------------------------------------------------------------------------
# 5. candidate verification
# ----------------------------------------------------------------------------

def _value_chroma(rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """max-channel value and chroma (max-min)/max at pixel positions (nearest)."""
    h, w = rgb.shape[:2]
    ui = np.clip(np.rint(u).astype(np.int64), 0, w - 1)
    vi = np.clip(np.rint(v).astype(np.int64), 0, h - 1)
    px = rgb[vi, ui].astype(np.float32)
    vmax = px.max(axis=-1)
    return vmax, (vmax - px.min(axis=-1)) / np.maximum(vmax, 1.0)


def _band_centroid(V: dict, path: np.ndarray, rows: np.ndarray, ncol: int, W: int) -> np.ndarray:
    """Centroid of (flank - intensity)+ over the dark run that contains the
    darkest sample within +-W columns of the path."""
    offs = np.arange(-W, W + 1)
    jj = np.clip(path[:, None] + offs[None, :], 0, ncol - 1)
    cen = V["centre"][rows[:, None], jj]
    fl = V["flank"][rows, path]
    fl = np.where(np.isfinite(fl), fl, np.nanmax(np.where(np.isfinite(cen), cen, -np.inf), axis=1))
    dark = np.nan_to_num(fl[:, None] - cen, nan=0.0)
    dark = np.maximum(dark, 0.0)
    peak = np.argmax(dark, axis=1)
    # keep only the connected run of positive darkness (>= 35% of the peak) around the peak
    thr = 0.35 * dark[np.arange(len(rows)), peak]
    keep = dark >= thr[:, None]
    idx = np.arange(len(offs))[None, :]
    left_break = np.where(~keep & (idx < peak[:, None]), idx, -1).max(axis=1)
    right_break = np.where(~keep & (idx > peak[:, None]), idx, len(offs)).min(axis=1)
    mask = (idx > left_break[:, None]) & (idx < right_break[:, None])
    wsum = np.sum(dark * mask, axis=1)
    cent = np.where(wsum > 1e-6, np.sum(dark * mask * offs[None, :], axis=1) / np.maximum(wsum, 1e-6), 0.0)
    return path + cent


def _path_score(V: dict, pipe_ok: np.ndarray, thr_abs: float) -> np.ndarray:
    cap = 3.0 * thr_abs
    score = np.where(pipe_ok, np.minimum(V["resp"], cap), 0.0)
    return np.where(score > 0, score, -0.25 * thr_abs)


def _evaluate_candidate(col0: int, path: np.ndarray, V: dict, E: np.ndarray, pipe_ok: np.ndarray,
                        surf: np.ndarray, ug: dict, k: np.ndarray, cyl: Cylinder, p: dict,
                        thr_abs: float, relax: float = 1.0, contra: np.ndarray | None = None) -> dict:
    resp, rel = V["resp"], V["rel"]
    nrow, ncol = resp.shape
    rows = np.arange(nrow)
    on = ug["inside"][rows, path] & pipe_ok[rows, path]
    ev = on & E[rows, path]
    out: dict[str, Any] = {"col0": int(col0), "ok": False, "reason": "", "relax": relax}
    if ev.sum() < 3:
        out["reason"] = "no_evidence"
        return out
    # sub-pixel line centre per row: centroid of the darkness of the band
    # around the path (the capped path score is flat across a wide band).
    colf = _band_centroid(V, path, rows, ncol, int(p["centroid_halfwin_px"]))
    jf = np.clip(np.floor(colf).astype(np.int64), 0, ncol - 2)
    wf = colf - jf
    u_img = (1 - wf) * ug["u"][rows, jf] + wf * ug["u"][rows, jf + 1]
    v_img = (1 - wf) * ug["v"][rows, jf] + wf * ug["v"][rows, jf + 1]
    # longest run of evidence rows, bridging image-space holes up to max_hole_px
    ev_idx = np.flatnonzero(ev)
    pts = np.column_stack([u_img[ev_idx], v_img[ev_idx]])
    gaps = np.hypot(*np.diff(pts, axis=0).T) if len(pts) > 1 else np.zeros(0)
    brk = np.flatnonzero(gaps > p["max_hole_px"] * (1.0 / max(relax, 0.5)))
    seg_starts = np.r_[0, brk + 1]
    seg_ends = np.r_[brk + 1, len(ev_idx)]
    seg_len = [float(np.sum(gaps[a:b - 1])) if b - a > 1 else 0.0 for a, b in zip(seg_starts, seg_ends)]
    best = int(np.argmax(seg_len))
    a, b = seg_starts[best], seg_ends[best]
    run_rows = ev_idx[a:b]
    r0, r1 = run_rows[0], run_rows[-1]
    span_rows = np.arange(r0, r1 + 1)
    span_on = on[span_rows]
    line_len = seg_len[best]
    run_frac = len(run_rows) / max(1, int(span_on.sum()))
    line_uv = np.column_stack([u_img[run_rows], v_img[run_rows]])
    uniq = len(np.unique(np.rint(line_uv).astype(np.int64), axis=0))
    # Localisation only: follow the same path into weaker (blurred) evidence at
    # both ends, bridging small holes.  Acceptance still uses the strong run.
    weak = on & (resp[rows, path] >= p["weak_evidence_factor"] * thr_abs) \
        & (rel[rows, path] >= p["weak_evidence_factor"] * p["valley_min_rel"])
    loc_rows = list(run_rows)
    for direction in (-1, 1):
        r = int(run_rows[0] if direction < 0 else run_rows[-1])
        last_uv = np.array([u_img[r], v_img[r]])
        while True:
            nxt = None
            rr = r + direction
            while 0 <= rr < nrow:
                if weak[rr]:
                    nxt = rr
                    break
                if not on[rr]:
                    break
                rr += direction
            if nxt is None:
                break
            uvn = np.array([u_img[nxt], v_img[nxt]])
            if np.hypot(*(uvn - last_uv)) > p["max_hole_px"] * 1.5:
                break
            loc_rows.append(nxt)
            r, last_uv = nxt, uvn
    loc_rows = np.array(sorted(set(loc_rows)))
    loc_uv = np.column_stack([u_img[loc_rows], v_img[loc_rows]])
    out["loc_rows"] = int(len(loc_rows))
    # share of the visible ring (rows on the pipe at this station) that shows the valley
    out["coverage"] = float(ev.sum()) / max(1, int(on.sum()))
    out.update(line_len_px=line_len, run_rows=int(len(run_rows)), run_frac=float(run_frac), unique_px=int(uniq),
               station_col=float(np.median(colf[run_rows])))
    # diffuse-row support inside the extent of the observed line.  A specular
    # highlight is brighter AND whiter (dichromatic model) than the body colour;
    # shading alone never makes a row specular.
    span = np.arange(r0, r1 + 1)
    span = span[on[span]]
    # The flanks are sampled where the valley of that row takes them (its
    # winning scale, at least 6 samples out): a wide interruption of a
    # reflection would otherwise be sampled inside the interruption.
    fo = np.maximum(V["off"][span, path[span]], 6)
    val = np.zeros(len(span), np.float32)
    chr_ = np.full(len(span), np.inf, np.float32)
    for d_off in (0, 1, 2):                    # the flank window of that scale and just beyond
        for sgn in (-1, 1):
            jj = np.clip(path[span] + sgn * (fo + d_off), 0, ncol - 1)
            vs, cs = _value_chroma(V["rgb"], ug["u"][span, jj], ug["v"][span, jj])
            val = np.maximum(val, vs)
            chr_ = np.minimum(chr_, cs)
    on_all = np.flatnonzero(on)
    ja = np.clip(path[on_all] - 6, 0, ncol - 1)            # body colour reference
    va, ca = _value_chroma(V["rgb"], ug["u"][on_all, ja], ug["v"][on_all, ja])
    ref_val = float(np.median(va)) if len(on_all) else 1.0
    ref_chr = float(np.median(ca)) if len(on_all) else 0.0
    # a clipped sample counts as a highlight only if it is clipped WHITE: a
    # saturated body colour (one channel at 255) is not a reflection
    spec = (val > 1.12 * ref_val) & ((chr_ < 0.6 * ref_chr) | ((val >= 245) & (chr_ < 0.1)))
    if p["achromatic_specular"] and ref_chr < p["achromatic_max_chroma"] and out["coverage"] < p["achromatic_partial_coverage"]:
        # No colour to tell a highlight from the body (grey pipe, monochrome
        # image).  A line over only part of the ring can be an interrupted
        # reflection: the reflection is a band whose brightness drops sharply
        # right outside the line, and the valley floor is the plain surface,
        # as bright as the rows next to the band.  A gap is darker than the
        # surface around it; a lit side of the pipe has soft edges.
        k_end, gap_rows, n_side = int(p["achromatic_edge_rows"]), 2, int(p["achromatic_side_rows"])
        sides = []
        for sgn, r_edge, end_val in ((-1, r0, val[:k_end]), (1, r1, val[-k_end:])):
            rr = r_edge + sgn * np.arange(gap_rows + 1, gap_rows + 1 + n_side)
            rr = rr[(rr >= 0) & (rr < nrow)]
            rr = rr[on[rr]]
            if len(rr) < 3:
                continue                        # the line reaches the end of the observed ring here
            outside = np.zeros(len(rr), np.float32)
            for off in (6, 8):
                for sg in (-1, 1):
                    jj = np.clip(path[rr] + sg * off, 0, ncol - 1)
                    outside = np.maximum(outside, _value_chroma(V["rgb"], ug["u"][rr, jj], ug["v"][rr, jj])[0])
            sides.append((float(np.median(end_val)), float(np.median(outside))))
        if sides:
            cc = np.rint(colf[span]).astype(np.int64)
            floor = np.full(len(span), np.inf, np.float32)
            for dj in (-1, 0, 1):
                jj = np.clip(cc + dj, 0, ncol - 1)
                floor = np.minimum(floor, _value_chroma(V["rgb"], ug["u"][span, jj], ug["v"][span, jj])[0])
            band = all(e > p["achromatic_step_ratio"] * o + p["achromatic_step_abs"] for e, o in sides)
            surface = float(np.median(floor)) >= p["achromatic_cavity_ratio"] * min(o for _, o in sides)
            out["achromatic_band"] = [round(e, 1) for e, _ in sides] + [round(o, 1) for _, o in sides]
            if band and surface:
                spec[:] = True                  # the whole line is an interrupted reflection
    diffuse_rows = span[~spec]
    dfrac = float(ev[diffuse_rows].mean()) if len(diffuse_rows) else 0.0
    out.update(diffuse_rows=int(len(diffuse_rows)), diffuse_frac=dfrac, specular_rows=int(spec.sum()),
               spec_ev_frac=float(ev[span[spec]].mean()) if spec.any() else 0.0,
               ref_val=ref_val, ref_chroma=ref_chr)
    # depth beyond a border-touching line: if the strip between the line and the
    # image edge is observed and is NOT pipe, the line is a pipe end.
    beyond = []
    for sgn in (-1, 1):
        cnt_in = cnt_bad = 0
        for dj in range(3, 12):
            jj = np.rint(colf[run_rows]).astype(int) + sgn * dj
            okj = (jj >= 0) & (jj < ncol)
            jjc = np.clip(jj, 0, ncol - 1)
            ins = okj & ug["inside"][run_rows, jjc]
            cnt_in += int(ins.sum())
            cnt_bad += int((ins & contra[run_rows, jjc]).sum())
        beyond.append((cnt_in, cnt_bad))
    out["beyond"] = beyond
    # walk outward from the line on each side: a pipe END shows contradicting
    # depth (background) before the pipe surface resumes; a junction does not.
    nwalk = max(3, int(round(p["end_walk_m"] / ug["step_m"])))
    need_surf = max(2, int(round(p["end_resume_m"] / ug["step_m"])))
    cr = np.rint(colf[run_rows]).astype(int)
    ends = []
    for sgn in (-1, 1):
        offs = sgn * np.arange(2, nwalk + 1)
        jj = cr[:, None] + offs[None, :]
        okj = (jj >= 0) & (jj < ncol)
        jjc = np.clip(jj, 0, ncol - 1)
        rr = run_rows[:, None]
        ins = okj & ug["inside"][rr, jjc]
        con = ins & contra[rr, jjc]
        srf = ins & surf[rr, jjc]
        big = nwalk + 10
        first_con = np.where(con.any(axis=1), np.argmax(con, axis=1), big)
        first_out = np.where((~ins).any(axis=1), np.argmax(~ins, axis=1), big)
        csum = np.cumsum(srf, axis=1)
        idx = np.clip(first_con, 0, jj.shape[1] - 1)
        surf_before = np.where(first_con < big, csum[np.arange(len(cr)), idx], csum[:, -1])
        # a junction whose physical gap is resolved by the depth camera also shows
        # non-pipe depth right at the line, but the surface resumes after the gap
        surf_after = csum[:, -1] - np.where(first_con < big, csum[np.arange(len(cr)), idx], csum[:, -1])
        ended = (first_con < first_out) & (surf_before < need_surf) & (surf_after < need_surf)
        ends.append(float(ended.mean()) if len(ended) else 0.0)
    out["end_frac"] = ends
    # depth on the pipe flanks (metric offset along the axis)
    off = max(1, int(round(p["flank_offset_m"] / ug["step_m"])))
    side_stats = []
    for sgn in (-1, 1):
        jj = np.clip(np.rint(colf[run_rows]).astype(int) + sgn * off, -1, ncol)
        inimg = (jj >= 0) & (jj < ncol)
        jjc = np.clip(jj, 0, ncol - 1)
        inimg &= ug["inside"][run_rows, jjc]
        match = inimg & surf[run_rows, jjc]
        bad = inimg & contra[run_rows, jjc]
        side_stats.append((int(inimg.sum()), int(match.sum()), int(bad.sum())))
    out["flanks"] = side_stats
    # radial step across the line.  A butt junction joins two surfaces of the
    # same radius; the edge of a socket / flange / clamp is a step of its wall
    # thickness.  Radial offset of a measured sample from the model surface:
    # delta = (R - D cos(theta)) * (z_meas - z_model) / z_model  (> 0 outside).
    lo = max(1, int(round(p["step_inner_m"] / ug["step_m"])))
    hi = max(lo + 2, int(round(p["step_outer_m"] / ug["step_m"])))
    wrow = cyl.radius - float(np.linalg.norm(cyl.point)) * np.cos(ug["thetas"][run_rows])
    radial = []
    for sgn in (-1, 1):
        jj = cr[:, None] + sgn * np.arange(lo, hi + 1)[None, :]
        okj = (jj >= 0) & (jj < ncol)
        jjc = np.clip(jj, 0, ncol - 1)
        rr = run_rows[:, None]
        zm = V["meas"][rr, jjc]
        zz = ug["z"][rr, jjc]
        okm = okj & ug["inside"][rr, jjc] & (zm > 0) & (np.abs(zm - zz) < p["flank_depth_tol_m"])
        dl = (wrow[:, None] * (zm - zz) / np.maximum(zz, 1e-6))[okm]
        radial.append(float(np.median(dl)) if dl.size >= p["step_min_samples"] else float("nan"))
    out["radial_mm"] = [1e3 * x for x in radial]
    out["radial_step_mm"] = abs(out["radial_mm"][0] - out["radial_mm"][1]) if all(np.isfinite(radial)) else float("nan")
    # ring fit on the observed line
    ring = fit_ring(loc_uv, k, cyl, p)
    if ring["ok"]:
        # 121 samples over <= 180 deg of ring: chord sagitta < 0.05 px, far below the gate
        curve = ring_pixels(ring["center"], ring["normal"], cyl.radius, k, n=121)
        dist = point_curve_distance(line_uv, curve)
        out.update(ring=ring, ring_med_px=float(np.median(dist)), ring_p90_px=float(np.percentile(dist, 90)),
                   tilt_deg=ring["tilt_deg"])
    # strength
    rels = rel[run_rows, path[run_rows]]
    out["rel_median"] = float(np.median(rels))
    out["resp_median"] = float(np.median(resp[run_rows, path[run_rows]]))
    out["score"] = float(line_len * min(out["rel_median"], 0.5) * max(dfrac, 0.05))
    out["line_uv"] = loc_uv
    # ---- decisions (order: physical vetoes first)
    (inA, mA, _), (inB, mB, _) = side_stats
    minr = p["min_flank_rows"]
    for n_in, n_m, n_b in side_stats:
        if n_in >= minr and n_b > 0.5 * n_in and n_m < p["pipe_end_max_match"] * n_in:
            out["reason"] = "pipe_end_or_occluder"
            return out
    sideA_ok, sideB_ok = mA >= minr, mB >= minr
    # a flank whose samples fall mostly outside the image is a border case
    out_lim = max(minr, p["flank_outside_frac"] * len(run_rows))
    sideA_out, sideB_out = inA < out_lim, inB < out_lim
    if max(out["end_frac"]) >= p["end_min_row_frac"]:
        out["reason"] = "pipe_end_near_line"
        return out
    for (n_in, n_bad), side_out in zip(beyond, (sideA_out, sideB_out)):
        if side_out and n_in >= 3 * minr and n_bad > 0.5 * n_in:
            out["reason"] = "pipe_end_at_border"
            return out
    if not ((sideA_ok and (sideB_ok or sideB_out)) or (sideB_ok and (sideA_ok or sideA_out))):
        out["reason"] = "no_pipe_on_flanks"
        return out
    hh, ww = ug["shape"]
    bd = np.minimum.reduce([line_uv[:, 0], ww - 1 - line_uv[:, 0], line_uv[:, 1], hh - 1 - line_uv[:, 1]])
    out["border_dist_px"] = float(np.median(bd))
    out["border_hug"] = bool((sideA_out or sideB_out) and out["border_dist_px"] < p["border_min_px"])
    if out["diffuse_rows"] >= p["min_diffuse_rows"] and out["diffuse_frac"] < p["min_diffuse_fraction"]:
        out["reason"] = "specular_interruption"
        return out
    if out["diffuse_rows"] < p["min_diffuse_rows"] and out.get("specular_rows", 0) >= out["diffuse_rows"]:
        out["reason"] = "specular_only_arc"
        return out
    if uniq < p["min_line_pixels"] * relax or line_len < p["min_line_pixels"] * relax:
        out["reason"] = "short_line"
        return out
    if run_frac < p["min_run_fraction"] * relax:
        out["reason"] = "fragmented_line"
        return out
    if not ring["ok"]:
        out["reason"] = "ring_fit_failed"
        return out
    if out["ring_p90_px"] > 4.0 or out["ring_med_px"] > 2.0:
        out["reason"] = "line_not_a_ring"
        return out
    if out["border_hug"]:
        out["reason"] = "border_unverifiable"   # passes only as continuation of a live track
        return out
    out["ok"] = True
    out["reason"] = "ok"
    return out


def track_distance(line_uv: np.ndarray | None, track_uv: np.ndarray | None) -> float:
    """Median distance (px) from a candidate line to the previously tracked line
    (point-to-polyline), the association metric for line-shaped features."""
    if line_uv is None or track_uv is None or len(line_uv) == 0 or len(track_uv) < 2:
        return float("inf")
    pts = line_uv[:: max(1, len(line_uv) // 60)]
    curve = track_uv[:: max(1, len(track_uv) // 200)]
    return float(np.median(point_curve_distance(pts, curve)))


def pipe_intensity_weights(Uc: np.ndarray, on_pipe: np.ndarray, p: dict) -> np.ndarray:
    """Channel weights matched to the pipe colour (see DEFAULTS "intensity_mode").

    Uc: unwrapped RGB samples (rows, cols, 3); on_pipe: samples the depth puts on
    the cylinder.  Symmetric in the channels, so any permutation of the colour
    channels permutes the weights and leaves the intensity image unchanged."""
    st = max(1, int(math.sqrt(on_pipe.size / 16000.0)))  # regular sub-grid: the albedo is smooth
    sub, msk = Uc[::st, ::st], on_pipe[::st, ::st]
    px = sub[msk]
    if len(px) < 50:
        px = sub.reshape(-1, 3)
    if len(px) > 4000:                                   # deterministic subsample
        px = px[:: len(px) // 4000]
    if len(px) == 0:
        return np.full(3, 1.0 / 3.0)
    alb = np.median(px, axis=0).astype(np.float64)
    clipped = (px >= p["intensity_clip_level"]).mean(axis=0)
    w = np.power(np.maximum(alb, 1.0), p["intensity_albedo_power"]) * np.power(1.0 - clipped, p["intensity_clip_power"])
    tot = float(w.sum())
    return w / tot if tot > 1e-9 else np.full(3, 1.0 / 3.0)


def _detect_on_cylinder(rgb: np.ndarray, depth_m: np.ndarray, k: np.ndarray, cyl: Cylinder, p: dict,
                        track_uv: np.ndarray | None, debug: dict | None, shared: dict) -> Detection:
    t0 = time.perf_counter()
    t2 = t0
    stats: dict[str, Any] = {"cyl_warm": cyl.warm}
    h, w = depth_m.shape
    ug = unwrap_grid(cyl, k, (h, w), p)
    if ug is not None:
        lo_s, hi_s = cyl.axial_range
        cols_sup = (ug["stations"] >= lo_s) & (ug["stations"] <= hi_s)
        rows_in = ug["inside"][:, cols_sup].any(axis=1) if cols_sup.any() else ug["inside"].any(axis=1)
        th = ug["thetas"][rows_in]
        cyl.stats["image_azimuth_deg"] = float(np.degrees(th.max() - th.min())) if len(th) else 0.0
    if ug is None:
        stats["ms_total"] = 1000 * (time.perf_counter() - t0)
        return Detection(False, "unwrap_failed", cyl, stats=stats)
    surf, contra, meas = surface_match(ug, depth_m, p["flank_depth_tol_m"])
    stats["pipe_ends"] = pipe_ends(ug, meas, cyl, p)
    if p["intensity_mode"] == "luma":
        if "gray" not in shared:
            shared["gray"] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        U = cv2.remap(shared["gray"], ug["u"], ug["v"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        Uc = cv2.remap(rgb, ug["u"], ug["v"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wts = pipe_intensity_weights(Uc, surf & ug["inside"], p)
        stats["intensity_weights"] = [round(float(x), 4) for x in wts]
        U = Uc.astype(np.float32) @ wts.astype(np.float32)
    U = np.where(ug["inside"], U, np.nan).astype(np.float32)
    gap_cols = max(2, int(round(0.012 / ug["step_m"])))
    pipe_ok = cv2.dilate(surf.astype(np.uint8), np.ones((3, 2 * gap_cols + 1), np.uint8)).astype(bool) & ug["inside"]
    if p["occluder_mask"]:
        # An object IN FRONT of the pipe (the welding torch, a cable, a hand)
        # hides the surface, and its silhouette on the pipe is a dark edge that
        # must not be traced as the junction line.  A physical gap is behind or
        # on the surface, never in front, so it is not affected.  No margin
        # around the object: a metric margin is many pixels at close range and
        # cuts the visible seam next to the torch tip (Gazebo weld-stress bags).
        front = ug["inside"] & (meas > 0) & (meas < ug["z"] - p["occluder_front_m"])
        if front.any():
            pipe_ok &= ~front
            stats["occluded_frac"] = round(float(front.sum()) / max(1, int(ug["inside"].sum())), 4)
    t3 = time.perf_counter()
    V = valley_response(U, ug["inside"], p)
    V["rgb"] = rgb          # chroma/value are sampled lazily at the few flank samples
    V["meas"] = meas        # measured depth at every unwrapped sample (radial step test)
    sigma = _noise_level(np.where(pipe_ok, U, np.nan))
    thr_abs = max(p["valley_min_abs"], p["valley_noise_k"] * sigma)
    E = (V["resp"] >= thr_abs) & (V["rel"] >= p["valley_min_rel"]) & pipe_ok
    Ed = cv2.dilate(E.astype(np.uint8), np.ones((1, 3), np.uint8))
    colcount = Ed.sum(axis=0).astype(np.float64)
    t4 = time.perf_counter()
    # candidate columns (non-maximum suppression)
    sep = int(p["candidate_separation_px"])
    order = np.argsort(-colcount)
    chosen: list[int] = []
    min_count = max(3.0, p["candidate_min_rel_count"] * float(colcount[order[0]])) if len(order) else 3.0
    for jcol in order:
        if colcount[jcol] < min_count:
            break
        if all(abs(jcol - c0) > sep for c0 in chosen):
            chosen.append(int(jcol))
        if len(chosen) >= p["candidates"]:
            break
    # a live track adds its predicted column as a candidate
    if track_uv is not None and len(track_uv):
        tu = np.nanmedian(track_uv, axis=0)
        du = np.hypot(ug["u"] - tu[0], ug["v"] - tu[1])
        du[~ug["inside"]] = np.inf
        rr, cc = np.unravel_index(int(np.argmin(du)), du.shape)
        if np.isfinite(du[rr, cc]) and du[rr, cc] < p["track_max_jump_px"] and all(abs(cc - c0) > 3 for c0 in chosen):
            chosen.append(int(cc))
    score = _path_score(V, pipe_ok, thr_abs)
    paths = trace_paths(score, chosen, int(p["path_halfband_px"])) if chosen else np.zeros((0, V["resp"].shape[0]), np.int64)
    evals = [_evaluate_candidate(c0, paths[n], V, E, pipe_ok, surf, ug, k, cyl, p, thr_abs, contra=contra)
             for n, c0 in enumerate(chosen)]
    t5 = time.perf_counter()
    if debug is not None:
        debug.update(ug=ug, U=U, V=V, E=E, pipe_ok=pipe_ok, surf=surf, evals=evals, colcount=colcount, thr_abs=thr_abs)
    stats.update(ms_unwrap=1000 * (t3 - t2), ms_valley=1000 * (t4 - t3), ms_eval=1000 * (t5 - t4),
                 thr_abs=thr_abs, sigma=sigma, unwrap_shape=list(U.shape), n_candidates=len(chosen))
    for e in evals:
        if e.get("line_uv") is not None and len(e["line_uv"]):
            e["uv_med"] = np.median(e["line_uv"], axis=0)
    live = track_uv is not None and len(track_uv) >= 2
    for e in evals:
        e["track_dist"] = track_distance(e.get("line_uv"), track_uv) if live else float("inf")
    if live:
        for e in evals:
            if e["reason"] == "border_unverifiable" and e["track_dist"] < p["track_max_jump_px"]:
                e["ok"], e["reason"] = True, "ok_border_continuation"
    passed = [e for e in evals if e["ok"]]
    # tracking relaxation: a failing candidate that continues the live track
    relaxed = False
    if not passed and live:
        near = sorted([n for n, e in enumerate(evals) if e["track_dist"] < p["track_max_jump_px"]
                       and e["reason"] in ("short_line", "fragmented_line")], key=lambda n: evals[n]["track_dist"])
        for n in near:
            e2 = _evaluate_candidate(evals[n]["col0"], paths[n], V, E, pipe_ok, surf, ug, k, cyl,
                                     p, thr_abs, relax=p["track_weak_factor"], contra=contra)
            if e2["ok"]:
                e2["track_dist"] = track_distance(e2.get("line_uv"), track_uv)
                passed.append(e2)
                relaxed = True
                break
    stats["candidates"] = [{kk: vv for kk, vv in e.items() if kk not in ("line_uv", "ring", "uv_med")} for e in evals]
    stats["ms_total"] = 1000 * (time.perf_counter() - t0)
    if not passed:
        why = evals[0]["reason"] if evals else "no_candidate"
        return Detection(False, "no_junction:" + why, cyl, stats=stats)
    passed.sort(key=lambda e: -e["score"])
    best = passed[0]
    if len(passed) > 1:
        resolved = False
        if live:
            # Evidence first; the track only breaks near-ties.  Under fast motion
            # the junction can jump tens of pixels and a weaker blur trail may
            # remain where it was: continuity must not override stronger evidence.
            cont = min(passed, key=lambda e: e.get("track_dist", float("inf")))
            if cont.get("track_dist", float("inf")) < p["track_tie_px"]:
                if cont is best or cont["score"] * p["track_tie_ratio"] >= best["score"]:
                    best, resolved = cont, True
        if not resolved:
            # no continuity to rely on: two comparable lines at different
            # stations are ambiguous
            other = passed[1] if best is passed[0] else passed[0]
            if best["score"] < p["ambiguity_ratio"] * other["score"] \
                    and abs(best["ring"]["station"] - other["ring"]["station"]) > 0.02:
                return Detection(False, "ambiguous_lines", cyl, stats=stats)
    ring = best["ring"]
    center, normal = ring["center"], ring["normal"]
    radial = -center + (center @ normal) * normal
    surface = center + cyl.radius * radial / max(np.linalg.norm(radial), 1e-9)
    stats.update(best={kk: vv for kk, vv in best.items() if kk not in ("line_uv", "ring", "uv_med")}, relaxed=relaxed)
    return Detection(True, "tracked_relaxed" if relaxed else "measured", cyl, center, normal, surface,
                     ring["station"], best["line_uv"], stats)




def detect_frame(rgb: np.ndarray, depth_m: np.ndarray, k: np.ndarray, radius: float, p: dict,
                 prior: Cylinder | None = None, track_uv: np.ndarray | None = None,
                 debug: dict | None = None, global_search: bool = True) -> Detection:
    t0 = time.perf_counter()
    grid = depth_grid(depth_m, k, p)
    t1 = time.perf_counter()
    models = fit_cylinders(grid, radius, p, prior, global_search)
    t2 = time.perf_counter()
    valid = [m for m in models if m.valid][: int(p["max_pipe_models"])]
    base = {"ms_grid": 1000 * (t1 - t0), "ms_cyl": 1000 * (t2 - t1), "n_models": len(valid)}
    if not valid:
        base["ms_total"] = 1000 * (time.perf_counter() - t0)
        return Detection(False, "cylinder:" + models[0].reason, models[0], stats=base)
    shared: dict = {}
    first = None
    for n, cyl in enumerate(valid):
        det = _detect_on_cylinder(rgb, depth_m, k, cyl, p, track_uv, debug if n == 0 else None, shared)
        det.stats.update(base)
        det.stats["model_rank"] = n
        if det.accepted:
            rf = float("nan")
            checkable = cyl.azimuth_deg >= p["radius_check_min_azimuth_deg"]
            if checkable:      # the radius is only well conditioned on a wide arc
                Pg = grid["P"].reshape(-1, 3); Ng = grid["N"].reshape(-1, 3)
                okp = np.isfinite(Pg[:, 2]) & np.isfinite(Ng[:, 0])
                rf = free_radius(Pg[okp], Ng[okp], cyl.axis, cyl.point, radius)
            det.stats["free_radius_m"] = rf
            lo, hi = p["radius_ratio_range"]
            if checkable and not (np.isfinite(rf) and lo * radius <= rf <= hi * radius):
                det = Detection(False, "no_junction:configured_radius_contradicted", cyl, stats=det.stats)
            else:
                det.stats["ms_total"] = 1000 * (time.perf_counter() - t0)
                return det
        if first is None:
            first = det
    first.stats["ms_total"] = 1000 * (time.perf_counter() - t0)
    return first


class VC1Detector:
    """Stateful wrapper: warm-started pipe, track-guided search, no published prediction."""

    def __init__(self, radius_m: float, params: dict | None = None, stateless: bool = False):
        self.p = dict(DEFAULTS)
        if params:
            self.p.update(params)
        self.radius = float(radius_m)
        self.stateless = stateless
        self.prev_cyl: Cylinder | None = None
        self.track_uv: np.ndarray | None = None
        self.track_age = 0
        self.frames_since_global = 0
        self.last_stamp_s: float | None = None

    def reset(self):
        self.prev_cyl, self.track_uv, self.track_age, self.frames_since_global = None, None, 0, 0

    def process(self, rgb: np.ndarray, depth_m: np.ndarray, k: np.ndarray,
                stamp_s: float | None = None) -> Detection:
        """One RGB-D tuple.  `stamp_s` (sensor time) is optional; when given, a
        gap longer than `reset_gap_s` or time going backwards (bag loop, sim
        reset) clears the search state first."""
        if stamp_s is not None:
            if self.last_stamp_s is not None and not (
                    0.0 <= stamp_s - self.last_stamp_s <= self.p["reset_gap_s"]):
                self.reset()
            self.last_stamp_s = float(stamp_s)
        prior = None if self.stateless else self.prev_cyl
        track = None if self.stateless else self.track_uv
        self.frames_since_global += 1
        need_global = (prior is None or self.frames_since_global >= self.p["global_every_n"])
        det = detect_frame(rgb, depth_m, k, self.radius, self.p, prior, track, global_search=need_global)
        if need_global or (det.cylinder is not None and not det.cylinder.warm):
            self.frames_since_global = 0
        if det.cylinder is not None and det.cylinder.valid:
            self.prev_cyl = det.cylinder
        else:
            self.prev_cyl = None
        if det.accepted:
            self.track_uv, self.track_age = det.line_uv, 0
        else:
            self.track_age += 1
            if self.track_age > self.p["track_max_misses"]:
                self.track_uv = None
        return det
