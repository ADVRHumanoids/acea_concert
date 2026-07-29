#!/usr/bin/env python3
"""Current-frame, color-independent continuity for the V15 detector.

The regular detector intentionally fails closed when a close or border view no
longer contains enough pipe pixels for a trustworthy full-cylinder strip.  A
visible junction can nevertheless remain measurable in RGB-D.  This module is
an independent recovery path with three simultaneous requirements:

* a dark line is measured in the *current* RGB frame near a flow prediction;
* current depth exists on the same surface on both sides of that line; and
* current depth normals vote a task-radius cylinder whose projected axis is
  perpendicular to the measured junction line.

No previous pose is published.  Previous state only bounds the image search;
every accepted result contains a newly measured line and a newly fitted local
cylinder.  The helper is V15-only and is disabled by default in the detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class FullFrameRecovery:
    accepted: bool = False
    reason: str = "not_evaluated"
    line_uv: np.ndarray | None = None
    support_line_uv: np.ndarray | None = None
    axis_camera_xyz: np.ndarray | None = None
    axis_point_camera_xyz_m: np.ndarray | None = None
    center_camera_xyz_m: np.ndarray | None = None
    surface_camera_xyz_m: np.ndarray | None = None
    radius_m: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image.astype(np.float32),
        points[:, 0].astype(np.float32).reshape(-1, 1),
        points[:, 1].astype(np.float32).reshape(-1, 1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).reshape(-1)


def _longest_run_bounds(mask: np.ndarray) -> tuple[int, int] | None:
    best_start = best_end = -1
    start = -1
    for index, value in enumerate(mask):
        if bool(value) and start < 0:
            start = index
        if start >= 0 and (not bool(value) or index == len(mask) - 1):
            end = index if bool(value) and index == len(mask) - 1 else index - 1
            if end - start > best_end - best_start:
                best_start, best_end = start, end
            start = -1
    return None if best_start < 0 else (best_start, best_end)


def _run_length(mask: np.ndarray) -> int:
    bounds = _longest_run_bounds(mask)
    return 0 if bounds is None else int(bounds[1] - bounds[0] + 1)


def _rigid_image_flow(
    previous: np.ndarray, current: np.ndarray, line: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    previous_u8 = np.clip(previous * 255.0, 0, 255).astype(np.uint8)
    current_u8 = np.clip(current * 255.0, 0, 255).astype(np.uint8)
    points = cv2.goodFeaturesToTrack(
        previous_u8,
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=7.0,
        blockSize=7,
    )
    if points is None or len(points) < 8:
        return line.copy(), {"flow_reason": "no_features", "flow_points": 0}
    next_points, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
        previous_u8,
        current_u8,
        points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or forward_ok is None:
        return line.copy(), {"flow_reason": "forward_failed", "flow_points": 0}
    back_points, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
        current_u8,
        previous_u8,
        next_points,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if back_points is None or backward_ok is None:
        return line.copy(), {"flow_reason": "backward_failed", "flow_points": 0}
    p0 = points.reshape(-1, 2)
    p1 = next_points.reshape(-1, 2)
    pb = back_points.reshape(-1, 2)
    good = (
        forward_ok.reshape(-1).astype(bool)
        & backward_ok.reshape(-1).astype(bool)
        & (np.linalg.norm(pb - p0, axis=1) <= 1.5)
    )
    p0 = p0[good]
    p1 = p1[good]
    if len(p0) < 6:
        return line.copy(), {
            "flow_reason": "too_few_forward_backward",
            "flow_points": int(len(p0)),
        }
    # OpenCV's RANSAC otherwise depends on global RNG state, which made the
    # same extracted frames choose different four-point affine hypotheses.
    cv2.setRNGSeed(170117)
    affine, inliers = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    inlier_fraction = float(inlier_count / max(1, len(p0)))
    scale = (
        float(np.hypot(affine[0, 0], affine[0, 1]))
        if affine is not None
        else float("nan")
    )
    if (
        affine is None
        or inlier_count < 8
        or inlier_fraction < 0.20
        or not (0.90 <= scale <= 1.10)
    ):
        delta = np.median(p1 - p0, axis=0)
        return line + delta[None, :], {
            "flow_reason": "median_translation_low_affine_consensus",
            "flow_points": int(len(p0)),
            "flow_inliers": inlier_count,
            "flow_inlier_fraction": inlier_fraction,
            "flow_dx_px": float(delta[0]),
            "flow_dy_px": float(delta[1]),
        }
    transformed = cv2.transform(
        line.reshape(1, -1, 2).astype(np.float32), affine
    )[0].astype(np.float64)
    return transformed, {
        "flow_reason": "affine",
        "flow_points": int(len(p0)),
        "flow_inliers": inlier_count,
        "flow_inlier_fraction": inlier_fraction,
        "flow_dx_px": float(affine[0, 2]),
        "flow_dy_px": float(affine[1, 2]),
        "flow_scale": scale,
    }


def _line_evidence(
    gray: np.ndarray,
    depth: np.ndarray,
    line: np.ndarray,
    search_radius: int,
    dark_threshold: float,
    strong_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    direction = np.asarray(line[1] - line[0], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length < 2.0:
        return None, None, {"proof_reason": "line_too_short"}
    tangent = direction / length
    normal = np.array([tangent[1], -tangent[0]], dtype=np.float64)
    samples = max(8, int(round(length)) + 1)
    alpha = np.linspace(0.0, 1.0, samples)
    base = line[0][None, :] + alpha[:, None] * direction[None, :]
    h, w = gray.shape
    rows: list[dict[str, Any]] = []
    for offset in range(-int(search_radius), int(search_radius) + 1):
        points = base + float(offset) * normal[None, :]
        inside = (
            (points[:, 0] >= 1.0)
            & (points[:, 0] < w - 1.0)
            & (points[:, 1] >= 1.0)
            & (points[:, 1] < h - 1.0)
        )
        if int(inside.sum()) < 3:
            continue
        center = _sample(gray, points)
        deficits = []
        depth_pair: tuple[np.ndarray, np.ndarray] | None = None
        for distance in (3.0, 6.0, 10.0):
            left_points = points - distance * normal[None, :]
            right_points = points + distance * normal[None, :]
            left = _sample(gray, left_points)
            right = _sample(gray, right_points)
            deficits.append(0.5 * (left + right) - center)
            if distance == 6.0:
                depth_pair = (
                    _sample(depth, left_points),
                    _sample(depth, right_points),
                )
        deficit_stack = np.stack(deficits)
        finite_deficit = np.any(np.isfinite(deficit_stack), axis=0)
        deficit = np.full(deficit_stack.shape[1], np.nan, dtype=np.float32)
        if np.any(finite_deficit):
            deficit[finite_deficit] = np.nanmax(
                deficit_stack[:, finite_deficit], axis=0
            )
        valid = inside & np.isfinite(deficit)
        dark = valid & (deficit >= float(dark_threshold))
        strong = valid & (deficit >= float(strong_threshold))
        assert depth_pair is not None
        left_depth, right_depth = depth_pair
        depth_valid = (
            inside
            & np.isfinite(left_depth)
            & np.isfinite(right_depth)
            & (left_depth > 0.05)
            & (right_depth > 0.05)
        )
        depth_same = depth_valid & (np.abs(left_depth - right_depth) <= 0.025)
        rows.append(
            {
                "offset": int(offset),
                "visible_samples": int(inside.sum()),
                "dark_count": int(dark.sum()),
                "strong_count": int(strong.sum()),
                "dark_run": _run_length(dark),
                "strong_run": _run_length(strong),
                "deficit_p90": (
                    float(np.nanpercentile(deficit[valid], 90.0))
                    if np.any(valid)
                    else 0.0
                ),
                "depth_valid_count": int(depth_valid.sum()),
                "depth_same_count": int(depth_same.sum()),
            }
        )
    if not rows:
        return None, None, {"proof_reason": "no_in_frame_samples"}
    best = max(
        rows,
        key=lambda row: (
            row["strong_run"],
            row["dark_run"],
            row["strong_count"],
            row["deficit_p90"],
            -abs(row["offset"]),
        ),
    )
    corrected = line + float(best["offset"]) * normal[None, :]
    corrected_points = corrected[0][None, :] + alpha[:, None] * (
        corrected[1] - corrected[0]
    )[None, :]
    center = _sample(gray, corrected_points)
    deficits = []
    for distance in (3.0, 6.0, 10.0):
        left = _sample(gray, corrected_points - distance * normal[None, :])
        right = _sample(gray, corrected_points + distance * normal[None, :])
        deficits.append(0.5 * (left + right) - center)
    corrected_stack = np.stack(deficits)
    finite_corrected = np.any(np.isfinite(corrected_stack), axis=0)
    corrected_deficit = np.full(
        corrected_stack.shape[1], np.nan, dtype=np.float32
    )
    if np.any(finite_corrected):
        corrected_deficit[finite_corrected] = np.nanmax(
            corrected_stack[:, finite_corrected], axis=0
        )
    corrected_inside = (
        (corrected_points[:, 0] >= 1.0)
        & (corrected_points[:, 0] < w - 1.0)
        & (corrected_points[:, 1] >= 1.0)
        & (corrected_points[:, 1] < h - 1.0)
        & np.isfinite(corrected_deficit)
    )
    bounds = _longest_run_bounds(
        corrected_inside & (corrected_deficit >= float(strong_threshold))
    ) or _longest_run_bounds(
        corrected_inside & (corrected_deficit >= float(dark_threshold))
    )
    support_line = corrected.copy()
    if bounds is not None:
        lo = max(0, int(bounds[0]) - 3)
        hi = min(samples - 1, int(bounds[1]) + 3)
        support_line = np.stack([corrected_points[lo], corrected_points[hi]])
        best["support_start"] = lo
        best["support_end"] = hi
    return corrected, support_line, {"proof_reason": "measured", **best}


def _backproject_uv(uv: np.ndarray, depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    x = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
    z = depth[y, x].astype(np.float64)
    return np.stack(
        [
            (uv[:, 0] - float(k[0, 2])) * z / float(k[0, 0]),
            (uv[:, 1] - float(k[1, 2])) * z / float(k[1, 1]),
            z,
        ],
        axis=1,
    )


def _normal_grid(
    depth: np.ndarray, k: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampled = depth[::stride, ::stride].astype(np.float32)
    smooth = cv2.GaussianBlur(sampled, (5, 5), 1.0)
    fx = float(k[0, 0]) / stride
    fy = float(k[1, 1]) / stride
    cx = float(k[0, 2]) / stride
    cy = float(k[1, 2]) / stride
    height, width = smooth.shape
    u = (np.arange(width, dtype=np.float32) - np.float32(cx)) / np.float32(fx)
    v = (np.arange(height, dtype=np.float32) - np.float32(cy)) / np.float32(fy)
    points = np.stack([u[None, :] * smooth, v[:, None] * smooth, smooth], axis=-1)
    dx = points[1:-1, 2:] - points[1:-1, :-2]
    dy = points[2:, 1:-1] - points[:-2, 1:-1]
    normals_core = np.cross(dx, dy)
    lengths = np.linalg.norm(normals_core, axis=-1)
    valid_depth = np.isfinite(sampled) & (sampled > 0.05)
    jump = 0.05 * stride
    continuous = (
        (np.abs(sampled[1:-1, 2:] - sampled[1:-1, :-2]) < jump)
        & (np.abs(sampled[2:, 1:-1] - sampled[:-2, 1:-1]) < jump)
    )
    ok_core = (
        continuous
        & (lengths > 1e-9)
        & np.isfinite(lengths)
        & valid_depth[1:-1, 1:-1]
    )
    normals = np.zeros_like(points)
    normals[1:-1, 1:-1][ok_core] = (
        normals_core[ok_core] / lengths[ok_core, None]
    )
    dot = np.einsum("ijk,ijk->ij", normals, points)
    normals[dot > 0.0] *= -1.0
    ok = np.zeros(sampled.shape, dtype=bool)
    ok[1:-1, 1:-1] = ok_core
    yy, xx = np.indices(sampled.shape)
    uv = np.stack([xx * stride, yy * stride], axis=-1).astype(np.float64)
    return points.astype(np.float64), normals.astype(np.float64), ok, uv


def _project_axis_direction(
    axis: np.ndarray, point: np.ndarray, k: np.ndarray
) -> np.ndarray | None:
    endpoints = np.stack([point - 0.15 * axis, point + 0.15 * axis])
    if np.any(endpoints[:, 2] <= 0.05):
        return None
    uv = np.stack(
        [
            float(k[0, 0]) * endpoints[:, 0] / endpoints[:, 2] + float(k[0, 2]),
            float(k[1, 1]) * endpoints[:, 1] / endpoints[:, 2] + float(k[1, 2]),
        ],
        axis=1,
    )
    direction = uv[1] - uv[0]
    norm = float(np.linalg.norm(direction))
    return None if norm < 1e-6 else direction / norm


def _axis_ray_center(
    line: np.ndarray, axis: np.ndarray, axis_point: np.ndarray, k: np.ndarray
) -> np.ndarray | None:
    midpoint = np.asarray(line, dtype=np.float64).mean(axis=0)
    ray = np.array(
        [
            (midpoint[0] - float(k[0, 2])) / float(k[0, 0]),
            (midpoint[1] - float(k[1, 2])) / float(k[1, 1]),
            1.0,
        ],
        dtype=np.float64,
    )
    ray /= np.linalg.norm(ray)
    matrix = np.stack([ray, -axis], axis=1)
    solution, *_ = np.linalg.lstsq(matrix, axis_point, rcond=None)
    center = axis_point + float(solution[1]) * axis
    if not np.isfinite(center).all() or center[2] <= 0.0:
        return None
    return center


def _line_cylinder_support(
    depth: np.ndarray,
    k: np.ndarray,
    support_line: np.ndarray,
    axis: np.ndarray,
    axis_point: np.ndarray,
    radius: float,
    params: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Require current task-radius depth immediately on both sides of the line.

    A broad local search can find the real pipe near an unrelated cloth line.
    This gate ties the visual cue to that cylinder: depth sampled at 3, 6 and
    10 pixels on each axial side must lie on the fitted surface. Both sides are
    required because a one-sided micro-sliver is not metrically actionable.
    """
    direction = np.asarray(support_line[1] - support_line[0], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length < 2.0:
        return False, {"line_cylinder_reason": "line_too_short"}
    tangent = direction / length
    pipe_direction = np.array([tangent[1], -tangent[0]], dtype=np.float64)
    samples = max(8, int(round(length)) + 1)
    alpha = np.linspace(0.0, 1.0, samples)
    base = support_line[0][None, :] + alpha[:, None] * direction[None, :]
    h, w = depth.shape
    tolerance = float(params.get("local_line_support_tol_m", 0.020))
    min_points = int(params.get("local_line_min_side_points", 6))
    min_fraction = float(params.get("local_line_min_side_fraction", 0.50))
    max_median = float(params.get("local_line_max_side_median_m", 0.015))
    diagnostics: dict[str, Any] = {}
    passed = True
    surface_ranges: list[np.ndarray] = []
    for name, offsets in (("negative", (-10.0, -6.0, -3.0)), ("positive", (3.0, 6.0, 10.0))):
        uv_groups: list[np.ndarray] = []
        for offset in offsets:
            uv = base + offset * pipe_direction[None, :]
            inside = (
                (uv[:, 0] >= 0.0)
                & (uv[:, 0] < w)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] < h)
            )
            if np.any(inside):
                uv_groups.append(uv[inside])
        uv = np.concatenate(uv_groups) if uv_groups else np.empty((0, 2))
        if len(uv):
            x = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
            y = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
            z = depth[y, x]
            valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
            uv = uv[valid]
        if len(uv):
            points = _backproject_uv(uv, depth, k)
            surface_ranges.append(np.linalg.norm(points, axis=1))
            q = points - axis_point[None, :]
            axial = q @ axis
            radial = np.sqrt(
                np.maximum(np.einsum("ij,ij->i", q, q) - axial * axial, 0.0)
            )
            residual = np.abs(radial - radius)
            median = float(np.median(residual))
            fraction = float((residual <= tolerance).mean())
        else:
            residual = np.empty(0, dtype=np.float64)
            median = float("inf")
            fraction = 0.0
        diagnostics[f"line_cylinder_{name}_points"] = int(len(residual))
        diagnostics[f"line_cylinder_{name}_median_mm"] = (
            None if not np.isfinite(median) else float(median * 1000.0)
        )
        diagnostics[f"line_cylinder_{name}_fraction"] = fraction
        side_ok = (
            len(residual) >= min_points
            and median <= max_median
            and fraction >= min_fraction
        )
        diagnostics[f"line_cylinder_{name}_passed"] = bool(side_ok)
        passed = passed and side_ok
    if surface_ranges:
        joined_ranges = np.concatenate(surface_ranges)
        diagnostics["line_cylinder_surface_range_p10_m"] = float(
            np.percentile(joined_ranges, 10.0)
        )
    else:
        diagnostics["line_cylinder_surface_range_p10_m"] = None
    diagnostics["line_cylinder_reason"] = "ok" if passed else "bilateral_surface_failed"
    return bool(passed), diagnostics


def _trusted_current_model_recovery(
    depth: np.ndarray,
    k: np.ndarray,
    support_line: np.ndarray,
    model: dict[str, Any] | None,
    params: dict[str, Any],
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    dict[str, Any],
]:
    """Corroborate a seam with the independently tracked current pipe model.

    The local crop can be underconstrained for one frame even though the main
    pipe tracker has already validated a task-radius cylinder from the full
    current depth image. That model is useful only as an independent proof: it
    is supplied by the wrapper exclusively for a current TRACK frame, and the
    measured line must still have current depth on both axial sides. A short
    micro-sliver is deliberately rejected because its axial station is not
    metrically constrained enough to replace the two-model local proof.
    """
    diagnostics: dict[str, Any] = {
        "trusted_current_model_checked": bool(model is not None),
        "trusted_current_model_passed": False,
    }
    if model is None:
        diagnostics["trusted_current_model_reason"] = "not_available"
        return None, None, None, None, diagnostics
    try:
        axis = np.asarray(model["axis"], dtype=np.float64).reshape(3)
        axis /= np.linalg.norm(axis)
        axis_point = np.asarray(model["axis_point"], dtype=np.float64).reshape(3)
        radius = float(model["radius_m"])
        source = str(model.get("fullframe_current_model_source", "unknown"))
    except Exception:
        diagnostics["trusted_current_model_reason"] = "invalid_model"
        return None, None, None, None, diagnostics
    diagnostics["trusted_current_model_source"] = source
    if (
        source != "current_track_component"
        or not np.isfinite(axis).all()
        or not np.isfinite(axis_point).all()
        or not np.isfinite(radius)
        or radius <= 1e-4
    ):
        diagnostics["trusted_current_model_reason"] = "model_not_current_or_finite"
        return None, None, None, None, diagnostics
    target_radius = float(params.get("pipe_radius_m", radius))
    radius_delta = abs(radius - target_radius)
    diagnostics["trusted_current_model_radius_m"] = radius
    diagnostics["trusted_current_model_radius_delta_mm"] = radius_delta * 1000.0
    if radius_delta > float(
        params.get("trusted_current_model_max_radius_delta_m", 0.010)
    ):
        diagnostics["trusted_current_model_reason"] = "radius_mismatch"
        return None, None, None, None, diagnostics

    direction = np.asarray(support_line[1] - support_line[0], dtype=np.float64)
    line_length = float(np.linalg.norm(direction))
    minimum_line_length = float(
        params.get("trusted_current_model_min_line_px", 70.0)
    )
    diagnostics["trusted_current_model_line_length_px"] = line_length
    diagnostics["trusted_current_model_min_line_px"] = minimum_line_length
    if line_length < minimum_line_length:
        diagnostics["trusted_current_model_reason"] = "line_too_short"
        return None, None, None, None, diagnostics
    tangent = direction / line_length
    pipe_direction = np.array([tangent[1], -tangent[0]], dtype=np.float64)
    projected_axis = _project_axis_direction(axis, axis_point, k)
    if projected_axis is None:
        diagnostics["trusted_current_model_reason"] = "axis_projection_failed"
        return None, None, None, None, diagnostics
    alignment = abs(float(np.dot(projected_axis, pipe_direction)))
    diagnostics["trusted_current_model_image_alignment"] = alignment
    minimum_alignment = float(
        np.cos(
            np.deg2rad(
                float(params.get("trusted_current_model_max_image_axis_delta_deg", 12.0))
            )
        )
    )
    if alignment < minimum_alignment:
        diagnostics["trusted_current_model_reason"] = "axis_image_mismatch"
        return None, None, None, None, diagnostics

    line_ok, line_diagnostics = _line_cylinder_support(
        depth, k, support_line, axis, axis_point, radius, params
    )
    diagnostics.update(line_diagnostics)
    if not line_ok:
        diagnostics["trusted_current_model_reason"] = "bilateral_surface_failed"
        return None, None, None, None, diagnostics

    surface_range = line_diagnostics.get("line_cylinder_surface_range_p10_m")
    if surface_range is None or not np.isfinite(float(surface_range)):
        diagnostics["trusted_current_model_reason"] = "surface_range_unavailable"
        return None, None, None, None, diagnostics
    axis_perpendicular = axis_point - float(np.dot(axis_point, axis)) * axis
    model_near_surface_range = float(np.linalg.norm(axis_perpendicular) - radius)
    foreground_excess = model_near_surface_range - float(surface_range)
    diagnostics["trusted_current_model_foreground_excess_mm"] = (
        foreground_excess * 1000.0
    )
    if foreground_excess > float(
        params.get("local_max_foreground_range_excess_m", 0.050)
    ):
        diagnostics["trusted_current_model_reason"] = "foreground_layer_mismatch"
        return None, None, None, None, diagnostics

    samples = max(8, int(round(line_length)) + 1)
    alpha = np.linspace(0.0, 1.0, samples)
    base = support_line[0][None, :] + alpha[:, None] * direction[None, :]
    h, w = depth.shape
    tolerance = float(params.get("local_line_support_tol_m", 0.020))
    side_points: dict[str, list[np.ndarray]] = {"negative": [], "positive": []}
    side_station: dict[str, list[np.ndarray]] = {"negative": [], "positive": []}
    for name, offsets in (
        ("negative", (-10.0, -6.0, -3.0)),
        ("positive", (3.0, 6.0, 10.0)),
    ):
        for offset in offsets:
            uv = base + offset * pipe_direction[None, :]
            inside = (
                (uv[:, 0] >= 0.0)
                & (uv[:, 0] < w)
                & (uv[:, 1] >= 0.0)
                & (uv[:, 1] < h)
            )
            uv = uv[inside]
            if not len(uv):
                continue
            x = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, w - 1)
            y = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, h - 1)
            z = depth[y, x]
            valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
            uv = uv[valid]
            if not len(uv):
                continue
            points = _backproject_uv(uv, depth, k)
            q = points - axis_point[None, :]
            station = q @ axis
            radial = np.sqrt(
                np.maximum(
                    np.einsum("ij,ij->i", q, q) - station * station,
                    0.0,
                )
            )
            supported = np.abs(radial - radius) <= tolerance
            if np.any(supported):
                side_points[name].append(points[supported])
                side_station[name].append(station[supported])
    if any(not side_points[name] for name in ("negative", "positive")):
        diagnostics["trusted_current_model_reason"] = "bilateral_station_unavailable"
        return None, None, None, None, diagnostics
    negative_points = np.concatenate(side_points["negative"])
    positive_points = np.concatenate(side_points["positive"])
    negative_station = np.concatenate(side_station["negative"])
    positive_station = np.concatenate(side_station["positive"])
    seam_station = 0.5 * (
        float(np.median(negative_station))
        + float(np.median(positive_station))
    )
    center = axis_point + seam_station * axis
    measured_surface = np.median(
        np.concatenate([negative_points, positive_points]), axis=0
    )
    radial = measured_surface - center
    radial -= float(np.dot(radial, axis)) * axis
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-9:
        diagnostics["trusted_current_model_reason"] = "surface_direction_degenerate"
        return None, None, None, None, diagnostics
    surface = center + radius * radial / radial_norm
    diagnostics.update(
        {
            "trusted_current_model_passed": True,
            "trusted_current_model_reason": "current_bilateral_cylinder",
            "trusted_current_model_negative_support": int(len(negative_points)),
            "trusted_current_model_positive_support": int(len(positive_points)),
        }
    )
    return axis, axis_point, center, surface, diagnostics


def _local_cylinder(
    depth: np.ndarray,
    k: np.ndarray,
    support_line: np.ndarray,
    radius: float,
    params: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    direction = support_line[1] - support_line[0]
    line_length = float(np.linalg.norm(direction))
    if line_length < 2.0:
        return None, None, None, {"local_cylinder_reason": "line_too_short"}
    tangent = direction / line_length
    pipe_direction = np.array([tangent[1], -tangent[0]], dtype=np.float64)
    midpoint = support_line.mean(axis=0)
    stride = max(1, int(params.get("normal_stride", 2)))
    points_grid, normals_grid, normal_ok, uv_grid = _normal_grid(depth, k, stride)
    delta = uv_grid - midpoint[None, None, :]
    axial_px = np.einsum("ijk,k->ij", delta, pipe_direction)
    cross_px = np.einsum("ijk,k->ij", delta, tangent)
    region = (
        normal_ok
        & (np.abs(axial_px) >= float(params.get("local_axial_min_px", 5.0)))
        & (np.abs(axial_px) <= float(params.get("local_axial_max_px", 110.0)))
        & (
            np.abs(cross_px)
            <= max(
                float(params.get("local_cross_min_px", 35.0)),
                float(params.get("local_cross_line_fraction", 0.65)) * line_length,
            )
        )
    )
    points = points_grid[region]
    normals = normals_grid[region]
    min_points = int(params.get("local_min_points", 40))
    if len(points) < min_points:
        return None, None, None, {
            "local_cylinder_points": int(len(points)),
            "local_cylinder_reason": "too_few_points",
        }
    side_depth = []
    for distance in (-10.0, -6.0, 6.0, 10.0):
        sample_point = midpoint[None, :] + distance * pipe_direction[None, :]
        side_depth.append(float(_sample(depth, sample_point)[0]))
    side_depth_array = np.asarray(side_depth, dtype=np.float64)
    side_depth_array = side_depth_array[
        np.isfinite(side_depth_array) & (side_depth_array > 0.05)
    ]
    if side_depth_array.size:
        reference_depth = float(np.median(side_depth_array))
        near = (
            np.abs(points[:, 2] - reference_depth)
            <= float(params.get("local_depth_layer_tol_m", 0.04))
        )
        points = points[near]
        normals = normals[near]
    if len(points) < min_points:
        return None, None, None, {
            "local_cylinder_points": int(len(points)),
            "local_cylinder_reason": "too_few_depth_layer",
        }
    votes = points - radius * normals
    # A fixed seed is intentional: adding the frame index made identical local
    # geometry intermittently select a perpendicular vote-line branch on tiny
    # slivers. The input points already change frame to frame; the estimator
    # itself must remain reproducible.
    rng = np.random.default_rng(int(params.get("ransac_seed", 170117)))
    iterations = int(params.get("local_ransac_iterations", 220))
    tolerance = float(params.get("local_vote_line_tol_m", 0.015))
    radial_support_tolerance = float(
        params.get("local_radial_support_tol_m", 0.020)
    )
    min_radial_support = int(params.get("local_min_radial_support", 30))
    min_alignment = float(
        np.cos(np.deg2rad(float(params.get("local_max_image_axis_delta_deg", 18.0))))
    )
    best: np.ndarray | None = None
    best_count = 0
    aligned_hypotheses = 0
    misaligned_hypotheses = 0
    refined_misaligned_hypotheses = 0
    unsupported_hypotheses = 0
    for a, b in rng.integers(0, len(votes), size=(iterations, 2)):
        span = votes[b] - votes[a]
        span_length = float(np.linalg.norm(span))
        if span_length < 0.025 or span_length > 0.60:
            continue
        axis = span / span_length
        # Competing structures in a close oblique crop can produce a denser
        # vote line perpendicular to the pipe. Selecting that line first and
        # rejecting it only after RANSAC creates avoidable three-frame holes.
        # Keep the same 18 degree safety constraint, but apply it while
        # selecting hypotheses so the best *geometrically admissible* cylinder
        # wins. The refined PCA axis is checked again below.
        projected_hypothesis = _project_axis_direction(axis, votes[a], k)
        if projected_hypothesis is None:
            continue
        hypothesis_alignment = abs(
            float(np.dot(projected_hypothesis, pipe_direction))
        )
        if hypothesis_alignment < min_alignment:
            misaligned_hypotheses += 1
            continue
        aligned_hypotheses += 1
        delta_votes = votes - votes[a]
        distance_sq = np.einsum("ij,ij->i", delta_votes, delta_votes) - (
            delta_votes @ axis
        ) ** 2
        inliers = distance_sq <= tolerance * tolerance
        count = int(inliers.sum())
        if count <= best_count:
            continue
        # A raw two-point direction can be admissible while the inlier cloud it
        # collects refines to a different, inadmissible branch. Check the PCA
        # model before promoting it to `best`; otherwise the globally largest
        # clutter branch is selected and only rejected after all alternatives
        # have already been discarded.
        hypothesis_core = votes[inliers]
        hypothesis_point = hypothesis_core.mean(axis=0)
        hypothesis_covariance = (
            (hypothesis_core - hypothesis_point).T
            @ (hypothesis_core - hypothesis_point)
        )
        _, hypothesis_vectors = np.linalg.eigh(hypothesis_covariance)
        refined_axis = hypothesis_vectors[:, -1]
        refined_axis /= np.linalg.norm(refined_axis)
        projected_refined = _project_axis_direction(
            refined_axis, hypothesis_point, k
        )
        if projected_refined is None or abs(
            float(np.dot(projected_refined, pipe_direction))
        ) < min_alignment:
            refined_misaligned_hypotheses += 1
            continue
        hypothesis_q = points - hypothesis_point[None, :]
        hypothesis_axial = hypothesis_q @ refined_axis
        hypothesis_radial = np.sqrt(
            np.maximum(
                np.einsum("ij,ij->i", hypothesis_q, hypothesis_q)
                - hypothesis_axial * hypothesis_axial,
                0.0,
            )
        )
        hypothesis_support = (
            np.abs(hypothesis_radial - radius) <= radial_support_tolerance
        )
        if int(hypothesis_support.sum()) < min_radial_support:
            unsupported_hypotheses += 1
            continue
        line_supported_hypothesis, _ = _line_cylinder_support(
            depth,
            k,
            support_line,
            refined_axis,
            hypothesis_point,
            radius,
            params,
        )
        if not line_supported_hypothesis:
            unsupported_hypotheses += 1
            continue
        best_count = count
        best = inliers
    min_votes = int(params.get("local_min_votes", 25))
    if best is None or best_count < min_votes:
        return None, None, None, {
            "local_cylinder_points": int(len(points)),
            "local_cylinder_votes": int(best_count),
            "local_cylinder_aligned_hypotheses": int(aligned_hypotheses),
            "local_cylinder_misaligned_hypotheses": int(misaligned_hypotheses),
            "local_cylinder_refined_misaligned_hypotheses": int(
                refined_misaligned_hypotheses
            ),
            "local_cylinder_unsupported_hypotheses": int(unsupported_hypotheses),
            "local_cylinder_reason": "no_vote_line",
        }
    core = votes[best]
    axis_point = core.mean(axis=0)
    covariance = (core - axis_point).T @ (core - axis_point)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis /= np.linalg.norm(axis)
    projected = _project_axis_direction(axis, axis_point, k)
    if projected is None:
        return None, None, None, {"local_cylinder_reason": "projection_failed"}
    image_alignment = abs(float(np.dot(projected, pipe_direction)))
    if image_alignment < min_alignment:
        return None, None, None, {
            "local_cylinder_points": int(len(points)),
            "local_cylinder_votes": int(best_count),
            "local_cylinder_aligned_hypotheses": int(aligned_hypotheses),
            "local_cylinder_misaligned_hypotheses": int(misaligned_hypotheses),
            "local_cylinder_refined_misaligned_hypotheses": int(
                refined_misaligned_hypotheses
            ),
            "local_cylinder_unsupported_hypotheses": int(unsupported_hypotheses),
            "local_cylinder_image_alignment": image_alignment,
            "local_cylinder_reason": "image_axis_mismatch",
        }
    q = points - axis_point[None, :]
    q_axis = q @ axis
    radial = np.sqrt(
        np.maximum(np.einsum("ij,ij->i", q, q) - q_axis * q_axis, 0.0)
    )
    residual = np.abs(radial - radius)
    support = residual <= radial_support_tolerance
    if int(support.sum()) < min_radial_support:
        return None, None, None, {
            "local_cylinder_points": int(len(points)),
            "local_cylinder_votes": int(best_count),
            "local_cylinder_support": int(support.sum()),
            "local_cylinder_reason": "radial_support_low",
        }
    line_supported, line_diagnostics = _line_cylinder_support(
        depth,
        k,
        support_line,
        axis,
        axis_point,
        radius,
        params,
    )
    if not line_supported:
        return None, None, None, {
            "local_cylinder_points": int(len(points)),
            "local_cylinder_votes": int(best_count),
            "local_cylinder_support": int(support.sum()),
            "local_cylinder_residual_med_mm": float(
                np.median(residual[support]) * 1000.0
            ),
            "local_cylinder_image_alignment": image_alignment,
            "local_cylinder_reason": "line_not_on_cylinder",
            **line_diagnostics,
        }
    foreground_diagnostics: dict[str, Any] = {}
    if bool(params.get("local_foreground_layer_guard_enabled", False)):
        surface_range_p10 = line_diagnostics.get(
            "line_cylinder_surface_range_p10_m"
        )
        if surface_range_p10 is None or not np.isfinite(
            float(surface_range_p10)
        ):
            return None, None, None, {
                "local_cylinder_reason": "foreground_range_unavailable",
                **line_diagnostics,
            }
        axis_perpendicular = axis_point - float(np.dot(axis_point, axis)) * axis
        model_near_surface_range = float(np.linalg.norm(axis_perpendicular) - radius)
        foreground_excess = model_near_surface_range - float(surface_range_p10)
        foreground_diagnostics = {
            "local_model_near_surface_range_m": model_near_surface_range,
            "local_foreground_range_excess_mm": foreground_excess * 1000.0,
        }
        if foreground_excess > float(
            params.get("local_max_foreground_range_excess_m", 0.050)
        ):
            return None, None, None, {
                "local_cylinder_reason": "foreground_layer_mismatch",
                **foreground_diagnostics,
                **line_diagnostics,
            }
    center = _axis_ray_center(support_line, axis, axis_point, k)
    if center is None:
        return None, None, None, {"local_cylinder_reason": "axis_ray_failed"}
    independent_diagnostics: dict[str, Any] = {}
    if bool(params.get("local_independent_model_check_enabled", False)):
        # RANSAC and a deterministic all-points estimate have different
        # failure modes. On a metrically useful crop they agree; on a tiny
        # grazing sliver several task-radius cylinders can explain the same
        # surface and their seam centres diverge by centimetres. Such a frame
        # is not actionable and must fail closed instead of publishing the
        # numerically strongest arbitrary branch.
        reference = np.median(points, axis=0)
        point_covariance = (points - reference).T @ (points - reference)
        _, point_axes = np.linalg.eigh(point_covariance)
        independent_candidates: list[tuple[Any, ...]] = []
        for column in range(3):
            independent_axis = point_axes[:, column]
            independent_axis /= np.linalg.norm(independent_axis)
            projected_independent = _project_axis_direction(
                independent_axis, reference, k
            )
            if projected_independent is None:
                continue
            independent_alignment = abs(
                float(np.dot(projected_independent, pipe_direction))
            )
            if independent_alignment < min_alignment:
                continue
            helper = (
                np.array([0.0, 0.0, 1.0])
                if abs(float(independent_axis[2])) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            basis_1 = np.cross(independent_axis, helper)
            basis_1 /= np.linalg.norm(basis_1)
            basis_2 = np.cross(independent_axis, basis_1)
            basis_2 /= np.linalg.norm(basis_2)
            independent_point = (
                reference
                + np.median((votes - reference) @ basis_1) * basis_1
                + np.median((votes - reference) @ basis_2) * basis_2
            )
            independent_q = points - independent_point[None, :]
            independent_axial = independent_q @ independent_axis
            independent_radial = np.sqrt(
                np.maximum(
                    np.einsum("ij,ij->i", independent_q, independent_q)
                    - independent_axial * independent_axial,
                    0.0,
                )
            )
            independent_residual = np.abs(independent_radial - radius)
            independent_support = int(
                (independent_residual <= radial_support_tolerance).sum()
            )
            if independent_support < min_radial_support:
                continue
            independent_center = _axis_ray_center(
                support_line, independent_axis, independent_point, k
            )
            if independent_center is None:
                continue
            independent_line_ok, independent_line = _line_cylinder_support(
                depth,
                k,
                support_line,
                independent_axis,
                independent_point,
                radius,
                params,
            )
            negative_median = independent_line.get(
                "line_cylinder_negative_median_mm"
            )
            positive_median = independent_line.get(
                "line_cylinder_positive_median_mm"
            )
            side_median_m = max(
                float("inf")
                if negative_median is None
                else 0.001 * float(negative_median),
                float("inf")
                if positive_median is None
                else 0.001 * float(positive_median),
            )
            independent_candidates.append(
                (
                    independent_support,
                    -float(np.median(independent_residual)),
                    independent_axis,
                    independent_point,
                    independent_center,
                    independent_alignment,
                    side_median_m,
                    bool(independent_line_ok),
                )
            )
        if not independent_candidates:
            return None, None, None, {
                "local_cylinder_reason": "independent_model_unavailable",
                **line_diagnostics,
            }
        independent = max(
            independent_candidates, key=lambda candidate: candidate[:2]
        )
        independent_axis = independent[2]
        independent_center = independent[4]
        axis_delta_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(abs(float(np.dot(axis, independent_axis))), 0.0, 1.0)
                )
            )
        )
        center_delta_m = float(np.linalg.norm(center - independent_center))
        independent_side_median_m = float(independent[6])
        independent_diagnostics = {
            "local_independent_axis_delta_deg": axis_delta_deg,
            "local_independent_center_delta_mm": center_delta_m * 1000.0,
            "local_independent_image_alignment": float(independent[5]),
            "local_independent_side_median_mm": independent_side_median_m
            * 1000.0,
            "local_independent_line_passed": bool(independent[7]),
            "local_independent_candidate_count": len(independent_candidates),
        }
        if (
            not bool(independent[7])
            or
            axis_delta_deg
            > float(params.get("local_independent_max_axis_delta_deg", 15.0))
            or center_delta_m
            > float(params.get("local_independent_max_center_delta_m", 0.025))
            or independent_side_median_m
            > float(params.get("local_independent_max_side_median_m", 0.025))
        ):
            return None, None, None, {
                "local_cylinder_reason": (
                    "independent_line_not_on_cylinder"
                    if not bool(independent[7])
                    else "local_model_ambiguous"
                ),
                **independent_diagnostics,
                **line_diagnostics,
            }
    return axis, axis_point, center, {
        "local_cylinder_points": int(len(points)),
        "local_cylinder_votes": int(best_count),
        "local_cylinder_aligned_hypotheses": int(aligned_hypotheses),
        "local_cylinder_misaligned_hypotheses": int(misaligned_hypotheses),
        "local_cylinder_refined_misaligned_hypotheses": int(
            refined_misaligned_hypotheses
        ),
        "local_cylinder_unsupported_hypotheses": int(unsupported_hypotheses),
        "local_cylinder_support": int(support.sum()),
        "local_cylinder_residual_med_mm": float(np.median(residual[support]) * 1000.0),
        "local_cylinder_image_alignment": image_alignment,
        "local_cylinder_reason": "ok",
        **foreground_diagnostics,
        **independent_diagnostics,
        **line_diagnostics,
    }


class FullFrameContinuityV15:
    """Stateful current-frame proof used only by the V15 detector."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.active = False
        self.line_uv: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None
        self.missed_frames = 0
        self.current_proof_streak = 0
        self.frame_index = 0
        self.last_reason = "not_initialized"
        self.current_pipe_model: dict[str, Any] | None = None

    def reset(self, reason: str) -> None:
        self.active = False
        self.line_uv = None
        self.missed_frames = 0
        self.current_proof_streak = 0
        self.last_reason = str(reason)

    def _gray(self, rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    @staticmethod
    def _line_innovation(
        reference: np.ndarray, measured: np.ndarray
    ) -> tuple[float, float]:
        ref_direction = np.asarray(reference[1] - reference[0], dtype=np.float64)
        measured_direction = np.asarray(measured[1] - measured[0], dtype=np.float64)
        ref_norm = float(np.linalg.norm(ref_direction))
        measured_norm = float(np.linalg.norm(measured_direction))
        if ref_norm < 1e-6 or measured_norm < 1e-6:
            return float("inf"), 180.0
        ref_tangent = ref_direction / ref_norm
        measured_tangent = measured_direction / measured_norm
        ref_normal = np.array([ref_tangent[1], -ref_tangent[0]])
        distance = abs(
            float(
                np.dot(
                    np.asarray(measured, dtype=np.float64).mean(axis=0)
                    - np.asarray(reference, dtype=np.float64).mean(axis=0),
                    ref_normal,
                )
            )
        )
        cosine = float(np.clip(abs(np.dot(ref_tangent, measured_tangent)), 0.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
        return distance, angle

    def _current_result(
        self,
        gray: np.ndarray,
        depth: np.ndarray,
        k: np.ndarray,
        predicted_line: np.ndarray,
        search_radius: int,
        source: str,
    ) -> FullFrameRecovery:
        measured, support_line, proof = _line_evidence(
            gray,
            depth,
            predicted_line,
            search_radius,
            float(self.params.get("dark_threshold", 0.025)),
            float(self.params.get("strong_threshold", 0.050)),
        )
        diagnostics = {"source": source, **proof}
        if measured is None or support_line is None:
            return FullFrameRecovery(False, str(proof.get("proof_reason")), diagnostics=diagnostics)
        if int(proof.get("strong_run", 0)) < int(self.params.get("min_strong_run_px", 4)):
            return FullFrameRecovery(False, "strong_run_low", measured, support_line, diagnostics=diagnostics)
        if int(proof.get("depth_same_count", 0)) < int(
            self.params.get("min_depth_same_count", 3)
        ):
            return FullFrameRecovery(False, "depth_support_low", measured, support_line, diagnostics=diagnostics)
        radius = float(self.params.get("pipe_radius_m", 0.10))
        local_params = dict(self.params)
        local_params["frame_index"] = int(self.frame_index)
        axis, axis_point, center, local = _local_cylinder(
            depth, k, support_line, radius, local_params
        )
        diagnostics.update(local)
        trusted_axis = trusted_axis_point = trusted_center = trusted_surface = None
        trusted_diagnostics: dict[str, Any] = {}
        if bool(self.params.get("trusted_current_model_enabled", False)):
            (
                trusted_axis,
                trusted_axis_point,
                trusted_center,
                trusted_surface,
                trusted_diagnostics,
            ) = _trusted_current_model_recovery(
                depth,
                k,
                support_line,
                self.current_pipe_model,
                local_params,
            )
            diagnostics.update(trusted_diagnostics)
        if (
            trusted_axis is not None
            and trusted_axis_point is not None
            and trusted_center is not None
            and trusted_surface is not None
        ):
            diagnostics["local_cylinder_attempt_reason"] = str(
                local.get("local_cylinder_reason", "not_available")
            )
            axis = trusted_axis
            axis_point = trusted_axis_point
            center = trusted_center
        if axis is None or axis_point is None or center is None:
            return FullFrameRecovery(
                False,
                str(local.get("local_cylinder_reason", "local_cylinder_failed")),
                measured,
                support_line,
                diagnostics=diagnostics,
            )
        if trusted_surface is not None:
            surface = trusted_surface
        else:
            radial_to_camera = -center - float(np.dot(-center, axis)) * axis
            radial_norm = float(np.linalg.norm(radial_to_camera))
            if radial_norm < 1e-8:
                return FullFrameRecovery(False, "surface_direction_degenerate", measured, support_line, diagnostics=diagnostics)
            surface = center + radius * radial_to_camera / radial_norm
        return FullFrameRecovery(
            True,
            "current_rgb_depth_cylinder",
            measured,
            support_line,
            axis,
            axis_point,
            center,
            surface,
            radius,
            diagnostics,
        )

    def evaluate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        k: np.ndarray,
        *,
        normal_line_uv: np.ndarray | None = None,
        normal_accepted: bool = False,
        normal_complete: bool = False,
        boundary_lines: list[np.ndarray] | None = None,
        current_pipe_model: dict[str, Any] | None = None,
        alternative_current_lines: list[tuple[str, np.ndarray]] | None = None,
    ) -> FullFrameRecovery:
        self.current_pipe_model = current_pipe_model
        gray = self._gray(rgb)
        flow_diagnostics: dict[str, Any] = {}
        predicted: np.ndarray | None = None
        unflowed: np.ndarray | None = None
        if self.active and self.line_uv is not None:
            unflowed = self.line_uv.copy()
            predicted = unflowed.copy()
            if self.previous_gray is not None:
                predicted, flow_diagnostics = _rigid_image_flow(
                    self.previous_gray, gray, predicted
                )

        result = FullFrameRecovery(False, "inactive")
        if normal_accepted and normal_line_uv is not None:
            # The normal pipeline already proved the frame. Re-evaluate only
            # when its 3D localization is incomplete or when it jumps away
            # from the flow-predicted, previously validated line. A large KLT
            # jump is overridden only by a complete current RGB-D+cylinder
            # proof, never by the prediction alone.
            normal_line = np.asarray(normal_line_uv, dtype=np.float64)
            outlier = False
            if predicted is not None:
                innovation_px, angle_delta_deg = self._line_innovation(
                    predicted, normal_line
                )
                flow_diagnostics["normal_line_innovation_px"] = innovation_px
                flow_diagnostics["normal_line_angle_delta_deg"] = angle_delta_deg
                outlier = (
                    innovation_px
                    > float(self.params.get("normal_line_max_innovation_px", 18.0))
                    or angle_delta_deg
                    > float(self.params.get("normal_line_max_angle_delta_deg", 15.0))
                )
            minimum_streak = int(
                self.params.get("normal_outlier_min_recovery_streak", 2)
            )
            override_outlier = bool(
                outlier
                and predicted is not None
                and self.current_proof_streak >= minimum_streak
            )
            post_recovery_check = bool(
                predicted is not None and self.current_proof_streak > 0
            )
            if (
                predicted is not None
                and (outlier or post_recovery_check)
                and not override_outlier
            ):
                # The first normal output after a current-frame recovery must
                # prove itself even when optical flow happened to move the
                # predicted line onto that output. This closes a measured case
                # where flow and the normal frontend agreed on the same false
                # dark line one frame after a correct recovery. A legitimate
                # rapid motion remains untouched because its normal line passes
                # this exact RGB/depth/task-cylinder proof at its own location.
                normal_self_check = self._current_result(
                    gray,
                    depth,
                    k,
                    normal_line,
                    int(self.params.get("normal_outlier_self_check_radius_px", 4)),
                    "normal_outlier_self_check",
                )
                self_check_visible = int(
                    normal_self_check.diagnostics.get("visible_samples", 0)
                )
                self_check_strong = int(
                    normal_self_check.diagnostics.get("strong_run", 0)
                )
                self_check_coverage = self_check_strong / max(
                    1, self_check_visible
                )
                self_check_min_coverage = float(
                    self.params.get("normal_self_check_min_coverage", 0.30)
                )
                normal_self_check.diagnostics[
                    "normal_self_check_coverage"
                ] = self_check_coverage
                if (
                    normal_self_check.accepted
                    and self_check_coverage < self_check_min_coverage
                ):
                    normal_self_check.accepted = False
                    normal_self_check.reason = "normal_self_check_coverage_low"
                flow_diagnostics["normal_outlier_self_check_passed"] = bool(
                    normal_self_check.accepted
                )
                flow_diagnostics["normal_outlier_self_check_reason"] = str(
                    normal_self_check.reason
                )
                flow_diagnostics["normal_post_recovery_self_check"] = bool(
                    post_recovery_check
                )
                flow_diagnostics["normal_self_check_coverage"] = float(
                    self_check_coverage
                )
                override_outlier = not bool(normal_self_check.accepted)
            if override_outlier and predicted is not None:
                result = self._current_result(
                    gray,
                    depth,
                    k,
                    predicted,
                    int(self.params.get("normal_outlier_search_radius_px", 40)),
                    "normal_outlier_recovery",
                )
                result.diagnostics.update(flow_diagnostics)
                if (
                    not result.accepted
                    and unflowed is not None
                    and float(np.max(np.abs(predicted - unflowed))) > 1.0
                ):
                    # Optical flow is only a search hint. On short oblique
                    # slivers it can follow the cloth/background layer and move
                    # the window away from a line that was completely proven in
                    # the preceding frame. Retry the pre-flow line, but accept
                    # it only through a fresh RGB/depth/two-cylinder proof.
                    flowed_reason = str(result.reason)
                    fallback = self._current_result(
                        gray,
                        depth,
                        k,
                        unflowed,
                        int(self.params.get("normal_outlier_search_radius_px", 40)),
                        "normal_outlier_recovery_unflowed",
                    )
                    fallback.diagnostics["flowed_recovery_reason"] = flowed_reason
                    fallback.diagnostics.update(flow_diagnostics)
                    if fallback.accepted:
                        result = fallback
                if not result.accepted and alternative_current_lines:
                    # A rapid camera sweep can leave both the local lock and its
                    # optical-flow search behind while a strong global visual
                    # candidate is already at the new junction location.  The
                    # alternative is only a search hint: it must independently
                    # pass this same current RGB/depth/cylinder proof.
                    failed_reason = str(result.reason)
                    for hint_source, hint_line in alternative_current_lines:
                        alternative = self._current_result(
                            gray,
                            depth,
                            k,
                            np.asarray(hint_line, dtype=np.float64),
                            int(
                                self.params.get(
                                    "global_candidate_search_radius_px", 10
                                )
                            ),
                            "normal_outlier_recovery",
                        )
                        alternative.diagnostics[
                            "normal_outlier_alternative_source"
                        ] = str(hint_source)
                        alternative.diagnostics[
                            "predicted_recovery_reason"
                        ] = failed_reason
                        alternative.diagnostics.update(flow_diagnostics)
                        if alternative.accepted:
                            result = alternative
                            break
                if result.accepted and result.line_uv is not None:
                    self.line_uv = result.line_uv.copy()
                    self.missed_frames = 0
                    self.current_proof_streak += 1
                else:
                    # The normal line is already known to disagree with a
                    # multi-frame sequence of complete current proofs. If its
                    # replacement is photometrically weak in this frame, fail
                    # closed: keep the validated search trajectory for the
                    # next frame and explicitly tell the wrapper to suppress
                    # the unverified normal pose.
                    self.line_uv = predicted.copy()
                    self.missed_frames += 1
                    self.current_proof_streak = max(
                        self.current_proof_streak,
                        int(
                            self.params.get(
                                "normal_outlier_min_recovery_streak", 2
                            )
                        ),
                    )
                    result.diagnostics["normal_outlier_output_blocked"] = True
            elif normal_complete:
                result = FullFrameRecovery(
                    False,
                    "normal_committed",
                    diagnostics=flow_diagnostics,
                )
            else:
                recovery_line = normal_line
                if predicted is not None:
                    normal_length = float(
                        np.linalg.norm(normal_line[1] - normal_line[0])
                    )
                    predicted_length = float(
                        np.linalg.norm(predicted[1] - predicted[0])
                    )
                    if (
                        normal_length > 2.0
                        and predicted_length > 1.25 * normal_length
                    ):
                        # A normal incomplete localization often returns only
                        # the tiny visible run. Replacing the continuity line
                        # with that segment irreversibly shrinks the next flow
                        # search and misses the seam when it grows back into
                        # view. Preserve the current measured centre/direction
                        # while retaining the validated track extent.
                        direction = (normal_line[1] - normal_line[0]) / normal_length
                        midpoint = normal_line.mean(axis=0)
                        half = 0.5 * predicted_length * direction
                        recovery_line = np.stack([midpoint - half, midpoint + half])
                result = self._current_result(
                    gray,
                    depth,
                    k,
                    recovery_line,
                    int(self.params.get("normal_incomplete_search_radius_px", 40)),
                    "normal_commit_incomplete",
                )
                if recovery_line is not normal_line:
                    result.diagnostics["normal_incomplete_line_extended"] = True
                    result.diagnostics["normal_incomplete_original_length_px"] = (
                        normal_length
                    )
                    result.diagnostics["normal_incomplete_tracking_length_px"] = (
                        predicted_length
                    )
            self.active = True
            if not override_outlier:
                if result.accepted and result.line_uv is not None:
                    self.line_uv = result.line_uv.copy()
                    self.current_proof_streak = 1
                else:
                    self.line_uv = normal_line.copy()
                    self.current_proof_streak = 0
                self.missed_frames = 0
        elif predicted is not None:
            result = self._current_result(
                gray,
                depth,
                k,
                predicted,
                int(self.params.get("search_radius_px", 70)),
                "tracked_recovery",
            )
            result.diagnostics.update(flow_diagnostics)
            if (
                not result.accepted
                and unflowed is not None
                and float(np.max(np.abs(predicted - unflowed))) > 1.0
            ):
                flowed_reason = str(result.reason)
                fallback = self._current_result(
                    gray,
                    depth,
                    k,
                    unflowed,
                    int(self.params.get("search_radius_px", 70)),
                    "tracked_recovery_unflowed",
                )
                fallback.diagnostics["flowed_recovery_reason"] = flowed_reason
                fallback.diagnostics.update(flow_diagnostics)
                if fallback.accepted:
                    result = fallback
            if result.accepted and result.line_uv is not None:
                self.line_uv = result.line_uv.copy()
                self.missed_frames = 0
                self.current_proof_streak += 1
            else:
                self.line_uv = predicted.copy()
                self.missed_frames += 1
                self.current_proof_streak = 0
        elif boundary_lines:
            candidates: list[FullFrameRecovery] = []
            for line in boundary_lines:
                candidate = self._current_result(
                    gray,
                    depth,
                    k,
                    np.asarray(line, dtype=np.float64),
                    int(self.params.get("fresh_border_search_radius_px", 24)),
                    "fresh_border",
                )
                if candidate.line_uv is not None:
                    height, width = gray.shape
                    midpoint_x = float(candidate.line_uv[:, 0].mean())
                    border_distance = min(midpoint_x, float(width - 1) - midpoint_x)
                    candidate.diagnostics["border_distance_px"] = border_distance
                    if border_distance > float(
                        self.params.get("fresh_border_max_distance_px", 20)
                    ):
                        candidate.accepted = False
                        candidate.reason = "fresh_candidate_not_at_border"
                candidates.append(candidate)
            accepted = [candidate for candidate in candidates if candidate.accepted]
            if accepted:
                result = max(
                    accepted,
                    key=lambda candidate: (
                        int(candidate.diagnostics.get("strong_run", 0)),
                        int(candidate.diagnostics.get("local_cylinder_votes", 0)),
                    ),
                )
                self.active = True
                self.line_uv = np.asarray(result.line_uv, dtype=np.float64).copy()
                self.missed_frames = 0
                self.current_proof_streak = 1
            elif candidates:
                result = max(
                    candidates,
                    key=lambda candidate: int(candidate.diagnostics.get("strong_run", 0)),
                )

        max_missed = int(self.params.get("max_missed_frames", 30))
        if self.active and self.missed_frames > max_missed:
            self.reset("missed_too_long")
        self.previous_gray = gray.copy()
        self.frame_index += 1
        self.last_reason = result.reason
        return result
