#!/usr/bin/env python3
"""ROS2 RGB-D pipe-junction detector for the ACEA Module 2 demo.

This node is the first online version of the offline ACEA RGB-D pipeline. It
subscribes to RGB, depth, and camera info, then runs a deterministic classical
pipeline:

RGB-D pair -> depth pipe tracker -> pipe-aligned strip -> seam score ->
temporal confirmation -> coarse camera-frame seam estimate.

It publishes JSON on std_msgs/String and RGB debug overlays on sensor_msgs/Image.
It does not use CartesIO, machine learning, custom ROS messages, or robot
commands.
"""

from __future__ import annotations

import json
import math
import os
import socket
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PilImage
from PIL import ImageDraw
from scipy import ndimage as scipy_ndimage

try:
    import cv2
    CV2_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - OpenCV is optional at runtime
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Bool, String
    from geometry_msgs.msg import PoseStamped
    from visualization_msgs.msg import Marker, MarkerArray
    RCLPY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - allows offline import/test without ROS
    RCLPY_IMPORT_ERROR = exc
    rclpy = None  # type: ignore

    class _StubParam:
        def __init__(self, value: Any) -> None:
            self.value = value

    class _StubLogger:
        def info(self, *a: Any, **k: Any) -> None: ...
        def warning(self, *a: Any, **k: Any) -> None: ...
        def warn(self, *a: Any, **k: Any) -> None: ...
        def error(self, *a: Any, **k: Any) -> None: ...

    class Node:  # type: ignore
        """Minimal offline stub: lets the node be instantiated WITHOUT ROS so the
        detector can be unit-tested on synthetic frames (declare/get_parameter
        return the declared defaults; pub/sub/timer are no-ops)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._declared: dict[str, Any] = {}

        def declare_parameter(self, name: str, default: Any) -> None:
            self._declared[name] = default

        def get_parameter(self, name: str) -> "_StubParam":
            return _StubParam(self._declared[name])

        def create_subscription(self, *a: Any, **k: Any) -> None: ...
        def create_publisher(self, *a: Any, **k: Any) -> "Node": return self
        def create_timer(self, *a: Any, **k: Any) -> None: ...
        def destroy_subscription(self, *a: Any, **k: Any) -> None: ...
        def publish(self, *a: Any, **k: Any) -> None: ...
        def get_logger(self) -> "_StubLogger": return _StubLogger()

    class ExternalShutdownException(Exception):  # type: ignore
        pass

    class _StubAny:
        def __call__(self, *a: Any, **k: Any) -> None: return None
        def __getattr__(self, _name: str) -> int: return 0

    HistoryPolicy = ReliabilityPolicy = QoSProfile = _StubAny()  # type: ignore
    Bool = CameraInfo = Image = String = PoseStamped = Marker = MarkerArray = None  # type: ignore

# Variant A deterministic RGB-only seam detector (black top-hat + vertical-run
# coherence, no depth). Imported from the same scripts dir (project convention).
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from acea_seam_detector import detect_seam as _variant_a_detect_seam
except Exception:  # pragma: no cover - Variant A is optional
    _variant_a_detect_seam = None

# Shared operational guards: single-instance lock (kills the "two detectors
# publishing alternating frame counts" failure) + a host:pid instance id stamped
# into status so duplicates are obvious in `ros2 topic echo`. ROS-free helper, so
# it imports in the offline path too; fall back to no-ops if it is ever missing.
try:
    from acea_detection_runtime import (
        DuplicateInstanceError,
        SingleInstanceLock,
        count_named_nodes,
        instance_id,
    )
except Exception:  # pragma: no cover - degrade gracefully if helper is absent
    class DuplicateInstanceError(RuntimeError):  # type: ignore
        pass

    class SingleInstanceLock:  # type: ignore
        def __init__(self, *_a: Any, **_k: Any) -> None:
            self.acquired = True
            self.holder = ""

        def release(self) -> None: ...

    def count_named_nodes(_node: Any) -> int:  # type: ignore
        return 1

    def instance_id() -> str:  # type: ignore
        return f"{socket.gethostname()}:{os.getpid()}"

# Weld-seam / gap-plane frame geometry (pure NumPy, ROS-free, unit-tested in
# acea_alignment/weld_seam.py). Turns (pipe axis, seam surface point) into a
# PoseStamped-ready frame for the IK / welding-tracking side (Arturo's request).
try:
    from acea_alignment.weld_seam import WeldSeamFrame, seam_frame_from_axis_and_surface
except Exception:  # pragma: no cover - geometry helper is optional at import time
    WeldSeamFrame = None  # type: ignore
    seam_frame_from_axis_and_surface = None  # type: ignore

try:
    from acea_alignment.pipe_pose import fit_pipe_pose as _fit_pipe_pose
except Exception:  # pragma: no cover - robust component validation is optional
    _fit_pipe_pose = None  # type: ignore

# RViz marker constants (visualization_msgs/Marker). Defined as literals so the
# values are available even when the message class is unavailable offline
# (Marker == None). They match the ROS message definition.
_MARKER_ADD = 0
_MARKER_DELETEALL = 3
_MARKER_CYLINDER = 3

# Fixed name for the single-instance lock (independent of the runtime ROS node
# name, which the launch file may remap). All copies of this detector contend
# for the same lock on a host.
_DETECTOR_LOCK_NAME = "acea_pipe_junction_node"


def _stamp_sec(msg: Image | CameraInfo) -> float:
    return float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector.astype(np.float64)
    return vector.astype(np.float64) / norm


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _round_list(values: np.ndarray | list[float] | None, digits: int = 6) -> list[float] | None:
    if values is None:
        return None
    return [_round(float(v), digits) for v in values]


def _pca(points: np.ndarray) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 3:
        raise ValueError("Need at least three points for PCA")
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        raise ValueError("PCA covariance contains non-finite values")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    return {
        "centroid": centroid,
        "eigenvalues": eigenvalues,
        "direction": _normalize(eigenvectors[:, 0]),
    }


def _cylinder_consensus_axis_fit(
    points: np.ndarray,
    *,
    max_iterations: int,
    radius_tolerance_m: float,
    min_inliers: int,
    min_inlier_fraction: float,
) -> dict[str, Any]:
    """Robust pipe-axis estimate inspired by cylinder RANSAC/consensus fitting.

    The visible pipe surface is an elongated cylinder patch, not a thin 3D line.
    Instead of requiring points to lie on the axis, estimate the dominant axis
    with PCA, keep points whose radial distance to that axis is consistent with
    one cylinder surface, and refit. This gives a deterministic, lightweight
    cylinder-consensus fit without adding a PCL dependency to the Python node.
    """

    clean = np.asarray(points, dtype=np.float64)
    clean = clean[np.all(np.isfinite(clean), axis=1)]
    if clean.shape[0] < 3:
        raise ValueError("Need at least three points for pipe-axis fitting")

    fit = _pca(clean)
    inliers = np.ones(clean.shape[0], dtype=bool)
    method = "pca"
    radius_m = None
    residual_m = None

    for _ in range(max(0, int(max_iterations))):
        centroid = fit["centroid"]
        direction = _normalize(fit["direction"])
        offsets = clean - centroid
        axial = offsets @ direction
        closest = centroid[None, :] + axial[:, None] * direction[None, :]
        radial = np.linalg.norm(clean - closest, axis=1)
        finite = np.isfinite(radial)
        if finite.sum() < min_inliers:
            break

        median_radius = float(np.median(radial[finite]))
        mad = float(np.median(np.abs(radial[finite] - median_radius)))
        robust_sigma = 1.4826 * mad
        tolerance = max(float(radius_tolerance_m), 3.0 * robust_sigma)
        candidate = finite & (np.abs(radial - median_radius) <= tolerance)

        if candidate.sum() >= min_inliers:
            axial_candidate = axial[candidate]
            lo, hi = np.percentile(axial_candidate, [1.0, 99.0])
            candidate &= (axial >= lo) & (axial <= hi)

        inlier_fraction = float(candidate.mean()) if candidate.size else 0.0
        if candidate.sum() < min_inliers or inlier_fraction < float(min_inlier_fraction):
            break

        inliers = candidate
        fit = _pca(clean[inliers])
        method = "cylinder_consensus_pca"
        radius_m = median_radius
        residual_m = float(np.median(np.abs(radial[inliers] - median_radius)))

    if radius_m is None or residual_m is None:
        centroid = fit["centroid"]
        direction = _normalize(fit["direction"])
        offsets = clean - centroid
        axial = offsets @ direction
        closest = centroid[None, :] + axial[:, None] * direction[None, :]
        radial = np.linalg.norm(clean - closest, axis=1)
        radius_m = float(np.median(radial))
        residual_m = float(np.median(np.abs(radial - radius_m)))

    return {
        "centroid": fit["centroid"],
        "eigenvalues": fit["eigenvalues"],
        "direction": _normalize(fit["direction"]),
        "points": clean,
        "inlier_mask": inliers,
        "inlier_count": int(inliers.sum()),
        "inlier_fraction": float(inliers.mean()) if inliers.size else 0.0,
        "radius_m": float(radius_m),
        "residual_m": float(residual_m),
        "method": method,
    }


def _depth_threshold_kmeans(depth: np.ndarray, valid: np.ndarray, sample_stride: int) -> dict[str, float | int]:
    sample = depth[::sample_stride, ::sample_stride]
    sample_valid = valid[::sample_stride, ::sample_stride]
    values = sample[sample_valid].astype(np.float64)
    if values.size < 32:
        raise ValueError("Not enough valid depth pixels for foreground selection")

    centers = np.percentile(values, [10.0, 90.0]).astype(np.float64)
    for _ in range(32):
        threshold = float(0.5 * (centers[0] + centers[1]))
        low = values[values <= threshold]
        high = values[values > threshold]
        if low.size == 0 or high.size == 0:
            break
        new_centers = np.array([low.mean(), high.mean()], dtype=np.float64)
        if np.linalg.norm(new_centers - centers) < 1e-9:
            centers = new_centers
            break
        centers = new_centers

    centers = np.sort(centers)
    threshold = float(0.5 * (centers[0] + centers[1]))
    return {
        "near_center_m": float(centers[0]),
        "far_center_m": float(centers[1]),
        "threshold_m": threshold,
        "sample_count": int(values.size),
    }


def _backproject(depth: np.ndarray, xs: np.ndarray, ys: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if not np.isfinite(fx) or not np.isfinite(fy) or abs(fx) < 1e-9 or abs(fy) < 1e-9:
        raise ValueError(f"Invalid camera intrinsics for backprojection: fx={fx}, fy={fy}")
    z = depth[ys, xs].astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def _project(points_camera: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    z = np.maximum(points_camera[:, 2], 1e-9)
    u = fx * points_camera[:, 0] / z + cx
    v = fy * points_camera[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def _line_box_segment(
    center_uv: np.ndarray,
    direction_uv: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    cx, cy = center_uv
    dx, dy = direction_uv
    points: list[tuple[float, float]] = []

    if abs(dx) > 1e-9:
        for x in (0.0, float(width - 1)):
            t = (x - cx) / dx
            y = cy + t * dy
            if 0.0 <= y <= height - 1:
                points.append((x, y))

    if abs(dy) > 1e-9:
        for y in (0.0, float(height - 1)):
            t = (y - cy) / dy
            x = cx + t * dx
            if 0.0 <= x <= width - 1:
                points.append((x, y))

    if len(points) < 2:
        return None

    best_pair = (points[0], points[1])
    best_dist = -1.0
    for i, p0 in enumerate(points):
        for p1 in points[i + 1:]:
            dist = float(np.linalg.norm(np.array(p0) - np.array(p1)))
            if dist > best_dist:
                best_dist = dist
                best_pair = (p0, p1)
    return best_pair


def _step_edge_profile(profile: np.ndarray, data_ok: np.ndarray, window_px: int, gap_px: int) -> np.ndarray:
    """|mean(right window) - mean(left window)| of the masked column profile.

    A socket/collar junction on a real pipe is a broad luminance step between
    two sustained plateaus (the two pipe segments have slightly different
    tone). Unlike the thin dark seam line the black-tophat needs, this step
    survives motion blur and stays visible on bright glossy PVC where the seam
    has almost no dark core. Columns where data_ok is False contribute nothing;
    a column needs at least half a window of real data on BOTH sides.
    """
    n = int(profile.size)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    ok = data_ok & np.isfinite(profile)
    vals = np.where(ok, profile, 0.0)
    win = max(4, int(window_px))
    gap = max(0, int(gap_px))
    csum = np.concatenate([[0.0], np.cumsum(vals)])
    ccnt = np.concatenate([[0.0], np.cumsum(ok.astype(np.float64))])
    x = np.arange(n)
    l0 = np.clip(x - gap - win, 0, n)
    l1 = np.clip(x - gap, 0, n)
    r0 = np.clip(x + gap + 1, 0, n)
    r1 = np.clip(x + gap + 1 + win, 0, n)
    lcnt = ccnt[l1] - ccnt[l0]
    rcnt = ccnt[r1] - ccnt[r0]
    lmean = (csum[l1] - csum[l0]) / np.maximum(lcnt, 1e-9)
    rmean = (csum[r1] - csum[r0]) / np.maximum(rcnt, 1e-9)
    enough = (lcnt >= 0.5 * win) & (rcnt >= 0.5 * win)
    out[enough] = np.abs(rmean - lmean)[enough]
    return out


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    half = window // 2
    n = values.size
    output = np.empty_like(values, dtype=np.float64)
    if n >= window:
        # Full windows vectorized; only the shrinking edge windows loop.
        core = np.lib.stride_tricks.sliding_window_view(values, window)
        output[half:n - half] = np.median(core, axis=1)
        edges = list(range(half)) + list(range(n - half, n))
    else:
        edges = list(range(n))
    for i in edges:
        output[i] = float(np.median(values[max(0, i - half):min(n, i + half + 1)]))
    return output


def _inverse_rotate_uv(points_uv: np.ndarray, image_size_wh: tuple[int, int], angle_deg: float) -> np.ndarray:
    width, height = image_size_wh
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    shifted = points_uv.astype(np.float64) - np.array([[cx, cy]], dtype=np.float64)
    x = shifted[:, 0]
    y = shifted[:, 1]
    # Inverse of PIL Image.rotate(angle_deg) (CCW). The strip is built with
    # rgb_image.rotate(angle_deg); to map a strip/rotated pixel back to the
    # original image we must rotate by -angle_deg. The previous signs rotated the
    # WRONG way: correct at angle=0 but the error grew ~lever*sin(2*angle) with
    # camera roll (~67px at ~12deg), corrupting both the published junction line
    # AND the 3D gap localization (_localize_confirmed_seam -> /gap/pose_robot).
    original_x = cos_t * x - sin_t * y + cx
    original_y = sin_t * x + cos_t * y + cy
    return np.stack([original_x, original_y], axis=1)


def _depth_visual(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < 100.0)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [1.0, 99.0])
        image = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        image = np.zeros_like(depth, dtype=np.float32)
    gray = (image * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


@dataclass
class TrackerResult:
    pipe_mask: np.ndarray
    pipe_pixels: int
    pipe_fraction: float
    bbox_uv: list[int]
    image_centroid_uv: np.ndarray
    image_direction_uv: np.ndarray
    image_axis_angle_deg: float
    image_line_segment_uv: tuple[tuple[float, float], tuple[float, float]] | None
    centroid_xyz_m: np.ndarray
    pipe_axis_xyz: np.ndarray
    stand_off_m: float
    lateral_offset_m: float
    vertical_offset_m: float
    yaw_error_deg: float
    pipe_pose_fit_method: str
    pipe_pose_inlier_count: int
    pipe_pose_inlier_fraction: float
    pipe_pose_radius_m: float
    pipe_pose_residual_m: float
    threshold_info: dict[str, float | int]


@dataclass
class TemporalChangeResult:
    score: float
    dark_delta: float
    z_score: float
    accepted: bool
    reference_ready: bool
    reference_frame_count: int
    reason: str


@dataclass
class SeamResult:
    candidate_x_strip_px: int
    candidate_x_rotated_px: int
    classical_candidate_x_strip_px: int
    classical_candidate_contrast: float
    classical_candidate_z_score: float
    candidate_contrast: float
    candidate_z_score: float
    confidence: float
    visual_frontend: str
    visual_frontend_accepted: bool
    rgb_dark_score: float
    rgb_local_contrast_score: float
    rgb_dark_threshold_used: float
    rgb_dark_accepted: bool
    depth_gap_score: float
    depth_gap_accepted: bool
    depth_gap_raw_accepted: bool
    depth_gap_score_plausible: bool
    depth_gap_depth_jump_m: float
    depth_gap_coverage_drop: float
    negative_gate_reason: str
    local_candidate_accepted: bool
    temporal_change_score: float
    temporal_change_dark_delta: float
    temporal_change_z_score: float
    temporal_change_accepted: bool
    temporal_change_gate_enabled: bool
    temporal_reference_ready: bool
    temporal_reference_frame_count: int
    temporal_change_reason: str
    junction_acceptance_mode: str
    variant_a_orientation_deg: float
    variant_a_classical_fallback_used: bool
    rgb_temporal_accepted: bool
    rgb_temporal_score: float
    rgb_vertical_edge_score: float
    rgb_luminance_edge_score: float
    rgb_chromatic_edge_score: float
    rgb_edge_chromaticity_ratio: float
    rgb_shadow_like_score: float
    rgb_shadow_like_rejected: bool
    rgb_surface_continuity_score: float
    rgb_surface_continuity_rejected: bool
    rgb_low_contrast_rejected: bool
    rgb_temporal_candidate_reject_reason: str
    rgb_line_support_fraction: float
    rgb_line_width_px: int
    rgb_track_id: int
    rgb_track_streak: int
    rgb_track_missed_frames: int
    rgb_candidate_velocity_px_per_frame: float
    klt_status: str
    klt_points: int
    klt_dx_px: float
    klt_predicted_x_strip_px: float | None
    pipe_end_rejected: bool
    pipe_support_left_cols: int
    pipe_support_right_cols: int
    pipe_support_left_coverage: float
    pipe_support_right_coverage: float
    accepted: bool
    edge_margin_px: int
    crop_xyxy: list[int]
    strip_size_wh: list[int]
    strip_mask: np.ndarray
    rotated_mask: np.ndarray
    rotation_deg: float
    strip_profile: np.ndarray
    strip_profile_valid: np.ndarray
    appearance_veto: bool = False
    appearance_ncc: float = 1.0
    # Collar/step-edge cue at the FINAL candidate column (see _step_edge_profile).
    # Raw per-frame measurement before output smoothing (-1 = not set).
    candidate_x_raw_strip_px: int = -1
    candidate_step_abs: float = 0.0
    candidate_step_z: float = 0.0
    # max(dark z, step z): the "some seam-like evidence here" scalar the
    # junction lock uses to keep tracking through motion blur.
    candidate_evidence_z: float = 0.0
    step_fallback_used: bool = False


@dataclass
class LocalizationResult:
    visible_surface_center_xyz_m: np.ndarray | None
    pipe_center_estimate_xyz_m: np.ndarray | None
    support_pixel_count: int
    center_method: str = ""


class OnlinePipeJunctionDetector:
    """Classical RGB-D detector with temporal confirmation state."""

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.state = "SCAN"
        self.candidate_streak = 0
        self.processed_frame_count = 0
        self.confirmed_frame_count: int | None = None
        self.previous_geometry: dict[str, float | None] | None = None
        self.previous_candidate_x: int | None = None
        self.temporal_reference_profile: np.ndarray | None = None
        self.temporal_reference_valid: np.ndarray | None = None
        self.temporal_reference_frame_count = 0
        self.rgb_track_id = 0
        self.rgb_track_x: int | None = None
        self.rgb_track_streak = 0
        self.rgb_track_missed_frames = 0
        self.junction_lock_active = False
        self.pipe_component_selection_info: dict[str, Any] = {}
        self.pipe_mask_reproject_info: dict[str, Any] = {}
        self._reproject_surface_points: np.ndarray | None = None
        self._reproject_ray_cache: tuple | None = None
        self.pipe_component_selected_model: dict[str, Any] | None = None
        self.pipe_component_selected_image_model: dict[str, Any] | None = None
        self.pipe_lock_model: dict[str, Any] | None = None
        self.pipe_image_lock_model: dict[str, Any] | None = None
        self.pipe_lock_missed_frames = 0
        self.pipe_lock_source = "none"
        self.pipe_image_lock_source = "none"
        self._pipe_lock_missed_frame = -1
        self.junction_lock_x: float | None = None
        self.junction_last_valid_x: float | None = None
        self.junction_last_valid_frame: int | None = None
        self.junction_lock_velocity_px = 0.0
        self._junction_lag_ema = 0.0
        self._junction_center_state: np.ndarray | None = None
        self._junction_center_lag = np.zeros(3, dtype=np.float64)
        # Color prior + model smoothing + impostor-escape state.
        self._warm_mask_current: np.ndarray | None = None
        self._warm_scene_ok = False
        self._reproject_axial_extent: tuple[float, float] | None = None
        self._pipe_mask_col_extent: tuple[int, int, int] | None = None
        # Confirmed pipe-END memory per side: [s_axial, frames_since_seen].
        # A socket end seen clearly INSIDE the frame stays banned as junction
        # even after camera motion pushes it over the image border (a single
        # frame cannot distinguish end-at-border from pipe-continuing).
        self._pipe_end_memory: dict[str, list[float]] = {}
        # Streak of consecutive frames where a side failed the full support
        # requirement with a stable s: {side: [s_end, count]}. Memory is
        # only created after pipe_end_memory_confirm_frames of these — a
        # single spurious short-extent frame must not ban a region.
        self._pipe_end_streak: dict[str, list[float]] = {}
        self._pipe_end_prev_components: tuple[np.ndarray, np.ndarray] | None = None
        self._pipe_model_filt: dict[str, np.ndarray] | None = None
        self._low_warm_streak = 0
        self.pipe_mask_warm_fraction: float | None = None
        self.pipe_mask_normal_fraction: float | None = None
        self._normal_image_cache: tuple | None = None
        # Color-independent cylinder guard state (active only when the warm
        # prior has no scene evidence): azimuthal normal-rotation features of
        # the current model, acquire streak toward the metric lock, and the
        # impostor fail streak that releases a lock held on non-cylindrical
        # surfaces (cloth/wall force-fitted as a tangent cylinder).
        self._coloroff_guard_frame_active = False
        self._coloroff_cyl_features: dict[str, float] | None = None
        self._coloroff_cyl_ok = True
        self._coloroff_cyl_reason = "not_run"
        self._coloroff_acquire_streak = 0
        self._coloroff_acquire_prev: tuple[np.ndarray, np.ndarray] | None = None
        self._coloroff_fail_streak = 0
        self._coloroff_hold_count = 0
        # v7 acquisition stickiness: the last cylinder-consistent model seen in
        # ACQUIRE, reprojected on the next frame so the search does not flip
        # back to the cloth between non-deterministic RANSAC hits.
        self._coloroff_provisional_model: dict[str, Any] | None = None
        self._coloroff_provisional_age = 0
        # v8: True whenever a valid pipe may be shown/published. Color-ON keeps
        # v5 behaviour (always True). Color-less ACQUIRE sets it False until the
        # first cylinder-consistent model is found, so the overlay does not draw
        # the default/cloth axis that flickers before the pipe is acquired.
        self._coloroff_pipe_visible = True
        self._pipe_lock_update_accepted = 0
        self._pipe_lock_update_held = 0
        self._pipe_lock_update_count_frame = -1
        # Explicit tracker FSM state (single source of truth for the reported
        # state AND for the model-update gating). Transitions are driven by
        # _advance_pipe_tracker_state; ACQUIRE searches globally, TRACK updates
        # only on a passing cylinder gate, HOLD freezes axis/point/radius while
        # projecting the held cylinder, LOST resets after persistent failure.
        self._pipe_state = "ACQUIRE"
        self._pipe_tracker_hold_frame = False
        self.junction_lock_streak = 0
        self.junction_lock_start_frame: int | None = None
        self.junction_lock_missed_frames = 0
        self.junction_lock_confidence = 0.0
        self.junction_lock_source = "none"
        self.klt_prev_gray: np.ndarray | None = None
        self.klt_prev_points: np.ndarray | None = None
        self.klt_prev_lock_x: float | None = None
        self.klt_last_dx_px = 0.0
        self.klt_last_points = 0
        self.klt_status = "not_initialized"
        # Optional appearance-identity veto (off by default): a small NCC template of
        # the locked junction's local strip patch, used only to REJECT a candidate
        # that hopped to a differently-looking line. Purely subtractive and cleared
        # when the lock is lost, so it can never force a hold/phantom.
        self.appearance_template: np.ndarray | None = None
        self._last_appearance_patch: np.ndarray | None = None

    def process(self, rgb: np.ndarray, depth: np.ndarray, k: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        self.processed_frame_count += 1
        self._last_depth_image = depth
        self._warm_mask_current = self._compute_warm_mask(rgb)
        # The prior is only INFORMATIVE when the scene contains warm evidence
        # at all: on a gray/synthetic pipe the warm mask is empty and every
        # warm gate must stand down (graceful fallback to depth-only).
        self._warm_scene_ok = bool(
            self._warm_mask_current is not None
            and int(self._warm_mask_current[::4, ::4].sum()) * 16 >= int(self.params["min_pipe_pixels"])
        )
        try:
            tracker = self._track_pipe(depth, k)
        except ValueError as exc:
            self._mark_pipe_lock_missed(str(exc))
            if self.pipe_lock_model is None:
                self._release_junction_lock("pipe_not_valid")
            status = self._no_pipe_status(str(exc))
            return status, rgb.copy(), _depth_visual(depth)
        if self.pipe_lock_model is not None and self.pipe_component_selected_model is not None:
            self._update_pipe_lock(tracker)
        if self.pipe_image_lock_model is not None and self.pipe_component_selected_image_model is not None:
            self._update_pipe_image_lock()
        try:
            seam = self._detect_seam(rgb, depth, tracker)
        except ValueError as exc:
            # Degenerate strip (e.g. the camera panned off the pipe and the
            # mask collapsed): report a clean no-detection frame instead of
            # crashing the node mid-experiment. Locks stay untouched for one
            # frame; the miss/escape logic handles persistent cases.
            status = self._no_pipe_status(f"seam_stage_failed:{exc}")
            return status, rgb.copy(), _depth_visual(depth)
        state_info = self._update_state(tracker, seam)
        self._update_temporal_reference(seam)
        localization = None
        if self.state in ("CONFIRMED", "STOP_AND_LOCALIZE"):
            localization = self._localize_confirmed_seam(depth, k, tracker, seam)

        rgb_overlay = self._draw_overlay(rgb, tracker, seam, localization, state_info)
        depth_overlay = self._draw_overlay(_depth_visual(depth), tracker, seam, localization, state_info)
        status = self._status_dict(tracker, seam, localization, state_info)
        return status, rgb_overlay, depth_overlay

    @staticmethod
    def _axis_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
        aa = _normalize(np.asarray(a, dtype=np.float64))
        bb = _normalize(np.asarray(b, dtype=np.float64))
        dot = abs(float(np.clip(np.dot(aa, bb), -1.0, 1.0)))
        return math.degrees(math.acos(dot))

    def _no_pipe_status(self, reason: str) -> dict[str, Any]:
        return {
            "state": "SCAN",
            "detected": False,
            "accepted": False,
            "detector_accepted": False,
            "confidence": None,
            "candidate_x_strip_px": None,
            "candidate_x_image_px": None,
            "reason": f"pipe_not_valid:{reason}",
            "processed_frame_count": int(self.processed_frame_count),
            "pipe_component_selection_method": self.pipe_component_selection_info.get("method"),
            "pipe_component_count": self.pipe_component_selection_info.get("component_count"),
            "pipe_component_candidate_count": self.pipe_component_selection_info.get("candidate_count"),
            "pipe_component_rejected_by_shape": self.pipe_component_selection_info.get("rejected_by_shape"),
            "pipe_component_cylinder_evaluated": self.pipe_component_selection_info.get("cylinder_evaluated"),
            "pipe_component_cylinder_valid": self.pipe_component_selection_info.get("cylinder_valid"),
            "pipe_component_selected_label": self.pipe_component_selection_info.get("selected_label"),
            "pipe_component_fallback_label": self.pipe_component_selection_info.get("fallback_label"),
            "pipe_component_band_valid": self.pipe_component_selection_info.get("band_valid"),
            "pipe_component_band_score": self.pipe_component_selection_info.get("band_score"),
            "pipe_component_band_width_fraction": self.pipe_component_selection_info.get("band_width_fraction"),
            "pipe_component_band_column_coverage": self.pipe_component_selection_info.get("band_column_coverage"),
            "pipe_component_band_pixels": self.pipe_component_selection_info.get("band_pixels"),
            "pipe_component_band_method": self.pipe_component_selection_info.get("band_method"),
            "pipe_mask_reproject_applied": self.pipe_mask_reproject_info.get("applied"),
            "pipe_mask_reproject_source": self.pipe_mask_reproject_info.get("source"),
            "pipe_mask_reproject_reason": self.pipe_mask_reproject_info.get("reason"),
            "pipe_mask_reproject_radius_m": self.pipe_mask_reproject_info.get("radius_m"),
            "pipe_mask_reproject_mask_px": self.pipe_mask_reproject_info.get("mask_px"),
            "pipe_lock_active": bool(self.pipe_lock_model is not None),
            "pipe_image_lock_active": bool(self.pipe_image_lock_model is not None),
            "pipe_lock_missed_frames": int(self.pipe_lock_missed_frames),
            "pipe_lock_source": self.pipe_lock_source,
            "pipe_image_lock_source": self.pipe_image_lock_source,
            "pipe_lock_selection_score": self.pipe_component_selection_info.get("score"),
            "pipe_lock_axis_delta_deg": self.pipe_component_selection_info.get("lock_axis_delta_deg"),
            "pipe_lock_radius_delta_m": self.pipe_component_selection_info.get("lock_radius_delta_m"),
            "pipe_lock_stand_delta_m": self.pipe_component_selection_info.get("lock_stand_delta_m"),
            "pipe_lock_axis_point_delta_m": self.pipe_component_selection_info.get("lock_axis_point_delta_m"),
            "pipe_image_lock_axis_delta_deg": self.pipe_component_selection_info.get("image_lock_axis_delta_deg"),
            "pipe_image_lock_center_delta_px": self.pipe_component_selection_info.get("image_lock_center_delta_px"),
            "pipe_image_lock_depth_delta_m": self.pipe_component_selection_info.get("image_lock_depth_delta_m"),
            "junction_lock_active": bool(self.junction_lock_active),
            "junction_lock_missed_frames": int(self.junction_lock_missed_frames),
            "junction_lock_confidence": _round(self.junction_lock_confidence),
            "junction_lock_source": self.junction_lock_source,
            "gap_plane_available": False,
            "weld_seam_pose_available": False,
        }

    def _mark_pipe_lock_missed(self, reason: str) -> None:
        if self.pipe_lock_model is None and self.pipe_image_lock_model is None:
            self.pipe_lock_source = "none"
            self.pipe_image_lock_source = "none"
            return
        frame = int(getattr(self, "processed_frame_count", -1))
        if frame >= 0 and frame == int(getattr(self, "_pipe_lock_missed_frame", -2)):
            return
        self._pipe_lock_missed_frame = frame
        self.pipe_lock_missed_frames += 1
        self.pipe_lock_source = f"missed:{reason}"
        self.pipe_image_lock_source = f"missed:{reason}"
        if (
            bool(self.params["pipe_lock_release_on_missed"])
            and self.pipe_lock_missed_frames >= int(self.params["pipe_lock_max_missed_frames"])
        ):
            self.pipe_lock_model = None
            self.pipe_image_lock_model = None
            self._pipe_model_filt = None
            self._low_warm_streak = 0
            self.pipe_lock_source = "released:missed_too_long"
            self.pipe_image_lock_source = "released:missed_too_long"

    def _pipe_fit_model(self, fit: Any) -> dict[str, Any] | None:
        try:
            axis = _normalize(np.asarray(getattr(fit, "axis_camera_xyz"), dtype=np.float64))
            axis_point = np.asarray(getattr(fit, "axis_point_camera_xyz_m"), dtype=np.float64).reshape(3)
            if not np.isfinite(axis_point).all():
                return None
            return {
                "axis": axis,
                "axis_point": axis_point,
                "radius_m": float(getattr(fit, "radius_m")),
                "stand_off_m": float(getattr(fit, "stand_off_m")),
                "residual_m": float(getattr(fit, "residual_m")),
                "inlier_fraction": float(getattr(fit, "inlier_fraction")),
            }
        except Exception:
            return None

    def _tracker_pipe_model(self, tracker: TrackerResult) -> dict[str, Any] | None:
        try:
            axis = _normalize(np.asarray(tracker.pipe_axis_xyz, dtype=np.float64))
            centroid = np.asarray(tracker.centroid_xyz_m, dtype=np.float64).reshape(3)
            if not np.isfinite(centroid).all():
                return None
            return {
                "axis": axis,
                "axis_point": centroid,
                "radius_m": float(tracker.pipe_pose_radius_m),
                "stand_off_m": float(tracker.stand_off_m),
                "residual_m": float(tracker.pipe_pose_residual_m),
                "inlier_fraction": float(tracker.pipe_pose_inlier_fraction),
            }
        except Exception:
            return None

    def _pipe_model_valid_for_lock(self, model: dict[str, Any] | None) -> bool:
        if model is None:
            return False
        nominal = max(1e-6, float(self.params["pipe_radius_m"]))
        radius_margin = max(
            float(self.params["pipe_lock_radius_abs_margin_m"]),
            float(self.params["pipe_lock_radius_rel_margin"]) * nominal,
        )
        radius = float(model.get("radius_m", 0.0))
        residual = float(model.get("residual_m", float("inf")))
        inlier_fraction = float(model.get("inlier_fraction", 0.0))
        return (
            abs(radius - nominal) <= radius_margin
            and residual <= float(self.params["pipe_lock_max_residual_m"])
            and inlier_fraction >= float(self.params["pipe_lock_min_inlier_fraction"])
        )

    def _update_pipe_lock(self, tracker: TrackerResult) -> None:
        if not bool(self.params["enable_pipe_temporal_lock"]):
            return
        if (
            bool(self.params["pipe_lock_require_fit_model"])
            and self.pipe_component_selected_model is None
        ):
            # A band-fallback mask has no cylinder-validated model behind it:
            # locking onto its tracker-PCA "cylinder" is exactly how the black
            # cloth captured the pipe lock at acquisition.
            self._mark_pipe_lock_missed("no_fit_validated_model")
            return
        model = self.pipe_component_selected_model or self._tracker_pipe_model(tracker)
        if not self._pipe_model_valid_for_lock(model):
            self._mark_pipe_lock_missed("tracker_model_invalid")
            return
        if self._coloroff_guard_frame_active and self._pipe_tracker_hold_frame:
            # FSM HOLD: keep projecting the held cylinder, never rewrite
            # axis/point/radius from a mask the normal field does not confirm.
            if self.processed_frame_count != self._pipe_lock_update_count_frame:
                self._pipe_lock_update_count_frame = int(self.processed_frame_count)
                self._pipe_lock_update_held += 1
            self._mark_pipe_lock_missed(f"coloroff_hold:{self._coloroff_cyl_reason}")
            return
        assert model is not None
        self.pipe_lock_model = model
        self.pipe_lock_missed_frames = 0
        self.pipe_lock_source = "tracker_update"
        if self.processed_frame_count != self._pipe_lock_update_count_frame:
            self._pipe_lock_update_count_frame = int(self.processed_frame_count)
            self._pipe_lock_update_accepted += 1

    def _pipe_image_model_from_candidate(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        try:
            point = candidate.get("band_line_point_uv")
            direction = candidate.get("band_line_direction_uv")
            if point is None or direction is None:
                selection_mask = np.asarray(candidate.get("selection_mask"), dtype=bool)
                ys, xs = np.nonzero(selection_mask)
                if xs.size < 8:
                    return None
                fit = _pca(np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1))
                point = fit["centroid"]
                direction = fit["direction"]
            point_uv = np.asarray(point, dtype=np.float64).reshape(2)
            direction_uv = _normalize(np.asarray(direction, dtype=np.float64).reshape(2))
            if not np.isfinite(point_uv).all() or not np.isfinite(direction_uv).all():
                return None
            if float(direction_uv[0]) < 0.0:
                direction_uv *= -1.0
            depth_median = float(candidate.get("band_depth_median_m", float("nan")))
            return {
                "point_uv": point_uv,
                "direction_uv": direction_uv,
                "depth_median_m": depth_median,
                "width_fraction": float(candidate.get("band_width_fraction", 0.0)),
                "score": float(candidate.get("band_score", candidate.get("score", 0.0))),
                "label": int(candidate.get("label", -1)),
            }
        except Exception:
            return None

    def _pipe_image_model_valid_for_lock(self, model: dict[str, Any] | None) -> bool:
        if model is None:
            return False
        point = np.asarray(model.get("point_uv"), dtype=np.float64)
        direction = np.asarray(model.get("direction_uv"), dtype=np.float64)
        return (
            point.shape == (2,)
            and direction.shape == (2,)
            and np.isfinite(point).all()
            and np.isfinite(direction).all()
            and float(model.get("width_fraction", 0.0)) >= float(self.params["pipe_image_lock_min_width_fraction"])
            and float(model.get("score", 0.0)) >= float(self.params["pipe_image_lock_min_band_score"])
        )

    def _update_pipe_image_lock(self) -> None:
        if not bool(self.params["enable_pipe_image_temporal_lock"]):
            return
        model = self.pipe_component_selected_image_model
        if not self._pipe_image_model_valid_for_lock(model):
            return
        assert model is not None
        self.pipe_image_lock_model = model
        self.pipe_lock_missed_frames = 0
        self.pipe_image_lock_source = "band_update"

    def _pipe_image_lock_compatible_score(self, model: dict[str, Any]) -> tuple[bool, float, dict[str, float]]:
        if self.pipe_image_lock_model is None:
            return False, 0.0, {}
        lock = self.pipe_image_lock_model
        direction = _normalize(np.asarray(model["direction_uv"], dtype=np.float64).reshape(2))
        lock_direction = _normalize(np.asarray(lock["direction_uv"], dtype=np.float64).reshape(2))
        axis_delta = self._axis_delta_deg(direction, lock_direction)
        normal = np.array([-lock_direction[1], lock_direction[0]], dtype=np.float64)
        center_delta = abs(float(np.dot(np.asarray(model["point_uv"]) - np.asarray(lock["point_uv"]), normal)))
        depth = float(model.get("depth_median_m", float("nan")))
        lock_depth = float(lock.get("depth_median_m", float("nan")))
        if np.isfinite(depth) and np.isfinite(lock_depth):
            depth_delta = abs(depth - lock_depth)
        else:
            depth_delta = 0.0

        max_axis = float(self.params["pipe_image_lock_max_axis_delta_deg"])
        max_center = float(self.params["pipe_image_lock_max_center_shift_px"])
        max_depth = float(self.params["pipe_image_lock_max_depth_delta_m"])
        compatible = axis_delta <= max_axis and center_delta <= max_center and depth_delta <= max_depth
        score = (
            math.exp(-axis_delta / max(max_axis, 1e-6))
            * math.exp(-center_delta / max(max_center, 1e-6))
            * math.exp(-depth_delta / max(max_depth, 1e-6))
            * max(0.05, min(1.0, float(model.get("score", 0.0)) / max(float(self.params["pipe_component_band_fallback_min_score"]), 1e-6)))
        )
        debug = {
            "axis_delta_deg": axis_delta,
            "center_delta_px": center_delta,
            "depth_delta_m": depth_delta,
        }
        return compatible, float(score), debug

    def _pipe_lock_compatible_score(self, model: dict[str, Any]) -> tuple[bool, float, dict[str, float]]:
        if self.pipe_lock_model is None:
            return False, 0.0, {}
        lock = self.pipe_lock_model
        axis_delta = self._axis_delta_deg(model["axis"], lock["axis"])
        radius_delta = abs(float(model["radius_m"]) - float(lock["radius_m"]))
        stand_delta = abs(float(model["stand_off_m"]) - float(lock["stand_off_m"]))
        point_delta = float(np.linalg.norm(np.asarray(model["axis_point"]) - np.asarray(lock["axis_point"])))
        residual = max(0.0, float(model["residual_m"]))
        inlier_fraction = max(0.0, min(1.0, float(model["inlier_fraction"])))

        max_axis = float(self.params["pipe_lock_max_axis_delta_deg"])
        max_radius = float(self.params["pipe_lock_max_radius_delta_m"])
        max_stand = float(self.params["pipe_lock_max_standoff_delta_m"])
        max_point = float(self.params["pipe_lock_max_axis_point_delta_m"])
        compatible = (
            axis_delta <= max_axis
            and radius_delta <= max_radius
            and stand_delta <= max_stand
            and point_delta <= max_point
            and residual <= float(self.params["pipe_lock_max_residual_m"])
            and inlier_fraction >= float(self.params["pipe_lock_min_inlier_fraction"])
        )
        score = (
            math.exp(-axis_delta / max(max_axis, 1e-6))
            * math.exp(-radius_delta / max(max_radius, 1e-6))
            * math.exp(-stand_delta / max(max_stand, 1e-6))
            * math.exp(-point_delta / max(max_point, 1e-6))
            * math.exp(-residual / max(float(self.params["pipe_lock_max_residual_m"]), 1e-6))
            * max(0.05, inlier_fraction)
        )
        debug = {
            "axis_delta_deg": axis_delta,
            "radius_delta_m": radius_delta,
            "stand_delta_m": stand_delta,
            "axis_point_delta_m": point_delta,
            "residual_m": residual,
            "inlier_fraction": inlier_fraction,
        }
        return compatible, float(score), debug

    def _pipe_band_candidate(
        self,
        component_mask: np.ndarray,
        depth: np.ndarray,
        expected_diameter_px: float,
    ) -> dict[str, Any]:
        """Extract a long pipe-like image band from one depth component.

        A close/partial cylinder often fails a strict metric cylinder fit, and
        real depth can connect the pipe to support/background pixels. This
        helper keeps the robust operational assumption instead: the pipe is an
        elongated visible band in the image, possibly tilted. It returns a
        filtered mask for RGB junction search, while the metric cylinder fit
        remains a quality signal for pose/lock when it is reliable.
        """

        ys, xs = np.nonzero(component_mask)
        total_pixels = int(xs.size)
        if total_pixels <= 0:
            return {"valid": False, "reason": "empty_component", "score": 0.0, "mask": component_mask}

        image_h, image_w = component_mask.shape[:2]
        min_pixels = int(self.params["pipe_component_band_min_pixels"])
        min_width_fraction = float(self.params["pipe_component_band_min_width_fraction"])
        min_column_coverage = float(self.params["pipe_component_band_min_column_coverage"])
        min_col_pixels = max(1, int(self.params["pipe_component_band_min_col_pixels"]))
        half_min = float(self.params["pipe_component_band_half_width_min_px"])
        half_max = float(self.params["pipe_component_band_half_width_max_px"])
        expected_scale = float(self.params["pipe_component_band_expected_diameter_scale"])
        max_hypothesis_points = max(256, int(self.params["pipe_component_band_max_hypothesis_points"]))
        run_gap_px = max(1, int(self.params["pipe_component_band_run_gap_px"]))
        upper_run_weight = float(self.params["pipe_component_band_upper_run_weight"])

        if expected_diameter_px > 1.0:
            expected_half = 0.5 * expected_scale * expected_diameter_px
        else:
            expected_half = 0.5 * (half_min + half_max)
        expected_half = float(np.clip(expected_half, half_min, half_max))
        half_widths = sorted({
            float(np.clip(half_min, half_min, half_max)),
            float(np.clip(expected_half, half_min, half_max)),
            float(np.clip(min(half_max, 1.7 * expected_half), half_min, half_max)),
        })

        col_counts = np.bincount(xs, minlength=image_w)
        # Pre-sort the component points once by (x, y): per-column slices replace
        # the per-column boolean scans (xs == col), which were O(points*columns)
        # and dominated the whole frame budget on large real-camera components.
        sort_order = np.lexsort((ys, xs))
        sorted_x = xs[sort_order]
        sorted_y = ys[sort_order]
        col_bounds = np.searchsorted(sorted_x, np.arange(image_w + 1))

        # Primary hypothesis for real camera close-ups: for each image column,
        # take the upper-most contiguous depth run of the component. The pipe is
        # expected to be the visible object in front after the upstream
        # alignment/scanning phase; supports/background attached below the pipe
        # must not drag the centreline downward.
        upper_x: list[float] = []
        upper_y: list[float] = []
        upper_half: list[float] = []
        target_height = max(2.0 * half_min, min(2.0 * half_max, 2.0 * expected_half))
        cols_for_runs = np.flatnonzero(col_counts >= min_col_pixels)
        for col in cols_for_runs:
            y_col = sorted_y[col_bounds[col]:col_bounds[col + 1]]
            if y_col.size < min_col_pixels:
                continue
            splits = np.flatnonzero(np.diff(y_col) > run_gap_px) + 1
            runs = np.split(y_col, splits)
            chosen: np.ndarray | None = None
            for run in runs:
                if run.size >= min_col_pixels:
                    chosen = run
                    break
            if chosen is None:
                continue
            y0 = int(chosen[0])
            y1 = int(chosen[-1])
            # If a support is connected to the pipe in depth, the first run can
            # become very tall. Cap it from the top by the expected pipe height
            # so the band stays on the pipe body instead of the structure below.
            y1_cap = min(y1, int(round(y0 + target_height)))
            if y1_cap <= y0:
                continue
            upper_x.append(float(col))
            upper_y.append(0.5 * float(y0 + y1_cap))
            upper_half.append(max(half_min, 0.5 * float(y1_cap - y0 + 1)))

        if total_pixels > max_hypothesis_points:
            idx = np.linspace(0, total_pixels - 1, max_hypothesis_points, dtype=np.int64)
            hx = xs[idx].astype(np.float64)
            hy = ys[idx].astype(np.float64)
        else:
            hx = xs.astype(np.float64)
            hy = ys.astype(np.float64)

        hypotheses: list[tuple[str, np.ndarray, np.ndarray]] = []
        if len(upper_x) >= 8:
            ux = np.asarray(upper_x, dtype=np.float64)
            uy = np.asarray(upper_y, dtype=np.float64)
            try:
                slope, intercept = np.polyfit(ux, uy, 1)
                residual = uy - (slope * ux + intercept)
                mad = float(np.median(np.abs(residual - np.median(residual))))
                keep = np.abs(residual - np.median(residual)) <= max(4.0, 3.5 * 1.4826 * mad)
                if int(keep.sum()) >= 8:
                    slope, intercept = np.polyfit(ux[keep], uy[keep], 1)
                point_x = float(np.median(ux))
                point = np.array([point_x, slope * point_x + intercept], dtype=np.float64)
                direction = _normalize(np.array([1.0, slope], dtype=np.float64))
                hypotheses.append(("upper_column_run_band", point, direction))
                if upper_half:
                    half_widths.append(float(np.clip(np.median(upper_half), half_min, half_max)))
                    half_widths = sorted(set(float(v) for v in half_widths))
            except Exception:
                pass
        if hx.size >= 3:
            try:
                uv = np.stack([hx, hy], axis=1)
                fit = _pca(uv)
                direction = _normalize(np.asarray(fit["direction"], dtype=np.float64))
                if abs(float(direction[0])) >= float(self.params["pipe_component_band_min_axis_dx"]):
                    if direction[0] < 0.0:
                        direction *= -1.0
                    hypotheses.append(("pca_band", np.asarray(fit["centroid"], dtype=np.float64), direction))
            except Exception:
                pass

        # A second deterministic hypothesis: robust line through per-column
        # medians. This is less sensitive than PCA when a depth component also
        # contains sparse support pixels connected to the pipe.
        cols = np.flatnonzero(col_counts >= min_col_pixels)
        if cols.size >= 8:
            med_y: list[float] = []
            med_x: list[float] = []
            for col in cols:
                y_col = sorted_y[col_bounds[col]:col_bounds[col + 1]]
                if y_col.size >= min_col_pixels:
                    med_x.append(float(col))
                    med_y.append(float(np.median(y_col)))
            if len(med_x) >= 8:
                mx = np.asarray(med_x, dtype=np.float64)
                my = np.asarray(med_y, dtype=np.float64)
                try:
                    # Least-squares on medians, then one residual trim pass.
                    slope, intercept = np.polyfit(mx, my, 1)
                    residual = my - (slope * mx + intercept)
                    mad = float(np.median(np.abs(residual - np.median(residual))))
                    keep = np.abs(residual - np.median(residual)) <= max(4.0, 3.5 * 1.4826 * mad)
                    if int(keep.sum()) >= 8:
                        slope, intercept = np.polyfit(mx[keep], my[keep], 1)
                    point_x = float(np.median(mx))
                    point = np.array([point_x, slope * point_x + intercept], dtype=np.float64)
                    direction = _normalize(np.array([1.0, slope], dtype=np.float64))
                    hypotheses.append(("column_median_band", point, direction))
                except Exception:
                    pass

        if not hypotheses:
            return {"valid": False, "reason": "no_line_hypothesis", "score": 0.0, "mask": component_mask}

        best: dict[str, Any] | None = None
        px = xs.astype(np.float64)
        py = ys.astype(np.float64)
        z_all = depth[ys, xs].astype(np.float64)
        for method, point, direction in hypotheses:
            dx = float(direction[0])
            dy = float(direction[1])
            dist = np.abs((px - float(point[0])) * dy - (py - float(point[1])) * dx)
            for half_width in half_widths:
                in_band = dist <= half_width
                band_count = int(in_band.sum())
                if band_count <= 0:
                    continue
                bx = xs[in_band]
                by = ys[in_band]
                width_px = int(bx.max() - bx.min() + 1)
                width_fraction = float(width_px) / max(float(image_w), 1.0)
                band_col_counts = np.bincount(bx, minlength=image_w)
                active_cols = int(np.count_nonzero(band_col_counts[bx.min(): bx.max() + 1] >= min_col_pixels))
                column_coverage = float(active_cols) / max(float(width_px), 1.0)
                z_band = z_all[in_band]
                z_band = z_band[np.isfinite(z_band) & (z_band > 0.0)]
                if z_band.size:
                    depth_median = float(np.median(z_band))
                    depth_mad = float(np.median(np.abs(z_band - depth_median)))
                else:
                    depth_median = float("nan")
                    depth_mad = float("inf")
                band_area = max(1.0, float(width_px) * max(1.0, 2.0 * half_width + 1.0))
                density = min(1.0, float(band_count) / band_area)
                if expected_diameter_px > 1.0:
                    height_error = abs(math.log(max(2.0 * half_width, 1.0) / max(expected_diameter_px, 1.0)))
                    height_score = math.exp(-0.45 * height_error)
                else:
                    height_score = 1.0
                depth_score = math.exp(-depth_mad / max(0.02, 0.03 * max(depth_median, 0.1)))
                valid = (
                    band_count >= min_pixels
                    and width_fraction >= min_width_fraction
                    and column_coverage >= min_column_coverage
                )
                score = (
                    max(1.0, float(width_px))
                    * math.sqrt(float(band_count))
                    * max(0.02, width_fraction)
                    * max(0.02, column_coverage)
                    * max(0.05, density)
                    * max(0.05, height_score)
                    * max(0.05, depth_score)
                )
                if method == "upper_column_run_band":
                    score *= upper_run_weight
                band_mask = np.zeros_like(component_mask, dtype=bool)
                band_mask[by, bx] = True
                record = {
                    "valid": bool(valid),
                    "reason": "ok" if valid else "below_band_thresholds",
                    "method": method,
                    "score": float(score),
                    "mask": band_mask,
                    "line_point_uv": np.asarray(point, dtype=np.float64).copy(),
                    "line_direction_uv": np.asarray(direction, dtype=np.float64).copy(),
                    "pixels": band_count,
                    "width_px": width_px,
                    "width_fraction": width_fraction,
                    "column_coverage": column_coverage,
                    "density": float(density),
                    "half_width_px": float(half_width),
                    "depth_median_m": depth_median,
                    "depth_mad_m": depth_mad,
                    "height_score": float(height_score),
                }
                if best is None or float(record["score"]) > float(best["score"]):
                    best = record

        if best is None:
            return {"valid": False, "reason": "no_band_support", "score": 0.0, "mask": component_mask}
        return best

    def _compute_warm_mask(self, rgb: np.ndarray) -> np.ndarray | None:
        """Task color prior: the pipe is orange PVC. A generous warm-hue mask
        (plus a bright rescue for washed-out specular cores) separates the
        pipe from black cloth / gray bricks / floor, which live at the SAME
        depth in the field scenes and are radially indistinguishable from the
        pipe surface. Used to score acquisition candidates, to keep clutter
        out of the refit/recenter seed, and to validate the silhouette
        interior. Configurable and fully bypassed when disabled."""
        if not bool(self.params["pipe_color_prior_enabled"]) or cv2 is None:
            return None
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        warm = (
            (h >= int(self.params["pipe_color_hue_min"]))
            & (h <= int(self.params["pipe_color_hue_max"]))
            & (s >= int(self.params["pipe_color_sat_min"]))
            & (v >= int(self.params["pipe_color_val_min"]))
        )
        # Specular cores lose hue/saturation but stay bright; dark cloth and
        # gray concrete never reach this value band.
        warm |= v >= int(self.params["pipe_color_bright_rescue_val"])
        return warm

    def _get_normal_image(self, depth: np.ndarray, k: np.ndarray, stride: int):
        """Per-frame cached strided normal field (unit normals oriented toward
        the camera) + 3D point grid. COLOR-INDEPENDENT discriminator: cloth
        and bricks at the pipe's own depth are radially indistinguishable, but
        their normals point at the camera / upward, while true pipe normals
        are RADIAL around the axis."""
        cache = getattr(self, "_normal_image_cache", None)
        if cache is not None and cache[0] == self.processed_frame_count and cache[1] == stride:
            return cache[2], cache[3], cache[4]
        d = depth[::stride, ::stride].astype(np.float32)
        if cv2 is not None:
            ds = cv2.GaussianBlur(d, (5, 5), 1.0)
        else:
            ds = d
        fx, fy = float(k[0, 0]) / stride, float(k[1, 1]) / stride
        cx, cy = float(k[0, 2]) / stride, float(k[1, 2]) / stride
        h2, w2 = ds.shape
        us = (np.arange(w2, dtype=np.float32) - np.float32(cx)) / np.float32(fx)
        vs = (np.arange(h2, dtype=np.float32) - np.float32(cy)) / np.float32(fy)
        grid = np.stack([us[None, :] * ds, vs[:, None] * ds, ds], axis=-1)
        dx = grid[1:-1, 2:] - grid[1:-1, :-2]
        dy = grid[2:, 1:-1] - grid[:-2, 1:-1]
        normals = np.cross(dx, dy)
        ln = np.linalg.norm(normals, axis=-1)
        valid_d = np.isfinite(d) & (d > 0.05)
        jump = np.float32(0.05 * stride)
        cont = (
            (np.abs(d[1:-1, 2:] - d[1:-1, :-2]) < jump)
            & (np.abs(d[2:, 1:-1] - d[:-2, 1:-1]) < jump)
        )
        ok_core = cont & (ln > 1e-9) & np.isfinite(ln) & valid_d[1:-1, 1:-1]
        unit = np.zeros_like(normals)
        unit[ok_core] = normals[ok_core] / ln[ok_core][:, None]
        dots = np.einsum("ijk,ijk->ij", unit, grid[1:-1, 1:-1])
        flip = dots > 0
        unit[flip] = -unit[flip]
        nrm_img = np.zeros((h2, w2, 3), np.float32)
        ok_img = np.zeros((h2, w2), bool)
        nrm_img[1:-1, 1:-1] = unit
        ok_img[1:-1, 1:-1] = ok_core
        self._normal_image_cache = (self.processed_frame_count, stride, nrm_img, ok_img, grid)
        return nrm_img, ok_img, grid

    @staticmethod
    def _normal_radial_cos(points: np.ndarray, normals: np.ndarray,
                           axis: np.ndarray, axis_point: np.ndarray) -> np.ndarray:
        q = points - axis_point[None, :]
        qd = q @ axis
        radial = q - qd[:, None] * axis[None, :]
        rn = np.linalg.norm(radial, axis=-1)
        rn = np.where(rn < 1e-9, 1.0, rn)
        radial = radial / rn[:, None]
        return np.einsum("ij,ij->i", normals, radial)

    @staticmethod
    def _cylinder_azimuth_features(
        points: np.ndarray,
        normals: np.ndarray,
        axis: np.ndarray,
        axis_point: np.ndarray,
        radius_m: float,
    ) -> dict[str, float] | None:
        """Azimuthal normal-rotation statistics of a candidate cylinder.

        The per-point radial-cos test is necessary but NOT sufficient: a plane
        force-fitted as a tangent cylinder keeps every normal aligned with the
        radial direction of its tangency band (the black cloth passed it at
        0.94+). What a plane can never fake is the ROTATION of the normal
        field around the axis: position azimuth phi and normal azimuth psi are
        measured in a basis whose e1 points from the axis toward the camera —
        on a true cylinder psi follows phi 1:1 (slope ~1) over a wide inlier
        arc, on cloth psi stays constant (slope ~0), on a gentle fold of
        radius R it slopes ~r/R.
        """
        axis = _normalize(np.asarray(axis, dtype=np.float64).reshape(3))
        axis_point = np.asarray(axis_point, dtype=np.float64).reshape(3)
        c_perp = -axis_point + float(axis_point @ axis) * axis
        cn = float(np.linalg.norm(c_perp))
        if cn < 1e-6:
            return None
        e1 = c_perp / cn
        e2 = np.cross(axis, e1)
        pts = np.asarray(points, dtype=np.float64)
        nrm = np.asarray(normals, dtype=np.float64)
        q = pts - axis_point[None, :]
        qd = q @ axis
        w = q - qd[:, None] * axis[None, :]
        wn = np.linalg.norm(w, axis=1)
        n_perp = nrm - (nrm @ axis)[:, None] * axis[None, :]
        nn = np.linalg.norm(n_perp, axis=1)
        good = (wn > 1e-6) & (nn > 0.2)
        if int(good.sum()) < 60:
            return None
        phi = np.arctan2(w[good] @ e2, w[good] @ e1)
        psi = np.arctan2(n_perp[good] @ e2, n_perp[good] @ e1)
        var_phi = float(np.var(phi))
        slope = 0.0
        if var_phi > (math.radians(4.0) ** 2):
            slope = float(np.mean((phi - phi.mean()) * (psi - psi.mean())) / var_phi)
        align_frac = float((np.cos(psi - phi) >= math.cos(math.radians(25.0))).mean())
        inl = np.abs(wn[good] - float(radius_m)) <= 0.02
        cov_inlier_deg = 0.0
        if int(inl.sum()) >= 30:
            cov_inlier_deg = float(
                np.degrees(np.percentile(phi[inl], 97.0) - np.percentile(phi[inl], 3.0))
            )
        return {
            "slope": slope,
            "align_frac": align_frac,
            "cov_inlier_deg": cov_inlier_deg,
            "cov_deg": float(np.degrees(np.percentile(phi, 97.0) - np.percentile(phi, 3.0))),
            "samples": float(good.sum()),
        }

    def _update_coloroff_normal_stats(self, pipe_mask: np.ndarray, depth: np.ndarray, k: np.ndarray) -> None:
        """Normal-consistency fraction + azimuthal cylinder features of the
        current mask against the current component model (color-less path)."""
        self.pipe_mask_normal_fraction = None
        self._coloroff_cyl_features = None
        if (
            bool(self.params["pipe_normal_consistency_enabled"])
            and not self._warm_scene_ok
            and self.pipe_component_selected_model is not None
        ):
            comps = self._pipe_axis_model_components_from_model(self.pipe_component_selected_model)
            if comps is not None:
                n_stride = max(1, int(self.params["pipe_mask_reproject_sample_stride"]))
                nrm_img_q, ok_img_q, grid_q = self._get_normal_image(depth, k, n_stride)
                m_str = pipe_mask[::n_stride, ::n_stride]
                if m_str.shape == ok_img_q.shape:
                    sel_q = m_str & ok_img_q
                    if int(sel_q.sum()) >= 100:
                        cos_q = self._normal_radial_cos(grid_q[sel_q], nrm_img_q[sel_q], comps[0], comps[1])
                        self.pipe_mask_normal_fraction = float(
                            (cos_q >= float(self.params["pipe_normal_cos_min"])).mean()
                        )
                        self._coloroff_cyl_features = self._cylinder_azimuth_features(
                            grid_q[sel_q],
                            nrm_img_q[sel_q],
                            comps[0],
                            comps[1],
                            float(
                                self.pipe_component_selected_model.get(
                                    "radius_m", float(self.params["pipe_radius_m"])
                                )
                            ),
                        )

    def _advance_pipe_tracker_state(
        self,
        depth: np.ndarray,
        k: np.ndarray,
        valid: np.ndarray,
        pipe_mask: np.ndarray,
        low_warm: bool,
        low_normal: bool,
    ) -> np.ndarray:
        """Single source of truth for the tracker state and model-update gate.

        ACQUIRE: search globally (component selection + color-less RANSAC) and
            only promote to TRACK after a streak of cylinder-consistent,
            axis-compatible models — this is where the metric lock is born.
        TRACK: the lock exists; the frame's cylinder gate decides whether
            _update_pipe_lock rewrites axis/point/radius (pass) or the model is
            frozen (fail -> HOLD).
        HOLD: never touch axis/point/radius; keep projecting the held cylinder
            and let the junction lock coast. Recover to TRACK on a passing gate.
        LOST: after pipe_coloroff_release_frames of HOLD the lock is dropped and
            the search restarts from ACQUIRE.

        Color/warm mode keeps the proven v5 model-update path unchanged; the
        FSM there only reflects the state as a real transitioned field so the
        reported ACQUIRE/TRACK/HOLD/LOST is consistent across both modes.
        """
        guard_active = (
            bool(self.params["pipe_coloroff_cylinder_guard_enabled"])
            and bool(self.params["pipe_normal_consistency_enabled"])
            and not self._warm_scene_ok
        )
        self._coloroff_guard_frame_active = guard_active

        if not guard_active:
            # Warm scene: do not alter v5's model-update decisions. Mirror them
            # into an explicit, transitioned state field for reporting only.
            self._coloroff_cyl_ok = True
            self._coloroff_cyl_reason = "guard_inactive"
            self._coloroff_acquire_streak = 0
            self._coloroff_acquire_prev = None
            self._coloroff_fail_streak = 0
            self._coloroff_hold_count = 0
            self._pipe_tracker_hold_frame = False
            self._coloroff_pipe_visible = True
            if self.pipe_lock_model is None:
                self._pipe_state = "LOST" if "released" in str(self.pipe_lock_source) else "ACQUIRE"
            elif self.pipe_lock_missed_frames > 0 or low_warm or low_normal:
                self._pipe_state = "HOLD"
            else:
                self._pipe_state = "TRACK"
            return pipe_mask

        cyl_ok, reason = self._coloroff_eval_cyl_gate()
        state = self._pipe_state if self.pipe_lock_model is not None else "ACQUIRE"

        if state in ("ACQUIRE", "LOST"):
            # v7 acquisition stickiness: before the (non-deterministic) RANSAC,
            # reproject from the last cylinder-consistent model found in this
            # ACQUIRE episode. This keeps the search on the pipe frame-to-frame
            # instead of flipping back to the cloth when the component fit or a
            # fresh RANSAC misses it — the source of the start-up axis jumps.
            if (
                not cyl_ok
                and bool(self.params["pipe_coloroff_acquire_sticky"])
                and self._coloroff_provisional_model is not None
                and self._coloroff_provisional_age
                < int(self.params["pipe_coloroff_provisional_max_age"])
            ):
                prev_model = self.pipe_component_selected_model
                prev_info = self.pipe_component_selection_info
                self.pipe_component_selected_model = dict(self._coloroff_provisional_model)
                sticky_mask = self._cylinder_reproject_mask(
                    np.zeros_like(valid, dtype=bool), valid, depth, k
                )
                if bool(self.pipe_mask_reproject_info.get("applied")):
                    pipe_mask = sticky_mask
                    self.pipe_component_selection_info = {
                        "method": "coloroff_sticky_acquire",
                        "component_count": None,
                        "selected_label": None,
                    }
                    self.pipe_mask_warm_fraction = self._mask_warm_fraction(
                        pipe_mask, self._warm_mask_current
                    )
                    self._update_coloroff_normal_stats(pipe_mask, depth, k)
                    cyl_ok, reason = self._coloroff_eval_cyl_gate()
                    reason = f"sticky:{reason}"
                else:
                    self.pipe_component_selected_model = prev_model
                    self.pipe_component_selection_info = prev_info
            # Global color-less acquisition. The near-depth component fuses
            # pipe+cloth without color, so when its model is not cylinder-ok
            # search the whole cloud: reliable normals vote an axis centre at
            # p - r*n (votes collapse onto a line only for a true task-radius
            # cylinder), RANSAC picks the line, and the render-and-compare
            # stage rebuilds the mask from it.
            if not cyl_ok and bool(self.params["pipe_coloroff_ransac_enabled"]):
                prev_model = self.pipe_component_selected_model
                prev_info = self.pipe_component_selection_info
                ransac_model = self._coloroff_ransac_cylinder(depth, k, valid)
                if ransac_model is not None:
                    self.pipe_component_selected_model = ransac_model
                    new_mask = self._cylinder_reproject_mask(
                        np.zeros_like(valid, dtype=bool), valid, depth, k
                    )
                    if bool(self.pipe_mask_reproject_info.get("applied")):
                        pipe_mask = new_mask
                        self.pipe_component_selection_info = {
                            "method": "coloroff_ransac_acquire",
                            "component_count": None,
                            "selected_label": None,
                        }
                        self.pipe_mask_warm_fraction = self._mask_warm_fraction(
                            pipe_mask, self._warm_mask_current
                        )
                        self._update_coloroff_normal_stats(pipe_mask, depth, k)
                        cyl_ok, reason = self._coloroff_eval_cyl_gate()
                        reason = f"ransac:{reason}"
                    else:
                        self.pipe_component_selected_model = prev_model
                        self.pipe_component_selection_info = prev_info
            self._coloroff_cyl_ok = cyl_ok
            self._coloroff_cyl_reason = reason
            self._coloroff_fail_streak = 0
            self._coloroff_hold_count = 0
            streak_ok = False
            if cyl_ok:
                comps_now = self._pipe_axis_model_components_from_model(
                    self.pipe_component_selected_model
                )
                if comps_now is not None:
                    axis_now, point_now = comps_now
                    prev = self._coloroff_acquire_prev
                    if prev is None:
                        streak_ok = True
                    else:
                        prev_axis, prev_point = prev
                        dp = point_now - prev_point
                        line_dist = float(np.linalg.norm(dp - float(dp @ prev_axis) * prev_axis))
                        streak_ok = (
                            self._axis_delta_deg(axis_now, prev_axis)
                            <= float(self.params["pipe_coloroff_acquire_max_axis_delta_deg"])
                            and line_dist
                            <= float(self.params["pipe_coloroff_acquire_max_line_dist_m"])
                        )
                    self._coloroff_acquire_prev = (axis_now.copy(), point_now.copy())
            if streak_ok:
                self._coloroff_acquire_streak += 1
            else:
                self._coloroff_acquire_streak = 1 if cyl_ok else 0
                if not cyl_ok:
                    self._coloroff_acquire_prev = None
            # Provisional-model memory for acquisition stickiness (v7): remember
            # the pipe once found so the next frame reprojects from it instead
            # of re-searching from scratch and flipping onto the cloth.
            if cyl_ok:
                self._coloroff_provisional_model = dict(self.pipe_component_selected_model)
                self._coloroff_provisional_age = 0
            elif self._coloroff_provisional_model is not None:
                self._coloroff_provisional_age += 1
                if self._coloroff_provisional_age >= int(
                    self.params["pipe_coloroff_provisional_max_age"]
                ):
                    self._coloroff_provisional_model = None
            if (
                self._coloroff_acquire_streak
                >= int(self.params["pipe_coloroff_acquire_stable_frames"])
                and self._pipe_model_valid_for_lock(self.pipe_component_selected_model)
            ):
                self.pipe_lock_model = dict(self.pipe_component_selected_model)
                self.pipe_lock_missed_frames = 0
                self.pipe_lock_source = f"coloroff_stable_acquire:{self._coloroff_acquire_streak}"
                self._pipe_state = "TRACK"
                self._coloroff_provisional_model = None
                self._coloroff_provisional_age = 0
            else:
                self._pipe_state = "ACQUIRE"
        elif state == "TRACK":
            self._coloroff_cyl_ok = cyl_ok
            self._coloroff_cyl_reason = reason
            self._coloroff_acquire_streak = 0
            self._coloroff_acquire_prev = None
            model_valid = self._pipe_model_valid_for_lock(self.pipe_component_selected_model)
            if cyl_ok and model_valid:
                self._coloroff_fail_streak = 0
                self._pipe_state = "TRACK"
            else:
                self._coloroff_fail_streak = 1
                self._coloroff_hold_count = 1
                self._pipe_state = "HOLD"
        else:  # HOLD
            self._coloroff_cyl_ok = cyl_ok
            self._coloroff_cyl_reason = reason
            self._coloroff_acquire_streak = 0
            self._coloroff_acquire_prev = None
            model_valid = self._pipe_model_valid_for_lock(self.pipe_component_selected_model)
            if cyl_ok and model_valid:
                self._coloroff_fail_streak = 0
                self._coloroff_hold_count = 0
                self._pipe_state = "TRACK"
            else:
                self._coloroff_hold_count += 1
                self._coloroff_fail_streak += 1
                if self._coloroff_hold_count >= int(self.params["pipe_coloroff_release_frames"]):
                    # Impostor escape without color: a lock that stays
                    # non-cylindrical this long is cloth/clutter, not the pipe.
                    self.pipe_lock_model = None
                    self.pipe_image_lock_model = None
                    self.pipe_lock_source = "released:coloroff_not_cylindrical"
                    self.pipe_image_lock_source = "released:coloroff_not_cylindrical"
                    self._pipe_model_filt = None
                    self._release_junction_lock("coloroff_not_cylindrical")
                    self._coloroff_hold_count = 0
                    self._coloroff_fail_streak = 0
                    self._pipe_state = "LOST"
                else:
                    self._pipe_state = "HOLD"

        self._pipe_tracker_hold_frame = self._pipe_state == "HOLD"
        # Pipe is "visible" (drawable/publishable) once anything trustworthy
        # backs it: an active lock, a provisional model from this ACQUIRE
        # episode, or a cylinder-consistent frame. Before the first such frame
        # the published axis is the default/cloth fallback and must be hidden.
        self._coloroff_pipe_visible = bool(
            self.pipe_lock_model is not None
            or self._coloroff_provisional_model is not None
            or self._coloroff_cyl_ok
        )
        return pipe_mask

    def _coloroff_eval_cyl_gate(self) -> tuple[bool, str]:
        feats = self._coloroff_cyl_features
        if self.pipe_component_selected_model is None:
            return False, "no_model"
        if feats is None:
            return False, "no_features"
        cyl_ok = (
            feats["slope"] >= float(self.params["pipe_cyl_guard_min_slope"])
            and feats["align_frac"] >= float(self.params["pipe_cyl_guard_min_align_frac"])
            and feats["cov_inlier_deg"] >= float(self.params["pipe_cyl_guard_min_inlier_cov_deg"])
        )
        if cyl_ok:
            return True, "ok"
        return False, (
            f"slope={feats['slope']:.2f}"
            f",align={feats['align_frac']:.2f}"
            f",cov={feats['cov_inlier_deg']:.0f}"
        )

    def _coloroff_ransac_cylinder(self, depth: np.ndarray, k: np.ndarray, valid: np.ndarray) -> dict[str, Any] | None:
        """Color-free global cylinder acquisition with the task radius prior.

        Every reliable surface normal votes an axis centre at p - r*n: on a
        true cylinder of radius r the votes collapse onto the axis LINE, on a
        plane/cloth they form a shifted sheet with no line concentration that
        also survives the azimuthal gate. A small deterministic RANSAC finds
        the best line; the model is validated on raw surface points (radial
        residual, axial extent, normal-rotation features) before being offered
        to the reproject stage.
        """
        stride = max(1, int(self.params["pipe_mask_reproject_sample_stride"]))
        nrm_img, ok_img, grid = self._get_normal_image(depth, k, stride)
        v_str = valid[::stride, ::stride]
        if v_str.shape != ok_img.shape:
            return None
        sel = ok_img & v_str
        pts = grid[sel].astype(np.float64)
        nrm = nrm_img[sel].astype(np.float64)
        if pts.shape[0] < int(self.params["pipe_coloroff_ransac_min_points"]):
            return None
        radius = float(self.params["pipe_radius_m"])
        centers = pts - radius * nrm
        max_votes = int(self.params["pipe_coloroff_ransac_max_votes"])
        rng = np.random.default_rng(986533 + int(self.processed_frame_count))
        if centers.shape[0] > max_votes:
            keep = rng.choice(centers.shape[0], size=max_votes, replace=False)
            votes = centers[keep]
        else:
            votes = centers
        tol = float(self.params["pipe_coloroff_ransac_center_tol_m"])
        iterations = int(self.params["pipe_coloroff_ransac_iterations"])
        pair_idx = rng.integers(0, votes.shape[0], size=(iterations, 2))
        best_count = 0
        best_inl: np.ndarray | None = None
        for a_i, b_i in pair_idx:
            span = votes[b_i] - votes[a_i]
            span_n = float(np.linalg.norm(span))
            if span_n < 0.08 or span_n > 1.5:
                continue
            axis_h = span / span_n
            d = votes - votes[a_i][None, :]
            dd = np.einsum("ij,ij->i", d, d)
            da = d @ axis_h
            dist2 = dd - da * da
            inl = dist2 <= tol * tol
            count = int(inl.sum())
            if count > best_count:
                best_count = count
                best_inl = inl
        if best_inl is None or best_count < int(self.params["pipe_coloroff_ransac_min_inliers"]):
            return None
        core = votes[best_inl]
        line_fit = _pca(core)
        axis = _normalize(np.asarray(line_fit["direction"], dtype=np.float64))
        axis_point = np.median(core, axis=0)
        # Validate on raw surface points against the hypothesised cylinder.
        q = pts - axis_point[None, :]
        qd = q @ axis
        dist = np.sqrt(np.maximum(np.einsum("ij,ij->i", q, q) - qd * qd, 0.0))
        band = np.abs(dist - radius) <= 0.05
        surf = np.abs(dist - radius) <= 0.02
        n_band = int(band.sum())
        n_surf = int(surf.sum())
        if n_band < 50 or n_surf < 50:
            return None
        extent = float(np.percentile(qd[surf], 95.0) - np.percentile(qd[surf], 5.0))
        if extent < float(self.params["pipe_coloroff_ransac_min_extent_m"]):
            return None
        feats = self._cylinder_azimuth_features(pts[surf], nrm[surf], axis, axis_point, radius)
        if feats is None:
            return None
        if (
            feats["slope"] < float(self.params["pipe_cyl_guard_min_slope"])
            or feats["align_frac"] < float(self.params["pipe_cyl_guard_min_align_frac"])
        ):
            return None
        residual = float(np.median(np.abs(dist[surf] - radius)))
        model = {
            "axis": axis,
            "axis_point": axis_point,
            "radius_m": radius,
            "stand_off_m": float(np.median(pts[surf][:, 2])),
            "residual_m": residual,
            "inlier_fraction": float(n_surf) / float(max(1, n_band)),
        }
        return model

    @staticmethod
    def _mask_warm_fraction(mask: np.ndarray, warm: np.ndarray | None, stride: int = 3) -> float | None:
        if warm is None:
            return None
        m = mask[::stride, ::stride]
        if not np.any(m):
            return None
        return float(warm[::stride, ::stride][m].mean())

    @staticmethod
    def _recenter_axis_known_radius(
        surface_points: np.ndarray,
        axis: np.ndarray,
        axis_point: np.ndarray,
        radius_nominal: float,
    ) -> tuple[np.ndarray, float] | None:
        """Re-solve the cylinder section centre with the radius held fixed.

        Points are projected onto the plane perpendicular to the axis and the
        2-DOF centre is found by Gauss-Newton on (|p - c| - r_nom), with one
        MAD trim in between to shed flying pixels behind the grazing edges.
        Returns (axis_point_on_axis_nearest_origin, residual) or None.
        """
        pts = np.asarray(surface_points, dtype=np.float64)
        if pts.shape[0] < 50:
            return None
        if pts.shape[0] > 30000:
            idx = np.linspace(0, pts.shape[0] - 1, 30000, dtype=np.int64)
            pts = pts[idx]
        # Deterministic plane basis perpendicular to the axis.
        helper = np.array([0.0, 0.0, 1.0]) if abs(float(axis[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = _normalize(np.cross(axis, helper))
        v = np.cross(axis, u)
        a = pts @ u
        b = pts @ v
        c = np.array([float(axis_point @ u), float(axis_point @ v)], dtype=np.float64)
        keep = np.ones(a.shape[0], dtype=bool)
        residual = float("inf")
        for iteration in range(5):
            da = a[keep] - c[0]
            db = b[keep] - c[1]
            dist = np.hypot(da, db)
            ok = dist > 1e-6
            if int(ok.sum()) < 50:
                return None
            unit_a = da[ok] / dist[ok]
            unit_b = db[ok] / dist[ok]
            resid = dist[ok] - radius_nominal
            h00 = float(unit_a @ unit_a)
            h01 = float(unit_a @ unit_b)
            h11 = float(unit_b @ unit_b)
            det = h00 * h11 - h01 * h01
            if abs(det) < 1e-9:
                return None
            g0 = float(unit_a @ resid)
            g1 = float(unit_b @ resid)
            c = c + np.array([(h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det])
            if iteration == 1:
                # One robust trim: drop points far off the pinned-radius circle.
                da_all = a - c[0]
                db_all = b - c[1]
                resid_all = np.hypot(da_all, db_all) - radius_nominal
                med = float(np.median(resid_all))
                mad = float(np.median(np.abs(resid_all - med)))
                keep = np.abs(resid_all - med) <= max(0.008, 3.0 * 1.4826 * mad)
                if int(keep.sum()) < 50:
                    return None
            residual = float(np.median(np.abs(resid)))
        if not np.isfinite(c).all() or not np.isfinite(residual):
            return None
        centre_3d = c[0] * u + c[1] * v
        # Same convention as fit_pipe_pose: the axis point nearest the camera
        # origin lies fully in the section plane, so this is already it.
        return np.asarray(centre_3d, dtype=np.float64), residual

    def _smooth_pipe_model(self, axis: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prev = self._pipe_model_filt
        axis = _normalize(np.asarray(axis, dtype=np.float64).reshape(3))
        point = np.asarray(point, dtype=np.float64).reshape(3)
        if prev is None:
            self._pipe_model_filt = {
                "axis": axis.copy(),
                "point": point.copy(),
                "lag": np.zeros(3, dtype=np.float64),
            }
            return axis, point
        prev_axis = prev["axis"]
        prev_point = prev["point"]
        if float(np.dot(axis, prev_axis)) < 0.0:
            axis = -axis
        angle_deg = math.degrees(
            math.acos(float(np.clip(abs(np.dot(axis, prev_axis)), -1.0, 1.0)))
        )
        gain = float(self.params["pipe_model_smooth_gain"])
        if angle_deg >= float(self.params["pipe_model_smooth_axis_snap_deg"]):
            axis_s = axis
        else:
            axis_s = _normalize(prev_axis * (1.0 - gain) + axis * gain)
        innovation = point - prev_point
        dist = float(np.linalg.norm(innovation))
        soft_m = float(self.params["pipe_model_smooth_point_soft_m"])
        # Same lag guard as the x and centre filters: a sustained one-sided
        # innovation is REAL motion (e.g. approaching the pipe), and without
        # the boost the filtered axis point trails by centimetres, rendering
        # a displaced/thinner silhouette (mask lost half its height on the
        # old bag's approach segments).
        lag = 0.7 * prev.get("lag", np.zeros(3)) + 0.3 * innovation
        if dist >= float(self.params["pipe_model_smooth_point_snap_m"]):
            point_s = point
            lag = np.zeros(3, dtype=np.float64)
        else:
            point_gain = gain if dist <= soft_m else max(0.25, gain * 0.6)
            if float(np.linalg.norm(lag)) > soft_m:
                point_gain = max(point_gain, 0.8)
            point_s = prev_point + point_gain * innovation
        self._pipe_model_filt = {"axis": axis_s.copy(), "point": point_s.copy(), "lag": lag}
        return axis_s, point_s

    def _cylinder_reproject_mask(
        self,
        selection_mask: np.ndarray,
        valid: np.ndarray,
        depth: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """Rebuild the pipe mask geometrically from the fitted cylinder.

        The near-depth threshold + component selection can cut the pipe (the
        close-up pipe spans the kmeans threshold itself) or drift onto an
        image-space band that leaves the pipe body. This stage goes the other
        way, as in classic cylinder RANSAC segmentation plus model rendering:

        1. Seed: valid-depth pixels near the surface of the best cylinder
           model available (this frame's component fit, else the temporal
           lock; a wide seed band absorbs lock axis_point bias).
        2. Refit the cylinder on the seed when the model needs it (lock
           source, or refit_when_selected); a fresh same-frame component fit
           is already exact, so by default it is reused as-is.
        3. Known-radius recenter: partial-arc circle fits inflate the radius
           and drag the centre (flying pixels behind the grazing edges); the
           radius is a task constant, so re-solve only the 2-DOF section
           centre with the radius pinned to nominal.
        4. Render-and-compare: analytic ray/cylinder intersection gives the
           exact silhouette of the cylinder, bounded by the axial extent of
           the observed surface. Measured depth in FRONT of the predicted
           surface is a real occluder and carves the mask. Invalid depth
           inside the silhouette stays pipe (specular holes, grazing edges).

        Point statistics (seed/surface/recenter) run on a spatially strided
        pixel grid; only the final silhouette render is full resolution.

        On failure of any gate the selection mask is returned unchanged.
        """
        info: dict[str, Any] = {"applied": False, "reason": "", "source": "none"}
        self.pipe_mask_reproject_info = info
        self._reproject_surface_points = None
        self._reproject_axial_extent = None
        if _fit_pipe_pose is None:
            info["reason"] = "fit_unavailable"
            return selection_mask
        model = self.pipe_component_selected_model
        source = "selected"
        if model is None and bool(self.params["pipe_mask_reproject_use_lock_model"]):
            model = self.pipe_lock_model
            source = "lock"
        if model is None:
            info["reason"] = "no_model"
            return selection_mask
        info["source"] = source
        try:
            axis = _normalize(np.asarray(model["axis"], dtype=np.float64).reshape(3))
            axis_point = np.asarray(model["axis_point"], dtype=np.float64).reshape(3)
            radius = float(model["radius_m"])
        except Exception:
            info["reason"] = "bad_model"
            return selection_mask
        if not np.isfinite(axis_point).all() or not np.isfinite(axis).all() or radius <= 1e-4:
            info["reason"] = "bad_model"
            return selection_mask

        min_pipe_pixels = int(self.params["min_pipe_pixels"])
        stride = max(1, int(self.params["pipe_mask_reproject_sample_stride"]))
        px_scale = stride * stride
        ys_s, xs_s = np.nonzero(valid[::stride, ::stride])
        ys_v = ys_s * stride
        xs_v = xs_s * stride
        if xs_v.size * px_scale < min_pipe_pixels:
            info["reason"] = "too_few_valid_depth"
            return selection_mask
        points = _backproject(depth, xs_v, ys_v, k)

        tol_abs = float(self.params["pipe_mask_reproject_radius_tol_abs_m"])
        tol_rel = float(self.params["pipe_mask_reproject_radius_tol_rel"])
        tol = max(tol_abs, tol_rel * max(radius, 1e-6))
        # The seed band is wider: a lock model whose axis_point came from the
        # tracker centroid sits near the surface rather than on the axis, and
        # the refit below straightens exactly that bias out.
        seed_tol = max(tol, float(self.params["pipe_mask_reproject_seed_radius_tol_m"]))

        def radial_to(axis_v: np.ndarray, point_v: np.ndarray) -> np.ndarray:
            # |q x d| for unit d, without allocating the cross product:
            # radial^2 = |q|^2 - (q . d)^2
            q = points - point_v[None, :]
            q2 = np.einsum("ij,ij->i", q, q)
            qd = q @ axis_v
            return np.sqrt(np.maximum(q2 - qd * qd, 0.0))

        radial = radial_to(axis, axis_point)
        seed_sel = np.abs(radial - radius) <= seed_tol
        # Color prior: clutter at the pipe's own depth (black cloth right
        # behind the top edge) lives INSIDE the radial seed band and drags
        # the refit/recenter up and down. Keep only warm points when enough
        # of them exist; fall back to the unfiltered band otherwise.
        warm_prior = getattr(self, "_warm_mask_current", None)
        warm_pts = warm_prior[ys_v, xs_v] if warm_prior is not None else None
        normal_pts_ok = None
        nrm_at = None
        if bool(self.params["pipe_normal_consistency_enabled"]) and warm_pts is None:
            nrm_img, ok_img, _grid = self._get_normal_image(depth, k, stride)
            nrm_at = nrm_img[ys_s, xs_s]
            ok_at = ok_img[ys_s, xs_s]
            cosr = self._normal_radial_cos(points, nrm_at, axis, axis_point)
            normal_pts_ok = ok_at & (cosr >= float(self.params["pipe_normal_cos_min"]))
        pipe_like_pts = None
        if warm_pts is not None and normal_pts_ok is not None:
            pipe_like_pts = warm_pts | normal_pts_ok
        elif warm_pts is not None:
            pipe_like_pts = warm_pts
        elif normal_pts_ok is not None:
            pipe_like_pts = normal_pts_ok
        if pipe_like_pts is not None:
            seed_like = seed_sel & pipe_like_pts
            if int(seed_like.sum()) * px_scale >= min_pipe_pixels:
                seed_sel = seed_like
        seed_px = int(seed_sel.sum()) * px_scale
        info["seed_px"] = seed_px
        if seed_px < min_pipe_pixels:
            info["reason"] = "seed_too_small"
            return selection_mask

        nominal = max(1e-6, float(self.params["pipe_radius_m"]))
        refit_model: dict[str, Any] = model
        run_refit = source == "lock" or bool(self.params["pipe_mask_reproject_refit_when_selected"])
        if run_refit:
            seed_mask = np.zeros_like(selection_mask, dtype=bool)
            seed_mask[ys_v[seed_sel], xs_v[seed_sel]] = True
            nominal_margin = max(
                float(self.params["cylinder_component_radius_abs_margin_m"]),
                float(self.params["cylinder_component_radius_rel_margin"]) * nominal,
            )
            fit_params = {
                "min_depth_m": float(self.params["min_depth_m"]),
                "max_depth_m": float(self.params["max_depth_m"]),
                "min_pipe_pixels": max(1, min_pipe_pixels // (4 * px_scale)),
                "max_fit_points": int(self.params["pipe_mask_reproject_max_fit_points"]),
                "sample_stride": int(self.params["sample_stride"]),
                "consensus_iterations": int(self.params["cylinder_component_consensus_iterations"]),
                "radius_tolerance_m": float(self.params["cylinder_component_radius_tolerance_m"]),
                "min_inliers": int(self.params["cylinder_component_min_inliers"]),
                "min_inlier_fraction": float(self.params["cylinder_component_min_inlier_fraction"]),
                "radius_min_m": max(1e-4, nominal - nominal_margin),
                "radius_max_m": nominal + nominal_margin,
                "max_residual_m": float(self.params["cylinder_component_max_residual_m"]),
                # The seed is already a clean cylinder-surface band: PCA gives
                # the axis reliably and the normal-field pass costs ~25 ms.
                "use_normal_axis": 1 if bool(self.params["pipe_mask_reproject_refit_use_normal_axis"]) else 0,
                "use_ransac_axis": 1 if bool(self.params["pipe_fit_use_ransac_axis"]) else 0,
            }
            try:
                fit = _fit_pipe_pose(depth, k, mask=seed_mask, params=fit_params)
            except Exception:
                info["reason"] = "refit_exception"
                return selection_mask
            if not bool(getattr(fit, "valid", False)):
                info["reason"] = f"refit_invalid:{getattr(fit, 'reason', '')}"
                return selection_mask
            fitted = self._pipe_fit_model(fit)
            if fitted is None:
                info["reason"] = "refit_model_none"
                return selection_mask
            refit_model = fitted
            if source == "lock":
                compatible, _score, _debug = self._pipe_lock_compatible_score(refit_model)
                if not compatible:
                    info["reason"] = "refit_incompatible_with_lock"
                    return selection_mask

        refit_axis = _normalize(np.asarray(refit_model["axis"], dtype=np.float64).reshape(3))
        refit_point = np.asarray(refit_model["axis_point"], dtype=np.float64).reshape(3)
        refit_radius = float(refit_model["radius_m"])
        tol_final = max(tol_abs, tol_rel * max(refit_radius, 1e-6))
        if run_refit:
            radial = radial_to(refit_axis, refit_point)
        surface_sel = np.abs(radial - refit_radius) <= tol_final
        surface_like = pipe_like_pts
        if nrm_at is not None and run_refit:
            # Re-evaluate the normal test against the refitted axis.
            cosr2 = self._normal_radial_cos(points, nrm_at, refit_axis, refit_point)
            normal_ok2 = (cosr2 >= float(self.params["pipe_normal_cos_min"]))
            surface_like = normal_ok2 if warm_pts is None else (warm_pts | normal_ok2)
        if surface_like is not None:
            surface_filtered = surface_sel & surface_like
            if int(surface_filtered.sum()) * px_scale >= min_pipe_pixels:
                surface_sel = surface_filtered
        if int(surface_sel.sum()) * px_scale < min_pipe_pixels:
            info["reason"] = "too_few_surface_inliers"
            return selection_mask
        surface_points = points[surface_sel]

        render_point = refit_point
        render_radius = refit_radius
        if bool(self.params["pipe_mask_reproject_known_radius_recenter"]):
            recentered = self._recenter_axis_known_radius(
                surface_points, refit_axis, refit_point, nominal
            )
            if recentered is not None:
                render_point, recenter_residual = recentered
                render_radius = nominal
                info["recentered"] = True
                info["recenter_residual_m"] = _round(recenter_residual, 4)
                refit_model = dict(refit_model)
                refit_model["axis_point"] = render_point
                refit_model["radius_m"] = float(nominal)
                refit_model["residual_m"] = float(recenter_residual)
        if bool(self.params["enable_pipe_model_smoothing"]):
            # The lock model used to be replaced wholesale every frame, so fit
            # noise went straight into mask -> strip -> junction. Same
            # innovation-gated filtering as the x and centre tracks.
            refit_axis, render_point = self._smooth_pipe_model(refit_axis, render_point)
            refit_model = dict(refit_model)
            refit_model["axis"] = refit_axis
            refit_model["axis_point"] = render_point
        render_radius += float(self.params["pipe_mask_reproject_render_radius_margin_m"])

        # Axial extent of the observed surface bounds the infinite fitted
        # cylinder so pixels beyond the real pipe ends are never claimed.
        axial_surface = (surface_points - render_point[None, :]) @ refit_axis
        s_lo, s_hi = np.percentile(axial_surface, [1.0, 99.0])
        # Raw observed extent (pre-margin): the junction acceptance uses it to
        # tell a pipe END (surface stops right after the candidate) from a
        # junction (pipe continues on both sides).
        self._reproject_axial_extent = (float(s_lo), float(s_hi))
        axial_margin = float(self.params["pipe_mask_reproject_axial_margin_m"])
        s_lo -= axial_margin
        s_hi += axial_margin

        # Render-and-compare: analytic ray/cylinder intersection per pixel.
        # A pixel belongs to the pipe if its ray hits the cylinder within the
        # axial extent AND the measured depth does not sit in front of the
        # predicted surface (a real occluder carves the mask). Invalid depth
        # inside the silhouette stays pipe: that is exactly the specular-hole
        # / grazing-edge case the depth test cannot judge.
        image_h, image_w = selection_mask.shape[:2]
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        ray_key = (image_w, image_h, round(fx, 6), round(fy, 6), round(cx, 6), round(cy, 6))
        cached = getattr(self, "_reproject_ray_cache", None)
        if cached is not None and cached[0] == ray_key:
            ray_x, ray_y = cached[1], cached[2]
        else:
            us = ((np.arange(image_w, dtype=np.float32) - np.float32(cx)) / np.float32(fx))
            vs = ((np.arange(image_h, dtype=np.float32) - np.float32(cy)) / np.float32(fy))
            ray_x, ray_y = np.meshgrid(us, vs)
            self._reproject_ray_cache = (ray_key, ray_x, ray_y)
        # Unnormalized ray direction (x, y, 1): the line parameter is then the
        # metric depth z directly, comparable with the depth image.
        d0, d1, d2 = (np.float32(refit_axis[0]), np.float32(refit_axis[1]), np.float32(refit_axis[2]))
        w_x = ray_y * d2 - d1
        w_y = d0 - ray_x * d2
        w_z = ray_x * d1 - ray_y * d0
        q_vec = np.cross(render_point, refit_axis).astype(np.float32)
        a_coef = w_x * w_x + w_y * w_y + w_z * w_z
        b_coef = -2.0 * (w_x * q_vec[0] + w_y * q_vec[1] + w_z * q_vec[2])
        c_coef = np.float32(float(q_vec @ q_vec) - render_radius * render_radius)
        disc = b_coef * b_coef - 4.0 * a_coef * c_coef
        hit = (disc >= 0.0) & (a_coef > 1e-12)
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_front = (-b_coef - sqrt_disc) / (2.0 * a_coef)
            z_back = (-b_coef + sqrt_disc) / (2.0 * a_coef)
        hit &= z_front > np.float32(self.params["min_depth_m"])
        # Axial coordinate of the front intersection point.
        p0, p1, p2 = (np.float32(render_point[0]), np.float32(render_point[1]), np.float32(render_point[2]))
        s_front = (
            (z_front * ray_x - p0) * d0
            + (z_front * ray_y - p1) * d1
            + (z_front - p2) * d2
        )
        hit &= (s_front >= np.float32(s_lo)) & (s_front <= np.float32(s_hi))

        occl_tol = float(self.params["pipe_mask_reproject_occlusion_tol_m"])
        depth_f = depth.astype(np.float32, copy=False)
        occluder = valid & hit & (depth_f < (z_front - np.float32(occl_tol)))
        # Bounded include-behind: the blanket keep-everything-behind policy
        # was meant for 10-15 px of aligned-depth bleed at the grazing edges,
        # but at close range the silhouette band is huge and it swallowed
        # bricks/floor/cloth (measured 40% non-pipe pixels). Keep the edge
        # ring unconditionally (bleed zone); the interior must sit near the
        # predicted surface or look like the pipe (color prior).
        ring_px = int(self.params["pipe_mask_reproject_edge_ring_px"])
        if ring_px > 0 and cv2 is not None:
            ring_kernel = np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)
            inner = cv2.erode(hit.astype(np.uint8), ring_kernel).astype(bool)
        else:
            inner = hit
        use_warm_interior = warm_prior is not None and getattr(self, "_warm_scene_ok", False)
        use_normal_interior = (
            bool(self.params["pipe_normal_consistency_enabled"])
            and cv2 is not None
            and not use_warm_interior
        )
        if not use_warm_interior and not use_normal_interior:
            interior_keep = inner
        else:
            # Interior = near the predicted surface AND pipe-looking, where
            # pipe-looking is warm (task color prior) OR normal-consistent
            # (COLOR-INDEPENDENT: cloth/brick normals point at the camera or
            # upward, true pipe normals are radial around the axis). The
            # plain near-surface OR admitted the brick past the pipe end.
            behind_tol_i = np.float32(self.params["pipe_mask_reproject_interior_behind_tol_m"])
            near_surface = valid & (depth_f <= (z_front + behind_tol_i))
            keep = np.zeros_like(inner)
            if use_warm_interior:
                keep |= warm_prior & (near_surface | ~valid)
            if use_normal_interior:
                nrm_img_i, ok_img_i, grid_i = self._get_normal_image(depth, k, stride)
                cos_grid = self._normal_radial_cos(
                    grid_i.reshape(-1, 3), nrm_img_i.reshape(-1, 3), refit_axis, render_point
                ).reshape(ok_img_i.shape)
                normal_strided = ok_img_i & (cos_grid >= float(self.params["pipe_normal_cos_min"]))
                normal_full = cv2.resize(
                    normal_strided.astype(np.uint8), (image_w, image_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                keep |= normal_full & near_surface
            interior_keep = inner & keep
        # A silhouette whose interior keeps (almost) nothing is a model
        # pointing at something that is not the pipe (e.g. the camera panned
        # onto the cloth): a ring-only mask would degenerate the seam strip.
        if int(interior_keep.sum()) < min_pipe_pixels // 2:
            info["reason"] = "interior_empty"
            return selection_mask
        final_mask = ((hit & ~inner) | interior_keep) & ~occluder
        if bool(self.params["pipe_mask_reproject_exclude_behind"]):
            # Off by default: aligned depth bleeds background values over the
            # grazing top/bottom edges of the pipe (10-15 px at ~1 m), so a
            # behind-the-surface exclusion erases real pipe there. The axial
            # bound already stops the mask past the pipe ends; enable this
            # only for scenes with a visible open pipe end inside the view.
            behind_tol = np.float32(self.params["pipe_mask_reproject_behind_tol_m"])
            final_mask &= ~(valid & hit & (depth_f > (z_back + behind_tol)))

        # Drop small blobs that match the radius by coincidence (block edges,
        # far clutter grazing the cylinder surface).
        labels, count = scipy_ndimage.label(final_mask)
        if count == 0:
            info["reason"] = "empty_after_reproject"
            return selection_mask
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        largest = int(sizes.max())
        keep_min = max(
            int(self.params["pipe_mask_reproject_min_component_px"]),
            int(float(self.params["pipe_mask_reproject_component_keep_frac"]) * largest),
        )
        keep_labels = np.flatnonzero(sizes >= keep_min)
        final_mask = np.isin(labels, keep_labels)
        inlier_px = int(final_mask.sum())
        info["inlier_px"] = inlier_px
        if inlier_px < min_pipe_pixels:
            info["reason"] = "too_few_inliers"
            return selection_mask

        close_px = int(self.params["pipe_mask_reproject_close_px"])
        if close_px > 1:
            if cv2 is not None:
                kernel = np.ones((close_px, close_px), np.uint8)
                final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            else:
                structure = np.ones((close_px, close_px), dtype=bool)
                final_mask = scipy_ndimage.binary_closing(final_mask, structure=structure)
        if bool(self.params["pipe_mask_reproject_fill_holes"]):
            final_mask = scipy_ndimage.binary_fill_holes(final_mask)

        self.pipe_component_selected_model = refit_model
        self._reproject_surface_points = surface_points
        info.update(
            {
                "applied": True,
                "reason": "ok",
                "refit": bool(run_refit),
                "radius_m": _round(float(refit_model["radius_m"]), 4),
                "residual_m": _round(float(refit_model.get("residual_m", 0.0)), 4),
                "tol_m": _round(tol_final, 4),
                "mask_px": int(final_mask.sum()),
            }
        )
        return final_mask

    def _project_axis_to_image(
        self, model: dict[str, Any], k: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Image-space pipe axis by PROJECTING the fitted 3D axis.

        The drawn axis / strip rotation used to come from the 2D PCA of the
        pipe mask, so any mask impurity (cloth ring, brick) visibly tilted
        the axis even when the 3D cylinder model was correct. Projecting the
        (smoothed, recentered) model makes the axis immune to the mask: the
        brick can stay in the visual mask without deforming anything.
        """
        try:
            axis = _normalize(np.asarray(model["axis"], dtype=np.float64).reshape(3))
            point = np.asarray(model["axis_point"], dtype=np.float64).reshape(3)
        except Exception:
            return None
        if not np.isfinite(axis).all() or not np.isfinite(point).all():
            return None
        half = 0.3
        p0 = point - half * axis
        p1 = point + half * axis
        if p0[2] <= 0.05 or p1[2] <= 0.05 or point[2] <= 0.05:
            return None
        uv = _project(np.stack([p0, p1]), k)
        d = uv[1] - uv[0]
        norm = float(np.linalg.norm(d))
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        direction = d / norm
        if direction[0] < 0.0:
            direction = -direction
        centre = _project(point.reshape(1, 3), k)[0]
        if not np.isfinite(centre).all() or not np.isfinite(direction).all():
            return None
        return centre, direction

    def _track_pipe(self, depth: np.ndarray, k: np.ndarray) -> TrackerResult:
        self.pipe_mask_reproject_info = {"applied": False, "reason": "not_run", "source": "none"}
        min_depth = float(self.params["min_depth_m"])
        max_depth = float(self.params["max_depth_m"])
        valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        threshold_info = _depth_threshold_kmeans(depth, valid, int(self.params["sample_stride"]))
        pipe_mask_raw = valid & (depth <= float(threshold_info["threshold_m"]))
        pipe_mask: np.ndarray | None = None
        fast_path_used = False
        reanchor_every = int(self.params["pipe_mask_fast_path_reanchor_every"])
        reanchor_frame = reanchor_every > 0 and (self.processed_frame_count % reanchor_every == 0)
        if (
            bool(self.params["enable_pipe_mask_cylinder_reproject"])
            and bool(self.params["pipe_mask_fast_path_when_locked"])
            and bool(self.params["enable_pipe_temporal_lock"])
            and self.pipe_lock_model is not None
            and not reanchor_frame
        ):
            # Fast path while the metric lock holds: the reprojection stage can
            # seed straight from the lock model over the full image, so the
            # component/band selection machinery (the most expensive stage) is
            # only needed for acquisition or when the lock-seeded refit fails
            # its compatibility gates (then we fall through to it unchanged).
            self.pipe_component_selected_model = None
            self.pipe_component_selected_image_model = None
            candidate_mask = self._cylinder_reproject_mask(
                np.zeros_like(valid, dtype=bool), valid, depth, k
            )
            if bool(self.pipe_mask_reproject_info.get("applied")):
                pipe_mask = candidate_mask
                fast_path_used = True
                self.pipe_component_selection_info = {
                    "method": "lock_seed_fast_path",
                    "component_count": None,
                    "selected_label": None,
                }
        if pipe_mask is None:
            pipe_mask = self._select_pipe_connected_component(pipe_mask_raw, depth, k)
            if bool(self.params["enable_pipe_mask_cylinder_reproject"]):
                pipe_mask = self._cylinder_reproject_mask(pipe_mask, valid, depth, k)
            else:
                self.pipe_mask_reproject_info = {"applied": False, "reason": "disabled", "source": "none"}
        self.pipe_mask_warm_fraction = self._mask_warm_fraction(pipe_mask, self._warm_mask_current)
        self._update_coloroff_normal_stats(pipe_mask, depth, k)
        low_warm = (
            self._warm_scene_ok
            and self.pipe_mask_warm_fraction is not None
            and self.pipe_mask_warm_fraction < float(self.params["junction_min_mask_warm_fraction"])
        )
        low_normal = (
            self.pipe_mask_normal_fraction is not None
            and self.pipe_mask_normal_fraction < float(self.params["pipe_mask_min_normal_fraction"])
        )
        # When the color-less FSM is active it OWNS every lock release (HOLD ->
        # LOST after pipe_coloroff_release_frames); the v5 low_warm/low_normal
        # escape below must not fire in parallel. It stays the release path in
        # warm/color-ON mode, where the FSM only mirrors the state.
        fsm_owns_release = (
            bool(self.params["pipe_coloroff_cylinder_guard_enabled"])
            and bool(self.params["pipe_normal_consistency_enabled"])
            and not self._warm_scene_ok
        )
        if (
            (low_warm or low_normal)
            and self.pipe_lock_model is not None
            and not fsm_owns_release
        ):
            self._low_warm_streak += 1
            if self._low_warm_streak >= int(self.params["pipe_lock_low_warm_release_frames"]):
                # Impostor escape: a locked "pipe" whose mask stays this
                # un-pipe-colored is cloth/clutter. Drop every lock and
                # re-acquire from scratch instead of self-confirming forever.
                self.pipe_lock_model = None
                self.pipe_image_lock_model = None
                self.pipe_lock_source = "released:low_warm_mask"
                self.pipe_image_lock_source = "released:low_warm_mask"
                self._pipe_model_filt = None
                self._release_junction_lock("low_warm_mask")
                self._low_warm_streak = 0
        elif not fsm_owns_release:
            # Decay instead of hard reset: a mask oscillating between the
            # pipe sliver and the cloth must still accumulate towards the
            # escape, otherwise the impostor survives indefinitely.
            self._low_warm_streak = max(0, self._low_warm_streak - 1)

        # ---- Explicit ACQUIRE/TRACK/HOLD/LOST tracker FSM ----------------
        pipe_mask = self._advance_pipe_tracker_state(
            depth, k, valid, pipe_mask, low_warm, low_normal
        )
        ys, xs = np.nonzero(pipe_mask)
        if xs.size < int(self.params["min_pipe_pixels"]):
            raise ValueError(f"Only {xs.size} pipe pixels selected")
        self._pipe_mask_col_extent = (int(xs.min()), int(xs.max()), int(pipe_mask.shape[1]))
        self._update_pipe_end_memory(k)

        # Morphological hole filling can add pixels without measured depth
        # (specular holes): the 3D fit must only see measured pixels, while the
        # image-space axis/bbox/strip keep the complete mask.
        fit_keep = valid[ys, xs]
        ys_fit = ys[fit_keep]
        xs_fit = xs[fit_keep]
        if xs_fit.size < max(3, int(self.params["min_pipe_pixels"]) // 4):
            ys_fit, xs_fit = ys, xs

        reuse_model: dict[str, Any] | None = None
        if (
            bool(self.params["pipe_pose_reuse_reproject_model"])
            and bool(self.pipe_mask_reproject_info.get("applied"))
            and self.pipe_component_selected_model is not None
            and self._reproject_surface_points is not None
            and self._reproject_surface_points.shape[0] >= 64
        ):
            reuse_model = self.pipe_component_selected_model

        if reuse_model is not None:
            # The reprojection stage just fitted (and recentered) this very
            # surface: running the consensus axis fit again on the same pixels
            # costs ~20 ms for the same answer.
            reuse_points = self._reproject_surface_points
            xyz_fit = {
                "centroid": np.median(reuse_points, axis=0),
                "eigenvalues": None,
                "direction": np.asarray(reuse_model["axis"], dtype=np.float64),
                "points": reuse_points,
                "inlier_mask": np.ones(reuse_points.shape[0], dtype=bool),
                "inlier_count": int(reuse_points.shape[0]),
                "inlier_fraction": float(reuse_model.get("inlier_fraction", 1.0)),
                "radius_m": float(reuse_model.get("radius_m", 0.0)),
                "residual_m": float(reuse_model.get("residual_m", 0.0)),
                "method": "reproject_model",
            }
        else:
            max_pca_points = int(self.params["max_pca_points"])
            if xs_fit.size > max_pca_points:
                indices = np.linspace(0, xs_fit.size - 1, max_pca_points, dtype=np.int64)
                xs_pca = xs_fit[indices]
                ys_pca = ys_fit[indices]
            else:
                xs_pca = xs_fit
                ys_pca = ys_fit
            points_camera = _backproject(depth, xs_pca, ys_pca, k)
            if bool(self.params["use_cylinder_consensus_pipe_pose"]):
                xyz_fit = _cylinder_consensus_axis_fit(
                    points_camera,
                    max_iterations=int(self.params["pipe_pose_consensus_iterations"]),
                    radius_tolerance_m=float(self.params["pipe_pose_radius_tolerance_m"]),
                    min_inliers=int(self.params["pipe_pose_min_inliers"]),
                    min_inlier_fraction=float(self.params["pipe_pose_min_inlier_fraction"]),
                )
            else:
                xyz_pca = _pca(points_camera)
                xyz_fit = {
                    "centroid": xyz_pca["centroid"],
                    "eigenvalues": xyz_pca["eigenvalues"],
                    "direction": xyz_pca["direction"],
                    "points": points_camera,
                    "inlier_mask": np.ones(points_camera.shape[0], dtype=bool),
                    "inlier_count": int(points_camera.shape[0]),
                    "inlier_fraction": 1.0,
                    "radius_m": 0.0,
                    "residual_m": 0.0,
                    "method": "pca",
                }

        image_centroid: np.ndarray | None = None
        image_direction: np.ndarray | None = None
        if bool(self.params["pipe_image_axis_from_model"]) and self.pipe_component_selected_model is not None:
            projected = self._project_axis_to_image(self.pipe_component_selected_model, k)
            if projected is not None:
                image_centroid, image_direction = projected
        if image_direction is None:
            uv_pca = _pca(np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1))
            image_direction = _normalize(uv_pca["direction"])
            image_centroid = np.asarray(uv_pca["centroid"], dtype=np.float64)
        if image_direction[0] < 0.0:
            image_direction = -image_direction
        image_axis_angle_deg = math.degrees(math.atan2(float(image_direction[1]), float(image_direction[0])))

        axis_camera = _normalize(xyz_fit["direction"])
        if axis_camera[0] < 0.0:
            axis_camera *= -1.0
        pipe_depths = depth[ys_fit, xs_fit]
        fit_points = np.asarray(xyz_fit["points"], dtype=np.float64)
        fit_inliers = np.asarray(xyz_fit["inlier_mask"], dtype=bool)
        inlier_points = fit_points[fit_inliers] if fit_inliers.size == fit_points.shape[0] else fit_points
        if inlier_points.shape[0] < 3:
            inlier_points = fit_points
        fit_centroid = np.median(inlier_points, axis=0) if inlier_points.shape[0] else xyz_fit["centroid"]
        yaw_error_deg = math.degrees(math.atan2(float(axis_camera[2]), float(axis_camera[0])))
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        if fast_path_used:
            # Keep the image-space lock fresh too: the rendered silhouette is
            # the strongest band evidence there is, so publish its axis line
            # as this frame's image model (used by the selection fallback).
            self.pipe_component_selected_image_model = self._pipe_image_model_from_candidate(
                {
                    "band_line_point_uv": np.asarray(image_centroid, dtype=np.float64),
                    "band_line_direction_uv": image_direction,
                    "band_depth_median_m": float(np.median(pipe_depths)) if pipe_depths.size else float("nan"),
                    "band_width_fraction": float(x1 - x0 + 1) / max(1.0, float(pipe_mask.shape[1])),
                    "band_score": 2.0 * float(self.params["pipe_image_lock_min_band_score"]),
                    "label": -2,
                }
            )

        tracker = TrackerResult(
            pipe_mask=pipe_mask,
            pipe_pixels=int(pipe_mask.sum()),
            pipe_fraction=float(pipe_mask.mean()),
            bbox_uv=[x0, y0, x1, y1],
            image_centroid_uv=image_centroid,
            image_direction_uv=image_direction,
            image_axis_angle_deg=image_axis_angle_deg,
            image_line_segment_uv=_line_box_segment(image_centroid, image_direction, depth.shape[1], depth.shape[0]),
            centroid_xyz_m=np.asarray(fit_centroid, dtype=np.float64),
            pipe_axis_xyz=axis_camera,
            stand_off_m=float(np.median(inlier_points[:, 2])) if inlier_points.shape[0] else float(np.median(pipe_depths)),
            lateral_offset_m=float(fit_centroid[0]),
            vertical_offset_m=float(fit_centroid[1]),
            yaw_error_deg=yaw_error_deg,
            pipe_pose_fit_method=str(xyz_fit["method"]),
            pipe_pose_inlier_count=int(xyz_fit["inlier_count"]),
            pipe_pose_inlier_fraction=float(xyz_fit["inlier_fraction"]),
            pipe_pose_radius_m=float(xyz_fit["radius_m"]),
            pipe_pose_residual_m=float(xyz_fit["residual_m"]),
            threshold_info=threshold_info,
        )
        return tracker

    def _select_pipe_connected_component(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """Keep the connected depth component that best matches the pipe body.

        The raw near-depth mask can include small disconnected pieces from the
        robot, cut edges, sensor artifacts, or background fragments. Treating all
        those pixels as one "pipe" destabilizes PCA/coverage/localization.

        The real pipe is not always the largest component in Gazebo: a nearby
        robot, wall, or floor fragment can be larger. Before a pipe is locked,
        candidates are validated by cylinder fit and projected diameter. After
        a lock exists, the selected component must be compatible with the
        previous pipe model (axis, radius, depth, and 3D position); otherwise
        the pipe is marked invalid instead of jumping to another object.
        """
        self.pipe_component_selected_model = None
        self.pipe_component_selected_image_model = None
        if mask.ndim != 2 or not np.any(mask):
            self.pipe_component_selection_info = {
                "method": "raw_mask_empty_or_invalid",
                "component_count": 0,
                "selected_label": None,
            }
            return mask
        min_pixels = max(1, int(self.params["min_pipe_pixels"]) // 4)

        structure = np.array(
            [
                [False, True, False],
                [True, True, True],
                [False, True, False],
            ],
            dtype=bool,
        )
        labels, count = scipy_ndimage.label(mask, structure=structure)
        if count == 0:
            self.pipe_component_selection_info = {
                "method": "no_connected_component",
                "component_count": int(count),
                "selected_label": None,
            }
            return mask
        sizes = np.bincount(labels.ravel())
        if sizes.size <= 1:
            self.pipe_component_selection_info = {
                "method": "label_failure",
                "component_count": int(count),
                "selected_label": None,
            }
            return mask
        sizes[0] = 0

        fy = float(k[1, 1]) if k.shape == (3, 3) else 0.0
        radius_m = float(self.params["pipe_radius_m"])
        image_h, image_w = mask.shape[:2]
        shape_prior_enabled = bool(self.params.get("pipe_component_shape_prior_enabled", False))
        min_width_fraction = float(self.params.get("pipe_component_min_width_fraction", 0.28))
        bottom_margin_fraction = float(self.params.get("pipe_component_bottom_margin_fraction", 0.05))
        bottom_allow_width_fraction = float(self.params.get("pipe_component_bottom_allow_width_fraction", 0.55))
        candidates_all: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        rejected_by_shape = 0
        for label in range(1, sizes.size):
            count_i = int(sizes[label])
            if count_i < min_pixels:
                continue
            ys, xs = np.nonzero(labels == label)
            if xs.size == 0:
                continue
            z = depth[ys, xs].astype(np.float64)
            z = z[np.isfinite(z) & (z > 0.0)]
            if z.size == 0:
                continue
            bbox_w = float(xs.max() - xs.min() + 1)
            bbox_h = float(ys.max() - ys.min() + 1)
            median_z = float(np.median(z))
            expected_diameter_px = 0.0
            if fy > 1e-6 and radius_m > 1e-6 and median_z > 1e-6:
                expected_diameter_px = 2.0 * radius_m * fy / median_z
            if expected_diameter_px > 1.0 and bbox_h > 1.0:
                height_error = abs(math.log(max(bbox_h, 1.0) / expected_diameter_px))
                height_score = math.exp(-height_error)
            else:
                height_score = 1.0
            component_mask = labels == label
            # Color prior: clutter at the SAME depth (black cloth, bricks)
            # fuses with the pipe into one component; evaluating the warm
            # subset lets the band/fit machinery find the pipe INSIDE the
            # blob. Falls back to the full component when the warm subset is
            # too small (prior disabled, non-orange pipe, heavy shading).
            warm_prior = getattr(self, "_warm_mask_current", None) if getattr(self, "_warm_scene_ok", False) else None
            eval_mask = component_mask
            component_warm_fraction: float | None = None
            if warm_prior is not None:
                warm_inter = component_mask & warm_prior
                n_warm = int(warm_inter.sum())
                component_warm_fraction = n_warm / max(1, count_i)
                if n_warm >= max(min_pixels, int(count_i * float(self.params["pipe_color_min_seed_px_frac"]))):
                    eval_mask = warm_inter
            band_info: dict[str, Any] = {
                "valid": False,
                "reason": "disabled",
                "score": 0.0,
                "mask": eval_mask,
            }
            selection_mask = eval_mask
            width_fraction = bbox_w / max(float(image_w), 1.0)
            touches_bottom = float(ys.max()) >= (1.0 - bottom_margin_fraction) * max(float(image_h - 1), 1.0)
            # Width matters because the pipe spans the camera horizontally in
            # this task, but size is damped so large background components do
            # not dominate when their projected height is wrong.
            score = bbox_w * math.sqrt(float(count_i)) * height_score
            if component_warm_fraction is not None:
                # Prefer components that look like the pipe; never a hard veto
                # here (acquisition still needs a fallback path).
                score *= 0.25 + 0.75 * component_warm_fraction
            if bool(self.params["pipe_component_use_band_filter"]):
                band_info = self._pipe_band_candidate(eval_mask, depth, expected_diameter_px)
                if bool(band_info.get("valid", False)):
                    selection_mask = np.asarray(band_info["mask"], dtype=bool)
                    score = max(score, float(band_info.get("score", 0.0)))
            candidate = {
                "label": int(label),
                "count": count_i,
                "score": float(score),
                "bbox_w": float(bbox_w),
                "bbox_h": float(bbox_h),
                "height_score": float(height_score),
                "width_fraction": float(width_fraction),
                "touches_bottom": bool(touches_bottom),
                "selection_mask": selection_mask,
                "band_valid": bool(band_info.get("valid", False)),
                "band_reason": band_info.get("reason"),
                "band_method": band_info.get("method"),
                "band_score": float(band_info.get("score", 0.0)),
                "band_pixels": int(band_info.get("pixels", 0)),
                "band_width_fraction": float(band_info.get("width_fraction", 0.0)),
                "band_column_coverage": float(band_info.get("column_coverage", 0.0)),
                "band_density": float(band_info.get("density", 0.0)),
                "band_half_width_px": float(band_info.get("half_width_px", 0.0)),
                "band_depth_median_m": float(band_info.get("depth_median_m", median_z)),
                "band_line_point_uv": band_info.get("line_point_uv"),
                "band_line_direction_uv": band_info.get("line_direction_uv"),
                "warm_fraction": component_warm_fraction,
            }
            candidates_all.append(candidate)
            shape_rejected = shape_prior_enabled and (
                width_fraction < min_width_fraction
                or (touches_bottom and width_fraction < bottom_allow_width_fraction)
            )
            if shape_rejected:
                rejected_by_shape += 1
                continue
            candidates.append(candidate)

        # If every component fails the shape prior, fall back to the old
        # candidates instead of crashing the detector. This preserves operation
        # in unusual views, but status exposes rejected_by_shape so the case is
        # visible during debugging.
        if not candidates and candidates_all:
            candidates = candidates_all

        if not candidates:
            self.pipe_component_selection_info = {
                "method": "no_large_component",
                "component_count": int(count),
                "selected_label": None,
                "rejected_by_shape": int(rejected_by_shape),
            }
            return mask
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        fallback_label = int(candidates[0]["label"])
        cylinder_evaluated = 0
        cylinder_valid = 0

        if bool(self.params.get("use_cylinder_component_selection", True)) and _fit_pipe_pose is not None:
            best_label = 0
            best_score = -float("inf")
            best_model: dict[str, Any] | None = None
            fit_candidates: list[dict[str, Any]] = []
            metric_lock_active = (
                bool(self.params["enable_pipe_temporal_lock"])
                and self.pipe_lock_model is not None
            )
            image_lock_active = (
                bool(self.params["enable_pipe_image_temporal_lock"])
                and self.pipe_image_lock_model is not None
            )
            lock_active = metric_lock_active or image_lock_active
            max_components = max(1, int(self.params["cylinder_component_max_components"]))
            nominal_radius = max(1e-6, float(self.params["pipe_radius_m"]))
            radius_abs_margin = float(self.params["cylinder_component_radius_abs_margin_m"])
            radius_rel_margin = float(self.params["cylinder_component_radius_rel_margin"])
            radius_tolerance_m = float(self.params["cylinder_component_radius_tolerance_m"])
            min_inliers = int(self.params["cylinder_component_min_inliers"])
            min_inlier_fraction = float(self.params["cylinder_component_min_inlier_fraction"])
            max_residual_m = float(self.params["cylinder_component_max_residual_m"])
            if lock_active:
                radius_abs_margin = float(self.params["pipe_tracking_component_radius_abs_margin_m"])
                radius_rel_margin = float(self.params["pipe_tracking_component_radius_rel_margin"])
                radius_tolerance_m = float(self.params["pipe_tracking_component_radius_tolerance_m"])
                min_inliers = int(self.params["pipe_tracking_component_min_inliers"])
                min_inlier_fraction = float(self.params["pipe_tracking_component_min_inlier_fraction"])
                max_residual_m = float(self.params["pipe_tracking_component_max_residual_m"])
            radius_margin = max(
                radius_abs_margin,
                radius_rel_margin * nominal_radius,
            )
            fit_params = {
                "min_depth_m": float(self.params["min_depth_m"]),
                "max_depth_m": float(self.params["max_depth_m"]),
                "min_pipe_pixels": min_pixels,
                "max_fit_points": int(self.params["cylinder_component_max_fit_points"]),
                "use_ransac_axis": 1 if bool(self.params["pipe_fit_use_ransac_axis"]) else 0,
                "sample_stride": int(self.params["sample_stride"]),
                "consensus_iterations": int(self.params["cylinder_component_consensus_iterations"]),
                "radius_tolerance_m": radius_tolerance_m,
                "min_inliers": min_inliers,
                "min_inlier_fraction": min_inlier_fraction,
                "radius_min_m": max(1e-4, nominal_radius - radius_margin),
                "radius_max_m": nominal_radius + radius_margin,
                "max_residual_m": max_residual_m,
            }
            for candidate in candidates[:max_components]:
                label = int(candidate["label"])
                component_mask = np.asarray(candidate.get("selection_mask", labels == label), dtype=bool)
                cylinder_evaluated += 1
                try:
                    fit = _fit_pipe_pose(depth, k, mask=component_mask, params=fit_params)
                except Exception:
                    continue
                if not bool(getattr(fit, "valid", False)):
                    continue
                cylinder_valid += 1
                radius_error = abs(float(getattr(fit, "radius_m", nominal_radius)) - nominal_radius)
                radius_score = math.exp(-radius_error / max(radius_margin, 1e-6))
                residual = max(0.0, float(getattr(fit, "residual_m", 0.0)))
                residual_score = math.exp(-residual / max(max_residual_m, 1e-6))
                axis = np.asarray(getattr(fit, "axis_camera_xyz", np.zeros(3)), dtype=np.float64)
                axis = _normalize(axis)
                cylinder_score = (
                    float(candidate["score"])
                    * max(0.05, float(getattr(fit, "inlier_fraction", 0.0)))
                    * radius_score
                    * residual_score
                )
                fit_model = self._pipe_fit_model(fit)
                fit_record = {
                    **candidate,
                    "fit": fit,
                    "fit_model": fit_model,
                    "selection_mask": component_mask,
                    "cylinder_score": float(cylinder_score),
                    "radius_m": float(getattr(fit, "radius_m", 0.0)),
                    "residual_m": residual,
                    "inlier_fraction": float(getattr(fit, "inlier_fraction", 0.0)),
                }
                fit_candidates.append(fit_record)
                if cylinder_score > best_score:
                    best_score = cylinder_score
                    best_label = label
                    best_model = fit_model

            if lock_active:
                best_temporal: dict[str, Any] | None = None
                best_temporal_score = -float("inf")
                best_temporal_debug: dict[str, float] = {}
                best_rejected_score = -float("inf")
                best_rejected_debug: dict[str, float] = {}
                if metric_lock_active:
                    for candidate in fit_candidates:
                        model = candidate.get("fit_model")
                        if not self._pipe_model_valid_for_lock(model):
                            continue
                        compatible, compat_score, compat_debug = self._pipe_lock_compatible_score(model)
                        if compat_score > best_rejected_score:
                            best_rejected_score = compat_score
                            best_rejected_debug = compat_debug
                        if not compatible:
                            continue
                        temporal_score = compat_score * max(0.05, float(candidate["inlier_fraction"]))
                        if temporal_score > best_temporal_score:
                            best_temporal_score = temporal_score
                            best_temporal = candidate
                            best_temporal_debug = compat_debug
                if (
                    best_temporal is not None
                    and best_temporal_score >= float(self.params["pipe_lock_min_compatibility_score"])
                ):
                    selected_label = int(best_temporal["label"])
                    self.pipe_component_selected_model = best_temporal.get("fit_model")
                    self.pipe_component_selected_image_model = self._pipe_image_model_from_candidate(best_temporal)
                    self.pipe_component_selection_info = {
                        "method": "temporal_pipe_lock",
                        "component_count": int(count),
                        "candidate_count": int(len(candidates)),
                        "rejected_by_shape": int(rejected_by_shape),
                        "cylinder_evaluated": int(cylinder_evaluated),
                        "cylinder_valid": int(cylinder_valid),
                        "selected_label": selected_label,
                        "fallback_label": int(fallback_label),
                        "score": _round(float(best_temporal_score)),
                        "band_valid": bool(best_temporal.get("band_valid", False)),
                        "band_score": _round(float(best_temporal.get("band_score", 0.0))),
                        "band_width_fraction": _round(float(best_temporal.get("band_width_fraction", 0.0))),
                        "band_column_coverage": _round(float(best_temporal.get("band_column_coverage", 0.0))),
                        "band_pixels": int(best_temporal.get("band_pixels", 0)),
                        "band_method": best_temporal.get("band_method"),
                        "lock_axis_delta_deg": _round(best_temporal_debug.get("axis_delta_deg")),
                        "lock_radius_delta_m": _round(best_temporal_debug.get("radius_delta_m")),
                        "lock_stand_delta_m": _round(best_temporal_debug.get("stand_delta_m")),
                        "lock_axis_point_delta_m": _round(best_temporal_debug.get("axis_point_delta_m")),
                    }
                    return np.asarray(best_temporal.get("selection_mask", labels == selected_label), dtype=bool)

                best_image: dict[str, Any] | None = None
                best_image_model: dict[str, Any] | None = None
                best_image_score = -float("inf")
                best_image_debug: dict[str, float] = {}
                best_image_rejected_score = -float("inf")
                best_image_rejected_debug: dict[str, float] = {}
                if image_lock_active:
                    for candidate in candidates:
                        if not bool(candidate.get("band_valid", False)):
                            continue
                        model = self._pipe_image_model_from_candidate(candidate)
                        if not self._pipe_image_model_valid_for_lock(model):
                            continue
                        assert model is not None
                        compatible, compat_score, compat_debug = self._pipe_image_lock_compatible_score(model)
                        if compat_score > best_image_rejected_score:
                            best_image_rejected_score = compat_score
                            best_image_rejected_debug = compat_debug
                        if not compatible:
                            continue
                        image_score = compat_score * max(
                            0.05,
                            min(
                                1.0,
                                float(candidate.get("band_score", 0.0))
                                / max(float(self.params["pipe_component_band_fallback_min_score"]), 1e-6),
                            ),
                        )
                        if image_score > best_image_score:
                            best_image_score = image_score
                            best_image = candidate
                            best_image_model = model
                            best_image_debug = compat_debug
                    if (
                        best_image is not None
                        and best_image_model is not None
                        and best_image_score >= float(self.params["pipe_image_lock_min_compatibility_score"])
                    ):
                        selected_label = int(best_image["label"])
                        self.pipe_component_selected_image_model = best_image_model
                        self.pipe_component_selection_info = {
                            "method": "temporal_pipe_image_lock",
                            "component_count": int(count),
                            "candidate_count": int(len(candidates)),
                            "rejected_by_shape": int(rejected_by_shape),
                            "cylinder_evaluated": int(cylinder_evaluated),
                            "cylinder_valid": int(cylinder_valid),
                            "selected_label": selected_label,
                            "fallback_label": int(fallback_label),
                            "score": _round(float(best_image_score)),
                            "band_valid": bool(best_image.get("band_valid", False)),
                            "band_score": _round(float(best_image.get("band_score", 0.0))),
                            "band_width_fraction": _round(float(best_image.get("band_width_fraction", 0.0))),
                            "band_column_coverage": _round(float(best_image.get("band_column_coverage", 0.0))),
                            "band_pixels": int(best_image.get("band_pixels", 0)),
                            "band_method": best_image.get("band_method"),
                            "image_lock_axis_delta_deg": _round(best_image_debug.get("axis_delta_deg")),
                            "image_lock_center_delta_px": _round(best_image_debug.get("center_delta_px")),
                            "image_lock_depth_delta_m": _round(best_image_debug.get("depth_delta_m")),
                        }
                        return np.asarray(best_image.get("selection_mask", labels == selected_label), dtype=bool)

                self._mark_pipe_lock_missed("no_compatible_component")
                if bool(self.params["pipe_lock_reject_global_when_locked"]):
                    self.pipe_component_selection_info = {
                        "method": "temporal_pipe_lock_no_compatible",
                        "component_count": int(count),
                        "candidate_count": int(len(candidates)),
                        "rejected_by_shape": int(rejected_by_shape),
                        "cylinder_evaluated": int(cylinder_evaluated),
                        "cylinder_valid": int(cylinder_valid),
                        "selected_label": None,
                        "fallback_label": int(fallback_label),
                        "pipe_lock_missed_frames": int(self.pipe_lock_missed_frames),
                        "score": _round(best_rejected_score) if np.isfinite(best_rejected_score) else None,
                        "lock_axis_delta_deg": _round(best_rejected_debug.get("axis_delta_deg")),
                        "lock_radius_delta_m": _round(best_rejected_debug.get("radius_delta_m")),
                        "lock_stand_delta_m": _round(best_rejected_debug.get("stand_delta_m")),
                        "lock_axis_point_delta_m": _round(best_rejected_debug.get("axis_point_delta_m")),
                        "image_lock_axis_delta_deg": _round(best_image_rejected_debug.get("axis_delta_deg")),
                        "image_lock_center_delta_px": _round(best_image_rejected_debug.get("center_delta_px")),
                        "image_lock_depth_delta_m": _round(best_image_rejected_debug.get("depth_delta_m")),
                    }
                    return np.zeros_like(mask, dtype=bool)

            if best_label > 0:
                self.pipe_component_selected_model = best_model
                best_candidate = next((c for c in fit_candidates if int(c["label"]) == int(best_label)), None)
                if best_candidate is not None:
                    self.pipe_component_selected_image_model = self._pipe_image_model_from_candidate(best_candidate)
                self.pipe_component_selection_info = {
                    "method": "cylinder_fit",
                    "component_count": int(count),
                    "candidate_count": int(len(candidates)),
                    "rejected_by_shape": int(rejected_by_shape),
                    "cylinder_evaluated": int(cylinder_evaluated),
                    "cylinder_valid": int(cylinder_valid),
                    "selected_label": int(best_label),
                    "fallback_label": int(fallback_label),
                    "score": _round(float(best_score)),
                    "band_valid": bool(best_candidate.get("band_valid", False)) if best_candidate else None,
                    "band_score": _round(float(best_candidate.get("band_score", 0.0))) if best_candidate else None,
                    "band_width_fraction": _round(float(best_candidate.get("band_width_fraction", 0.0))) if best_candidate else None,
                    "band_column_coverage": _round(float(best_candidate.get("band_column_coverage", 0.0))) if best_candidate else None,
                    "band_pixels": int(best_candidate.get("band_pixels", 0)) if best_candidate else None,
                    "band_method": best_candidate.get("band_method") if best_candidate else None,
                }
                if best_candidate is not None:
                    return np.asarray(best_candidate.get("selection_mask", labels == best_label), dtype=bool)
                return labels == best_label

            if bool(self.params["pipe_component_require_valid_cylinder"]):
                best_band = max(
                    (c for c in candidates if bool(c.get("band_valid", False))),
                    key=lambda item: float(item.get("band_score", 0.0)),
                    default=None,
                )
                band_warm_ok = True
                if best_band is not None and getattr(self, "_warm_scene_ok", False):
                    band_warm = self._mask_warm_fraction(
                        np.asarray(best_band.get("selection_mask"), dtype=bool),
                        self._warm_mask_current,
                    )
                    band_warm_ok = (
                        band_warm is not None
                        and band_warm >= float(self.params["pipe_color_min_band_warm_frac"])
                    )
                if (
                    best_band is not None
                    and band_warm_ok
                    and bool(self.params["pipe_component_allow_band_fallback_without_valid_cylinder"])
                    and float(best_band.get("band_score", 0.0)) >= float(self.params["pipe_component_band_fallback_min_score"])
                ):
                    self.pipe_component_selected_image_model = self._pipe_image_model_from_candidate(best_band)
                    self.pipe_component_selection_info = {
                        "method": "band_mask_fallback_no_valid_cylinder",
                        "component_count": int(count),
                        "candidate_count": int(len(candidates)),
                        "rejected_by_shape": int(rejected_by_shape),
                        "cylinder_evaluated": int(cylinder_evaluated),
                        "cylinder_valid": int(cylinder_valid),
                        "selected_label": int(best_band["label"]),
                        "fallback_label": int(fallback_label),
                        "score": _round(float(best_band.get("band_score", 0.0))),
                        "band_valid": bool(best_band.get("band_valid", False)),
                        "band_score": _round(float(best_band.get("band_score", 0.0))),
                        "band_width_fraction": _round(float(best_band.get("band_width_fraction", 0.0))),
                        "band_column_coverage": _round(float(best_band.get("band_column_coverage", 0.0))),
                        "band_pixels": int(best_band.get("band_pixels", 0)),
                        "band_method": best_band.get("band_method"),
                    }
                    return np.asarray(best_band.get("selection_mask", labels == int(best_band["label"])), dtype=bool)
                self.pipe_component_selection_info = {
                    "method": "no_valid_cylinder_component",
                    "component_count": int(count),
                    "candidate_count": int(len(candidates)),
                    "rejected_by_shape": int(rejected_by_shape),
                    "cylinder_evaluated": int(cylinder_evaluated),
                    "cylinder_valid": int(cylinder_valid),
                    "selected_label": None,
                    "fallback_label": int(fallback_label),
                }
                return np.zeros_like(mask, dtype=bool)

        if (
            bool(self.params.get("pipe_component_require_valid_cylinder", False))
            and bool(self.params.get("use_cylinder_component_selection", True))
            and _fit_pipe_pose is None
        ):
            self.pipe_component_selection_info = {
                "method": "cylinder_fit_unavailable",
                "component_count": int(count),
                "candidate_count": int(len(candidates)),
                "rejected_by_shape": int(rejected_by_shape),
                "cylinder_evaluated": int(cylinder_evaluated),
                "cylinder_valid": int(cylinder_valid),
                "selected_label": None,
                "fallback_label": int(fallback_label),
            }
            return np.zeros_like(mask, dtype=bool)

        self.pipe_component_selection_info = {
            "method": "bbox_fallback",
            "component_count": int(count),
            "candidate_count": int(len(candidates)),
            "rejected_by_shape": int(rejected_by_shape),
            "cylinder_evaluated": int(cylinder_evaluated),
            "cylinder_valid": int(cylinder_valid),
            "selected_label": int(fallback_label),
        }
        return labels == fallback_label

    def _variant_a_params(self) -> dict[str, Any]:
        """Map node params -> acea_seam_detector.detect_seam params (rest = its defaults)."""
        return {
            "tophat_se_len_px": int(self.params["variant_a_tophat_se_len_px"]),
            "min_vertical_run_px": int(self.params["variant_a_min_vertical_run_px"]),
            "min_significance_z": float(self.params["variant_a_min_significance_z"]),
            "max_seam_width_px": float(self.params["variant_a_max_seam_width_px"]),
            "border_margin_px": int(self.params["variant_a_border_margin_px"]),
            "orientation_search_deg": float(self.params["variant_a_orientation_search_deg"]),
            "orientation_search_step_deg": float(self.params["variant_a_orientation_search_step_deg"]),
        }

    @staticmethod
    def _variant_a_center_x_in_strip(det: Any, strip_shape: tuple[int, int]) -> int:
        """Map Variant-A rotated-frame column back to the unrotated strip x."""
        h, w = int(strip_shape[0]), int(strip_shape[1])
        x_rot = float(getattr(det, "x_px", 0.0))
        angle_deg = float(getattr(det, "orientation_deg", 0.0))
        if abs(angle_deg) < 1e-6:
            return int(np.clip(round(x_rot), 0, max(w - 1, 0)))

        cx = 0.5 * float(w - 1)
        cy = 0.5 * float(h - 1)
        theta = math.radians(angle_deg)
        c = math.cos(theta)
        s = math.sin(theta)

        def inverse_x(x: float, y: float) -> float:
            dx = float(x) - cx
            dy = float(y) - cy
            return c * dx - s * dy + cx

        x0 = inverse_x(x_rot, 0.0)
        x1 = inverse_x(x_rot, float(h - 1))
        return int(np.clip(round(0.5 * (x0 + x1)), 0, max(w - 1, 0)))

    def _appearance_patch(self, gray: np.ndarray, candidate_x: int) -> np.ndarray | None:
        """Fixed-size grayscale patch around the candidate column in the pipe-aligned
        strip (numpy-only). The strip is already de-rotated, so this is rotation-stable."""
        half = int(self.params.get("appearance_template_half_px", 28))
        h, w = gray.shape
        if w < 2 * half + 1 or h < 8:
            return None
        x0 = int(np.clip(candidate_x - half, 0, max(0, w - (2 * half + 1))))
        patch = gray[:, x0:x0 + 2 * half + 1]
        if patch.shape[1] < 2 * half + 1:
            patch = np.pad(patch, ((0, 0), (0, 2 * half + 1 - patch.shape[1])), mode="edge")
        idx = np.linspace(0, h - 1, 48).round().astype(int)  # canonical height
        return patch[idx, :].astype(np.float64)

    @staticmethod
    def _appearance_ncc(a: np.ndarray, b: np.ndarray) -> float:
        """Zero-mean normalized cross-correlation in [-1, 1]."""
        if a.shape != b.shape:
            return 0.0
        a = a - float(a.mean())
        b = b - float(b.mean())
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a.ravel(), b.ravel()) / (na * nb))

    def _detect_seam(self, rgb: np.ndarray, depth: np.ndarray, tracker: TrackerResult) -> SeamResult:
        angle_deg = tracker.image_axis_angle_deg
        if cv2 is not None:
            # warpAffine is ~4x faster than three full-frame PIL rotates and
            # uses the same rotation centre convention as _inverse_rotate_uv
            # ((w-1)/2, (h-1)/2), so the rotated->original candidate mapping
            # is exact.
            img_h, img_w = rgb.shape[:2]
            rot_m = cv2.getRotationMatrix2D(((img_w - 1) * 0.5, (img_h - 1) * 0.5), angle_deg, 1.0)
            rotated_rgb_arr = cv2.warpAffine(
                rgb, rot_m, (img_w, img_h), flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            rotated_depth = cv2.warpAffine(
                depth.astype(np.float32), rot_m, (img_w, img_h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"),
            ).astype(np.float64)
            rotated_mask = cv2.warpAffine(
                tracker.pipe_mask.astype(np.uint8), rot_m, (img_w, img_h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ) > 0
        else:
            rgb_image = PilImage.fromarray(rgb, mode="RGB")
            depth_image = PilImage.fromarray(depth.astype(np.float32), mode="F")
            mask_image = PilImage.fromarray((tracker.pipe_mask.astype(np.uint8) * 255), mode="L")
            rotated_rgb_arr = np.asarray(
                rgb_image.rotate(angle_deg, resample=PilImage.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
            )
            rotated_depth = np.asarray(
                depth_image.rotate(angle_deg, resample=PilImage.Resampling.BILINEAR, expand=False, fillcolor=np.nan),
                dtype=np.float64,
            )
            rotated_mask = np.asarray(
                mask_image.rotate(angle_deg, resample=PilImage.Resampling.NEAREST, expand=False, fillcolor=0)
            ) > 0

        ys, _ = np.nonzero(rotated_mask)
        if ys.size == 0:
            raise ValueError("Rotated pipe mask is empty")

        height, width = rotated_rgb_arr.shape[:2]
        vertical_margin = int(self.params["strip_vertical_margin_px"])
        x0, x1 = 0, width - 1
        y0 = max(0, int(ys.min()) - vertical_margin)
        y1 = min(height - 1, int(ys.max()) + vertical_margin)
        # The full geometric mask can span the whole close-up frame; the seam
        # machinery (rotations, tophat search, profiles, KLT) scales with the
        # strip height while the junction line only needs the central band of
        # the pipe. Cap the strip around the mask row centroid.
        strip_cap = int(self.params["strip_max_height_px"])
        if strip_cap > 0 and (y1 - y0 + 1) > strip_cap:
            center = int(round(float(ys.mean())))
            y0_cap = center - strip_cap // 2
            y1_cap = y0_cap + strip_cap - 1
            if y0_cap < y0:
                y0_cap, y1_cap = y0, y0 + strip_cap - 1
            if y1_cap > y1:
                y1_cap, y0_cap = y1, y1 - strip_cap + 1
            y0, y1 = max(y0, y0_cap), min(y1, y1_cap)
        strip_arr = rotated_rgb_arr[y0:y1 + 1, x0:x1 + 1]
        strip_mask = rotated_mask[y0:y1 + 1, x0:x1 + 1]
        strip_h, strip_w = strip_arr.shape[:2]

        strip_rgb = strip_arr.astype(np.float64) / 255.0
        # ITU-R 601-2 luma, same weights PIL's "L" convert uses.
        gray = (
            strip_rgb[:, :, 0] * 0.299 + strip_rgb[:, :, 1] * 0.587 + strip_rgb[:, :, 2] * 0.114
        )
        profile, counts = self._column_profile(gray, strip_mask)
        background = _median_filter_1d(profile, int(self.params["background_window_px"]))
        residual = background - profile
        edge_profile, edge_support_profile = self._rgb_vertical_edge_profile(gray, strip_mask)

        strip_width = profile.size
        edge_margin = max(
            int(self.params["edge_margin_px"]),
            int(round(strip_width * float(self.params["edge_margin_fraction"]))),
        )
        min_count = max(8, int(float(self.params["min_valid_column_fraction"]) * strip_h))
        valid = np.isfinite(profile) & (counts >= min_count)
        columns = np.arange(strip_width)
        valid &= columns >= edge_margin
        valid &= columns < strip_width - edge_margin
        if not valid.any():
            raise ValueError("No valid interior strip columns for seam scoring")

        classical_candidate_x = int(np.argmax(np.where(valid, residual, -np.inf)))

        interior_residual = residual[valid]
        residual_median = float(np.median(interior_residual))
        residual_mad = float(np.median(np.abs(interior_residual - residual_median)))
        robust_sigma = max(1.4826 * residual_mad, 1.0 / 255.0)
        dark_z_profile = (residual - residual_median) / robust_sigma

        # Collar/step-edge cue. Computed on data-backed columns (raw counts),
        # NOT on the edge-margin-restricted 'valid' mask, so the averaging
        # windows near the image border still see real profile data.
        step_data_ok = counts >= min_count
        step_profile = _step_edge_profile(
            profile,
            step_data_ok,
            int(self.params["step_edge_window_px"]),
            int(self.params["step_edge_gap_px"]),
        )
        step_interior = step_profile[valid]
        if step_interior.size:
            step_median = float(np.median(step_interior))
            step_mad = float(np.median(np.abs(step_interior - step_median)))
        else:
            step_median, step_mad = 0.0, 0.0
        step_sigma = max(1.4826 * step_mad, 1e-3)
        step_z_profile = (step_profile - step_median) / step_sigma

        # Columns whose pipe-mask support says "pipe end / border", shared by the
        # Variant A crop and the step-edge candidate search below.
        safe_cols = ~self._pipe_end_rejected_columns(strip_mask)

        mode = str(self.params["junction_acceptance_mode"]).strip().lower()
        if mode not in ("rgb_temporal", "variant_a_rgb"):
            mode = "variant_a_rgb"

        # Variant A (deterministic RGB-only): run on the already-rotated, pipe-only
        # strip (pipe horizontal -> seam vertical -> pipe_axis_angle_deg=0).
        variant_a_result = None
        variant_a_x_offset = 0
        variant_a_shape: tuple[int, int] = gray.shape
        if mode == "variant_a_rgb" and _variant_a_detect_seam is not None:
            variant_a_rgb = strip_arr
            if (
                bool(self.params["variant_a_suppress_pipe_end_columns"])
                and safe_cols.any()
                and not safe_cols.all()
            ):
                # CROP to the safe column range instead of repainting unsafe
                # columns with a constant color. The repaint created an
                # artificial luminance step exactly at the fill boundary and the
                # tophat then kept electing that boundary (a constant junk
                # candidate a few px inside the border margin) over the real
                # seam - observed on the real bag as the serial
                # candidate_near_border rejects at x~17.
                first_safe = int(np.argmax(safe_cols))
                last_safe = int(strip_width - 1 - np.argmax(safe_cols[::-1]))
                if last_safe - first_safe + 1 >= 32:
                    variant_a_rgb = variant_a_rgb[:, first_safe:last_safe + 1]
                    variant_a_x_offset = first_safe
                    variant_a_shape = (gray.shape[0], variant_a_rgb.shape[1])
            variant_a_result = _variant_a_detect_seam(
                variant_a_rgb,
                params=self._variant_a_params(),
                pipe_axis_angle_deg=0.0,
            )

        if mode == "rgb_temporal":
            residual_score_profile = np.clip(
                residual / max(float(self.params["strong_dark_contrast"]), 1e-6),
                0.0,
                1.0,
            )
            temporal_candidate_score = (
                float(self.params["rgb_temporal_visual_weight"]) * residual_score_profile
                + float(self.params["rgb_temporal_edge_score_weight"]) * edge_profile
            )
            classical_candidate_x = int(np.argmax(np.where(valid, temporal_candidate_score, -np.inf)))

        candidate_source = "classical_rgb_dark"
        variant_a_candidate_x = None
        if mode == "variant_a_rgb" and variant_a_result is not None:
            candidate_x = int(np.clip(
                variant_a_x_offset + self._variant_a_center_x_in_strip(variant_a_result, variant_a_shape),
                0,
                strip_width - 1,
            ))
            variant_a_candidate_x = int(candidate_x)
            candidate_source = "variant_a"
        else:
            candidate_x = classical_candidate_x
        variant_a_classical_fallback_used = False
        if (
            mode == "variant_a_rgb"
            and variant_a_result is not None
            and bool(self.params["variant_a_fallback_to_classical_on_border"])
            and not self.junction_lock_active
        ):
            variant_pipe_end = self._pipe_end_rejected(strip_mask, candidate_x)
            variant_near_border = not (edge_margin <= candidate_x < strip_width - edge_margin)
            classical_z_for_fallback = (
                float(residual[classical_candidate_x]) - residual_median
            ) / robust_sigma
            classical_contrast_for_fallback = float(max(0.0, residual[classical_candidate_x]))
            classical_strong = bool(
                classical_contrast_for_fallback >= float(self.params["variant_a_classical_fallback_min_contrast"])
                and classical_z_for_fallback >= float(self.params["variant_a_classical_fallback_min_z"])
            )
            classical_safe = bool(
                valid[classical_candidate_x]
                and not self._pipe_end_rejected(strip_mask, classical_candidate_x)
            )
            max_fallback_distance = float(self.params["variant_a_classical_fallback_max_distance_px"])
            classical_near_variant = abs(int(classical_candidate_x) - int(candidate_x)) <= max_fallback_distance
            if (
                (variant_pipe_end or variant_near_border)
                and classical_strong
                and classical_safe
                and classical_near_variant
            ):
                candidate_x = classical_candidate_x
                candidate_source = "classical_fallback"
                variant_a_classical_fallback_used = True
        # Collar/step-edge fallback: when the tophat frontend has NO acceptable
        # thin dark line (blurred seam during camera motion, glossy PVC with a
        # tone-step collar instead of a dark gap), take the strongest broad
        # luminance step on pipe-supported interior columns. This is what makes
        # the real socket junction detectable at all in a large part of the bag.
        if (
            mode == "variant_a_rgb"
            and bool(self.params["enable_step_edge_candidate"])
            and not variant_a_classical_fallback_used
        ):
            variant_would_accept = bool(
                variant_a_result is not None
                and bool(variant_a_result.accepted)
                and not self._pipe_end_rejected(strip_mask, candidate_x)
            )
            if not variant_would_accept:
                step_scores = np.where(valid & safe_cols, step_profile, -np.inf)
                step_x = int(np.argmax(step_scores))
                if (
                    np.isfinite(step_scores[step_x])
                    and float(step_z_profile[step_x]) >= float(self.params["step_edge_min_z"])
                    and float(step_profile[step_x]) >= float(self.params["step_edge_min_abs"])
                    and not self._pipe_end_rejected(strip_mask, step_x)
                ):
                    candidate_x = int(step_x)
                    candidate_source = "step_edge"
        # The junction-lock local search scores BOTH cues: thin dark line OR
        # collar step. Under motion blur only the step survives.
        evidence_z_profile = np.maximum(dark_z_profile, step_z_profile)
        klt_prediction = self._klt_predict_junction_x(gray, strip_mask, evidence_z_profile, valid)
        if klt_prediction["available"] and self.junction_lock_active:
            klt_candidate_x = int(klt_prediction["candidate_x"])
            visual_is_strong = bool(variant_a_result is not None and bool(variant_a_result.accepted))
            visual_disagrees = bool(
                visual_is_strong
                and variant_a_candidate_x is not None
                and abs(klt_candidate_x - int(variant_a_candidate_x)) > int(self.params["klt_max_visual_disagreement_px"])
            )
            if not visual_disagrees:
                candidate_x = klt_candidate_x
                candidate_source = "klt_prediction"
                variant_a_classical_fallback_used = False
        klt_predicted_x = klt_prediction.get("predicted_x")

        # Optional appearance-identity veto (off by default): compare the local strip
        # patch at the current candidate column to the template captured at lock. Only
        # marks a veto; the template is captured/cleared by the state machine.
        appearance_ncc = 1.0
        appearance_veto = False
        self._last_appearance_patch = None
        if bool(self.params.get("enable_appearance_veto", False)):
            patch = self._appearance_patch(gray, candidate_x)
            self._last_appearance_patch = patch
            if patch is not None and self.appearance_template is not None:
                appearance_ncc = self._appearance_ncc(patch, self.appearance_template)
                if appearance_ncc < float(self.params.get("appearance_veto_min_ncc", 0.35)):
                    appearance_veto = True

        contrast = float(max(0.0, residual[candidate_x]))
        min_dark = float(self.params["min_dark_contrast"])
        strong_dark = float(self.params["strong_dark_contrast"])
        rgb_dark_score = float(np.clip((contrast - min_dark) / max(strong_dark - min_dark, 1e-6), 0.0, 1.0))

        z_score = (float(residual[candidate_x]) - residual_median) / robust_sigma
        classical_z_score = (float(residual[classical_candidate_x]) - residual_median) / robust_sigma
        candidate_step_abs = float(step_profile[candidate_x])
        candidate_step_z = float(step_z_profile[candidate_x])
        candidate_evidence_z = float(max(z_score, candidate_step_z))
        min_local_z = float(self.params["rgb_local_min_z_score"])
        strong_local_z = float(self.params["rgb_local_strong_z_score"])
        rgb_local_contrast_score = float(
            np.clip((z_score - min_local_z) / max(strong_local_z - min_local_z, 1e-6), 0.0, 1.0)
        )
        depth_gap = self._depth_gap_evidence(rotated_depth, rotated_mask, [x0, y0, x1, y1], candidate_x)
        depth_gap_raw_accepted = bool(
            depth_gap["depth_jump_m"] >= float(self.params["depth_gap_diagnostic_min_score_m"])
            or depth_gap["coverage_drop"] >= float(self.params["depth_gap_diagnostic_min_coverage_drop"])
        )
        depth_gap_score_plausible = True
        depth_gap_accepted = depth_gap_raw_accepted and depth_gap_score_plausible
        confidence = float(np.clip(rgb_dark_score, 0.0, 1.0))
        rgb_dark_threshold_used = min_dark
        rgb_dark_accepted = bool(rgb_dark_score >= float(self.params["accept_confidence"]))

        visual_frontend = "classical_rgb_dark"
        visual_frontend_accepted = rgb_dark_accepted
        negative_gate_reason = "classical_rgb_dark" if visual_frontend_accepted else "rgb_dark_rejected"
        temporal_change = self._temporal_scan_change(profile, valid, candidate_x)
        temporal_gate_enabled = bool(self.params["enable_temporal_scan_change"]) and bool(
            self.params["use_temporal_scan_change_gate"]
        )
        line_support_fraction = float(edge_support_profile[candidate_x]) if edge_support_profile.size else 0.0
        rgb_vertical_edge_score = float(edge_profile[candidate_x]) if edge_profile.size else 0.0
        rgb_line_width_px = self._rgb_line_width_px(residual, candidate_x, valid)
        pipe_support = self._pipe_support_around_candidate(strip_mask, candidate_x)
        pipe_end_rejected = bool(pipe_support["pipe_end_rejected"])
        rgb_shadow_features = self._rgb_shadow_continuity_features(
            strip_rgb,
            gray,
            strip_mask,
            candidate_x,
            rgb_line_width_px,
            contrast,
        )
        rgb_temporal_base = max(rgb_dark_score, rgb_local_contrast_score)
        temporal_score = temporal_change.score if temporal_change.reference_ready else 0.0
        rgb_temporal_score = float(
            np.clip(
                float(self.params["rgb_temporal_visual_weight"]) * rgb_temporal_base
                + float(self.params["rgb_temporal_edge_score_weight"]) * rgb_vertical_edge_score
                + float(self.params["rgb_temporal_temporal_weight"]) * temporal_score,
                0.0,
                1.0,
            )
        )
        if bool(self.params["rgb_temporal_boost_on_temporal_change"]) and temporal_change.accepted:
            rgb_temporal_score = max(rgb_temporal_score, float(self.params["rgb_temporal_temporal_boost_score"]))

        rgb_low_contrast_rejected = bool(
            bool(self.params["rgb_temporal_low_contrast_reject_enabled"])
            and contrast < float(self.params["rgb_temporal_min_candidate_contrast"])
        )
        rgb_shadow_like_rejected = bool(rgb_shadow_features["shadow_like_rejected"])
        rgb_surface_continuity_rejected = bool(rgb_shadow_features["surface_continuity_rejected"])
        rgb_temporal_reject_reasons: list[str] = []
        if pipe_end_rejected:
            rgb_temporal_reject_reasons.append("pipe_end_rejected")
        if rgb_low_contrast_rejected:
            rgb_temporal_reject_reasons.append("rgb_low_contrast")
        if rgb_shadow_like_rejected:
            rgb_temporal_reject_reasons.append("rgb_shadow_like")
        if rgb_surface_continuity_rejected:
            rgb_temporal_reject_reasons.append("rgb_surface_continuity")
        if rgb_temporal_score < float(self.params["rgb_temporal_min_score"]):
            rgb_temporal_reject_reasons.append("rgb_temporal_score_low")
        if line_support_fraction < float(self.params["rgb_temporal_min_line_support_fraction"]):
            rgb_temporal_reject_reasons.append("rgb_line_support_low")
        if rgb_line_width_px > int(self.params["rgb_temporal_max_line_width_px"]):
            rgb_temporal_reject_reasons.append("rgb_line_too_wide")
        rgb_temporal_candidate_reject_reason = (
            "ok" if not rgb_temporal_reject_reasons else ";".join(rgb_temporal_reject_reasons)
        )
        rgb_temporal_candidate_ok = bool(
            not pipe_end_rejected
            and not rgb_low_contrast_rejected
            and not rgb_shadow_like_rejected
            and not rgb_surface_continuity_rejected
            and rgb_temporal_score >= float(self.params["rgb_temporal_min_score"])
            and line_support_fraction >= float(self.params["rgb_temporal_min_line_support_fraction"])
            and rgb_line_width_px <= int(self.params["rgb_temporal_max_line_width_px"])
        )
        track = self._update_rgb_temporal_track(candidate_x, rgb_temporal_candidate_ok)
        rgb_temporal_accepted = bool(
            rgb_temporal_candidate_ok
            and track["streak"] >= int(self.params["rgb_temporal_min_track_streak"])
        )

        if mode == "rgb_temporal":
            visual_frontend = "rgb_temporal_edge"
            visual_frontend_accepted = bool(rgb_temporal_candidate_ok)
            confidence = rgb_temporal_score
            rgb_dark_accepted = bool(rgb_temporal_candidate_ok)
            local_candidate_accepted = rgb_temporal_accepted
            accepted = rgb_temporal_accepted
            if not rgb_temporal_candidate_ok:
                negative_gate_reason = rgb_temporal_candidate_reject_reason
            elif not rgb_temporal_accepted:
                negative_gate_reason = "rgb_temporal_tracking"
            else:
                negative_gate_reason = "rgb_temporal_edge_track"
        else:
            if variant_a_classical_fallback_used:
                va_seam = True
                va_sig = float(classical_z_score)
                va_reason = "classical_fallback_from_variant_a_border"
            elif candidate_source == "step_edge":
                # Already passed the step z/abs gates at selection time; the
                # pipe-end and depth-end gates below still apply.
                va_seam = True
                va_sig = float(step_z_profile[candidate_x])
                va_reason = "step_edge_fallback"
            elif candidate_source == "klt_prediction":
                # KLT is a temporal prior, not a detector. If KLT points to a
                # different column than Variant A, do not inherit Variant A's
                # acceptance/significance from that other column. The lock
                # reacquire gate below must validate this candidate locally.
                va_seam = False
                va_sig = float(z_score)
                va_reason = "klt_prediction_requires_local_reacquire"
            else:
                va_seam = bool(variant_a_result.accepted) if variant_a_result is not None else False
                va_sig = float(variant_a_result.significance) if variant_a_result is not None else 0.0
                va_reason = variant_a_result.reason if variant_a_result is not None else "variant_a_unavailable"
            z_strong = max(float(self.params["variant_a_z_confidence_strong"]), 1e-6)
            # Pipe-END / start discrimination (an end is NOT a junction):
            #  (1) RGB structural: a real junction has pipe surface on BOTH sides
            #      (reuse _pipe_end_rejected, computed from the pipe mask coverage).
            #  (2) coarse depth: a pipe end has a HUGE depth jump (surface->background,
            #      ~0.7 m observed); a real 3 mm seam has ~none. Depth is used ONLY for
            #      the gross end here, NOT to detect the seam (too noisy at 3 mm).
            va_pipe_end_depth = bool(
                bool(self.params["variant_a_use_depth_pipe_end_gate"])
                and float(depth_gap["depth_jump_m"]) > float(self.params["variant_a_pipe_end_max_depth_jump_m"])
            )
            va_accepted = bool(va_seam and not pipe_end_rejected and not va_pipe_end_depth)
            visual_frontend = "variant_a_tophat_rgb"
            visual_frontend_accepted = va_accepted
            # Map MAD z-score -> confidence; an accepted seam (z>=min_significance_z)
            # clears min_confidence so _update_state's confidence_ok passes.
            confidence = float(np.clip(va_sig / z_strong, 0.0, 1.0))
            rgb_dark_accepted = va_accepted
            local_candidate_accepted = va_accepted
            # _update_state adds the temporal confirmation.
            accepted = va_accepted
            if va_accepted:
                negative_gate_reason = "variant_a_tophat"
            elif pipe_end_rejected:
                negative_gate_reason = "variant_a:pipe_end_rgb"
            elif va_pipe_end_depth:
                negative_gate_reason = f"variant_a:pipe_end_depth={float(depth_gap['depth_jump_m']):.2f}m"
            else:
                negative_gate_reason = f"variant_a:{va_reason}"
        if accepted and bool(self.params["enable_opencv_klt_tracking"]):
            self._klt_update_reference(np.clip(gray * 255.0, 0, 255).astype(np.uint8), strip_mask, float(candidate_x))

        return SeamResult(
            candidate_x_strip_px=candidate_x,
            candidate_x_rotated_px=int(x0 + candidate_x),
            classical_candidate_x_strip_px=classical_candidate_x,
            classical_candidate_contrast=float(max(0.0, residual[classical_candidate_x])),
            classical_candidate_z_score=float(classical_z_score),
            candidate_contrast=contrast,
            candidate_z_score=float(z_score),
            confidence=confidence,
            visual_frontend=visual_frontend,
            visual_frontend_accepted=visual_frontend_accepted,
            rgb_dark_score=rgb_dark_score,
            rgb_local_contrast_score=rgb_local_contrast_score,
            rgb_dark_threshold_used=rgb_dark_threshold_used,
            rgb_dark_accepted=rgb_dark_accepted,
            depth_gap_score=float(depth_gap["score"]),
            depth_gap_accepted=depth_gap_accepted,
            depth_gap_raw_accepted=depth_gap_raw_accepted,
            depth_gap_score_plausible=depth_gap_score_plausible,
            depth_gap_depth_jump_m=float(depth_gap["depth_jump_m"]),
            depth_gap_coverage_drop=float(depth_gap["coverage_drop"]),
            negative_gate_reason=negative_gate_reason,
            local_candidate_accepted=local_candidate_accepted,
            temporal_change_score=temporal_change.score,
            temporal_change_dark_delta=temporal_change.dark_delta,
            temporal_change_z_score=temporal_change.z_score,
            temporal_change_accepted=temporal_change.accepted,
            temporal_change_gate_enabled=temporal_gate_enabled,
            temporal_reference_ready=temporal_change.reference_ready,
            temporal_reference_frame_count=temporal_change.reference_frame_count,
            temporal_change_reason=temporal_change.reason,
            junction_acceptance_mode=mode,
            variant_a_orientation_deg=0.0
            if variant_a_result is None
            else float(getattr(variant_a_result, "orientation_deg", 0.0)),
            variant_a_classical_fallback_used=variant_a_classical_fallback_used,
            rgb_temporal_accepted=rgb_temporal_accepted,
            rgb_temporal_score=rgb_temporal_score,
            rgb_vertical_edge_score=rgb_vertical_edge_score,
            rgb_luminance_edge_score=float(rgb_shadow_features["luminance_edge_score"]),
            rgb_chromatic_edge_score=float(rgb_shadow_features["chromatic_edge_score"]),
            rgb_edge_chromaticity_ratio=float(rgb_shadow_features["edge_chromaticity_ratio"]),
            rgb_shadow_like_score=float(rgb_shadow_features["shadow_like_score"]),
            rgb_shadow_like_rejected=rgb_shadow_like_rejected,
            rgb_surface_continuity_score=float(rgb_shadow_features["surface_continuity_score"]),
            rgb_surface_continuity_rejected=rgb_surface_continuity_rejected,
            rgb_low_contrast_rejected=rgb_low_contrast_rejected,
            rgb_temporal_candidate_reject_reason=rgb_temporal_candidate_reject_reason,
            rgb_line_support_fraction=line_support_fraction,
            rgb_line_width_px=rgb_line_width_px,
            rgb_track_id=int(track["id"]),
            rgb_track_streak=int(track["streak"]),
            rgb_track_missed_frames=int(track["missed"]),
            rgb_candidate_velocity_px_per_frame=float(track["velocity"]),
            klt_status=str(klt_prediction.get("reason", "unavailable")),
            klt_points=int(klt_prediction.get("points", 0) or 0),
            klt_dx_px=float(klt_prediction.get("dx_px", 0.0) or 0.0),
            klt_predicted_x_strip_px=None if klt_predicted_x is None else float(klt_predicted_x),
            pipe_end_rejected=pipe_end_rejected,
            pipe_support_left_cols=int(pipe_support["left_cols"]),
            pipe_support_right_cols=int(pipe_support["right_cols"]),
            pipe_support_left_coverage=float(pipe_support["left_coverage"]),
            pipe_support_right_coverage=float(pipe_support["right_coverage"]),
            accepted=accepted,
            edge_margin_px=edge_margin,
            crop_xyxy=[x0, y0, x1, y1],
            strip_size_wh=[strip_w, strip_h],
            strip_mask=strip_mask,
            rotated_mask=rotated_mask,
            rotation_deg=angle_deg,
            strip_profile=profile,
            strip_profile_valid=valid,
            appearance_veto=appearance_veto,
            appearance_ncc=appearance_ncc,
            candidate_step_abs=candidate_step_abs,
            candidate_step_z=candidate_step_z,
            candidate_evidence_z=candidate_evidence_z,
            step_fallback_used=bool(candidate_source == "step_edge"),
        )

    def _rgb_shadow_continuity_features(
        self,
        strip_rgb: np.ndarray,
        gray: np.ndarray,
        strip_mask: np.ndarray,
        candidate_x: int,
        line_width_px: int,
        contrast: float,
    ) -> dict[str, float | bool]:
        defaults: dict[str, float | bool] = {
            "luminance_edge_score": 0.0,
            "chromatic_edge_score": 0.0,
            "edge_chromaticity_ratio": 0.0,
            "shadow_like_score": 0.0,
            "shadow_like_rejected": False,
            "surface_continuity_score": 0.0,
            "surface_continuity_rejected": False,
        }
        if strip_rgb.ndim != 3 or strip_rgb.shape[:2] != strip_mask.shape:
            return defaults

        width = int(strip_rgb.shape[1])
        x = int(np.clip(candidate_x, 0, max(width - 1, 0)))
        side_width = max(2, int(self.params["rgb_temporal_continuity_side_width_px"]))
        base_gap = max(1, int(self.params["rgb_temporal_continuity_side_gap_px"]))
        half_line = max(1, int(math.ceil(max(1, int(line_width_px)) / 2.0)))
        side_gap = max(base_gap, half_line)

        left0 = max(0, x - side_gap - side_width)
        left1 = max(0, x - side_gap)
        right0 = min(width, x + side_gap + 1)
        right1 = min(width, x + side_gap + 1 + side_width)
        if left1 <= left0 or right1 <= right0:
            return defaults

        eps = 1e-6
        rgb = np.clip(strip_rgb, 0.0, 1.0)
        luma = np.asarray(gray, dtype=np.float64)
        rgb_sum = np.maximum(rgb.sum(axis=2, keepdims=True), eps)
        chroma = rgb / rgb_sum

        local_gradient = np.zeros_like(luma, dtype=np.float64)
        if width > 2:
            local_gradient[:, 1:-1] = 0.5 * np.abs(luma[:, 2:] - luma[:, :-2])
            local_gradient[:, 0] = local_gradient[:, 1]
            local_gradient[:, -1] = local_gradient[:, -2]

        def band_stats(x0: int, x1: int) -> dict[str, float | np.ndarray | int]:
            mask = strip_mask[:, x0:x1] & np.isfinite(luma[:, x0:x1])
            count = int(mask.sum())
            coverage = float(mask.mean()) if mask.size else 0.0
            if count < int(self.params["rgb_temporal_continuity_min_pixels"]):
                return {
                    "valid": False,
                    "count": count,
                    "coverage": coverage,
                    "luma": 0.0,
                    "chroma": np.zeros(3, dtype=np.float64),
                    "texture": 0.0,
                }
            luma_values = luma[:, x0:x1][mask]
            chroma_values = chroma[:, x0:x1, :][mask]
            gradient_values = local_gradient[:, x0:x1][mask]
            return {
                "valid": True,
                "count": count,
                "coverage": coverage,
                "luma": float(np.median(luma_values)),
                "chroma": np.median(chroma_values, axis=0),
                "texture": float(np.median(gradient_values)),
            }

        left = band_stats(left0, left1)
        right = band_stats(right0, right1)
        if not bool(left["valid"]) or not bool(right["valid"]):
            return defaults

        left_chroma = np.asarray(left["chroma"], dtype=np.float64)
        right_chroma = np.asarray(right["chroma"], dtype=np.float64)
        luma_delta = abs(float(left["luma"]) - float(right["luma"]))
        chroma_delta = float(np.linalg.norm(left_chroma - right_chroma))
        texture_delta = abs(float(left["texture"]) - float(right["texture"]))
        coverage_delta = abs(float(left["coverage"]) - float(right["coverage"]))

        luma_strong = max(float(self.params["rgb_temporal_shadow_luma_delta_strong"]), eps)
        chroma_strong = max(float(self.params["rgb_temporal_shadow_chroma_delta_strong"]), eps)
        luminance_edge_score = float(np.clip(luma_delta / luma_strong, 0.0, 1.0))
        chromatic_edge_score = float(np.clip(chroma_delta / chroma_strong, 0.0, 1.0))
        edge_chromaticity_ratio = float(chroma_delta / max(luma_delta, eps))

        max_ratio = max(float(self.params["rgb_temporal_shadow_max_chromaticity_ratio"]), eps)
        chroma_absence_score = float(np.clip(1.0 - edge_chromaticity_ratio / max_ratio, 0.0, 1.0))
        shadow_like_score = float(np.clip(luminance_edge_score * chroma_absence_score, 0.0, 1.0))
        shadow_like_rejected = bool(
            bool(self.params["rgb_temporal_shadow_reject_enabled"])
            and shadow_like_score >= float(self.params["rgb_temporal_shadow_like_min_score"])
            and int(line_width_px) >= int(self.params["rgb_temporal_shadow_min_line_width_px"])
            and float(contrast) <= float(self.params["rgb_temporal_shadow_max_candidate_contrast"])
        )

        chroma_similarity = 1.0 - np.clip(
            chroma_delta / max(float(self.params["rgb_temporal_continuity_chroma_delta_reject"]), eps),
            0.0,
            1.0,
        )
        texture_similarity = 1.0 - np.clip(
            texture_delta / max(float(self.params["rgb_temporal_continuity_texture_delta_reject"]), eps),
            0.0,
            1.0,
        )
        coverage_similarity = 1.0 - np.clip(
            coverage_delta / max(float(self.params["rgb_temporal_continuity_coverage_delta_reject"]), eps),
            0.0,
            1.0,
        )
        surface_continuity_score = float(
            np.clip(0.50 * chroma_similarity + 0.25 * texture_similarity + 0.25 * coverage_similarity, 0.0, 1.0)
        )
        surface_continuity_rejected = bool(
            bool(self.params["rgb_temporal_surface_continuity_reject_enabled"])
            and surface_continuity_score >= float(self.params["rgb_temporal_surface_continuity_min_score"])
            and float(contrast) <= float(self.params["rgb_temporal_surface_continuity_max_contrast"])
        )

        return {
            "luminance_edge_score": luminance_edge_score,
            "chromatic_edge_score": chromatic_edge_score,
            "edge_chromaticity_ratio": edge_chromaticity_ratio,
            "shadow_like_score": shadow_like_score,
            "shadow_like_rejected": shadow_like_rejected,
            "surface_continuity_score": surface_continuity_score,
            "surface_continuity_rejected": surface_continuity_rejected,
        }

    def _klt_predict_junction_x(
        self,
        gray: np.ndarray,
        strip_mask: np.ndarray,
        evidence_profile: np.ndarray,
        valid: np.ndarray,
    ) -> dict[str, Any]:
        # evidence_profile: per-column seam evidence to argmax inside the local
        # window (currently max(dark-line z, collar-step z), see _detect_seam).
        if not bool(self.params["enable_opencv_klt_tracking"]):
            return {"available": False, "reason": "disabled", "candidate_x": None}
        if cv2 is None:
            self.klt_status = f"opencv_unavailable:{type(CV2_IMPORT_ERROR).__name__}"
            return {"available": False, "reason": self.klt_status, "candidate_x": None}
        if gray.ndim != 2 or strip_mask.shape != gray.shape or evidence_profile.size != gray.shape[1]:
            return {"available": False, "reason": "invalid_strip", "candidate_x": None}
        if not self.junction_lock_active or self.junction_lock_x is None:
            self._klt_reset()
            return {"available": False, "reason": "not_locked", "candidate_x": None}

        gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        # The strip height follows the pipe mask rows and changes frame to
        # frame, but calcOpticalFlowPyrLK requires prev/next images of the SAME
        # size - it was throwing on almost every other frame (klt_status
        # "klt_error:error") and silently disabling the temporal prior. Track in
        # a canonical-height plane: x (the only coordinate we use) is preserved.
        canon_h = max(32, int(self.params["klt_canonical_height_px"]))
        w = gray_u8.shape[1]
        gray_u8 = cv2.resize(gray_u8, (w, canon_h), interpolation=cv2.INTER_AREA)
        strip_mask = cv2.resize(
            strip_mask.astype(np.uint8), (w, canon_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        h = canon_h
        prev_x = float(self.junction_lock_x)
        predicted = prev_x + float(self.junction_lock_velocity_px)
        points_now = 0

        if self.klt_prev_gray is not None and self.klt_prev_points is not None and self.klt_prev_points.size > 0:
            try:
                next_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                    self.klt_prev_gray,
                    gray_u8,
                    self.klt_prev_points,
                    None,
                    winSize=(int(self.params["klt_win_size_px"]), int(self.params["klt_win_size_px"])),
                    maxLevel=int(self.params["klt_max_level"]),
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        int(self.params["klt_term_count"]),
                        float(self.params["klt_term_eps"]),
                    ),
                )
            except Exception as exc:
                self.klt_status = f"klt_error:{type(exc).__name__}"
                next_pts = status = None
            if next_pts is not None and status is not None:
                good = status.reshape(-1).astype(bool)
                prev = self.klt_prev_points.reshape(-1, 2)[good]
                nxt = next_pts.reshape(-1, 2)[good]
                in_bounds = (
                    (nxt[:, 0] >= 0.0)
                    & (nxt[:, 0] < float(w))
                    & (nxt[:, 1] >= 0.0)
                    & (nxt[:, 1] < float(h))
                ) if nxt.size else np.zeros(0, dtype=bool)
                prev = prev[in_bounds]
                nxt = nxt[in_bounds]
                points_now = int(nxt.shape[0])
                if points_now >= int(self.params["klt_min_valid_points"]):
                    dx = float(np.median(nxt[:, 0] - prev[:, 0]))
                    self.klt_last_dx_px = dx
                    predicted = prev_x + dx
                    self.klt_status = "tracked"
                else:
                    self.klt_status = f"too_few_points:{points_now}"

        radius = int(self.params["junction_lock_search_radius_px"])
        center = int(round(np.clip(predicted, 0, max(w - 1, 0))))
        lo = max(0, center - radius)
        hi = min(w - 1, center + radius)
        local_valid = valid.copy()
        local_valid[:lo] = False
        local_valid[hi + 1:] = False
        if not local_valid.any():
            self._klt_update_reference(gray_u8, strip_mask, prev_x)
            return {"available": False, "reason": "no_valid_local_columns", "candidate_x": None}

        # Distance-penalized local argmax: without it, a nearby competitor (the
        # collar's second edge ~60 px away, or a static reflection stripe on
        # the glossy pipe) inside the window wins whenever its instantaneous
        # score fluctuates above the tracked seam -> flip-flop / track capture.
        # A competitor must now beat the tracked seam by ~penalty*distance in
        # robust-z to steal the lock.
        penalty = float(self.params["junction_lock_distance_penalty_z_per_px"])
        columns = np.arange(evidence_profile.size, dtype=np.float64)
        local_score = evidence_profile - penalty * np.abs(columns - float(predicted))
        candidate_x = int(np.argmax(np.where(local_valid, local_score, -np.inf)))
        self._klt_update_reference(gray_u8, strip_mask, float(candidate_x))
        self.klt_last_points = points_now
        return {
            "available": True,
            "reason": self.klt_status,
            "candidate_x": candidate_x,
            "predicted_x": float(predicted),
            "dx_px": float(self.klt_last_dx_px),
            "points": int(points_now),
        }

    def _klt_update_reference(self, gray_u8: np.ndarray, strip_mask: np.ndarray, center_x: float) -> None:
        if cv2 is None or gray_u8.ndim != 2:
            return
        # Keep every stored reference in the canonical-height plane (see
        # _klt_predict_junction_x): callers pass strips of varying height.
        canon_h = max(32, int(self.params["klt_canonical_height_px"]))
        if gray_u8.shape[0] != canon_h:
            width = gray_u8.shape[1]
            gray_u8 = cv2.resize(gray_u8, (width, canon_h), interpolation=cv2.INTER_AREA)
            strip_mask = cv2.resize(
                strip_mask.astype(np.uint8), (width, canon_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        h, w = gray_u8.shape
        radius = int(self.params["klt_feature_radius_px"])
        x0 = max(0, int(round(center_x)) - radius)
        x1 = min(w - 1, int(round(center_x)) + radius)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, x0:x1 + 1] = strip_mask[:, x0:x1 + 1].astype(np.uint8) * 255
        try:
            pts = cv2.goodFeaturesToTrack(
                gray_u8,
                maxCorners=int(self.params["klt_max_features"]),
                qualityLevel=float(self.params["klt_quality_level"]),
                minDistance=float(self.params["klt_min_distance_px"]),
                mask=mask,
                blockSize=int(self.params["klt_block_size_px"]),
            )
        except Exception:
            pts = None
        self.klt_prev_gray = gray_u8.copy()
        self.klt_prev_points = pts.astype(np.float32) if pts is not None else None
        self.klt_prev_lock_x = float(center_x)

    def _klt_reset(self) -> None:
        self.klt_prev_gray = None
        self.klt_prev_points = None
        self.klt_prev_lock_x = None
        self.klt_last_dx_px = 0.0
        self.klt_last_points = 0
        self.klt_status = "not_locked"

    def _rgb_vertical_edge_profile(self, gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = gray.shape
        gradient = np.zeros_like(gray, dtype=np.float64)
        if width > 2:
            gradient[:, 1:-1] = 0.5 * np.abs(gray[:, 2:] - gray[:, :-2])
            gradient[:, 0] = gradient[:, 1]
            gradient[:, -1] = gradient[:, -2]

        valid_values = gradient[mask & np.isfinite(gradient)]
        if valid_values.size:
            median = float(np.median(valid_values))
            mad = float(np.median(np.abs(valid_values - median)))
            sigma = max(1.4826 * mad, 1.0 / 255.0)
            threshold = max(
                float(self.params["rgb_temporal_min_edge_gradient"]),
                median + float(self.params["rgb_temporal_edge_z_score"]) * sigma,
            )
        else:
            threshold = float(self.params["rgb_temporal_min_edge_gradient"])

        support = np.zeros(width, dtype=np.float64)
        counts = mask.sum(axis=0)
        for x in range(width):
            if counts[x] <= 0:
                continue
            column_valid = mask[:, x] & np.isfinite(gradient[:, x])
            if not column_valid.any():
                continue
            support[x] = float((gradient[:, x][column_valid] >= threshold).sum()) / float(column_valid.sum())

        strong_support = max(float(self.params["rgb_temporal_strong_line_support_fraction"]), 1e-6)
        score = np.clip(support / strong_support, 0.0, 1.0)
        return score, support

    def _rgb_line_width_px(self, residual: np.ndarray, candidate_x: int, valid: np.ndarray) -> int:
        if residual.size == 0:
            return 0
        valid_residual = residual[valid & np.isfinite(residual)]
        if valid_residual.size == 0:
            return 0
        threshold = max(
            float(self.params["rgb_temporal_line_width_min_residual"]),
            float(np.median(valid_residual)),
        )
        x = int(candidate_x)
        left = x
        while left > 0 and valid[left - 1] and np.isfinite(residual[left - 1]) and residual[left - 1] >= threshold:
            left -= 1
        right = x
        while (
            right < residual.size - 1
            and valid[right + 1]
            and np.isfinite(residual[right + 1])
            and residual[right + 1] >= threshold
        ):
            right += 1
        return int(right - left + 1)

    def _pipe_support_around_candidate(self, strip_mask: np.ndarray, candidate_x: int) -> dict[str, float | int | bool]:
        width = strip_mask.shape[1]
        x = int(candidate_x)
        side_width = int(self.params["rgb_temporal_pipe_end_side_width_px"])
        min_cols = int(self.params["rgb_temporal_pipe_end_min_side_columns_px"])
        left0 = max(0, x - side_width)
        left1 = max(0, x)
        right0 = min(width, x + 1)
        right1 = min(width, x + 1 + side_width)
        left = strip_mask[:, left0:left1]
        right = strip_mask[:, right0:right1]
        left_cols = max(0, left1 - left0)
        right_cols = max(0, right1 - right0)
        left_cov = float(left.mean()) if left.size else 0.0
        right_cov = float(right.mean()) if right.size else 0.0
        min_cov = float(self.params["rgb_temporal_pipe_end_min_side_coverage"])
        max_delta = float(self.params["rgb_temporal_pipe_end_max_coverage_delta"])
        pipe_end_rejected = bool(
            left_cols < min_cols
            or right_cols < min_cols
            or left_cov < min_cov
            or right_cov < min_cov
            or abs(left_cov - right_cov) > max_delta
        )
        return {
            "pipe_end_rejected": pipe_end_rejected,
            "left_cols": int(left_cols),
            "right_cols": int(right_cols),
            "left_coverage": left_cov,
            "right_coverage": right_cov,
        }

    def _pipe_end_rejected(self, strip_mask: np.ndarray, candidate_x: int) -> bool:
        return bool(self._pipe_support_around_candidate(strip_mask, candidate_x)["pipe_end_rejected"])

    def _pipe_end_rejected_columns(self, strip_mask: np.ndarray) -> np.ndarray:
        """Vectorized _pipe_end_rejected for every column at once.

        Same windowed side-coverage math as _pipe_support_around_candidate
        (block mean == mean of per-column coverages, all columns share the
        strip height), computed with one cumulative sum instead of one pair
        of 2D slices per column.
        """
        height, width = strip_mask.shape[:2]
        side_width = int(self.params["rgb_temporal_pipe_end_side_width_px"])
        min_cols = int(self.params["rgb_temporal_pipe_end_min_side_columns_px"])
        min_cov = float(self.params["rgb_temporal_pipe_end_min_side_coverage"])
        max_delta = float(self.params["rgb_temporal_pipe_end_max_coverage_delta"])
        xs = np.arange(width)
        left0 = np.maximum(0, xs - side_width)
        left1 = xs
        right0 = np.minimum(width, xs + 1)
        right1 = np.minimum(width, xs + 1 + side_width)
        left_cols = left1 - left0
        right_cols = right1 - right0
        col_sum = strip_mask.sum(axis=0, dtype=np.float64)
        csum = np.concatenate(([0.0], np.cumsum(col_sum)))
        denom_h = float(max(1, height))
        with np.errstate(divide="ignore", invalid="ignore"):
            left_cov = np.where(left_cols > 0, (csum[left1] - csum[left0]) / (denom_h * np.maximum(left_cols, 1)), 0.0)
            right_cov = np.where(right_cols > 0, (csum[right1] - csum[right0]) / (denom_h * np.maximum(right_cols, 1)), 0.0)
        return (
            (left_cols < min_cols)
            | (right_cols < min_cols)
            | (left_cov < min_cov)
            | (right_cov < min_cov)
            | (np.abs(left_cov - right_cov) > max_delta)
        )

    def _update_rgb_temporal_track(self, candidate_x: int, candidate_ok: bool) -> dict[str, float | int]:
        x = int(candidate_x)
        velocity = 0.0
        max_jump = int(self.params["rgb_temporal_track_max_jump_px"])
        max_missed = int(self.params["rgb_temporal_track_missed_max"])
        if not candidate_ok:
            self.rgb_track_missed_frames += 1
            if self.rgb_track_missed_frames > max_missed:
                self.rgb_track_x = None
                self.rgb_track_streak = 0
            return {
                "id": self.rgb_track_id,
                "streak": self.rgb_track_streak,
                "missed": self.rgb_track_missed_frames,
                "velocity": 0.0,
            }

        if self.rgb_track_x is None or abs(x - self.rgb_track_x) > max_jump:
            self.rgb_track_id += 1
            self.rgb_track_streak = 1
            self.rgb_track_missed_frames = 0
            velocity = 0.0
        else:
            velocity = float(x - self.rgb_track_x)
            self.rgb_track_streak += 1
            self.rgb_track_missed_frames = 0
        self.rgb_track_x = x
        return {
            "id": self.rgb_track_id,
            "streak": self.rgb_track_streak,
            "missed": self.rgb_track_missed_frames,
            "velocity": velocity,
        }

    def _temporal_scan_change(self, profile: np.ndarray, valid: np.ndarray, candidate_x: int) -> TemporalChangeResult:
        min_frames = int(self.params["temporal_reference_min_frames"])
        reference_ready = (
            self.temporal_reference_profile is not None
            and self.temporal_reference_valid is not None
            and self.temporal_reference_frame_count >= min_frames
            and self.temporal_reference_profile.shape == profile.shape
        )
        if not reference_ready:
            return TemporalChangeResult(
                score=0.0,
                dark_delta=0.0,
                z_score=0.0,
                accepted=False,
                reference_ready=False,
                reference_frame_count=self.temporal_reference_frame_count,
                reason="temporal_reference_not_ready",
            )

        reference = self.temporal_reference_profile
        reference_valid = self.temporal_reference_valid
        compare_valid = valid & reference_valid & np.isfinite(profile) & np.isfinite(reference)
        if int(compare_valid.sum()) < int(self.params["temporal_min_compare_columns"]):
            return TemporalChangeResult(
                score=0.0,
                dark_delta=0.0,
                z_score=0.0,
                accepted=False,
                reference_ready=True,
                reference_frame_count=self.temporal_reference_frame_count,
                reason="not_enough_temporal_compare_columns",
            )

        delta = reference - profile
        half_width = int(self.params["temporal_change_band_half_width_px"])
        x0 = max(0, int(candidate_x) - half_width)
        x1 = min(profile.size - 1, int(candidate_x) + half_width)
        band_valid = compare_valid[x0:x1 + 1]
        if int(band_valid.sum()) < int(self.params["temporal_min_band_columns"]):
            return TemporalChangeResult(
                score=0.0,
                dark_delta=0.0,
                z_score=0.0,
                accepted=False,
                reference_ready=True,
                reference_frame_count=self.temporal_reference_frame_count,
                reason="not_enough_temporal_band_columns",
            )

        dark_delta = float(max(0.0, np.median(delta[x0:x1 + 1][band_valid])))
        interior_delta = delta[compare_valid]
        median_delta = float(np.median(interior_delta))
        mad = float(np.median(np.abs(interior_delta - median_delta)))
        robust_sigma = max(1.4826 * mad, 1.0 / 255.0)
        z_score = (dark_delta - median_delta) / robust_sigma
        min_delta = float(self.params["min_temporal_change_dark_delta"])
        min_z = float(self.params["min_temporal_change_z_score"])
        score = float(np.clip((dark_delta - min_delta) / max(float(self.params["strong_temporal_change_dark_delta"]) - min_delta, 1e-6), 0.0, 1.0))
        accepted = bool(dark_delta >= min_delta and z_score >= min_z)
        reason = "temporal_change_ok" if accepted else "temporal_change_too_weak"
        return TemporalChangeResult(
            score=score,
            dark_delta=dark_delta,
            z_score=float(z_score),
            accepted=accepted,
            reference_ready=True,
            reference_frame_count=self.temporal_reference_frame_count,
            reason=reason,
        )

    def _update_temporal_reference(self, seam: SeamResult) -> None:
        if not bool(self.params["enable_temporal_scan_change"]):
            return
        if bool(self.params["temporal_reference_update_on_reject_only"]) and seam.local_candidate_accepted:
            return
        profile = seam.strip_profile.astype(np.float64)
        valid = seam.strip_profile_valid.astype(bool) & np.isfinite(profile)
        if int(valid.sum()) < int(self.params["temporal_min_compare_columns"]):
            return
        if self.temporal_reference_profile is None or self.temporal_reference_profile.shape != profile.shape:
            self.temporal_reference_profile = profile.copy()
            self.temporal_reference_valid = valid.copy()
            self.temporal_reference_frame_count = 1
            return
        alpha = float(self.params["temporal_reference_alpha"])
        ref_valid = self.temporal_reference_valid & valid
        if ref_valid.any():
            self.temporal_reference_profile[ref_valid] = (
                (1.0 - alpha) * self.temporal_reference_profile[ref_valid] + alpha * profile[ref_valid]
            )
        newly_valid = valid & ~self.temporal_reference_valid
        self.temporal_reference_profile[newly_valid] = profile[newly_valid]
        self.temporal_reference_valid |= valid
        self.temporal_reference_frame_count += 1

    def _depth_gap_evidence(
        self,
        rotated_depth: np.ndarray,
        rotated_mask: np.ndarray,
        crop_xyxy: list[int],
        candidate_x_strip_px: int,
    ) -> dict[str, float]:
        x0, y0, x1, y1 = [int(v) for v in crop_xyxy]
        x = x0 + int(candidate_x_strip_px)
        y0 = max(0, y0)
        y1 = min(rotated_depth.shape[0] - 1, y1)
        half_width = int(self.params["depth_gap_band_half_width_px"])
        neighbor_offset = int(self.params["depth_gap_neighbor_offset_px"])

        def band(cx: int) -> tuple[np.ndarray, float]:
            bx0 = max(0, cx - half_width)
            bx1 = min(rotated_depth.shape[1] - 1, cx + half_width)
            depth_band = rotated_depth[y0:y1 + 1, bx0:bx1 + 1]
            mask_band = rotated_mask[y0:y1 + 1, bx0:bx1 + 1]
            valid = mask_band & np.isfinite(depth_band) & (depth_band > 0.0)
            coverage = float(valid.sum()) / max(float(valid.size), 1.0)
            return depth_band[valid], coverage

        center_depth, center_coverage = band(x)
        left_depth, left_coverage = band(x - neighbor_offset)
        right_depth, right_coverage = band(x + neighbor_offset)
        neighbor_depth = np.concatenate([left_depth, right_depth]) if left_depth.size and right_depth.size else np.array([])
        neighbor_coverage = 0.5 * (left_coverage + right_coverage)

        if center_depth.size < int(self.params["min_depth_gap_samples"]) or neighbor_depth.size < int(self.params["min_depth_gap_samples"]):
            depth_jump = 0.0
        else:
            depth_jump = abs(float(np.median(center_depth)) - float(np.median(neighbor_depth)))
        coverage_drop = max(0.0, neighbor_coverage - center_coverage)
        return {
            "score": max(depth_jump, coverage_drop * float(self.params["depth_gap_coverage_score_scale_m"])),
            "depth_jump_m": depth_jump,
            "coverage_drop": coverage_drop,
            "center_coverage": center_coverage,
            "neighbor_coverage": neighbor_coverage,
        }

    def _column_profile(self, gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = gray.shape
        counts = mask.sum(axis=0)
        min_count = max(8, int(float(self.params["min_valid_column_fraction"]) * height))
        masked = np.where(mask, gray, np.nan)
        with np.errstate(all="ignore"):
            profile = np.nanmedian(masked, axis=0)
        profile[counts < min_count] = np.nan

        good = np.isfinite(profile)
        if not good.any():
            raise ValueError("No valid strip columns for seam scoring")
        if not good.all():
            xi = np.arange(width)
            profile[~good] = np.interp(xi[~good], xi[good], profile[good])
        return profile, counts

    def _update_state(self, tracker: TrackerResult, seam: SeamResult) -> dict[str, Any]:
        current_geometry = {
            "pipe_axis_angle_deg": tracker.image_axis_angle_deg,
            "stand_off_m": tracker.stand_off_m,
            "yaw_error_deg": tracker.yaw_error_deg,
        }
        if seam.junction_acceptance_mode in ("rgb_temporal", "variant_a_rgb"):
            geometry_ok, geometry_failures = True, []
        else:
            geometry_ok, geometry_failures = self._geometry_consistent(current_geometry)
        candidate_not_border = self._candidate_not_border(seam)
        confidence_ok = seam.confidence >= float(self.params["min_confidence"])
        jump_ok = True
        jump_reason = ""
        if seam.junction_acceptance_mode != "rgb_temporal" and self.previous_candidate_x is not None:
            jump = abs(seam.candidate_x_strip_px - self.previous_candidate_x)
            if jump > int(self.params["max_candidate_jump_px"]):
                jump_ok = False
                jump_reason = f"candidate_jump={jump}px"

        eligible = (seam.accepted and confidence_ok and candidate_not_border
                    and geometry_ok and jump_ok and not seam.appearance_veto)
        fresh_reacquire_ok = True
        fresh_reacquire_reason = ""
        if (
            eligible
            and not self.junction_lock_active
            and self.junction_last_valid_x is not None
            and (self.pipe_lock_model is not None or self.pipe_image_lock_model is not None)
            and str(self.junction_lock_source).startswith("released:")
        ):
            fresh_jump = abs(float(seam.candidate_x_strip_px) - float(self.junction_last_valid_x))
            blind_frames = 1
            if self.junction_last_valid_frame is not None:
                blind_frames = max(1, int(self.processed_frame_count) - int(self.junction_last_valid_frame))
            # The longer the detector has been blind, the farther the junction
            # can legitimately have moved (measured sweep speed on the real bag:
            # up to ~20 px/frame). A fixed gate blocked valid re-locks after a
            # long outage; the cap still blocks single-frame teleports.
            max_fresh_jump = min(
                float(self.params["junction_fresh_reacquire_max_jump_px"]),
                float(self.params["junction_fresh_reacquire_base_px"])
                + float(self.params["junction_fresh_reacquire_px_per_frame"]) * float(blind_frames),
            )
            if fresh_jump > max_fresh_jump:
                eligible = False
                fresh_reacquire_ok = False
                fresh_reacquire_reason = f"fresh_reacquire_jump={int(round(fresh_jump))}px>{int(round(max_fresh_jump))}px"
        mask_gate_reason: str | None = None
        if eligible:
            # A physical junction always leaves SOME strip-level trace (dark
            # line or luminance step). A fresh candidate whose strip evidence
            # is flat is a rendering/rotation artifact (e.g. Variant A corner
            # fill), never a seam. Lock hold/coast paths are not affected.
            strip_evidence = max(
                float(seam.candidate_z_score),
                float(seam.candidate_step_z),
                float(seam.candidate_evidence_z),
            )
            fresh_min_evidence = float(self.params["junction_fresh_min_evidence_z"])
            if self._coloroff_guard_frame_active and not self.junction_lock_active:
                # Color-less mode lacks the warm mask that keeps the seam strip
                # on the true pipe, so the seam frontend fires on smooth-pipe
                # specular/edge steps (measured false f65: strip evidence 5.0)
                # long before the real collar (7.8-12.6). Demand a strong,
                # collar-grade step before a FRESH color-less junction may lock;
                # tracking/coast frames are unaffected (they never reach here).
                fresh_min_evidence = max(
                    fresh_min_evidence,
                    float(self.params["junction_fresh_min_evidence_z_coloroff"]),
                )
            if strip_evidence < fresh_min_evidence:
                eligible = False
                mask_gate_reason = f"no_strip_evidence={strip_evidence:.2f}"
        if eligible and bool(self.params["junction_require_cylinder_mask"]) and self.pipe_component_selected_model is None:
            eligible = False
            mask_gate_reason = "no_cylinder_validated_mask"
        if eligible and self._coloroff_guard_frame_active:
            # No color anchor: a fresh junction may only come from a pipe the
            # cylinder guard has locked, and never from a frame whose mask is
            # not cylindrical (lock hold/coast still carries short outages).
            if self.pipe_lock_model is None:
                eligible = False
                need = int(self.params["pipe_coloroff_acquire_stable_frames"])
                mask_gate_reason = (
                    f"coloroff_waiting_stable_cylinder={self._coloroff_acquire_streak}/{need}"
                )
            elif not self._coloroff_cyl_ok:
                eligible = False
                mask_gate_reason = f"coloroff_not_cylindrical:{self._coloroff_cyl_reason}"
        if eligible and bool(self.params["enable_junction_axial_support_gate"]):
            support_ok, support_reason = self._junction_axial_support_ok(seam)
            if not support_ok:
                eligible = False
                mask_gate_reason = support_reason
        if (
            eligible
            and self._warm_scene_ok
            and self.pipe_mask_warm_fraction is not None
            and self.pipe_mask_warm_fraction < float(self.params["junction_min_mask_warm_fraction"])
        ):
            eligible = False
            mask_gate_reason = f"low_warm_mask={self.pipe_mask_warm_fraction:.2f}"
        if (
            eligible
            and self.pipe_mask_normal_fraction is not None
            and self.pipe_mask_normal_fraction < float(self.params["pipe_mask_min_normal_fraction"])
        ):
            eligible = False
            mask_gate_reason = f"low_normal_mask={self.pipe_mask_normal_fraction:.2f}"
        lock_used = False
        lock_reason = "none"

        if eligible:
            self._update_junction_lock(seam)
            self.candidate_streak += 1
            if self.candidate_streak >= int(self.params["min_confirm_frames"]):
                self.state = "STOP_AND_LOCALIZE"
                if self.confirmed_frame_count is None:
                    self.confirmed_frame_count = self.processed_frame_count
                if (bool(self.params.get("enable_appearance_veto", False))
                        and self.appearance_template is None
                        and self._last_appearance_patch is not None):
                    self.appearance_template = self._last_appearance_patch
            else:
                self.state = "CANDIDATE"
        else:
            lock_used, lock_reason = self._try_hold_junction_lock(seam, geometry_ok)
            if lock_used:
                eligible = True
                confidence_ok = True
                candidate_not_border = self._candidate_not_border(seam)
                self.candidate_streak = max(self.candidate_streak, int(self.params["min_confirm_frames"]))
                self.state = "STOP_AND_LOCALIZE"
                if self.confirmed_frame_count is None:
                    self.confirmed_frame_count = self.processed_frame_count
            else:
                self.candidate_streak = 0
                self.state = "SCAN"
                self.confirmed_frame_count = None
                self.appearance_template = None

        if eligible:
            self._update_pipe_lock(tracker)
            self._update_pipe_image_lock()
            self.previous_geometry = current_geometry
            self.previous_candidate_x = seam.candidate_x_strip_px
            self.junction_last_valid_x = float(seam.candidate_x_strip_px)
            self.junction_last_valid_frame = int(self.processed_frame_count)
        elif bool(self.params["reset_geometry_on_reject"]):
            self.previous_geometry = None
            self.previous_candidate_x = None

        reason_parts = []
        if mask_gate_reason:
            reason_parts.append(mask_gate_reason)
        if not seam.accepted:
            reason_parts.append("detector_rejected")
        if not seam.visual_frontend_accepted:
            reason_parts.append("rgb_dark_rejected")
        if seam.temporal_change_gate_enabled and not seam.temporal_change_accepted:
            reason_parts.append(f"temporal_change_rejected:{seam.temporal_change_reason}")
        if not confidence_ok:
            reason_parts.append("low_confidence")
        if not candidate_not_border:
            reason_parts.append("candidate_near_border")
        if not geometry_ok:
            reason_parts.extend(geometry_failures)
        if not jump_ok:
            reason_parts.append(jump_reason)
        if seam.appearance_veto:
            reason_parts.append(f"appearance_veto={seam.appearance_ncc:.2f}")
        if not fresh_reacquire_ok:
            reason_parts.append(fresh_reacquire_reason)
        if lock_used:
            reason_parts.append(f"junction_track:{lock_reason}")
        if eligible:
            reason_parts.append("eligible")
        if self.state == "STOP_AND_LOCALIZE":
            reason_parts.append("stop_and_localize")

        return {
            "eligible": eligible,
            "jump_ok": jump_ok,
            "fresh_reacquire_ok": fresh_reacquire_ok,
            "fresh_reacquire_reason": fresh_reacquire_reason,
            "candidate_not_border": candidate_not_border,
            "geometry_consistent": geometry_ok,
            "reason": ";".join(reason_parts),
            "junction_lock_active": self.junction_lock_active,
            "junction_lock_used": lock_used,
            "junction_lock_reason": lock_reason,
            "junction_lock_x_strip_px": None if self.junction_lock_x is None else float(self.junction_lock_x),
            "junction_last_valid_x_strip_px": None if self.junction_last_valid_x is None else float(self.junction_last_valid_x),
            "junction_last_valid_frame": self.junction_last_valid_frame,
            "junction_lock_velocity_px_per_frame": float(self.junction_lock_velocity_px),
            "junction_lock_streak": int(self.junction_lock_streak),
            "junction_lock_missed_frames": int(self.junction_lock_missed_frames),
            "junction_lock_confidence": float(self.junction_lock_confidence),
            "junction_lock_source": self.junction_lock_source,
        }

    def _fuse_junction_measurement(self, seam: SeamResult, measured_x: float) -> float:
        """Predictor-corrector on the published junction column.

        The raw per-frame candidate wobbles by +-20-30 px during fast pans
        (the collar has two edges and motion blur moves the strongest cue
        between them), so publishing the measurement directly makes the
        junction line JUMP. The junction is rigidly attached to the pipe:
        the KLT median flow of the pipe texture predicts how the locked
        column moved, and the measurement only corrects that prediction with
        an innovation-dependent gain. Large innovations snap (a re-detection
        after a blind stretch is real motion, not noise), and a persistent
        one-sided lag boosts the gain so the track can never trail behind a
        sustained drift the flow under-measures.
        """
        if (
            not bool(self.params["enable_junction_output_smoothing"])
            or self.junction_lock_x is None
            or not self.junction_lock_active
        ):
            self._junction_lag_ema = 0.0
            return float(measured_x)
        lock_x = float(self.junction_lock_x)
        klt_dx = float(seam.klt_dx_px)
        if str(seam.klt_status) == "tracked" and np.isfinite(klt_dx) and abs(klt_dx) <= float(
            self.params["junction_lock_coast_max_dx_px"]
        ):
            predicted = lock_x + klt_dx
        else:
            predicted = lock_x + float(self.junction_lock_velocity_px)
        innovation = float(measured_x) - predicted
        snap_px = float(self.params["junction_smooth_snap_px"])
        if abs(innovation) >= snap_px:
            self._junction_lag_ema = 0.0
            return float(measured_x)
        soft_px = float(self.params["junction_smooth_innovation_soft_px"])
        gain = (
            float(self.params["junction_smooth_gain"])
            if abs(innovation) <= soft_px
            else float(self.params["junction_smooth_gain_far"])
        )
        # Sustained one-sided innovation = the filter is trailing real motion:
        # raise the gain until the bias is gone.
        self._junction_lag_ema = 0.7 * self._junction_lag_ema + 0.3 * innovation
        if abs(self._junction_lag_ema) > soft_px:
            gain = max(gain, 0.7)
        fused = predicted + gain * innovation
        width = int(seam.strip_size_wh[0])
        return float(np.clip(fused, 0.0, float(max(width - 1, 0))))

    def _apply_smoothed_candidate(self, seam: SeamResult, fused_x: float) -> None:
        seam.candidate_x_raw_strip_px = int(seam.candidate_x_strip_px)
        fused_i = int(round(fused_x))
        seam.candidate_x_strip_px = fused_i
        seam.candidate_x_rotated_px = int(seam.crop_xyxy[0] + fused_i)

    def _update_junction_lock(self, seam: SeamResult) -> None:
        fused = self._fuse_junction_measurement(seam, float(seam.candidate_x_strip_px))
        if self.junction_lock_x is None or not self.junction_lock_active:
            self.junction_lock_velocity_px = 0.0
            self.junction_lock_streak = 1
            self.junction_lock_start_frame = int(self.processed_frame_count)
        else:
            measured_velocity = fused - float(self.junction_lock_x)
            # Smooth velocity so camera/base motion can be followed without
            # jumping to one-frame outliers.
            self.junction_lock_velocity_px = 0.7 * self.junction_lock_velocity_px + 0.3 * measured_velocity
            self.junction_lock_streak += 1
        self.junction_lock_x = fused
        self._apply_smoothed_candidate(seam, fused)
        self.junction_lock_active = True
        self.junction_lock_missed_frames = 0
        self.junction_lock_confidence = max(float(seam.confidence), float(self.params["junction_lock_min_confidence"]))
        self.junction_lock_source = "fresh_detection"

    def _release_junction_lock(self, reason: str) -> tuple[bool, str]:
        self._junction_lag_ema = 0.0
        self._junction_center_state = None
        self._junction_center_lag = np.zeros(3, dtype=np.float64)
        self.junction_lock_active = False
        self.junction_lock_x = None
        self.junction_lock_velocity_px = 0.0
        self.junction_lock_streak = 0
        self.junction_lock_start_frame = None
        self.junction_lock_missed_frames = 0
        self.junction_lock_confidence = 0.0
        self.junction_lock_source = f"released:{reason}"
        # Track lost: clear the continuity references too, so the NEXT fresh
        # detection is re-acquired clean and is NOT gated against the stale
        # pre-loss column. Without this the detector stays poisoned after an exit
        # (a good re-appearing junction is rejected forever with
        # candidate_jump=<big>px, because previous_candidate_x kept the old value).
        self.previous_candidate_x = None
        self.previous_geometry = None
        # Same poisoning applies to the fresh-reacquire anchor: after the end
        # of the pipe left the view, the REAL junction entering from the other
        # side was rejected for ~6 s with fresh_reacquire_jump=500px>300px
        # because junction_last_valid_x still pointed at the dead track.
        self.junction_last_valid_x = None
        self.junction_last_valid_frame = None
        return False, reason

    def _try_hold_junction_lock(self, seam: SeamResult, geometry_ok: bool) -> tuple[bool, str]:
        if not bool(self.params["enable_junction_lock"]):
            return False, "disabled"
        if (
            self._warm_scene_ok
            and self.pipe_mask_warm_fraction is not None
            and self.pipe_mask_warm_fraction < float(self.params["junction_min_mask_warm_fraction"])
        ):
            # Never hold/coast-publish on a mask that does not look like the
            # pipe: this is what kept the cloth junction alive for 150 frames.
            return False, "low_warm_mask"
        if (
            self.pipe_mask_normal_fraction is not None
            and self.pipe_mask_normal_fraction < float(self.params["pipe_mask_min_normal_fraction"])
        ):
            return False, "low_normal_mask"
        if not self.junction_lock_active or self.junction_lock_x is None:
            return False, "not_locked"
        if not geometry_ok:
            return self._release_junction_lock("geometry_changed")

        self.junction_lock_missed_frames += 1
        max_missed = int(self.params["junction_lock_max_missed_frames"])
        if self.junction_lock_missed_frames > max_missed:
            return self._release_junction_lock("missed_too_long")

        width = int(seam.strip_size_wh[0])
        if width <= 0:
            return self._release_junction_lock("invalid_strip")
        predicted = float(self.junction_lock_x) + float(self.junction_lock_velocity_px)
        measured_i = int(seam.candidate_x_strip_px)
        search_radius = float(self.params["junction_lock_search_radius_px"])
        if abs(float(measured_i) - predicted) > search_radius:
            self._coast_lock_with_flow(seam)
            if self._publish_coasted_lock(seam):
                return True, "coasted"
            return False, "measurement_far_from_track"

        # The lock is a prior, not a hallucination: keep publishing only if the
        # current frame still contains seam-like evidence near the predicted
        # position. This lets the seam move with camera/robot motion while
        # avoiding a static pose in empty image space. TWO kinds of evidence
        # count: the thin dark seam line (sharp, close, static) OR the collar's
        # broad luminance step (survives motion blur on the real pipe).
        dark_reacquire = bool(
            seam.rgb_line_width_px <= int(self.params["rgb_temporal_max_line_width_px"])
            and float(seam.candidate_z_score) >= float(self.params["junction_lock_min_reacquire_z"])
            and float(seam.candidate_contrast) >= float(self.params["junction_lock_min_reacquire_contrast"])
        )
        step_reacquire = bool(
            float(seam.candidate_step_z) >= float(self.params["step_edge_reacquire_min_z"])
            and float(seam.candidate_step_abs) >= float(self.params["step_edge_min_abs"])
        )
        visual_reacquire = bool(not seam.pipe_end_rejected and (dark_reacquire or step_reacquire))
        if not visual_reacquire:
            self._coast_lock_with_flow(seam)
            if self._publish_coasted_lock(seam):
                return True, "coasted"
            return False, "no_visual_reacquire"

        if not (seam.edge_margin_px <= measured_i < width - seam.edge_margin_px):
            return self._release_junction_lock("out_of_view")

        if self._pipe_end_memory and self.pipe_component_selected_model is not None:
            comps = self._pipe_axis_model_components_from_model(self.pipe_component_selected_model)
            if comps is not None:
                k_cam = np.array(
                    [
                        [float(self.params["last_fx"]), 0.0, float(self.params["last_cx"])],
                        [0.0, float(self.params["last_fy"]), float(self.params["last_cy"])],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                centre = self._junction_axis_center_from_image_line(seam, k_cam, comps)
                if centre is not None:
                    s_now = float(np.dot(centre - comps[1], comps[0]))
                    mem = self._remembered_end_near(s_now)
                    if mem is not None:
                        # Relative age decides who is the impostor. A memory
                        # confirmed around the time the lock formed means the
                        # track was acquired ON a pipe end: kill it, an end
                        # never becomes a junction. A memory born when the
                        # lock was already mature is the artifact (specular /
                        # FOV shortening of the extent around the tracked
                        # junction at close range, measured f849 on the cloth
                        # bag: it banned the true junction for ~400 frames) —
                        # the lock's accumulated evidence wins.
                        start = self.junction_lock_start_frame
                        mem_birth = float(mem[2]) if len(mem) > 2 else 0.0
                        if start is None or mem_birth <= float(start) + 10.0:
                            return self._release_junction_lock("near_remembered_pipe_end")
                        # Misdiagnosed end: delete it, or it keeps banning
                        # fresh reacquisition if the lock later dies on blur.
                        for key, val in list(self._pipe_end_memory.items()):
                            if val is mem:
                                del self._pipe_end_memory[key]
                                self._pipe_end_streak.pop(key, None)

        decay = float(self.params["junction_lock_confidence_decay"])
        self.junction_lock_confidence = max(
            self.junction_lock_confidence * decay,
            float(seam.confidence) * decay,
            float(self.params["junction_lock_min_confidence"]),
        )
        if self.junction_lock_confidence < float(self.params["junction_lock_min_confidence"]):
            return self._release_junction_lock("confidence_decayed")

        fused = self._fuse_junction_measurement(seam, float(measured_i))
        measured_velocity = fused - float(self.junction_lock_x)
        self.junction_lock_velocity_px = 0.7 * self.junction_lock_velocity_px + 0.3 * measured_velocity
        self.junction_lock_x = fused
        self.junction_lock_source = "reacquired_measurement"
        # The lock FOUND seam evidence near the predicted position this frame, so
        # this is NOT a miss -> reset the miss counter. Otherwise missed_frames
        # (only reset on a fresh, fully-gated detection in _update_junction_lock)
        # keeps climbing while the lock tracks the junction fine via the relaxed
        # reacquire, and the lock is force-released after max_missed frames even
        # though the junction is visible the whole time. Observed live: a clearly
        # framed junction was held by the lock for exactly 12 reacquired frames
        # (junction_lock_source=reacquired_measurement, missed 9->12), then
        # released "missed_too_long" and the candidate jumped to the image border.
        # missed_frames must count only consecutive frames with NO evidence
        # (no_visual_reacquire / measurement_far), so the lock can bridge an
        # arbitrarily long run of strict-detection failures as long as the seam is
        # still there.
        self.junction_lock_missed_frames = 0

        seam.candidate_x_raw_strip_px = int(measured_i)
        fused_i = int(round(fused))
        seam.candidate_x_strip_px = fused_i
        seam.candidate_x_rotated_px = int(seam.crop_xyxy[0] + fused_i)
        seam.confidence = float(self.junction_lock_confidence)
        seam.accepted = True
        seam.local_candidate_accepted = True
        seam.visual_frontend_accepted = True
        seam.rgb_dark_accepted = True
        seam.negative_gate_reason = "junction_lock_reacquired"
        return True, "reacquired_measurement"

    def _coast_lock_with_flow(self, seam: SeamResult) -> None:
        """Dead-reckon the locked junction column on the pipe's optical flow.

        The junction is rigidly attached to the pipe, so when the seam itself
        has no visual evidence (motion blur during a fast sweep) the KLT median
        dx of the pipe texture still measures how far the junction moved in the
        image. Without this the lock keeps predicting from a stale column and,
        after a long blind stretch, the true junction reappears OUTSIDE the
        search radius -> release -> fresh-reacquire trouble. Coasting only
        recentres the search window; it never publishes a pose by itself.
        """
        if not self.junction_lock_active or self.junction_lock_x is None:
            return
        if str(seam.klt_status) != "tracked":
            return
        dx = float(seam.klt_dx_px)
        max_dx = float(self.params["junction_lock_coast_max_dx_px"])
        if not np.isfinite(dx) or abs(dx) > max_dx:
            return
        width = int(seam.strip_size_wh[0])
        self.junction_lock_x = float(np.clip(self.junction_lock_x + dx, 0.0, float(max(width - 1, 0))))
        self.junction_lock_velocity_px = 0.7 * self.junction_lock_velocity_px + 0.3 * dx

    def _publish_coasted_lock(self, seam: SeamResult) -> bool:
        """Publish the KLT-coasted lock during SHORT blind stretches.

        Without this the junction line (and the centre marker) disappears the
        moment motion blur kills the seam evidence, even though the lock knows
        exactly where the junction moved (the pipe's optical flow measured
        it). Dead-reckoning is published only while it stays trustworthy:
        flow tracked this frame, a bounded number of consecutive blind
        frames, and the coasted column safely inside the strip (a junction
        actually LEAVING the view must not be dragged along the border).
        """
        if not bool(self.params["enable_junction_coast_publish"]):
            return False
        if not self.junction_lock_active or self.junction_lock_x is None:
            return False
        if self.junction_lock_missed_frames > int(self.params["junction_publish_coast_max_frames"]):
            return False
        if str(seam.klt_status) != "tracked":
            return False
        width = int(seam.strip_size_wh[0])
        margin = int(seam.edge_margin_px) + int(self.params["junction_coast_publish_edge_margin_px"])
        x = float(self.junction_lock_x)
        if not (margin <= x < width - margin):
            return False
        decay = float(self.params["junction_lock_confidence_decay"])
        self.junction_lock_confidence = max(
            self.junction_lock_confidence * decay,
            float(self.params["junction_lock_min_confidence"]),
        )
        seam.candidate_x_raw_strip_px = int(seam.candidate_x_strip_px)
        fused_i = int(round(x))
        seam.candidate_x_strip_px = fused_i
        seam.candidate_x_rotated_px = int(seam.crop_xyxy[0] + fused_i)
        seam.confidence = float(self.junction_lock_confidence)
        seam.accepted = True
        seam.local_candidate_accepted = True
        seam.visual_frontend_accepted = True
        seam.rgb_dark_accepted = True
        seam.negative_gate_reason = "junction_lock_coasted"
        self.junction_lock_source = "coasted"
        return True

    def _geometry_consistent(self, current: dict[str, float | None]) -> tuple[bool, list[str]]:
        if self.previous_geometry is None:
            return True, []

        failures: list[str] = []
        axis_delta = abs(float(current["pipe_axis_angle_deg"]) - float(self.previous_geometry["pipe_axis_angle_deg"]))
        if axis_delta > float(self.params["max_axis_angle_delta_deg"]):
            failures.append(f"axis_delta={axis_delta:.3f}deg")

        stand_delta = abs(float(current["stand_off_m"]) - float(self.previous_geometry["stand_off_m"]))
        if stand_delta > float(self.params["max_stand_off_delta_m"]):
            failures.append(f"stand_off_delta={stand_delta:.3f}m")

        yaw_delta = abs(float(current["yaw_error_deg"]) - float(self.previous_geometry["yaw_error_deg"]))
        if yaw_delta > float(self.params["max_yaw_delta_deg"]):
            failures.append(f"yaw_delta={yaw_delta:.3f}deg")

        return not failures, failures

    def _candidate_not_border(self, seam: SeamResult) -> bool:
        if seam.pipe_end_rejected:
            return False
        width = int(seam.strip_size_wh[0])
        x = int(seam.candidate_x_strip_px)
        if seam.edge_margin_px <= x < width - seam.edge_margin_px:
            return True
        return bool(
            bool(self.params["candidate_border_allow_if_pipe_supported"])
            and not seam.pipe_end_rejected
        )

    def _pipe_axis_model_components(
        self,
        tracker: TrackerResult,
        model: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            if model is not None:
                axis = _normalize(np.asarray(model["axis"], dtype=np.float64).reshape(3))
                axis_point = np.asarray(model["axis_point"], dtype=np.float64).reshape(3)
            else:
                axis = _normalize(np.asarray(tracker.pipe_axis_xyz, dtype=np.float64).reshape(3))
                axis_point = np.asarray(tracker.centroid_xyz_m, dtype=np.float64).reshape(3)
            if not np.isfinite(axis).all() or not np.isfinite(axis_point).all():
                return None
            if float(np.linalg.norm(axis)) < 1e-9:
                return None
            return axis, axis_point
        except Exception:
            return None

    def _junction_axis_center_from_image_line(
        self,
        seam: SeamResult,
        k: np.ndarray,
        components: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray | None:
        """Junction centre from geometry only: closest point between the pipe
        AXIS and the camera ray through the junction line's midpoint. No depth
        is sampled at the collar, so specular holes / blur cannot move it."""
        axis, axis_point = components
        try:
            line_uv = self._candidate_line_original_uv(seam)
            u = 0.5 * (float(line_uv[0][0]) + float(line_uv[1][0]))
            v = 0.5 * (float(line_uv[0][1]) + float(line_uv[1][1]))
            fx, fy = float(self.params["last_fx"]), float(self.params["last_fy"])
            cx, cy = float(self.params["last_cx"]), float(self.params["last_cy"])
            if k is not None and k.shape == (3, 3):
                fx, fy = float(k[0, 0]), float(k[1, 1])
                cx, cy = float(k[0, 2]), float(k[1, 2])
            ray = _normalize(np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64))
            a_mat = np.stack([ray, -axis], axis=1)
            sol, *_ = np.linalg.lstsq(a_mat, axis_point, rcond=None)
            s = float(sol[1])
            center = axis_point + s * axis
            if not np.isfinite(center).all() or center[2] <= 0.0:
                return None
            return center
        except Exception:
            return None

    def _update_pipe_end_memory(self, k: np.ndarray) -> None:
        extent = self._reproject_axial_extent
        model = self.pipe_component_selected_model
        max_age = int(self.params["pipe_end_memory_max_age_frames"])
        # age existing entries; drop stale ones
        for side in list(self._pipe_end_memory.keys()):
            self._pipe_end_memory[side][1] += 1
            if self._pipe_end_memory[side][1] > max_age:
                del self._pipe_end_memory[side]
        if extent is None or model is None:
            self._pipe_end_prev_components = None
            return
        comps = self._pipe_axis_model_components_from_model(model)
        if comps is None:
            self._pipe_end_prev_components = None
            return
        axis, point = comps
        # The s origin is the model axis point IN CAMERA FRAME: camera motion
        # shifts every physical point's s by the projection of the axis-point
        # displacement. Memories are positions of PHYSICAL ends — move them
        # with the parameterization, NOT with the observed extent end (which
        # during an approach is the FOV cutoff: following it dragged the f4
        # socket memory from s=0.414 through the true junction's s, banning
        # ~130 frames of f200-799 on the cloth bag).
        prev = getattr(self, "_pipe_end_prev_components", None)
        if prev is not None and self._pipe_end_memory:
            prev_axis, prev_point = prev
            if float(np.dot(prev_axis, axis)) > 0.9:
                delta_s = float(np.dot(prev_point - point, axis))
                if abs(delta_s) <= 0.25:
                    for mem in self._pipe_end_memory.values():
                        mem[0] += delta_s
        self._pipe_end_prev_components = (axis.copy(), point.copy())
        # NO reattach-refresh here: while an end is still measurable, the
        # axial gate itself keeps rejecting it AND its streak/creation path
        # re-anchors the memory on the measured extent. This method only
        # carries memories through camera motion (parameterization shift
        # above) and ages them out ~90 frames after the end was last
        # confirmed. A reattach heuristic that followed nearby extent ends
        # hijacked onto specular/FOV cutoffs and dragged the f4 socket ban
        # through the true junction's s (age stayed 0 from f5 to f344+).
        if os.environ.get("ACEA_END_MEM_DEBUG") is not None and self._pipe_end_memory:
            state = {
                k2: (round(v[0], 3), int(v[1]), int(v[2]) if len(v) > 2 else -1)
                for k2, v in self._pipe_end_memory.items()
            }
            print(f"ENDMEM f{self.processed_frame_count - 1} STATE {state}", file=sys.stderr)

    def _remembered_end_near(self, s_j: float) -> list[float] | None:
        ban = float(self.params["pipe_end_memory_ban_radius_m"])
        for mem in self._pipe_end_memory.values():
            if abs(s_j - float(mem[0])) <= ban:
                return mem
        return None

    def _junction_near_remembered_end(self, s_j: float) -> bool:
        return self._remembered_end_near(s_j) is not None

    def _end_unmeasured_beyond(
        self,
        u_end: float,
        v_end: float,
        du: float,
        dv: float,
    ) -> bool:
        """True when the image region just BEYOND a measured extent end (in
        the axis direction away from the pipe body) is mostly depth-invalid:
        the surface visibly continues but cannot be measured (grazing angle /
        defocus at close range), so the extent end is a sensing artifact, not
        a physical pipe end. A real end shows VALID far depth beyond it
        (background behind the socket mouth)."""
        depth = getattr(self, "_last_depth_image", None)
        if depth is None:
            return False
        h, w = depth.shape[:2]
        norm = float(np.hypot(du, dv))
        if norm < 1e-6:
            return False
        du, dv = du / norm, dv / norm
        total = 0
        invalid = 0
        for t in range(6, 50, 4):
            for off in (-6.0, 0.0, 6.0):
                u = int(round(u_end + du * t - dv * off))
                v = int(round(v_end + dv * t + du * off))
                if not (0 <= u < w and 0 <= v < h):
                    continue
                total += 1
                z = float(depth[v, u])
                if not np.isfinite(z) or z <= 0.0:
                    invalid += 1
        if total < 10:
            return False
        return invalid >= 0.5 * total

    def _junction_axial_support_ok(self, seam: SeamResult) -> tuple[bool, str]:
        """A junction has pipe on BOTH sides along the axis; a pipe END does
        not. Requires the observed cylinder surface to extend at least
        junction_min_axial_support_m beyond the candidate on each side —
        except where the view itself is truncated by the image border (there
        the missing side gets the benefit of the doubt, so a junction
        entering the frame is not delayed)."""
        model = self.pipe_component_selected_model
        extent = self._reproject_axial_extent
        if model is None or extent is None:
            return True, ""
        components = self._pipe_axis_model_components_from_model(model)
        if components is None:
            return True, ""
        axis, point = components
        k = np.array(
            [
                [float(self.params["last_fx"]), 0.0, float(self.params["last_cx"])],
                [0.0, float(self.params["last_fy"]), float(self.params["last_cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        centre = self._junction_axis_center_from_image_line(seam, k, components)
        if centre is None:
            return True, ""
        s_j = float(np.dot(centre - point, axis))
        # NOTE: the ban is checked AFTER the streak/creation update below, so
        # a banned-but-still-visible end keeps re-anchoring its memory on the
        # measured extent every frame (an early return here froze the memory,
        # and the reattach heuristic that compensated hijacked onto specular
        # extent ends, dragging the ban onto the true junction).
        min_side = float(self.params["junction_min_axial_support_m"])

        def border_s(x_border: float) -> float | None:
            projected = self._project_axis_to_image(model, k)
            if projected is None:
                return None
            c_uv, d_uv = projected
            if abs(float(d_uv[0])) < 1e-6:
                return None
            v = float(c_uv[1]) + float(d_uv[1]) / float(d_uv[0]) * (x_border - float(c_uv[0]))
            fx, fy = float(k[0, 0]), float(k[1, 1])
            cx, cy = float(k[0, 2]), float(k[1, 2])
            ray = _normalize(np.array([(x_border - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64))
            a_mat = np.stack([ray, -axis], axis=1)
            sol, *_ = np.linalg.lstsq(a_mat, point, rcond=None)
            return float(sol[1])

        width = float(self.params["last_cx"]) * 2.0
        s_b0 = border_s(4.0)
        s_b1 = border_s(width - 4.0)
        if s_b0 is None or s_b1 is None:
            return True, ""
        s_min_vis, s_max_vis = min(s_b0, s_b1), max(s_b0, s_b1)
        support_left = s_j - extent[0]
        support_right = extent[1] - s_j
        # Benefit of the doubt ONLY where the pipe MASK itself touches the
        # image border in pixels (the pipe visibly continues out of frame).
        # The socket end shows its mouth INSIDE the frame, so no doubt: the
        # bell plus its visible inner wall measure ~12 cm of "support" while
        # the pipe genuinely ends there — only the full requirement separates
        # that from a real junction.
        truncated_left = False
        truncated_right = False
        unmeasured_left = False
        unmeasured_right = False
        img_w = float(self.params["last_cx"]) * 2.0
        img_h = float(self.params["last_cy"]) * 2.0
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        # Doubt is granted generously near the border: the confirmed-end
        # memory above is what protects against a real end sliding out.
        margin_px = 45.0
        for side, s_end, s_dir in (("left", extent[0], -1.0), ("right", extent[1], 1.0)):
            p_end = point + s_end * axis
            if p_end[2] <= 0.05:
                continue
            u_end = fx * p_end[0] / p_end[2] + cx
            v_end = fy * p_end[1] / p_end[2] + cy
            # The OBSERVED surface end lies at/beyond the image border: the
            # pipe visibly continues out of frame on that side. If instead
            # the surface end projects INSIDE the frame, the end is real and
            # fully measured (socket mouth case) -> full requirement.
            beyond = (
                u_end <= margin_px
                or u_end >= img_w - margin_px
                or v_end <= margin_px
                or v_end >= img_h - margin_px
            )
            # Image direction of "beyond the end": project a point slightly
            # further along the axis on this side.
            p_out = point + (s_end + s_dir * 0.05) * axis
            unmeasured = False
            if p_out[2] > 0.05:
                u_out = fx * p_out[0] / p_out[2] + cx
                v_out = fy * p_out[1] / p_out[2] + cy
                unmeasured = self._end_unmeasured_beyond(u_end, v_end, u_out - u_end, v_out - v_end)
            if side == "left":
                truncated_left = beyond
                unmeasured_left = unmeasured
            else:
                truncated_right = beyond
                unmeasured_right = unmeasured
        # The FULL requirement scales with stand-off: at 0.3 m the FOV shows
        # ~0.28 m of pipe in total, so demanding 0.15 m per side is
        # geometrically impossible and rejected the TRUE junction for ~130
        # frames (f852-999, L=0.13-0.14/0.15). Far away (socket at 0.7 m) the
        # scale caps at min_side and the pipe-end gate keeps its full bite.
        stand = float(np.linalg.norm(centre))
        needed_full = min(
            min_side,
            max(0.06, float(self.params["junction_axial_support_standoff_scale"]) * stand),
        )
        # Scale-invariant cap: each side must hold a SHARE of the visible
        # extent, never more than the extent can offer. At close range the
        # whole visible pipe is ~0.27 m, so two absolute 0.14 m sides are
        # geometrically impossible (measured f852-999: the candidate
        # oscillates between the collar edges and one side always fails by
        # millimetres). A pipe-end candidate holds a tiny share of a LONG
        # extent (socket: 0.10/0.81 = 12%) and stays rejected.
        extent_total = max(1e-6, float(extent[1]) - float(extent[0]))
        needed_full = min(
            needed_full,
            float(self.params["junction_axial_support_extent_share"]) * extent_total,
        )
        needed_left = min(needed_full, max(0.0, s_j - s_min_vis - 0.02)) if truncated_left else needed_full
        needed_right = min(needed_full, max(0.0, s_max_vis - s_j - 0.02)) if truncated_right else needed_full
        # Color-less masks systematically stop 3-4 cm short of the image
        # border (grazing-depth dropout that the warm mask fills in color
        # mode), so a border-truncated side under-measures its support by
        # that much. Grant the deficit to the MEASUREMENT, not to the
        # requirement: a real end keeps failing by far more than 4 cm, and
        # non-truncated sides (socket mouth inside the frame) get nothing.
        support_left_eff = support_left
        support_right_eff = support_right
        if self._coloroff_guard_frame_active:
            mask_deficit = float(self.params["junction_axial_support_coloroff_mask_deficit_m"])
            if truncated_left:
                support_left_eff += mask_deficit
            if truncated_right:
                support_right_eff += mask_deficit
        # An extent end whose beyond-region is depth-INVALID is a sensing
        # artifact (grazing angle / defocus at close range), not a physical
        # end: grant the doubt, but only when the side already shows a
        # substantial run (>= 0.6*min) — a genuinely short stub must not
        # pass. Measured on the cloth bag f852-999: the TRUE junction sat at
        # L=0.13-0.14/0.15 for ~130 frames while the pipe visibly ran to the
        # image border with no depth on it.
        if unmeasured_left and support_left >= 0.6 * min_side:
            needed_left = min(needed_left, max(0.0, support_left))
        if unmeasured_right and support_right >= 0.6 * min_side:
            needed_right = min(needed_right, max(0.0, support_right))
        # Confirm a pipe END on any side whose FULL requirement fails while
        # the observed surface end projects INSIDE the frame (>=10 px from the
        # border): the short support was actually measured, not truncated.
        # This must happen regardless of the border-doubt outcome — the doubt
        # requirement flickers around the true support when the bell nears
        # the border, and a single flicker used to seed a lock on the end.
        def end_inside(s_end: float) -> bool:
            inside_px = float(self.params["pipe_end_memory_confirm_inside_px"])
            if self._coloroff_guard_frame_active:
                # Kept at the same 10 px by default: the f46 color-less
                # memory this margin was once raised against turned out to be
                # the REAL socket end (its ban is what keeps the fine-tubo
                # candidates out at f60-92 while border doubt is active).
                inside_px = max(inside_px, float(self.params["pipe_end_memory_coloroff_inside_px"]))
            p_end = point + s_end * axis
            if p_end[2] <= 0.05:
                return False
            u_end = fx * p_end[0] / p_end[2] + cx
            v_end = fy * p_end[1] / p_end[2] + cy
            return (
                inside_px <= u_end <= img_w - inside_px
                and inside_px <= v_end <= img_h - inside_px
            )
        debug = os.environ.get("ACEA_END_MEM_DEBUG") is not None
        streak_need = int(self.params["pipe_end_memory_confirm_frames"])
        # A REAL pipe end fails DECISIVELY (support well under the requirement:
        # the bell+inner wall of the socket measure ~0.10-0.12 m) and has ample
        # pipe on the OTHER side. A short observed extent on BOTH sides is a
        # field-of-view / specular-shortening artifact around a tracked
        # junction (measured: supL~0.14 supR~0.13 while s_j~0 = the TRUE
        # junction) and must never seed a ban zone.
        decisive = min_side * float(self.params["pipe_end_memory_decisive_factor"])
        # Confirming a real end also demands a LONG measured run of pipe on
        # the other side (the f0-89 socket showed 0.71 m). At close range the
        # whole visible extent is ~0.3 m and marginal dips to 0.09-0.12 m are
        # specular/FOV artifacts around the true junction (f879/f920/f1109
        # phantoms) — with other-side at only 0.16-0.19 m they must not
        # confirm an end.
        other_need = min_side * float(self.params["pipe_end_memory_other_side_factor"])
        for side, failed, s_end, support_other, unmeasured in (
            ("left", support_left + 1e-6 < decisive, float(extent[0]), support_right, unmeasured_left),
            ("right", support_right + 1e-6 < decisive, float(extent[1]), support_left, unmeasured_right),
        ):
            if failed and support_other >= other_need and end_inside(s_end) and not unmeasured:
                prev = self._pipe_end_streak.get(side)
                if prev is not None and abs(s_end - prev[0]) <= 0.06:
                    self._pipe_end_streak[side] = [s_end, prev[1] + 1]
                else:
                    self._pipe_end_streak[side] = [s_end, 1]
                if debug:
                    print(
                        f"ENDMEM f{self.processed_frame_count - 1} streak {side} "
                        f"s={s_end:.3f} n={self._pipe_end_streak[side][1]} "
                        f"supL={support_left:.3f} supR={support_right:.3f} s_j={s_j:.3f}",
                        file=sys.stderr,
                    )
                if self._pipe_end_streak[side][1] >= streak_need:
                    if debug and side not in self._pipe_end_memory:
                        print(
                            f"ENDMEM f{self.processed_frame_count - 1} CREATE {side} s={s_end:.3f}",
                            file=sys.stderr,
                        )
                    birth = (
                        self._pipe_end_memory[side][2]
                        if side in self._pipe_end_memory
                        else float(self.processed_frame_count)
                    )
                    self._pipe_end_memory[side] = [s_end, 0, birth]
            else:
                self._pipe_end_streak.pop(side, None)
        if self._junction_near_remembered_end(s_j):
            if debug:
                print(
                    f"ENDMEM f{self.processed_frame_count - 1} BAN s_j={s_j:.3f} "
                    f"mem={ {k: round(v[0], 3) for k, v in self._pipe_end_memory.items()} }",
                    file=sys.stderr,
                )
            return False, "near_remembered_pipe_end"
        if support_left_eff + 1e-6 < needed_left or support_right_eff + 1e-6 < needed_right:
            if debug:
                uv = []
                for s_end in (extent[0], extent[1]):
                    p_end = point + float(s_end) * axis
                    if p_end[2] > 0.05:
                        uv.append((round(fx * p_end[0] / p_end[2] + cx), round(fy * p_end[1] / p_end[2] + cy)))
                    else:
                        uv.append(None)
                print(
                    f"ENDMEM f{self.processed_frame_count - 1} REJSUP "
                    f"L={support_left_eff:.3f}/{needed_left:.3f} R={support_right_eff:.3f}/{needed_right:.3f} "
                    f"uvL={uv[0]} uvR={uv[1]} s_j={s_j:.3f} ext=({extent[0]:.3f},{extent[1]:.3f}) "
                    f"maskcols={self._pipe_mask_col_extent}",
                    file=sys.stderr,
                )
            return False, (
                f"pipe_end_axial_support L={support_left_eff:.2f}/{needed_left:.2f} "
                f"R={support_right_eff:.2f}/{needed_right:.2f}"
            )
        return True, ""

    @staticmethod
    def _pipe_axis_model_components_from_model(model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        return OnlinePipeJunctionDetector._pipe_axis_components_impl(model)

    @staticmethod
    def _pipe_axis_components_impl(model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            axis = _normalize(np.asarray(model["axis"], dtype=np.float64).reshape(3))
            point = np.asarray(model["axis_point"], dtype=np.float64).reshape(3)
            if not np.isfinite(axis).all() or not np.isfinite(point).all():
                return None
            return axis, point
        except Exception:
            return None

    def _pipe_tracker_state(self) -> str:
        """Return the explicit FSM state maintained by
        _advance_pipe_tracker_state (single source of truth)."""
        return self._pipe_state

    def _filter_junction_center(self, center_meas: np.ndarray) -> np.ndarray:
        """Innovation-gated smoothing of the published 3D junction centre,
        same philosophy as the image-space x filter: small innovations are
        blended, big ones snap (a re-detection after a gap is real motion),
        and a persistent one-sided lag raises the gain."""
        if not bool(self.params["enable_junction_center_smoothing"]):
            return center_meas
        prev = self._junction_center_state
        if prev is None:
            self._junction_center_state = center_meas
            self._junction_center_lag = np.zeros(3, dtype=np.float64)
            return center_meas
        innovation = center_meas - prev
        dist = float(np.linalg.norm(innovation))
        snap_m = float(self.params["junction_center_snap_m"])
        if dist >= snap_m:
            self._junction_center_state = center_meas
            self._junction_center_lag = np.zeros(3, dtype=np.float64)
            return center_meas
        soft_m = float(self.params["junction_center_soft_m"])
        gain = (
            float(self.params["junction_center_gain"])
            if dist <= soft_m
            else float(self.params["junction_center_gain_far"])
        )
        self._junction_center_lag = 0.7 * self._junction_center_lag + 0.3 * innovation
        if float(np.linalg.norm(self._junction_center_lag)) > soft_m:
            gain = max(gain, 0.7)
        state = prev + gain * innovation
        self._junction_center_state = state
        return state

    def _localize_confirmed_seam(
        self,
        depth: np.ndarray,
        k: np.ndarray,
        tracker: TrackerResult,
        seam: SeamResult,
    ) -> LocalizationResult:
        pipe_model = self.pipe_component_selected_model or self.pipe_lock_model
        components = self._pipe_axis_model_components(tracker, pipe_model)

        def ray_result(support: int, method: str) -> LocalizationResult:
            if components is None:
                return LocalizationResult(None, None, support, "no_axis_model")
            center = self._junction_axis_center_from_image_line(seam, k, components)
            if center is None:
                return LocalizationResult(None, None, support, "ray_failed")
            return LocalizationResult(None, self._filter_junction_center(center), support, method)

        x_center = int(seam.candidate_x_rotated_px)
        y0, y1 = int(seam.crop_xyxy[1]), int(seam.crop_xyxy[3])
        half_width = int(self.params["surface_band_half_width_px"])
        x0 = max(0, x_center - half_width)
        x1 = min(depth.shape[1] - 1, x_center + half_width)
        y0 = max(0, y0)
        y1 = min(depth.shape[0] - 1, y1)

        band_mask = np.zeros(seam.rotated_mask.shape, dtype=bool)
        band_mask[y0:y1 + 1, x0:x1 + 1] = seam.rotated_mask[y0:y1 + 1, x0:x1 + 1]
        ys_rot, xs_rot = np.nonzero(band_mask)
        if xs_rot.size == 0:
            return ray_result(0, "axis_ray_no_band")

        rotated_uv = np.stack([xs_rot.astype(np.float64), ys_rot.astype(np.float64)], axis=1)
        original_uv = _inverse_rotate_uv(rotated_uv, (depth.shape[1], depth.shape[0]), seam.rotation_deg)
        xs = np.rint(original_uv[:, 0]).astype(np.int64)
        ys = np.rint(original_uv[:, 1]).astype(np.int64)
        in_bounds = (xs >= 0) & (xs < depth.shape[1]) & (ys >= 0) & (ys < depth.shape[0])
        xs = xs[in_bounds]
        ys = ys[in_bounds]
        valid = tracker.pipe_mask[ys, xs] & np.isfinite(depth[ys, xs]) & (depth[ys, xs] > 0.0)
        xs = xs[valid]
        ys = ys[valid]
        if xs.size < int(self.params["min_surface_points"]):
            return ray_result(int(xs.size), "axis_ray_low_support")

        unique_uv = np.unique(np.stack([xs, ys], axis=1), axis=0)
        unique_uv = self._coherent_depth_support_uv(unique_uv, depth)
        if unique_uv.shape[0] < int(self.params["min_surface_points"]):
            return ray_result(int(unique_uv.shape[0]), "axis_ray_low_coherent_support")
        points_camera = _backproject(depth, unique_uv[:, 0], unique_uv[:, 1], k)
        # Use a real point close to the median support instead of publishing a
        # component-wise median that may combine y from one depth component and z
        # from another. That was fragile when the seam column crossed both pipe
        # surface and background/pipe-end depth.
        median_point = np.median(points_camera, axis=0)
        surface_center = points_camera[int(np.argmin(np.linalg.norm(points_camera - median_point, axis=1)))]

        # The published centre lives ON the fitted pipe axis: local depth near
        # the collar only estimates the LONGITUDINAL coordinate (projection of
        # the surface point onto the axis); if its support is thin, the image
        # ray gives that coordinate instead. Either way the lateral position
        # comes from the recentered cylinder model, which is far more stable
        # than lifting a local surface point by one radius.
        support = int(unique_uv.shape[0])
        if components is not None:
            axis, axis_point = components
            method = "axis_surface"
            center = axis_point + float(np.dot(surface_center - axis_point, axis)) * axis
            if support < int(self.params["junction_axis_center_min_support_px"]):
                ray_center = self._junction_axis_center_from_image_line(seam, k, components)
                if ray_center is not None:
                    center = ray_center
                    method = f"axis_ray_support={support}"
            if np.isfinite(center).all() and center[2] > 0.0:
                return LocalizationResult(surface_center, self._filter_junction_center(center), support, method)

        # No usable axis model: legacy surface-lift fallback.
        view_direction = _normalize(surface_center)
        axis = _normalize(tracker.pipe_axis_xyz)
        radial_direction = view_direction - float(np.dot(view_direction, axis)) * axis
        radial_direction = _normalize(radial_direction)
        pipe_center = surface_center + float(self.params["pipe_radius_m"]) * radial_direction
        return LocalizationResult(surface_center, self._filter_junction_center(pipe_center), support, "surface_lift_fallback")

    def _coherent_depth_support_uv(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Keep the most depth-coherent vertical component in the seam support.

        The raw support is a thin column through the pipe mask. In Gazebo, and
        near a cut pipe, that column may contain disconnected depth components
        (real pipe surface plus background / lower surface / edge artifacts).
        Taking all pixels together can create a 3D point that is numerically
        valid but geometrically wrong. Prefer the connected row segment with the
        smallest robust depth spread; fall back to the full support if the split
        is not informative.
        """
        if uv.shape[0] == 0:
            return uv
        xs = uv[:, 0].astype(np.int64)
        ys = uv[:, 1].astype(np.int64)
        z = depth[ys, xs].astype(np.float64)
        finite = np.isfinite(z) & (z > 0.0)
        uv = uv[finite]
        ys = ys[finite]
        z = z[finite]
        if uv.shape[0] == 0:
            return uv

        rows = np.unique(ys)
        if rows.size <= 1:
            return uv

        segments: list[tuple[int, int]] = []
        start = int(rows[0])
        prev = int(rows[0])
        for row in rows[1:]:
            row_i = int(row)
            if row_i - prev > 2:
                segments.append((start, prev))
                start = row_i
            prev = row_i
        segments.append((start, prev))

        min_points = max(12, min(int(self.params["min_surface_points"]), 40))
        best_mask = None
        best_score = float("inf")
        for start, end in segments:
            mask = (ys >= start) & (ys <= end)
            count = int(mask.sum())
            if count < min_points:
                continue
            z_seg = z[mask]
            q10, q50, q90 = np.percentile(z_seg, [10.0, 50.0, 90.0])
            robust_spread = float(q90 - q10)
            row_span = max(1, end - start + 1)
            # Prefer compact, coherent depth components. The light count bonus
            # breaks ties in favour of better-supported components without
            # letting large background components dominate.
            score = robust_spread + 0.002 / math.sqrt(float(count)) + 0.0002 * float(row_span)
            if np.isfinite(q50) and score < best_score:
                best_score = score
                best_mask = mask

        if best_mask is None:
            return uv
        return uv[best_mask]

    def _pipe_side_coverage(self, pipe_mask: np.ndarray) -> tuple[float, float]:
        """Fraction of pipe-surface pixels in the LEFT vs RIGHT third of the image.

        Robust pipe-END signal from the depth-derived pipe mask: a real junction
        keeps the pipe body on BOTH sides; a pipe end/start leaves one side at ~0.
        Scan-control uses this to reverse the scan (event-driven, no odometry)."""
        if pipe_mask is None or pipe_mask.ndim != 2 or pipe_mask.shape[1] < 3:
            return 0.0, 0.0
        w = pipe_mask.shape[1]
        third = max(1, w // 3)
        left = float(pipe_mask[:, :third].mean())
        right = float(pipe_mask[:, w - third:].mean())
        return left, right

    def _status_dict(
        self,
        tracker: TrackerResult,
        seam: SeamResult,
        localization: LocalizationResult | None,
        state_info: dict[str, Any],
    ) -> dict[str, Any]:
        pipe_cov_left, pipe_cov_right = self._pipe_side_coverage(tracker.pipe_mask)
        return {
            "pipe_coverage_left_frac": _round(pipe_cov_left),
            "pipe_coverage_right_frac": _round(pipe_cov_right),
            "state": self.state,
            "processed_frame_count": self.processed_frame_count,
            "confirmed_frame_count": self.confirmed_frame_count,
            "candidate_streak": self.candidate_streak,
            "junction_lock_active": bool(state_info.get("junction_lock_active", False)),
            "junction_lock_used": bool(state_info.get("junction_lock_used", False)),
            "junction_lock_reason": state_info.get("junction_lock_reason"),
            "junction_lock_x_strip_px": _round(state_info.get("junction_lock_x_strip_px")),
            "junction_last_valid_x_strip_px": _round(state_info.get("junction_last_valid_x_strip_px")),
            "junction_last_valid_frame": state_info.get("junction_last_valid_frame"),
            "fresh_reacquire_ok": bool(state_info.get("fresh_reacquire_ok", True)),
            "fresh_reacquire_reason": state_info.get("fresh_reacquire_reason"),
            "junction_lock_velocity_px_per_frame": _round(state_info.get("junction_lock_velocity_px_per_frame")),
            "junction_lock_streak": int(state_info.get("junction_lock_streak", 0)),
            "junction_lock_missed_frames": int(state_info.get("junction_lock_missed_frames", 0)),
            "junction_lock_confidence": _round(state_info.get("junction_lock_confidence")),
            "junction_lock_source": state_info.get("junction_lock_source"),
            "confidence": _round(seam.confidence),
            "junction_acceptance_mode": seam.junction_acceptance_mode,
            "variant_a_orientation_deg": _round(seam.variant_a_orientation_deg),
            "variant_a_classical_fallback_used": bool(seam.variant_a_classical_fallback_used),
            "visual_frontend": seam.visual_frontend,
            "visual_frontend_accepted": bool(seam.visual_frontend_accepted),
            "candidate_x_strip_px": int(seam.candidate_x_strip_px),
            "candidate_x_raw_strip_px": int(seam.candidate_x_raw_strip_px),
            "junction_smooth_lag_ema_px": _round(float(self._junction_lag_ema), 2),
            "junction_center_method": localization.center_method if localization is not None else None,
            "pipe_tracker_state": self._pipe_tracker_state(),
            "warm_scene_ok": bool(self._warm_scene_ok),
            "coloroff_guard_active": bool(self._coloroff_guard_frame_active),
            "coloroff_pipe_visible": bool(getattr(self, "_coloroff_pipe_visible", True)),
            "coloroff_cyl_ok": bool(self._coloroff_cyl_ok),
            "coloroff_cyl_reason": self._coloroff_cyl_reason,
            "coloroff_acquire_streak": int(self._coloroff_acquire_streak),
            "coloroff_fail_streak": int(self._coloroff_fail_streak),
            "pipe_cyl_slope": _round(self._coloroff_cyl_features["slope"], 3) if self._coloroff_cyl_features else None,
            "pipe_cyl_align_frac": _round(self._coloroff_cyl_features["align_frac"], 3) if self._coloroff_cyl_features else None,
            "pipe_cyl_cov_inlier_deg": _round(self._coloroff_cyl_features["cov_inlier_deg"], 1) if self._coloroff_cyl_features else None,
            "pipe_cyl_cov_deg": _round(self._coloroff_cyl_features["cov_deg"], 1) if self._coloroff_cyl_features else None,
            "pipe_cyl_samples": int(self._coloroff_cyl_features["samples"]) if self._coloroff_cyl_features else None,
            "pipe_lock_update_accepted_count": int(self._pipe_lock_update_accepted),
            "pipe_lock_update_held_count": int(self._pipe_lock_update_held),
            "pipe_mask_warm_fraction": _round(self.pipe_mask_warm_fraction, 3) if self.pipe_mask_warm_fraction is not None else None,
            "pipe_mask_normal_fraction": _round(self.pipe_mask_normal_fraction, 3) if getattr(self, "pipe_mask_normal_fraction", None) is not None else None,
            "pipe_low_warm_streak": int(self._low_warm_streak),
            # Junction line in ORIGINAL image pixels (the two endpoints the overlay
            # draws): [[u0,v0],[u1,v1]]. Lets an external projected ground truth
            # compare in image space regardless of pipe roll. None if unavailable.
            "candidate_line_image_uv": (
                [[round(float(p[0]), 1), round(float(p[1]), 1)]
                 for p in self._candidate_line_original_uv(seam)]
                if seam.candidate_x_rotated_px is not None else None
            ),
            "candidate_contrast": _round(seam.candidate_contrast),
            "candidate_z_score": _round(seam.candidate_z_score),
            "candidate_step_abs": _round(seam.candidate_step_abs),
            "candidate_step_z": _round(seam.candidate_step_z),
            "candidate_evidence_z": _round(seam.candidate_evidence_z),
            "step_fallback_used": bool(seam.step_fallback_used),
            "classical_candidate_x_strip_px": int(seam.classical_candidate_x_strip_px),
            "classical_candidate_contrast": _round(seam.classical_candidate_contrast),
            "classical_candidate_z_score": _round(seam.classical_candidate_z_score),
            "rgb_dark_score": _round(seam.rgb_dark_score),
            "rgb_local_contrast_score": _round(seam.rgb_local_contrast_score),
            "rgb_dark_threshold_used": _round(seam.rgb_dark_threshold_used),
            # A raw seam acceptance that the state machine flagged as a large
            # candidate jump (hop to another line) must NOT be surfaced as accepted
            # nor propagated downstream: the jump guard already refused to lock it.
            "detector_accepted": bool(state_info.get("eligible", False)),
            "rgb_dark_accepted": bool(seam.rgb_dark_accepted),
            "rgb_temporal_accepted": bool(seam.rgb_temporal_accepted),
            "rgb_temporal_score": _round(seam.rgb_temporal_score),
            "rgb_vertical_edge_score": _round(seam.rgb_vertical_edge_score),
            "rgb_luminance_edge_score": _round(seam.rgb_luminance_edge_score),
            "rgb_chromatic_edge_score": _round(seam.rgb_chromatic_edge_score),
            "rgb_edge_chromaticity_ratio": _round(seam.rgb_edge_chromaticity_ratio),
            "rgb_shadow_like_score": _round(seam.rgb_shadow_like_score),
            "rgb_shadow_like_rejected": bool(seam.rgb_shadow_like_rejected),
            "rgb_surface_continuity_score": _round(seam.rgb_surface_continuity_score),
            "rgb_surface_continuity_rejected": bool(seam.rgb_surface_continuity_rejected),
            "rgb_low_contrast_rejected": bool(seam.rgb_low_contrast_rejected),
            "rgb_temporal_candidate_reject_reason": seam.rgb_temporal_candidate_reject_reason,
            "rgb_line_support_fraction": _round(seam.rgb_line_support_fraction),
            "rgb_line_width_px": int(seam.rgb_line_width_px),
            "rgb_track_id": int(seam.rgb_track_id),
            "rgb_track_streak": int(seam.rgb_track_streak),
            "rgb_track_missed_frames": int(seam.rgb_track_missed_frames),
            "rgb_candidate_velocity_px_per_frame": _round(seam.rgb_candidate_velocity_px_per_frame),
            "klt_status": seam.klt_status,
            "klt_points": int(seam.klt_points),
            "klt_dx_px": _round(seam.klt_dx_px),
            "klt_predicted_x_strip_px": _round(seam.klt_predicted_x_strip_px),
            "pipe_end_rejected": bool(seam.pipe_end_rejected),
            "pipe_support_left_cols": int(seam.pipe_support_left_cols),
            "pipe_support_right_cols": int(seam.pipe_support_right_cols),
            "pipe_support_left_coverage": _round(seam.pipe_support_left_coverage),
            "pipe_support_right_coverage": _round(seam.pipe_support_right_coverage),
            "depth_gap_score": _round(seam.depth_gap_score),
            "depth_gap_accepted": bool(seam.depth_gap_accepted),
            "depth_gap_raw_accepted": bool(seam.depth_gap_raw_accepted),
            "depth_gap_score_plausible": bool(seam.depth_gap_score_plausible),
            "depth_gap_depth_jump_m": _round(seam.depth_gap_depth_jump_m),
            "depth_gap_coverage_drop": _round(seam.depth_gap_coverage_drop),
            "negative_gate_reason": seam.negative_gate_reason,
            "local_candidate_accepted": bool(seam.local_candidate_accepted),
            "temporal_scan_change_enabled": bool(self.params["enable_temporal_scan_change"]),
            "temporal_change_gate_enabled": bool(seam.temporal_change_gate_enabled),
            "temporal_change_score": _round(seam.temporal_change_score),
            "temporal_change_dark_delta": _round(seam.temporal_change_dark_delta),
            "temporal_change_z_score": _round(seam.temporal_change_z_score),
            "temporal_change_accepted": bool(seam.temporal_change_accepted),
            "temporal_reference_ready": bool(seam.temporal_reference_ready),
            "temporal_reference_frame_count": int(seam.temporal_reference_frame_count),
            "temporal_change_reason": seam.temporal_change_reason,
            "eligible": bool(state_info["eligible"]),
            "candidate_not_border": bool(state_info["candidate_not_border"]),
            "geometry_consistent": bool(state_info["geometry_consistent"]),
            "reason": state_info["reason"],
            "stand_off_m": _round(tracker.stand_off_m),
            "lateral_offset_m": _round(tracker.lateral_offset_m),
            "vertical_offset_m": _round(tracker.vertical_offset_m),
            "yaw_error_deg": _round(tracker.yaw_error_deg),
            "image_pipe_axis_angle_deg": _round(tracker.image_axis_angle_deg),
            "pipe_pixels": int(tracker.pipe_pixels),
            "pipe_fraction_of_image": _round(tracker.pipe_fraction),
            "pipe_component_selection_method": self.pipe_component_selection_info.get("method"),
            "pipe_component_count": self.pipe_component_selection_info.get("component_count"),
            "pipe_component_candidate_count": self.pipe_component_selection_info.get("candidate_count"),
            "pipe_component_rejected_by_shape": self.pipe_component_selection_info.get("rejected_by_shape"),
            "pipe_component_cylinder_evaluated": self.pipe_component_selection_info.get("cylinder_evaluated"),
            "pipe_component_cylinder_valid": self.pipe_component_selection_info.get("cylinder_valid"),
            "pipe_component_selected_label": self.pipe_component_selection_info.get("selected_label"),
            "pipe_component_fallback_label": self.pipe_component_selection_info.get("fallback_label"),
            "pipe_component_band_valid": self.pipe_component_selection_info.get("band_valid"),
            "pipe_component_band_score": self.pipe_component_selection_info.get("band_score"),
            "pipe_component_band_width_fraction": self.pipe_component_selection_info.get("band_width_fraction"),
            "pipe_component_band_column_coverage": self.pipe_component_selection_info.get("band_column_coverage"),
            "pipe_component_band_pixels": self.pipe_component_selection_info.get("band_pixels"),
            "pipe_component_band_method": self.pipe_component_selection_info.get("band_method"),
            "pipe_mask_reproject_applied": self.pipe_mask_reproject_info.get("applied"),
            "pipe_mask_reproject_source": self.pipe_mask_reproject_info.get("source"),
            "pipe_mask_reproject_reason": self.pipe_mask_reproject_info.get("reason"),
            "pipe_mask_reproject_radius_m": self.pipe_mask_reproject_info.get("radius_m"),
            "pipe_mask_reproject_mask_px": self.pipe_mask_reproject_info.get("mask_px"),
            "pipe_lock_active": bool(self.pipe_lock_model is not None),
            "pipe_image_lock_active": bool(self.pipe_image_lock_model is not None),
            "pipe_lock_missed_frames": int(self.pipe_lock_missed_frames),
            "pipe_lock_source": self.pipe_lock_source,
            "pipe_image_lock_source": self.pipe_image_lock_source,
            "pipe_lock_selection_score": self.pipe_component_selection_info.get("score"),
            "pipe_lock_axis_delta_deg": self.pipe_component_selection_info.get("lock_axis_delta_deg"),
            "pipe_lock_radius_delta_m": self.pipe_component_selection_info.get("lock_radius_delta_m"),
            "pipe_lock_stand_delta_m": self.pipe_component_selection_info.get("lock_stand_delta_m"),
            "pipe_lock_axis_point_delta_m": self.pipe_component_selection_info.get("lock_axis_point_delta_m"),
            "pipe_image_lock_axis_delta_deg": self.pipe_component_selection_info.get("image_lock_axis_delta_deg"),
            "pipe_image_lock_center_delta_px": self.pipe_component_selection_info.get("image_lock_center_delta_px"),
            "pipe_image_lock_depth_delta_m": self.pipe_component_selection_info.get("image_lock_depth_delta_m"),
            "pipe_axis_camera_xyz": _round_list(tracker.pipe_axis_xyz),
            "pipe_centroid_camera_xyz_m": _round_list(tracker.centroid_xyz_m),
            "pipe_pose_fit_method": tracker.pipe_pose_fit_method,
            "pipe_pose_inlier_count": int(tracker.pipe_pose_inlier_count),
            "pipe_pose_inlier_fraction": _round(tracker.pipe_pose_inlier_fraction),
            "pipe_pose_radius_m": _round(tracker.pipe_pose_radius_m),
            "pipe_pose_residual_m": _round(tracker.pipe_pose_residual_m),
            "coarse_seam_visible_surface_camera_xyz_m": None
            if localization is None
            else _round_list(localization.visible_surface_center_xyz_m),
            "coarse_seam_center_camera_xyz_m": None
            if localization is None
            else _round_list(localization.pipe_center_estimate_xyz_m),
            "coarse_seam_support_pixel_count": None if localization is None else localization.support_pixel_count,
        }

    def _draw_overlay(
        self,
        rgb: np.ndarray,
        tracker: TrackerResult,
        seam: SeamResult,
        localization: LocalizationResult | None,
        state_info: dict[str, Any],
    ) -> np.ndarray:
        base = rgb
        # v8: before the first cylinder-consistent model in a color-less
        # ACQUIRE episode, the mask/axis are the default or cloth fallback.
        # Hide them so the overlay does not flicker between the cloth and the
        # pipe until the pipe is genuinely acquired.
        pipe_visible = getattr(self, "_coloroff_pipe_visible", True)
        mask = tracker.pipe_mask
        if pipe_visible and mask is not None and mask.any():
            # Same visual as the old RGBA alpha_composite (alpha 70/255 green
            # tint), an order of magnitude cheaper than two full-frame
            # composites per frame.
            if cv2 is not None:
                tint_img = np.empty_like(rgb)
                tint_img[:, :] = (20, 190, 90)
                blended = cv2.addWeighted(rgb, 1.0 - 70.0 / 255.0, tint_img, 70.0 / 255.0, 0.0)
                base = np.where(mask[:, :, None], blended, rgb)
            else:
                base = rgb.copy()
                alpha = 70.0 / 255.0
                tint = np.array([20.0, 190.0, 90.0], dtype=np.float32)
                base[mask] = (base[mask].astype(np.float32) * (1.0 - alpha) + tint * alpha).astype(np.uint8)
        image = PilImage.fromarray(np.ascontiguousarray(base), mode="RGB")
        draw = ImageDraw.Draw(image, "RGBA")

        if pipe_visible and tracker.image_line_segment_uv is not None:
            p0, p1 = tracker.image_line_segment_uv
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(255, 40, 40, 255), width=3)

        if pipe_visible:
            u, v = tracker.image_centroid_uv
            draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(40, 120, 255, 255), width=3)

        line_uv = self._candidate_line_original_uv(seam)
        final_accepted = bool(state_info.get("eligible", False))
        if final_accepted:
            color = (0, 220, 255, 255)
            label = "JUNCTION"
            width = 4
        elif seam.local_candidate_accepted and bool(self.params["draw_rejected_junction_candidates"]):
            color = (255, 210, 0, 255)
            label = "VERIFY"
            width = 3
        else:
            if not bool(self.params["draw_rejected_junction_candidates"]):
                return np.asarray(image.convert("RGB"))
            color = (255, 180, 0, 255)
            label = "SCAN SEARCH"
            width = 3
        draw.line((line_uv[0][0], line_uv[0][1], line_uv[1][0], line_uv[1][1]), fill=color, width=width)
        label_x = max(4, min(image.width - 120, int(line_uv[0][0]) + 6))
        label_y = max(4, min(image.height - 22, int(line_uv[0][1]) + 6))
        draw.rectangle((label_x - 3, label_y - 2, label_x + 112, label_y + 15), fill=(0, 0, 0, 120))
        draw.text((label_x, label_y), label, fill=color)

        if localization is not None and localization.pipe_center_estimate_xyz_m is not None:
            k = np.array(
                [
                    [float(self.params["last_fx"]), 0.0, float(self.params["last_cx"])],
                    [0.0, float(self.params["last_fy"]), float(self.params["last_cy"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            uv = _project(localization.pipe_center_estimate_xyz_m.reshape(1, 3), k)[0]
            draw.ellipse((uv[0] - 8, uv[1] - 8, uv[0] + 8, uv[1] + 8), outline=(255, 210, 0, 255), width=3)

        return np.asarray(image.convert("RGB"))

    def _candidate_line_original_uv(self, seam: SeamResult) -> tuple[tuple[float, float], tuple[float, float]]:
        y0, y1 = float(seam.crop_xyxy[1]), float(seam.crop_xyxy[3])
        x = float(seam.candidate_x_rotated_px)
        points_rot = np.array([[x, y0], [x, y1]], dtype=np.float64)
        original = _inverse_rotate_uv(points_rot, (seam.rotated_mask.shape[1], seam.rotated_mask.shape[0]), seam.rotation_deg)
        return (float(original[0, 0]), float(original[0, 1])), (float(original[1, 0]), float(original[1, 1]))


class AceaPipeJunctionNode(Node):
    def __init__(self) -> None:
        super().__init__("acea_pipe_junction_detector")
        self.params = self._declare_params()
        self.instance_id = instance_id()

        # Refuse to start a second detector on the same host: two of them publish
        # CONFLICTING detections to the same topics (the alternating
        # processed_frame_count symptom). Override with allow_duplicate:=true.
        self._single_instance: SingleInstanceLock | None = None
        if rclpy is not None:
            self._single_instance = SingleInstanceLock(_DETECTOR_LOCK_NAME)
            if not self._single_instance.acquired and not bool(self.params.get("allow_duplicate", False)):
                raise DuplicateInstanceError(
                    f"another '{_DETECTOR_LOCK_NAME}' is already running on this host "
                    f"(holder {self._single_instance.holder or 'unknown'}). Two detectors "
                    "publish conflicting results to the same topics — you get alternating "
                    "processed_frame_count and contradictory state (one SCAN, one "
                    "STOP_AND_LOCALIZE), and config edits look ignored because the old "
                    f"instance is still alive. Stop it first:  pkill -f {_DETECTOR_LOCK_NAME} "
                    "  (or close its terminal/launch), then start exactly one. To run two "
                    "on purpose, pass -p allow_duplicate:=true."
                )

        self.detector = OnlinePipeJunctionDetector(self.params)
        self.rgb_queue: deque[tuple[float, Image]] = deque(maxlen=int(self.params["queue_size"]))
        self.depth_queue: deque[tuple[float, Image]] = deque(maxlen=int(self.params["queue_size"]))
        self.info_queue: deque[tuple[float, CameraInfo]] = deque(maxlen=int(self.params["queue_size"]))
        self.received_count = 0
        self.last_processed_rgb_time: float | None = None
        self.last_status_publish_time = 0.0

        # Per-stream arrival timestamps (node clock) for sync diagnostics + the
        # auto receive-time fallback. Effective sync clock can flip at runtime.
        self._rgb_recv: deque[float] = deque(maxlen=30)
        self._depth_recv: deque[float] = deque(maxlen=30)
        self._info_recv: deque[float] = deque(maxlen=30)
        self._effective_use_receive_time = bool(self.params.get("use_receive_time_for_sync", False))
        self._receive_time_fallback_logged = False
        self._last_subscription_reset_time = 0.0

        camera_qos_reliability = str(self.params.get("camera_qos_reliability", "best_effort")).strip().lower()
        if camera_qos_reliability in ("reliable", "rel"):
            reliability = ReliabilityPolicy.RELIABLE
        else:
            reliability = ReliabilityPolicy.BEST_EFFORT
        self._camera_qos = QoSProfile(
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.params["queue_size"]),
        )
        self._rgb_sub = None
        self._depth_sub = None
        self._info_sub = None
        self._create_camera_subscriptions()

        self.detection_pub = self.create_publisher(String, str(self.params["detection_topic"]), 10)
        self.detected_pub = self.create_publisher(Bool, str(self.params["detected_topic"]), 10)
        self.status_pub = self.create_publisher(String, str(self.params["status_topic"]), 10)
        self.rgb_overlay_pub = self.create_publisher(Image, str(self.params["rgb_overlay_topic"]), 10)
        self.depth_overlay_pub = self.create_publisher(Image, str(self.params["depth_overlay_topic"]), 10)
        self.weld_seam_pose_pub = None
        self.weld_gap_plane_pub = None
        self.weld_marker_pub = None
        if bool(self.params["publish_weld_gap_geometry"]):
            self.weld_seam_pose_pub = self.create_publisher(
                PoseStamped,
                str(self.params["weld_seam_pose_topic"]),
                10,
            )
            self.weld_gap_plane_pub = self.create_publisher(
                String,
                str(self.params["weld_gap_plane_topic"]),
                10,
            )
            self.weld_marker_pub = self.create_publisher(
                MarkerArray,
                str(self.params["weld_marker_topic"]),
                10,
            )
        if bool(self.params["publish_waiting_status"]):
            self.create_timer(float(self.params["waiting_status_period_s"]), self._publish_waiting_status)
        # Periodic monitor for duplicates the file lock cannot see (another
        # container / host on the same DDS graph publishing the same topics).
        if rclpy is not None:
            self.create_timer(5.0, self._check_duplicate_graph)
        self.get_logger().info(
            f"ACEA pipe-junction detector [{self.instance_id}] listening to "
            f"{self.params['rgb_topic']}, {self.params['depth_topic']}, {self.params['camera_info_topic']}"
        )

    def _declare_params(self) -> dict[str, Any]:
        declarations = {
            "rgb_topic": "/camera/rgb",
            "depth_topic": "/camera/depth",
            "camera_info_topic": "/camera/camera_info",
            "detection_topic": "/acea/pipe_junction/detection",
            "detected_topic": "/acea/pipe_junction/detected",
            "status_topic": "/acea/pipe_junction/status",
            "rgb_overlay_topic": "/acea/pipe_junction/debug/rgb_overlay",
            "depth_overlay_topic": "/acea/pipe_junction/debug/depth_overlay",
            "publish_weld_gap_geometry": True,
            "weld_seam_pose_topic": "/acea/weld_seam/pose",
            "weld_gap_plane_topic": "/acea/weld_seam/gap_plane",
            "weld_gap_require_detector_accepted": True,
            "weld_marker_topic": "/acea/weld_seam/markers",
            "weld_marker_cylinder_length_m": 0.6,
            "weld_marker_plane_scale": 1.3,
            # Gazebo image/depth bridges can use inconsistent header stamps on
            # different machines. Default to receive-time sync so the detector
            # is robust even if detector.yaml was not installed/loaded.
            "sync_slop_s": 1.0,
            "use_receive_time_for_sync": True,
            # If RGB and depth both arrive but never sync on header stamps (zero
            # stamps, or mixed sim/wall clock), automatically fall back to
            # receive-time sync once and warn. Set use_receive_time_for_sync:=true
            # to make it the default and silence the warning.
            "auto_receive_time_fallback": True,
            # Refuse to start if another detector is already running on this host.
            "allow_duplicate": False,
            "allow_stale_camera_info": True,
            "queue_size": 10,
            "camera_qos_reliability": "best_effort",
            "publish_waiting_status": True,
            "waiting_status_period_s": 1.0,
            "draw_rejected_junction_candidates": False,
            "stream_stale_s": 2.0,
            "stale_subscription_reset_s": 5.0,
            "process_every_n": 1,
            "allow_nominal_intrinsics_fallback": True,
            "nominal_fx_px": 733.0,
            "nominal_fy_px": 733.0,
            "nominal_cx_px": -1.0,
            "nominal_cy_px": -1.0,
            "min_depth_m": 0.05,
            "max_depth_m": 20.0,
            "sample_stride": 4,
            "max_pca_points": 60000,
            "min_pipe_pixels": 1000,
            "use_cylinder_component_selection": True,
            "cylinder_component_max_components": 8,
            "cylinder_component_consensus_iterations": 2,
            "cylinder_component_radius_tolerance_m": 0.035,
            "cylinder_component_min_inliers": 180,
            "cylinder_component_min_inlier_fraction": 0.22,
            "cylinder_component_radius_abs_margin_m": 0.03,
            "cylinder_component_radius_rel_margin": 0.35,
            "cylinder_component_max_residual_m": 0.05,
            "pipe_tracking_component_radius_tolerance_m": 0.06,
            "pipe_tracking_component_min_inliers": 100,
            "pipe_tracking_component_min_inlier_fraction": 0.12,
            "pipe_tracking_component_radius_abs_margin_m": 0.06,
            "pipe_tracking_component_radius_rel_margin": 0.65,
            "pipe_tracking_component_max_residual_m": 0.08,
            "pipe_component_require_valid_cylinder": True,
            "pipe_component_shape_prior_enabled": False,
            "pipe_component_min_width_fraction": 0.28,
            "pipe_component_bottom_margin_fraction": 0.05,
            "pipe_component_bottom_allow_width_fraction": 0.55,
            "pipe_component_use_band_filter": True,
            "pipe_component_allow_band_fallback_without_valid_cylinder": True,
            "pipe_component_band_min_pixels": 600,
            "pipe_component_band_min_width_fraction": 0.22,
            "pipe_component_band_min_column_coverage": 0.28,
            "pipe_component_band_min_col_pixels": 3,
            "pipe_component_band_half_width_min_px": 8.0,
            "pipe_component_band_half_width_max_px": 90.0,
            "pipe_component_band_expected_diameter_scale": 0.90,
            "pipe_component_band_min_axis_dx": 0.18,
            "pipe_component_band_max_hypothesis_points": 12000,
            "pipe_component_band_run_gap_px": 3,
            "pipe_component_band_upper_run_weight": 1.8,
            "pipe_component_band_fallback_min_score": 700.0,
            "enable_pipe_mask_cylinder_reproject": True,
            "pipe_mask_reproject_use_lock_model": True,
            "pipe_mask_reproject_radius_tol_abs_m": 0.030,
            "pipe_mask_reproject_radius_tol_rel": 0.25,
            "pipe_mask_reproject_seed_radius_tol_m": 0.050,
            "pipe_mask_reproject_max_fit_points": 20000,
            "pipe_mask_reproject_sample_stride": 2,
            "pipe_mask_fast_path_when_locked": True,
            "pipe_mask_fast_path_reanchor_every": 15,
            "strip_max_height_px": 160,
            "pipe_mask_reproject_refit_when_selected": False,
            "pipe_mask_reproject_refit_use_normal_axis": True,
            "pipe_mask_reproject_known_radius_recenter": True,
            "pipe_mask_reproject_render_radius_margin_m": 0.005,
            "pipe_mask_reproject_axial_margin_m": 0.10,
            "pipe_mask_reproject_occlusion_tol_m": 0.040,
            "pipe_mask_reproject_exclude_behind": False,
            "pipe_mask_reproject_behind_tol_m": 0.10,
            "pipe_mask_reproject_min_component_px": 400,
            "pipe_mask_reproject_component_keep_frac": 0.02,
            "pipe_mask_reproject_close_px": 5,
            "pipe_mask_reproject_fill_holes": False,
            "enable_pipe_temporal_lock": True,
            "pipe_lock_reject_global_when_locked": True,
            "pipe_lock_release_on_missed": True,
            "pipe_lock_max_missed_frames": 6,
            "pipe_lock_radius_abs_margin_m": 0.05,
            "pipe_lock_radius_rel_margin": 0.40,
            "pipe_lock_max_radius_delta_m": 0.05,
            "pipe_lock_max_axis_delta_deg": 50.0,
            "pipe_lock_max_standoff_delta_m": 0.75,
            "pipe_lock_max_axis_point_delta_m": 0.50,
            "pipe_lock_min_inlier_fraction": 0.30,
            "pipe_lock_max_residual_m": 0.07,
            "pipe_lock_min_compatibility_score": 0.005,
            "enable_pipe_image_temporal_lock": True,
            "pipe_image_lock_max_axis_delta_deg": 18.0,
            "pipe_image_lock_max_center_shift_px": 90.0,
            "pipe_image_lock_max_depth_delta_m": 0.45,
            "pipe_image_lock_min_compatibility_score": 0.02,
            "pipe_image_lock_min_width_fraction": 0.18,
            "pipe_image_lock_min_band_score": 450.0,
            "cylinder_component_max_fit_points": 12000,
            "pipe_pose_reuse_reproject_model": True,
            "pipe_image_axis_from_model": True,
            "pipe_fit_use_ransac_axis": True,
            "pipe_normal_consistency_enabled": True,
            "pipe_normal_cos_min": 0.5,
            "pipe_mask_min_normal_fraction": 0.35,
            # Color-independent cylinder guard (active only without warm scene
            # evidence): thresholds on the azimuthal normal-rotation features.
            # Measured on the cloth bag (color prior off): cloth tangent fits
            # score slope 0.02-0.06 / align 0.43-0.48, the true pipe scores
            # slope 0.50-0.66 / align 0.89-0.93 — both gates sit mid-gap.
            # Inlier coverage does NOT separate (the 0.02 m tolerance band on
            # the cloth spans 90deg+), so its gate is disabled by default.
            "pipe_coloroff_cylinder_guard_enabled": True,
            "pipe_cyl_guard_min_slope": 0.35,
            "pipe_cyl_guard_min_align_frac": 0.70,
            "pipe_cyl_guard_min_inlier_cov_deg": 0.0,
            "pipe_coloroff_acquire_stable_frames": 4,
            "pipe_coloroff_acquire_max_axis_delta_deg": 8.0,
            "pipe_coloroff_acquire_max_line_dist_m": 0.08,
            "pipe_coloroff_release_frames": 12,
            "junction_fresh_min_evidence_z_coloroff": 12.0,
            "pipe_coloroff_acquire_sticky": True,
            "pipe_coloroff_provisional_max_age": 8,
            # Border compensation for the color-less mask (grazing-depth
            # dropout the warm mask normally fills): support grant on
            # truncated sides and wider inside margin for end-memory creation.
            "junction_axial_support_coloroff_mask_deficit_m": 0.04,
            "pipe_end_memory_coloroff_inside_px": 10.0,
            # Normal-voting RANSAC used ONLY in color-less ACQUIRE frames
            # whose component selection produced no cylinder-consistent model.
            "pipe_coloroff_ransac_enabled": True,
            "pipe_coloroff_ransac_iterations": 48,
            "pipe_coloroff_ransac_center_tol_m": 0.025,
            "pipe_coloroff_ransac_min_points": 400,
            "pipe_coloroff_ransac_max_votes": 6000,
            "pipe_coloroff_ransac_min_inliers": 250,
            "pipe_coloroff_ransac_min_extent_m": 0.20,
            "use_cylinder_consensus_pipe_pose": True,
            "pipe_pose_consensus_iterations": 2,
            "pipe_pose_radius_tolerance_m": 0.08,
            "pipe_pose_min_inliers": 600,
            "pipe_pose_min_inlier_fraction": 0.35,
            "strip_vertical_margin_px": 6,
            "min_valid_column_fraction": 0.25,
            "background_window_px": 41,
            "edge_margin_fraction": 0.06,
            "edge_margin_px": 20,
            "min_dark_contrast": 0.015,
            "strong_dark_contrast": 0.06,
            "accept_confidence": 0.25,
            "min_confidence": 0.25,
            "junction_acceptance_mode": "variant_a_rgb",
            # Variant A (deterministic RGB-only) frontend params (mode == "variant_a_rgb").
            "variant_a_tophat_se_len_px": 21,
            "variant_a_min_vertical_run_px": 100,
            "variant_a_min_significance_z": 5.0,
            "variant_a_max_seam_width_px": 14.0,
            "variant_a_border_margin_px": 15,
            "variant_a_z_confidence_strong": 10.0,
            "variant_a_orientation_search_deg": 25.0,
            "variant_a_orientation_search_step_deg": 2.0,
            "variant_a_suppress_pipe_end_columns": True,
            "variant_a_fallback_to_classical_on_border": True,
            "variant_a_classical_fallback_min_z": 8.0,
            "variant_a_classical_fallback_min_contrast": 0.05,
            "variant_a_classical_fallback_max_distance_px": 120,
            # Pipe-end rejection for Variant A (an end/start is not a junction):
            "variant_a_use_depth_pipe_end_gate": True,   # coarse depth: reject a huge jump
            "variant_a_pipe_end_max_depth_jump_m": 0.15,  # >> 3 mm seam, << pipe-end (~0.7 m)
            "rgb_temporal_visual_weight": 0.45,
            "rgb_temporal_edge_score_weight": 0.40,
            "rgb_temporal_temporal_weight": 0.15,
            "rgb_temporal_min_score": 0.35,
            "rgb_temporal_min_line_support_fraction": 0.04,
            "rgb_temporal_strong_line_support_fraction": 0.18,
            "rgb_temporal_min_edge_gradient": 0.020,
            "rgb_temporal_edge_z_score": 3.0,
            "rgb_temporal_line_width_min_residual": 0.008,
            "rgb_temporal_max_line_width_px": 18,
            "rgb_temporal_track_max_jump_px": 180,
            "rgb_temporal_track_missed_max": 5,
            "rgb_temporal_min_track_streak": 2,
            # Collar/step-edge cue (see _step_edge_profile): the real socket
            # junction is a broad luminance step, not only a thin dark line.
            "enable_step_edge_candidate": True,
            "step_edge_window_px": 30,
            "step_edge_gap_px": 4,
            "step_edge_min_z": 3.5,
            "step_edge_min_abs": 0.02,
            "step_edge_reacquire_min_z": 3.0,
            "enable_junction_lock": True,
            "junction_lock_max_missed_frames": 12,
            "junction_lock_search_radius_px": 80,
            # Fresh-reacquire gate scales with how long the detector was blind:
            # allowed = min(max_jump, base + px_per_frame * blind_frames).
            "junction_fresh_reacquire_max_jump_px": 300,
            "junction_fresh_reacquire_base_px": 80.0,
            "junction_fresh_reacquire_px_per_frame": 8.0,
            # Blind coasting on pipe optical flow (see _coast_lock_with_flow).
            "junction_lock_coast_max_dx_px": 40.0,
            "enable_junction_output_smoothing": True,
            "junction_smooth_gain": 0.5,
            "junction_smooth_gain_far": 0.25,
            "junction_smooth_innovation_soft_px": 10.0,
            "junction_smooth_snap_px": 45.0,
            "enable_junction_coast_publish": True,
            "junction_publish_coast_max_frames": 8,
            "junction_coast_publish_edge_margin_px": 12,
            "enable_junction_center_smoothing": True,
            "junction_center_gain": 0.45,
            "junction_center_gain_far": 0.25,
            "junction_center_soft_m": 0.02,
            "junction_center_snap_m": 0.12,
            "junction_axis_center_min_support_px": 200,
            "pipe_color_prior_enabled": True,
            "pipe_color_hue_min": 2,
            "pipe_color_hue_max": 30,
            "pipe_color_sat_min": 60,
            "pipe_color_val_min": 40,
            "pipe_color_bright_rescue_val": 200,
            "pipe_color_min_band_warm_frac": 0.50,
            "pipe_color_min_seed_px_frac": 0.25,
            "junction_min_mask_warm_fraction": 0.35,
            "junction_fresh_min_evidence_z": 1.0,
            "enable_junction_axial_support_gate": True,
            "junction_min_axial_support_m": 0.15,
            "pipe_end_memory_max_age_frames": 90,
            "pipe_end_memory_confirm_inside_px": 10.0,
            "pipe_end_memory_ban_radius_m": 0.13,
            "pipe_end_memory_confirm_frames": 5,
            "pipe_end_memory_decisive_factor": 0.8,
            "pipe_end_memory_other_side_factor": 2.0,
            "junction_axial_support_standoff_scale": 0.4,
            "junction_axial_support_extent_share": 0.45,
            "junction_require_cylinder_mask": True,
            "pipe_lock_require_fit_model": True,
            "pipe_lock_low_warm_release_frames": 10,
            "pipe_mask_reproject_edge_ring_px": 14,
            "pipe_mask_reproject_interior_behind_tol_m": 0.06,
            "enable_pipe_model_smoothing": True,
            "pipe_model_smooth_gain": 0.4,
            "pipe_model_smooth_axis_snap_deg": 8.0,
            "pipe_model_smooth_point_soft_m": 0.010,
            "pipe_model_smooth_point_snap_m": 0.08,
            "junction_lock_min_reacquire_z": 3.0,
            "junction_lock_min_reacquire_contrast": 0.006,
            "junction_lock_confidence_decay": 0.96,
            "junction_lock_min_confidence": 0.25,
            "enable_opencv_klt_tracking": True,
            "klt_canonical_height_px": 160,
            "junction_lock_distance_penalty_z_per_px": 0.04,
            "klt_max_features": 80,
            "klt_feature_radius_px": 80,
            "klt_min_valid_points": 4,
            "klt_quality_level": 0.01,
            "klt_min_distance_px": 5.0,
            "klt_block_size_px": 7,
            "klt_win_size_px": 21,
            "klt_max_level": 3,
            "klt_term_count": 20,
            "klt_term_eps": 0.03,
            "klt_max_visual_disagreement_px": 80,
            "rgb_temporal_pipe_end_side_width_px": 32,
            "rgb_temporal_pipe_end_min_side_columns_px": 24,
            "rgb_temporal_pipe_end_min_side_coverage": 0.18,
            "rgb_temporal_pipe_end_max_coverage_delta": 0.65,
            "candidate_border_allow_if_pipe_supported": True,
            "rgb_temporal_boost_on_temporal_change": True,
            "rgb_temporal_temporal_boost_score": 0.55,
            "rgb_temporal_low_contrast_reject_enabled": True,
            "rgb_temporal_min_candidate_contrast": 0.025,
            "rgb_temporal_shadow_reject_enabled": True,
            "rgb_temporal_shadow_luma_delta_strong": 0.080,
            "rgb_temporal_shadow_chroma_delta_strong": 0.025,
            "rgb_temporal_shadow_max_chromaticity_ratio": 0.35,
            "rgb_temporal_shadow_like_min_score": 0.55,
            "rgb_temporal_shadow_min_line_width_px": 6,
            "rgb_temporal_shadow_max_candidate_contrast": 0.12,
            "rgb_temporal_surface_continuity_reject_enabled": True,
            "rgb_temporal_surface_continuity_min_score": 0.72,
            "rgb_temporal_surface_continuity_max_contrast": 0.10,
            "rgb_temporal_continuity_side_width_px": 14,
            "rgb_temporal_continuity_side_gap_px": 2,
            "rgb_temporal_continuity_min_pixels": 16,
            "rgb_temporal_continuity_chroma_delta_reject": 0.035,
            "rgb_temporal_continuity_texture_delta_reject": 0.020,
            "rgb_temporal_continuity_coverage_delta_reject": 0.25,
            "rgb_local_min_z_score": 4.0,
            "rgb_local_strong_z_score": 8.0,
            "depth_gap_neighbor_offset_px": 10,
            "depth_gap_band_half_width_px": 2,
            "depth_gap_diagnostic_min_score_m": 0.00038,
            "depth_gap_diagnostic_min_coverage_drop": 0.015,
            "depth_gap_coverage_score_scale_m": 0.1,
            "min_depth_gap_samples": 32,
            "enable_temporal_scan_change": True,
            "use_temporal_scan_change_gate": False,
            "temporal_reference_min_frames": 8,
            "temporal_reference_alpha": 0.05,
            "temporal_reference_update_on_reject_only": True,
            "temporal_change_band_half_width_px": 4,
            "min_temporal_change_dark_delta": 0.012,
            "strong_temporal_change_dark_delta": 0.04,
            "min_temporal_change_z_score": 3.0,
            "temporal_min_compare_columns": 64,
            "temporal_min_band_columns": 3,
            "min_confirm_frames": 2,
            "max_axis_angle_delta_deg": 3.0,
            "max_stand_off_delta_m": 0.15,
            "max_yaw_delta_deg": 5.0,
            "max_candidate_jump_px": 120,
            "reset_geometry_on_reject": False,
            "pipe_radius_m": 0.45,
            "surface_band_half_width_px": 4,
            "min_surface_points": 50,
            "last_fx": 1.0,
            "last_fy": 1.0,
            "last_cx": 0.0,
            "last_cy": 0.0,
        }
        values: dict[str, Any] = {}
        for name, default in declarations.items():
            self.declare_parameter(name, default)
            values[name] = self.get_parameter(name).value
        return values

    def _rgb_cb(self, msg: Image) -> None:
        self._rgb_recv.append(self._now_sec())
        self.rgb_queue.append((self._message_time(msg), msg))
        self._try_process()

    def _depth_cb(self, msg: Image) -> None:
        self._depth_recv.append(self._now_sec())
        self.depth_queue.append((self._message_time(msg), msg))
        self._try_process()

    def _info_cb(self, msg: CameraInfo) -> None:
        self._info_recv.append(self._now_sec())
        self.info_queue.append((self._message_time(msg), msg))
        self._try_process()

    def _message_time(self, msg: Image | CameraInfo) -> float:
        if self._effective_use_receive_time:
            return 1e-9 * float(self.get_clock().now().nanoseconds)
        stamp = _stamp_sec(msg)
        if stamp > 0.0:
            return stamp
        return 1e-9 * float(self.get_clock().now().nanoseconds)

    def _try_process(self) -> None:
        if not self.rgb_queue or not self.depth_queue:
            return

        rgb_time, rgb_msg = self.rgb_queue[-1]
        depth_pair = self._closest(self.depth_queue, rgb_time)
        info_pair = self._camera_info_pair(rgb_time)
        if depth_pair is None or info_pair is None:
            return

        depth_time, depth_msg = depth_pair
        info_time, info_msg = info_pair
        slop = float(self.params["sync_slop_s"])
        if abs(depth_time - rgb_time) > slop:
            return
        if not bool(self.params["allow_stale_camera_info"]) and abs(info_time - rgb_time) > slop:
            return
        if self.last_processed_rgb_time is not None and abs(rgb_time - self.last_processed_rgb_time) < 1e-12:
            return
        self.last_processed_rgb_time = rgb_time

        self.received_count += 1
        every_n = max(1, int(self.params["process_every_n"]))
        if self.received_count % every_n != 0:
            return

        try:
            rgb = self._image_to_rgb_array(rgb_msg)
            depth = self._image_to_depth_m(depth_msg)
            k, intrinsics_source = self._camera_matrix(info_msg, rgb_msg)
            self.params["last_fx"] = float(k[0, 0])
            self.params["last_fy"] = float(k[1, 1])
            self.params["last_cx"] = float(k[0, 2])
            self.params["last_cy"] = float(k[1, 2])
            status, rgb_overlay, depth_overlay = self.detector.process(rgb, depth, k)
            status.update(
                {
                    "stamp": {"sec": int(rgb_msg.header.stamp.sec), "nanosec": int(rgb_msg.header.stamp.nanosec)},
                    "frame_id": rgb_msg.header.frame_id,
                    "node_instance": self.instance_id,
                    "rgb_depth_dt_s": _round(depth_time - rgb_time),
                    "rgb_info_dt_s": _round(info_time - rgb_time),
                    "camera_intrinsics_source": intrinsics_source,
                }
            )
            self._publish(status, rgb_overlay, depth_overlay, rgb_msg.header)
        except Exception as exc:
            status = {
                "state": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "processed_frame_count": self.detector.processed_frame_count,
            }
            self.detection_pub.publish(String(data=json.dumps(status, sort_keys=True)))
            self.last_status_publish_time = self._now_sec()
            self.get_logger().warn(status["error"], throttle_duration_sec=2.0)

    @staticmethod
    def _closest(queue: deque[tuple[float, Any]], target: float) -> tuple[float, Any] | None:
        if not queue:
            return None
        return min(queue, key=lambda pair: abs(pair[0] - target))

    def _camera_info_pair(self, target: float) -> tuple[float, CameraInfo] | None:
        valid_pairs = [(stamp, msg) for stamp, msg in self.info_queue if self._camera_info_valid(msg)]
        if not valid_pairs and bool(self.params["allow_nominal_intrinsics_fallback"]):
            valid_pairs = list(self.info_queue)
        if not valid_pairs and bool(self.params["allow_nominal_intrinsics_fallback"]):
            return target, CameraInfo()
        if not valid_pairs:
            return None
        if bool(self.params["allow_stale_camera_info"]):
            return valid_pairs[-1]
        return min(valid_pairs, key=lambda pair: abs(pair[0] - target))

    def _create_camera_subscriptions(self) -> None:
        self._rgb_sub = self.create_subscription(
            Image,
            str(self.params["rgb_topic"]),
            self._rgb_cb,
            self._camera_qos,
        )
        self._depth_sub = self.create_subscription(
            Image,
            str(self.params["depth_topic"]),
            self._depth_cb,
            self._camera_qos,
        )
        self._info_sub = self.create_subscription(
            CameraInfo,
            str(self.params["camera_info_topic"]),
            self._info_cb,
            self._camera_qos,
        )

    def _reset_camera_subscriptions(self, reason: str) -> None:
        now = self._now_sec()
        min_period = float(self.params["stale_subscription_reset_s"])
        if now - self._last_subscription_reset_time < min_period:
            return
        self._last_subscription_reset_time = now

        for sub in (self._rgb_sub, self._depth_sub, self._info_sub):
            if sub is None:
                continue
            try:
                self.destroy_subscription(sub)
            except Exception as exc:
                self.get_logger().warn(
                    f"camera subscription destroy failed during stale recovery: {type(exc).__name__}: {exc}",
                    throttle_duration_sec=10.0,
                )

        self.rgb_queue.clear()
        self.depth_queue.clear()
        self.info_queue.clear()
        self._rgb_recv.clear()
        self._depth_recv.clear()
        self._info_recv.clear()
        self.last_processed_rgb_time = None
        self._create_camera_subscriptions()
        self.get_logger().warn(
            f"camera streams became stale ({reason}); recreated RGB/depth/camera_info subscriptions "
            "to recover after a Gazebo/ros_gz_bridge restart.",
            throttle_duration_sec=5.0,
        )

    @staticmethod
    def _camera_info_valid(msg: CameraInfo) -> bool:
        try:
            k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        except Exception:
            return False
        fx, fy = float(k[0, 0]), float(k[1, 1])
        return bool(np.isfinite(k).all() and abs(fx) > 1e-9 and abs(fy) > 1e-9)

    def _camera_matrix(self, info_msg: CameraInfo, rgb_msg: Image) -> tuple[np.ndarray, str]:
        if self._camera_info_valid(info_msg):
            return np.asarray(info_msg.k, dtype=np.float64).reshape(3, 3), "camera_info"
        if not bool(self.params["allow_nominal_intrinsics_fallback"]):
            raise ValueError("CameraInfo K is invalid and nominal intrinsics fallback is disabled")

        fx = float(self.params["nominal_fx_px"])
        fy = float(self.params["nominal_fy_px"])
        cx = float(self.params["nominal_cx_px"])
        cy = float(self.params["nominal_cy_px"])
        if cx < 0.0:
            cx = 0.5 * float(rgb_msg.width)
        if cy < 0.0:
            cy = 0.5 * float(rgb_msg.height)
        if not all(np.isfinite(v) for v in (fx, fy, cx, cy)) or abs(fx) < 1e-9 or abs(fy) < 1e-9:
            raise ValueError("Nominal camera intrinsics fallback is invalid")
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64), "nominal_fallback"

    @staticmethod
    def _status_vector3(status: dict[str, Any], key: str) -> np.ndarray | None:
        value = status.get(key)
        if value is None:
            return None
        try:
            vector = np.asarray(value, dtype=np.float64).reshape(3)
        except Exception:
            return None
        if not np.all(np.isfinite(vector)):
            return None
        return vector

    def _build_weld_gap_geometry(
        self,
        status: dict[str, Any],
        header: Any,
    ) -> tuple[dict[str, Any] | None, Any | None, Any | None]:
        if not bool(self.params["publish_weld_gap_geometry"]):
            return None, None, None

        detector_accepted = bool(status.get("detector_accepted", False))
        require_accepted = bool(self.params["weld_gap_require_detector_accepted"])
        axis = self._status_vector3(status, "pipe_axis_camera_xyz")
        center_point = self._status_vector3(status, "coarse_seam_center_camera_xyz_m")
        surface_point = self._status_vector3(status, "coarse_seam_visible_surface_camera_xyz_m")
        support_pixel_count = status.get("coarse_seam_support_pixel_count")

        reason = "ok"
        plane_valid = True
        if require_accepted and not detector_accepted:
            reason = "detector_not_accepted"
            plane_valid = False
        elif axis is None:
            reason = "missing_pipe_axis_camera_xyz"
            plane_valid = False
        elif surface_point is None:
            reason = "missing_coarse_seam_visible_surface_camera_xyz_m"
            plane_valid = False

        seam_frame = None
        pose_msg = None
        pose_valid = False
        pose_reason = reason
        if plane_valid:
            pose_reason = "ok"
            if surface_point is None:
                pose_reason = "missing_coarse_seam_visible_surface_camera_xyz_m"
            elif seam_frame_from_axis_and_surface is None:
                pose_reason = "weld_seam_helper_unavailable"
            else:
                seam_frame = seam_frame_from_axis_and_surface(axis, surface_point)
                if seam_frame is None:
                    pose_reason = "weld_seam_frame_degenerate_or_invalid"
                else:
                    pose_valid = True

        if pose_valid and PoseStamped is not None and seam_frame is not None:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = header.stamp
            pose_msg.header.frame_id = header.frame_id
            pose_msg.pose.position.x = float(seam_frame.origin_xyz_m[0])
            pose_msg.pose.position.y = float(seam_frame.origin_xyz_m[1])
            pose_msg.pose.position.z = float(seam_frame.origin_xyz_m[2])
            pose_msg.pose.orientation.x = float(seam_frame.quat_xyzw[0])
            pose_msg.pose.orientation.y = float(seam_frame.quat_xyzw[1])
            pose_msg.pose.orientation.z = float(seam_frame.quat_xyzw[2])
            pose_msg.pose.orientation.w = float(seam_frame.quat_xyzw[3])

        payload = {
            "valid": bool(plane_valid),
            "reason": reason,
            "pose_valid": bool(pose_valid),
            "pose_reason": pose_reason,
            "source": "acea_pipe_junction_node",
            "metric_3d_available": bool(plane_valid and surface_point is not None),
            "uses_assumed_depth": False,
            "gap_plane_point_source": "depth_backprojected_visible_surface_median",
            "gap_plane_center_source": "depth_surface_point_plus_pipe_radius_radial_projection",
            "gap_plane_normal_source": "depth_pipe_tracker_axis_camera_xyz",
            "frame_id": header.frame_id,
            "stamp": status.get("stamp"),
            "state": status.get("state"),
            "detector_accepted": detector_accepted,
            "confidence": status.get("confidence"),
            "candidate_streak": status.get("candidate_streak"),
            "support_pixel_count": support_pixel_count,
            "gap_plane_point_camera_xyz_m": _round_list(surface_point),
            "gap_plane_center_camera_xyz_m": _round_list(center_point),
            "gap_plane_normal_camera_xyz": _round_list(axis),
            "seam_visible_surface_camera_xyz_m": _round_list(surface_point),
            "seam_frame_origin_camera_xyz_m": None
            if seam_frame is None
            else _round_list(seam_frame.origin_xyz_m),
            "seam_frame_quaternion_xyzw": None
            if seam_frame is None
            else _round_list(seam_frame.quat_xyzw),
            "seam_frame_surface_normal_camera_xyz": None
            if seam_frame is None
            else _round_list(seam_frame.surface_normal),
            "seam_frame_tangent_camera_xyz": None
            if seam_frame is None
            else _round_list(seam_frame.seam_tangent),
            "seam_frame_degenerate": None if seam_frame is None else bool(seam_frame.degenerate),
            "weld_seam_pose_topic": str(self.params["weld_seam_pose_topic"]),
            "weld_gap_plane_topic": str(self.params["weld_gap_plane_topic"]),
            "convention": {
                "gap_plane_normal": "pipe_axis_camera_xyz",
                "gap_plane_point": "coarse_seam_visible_surface_camera_xyz_m",
                "gap_plane_center": "coarse_seam_center_camera_xyz_m",
                "pose_origin": "coarse_seam_visible_surface_camera_xyz_m",
                "pose_x": "outward_surface_normal_camera_xyz",
                "pose_y": "circumferential_seam_tangent_camera_xyz",
                "pose_z": "gap_plane_normal_pipe_axis_camera_xyz",
            },
        }
        marker_array = self._build_weld_markers(payload, seam_frame, center_point, status, header)
        return payload, pose_msg, marker_array

    def _weld_cylinder_radius(self, status: dict[str, Any]) -> float:
        """Radius for the RViz cylinder: the depth-fitted pipe radius when
        available, else the nominal ``pipe_radius_m`` parameter."""
        fitted = status.get("pipe_pose_radius_m")
        if fitted is not None and math.isfinite(float(fitted)) and float(fitted) > 1e-3:
            return float(fitted)
        return float(self.params["pipe_radius_m"])

    def _make_marker(
        self,
        header: Any,
        marker_id: int,
        position: Any,
        quat_xyzw: Any,
        scale_xyz: tuple[float, float, float],
        color_rgba: tuple[float, float, float, float],
    ) -> Any:
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = header.frame_id
        marker.ns = "acea_weld_seam"
        marker.id = int(marker_id)
        marker.type = _MARKER_CYLINDER
        marker.action = _MARKER_ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.x = float(quat_xyzw[0])
        marker.pose.orientation.y = float(quat_xyzw[1])
        marker.pose.orientation.z = float(quat_xyzw[2])
        marker.pose.orientation.w = float(quat_xyzw[3])
        marker.scale.x = float(scale_xyz[0])
        marker.scale.y = float(scale_xyz[1])
        marker.scale.z = float(scale_xyz[2])
        marker.color.r = float(color_rgba[0])
        marker.color.g = float(color_rgba[1])
        marker.color.b = float(color_rgba[2])
        marker.color.a = float(color_rgba[3])
        return marker

    def _build_weld_markers(
        self,
        payload: dict[str, Any],
        seam_frame: Any,
        center_point: Any,
        status: dict[str, Any],
        header: Any,
    ) -> Any | None:
        """RViz MarkerArray: the fitted pipe cylinder + a thin disk for the gap
        plane, both oriented by the seam frame (local z = pipe axis). Returns a
        single DELETEALL array when there is no valid pose so stale markers clear."""
        if Marker is None or MarkerArray is None or self.weld_marker_pub is None:
            return None
        array = MarkerArray()
        if not payload.get("pose_valid") or seam_frame is None or center_point is None:
            clear = Marker()
            clear.header.stamp = header.stamp
            clear.header.frame_id = header.frame_id
            clear.ns = "acea_weld_seam"
            clear.action = _MARKER_DELETEALL
            array.markers = [clear]
            return array
        radius = self._weld_cylinder_radius(status)
        quat = seam_frame.quat_xyzw
        length = float(self.params["weld_marker_cylinder_length_m"])
        plane_scale = float(self.params["weld_marker_plane_scale"])
        center = [float(c) for c in center_point]
        # id 0: the fitted pipe as a translucent cylinder along the pipe axis.
        cylinder = self._make_marker(
            header, 0, center, quat,
            (2.0 * radius, 2.0 * radius, length),
            (0.2, 0.5, 1.0, 0.35),
        )
        # id 1: the gap plane as a thin disk perpendicular to the axis at the seam.
        disk = self._make_marker(
            header, 1, center, quat,
            (2.0 * radius * plane_scale, 2.0 * radius * plane_scale, 0.005),
            (1.0, 0.55, 0.1, 0.55),
        )
        array.markers = [cylinder, disk]
        return array

    def _publish_weld_gap_geometry(
        self, payload: dict[str, Any] | None, pose_msg: Any | None, marker_array: Any | None
    ) -> None:
        if payload is None:
            return
        if self.weld_gap_plane_pub is not None:
            self.weld_gap_plane_pub.publish(String(data=json.dumps(payload, sort_keys=True, allow_nan=False)))
        if payload.get("pose_valid") is True and pose_msg is not None and self.weld_seam_pose_pub is not None:
            self.weld_seam_pose_pub.publish(pose_msg)
        if marker_array is not None and self.weld_marker_pub is not None:
            self.weld_marker_pub.publish(marker_array)

    def _publish_debug_status(self, status: dict[str, Any]) -> None:
        accepted = bool(status.get("detector_accepted", status.get("accepted", False)))
        compact = {
            "state": status.get("state"),
            "detected": accepted,
            "detector_accepted": accepted,
            "node_instance": self.instance_id,
            "reason": status.get("reason"),
            "hint": status.get("hint"),
            "confidence": status.get("confidence"),
            "candidate_x_strip_px": status.get("candidate_x_strip_px"),
            "candidate_x_image_px": status.get("candidate_x_image_px"),
            "variant_a_orientation_deg": status.get("variant_a_orientation_deg"),
            "junction_lock_active": bool(status.get("junction_lock_active", False)),
            "junction_lock_used": bool(status.get("junction_lock_used", False)),
            "junction_lock_missed_frames": status.get("junction_lock_missed_frames"),
            "junction_lock_confidence": status.get("junction_lock_confidence"),
            "junction_lock_source": status.get("junction_lock_source"),
            "junction_last_valid_x_strip_px": status.get("junction_last_valid_x_strip_px"),
            "junction_last_valid_frame": status.get("junction_last_valid_frame"),
            "fresh_reacquire_ok": status.get("fresh_reacquire_ok"),
            "fresh_reacquire_reason": status.get("fresh_reacquire_reason"),
            "klt_status": status.get("klt_status"),
            "klt_points": status.get("klt_points"),
            "klt_dx_px": status.get("klt_dx_px"),
            "rgb_dark_accepted": status.get("rgb_dark_accepted"),
            "depth_gap_accepted": status.get("depth_gap_accepted"),
            "pipe_end_rejected": status.get("pipe_end_rejected"),
            "pipe_support_left_cols": status.get("pipe_support_left_cols"),
            "pipe_support_right_cols": status.get("pipe_support_right_cols"),
            "pipe_support_left_coverage": status.get("pipe_support_left_coverage"),
            "pipe_support_right_coverage": status.get("pipe_support_right_coverage"),
            "candidate_not_border": status.get("candidate_not_border"),
            "pipe_component_selection_method": status.get("pipe_component_selection_method"),
            "pipe_component_count": status.get("pipe_component_count"),
            "pipe_component_candidate_count": status.get("pipe_component_candidate_count"),
            "pipe_component_rejected_by_shape": status.get("pipe_component_rejected_by_shape"),
            "pipe_component_cylinder_evaluated": status.get("pipe_component_cylinder_evaluated"),
            "pipe_component_cylinder_valid": status.get("pipe_component_cylinder_valid"),
            "pipe_component_selected_label": status.get("pipe_component_selected_label"),
            "pipe_component_fallback_label": status.get("pipe_component_fallback_label"),
            "pipe_component_band_valid": status.get("pipe_component_band_valid"),
            "pipe_component_band_score": status.get("pipe_component_band_score"),
            "pipe_component_band_width_fraction": status.get("pipe_component_band_width_fraction"),
            "pipe_component_band_column_coverage": status.get("pipe_component_band_column_coverage"),
            "pipe_component_band_pixels": status.get("pipe_component_band_pixels"),
            "pipe_component_band_method": status.get("pipe_component_band_method"),
            "pipe_lock_active": status.get("pipe_lock_active"),
            "pipe_image_lock_active": status.get("pipe_image_lock_active"),
            "pipe_lock_missed_frames": status.get("pipe_lock_missed_frames"),
            "pipe_lock_source": status.get("pipe_lock_source"),
            "pipe_image_lock_source": status.get("pipe_image_lock_source"),
            "pipe_lock_selection_score": status.get("pipe_lock_selection_score"),
            "pipe_lock_axis_delta_deg": status.get("pipe_lock_axis_delta_deg"),
            "pipe_lock_radius_delta_m": status.get("pipe_lock_radius_delta_m"),
            "pipe_lock_stand_delta_m": status.get("pipe_lock_stand_delta_m"),
            "pipe_lock_axis_point_delta_m": status.get("pipe_lock_axis_point_delta_m"),
            "pipe_image_lock_axis_delta_deg": status.get("pipe_image_lock_axis_delta_deg"),
            "pipe_image_lock_center_delta_px": status.get("pipe_image_lock_center_delta_px"),
            "pipe_image_lock_depth_delta_m": status.get("pipe_image_lock_depth_delta_m"),
            "gap_plane_available": bool(status.get("gap_plane_available", False)),
            "weld_seam_pose_available": bool(status.get("weld_seam_pose_available", False)),
            "frame_id": status.get("frame_id"),
            "processed_frame_count": status.get("processed_frame_count"),
            "rgb_queue": status.get("rgb_queue"),
            "depth_queue": status.get("depth_queue"),
            "camera_info_queue": status.get("camera_info_queue"),
            "rgb_topic": str(self.params["rgb_topic"]),
            "depth_topic": str(self.params["depth_topic"]),
            "camera_info_topic": str(self.params["camera_info_topic"]),
            "rgb_hz": status.get("rgb_hz"),
            "depth_hz": status.get("depth_hz"),
            "camera_info_hz": status.get("camera_info_hz"),
            "rgb_age_s": status.get("rgb_age_s"),
            "depth_age_s": status.get("depth_age_s"),
            "camera_info_age_s": status.get("camera_info_age_s"),
            "sync_time_source": status.get("sync_time_source"),
            "sync_slop_s": status.get("sync_slop_s"),
            "latest_rgb_depth_dt_s": status.get("latest_rgb_depth_dt_s"),
            "latest_rgb_info_dt_s": status.get("latest_rgb_info_dt_s"),
            "received_synced_candidate_count": status.get("received_synced_candidate_count"),
        }
        self.detected_pub.publish(Bool(data=accepted))
        self.status_pub.publish(String(data=json.dumps(compact, sort_keys=True, allow_nan=False)))

    def _publish(self, status: dict[str, Any], rgb_overlay: np.ndarray, depth_overlay: np.ndarray, header: Any) -> None:
        # Stamp the detection with the processed-RGB header time so an external
        # tool can pair this detection to the EXACT recorded frame (by stamp, not
        # latest-arrival) -- removes async during fast sweeps for projected-GT.
        if header is not None and getattr(header, "stamp", None) is not None:
            status["rgb_stamp_s"] = round(
                float(header.stamp.sec) + 1e-9 * float(header.stamp.nanosec), 9)
        gap_payload, pose_msg, marker_array = self._build_weld_gap_geometry(status, header)
        if gap_payload is not None:
            status.update(
                {
                    "gap_plane_available": bool(gap_payload["valid"]),
                    "gap_plane_reason": gap_payload["reason"],
                    "gap_plane_metric_3d_available": bool(gap_payload["metric_3d_available"]),
                    "gap_plane_uses_assumed_depth": bool(gap_payload["uses_assumed_depth"]),
                    "gap_plane_point_source": gap_payload["gap_plane_point_source"],
                    "gap_plane_point_camera_xyz_m": gap_payload["gap_plane_point_camera_xyz_m"],
                    "gap_plane_center_camera_xyz_m": gap_payload["gap_plane_center_camera_xyz_m"],
                    "gap_plane_normal_camera_xyz": gap_payload["gap_plane_normal_camera_xyz"],
                    "weld_seam_pose_available": bool(gap_payload["pose_valid"]),
                    "weld_seam_pose_reason": gap_payload["pose_reason"],
                    "weld_seam_pose_topic": gap_payload["weld_seam_pose_topic"],
                    "weld_gap_plane_topic": gap_payload["weld_gap_plane_topic"],
                    "weld_marker_topic": str(self.params["weld_marker_topic"]),
                }
            )
        self.detection_pub.publish(String(data=json.dumps(status, sort_keys=True, allow_nan=False)))
        self._publish_debug_status(status)
        self.last_status_publish_time = self._now_sec()
        self._publish_weld_gap_geometry(gap_payload, pose_msg, marker_array)
        self.rgb_overlay_pub.publish(self._rgb_array_to_msg(rgb_overlay, header))
        self.depth_overlay_pub.publish(self._rgb_array_to_msg(depth_overlay, header))

    def _publish_waiting_status(self) -> None:
        now = self._now_sec()
        if now - self.last_status_publish_time < 0.8 * float(self.params["waiting_status_period_s"]):
            return

        rgb_time = self.rgb_queue[-1][0] if self.rgb_queue else None
        depth_pair = self._closest(self.depth_queue, rgb_time) if rgb_time is not None else None
        info_pair = self._camera_info_pair(rgb_time) if rgb_time is not None else None
        depth_dt = None if depth_pair is None or rgb_time is None else depth_pair[0] - rgb_time
        info_dt = None if info_pair is None or rgb_time is None else info_pair[0] - rgb_time
        latest_camera_info_valid = self._camera_info_valid(self.info_queue[-1][1]) if self.info_queue else False
        rgb_age = self._stream_age(self._rgb_recv)
        depth_age = self._stream_age(self._depth_recv)
        info_age = self._stream_age(self._info_recv)
        stale_s = float(self.params["stream_stale_s"])

        # Self-heal the worst sync failure: both streams flow but nothing ever
        # syncs because the header stamps are unusable (zero, or sim vs wall).
        self._maybe_engage_receive_time_fallback()

        if not self.rgb_queue:
            reason = "waiting_for_rgb"
        elif not self.depth_queue:
            reason = "waiting_for_depth"
        elif not self.info_queue:
            reason = "waiting_for_camera_info"
        elif rgb_age is not None and rgb_age > stale_s:
            reason = "waiting_for_rgb_stale"
        elif depth_age is not None and depth_age > stale_s:
            reason = "waiting_for_depth_stale"
        elif info_age is not None and info_age > stale_s:
            reason = "waiting_for_camera_info_stale"
        elif info_pair is None:
            reason = "waiting_for_valid_camera_info"
        elif depth_dt is not None and abs(depth_dt) > float(self.params["sync_slop_s"]):
            reason = "waiting_for_rgb_depth_sync"
        else:
            reason = "waiting_for_next_frame"

        if reason in {
            "waiting_for_rgb_stale",
            "waiting_for_depth_stale",
            "waiting_for_camera_info_stale",
        }:
            self._reset_camera_subscriptions(reason)

        status = {
            "state": "WAITING_FOR_SYNC",
            "reason": reason,
            "hint": self._waiting_hint(reason),
            "node_instance": self.instance_id,
            "rgb_queue": len(self.rgb_queue),
            "depth_queue": len(self.depth_queue),
            "camera_info_queue": len(self.info_queue),
            "rgb_hz": _round(self._stream_hz(self._rgb_recv), 2),
            "depth_hz": _round(self._stream_hz(self._depth_recv), 2),
            "camera_info_hz": _round(self._stream_hz(self._info_recv), 2),
            "rgb_age_s": _round(rgb_age, 2),
            "depth_age_s": _round(depth_age, 2),
            "camera_info_age_s": _round(info_age, 2),
            "sync_time_source": "receive" if self._effective_use_receive_time else "header",
            "latest_camera_info_valid": latest_camera_info_valid,
            "processed_frame_count": self.detector.processed_frame_count,
            "received_synced_candidate_count": self.received_count,
            "sync_slop_s": _round(float(self.params["sync_slop_s"])),
            "latest_rgb_depth_dt_s": _round(depth_dt),
            "latest_rgb_info_dt_s": _round(info_dt),
            "allow_stale_camera_info": bool(self.params["allow_stale_camera_info"]),
            "allow_nominal_intrinsics_fallback": bool(self.params["allow_nominal_intrinsics_fallback"]),
        }
        self.detection_pub.publish(String(data=json.dumps(status, sort_keys=True, allow_nan=False)))
        self._publish_debug_status(status)
        self.last_status_publish_time = now

    def _stream_hz(self, recv: deque[float]) -> float | None:
        if len(recv) < 2:
            return None
        span = recv[-1] - recv[0]
        if span <= 0.0:
            return None
        return (len(recv) - 1) / span

    def _stream_age(self, recv: deque[float]) -> float | None:
        if not recv:
            return None
        return max(0.0, self._now_sec() - recv[-1])

    def _waiting_hint(self, reason: str) -> str:
        depth_age = self._stream_age(self._depth_recv)
        if reason == "waiting_for_rgb":
            return "no RGB frames arriving: check rgb_topic and that the camera/bridge publishes it."
        if reason == "waiting_for_depth":
            return ("no depth frames arriving: check depth_topic and the depth bridge "
                    "(Gazebo depth sensor / RealSense aligned depth).")
        if reason == "waiting_for_camera_info":
            return "no CameraInfo yet: check camera_info_topic (nominal-intrinsics fallback may apply)."
        if reason == "waiting_for_valid_camera_info":
            return "CameraInfo received but its K matrix is zero/invalid."
        if reason == "waiting_for_rgb_stale":
            return ("RGB topic exists but this detector is not receiving fresh RGB frames. "
                    "Check topic mismatch, stale detector instance, QoS, or the RGB bridge.")
        if reason == "waiting_for_depth_stale":
            return ("Depth topic exists but this detector is not receiving fresh depth frames. "
                    "Check topic mismatch, stale detector instance, QoS, or the depth bridge.")
        if reason == "waiting_for_camera_info_stale":
            return ("CameraInfo topic exists but this detector is not receiving fresh CameraInfo. "
                    "Check topic mismatch, stale detector instance, or camera_info bridge.")
        if reason == "waiting_for_rgb_depth_sync":
            if depth_age is not None and depth_age > 1.0:
                return ("depth stream stalled or much slower than RGB; check the depth rate "
                        "or raise sync_slop_s.")
            return ("RGB and depth stamps differ by more than sync_slop_s; likely inconsistent "
                    "clocks. receive-time fallback engages automatically if it persists.")
        return "frames flowing; the next synced pair will be processed."

    def _maybe_engage_receive_time_fallback(self) -> None:
        if (
            self._effective_use_receive_time
            or not bool(self.params.get("auto_receive_time_fallback", True))
            or self.received_count > 0
            or len(self._rgb_recv) < 5
            or len(self._depth_recv) < 5
        ):
            return
        # Both streams are clearly flowing yet not a single pair has synced -> the
        # header stamps are unusable. Switch to arrival-time sync and reset the
        # queues so old (header-time) entries do not mix with new (receive-time).
        self._effective_use_receive_time = True
        self.rgb_queue.clear()
        self.depth_queue.clear()
        self.info_queue.clear()
        if not self._receive_time_fallback_logged:
            self._receive_time_fallback_logged = True
            self.get_logger().warn(
                "RGB and depth are both arriving but no pair synced on header stamps; "
                "switching to receive-time sync (stamps look zero or mixed sim/wall clock). "
                "Set use_receive_time_for_sync:=true to make this the default.")

    def _check_duplicate_graph(self) -> None:
        if count_named_nodes(self) > 1:
            self.get_logger().error(
                f"Multiple live nodes named '{self.get_name()}' on the ROS graph "
                f"(this one: {self.instance_id}). They publish conflicting detections to "
                "the same topics. A same-host copy is already blocked, so this is most "
                "likely a second container/host — keep exactly one detector running.",
                throttle_duration_sec=10.0)

    def _now_sec(self) -> float:
        return 1e-9 * float(self.get_clock().now().nanoseconds)

    @staticmethod
    def _image_to_rgb_array(msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        height, width = int(msg.height), int(msg.width)
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        if encoding in ("rgb8", "bgr8"):
            channels = 3
            rows = raw.reshape(height, int(msg.step))
            array = rows[:, :width * channels].reshape(height, width, channels).copy()
            if encoding == "bgr8":
                array = array[:, :, ::-1].copy()
            return array
        if encoding in ("rgba8", "bgra8"):
            channels = 4
            rows = raw.reshape(height, int(msg.step))
            array = rows[:, :width * channels].reshape(height, width, channels).copy()
            if encoding == "bgra8":
                array = array[:, :, [2, 1, 0, 3]]
            return array[:, :, :3].copy()
        if encoding in ("mono8", "8uc1"):
            rows = raw.reshape(height, int(msg.step))
            gray = rows[:, :width].copy()
            return np.repeat(gray[:, :, None], 3, axis=2)
        raise ValueError(f"Unsupported RGB image encoding: {msg.encoding}")

    @staticmethod
    def _image_to_depth_m(msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        height, width = int(msg.height), int(msg.width)
        if encoding in ("32fc1", "passthrough"):
            raw = np.frombuffer(msg.data, dtype=np.float32)
            row_floats = int(msg.step) // np.dtype(np.float32).itemsize
            return raw.reshape(height, row_floats)[:, :width].astype(np.float32).copy()
        if encoding == "16uc1":
            raw = np.frombuffer(msg.data, dtype=np.uint16)
            row_values = int(msg.step) // np.dtype(np.uint16).itemsize
            return (raw.reshape(height, row_values)[:, :width].astype(np.float32) / 1000.0).copy()
        raise ValueError(f"Unsupported depth image encoding: {msg.encoding}")

    @staticmethod
    def _rgb_array_to_msg(array: np.ndarray, header: Any) -> Image:
        output = Image()
        output.header = header
        output.height = int(array.shape[0])
        output.width = int(array.shape[1])
        output.encoding = "rgb8"
        output.is_bigendian = 0
        output.step = int(array.shape[1] * 3)
        output.data = np.ascontiguousarray(array.astype(np.uint8)).tobytes()
        return output


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = AceaPipeJunctionNode()
        rclpy.spin(node)
    except DuplicateInstanceError as exc:
        # Half-constructed node may exist; use plain stderr (visible in launch log).
        print(f"[acea_pipe_junction_node] refusing to start: {exc}", file=sys.stderr)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            lock = getattr(node, "_single_instance", None)
            if lock is not None:
                lock.release()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
