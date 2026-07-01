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


def _median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    half = window // 2
    output = np.empty_like(values, dtype=np.float64)
    for i in range(values.size):
        output[i] = float(np.median(values[max(0, i - half):min(values.size, i + half + 1)]))
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


@dataclass
class LocalizationResult:
    visible_surface_center_xyz_m: np.ndarray | None
    pipe_center_estimate_xyz_m: np.ndarray | None
    support_pixel_count: int


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
        self.pipe_component_selected_model: dict[str, Any] | None = None
        self.pipe_lock_model: dict[str, Any] | None = None
        self.pipe_lock_missed_frames = 0
        self.pipe_lock_source = "none"
        self.junction_lock_x: float | None = None
        self.junction_lock_velocity_px = 0.0
        self.junction_lock_streak = 0
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
        try:
            tracker = self._track_pipe(depth, k)
        except ValueError as exc:
            self._mark_pipe_lock_missed(str(exc))
            self._release_junction_lock("pipe_not_valid")
            status = self._no_pipe_status(str(exc))
            return status, rgb.copy(), _depth_visual(depth)
        seam = self._detect_seam(rgb, depth, tracker)
        state_info = self._update_state(tracker, seam)
        self._update_temporal_reference(seam)
        localization = None
        if self.state in ("CONFIRMED", "STOP_AND_LOCALIZE"):
            localization = self._localize_confirmed_seam(depth, k, tracker, seam)

        rgb_overlay = self._draw_overlay(rgb, tracker, seam, localization)
        depth_overlay = self._draw_overlay(_depth_visual(depth), tracker, seam, localization)
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
            "pipe_lock_active": bool(self.pipe_lock_model is not None),
            "pipe_lock_missed_frames": int(self.pipe_lock_missed_frames),
            "pipe_lock_source": self.pipe_lock_source,
            "pipe_lock_selection_score": self.pipe_component_selection_info.get("score"),
            "pipe_lock_axis_delta_deg": self.pipe_component_selection_info.get("lock_axis_delta_deg"),
            "pipe_lock_radius_delta_m": self.pipe_component_selection_info.get("lock_radius_delta_m"),
            "pipe_lock_stand_delta_m": self.pipe_component_selection_info.get("lock_stand_delta_m"),
            "pipe_lock_axis_point_delta_m": self.pipe_component_selection_info.get("lock_axis_point_delta_m"),
            "junction_lock_active": bool(self.junction_lock_active),
            "junction_lock_missed_frames": int(self.junction_lock_missed_frames),
            "junction_lock_confidence": _round(self.junction_lock_confidence),
            "junction_lock_source": self.junction_lock_source,
            "gap_plane_available": False,
            "weld_seam_pose_available": False,
        }

    def _mark_pipe_lock_missed(self, reason: str) -> None:
        if self.pipe_lock_model is None:
            self.pipe_lock_source = "none"
            return
        self.pipe_lock_missed_frames += 1
        self.pipe_lock_source = f"missed:{reason}"
        if (
            bool(self.params["pipe_lock_release_on_missed"])
            and self.pipe_lock_missed_frames > int(self.params["pipe_lock_max_missed_frames"])
        ):
            self.pipe_lock_model = None
            self.pipe_lock_source = "released:missed_too_long"

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
        model = self.pipe_component_selected_model or self._tracker_pipe_model(tracker)
        if not self._pipe_model_valid_for_lock(model):
            self._mark_pipe_lock_missed("tracker_model_invalid")
            return
        assert model is not None
        self.pipe_lock_model = model
        self.pipe_lock_missed_frames = 0
        self.pipe_lock_source = "tracker_update"

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

    def _track_pipe(self, depth: np.ndarray, k: np.ndarray) -> TrackerResult:
        min_depth = float(self.params["min_depth_m"])
        max_depth = float(self.params["max_depth_m"])
        valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        threshold_info = _depth_threshold_kmeans(depth, valid, int(self.params["sample_stride"]))
        pipe_mask_raw = valid & (depth <= float(threshold_info["threshold_m"]))
        pipe_mask = self._select_pipe_connected_component(pipe_mask_raw, depth, k)
        ys, xs = np.nonzero(pipe_mask)
        if xs.size < int(self.params["min_pipe_pixels"]):
            raise ValueError(f"Only {xs.size} pipe pixels selected")

        max_pca_points = int(self.params["max_pca_points"])
        if xs.size > max_pca_points:
            indices = np.linspace(0, xs.size - 1, max_pca_points, dtype=np.int64)
            xs_pca = xs[indices]
            ys_pca = ys[indices]
        else:
            xs_pca = xs
            ys_pca = ys

        points_camera = _backproject(depth, xs_pca, ys_pca, k)
        uv_pca = _pca(np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1))
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

        image_direction = _normalize(uv_pca["direction"])
        if image_direction[0] < 0.0:
            image_direction *= -1.0
        image_axis_angle_deg = math.degrees(math.atan2(float(image_direction[1]), float(image_direction[0])))

        axis_camera = _normalize(xyz_fit["direction"])
        if axis_camera[0] < 0.0:
            axis_camera *= -1.0
        pipe_depths = depth[pipe_mask]
        fit_points = np.asarray(xyz_fit["points"], dtype=np.float64)
        fit_inliers = np.asarray(xyz_fit["inlier_mask"], dtype=bool)
        inlier_points = fit_points[fit_inliers] if fit_inliers.size == fit_points.shape[0] else fit_points
        if inlier_points.shape[0] < 3:
            inlier_points = fit_points
        fit_centroid = np.median(inlier_points, axis=0) if inlier_points.shape[0] else xyz_fit["centroid"]
        yaw_error_deg = math.degrees(math.atan2(float(axis_camera[2]), float(axis_camera[0])))
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        tracker = TrackerResult(
            pipe_mask=pipe_mask,
            pipe_pixels=int(pipe_mask.sum()),
            pipe_fraction=float(pipe_mask.mean()),
            bbox_uv=[x0, y0, x1, y1],
            image_centroid_uv=uv_pca["centroid"],
            image_direction_uv=image_direction,
            image_axis_angle_deg=image_axis_angle_deg,
            image_line_segment_uv=_line_box_segment(uv_pca["centroid"], image_direction, depth.shape[1], depth.shape[0]),
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
        if count <= 1:
            self.pipe_component_selection_info = {
                "method": "single_component",
                "component_count": int(count),
                "selected_label": 1 if count == 1 else None,
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
            width_fraction = bbox_w / max(float(image_w), 1.0)
            touches_bottom = float(ys.max()) >= (1.0 - bottom_margin_fraction) * max(float(image_h - 1), 1.0)
            # Width matters because the pipe spans the camera horizontally in
            # this task, but size is damped so large background components do
            # not dominate when their projected height is wrong.
            score = bbox_w * math.sqrt(float(count_i)) * height_score
            candidate = {
                "label": int(label),
                "count": count_i,
                "score": float(score),
                "bbox_w": float(bbox_w),
                "bbox_h": float(bbox_h),
                "height_score": float(height_score),
                "width_fraction": float(width_fraction),
                "touches_bottom": bool(touches_bottom),
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
            max_components = max(1, int(self.params["cylinder_component_max_components"]))
            nominal_radius = max(1e-6, float(self.params["pipe_radius_m"]))
            radius_margin = max(
                float(self.params["cylinder_component_radius_abs_margin_m"]),
                float(self.params["cylinder_component_radius_rel_margin"]) * nominal_radius,
            )
            fit_params = {
                "min_depth_m": float(self.params["min_depth_m"]),
                "max_depth_m": float(self.params["max_depth_m"]),
                "min_pipe_pixels": min_pixels,
                "max_fit_points": int(self.params["max_pca_points"]),
                "sample_stride": int(self.params["sample_stride"]),
                "consensus_iterations": int(self.params["cylinder_component_consensus_iterations"]),
                "radius_tolerance_m": float(self.params["cylinder_component_radius_tolerance_m"]),
                "min_inliers": int(self.params["cylinder_component_min_inliers"]),
                "min_inlier_fraction": float(self.params["cylinder_component_min_inlier_fraction"]),
                "radius_min_m": max(1e-4, nominal_radius - radius_margin),
                "radius_max_m": nominal_radius + radius_margin,
                "max_residual_m": float(self.params["cylinder_component_max_residual_m"]),
            }
            for candidate in candidates[:max_components]:
                label = int(candidate["label"])
                component_mask = labels == label
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
                residual_score = math.exp(-residual / max(float(self.params["cylinder_component_max_residual_m"]), 1e-6))
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

            lock_active = (
                bool(self.params["enable_pipe_temporal_lock"])
                and self.pipe_lock_model is not None
            )
            if lock_active:
                best_temporal: dict[str, Any] | None = None
                best_temporal_score = -float("inf")
                best_temporal_debug: dict[str, float] = {}
                best_rejected_score = -float("inf")
                best_rejected_debug: dict[str, float] = {}
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
                        "lock_axis_delta_deg": _round(best_temporal_debug.get("axis_delta_deg")),
                        "lock_radius_delta_m": _round(best_temporal_debug.get("radius_delta_m")),
                        "lock_stand_delta_m": _round(best_temporal_debug.get("stand_delta_m")),
                        "lock_axis_point_delta_m": _round(best_temporal_debug.get("axis_point_delta_m")),
                    }
                    return labels == selected_label

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
                    }
                    return np.zeros_like(mask, dtype=bool)

            if best_label > 0:
                self.pipe_component_selected_model = best_model
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
                }
                return labels == best_label

            if bool(self.params["pipe_component_require_valid_cylinder"]):
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
        rgb_image = PilImage.fromarray(rgb, mode="RGB")
        depth_image = PilImage.fromarray(depth.astype(np.float32), mode="F")
        mask_image = PilImage.fromarray((tracker.pipe_mask.astype(np.uint8) * 255), mode="L")
        angle_deg = tracker.image_axis_angle_deg
        rotated_rgb = rgb_image.rotate(angle_deg, resample=PilImage.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
        rotated_depth = np.asarray(
            depth_image.rotate(angle_deg, resample=PilImage.Resampling.BILINEAR, expand=False, fillcolor=np.nan),
            dtype=np.float64,
        )
        rotated_mask_img = mask_image.rotate(angle_deg, resample=PilImage.Resampling.NEAREST, expand=False, fillcolor=0)
        rotated_mask = np.asarray(rotated_mask_img) > 0

        ys, _ = np.nonzero(rotated_mask)
        if ys.size == 0:
            raise ValueError("Rotated pipe mask is empty")

        width, height = rotated_rgb.size
        vertical_margin = int(self.params["strip_vertical_margin_px"])
        x0, x1 = 0, width - 1
        y0 = max(0, int(ys.min()) - vertical_margin)
        y1 = min(height - 1, int(ys.max()) + vertical_margin)
        strip = rotated_rgb.crop((x0, y0, x1 + 1, y1 + 1))
        strip_mask = rotated_mask[y0:y1 + 1, x0:x1 + 1]

        strip_rgb = np.asarray(strip.convert("RGB"), dtype=np.float64) / 255.0
        gray = np.asarray(strip.convert("L"), dtype=np.float64) / 255.0
        profile, counts = self._column_profile(gray, strip_mask)
        background = _median_filter_1d(profile, int(self.params["background_window_px"]))
        residual = background - profile
        edge_profile, edge_support_profile = self._rgb_vertical_edge_profile(gray, strip_mask)

        strip_width = profile.size
        edge_margin = max(
            int(self.params["edge_margin_px"]),
            int(round(strip_width * float(self.params["edge_margin_fraction"]))),
        )
        min_count = max(8, int(float(self.params["min_valid_column_fraction"]) * strip.height))
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

        mode = str(self.params["junction_acceptance_mode"]).strip().lower()
        if mode not in ("rgb_temporal", "variant_a_rgb"):
            mode = "variant_a_rgb"

        # Variant A (deterministic RGB-only): run on the already-rotated, pipe-only
        # strip (pipe horizontal -> seam vertical -> pipe_axis_angle_deg=0).
        variant_a_result = None
        if mode == "variant_a_rgb" and _variant_a_detect_seam is not None:
            variant_a_result = _variant_a_detect_seam(
                np.asarray(strip.convert("RGB"), dtype=np.uint8),
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

        if mode == "variant_a_rgb" and variant_a_result is not None:
            candidate_x = self._variant_a_center_x_in_strip(variant_a_result, gray.shape)
        else:
            candidate_x = classical_candidate_x
        klt_prediction = self._klt_predict_junction_x(gray, strip_mask, residual, valid)
        if klt_prediction["available"] and self.junction_lock_active:
            candidate_x = int(klt_prediction["candidate_x"])
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
        pipe_end_rejected = self._pipe_end_rejected(strip_mask, candidate_x)
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
            accepted=accepted,
            edge_margin_px=edge_margin,
            crop_xyxy=[x0, y0, x1, y1],
            strip_size_wh=[strip.width, strip.height],
            strip_mask=strip_mask,
            rotated_mask=rotated_mask,
            rotation_deg=angle_deg,
            strip_profile=profile,
            strip_profile_valid=valid,
            appearance_veto=appearance_veto,
            appearance_ncc=appearance_ncc,
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
        residual: np.ndarray,
        valid: np.ndarray,
    ) -> dict[str, Any]:
        if not bool(self.params["enable_opencv_klt_tracking"]):
            return {"available": False, "reason": "disabled", "candidate_x": None}
        if cv2 is None:
            self.klt_status = f"opencv_unavailable:{type(CV2_IMPORT_ERROR).__name__}"
            return {"available": False, "reason": self.klt_status, "candidate_x": None}
        if gray.ndim != 2 or strip_mask.shape != gray.shape or residual.size != gray.shape[1]:
            return {"available": False, "reason": "invalid_strip", "candidate_x": None}
        if not self.junction_lock_active or self.junction_lock_x is None:
            self._klt_reset()
            return {"available": False, "reason": "not_locked", "candidate_x": None}

        gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        h, w = gray_u8.shape
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

        candidate_x = int(np.argmax(np.where(local_valid, residual, -np.inf)))
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

    def _pipe_end_rejected(self, strip_mask: np.ndarray, candidate_x: int) -> bool:
        width = strip_mask.shape[1]
        x = int(candidate_x)
        side_width = int(self.params["rgb_temporal_pipe_end_side_width_px"])
        left0 = max(0, x - side_width)
        left1 = max(0, x - 1)
        right0 = min(width - 1, x + 1)
        right1 = min(width - 1, x + side_width)
        # A real junction has FULL pipe support on BOTH sides. If the candidate is
        # within side_width of the strip/image border, that support band is clipped
        # by the frame (the junction is exiting / the pipe is cut off) -> treat it
        # as a pipe-end so acceptance stops BEFORE the junction leaves view, not
        # after (avoids latching onto clipped pipe edges during an exit).
        if (x - side_width) < 0 or (x + side_width) > (width - 1):
            return True
        left = strip_mask[:, left0:left1 + 1]
        right = strip_mask[:, right0:right1 + 1]
        left_cov = float(left.mean()) if left.size else 0.0
        right_cov = float(right.mean()) if right.size else 0.0
        min_cov = float(self.params["rgb_temporal_pipe_end_min_side_coverage"])
        max_delta = float(self.params["rgb_temporal_pipe_end_max_coverage_delta"])
        return bool(left_cov < min_cov or right_cov < min_cov or abs(left_cov - right_cov) > max_delta)

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
        profile = np.full(width, np.nan, dtype=np.float64)
        counts = mask.sum(axis=0)
        min_count = max(8, int(float(self.params["min_valid_column_fraction"]) * height))
        for x in range(width):
            if counts[x] >= min_count:
                profile[x] = float(np.median(gray[mask[:, x], x]))

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
            self.previous_geometry = current_geometry
            self.previous_candidate_x = seam.candidate_x_strip_px
        elif bool(self.params["reset_geometry_on_reject"]):
            self.previous_geometry = None
            self.previous_candidate_x = None

        reason_parts = []
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
        if lock_used:
            reason_parts.append(f"junction_track:{lock_reason}")
        if eligible:
            reason_parts.append("eligible")
        if self.state == "STOP_AND_LOCALIZE":
            reason_parts.append("stop_and_localize")

        return {
            "eligible": eligible,
            "jump_ok": jump_ok,
            "candidate_not_border": candidate_not_border,
            "geometry_consistent": geometry_ok,
            "reason": ";".join(reason_parts),
            "junction_lock_active": self.junction_lock_active,
            "junction_lock_used": lock_used,
            "junction_lock_reason": lock_reason,
            "junction_lock_x_strip_px": None if self.junction_lock_x is None else float(self.junction_lock_x),
            "junction_lock_velocity_px_per_frame": float(self.junction_lock_velocity_px),
            "junction_lock_streak": int(self.junction_lock_streak),
            "junction_lock_missed_frames": int(self.junction_lock_missed_frames),
            "junction_lock_confidence": float(self.junction_lock_confidence),
            "junction_lock_source": self.junction_lock_source,
        }

    def _update_junction_lock(self, seam: SeamResult) -> None:
        x = float(seam.candidate_x_strip_px)
        if self.junction_lock_x is None or not self.junction_lock_active:
            self.junction_lock_velocity_px = 0.0
            self.junction_lock_streak = 1
        else:
            measured_velocity = x - float(self.junction_lock_x)
            # Smooth velocity so camera/base motion can be followed without
            # jumping to one-frame outliers.
            self.junction_lock_velocity_px = 0.7 * self.junction_lock_velocity_px + 0.3 * measured_velocity
            self.junction_lock_streak += 1
        self.junction_lock_x = x
        self.junction_lock_active = True
        self.junction_lock_missed_frames = 0
        self.junction_lock_confidence = max(float(seam.confidence), float(self.params["junction_lock_min_confidence"]))
        self.junction_lock_source = "fresh_detection"

    def _release_junction_lock(self, reason: str) -> tuple[bool, str]:
        self.junction_lock_active = False
        self.junction_lock_x = None
        self.junction_lock_velocity_px = 0.0
        self.junction_lock_streak = 0
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
        return False, reason

    def _try_hold_junction_lock(self, seam: SeamResult, geometry_ok: bool) -> tuple[bool, str]:
        if not bool(self.params["enable_junction_lock"]):
            return False, "disabled"
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
            return False, "measurement_far_from_track"

        # The lock is a prior, not a hallucination: keep publishing only if the
        # current frame still contains seam-like evidence near the predicted
        # position. This lets the seam move with camera/robot motion while
        # avoiding a static pose in empty image space.
        visual_reacquire = bool(
            not seam.pipe_end_rejected
            and seam.rgb_line_width_px <= int(self.params["rgb_temporal_max_line_width_px"])
            and (
                float(seam.candidate_z_score) >= float(self.params["junction_lock_min_reacquire_z"])
                or float(seam.candidate_contrast) >= float(self.params["junction_lock_min_reacquire_contrast"])
            )
        )
        if not visual_reacquire:
            return False, "no_visual_reacquire"

        if not (seam.edge_margin_px <= measured_i < width - seam.edge_margin_px):
            return self._release_junction_lock("out_of_view")

        decay = float(self.params["junction_lock_confidence_decay"])
        self.junction_lock_confidence = max(
            self.junction_lock_confidence * decay,
            float(seam.confidence) * decay,
            float(self.params["junction_lock_min_confidence"]),
        )
        if self.junction_lock_confidence < float(self.params["junction_lock_min_confidence"]):
            return self._release_junction_lock("confidence_decayed")

        measured_velocity = float(measured_i) - float(self.junction_lock_x)
        self.junction_lock_velocity_px = 0.7 * self.junction_lock_velocity_px + 0.3 * measured_velocity
        self.junction_lock_x = float(measured_i)
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

        seam.candidate_x_strip_px = measured_i
        seam.candidate_x_rotated_px = int(seam.crop_xyxy[0] + measured_i)
        seam.confidence = float(self.junction_lock_confidence)
        seam.accepted = True
        seam.local_candidate_accepted = True
        seam.visual_frontend_accepted = True
        seam.rgb_dark_accepted = True
        seam.negative_gate_reason = "junction_lock_reacquired"
        return True, "reacquired_measurement"

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
        width = int(seam.strip_size_wh[0])
        x = int(seam.candidate_x_strip_px)
        return seam.edge_margin_px <= x < width - seam.edge_margin_px

    def _localize_confirmed_seam(
        self,
        depth: np.ndarray,
        k: np.ndarray,
        tracker: TrackerResult,
        seam: SeamResult,
    ) -> LocalizationResult:
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
            return LocalizationResult(None, None, 0)

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
            return LocalizationResult(None, None, int(xs.size))

        unique_uv = np.unique(np.stack([xs, ys], axis=1), axis=0)
        unique_uv = self._coherent_depth_support_uv(unique_uv, depth)
        if unique_uv.shape[0] < int(self.params["min_surface_points"]):
            return LocalizationResult(None, None, int(unique_uv.shape[0]))
        points_camera = _backproject(depth, unique_uv[:, 0], unique_uv[:, 1], k)
        # Use a real point close to the median support instead of publishing a
        # component-wise median that may combine y from one depth component and z
        # from another. That was fragile when the seam column crossed both pipe
        # surface and background/pipe-end depth.
        median_point = np.median(points_camera, axis=0)
        surface_center = points_camera[int(np.argmin(np.linalg.norm(points_camera - median_point, axis=1)))]

        view_direction = _normalize(surface_center)
        axis = _normalize(tracker.pipe_axis_xyz)
        radial_direction = view_direction - float(np.dot(view_direction, axis)) * axis
        radial_direction = _normalize(radial_direction)
        pipe_center = surface_center + float(self.params["pipe_radius_m"]) * radial_direction
        return LocalizationResult(surface_center, pipe_center, int(unique_uv.shape[0]))

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
            "junction_lock_velocity_px_per_frame": _round(state_info.get("junction_lock_velocity_px_per_frame")),
            "junction_lock_streak": int(state_info.get("junction_lock_streak", 0)),
            "junction_lock_missed_frames": int(state_info.get("junction_lock_missed_frames", 0)),
            "junction_lock_confidence": _round(state_info.get("junction_lock_confidence")),
            "junction_lock_source": state_info.get("junction_lock_source"),
            "confidence": _round(seam.confidence),
            "junction_acceptance_mode": seam.junction_acceptance_mode,
            "variant_a_orientation_deg": _round(seam.variant_a_orientation_deg),
            "visual_frontend": seam.visual_frontend,
            "visual_frontend_accepted": bool(seam.visual_frontend_accepted),
            "candidate_x_strip_px": int(seam.candidate_x_strip_px),
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
            "classical_candidate_x_strip_px": int(seam.classical_candidate_x_strip_px),
            "classical_candidate_contrast": _round(seam.classical_candidate_contrast),
            "classical_candidate_z_score": _round(seam.classical_candidate_z_score),
            "rgb_dark_score": _round(seam.rgb_dark_score),
            "rgb_local_contrast_score": _round(seam.rgb_local_contrast_score),
            "rgb_dark_threshold_used": _round(seam.rgb_dark_threshold_used),
            # A raw seam acceptance that the state machine flagged as a large
            # candidate jump (hop to another line) must NOT be surfaced as accepted
            # nor propagated downstream: the jump guard already refused to lock it.
            "detector_accepted": bool(seam.accepted and state_info.get("jump_ok", True)),
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
            "pipe_lock_active": bool(self.pipe_lock_model is not None),
            "pipe_lock_missed_frames": int(self.pipe_lock_missed_frames),
            "pipe_lock_source": self.pipe_lock_source,
            "pipe_lock_selection_score": self.pipe_component_selection_info.get("score"),
            "pipe_lock_axis_delta_deg": self.pipe_component_selection_info.get("lock_axis_delta_deg"),
            "pipe_lock_radius_delta_m": self.pipe_component_selection_info.get("lock_radius_delta_m"),
            "pipe_lock_stand_delta_m": self.pipe_component_selection_info.get("lock_stand_delta_m"),
            "pipe_lock_axis_point_delta_m": self.pipe_component_selection_info.get("lock_axis_point_delta_m"),
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
    ) -> np.ndarray:
        image = PilImage.fromarray(rgb, mode="RGB").convert("RGBA")
        overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        overlay[tracker.pipe_mask] = (20, 190, 90, 70)
        image = PilImage.alpha_composite(image, PilImage.fromarray(overlay, mode="RGBA"))
        draw = ImageDraw.Draw(image)

        if tracker.image_line_segment_uv is not None:
            p0, p1 = tracker.image_line_segment_uv
            draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(255, 40, 40, 255), width=3)

        u, v = tracker.image_centroid_uv
        draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(40, 120, 255, 255), width=3)

        line_uv = self._candidate_line_original_uv(seam)
        if seam.accepted:
            color = (0, 220, 255, 255)
            label = "JUNCTION"
            width = 4
        elif seam.local_candidate_accepted:
            color = (255, 210, 0, 255)
            label = "VERIFY"
            width = 3
        else:
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
            "cylinder_component_radius_abs_margin_m": 0.08,
            "cylinder_component_radius_rel_margin": 0.75,
            "cylinder_component_max_residual_m": 0.05,
            "pipe_component_require_valid_cylinder": True,
            "pipe_component_shape_prior_enabled": False,
            "pipe_component_min_width_fraction": 0.28,
            "pipe_component_bottom_margin_fraction": 0.05,
            "pipe_component_bottom_allow_width_fraction": 0.55,
            "enable_pipe_temporal_lock": True,
            "pipe_lock_reject_global_when_locked": True,
            "pipe_lock_release_on_missed": False,
            "pipe_lock_max_missed_frames": 6,
            "pipe_lock_radius_abs_margin_m": 0.09,
            "pipe_lock_radius_rel_margin": 0.80,
            "pipe_lock_max_radius_delta_m": 0.10,
            "pipe_lock_max_axis_delta_deg": 50.0,
            "pipe_lock_max_standoff_delta_m": 0.75,
            "pipe_lock_max_axis_point_delta_m": 0.90,
            "pipe_lock_min_inlier_fraction": 0.20,
            "pipe_lock_max_residual_m": 0.07,
            "pipe_lock_min_compatibility_score": 0.005,
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
            "enable_junction_lock": True,
            "junction_lock_max_missed_frames": 12,
            "junction_lock_search_radius_px": 180,
            "junction_lock_min_reacquire_z": 3.0,
            "junction_lock_min_reacquire_contrast": 0.006,
            "junction_lock_confidence_decay": 0.96,
            "junction_lock_min_confidence": 0.25,
            "enable_opencv_klt_tracking": True,
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
            "rgb_temporal_pipe_end_side_width_px": 32,
            "rgb_temporal_pipe_end_min_side_coverage": 0.18,
            "rgb_temporal_pipe_end_max_coverage_delta": 0.65,
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
            "klt_status": status.get("klt_status"),
            "klt_points": status.get("klt_points"),
            "klt_dx_px": status.get("klt_dx_px"),
            "rgb_dark_accepted": status.get("rgb_dark_accepted"),
            "depth_gap_accepted": status.get("depth_gap_accepted"),
            "pipe_component_selection_method": status.get("pipe_component_selection_method"),
            "pipe_component_count": status.get("pipe_component_count"),
            "pipe_component_candidate_count": status.get("pipe_component_candidate_count"),
            "pipe_component_rejected_by_shape": status.get("pipe_component_rejected_by_shape"),
            "pipe_component_cylinder_evaluated": status.get("pipe_component_cylinder_evaluated"),
            "pipe_component_cylinder_valid": status.get("pipe_component_cylinder_valid"),
            "pipe_component_selected_label": status.get("pipe_component_selected_label"),
            "pipe_component_fallback_label": status.get("pipe_component_fallback_label"),
            "pipe_lock_active": status.get("pipe_lock_active"),
            "pipe_lock_missed_frames": status.get("pipe_lock_missed_frames"),
            "pipe_lock_source": status.get("pipe_lock_source"),
            "pipe_lock_selection_score": status.get("pipe_lock_selection_score"),
            "pipe_lock_axis_delta_deg": status.get("pipe_lock_axis_delta_deg"),
            "pipe_lock_radius_delta_m": status.get("pipe_lock_radius_delta_m"),
            "pipe_lock_stand_delta_m": status.get("pipe_lock_stand_delta_m"),
            "pipe_lock_axis_point_delta_m": status.get("pipe_lock_axis_point_delta_m"),
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
