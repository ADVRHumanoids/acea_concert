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
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PilImage
from PIL import ImageDraw

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
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
        def publish(self, *a: Any, **k: Any) -> None: ...
        def get_logger(self) -> "_StubLogger": return _StubLogger()

    class ExternalShutdownException(Exception):  # type: ignore
        pass

    class _StubAny:
        def __call__(self, *a: Any, **k: Any) -> None: return None
        def __getattr__(self, _name: str) -> int: return 0

    HistoryPolicy = ReliabilityPolicy = QoSProfile = _StubAny()  # type: ignore
    CameraInfo = Image = String = PoseStamped = Marker = MarkerArray = None  # type: ignore

# Variant A deterministic RGB-only seam detector (black top-hat + vertical-run
# coherence, no depth). Optional frontend selected by junction_acceptance_mode
# == "variant_a_rgb". Imported from the same scripts dir (project convention).
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from acea_seam_detector import detect_seam as _variant_a_detect_seam
except Exception:  # pragma: no cover - Variant A is optional
    _variant_a_detect_seam = None

# Weld-seam / gap-plane frame geometry (pure NumPy, ROS-free, unit-tested in
# acea_alignment/weld_seam.py). Turns (pipe axis, seam surface point) into a
# PoseStamped-ready frame for the IK / welding-tracking side (Arturo's request).
try:
    from acea_alignment.weld_seam import WeldSeamFrame, seam_frame_from_axis_and_surface
except Exception:  # pragma: no cover - geometry helper is optional at import time
    WeldSeamFrame = None  # type: ignore
    seam_frame_from_axis_and_surface = None  # type: ignore

# RViz marker constants (visualization_msgs/Marker). Defined as literals so the
# values are available even when the message class is unavailable offline
# (Marker == None). They match the ROS message definition.
_MARKER_ADD = 0
_MARKER_DELETEALL = 3
_MARKER_CYLINDER = 3


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
    original_x = cos_t * x + sin_t * y + cx
    original_y = -sin_t * x + cos_t * y + cy
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
class YoloSegCandidate:
    enabled: bool = False
    available: bool = False
    detected: bool = False
    accepted: bool = False
    confidence: float | None = None
    mask_area_px: int = 0
    mask_area_fraction: float = 0.0
    strip_overlap_px: int = 0
    candidate_x_strip_px: int | None = None
    candidate_x_rotated_px: int | None = None
    candidate_x_image_px: int | None = None
    class_name: str | None = None
    reason: str = "disabled"
    mask: np.ndarray | None = None


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
    weak_rgb_depth_supported: bool
    rgb_dark_accepted: bool
    depth_gap_score: float
    depth_gap_accepted: bool
    depth_gap_raw_accepted: bool
    depth_gap_score_plausible: bool
    depth_gap_threshold_used_m: float
    min_depth_gap_score_m: float
    max_depth_gap_score_m: float
    depth_gap_depth_jump_m: float
    depth_gap_coverage_drop: float
    yolo_weak_depth_gap_supported: bool
    yolo_weak_depth_gap_threshold_m: float
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
    yolo_seg_enabled: bool
    yolo_seg_available: bool
    yolo_seg_detected: bool
    yolo_seg_confidence: float | None
    yolo_seg_mask_area_px: int
    yolo_seg_mask_area_fraction: float
    yolo_seg_strip_overlap_px: int
    yolo_seg_candidate_x_strip_px: int | None
    yolo_seg_candidate_x_rotated_px: int | None
    yolo_seg_candidate_x_image_px: int | None
    yolo_seg_class_name: str | None
    yolo_seg_reason: str
    yolo_seg_mask: np.ndarray | None
    junction_acceptance_mode: str
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
    pipe_end_rejected: bool
    depth_gap_used_for_acceptance: bool
    accepted: bool
    edge_margin_px: int
    crop_xyxy: list[int]
    strip_size_wh: list[int]
    strip_mask: np.ndarray
    rotated_mask: np.ndarray
    rotation_deg: float
    strip_profile: np.ndarray
    strip_profile_valid: np.ndarray


@dataclass
class LocalizationResult:
    visible_surface_center_xyz_m: np.ndarray | None
    pipe_center_estimate_xyz_m: np.ndarray | None
    support_pixel_count: int


class YoloSegFrontend:
    """Lazy optional Ultralytics YOLO-seg adapter.

    The adapter is deliberately defensive: if Ultralytics or the model cannot be
    loaded, the detector keeps running and reports a rejected YOLO candidate
    instead of crashing the ROS node.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.model: Any | None = None
        self.yolo_class: Any | None = None
        self.load_error: str | None = None
        self.device_override: str | None = None

    def _candidate_site_packages(self) -> list[Path]:
        configured = str(self.params.get("yolo_python_site_packages", "")).strip()
        candidates: list[Path] = []
        if configured:
            candidates.append(Path(configured))

        common_roots = [
            Path.cwd(),
            Path("/workspace/iit-concert-ros-pkg"),
            Path("/home/user/xbot2_ws/src/iit-concert-ros-pkg"),
            Path(__file__).resolve().parents[2],
        ]
        seen: set[str] = set()
        for root in common_roots:
            for path in sorted((root / ".venv-yolo" / "lib").glob("python*/site-packages")):
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    candidates.append(path)
        return candidates

    def _prepare_pythonpath(self) -> None:
        config_dir_text = str(self.params.get("yolo_config_dir", "")).strip() or "/tmp/acea_yolo_ultralytics"
        config_dir = Path(config_dir_text)
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
        except OSError:
            pass
        for path in self._candidate_site_packages():
            if path.exists():
                text = str(path)
                if text not in sys.path:
                    sys.path.insert(0, text)

    def _resolve_model_path(self) -> Path | None:
        model_text = str(self.params.get("yolo_model_path", "")).strip()
        if not model_text:
            return None
        path = Path(model_text)
        if path.is_absolute() and path.exists():
            return path
        candidates = [
            Path.cwd() / path,
            Path("/workspace/iit-concert-ros-pkg") / path,
            Path("/home/user/xbot2_ws/src/iit-concert-ros-pkg") / path,
            Path(__file__).resolve().parents[2] / path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if self.load_error is not None:
            return False
        model_path = self._resolve_model_path()
        if model_path is None:
            self.load_error = "model_path_empty"
            return False
        if not model_path.exists():
            self.load_error = f"model_not_found:{model_path}"
            return False
        try:
            self._prepare_pythonpath()
            from ultralytics import YOLO  # type: ignore

            self.yolo_class = YOLO
            self.model = YOLO(str(model_path))
            return True
        except Exception as exc:  # pragma: no cover - depends on local YOLO env
            self.load_error = f"{type(exc).__name__}:{exc}"
            return False

    def _device_arg(self) -> str:
        if self.device_override is not None:
            return self.device_override
        return str(self.params.get("yolo_device", "")).strip()

    def _predict_once(self, rgb: np.ndarray) -> Any:
        assert self.model is not None
        kwargs: dict[str, Any] = {
            "source": PilImage.fromarray(rgb, mode="RGB"),
            "imgsz": int(self.params["yolo_imgsz"]),
            "conf": float(self.params["yolo_conf_threshold"]),
            "verbose": False,
        }
        device = self._device_arg()
        if device:
            kwargs["device"] = device
        return self.model.predict(**kwargs)

    def predict(self, rgb: np.ndarray) -> YoloSegCandidate:
        if not bool(self.params.get("use_yolo_seg_frontend", False)):
            return YoloSegCandidate(enabled=False, reason="disabled")
        if not self._ensure_model():
            return YoloSegCandidate(enabled=True, available=False, reason=f"unavailable:{self.load_error}")

        assert self.model is not None
        try:
            results = self._predict_once(rgb)
        except Exception as exc:  # pragma: no cover - depends on local YOLO env
            error_text = f"{type(exc).__name__}:{exc}"
            if bool(self.params.get("yolo_allow_cpu_fallback", True)) and "Invalid CUDA" in error_text:
                self.device_override = "cpu"
                try:
                    results = self._predict_once(rgb)
                except Exception as retry_exc:  # pragma: no cover - depends on local YOLO env
                    retry_text = f"{type(retry_exc).__name__}:{retry_exc}"
                    return YoloSegCandidate(
                        enabled=True,
                        available=True,
                        reason=f"inference_error:{error_text};cpu_fallback_error:{retry_text}",
                    )
            else:
                return YoloSegCandidate(enabled=True, available=True, reason=f"inference_error:{error_text}")

        height, width = rgb.shape[:2]
        min_area = int(self.params["yolo_min_mask_area_px"])
        max_area_fraction = float(self.params["yolo_max_mask_area_fraction"])
        class_filter = str(self.params.get("yolo_class_name", "")).strip()
        best: YoloSegCandidate | None = None
        for result in results:
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            if boxes is None or masks is None or getattr(masks, "data", None) is None:
                continue
            cls_values = [] if getattr(boxes, "cls", None) is None else boxes.cls.cpu().numpy().tolist()
            conf_values = [] if getattr(boxes, "conf", None) is None else boxes.conf.cpu().numpy().tolist()
            mask_values = masks.data.cpu().numpy()
            names = {int(k): str(v) for k, v in getattr(self.model, "names", {}).items()}
            for idx, mask_float in enumerate(mask_values):
                class_id = int(cls_values[idx]) if idx < len(cls_values) else 0
                class_name = names.get(class_id, str(class_id))
                if class_filter and class_name != class_filter:
                    continue
                mask = np.asarray(mask_float, dtype=np.float32)
                if mask.shape != (height, width):
                    mask_img = PilImage.fromarray((mask > 0.5).astype(np.uint8) * 255, mode="L")
                    mask_img = mask_img.resize((width, height), resample=PilImage.Resampling.NEAREST)
                    mask_bool = np.asarray(mask_img) > 0
                else:
                    mask_bool = mask > 0.5
                area = int(mask_bool.sum())
                area_fraction = float(area) / max(float(width * height), 1.0)
                if area < min_area:
                    continue
                if max_area_fraction > 0.0 and area_fraction > max_area_fraction:
                    continue
                ys, xs = np.nonzero(mask_bool)
                if xs.size == 0:
                    continue
                confidence = float(conf_values[idx]) if idx < len(conf_values) else 0.0
                candidate = YoloSegCandidate(
                    enabled=True,
                    available=True,
                    detected=True,
                    accepted=True,
                    confidence=confidence,
                    mask_area_px=area,
                    mask_area_fraction=area_fraction,
                    candidate_x_image_px=int(np.median(xs)),
                    class_name=class_name,
                    reason="mask_detected",
                    mask=mask_bool,
                )
                if best is None or (candidate.confidence or 0.0) > (best.confidence or 0.0):
                    best = candidate

        if best is None:
            return YoloSegCandidate(enabled=True, available=True, reason="no_valid_mask")
        return best


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
        self.yolo_frontend = YoloSegFrontend(params)

    def process(self, rgb: np.ndarray, depth: np.ndarray, k: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        self.processed_frame_count += 1
        tracker = self._track_pipe(depth, k)
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

    def _track_pipe(self, depth: np.ndarray, k: np.ndarray) -> TrackerResult:
        min_depth = float(self.params["min_depth_m"])
        max_depth = float(self.params["max_depth_m"])
        valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
        threshold_info = _depth_threshold_kmeans(depth, valid, int(self.params["sample_stride"]))
        pipe_mask = valid & (depth <= float(threshold_info["threshold_m"]))
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

        return TrackerResult(
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

    def _variant_a_params(self) -> dict[str, Any]:
        """Map node params -> acea_seam_detector.detect_seam params (rest = its defaults)."""
        return {
            "tophat_se_len_px": int(self.params["variant_a_tophat_se_len_px"]),
            "min_vertical_run_px": int(self.params["variant_a_min_vertical_run_px"]),
            "min_significance_z": float(self.params["variant_a_min_significance_z"]),
            "max_seam_width_px": float(self.params["variant_a_max_seam_width_px"]),
            "border_margin_px": int(self.params["variant_a_border_margin_px"]),
        }

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
        if mode not in ("rgb_depth", "rgb_temporal", "variant_a_rgb"):
            mode = "rgb_depth"

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

        yolo_candidate = self._yolo_candidate_for_strip(
            rgb,
            angle_deg,
            [x0, y0, x1, y1],
            strip_mask,
            valid,
            residual,
            rotated_depth,
            rotated_mask,
        )
        use_yolo = bool(self.params["use_yolo_seg_frontend"])
        if mode == "variant_a_rgb" and variant_a_result is not None:
            candidate_x = int(np.clip(int(variant_a_result.x_px), 0, strip_width - 1))
        elif use_yolo and yolo_candidate.candidate_x_strip_px is not None:
            candidate_x = int(yolo_candidate.candidate_x_strip_px)
            candidate_x = int(np.clip(candidate_x, 0, strip_width - 1))
        else:
            candidate_x = classical_candidate_x

        contrast = float(max(0.0, residual[candidate_x]))
        min_dark = float(self.params["min_dark_contrast"])
        strong_dark = float(self.params["strong_dark_contrast"])
        rgb_dark_score = float(np.clip((contrast - min_dark) / max(strong_dark - min_dark, 1e-6), 0.0, 1.0))

        z_score = (float(residual[candidate_x]) - residual_median) / robust_sigma
        classical_z_score = (float(residual[classical_candidate_x]) - residual_median) / robust_sigma
        min_local_z = float(self.params["weak_rgb_min_z_score"])
        strong_local_z = float(self.params["weak_rgb_strong_z_score"])
        rgb_local_contrast_score = float(
            np.clip((z_score - min_local_z) / max(strong_local_z - min_local_z, 1e-6), 0.0, 1.0)
        )
        depth_gap = self._depth_gap_evidence(rotated_depth, rotated_mask, [x0, y0, x1, y1], candidate_x)
        standard_depth_gap_threshold = float(self.params["min_depth_gap_m"])
        depth_gap_raw_accepted = bool(
            depth_gap["depth_jump_m"] >= standard_depth_gap_threshold
            or depth_gap["coverage_drop"] >= float(self.params["min_depth_gap_coverage_drop"])
        )
        yolo_weak_depth_gap_threshold = float(self.params["yolo_weak_depth_gap_min_score_m"])
        yolo_confidence_for_gate = 0.0 if yolo_candidate.confidence is None else float(yolo_candidate.confidence)
        yolo_weak_depth_gap_supported = bool(
            use_yolo
            and bool(self.params["enable_yolo_weak_depth_gap_support"])
            and bool(yolo_candidate.accepted)
            and yolo_confidence_for_gate >= float(self.params["yolo_weak_depth_gap_min_confidence"])
            and float(depth_gap["score"]) >= yolo_weak_depth_gap_threshold
        )
        if yolo_weak_depth_gap_supported:
            depth_gap_raw_accepted = True
        depth_gap_threshold_used = yolo_weak_depth_gap_threshold if yolo_weak_depth_gap_supported else standard_depth_gap_threshold
        max_depth_gap_score = float(self.params["max_depth_gap_score_m"])
        min_depth_gap_score = float(self.params["min_depth_gap_score_m"])
        depth_gap_score_plausible = bool(
            (min_depth_gap_score <= 0.0 or float(depth_gap["score"]) >= min_depth_gap_score)
            and (max_depth_gap_score <= 0.0 or float(depth_gap["score"]) <= max_depth_gap_score)
        )
        depth_gap_accepted = depth_gap_raw_accepted and depth_gap_score_plausible
        if not bool(self.params["use_depth_gap_gate"]):
            depth_gap_raw_accepted = True
            depth_gap_score_plausible = True
            depth_gap_accepted = True
            depth_gap_threshold_used = 0.0
        weak_rgb_depth_supported = bool(
            self.params["enable_weak_rgb_depth_support"]
            and bool(self.params["use_depth_gap_gate"])
            and depth_gap_accepted
            and contrast >= float(self.params["weak_rgb_min_dark_contrast"])
            and z_score >= min_local_z
        )
        confidence = rgb_dark_score
        if weak_rgb_depth_supported:
            confidence = max(confidence, rgb_local_contrast_score * float(self.params["weak_rgb_confidence_scale"]))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        rgb_dark_threshold_used = min_dark
        if weak_rgb_depth_supported and rgb_dark_score < float(self.params["accept_confidence"]):
            rgb_dark_threshold_used = float(self.params["weak_rgb_min_dark_contrast"])
        rgb_dark_accepted = bool(
            rgb_dark_score >= float(self.params["accept_confidence"]) or weak_rgb_depth_supported
        )

        visual_frontend = "yolo_seg_mask" if use_yolo else "classical_rgb_dark"
        visual_frontend_accepted = bool(yolo_candidate.accepted) if use_yolo else rgb_dark_accepted
        if use_yolo:
            yolo_confidence = 0.0 if yolo_candidate.confidence is None else float(yolo_candidate.confidence)
            confidence = float(np.clip(yolo_confidence, 0.0, 1.0))
            if weak_rgb_depth_supported:
                confidence = max(confidence, rgb_local_contrast_score * float(self.params["weak_rgb_confidence_scale"]))

        if use_yolo and yolo_weak_depth_gap_supported:
            success_reason = "yolo_seg_and_weak_depth_gap"
        else:
            success_reason = "yolo_seg_and_depth_gap" if use_yolo else "rgb_and_depth_gap"
        visual_reject_reason = (
            "yolo_seg_rejected" if not use_yolo else f"yolo_seg_rejected:{yolo_candidate.reason}"
        )
        if not use_yolo:
            visual_reject_reason = "rgb_dark_rejected"

        negative_gate_reason = success_reason
        if not visual_frontend_accepted and not depth_gap_accepted:
            negative_gate_reason = f"{visual_reject_reason};depth_gap_rejected"
        elif not visual_frontend_accepted:
            negative_gate_reason = visual_reject_reason
        elif not depth_gap_accepted:
            if depth_gap_raw_accepted and not depth_gap_score_plausible:
                if min_depth_gap_score > 0.0 and float(depth_gap["score"]) < min_depth_gap_score:
                    negative_gate_reason = "depth_gap_too_small"
                else:
                    negative_gate_reason = "depth_gap_too_large"
            else:
                negative_gate_reason = "depth_gap_rejected"
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

        depth_gap_used_for_acceptance = bool(mode == "rgb_depth" and self.params["use_depth_gap_gate"])
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
        elif mode == "variant_a_rgb":
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
        else:
            local_candidate_accepted = visual_frontend_accepted and depth_gap_accepted
            accepted = local_candidate_accepted and (not temporal_gate_enabled or temporal_change.accepted)
            if local_candidate_accepted and temporal_gate_enabled and not temporal_change.accepted:
                if negative_gate_reason == success_reason:
                    negative_gate_reason = f"temporal_change_rejected:{temporal_change.reason}"
                else:
                    negative_gate_reason = f"{negative_gate_reason};temporal_change_rejected:{temporal_change.reason}"

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
            weak_rgb_depth_supported=weak_rgb_depth_supported,
            rgb_dark_accepted=rgb_dark_accepted,
            depth_gap_score=float(depth_gap["score"]),
            depth_gap_accepted=depth_gap_accepted,
            depth_gap_raw_accepted=depth_gap_raw_accepted,
            depth_gap_score_plausible=depth_gap_score_plausible,
            depth_gap_threshold_used_m=depth_gap_threshold_used,
            min_depth_gap_score_m=min_depth_gap_score,
            max_depth_gap_score_m=max_depth_gap_score,
            depth_gap_depth_jump_m=float(depth_gap["depth_jump_m"]),
            depth_gap_coverage_drop=float(depth_gap["coverage_drop"]),
            yolo_weak_depth_gap_supported=yolo_weak_depth_gap_supported,
            yolo_weak_depth_gap_threshold_m=yolo_weak_depth_gap_threshold,
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
            yolo_seg_enabled=bool(yolo_candidate.enabled),
            yolo_seg_available=bool(yolo_candidate.available),
            yolo_seg_detected=bool(yolo_candidate.detected),
            yolo_seg_confidence=yolo_candidate.confidence,
            yolo_seg_mask_area_px=int(yolo_candidate.mask_area_px),
            yolo_seg_mask_area_fraction=float(yolo_candidate.mask_area_fraction),
            yolo_seg_strip_overlap_px=int(yolo_candidate.strip_overlap_px),
            yolo_seg_candidate_x_strip_px=yolo_candidate.candidate_x_strip_px,
            yolo_seg_candidate_x_rotated_px=yolo_candidate.candidate_x_rotated_px,
            yolo_seg_candidate_x_image_px=yolo_candidate.candidate_x_image_px,
            yolo_seg_class_name=yolo_candidate.class_name,
            yolo_seg_reason=yolo_candidate.reason,
            yolo_seg_mask=yolo_candidate.mask,
            junction_acceptance_mode=mode,
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
            pipe_end_rejected=pipe_end_rejected,
            depth_gap_used_for_acceptance=depth_gap_used_for_acceptance,
            accepted=accepted,
            edge_margin_px=edge_margin,
            crop_xyxy=[x0, y0, x1, y1],
            strip_size_wh=[strip.width, strip.height],
            strip_mask=strip_mask,
            rotated_mask=rotated_mask,
            rotation_deg=angle_deg,
            strip_profile=profile,
            strip_profile_valid=valid,
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
        if left1 < left0 or right1 < right0:
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

    def _yolo_candidate_for_strip(
        self,
        rgb: np.ndarray,
        angle_deg: float,
        crop_xyxy: list[int],
        strip_mask: np.ndarray,
        valid_columns: np.ndarray,
        residual: np.ndarray,
        rotated_depth: np.ndarray,
        rotated_mask: np.ndarray,
    ) -> YoloSegCandidate:
        candidate = self.yolo_frontend.predict(rgb)
        if not candidate.enabled or candidate.mask is None:
            return candidate

        x0, y0, x1, y1 = [int(v) for v in crop_xyxy]
        mask_image = PilImage.fromarray((candidate.mask.astype(np.uint8) * 255), mode="L")
        rotated_mask_img = mask_image.rotate(angle_deg, resample=PilImage.Resampling.NEAREST, expand=False, fillcolor=0)
        rotated_mask = np.asarray(rotated_mask_img) > 0
        mask_strip = rotated_mask[y0:y1 + 1, x0:x1 + 1] & strip_mask
        ys, xs = np.nonzero(mask_strip)
        candidate.strip_overlap_px = int(xs.size)

        if xs.size < int(self.params["yolo_min_strip_overlap_px"]):
            candidate.accepted = False
            candidate.candidate_x_strip_px = None
            candidate.candidate_x_rotated_px = None
            if candidate.detected:
                candidate.reason = "mask_not_on_pipe_strip"
            return candidate

        unique_x = np.unique(xs)
        valid_unique_x = unique_x[valid_columns[unique_x]] if unique_x.size else unique_x
        if valid_unique_x.size == 0:
            valid_unique_x = unique_x

        best_x = int(np.median(valid_unique_x))
        best_depth_score = -1.0
        best_residual = -1.0
        for column in valid_unique_x:
            depth_gap = self._depth_gap_evidence(rotated_depth, rotated_mask, crop_xyxy, int(column))
            depth_score = float(depth_gap["score"])
            column_residual = float(max(0.0, residual[int(column)])) if np.isfinite(residual[int(column)]) else 0.0
            if depth_score > best_depth_score + 1e-12 or (
                abs(depth_score - best_depth_score) <= 1e-12 and column_residual > best_residual
            ):
                best_x = int(column)
                best_depth_score = depth_score
                best_residual = column_residual

        candidate.candidate_x_strip_px = best_x
        candidate.candidate_x_rotated_px = int(x0 + candidate.candidate_x_strip_px)
        candidate.accepted = bool(candidate.detected)
        if candidate.accepted:
            candidate.reason = "mask_on_pipe_strip;column_selected_by_depth_gap"
        return candidate

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

        eligible = seam.accepted and confidence_ok and candidate_not_border and geometry_ok and jump_ok

        if eligible:
            self.candidate_streak += 1
            if self.candidate_streak >= int(self.params["min_confirm_frames"]):
                self.state = "STOP_AND_LOCALIZE"
                if self.confirmed_frame_count is None:
                    self.confirmed_frame_count = self.processed_frame_count
            else:
                self.state = "CANDIDATE"
        else:
            self.candidate_streak = 0
            self.state = "SCAN"
            self.confirmed_frame_count = None

        if eligible:
            self.previous_geometry = current_geometry
            self.previous_candidate_x = seam.candidate_x_strip_px
        elif bool(self.params["reset_geometry_on_reject"]):
            self.previous_geometry = None
            self.previous_candidate_x = None

        reason_parts = []
        if not seam.accepted:
            reason_parts.append("detector_rejected")
        if not seam.visual_frontend_accepted:
            if seam.visual_frontend == "yolo_seg_mask":
                reason_parts.append(f"yolo_seg_rejected:{seam.yolo_seg_reason}")
            else:
                reason_parts.append("rgb_dark_rejected")
        if seam.depth_gap_used_for_acceptance and not seam.depth_gap_accepted:
            reason_parts.append("depth_gap_rejected")
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
        if eligible:
            reason_parts.append("eligible")
        if self.state == "STOP_AND_LOCALIZE":
            reason_parts.append("stop_and_localize")

        return {
            "eligible": eligible,
            "candidate_not_border": candidate_not_border,
            "geometry_consistent": geometry_ok,
            "reason": ";".join(reason_parts),
        }

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
        points_camera = _backproject(depth, unique_uv[:, 0], unique_uv[:, 1], k)
        surface_center = np.median(points_camera, axis=0)

        view_direction = _normalize(surface_center)
        axis = _normalize(tracker.pipe_axis_xyz)
        radial_direction = view_direction - float(np.dot(view_direction, axis)) * axis
        radial_direction = _normalize(radial_direction)
        pipe_center = surface_center + float(self.params["pipe_radius_m"]) * radial_direction
        return LocalizationResult(surface_center, pipe_center, int(unique_uv.shape[0]))

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
            "confidence": _round(seam.confidence),
            "junction_acceptance_mode": seam.junction_acceptance_mode,
            "visual_frontend": seam.visual_frontend,
            "visual_frontend_accepted": bool(seam.visual_frontend_accepted),
            "candidate_x_strip_px": int(seam.candidate_x_strip_px),
            "candidate_contrast": _round(seam.candidate_contrast),
            "candidate_z_score": _round(seam.candidate_z_score),
            "classical_candidate_x_strip_px": int(seam.classical_candidate_x_strip_px),
            "classical_candidate_contrast": _round(seam.classical_candidate_contrast),
            "classical_candidate_z_score": _round(seam.classical_candidate_z_score),
            "rgb_dark_score": _round(seam.rgb_dark_score),
            "rgb_local_contrast_score": _round(seam.rgb_local_contrast_score),
            "rgb_dark_threshold_used": _round(seam.rgb_dark_threshold_used),
            "weak_rgb_depth_supported": bool(seam.weak_rgb_depth_supported),
            "detector_accepted": bool(seam.accepted),
            "rgb_dark_accepted": bool(seam.rgb_dark_accepted),
            "yolo_seg_enabled": bool(seam.yolo_seg_enabled),
            "yolo_seg_available": bool(seam.yolo_seg_available),
            "yolo_seg_detected": bool(seam.yolo_seg_detected),
            "yolo_seg_confidence": _round(seam.yolo_seg_confidence),
            "yolo_seg_mask_area_px": int(seam.yolo_seg_mask_area_px),
            "yolo_seg_mask_area_fraction": _round(seam.yolo_seg_mask_area_fraction),
            "yolo_seg_strip_overlap_px": int(seam.yolo_seg_strip_overlap_px),
            "yolo_seg_candidate_x_strip_px": None
            if seam.yolo_seg_candidate_x_strip_px is None
            else int(seam.yolo_seg_candidate_x_strip_px),
            "yolo_seg_candidate_x_rotated_px": None
            if seam.yolo_seg_candidate_x_rotated_px is None
            else int(seam.yolo_seg_candidate_x_rotated_px),
            "yolo_seg_candidate_x_image_px": None
            if seam.yolo_seg_candidate_x_image_px is None
            else int(seam.yolo_seg_candidate_x_image_px),
            "yolo_seg_class_name": seam.yolo_seg_class_name,
            "yolo_seg_reason": seam.yolo_seg_reason,
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
            "pipe_end_rejected": bool(seam.pipe_end_rejected),
            "depth_gap_used_for_acceptance": bool(seam.depth_gap_used_for_acceptance),
            "depth_gap_score": _round(seam.depth_gap_score),
            "depth_gap_accepted": bool(seam.depth_gap_accepted),
            "depth_gap_raw_accepted": bool(seam.depth_gap_raw_accepted),
            "depth_gap_score_plausible": bool(seam.depth_gap_score_plausible),
            "depth_gap_threshold_used_m": _round(seam.depth_gap_threshold_used_m),
            "min_depth_gap_score_m": _round(seam.min_depth_gap_score_m),
            "max_depth_gap_score_m": _round(seam.max_depth_gap_score_m),
            "depth_gap_depth_jump_m": _round(seam.depth_gap_depth_jump_m),
            "depth_gap_coverage_drop": _round(seam.depth_gap_coverage_drop),
            "yolo_weak_depth_gap_supported": bool(seam.yolo_weak_depth_gap_supported),
            "yolo_weak_depth_gap_threshold_m": _round(seam.yolo_weak_depth_gap_threshold_m),
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
        if seam.yolo_seg_mask is not None and seam.yolo_seg_detected:
            overlay[seam.yolo_seg_mask] = (40, 120, 255, 95)
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
        self.detector = OnlinePipeJunctionDetector(self.params)
        self.rgb_queue: deque[tuple[float, Image]] = deque(maxlen=int(self.params["queue_size"]))
        self.depth_queue: deque[tuple[float, Image]] = deque(maxlen=int(self.params["queue_size"]))
        self.info_queue: deque[tuple[float, CameraInfo]] = deque(maxlen=int(self.params["queue_size"]))
        self.received_count = 0
        self.last_processed_rgb_time: float | None = None
        self.last_status_publish_time = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=int(self.params["queue_size"]),
        )
        self.create_subscription(Image, str(self.params["rgb_topic"]), self._rgb_cb, qos)
        self.create_subscription(Image, str(self.params["depth_topic"]), self._depth_cb, qos)
        self.create_subscription(CameraInfo, str(self.params["camera_info_topic"]), self._info_cb, qos)

        self.detection_pub = self.create_publisher(String, str(self.params["detection_topic"]), 10)
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
        self.get_logger().info(
            "ACEA pipe-junction detector listening to "
            f"{self.params['rgb_topic']}, {self.params['depth_topic']}, {self.params['camera_info_topic']}"
        )

    def _declare_params(self) -> dict[str, Any]:
        declarations = {
            "rgb_topic": "/camera/rgb",
            "depth_topic": "/camera/depth",
            "camera_info_topic": "/camera/camera_info",
            "detection_topic": "/acea/pipe_junction/detection",
            "rgb_overlay_topic": "/acea/pipe_junction/debug/rgb_overlay",
            "depth_overlay_topic": "/acea/pipe_junction/debug/depth_overlay",
            "publish_weld_gap_geometry": True,
            "weld_seam_pose_topic": "/acea/weld_seam/pose",
            "weld_gap_plane_topic": "/acea/weld_seam/gap_plane",
            "weld_gap_require_detector_accepted": True,
            "weld_marker_topic": "/acea/weld_seam/markers",
            "weld_marker_cylinder_length_m": 0.6,
            "weld_marker_plane_scale": 1.3,
            "sync_slop_s": 0.08,
            "allow_stale_camera_info": True,
            "queue_size": 10,
            "publish_waiting_status": True,
            "waiting_status_period_s": 1.0,
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
            "junction_acceptance_mode": "rgb_depth",
            # Variant A (deterministic RGB-only) frontend params (mode == "variant_a_rgb").
            "variant_a_tophat_se_len_px": 21,
            "variant_a_min_vertical_run_px": 100,
            "variant_a_min_significance_z": 5.0,
            "variant_a_max_seam_width_px": 14.0,
            "variant_a_border_margin_px": 15,
            "variant_a_z_confidence_strong": 10.0,
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
            "use_yolo_seg_frontend": False,
            "yolo_model_path": "",
            "yolo_python_site_packages": "",
            "yolo_config_dir": "/tmp/acea_yolo_ultralytics",
            "yolo_conf_threshold": 0.25,
            "yolo_imgsz": 640,
            "yolo_device": 0,
            "yolo_allow_cpu_fallback": True,
            "yolo_class_name": "",
            "yolo_min_mask_area_px": 12,
            "yolo_max_mask_area_fraction": 0.20,
            "yolo_min_strip_overlap_px": 6,
            "enable_yolo_weak_depth_gap_support": True,
            "yolo_weak_depth_gap_min_score_m": 0.00020,
            "yolo_weak_depth_gap_min_confidence": 0.25,
            "enable_weak_rgb_depth_support": True,
            "weak_rgb_min_dark_contrast": 0.006,
            "weak_rgb_min_z_score": 4.0,
            "weak_rgb_strong_z_score": 8.0,
            "weak_rgb_confidence_scale": 1.0,
            "use_depth_gap_gate": True,
            "depth_gap_neighbor_offset_px": 10,
            "depth_gap_band_half_width_px": 2,
            "min_depth_gap_m": 0.00038,
            "min_depth_gap_coverage_drop": 0.015,
            "min_depth_gap_score_m": 0.0,
            "max_depth_gap_score_m": 0.0,
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
        self.rgb_queue.append((self._message_time(msg), msg))
        self._try_process()

    def _depth_cb(self, msg: Image) -> None:
        self.depth_queue.append((self._message_time(msg), msg))
        self._try_process()

    def _info_cb(self, msg: CameraInfo) -> None:
        self.info_queue.append((self._message_time(msg), msg))
        self._try_process()

    def _message_time(self, msg: Image | CameraInfo) -> float:
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

    def _publish(self, status: dict[str, Any], rgb_overlay: np.ndarray, depth_overlay: np.ndarray, header: Any) -> None:
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

        if not self.rgb_queue:
            reason = "waiting_for_rgb"
        elif not self.depth_queue:
            reason = "waiting_for_depth"
        elif not self.info_queue:
            reason = "waiting_for_camera_info"
        elif info_pair is None:
            reason = "waiting_for_valid_camera_info"
        elif depth_dt is not None and abs(depth_dt) > float(self.params["sync_slop_s"]):
            reason = "waiting_for_rgb_depth_sync"
        else:
            reason = "waiting_for_next_frame"

        status = {
            "state": "WAITING_FOR_SYNC",
            "reason": reason,
            "rgb_queue": len(self.rgb_queue),
            "depth_queue": len(self.depth_queue),
            "camera_info_queue": len(self.info_queue),
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
        self.last_status_publish_time = now

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
    node = AceaPipeJunctionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
