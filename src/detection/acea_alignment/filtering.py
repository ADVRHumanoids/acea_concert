"""Temporal filtering + quality gating for the pipe pose.

The per-frame fit from ``fit_pipe_pose`` is accurate but raw: feeding it straight
into a velocity controller is exactly what made the previous alignment jittery.
This filter turns the raw stream into a smooth, trustworthy alignment signal:

  * Quality gate  : drop frames the estimator already flagged invalid.
  * Outlier gate  : once locked, drop frames whose yaw / stand-off jump implausibly
                    far from the current estimate (a single corrupted depth frame
                    must not move the robot).
  * EMA smoothing : low-pass the accepted estimates (axis direction is smoothed as
                    a vector and re-normalised, never as a wrapping angle).
  * Ready logic   : only report ``ready`` after a few consecutive good frames, and
                    drop ``ready`` after a run of rejects (so control stops instead
                    of acting on a stale pose).

It is deterministic and ROS-free, so the closed-loop behaviour can be validated
offline before any live run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .pipe_pose import PipePose, _canonical_axis

DEFAULT_FILTER_PARAMS: dict[str, float | int] = {
    "ema_alpha": 0.35,          # weight of the newest accepted frame
    "max_yaw_jump_deg": 8.0,    # reject frames that jump more than this once locked
    "max_standoff_jump_m": 0.20,
    "min_ready_frames": 4,      # consecutive good frames before 'ready'
    "max_reject_streak": 6,     # consecutive rejects before dropping the lock
}


@dataclass
class FilteredPose:
    valid: bool                 # is there a usable (locked) estimate right now
    ready: bool                 # locked AND enough good frames to trust for control
    reason: str
    yaw_error_deg: float = 0.0
    surface_distance_m: float = 0.0
    stand_off_m: float = 0.0
    lateral_offset_m: float = 0.0
    vertical_offset_m: float = 0.0
    radius_m: float = 0.0
    axis_camera_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accepted_frame_count: int = 0
    reject_streak: int = 0
    last_action: str = ""       # accepted | gated_invalid | outlier | relocked

    def to_json(self) -> dict[str, Any]:
        axis = self.axis_camera_xyz
        return {
            "filtered_valid": bool(self.valid),
            "filtered_ready": bool(self.ready),
            "filtered_reason": self.reason,
            "filtered_yaw_error_deg": round(float(self.yaw_error_deg), 6),
            "filtered_surface_distance_m": round(float(self.surface_distance_m), 6),
            "filtered_stand_off_m": round(float(self.stand_off_m), 6),
            "filtered_lateral_offset_m": round(float(self.lateral_offset_m), 6),
            "filtered_pipe_axis_camera_xyz": [round(float(v), 6) for v in axis]
            if np.all(np.isfinite(axis)) else None,
            "filtered_accepted_frame_count": int(self.accepted_frame_count),
            "filtered_reject_streak": int(self.reject_streak),
            "filtered_last_action": self.last_action,
        }


class PoseFilter:
    """Stateful EMA filter + quality/outlier gate over a stream of ``PipePose``."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(DEFAULT_FILTER_PARAMS)
        if params:
            self.params.update(params)
        self.reset()

    def reset(self) -> None:
        self._locked = False
        self._yaw = 0.0
        self._surface = 0.0
        self._stand_off = 0.0
        self._lateral = 0.0
        self._vertical = 0.0
        self._radius = 0.0
        self._axis = np.array([1.0, 0.0, 0.0])
        self._accepted = 0
        self._reject_streak = 0

    def _ema(self, prev: float, new: float, alpha: float) -> float:
        return alpha * new + (1.0 - alpha) * prev

    def _lock_to(self, pose: PipePose) -> None:
        self._yaw = pose.yaw_error_deg
        self._surface = pose.surface_distance_m
        self._stand_off = pose.stand_off_m
        self._lateral = pose.lateral_offset_m
        self._vertical = pose.vertical_offset_m
        self._radius = pose.radius_m
        self._axis = _canonical_axis(np.asarray(pose.axis_camera_xyz, dtype=float))
        self._locked = True

    def update(self, pose: PipePose) -> FilteredPose:
        alpha = float(self.params["ema_alpha"])

        # 1) Quality gate: the estimator already validated radius/inliers/residual.
        if not pose.valid:
            self._reject(f"gated_invalid:{pose.reason}")
            return self._emit("gated_invalid", f"gated_invalid:{pose.reason}")

        # 2) Outlier gate (only meaningful once locked).
        if self._locked:
            dyaw = abs(pose.yaw_error_deg - self._yaw)
            dstand = abs(pose.surface_distance_m - self._surface)
            if (dyaw > float(self.params["max_yaw_jump_deg"])
                    or dstand > float(self.params["max_standoff_jump_m"])):
                self._reject(f"outlier:dyaw={dyaw:.2f},dstand={dstand:.3f}")
                return self._emit("outlier", f"outlier:dyaw={dyaw:.2f},dstand={dstand:.3f}")

        # 3) Accept.
        if not self._locked:
            self._lock_to(pose)
            action = "relocked"
        else:
            self._yaw = self._ema(self._yaw, pose.yaw_error_deg, alpha)
            self._surface = self._ema(self._surface, pose.surface_distance_m, alpha)
            self._stand_off = self._ema(self._stand_off, pose.stand_off_m, alpha)
            self._lateral = self._ema(self._lateral, pose.lateral_offset_m, alpha)
            self._vertical = self._ema(self._vertical, pose.vertical_offset_m, alpha)
            self._radius = self._ema(self._radius, pose.radius_m, alpha)
            new_axis = self._axis * (1.0 - alpha) + np.asarray(pose.axis_camera_xyz) * alpha
            self._axis = _canonical_axis(new_axis)
            action = "accepted"
        self._accepted += 1
        self._reject_streak = 0
        return self._emit(action, "ok")

    def _reject(self, _reason: str) -> None:
        self._reject_streak += 1
        if self._reject_streak >= int(self.params["max_reject_streak"]):
            # Drop the lock entirely; control must stop until we re-acquire.
            self._locked = False
            self._accepted = 0

    def _emit(self, action: str, reason: str) -> FilteredPose:
        ready = (self._locked
                 and self._accepted >= int(self.params["min_ready_frames"]))
        return FilteredPose(
            valid=self._locked,
            ready=ready,
            reason=reason,
            yaw_error_deg=self._yaw,
            surface_distance_m=self._surface,
            stand_off_m=self._stand_off,
            lateral_offset_m=self._lateral,
            vertical_offset_m=self._vertical,
            radius_m=self._radius,
            axis_camera_xyz=self._axis.copy(),
            accepted_frame_count=self._accepted,
            reject_streak=self._reject_streak,
            last_action=action,
        )
