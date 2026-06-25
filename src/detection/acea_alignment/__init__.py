"""ACEA Module 2 deterministic pipe-alignment core.

This package is the clean, deterministic replacement for the scattered
alignment logic that previously lived inside ``acea_pipe_junction_node.py``
(Python PCA-consensus) and ``acea_pipe_pose_pcl_node.cpp`` (randomized
PCL/RANSAC). It is pure NumPy/SciPy, has no ROS dependency, and is designed to
be validated offline (synthetic + logged frames) before any live Isaac/XBot2
run.

Design goals:
  * Strictly deterministic (no RANSAC randomness).
  * One pinned camera-frame convention, documented in ``pipe_pose``.
  * Derived sign conventions (no guessed ``*_sign`` flags).
  * Drop-in JSON contract for ``/acea/pipe_junction/pipe_pose`` so the existing
    scan-control node consumes it unchanged.

See ``acea_alignment_offline_eval.py`` for the validation harness.
"""

from __future__ import annotations

from .filtering import DEFAULT_FILTER_PARAMS, FilteredPose, PoseFilter
from .pipe_pose import (
    DEFAULT_PIPE_POSE_PARAMS,
    PipePose,
    backproject,
    fit_pipe_pose,
    foreground_mask,
)

__all__ = [
    "DEFAULT_PIPE_POSE_PARAMS",
    "PipePose",
    "backproject",
    "fit_pipe_pose",
    "foreground_mask",
    "DEFAULT_FILTER_PARAMS",
    "FilteredPose",
    "PoseFilter",
]
