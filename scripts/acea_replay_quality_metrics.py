#!/usr/bin/env python3
"""Summarize detector quality metrics from offline replay status.jsonl files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


def _vec(row: dict[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(3)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _axis_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = a / max(float(np.linalg.norm(a)), 1e-12)
    bb = b / max(float(np.linalg.norm(b)), 1e-12)
    dot = abs(float(np.clip(np.dot(aa, bb), -1.0, 1.0)))
    return float(np.degrees(np.arccos(dot)))


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.median(arr)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def summarize(run: Path, visible_from: int | None, initial_until: int | None) -> dict[str, Any]:
    rows = [json.loads(line) for line in (run / "status.jsonl").read_text().splitlines() if line.strip()]
    accepted = [r for r in rows if bool(r.get("detector_accepted"))]
    first_accepted = accepted[0].get("frame_index") if accepted else None
    state_counts = Counter(str(r.get("pipe_tracker_state")) for r in rows)
    reason_counts = Counter(str(r.get("reason")) for r in rows if not bool(r.get("detector_accepted")))

    false_initial = None
    if initial_until is not None:
        false_initial = sum(1 for r in accepted if int(r.get("frame_index", -1)) < initial_until)
    misses_visible = None
    visible_frames = None
    if visible_from is not None:
        visible = [r for r in rows if int(r.get("frame_index", -1)) >= visible_from]
        visible_frames = len(visible)
        misses_visible = sum(1 for r in visible if not bool(r.get("detector_accepted")))

    axis_jitter: list[float] = []
    prev_axis: np.ndarray | None = None
    for r in rows:
        if str(r.get("pipe_tracker_state")) != "TRACK":
            continue
        axis = _vec(r, "pipe_axis_camera_xyz")
        if axis is None:
            continue
        if prev_axis is not None:
            axis_jitter.append(_axis_delta_deg(prev_axis, axis))
        prev_axis = axis

    junction_jitter: list[float] = []
    prev_center: np.ndarray | None = None
    for r in rows:
        if not bool(r.get("detector_accepted")):
            continue
        center = _vec(r, "coarse_seam_center_camera_xyz_m")
        if center is None:
            continue
        if prev_center is not None:
            junction_jitter.append(1000.0 * float(np.linalg.norm(center - prev_center)))
        prev_center = center

    summary_path = run / "replay_summary.json"
    runtime = None
    if summary_path.exists():
        runtime = json.loads(summary_path.read_text()).get("process_ms")

    return {
        "run": str(run),
        "frames": len(rows),
        "accepted": len(accepted),
        "accepted_fraction": round(len(accepted) / max(1, len(rows)), 4),
        "first_accepted_frame": first_accepted,
        "initial_false_accepts": false_initial,
        "visible_from_frame": visible_from,
        "visible_frames": visible_frames,
        "misses_visible": misses_visible,
        "pipe_end_related_rejects": sum(v for k, v in reason_counts.items() if "pipe_end" in k),
        "pipe_tracker_state_counts": dict(state_counts),
        "base_lock_active_frames": sum(1 for r in rows if bool(r.get("pipe_base_lock_active"))),
        "base_lock_sources": dict(Counter(str(r.get("pipe_base_lock_source")) for r in rows)),
        "axis_jitter_deg_per_frame": _percentiles(axis_jitter),
        "junction_3d_jitter_mm_per_frame": _percentiles(junction_jitter),
        "runtime_ms": runtime,
        "top_reject_reasons": reason_counts.most_common(8),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--visible-from", type=int, default=None)
    parser.add_argument("--initial-until", type=int, default=None)
    args = parser.parse_args()
    for run in args.runs:
        print(json.dumps(summarize(run, args.visible_from, args.initial_until), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
