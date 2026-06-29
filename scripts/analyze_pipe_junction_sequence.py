#!/usr/bin/env python3
"""Summarize a recorded pipe-junction sequence: stability AND ground-truth accuracy.

Input is the folder produced by record_pipe_junction_sequence.py.

Two layers:
  * STABILITY (ground-truth-free): acceptance fraction, x jumps, depth-gate
    availability, lock continuity, contact sheet of notable frames.
  * ACCURACY (ground truth from the raw sim frame): the weld gap is a black
    marker on the bright pipe, so the true gap column is the peak of a horizontal
    black-hat (a dark vertical line on the bright horizontal pipe). We compare the
    detector's committed column (candidate_x_strip_px; the pipe is near-horizontal
    in framed sweeps so strip x ~ image x) against this true column on every frame
    where the gap is actually visible. This catches a detector that is *stably
    wrong* (high acceptance + no jumps, but tracking a column beside the gap) and
    flags 'gap visible but rejected' frames (e.g. the candidate_jump poisoning).

Accuracy needs the raw frames/*.png (always recorded); it auto-disables if the
images or Pillow are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _bool(value: Any) -> bool:
    return bool(value)


def _row_scalar(row: dict[str, Any], key: str) -> Any:
    status = row.get("status") if isinstance(row.get("status"), dict) else {}
    detection = row.get("detection") if isinstance(row.get("detection"), dict) else {}
    if key in status:
        return status.get(key)
    return detection.get(key)


# --------------------------------------------------------------------------- #
# Ground truth: true gap column from the raw sim frame
# --------------------------------------------------------------------------- #
def gap_col_and_prominence(rgb: np.ndarray, upper_frac: float = 0.62,
                           halfwin: int = 14, smooth: int = 5) -> tuple[int, float]:
    """True weld-gap column = peak of a horizontal black-hat (dark vertical line
    on the bright horizontal pipe). Returns (column, prominence = peak/median).

    The horizontal black-hat rejects the lower shadow band (dark, but its
    horizontal neighbours are equally dark) and uniform background; the gripper is
    excluded by upper_frac. Low prominence => the gap is not framed (clipped/out)."""
    g = rgb[: int(upper_frac * rgb.shape[0])].mean(axis=2).astype(np.float32)
    left = np.empty_like(g)
    right = np.empty_like(g)
    left[:, halfwin:] = g[:, :-halfwin]
    left[:, :halfwin] = g[:, :1]
    right[:, :-halfwin] = g[:, halfwin:]
    right[:, -halfwin:] = g[:, -1:]
    resp = np.clip(0.5 * (left + right) - g, 0.0, None)      # dark-on-bright
    score = resp.sum(axis=0)
    if smooth > 1:
        score = np.convolve(score, np.ones(smooth) / smooth, mode="same")
    gap = int(np.argmax(score))
    med = float(np.median(score)) + 1e-6
    return gap, float(score[gap]) / med


# ----- projected ground truth (gz gap pose + per-frame camera tf), tilt/contrast-free -----
def _quat_to_R(q: list[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _project_gap(cam_pose: dict[str, Any], gap_base: np.ndarray, K: tuple[float, float, float, float]):
    """Project the gap point (in the tf-base frame) into the image using the logged
    base->camera-optical transform. Returns (u, v) or None (behind camera)."""
    t = np.array([float(v) for v in cam_pose["t"]], dtype=np.float64)
    R = _quat_to_R(cam_pose["q"])           # base<-cam rotation (cam pose in base)
    p_cam = R.T @ (gap_base - t)            # base -> camera optical (+z forward)
    if p_cam[2] <= 1e-6:
        return None
    fx, fy, cx, cy = K
    return (fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy)


def _point_to_segment_dist(p, line) -> float | None:
    try:
        a = np.array([float(line[0][0]), float(line[0][1])], dtype=np.float64)
        b = np.array([float(line[1][0]), float(line[1][1])], dtype=np.float64)
    except Exception:
        return None
    pp = np.array([float(p[0]), float(p[1])], dtype=np.float64)
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return float(np.linalg.norm(pp - a))
    tt = float((pp - a) @ ab) / denom        # perpendicular distance to the (infinite) junction line
    return float(np.linalg.norm(pp - (a + tt * ab)))


def _compute_gt(seq: Path, rows: list[dict[str, Any]], visible_min_prominence: float
                ) -> tuple[bool, dict[int, dict[str, Any]]]:
    """Per-saved_frame GT. Returns (available, {saved_frame: {...}})."""
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return False, {}
    from PIL import Image
    gt: dict[int, dict[str, Any]] = {}
    seen_any = False
    for row in rows:
        rel = row.get("rgb")
        if not rel:
            continue
        img_path = seq / str(rel)
        if not img_path.exists():
            continue
        seen_any = True
        rgb = np.asarray(Image.open(img_path).convert("RGB"))
        gap, prom = gap_col_and_prominence(rgb)
        gt[int(row.get("saved_frame"))] = {
            "gap_col": gap,
            "gap_prominence": round(prom, 2),
            "gap_visible": bool(prom >= visible_min_prominence),
        }
    return seen_any, gt


def _write_csv(rows: list[dict[str, Any]], path: Path, jump_threshold_px: float,
               gt: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    last_accepted_x: float | None = None
    for row in rows:
        x = _num(_row_scalar(row, "candidate_x_strip_px"))
        accepted = _bool(_row_scalar(row, "detector_accepted"))
        dx = None
        jump = False
        if accepted and x is not None and last_accepted_x is not None:
            dx = x - last_accepted_x
            jump = abs(dx) > jump_threshold_px
        if accepted and x is not None:
            last_accepted_x = x
        g = gt.get(int(row.get("saved_frame"))) if row.get("saved_frame") is not None else None
        gap_col = g.get("gap_col") if g else None
        gap_visible = g.get("gap_visible") if g else None
        gt_err = None
        gt_abs_err = None
        if g and gap_visible and x is not None:
            gt_err = float(x) - float(gap_col)
            gt_abs_err = abs(gt_err)
        out = {
            "saved_frame": row.get("saved_frame"),
            "time_s": row.get("time_s"),
            "detected": _bool(_row_scalar(row, "detected")),
            "accepted": accepted,
            "x": x,
            "dx_from_prev_accepted": dx,
            "jump": jump,
            "gap_col_gt": gap_col,
            "gap_visible": gap_visible,
            "gap_prominence": g.get("gap_prominence") if g else None,
            "gt_err_px": None if gt_err is None else round(gt_err, 1),
            "gt_abs_err_px": None if gt_abs_err is None else round(gt_abs_err, 1),
            "depth_gap_accepted": _bool(_row_scalar(row, "depth_gap_accepted")),
            "gap_plane_available": _bool(_row_scalar(row, "gap_plane_available")),
            "weld_seam_pose_available": _bool(_row_scalar(row, "weld_seam_pose_available")),
            "junction_lock_active": _bool(_row_scalar(row, "junction_lock_active")),
            "junction_lock_source": _row_scalar(row, "junction_lock_source"),
            "junction_lock_missed_frames": _row_scalar(row, "junction_lock_missed_frames"),
            "klt_points": _row_scalar(row, "klt_points"),
            "klt_status": _row_scalar(row, "klt_status"),
            "reason": _row_scalar(row, "reason"),
            "rgb_overlay": row.get("rgb_overlay"),
        }
        out_rows.append(out)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["saved_frame"])
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def _fraction(values: list[bool]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _make_contact_sheet(seq_dir: Path, rows: list[dict[str, Any]], path: Path, max_tiles: int = 24) -> None:
    def _high_err(row: dict[str, Any]) -> bool:
        e = row.get("gt_abs_err_px")
        return e is not None and float(e) > 20.0

    selected: list[dict[str, Any]] = []
    for row in rows:
        if (row.get("jump") or not row.get("accepted")
                or row.get("junction_lock_missed_frames") not in (None, 0, "0")
                or _high_err(row)
                or (row.get("gap_visible") and not row.get("accepted"))):
            selected.append(row)
    if not selected:
        selected = rows[::max(1, len(rows) // max_tiles)] if rows else []
    selected = selected[:max_tiles]
    if not selected:
        return

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    thumbs = []
    for row in selected:
        rel = row.get("rgb_overlay")
        if not rel:
            continue
        img_path = seq_dir / str(rel)
        if not img_path.exists():
            continue
        im = Image.open(img_path).convert("RGB").resize((320, 240))
        draw = ImageDraw.Draw(im)
        err = row.get("gt_abs_err_px")
        label = (
            f"f={row.get('saved_frame')} acc={int(bool(row.get('accepted')))} "
            f"x={row.get('x')} err={'-' if err is None else err}"
        )
        draw.rectangle((0, 0, 320, 22), fill=(0, 0, 0))
        draw.text((6, 5), label[:54], fill=(255, 255, 255))
        thumbs.append(im)
    if not thumbs:
        return
    cols = 3
    rows_n = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 320, rows_n * 240), (30, 30, 30))
    for idx, im in enumerate(thumbs):
        sheet.paste(im, ((idx % cols) * 320, (idx // cols) * 240))
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", required=True, type=Path)
    parser.add_argument("--jump-threshold-px", type=float, default=50.0)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to write analysis outputs (default: the sequence folder).")
    parser.add_argument("--gt", dest="gt", action="store_true", default=True,
                        help="Compute ground-truth accuracy from the raw frames (default on).")
    parser.add_argument("--no-gt", dest="gt", action="store_false")
    parser.add_argument("--gt-visible-min-prominence", type=float, default=8.0,
                        help="Min horizontal-black-hat prominence to call the gap visible.")
    parser.add_argument("--gt-median-thr-px", type=float, default=12.0,
                        help="Accuracy PASS needs median |error| <= this.")
    parser.add_argument("--gt-p90-thr-px", type=float, default=30.0,
                        help="Accuracy PASS needs p90 |error| <= this.")
    parser.add_argument("--image-width", type=int, default=640,
                        help="Image width, for the near-border false-accept check.")
    parser.add_argument("--near-border-px", type=float, default=50.0,
                        help="A candidate within this many px of the image edge counts as near-border.")
    parser.add_argument("--gap-base", default="2.0,0.0,0.003",
                        help="Gap position in the tf-base frame [m] 'x,y,z' (gz gap_world - base_world). "
                             "Enables the tilt/contrast-free PROJECTED GT when cam_pose is logged.")
    parser.add_argument("--fx", type=float, default=462.1)
    parser.add_argument("--fy", type=float, default=462.1)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    args = parser.parse_args()

    seq = args.seq
    out_dir = args.out_dir or seq
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_jsonl = seq / "frames.jsonl"
    if not frames_jsonl.exists():
        raise SystemExit(f"missing {frames_jsonl}")
    rows = _read_jsonl(frames_jsonl)

    gt_available, gt = (False, {})
    if args.gt:
        gt_available, gt = _compute_gt(seq, rows, float(args.gt_visible_min_prominence))

    csv_rows = _write_csv(rows, out_dir / "analysis.csv", float(args.jump_threshold_px), gt)

    xs = np.asarray([r["x"] for r in csv_rows if r.get("accepted") and r.get("x") is not None], dtype=np.float64)
    dx = np.asarray([r["dx_from_prev_accepted"] for r in csv_rows if r.get("dx_from_prev_accepted") is not None], dtype=np.float64)
    jumps = [r for r in csv_rows if r.get("jump")]
    reasons: dict[str, int] = {}
    for row in csv_rows:
        reason = str(row.get("reason") or "")
        reasons[reason] = reasons.get(reason, 0) + 1

    # ---- accuracy (ground truth) ----
    visible = [r for r in csv_rows if r.get("gap_visible")]
    acc_vis = [r for r in visible if r.get("accepted") and r.get("gt_abs_err_px") is not None]
    abs_err = np.asarray([float(r["gt_abs_err_px"]) for r in acc_vis], dtype=np.float64)
    signed = np.asarray([float(r["gt_err_px"]) for r in acc_vis], dtype=np.float64)
    visible_rejected = [r for r in visible if not r.get("accepted")]
    # bug2 / clipped-edge residual: the detector ACCEPTED while no gap is visible
    # (a phantom / clipped-pipe-edge), and the subset near the image border (the
    # clipped-pipe-edge signature). NOTE: gap-not-visible can also be a GT miss
    # (faint real gap), so the near-border subset is the cleaner false-accept signal.
    _near = float(args.near_border_px)
    _imgw = float(args.image_width)
    accepted_gap_absent = [r for r in csv_rows if r.get("accepted") and not r.get("gap_visible")]
    accepted_near_border_no_gap = [
        r for r in accepted_gap_absent
        if r.get("x") is not None and (float(r["x"]) < _near or float(r["x"]) > _imgw - _near)
    ]
    accuracy_block: dict[str, Any] = {"gt_available": bool(gt_available)}
    accuracy_pass: bool | None = None
    if gt_available:
        median_abs = float(np.median(abs_err)) if abs_err.size else None
        p90_abs = float(np.percentile(abs_err, 90)) if abs_err.size else None
        vis_rej_frac = float(len(visible_rejected)) / max(1, len(visible))
        accuracy_pass = bool(
            abs_err.size > 0
            and median_abs is not None and median_abs <= float(args.gt_median_thr_px)
            and p90_abs is not None and p90_abs <= float(args.gt_p90_thr_px)
            and vis_rej_frac <= 0.10
        )
        accuracy_block.update({
            "gap_visible_frames": len(visible),
            "accuracy_eval_frames": int(abs_err.size),
            "median_abs_err_px": None if median_abs is None else round(median_abs, 2),
            "p90_abs_err_px": None if p90_abs is None else round(p90_abs, 2),
            "max_abs_err_px": float(np.max(abs_err)) if abs_err.size else None,
            "mean_signed_err_px": round(float(np.mean(signed)), 2) if signed.size else None,
            "gap_visible_but_rejected": len(visible_rejected),
            "gap_visible_but_rejected_fraction": round(vis_rej_frac, 3),
            "accepted_gap_not_visible": len(accepted_gap_absent),
            "accepted_near_border_no_gap": len(accepted_near_border_no_gap),
            "median_thr_px": float(args.gt_median_thr_px),
            "p90_thr_px": float(args.gt_p90_thr_px),
            "accuracy_pass": accuracy_pass,
        })

    # ---- PROJECTED ground truth (rigorous, tilt/contrast-free, ASYNC-FREE) ----
    # Projects the known gz gap into the image via the per-frame camera pose and
    # measures the perpendicular distance to the detector's junction line. The line
    # is paired to each frame by EXACT processed-RGB stamp (detector's rgb_stamp_s
    # vs the frame's rgb_stamp_s, from detection.jsonl) -- NOT latest-arrival -- so
    # there is no async during fast sweeps and the error is the pure localization
    # error on EVERY accepted frame regardless of pipe roll or gap contrast.
    projected_block: dict[str, Any] = {"available": False}
    try:
        gap_base = np.array([float(v) for v in str(args.gap_base).split(",")], dtype=np.float64)
    except Exception:
        gap_base = None
    K = (float(args.fx), float(args.fy), float(args.cx), float(args.cy))
    det_by_stamp: dict[float, dict[str, Any]] = {}
    det_path = seq / "detection.jsonl"
    if det_path.exists():
        for d in _read_jsonl(det_path):
            st = d.get("rgb_stamp_s")
            if st is not None and d.get("candidate_line_image_uv"):
                det_by_stamp[round(float(st), 6)] = d
    proj_err: list[float] = []
    stamp_matched = 0
    for row in rows:
        cam = row.get("cam_pose")
        stamp = row.get("rgb_stamp_s")
        if not isinstance(cam, dict) or stamp is None:
            continue
        det = det_by_stamp.get(round(float(stamp), 6))
        if det is None:
            continue
        stamp_matched += 1
        if not _bool(det.get("detector_accepted")):
            continue
        line = det.get("candidate_line_image_uv")
        if not line or gap_base is None:
            continue
        uv = _project_gap(cam, gap_base, K)
        if uv is None:
            continue
        d = _point_to_segment_dist(uv, line)
        if d is not None:
            proj_err.append(d)
    if gap_base is not None and proj_err:
        pe = np.asarray(proj_err, dtype=np.float64)
        projected_block = {
            "available": True,
            "stamp_synced": True,
            "gap_base_m": gap_base.tolist(),
            "eval_frames": int(pe.size),
            "stamp_matched_frames": int(stamp_matched),
            "median_err_px": round(float(np.median(pe)), 2),
            "p90_err_px": round(float(np.percentile(pe, 90)), 2),
            "max_err_px": round(float(np.max(pe)), 2),
        }
    elif not det_by_stamp:
        projected_block = {"available": False,
                           "note": "detection.jsonl has no rgb_stamp_s (old detector) or no cam_pose; "
                                   "re-record with the stamp-publishing detector + --log-camera-pose"}
    elif stamp_matched == 0:
        projected_block = {"available": False,
                           "note": "no frame rgb_stamp_s matched a detection stamp (check clocks/recording)"}

    summary = {
        "sequence": str(seq),
        "frames": len(csv_rows),
        "detected_fraction": _fraction([bool(r.get("detected")) for r in csv_rows]),
        "accepted_fraction": _fraction([bool(r.get("accepted")) for r in csv_rows]),
        "depth_gap_accepted_fraction": _fraction([bool(r.get("depth_gap_accepted")) for r in csv_rows]),
        "gap_plane_available_fraction": _fraction([bool(r.get("gap_plane_available")) for r in csv_rows]),
        "lock_active_fraction": _fraction([bool(r.get("junction_lock_active")) for r in csv_rows]),
        "jump_threshold_px": float(args.jump_threshold_px),
        "jump_count": len(jumps),
        "x_min": float(np.min(xs)) if xs.size else None,
        "x_max": float(np.max(xs)) if xs.size else None,
        "x_range_px": float(np.max(xs) - np.min(xs)) if xs.size else None,
        "dx_abs_median_px": float(np.median(np.abs(dx))) if dx.size else None,
        "dx_abs_max_px": float(np.max(np.abs(dx))) if dx.size else None,
        "max_lock_missed_frames": max([int(r.get("junction_lock_missed_frames") or 0) for r in csv_rows], default=0),
        "reason_counts": dict(sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:12]),
        "jump_frames": [r.get("saved_frame") for r in jumps[:50]],
        "accuracy": accuracy_block,
        "projected_accuracy": projected_block,
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _make_contact_sheet(seq, csv_rows, out_dir / "analysis_contact_sheet.png")

    print("[analysis] sequence:", seq)
    print(f"[analysis] frames={summary['frames']} accepted={summary['accepted_fraction']:.3f} "
          f"depth={summary['depth_gap_accepted_fraction']:.3f} lock={summary['lock_active_fraction']:.3f}")
    print(f"[analysis] STABILITY x_range={summary['x_range_px']}px "
          f"median|dx|={summary['dx_abs_median_px']}px max|dx|={summary['dx_abs_max_px']}px "
          f"jumps>{summary['jump_threshold_px']}px={summary['jump_count']} max_lock_missed={summary['max_lock_missed_frames']}")
    if gt_available:
        ab = accuracy_block
        if ab.get("accuracy_eval_frames"):
            print(f"[analysis] ACCURACY(GT) median|err|={ab['median_abs_err_px']}px "
                  f"p90={ab['p90_abs_err_px']}px max={ab['max_abs_err_px']}px "
                  f"bias={ab['mean_signed_err_px']:+}px (n={ab['accuracy_eval_frames']}/{ab['gap_visible_frames']} visible)")
        print(f"[analysis] ACCURACY(GT) gap_visible_but_rejected={ab['gap_visible_but_rejected']}/"
              f"{ab['gap_visible_frames']} ({ab['gap_visible_but_rejected_fraction']:.0%})  "
              f"=> accuracy_pass={ab.get('accuracy_pass')}")
        print(f"[analysis] BUG2-residual accepted_gap_not_visible={ab['accepted_gap_not_visible']} "
              f"near_border_no_gap={ab['accepted_near_border_no_gap']}  "
              f"(clipped/border false-accepts; high near_border_no_gap => tune "
              f"rgb_temporal_pipe_end_min_side_coverage)")
    else:
        print("[analysis] ACCURACY(GT): disabled (no raw frames or Pillow missing)")
    if projected_block.get("available"):
        pb = projected_block
        print(f"[analysis] PROJECTED-GT (tilt-free, STAMP-SYNCED) median={pb['median_err_px']}px "
              f"p90={pb['p90_err_px']}px max={pb['max_err_px']}px "
              f"(n={pb['eval_frames']} accepted, {pb['stamp_matched_frames']} stamp-matched)")
    elif "note" in projected_block:
        print(f"[analysis] PROJECTED-GT: unavailable - {projected_block['note']}")
    print(f"[analysis] csv: {out_dir / 'analysis.csv'}")
    print(f"[analysis] summary: {out_dir / 'analysis_summary.json'}")
    if (out_dir / "analysis_contact_sheet.png").exists():
        print(f"[analysis] contact sheet: {out_dir / 'analysis_contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
