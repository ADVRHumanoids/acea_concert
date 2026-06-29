#!/usr/bin/env python3
"""Run one clean pipe-junction final validation sequence.

This wrapper assumes Gazebo, camera bridges, detector, xbot2, and the robot home
pose are already up. It only orchestrates:

  1. record_pipe_junction_sequence.py into a fresh motion_sequences/<RUN_ID>
  2. pipe_junction_camera_sweep.py for the repeatable wrist sweep
  3. analyze_pipe_junction_sequence.py
  4. threshold checks for the final validation metrics

It writes the latest run id to /tmp/acea_last_run_id for the usual follow-up
commands.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(str(part) for part in cmd)


def _run(cmd: list[str]) -> None:
    print(f"[final-validation] run: {_cmd_str(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _threshold(name: str, value: float | int | None, limit: float, op: str) -> tuple[bool, str]:
    if value is None:
        return False, f"{name}: missing"
    if op == "<=":
        ok = float(value) <= float(limit)
        return ok, f"{name}: {value} <= {limit}"
    if op == ">=":
        ok = float(value) >= float(limit)
        return ok, f"{name}: {value} >= {limit}"
    raise ValueError(op)


def _evaluate(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    accuracy = summary.get("accuracy") if isinstance(summary.get("accuracy"), dict) else {}
    projected = summary.get("projected_accuracy") if isinstance(summary.get("projected_accuracy"), dict) else {}

    checks: list[tuple[bool, str]] = []
    checks.append(_threshold("accepted_fraction", summary.get("accepted_fraction"), args.min_accepted_fraction, ">="))
    checks.append(_threshold("lock_active_fraction", summary.get("lock_active_fraction"), args.min_lock_fraction, ">="))
    checks.append(_threshold("jump_count", summary.get("jump_count"), args.max_jumps, "<="))
    checks.append(_threshold(
        "gap_visible_but_rejected",
        accuracy.get("gap_visible_but_rejected"),
        args.max_gap_visible_rejected,
        "<=",
    ))
    checks.append(_threshold(
        "near_border_no_gap",
        accuracy.get("accepted_near_border_no_gap"),
        args.max_near_border_no_gap,
        "<=",
    ))
    checks.append((bool(projected.get("available")), "PROJECTED-GT available"))
    if projected.get("available"):
        checks.append(_threshold("PROJECTED-GT median", projected.get("median_err_px"), args.max_projected_median_px, "<="))
        checks.append(_threshold("PROJECTED-GT p90", projected.get("p90_err_px"), args.max_projected_p90_px, "<="))
        checks.append(_threshold("PROJECTED-GT max", projected.get("max_err_px"), args.max_projected_max_px, "<="))
        checks.append(_threshold(
            "PROJECTED-GT eval_frames",
            projected.get("eval_frames"),
            args.min_projected_eval_frames,
            ">=",
        ))

    print("[final-validation] checks:")
    all_ok = True
    for ok, text in checks:
        all_ok = all_ok and ok
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {text}")
    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="",
                        help="Default: pipe_motion_final_validation_<timestamp>.")
    parser.add_argument("--prefix", default="pipe_motion_final_validation")
    parser.add_argument("--out-root", type=Path, default=Path("motion_sequences"))
    parser.add_argument("--record-duration", type=float, default=55.0)
    parser.add_argument("--sweep-start-delay", type=float, default=3.0)
    parser.add_argument("--sweep-duration", type=float, default=45.0)
    parser.add_argument("--sweep-cycles", type=float, default=1.0)
    parser.add_argument("--sweep-dt", type=float, default=0.02)
    parser.add_argument("--sweep-ramp-s", type=float, default=3.0)
    parser.add_argument("--j4-f-deg", type=float, default=5.0)
    parser.add_argument("--j5-f-deg", type=float, default=3.0)
    parser.add_argument("--j6-f-deg", type=float, default=8.0)
    parser.add_argument("--save-every-n", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=15.0)
    parser.add_argument("--jump-threshold-px", type=float, default=50.0)
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Only record/analyze; useful if you move the camera manually.")
    parser.add_argument("--skip-record", action="store_true",
                        help="Analyze an existing --run-id folder.")
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")

    parser.add_argument("--min-accepted-fraction", type=float, default=0.95)
    parser.add_argument("--min-lock-fraction", type=float, default=0.95)
    parser.add_argument("--max-jumps", type=int, default=0)
    parser.add_argument("--max-gap-visible-rejected", type=int, default=2)
    parser.add_argument("--max-near-border-no-gap", type=int, default=0)
    parser.add_argument("--max-projected-median-px", type=float, default=4.0)
    parser.add_argument("--max-projected-p90-px", type=float, default=6.0)
    parser.add_argument("--max-projected-max-px", type=float, default=8.0)
    parser.add_argument("--min-projected-eval-frames", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id.strip() or f"{args.prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    seq = args.out_root / run_id

    if seq.exists() and not args.allow_existing:
        raise SystemExit(f"{seq} already exists. Use --run-id NEW_NAME or --allow-existing.")
    seq.parent.mkdir(parents=True, exist_ok=True)
    Path("/tmp/acea_last_run_id").write_text(run_id + "\n", encoding="utf-8")

    print(f"[final-validation] RUN_ID={run_id}")
    print(f"[final-validation] sequence={seq}")

    recorder: subprocess.Popen[Any] | None = None
    if not args.skip_record:
        record_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "record_pipe_junction_sequence.py"),
            "--out",
            str(seq),
            "--duration",
            str(float(args.record_duration)),
            "--save-every-n",
            str(int(args.save_every_n)),
            "--best-effort",
            "--log-camera-pose",
            "--video-fps",
            str(float(args.video_fps)),
        ]
        print(f"[final-validation] start recorder: {_cmd_str(record_cmd)}", flush=True)
        recorder = subprocess.Popen(record_cmd)
        time.sleep(max(0.0, float(args.sweep_start_delay)))

        try:
            if not args.skip_sweep:
                sweep_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "pipe_junction_camera_sweep.py"),
                    "--duration",
                    str(float(args.sweep_duration)),
                    "--dt",
                    str(float(args.sweep_dt)),
                    "--cycles",
                    str(float(args.sweep_cycles)),
                    "--ramp-s",
                    str(float(args.sweep_ramp_s)),
                    "--j4-f-deg",
                    str(float(args.j4_f_deg)),
                    "--j5-f-deg",
                    str(float(args.j5_f_deg)),
                    "--j6-f-deg",
                    str(float(args.j6_f_deg)),
                    "--return-home",
                ]
                _run(sweep_cmd)

            timeout = max(5.0, float(args.record_duration) + 15.0)
            recorder.wait(timeout=timeout)
        except BaseException:
            if recorder.poll() is None:
                recorder.terminate()
                try:
                    recorder.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    recorder.kill()
            raise

        if recorder.returncode != 0:
            raise SystemExit(f"recorder failed with exit code {recorder.returncode}")

    if args.no_analyze:
        return 0

    analyze_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "analyze_pipe_junction_sequence.py"),
        "--seq",
        str(seq),
        "--jump-threshold-px",
        str(float(args.jump_threshold_px)),
    ]
    _run(analyze_cmd)

    summary_path = seq / "analysis_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"analysis did not write {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    ok = _evaluate(summary, args)

    print(f"[final-validation] rgb overlay: {seq / 'rgb_overlay.mp4'}")
    print(f"[final-validation] summary: {summary_path}")
    print(f"[final-validation] csv: {seq / 'analysis.csv'}")
    print(f"[final-validation] contact sheet: {seq / 'analysis_contact_sheet.png'}")
    print("[final-validation] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
