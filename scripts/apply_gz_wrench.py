#!/usr/bin/env python3
"""
Publish a Gazebo link/model wrench from the command line.

Example:
    python3 scripts/apply_gz_wrench.py --link L_6_F --force-y 20 --duration 1.0
"""

import argparse
import shutil
import subprocess
import sys
from time import sleep


def fmt_vec(name, values):
    x, y, z = values
    return f"{name}: {{x: {x:g}, y: {y:g}, z: {z:g}}}"


def run_gz_topic(topic, msg_type, payload, dry_run=False):
    cmd = ["gz", "topic", "-t", topic, "-m", msg_type, "-p", payload]
    print("+ " + " ".join(f"'{arg}'" if " " in arg else arg for arg in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_entity_name(args):
    if args.entity_name:
        return args.entity_name

    if args.entity_type == "MODEL":
        if not args.model:
            raise ValueError("--model is required for entity type MODEL")
        return args.model

    if not args.model:
        raise ValueError("--model is required when targeting a link")

    if "::" in args.link:
        return args.link

    return f"{args.model}::{args.link}"


def main():
    parser = argparse.ArgumentParser(
        description="Apply or clear a Gazebo wrench on a model or link.",
    )
    parser.add_argument("--world", default="default", help="Gazebo world name.")
    parser.add_argument("--model", default="ModularBot", help="Gazebo model name, from `gz model --list`.")
    parser.add_argument("--link", default="L_6_F", help="Link name inside the model.")
    parser.add_argument(
        "--entity-name",
        help="Exact Gazebo entity name. Overrides --model/--link.",
    )
    parser.add_argument(
        "--entity-type",
        choices=("LINK", "MODEL"),
        default="LINK",
        help="Gazebo entity type to target.",
    )
    parser.add_argument(
        "--force",
        nargs=3,
        type=float,
        metavar=("FX", "FY", "FZ"),
        help="Force vector in world frame [N].",
    )
    parser.add_argument(
        "--force-y",
        type=float,
        default=20.0,
        help="Convenience force in world Y [N], used when --force is omitted.",
    )
    parser.add_argument(
        "--torque",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("TX", "TY", "TZ"),
        help="Torque vector [Nm].",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Seconds to keep a persistent wrench before clearing it. Use 0 to leave it active.",
    )
    parser.add_argument(
        "--oneshot",
        action="store_true",
        help="Apply for one simulation step instead of using a persistent wrench.",
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Only clear the selected persistent wrench.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Gazebo commands without running them.",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=0.0,
        help="Ramp up the wrench over this many seconds (default: 0, no ramp).",
    )
    parser.add_argument(
        "--ramp-steps",
        type=int,
        default=20,
        help="Number of steps for the ramp (default: 20).",
    )
    args = parser.parse_args()

    if shutil.which("gz") is None:
        print("error: `gz` command not found. Source your ROS/Gazebo environment first.", file=sys.stderr)
        return 1

    try:
        entity_name = build_entity_name(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    prefix = f"/world/{args.world}/wrench"
    clear_payload = f"name: '{entity_name}', type: {args.entity_type}"

    if args.clear_only:
        run_gz_topic(f"{prefix}/clear", "gz.msgs.Entity", clear_payload, args.dry_run)
        return 0

    force = tuple(args.force) if args.force is not None else (0.0, args.force_y, 0.0)
    torque = tuple(args.torque)
    payload = (
        f"entity: {{name: '{entity_name}', type: {args.entity_type}}}, "
        f"wrench: {{{fmt_vec('force', force)}, {fmt_vec('torque', torque)}}}"
    )

    if args.oneshot:
        run_gz_topic(prefix, "gz.msgs.EntityWrench", payload, args.dry_run)
        return 0

    # Ramping logic
    if args.ramp_time > 0.0:
        ramp_steps = args.ramp_steps
        for i in range(1, ramp_steps + 1):
            frac = i / ramp_steps
            ramp_force = tuple(frac * f for f in force)
            ramp_torque = tuple(frac * t for t in torque)
            ramp_payload = (
                f"entity: {{name: '{entity_name}', type: {args.entity_type}}}, "
                f"wrench: {{{fmt_vec('force', ramp_force)}, {fmt_vec('torque', ramp_torque)}}}"
            )
            run_gz_topic(f"{prefix}/persistent", "gz.msgs.EntityWrench", ramp_payload, args.dry_run)
            sleep(args.ramp_time / ramp_steps)
        # After ramp, hold at full value for the remaining duration
        hold_time = max(0.0, args.duration - args.ramp_time)
        if hold_time > 0.0:
            print(f"Holding wrench at full value for {hold_time:g} s...")
            if not args.dry_run:
                sleep(hold_time)
        run_gz_topic(f"{prefix}/clear", "gz.msgs.Entity", clear_payload, args.dry_run)
        return 0

    run_gz_topic(f"{prefix}/persistent", "gz.msgs.EntityWrench", payload, args.dry_run)

    if args.duration > 0.0:
        print(f"Keeping wrench active for {args.duration:g} s...")
        if not args.dry_run:
            sleep(args.duration)
        run_gz_topic(f"{prefix}/clear", "gz.msgs.Entity", clear_payload, args.dry_run)
    else:
        print("Persistent wrench left active. Clear it with --clear-only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
