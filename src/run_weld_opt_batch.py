#!/usr/bin/env python3
"""Run weld_opt.py repeatedly and collect pipe-height optimization data."""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from scipy.io import loadmat


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WELD_OPT = PACKAGE_ROOT / "src" / "weld_opt.py"
DEFAULT_PLOT_SCRIPT = PACKAGE_ROOT / "src" / "plot_weld_opt_batch.py"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "mat_files" / "pipe_height_runs"

SCENARIO_SETS = {
    "manual": [],
    "current": [
        ("current_bottom_half", np.pi, 2.0 * np.pi, True),
    ],
    "quarters": [
        ("quarter_right_to_top", 0.0, 0.5 * np.pi, False),
        ("quarter_top_to_left", 0.5 * np.pi, np.pi, False),
        ("quarter_left_to_bottom", np.pi, 1.5 * np.pi, True),
        ("quarter_bottom_to_right", 1.5 * np.pi, 2.0 * np.pi, True),
    ],
    "halves": [
        ("half_top", 0.0, np.pi, False),
        ("half_bottom", np.pi, 2.0 * np.pi, True),
        ("half_start_above", 0.5 * np.pi, 1.5 * np.pi, False),
        ("half_start_bottom", 1.5 * np.pi, 2.5 * np.pi, True),
    ],
}
SCENARIO_SETS["pipe-arcs"] = SCENARIO_SETS["quarters"] + SCENARIO_SETS["halves"]

SUMMARY_FIELDS = [
    "run",
    "scenario_run",
    "scenario",
    "status",
    "returncode",
    "duration_s",
    "seed",
    "angle_weld_start",
    "angle_weld_end",
    "angle_weld_span",
    "weld_upside_down",
    "pipe_z",
    "pipe_z_nominal",
    "pipe_z_delta",
    "pipe_z_bound_low",
    "pipe_z_bound_high",
    "initial_base_x",
    "initial_base_y",
    "initial_base_z",
    "weld_standoff_from_pipe",
    "weld_trajectory_radius",
    "max_abs_critical_static_tau",
    "mean_abs_critical_static_tau",
    "mat_file",
    "log_file",
    "error",
]


def _scalar(data, key, default=np.nan):
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)
    if value.size == 0:
        return default
    return float(value[0])


def _vector(data, key):
    if key not in data:
        return np.array([])
    return np.asarray(data[key], dtype=float).reshape(-1)


def _text(data, key, default=""):
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)
    if value.size == 0:
        return default
    return str(value[0]).strip()


def _extract_metrics(mat_file: Path) -> dict[str, float]:
    data = loadmat(str(mat_file))

    pipe_z = _scalar(data, "pipe_z")
    pipe_z_nominal = _scalar(data, "pipe_z_nominal")
    pipe_z_bounds = _vector(data, "pipe_z_bounds")
    initial_robot_pose = _vector(data, "initial_robot_pose")

    metrics = {
        "scenario": _text(data, "trajectory_scenario_name"),
        "angle_weld_start": _scalar(data, "angle_weld_start"),
        "angle_weld_end": _scalar(data, "angle_weld_end"),
        "angle_weld_span": _scalar(data, "angle_weld_span"),
        "weld_upside_down": bool(_scalar(data, "weld_upside_down", 0.0)),
        "pipe_z": pipe_z,
        "pipe_z_nominal": pipe_z_nominal,
        "pipe_z_delta": pipe_z - pipe_z_nominal,
        "pipe_z_bound_low": pipe_z_bounds[0] if pipe_z_bounds.size > 0 else np.nan,
        "pipe_z_bound_high": pipe_z_bounds[1] if pipe_z_bounds.size > 1 else np.nan,
        "initial_base_x": initial_robot_pose[0] if initial_robot_pose.size > 0 else np.nan,
        "initial_base_y": initial_robot_pose[1] if initial_robot_pose.size > 1 else np.nan,
        "initial_base_z": initial_robot_pose[2] if initial_robot_pose.size > 2 else np.nan,
        "weld_standoff_from_pipe": _scalar(data, "weld_standoff_from_pipe"),
        "weld_trajectory_radius": _scalar(data, "weld_trajectory_radius"),
    }

    if "tau" in data and "critical_torque_indices" in data:
        tau = np.asarray(data["tau"], dtype=float)
        indices = np.asarray(data["critical_torque_indices"], dtype=int).reshape(-1)
        valid = indices[(indices >= 0) & (indices < tau.shape[0])]
        if valid.size:
            critical_tau = tau[valid, :]
            metrics["max_abs_critical_static_tau"] = float(
                np.max(np.abs(critical_tau)))
            metrics["mean_abs_critical_static_tau"] = float(
                np.mean(np.abs(critical_tau)))
        else:
            metrics["max_abs_critical_static_tau"] = np.nan
            metrics["mean_abs_critical_static_tau"] = np.nan
    else:
        metrics["max_abs_critical_static_tau"] = np.nan
        metrics["mean_abs_critical_static_tau"] = np.nan

    return metrics


def _parse_angle(value: str) -> float:
    return float(eval(value, {"__builtins__": {}}, {"pi": np.pi, "np": np}))


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected true or false for --upside-down"
    )


def _scenario_from_tuple(item: tuple) -> dict:
    name, angle_start, angle_end, upside_down = item
    return {
        "name": name,
        "angle_start": float(angle_start),
        "angle_end": float(angle_end),
        "upside_down": bool(upside_down),
    }


def _scenario_by_name(name: str) -> dict | None:
    for scenario_set in SCENARIO_SETS.values():
        for item in scenario_set:
            scenario = _scenario_from_tuple(item)
            if scenario["name"] == name:
                return scenario
    return None


def _slug_angle(value: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    return "_".join(part for part in slug.split("_") if part) or "angle"


def _scenario_from_angle_span(parts: list[str]) -> dict:
    angle_start = _parse_angle(parts[0])
    angle_end = _parse_angle(parts[1])
    return {
        "name": f"arc_{_slug_angle(parts[0])}_to_{_slug_angle(parts[1])}",
        "angle_start": angle_start,
        "angle_end": angle_end,
        "upside_down": True,
    }


def _parse_scenario_arg(text: str) -> dict:
    text = text.strip()
    named = _scenario_by_name(text)
    if named is not None:
        return named

    parts = [part.strip() for part in text.split(":")]
    if len(parts) == 2:
        return _scenario_from_angle_span(parts)

    raise ValueError(
        "scenario must be either a preset name or start:end; "
        "for example quarter_right_to_top or 0:pi/2"
    )


def _apply_upside_down_override(scenarios: list[dict],
                                upside_down: bool | None) -> list[dict]:
    if upside_down is None:
        return scenarios
    return [{**scenario, "upside_down": upside_down} for scenario in scenarios]


def _scenario_list(args) -> list[dict]:
    if args.scenario:
        return _apply_upside_down_override(
            [_parse_scenario_arg(item) for item in args.scenario],
            args.upside_down,
        )
    if args.scenario_set == "manual":
        return _apply_upside_down_override([{
            "name": "manual",
            "angle_start": np.nan,
            "angle_end": np.nan,
            "upside_down": np.nan,
            "use_weld_opt_defaults": True,
        }], args.upside_down)
    return _apply_upside_down_override(
        [_scenario_from_tuple(item) for item in SCENARIO_SETS[args.scenario_set]],
        args.upside_down,
    )


def _write_summaries(rows: list[dict], output_dir: Path) -> None:
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)


def _track(run_id: str, message: str) -> None:
    print(f"    [{run_id}] {message}", flush=True)


def _run_weld_opt(cmd: list[str], cwd: Path, env: dict, log_file: Path,
                  timeout: float) -> tuple[int | None, bool]:
    start = perf_counter()

    with log_file.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

        while True:
            returncode = proc.poll()
            if returncode is not None:
                return returncode, False

            if perf_counter() - start > timeout:
                proc.kill()
                proc.wait()
                return None, True

            sleep(0.5)


def _run_once(args, run_index: int, scenario_run: int,
              scenario: dict, output_dir: Path) -> dict:
    run_id = f"run_{run_index:04d}_{scenario['name']}_{scenario_run:03d}"
    mat_file = output_dir / f"{run_id}.mat"
    log_file = output_dir / f"{run_id}.log"
    seed = None if args.seed_base is None else args.seed_base + run_index

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["WELD_OPT_BATCH"] = "1"
    env["WELD_OPT_OUTPUT_PATH"] = str(mat_file)
    env["WELD_OPT_MAX_ATTEMPTS"] = str(args.max_attempts)
    env["WELD_OPT_SCENARIO_NAME"] = scenario["name"]
    if not scenario.get("use_weld_opt_defaults", False):
        env["WELD_OPT_ANGLE_START"] = str(scenario["angle_start"])
        env["WELD_OPT_ANGLE_END"] = str(scenario["angle_end"])
    if isinstance(scenario["upside_down"], bool):
        env["WELD_OPT_WELD_UPSIDE_DOWN"] = "1" if scenario["upside_down"] else "0"
    if seed is not None:
        env["WELD_OPT_SEED"] = str(seed)

    cmd = [args.python, str(args.weld_opt)]
    print(f"[batch] {run_id}: starting", flush=True)
    _track(run_id, f"scenario={scenario['name']}")
    _track(run_id, f"log -> {log_file}")
    _track(run_id, "launching weld_opt.py")
    start = perf_counter()

    row = {
        "run": run_index,
        "scenario_run": scenario_run,
        "scenario": scenario["name"],
        "status": "failed",
        "returncode": "",
        "duration_s": np.nan,
        "seed": "" if seed is None else seed,
        "angle_weld_start": scenario["angle_start"],
        "angle_weld_end": scenario["angle_end"],
        "angle_weld_span": scenario["angle_end"] - scenario["angle_start"],
        "weld_upside_down": scenario["upside_down"],
        "mat_file": str(mat_file),
        "log_file": str(log_file),
        "error": "",
    }

    try:
        returncode, timed_out = _run_weld_opt(
            cmd,
            PACKAGE_ROOT,
            env,
            log_file,
            args.timeout,
        )
        row["returncode"] = "timeout" if timed_out else returncode
        row["duration_s"] = round(perf_counter() - start, 3)

        if timed_out:
            row["status"] = "timeout"
            row["error"] = f"timed out after {args.timeout:.1f}s"
            _track(run_id, row["error"])
            return row

        if returncode != 0:
            row["error"] = f"weld_opt.py exited with {returncode}"
            _track(run_id, f"optimization failed: {row['error']}")
            return row
        if not mat_file.exists():
            row["error"] = "MAT file was not produced"
            _track(run_id, row["error"])
            return row

        _track(run_id, "optimization finished; extracting metrics")
        row.update(_extract_metrics(mat_file))
        row["status"] = "ok"
        _track(
            run_id,
            f"ok: pipe_z={row['pipe_z']:.4f} m, "
            f"delta={row['pipe_z_delta']:+.4f} m, "
            f"duration={row['duration_s']:.1f}s"
        )
        return row

    except Exception as exc:
        row["duration_s"] = round(perf_counter() - start, 3)
        row["error"] = repr(exc)
        _track(run_id, f"failed: {exc}")
        return row


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run weld_opt.py many times and summarize optimized pipe height "
            "across weld-arc scenarios."
        ))
    parser.add_argument("--runs", type=int, default=20,
                        help="Number of random initial guesses per scenario.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--weld-opt", type=Path, default=DEFAULT_WELD_OPT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-attempts", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--scenario-set", choices=sorted(SCENARIO_SETS),
                        default="manual",
                        help="Default 'manual' uses the angles set in weld_opt.py.")
    parser.add_argument(
        "--scenario",
        action="append",
        help=(
            "Scenario as either a preset name or start:end. Angles may use "
            "pi, for example quarter_right_to_top or 0:pi/2. Can be repeated."
        ),
    )
    parser.add_argument(
        "--upside-down",
        type=_parse_bool,
        default=None,
        help=(
            "Override whether scenarios weld upside down. Use true or false. "
            "If omitted, presets use the value defined in SCENARIO_SETS."
        ),
    )
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--no-plots", action="store_true",
                        help="Do not generate PNG plots after the batch.")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if not args.weld_opt.exists():
        parser.error(f"--weld-opt does not exist: {args.weld_opt}")
    return args


def main() -> int:
    args = _parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _scenario_list(args)
    print(
        f"[batch] running {args.runs} optimization(s) for each of "
        f"{len(scenarios)} scenario(s)."
    )
    print(f"[batch] output directory: {output_dir}")

    rows = []
    run_index = 0
    stop = False
    for scenario in scenarios:
        for scenario_run in range(1, args.runs + 1):
            run_index += 1
            row = _run_once(args, run_index, scenario_run, scenario, output_dir)
            rows.append(row)
            _write_summaries(rows, output_dir)
            print(f"    [summary] updated {output_dir / 'summary.csv'}", flush=True)
            if args.stop_on_failure and row["status"] != "ok":
                stop = True
                break
        if stop:
            break

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    if ok_count and not args.no_plots:
        print("[batch] generating plots")
        completed = subprocess.run(
            [args.python, str(DEFAULT_PLOT_SCRIPT), str(output_dir)],
            check=False,
        )
        if completed.returncode != 0:
            print("[batch] warning: plot generation failed")

    print(
        f"[batch] finished: {ok_count}/{len(rows)} successful runs. "
        f"Summary: {output_dir / 'summary.csv'}"
    )
    return 0 if ok_count == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
