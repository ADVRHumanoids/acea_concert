#!/usr/bin/env python3
"""Plot summary data produced by run_weld_opt_batch.py."""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _float(row, key, default=np.nan):
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _read_rows(summary_csv: Path) -> list[dict]:
    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    ok_rows = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        pipe_z = _float(row, "pipe_z")
        if not math.isfinite(pipe_z):
            continue
        ok_rows.append(row)
    return ok_rows


def _scenario_names(rows: list[dict]) -> list[str]:
    names = []
    for row in rows:
        name = row.get("scenario") or "manual"
        if name not in names:
            names.append(name)
    return names


def _colors(names: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap("tab10" if len(names) <= 10 else "tab20")
    return {name: cmap(i % cmap.N) for i, name in enumerate(names)}


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[plot_batch] wrote {path}")


def _plot_pipe_height_by_run(rows: list[dict], out_dir: Path):
    names = _scenario_names(rows)
    colors = _colors(names)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    for name in names:
        selected = [row for row in rows if (row.get("scenario") or "manual") == name]
        x = [_float(row, "run") for row in selected]
        y = [_float(row, "pipe_z") for row in selected]
        ax.scatter(x, y, s=36, label=name, color=colors[name], alpha=0.85)

    nominal = _float(rows[0], "pipe_z_nominal")
    low = _float(rows[0], "pipe_z_bound_low")
    high = _float(rows[0], "pipe_z_bound_high")
    if math.isfinite(nominal):
        ax.axhline(nominal, color="0.25", linestyle="--", linewidth=1.2,
                   label="nominal height")
    if math.isfinite(low):
        ax.axhline(low, color="0.65", linestyle=":", linewidth=1.0,
                   label="height bounds")
    if math.isfinite(high):
        ax.axhline(high, color="0.65", linestyle=":", linewidth=1.0)

    ax.set_title("Optimized Pipe Height By Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Pipe height z [m]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_dir / "pipe_height_by_run.png")


def _plot_pipe_height_delta_hist(rows: list[dict], out_dir: Path):
    names = _scenario_names(rows)
    colors = _colors(names)
    fig, ax = plt.subplots(figsize=(8, 4.8))

    plotted = False
    for name in names:
        values = np.array([
            _float(row, "pipe_z_delta")
            for row in rows
            if (row.get("scenario") or "manual") == name
        ], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax.hist(values, bins=min(12, max(3, values.size)), alpha=0.45,
                label=name, color=colors[name])
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0)
    ax.set_title("Optimized Height Change From Nominal")
    ax.set_xlabel("pipe_z - pipe_z_nominal [m]")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    _save(fig, out_dir / "pipe_height_delta_hist.png")


def _plot_pipe_height_by_scenario(rows: list[dict], out_dir: Path):
    names = _scenario_names(rows)
    if len(names) < 2:
        return

    data = []
    labels = []
    for name in names:
        values = np.array([
            _float(row, "pipe_z")
            for row in rows
            if (row.get("scenario") or "manual") == name
        ], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            data.append(values)
            labels.append(name)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5.0))
    ax.boxplot(data, labels=labels, showmeans=True)
    for index, values in enumerate(data, start=1):
        jitter = np.linspace(-0.12, 0.12, values.size) if values.size > 1 else [0.0]
        ax.scatter(np.asarray(jitter) + index, values, s=24, color="0.2", alpha=0.55)
    ax.set_title("Optimized Pipe Height By Trajectory Scenario")
    ax.set_ylabel("Pipe height z [m]")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    _save(fig, out_dir / "pipe_height_by_scenario.png")


def _plot_height_over_initial_base(rows: list[dict], out_dir: Path):
    x = np.array([_float(row, "initial_base_x") for row in rows], dtype=float)
    y = np.array([_float(row, "initial_base_y") for row in rows], dtype=float)
    z = np.array([_float(row, "pipe_z") for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5.8))
    sc = ax.scatter(x[valid], y[valid], c=z[valid], s=48, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="Optimized pipe height z [m]")
    ax.set_title("Pipe Height Over Initial Base Guess")
    ax.set_xlabel("Initial base x [m]")
    ax.set_ylabel("Initial base y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    _save(fig, out_dir / "pipe_height_vs_initial_base_xy.png")


def _plot_torque_vs_height(rows: list[dict], out_dir: Path):
    x = np.array([_float(row, "pipe_z") for row in rows], dtype=float)
    y = np.array([_float(row, "max_abs_critical_static_tau") for row in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5.0))
    ax.scatter(x[valid], y[valid], s=42, alpha=0.8)
    ax.set_title("Critical Static Torque Vs Optimized Pipe Height")
    ax.set_xlabel("Pipe height z [m]")
    ax.set_ylabel("max |critical static tau| [Nm]")
    ax.grid(True, alpha=0.25)
    _save(fig, out_dir / "critical_static_torque_vs_pipe_height.png")


def plot_batch(summary_csv: Path, out_dir: Path):
    rows = _read_rows(summary_csv)
    if not rows:
        raise RuntimeError(f"No successful rows found in {summary_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_pipe_height_by_run(rows, out_dir)
    _plot_pipe_height_delta_hist(rows, out_dir)
    _plot_pipe_height_by_scenario(rows, out_dir)
    _plot_height_over_initial_base(rows, out_dir)
    _plot_torque_vs_height(rows, out_dir)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create plots from a run_weld_opt_batch.py summary.csv.")
    parser.add_argument(
        "path",
        type=Path,
        help="Batch output directory or path to summary.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_csv = args.path / "summary.csv" if args.path.is_dir() else args.path
    if not summary_csv.exists():
        print(f"summary file not found: {summary_csv}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or summary_csv.parent / "plots"
    plot_batch(summary_csv, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
