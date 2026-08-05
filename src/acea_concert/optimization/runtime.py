from dataclasses import dataclass
from math import pi
import os
from pathlib import Path


def _env_flag(environ, name, default=False):
    value = environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(environ, name, default):
    value = environ.get(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class WeldOptRuntime:
    batch: bool
    skip_rviz_scene: bool
    skip_replay: bool
    max_random_initial_pose_attempts: int
    seed: int | None


@dataclass(frozen=True)
class WeldScenario:
    name: str
    angle_start: float
    angle_end: float
    upside_down: bool


def weld_opt_runtime_from_env(environ=None) -> WeldOptRuntime:
    environ = os.environ if environ is None else environ
    batch = _env_flag(environ, "WELD_OPT_BATCH")
    seed_env = environ.get("WELD_OPT_SEED")
    return WeldOptRuntime(
        batch=batch,
        skip_rviz_scene=batch or _env_flag(environ, "WELD_OPT_SKIP_RVIZ"),
        skip_replay=batch or _env_flag(environ, "WELD_OPT_SKIP_REPLAY"),
        max_random_initial_pose_attempts=int(
            environ.get("WELD_OPT_MAX_ATTEMPTS", "0")),
        seed=int(seed_env) if seed_env is not None else None,
    )


def weld_scenario_from_env(environ=None) -> WeldScenario:
    environ = os.environ if environ is None else environ
    return WeldScenario(
        name=environ.get("WELD_OPT_SCENARIO_NAME", "manual"),
        angle_start=_env_float(environ, "WELD_OPT_ANGLE_START", pi),
        angle_end=_env_float(environ, "WELD_OPT_ANGLE_END", 0.5 * pi),
        upside_down=_env_flag(environ, "WELD_OPT_WELD_UPSIDE_DOWN", False),
    )


def weld_output_path(package_root: Path, environ=None) -> Path:
    environ = os.environ if environ is None else environ
    output_path = environ.get("WELD_OPT_OUTPUT_PATH")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    output_name = environ.get("WELD_OPT_OUTPUT_NAME", "weld_concert")
    output_dir = package_root / "mat_files"
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"{output_name}.mat"
