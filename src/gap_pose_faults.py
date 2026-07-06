"""Optional noise/dropout injection for /gap/pose_robot test messages."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class GapPoseFaultConfig:
    position_std: float = 0.0
    orientation_std: float = 0.0
    dropout_probability: float = 0.0
    disconnect_every: float = 0.0
    disconnect_duration: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        if self.position_std < 0.0:
            raise ValueError("position_std must be >= 0")
        if self.orientation_std < 0.0:
            raise ValueError("orientation_std must be >= 0")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1]")
        if self.disconnect_every < 0.0:
            raise ValueError("disconnect_every must be >= 0")
        if self.disconnect_duration < 0.0:
            raise ValueError("disconnect_duration must be >= 0")
        if self.disconnect_every == 0.0 and self.disconnect_duration > 0.0:
            raise ValueError("disconnect_duration requires disconnect_every")

    @property
    def enabled(self) -> bool:
        return any((
            self.position_std > 0.0,
            self.orientation_std > 0.0,
            self.dropout_probability > 0.0,
            self.disconnect_every > 0.0 and self.disconnect_duration > 0.0,
        ))


class GapPoseFaultInjector:
    def __init__(self, config: GapPoseFaultConfig):
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._start_s = perf_counter()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def apply(self, xyz: np.ndarray,
              quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if self._is_disconnected():
            return None
        if self._config.dropout_probability > 0.0:
            if self._rng.random() < self._config.dropout_probability:
                return None

        noisy_xyz = np.array(xyz, dtype=float, copy=True)
        noisy_quat = np.array(quat_xyzw, dtype=float, copy=True)

        if self._config.position_std > 0.0:
            noisy_xyz += self._rng.normal(
                0.0, self._config.position_std, size=3)

        if self._config.orientation_std > 0.0:
            rot_noise = R.from_rotvec(
                self._rng.normal(0.0, self._config.orientation_std, size=3))
            noisy_quat = (rot_noise * R.from_quat(noisy_quat)).as_quat()

        return noisy_xyz, noisy_quat

    def _is_disconnected(self) -> bool:
        cfg = self._config
        if cfg.disconnect_every <= 0.0 or cfg.disconnect_duration <= 0.0:
            return False
        return (perf_counter() - self._start_s) % cfg.disconnect_every < min(
            cfg.disconnect_duration, cfg.disconnect_every)


def _self_check():
    xyz = np.array([1.0, 2.0, 3.0])
    quat = np.array([0.0, 0.0, 0.0, 1.0])

    unchanged = GapPoseFaultInjector(GapPoseFaultConfig()).apply(xyz, quat)
    assert unchanged is not None
    assert np.allclose(unchanged[0], xyz)
    assert np.allclose(unchanged[1], quat)

    dropped = GapPoseFaultInjector(
        GapPoseFaultConfig(dropout_probability=1.0)).apply(xyz, quat)
    assert dropped is None


if __name__ == "__main__":
    _self_check()
