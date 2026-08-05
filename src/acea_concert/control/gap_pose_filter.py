"""Small low-pass filter for measured gap pose."""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R


class GapPoseLowPass:
    def __init__(
        self,
        tau_s: float = 0.0,
        history_size: int = 1,
        max_position_jump_m: float = 0.0,
        max_angle_jump_deg: float = 0.0,
    ):
        if tau_s < 0.0:
            raise ValueError("tau_s must be >= 0")
        if history_size < 1:
            raise ValueError("history_size must be >= 1")
        if max_position_jump_m < 0.0:
            raise ValueError("max_position_jump_m must be >= 0")
        if max_angle_jump_deg < 0.0:
            raise ValueError("max_angle_jump_deg must be >= 0")

        self._tau_s = float(tau_s)
        self._history_size = int(history_size)
        self._max_position_jump_m = float(max_position_jump_m)
        self._max_angle_jump_rad = np.deg2rad(float(max_angle_jump_deg))
        self._xyz = None
        self._rot = None
        self._last_time_s = None
        self._history_xyz = deque(maxlen=self._history_size)
        self._history_rot = deque(maxlen=self._history_size)
        self._accepted_samples = 0
        self._rejected_samples = 0

    @property
    def enabled(self) -> bool:
        return (
            self._tau_s > 0.0
            or self._history_size > 1
            or self._max_position_jump_m > 0.0
            or self._max_angle_jump_rad > 0.0
        )

    @property
    def rejected_samples(self) -> int:
        return self._rejected_samples

    def update(self, xyz: np.ndarray, rotation_matrix: np.ndarray,
               time_s: float) -> tuple[np.ndarray, np.ndarray]:
        xyz = np.asarray(xyz, dtype=float)
        rot = R.from_matrix(np.asarray(rotation_matrix, dtype=float))
        time_s = float(time_s)

        if not self.enabled:
            self._set_state(xyz, rot, time_s)
            return xyz.copy(), rot.as_matrix()

        if self._is_outlier(xyz, rot):
            self._rejected_samples += 1
            self._last_time_s = time_s
            return self._xyz.copy(), self._rot.as_matrix()

        self._history_xyz.append(xyz.copy())
        self._history_rot.append(rot)
        self._accepted_samples += 1

        target_xyz, target_rot = self._history_estimate()
        if self._xyz is None or self._rot is None or self._last_time_s is None:
            self._set_state(target_xyz, target_rot, time_s)
            return self._current()

        dt = max(0.0, time_s - self._last_time_s)
        if self._tau_s > 0.0 and dt > 0.0:
            alpha = dt / (self._tau_s + dt)
            self._xyz = (1.0 - alpha) * self._xyz + alpha * target_xyz
            delta = self._rot.inv() * target_rot
            self._rot = self._rot * R.from_rotvec(alpha * delta.as_rotvec())
        else:
            self._xyz = target_xyz.copy()
            self._rot = target_rot
        self._last_time_s = time_s
        return self._current()

    def _set_state(self, xyz: np.ndarray, rot: R, time_s: float):
        self._xyz = xyz.copy()
        self._rot = rot
        self._last_time_s = time_s

    def _is_outlier(self, xyz: np.ndarray, rot: R) -> bool:
        if self._xyz is None or self._rot is None:
            return False
        if self._accepted_samples < min(3, self._history_size):
            return False
        if (self._max_position_jump_m > 0.0
                and np.linalg.norm(xyz - self._xyz) > self._max_position_jump_m):
            return True
        angle = (self._rot.inv() * rot).magnitude()
        return self._max_angle_jump_rad > 0.0 and angle > self._max_angle_jump_rad

    def _history_estimate(self) -> tuple[np.ndarray, R]:
        if not self._history_xyz:
            if self._xyz is None or self._rot is None:
                raise RuntimeError("gap pose filter has no pose estimate")
            return self._xyz.copy(), self._rot

        xyz = np.median(np.stack(self._history_xyz), axis=0)
        ref = self._rot if self._rot is not None else self._history_rot[-1]
        rotvecs = np.stack(
            [(ref.inv() * hist_rot).as_rotvec()
             for hist_rot in self._history_rot]
        )
        rot = ref * R.from_rotvec(np.median(rotvecs, axis=0))
        return xyz, rot

    def _current(self) -> tuple[np.ndarray, np.ndarray]:
        return self._xyz.copy(), self._rot.as_matrix()


def _self_check():
    filt = GapPoseLowPass(tau_s=1.0)
    eye = np.eye(3)
    xyz0, _ = filt.update(np.array([0.0, 0.0, 0.0]), eye, 0.0)
    xyz1, _ = filt.update(np.array([2.0, 0.0, 0.0]), eye, 1.0)
    assert np.allclose(xyz0, [0.0, 0.0, 0.0])
    assert np.allclose(xyz1, [1.0, 0.0, 0.0])

    target = R.from_euler("z", 90.0, degrees=True).as_matrix()
    _, rot = filt.update(np.array([2.0, 0.0, 0.0]), target, 2.0)
    yaw = R.from_matrix(rot).as_euler("zyx", degrees=True)[0]
    assert 40.0 < yaw < 50.0

    filt = GapPoseLowPass(
        tau_s=0.0,
        history_size=3,
        max_position_jump_m=0.5,
    )
    filt.update(np.array([0.0, 0.0, 0.0]), eye, 0.0)
    xyz, _ = filt.update(np.array([0.1, 0.0, 0.0]), eye, 0.1)
    assert np.allclose(xyz, [0.05, 0.0, 0.0])
    xyz, _ = filt.update(np.array([10.0, 0.0, 0.0]), eye, 0.2)
    assert np.allclose(xyz, [0.1, 0.0, 0.0])
    xyz, _ = filt.update(np.array([10.0, 0.0, 0.0]), eye, 0.3)
    assert np.allclose(xyz, [0.1, 0.0, 0.0])
    assert filt.rejected_samples == 1

    filt = GapPoseLowPass(history_size=5, max_position_jump_m=0.5)
    filt.update(np.array([10.0, 0.0, 0.0]), eye, 0.0)
    filt.update(np.array([0.0, 0.0, 0.0]), eye, 0.1)
    xyz, _ = filt.update(np.array([0.1, 0.0, 0.0]), eye, 0.2)
    assert np.allclose(xyz, [0.1, 0.0, 0.0])
    xyz, _ = filt.update(np.array([10.0, 0.0, 0.0]), eye, 0.3)
    assert np.allclose(xyz, [0.1, 0.0, 0.0])
    assert filt.rejected_samples == 1


if __name__ == "__main__":
    _self_check()
