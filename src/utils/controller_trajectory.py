import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


class CyclicPosturalTrajectory:
    """Forward/backward cyclic interpolation for actuated joint references."""

    def __init__(self, q_act: np.ndarray, joint_names: list[str], dt: float):
        self.q_act = np.asarray(q_act, dtype=float)
        self.joint_names = list(joint_names)
        self.dt = float(dt)
        self.num_nodes = self.q_act.shape[1]
        self.duration = (self.num_nodes - 1) * self.dt
        self.cycle_duration = 2.0 * self.duration

        q_cycle = np.concatenate([self.q_act, self.q_act[:, ::-1]], axis=1)
        t_nodes = np.linspace(0.0, self.cycle_duration, q_cycle.shape[1])
        self._interp = interp1d(
            t_nodes,
            q_cycle,
            axis=1,
            kind='linear',
            fill_value='extrapolate',
        )

    def postural_map(self, t_global: float) -> dict:
        """Return {joint_name: angle} for all actuated joints at time t."""
        t_mod = t_global % self.cycle_duration
        q_now = self._interp(t_mod)
        return {
            name: float(q_now[i])
            for i, name in enumerate(self.joint_names)
        }


class CyclicVectorTrajectory:
    """Forward/backward cyclic interpolation for vector-valued references."""

    def __init__(self, values: np.ndarray, dt: float):
        self.values = np.asarray(values, dtype=float)
        self.dt = float(dt)
        self.num_nodes = self.values.shape[1]
        self.duration = (self.num_nodes - 1) * self.dt
        self.cycle_duration = 2.0 * self.duration

        value_cycle = np.concatenate(
            [self.values, self.values[:, ::-1]], axis=1)
        t_nodes = np.linspace(0.0, self.cycle_duration, value_cycle.shape[1])
        self._interp = interp1d(
            t_nodes,
            value_cycle,
            axis=1,
            kind='linear',
            fill_value='extrapolate',
        )

    def value(self, t_global: float) -> np.ndarray:
        """Return the interpolated vector at time t."""
        t_mod = t_global % self.cycle_duration
        return np.asarray(self._interp(t_mod), dtype=float)


class CyclicQuaternionTrajectory:
    """Forward/backward cyclic interpolation for quaternion references."""

    def __init__(self, quaternions: np.ndarray, dt: float):
        self.quaternions = np.asarray(quaternions, dtype=float)
        if self.quaternions.shape[0] != 4 and self.quaternions.shape[1] == 4:
            self.quaternions = self.quaternions.T
        if self.quaternions.shape[0] != 4:
            raise ValueError("quaternions must have shape (4, N) or (N, 4)")

        self.dt = float(dt)
        self.num_nodes = self.quaternions.shape[1]
        self.duration = (self.num_nodes - 1) * self.dt
        self.cycle_duration = 2.0 * self.duration

        quat_cycle = np.concatenate(
            [self.quaternions, self.quaternions[:, ::-1]], axis=1)
        t_nodes = np.linspace(0.0, self.cycle_duration, quat_cycle.shape[1])
        self._slerp = Slerp(t_nodes, Rotation.from_quat(quat_cycle.T))

    def rotation(self, t_global: float) -> Rotation:
        """Return the interpolated orientation as a scipy Rotation."""
        t_mod = t_global % self.cycle_duration
        return self._slerp([t_mod])[0]

    def matrix(self, t_global: float) -> np.ndarray:
        """Return the interpolated orientation as a rotation matrix."""
        return self.rotation(t_global).as_matrix()


def ee_pose_from_postural(model_target, postural_map: dict,
                          distal_link: str, base_link: str):
    """Compute EE pose for a given postural joint map using a temporary model."""
    model_target.setJointPosition(postural_map)
    model_target.update()
    return model_target.getPose(distal_link, base_link)
