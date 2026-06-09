import numpy as np
from scipy.interpolate import interp1d


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


def ee_pose_from_postural(model_target, postural_map: dict,
                          distal_link: str, base_link: str):
    """Compute EE pose for a given postural joint map using a temporary model."""
    model_target.setJointPosition(postural_map)
    model_target.update()
    return model_target.getPose(distal_link, base_link)
