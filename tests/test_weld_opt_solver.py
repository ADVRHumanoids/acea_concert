"""Small dependency-free-of-robot check for weld solution selection."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import weld_opt_solver as solver


class _Log:
    path = "/tmp/not-written"

    def write(self, *args, **kwargs):
        pass


class _QRow:
    def __init__(self, model, index):
        self.model = model
        self.index = index

    def setBounds(self, low, high):
        self.model.fixed[self.index] = float(np.asarray(low).reshape(-1)[0])


class _Q:
    def __init__(self, model):
        self.model = model

    def __getitem__(self, index):
        return _QRow(self.model, index)

    def setInitialGuess(self, value):
        pass


class _Model:
    def __init__(self):
        self.q0 = np.zeros(7)
        self.fixed = {}
        self.q = _Q(self)


class _TaskInterface:
    def __init__(self):
        self.model = _Model()
        self.solution = None

    def bootstrap(self):
        q = np.zeros((7, 2))
        q[0, :] = self.model.fixed[0]
        q[1, :] = self.model.fixed[1]
        self.solution = {
            "q": q,
            "v": np.zeros((7, 2)),
            "a": np.zeros((7, 1)),
        }
        return True


class _CollisionChecker:
    def compute_collisions(self, q):
        colliding = q[0] < 0.0
        return colliding, ["collision"] if colliding else []


class _InverseDynamics:
    def call(self, q, velocity, acceleration):
        tau = np.zeros(7)
        tau[6] = 10.0 * abs(q[0])
        return tau


def main():
    solver.WeldOptAttemptLog = _Log
    result = solver.solve_weld_problem(
        task_interface=_TaskInterface(),
        n_intervals=1,
        base_bounds=(-2.0, 2.0, -1.0, 1.0),
        base_search_points=np.array([
            [2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ]),
        max_random_attempts=0,
        nominal_pipe_center=[1.5, 0.0, 1.0],
        optimize_pipe_height=False,
        make_collision_checker=lambda center: _CollisionChecker(),
        inverse_dynamics=_InverseDynamics(),
        critical_torque_indices=[6],
    )
    assert result["q"][0, 0] == 1.0


if __name__ == "__main__":
    main()
