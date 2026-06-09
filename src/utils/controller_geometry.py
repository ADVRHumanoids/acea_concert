import numpy as np


BASE_X_AXIS_ROBOT = np.array([1.0, 0.0, 0.0])
BASE_Y_AXIS_ROBOT = np.array([0.0, 1.0, 0.0])
BASE_Z_AXIS_ROBOT = np.array([0.0, 0.0, 1.0])


def unit_vector(v, fallback=None):
    """Return v normalized, or fallback if v is too small."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm > 1e-9:
        return v / norm
    if fallback is None:
        return None
    return np.asarray(fallback, dtype=float)


def axis_orthogonal_to(axis: np.ndarray) -> np.ndarray:
    """Pick a deterministic unit vector orthogonal to axis."""
    for candidate in (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ):
        orth = candidate - np.dot(candidate, axis) * axis
        orth = unit_vector(orth)
        if orth is not None:
            return orth
    return BASE_X_AXIS_ROBOT.copy()


def gap_tangent_axis(gap_y_axis: np.ndarray) -> np.ndarray:
    """Return the pipe/gap tangent axis, expressed in base_link."""
    gap_y_axis = unit_vector(gap_y_axis, BASE_Y_AXIS_ROBOT)

    # The pipes are horizontal, so use the horizontal direction perpendicular
    # to the measured gap normal and keep its sign close to base +X.
    tangent = np.array([gap_y_axis[1], -gap_y_axis[0], 0.0])
    tangent = unit_vector(tangent)
    if tangent is None:
        tangent = (
            BASE_X_AXIS_ROBOT
            - np.dot(BASE_X_AXIS_ROBOT, gap_y_axis) * gap_y_axis
        )
        tangent = unit_vector(tangent, axis_orthogonal_to(gap_y_axis))
    if np.dot(tangent, BASE_X_AXIS_ROBOT) < 0.0:
        tangent = -tangent
    return tangent


def gap_frame_axes(gap_y_axis: np.ndarray):
    """Return a right-handed gap frame (x along pipe, y across gap, z up-ish)."""
    y_axis = unit_vector(gap_y_axis, BASE_Y_AXIS_ROBOT)
    x_axis = gap_tangent_axis(y_axis)
    z_axis = unit_vector(np.cross(x_axis, y_axis), BASE_Z_AXIS_ROBOT)
    x_axis = unit_vector(np.cross(y_axis, z_axis), x_axis)
    return x_axis, y_axis, z_axis


def rotation_with_y_axis(R_hint: np.ndarray,
                         gap_y_axis: np.ndarray) -> np.ndarray:
    """
    Align the EE local Y axis with the y-gap axis, preserving the current
    postural tool direction as much as possible.
    """
    R_hint = np.asarray(R_hint, dtype=float)
    gap_y_axis = unit_vector(gap_y_axis, BASE_Y_AXIS_ROBOT)

    # Keep the sign closest to the postural reference to avoid 180 deg flips.
    target_y = gap_y_axis
    if np.dot(R_hint[:, 1], target_y) < 0.0:
        target_y = -target_y

    target_z = R_hint[:, 2] - np.dot(R_hint[:, 2], target_y) * target_y
    target_z = unit_vector(target_z)
    if target_z is None:
        target_z = axis_orthogonal_to(target_y)

    target_x = unit_vector(np.cross(target_y, target_z))
    target_z = unit_vector(np.cross(target_x, target_y))
    return np.column_stack([target_x, target_y, target_z])


def rotation_correction(R_target: np.ndarray, R_reference: np.ndarray):
    """Return angle and rotation vector taking R_reference to R_target."""
    R_delta = (
        np.asarray(R_target, dtype=float)
        @ np.asarray(R_reference, dtype=float).T
    )
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    if angle < 1e-9:
        return 0.0, np.zeros(3)

    axis = np.array([
        R_delta[2, 1] - R_delta[1, 2],
        R_delta[0, 2] - R_delta[2, 0],
        R_delta[1, 0] - R_delta[0, 1],
    ]) / (2.0 * np.sin(angle))
    return angle, axis * angle
