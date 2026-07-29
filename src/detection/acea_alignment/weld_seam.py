"""Weld-seam / gap-plane frame from the pipe axis + a seam surface point.

For a butt weld between two coaxial pipe segments the gap is a circumferential
seam lying in the plane **perpendicular to the pipe axis**. So the "plane of the
gap w.r.t. the camera" (what the IK tracking side asked for) is fully defined by
one point on the seam and the gap-plane normal:

    plane point  = a 3D point on the seam (camera frame)
    plane normal = the pipe axis direction (camera frame)

Both quantities are already produced by the detector
(``acea_pipe_junction_node``):
    * ``pipe_axis_camera_xyz``                     -> plane normal
    * ``coarse_seam_visible_surface_camera_xyz_m`` -> a point on the seam surface

This module turns those two vectors into a full, right-handed **seam frame**
(a pose) so the welding torch gets everything in one message:

    convention (camera frame, columns of R = [x | y | z]):
      z = plane_normal   = pipe axis            (gap/seam lies in the x-y plane)
      x = surface_normal = outward pipe-surface normal (points toward the camera;
                           the torch approaches the surface along -x)
      y = seam_tangent   = z x x = circumferential direction the torch travels

The math is pure NumPy, deterministic, and ROS-free so it can be unit-tested
offline (run ``python3 weld_seam.py``).  The ROS node only imports
``seam_frame_from_axis_and_surface`` and publishes the resulting pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


def _unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return v
    return v / n


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a right-handed 3x3 rotation matrix into a ROS ``xyzw`` quaternion.

    Uses Shepperd's branch-by-largest-diagonal method for numerical stability.
    """
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 0.0)) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 0.0)) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 0.0)) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        return [0.0, 0.0, 0.0, 1.0]
    q = q / n
    # Canonical sign: keep w >= 0 so the same orientation maps to one quaternion.
    if q[3] < 0.0:
        q = -q
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


@dataclass
class WeldSeamFrame:
    """Gap-plane / weld-seam frame in the camera frame (see module docstring)."""

    origin_xyz_m: np.ndarray      # point on the seam surface = a point on the gap plane
    quat_xyzw: list[float]        # orientation of R = [x | y | z]
    plane_normal: np.ndarray      # z axis = pipe axis = gap-plane normal
    surface_normal: np.ndarray    # x axis = outward surface normal (torch approaches along -x)
    seam_tangent: np.ndarray      # y axis = circumferential travel direction
    degenerate: bool              # True if the radial direction had to be faked (axis seen end-on)


def _fallback_radial(z: np.ndarray) -> np.ndarray:
    """A unit vector orthogonal to ``z``, used only in the degenerate case."""
    # Pick the world axis least aligned with z, then orthogonalize it against z.
    basis = np.eye(3, dtype=np.float64)
    ref = basis[int(np.argmin(np.abs(z)))]
    x = ref - float(np.dot(ref, z)) * z
    return _unit(x)


def seam_frame_from_axis_and_surface(
    axis_xyz: np.ndarray,
    surface_xyz: np.ndarray,
    axis_point_xyz: np.ndarray | None = None,
) -> WeldSeamFrame | None:
    """Build the weld-seam / gap-plane frame from the pipe axis and a seam point.

    Parameters
    ----------
    axis_xyz : pipe axis direction in the camera frame (need not be unit; sign is
        preserved -- the detector already fixes it so axis_x >= 0).
    surface_xyz : a 3D point on the visible pipe surface at the seam (camera
        frame), e.g. ``coarse_seam_visible_surface_camera_xyz_m``.
    axis_point_xyz : optional point on the cylinder axis at the seam.  When
        supplied, the outward radial is the direction from the closest point
        on that axis line toward the camera origin.  This is stable even when
        depth is missing or noisy on the visible seam edge.  Omitting it
        preserves the legacy surface-point estimate.

    Returns
    -------
    WeldSeamFrame, or ``None`` if the axis is null / inputs are non-finite.
    """
    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    surface = np.asarray(surface_xyz, dtype=np.float64).reshape(3)
    axis_point = None
    if axis_point_xyz is not None:
        axis_point = np.asarray(axis_point_xyz, dtype=np.float64).reshape(3)
    if not (
        np.all(np.isfinite(axis))
        and np.all(np.isfinite(surface))
        and (axis_point is None or np.all(np.isfinite(axis_point)))
    ):
        return None
    z = _unit(axis)
    if float(np.linalg.norm(z)) < _EPS:
        return None

    if axis_point is None:
        # Legacy estimate: infer the visible outward direction from the camera
        # origin.  This is retained for backward compatibility but changes when
        # the same cylinder is translated in the camera frame.
        radial = -(surface - float(np.dot(surface, z)) * z)
    else:
        # Cylinder geometry: find the point on the axis line closest to the
        # camera origin, then point from the axis toward the camera.
        closest_axis_point = axis_point - float(np.dot(axis_point, z)) * z
        radial = -closest_axis_point
    pn = float(np.linalg.norm(radial))
    degenerate = pn < _EPS * max(float(np.linalg.norm(surface)), 1.0)
    if degenerate:
        # Pipe seen (nearly) end-on: radial undefined -> pick any frame in-plane.
        x = _fallback_radial(z)
    else:
        x = radial / pn  # outward surface normal (toward the camera)

    y = _unit(np.cross(z, x))          # seam tangent (circumferential)
    x = _unit(np.cross(y, z))          # re-orthonormalize x against (y, z)
    rotation = np.column_stack([x, y, z])
    quat = rotation_matrix_to_quaternion_xyzw(rotation)
    return WeldSeamFrame(
        origin_xyz_m=surface.copy(),
        quat_xyzw=quat,
        plane_normal=z,
        surface_normal=x,
        seam_tangent=y,
        degenerate=bool(degenerate),
    )


# --------------------------------------------------------------------------- #
# Offline self-test:  python3 weld_seam.py
# --------------------------------------------------------------------------- #
def _quat_xyzw_to_R(q: list[float]) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _selftest() -> int:
    results: list[tuple[bool, str]] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        results.append((bool(ok), label + (f"  [{detail}]" if detail else "")))

    # 1) canonical case: axis=+x, surface straight ahead at z=1m.
    f = seam_frame_from_axis_and_surface([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    assert f is not None
    check(np.allclose(f.plane_normal, [1.0, 0.0, 0.0]), "canonical plane_normal = +x",
          str(np.round(f.plane_normal, 4)))
    check(np.allclose(f.surface_normal, [0.0, 0.0, -1.0]), "canonical surface_normal = -z (toward camera)",
          str(np.round(f.surface_normal, 4)))
    expected_q = np.array([0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)])
    q = np.array(f.quat_xyzw)
    check(np.allclose(q, expected_q, atol=1e-6) or np.allclose(q, -expected_q, atol=1e-6),
          "canonical quaternion = (0, v2/2, 0, v2/2)", str(np.round(q, 4)))

    rng = np.random.default_rng(0)
    max_ortho_err = 0.0
    max_rt_err = 0.0
    max_coplanar_err = 0.0
    for _ in range(2000):
        axis = rng.normal(size=3)
        if np.linalg.norm(axis) < 1e-3:
            continue
        if axis[0] < 0:
            axis = -axis  # mimic the node's sign fix
        surface = np.array([rng.normal(0, 0.3), rng.normal(0, 0.3), rng.uniform(0.5, 3.0)])
        fr = seam_frame_from_axis_and_surface(axis, surface)
        assert fr is not None
        x, y, z = fr.surface_normal, fr.seam_tangent, fr.plane_normal
        R = np.column_stack([x, y, z])
        # 2) orthonormal + right-handed + z == axis direction
        max_ortho_err = max(max_ortho_err, float(np.abs(R.T @ R - np.eye(3)).max()))
        det_ok = abs(float(np.linalg.det(R)) - 1.0) < 1e-9
        z_ok = np.allclose(z, _unit(axis), atol=1e-9)
        y_ok = np.allclose(y, np.cross(z, x), atol=1e-9)
        if not (det_ok and z_ok and y_ok):
            check(False, "orthonormal/handed/axis", f"det={np.linalg.det(R):.6f}")
            break
        # 3) quaternion round-trip
        R2 = _quat_xyzw_to_R(fr.quat_xyzw)
        max_rt_err = max(max_rt_err, float(np.abs(R2 - R).max()))
        # 4) seam circle points are coplanar (in the x-y plane through origin)
        r = 0.15
        for theta in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            p = fr.origin_xyz_m + r * (math.cos(theta) * x + math.sin(theta) * y)
            max_coplanar_err = max(max_coplanar_err, abs(float(np.dot(p - fr.origin_xyz_m, z))))
    else:
        check(max_ortho_err < 1e-9, "R orthonormal over 2000 random cases", f"max|RtR-I|={max_ortho_err:.2e}")
        check(max_rt_err < 1e-9, "quaternion round-trips to R", f"max err={max_rt_err:.2e}")
        check(max_coplanar_err < 1e-9, "seam circle lies in the gap plane", f"max |.normal|={max_coplanar_err:.2e}")

    # 5) degenerate (pipe seen end-on): must not crash, still orthonormal, z=axis
    fd = seam_frame_from_axis_and_surface([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    assert fd is not None
    Rd = np.column_stack([fd.surface_normal, fd.seam_tangent, fd.plane_normal])
    check(fd.degenerate and np.abs(Rd.T @ Rd - np.eye(3)).max() < 1e-9
          and np.allclose(fd.plane_normal, [1.0, 0.0, 0.0]),
          "degenerate axis-end-on handled (orthonormal, z=axis)")
    check(all(math.isfinite(c) for c in fd.quat_xyzw), "degenerate quaternion finite", str(np.round(fd.quat_xyzw, 4)))

    # 6) determinism: identical inputs -> identical outputs
    a = seam_frame_from_axis_and_surface([0.2, -0.1, 0.97], [0.1, 0.05, 1.4])
    b = seam_frame_from_axis_and_surface([0.2, -0.1, 0.97], [0.1, 0.05, 1.4])
    check(a is not None and b is not None and a.quat_xyzw == b.quat_xyzw
          and np.array_equal(a.plane_normal, b.plane_normal), "deterministic (bit-identical repeat)")

    # 7) The fitted-axis construction is independent of which point along the
    # same axis line represents the cylinder.
    ft = seam_frame_from_axis_and_surface(
        [0.0, 1.0, 0.0],
        [0.3, 0.2, 2.0],
        [0.4, 0.2, 2.0],
    )
    ft_shifted = seam_frame_from_axis_and_surface(
        [0.0, 1.0, 0.0],
        [0.3, 1.2, 2.0],
        [0.4, 1.2, 2.0],
    )
    assert ft is not None
    expected_radial = _unit(np.array([-0.4, 0.0, -2.0]))
    check(
        ft_shifted is not None
        and np.allclose(ft.surface_normal, expected_radial, atol=1e-9)
        and np.allclose(ft_shifted.surface_normal, expected_radial, atol=1e-9),
        "axis-to-camera radial is invariant along the axis line",
        str(np.round(ft.surface_normal, 4)),
    )

    # 8) None on bad input
    check(seam_frame_from_axis_and_surface([0, 0, 0], [0, 0, 1]) is None, "null axis -> None")
    check(seam_frame_from_axis_and_surface([1, 0, 0], [np.nan, 0, 1]) is None, "non-finite surface -> None")
    check(
        seam_frame_from_axis_and_surface([1, 0, 0], [0, 0, 1], [np.nan, 0, 1]) is None,
        "non-finite axis point -> None",
    )

    ok = sum(1 for r, _ in results if r)
    for passed, label in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print(f"weld_seam self-test: {ok}/{len(results)} passed")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
