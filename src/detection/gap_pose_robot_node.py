#!/usr/bin/env python3
"""
gap_pose_robot_node.py — perception-driven ``/gap/pose_robot`` publisher.

Drop-in replacement for the Gazebo ground-truth ``gap_pose_publisher.py``.
Instead of reading pipe model poses from gz transport, it consumes the live
weld-seam gap geometry produced by the pipe-junction detector
(``acea_pipe_junction_node.py``, camera frame) and republishes the SAME contract
the rest of ``acea_concert`` already expects:

    /gap/pose_robot   geometry_msgs/PoseStamped   frame_id: base_link

The gap frame matches ``gap_pose_publisher.py`` exactly:

    y = pipe axis           (gap-plane normal, "centre-to-centre")
    x = radial weld tangent (outward pipe-surface normal, projected perpendicular to y)
    z = x X y               (right-handed)

Camera -> base_link transform uses tf2 (camera optical frame -> ``target_frame``).
A configurable static extrinsic fallback exists for lab/offline tests, but it is
disabled by default and the identity fallback is blocked unless explicitly
allowed, because publishing camera-frame coordinates as ``base_link`` would be
unsafe for the controller.

Optional convenience: the 4x4 homogeneous ``base_link <- gap`` transform is
published on a SEPARATE topic

    /gap/pose_robot_matrix   std_msgs/Float64MultiArray   (16 values, row-major)

It is deliberately kept OFF the pose topic — it is redundant with the
quaternion, and the canonical pose contract is PoseStamped(position + quaternion).

Run the geometry self-test (no ROS needed):
    python3 gap_pose_robot_node.py --selftest
"""
from __future__ import annotations

from copy import deepcopy
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the package's pure-NumPy, unit-tested quaternion helper (no scipy dep).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from acea_alignment.weld_seam import rotation_matrix_to_quaternion_xyzw  # noqa: E402

# Shared single-instance guard: a second bridge would publish a conflicting
# /gap/pose_robot. ROS-free helper; degrade to no-ops if it is ever missing.
try:
    from acea_detection_runtime import (  # noqa: E402
        DuplicateInstanceError,
        SingleInstanceLock,
        count_named_nodes,
        instance_id,
    )
except Exception:  # pragma: no cover - degrade gracefully if helper is absent
    import os as _os
    import socket as _socket

    class DuplicateInstanceError(RuntimeError):  # type: ignore
        pass

    class SingleInstanceLock:  # type: ignore
        def __init__(self, *_a, **_k):
            self.acquired = True
            self.holder = ""

        def release(self):
            ...

    def count_named_nodes(_node):  # type: ignore
        return 1

    def instance_id():  # type: ignore
        return f"{_socket.gethostname()}:{_os.getpid()}"

_BRIDGE_LOCK_NAME = "gap_pose_robot_node"
_EPS = 1e-9


# ── Pure geometry (ROS-free, unit-testable) ────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < _EPS or not np.isfinite(n):
        return None
    return v / n


def _quat_xyzw_to_R(q) -> np.ndarray:
    """Rotation matrix from a ROS xyzw quaternion."""
    x, y, z, w = (float(c) for c in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < _EPS:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _vector3_or_none(value) -> np.ndarray | None:
    try:
        vec = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(vec)):
        return None
    return vec


def _is_identity_static_transform(xyz: np.ndarray, quat_xyzw: np.ndarray) -> bool:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    q_norm = float(np.linalg.norm(q))
    if q_norm > _EPS:
        q = q / q_norm
    return bool(
        np.allclose(xyz, [0.0, 0.0, 0.0], atol=1e-12)
        and (
            np.allclose(q, [0.0, 0.0, 0.0, 1.0], atol=1e-12)
            or np.allclose(q, [0.0, 0.0, 0.0, -1.0], atol=1e-12)
        )
    )


def gap_frame_from_axis_and_radial(axis: np.ndarray, radial: np.ndarray):
    """Build the colleague's gap frame and return (R_3x3, quat_xyzw) or None.

    Convention (identical to gap_pose_publisher.py ``_gap_frame_axes``):
        y = unit(axis)                          # pipe axis = gap normal
        x = unit(radial projected perpendicular to y)
        z = unit(x X y)                         # right-handed
        x = unit(y X z)                         # re-orthogonalise
    Columns of R are [x | y | z].
    """
    y = _unit(np.asarray(axis, dtype=np.float64).reshape(3))
    if y is None:
        return None
    r = np.asarray(radial, dtype=np.float64).reshape(3)
    x = _unit(r - np.dot(r, y) * y)
    if x is None:  # radial parallel to axis -> pick any vector perpendicular to y
        seed = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = _unit(seed - np.dot(seed, y) * y)
        if x is None:
            return None
    z = _unit(np.cross(x, y))
    if z is None:
        return None
    x = _unit(np.cross(y, z))
    if x is None:
        return None
    R = np.column_stack([x, y, z])
    return R, rotation_matrix_to_quaternion_xyzw(R)


def _reference_vector_or_none(value) -> np.ndarray | None:
    vec = _vector3_or_none(value)
    if vec is None:
        return None
    return _unit(vec)


def _canonicalize_axis_and_radial(
    axis: np.ndarray,
    radial: np.ndarray,
    axis_reference: np.ndarray | None,
    radial_reference: np.ndarray | None,
    previous_axis: np.ndarray | None = None,
    previous_radial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, bool]]:
    """Resolve the cylinder sign ambiguity before building /gap/pose_robot.

    A fitted cylinder has no intrinsic +axis direction: ``axis`` and ``-axis``
    describe the same pipe. The controller, however, expects the historical
    Gazebo gap frame where +Y has a fixed meaning. First align to configured
    reference directions in the target frame; if a reference is disabled, keep
    temporal continuity with the last published axes.
    """
    a = np.asarray(axis, dtype=np.float64).reshape(3).copy()
    r = np.asarray(radial, dtype=np.float64).reshape(3).copy()
    flipped = {"axis_flipped": False, "radial_flipped": False}

    a_unit = _unit(a)
    if a_unit is None:
        return a, r, flipped

    axis_ref = axis_reference if axis_reference is not None else previous_axis
    if axis_ref is not None and float(np.dot(a_unit, axis_ref)) < 0.0:
        a = -a
        a_unit = -a_unit
        flipped["axis_flipped"] = True

    # Radial sign is meaningful only after removing the axis component.
    r_proj = r - float(np.dot(r, a_unit)) * a_unit
    r_unit = _unit(r_proj)
    radial_ref = radial_reference if radial_reference is not None else previous_radial
    if r_unit is not None and radial_ref is not None:
        ref_proj = radial_ref - float(np.dot(radial_ref, a_unit)) * a_unit
        ref_unit = _unit(ref_proj)
        if ref_unit is not None and float(np.dot(r_unit, ref_unit)) < 0.0:
            r = -r
            flipped["radial_flipped"] = True

    return a, r, flipped


def _selftest() -> int:
    """Validate the gap-frame convention without ROS. Returns 0 on success."""
    checks: list[tuple[str, bool]] = []

    # axis=+Y, radial=+X  ->  frame == identity (x=+X, y=+Y, z=+Z)
    out = gap_frame_from_axis_and_radial([0, 1, 0], [1, 0, 0])
    assert out is not None
    R, q = out
    checks.append(("axis=+Y radial=+X gives identity R", np.allclose(R, np.eye(3), atol=1e-9)))
    checks.append(("identity quat is (0,0,0,1)", np.allclose(q, [0, 0, 0, 1], atol=1e-9)))

    # Columns honour the convention for a tilted, non-orthogonal radial seed.
    axis = _unit(np.array([0.2, 1.0, -0.1]))
    radial_seed = np.array([1.0, 0.3, 0.0])  # not perpendicular to axis on purpose
    out = gap_frame_from_axis_and_radial(axis, radial_seed)
    assert out is not None
    R, q = out
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    checks.append(("col y == pipe axis", np.allclose(y, axis, atol=1e-9)))
    checks.append(("x perpendicular to y", abs(float(np.dot(x, y))) < 1e-9))
    checks.append(("z == x X y (right handed)", np.allclose(z, np.cross(x, y), atol=1e-9)))
    checks.append(("x stays in the radial half-space", float(np.dot(x, radial_seed)) > 0.0))
    checks.append(("R orthonormal", np.allclose(R.T @ R, np.eye(3), atol=1e-9)))
    checks.append(("det(R) == +1", abs(float(np.linalg.det(R)) - 1.0) < 1e-9))

    # quat -> R round trip matches.
    checks.append(("quat<->R round trip", np.allclose(_quat_xyzw_to_R(q), R, atol=1e-9)))

    # A cylinder fit can return the same physical pipe with both signs flipped.
    # References canonicalize it back to the Gazebo contract (+Y axis, +X radial).
    axis_c, radial_c, info = _canonicalize_axis_and_radial(
        np.array([0.0, -1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    out = gap_frame_from_axis_and_radial(axis_c, radial_c)
    assert out is not None
    R, _ = out
    checks.append(("reference canonicalizes flipped cylinder signs", np.allclose(R, np.eye(3), atol=1e-9)))
    checks.append(("canonicalization reports both flips", info["axis_flipped"] and info["radial_flipped"]))

    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"gap_pose_robot self-test: {sum(p for _, p in checks)}/{len(checks)} checks passed")
    return 0 if ok else 1


# ── ROS node ───────────────────────────────────────────────────────────────────

def _build_node_class():
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from std_msgs.msg import Float64MultiArray, String
    import tf2_ros

    class GapPoseRobotNode(Node):
        """Transforms the detector's camera-frame gap geometry into base_link."""

        def __init__(self) -> None:
            super().__init__("gap_pose_robot_node")
            p = self.declare_parameter
            self.instance_id = instance_id()

            # Refuse a second bridge on this host: two would publish a conflicting
            # /gap/pose_robot. Override with allow_duplicate:=true.
            self.allow_duplicate = bool(p("allow_duplicate", False).value)
            self._single_instance = SingleInstanceLock(_BRIDGE_LOCK_NAME)
            if not self._single_instance.acquired and not self.allow_duplicate:
                raise DuplicateInstanceError(
                    f"another '{_BRIDGE_LOCK_NAME}' is already running on this host "
                    f"(holder {self._single_instance.holder or 'unknown'}). Two of them "
                    "publish a conflicting /gap/pose_robot. Stop the other first:  pkill -f "
                    f"{_BRIDGE_LOCK_NAME}  (or close its terminal/launch), then start one. "
                    "To run two on purpose, pass -p allow_duplicate:=true."
                )

            self.input_topic = p("input_gap_plane_topic", "/acea/weld_seam/gap_plane").value
            self.output_pose_topic = p("output_pose_topic", "/gap/pose_robot").value
            self.output_matrix_topic = p("output_matrix_topic", "/gap/pose_robot_matrix").value
            self.publish_matrix = bool(p("publish_matrix", True).value)
            self.target_frame = p("target_frame", "base_link").value
            self.camera_frame_override = p("camera_frame_override", "").value
            self.use_tf = bool(p("use_tf", True).value)
            self.tf_timeout_s = float(p("tf_timeout_s", 0.1).value)
            self.static_fallback_enabled = bool(p("static_fallback_enabled", False).value)
            self.allow_identity_static_fallback = bool(
                p("allow_identity_static_fallback", False).value)
            self.static_xyz = np.asarray(
                p("static_cam_to_base_xyz", [0.0, 0.0, 0.0]).value, dtype=np.float64)
            self.static_quat = np.asarray(
                p("static_cam_to_base_quat_xyzw", [0.0, 0.0, 0.0, 1.0]).value, dtype=np.float64)
            self.axis_sign = float(p("axis_sign", 1.0).value)
            self.radial_sign = float(p("radial_sign", 1.0).value)
            self.axis_reference_target_xyz = _reference_vector_or_none(
                p("axis_reference_target_xyz", [0.0, 1.0, 0.0]).value)
            self.radial_reference_target_xyz = _reference_vector_or_none(
                p("radial_reference_target_xyz", [1.0, 0.0, 0.0]).value)
            self.use_reference_sign_canonicalization = bool(
                p("use_reference_sign_canonicalization", True).value)
            self.use_temporal_sign_continuity = bool(
                p("use_temporal_sign_continuity", True).value)
            self.require_pose_valid = bool(p("require_pose_valid", True).value)
            self.require_metric_3d = bool(p("require_metric_3d", True).value)
            self.reject_assumed_depth = bool(p("reject_assumed_depth", True).value)
            self.broadcast_tf = bool(p("broadcast_tf", False).value)
            self.gap_tf_child_frame = p("gap_tf_child_frame", "gap_seam").value
            self.hold_last_pose = bool(p("hold_last_pose", False).value)
            self.hold_publish_rate_hz = float(p("hold_publish_rate_hz", 20.0).value)
            self.hold_max_age_s = float(p("hold_max_age_s", 2.0).value)

            self._warned_fallback = False
            self._warned_no_transform = False
            self._warned_identity_fallback = False
            self._last_pose_msg = None
            self._last_matrix_data = None
            self._last_tf_msg = None
            self._last_pose_time_ns = None
            self._last_axis_b = None
            self._last_radial_b = None
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self) if self.broadcast_tf else None

            self.pose_pub = self.create_publisher(PoseStamped, self.output_pose_topic, 10)
            self.matrix_pub = (
                self.create_publisher(Float64MultiArray, self.output_matrix_topic, 10)
                if self.publish_matrix else None)
            self.create_subscription(String, self.input_topic, self._on_gap_plane, 10)

            # Monitor for duplicates the file lock cannot see (another container/host).
            self.create_timer(5.0, self._check_duplicate_graph)
            if self.hold_last_pose and self.hold_publish_rate_hz > 0.0:
                self.create_timer(1.0 / self.hold_publish_rate_hz, self._republish_last_pose)

            self.get_logger().info(
                f"gap_pose_robot [{self.instance_id}]: {self.input_topic} (camera frame) -> "
                f"{self.output_pose_topic} (frame {self.target_frame}); "
                f"tf={'on' if self.use_tf else 'off'}, "
                f"static_fallback={'on' if self.static_fallback_enabled else 'off'}, "
                f"hold_last_pose={'on' if self.hold_last_pose else 'off'}")

        def _check_duplicate_graph(self) -> None:
            if count_named_nodes(self) > 1:
                self.get_logger().error(
                    f"Multiple live nodes named '{self.get_name()}' on the ROS graph "
                    f"(this one: {self.instance_id}) — they publish a conflicting "
                    f"{self.output_pose_topic}. Keep exactly one bridge running.",
                    throttle_duration_sec=10.0)

        def _remember_last_pose(self, pose, matrix_data, tf_msg) -> None:
            if not self.hold_last_pose:
                return
            self._last_pose_msg = deepcopy(pose)
            self._last_matrix_data = None if matrix_data is None else list(matrix_data)
            self._last_tf_msg = None if tf_msg is None else deepcopy(tf_msg)
            self._last_pose_time_ns = int(self.get_clock().now().nanoseconds)

        def _republish_last_pose(self) -> None:
            if not self.hold_last_pose or self._last_pose_msg is None or self._last_pose_time_ns is None:
                return
            now = self.get_clock().now()
            age_s = 1e-9 * float(int(now.nanoseconds) - int(self._last_pose_time_ns))
            if age_s > max(0.0, self.hold_max_age_s):
                return

            stamp = now.to_msg()
            pose = deepcopy(self._last_pose_msg)
            pose.header.stamp = stamp
            self.pose_pub.publish(pose)

            if self.matrix_pub is not None and self._last_matrix_data is not None:
                arr = Float64MultiArray()
                arr.data = list(self._last_matrix_data)
                self.matrix_pub.publish(arr)

            if self.tf_broadcaster is not None and self._last_tf_msg is not None:
                tf_msg = deepcopy(self._last_tf_msg)
                tf_msg.header.stamp = stamp
                self.tf_broadcaster.sendTransform(tf_msg)

        # -- camera -> target transform ---------------------------------------
        def _resolve_transform(self, camera_frame: str):
            """Return (R_3x3, t_3) for target_frame <- camera_frame, or None."""
            if self.use_tf and camera_frame:
                try:
                    import rclpy.time
                    from rclpy.duration import Duration
                    tf = self.tf_buffer.lookup_transform(
                        self.target_frame,
                        camera_frame,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=max(self.tf_timeout_s, 0.0)),
                    )
                    tr = tf.transform.translation
                    rot = tf.transform.rotation
                    R = _quat_xyzw_to_R([rot.x, rot.y, rot.z, rot.w])
                    t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
                    return R, t
                except Exception as exc:  # noqa: BLE001 - any tf2 failure -> fallback
                    if not self._warned_fallback:
                        self.get_logger().warn(
                            f"tf2 lookup {self.target_frame}<-{camera_frame} failed "
                            f"({exc}); using static extrinsic fallback. Set "
                            f"static_cam_to_base_xyz / _quat_xyzw for your robot.")
                        self._warned_fallback = True
            if self.static_fallback_enabled:
                if (
                    _is_identity_static_transform(self.static_xyz, self.static_quat)
                    and not self.allow_identity_static_fallback
                ):
                    if not self._warned_identity_fallback:
                        self.get_logger().error(
                            "Static camera->base fallback is identity and is blocked. "
                            "Provide a real static_cam_to_base_* transform, disable "
                            "static_fallback_enabled, or explicitly set "
                            "allow_identity_static_fallback:=true only for offline tests.")
                        self._warned_identity_fallback = True
                    return None
                return _quat_xyzw_to_R(self.static_quat), self.static_xyz.copy()
            if not self._warned_no_transform:
                self.get_logger().warn(
                    f"No transform available for {self.target_frame}<-{camera_frame}; "
                    "not publishing /gap/pose_robot until tf2 or an explicit static "
                    "camera extrinsic is configured.")
                self._warned_no_transform = True
            return None

        def _on_gap_plane(self, msg) -> None:
            try:
                data = json.loads(msg.data)
            except (ValueError, TypeError):
                return
            if self.require_pose_valid and not bool(data.get("pose_valid", False)):
                return
            if self.require_metric_3d and not bool(data.get("metric_3d_available", False)):
                return
            if self.reject_assumed_depth and bool(data.get("uses_assumed_depth", False)):
                return
            axis = _vector3_or_none(data.get("gap_plane_normal_camera_xyz"))
            radial = _vector3_or_none(data.get("seam_frame_surface_normal_camera_xyz"))
            center = _vector3_or_none(data.get("gap_plane_center_camera_xyz_m"))
            if axis is None or radial is None or center is None:
                self.get_logger().warn(
                    "gap plane message missing finite axis/radial/center; skipping",
                    throttle_duration_sec=2.0)
                return
            camera_frame = self.camera_frame_override or str(data.get("frame_id", ""))
            if not camera_frame:
                self.get_logger().warn(
                    "gap plane message has no frame_id and camera_frame_override is empty; skipping",
                    throttle_duration_sec=2.0)
                return

            resolved = self._resolve_transform(camera_frame)
            if resolved is None:
                return
            R_bc, t_bc = resolved

            axis_b = self.axis_sign * (R_bc @ axis)
            radial_b = self.radial_sign * (R_bc @ radial)
            center_b = R_bc @ center + t_bc

            axis_ref = (
                self.axis_reference_target_xyz
                if self.use_reference_sign_canonicalization else None)
            radial_ref = (
                self.radial_reference_target_xyz
                if self.use_reference_sign_canonicalization else None)
            previous_axis = self._last_axis_b if self.use_temporal_sign_continuity else None
            previous_radial = self._last_radial_b if self.use_temporal_sign_continuity else None
            axis_b, radial_b, _flip_info = _canonicalize_axis_and_radial(
                axis_b, radial_b, axis_ref, radial_ref, previous_axis, previous_radial)

            frame = gap_frame_from_axis_and_radial(axis_b, radial_b)
            if frame is None:
                self.get_logger().warn("degenerate gap frame; skipping", throttle_duration_sec=2.0)
                return
            R_gap, quat = frame
            stamp = self.get_clock().now().to_msg()

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.target_frame
            pose.pose.position.x = float(center_b[0])
            pose.pose.position.y = float(center_b[1])
            pose.pose.position.z = float(center_b[2])
            pose.pose.orientation.x = float(quat[0])
            pose.pose.orientation.y = float(quat[1])
            pose.pose.orientation.z = float(quat[2])
            pose.pose.orientation.w = float(quat[3])
            self.pose_pub.publish(pose)

            matrix_data = None
            if self.matrix_pub is not None:
                T = np.eye(4)
                T[:3, :3] = R_gap
                T[:3, 3] = center_b
                arr = Float64MultiArray()
                arr.data = [float(v) for v in T.reshape(-1)]
                self.matrix_pub.publish(arr)
                matrix_data = arr.data

            tf_msg = None
            if self.tf_broadcaster is not None:
                tf_msg = TransformStamped()
                tf_msg.header.stamp = stamp
                tf_msg.header.frame_id = self.target_frame
                tf_msg.child_frame_id = self.gap_tf_child_frame
                tf_msg.transform.translation.x = float(center_b[0])
                tf_msg.transform.translation.y = float(center_b[1])
                tf_msg.transform.translation.z = float(center_b[2])
                tf_msg.transform.rotation.x = float(quat[0])
                tf_msg.transform.rotation.y = float(quat[1])
                tf_msg.transform.rotation.z = float(quat[2])
                tf_msg.transform.rotation.w = float(quat[3])
                self.tf_broadcaster.sendTransform(tf_msg)

            self._remember_last_pose(pose, matrix_data, tf_msg)
            self._last_axis_b = _unit(axis_b)
            y = self._last_axis_b
            if y is not None:
                radial_proj = radial_b - float(np.dot(radial_b, y)) * y
                self._last_radial_b = _unit(radial_proj)

    return GapPoseRobotNode


def main(args=None) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = _build_node_class()()
        rclpy.spin(node)
    except DuplicateInstanceError as exc:
        print(f"[gap_pose_robot_node] refusing to start: {exc}", file=sys.stderr)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            lock = getattr(node, "_single_instance", None)
            if lock is not None:
                lock.release()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    main()
