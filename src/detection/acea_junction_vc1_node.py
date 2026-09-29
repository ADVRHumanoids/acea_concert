#!/usr/bin/env python3
"""ROS 2 node for the VC1 pipe-junction detector.

Inputs : RGB, aligned depth and CameraInfo with identical sensor stamps.
Output : the same detection JSON as the V17 node (``/acea/pipe_junction/detection``),
         consumed unchanged by the sensor-stamped gap-pose bridge
         (``gap_pose_robot_node_v14_dev.py``) which publishes ``/gap/pose_robot``
         in ``base_link`` with the CAMERA stamp preserved.

Every accepted pose is measured in the current RGB-D tuple.  The detector keeps
only search state (warm-started pipe, track position, vetoed features); a held
or predicted pose is never published.  Frames without a synchronized depth are
reported as rejected, never filled with an old depth.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import threading
import math
from pathlib import Path
import sys
import time
from typing import Any

import socket

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acea_junction_vc1 as vc1  # noqa: E402
from acea_alignment.weld_seam import seam_frame_from_axis_and_surface  # noqa: E402

try:  # noqa: E402 - installed next to this executable by CMake (shared with V16/V17)
    from acea_detection_runtime import DuplicateInstanceError, SingleInstanceLock, count_named_nodes, instance_id
except Exception:  # pragma: no cover - fail-safe fallback: never block startup
    class DuplicateInstanceError(RuntimeError):
        pass

    class SingleInstanceLock:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            self.acquired, self.holder = True, ""

        def release(self) -> None:
            return None

    def count_named_nodes(_node) -> int:
        return 1

    def instance_id() -> str:
        return f"{socket.gethostname()}:unknown"

SCHEMA = "acea.pipe_junction_detection/vc1"
# Same lock as the V16/V17 nodes: only one ACEA detector may publish the
# junction topics on a host (two detectors would feed the bridge contradictory poses).
DETECTOR_LOCK_NAME = "acea_pipe_junction_node"


def stamp_ns(msg) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def rgb_array(msg: Image) -> np.ndarray:
    enc = str(msg.encoding).lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.step))
    w = int(msg.width)
    if enc in ("rgb8", "bgr8"):
        img = raw[:, : 3 * w].reshape(int(msg.height), w, 3)
        return np.ascontiguousarray(img[:, :, ::-1] if enc == "bgr8" else img)
    if enc in ("rgba8", "bgra8"):
        img = raw[:, : 4 * w].reshape(int(msg.height), w, 4)[:, :, :3]
        return np.ascontiguousarray(img[:, :, ::-1] if enc == "bgra8" else img)
    if enc in ("mono8", "8uc1"):
        return np.ascontiguousarray(np.repeat(raw[:, :w, None], 3, axis=2))
    raise ValueError(f"unsupported_rgb_encoding:{msg.encoding}")


def depth_array_m(msg: Image) -> np.ndarray:
    enc = str(msg.encoding).upper()
    bo = ">" if bool(msg.is_bigendian) else "<"
    if enc == "32FC1":
        d = np.frombuffer(msg.data, dtype=np.dtype(f"{bo}f4")).reshape(int(msg.height), int(msg.step) // 4)[:, : int(msg.width)]
        d = d.astype(np.float32, copy=True)
    elif enc in ("16UC1", "MONO16"):
        d = np.frombuffer(msg.data, dtype=np.dtype(f"{bo}u2")).reshape(int(msg.height), int(msg.step) // 2)[:, : int(msg.width)]
        d = d.astype(np.float32) * 0.001
    else:
        raise ValueError(f"unsupported_depth_encoding:{msg.encoding}")
    d[~np.isfinite(d)] = 0.0
    return np.ascontiguousarray(d)


def seam_axis(det) -> np.ndarray:
    """Axis of the published seam frame: the depth-fitted pipe (cylinder) axis.

    The junction ring normal also carries the small tilt fitted to the RGB line
    (it helps to follow the visible line, so the ring centre and surface point
    keep it), but as an orientation it is much noisier: in simulation with
    ground truth the cylinder axis errs 0.03 deg (p50) vs 0.9 deg for the ring
    normal, and on the real bags it is also the steadier of the two.  A butt
    joint's seam plane is perpendicular to the pipe axis."""
    ring = canonical_axis(det.axis)
    cyl = det.cylinder.axis if det.cylinder is not None else None
    if cyl is None or not np.isfinite(cyl).all():
        return ring
    cyl = np.asarray(cyl, float) / np.linalg.norm(cyl)
    return cyl if float(cyl @ ring) >= 0.0 else -cyl


def canonical_axis(axis: np.ndarray) -> np.ndarray:
    """Same camera-derived sign convention as V17 (x >= 0, then y >= 0), so the
    existing bridge configuration (axis_sign=-1) maps it identically."""
    a = np.asarray(axis, float)
    if a[0] < 0.0 or (abs(a[0]) < 1e-9 and a[1] < 0.0):
        a = -a
    return a


def finite_list(v) -> list[float] | None:
    if v is None:
        return None
    arr = np.asarray(v, float).reshape(-1)
    return [float(x) for x in arr] if np.isfinite(arr).all() else None


class TupleSync:
    """RGB / aligned depth / CameraInfo pairing by sensor stamp (no ROS inside).

    * The oldest complete tuple is processed first; an RGB frame still waiting
      for its depth never blocks a newer complete tuple; beyond `max_backlog`
      complete tuples the oldest are superseded (real time: newest data win).
    * Order: a tuple is never handed out after a newer one.  An RGB frame whose
      depth or info had not arrived when a newer tuple was processed is dropped
      ("overtaken": usually its depth was lost; if it arrives late it would
      move the detector state and the published poses back in time).
    * Time going back (bag loop, simulation reset) starts a new timeline:
      queues, processed stamps and the order are reset, so repeated stamps
      are processed again.  A jump of more than `reset_jump_ns` resets at once;
      a smaller one as soon as a second consecutive RGB frame is also behind
      (a single late RGB frame is not a new timeline; it is dropped).
    """

    def __init__(self, queue_size: int, slop_ns: int, expiry_ns: int, max_backlog: int,
                 count: dict[str, int] | None = None, reset_jump_ns: int = 1_000_000_000) -> None:
        self.queue_size, self.slop_ns, self.expiry_ns = queue_size, slop_ns, expiry_ns
        self.max_backlog, self.reset_jump_ns = max_backlog, reset_jump_ns
        self.count = count if count is not None else {}
        self.rgb_q: OrderedDict[int, Any] = OrderedDict()
        self.depth_q: OrderedDict[int, Any] = OrderedDict()
        self.info_q: OrderedDict[int, Any] = OrderedDict()
        self.done: OrderedDict[int, None] = OrderedDict()
        self.newest_rgb_ns: int | None = None
        self.last_processed_ns: int | None = None
        self.rgb_behind = 0

    def _bump(self, key: str) -> None:
        self.count[key] = self.count.get(key, 0) + 1

    def _insert(self, q: OrderedDict, st: int, msg) -> None:
        q[st] = msg
        q.move_to_end(st)
        while len(q) > self.queue_size:
            q.popitem(last=False)

    def _mark(self, t: int) -> None:
        self.done[t] = None
        while len(self.done) > 4 * self.queue_size:
            self.done.popitem(last=False)

    def _new_timeline(self) -> None:
        self.rgb_q.clear(); self.depth_q.clear(); self.info_q.clear(); self.done.clear()
        self.newest_rgb_ns = self.last_processed_ns = None
        self.rgb_behind = 0
        self._bump("time_resets")

    def add_rgb(self, st: int, msg) -> None:
        if self.newest_rgb_ns is not None and st < self.newest_rgb_ns:
            if self.newest_rgb_ns - st > self.reset_jump_ns or self.rgb_behind >= 1:
                self._new_timeline()
            else:
                self.rgb_behind += 1        # late frame or first frame of a new timeline
        else:
            self.rgb_behind = 0
        self.newest_rgb_ns = st if self.newest_rgb_ns is None else max(self.newest_rgb_ns, st)
        self._insert(self.rgb_q, st, msg)

    def add_depth(self, st: int, msg) -> None:
        self._insert(self.depth_q, st, msg)

    def add_info(self, st: int, msg) -> None:
        self._insert(self.info_q, st, msg)

    @staticmethod
    def _nearest(q: OrderedDict, t: int):
        return min(q, key=lambda s: (abs(s - t), s)) if q else None

    def next_tuple(self):
        """(rgb, depth, info) of the oldest complete tuple newer than the last
        one handed out, or None."""
        if self.last_processed_ns is not None:
            for t in [t for t in self.rgb_q if t <= self.last_processed_ns]:
                self.rgb_q.pop(t)
                self._mark(t)
                self._bump("overtaken")
        complete = []
        for t in sorted(self.rgb_q):
            if t in self.done:
                continue
            td, ti = self._nearest(self.depth_q, t), self._nearest(self.info_q, t)
            if td is not None and ti is not None and abs(td - t) <= self.slop_ns and abs(ti - t) <= self.slop_ns:
                complete.append((t, td, ti))
        while len(complete) > self.max_backlog:
            t, td, ti = complete.pop(0)
            self.rgb_q.pop(t, None); self.depth_q.pop(td, None); self.info_q.pop(ti, None)
            self._mark(t)
            self._bump("superseded")
        newest_d = max(self.depth_q) if self.depth_q else None
        if newest_d is not None:
            for old in [s for s in self.rgb_q if newest_d > s + self.expiry_ns and s not in [c[0] for c in complete]]:
                self.rgb_q.pop(old)
                self._mark(old)
                self._bump("no_depth")
        if not complete:
            return None
        t, td, ti = complete[0]
        rgb, dep, inf = self.rgb_q.pop(t), self.depth_q.pop(td), self.info_q.pop(ti)
        self._mark(t)
        self.last_processed_ns = t
        return rgb, dep, inf


class VC1Node(Node):
    def __init__(self) -> None:
        super().__init__("acea_pipe_junction_node")
        decl = {
            "rgb_topic": "/camera_E/color/image_raw",
            "depth_topic": "/camera_E/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera_E/color/camera_info",
            "detection_topic": "/acea/pipe_junction/detection",
            "status_topic": "/acea/pipe_junction/status",
            "detected_topic": "/acea/pipe_junction/detected",
            "rgb_overlay_topic": "/acea/pipe_junction/debug/rgb_overlay",
            "weld_gap_plane_topic": "/acea/weld_seam/gap_plane",
            "publish_rgb_overlay": False,
            "camera_qos_reliability": "best_effort",
            "queue_size": 60,
            "sync_slop_s": 0.002,
            "sync_expiry_s": 0.25,
            "max_backlog": 2,
            "pipe_diameter_m": 0.20,
            "stateless": False,
            "core_params_json": "{}",
            "allow_duplicate": False,
        }
        for name, default in decl.items():
            self.declare_parameter(name, default)
        g = lambda n: self.get_parameter(n).value
        self.instance = instance_id()
        self._single_instance = SingleInstanceLock(DETECTOR_LOCK_NAME)
        if not self._single_instance.acquired and not bool(g("allow_duplicate")):
            raise DuplicateInstanceError(
                f"another ACEA detector holds the single-instance lock ({self._single_instance.holder or 'unknown'})")
        self.radius_m = 0.5 * float(g("pipe_diameter_m"))
        if not (math.isfinite(self.radius_m) and self.radius_m > 0):
            raise ValueError("pipe_diameter_m must be positive (the only workpiece prior)")
        overrides = json.loads(str(g("core_params_json")))
        unknown = sorted(set(overrides) - set(vc1.DEFAULTS))
        if unknown:
            raise ValueError(f"unknown VC1 parameters: {unknown}")
        self.detector = vc1.VC1Detector(self.radius_m, overrides, stateless=bool(g("stateless")))
        self.queue_size = max(3, int(g("queue_size")))
        self.slop_ns = int(round(float(g("sync_slop_s")) * 1e9))
        self.expiry_ns = int(round(max(float(g("sync_slop_s")), float(g("sync_expiry_s"))) * 1e9))
        self.max_backlog = max(1, int(g("max_backlog")))
        self.count = {"processed": 0, "accepted": 0, "rejected": 0}
        self.sync = TupleSync(self.queue_size, self.slop_ns, self.expiry_ns, self.max_backlog, self.count)
        self._last_warn: dict[str, float] = {}
        rel = str(g("camera_qos_reliability")).lower()
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE if rel in ("reliable", "rel") else ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=self.queue_size)
        self.create_subscription(Image, str(g("rgb_topic")), self._on_rgb, qos)
        self.create_subscription(Image, str(g("depth_topic")), self._on_depth, qos)
        self.create_subscription(CameraInfo, str(g("camera_info_topic")), self._on_info, qos)
        self.det_pub = self.create_publisher(String, str(g("detection_topic")), 100)
        self.status_pub = self.create_publisher(String, str(g("status_topic")), 20)
        self.detected_pub = self.create_publisher(Bool, str(g("detected_topic")), 20)
        self.gap_pub = self.create_publisher(String, str(g("weld_gap_plane_topic")), 20)
        self.overlay_pub = self.create_publisher(Image, str(g("rgb_overlay_topic")), 5) if bool(g("publish_rgb_overlay")) else None
        # Reception never blocks: callbacks only enqueue, a worker thread runs the detector.
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()
        self.create_timer(10.0, self._report)
        self.get_logger().info(f"VC1 detector: radius={self.radius_m:.4f} m, RGB={g('rgb_topic')}, depth={g('depth_topic')}")

    def _report(self) -> None:
        self.get_logger().info(f"VC1 counters: {self.count}")
        if count_named_nodes(self) > 1:        # a copy on another host/container shares the graph
            self.get_logger().error("multiple ACEA detector nodes are live: outputs are unsafe")

    def _warn(self, key: str, text: str, period_s: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._last_warn.get(key, -1e9) >= period_s:
            self._last_warn[key] = now
            self.get_logger().warning(text)

    def _work(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.1)
            self._wake.clear()
            while not self._stop.is_set():
                try:
                    with self._lock:
                        job = self.sync.next_tuple()
                    if job is None:
                        break
                    self._process(*job)
                except Exception as exc:     # the worker must never die silently
                    if self._stop.is_set() or not rclpy.ok():
                        return               # shutting down: the frame was mid-flight, nothing to report
                    self.count["worker_errors"] = self.count.get("worker_errors", 0) + 1
                    self._warn("worker", f"VC1 worker error (frame dropped): {type(exc).__name__}: {exc}")

    def destroy_node(self):
        self._stop.set()
        self._wake.set()
        worker = getattr(self, "_worker", None)
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)
        lock = getattr(self, "_single_instance", None)
        if lock is not None:
            lock.release()
        return super().destroy_node()

    # ---------------------------------------------------------------- sync
    def _on_rgb(self, msg):
        with self._lock:
            self.sync.add_rgb(stamp_ns(msg), msg)
        self._wake.set()

    def _on_depth(self, msg):
        with self._lock:
            self.sync.add_depth(stamp_ns(msg), msg)
        self._wake.set()

    def _on_info(self, msg):
        with self._lock:
            self.sync.add_info(stamp_ns(msg), msg)
        self._wake.set()

    # ---------------------------------------------------------- detection
    def _process(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo) -> None:
        t0 = time.perf_counter()
        det, reason = None, None
        try:
            rgb = rgb_array(rgb_msg)
            depth = depth_array_m(depth_msg)
            k = np.asarray(info_msg.k, float).reshape(3, 3)
            if rgb.shape[:2] != depth.shape:
                raise ValueError(f"rgb_depth_shape_mismatch:{rgb.shape[:2]}!={depth.shape} (depth must be aligned to color)")
            if not (np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0):
                raise ValueError("invalid_camera_info_k")
            det = self.detector.process(rgb, depth, k, stamp_s=stamp_ns(rgb_msg) * 1e-9)
            reason = det.reason
        except Exception as exc:        # fail closed on malformed inputs
            reason = f"input_error:{type(exc).__name__}:{exc}"
            rgb, k, det = None, None, None
            self._warn("input", f"VC1 input rejected: {reason}")
        ms = 1000.0 * (time.perf_counter() - t0)
        seam = None
        if det is not None and det.accepted:
            if not all(v is not None and np.isfinite(v).all() for v in (det.axis, det.center, det.surface)):
                det.accepted, det.reason, reason = False, "non_finite_pose", "non_finite_pose"
        if det is not None and det.accepted:
            seam = seam_frame_from_axis_and_surface(seam_axis(det), det.surface, det.center)
            if seam is None or not np.isfinite(np.asarray(seam.quat_xyzw, float)).all():
                seam = None          # degenerate frame: fail closed
                det.accepted, det.reason, reason = False, "metric_seam_frame_invalid", "metric_seam_frame_invalid"
        self._publish(self._payload(rgb_msg, ms, det, reason))
        if seam is not None:
            self._publish_gap_plane(rgb_msg, det, seam)
        if self.overlay_pub is not None and rgb is not None:
            self.overlay_pub.publish(self._overlay_msg(rgb, k, det, rgb_msg))

    def _payload(self, rgb_msg: Image, ms: float, det, reason: str) -> dict[str, Any]:
        acc = bool(det is not None and det.accepted)
        self.count["processed"] += 1
        self.count["accepted" if acc else "rejected"] += 1
        axis = canonical_axis(det.axis) if acc else None
        st = rgb_msg.header.stamp
        cyl = det.cylinder if det is not None else None
        return {
            "schema": SCHEMA, "detector_version": "vc1", "instance": self.instance,
            "stamp": {"sec": int(st.sec), "nanosec": int(st.nanosec)},
            "rgb_stamp_ns": stamp_ns(rgb_msg), "frame_id": str(rgb_msg.header.frame_id),
            "process_ms": float(ms), "pipe_radius_m": float(self.radius_m),
            "runtime_inputs": ["current_rgb", "current_aligned_depth", "current_camera_info", "known_pipe_radius"],
            "temporal_pose_used": False, "ground_truth_used": False, "fixture_pose_used": False,
            "detector_accepted": acc, "eligible": acc, "detector_pose_complete": acc,
            "gap_plane_metric_3d_available": acc, "gap_plane_uses_assumed_depth": False,
            "current_frame_proof": acc, "state": "MEASURED" if acc else "REJECTED",
            "junction_lock_source": ("tracked_current_frame" if acc and det.reason == "tracked_relaxed"
                                     else "fresh_current_frame" if acc else "none"),
            "reason": reason, "source": reason,
            "pipe_axis_camera_xyz": finite_list(axis),
            "coarse_seam_center_camera_xyz_m": finite_list(det.center) if acc else None,
            "coarse_seam_visible_surface_camera_xyz_m": finite_list(det.surface) if acc else None,
            "station_m": (float(det.station_m) if acc and det.station_m is not None
                          and math.isfinite(det.station_m) else None),
            "cylinder_valid": bool(cyl is not None and cyl.valid),
            "cylinder_axis_camera_xyz": (finite_list(canonical_axis(cyl.axis))
                                         if cyl is not None and cyl.axis is not None else None),
            "cylinder_axis_point_camera_xyz_m": finite_list(cyl.point) if cyl is not None and cyl.point is not None else None,
            # observed pipe extent along the axis, from cylinder_axis_point (visualisation / mission)
            "cylinder_axial_range_m": ([float(cyl.axial_range[0]), float(cyl.axial_range[1])]
                                       if cyl is not None and cyl.valid else None),
            # ends of the measured pipe seen in this frame (points on the axis): the
            # junction search turns there
            "pipe_end_points_camera_xyz_m": ([e["point_camera_xyz_m"] for e in det.stats.get("pipe_ends") or []]
                                             if det is not None else []),
            "accepted_count": self.count["accepted"], "rejected_count": self.count["rejected"],
        }

    def _publish(self, payload: dict) -> None:
        msg = String(data=json.dumps(payload, sort_keys=True, allow_nan=False, default=float))
        self.det_pub.publish(msg)
        self.status_pub.publish(msg)
        self.detected_pub.publish(Bool(data=bool(payload["detector_accepted"])))

    def _publish_gap_plane(self, rgb_msg: Image, det, seam) -> None:
        """Same schema/fields as the V17 node: consumed by gap_pose_robot_node_v14_dev."""
        st = rgb_msg.header.stamp
        axis = seam_axis(det)
        gap = {
            "schema": "acea.weld_gap_plane/vc1", "valid": True, "reason": "current_frame_metric_proof",
            "pose_valid": True, "pose_reason": "current_frame_metric_proof", "detector_accepted": True,
            "metric_3d_available": True, "uses_assumed_depth": False, "source": "acea_junction_vc1_node",
            "gap_plane_point_source": "current_known_radius_cylinder_surface",
            "gap_plane_center_source": "current_rgb_ring_on_known_radius_cylinder",
            "gap_plane_normal_source": "current_depth_cylinder_axis",
            "seam_frame_radial_source": "current_axis_line_toward_camera",
            "frame_id": str(rgb_msg.header.frame_id),
            "stamp": {"sec": int(st.sec), "nanosec": int(st.nanosec)},
            "gap_plane_point_camera_xyz_m": finite_list(det.surface),
            "gap_plane_center_camera_xyz_m": finite_list(det.center),
            "gap_plane_normal_camera_xyz": finite_list(axis),
            "seam_visible_surface_camera_xyz_m": finite_list(det.surface),
            "seam_frame_origin_camera_xyz_m": finite_list(seam.origin_xyz_m),
            "seam_frame_quaternion_xyzw": [float(x) for x in seam.quat_xyzw],
            "seam_frame_surface_normal_camera_xyz": finite_list(seam.surface_normal),
            "seam_frame_tangent_camera_xyz": finite_list(seam.seam_tangent),
            "seam_frame_degenerate": bool(seam.degenerate),
            "proof_source": det.reason,
        }
        self.gap_pub.publish(String(data=json.dumps(gap, sort_keys=True, allow_nan=False)))

    def _overlay_msg(self, rgb: np.ndarray, k, det, src: Image) -> Image:
        img = rgb.copy()
        if det is not None and det.accepted and k is not None:
            ring = vc1.ring_pixels(det.center, det.axis, self.radius_m, k)
            if len(ring):
                cv2.polylines(img, [np.rint(ring).astype(np.int32)], False, (255, 0, 0), 2)
        txt = "VC1 " + ("MEASURED" if det is not None and det.accepted else (det.reason if det is not None else "no input"))
        cv2.putText(img, txt[:60], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        out = Image()
        out.header = src.header
        out.height, out.width = img.shape[:2]
        out.encoding = "rgb8"
        out.step = 3 * img.shape[1]
        out.data = img.tobytes()
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    code = 0
    try:
        node = VC1Node()
        rclpy.spin(node)
    except DuplicateInstanceError as exc:
        print(f"[acea_junction_vc1_node] refusing to start: {exc}", file=sys.stderr)
        code = 2
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # rclpy (Jazzy) shutdown race: after SIGINT the context is closed while the
        # executor still takes a message from a subscription that was ready
        # ("Unable to convert call argument '0' ..."); an error while running is raised
        if rclpy.ok():
            raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
