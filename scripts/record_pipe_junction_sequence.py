#!/usr/bin/env python3
"""Record camera_F RGB-D frames, detector overlays, and detector status.

This is a data-capture helper for pipe-junction tracking tests. It does not
modify or call the detector internals: it only subscribes to the live ROS topics
and writes a replayable folder with images plus JSONL metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _decode_image(msg: Any) -> np.ndarray:
    enc = str(msg.encoding).lower()
    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 3)
        if enc == "bgr8":
            arr = arr[..., [2, 1, 0]]
        return arr
    if enc in ("rgba8", "bgra8"):
        arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width, 4)[..., :3]
        if enc == "bgra8":
            arr = arr[..., [2, 1, 0]]
        return arr
    if enc in ("mono8", "8uc1"):
        arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.width)
        return np.repeat(arr[..., None], 3, axis=2)
    if enc in ("16uc1", "mono16"):
        arr = np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width)
        return arr
    if enc in ("32fc1",):
        return np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width)
    raise RuntimeError(f"unsupported image encoding: {msg.encoding}")


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0.0)
    if not bool(valid.any()):
        return np.zeros(d.shape[:2], dtype=np.uint8)
    lo, hi = np.percentile(d[valid], [1.0, 99.0])
    return (np.clip((d - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image as PImage

        PImage.fromarray(np.ascontiguousarray(arr)).save(path)
    except Exception:
        np.save(path.with_suffix(".npy"), arr)


def _stamp_s(msg: Any) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0.0
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _parse_json_text(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> Any | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), size)
    return writer if writer.isOpened() else None


class SequenceRecorder:
    def __init__(self, args: argparse.Namespace) -> None:
        import rclpy
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self.args = args
        self.out = Path(args.out)
        self.frames_dir = self.out / "frames"
        self.overlay_dir = self.out / "overlays"
        self.depth_dir = self.out / "depth_preview"
        self.out.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        self.latest: dict[str, Any] = {}
        self.latest_status: dict[str, Any] | None = None
        self.latest_detection: dict[str, Any] | None = None
        self.frame_idx = 0
        self.saved_idx = 0
        self.start_monotonic = time.monotonic()
        self.last_save_monotonic = 0.0
        self.video_writer: Any | None = None

        reliability = ReliabilityPolicy.RELIABLE if args.reliable else ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(depth=20, reliability=reliability, history=HistoryPolicy.KEEP_LAST)

        self.rclpy = rclpy
        self.node = rclpy.create_node("record_pipe_junction_sequence")
        self.tf_buffer = None
        if bool(getattr(args, "log_camera_pose", False)):
            try:
                from tf2_ros import Buffer, TransformListener

                self.tf_buffer = Buffer()
                self._tf_listener = TransformListener(self.tf_buffer, self.node)
            except Exception as exc:  # pragma: no cover - tf2 optional
                print(f"[record] tf2 unavailable, camera pose not logged: {exc}")
                self.tf_buffer = None
        self.node.create_subscription(Image, args.rgb_topic, self._on_rgb, qos)
        self.node.create_subscription(Image, args.depth_topic, self._on_depth, qos)
        self.node.create_subscription(Image, args.rgb_overlay_topic, self._on_rgb_overlay, qos)
        self.node.create_subscription(Image, args.depth_overlay_topic, self._on_depth_overlay, qos)
        self.node.create_subscription(String, args.status_topic, self._on_status, qos)
        self.node.create_subscription(String, args.detection_topic, self._on_detection, qos)
        self.rows = (self.out / "frames.jsonl").open("w", encoding="utf-8")
        self.status_rows = (self.out / "status.jsonl").open("w", encoding="utf-8")
        self.detection_rows = (self.out / "detection.jsonl").open("w", encoding="utf-8")

    def close(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
        self.rows.close()
        self.status_rows.close()
        self.detection_rows.close()
        self.node.destroy_node()

    def _on_status(self, msg: Any) -> None:
        payload = _parse_json_text(msg.data)
        if payload is not None:
            self.latest_status = payload
            self.status_rows.write(json.dumps(payload, sort_keys=True) + "\n")
            self.status_rows.flush()

    def _on_detection(self, msg: Any) -> None:
        payload = _parse_json_text(msg.data)
        if payload is not None:
            self.latest_detection = payload
            self.detection_rows.write(json.dumps(payload, sort_keys=True) + "\n")
            self.detection_rows.flush()

    def _on_depth(self, msg: Any) -> None:
        self.latest["depth"] = msg

    def _on_rgb_overlay(self, msg: Any) -> None:
        self.latest["rgb_overlay"] = msg

    def _on_depth_overlay(self, msg: Any) -> None:
        self.latest["depth_overlay"] = msg

    def _on_rgb(self, msg: Any) -> None:
        self.frame_idx += 1
        now = time.monotonic()
        min_period = 1.0 / max(float(self.args.save_fps), 1e-6) if self.args.save_fps > 0.0 else 0.0
        if self.args.save_every_n > 1 and self.frame_idx % int(self.args.save_every_n) != 0:
            return
        if min_period > 0.0 and now - self.last_save_monotonic < min_period:
            return
        self.last_save_monotonic = now
        self.saved_idx += 1

        rgb = _decode_image(msg)
        rgb_name = f"rgb_{self.saved_idx:06d}.png"
        _save_png(rgb, self.frames_dir / rgb_name)

        overlay_name = None
        overlay_msg = self.latest.get("rgb_overlay")
        overlay = None
        if overlay_msg is not None:
            overlay = _decode_image(overlay_msg)
            overlay_name = f"rgb_overlay_{self.saved_idx:06d}.png"
            _save_png(overlay, self.overlay_dir / overlay_name)
            if self.args.video and overlay.ndim == 3:
                if self.video_writer is None:
                    h, w = overlay.shape[:2]
                    self.video_writer = _open_video_writer(self.out / "rgb_overlay.mp4", self.args.video_fps, (w, h))
                if self.video_writer is not None:
                    import cv2  # type: ignore

                    self.video_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        depth_preview_name = None
        depth_msg = self.latest.get("depth")
        if depth_msg is not None:
            depth = _decode_image(depth_msg)
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) / 1000.0
            else:
                depth_m = depth.astype(np.float32)
            depth_preview_name = f"depth_{self.saved_idx:06d}.png"
            _save_png(_depth_preview(depth_m), self.depth_dir / depth_preview_name)
            if self.args.save_depth_npy:
                np.save(self.depth_dir / f"depth_{self.saved_idx:06d}.npy", depth_m)

        row = {
            "saved_frame": self.saved_idx,
            "rgb_frame_count": self.frame_idx,
            "time_s": round(now - self.start_monotonic, 6),
            "rgb_stamp_s": round(_stamp_s(msg), 9),
            "rgb": str(Path("frames") / rgb_name),
            "rgb_overlay": str(Path("overlays") / overlay_name) if overlay_name else None,
            "depth_preview": str(Path("depth_preview") / depth_preview_name) if depth_preview_name else None,
            "status": self.latest_status,
            "detection": self.latest_detection,
            "cam_pose": self._lookup_cam_pose(msg),
        }
        self.rows.write(json.dumps(row, sort_keys=True) + "\n")
        self.rows.flush()

        status = self.latest_status or {}
        print(
            f"\r[record] saved={self.saved_idx:05d} "
            f"det={status.get('detected')} acc={status.get('detector_accepted')} "
            f"x={status.get('candidate_x_strip_px')} "
            f"lock={status.get('junction_lock_active')} "
            f"src={status.get('junction_lock_source')} "
            f"miss={status.get('junction_lock_missed_frames')}",
            end="",
            flush=True,
        )

    def _lookup_cam_pose(self, msg: Any) -> dict[str, Any] | None:
        if self.tf_buffer is None:
            return None
        try:
            import rclpy

            tr = self.tf_buffer.lookup_transform(
                str(self.args.tf_base_frame), msg.header.frame_id, rclpy.time.Time())
            t = tr.transform.translation
            q = tr.transform.rotation
            return {"target": str(self.args.tf_base_frame), "source": str(msg.header.frame_id),
                    "t": [float(t.x), float(t.y), float(t.z)],
                    "q": [float(q.x), float(q.y), float(q.z), float(q.w)]}
        except Exception:
            return None

    def run(self) -> None:
        deadline = time.monotonic() + float(self.args.duration)
        while self.rclpy.ok() and time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.1)
        print()
        summary = {
            "out": str(self.out),
            "duration_s": float(self.args.duration),
            "rgb_frames_seen": int(self.frame_idx),
            "frames_saved": int(self.saved_idx),
            "latest_status": self.latest_status,
        }
        (self.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[record] summary: {self.out / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="_pipe_junction_sequence")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rgb-topic", default="/camera_F/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera_F/depth_image")
    parser.add_argument("--rgb-overlay-topic", default="/acea/pipe_junction/debug/rgb_overlay")
    parser.add_argument("--depth-overlay-topic", default="/acea/pipe_junction/debug/depth_overlay")
    parser.add_argument("--status-topic", default="/acea/pipe_junction/status")
    parser.add_argument("--detection-topic", default="/acea/pipe_junction/detection")
    parser.add_argument("--save-every-n", type=int, default=1)
    parser.add_argument("--save-fps", type=float, default=0.0,
                        help="Optional max image save FPS. 0 saves every selected RGB frame.")
    parser.add_argument("--video", action="store_true", default=True,
                        help="Write rgb_overlay.mp4 when OpenCV is available.")
    parser.add_argument("--no-video", dest="video", action="store_false")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--save-depth-npy", action="store_true")
    parser.add_argument("--best-effort", dest="reliable", action="store_false", default=False)
    parser.add_argument("--reliable", dest="reliable", action="store_true")
    parser.add_argument("--log-camera-pose", dest="log_camera_pose", action="store_true", default=True,
                        help="Log <tf-base-frame>->camera tf per frame (for projected-GT accuracy).")
    parser.add_argument("--no-log-camera-pose", dest="log_camera_pose", action="store_false")
    parser.add_argument("--tf-base-frame", default="base_link",
                        help="Fixed frame the per-frame camera pose is expressed in.")
    return parser.parse_args()


def main() -> int:
    import rclpy

    rclpy.init()
    recorder = SequenceRecorder(parse_args())
    try:
        recorder.run()
    finally:
        recorder.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
