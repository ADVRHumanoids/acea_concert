#!/usr/bin/env python3
"""Grab one frame from a camera topic and save the RAW image(s) for an offline
orientation / framing check -- NO detection overlay. Used to verify the arm
camera (camera_F) is oriented correctly on the pipe junction once it is bridged.

  # inside docker-dev-1, from the bind-mounted acea_concert dir so the host can read it:
  python3 scripts/grab_camera_frame.py \
      --rgb-topic /camera_F/color/image_raw \
      --depth-topic /camera_F/depth_image \
      --camera-info-topic /camera_F/camera_info \
      --out _camera_F_grab

Writes <out>/rgb.png, <out>/depth.npy + depth_preview.png (if a depth topic is
given), and info.json (frame_id / encoding / size / intrinsics). Works with
SensorDataQoS publishers (best-effort). No cv2/cv_bridge dependency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _save_png(arr: np.ndarray, path: Path) -> bool:
    try:
        from PIL import Image as PImage
        PImage.fromarray(np.ascontiguousarray(arr)).save(path)
        return True
    except Exception:
        np.save(path.with_suffix(".npy"), arr)
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb-topic", default="/camera_F/color/image_raw")
    ap.add_argument("--depth-topic", default="")
    ap.add_argument("--camera-info-topic", default="")
    ap.add_argument("--out", type=Path, default=Path("camera_grab"))
    ap.add_argument("--timeout", type=float, default=10.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    n = rclpy.create_node("grab_camera_frame")
    box: dict[str, object] = {}
    n.create_subscription(Image, a.rgb_topic, lambda m: box.setdefault("rgb", m), qos_profile_sensor_data)
    need = {"rgb"}
    if a.depth_topic:
        n.create_subscription(Image, a.depth_topic, lambda m: box.setdefault("depth", m), qos_profile_sensor_data)
        need.add("depth")
    if a.camera_info_topic:
        n.create_subscription(CameraInfo, a.camera_info_topic, lambda m: box.setdefault("info", m), qos_profile_sensor_data)
        need.add("info")

    t0 = time.time()
    while not need.issubset(box) and time.time() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.2)

    info_out: dict[str, object] = {}
    rgb = box.get("rgb")
    if rgb is None:
        print(f"[grab] NO RGB on {a.rgb_topic} within {a.timeout:.0f}s -- is it bridged/published?")
        rclpy.shutdown()
        return 1
    arr = np.frombuffer(bytes(rgb.data), np.uint8).reshape(rgb.height, rgb.width, -1)
    if rgb.encoding.startswith("bgr"):
        arr = arr[..., [2, 1, 0]] if arr.shape[2] >= 3 else arr
    _save_png(arr[..., :3], a.out / "rgb.png")
    info_out["rgb"] = {"frame_id": rgb.header.frame_id, "encoding": rgb.encoding,
                       "width": rgb.width, "height": rgb.height}
    print(f"[grab] RGB   frame_id={rgb.header.frame_id} enc={rgb.encoding} {rgb.width}x{rgb.height}")

    depth = box.get("depth")
    if depth is not None:
        if depth.encoding in ("16UC1", "mono16"):
            d = np.frombuffer(bytes(depth.data), np.uint16).reshape(depth.height, depth.width).astype(np.float32) / 1000.0
        else:  # 32FC1 (meters) and fallbacks
            d = np.frombuffer(bytes(depth.data), np.float32).reshape(depth.height, depth.width)
        np.save(a.out / "depth.npy", d)
        valid = np.isfinite(d) & (d > 0)
        if valid.any():
            lo, hi = np.percentile(d[valid], [1.0, 99.0])
            vis = (np.clip((d - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
            _save_png(vis, a.out / "depth_preview.png")
        info_out["depth"] = {"frame_id": depth.header.frame_id, "encoding": depth.encoding,
                             "min_m": float(np.nanmin(d[valid])) if valid.any() else None,
                             "max_m": float(np.nanmax(d[valid])) if valid.any() else None}
        print(f"[grab] DEPTH frame_id={depth.header.frame_id} enc={depth.encoding} "
              f"range=[{info_out['depth']['min_m']},{info_out['depth']['max_m']}] m")

    ci = box.get("info")
    if ci is not None:
        info_out["camera_info"] = {"frame_id": ci.header.frame_id, "width": ci.width,
                                   "height": ci.height, "K": [float(x) for x in ci.k]}
        print(f"[grab] INFO  frame_id={ci.header.frame_id} {ci.width}x{ci.height} "
              f"fx={ci.k[0]:.1f} fy={ci.k[4]:.1f} cx={ci.k[2]:.1f} cy={ci.k[5]:.1f}")

    (a.out / "info.json").write_text(json.dumps(info_out, indent=2) + "\n")
    print(f"[grab] saved -> {a.out}")
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
