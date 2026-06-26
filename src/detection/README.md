# ACEA Detection Module (`acea_concert`)

Perception module that produces the weld-seam **gap pose in the robot's
`base_link` frame** from a depth camera, as a **drop-in replacement** for the
Gazebo ground-truth `src/gap_pose_publisher.py` (which fakes the gap from `gz`
model poses, without a camera).

It publishes the exact same contract the rest of `acea_concert` already consumes
(e.g. `drive_base_to_weld_pose.py`):

```
/gap/pose_robot   geometry_msgs/PoseStamped   frame_id: base_link
```

## Two nodes

| Node | In | Out |
|------|----|-----|
| `acea_pipe_junction_node.py` | `/camera/rgb`, `/camera/depth`, `/camera/camera_info` | `/acea/pipe_junction/detection`, `/acea/weld_seam/gap_plane` (JSON, **camera frame**), `/acea/weld_seam/pose`, `/acea/weld_seam/markers` |
| `gap_pose_robot_node.py` | `/acea/weld_seam/gap_plane` | **`/gap/pose_robot`** (PoseStamped, **base_link**), `/gap/pose_robot_matrix` (optional 4×4) |

The detector estimates the pipe axis + back-projects the seam surface from **real
depth** (not assumed); the gap plane is `{point = seam surface, normal = pipe
axis}`. `gap_pose_robot_node` transforms that geometry `camera → base_link` and
rebuilds the gap frame in the **same convention as `gap_pose_publisher.py`**:

```
y = pipe axis            (gap-plane normal)
x = radial weld tangent  (outward surface normal, projected ⟂ y)
z = x × y                (right-handed)
```

## Gap pose: what goes on the topic

`/gap/pose_robot` carries **position (x,y,z) + quaternion** in `base_link`. That
is the full pose — the rotation matrix is **not** put on the pose topic (it is
redundant with the quaternion). For debugging only, the 4×4 homogeneous
`base_link ← gap` transform can be published **separately** on
`/gap/pose_robot_matrix` (`std_msgs/Float64MultiArray`, 16 row-major values);
enable it with `publish_matrix:=true`.

## Camera → base_link transform

`gap_pose_robot_node` first tries a **tf2 lookup** `base_link ← <camera optical
frame>` (frame taken from the detector message, since `robot_state_publisher`
provides the TF tree). This is the preferred online path. If that frame is not
in the tree, the node does **not** publish `/gap/pose_robot` unless an explicit
static extrinsic fallback is enabled and configured:

```yaml
static_cam_to_base_xyz: [x, y, z]
static_cam_to_base_quat_xyzw: [x, y, z, w]
```

The identity fallback is blocked by default. Do not enable it for real robot or
controller tests, otherwise camera-frame coordinates could be mislabeled as
`base_link`.

If the perceived frame points the opposite way to the GT convention, flip
`axis_sign` / `radial_sign` (verify once against `gap_pose_publisher.py` live).

## Run it

```bash
# build + source the workspace, then:
ros2 launch acea_concert detection.launch.py

# point at non-default camera topics / enable YOLO frontend:
ros2 launch acea_concert detection.launch.py use_yolo_seg_frontend:=true
```

Or each node standalone:

```bash
ros2 run acea_concert acea_pipe_junction_node.py --ros-args \
  --params-file $(ros2 pkg prefix acea_concert)/share/acea_concert/config/detector.yaml
ros2 run acea_concert gap_pose_robot_node.py --ros-args \
  --params-file $(ros2 pkg prefix acea_concert)/share/acea_concert/config/gap_pose_robot.yaml
```

Check the output:

```bash
ros2 topic echo /gap/pose_robot
```

Config lives in `config/detector.yaml` and `config/gap_pose_robot.yaml`.

## Geometry self-test (no ROS / no GPU)

```bash
python3 src/detection/gap_pose_robot_node.py --selftest
```

Validates the gap-frame convention and the quaternion math (9/9 checks).

## Robustness (operational)

Two recurring runtime failures are guarded against (see the package README,
"Run Exactly One Detector", for the full reproduce/verify steps):

- **Duplicate nodes** — a second detector or bridge on the same host refuses to
  start (race-free file lock; override `-p allow_duplicate:=true`). A periodic
  graph check also warns about cross-host/container copies. Every status message
  carries `node_instance` (`host:pid`).
- **RGB/depth sync** — if both streams flow but never sync on header stamps
  (zero or mixed sim/wall clock), the detector switches to receive-time sync
  automatically (`auto_receive_time_fallback`). `WAITING_FOR_SYNC` status reports
  `rgb_hz` / `depth_hz` / `camera_info_hz`, `*_age_s`, `sync_time_source`, and a
  human `hint`. Note: a node keeps the params it loaded at startup — restart it
  after editing/rebuilding the config.

## Notes

- The detector needs a working **depth** stream; in Gazebo bridge the depth
  camera sensor to `/camera/depth` (+ `rgb`, `camera_info`) or remap the topics.
- The contract (`/gap/pose_robot` in `base_link`) is what `acea_concert`
  integrates against. The bridge transform is exact; if perception disagrees with
  the ground truth, suspect the camera->`base_link` TF (see the package README,
  "Compare Perception Against Ground Truth").
