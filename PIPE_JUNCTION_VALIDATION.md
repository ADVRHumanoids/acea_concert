# Pipe Junction Detection And Tracking Validation

Date: 2026-06-29

This document records the current validated state of the Gazebo arm-camera pipe
junction detector/tracker. The scope here is detection/tracking and publication
of a correct junction pose. Robot control/closed-loop motion is owned by the
controller side and is intentionally outside this validation.

## Scope

The detector/tracker is responsible for:

- detecting the pipe junction in `camera_F` RGB/depth;
- keeping the junction track stable during camera roll/sweep;
- estimating the gap plane and seam pose from aligned depth;
- publishing the junction pose/geometry for downstream control;
- marking output invalid when detection is not reliable.

The detector/tracker is not responsible for commanding the robot.

## Output Contract

The downstream controller should consume only valid perception outputs.

Primary topic:

```text
/gap/pose_robot   geometry_msgs/PoseStamped   frame_id: base_link
```

Useful detector/debug topics:

```text
/acea/pipe_junction/status
/acea/pipe_junction/detection
/acea/pipe_junction/debug/rgb_overlay
/acea/weld_seam/gap_plane
/acea/weld_seam/pose
```

Recommended validity gates for downstream consumers:

```text
detected == true
detector_accepted == true
gap_plane_available == true
weld_seam_pose_available == true
junction_lock_active == true
```

If a consumer wants continuous behavior, it may hold the last valid pose for a
very small number of frames, but the pose should be marked stale/held. The
detector itself should not be tuned to accept suspicious jumps only to avoid a
one-frame visual drop.

## Bugs Fixed And Validated

### 1. Candidate Jump Poisoning

Problem:

After an aggressive exit/re-entry, a rejected far candidate could poison the
continuity state. The detector then kept rejecting the true junction as a large
`candidate_jump`, even when the gap was visible again.

Validation signal:

```text
gap_visible_but_rejected ~= 0
```

Final validation:

```text
gap_visible_but_rejected = 0/280
```

### 2. Clipped/Border False Accepts

Problem:

When the pipe was clipped near the image edge, the detector could accept a pipe
end or border as if it were the junction.

Validation signal:

```text
near_border_no_gap == 0
```

Final validation:

```text
near_border_no_gap = 0
```

`accepted_gap_not_visible` alone is not a clean bug metric. The image-based GT
can miss faint or rolled gaps. The cleaner clipped-edge signal is
`near_border_no_gap`.

### 3. Roll Localization Drift

Problem:

The detector was stable under camera roll, but it was stably wrong. The
published junction line and 3D gap localization drifted sideways when the camera
rolled.

Root cause:

`_inverse_rotate_uv()` used the wrong sine signs when mapping pixels from the
rotated detector strip back to the original image. This was invisible at
`angle=0` but grew with roll:

```text
buggy inverse:  ~73 px error at 10 deg, ~110 px at 15 deg
fixed inverse:  <1 px round-trip error
```

Impact:

This affected both the displayed junction line and the 3D seam localization used
to publish the robot-frame gap pose. It was therefore a real robot-target bug,
not just an overlay issue.

Validation signal:

```text
PROJECTED-GT (tilt-free, stamp-synced)
```

Before fix:

```text
median ~= 42 px
p90    ~= 87 px
max    ~= 118 px
```

After fix, final validation:

```text
median = 2.57 px
p90    = 3.51 px
max    = 4.19 px
```

## Ground Truth Methods

Two GT layers are used.

### Image GT

The analyzer finds the dark gap from the raw RGB frame using a horizontal
black-hat style contrast score. This is useful for quick checks:

```text
ACCURACY(GT) median|err|
ACCURACY(GT) p90
gap_visible_but_rejected
```

Limitation:

Image GT depends on contrast and can be imperfect when the pipe is rolled,
partially visible, or visually ambiguous.

### Projected GT

Projected GT is the rigorous metric for final validation.

It uses:

- logged per-frame camera pose from TF;
- known Gazebo gap position in `base_link`;
- camera intrinsics;
- exact detector/frame stamp matching.

The known gap is projected into the camera image and compared to the detector's
published junction line. This is independent of contrast and roll.

This is the metric that exposed the roll-localization bug.

## Final Validation Run

Final clean run:

```text
motion_sequences/pipe_motion_final_validation_20260629_125748
```

Result:

```text
[final-validation] RESULT: PASS
```

Key metrics:

```text
frames                      = 359
accepted                    = 1.000
lock                        = 1.000
depth                       = 0.766
x_range                     = 51.0 px
median|dx|                  = 0.0 px
max|dx|                     = 4.0 px
jumps > 50 px               = 0
max_lock_missed             = 3

ACCURACY(GT) median|err|    = 10.0 px
ACCURACY(GT) p90            = 25.0 px
ACCURACY(GT) max            = 30.0 px
gap_visible_but_rejected    = 0/280
near_border_no_gap          = 0

PROJECTED-GT median         = 2.57 px
PROJECTED-GT p90            = 3.51 px
PROJECTED-GT max            = 4.19 px
PROJECTED-GT eval frames    = 82
```

Interpretation:

- The detector/tracker is stable during the aggressive wrist sweep.
- The projected geometric junction error is about 2-4 px.
- No large tracking jumps were observed.
- No visible-gap rejection remained in the final run.
- No clipped/border false accept was observed.
- Image GT has larger error than projected GT because it is a contrast-based
  heuristic; projected GT is the final accuracy metric.

## Final Validation Script

The wrapper script is:

```text
scripts/run_pipe_junction_final_validation.py
```

It assumes Gazebo, camera bridges, detector, XBot2, and the robot home pose are
already up. It then:

1. creates a fresh `motion_sequences/<RUN_ID>` folder;
2. records RGB/depth/status/detection/TF camera pose;
3. writes `rgb_overlay.mp4`;
4. runs the wrist sweep;
5. runs `analyze_pipe_junction_sequence.py`;
6. checks final PASS/FAIL thresholds.

Default PASS thresholds:

```text
accepted_fraction        >= 0.95
lock_active_fraction     >= 0.95
jump_count               <= 0
gap_visible_but_rejected <= 2
near_border_no_gap       <= 0
PROJECTED-GT median      <= 4 px
PROJECTED-GT p90         <= 6 px
PROJECTED-GT max         <= 8 px
PROJECTED-GT eval_frames >= 40
```

## Commands

Run these in the container unless noted otherwise.

### 1. Source Environment

```bash
cd /home/user/concert_ws/src/acea_concert

source /opt/ros/jazzy/setup.bash
source /opt/xbot/setup.bash 2>/dev/null || true
source /opt/xbot/share/xbot_msgs/local_setup.bash
source /opt/xbot/share/cartesian_interface_ros/local_setup.bash
source /home/user/concert_ws/install/setup.bash
```

### 2. Start A Clean Detector

Stop old copies first:

```bash
pkill -f acea_pipe_junction_node || true
pkill -f gap_pose_robot_node || true
```

Start the arm-camera detector:

```bash
ros2 launch /home/user/concert_ws/src/acea_concert/launch/arm_camera_detection.launch.py \
  detector_start_delay_s:=3.0 \
  camera_qos_reliability:=reliable
```

### 3. Home The Robot

```bash
cd /home/user/concert_ws/src/acea_concert

source /opt/ros/jazzy/setup.bash
source /opt/xbot/setup.bash 2>/dev/null || true
source /opt/xbot/share/xbot_msgs/local_setup.bash
source /opt/xbot/share/cartesian_interface_ros/local_setup.bash
source /home/user/concert_ws/install/setup.bash

timeout 5 ros2 service call /xbotcore/ros_ctrl/switch std_srvs/srv/SetBool "{data: true}" || true

ros2 run acea_concert home_to_weld_start.py --duration 8.0
```

Quick detector check:

```bash
ros2 topic echo /acea/pipe_junction/status --once --full-length
```

Expected healthy state:

```text
detected: true
detector_accepted: true
junction_lock_active: true
gap_plane_available: true
weld_seam_pose_available: true
```

### 4. Run Final Validation

```bash
cd /home/user/concert_ws/src/acea_concert

source /opt/ros/jazzy/setup.bash
source /opt/xbot/setup.bash 2>/dev/null || true
source /opt/xbot/share/xbot_msgs/local_setup.bash
source /opt/xbot/share/cartesian_interface_ros/local_setup.bash
source /home/user/concert_ws/install/setup.bash

python3 scripts/run_pipe_junction_final_validation.py
```

Expected end line:

```text
[final-validation] RESULT: PASS
```

The latest run id is written to:

```text
/tmp/acea_last_run_id
```

Inspect outputs:

```bash
export RUN_ID=$(cat /tmp/acea_last_run_id)

ls -lh motion_sequences/$RUN_ID/rgb_overlay.mp4
ls -lh motion_sequences/$RUN_ID/analysis_summary.json
ls -lh motion_sequences/$RUN_ID/analysis.csv
ls -lh motion_sequences/$RUN_ID/analysis_contact_sheet.png

xdg-open motion_sequences/$RUN_ID/rgb_overlay.mp4
```

Re-run only the analyzer:

```bash
python3 scripts/analyze_pipe_junction_sequence.py \
  --seq motion_sequences/$RUN_ID \
  --jump-threshold-px 50
```

## Artifacts To Keep

For a final validation run, keep:

```text
motion_sequences/<RUN_ID>/rgb_overlay.mp4
motion_sequences/<RUN_ID>/analysis_summary.json
motion_sequences/<RUN_ID>/analysis.csv
motion_sequences/<RUN_ID>/analysis_contact_sheet.png
motion_sequences/<RUN_ID>/frames.jsonl
motion_sequences/<RUN_ID>/detection.jsonl
motion_sequences/<RUN_ID>/status.jsonl
```

`frames.jsonl` contains the per-frame RGB stamp and camera pose used by
projected GT. `detection.jsonl` contains the exact detector output matched by
RGB timestamp.

## Known Acceptable Behavior

Short one- or two-frame `SCAN SEARCH` visual drops can happen when the raw
detector proposes a large suspicious candidate jump. If the lock remains active
and the next frames re-acquire correctly, this is preferable to accepting a false
jump.

Do not relax the jump gate only to remove rare micro-drops unless downstream
control explicitly requires a different stale-pose policy.

## Current Recommendation

Do not tune detector thresholds further based on this validation run.

The next useful detection-side work is documentation/contract hardening for the
controller consumer:

- which topic to use;
- which validity flags must be true;
- what to do on one- or two-frame misses;
- maximum acceptable pose age;
- how to log final integration tests.
