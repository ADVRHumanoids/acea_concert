#!/usr/bin/env python3
"""
gap_pose_publisher.py
─────────────────────
Reads Gazebo entity poses directly from the gz transport layer via a
background subprocess (gz topic -e) and publishes:

  /gap/pose_world  (PoseStamped, frame: world)      – gap centre in world
  /gap/pose_robot  (PoseStamped, frame: base_link)  – gap in robot base frame
  /gap/y_axis_world (Vector3Stamped, frame: world)  – unit y-gap direction
  /gap/y_axis_robot (Vector3Stamped, frame: base_link) – direction in base
  /robot/base_pose (PoseStamped, frame: world)      – robot base in world

No ros_gz_bridge configuration needed — works out of the box.

Usage:
  python3 src/gap_pose_publisher.py
"""

import re
import subprocess
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped

# ── Gazebo model names (from gz topic -e /world/default/pose/info) ─────────
ROBOT_MODEL     = 'ModularBot'        # capital M and B — verified from Gazebo
ROBOT_BASE_LINK = 'ModularBot::base_link'
GAP_MODEL_LEFT  = 'weld_pipe_left'
GAP_MODEL_RIGHT = 'weld_pipe_right'

GZ_POSE_TOPIC = '/world/default/pose/info'
WORLD_FRAME   = 'world'
BASE_FRAME    = 'base_link'


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q = [x, y, z, w]."""
    t = 2.0 * np.cross(q[:3], v)
    return v + q[3] * t + np.cross(q[:3], t)


def _transform_world_to_frame(t_world: np.ndarray, q_world: np.ndarray,
                               point_world: np.ndarray) -> np.ndarray:
    """Express point_world in the local frame defined by (t_world, q_world)."""
    q_conj = np.array([-q_world[0], -q_world[1], -q_world[2], q_world[3]])
    return _quat_rotate(q_conj, point_world - t_world)


def _rotate_world_vector_to_frame(q_world: np.ndarray,
                                  vector_world: np.ndarray) -> np.ndarray:
    """Express a world-frame direction vector in the local frame q_world."""
    q_conj = np.array([-q_world[0], -q_world[1], -q_world[2], q_world[3]])
    return _quat_rotate(q_conj, vector_world)


def _gap_y_axis(xyz_left: np.ndarray, xyz_right: np.ndarray) -> np.ndarray:
    """Unit y-gap direction in world frame, from right pipe centre to left."""
    gap_y = xyz_left - xyz_right
    norm = np.linalg.norm(gap_y)
    if norm < 1e-9:
        raise ValueError("Cannot compute y-gap axis: pipe centres coincide")
    return gap_y / norm


def _pose_stamped(frame_id: str, xyz: np.ndarray,
                  quat_xyzw: np.ndarray, stamp) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.header.stamp    = stamp
    msg.pose.position.x    = float(xyz[0])
    msg.pose.position.y    = float(xyz[1])
    msg.pose.position.z    = float(xyz[2])
    msg.pose.orientation.x = float(quat_xyzw[0])
    msg.pose.orientation.y = float(quat_xyzw[1])
    msg.pose.orientation.z = float(quat_xyzw[2])
    msg.pose.orientation.w = float(quat_xyzw[3])
    return msg


def _vector3_stamped(frame_id: str, vector: np.ndarray, stamp) -> Vector3Stamped:
    msg = Vector3Stamped()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.vector.x = float(vector[0])
    msg.vector.y = float(vector[1])
    msg.vector.z = float(vector[2])
    return msg


# ── Protobuf text parser ───────────────────────────────────────────────────────

def _parse_pose_v(text: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Parse one gz.msgs.Pose_V text block into
      {model_name: (xyz [3], quat_xyzw [4])}

    Each pose block looks like:
      pose {
        name: "ModelName"
        position { x: 1.0  y: 2.0  z: 3.0 }
        orientation { x: 0.0  y: 0.0  z: 0.0  w: 1.0 }
      }
    Fields with value 0.0 are omitted by protobuf text format.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Split into individual pose blocks
    blocks = re.split(r'(?=^pose \{)', text, flags=re.MULTILINE)
    for block in blocks:
        m_name = re.search(r'name:\s*"([^"]+)"', block)
        if not m_name:
            continue
        name = m_name.group(1)

        m_pos = re.search(r'position \{([^}]*)\}', block, re.DOTALL)
        if m_pos:
            pos_txt = m_pos.group(1)

            def _getp(f):
                m = re.search(rf'{f}:\s*([-\d.eE+]+)', pos_txt)
                return float(m.group(1)) if m else 0.0

            xyz = np.array([_getp('x'), _getp('y'), _getp('z')])
        else:
            xyz = np.zeros(3)

        # orientation block — need to be careful: x/y/z/w can all be absent (→ 0)
        # but w defaults to 1 in a unit quaternion when absent only if no rotation
        # Gazebo omits w when it's 0, but prints it when non-zero.
        # Strategy: read raw, then if norm≈0 treat as identity
        # re-read w specifically from inside orientation block
        m_ori = re.search(r'orientation \{([^}]*)\}', block, re.DOTALL)
        if m_ori:
            ori_txt = m_ori.group(1)
            def _getf(f):
                m = re.search(rf'{f}:\s*([-\d.eE+]+)', ori_txt)
                return float(m.group(1)) if m else 0.0
            qx, qy, qz, qw = _getf('x'), _getf('y'), _getf('z'), _getf('w')
            # If all zero Gazebo omitted everything → identity
            if qx == 0.0 and qy == 0.0 and qz == 0.0 and qw == 0.0:
                qw = 1.0
        else:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

        quat = np.array([qx, qy, qz, qw])
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            quat /= norm

        result[name] = (xyz, quat)

    return result


# ──────────────────────────────────────────────────────────────────────────────

class GapPosePublisher(Node):

    def __init__(self):
        super().__init__('gap_pose_publisher')

        self._pub_gap_world  = self.create_publisher(PoseStamped, '/gap/pose_world',  10)
        self._pub_gap_robot  = self.create_publisher(PoseStamped, '/gap/pose_robot',  10)
        self._pub_gap_y_world = self.create_publisher(Vector3Stamped, '/gap/y_axis_world', 10)
        self._pub_gap_y_robot = self.create_publisher(Vector3Stamped, '/gap/y_axis_robot', 10)
        self._pub_robot_base = self.create_publisher(PoseStamped, '/robot/base_pose', 10)

        # Latest parsed poses {name: (xyz, quat)}
        self._poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._lock = threading.Lock()

        # Background thread reads gz topic stream
        self._reader = threading.Thread(target=self._gz_reader, daemon=True)
        self._reader.start()

        # Publish timer at 50 Hz
        self.create_timer(0.02, self._publish)

        self.get_logger().info(
            f"Reading '{GZ_POSE_TOPIC}' via gz subprocess. "
            f"Waiting for: [{ROBOT_BASE_LINK} or {ROBOT_MODEL}], "
            f"[{GAP_MODEL_LEFT}], [{GAP_MODEL_RIGHT}]"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _gz_reader(self):
        """Continuously read gz topic stream and update self._poses."""
        proc = subprocess.Popen(
            ['gz', 'topic', '-e', '-t', GZ_POSE_TOPIC],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        buffer = []
        for line in proc.stdout:
            buffer.append(line)
            # A complete Pose_V message ends when we see a new "header {" line
            # (the next message starts) or we accumulate enough lines.
            # Simplest heuristic: flush when we see a lone "}" at indent 0
            # followed by a blank or new header.
            if line.startswith('header {') and len(buffer) > 1:
                block = ''.join(buffer[:-1])  # exclude the new 'header {' line
                parsed = _parse_pose_v(block)
                if parsed:
                    with self._lock:
                        self._poses.update(parsed)
                buffer = [line]  # start new block with the 'header {' line

    # ──────────────────────────────────────────────────────────────────────────
    def _publish(self):
        with self._lock:
            poses = dict(self._poses)

        has_left  = GAP_MODEL_LEFT  in poses
        has_right = GAP_MODEL_RIGHT in poses
        has_robot_base = ROBOT_BASE_LINK in poses
        has_robot_model = ROBOT_MODEL in poses
        has_robot = has_robot_base or has_robot_model

        if not (has_left and has_right and has_robot):
            missing = [n for n, f in [(f'{ROBOT_BASE_LINK} or {ROBOT_MODEL}', has_robot),
                                       (GAP_MODEL_LEFT, has_left),
                                       (GAP_MODEL_RIGHT, has_right)] if not f]
            self.get_logger().warn(
                f"Still waiting for: {missing}", throttle_duration_sec=3.0)
            return

        stamp = self.get_clock().now().to_msg()

        robot_pose_name = ROBOT_BASE_LINK if has_robot_base else ROBOT_MODEL
        robot_xyz,  robot_quat  = poses[robot_pose_name]
        xyz_left,   _           = poses[GAP_MODEL_LEFT]
        xyz_right,  _           = poses[GAP_MODEL_RIGHT]

        # ── Gap centre (world) ────────────────────────────────────────────────
        gap_world = (xyz_left + xyz_right) / 2.0

        # ── Y-gap direction (world and base_link) ─────────────────────────────
        # This is the orientation information that is directly defined by the
        # two pipe centres: the unit vector from right pipe to left pipe.
        gap_y_world = _gap_y_axis(xyz_left, xyz_right)
        gap_y_robot = _rotate_world_vector_to_frame(robot_quat, gap_y_world)

        # ── Gap pose in robot base_link frame ─────────────────────────────────
        # Position: rotate + translate
        gap_in_robot = _transform_world_to_frame(robot_xyz, robot_quat, gap_world)
        identity_quat = np.array([0.0, 0.0, 0.0, 1.0])

        self._pub_gap_world.publish(
            _pose_stamped(WORLD_FRAME, gap_world,    identity_quat,  stamp))
        self._pub_gap_robot.publish(
            _pose_stamped(BASE_FRAME,  gap_in_robot, identity_quat,  stamp))
        self._pub_gap_y_world.publish(
            _vector3_stamped(WORLD_FRAME, gap_y_world, stamp))
        self._pub_gap_y_robot.publish(
            _vector3_stamped(BASE_FRAME, gap_y_robot, stamp))
        self._pub_robot_base.publish(
            _pose_stamped(WORLD_FRAME, robot_xyz,    robot_quat,      stamp))

        # Signed yaw offset of the y-gap direction from base_link +Y.
        gap_yaw_robot = float(np.arctan2(-gap_y_robot[0], gap_y_robot[1]))

        self.get_logger().info(
            f"gap world y={gap_world[1]:.4f}  |  "
            f"gap robot y={gap_in_robot[1]:.4f}  |  "
            f"gap_y_robot=[{gap_y_robot[0]:+.3f}, "
            f"{gap_y_robot[1]:+.3f}, {gap_y_robot[2]:+.3f}]  |  "
            f"y-gap yaw from base +Y={np.degrees(gap_yaw_robot):.2f}°",
            throttle_duration_sec=1.0,
        )


def main():
    rclpy.init()
    node = GapPosePublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
