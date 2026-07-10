"""
A minimal ROS 2 node that sends only gravity compensation torques to the robot.
"""

import rclpy
from rclpy.node import Node
from xbot_msgs.msg import JointCommand
from xbot2_interface import pyxbot2_interface as xbi

from controller_ros import fetch_robot_description


BASE_AND_WHEEL_JOINTS = (
    'J1_A', 'J1_B', 'J1_C', 'J1_D',
    'J_wheel_A', 'J_wheel_B', 'J_wheel_C', 'J_wheel_D',
)


def _is_gravity_comp_joint(name):
    lower_name = name.lower()
    return (
        name not in BASE_AND_WHEEL_JOINTS
        and not lower_name.startswith(('base', 'world', 'floating_base'))
    )


def _stamp_from_xbot_time(xbot_time):
    epoch = xbot_time.replace(
        year=1970, month=1, day=1, hour=0, minute=0, second=0,
        microsecond=0)
    delta = xbot_time - epoch
    sec = delta.days * 24 * 60 * 60 + delta.seconds
    nanosec = delta.microseconds * 1000
    return sec, nanosec


class GravityCompNode(Node):
    def __init__(self):
        super().__init__('gravity_comp_node')

        print("[gravity_comp] Waiting for robot description …")

        urdf, srdf = fetch_robot_description('gravity_comp_urdf_reader')
        print("[gravity_comp] URDF and SRDF received.")

        # ── Build ConfigOptions for RobotInterface2 ───────────────────────────────────
        cfg = xbi.ConfigOptions()
        cfg.set_urdf(urdf)
        cfg.set_srdf(srdf)
        cfg.set_string_parameter('model_type', 'pin')
        cfg.set_bool_parameter('is_model_floating_base', True)
        cfg.set_string_parameter('framework', 'ros2')

        # ── Create RobotInterface2 and sense initial state ────────────────────────────
        print("[controller] Connecting to RobotInterface2 …")
        self.robot = xbi.RobotInterface2(cfg)
        self.robot.sense()

        robot_q_map = self.robot.qToMap(self.robot.getJointPosition())

        # ── Build ModelInterface2 for the standalone solver ───────────────────────────
        self.model = xbi.ModelInterface2(urdf, srdf, 'pin')

        # Sync model to robot: actuated joints
        self.model.setJointPosition(robot_q_map)
        self.model.update()

        gcomp_map = self.model.vToMap(self.model.computeGravityCompensation())
        self.gravity_comp_joints = [
            name for name in robot_q_map
            if _is_gravity_comp_joint(name) and name in gcomp_map
        ]
        if not self.gravity_comp_joints:
            raise RuntimeError("No arm joints are available for gravity comp.")

        self.command_pub = self.create_publisher(
            JointCommand, '/xbotcore/command', 1)
        self.timer = self.create_timer(0.01, self.send_gravity_comp)
        self.get_logger().info(
            "Gravity compensation node started for joints: "
            f"{self.gravity_comp_joints}")

            
    def send_gravity_comp(self):
        # Get robot state
        self.robot.sense()
        # Sync model to robot
        self.model.setJointPosition(self.robot.qToMap(self.robot.getJointPosition()))
        # Update model
        self.model.update()
        # Compute and send gravity compensation torques
        gcomp_map = self.model.vToMap(self.model.computeGravityCompensation())
        msg = JointCommand()
        msg.header.stamp.sec, msg.header.stamp.nanosec = _stamp_from_xbot_time(
            self.robot.getTimestamp())
        msg.name = list(self.gravity_comp_joints)
        msg.effort = [
            float(gcomp_map[name])
            for name in self.gravity_comp_joints
        ]
        msg.ctrl_mode = [
            int(xbi.ControlMode.EFFORT)
            for _ in self.gravity_comp_joints
        ]
        self.command_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GravityCompNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
