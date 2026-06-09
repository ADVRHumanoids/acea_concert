"""
A minimal ROS 2 node that sends only gravity compensation torques to the robot.
"""

import rclpy
from rclpy.node import Node
from xbot2_interface import pyxbot2_interface as xbi

from controller_ros import fetch_robot_description

class GravityCompNode(Node):
    def __init__(self):
        super().__init__('gravity_comp_node')

        # ── Read URDF/SRDF from robot_description_publisher ───────────────────────────
        print("[controller] Waiting for robot_description ROS parameters …")

        urdf, srdf = fetch_robot_description('gravity_comp_urdf_reader')
        print("[controller] URDF and SRDF received.")

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
        self.robot.setControlMode(xbi.ControlMode.EFFORT)
        self.robot.sense()
        

        robot_q_map = self.robot.qToMap(self.robot.getJointPosition())

        # ── Build ModelInterface2 for the standalone solver ───────────────────────────
        self.model = xbi.ModelInterface2(urdf, srdf, 'pin')

        # Sync model to robot: actuated joints
        self.model.setJointPosition(robot_q_map)
        self.model.update()

        self.timer = self.create_timer(0.01, self.send_gravity_comp)
        self.get_logger().info('Gravity compensation node started.')

            
    def send_gravity_comp(self):
        # Get robot state
        self.robot.sense()
        # Sync model to robot
        self.model.setJointPosition(self.robot.qToMap(self.robot.getJointPosition()))
        # Update model
        self.model.update()
        # Compute and send gravity compensation torques
        self.robot.setEffortReference(self.model.computeGravityCompensation())
        # Send command to robot
        self.robot.move() 


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
