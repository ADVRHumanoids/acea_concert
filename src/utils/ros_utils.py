import rclpy
from rcl_interfaces.srv import GetParameters

def fetch_robot_description(node_name):
    node   = rclpy.create_node(node_name)
    client = node.create_client(GetParameters, '/robot_description_publisher/get_parameters')
    if not client.wait_for_service(timeout_sec=15.0):
        raise RuntimeError("[controller] /robot_description_publisher not available. Is the simulation running?")
    req       = GetParameters.Request()
    req.names = ['robot_description', 'robot_description_semantic']
    future    = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
    node.destroy_node()
    if future.result() is None:
        raise RuntimeError("[controller] Failed to read robot_description parameters.")
    vals = future.result().values
    return vals[0].string_value, vals[1].string_value