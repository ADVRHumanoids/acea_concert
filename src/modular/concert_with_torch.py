import contextlib
import os
import sys
import xml.etree.ElementTree as ET

from modular.URDF_writer import UrdfWriter, write_file_to_stdout

is_floating_base = True

WELD_TORCH_CAMERA_NAME = "camera_F"
WELD_TORCH_CAMERA_PARENT = "ee_F"
WELD_TORCH_CAMERA_XYZ = [0.1, 0.0, -0.05]
WELD_TORCH_CAMERA_RPY = [3.141593, -0.8, 0.0]  # 180 deg roll keeps the view direction, flips camera upright


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        try:
            sys.stdout = devnull
            yield
        finally:
            sys.stdout = old_stdout


def add_weld_torch_camera(urdf_writer):
    ET.SubElement(
        urdf_writer.root,
        "xacro:include",
        filename="${MODULAR_PATH}/modular_data/urdf/concert.sensors.urdf.xacro",
    )
    camera = ET.SubElement(
        urdf_writer.root,
        "xacro:add_realsense_d_camera",
        name=WELD_TORCH_CAMERA_NAME,
        parent_name=WELD_TORCH_CAMERA_PARENT,
        add_gazebo_sensor="true",
    )
    ET.SubElement(
        camera,
        "origin",
        xyz=" ".join(str(x) for x in WELD_TORCH_CAMERA_XYZ),
        rpy=" ".join(str(x) for x in WELD_TORCH_CAMERA_RPY),
    )
    urdf_writer.add_sensor_name("camera", WELD_TORCH_CAMERA_NAME)


with suppress_stdout():

    # create UrdfWriter object and joint map to store homing values
    urdf_writer = UrdfWriter(speedup=True, floating_base=is_floating_base)
    homing_joint_map = {}

    # add mobile base
    urdf_writer.add_module('concert/mobile_platform_concert.json', module_name='mobile_base')

    if is_floating_base:
        # leg + wheel 1
        data = urdf_writer.select_module_from_name('mobile_base_con1')
        wheel_data, steering_data = urdf_writer.add_wheel_module(wheel_filename='concert/module_wheel_concert.json', 
                                            steering_filename='concert/module_steering_concert_fl_rr.json')
        homing_joint_map[str(steering_data['name'])] = 0.0
        homing_joint_map[str(wheel_data['name'])] = 0.0

        # leg + wheel 2
        data = urdf_writer.select_module_from_name('mobile_base_con2')
        wheel_data, steering_data = urdf_writer.add_wheel_module(wheel_filename='concert/module_wheel_concert.json', 
                                            steering_filename='concert/module_steering_concert_fr_rl.json')
        homing_joint_map[str(steering_data['name'])] = 0.0
        homing_joint_map[str(wheel_data['name'])] = 0.0

        # leg + wheel 3
        data = urdf_writer.select_module_from_name('mobile_base_con3')
        wheel_data, steering_data = urdf_writer.add_wheel_module(wheel_filename='concert/module_wheel_concert.json', 
                                            steering_filename='concert/module_steering_concert_fr_rl.json')
        homing_joint_map[str(steering_data['name'])] = 0.0
        homing_joint_map[str(wheel_data['name'])] = 0.0

        # leg + wheel 4
        data = urdf_writer.select_module_from_name('mobile_base_con4')
        wheel_data, steering_data = urdf_writer.add_wheel_module(wheel_filename='concert/module_wheel_concert.json', 
                                            steering_filename='concert/module_steering_concert_fl_rr.json')
        homing_joint_map[str(steering_data['name'])] = 0.0
        homing_joint_map[str(wheel_data['name'])] = 0.0

    # manipulator
    data = urdf_writer.select_module_from_name('mobile_base_con5')

    # Override Gazebo PID inside the resource dict BEFORE adding the module,
    # so UrdfWriter reads the new values when it builds the URDF.
    elbow_A_dict = urdf_writer.modular_resources_manager.available_modules_dict['concert/module_joint_elbow_A_concert.json']
    elbow_A_dict['joints'][0]['control_parameters']['xbot_gz']['pid']['p'] = 5000.0
    elbow_A_dict['joints'][0]['control_parameters']['xbot_gz']['pid']['d'] = 80.0

    elbow_B_dict = urdf_writer.modular_resources_manager.available_modules_dict['concert/module_joint_elbow_B_concert.json']
    elbow_B_dict['joints'][0]['control_parameters']['xbot_gz']['pid']['p'] = 2000.0
    elbow_B_dict['joints'][0]['control_parameters']['xbot_gz']['pid']['d'] = 25.0

    prismatic_dict = urdf_writer.modular_resources_manager.available_modules_dict['experimental/module_joint_prismatic_concert.json']
    prismatic_dict['joints'][0]['control_parameters']['xbot_gz']['pid']['p'] = 5000.0
    prismatic_dict['joints'][0]['control_parameters']['xbot_gz']['pid']['d'] = 80.0

    # big yaw (a.k.a. ralla)
    data = urdf_writer.add_module('experimental/module_joint_yaw_XL_concert.json', offsets={'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0})
    homing_joint_map[str(data['name'])] = -3.14159265 / 2  # -pi/2, matches weld_opt.py

    # prismatic joint  →  J2_E
    data = urdf_writer.add_module('experimental/module_joint_prismatic_concert.json', offsets={'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0})
    homing_joint_map[str(data['name'])] = 1.0 

    data = urdf_writer.add_module('experimental/module_hub_prismatic_cart_concert.json', module_name='hub_prismatic_cart', offsets={'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0})
    homing_joint_map[str(data['name'])] = 0.0

    # # Left mounted interface
    data = urdf_writer.select_module_from_name('hub_prismatic_cart_con4')


    data = urdf_writer.add_module('concert/module_joint_elbow_A_concert.json')  # J1_F
    homing_joint_map[str(data['name'])] = -0.85

    data = urdf_writer.add_module('concert/module_joint_yaw_A_concert.json')    # J2_F
    homing_joint_map[str(data['name'])] = 0.0

    #add a 30cm passive link
    data = urdf_writer.add_module('concert/module_link_straight_300_concert.json')

    data = urdf_writer.add_module('concert/module_joint_elbow_A_concert.json')  # J3_F
    homing_joint_map[str(data['name'])] = -1.47

    data = urdf_writer.add_module('concert/module_joint_yaw_A_concert.json')    # J4_F
    homing_joint_map[str(data['name'])] = 0.0
    
    #add a 30cm passive link
    data = urdf_writer.add_module('concert/module_link_straight_300_concert.json')

    data = urdf_writer.add_module('concert/module_joint_elbow_B_concert.json')  # J5_F
    homing_joint_map[str(data['name'])] = 0.75

    data = urdf_writer.add_module('concert/module_joint_yaw_B_concert.json')    # J6_F
    homing_joint_map[str(data['name'])] = 0.0

    data = urdf_writer.add_module('experimental/module_weld_torch_dummy.json')
    add_weld_torch_camera(urdf_writer)

    # # Right mounted interface
    # data = urdf_writer.select_module_from_name('hub_prismatic_cart_con2')

    # Top mounted interface
    # data = urdf_writer.select_module_from_name('hub_prismatic_cart_con3')


write_file_to_stdout(urdf_writer, homing_joint_map)
