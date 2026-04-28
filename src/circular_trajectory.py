import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_circular_trajectory(ns, center, radius, angle_start=0.0, angle_end=2*np.pi, upside_down=False):
    """
    Generate a circular trajectory in the XZ plane (constant Y).

    The end-effector orientation is set so that the tool z-axis always points
    radially inward (normal to the circle toward the center).

    Args:
        ns: number of trajectory steps
        center: [x, y, z] center of the circle in world frame
        radius: radius of the circle in meters
        angle_start: starting angle in radians (default 0)
        angle_end: ending angle in radians (default 2*pi for full circle)

    Returns:
        positions: (3, ns+1) numpy array
        orientations: (4, ns+1) numpy array of quaternions (x, y, z, w)
    """
    center = np.asarray(center, dtype=float)
    positions = np.zeros((3, ns + 1))
    orientations = np.zeros((4, ns + 1))  # Quaternion: x, y, z, w

    for k in range(ns + 1):
        angle = angle_start + (angle_end - angle_start) * k / ns

        # Position: circle in XZ plane at constant Y
        positions[0, k] = center[0] + radius * np.cos(angle)
        positions[1, k] = center[1]
        positions[2, k] = center[2] + radius * np.sin(angle)


        # Add wiggling sinusoidal y motion
        # amplitude = 0.02  # amplitude of the wiggle (meters)
        # frequency = 10    # number of wiggles per full circle
        # positions[1, k] = center[1] + amplitude * np.sin(frequency * angle)

        # Orientation: tool z-axis points radially inward from center
        normal = np.array([-np.cos(angle), 0.0, -np.sin(angle)])
        # normal = np.array([np.sin(angle), 0.0, -np.cos(angle)])
        y_axis = np.array([0.0, -1.0, 0.0])

        z_tool = normal
        y_tool = y_axis
        x_tool = np.cross(y_tool, z_tool)
        rot_matrix = np.column_stack([x_tool, y_tool, z_tool])

        # Rotate the frame by 180 deg around z so x aligns with -x
        if upside_down:
            rot_z_180 = R.from_euler('z', 180, degrees=True).as_matrix()
            rot_matrix = rot_matrix @ rot_z_180
            
        orientations[:, k] = R.from_matrix(rot_matrix).as_quat()  # Quaternion (x, y, z, w)

    return positions, orientations
