import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Hardcoded mat file path
matfile = os.path.join(os.path.dirname(__file__), '../mat_files/weld_concert.mat')
if not os.path.exists(matfile):
    print(f"File not found: {matfile}")
    sys.exit(1)

data = scipy.io.loadmat(matfile)

# Extract relevant fields
q = data.get('q')
v = data.get('v')
a = data.get('a')
tau = data.get('tau')
pos_center_pipe = data.get('pos_center_pipe')
radius_pipe = data.get('radius_pipe')
angle_weld_start = data.get('angle_weld_start')
angle_weld_end = data.get('angle_weld_end')
joint_names = [str(j[0]) if isinstance(j, np.ndarray) else str(j) for j in data['joint_names'].squeeze()]

joint_names = joint_names[2:]


# Plot joint positions
if q is not None:
    plt.figure()
    for i in range(q[15:].shape[0]):
        label = joint_names[i + 15 - 7] if joint_names and i + 15 - 7 < len(joint_names) else None
        plt.plot(q[15:][i, :], label=label)
    plt.title('Joint Positions (q)')
    plt.xlabel('Node')
    plt.ylabel('Position')
    plt.grid(True)
    plt.legend()

# Plot joint velocities
if v is not None:
    plt.figure()
    for i in range(v[14:].shape[0]):
        label = joint_names[i + 14 - 6] if joint_names and i + 14 - 6 < len(joint_names) else None
        plt.plot(v[14:][i, :], label=label)
    plt.title('Joint Velocities (v)')
    plt.xlabel('Node')
    plt.ylabel('Velocity')
    plt.grid(True)
    plt.legend()

if a is not None:
    plt.figure()
    for i in range(a[14:].shape[0]):
        label = joint_names[i + 14 - 6] if joint_names and i + 14 - 6 < len(joint_names) else None
        plt.plot(a[14:][i, :], label=label)
    plt.title('Joint Accelerations (a)')
    plt.xlabel('Node')
    plt.ylabel('Acceleration')
    plt.grid(True)
    plt.legend()

# Plot joint torques
if tau is not None:
    plt.figure()
    for i in range(tau[14:].shape[0]):
        label = joint_names[i + 14 - 6] if joint_names and i + 14 - 6 < len(joint_names) else None
        plt.plot(tau[14:][i, :], label=label)
    plt.title('Joint Torques (tau)')
    plt.xlabel('Node')
    plt.ylabel('Torque')
    plt.grid(True)
    plt.legend()

# # Plot end-effector trajectory if possible
# if q is not None and pos_center_pipe is not None and radius_pipe is not None:
#     try:
#         # Assume first 3 joints are base position (x, y, z)
#         ee_xyz = q[:3, :]
#         plt.figure()
#         plt.plot(ee_xyz[0, :], ee_xyz[1, :], label='Trajectory (x vs y)')
#         plt.scatter(pos_center_pipe[0], pos_center_pipe[1], c='r', label='Pipe Center')
#         plt.title('End-Effector Trajectory (x-y)')
#         plt.xlabel('x')
#         plt.ylabel('y')
#         plt.axis('equal')
#         plt.legend()
#         plt.grid(True)
#     except Exception as e:
#         print(f"Could not plot end-effector trajectory: {e}")

plt.show()
