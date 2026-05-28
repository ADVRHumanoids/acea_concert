from scipy.io import loadmat
import os

# Path to your mat file
mat_path = '../mat_files/weld_concert.mat'
# Try alternative absolute path if not found
if not os.path.exists(mat_path):
    mat_path = os.path.join(os.path.dirname(__file__), '../mat_files/weld_concert.mat')
    if not os.path.exists(mat_path):
        mat_path = os.path.join(os.path.dirname(__file__), '../../mat_files/weld_concert.mat')
    if not os.path.exists(mat_path):
        mat_path = os.path.join(os.path.dirname(__file__), '../src/acea_concert/mat_files/weld_concert.mat')
print(f"[INFO] Using mat file: {mat_path}")
m = loadmat(mat_path)

# Extract initial robot base pose (first 3 values)
base_pose = m['initial_robot_pose'].flatten()
base_x, base_y, base_z = base_pose[0], base_pose[1], base_pose[2]

# These should match your weld_opt.py pipe center
pos_center_pipe = [1.5, 0.0, 0.75]  # [X, Y, Z]

pipe_x = pos_center_pipe[0] - base_x
pipe_y = pos_center_pipe[1] - base_y
pipe_z = pos_center_pipe[2] - base_z

print(f"# Copy these values to weld_sim.launch.py:")
print(f"PIPE_X = {pipe_x:.6f}")
print(f"PIPE_Y = {pipe_y:.6f}")
print(f"PIPE_Z = {pipe_z:.6f}")
