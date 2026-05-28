import subprocess

class InitScene:
    def __init__(self, path_ws, pos_center_pipe, radius_pipe, length_pipe, orientation_pipe, 
                 footprint_robot_x, footprint_robot_y, center_x, center_y, size_x, size_y, position):
        
        self.path_ws = path_ws
        self.pos_center_pipe = pos_center_pipe
        self.radius_pipe = radius_pipe
        self.length_pipe = length_pipe
        self.orientation_pipe = orientation_pipe
        self.footprint_robot_x = footprint_robot_x
        self.footprint_robot_y = footprint_robot_y
        self.center_x = center_x
        self.center_y = center_y
        self.size_x = size_x
        self.size_y = size_y
        self.position = position

    def kill_existing_markers(self):

        subprocess.run("pkill -f rviz_pipe_marker.py", shell=True)
        subprocess.run("pkill -f rviz_rectangle_marker.py", shell=True)
        subprocess.run("pkill -f rviz_line_marker.py", shell=True)

    def launch_scene(self):

        # Pipe marker
        subprocess.Popen([
            "python3", str(self.path_ws / "rviz_pipe_marker.py"),
            'weld_pipe',
            str(self.pos_center_pipe[0]), str(self.pos_center_pipe[1]), str(self.pos_center_pipe[2]),
            str(self.radius_pipe), str(self.length_pipe),
            str(self.orientation_pipe[0]), str(self.orientation_pipe[1]), str(self.orientation_pipe[2]), str(self.orientation_pipe[3])
        ])
        # Footprint marker
        subprocess.Popen([
            "python3", str(self.path_ws / "rviz_rectangle_marker.py"),
            'footprint_robot',
            str(0.0), str(0.0),
            str(self.footprint_robot_x), str(self.footprint_robot_y),
            'base_link',
            '0.0', '1.0', '0.0', '0.3'
        ])
        # Initial zone marker
        subprocess.Popen([
            "python3", str(self.path_ws / "rviz_rectangle_marker.py"),
            'initial_zone',
            str(self.center_x), str(self.center_y),
            str(self.size_x), str(self.size_y),
            'world',

        ])
        # Line marker
        line_points = []
        for k in range(self.position.shape[1]):
            line_points.extend([self.position[0, k], self.position[1, k], self.position[2, k]])
        subprocess.Popen([
            "python3", str(self.path_ws / "rviz_line_marker.py"),
            'weld_trajectory',
            *[str(x) for x in line_points]
        ])