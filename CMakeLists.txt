cmake_minimum_required(VERSION 3.8)
project(acea_concert)

find_package(ament_cmake REQUIRED)

# --- Python scripts (runnable via ros2 run or directly) ---
install(PROGRAMS
  scripts/apply_gz_wrench.py
  scripts/kill_concert_stack
  scripts/omnisteering_sine_command.py
  src/drive_base_to_weld_pose.py
  src/home_to_weld_start.py
  src/weld_opt.py
  src/run_weld_opt_batch.py
  src/plot_weld_opt_batch.py
  src/replayer.py
  src/controller.py
  src/circular_trajectory.py
  src/collision_checker.py
  src/homing_trajectory.py
  src/plan_homing_from_mat.py
  src/gap_pose_publisher.py
  src/plot_weld_result.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY src/modular/
  DESTINATION lib/${PROJECT_NAME}/modular
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY src/viz/
  DESTINATION lib/${PROJECT_NAME}/viz
  FILES_MATCHING PATTERN "*.py"
)

# --- ACEA perception detection module ---
# Pipe-junction detector (camera-frame weld-seam gap geometry) + the perception
# gap_pose_robot publisher that republishes /gap/pose_robot in base_link as a
# drop-in replacement for the Gazebo ground-truth gap_pose_publisher.py.
# See src/detection/README.md. acea_alignment/ is installed next to the nodes so
# their `from acea_alignment...` imports resolve under `ros2 run`.
install(PROGRAMS
  src/detection/acea_pipe_junction_node.py
  src/detection/gap_pose_robot_node.py
  src/detection/acea_seam_detector.py
  src/detection/acea_detection_runtime.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY src/detection/acea_alignment/
  DESTINATION lib/${PROJECT_NAME}/acea_alignment
  FILES_MATCHING PATTERN "*.py"
)

# --- ROS share files ---
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml" PATTERN "*.yml"
)

install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
  FILES_MATCHING PATTERN "*.rviz"
)

install(DIRECTORY world/
  DESTINATION share/${PROJECT_NAME}/world
  OPTIONAL
  FILES_MATCHING PATTERN "*.sdf"
)

install(FILES resource/acea_concert
  DESTINATION share/ament_index/resource_index/packages
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
