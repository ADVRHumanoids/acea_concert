cmake_minimum_required(VERSION 3.8)
project(acea_concert)

find_package(ament_cmake REQUIRED)

# --- Python scripts (runnable via ros2 run or directly) ---
install(PROGRAMS
  scripts/apply_gz_wrench.py
  scripts/kill_concert_stack
  scripts/omnisteering_sine_command.py
  scripts/read_ros_string_xml.py
  scripts/start_xbot2_after_clock.sh
  src/drive_base_to_weld_pose.py
  src/home_to_weld_start.py
  src/weld_opt.py
  src/run_weld_opt_batch.py
  src/plot_weld_opt_batch.py
  src/replayer.py
  src/controller.py
  src/plan_homing_from_mat.py
  src/gap_pose_publisher.py
  src/plot_weld_result.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY src/acea_concert/
  DESTINATION lib/${PROJECT_NAME}/acea_concert
  FILES_MATCHING PATTERN "*.py"
)

# --- ACEA perception detection module (validated V16 line) ---
# Publishes /gap/pose_robot (base_link) as a drop-in replacement for the
# Gazebo ground-truth gap_pose_publisher.py. acea_alignment/ is installed next
# to the nodes so their `from acea_alignment...` imports resolve via ros2 run.
install(PROGRAMS
  src/detection/acea_pipe_junction_node_v16_dev.py
  src/detection/acea_fullframe_continuity_v15.py
  src/detection/acea_seam_detector_v15.py
  src/detection/acea_detection_runtime.py
  src/detection/gap_pose_robot_node_v14_dev.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY src/detection/acea_alignment/
  DESTINATION lib/${PROJECT_NAME}/acea_alignment
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY src/modular/
  DESTINATION lib/${PROJECT_NAME}/modular
  FILES_MATCHING PATTERN "*.py"
)

install(DIRECTORY src/viz/
  DESTINATION lib/${PROJECT_NAME}/viz
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

install(FILES resource/acea_concert
  DESTINATION share/ament_index/resource_index/packages
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
