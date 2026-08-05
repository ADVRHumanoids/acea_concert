# Source this file in any terminal before running the CONCERT online stack.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file instead of executing it:"
    echo "  source ${BASH_SOURCE[0]}"
    exit 2
fi

export CONCERT_WS="${CONCERT_WS:-/home/user/concert_ws}"
export ACEA_CONCERT_DIR="${ACEA_CONCERT_DIR:-${CONCERT_WS}/src/acea_concert}"

# Environment ---------------------------------------------------------------

_concert_source_if_exists() {
    local file="$1"
    [[ -f "${file}" ]] && source "${file}"
}

_concert_source_if_exists_quietly() {
    local file="$1"
    [[ -f "${file}" ]] && source "${file}" 2>/dev/null
}

_concert_setup_environment() {
    local file
    local quiet_files=(
        /opt/xbot/setup.bash
        /opt/xbot/setup.sh
    )
    local files=(
        /opt/ros/jazzy/setup.bash
        /opt/xbot/share/xbot_msgs/local_setup.bash
        /opt/xbot/share/cartesian_interface_ros/local_setup.bash
        "${CONCERT_WS}/setup.bash"
        "${CONCERT_WS}/install/setup.bash"
    )

    for file in "${quiet_files[@]}"; do
        _concert_source_if_exists_quietly "${file}"
    done

    for file in "${files[@]}"; do
        _concert_source_if_exists "${file}"
    done
}

_concert_setup_graphics() {
    if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
        export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
    fi
}

_concert_setup_environment
_concert_setup_graphics

# Navigation ----------------------------------------------------------------

concert_ws() {
    cd "${CONCERT_WS}" || return
}

concert_cd() {
    cd "${ACEA_CONCERT_DIR}" || return
}

# Commands ------------------------------------------------------------------

_concert_python_script() {
    local script="$1"
    shift

    concert_cd || return
    python3 "src/${script}" "$@"
}

concert_build() {
    concert_ws || return
    colcon build --symlink-install --packages-select acea_concert "$@"
    _concert_source_if_exists "${CONCERT_WS}/install/setup.bash"
}

concert_sim() {
    local args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --optimize_pose|--optimize-pose|--optimized_pose|--optimized-pose)
                args+=("optimized_robot_pose:=true")
                ;;
            *)
                args+=("$1")
                ;;
        esac
        shift
    done

    concert_ws || return
    ros2 launch acea_concert weld_sim.launch.py "${args[@]}"
}

concert_xbot_gui() {
    if command -v xbot2_gui >/dev/null 2>&1; then
        xbot2_gui "$@"
    elif command -v xbot_gui >/dev/null 2>&1; then
        xbot_gui "$@"
    elif command -v xbot2_gui_server >/dev/null 2>&1; then
        xbot2_gui_server "$@"
    else
        echo "No XBot GUI command found: tried xbot2_gui, xbot_gui, xbot2_gui_server." >&2
        return 127
    fi
}

concert_gap() {
    _concert_python_script gap_pose_publisher.py "$@"
}

concert_gravity() {
    _concert_python_script gravity_comp_node.py "$@"
}

concert_home() {
    _concert_python_script home_to_weld_start.py "$@"
}

concert_optimize_homing() {
    _concert_python_script plan_homing_from_mat.py "$@"
}

concert_optimize_weld() {
    _concert_python_script weld_opt.py "$@"
}

concert_drive() {
    _concert_python_script drive_base_to_weld_pose.py "$@"
}

concert_controller() {
    _concert_python_script controller.py "$@"
}

concert_rviz() {
    rviz2 -d "${ACEA_CONCERT_DIR}/rviz/rviz_config_controller.rviz" "$@"
}

concert_kill() {
    "${ACEA_CONCERT_DIR}/scripts/kill_concert_stack" "$@"
}

# Help ----------------------------------------------------------------------

concert_help() {
    cat <<'EOF'
CONCERT commands:
  concert_sim [args...]   ros2 launch acea_concert weld_sim.launch.py
    alias: concert_sim --optimize_pose -> optimized_robot_pose:=true
  concert_xbot_gui        start XBot GUI/server
  concert_gap             publish /gap/pose_robot from Gazebo ground truth
  concert_gravity         run gravity compensation
  concert_optimize_weld   compute and save the optimized weld trajectory
  concert_optimize_homing compute and save a collision-aware q_homing
  concert_home            move the robot along the saved q_homing
  concert_drive           drive base to optimized weld pose
  concert_controller      run welding controller
  concert_rviz            open controller RViz config
  concert_kill            stop the CONCERT/ROS/Gazebo/XBot stack

Typical order:
  concert_sim
  concert_xbot_gui
  concert_gap
  concert_gravity
  concert_home
  concert_drive
  concert_controller
EOF
}

_concert_ready_help() {
    case "$1" in
        concert_optimize_weld)
            cat <<'EOF'

Weld Optimization

Ready:
  concert_optimize_weld

Main arguments:
  --use-prismatic-joint
  --[no-]upside-down

After it finishes, use the result with:
  concert_sim --optimize_pose
  concert_optimize_homing --initial-pose-from-gazebo --retry
EOF
            ;;
        concert_sim)
            cat <<'EOF'

Ready:
  concert_sim --optimize_pose

Main arguments:
  --optimize_pose       use the optimized robot/pipe pose
  mat_file:=PATH        select an optimization MAT file
EOF
            ;;
        concert_home)
            cat <<'EOF'

Ready:
  concert_home --homing-trajectory mat_files/weld_concert.mat

Main arguments:
  --homing-trajectory PATH

Direct fallback, without an optimized homing path:
  concert_home --duration SECONDS --dt SECONDS
This only interpolates from the current joints to weld q[:, 0].
EOF
            ;;
        concert_optimize_homing)
            cat <<'EOF'

Ready:
  concert_optimize_homing --initial-pose-from-gazebo --retry

Then switch to the Homing Execution pane without moving the robot.

Main arguments:
  --mat-file PATH
  --output PATH
  --duration SECONDS
  --dt SECONDS
  --planner-nodes N
  --initial-pose-from-gazebo
  --retry
  --rviz
EOF
            ;;
        concert_controller)
            cat <<'EOF'

Ready: 
  concert_controller

Main arguments:
  --open-loop
  --[no-]stop-on-gap-loss
  --[no-]tangent-correction
  --gap-pose-timeout SECONDS
  --gap-filter-tau SECONDS
  --gap-filter-history-size N
  --gap-filter-max-position-jump METERS
  --gap-filter-max-angle-jump DEGREES
EOF
            ;;
        *)
            printf '\nReady: %s\n' "$1"
            ;;
    esac
}

if [[ -n "${CONCERT_READY_COMMAND:-}" ]]; then
    _concert_ready_help "${CONCERT_READY_COMMAND}"
    unset CONCERT_READY_COMMAND
elif [[ "${CONCERT_ENV_QUIET:-0}" != "1" ]]; then
    concert_help
fi
