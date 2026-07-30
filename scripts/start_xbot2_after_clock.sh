#!/usr/bin/env bash

set -Eeuo pipefail

timeout_seconds=120
settle_seconds=1.0
xbot2_config=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout)
            timeout_seconds="$2"
            shift 2
            ;;
        --settle)
            settle_seconds="$2"
            shift 2
            ;;
        --config)
            xbot2_config="$2"
            shift 2
            ;;
        *)
            echo "start_xbot2_after_clock: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${xbot2_config}" ]]; then
    echo "start_xbot2_after_clock: --config is required" >&2
    exit 2
fi

echo "[start_xbot2_after_clock] waiting for an active Gazebo /clock"
timeout --signal=TERM "${timeout_seconds}" \
    ros2 topic echo /clock --once >/dev/null
sleep "${settle_seconds}"

echo "[start_xbot2_after_clock] clock active; starting xbot2-core"

# Keep a narrow retry for XBot's known XML-parser SIGSEGV (shell status 139).
# The simulation config uses read_ros_string_xml.py to prevent middleware
# diagnostics or YAML serialization from preceding the XML document. Every
# other failure remains immediately visible to the validation runner.
max_attempts=3
for ((attempt = 1; attempt <= max_attempts; ++attempt)); do
    core_pid=""
    forward_signal() {
        if [[ -n "${core_pid}" ]] && kill -0 "${core_pid}" 2>/dev/null; then
            kill -TERM "${core_pid}" 2>/dev/null || true
        fi
    }
    trap forward_signal INT TERM

    set +e
    xbot2-core -V --hw sim --simtime --config "${xbot2_config}" -- &
    core_pid=$!
    rc=$?
    set -e

    if [[ "${rc}" -eq 0 ]]; then
        ros_ctrl_started=false
        for ((poll = 0; poll < timeout_seconds * 2; ++poll)); do
            if ! kill -0 "${core_pid}" 2>/dev/null; then
                break
            fi
            set +e
            response="$(
                timeout 2 ros2 service call /xbotcore/ros_ctrl/switch \
                    std_srvs/srv/SetBool "{data: true}" 2>&1
            )"
            service_rc=$?
            set -e
            if [[ "${service_rc}" -eq 0 ]] \
                && grep -Eq 'success=True|success: true' <<<"${response}"; then
                echo "[start_xbot2_after_clock] ros_ctrl running"
                ros_ctrl_started=true
                break
            fi
            sleep 0.5
        done

        if [[ "${ros_ctrl_started}" != "true" ]] && kill -0 "${core_pid}" 2>/dev/null; then
            echo "[start_xbot2_after_clock] ros_ctrl did not start" >&2
            kill -TERM "${core_pid}" 2>/dev/null || true
        fi

        set +e
        wait "${core_pid}"
        rc=$?
        set -e
    fi
    core_pid=""
    trap - INT TERM

    if [[ "${rc}" -eq 0 ]]; then
        exit 0
    fi
    if [[ "${rc}" -ne 139 || "${attempt}" -ge "${max_attempts}" ]]; then
        exit "${rc}"
    fi
    echo "[start_xbot2_after_clock] xbot2-core XML parse crash; retry ${attempt}/${max_attempts}" >&2
    sleep 2
done
