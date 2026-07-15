#!/bin/bash
# Patches non-persistent files inside the container using copies from /home/user/patch.
# Run this after container startup or after rebuilding modular.

set -e

CONCERT_WS="/home/user/concert_ws"
PACKAGE_DIR="$CONCERT_WS/src/acea_concert"
PATCH_DIR="$PACKAGE_DIR/mount_files"
BUILD_DIR="$CONCERT_WS/build/acea_concert"
INSTALL_DIR="$CONCERT_WS/install"

echo "Patching modular config_file.yaml..."
cp "$PATCH_DIR/config_file.yaml" /home/user/concert_ws/src/modular/src/modular/config_file.yaml

echo "Patching modular concert_prismatic.py..."
cp "$PATCH_DIR/concert_prismatic.py" /home/user/concert_ws/src/acea_concert/src/modular/concert_prismatic.py

echo "Configuring acea_concert..."
source /opt/xbot/setup.sh
source "$CONCERT_WS/setup.bash"

cmake \
    -S "$PACKAGE_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"

echo "Building and installing acea_concert..."
cmake --build "$BUILD_DIR" --parallel "$(nproc)"
cmake --install "$BUILD_DIR"

echo "Container patches applied and acea_concert installed."
