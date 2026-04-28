#!/bin/bash
# Patches non-persistent files inside the container using copies from /home/user/patch.
# Run this after container startup or after rebuilding modular.

PATCH_DIR="/home/user/concert_ws/src/acea_concert/mount_files"

echo "Patching modular config_file.yaml..."
cp "$PATCH_DIR/config_file.yaml" /home/user/concert_ws/src/modular/src/modular/config_file.yaml

echo "Patching modular concert_prismatic.py..."
cp "$PATCH_DIR/concert_prismatic.py" /home/user/concert_ws/src/acea_concert/src/modular/concert_prismatic.py

echo "Done."
