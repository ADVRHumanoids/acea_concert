#!/bin/bash
# Patches non-persistent files inside the container using copies from concert_weld.
# Run this after container startup or after rebuilding modular.

WELD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Patching modular config_file.yaml..."
cp "$WELD_DIR/config_file.yaml" /home/user/concert_ws/src/modular/src/modular/config_file.yaml

echo "Done."
