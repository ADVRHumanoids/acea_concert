#!/bin/bash
# Patches non-persistent files inside the container using copies from /patch.
# Run this after container startup or after rebuilding modular.

echo "Patching modular config_file.yaml..."
cp "config_file.yaml" /home/user/concert_ws/src/modular/src/modular/config_file.yaml

echo "Patching modular concert_prismatic.py..."
cp "concert_prismatic.py" /home/user/concert_ws/src/concert_description/concert_examples/src/concert_prismatic.py

echo "Done."
