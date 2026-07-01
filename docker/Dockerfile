FROM hhcmhub/xbot2-noble-dev:latest

# create concert workspace and initialize forest
RUN mkdir -p ~/concert_ws && cd ~/concert_ws && forest init

# source the workspace
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
RUN echo "source /opt/xbot/setup.sh" >> ~/.bashrc
RUN echo "source ~/concert_ws/setup.bash" >> ~/.bashrc

RUN sudo apt-get update && sudo apt-get install -y \
    swig \
    python3-future \
    ros-jazzy-pinocchio \
    && sudo rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages anytree

# Gazebo RGB-D sensors need EGL. The NVIDIA runtime mounts libEGL_nvidia.so,
# but this base image may only ship Mesa's GLVND vendor JSON, causing OGRE to
# select Mesa and fail in headless Docker. Register NVIDIA as an EGL vendor.
RUN sudo mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '%s\n' \
      '{' \
      '    "file_format_version" : "1.0.0",' \
      '    "ICD" : {' \
      '        "library_path" : "libEGL_nvidia.so.0"' \
      '    }' \
      '}' | sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null

WORKDIR /home/user/concert_ws

# add recipes
RUN forest add-recipes git@github.com:ADVRHumanoids/multidof_recipes.git --clone-protocol https --tag acea

RUN forest grow moveit_compute_default_collisions --clone-protocol https -j7

RUN forest grow horizon --clone-protocol https -j7

RUN forest grow concert_description --clone-protocol https -j7

RUN forest grow cartesian_interface --clone-protocol https -j7

RUN forest grow centauro_cartesio --clone-protocol https -j7

RUN forest grow xbot2_gui_server --clone-protocol https -j7

RUN mkdir -p ~/patch
