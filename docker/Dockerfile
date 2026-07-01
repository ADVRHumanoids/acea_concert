FROM hhcmhub/xbot2-noble-dev:latest

# create concert workspace and initialize forest
RUN mkdir -p ~/concert_ws && cd ~/concert_ws && forest init

# source the workspace
RUN echo "source /opt/xbot/setup.sh" >> ~/.bashrc
RUN echo "source ~/concert_ws/setup.bash" >> ~/.bashrc

RUN sudo apt-get update && sudo apt-get install -y swig

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
