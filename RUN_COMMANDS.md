# CONCERT Quick Runbook

Run each long-lived command in a separate terminal. In every terminal:

```bash
source ~/concert_ws/install/setup.bash
cd ~/concert_ws/src/acea_concert
```

After changing source files, install with the workspace's normal build:

```bash
cd ~/concert_ws/build/acea_concert
make -j8 install
```

## 1. Simulator

Start with the pipes already at the optimized pose relative to the robot:

```bash
ros2 launch acea_concert weld_sim.launch.py optimized_robot_pose:=true
```

Start without optimized placement:

```bash
ros2 launch acea_concert weld_sim.launch.py
```

Wait until Gazebo and XBot are ready, then start ROS control manually:

```bash
ros2 service call /xbotcore/ros_ctrl/switch \
  std_srvs/srv/SetBool "{data: true}"
```

## 2. Gravity compensation

Keep this running:

```bash
python3 src/gravity_comp_node.py
```

## 3. Simulated gap pose

The drive and closed-loop welding controller need `/gap/pose_robot`. Keep this
running when perception is not providing that topic:

```bash
python3 src/gap_pose_publisher.py
```

## 4. Drive the mobile base

```bash
python3 src/drive_base_to_weld_pose.py
```

Skip this when the simulator was launched with
`optimized_robot_pose:=true`. For intelligent-fold homing in the normal
simulation, drive first and plan the homing trajectory afterward.

## 5. Choose one homing method

### Normal homing

Directly interpolate the arm joints to the first welding configuration:

```bash
python3 src/home_to_weld_start.py
```

### Intelligent-fold homing

Plan from the current robot and pipe poses:

```bash
python3 src/plan_homing_from_mat.py \
  --initial-pose-from-gazebo \
  --output /tmp/weld_concert_compact.mat \
  --duration 8
```

Then replay the planned trajectory:

```bash
python3 src/home_to_weld_start.py \
  --homing-trajectory /tmp/weld_concert_compact.mat
```

Do not move the robot between planning and replay. Intelligent-fold homing
replaces normal homing; do not run both.

## 6. Welding controller

After homing:

```bash
python3 src/controller.py
```

Open-loop mode:

```bash
python3 src/controller.py --open-loop
```

## Stop the simulation stack

```bash
scripts/kill_concert_stack
```
