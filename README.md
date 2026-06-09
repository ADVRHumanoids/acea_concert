# acea_concert

Tools for optimizing and executing the CONCERT welding trajectory.

## Pipeline

The workflow is split into two parts:

1. Offline optimization: generate the optimal trajectory and save it to a MAT file.
2. Online controller: run the simulation, publish the current gap pose, compensate gravity, and execute the controller.

## 1. Offline Optimization

Run the optimizer from the package root:

```bash
cd /home/user/concert_ws/src/acea_concert
python3 src/weld_opt.py
```

`src/weld_opt.py` solves the Horizon optimization problem and saves:

```text
mat_files/weld_concert.mat
```

The MAT file contains the optimized joint trajectory, the pipe/gap geometry, and the weld trajectory expressed in the gap frame. The online controller expects this file to exist before starting the simulation.

Use RViz to inspect the robot, pipe, and optimized weld trajectory:

```bash
rviz2 -d rviz/rviz_config.rviz
```

## 2. Online Controller

Start each component in a separate terminal.

### Simulation

```bash
cd /home/user/concert_ws
ros2 launch acea_concert weld_sim.launch.py
```

This launches the robot simulation and spawns the two pipe halves using the geometry stored in `mat_files/weld_concert.mat`.

### Gap Pose Ground Truth

```bash
cd /home/user/concert_ws/src/acea_concert
python3 src/gap_pose_publisher.py
```

This publishes the current gap pose with respect to `base_link`:

```text
/gap/pose_robot
/gap/x_axis_robot
/gap/y_axis_robot
/gap/z_axis_robot
```

This node is ground truth for simulation. In the real setup, it should be replaced by perception.

### Gravity Compensation

```bash
cd /home/user/concert_ws/src/acea_concert
python3 src/gravity_comp_node.py
```

### Welding Controller

```bash
cd /home/user/concert_ws/src/acea_concert
python3 src/controller.py
```

The controller uses:

```text
optimized posture trajectory from the MAT file
desired weld position in the gap frame
desired weld orientation in the gap frame
measured gap pose from ground truth/perception
```

At runtime it relocates the optimized weld trajectory onto the current measured gap:

```text
position:    base_p_ee_des = base_p_gap + base_R_gap * gap_p_ee_des
orientation: base_R_ee_des = base_R_gap * gap_R_ee_des
```

## Controller Pipeline

The controller does not blindly replay the optimized joint trajectory. The joint trajectory is used as a nominal posture, while the welding target is rebuilt online from the measured gap pose.

```mermaid
flowchart TD
    subgraph offline["offline optimization"]
        mat["weld_concert.mat"]
        qdes["q_des(t)<br/>optimal posture"]
        weldgap["gap_p_ee_des(t)<br/>gap_R_ee_des(t)<br/>weld target in gap frame"]
        mat --> qdes
        mat --> weldgap
    end

    subgraph online["online measurements"]
        perception["ground truth now<br/>perception later"]
        gappose["base_p_gap<br/>base_R_gap<br/>current gap pose"]
        feedback["robot feedback<br/>q_meas<br/>base_p_ee_meas<br/>base_R_ee_meas"]
        perception --> gappose
    end

    subgraph controller["controller.py"]
        relocate["relocate weld target<br/>base_p_ee_des = base_p_gap + base_R_gap * gap_p_ee_des<br/>base_R_ee_des = base_R_gap * gap_R_ee_des"]
        project["project error in gap frame<br/>normal direction<br/>tangent direction"]
        pd["PD correction<br/>normal velocity<br/>tangent velocity"]
        ik["CartesIO IK<br/>EE pose task<br/>postural task"]
    end

    output["joint position command"]
    diag["diagnostics<br/>PlotJuggler"]

    qdes --> ik
    weldgap --> relocate
    gappose --> relocate
    gappose --> project
    feedback --> project
    relocate --> project
    project --> pd
    pd --> ik
    relocate --> ik
    ik --> output
    output --> feedback
    project --> diag
    pd --> diag
```

### Controller Inputs

From the MAT file:

```text
q_des(t)              optimized joint posture trajectory
gap_p_ee_des(t)       desired weld position expressed in the gap frame
gap_R_ee_des(t)       desired tool orientation expressed in the gap frame
```

From ground truth now, and from perception later:

```text
base_p_gap            gap origin expressed in base_link
base_R_gap            gap orientation expressed in base_link
```

From robot feedback:

```text
q_meas                measured robot joint state
base_p_ee_meas        measured EE position from forward kinematics
base_R_ee_meas        measured EE orientation from forward kinematics
```

### Control Loop

At each controller tick:

1. Read `q_des(t)` and send it to the CartesIO postural task.
2. Read the desired weld position and orientation in the gap frame.
3. Read the current measured gap pose in `base_link`.
4. Convert the desired weld target from gap frame to `base_link`.
5. Compare the measured EE position with the desired weld target along:
   - the gap normal direction, across the pipes
   - the gap tangent direction, along the weld
6. Compute PD correction velocities along those two directions.
7. Command the EE pose task and postural task through CartesIO.
8. Publish diagnostics for PlotJuggler.

The key idea is that if the robot or gap moves, the weld target moves with the measured gap:

```text
base_p_ee_des = base_p_gap + base_R_gap * gap_p_ee_des
base_R_ee_des = base_R_gap * gap_R_ee_des
```

## Main Files

- `src/weld_opt.py`: offline trajectory optimization and MAT file generation.
- `launch/weld_sim.launch.py`: simulation launch file.
- `src/gap_pose_publisher.py`: simulation ground-truth gap pose publisher.
- `src/gravity_comp_node.py`: gravity compensation node.
- `src/controller.py`: online welding controller.
- `src/controller_ros.py`: ROS interface for the controller.
- `src/utils/`: controller geometry, trajectory, and diagnostic helpers.

## Configuration

- `config/weld.yaml`: Horizon optimization task configuration.
- `config/cartesio_stack.yaml`: CartesIO stack used by the online controller.
- `src/modular/`: robot model generation.
- `mat_files/`: generated optimization results.

## Dependencies

- ROS 2 Jazzy
- Horizon
- CasADi
- XBot2
- CartesIO
- NumPy, SciPy, Matplotlib
