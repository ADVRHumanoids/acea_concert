
# acea_concert

### Trajectory Generation & Optimization

- `src/weld_opt.py`: Sets up and solves the trajectory optimization problem using Horizon. Loads robot models, applies constraints/costs from `config/weld.yaml`, check for collisions (using XBot2) and saves results to `.mat` files.
- `src/replay.py`: Starts from a saved solution from a `.mat` file and resample it to a desired value. Also augments the trajectory with a wiggling in the y-direction.

### Replay & Visualization

- `src/viz/`: Contains utilities for RViz marker visualization and scene setup.

### Modular Robot Description

- `concert_with_torch.py`: Current script to generate URDF/SRDF robot descriptions from modular JSON files.

### Data Analysis

- `src/plot_weld_result.py`: Plots joint positions, velocities, and other results from `.mat` files.

## Usage

### 1. Build the Docker Container

```bash
cd docker
docker compose build
docker compose up
```

### 2. Run Trajectory Optimization

```bash
python3 src/weld_opt.py
```

### 3. Resample and Add Wiggle

```bash
python3 src/replayer.py
```

### 4. Visualize Results

Open RViz with the provided config:

```bash
rviz2 -d rviz/rviz_config.rviz
```

### 5. Plot Some Results

```bash
python3 src/plot_weld_result.py
```

## Configuration

- **Optimization settings**: `config/weld.yaml`
- **Robot model generation**: `src/modular/`
- **Example data**: `mat_files/`

## Dependencies

- ROS 2 (Jazzy)
- Horizon (trajectory optimization framework)
- CasADi
- XBot2 (for collision checking)
- NumPy, SciPy, Matplotlib