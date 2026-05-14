# 🚗 UniPD-DriveOps — BRAIN Workspace

> Autonomous driving software stack for the **Bosch Future Mobility Challenge (BFMC)** by Team UniPD-DriveOps, University of Padova.

---

## 📦 Repository Structure

```
luxonis_ws/
└── src/
    ├── camera_preprocessing/
    │   ├── camera_preprocessing/
    │   │   ├── luxonis_preprocessing.py   # Main lane following node
    │   │   ├── controller3.py             # Lateral/angular controller
    │   │   └── lane_keeper_small.onnx     # Trained CNN model
    │   ├── launch/
    │   │   └── sim.launch.py              # Simulation launch file
    │   ├── depthai-ros/
    │   ├── package.xml
    │   ├── setup.cfg
    │   └── setup.py
    └── traffic_sign/
        ├── traffic_sign/
        │   ├── traffic_sign_node.py       # Main traffic sign detection node
        │   └── best.onnx                  # Trained YOLOv8n model (9 classes)
        ├── config/
        │   └── params.yaml                # Per-class trigger distances and cooldowns
        ├── launch/
        │   └── traffic_sign.launch.py     # Launch file
        ├── test_signs.sh                  # Sequential sign spawner for Gazebo testing
        ├── package.xml
        ├── setup.cfg
        └── setup.py
```

---

## 🧠 Packages

### `camera_preprocessing`

CNN-based lane following pipeline using a Luxonis OAK-D camera (real or simulated).

**Pipeline:**
1. Subscribe to `/oak/rgb/image_raw` (`sensor_msgs/Image`, `rgb8`)
2. Preprocess: grayscale → crop → resize → Canny → blur → 32×32 blob
3. Run inference with `lane_keeper_small.onnx` → lateral error `e2`, angular error `e3`
4. Compute steering angle via `Controller` (pure pursuit + PD)
5. Publish speed and steering to `/automobile/command`

**Key topics:**

| Topic | Type | Direction |
|-------|------|-----------|
| `/oak/rgb/image_raw` | `sensor_msgs/Image` | Subscribed |
| `/automobile/command` | `std_msgs/String` | Published |

---

### `traffic_sign`

YOLOv8n-based traffic sign detection with spatial distance estimation using the OAK-D Pro stereo cameras.

**9 target classes:**

| ID | Class | Trigger dist | Action (BFMC rules) |
|----|-------|-------------|---------------------|
| 0 | `stop` | 0.60 m | Halt 3 seconds at intersection |
| 1 | `parking` | 0.80 m | Slow down, find parking spot |
| 2 | `priority` | 0.60 m | Enter intersection without stopping |
| 3 | `roundabout` | 0.70 m | Follow roundabout rules (CCW) |
| 4 | `one_way` | 0.70 m | Follow indicated direction |
| 5 | `no_entry` | 0.80 m | Avoid that road |
| 6 | `exit_highway` | 0.60 m | Switch to city rules (min 20 cm/s) |
| 7 | `entrance_highway` | 0.80 m | Switch to highway rules (min 40 cm/s) |
| 8 | `crosswalk` | 0.50 m | Slow down, yield to pedestrians |

**Pipeline:**
1. Get frame from OAK-D Pro RGB camera (or simulator topic / webcam in debug mode)
2. Run YOLOv8n ONNX inference via ONNX Runtime (CPU)
3. Compute distance from stereo disparity map (OAK mode only)
4. Select closest sign, check per-class trigger distance and cooldown
5. Publish trigger on `/traffic_sign/detection`
6. Publish annotated frame on `/traffic_sign/image`

**Three operating modes** (set in `config/params.yaml`):

| Mode | Config | Use case |
|------|--------|----------|
| SIM | `use_sim: true` | Gazebo simulator |
| MOCK | `use_mock_camera: true` | Webcam or video file |
| OAK | both false | OAK-D Pro real camera |

**Key topics:**

| Topic | Type | Direction |
|-------|------|-----------|
| `/oak/rgb/image_raw` | `sensor_msgs/Image` | Subscribed (SIM mode) |
| `/traffic_sign/detection` | `std_msgs/String` | Published |
| `/traffic_sign/image` | `sensor_msgs/Image` | Published |

**Detection message format:**
```json
{"sign": "stop", "distance_m": 3.42, "confidence": 0.91}
```
In SIM and MOCK mode `distance_m` is `null` (no real stereo depth available).

---

## ⚙️ Requirements

| Dependency | Version |
|-----------|---------|
| ROS 2 | Jazzy |
| Gazebo | Harmonic |
| Python | 3.10+ |
| OpenCV | `cv2` |
| ONNX Runtime | `onnxruntime` |
| depthai | `depthai` (OAK mode only) |
| cv_bridge | ROS 2 package |

```bash
pip install onnxruntime opencv-python numpy
pip install depthai  # only needed for OAK-D Pro mode
```

---

## 🚀 Installation

```bash
git clone https://github.com/UniPD-DriveOps-BFMC/BRAIN_driveops.git
cd BRAIN_driveops/luxonis_ws

source /opt/ros/jazzy/setup.bash
colcon build --packages-select camera_preprocessing traffic_sign
source install/setup.bash
```

---

## 🖥️ Usage

### Lane Following (run with Gazebo Simulator)

```bash
# Source both workspaces
cd BRAIN_driveops/luxonis_ws
source ~/BFMC/bfmc_ws/install/setup.bash
source install/setup.bash

# Launch simulation + lane following
ros2 launch camera_preprocessing sim.launch.py
```

**Launch arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `desired_speed` | `0.3` | Target speed [m/s] |
| `debug_view` | `true` | Show preprocessed frames with OpenCV |

Example with custom parameters:
```bash
ros2 launch camera_preprocessing sim.launch.py desired_speed:=0.1 debug_view:=true
```

### Lane Following on Real Hardware (OAK-D Camera)

First, connect the Luxonis OAK-D camera and launch the camera driver in a **separate terminal**:

```bash
# Terminal 1 — start the Luxonis camera driver
ros2 launch depthai_ros_driver camera.launch.py
```

Then, in a **second terminal**, launch the lane inference node:

```bash
# Terminal 2 — lane following node
ros2 run camera_preprocessing lane_inference_node \
  --ros-args \
  -p input_topic:=/oak/rgb/image_raw \
  -p command_topic:=/automobile/command \
  -p desired_speed:=0.2
```

---

## 🔧 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | bundled `.onnx` | Path to ONNX model |
| `input_topic` | string | `/oak/rgb/image_raw` | Camera topic |
| `command_topic` | string | `/automobile/command` | Command output topic |
| `desired_speed` | float | `0.3` | Target speed [m/s] |
| `debug_view` | bool | `true` | Enable OpenCV debug windows |
| `faster` | bool | `false` | Single inference (no flip augmentation) |

---

## 🧪 Diagnostics

Check topic connectivity:
```bash
ros2 topic info /oak/rgb/image_raw -v
ros2 topic hz /oak/rgb/image_raw
```

Visualize camera feed:
```bash
ros2 run rqt_image_view rqt_image_view
```

View node graph:
```bash
ros2 run rqt_graph rqt_graph
```

---

### Traffic Sign Detection

**Simulator mode** (default):

```bash
# Terminal 1 — start simulator with all objects
cd ~/BFMC/bfmc_ws/SimulatorROS2
source /opt/ros/jazzy/setup.bash
source ../install/setup.bash
just all-objects

# Terminal 2 — traffic sign node
source /opt/ros/jazzy/setup.bash
source ~/BFMC/luxonis_ws/install/setup.bash
ros2 launch traffic_sign traffic_sign.launch.py

# Terminal 3 — monitor detections
ros2 topic echo /traffic_sign/detection
```

**OAK-D Pro mode** — set in `config/params.yaml`:
```yaml
use_sim:         false
use_mock_camera: false
```
Then:
```bash
ros2 launch traffic_sign traffic_sign.launch.py
```

**Webcam / video debug mode:**
```yaml
use_sim:         false
use_mock_camera: true
mock_source:     "0"   # 0 = webcam, or "/path/to/video.mp4"
```

**Test all signs sequentially in Gazebo:**
```bash
bash src/traffic_sign/test_signs.sh
```

---

## 🔧 Parameters — `traffic_sign`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_sim` | `true` | Subscribe to ROS2 image topic (simulator) |
| `use_mock_camera` | `false` | Use webcam or video file |
| `mock_source` | `"0"` | Webcam index or video file path |
| `sim_image_topic` | `/oak/rgb/image_raw` | Image topic in SIM mode |
| `conf_threshold` | `0.50` | YOLO confidence threshold |
| `iou_threshold` | `0.40` | NMS IoU threshold |
| `depth_min_mm` | `100` | Minimum valid depth (mm) |
| `depth_max_mm` | `2000` | Maximum valid depth (mm) |
| `trigger_dist_<class>` | per-class | Distance (m) to trigger event |
| `cooldown_<class>` | per-class | Seconds between same-class triggers |
| `debug_view` | `true` | Show OpenCV annotated window |

---

## 🧪 Diagnostics

```bash
# Check active topics
ros2 topic list | grep -E "traffic_sign|oak"

# Detection rate
ros2 topic hz /traffic_sign/detection

# Visualize annotated feed
ros2 run rqt_image_view rqt_image_view /traffic_sign/image

# Node graph
ros2 run rqt_graph rqt_graph
```

---

## 👥 Team

**UniPD-DriveOps** — University of Padova
Bosch Future Mobility Challenge 2026

---

## 📄 License

This project is developed for academic and competition purposes.
© 2026 UniPD-DriveOps — University of Padova
