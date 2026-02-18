# 🏌️ Golf Ball Tracking System

## Overview

A robust **golf ball tracking system** using:
- **YOLOv8** for ball detection
- **Kalman Filter** for smooth trajectory prediction
- **Dynamic ROI** for efficient processing
- **Validation Layer** for false positive reduction

## 🎯 Features

### ✅ Core Capabilities
- **Initial Detection Phase**: Bootstrap tracking by detecting ball in full frame
- **Tracking Phase**: Kalman filter-based prediction and tracking
- **Dynamic ROI Management**: Adaptive region of interest for efficient processing
- **Validation Layer**: Multi-criteria validation to reject false positives
- **Smooth Trajectories**: Temporal consistency with Kalman filtering
- **Edge Case Handling**: Handles motion blur, temporary disappearance, and occlusions

### 🧠 Technical Highlights
- **State Vector**: `[x, y, vx, vy]` - position + velocity
- **Measurement Vector**: `[x, y]` - direct position measurements
- **Adaptive ROI**: Expands on miss, shrinks on detection success
- **Validation Criteria**:
  - Distance from predicted position
  - Bounding box size consistency
  - Confidence threshold
  - Aspect ratio (ball roundness)

## 📁 Project Structure

```
ball_tracking_python/
├── detector.py          # YOLOv8 detection wrapper
├── tracker.py           # Kalman filter tracker
├── roi_manager.py       # Dynamic ROI management
├── main.py              # Main orchestration pipeline
├── best.pt              # YOLOv8 model weights
├── input_3.mp4          # Input video
└── README.md            # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy

### Install Dependencies
```bash
pip install opencv-python ultralytics numpy
```

## 💻 Usage

### Basic Usage
```bash
python main.py
```

### Configuration
Edit the `Config` class in [main.py](main.py) to adjust parameters:

```python
class Config:
    # Model and Video Paths
    MODEL_PATH = "best.pt"
    VIDEO_INPUT = "input_3.mp4"
    VIDEO_OUTPUT = "output_kalman.mp4"
    
    # Detection Parameters
    CONFIDENCE_THRESHOLD = 0.35
    
    # Tracking Parameters
    MAX_MISSED_FRAMES = 15
    PROCESS_NOISE = 0.03
    MEASUREMENT_NOISE = 5.0
    
    # ROI Parameters
    INITIAL_ROI_SIZE = 200
    MIN_ROI_SIZE = 100
    MAX_ROI_SIZE = 400
    ROI_EXPANSION_FACTOR = 1.2
    
    # Validation Parameters
    MAX_DISTANCE = 150.0
    MAX_SIZE_CHANGE_RATIO = 2.5
```

## 🔧 Module Documentation

### 1. `detector.py` - Ball Detector

**Purpose**: Wraps YOLOv8 for golf ball detection with validation.

**Key Methods**:
- `detect_full_frame(frame)`: Run detection on full frame
- `detect_in_roi(frame, roi)`: Run detection only in ROI region
- `validate_detection(...)`: Multi-criteria detection validation

**Example**:
```python
from detector import BallDetector

detector = BallDetector("best.pt", confidence_threshold=0.35)
detection = detector.detect_full_frame(frame)
if detection:
    cx, cy, conf, bbox = detection
```

### 2. `tracker.py` - Kalman Filter Tracker

**Purpose**: Implements Kalman filter for smooth ball trajectory prediction.

**State Vector**: `[x, y, vx, vy]`
- `x, y`: Ball center position
- `vx, vy`: Velocity components

**Key Methods**:
- `predict()`: Predict next position
- `update(measurement, bbox)`: Update with new measurement
- `update_with_prediction()`: Update with prediction on miss
- `get_trajectory()`: Get full trajectory history

**Example**:
```python
from tracker import KalmanBallTracker

tracker = KalmanBallTracker(
    initial_position=(cx, cy),
    process_noise=0.03,
    measurement_noise=5.0,
    max_missed_frames=15
)

# Each frame:
pred_x, pred_y = tracker.predict()
if detection_found:
    tracker.update((cx, cy), bbox)
else:
    tracker.update_with_prediction()
```

### 3. `roi_manager.py` - ROI Manager

**Purpose**: Manages dynamic Region of Interest for efficient detection.

**Key Methods**:
- `get_roi(center)`: Get ROI around center position
- `expand_roi()`: Increase ROI size (on detection failure)
- `shrink_roi()`: Decrease ROI size (on detection success)
- `get_adaptive_roi(center, velocity)`: Velocity-aware ROI

**Example**:
```python
from roi_manager import ROIManager

roi_manager = ROIManager(
    frame_width=1920,
    frame_height=1080,
    initial_roi_size=200
)

roi = roi_manager.get_roi((pred_x, pred_y))
detection = detector.detect_in_roi(frame, roi)
```

### 4. `main.py` - Main Pipeline

**Purpose**: Orchestrates the complete tracking system.

**Pipeline Stages**:
1. **Detection Phase**: Bootstrap tracking with full-frame detection
2. **Tracking Phase**: Kalman prediction → ROI detection → Validation → Update
3. **Visualization**: Draw trajectories, ROI, predictions, info panel

## 🎨 Visualization

The output video includes:
- **Green trajectory**: Smoothed ball path
- **Cyan ROI rectangle**: Current search region
- **Red prediction point**: Kalman filter prediction
- **Info panel**: Real-time statistics

## ⚙️ Parameter Tuning Guide

### Detection Sensitivity
- **CONFIDENCE_THRESHOLD**: Lower = more detections (more false positives)
- **Recommended**: 0.3 - 0.4

### Tracking Smoothness
- **PROCESS_NOISE**: Higher = trust model less, adapt faster
- **MEASUREMENT_NOISE**: Higher = trust measurements less, smoother
- **Recommended**: Keep default unless trajectory too jittery

### ROI Behavior
- **INITIAL_ROI_SIZE**: Larger = more detections, slower
- **ROI_EXPANSION_FACTOR**: Higher = faster expansion on miss
- **Recommended**: Adjust based on ball speed

### Validation Strictness
- **MAX_DISTANCE**: Higher = accept detections farther from prediction
- **MAX_SIZE_CHANGE_RATIO**: Higher = accept larger size changes
- **Recommended**: Increase if ball is lost frequently

## 🐛 Troubleshooting

### Ball Not Detected Initially
- Lower `CONFIDENCE_THRESHOLD`
- Check if model is trained for golf balls
- Verify video quality

### Tracking Lost Frequently
- Increase `MAX_DISTANCE` validation threshold
- Increase `MAX_MISSED_FRAMES`
- Adjust `MEASUREMENT_NOISE` for smoother tracking

### Trajectory Too Jittery
- Increase `MEASUREMENT_NOISE` (trust detections less)
- Decrease `PROCESS_NOISE` (trust model more)

### False Positives
- Increase `CONFIDENCE_THRESHOLD`
- Decrease `MAX_DISTANCE`
- Decrease `MAX_SIZE_CHANGE_RATIO`

## 📊 Performance

- **Typical FPS**: 20-30 FPS (depends on hardware and ROI size)
- **Optimization**: ROI-based detection reduces inference time by 4-10x
- **Memory**: Low memory footprint, suitable for real-time processing

## 🔬 Kalman Filter Details

### Constant Velocity Model

**State Transition**:
```
x_next = x + vx * dt
y_next = y + vy * dt
vx_next = vx
vy_next = vy
```

**Why This Works**:
- Golf balls have relatively smooth motion
- Velocity changes gradually (gravity/air resistance)
- Simple model = fast computation
- Handles temporary occlusions well

### Tuning Tips
- **Fast-moving balls**: Decrease `MEASUREMENT_NOISE`
- **Erratic motion**: Increase `PROCESS_NOISE`
- **Smooth trajectories**: Increase `MEASUREMENT_NOISE`

## 📈 Future Enhancements

- [ ] Multi-ball tracking (assign track IDs)
- [ ] 3D trajectory reconstruction
- [ ] Ball spin detection
- [ ] Bounce detection and physics modeling
- [ ] Real-time visualization with OpenCV windows
- [ ] Export trajectory data to CSV/JSON

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Credits

- **YOLOv8**: Ultralytics
- **Kalman Filter**: OpenCV implementation
- **Architecture**: Modular design for maintainability

---

**Made with ❤️ for Computer Vision**
