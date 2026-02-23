# Golf Ball Tracking System — Technical Reference

A computer-vision pipeline that detects, tracks, and predicts the flight path of a golf ball in video footage. It combines a fine-tuned YOLOv8 object detector with a Kalman filter tracker, an adaptive Region-of-Interest (ROI) manager, and a direction-based trajectory extension module.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Processing Pipeline](#2-processing-pipeline)
3. [Module Reference](#3-module-reference)
   - [config.py — Centralized Configuration](#configpy--centralized-configuration)
   - [detector.py — BallDetector](#detectorpy--balldetector)
   - [tracker.py — KalmanBallTracker](#trackerpy--kalmanballtracker)
   - [roi_manager.py — ROIManager](#roi_managerpy--roimanager)
   - [trajectory_predictor.py — TrajectoryPredictor](#trajectory_predictorpy--trajectorypredictor)
   - [trajectory_utils.py — TrajectoryAnalyzer](#trajectory_utilspy--trajectoryanalyzer)
   - [main.py — Orchestration Pipeline](#mainpy--orchestration-pipeline)
4. [Algorithms In Depth](#4-algorithms-in-depth)
   - [Kalman Filter — Constant Velocity Model](#kalman-filter--constant-velocity-model)
   - [Validation Layer](#validation-layer)
   - [Trajectory Extension](#trajectory-extension)
5. [Configuration System](#5-configuration-system)
6. [Project Structure](#6-project-structure)
7. [Installation & Usage](#7-installation--usage)
8. [Parameter Tuning Guide](#8-parameter-tuning-guide)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     main.py (Pipeline)                  │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │ BallDetector │──▶│KalmanTracker │──▶│ ROIManager │  │
│  │  (YOLOv8)   │   │ (cv2.Kalman) │   │ (Adaptive) │  │
│  └──────────────┘   └──────────────┘   └────────────┘  │
│         │                  │                  │         │
│         └──────────────────▼──────────────────┘         │
│                    Validation Layer                      │
│                           │                             │
│                           ▼                             │
│              TrajectoryPredictor (Extension)            │
│                           │                             │
│                           ▼                             │
│                 Annotated Output Video                  │
└─────────────────────────────────────────────────────────┘
```

The system is split into two high-level **tracking states**:

| State | Description |
|---|---|
| `DETECTION` | Bootstrap phase — full-frame YOLO scan every frame until the ball is found for the first time. |
| `TRACKING` | Active tracking — Kalman prediction drives an adaptive ROI; full-frame acts as a live fallback when the ROI misses. |

---

## 2. Processing Pipeline

For every video frame the following sequence executes:

```
Read Frame
    │
    ▼
[DETECTION state]
    └─ detect_full_frame()
           │  Found? → Initialize KalmanTracker → switch to TRACKING
           │  Not found? → stay in DETECTION
    ▼
[TRACKING state]
    ├─ 1. tracker.predict()               ← Kalman prediction (x, y, vx, vy)
    ├─ 2. roi_manager.get_adaptive_roi()  ← velocity-shifted ROI
    ├─ 3. detect_in_roi()  (upscaled 2×)  ← fast ROI detection
    ├─ 4. detect_full_frame()             ← parallel full-frame safety net
    │
    ├─ ROI hit & valid?
    │       └─ tracker.update() → shrink ROI → record detection
    ├─ ROI hit but invalid? or ROI miss?
    │       ├─ Full-frame hit? → reset tracker at new position → record (gap bridged)
    │       └─ Both miss?     → tracker.update_with_prediction() → expand ROI
    │
    ├─ tracker.is_active == False? → back to DETECTION
    │
    ├─ draw_trajectory()              ← gap-bridging polyline over all_detections[]
    ├─ TrajectoryPredictor.predict() + draw_predicted()
    └─ Write annotated frame to output video
```

Both ROI and full-frame detection run **every tracking frame** (not with a delay). This guarantees immediate re-acquisition after a fast swing without waiting for several missed frames.

---

## 3. Module Reference

### `config.py` — Centralized Configuration

All tunable parameters live in one place so no module needs to be edited directly.

**Classes**

| Class | Purpose |
|---|---|
| `Config` | Base configuration (default values) |
| `FastModeConfig` | Smaller ROI, higher confidence threshold — optimized for speed |
| `AccurateModeConfig` | Larger ROI, lower confidence threshold — optimized for quality |
| `RobustModeConfig` | Higher miss tolerance, more aggressive ROI expansion — for occluded scenes |

**Helper functions**

| Function | Description |
|---|---|
| `get_config(mode)` | Returns a `Config` subclass instance by mode name (`"default"`, `"fast"`, `"accurate"`, `"robust"`) |
| `print_config(config)` | Pretty-prints all active parameter values |

**Key parameter groups**

| Group | Parameters |
|---|---|
| File paths | `MODEL_PATH`, `VIDEO_INPUT`, `VIDEO_OUTPUT` |
| Detection | `CONFIDENCE_THRESHOLD` |
| Kalman filter | `PROCESS_NOISE`, `MEASUREMENT_NOISE`, `MAX_MISSED_FRAMES` |
| ROI | `INITIAL_ROI_SIZE`, `MIN_ROI_SIZE`, `MAX_ROI_SIZE`, `ROI_EXPANSION_FACTOR` |
| Validation | `MAX_DISTANCE`, `MAX_SIZE_CHANGE_RATIO`, `MIN_BBOX_AREA`, `MIN/MAX_ASPECT_RATIO`, `MIN_DETECTION_DISTANCE` |
| Prediction | `PREDICTION_ENABLED`, `PREDICTION_MIN_CONSISTENT_SEGMENTS`, `PREDICTION_DIRECTION_TOLERANCE_DEG`, `PREDICTION_LOOKBACK_POINTS`, `PREDICTION_MAX_LENGTH_PX`, `PREDICTION_POINTS_PER_FRAME` |
| Visualization | `SHOW_ROI`, `SHOW_PREDICTION`, `TRAJECTORY_COLOR`, `TRAJECTORY_THICKNESS`, `ROI_COLOR`, `PRED_COLOR` |

---

### `detector.py` — BallDetector

Wraps the `ultralytics.YOLO` model with golf-ball-specific logic.

**Class: `BallDetector(model_path, confidence_threshold)`**

| Method | Signature | Description |
|---|---|---|
| `detect_full_frame` | `(frame) → Optional[tuple]` | Runs YOLO on the full resolution frame; returns `(cx, cy, conf, bbox)` of the highest-confidence detection, or `None`. |
| `detect_in_roi` | `(frame, roi, upscale_factor=2.0) → Optional[tuple]` | Crops the frame to `roi`, **upscales the crop by `upscale_factor`** before YOLO inference (improves detection of tiny in-flight balls), then maps the result back to global frame coordinates. Returns `(cx, cy, conf, bbox)` in global coordinates. |
| `validate_detection` | `(detection, predicted_pos, last_bbox, ...) → bool` | Runs 4 independent filters: confidence, distance from Kalman prediction, bounding-box area ratio, and aspect ratio. |

**ROI upscaling detail**

A golf ball during flight may be only a few pixels wide. `detect_in_roi` enlarges the crop before inference so the model sees a reasonably sized object, then divides all coordinates by `upscale_factor` to restore global positions:

```
Global bbox = (bbox_in_upscaled_crop / upscale_factor) + roi_origin
```

---

### `tracker.py` — KalmanBallTracker

Implements a **4-state, 2-measurement** Kalman filter using `cv2.KalmanFilter`.

**State vector**: `[x, y, vx, vy]`  
**Measurement vector**: `[x, y]`

**Class: `KalmanBallTracker(initial_position, process_noise, measurement_noise, max_missed_frames)`**

| Method | Description |
|---|---|
| `predict()` | Calls `kalman.predict()` and returns `(pred_x, pred_y)`. |
| `update(measurement, bbox)` | Calls `kalman.correct(meas)`, appends the corrected position to `trajectory`, resets `missed_frames`. |
| `update_with_prediction()` | No measurement available — appends the predicted state to `trajectory`, increments `missed_frames`, sets `is_active = False` if limit exceeded. |
| `get_trajectory()` | Returns a copy of the accumulated `(x, y)` list. |
| `get_velocity()` | Reads `vx, vy` directly from `kalman.statePost`. |
| `get_speed()` | Returns `sqrt(vx^2 + vy^2)`. |

**Class: `TrackingState`** — string constants `DETECTION` and `TRACKING` used as state labels throughout the pipeline.

---

### `roi_manager.py` — ROIManager

Manages a square search window that **grows on miss** and **shrinks on success**, keeping processing cost low while adapting to ball speed and direction.

**Class: `ROIManager(frame_width, frame_height, initial_roi_size, min_roi_size, max_roi_size, expansion_factor)`**

| Method | Description |
|---|---|
| `get_roi(center)` | Returns `(x1, y1, x2, y2)` of a square of `current_roi_size` centered at `center`, clamped to frame boundaries. |
| `expand_roi()` | `current_roi_size = min(size × expansion_factor, max_roi_size)` |
| `shrink_roi()` | `current_roi_size = max(size / expansion_factor, min_roi_size)` |
| `reset_roi()` | Resets to `initial_roi_size`. |
| `get_adaptive_roi(center, velocity, speed_factor=2.0)` | Shifts the center by `velocity × speed_factor` before calling `get_roi`, so the ROI leads the ball in its direction of motion. |
| `should_use_full_frame()` | Returns `True` when the ROI covers more than 80% of the frame area (full-frame is then not more expensive). |

---

### `trajectory_predictor.py` — TrajectoryPredictor

Extends the confirmed trajectory **forward** in a straight line when the ball is in consistent directional motion (e.g. after a clean shot).

**Class: `TrajectoryPredictor(frame_width, frame_height, min_consistent_segments, direction_tolerance_deg, lookback_points, max_prediction_length_px, prediction_step_px)`**

| Method | Description |
|---|---|
| `predict(trajectory)` | Returns a list of predicted `(x, y)` points; empty if direction is not sufficiently consistent. |
| `draw_predicted(frame, predicted_points, last_real_point, color, thickness)` | Draws a polyline from `last_real_point` through all predicted points. |

The prediction is **animated** in `main.py`: `PREDICTION_POINTS_PER_FRAME` controls how many new predicted points are revealed per frame, creating a smooth draw-forward effect. When the trajectory gains new real detections, the animation resets.

---

### `trajectory_utils.py` — TrajectoryAnalyzer

Offline analysis and optional data export. Not part of the live pipeline but can be used after processing.

**Class: `TrajectoryAnalyzer(trajectory)`**

| Method | Returns |
|---|---|
| `get_total_distance()` | Total arc length in pixels |
| `get_displacement()` | `(dx, dy, magnitude)` — straight-line distance from first to last point |
| `get_velocity_profile()` | Per-frame speed list `[px/frame]` |
| `get_acceleration_profile()` | Per-frame acceleration list `[px/frame^2]` |
| `get_average_speed()` | Mean of velocity profile |
| `get_max_speed()` | Peak speed in `px/frame` |
| `get_bounding_box()` | `(min_x, min_y, max_x, max_y)` of all trajectory points |
| `get_statistics()` | Dictionary combining all of the above |

---

### `main.py` — Orchestration Pipeline

The entry point that wires all modules together.

**Key internal structures**

| Variable | Type | Purpose |
|---|---|---|
| `all_detections` | `List[(x, y)]` | Every confirmed real detection (never Kalman-only points). Used for gap-bridging line drawing and trajectory extension. |
| `first_detection_bbox` | `Optional[bbox]` | Stores the very first detected bounding box so it can be drawn persistently as the "START" marker. |
| `fallback_reacquisitions` | `int` | Counter for how many times the full-frame fallback successfully re-acquired the ball after ROI failure. |
| `prediction_reveal_count` | `int` | Animation state — how many predicted points have been revealed so far. |

**`_add_detection(detections, point) → bool`**  
Appends a new confirmed point only if it is at least `MIN_DETECTION_DISTANCE` pixels from the last recorded one. Prevents cluttering the trajectory with duplicate stationary points.

**`draw_trajectory(frame, all_detections)`**  
Draws a polyline connecting every entry in `all_detections`. Because the list contains only real detections (not Kalman fill-ins), the line **bridges gaps** automatically — when the ball was lost for several frames but then re-acquired via full-frame fallback, the line jumps straight from the last known point to the new one, producing a clean connected path.

**`draw_info_panel(...)`**  
Appends a dark HUD strip below the frame showing: frame index, FPS, tracking state, ROI size, ball position, speed, missed-frame count, trajectory length, and total confirmed detections.

---

## 4. Algorithms In Depth

### Kalman Filter — Constant Velocity Model

The filter models the ball as a point moving at approximately constant velocity.

**State vector**

```
x_state = [x, y, vx, vy]
```

**State transition matrix** (dt = 1 frame)

```
F = | 1  0  1  0 |
    | 0  1  0  1 |
    | 0  0  1  0 |
    | 0  0  0  1 |
```

Encodes: `x_next = x + vx`, `y_next = y + vy`, velocities held constant.

**Measurement matrix** (position only)

```
H = | 1  0  0  0 |
    | 0  1  0  0 |
```

**Predict step**

```
x_predicted = F * x_prev
P_predicted = F * P_prev * F^T + Q
```

**Update step** (when a real detection is available)

```
K = P_predicted * H^T * (H * P_predicted * H^T + R)^-1   ← Kalman gain
x_corrected = x_predicted + K * (z - H * x_predicted)    ← fuse measurement
P_corrected = (I - K * H) * P_predicted
```

Where:
- `Q = PROCESS_NOISE × I₄` — models uncertainty in the constant velocity assumption  
- `R = MEASUREMENT_NOISE × I₂` — models detector noise  
- `P₀ = 1000 × I₄` — large initial uncertainty (position unknown)

When no detection is available, only the predict step runs and the predicted state is recorded in the trajectory via `update_with_prediction`.

---

### Validation Layer

`BallDetector.validate_detection()` applies four sequential gate tests. A candidate is **rejected** if it fails any one:

| Test | Condition | Purpose |
|---|---|---|
| Confidence | `conf >= CONFIDENCE_THRESHOLD` | Removes low-quality detections |
| Distance | `dist(detection, prediction) <= MAX_DISTANCE` | Rejects detections far from expected position |
| Size ratio | `max(curr_area/last_area, last_area/curr_area) <= MAX_SIZE_CHANGE_RATIO` | Prevents size jumps (e.g. shadow mistaken for ball) |
| Aspect ratio | `MIN_ASPECT_RATIO <= w/h <= MAX_ASPECT_RATIO` | Ball must be roughly circular |

`_add_detection` in `main.py` then acts as a **spatial deduplication** gate: points closer than `MIN_DETECTION_DISTANCE` pixels to the previously recorded detection are discarded, preventing duplicate stationary points.

---

### Trajectory Extension

After tracking the ball's real path, the predictor extends it forward geometrically:

1. **Tail selection** — use the last `PREDICTION_LOOKBACK_POINTS` entries from `all_detections`.
2. **Segment computation** — for each pair of consecutive tail points, compute `(dx, dy, length, angle_radians)`.
3. **Consistent run search** — scan for the longest consecutive run where `|angle_i − angle_{i-1}| <= tolerance_rad`. Using the longest run gives the most stable direction estimate.
4. **Direction averaging** — average `dx` and `dy` across the consistent run to get a mean direction vector.
5. **Prediction walk** — starting from the last real point, step forward by `step_size = avg_segment_length` in the unit direction, accumulating until `PREDICTION_MAX_LENGTH_PX` or the frame boundary.
6. **Animation** — `main.py` reveals `PREDICTION_POINTS_PER_FRAME` new predicted points each frame, producing a draw-forward animation that resets every time new real detections arrive.

---

## 5. Configuration System

### Default `Config` values

```python
# Detection
CONFIDENCE_THRESHOLD = 0.1

# Kalman
MAX_MISSED_FRAMES     = 1000
PROCESS_NOISE         = 0.03
MEASUREMENT_NOISE     = 5.0

# ROI
INITIAL_ROI_SIZE      = 400
MIN_ROI_SIZE          = 400
MAX_ROI_SIZE          = 600
ROI_EXPANSION_FACTOR  = 1.2

# Validation
MAX_DISTANCE          = 1_000_000.0   # effectively disabled
MAX_SIZE_CHANGE_RATIO = 2.5
MIN_BBOX_AREA         = 9
MIN_ASPECT_RATIO      = 0.3
MAX_ASPECT_RATIO      = 3.0
MIN_DETECTION_DISTANCE = 10.0

# Prediction
PREDICTION_ENABLED                  = True
PREDICTION_MIN_CONSISTENT_SEGMENTS = 2
PREDICTION_DIRECTION_TOLERANCE_DEG = 30.0
PREDICTION_LOOKBACK_POINTS         = 5
PREDICTION_MAX_LENGTH_PX           = 300.0
PREDICTION_POINTS_PER_FRAME        = 1
```

### Configuration Presets

| Preset class | Changes from default | When to use |
|---|---|---|
| `FastModeConfig` | Smaller ROI (150–300 px), higher confidence (0.4), lower measurement noise (3.0) | Real-time or low-end hardware |
| `AccurateModeConfig` | Larger ROI (250–500 px), lower confidence (0.3), higher measurement noise (7.0), max distance 200 | Offline high-quality processing |
| `RobustModeConfig` | More missed-frame tolerance (25), larger max distance (200), faster ROI expansion (1.3), higher noise values | Heavily occluded or erratic footage |

Select a preset programmatically:

```python
from config import get_config

config = get_config("robust")   # "default" | "fast" | "accurate" | "robust"
```

---

## 6. Project Structure

```
ball_tracking_python/
├── config.py                  # All tunable parameters + preset classes
├── detector.py                # YOLOv8 wrapper with ROI upscaling & validation
├── tracker.py                 # Kalman filter tracker (constant velocity)
├── roi_manager.py             # Adaptive search-window manager
├── trajectory_predictor.py    # Direction-based forward trajectory extension
├── trajectory_utils.py        # Offline trajectory analysis & export helpers
├── main.py                    # Pipeline orchestration & visualization
├── run.py                     # Thin launcher script
├── examples.py                # Usage examples
├── golfballyolov8n.pt         # Fine-tuned YOLOv8-nano weights
├── requirements.txt           # Python dependencies
└── inputs/                    # Input video files
```

---

## 7. Installation & Usage

### Prerequisites

- Python 3.8+
- OpenCV (`opencv-python`)
- Ultralytics YOLOv8 (`ultralytics`)
- NumPy

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Configure by editing `config.py` before running, or select a preset by importing the desired class:

```python
# in main.py, change the Config import line to:
from config import AccurateModeConfig as Config
```

### Output

The annotated video is written to `Config.VIDEO_OUTPUT` (default `output.mp4`). Each frame includes:

- **Green polyline** — full detected trajectory, bridging all gaps
- **Green predicted line** — forward extension drawn incrementally
- **Cyan ROI rectangle** — current adaptive search window (when `SHOW_ROI = True`)
- **Green bounding box + "DETECTED" label** — first detection anchor point
- **HUD strip** — frame number, FPS, tracking state, ball position, speed, missed-frame counter, trajectory length

---

## 8. Parameter Tuning Guide

| Problem | Fix |
|---|---|
| Ball not detected initially | Lower `CONFIDENCE_THRESHOLD` (try `0.05–0.2`); increase `INITIAL_ROI_SIZE` |
| Tracking lost mid-flight | Increase `MAX_MISSED_FRAMES`; increase `MAX_DISTANCE`; use `RobustModeConfig` |
| Trajectory too jittery | Increase `MEASUREMENT_NOISE`; decrease `PROCESS_NOISE` |
| Too many false detections | Increase `CONFIDENCE_THRESHOLD`; decrease `MAX_DISTANCE`; tighten `MAX_SIZE_CHANGE_RATIO` |
| Prediction line does not appear | Lower `PREDICTION_MIN_CONSISTENT_SEGMENTS` (try `2`); raise `PREDICTION_DIRECTION_TOLERANCE_DEG`; raise `PREDICTION_MAX_LENGTH_PX` |
| Processing too slow | Use `FastModeConfig`; reduce `MAX_ROI_SIZE` |

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Error: Cannot open video` | Wrong path in `VIDEO_INPUT` | Update `Config.VIDEO_INPUT` |
| Tracker resets every frame | `MAX_MISSED_FRAMES` too low | Increase `MAX_MISSED_FRAMES` |
| Prediction appears but is very short | `PREDICTION_MAX_LENGTH_PX` too small | Increase `PREDICTION_MAX_LENGTH_PX` |
| ROI always at frame edge | Ball near border — expected | ROI is intentionally clamped to frame boundaries |
| High `fallback_reacquisitions` count | ROI consistently misses fast ball | Increase `INITIAL_ROI_SIZE` / `MAX_ROI_SIZE` |

---

## Credits

- **YOLOv8** — [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Kalman Filter** — OpenCV `cv2.KalmanFilter`