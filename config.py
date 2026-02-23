"""
Configuration file for Golf Ball Tracking System
Centralized parameter management for easy tuning.
"""


class Config:
    """
    Centralized configuration for the golf ball tracking system.
    Modify these parameters to tune tracking behavior.
    """
    
    # ═══════════════════════════════════════════════════════════
    # FILE PATHS
    # ═══════════════════════════════════════════════════════════
    
    MODEL_PATH = "golfballyolov8n.pt"              # YOLOv8 model weights path
    VIDEO_INPUT = "indoor_4.mp4"         # Input video path
    VIDEO_OUTPUT = "output.mp4"  # Output video path
    
    # ═══════════════════════════════════════════════════════════
    # DETECTION PARAMETERS
    # ═══════════════════════════════════════════════════════════
    
    # Minimum confidence score for valid detection (0.0 - 1.0)
    # Lower = more detections but more false positives
    # Higher = fewer detections but more reliable
    # Recommended: 0.3 - 0.4
    CONFIDENCE_THRESHOLD = 0.1
    
    # ═══════════════════════════════════════════════════════════
    # TRACKING PARAMETERS (Kalman Filter)
    # ═══════════════════════════════════════════════════════════
    
    # Maximum consecutive frames without detection before tracking reset
    # Higher = more tolerant to temporary occlusions
    # Lower = faster reset but may lose track prematurely
    # Recommended: 10 - 20
    MAX_MISSED_FRAMES = 1000
    
    # Kalman filter process noise covariance
    # Higher = trust the motion model less, adapt faster to new measurements
    # Lower = trust the motion model more, smoother but less responsive
    # Recommended: 0.01 - 0.1
    PROCESS_NOISE = 0.03
    
    # Kalman filter measurement noise covariance
    # Higher = trust measurements less, produce smoother trajectories
    # Lower = trust measurements more, follow detections closely
    # Recommended: 1.0 - 10.0
    MEASUREMENT_NOISE = 5.0
    
    # ═══════════════════════════════════════════════════════════
    # ROI (Region of Interest) PARAMETERS
    # ═══════════════════════════════════════════════════════════
    
    # Initial ROI size in pixels (square region)
    # Larger = more detections but slower processing
    # Smaller = faster but may miss ball
    # Recommended: 150 - 250
    INITIAL_ROI_SIZE = 400
    
    # Minimum ROI size in pixels
    # Prevents ROI from becoming too small
    # Recommended: 80 - 150
    MIN_ROI_SIZE = 400
    
    # Maximum ROI size in pixels
    # Prevents ROI from becoming too large
    # Recommended: 300 - 500
    MAX_ROI_SIZE = 600
    
    # ROI expansion factor when detection fails
    # Higher = faster expansion (more aggressive search)
    # Lower = slower expansion (more conservative)
    # Recommended: 1.15 - 1.3
    ROI_EXPANSION_FACTOR = 1.2
    
    # ═══════════════════════════════════════════════════════════
    # VALIDATION PARAMETERS (False Positive Reduction)
    # ═══════════════════════════════════════════════════════════
    
    # Maximum distance (in pixels) from predicted position
    # Detections farther than this are rejected as false positives
    # Higher = more tolerant to sudden movements
    # Lower = stricter validation
    # Recommended: 100 - 200
    MAX_DISTANCE = 1000000.0
    
    # Maximum ratio of bounding box area change
    # Example: 2.5 means area can change by 2.5x between frames
    # Higher = accept larger size changes
    # Lower = stricter size consistency
    # Recommended: 2.0 - 3.0
    MAX_SIZE_CHANGE_RATIO = 2.5
    
    # Minimum bounding box area (in pixels²)
    # Detections smaller than this are rejected
    # Prevents tracking tiny false positives
    # Recommended: 30 - 100
    MIN_BBOX_AREA = 9
    
    # Aspect ratio range for valid detections3 (width/height)
    # Golf balls should be roughly circular
    # [min, max] range
    # Recommended: [0.3, 3.0]
    MIN_ASPECT_RATIO = 0.3
    MAX_ASPECT_RATIO = 3.0
    
    # Minimum distance (pixels) between consecutive recorded detections
    # Points closer than this are considered the same position and skipped
    # Prevents duplicate points when the ball is stationary
    # Recommended: 5 - 20
    MIN_DETECTION_DISTANCE = 10.0

    # ═══════════════════════════════════════════════════════════
    # PREDICTION PARAMETERS (Trajectory Extension)
    # ═══════════════════════════════════════════════════════════
    
    # Enable trajectory prediction / extension
    PREDICTION_ENABLED = True
    
    # Minimum consecutive segments with same direction to trigger prediction
    # At least 2 are needed to establish a direction
    # Recommended: 2 - 4
    PREDICTION_MIN_CONSISTENT_SEGMENTS = 2
    
    # Maximum angular difference (degrees) between two consecutive
    # segments to consider them "same direction"
    # Higher = more tolerant to slight curves
    # Lower = stricter straight-line requirement
    # Recommended: 20 - 45
    PREDICTION_DIRECTION_TOLERANCE_DEG = 30.0
    
    # How many of the most recent trajectory points to consider
    # when determining the direction for prediction
    # Recommended: 3 - 10
    PREDICTION_LOOKBACK_POINTS = 5
    
    # Maximum prediction extension distance in pixels
    # The predicted line will stop after this distance from the last real point
    # Recommended: 100 - 500
    PREDICTION_MAX_LENGTH_PX = 300.0
    
    # Prediction line color (BGR)
    PREDICTION_LINE_COLOR = (0, 255, 0)  # green 
    
    # Prediction line thickness
    PREDICTION_LINE_THICKNESS = 10
    
    # How many predicted points to reveal per frame (animation speed)
    # Higher = faster reveal, Lower = slower / smoother animation
    # Set to 0 or negative to reveal all at once (no animation)
    # Recommended: 1 - 5
    PREDICTION_POINTS_PER_FRAME = 1
    
    # ═══════════════════════════════════════════════════════════
    # VISUALIZATION PARAMETERS
    # ═══════════════════════════════════════════════════════════
    
    # Show ROI rectangle on output video
    SHOW_ROI = False
    
    # Show predicted position marker on output video
    SHOW_PREDICTION = True
    
    # Show velocity vector arrow
    SHOW_VELOCITY = False
    
    # Trajectory line thickness (pixels)
    TRAJECTORY_THICKNESS = 10
    
    # Trajectory color (BGR format)
    TRAJECTORY_COLOR = (0, 255, 0)  # Green
    
    # ROI rectangle color (BGR format)
    ROI_COLOR = (255, 255, 0)  # Cyan
    
    # Prediction marker color (BGR format)
    PRED_COLOR = (0, 0, 255)  # Red
    
    # ═══════════════════════════════════════════════════════════
    # PERFORMANCE PARAMETERS
    # ═══════════════════════════════════════════════════════════
    
    # Show progress updates every N frames
    PROGRESS_UPDATE_INTERVAL = 30
    
    # Enable verbose logging
    VERBOSE = True


# ═══════════════════════════════════════════════════════════
# PRESET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════

class FastModeConfig(Config):
    """
    Configuration optimized for speed.
    Use for real-time processing or low-end hardware.
    """
    INITIAL_ROI_SIZE = 150
    MAX_ROI_SIZE = 300
    CONFIDENCE_THRESHOLD = 0.4
    MEASUREMENT_NOISE = 3.0


class AccurateModeConfig(Config):
    """
    Configuration optimized for accuracy.
    Use for offline processing where quality > speed.
    """
    INITIAL_ROI_SIZE = 250
    MAX_ROI_SIZE = 500
    CONFIDENCE_THRESHOLD = 0.3
    MAX_DISTANCE = 200.0
    MEASUREMENT_NOISE = 7.0


class RobustModeConfig(Config):
    """
    Configuration optimized for robustness.
    Use when ball frequently disappears or has occlusions.
    """
    MAX_MISSED_FRAMES = 25
    MAX_DISTANCE = 200.0
    ROI_EXPANSION_FACTOR = 1.3
    PROCESS_NOISE = 0.05
    MEASUREMENT_NOISE = 8.0


# ═══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def get_config(mode: str = "default") -> Config:
    """
    Get configuration based on mode.
    
    Args:
        mode: Configuration mode ("default", "fast", "accurate", "robust")
        
    Returns:
        Config object
    """
    modes = {
        "default": Config,
        "fast": FastModeConfig,
        "accurate": AccurateModeConfig,
        "robust": RobustModeConfig
    }
    
    config_class = modes.get(mode.lower(), Config)
    return config_class()


def print_config(config: Config):
    """
    Print current configuration parameters.
    
    Args:
        config: Config object
    """
    print("=" * 60)
    print("Configuration Parameters")
    print("=" * 60)
    
    print("\n[File Paths]")
    print(f"  MODEL_PATH: {config.MODEL_PATH}")
    print(f"  VIDEO_INPUT: {config.VIDEO_INPUT}")
    print(f"  VIDEO_OUTPUT: {config.VIDEO_OUTPUT}")
    
    print("\n[Detection]")
    print(f"  CONFIDENCE_THRESHOLD: {config.CONFIDENCE_THRESHOLD}")
    
    print("\n[Tracking]")
    print(f"  MAX_MISSED_FRAMES: {config.MAX_MISSED_FRAMES}")
    print(f"  PROCESS_NOISE: {config.PROCESS_NOISE}")
    print(f"  MEASUREMENT_NOISE: {config.MEASUREMENT_NOISE}")
    
    print("\n[ROI]")
    print(f"  INITIAL_ROI_SIZE: {config.INITIAL_ROI_SIZE}")
    print(f"  MIN_ROI_SIZE: {config.MIN_ROI_SIZE}")
    print(f"  MAX_ROI_SIZE: {config.MAX_ROI_SIZE}")
    print(f"  ROI_EXPANSION_FACTOR: {config.ROI_EXPANSION_FACTOR}")
    
    print("\n[Validation]")
    print(f"  MAX_DISTANCE: {config.MAX_DISTANCE}")
    print(f"  MAX_SIZE_CHANGE_RATIO: {config.MAX_SIZE_CHANGE_RATIO}")
    
    print("=" * 60)
