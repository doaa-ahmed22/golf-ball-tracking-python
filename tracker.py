"""
Golf Ball Tracker Module
Implements Kalman filter-based tracking for smooth ball trajectory prediction.
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List


class KalmanBallTracker:
    """
    Kalman filter-based tracker for golf ball with constant velocity model.
    
    State vector: [x, y, vx, vy]
        x, y: ball center position
        vx, vy: velocity components
    
    Measurement vector: [x, y]
        Direct position measurements from detector
    """
    
    def __init__(
        self,
        initial_position: Tuple[float, float],
        process_noise: float = 0.03,
        measurement_noise: float = 5.0,
        max_missed_frames: int = 15
    ):
        """
        Initialize Kalman filter tracker.
        
        Args:
            initial_position: Initial (x, y) position
            process_noise: Process noise covariance (model uncertainty)
            measurement_noise: Measurement noise covariance (detection uncertainty)
            max_missed_frames: Maximum consecutive frames without detection before reset
        """
        self.max_missed_frames = max_missed_frames
        self.missed_frames = 0
        self.is_active = True
        
        # Initialize Kalman Filter (4 states, 2 measurements)
        self.kalman = cv2.KalmanFilter(4, 2, 0)
        
        # State transition matrix (constant velocity model)
        # x_next = x + vx * dt
        # y_next = y + vy * dt
        # vx_next = vx
        # vy_next = vy
        dt = 1.0  # Assume constant time step
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we measure position x, y directly)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance matrix
        # Higher values = trust the model less, adapt faster to measurements
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        
        # Measurement noise covariance matrix
        # Higher values = trust measurements less, smoother predictions
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Error covariance matrix (initial uncertainty)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1000
        
        # Initialize state with initial position and zero velocity
        x, y = initial_position
        self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        
        # Trajectory storage (smoothed positions)
        self.trajectory: List[Tuple[int, int]] = [(int(x), int(y))]
        
        # Last valid bounding box (for validation)
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        
    def predict(self) -> Tuple[float, float]:
        """
        Predict next position using Kalman filter.
        
        Returns:
            Predicted (x, y) position
        """
        prediction = self.kalman.predict()
        pred_x = float(prediction[0])
        pred_y = float(prediction[1])
        return (pred_x, pred_y)
    
    def update(self, measurement: Tuple[float, float], bbox: Optional[Tuple[int, int, int, int]] = None):
        """
        Update Kalman filter with new measurement.
        
        Args:
            measurement: Measured (x, y) position from detector
            bbox: Associated bounding box (x1, y1, x2, y2)
        """
        # Convert measurement to numpy array
        meas = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        
        # Correct Kalman filter with measurement
        self.kalman.correct(meas)
        
        # Store corrected position in trajectory
        corrected = self.kalman.statePost
        x = int(corrected[0])
        y = int(corrected[1])
        self.trajectory.append((x, y))
        
        # Update last valid bbox
        if bbox is not None:
            self.last_bbox = bbox
        
        # Reset missed frames counter
        self.missed_frames = 0
    
    def update_with_prediction(self):
        """
        Update trajectory with predicted position when no measurement is available.
        This keeps the trajectory smooth during temporary disappearances.
        """
        # Get predicted state
        predicted = self.kalman.statePre
        x = int(predicted[0])
        y = int(predicted[1])
        
        # Add predicted position to trajectory
        self.trajectory.append((x, y))
        
        # Increment missed frames counter
        self.missed_frames += 1
        
        # Check if tracking should be terminated
        if self.missed_frames >= self.max_missed_frames:
            self.is_active = False
    
    def get_current_position(self) -> Tuple[int, int]:
        """
        Get current tracked position (last in trajectory).
        
        Returns:
            Current (x, y) position
        """
        if len(self.trajectory) > 0:
            return self.trajectory[-1]
        return (0, 0)
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """
        Get full trajectory history.
        
        Returns:
            List of (x, y) positions
        """
        return self.trajectory.copy()
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate from Kalman filter.
        
        Returns:
            (vx, vy) velocity components
        """
        state = self.kalman.statePost
        vx = float(state[2])
        vy = float(state[3])
        return (vx, vy)
    
    def get_speed(self) -> float:
        """
        Get current speed magnitude.
        
        Returns:
            Speed in pixels per frame
        """
        vx, vy = self.get_velocity()
        return np.sqrt(vx**2 + vy**2)
    
    def reset(self):
        """
        Reset tracker state (called when tracking is lost).
        """
        self.is_active = False
        self.missed_frames = 0


class TrackingState:
    """
    Enum-like class for tracking states.
    """
    DETECTION = "detection"  # Initial detection phase
    TRACKING = "tracking"    # Active tracking with Kalman filter
    LOST = "lost"            # Tracking lost, need re-initialization
