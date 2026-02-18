"""
Golf Ball Detector Module
Wraps YOLOv8 for detecting golf balls with confidence filtering.
"""

from ultralytics import YOLO
import numpy as np
from typing import Optional, Tuple, List


class BallDetector:
    """
    YOLOv8-based golf ball detector with validation.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.35):
        """
        Initialize the ball detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (e.g., 'best.pt')
            confidence_threshold: Minimum confidence for valid detection
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
    def detect_full_frame(self, frame: np.ndarray) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        """
        Run detection on full frame and return the highest confidence ball detection.
        
        Args:
            frame: Input frame (full resolution)
            
        Returns:
            Tuple of (center_x, center_y, confidence, bbox) or None if no detection
            bbox is (x1, y1, x2, y2)
        """
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        
        if len(results[0].boxes) == 0:
            return None
        
        # Get highest confidence detection
        best_box = None
        best_conf = 0.0
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box
        
        if best_box is None:
            return None
        
        # Extract bounding box
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Calculate center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        return (cx, cy, best_conf, (x1, y1, x2, y2))
    
    def detect_in_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, float, Tuple[int, int, int, int]]]:
        """
        Run detection only within ROI region.
        
        Args:
            frame: Input frame (full resolution)
            roi: ROI coordinates (x1, y1, x2, y2)
            
        Returns:
            Tuple of (center_x, center_y, confidence, bbox) in GLOBAL coordinates or None
            bbox is (x1, y1, x2, y2) in global frame coordinates
        """
        x1_roi, y1_roi, x2_roi, y2_roi = roi
        
        # Validate ROI boundaries
        h, w = frame.shape[:2]
        x1_roi = max(0, min(x1_roi, w))
        y1_roi = max(0, min(y1_roi, h))
        x2_roi = max(0, min(x2_roi, w))
        y2_roi = max(0, min(y2_roi, h))
        
        # Extract ROI
        if x2_roi <= x1_roi or y2_roi <= y1_roi:
            return None
        
        roi_crop = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        
        # Run detection on ROI
        results = self.model(roi_crop, verbose=False, conf=self.confidence_threshold)
        
        if len(results[0].boxes) == 0:
            return None
        
        # Get highest confidence detection
        best_box = None
        best_conf = 0.0
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box
        
        if best_box is None:
            return None
        
        # Extract bounding box (in ROI coordinates)
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Convert to global coordinates
        x1_global = x1 + x1_roi
        y1_global = y1 + y1_roi
        x2_global = x2 + x1_roi
        y2_global = y2 + y1_roi
        
        # Calculate center in global coordinates
        cx = (x1_global + x2_global) // 2
        cy = (y1_global + y2_global) // 2
        
        return (cx, cy, best_conf, (x1_global, y1_global, x2_global, y2_global))
    
    def validate_detection(
        self,
        detection: Tuple[int, int, float, Tuple[int, int, int, int]],
        predicted_pos: Optional[Tuple[float, float]] = None,
        last_bbox: Optional[Tuple[int, int, int, int]] = None,
        max_distance: float = 150.0,
        max_size_change_ratio: float = 2.5
    ) -> bool:
        """
        Validate a detection against predicted position and last bbox.
        
        Args:
            detection: (cx, cy, confidence, bbox)
            predicted_pos: (pred_x, pred_y) from Kalman filter
            last_bbox: Previous bounding box (x1, y1, x2, y2)
            max_distance: Maximum allowed distance from predicted position
            max_size_change_ratio: Maximum ratio change in bbox area
            
        Returns:
            True if detection is valid, False otherwise
        """
        cx, cy, conf, bbox = detection
        x1, y1, x2, y2 = bbox
        
        # 1. Confidence check (already done in detect methods, but double-check)
        if conf < self.confidence_threshold:
            return False
        
        # 2. Distance check from predicted position
        if predicted_pos is not None:
            pred_x, pred_y = predicted_pos
            distance = np.sqrt((cx - pred_x)**2 + (cy - pred_y)**2)
            if distance > max_distance:
                return False
        
        # 3. Size consistency check
        if last_bbox is not None:
            last_x1, last_y1, last_x2, last_y2 = last_bbox
            last_area = (last_x2 - last_x1) * (last_y2 - last_y1)
            curr_area = (x2 - x1) * (y2 - y1)
            
            if last_area > 0 and curr_area > 0:
                size_ratio = max(curr_area / last_area, last_area / curr_area)
                if size_ratio > max_size_change_ratio:
                    return False
        
        # 4. Aspect ratio check (ball should be roughly circular)
        width = x2 - x1
        height = y2 - y1
        if height > 0:
            aspect_ratio = width / height
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                return False
        
        # 5. Minimum size check
        bbox_area = width * height
        if bbox_area < 50:  # Too small
            return False
        
        return True
