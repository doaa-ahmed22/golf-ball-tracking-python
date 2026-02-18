"""
ROI Manager Module
Manages dynamic Region of Interest (ROI) for efficient ball detection.
"""

import numpy as np
from typing import Tuple


class ROIManager:
    """
    Dynamic ROI manager that adapts size based on tracking confidence.
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        initial_roi_size: int = 200,
        min_roi_size: int = 100,
        max_roi_size: int = 400,
        expansion_factor: float = 1.2
    ):
        """
        Initialize ROI manager.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            initial_roi_size: Initial ROI size (square)
            min_roi_size: Minimum ROI size
            max_roi_size: Maximum ROI size
            expansion_factor: Factor to expand ROI when detection fails
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.initial_roi_size = initial_roi_size
        self.min_roi_size = min_roi_size
        self.max_roi_size = max_roi_size
        self.expansion_factor = expansion_factor
        
        # Current ROI size
        self.current_roi_size = initial_roi_size
        
    def get_roi(self, center: Tuple[float, float]) -> Tuple[int, int, int, int]:
        """
        Get ROI bounding box centered at given position.
        
        Args:
            center: (x, y) center position for ROI
            
        Returns:
            ROI coordinates (x1, y1, x2, y2) clamped to frame boundaries
        """
        cx, cy = center
        half_size = self.current_roi_size // 2
        
        # Calculate ROI bounds
        x1 = int(cx - half_size)
        y1 = int(cy - half_size)
        x2 = int(cx + half_size)
        y2 = int(cy + half_size)
        
        # Clamp to frame boundaries
        x1 = max(0, min(x1, self.frame_width))
        y1 = max(0, min(y1, self.frame_height))
        x2 = max(0, min(x2, self.frame_width))
        y2 = max(0, min(y2, self.frame_height))
        
        return (x1, y1, x2, y2)
    
    def expand_roi(self):
        """
        Expand ROI size (called when detection fails).
        Gradually increases search area to find the ball.
        """
        self.current_roi_size = int(self.current_roi_size * self.expansion_factor)
        self.current_roi_size = min(self.current_roi_size, self.max_roi_size)
    
    def shrink_roi(self):
        """
        Shrink ROI size (called when detection succeeds).
        Reduces search area for better performance.
        """
        self.current_roi_size = int(self.current_roi_size / self.expansion_factor)
        self.current_roi_size = max(self.current_roi_size, self.min_roi_size)
    
    def reset_roi(self):
        """
        Reset ROI to initial size.
        """
        self.current_roi_size = self.initial_roi_size
    
    def get_adaptive_roi(
        self,
        center: Tuple[float, float],
        velocity: Tuple[float, float],
        speed_factor: float = 2.0
    ) -> Tuple[int, int, int, int]:
        """
        Get velocity-adaptive ROI that shifts in the direction of motion.
        
        Args:
            center: (x, y) center position for ROI
            velocity: (vx, vy) velocity components
            speed_factor: Factor to shift ROI based on velocity
            
        Returns:
            ROI coordinates (x1, y1, x2, y2) clamped to frame boundaries
        """
        cx, cy = center
        vx, vy = velocity
        
        # Offset center by velocity prediction
        offset_x = vx * speed_factor
        offset_y = vy * speed_factor
        
        adjusted_cx = cx + offset_x
        adjusted_cy = cy + offset_y
        
        # Get ROI around adjusted center
        return self.get_roi((adjusted_cx, adjusted_cy))
    
    def get_current_size(self) -> int:
        """
        Get current ROI size.
        
        Returns:
            Current ROI size
        """
        return self.current_roi_size
    
    def set_size(self, size: int):
        """
        Manually set ROI size.
        
        Args:
            size: New ROI size
        """
        self.current_roi_size = max(self.min_roi_size, min(size, self.max_roi_size))
    
    def should_use_full_frame(self) -> bool:
        """
        Determine if ROI is large enough that full frame should be used instead.
        
        Returns:
            True if should use full frame detection
        """
        # If ROI covers more than 80% of frame, use full frame
        roi_area = self.current_roi_size * self.current_roi_size
        frame_area = self.frame_width * self.frame_height
        coverage = roi_area / frame_area
        
        return coverage > 0.8
