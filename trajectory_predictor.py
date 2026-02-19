"""
Trajectory Prediction Module
Extends a tracked trajectory forward by analyzing the direction of the last
few points.  When at least two consecutive segments share a consistent
direction (within an angular tolerance), the module projects additional
points along that direction until the frame boundary or a maximum distance
is reached.
"""

import math
import numpy as np
import cv2
from typing import List, Tuple, Optional


class TrajectoryPredictor:
    """
    Predict / extend a ball trajectory based on the direction established
    by the most recent detected points.

    Usage:
        predictor = TrajectoryPredictor(frame_width, frame_height)
        predicted = predictor.predict(all_detections)
        predictor.draw_predicted(frame, predicted)
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        min_consistent_segments: int = 2,
        direction_tolerance_deg: float = 30.0,
        lookback_points: int = 5,
        max_prediction_length_px: float = 2000.0,
        prediction_step_px: Optional[float] = None,
    ):
        """
        Args:
            frame_width:  Video frame width (used to clip predictions).
            frame_height: Video frame height.
            min_consistent_segments: Minimum number of consecutive segments
                that must agree in direction before prediction kicks in.
            direction_tolerance_deg: Maximum angular difference (degrees)
                between two consecutive segments to consider them
                "same direction".
            lookback_points: How many of the most recent trajectory points
                to consider when determining direction.
            max_prediction_length_px: Stop predicting after this cumulative
                distance (pixels) from the last real point.
            prediction_step_px: Distance between consecutive predicted
                points.  If None, uses the average step size of the
                consistent segments.
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_consistent_segments = min_consistent_segments
        self.direction_tolerance_deg = direction_tolerance_deg
        self.lookback_points = lookback_points
        self.max_prediction_length_px = max_prediction_length_px
        self.prediction_step_px = prediction_step_px

    # ──────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────

    def predict(
        self, trajectory: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Given detected trajectory points, return a list of *predicted*
        extension points (may be empty if direction is not consistent).

        Args:
            trajectory: List of (x, y) detected positions.

        Returns:
            List of predicted (x, y) positions extending the trajectory.
        """
        if len(trajectory) < self.min_consistent_segments + 1:
            return []

        # Take the last N points
        tail = trajectory[-self.lookback_points:]
        if len(tail) < 2:
            return []

        # Compute direction vectors between consecutive tail points
        segments = []
        for i in range(1, len(tail)):
            dx = tail[i][0] - tail[i - 1][0]
            dy = tail[i][1] - tail[i - 1][1]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue  # skip zero-length segments
            angle = math.atan2(dy, dx)
            segments.append((dx, dy, length, angle))

        if len(segments) < self.min_consistent_segments:
            return []

        # Find the longest run of consecutive segments with consistent direction
        best_run_start = 0
        best_run_len = 1
        current_start = 0
        current_len = 1

        for i in range(1, len(segments)):
            angle_diff = abs(segments[i][3] - segments[i - 1][3])
            # Normalize to [0, pi]
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
            tolerance_rad = math.radians(self.direction_tolerance_deg)

            if angle_diff <= tolerance_rad:
                current_len += 1
                if current_len > best_run_len:
                    best_run_len = current_len
                    best_run_start = current_start
            else:
                current_start = i
                current_len = 1

        if best_run_len < self.min_consistent_segments:
            return []

        # Use the consistent run to derive direction & step size
        consistent = segments[best_run_start: best_run_start + best_run_len]
        avg_dx = np.mean([s[0] for s in consistent])
        avg_dy = np.mean([s[1] for s in consistent])
        avg_length = math.hypot(avg_dx, avg_dy)

        if avg_length < 1e-6:
            return []

        # Unit direction vector
        ux = avg_dx / avg_length
        uy = avg_dy / avg_length

        # Step size
        step = self.prediction_step_px if self.prediction_step_px else avg_length

        # Starting point: last detection
        start_x, start_y = float(trajectory[-1][0]), float(trajectory[-1][1])

        # Generate predicted points
        predicted: List[Tuple[int, int]] = []
        cumulative = 0.0

        while cumulative < self.max_prediction_length_px:
            cumulative += step
            px = start_x + ux * cumulative
            py = start_y + uy * cumulative

            ix, iy = int(round(px)), int(round(py))

            # Stop if outside frame boundaries
            if ix < 0 or ix >= self.frame_width or iy < 0 or iy >= self.frame_height:
                break

            predicted.append((ix, iy))

        return predicted

    # ──────────────────────────────────────────────────────────
    # DRAWING
    # ──────────────────────────────────────────────────────────

    def draw_predicted(
        self,
        frame: np.ndarray,
        predicted_points: List[Tuple[int, int]],
        last_real_point: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 0, 255),
        thickness: int = 2,
    ):
        """
        Draw the predicted trajectory extension as a solid line.

        Args:
            frame: Video frame to draw on.
            predicted_points: List of predicted (x, y) points.
            last_real_point: The last confirmed detection point (to
                connect the line from).
            color: BGR colour for the prediction line.
            thickness: Line thickness.
        """
        if not predicted_points:
            return

        # Build the full chain: last_real → predicted_0 → predicted_1 → …
        chain = [last_real_point] + predicted_points

        for i in range(1, len(chain)):
            cv2.line(frame, chain[i - 1], chain[i], color, thickness, cv2.LINE_AA)

    # ──────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────

    def _clip_to_frame(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Optional[Tuple[int, int]]:
        """Return the point where the segment (x0,y0)→(x1,y1) intersects
        the frame boundary, or None."""
        dx = x1 - x0
        dy = y1 - y0

        t_values = []
        # Left boundary x=0
        if dx != 0:
            t = -x0 / dx
            if t > 0:
                t_values.append(t)
        # Right boundary x=width-1
        if dx != 0:
            t = (self.frame_width - 1 - x0) / dx
            if t > 0:
                t_values.append(t)
        # Top boundary y=0
        if dy != 0:
            t = -y0 / dy
            if t > 0:
                t_values.append(t)
        # Bottom boundary y=height-1
        if dy != 0:
            t = (self.frame_height - 1 - y0) / dy
            if t > 0:
                t_values.append(t)

        if not t_values:
            return None

        # Pick the smallest positive t that brings us inside or on the edge
        for t in sorted(t_values):
            bx = x0 + dx * t
            by = y0 + dy * t
            ix, iy = int(round(bx)), int(round(by))
            if 0 <= ix < self.frame_width and 0 <= iy < self.frame_height:
                return (ix, iy)

        return None

    @staticmethod
    def _draw_dashed_line(
        frame: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_length: int,
        gap_length: int,
    ):
        """Draw a dashed line between two points."""
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        total_length = math.hypot(dx, dy)
        if total_length < 1:
            return

        ux = dx / total_length
        uy = dy / total_length

        segment = dash_length + gap_length
        dist = 0.0
        drawing = True  # alternate dash / gap

        while dist < total_length:
            if drawing:
                end_dist = min(dist + dash_length, total_length)
                sx = int(pt1[0] + ux * dist)
                sy = int(pt1[1] + uy * dist)
                ex = int(pt1[0] + ux * end_dist)
                ey = int(pt1[1] + uy * end_dist)
                cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
                dist = end_dist
            else:
                dist += gap_length
            drawing = not drawing
