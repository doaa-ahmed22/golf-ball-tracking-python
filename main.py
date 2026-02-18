"""
Golf Ball Tracking System - Main Orchestration
Implements robust ball tracking with Kalman filtering, dynamic ROI, and validation.

Key improvement: Full-frame fallback detection + gap-bridging trajectory drawing.
When the ball moves far from the predicted ROI (fast golf swing), the system
falls back to full-frame detection to re-acquire the ball, then draws a line
connecting ALL detected points across gaps.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import time
import sys

from detector import BallDetector
from tracker import KalmanBallTracker, TrackingState
from roi_manager import ROIManager

# Import configuration
try:
    from config import Config, get_config, print_config
    USE_EXTERNAL_CONFIG = True
except ImportError:
    USE_EXTERNAL_CONFIG = False
    print("Warning: config.py not found, using default inline configuration")

    class Config:
        """Fallback configuration if config.py is not available."""
        MODEL_PATH = "best.pt"
        VIDEO_INPUT = "input_3.mp4"
        VIDEO_OUTPUT = "output_kalman.mp4"
        CONFIDENCE_THRESHOLD = 0.35
        MAX_MISSED_FRAMES = 15
        PROCESS_NOISE = 0.03
        MEASUREMENT_NOISE = 5.0
        INITIAL_ROI_SIZE = 200
        MIN_ROI_SIZE = 100
        MAX_ROI_SIZE = 400
        ROI_EXPANSION_FACTOR = 1.2
        MAX_DISTANCE = 150.0
        MAX_SIZE_CHANGE_RATIO = 2.5
        SHOW_ROI = True
        SHOW_PREDICTION = True
        TRAJECTORY_THICKNESS = 2
        TRAJECTORY_COLOR = (0, 255, 0)
        ROI_COLOR = (255, 255, 0)
        PRED_COLOR = (0, 0, 255)


# ═══════════════════════════════════════════════════════════
# TRAJECTORY DRAWING
# ═══════════════════════════════════════════════════════════

def draw_trajectory(frame: np.ndarray, all_detections: List[Tuple[int, int]]):
    """
    Draw the full trajectory line connecting all detected ball positions.
    This bridges gaps — even if the ball was lost for several frames,
    the line connects from the last known point to the newly re-acquired point.

    Args:
        frame: The video frame to draw on
        all_detections: List of (x, y) positions of every confirmed detection
    """
    if len(all_detections) < 2:
        return

    # Draw lines between consecutive detection points
    for i in range(1, len(all_detections)):
        pt1 = all_detections[i - 1]
        pt2 = all_detections[i]

        # Fade color from yellow (start) to green (recent)
        alpha = i / len(all_detections)
        color = (
            int(0 * alpha),           # B: 0
            int(255 * alpha),          # G: fades in
            int(255 * (1 - alpha))     # R: fades out
        )
        cv2.line(frame, pt1, pt2, color, Config.TRAJECTORY_THICKNESS + 1, cv2.LINE_AA)

    # Draw dots at each detection point
    for i, pt in enumerate(all_detections):
        alpha = i / max(len(all_detections) - 1, 1)
        dot_color = (
            int(0 * alpha),
            int(200 * alpha + 55),
            int(255 * (1 - alpha))
        )
        cv2.circle(frame, pt, 4, dot_color, -1)

    # Mark start and end
    cv2.circle(frame, all_detections[0], 8, (0, 255, 255), 2)   # Cyan = start
    cv2.putText(frame, "START", (all_detections[0][0] + 10, all_detections[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    if len(all_detections) > 1:
        cv2.circle(frame, all_detections[-1], 8, (0, 165, 255), 2)  # Orange = latest
        cv2.putText(frame, "LATEST", (all_detections[-1][0] + 10, all_detections[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)


# ═══════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════

def draw_info_panel(
    frame: np.ndarray,
    frame_idx: int,
    state: str,
    tracker: Optional[KalmanBallTracker],
    roi_size: int,
    fps: float,
    total_detections: int,
    fallback_count: int
):
    """Draw information panel on frame."""
    h, w = frame.shape[:2]
    panel_height = 120
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray background

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    line_height = 20
    x_start = 10
    y_start = 20

    cv2.putText(panel, f"Frame: {frame_idx}", (x_start, y_start),
                font, font_scale, color, thickness)
    cv2.putText(panel, f"FPS: {fps:.1f}", (x_start + 150, y_start),
                font, font_scale, color, thickness)
    cv2.putText(panel, f"Detections: {total_detections}", (x_start + 300, y_start),
                font, font_scale, (0, 255, 100), thickness)

    state_color = (0, 255, 0) if state == TrackingState.TRACKING else (0, 165, 255)
    cv2.putText(panel, f"State: {state}", (x_start, y_start + line_height),
                font, font_scale, state_color, thickness)
    cv2.putText(panel, f"Fallback detections: {fallback_count}", (x_start + 300, y_start + line_height),
                font, font_scale, (0, 200, 255), thickness)

    cv2.putText(panel, f"ROI Size: {roi_size}px", (x_start, y_start + line_height * 2),
                font, font_scale, color, thickness)

    if tracker and tracker.is_active:
        pos = tracker.get_current_position()
        speed = tracker.get_speed()
        missed = tracker.missed_frames

        cv2.putText(panel, f"Position: ({pos[0]}, {pos[1]})",
                    (x_start, y_start + line_height * 3),
                    font, font_scale, color, thickness)
        cv2.putText(panel, f"Speed: {speed:.1f} px/frame",
                    (x_start + 250, y_start + line_height * 3),
                    font, font_scale, color, thickness)
        cv2.putText(panel, f"Missed: {missed}/{Config.MAX_MISSED_FRAMES}",
                    (x_start, y_start + line_height * 4),
                    font, font_scale, (0, 255, 255) if missed > 5 else color, thickness)
        cv2.putText(panel, f"Trajectory: {len(tracker.trajectory)} points",
                    (x_start + 250, y_start + line_height * 4),
                    font, font_scale, color, thickness)

    combined = np.vstack([panel, frame])
    return combined


# ═══════════════════════════════════════════════════════════
# MAIN TRACKING LOGIC
# ═══════════════════════════════════════════════════════════

def main():
    """Main tracking pipeline."""

    print("=" * 60)
    print("Golf Ball Tracking System (YOLOv8 + Kalman Filter)")
    print("=" * 60)
    print(f"Model: {Config.MODEL_PATH}")
    print(f"Video: {Config.VIDEO_INPUT}")
    print(f"Output: {Config.VIDEO_OUTPUT}")
    print("-" * 60)

    # ─── Initialize Components ───
    print("Initializing detector...")
    detector = BallDetector(Config.MODEL_PATH, Config.CONFIDENCE_THRESHOLD)

    print("Opening video...")
    cap = cv2.VideoCapture(Config.VIDEO_INPUT)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {Config.VIDEO_INPUT}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print("-" * 60)

    roi_manager = ROIManager(
        width, height,
        Config.INITIAL_ROI_SIZE,
        Config.MIN_ROI_SIZE,
        Config.MAX_ROI_SIZE,
        Config.ROI_EXPANSION_FACTOR
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    panel_height = 120
    out = cv2.VideoWriter(
        Config.VIDEO_OUTPUT,
        fourcc,
        fps,
        (width, height + panel_height)
    )

    # ─── Tracking State Variables ───
    state = TrackingState.DETECTION
    tracker: Optional[KalmanBallTracker] = None
    frame_idx = 0
    detection_attempts = 0
    successful_tracks = 0

    # ─── Gap-bridging: accumulate ALL confirmed detections ───
    # This list stores every real detection (not Kalman predictions),
    # so we can draw a line connecting them even across large gaps.
    all_detections: List[Tuple[int, int]] = []

    # Count how many times we re-acquired via full-frame fallback
    fallback_reacquisitions = 0

    # How many consecutive missed frames before we try full-frame fallback
    # (separate from MAX_MISSED_FRAMES which resets the tracker entirely)
    FALLBACK_AFTER_MISSED = 3

    process_times = []
    start_time = time.time()

    print("Starting processing...")
    print("Key: Cyan circle = START, Orange circle = LATEST detection")
    print("     Yellow→Green line = trajectory (bridges gaps)")
    print("=" * 60)

    # ─── Main Processing Loop ───
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()
        frame_idx += 1

        # ═══════════════════════════════════════════════════════
        # PHASE 1: INITIAL DETECTION (Bootstrap)
        # ═══════════════════════════════════════════════════════
        if state == TrackingState.DETECTION:
            detection_attempts += 1

            detection = detector.detect_full_frame(frame)

            if detection is not None:
                cx, cy, conf, bbox = detection

                tracker = KalmanBallTracker(
                    initial_position=(cx, cy),
                    process_noise=Config.PROCESS_NOISE,
                    measurement_noise=Config.MEASUREMENT_NOISE,
                    max_missed_frames=Config.MAX_MISSED_FRAMES
                )
                tracker.last_bbox = bbox
                state = TrackingState.TRACKING
                successful_tracks += 1
                roi_manager.reset_roi()

                # Record this detection for trajectory drawing
                all_detections.append((cx, cy))

                print(f"✓ Frame {frame_idx}: Ball detected at ({cx}, {cy}) "
                      f"confidence={conf:.2f} → Tracking started")

                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
                cv2.putText(frame, f"DETECTED: {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ═══════════════════════════════════════════════════════
        # PHASE 2: TRACKING (Kalman Filter + ROI)
        # ═══════════════════════════════════════════════════════
        elif state == TrackingState.TRACKING and tracker is not None:

            # Step 1: Predict next position
            pred_x, pred_y = tracker.predict()

            # Step 2: Get velocity for adaptive ROI
            vx, vy = tracker.get_velocity()

            # Step 3: Create dynamic ROI around prediction
            if tracker.get_speed() > 10:
                roi = roi_manager.get_adaptive_roi((pred_x, pred_y), (vx, vy))
            else:
                roi = roi_manager.get_roi((pred_x, pred_y))

            # Step 4: Run detection in ROI
            detection = detector.detect_in_roi(frame, roi)

            # Step 5: Process detection result
            if detection is not None:
                cx, cy, conf, bbox = detection

                is_valid = detector.validate_detection(
                    detection,
                    predicted_pos=(pred_x, pred_y),
                    last_bbox=tracker.last_bbox,
                    max_distance=Config.MAX_DISTANCE,
                    max_size_change_ratio=Config.MAX_SIZE_CHANGE_RATIO
                )

                if is_valid:
                    tracker.update((cx, cy), bbox)
                    roi_manager.shrink_roi()

                    # Record this confirmed detection for trajectory
                    all_detections.append((cx, cy))

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    tracker.update_with_prediction()
                    roi_manager.expand_roi()

                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, "REJECTED", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            else:
                # ─── No detection in ROI ───
                # If we've missed enough frames, try a full-frame fallback
                # to re-acquire the ball even if it moved far away.
                if tracker.missed_frames >= FALLBACK_AFTER_MISSED:
                    fallback_detection = detector.detect_full_frame(frame)

                    if fallback_detection is not None:
                        fcx, fcy, fconf, fbbox = fallback_detection

                        # Re-initialize tracker at the newly found position
                        tracker = KalmanBallTracker(
                            initial_position=(fcx, fcy),
                            process_noise=Config.PROCESS_NOISE,
                            measurement_noise=Config.MEASUREMENT_NOISE,
                            max_missed_frames=Config.MAX_MISSED_FRAMES
                        )
                        tracker.last_bbox = fbbox
                        roi_manager.reset_roi()
                        fallback_reacquisitions += 1

                        # Record this re-acquired point — this is the key!
                        # The trajectory line will now bridge from the last
                        # known point to this newly found distant point.
                        all_detections.append((fcx, fcy))

                        print(f"  ↻ Frame {frame_idx}: Fallback re-acquired at "
                              f"({fcx}, {fcy}) conf={fconf:.2f} "
                              f"[gap bridged from {all_detections[-2] if len(all_detections) >= 2 else 'start'}]")

                        fx1, fy1, fx2, fy2 = fbbox
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 165, 0), 2)
                        cv2.circle(frame, (fcx, fcy), 8, (255, 165, 0), -1)
                        cv2.putText(frame, f"FALLBACK: {fconf:.2f}", (fx1, fy1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                    else:
                        tracker.update_with_prediction()
                        roi_manager.expand_roi()
                else:
                    tracker.update_with_prediction()
                    roi_manager.expand_roi()

            # Draw predicted position
            if Config.SHOW_PREDICTION:
                cv2.circle(frame, (int(pred_x), int(pred_y)), 8,
                          Config.PRED_COLOR, 2)
                cv2.putText(frame, "PRED", (int(pred_x) + 10, int(pred_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, Config.PRED_COLOR, 1)

            # Draw ROI
            if Config.SHOW_ROI:
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), Config.ROI_COLOR, 1)

            # Check if tracking lost (exceeded MAX_MISSED_FRAMES)
            if not tracker.is_active:
                print(f"✗ Frame {frame_idx}: Tracking lost "
                      f"(missed {Config.MAX_MISSED_FRAMES} frames) → Reset to detection")
                state = TrackingState.DETECTION
                tracker = None
                roi_manager.reset_roi()

        # ═══════════════════════════════════════════════════════
        # DRAW TRAJECTORY (gap-bridging line across all detections)
        # ═══════════════════════════════════════════════════════
        draw_trajectory(frame, all_detections)

        # ═══════════════════════════════════════════════════════
        # VISUALIZATION & OUTPUT
        # ═══════════════════════════════════════════════════════

        frame_time = time.time() - frame_start
        process_times.append(frame_time)
        current_fps = 1.0 / frame_time if frame_time > 0 else 0

        frame_with_info = draw_info_panel(
            frame, frame_idx, state, tracker,
            roi_manager.get_current_size(), current_fps,
            len(all_detections), fallback_reacquisitions
        )

        out.write(frame_with_info)

        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            avg_fps = 1.0 / np.mean(process_times[-30:]) if process_times else 0
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames}) "
                  f"| Avg FPS: {avg_fps:.1f} | Detections: {len(all_detections)}")

    # ─── Cleanup ───
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ─── Final Statistics ───
    total_time = time.time() - start_time
    avg_fps = len(process_times) / sum(process_times) if process_times else 0

    print("=" * 60)
    print("✅ Processing Complete!")
    print("=" * 60)
    print(f"Output saved: {Config.VIDEO_OUTPUT}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Detection attempts: {detection_attempts}")
    print(f"Successful tracks: {successful_tracks}")
    print(f"Total confirmed detections: {len(all_detections)}")
    print(f"Fallback re-acquisitions: {fallback_reacquisitions}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.1f}")

    if len(all_detections) >= 2:
        start_pt = all_detections[0]
        end_pt = all_detections[-1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        total_displacement = np.sqrt(dx**2 + dy**2)
        print(f"Trajectory: {len(all_detections)} points, "
              f"displacement={total_displacement:.1f}px "
              f"({start_pt} → {end_pt})")

    print("=" * 60)


if __name__ == "__main__":
    main()
