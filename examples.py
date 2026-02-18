"""
Example Usage Script for Golf Ball Tracking System
Demonstrates different configuration modes and use cases.
"""

import cv2
import numpy as np
from detector import BallDetector
from tracker import KalmanBallTracker, TrackingState
from roi_manager import ROIManager
from config import get_config, print_config


def example_1_basic_tracking():
    """
    Example 1: Basic tracking with default configuration.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Tracking with Default Config")
    print("=" * 60)
    
    # Use default configuration
    config = get_config("default")
    print_config(config)
    
    # Run main tracking (would call main.py here)
    print("\nRun: python main.py")


def example_2_fast_mode():
    """
    Example 2: Fast mode for real-time processing.
    """
    print("\n" + "=" * 60)
    print("Example 2: Fast Mode (Real-time Processing)")
    print("=" * 60)
    
    config = get_config("fast")
    print_config(config)
    
    print("\nThis mode is optimized for speed:")
    print("  - Smaller ROI sizes")
    print("  - Higher confidence threshold")
    print("  - Lower measurement noise")
    print("\nIdeal for: Real-time applications, low-end hardware")


def example_3_accurate_mode():
    """
    Example 3: Accurate mode for offline processing.
    """
    print("\n" + "=" * 60)
    print("Example 3: Accurate Mode (Quality over Speed)")
    print("=" * 60)
    
    config = get_config("accurate")
    print_config(config)
    
    print("\nThis mode is optimized for accuracy:")
    print("  - Larger ROI sizes")
    print("  - Lower confidence threshold")
    print("  - Higher measurement noise for smoothing")
    print("\nIdeal for: Offline analysis, high-quality output")


def example_4_robust_mode():
    """
    Example 4: Robust mode for challenging conditions.
    """
    print("\n" + "=" * 60)
    print("Example 4: Robust Mode (Handle Occlusions)")
    print("=" * 60)
    
    config = get_config("robust")
    print_config(config)
    
    print("\nThis mode is optimized for robustness:")
    print("  - More missed frames tolerance")
    print("  - Larger max distance")
    print("  - Faster ROI expansion")
    print("\nIdeal for: Challenging videos, frequent occlusions")


def example_5_custom_detector():
    """
    Example 5: Using detector module standalone.
    """
    print("\n" + "=" * 60)
    print("Example 5: Standalone Detector Usage")
    print("=" * 60)
    
    # Initialize detector
    detector = BallDetector("best.pt", confidence_threshold=0.35)
    
    # Load a test frame
    cap = cv2.VideoCapture("input_3.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Full frame detection
        detection = detector.detect_full_frame(frame)
        
        if detection:
            cx, cy, conf, bbox = detection
            print(f"\n✓ Ball detected:")
            print(f"  Center: ({cx}, {cy})")
            print(f"  Confidence: {conf:.2f}")
            print(f"  Bbox: {bbox}")
        else:
            print("\n✗ No ball detected")


def example_6_custom_tracker():
    """
    Example 6: Using tracker module standalone.
    """
    print("\n" + "=" * 60)
    print("Example 6: Standalone Tracker Usage")
    print("=" * 60)
    
    # Initialize tracker at position (500, 300)
    tracker = KalmanBallTracker(
        initial_position=(500, 300),
        process_noise=0.03,
        measurement_noise=5.0,
        max_missed_frames=15
    )
    
    print("\nTracker initialized at (500, 300)")
    
    # Simulate tracking over several frames
    measurements = [
        (510, 305),  # Frame 1
        (520, 310),  # Frame 2
        (530, 315),  # Frame 3
        None,        # Frame 4 (missed detection)
        (550, 325),  # Frame 5
    ]
    
    for i, meas in enumerate(measurements):
        # Predict
        pred_x, pred_y = tracker.predict()
        print(f"\nFrame {i+1}:")
        print(f"  Predicted: ({pred_x:.1f}, {pred_y:.1f})")
        
        if meas:
            # Update with measurement
            tracker.update(meas)
            print(f"  Measured: {meas}")
            print(f"  Status: Updated")
        else:
            # No measurement, use prediction
            tracker.update_with_prediction()
            print(f"  Measured: None")
            print(f"  Status: Using prediction")
    
    # Get trajectory
    trajectory = tracker.get_trajectory()
    print(f"\nFinal trajectory length: {len(trajectory)} points")
    print(f"Velocity: {tracker.get_velocity()}")
    print(f"Speed: {tracker.get_speed():.2f} px/frame")


def example_7_custom_roi():
    """
    Example 7: Using ROI manager standalone.
    """
    print("\n" + "=" * 60)
    print("Example 7: Standalone ROI Manager Usage")
    print("=" * 60)
    
    # Initialize ROI manager for 1920x1080 frame
    roi_manager = ROIManager(
        frame_width=1920,
        frame_height=1080,
        initial_roi_size=200,
        min_roi_size=100,
        max_roi_size=400
    )
    
    print(f"\nROI Manager initialized for 1920x1080 frame")
    print(f"Initial ROI size: {roi_manager.get_current_size()}px")
    
    # Get ROI around center position
    roi = roi_manager.get_roi((960, 540))
    print(f"\nROI at frame center: {roi}")
    
    # Simulate detection failures (expand ROI)
    print("\nSimulating 3 detection failures:")
    for i in range(3):
        roi_manager.expand_roi()
        print(f"  After failure {i+1}: {roi_manager.get_current_size()}px")
    
    # Simulate detection success (shrink ROI)
    print("\nSimulating 2 detection successes:")
    for i in range(2):
        roi_manager.shrink_roi()
        print(f"  After success {i+1}: {roi_manager.get_current_size()}px")
    
    # Adaptive ROI with velocity
    velocity = (10, 5)  # Moving right and down
    adaptive_roi = roi_manager.get_adaptive_roi((960, 540), velocity)
    print(f"\nAdaptive ROI with velocity {velocity}: {adaptive_roi}")


def example_8_export_trajectory():
    """
    Example 8: Export trajectory data to CSV.
    """
    print("\n" + "=" * 60)
    print("Example 8: Export Trajectory to CSV")
    print("=" * 60)
    
    # Simulate trajectory data
    trajectory = [(100 + i*10, 200 + i*5) for i in range(50)]
    
    # Export to CSV
    import csv
    output_file = "trajectory.csv"
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'X', 'Y'])
        for i, (x, y) in enumerate(trajectory):
            writer.writerow([i, x, y])
    
    print(f"\n✓ Trajectory exported to {output_file}")
    print(f"  Total points: {len(trajectory)}")


def main_menu():
    """
    Interactive menu for examples.
    """
    examples = {
        "1": ("Basic Tracking (Default Config)", example_1_basic_tracking),
        "2": ("Fast Mode (Real-time)", example_2_fast_mode),
        "3": ("Accurate Mode (Quality)", example_3_accurate_mode),
        "4": ("Robust Mode (Occlusions)", example_4_robust_mode),
        "5": ("Standalone Detector", example_5_custom_detector),
        "6": ("Standalone Tracker", example_6_custom_tracker),
        "7": ("Standalone ROI Manager", example_7_custom_roi),
        "8": ("Export Trajectory", example_8_export_trajectory),
    }
    
    print("\n" + "=" * 60)
    print("🏌️  Golf Ball Tracking System - Examples")
    print("=" * 60)
    print("\nAvailable Examples:")
    for key, (desc, _) in examples.items():
        print(f"  {key}. {desc}")
    print("  0. Run all examples")
    print("  q. Quit")
    print("=" * 60)
    
    choice = input("\nSelect example (0-8, q): ").strip().lower()
    
    if choice == 'q':
        print("Goodbye!")
        return
    elif choice == '0':
        for desc, func in examples.values():
            func()
    elif choice in examples:
        desc, func = examples[choice]
        func()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    # Run all examples by default
    print("\n" + "=" * 60)
    print("🏌️  Golf Ball Tracking System - Usage Examples")
    print("=" * 60)
    
    print("\nThis script demonstrates various usage patterns.")
    print("Uncomment examples below to run them.\n")
    
    # Uncomment to run specific examples:
    # example_1_basic_tracking()
    # example_2_fast_mode()
    # example_3_accurate_mode()
    # example_4_robust_mode()
    # example_5_custom_detector()
    example_6_custom_tracker()
    # example_7_custom_roi()
    # example_8_export_trajectory()
    
    # Or run interactive menu:
    # main_menu()
