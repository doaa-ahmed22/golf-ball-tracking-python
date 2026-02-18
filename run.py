"""
Quick Start Script for Golf Ball Tracking System
Easy command-line interface for running tracking with different modes.
"""

import sys
import os


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print(" " * 15 + "🏌️  GOLF BALL TRACKING SYSTEM")
    print(" " * 20 + "YOLOv8 + Kalman Filter")
    print("=" * 70)


def print_usage():
    """Print usage instructions."""
    print("\nUsage:")
    print("  python run.py [mode] [options]")
    print("\nModes:")
    print("  default    - Standard tracking (balanced)")
    print("  fast       - Fast mode (real-time, lower quality)")
    print("  accurate   - Accurate mode (slower, higher quality)")
    print("  robust     - Robust mode (handles occlusions better)")
    print("\nOptions:")
    print("  --input FILE      Input video file (default: input_3.mp4)")
    print("  --output FILE     Output video file (default: output_kalman.mp4)")
    print("  --model FILE      Model weights file (default: best.pt)")
    print("  --show-config     Display configuration and exit")
    print("  --examples        Run examples script")
    print("  --analyze         Run trajectory analysis demo")
    print("  --help            Show this help message")
    print("\nExamples:")
    print("  python run.py")
    print("  python run.py fast --input my_video.mp4")
    print("  python run.py accurate --output high_quality.mp4")
    print("  python run.py robust --model custom_model.pt")
    print("  python run.py --show-config")
    print("  python run.py --examples")
    print("=" * 70 + "\n")


def run_tracking(mode="default", input_video=None, output_video=None, model_path=None):
    """
    Run the tracking system with specified parameters.
    
    Args:
        mode: Configuration mode (default, fast, accurate, robust)
        input_video: Input video path
        output_video: Output video path
        model_path: Model weights path
    """
    print("\n" + "-" * 70)
    print(f"Starting tracking in '{mode.upper()}' mode...")
    print("-" * 70)
    
    # Import configuration
    try:
        from config import get_config, print_config
        config = get_config(mode)
    except ImportError:
        print("❌ Error: config.py not found!")
        print("Please ensure config.py is in the same directory.")
        return False
    
    # Override config with command-line arguments
    if input_video:
        config.VIDEO_INPUT = input_video
        print(f"Input video: {input_video}")
    else:
        print(f"Input video: {config.VIDEO_INPUT} (default)")
    
    if output_video:
        config.VIDEO_OUTPUT = output_video
        print(f"Output video: {output_video}")
    else:
        print(f"Output video: {config.VIDEO_OUTPUT} (default)")
    
    if model_path:
        config.MODEL_PATH = model_path
        print(f"Model: {model_path}")
    else:
        print(f"Model: {config.MODEL_PATH} (default)")
    
    print("-" * 70)
    
    # Check if files exist
    if not os.path.exists(config.MODEL_PATH):
        print(f"\n❌ Error: Model file not found: {config.MODEL_PATH}")
        print("Please ensure the model file exists.")
        return False
    
    if not os.path.exists(config.VIDEO_INPUT):
        print(f"\n❌ Error: Input video not found: {config.VIDEO_INPUT}")
        print("Please ensure the input video file exists.")
        return False
    
    # Run main tracking
    try:
        # Import and run main tracking function
        # We need to temporarily modify the Config class in main.py
        import main
        
        # Override main.Config with our config
        main.Config = config
        
        # Run tracking
        main.main()
        
        print("\n✅ Tracking completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during tracking: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_config(mode="default"):
    """Display configuration for specified mode."""
    try:
        from config import get_config, print_config
        config = get_config(mode)
        print_config(config)
    except ImportError:
        print("❌ Error: config.py not found!")


def run_examples():
    """Run examples script."""
    print("\nLaunching examples...\n")
    try:
        import examples
        examples.example_6_custom_tracker()
    except ImportError:
        print("❌ Error: examples.py not found!")


def run_analysis():
    """Run trajectory analysis demo."""
    print("\nLaunching trajectory analysis demo...\n")
    try:
        import trajectory_utils
        # Run the demo in the module
        exec(open("trajectory_utils.py").read())
    except ImportError:
        print("❌ Error: trajectory_utils.py not found!")


def main():
    """Main entry point."""
    print_banner()
    
    # Parse command-line arguments
    args = sys.argv[1:]
    
    if not args or "--help" in args or "-h" in args:
        print_usage()
        return
    
    # Handle special commands
    if "--examples" in args:
        run_examples()
        return
    
    if "--analyze" in args:
        run_analysis()
        return
    
    # Extract mode
    valid_modes = ["default", "fast", "accurate", "robust"]
    mode = "default"
    if args[0] in valid_modes:
        mode = args[0]
        args = args[1:]
    
    # Handle show-config
    if "--show-config" in args:
        show_config(mode)
        return
    
    # Parse options
    input_video = None
    output_video = None
    model_path = None
    
    i = 0
    while i < len(args):
        if args[i] == "--input" and i + 1 < len(args):
            input_video = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_video = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model_path = args[i + 1]
            i += 2
        else:
            print(f"⚠️  Warning: Unknown option: {args[i]}")
            i += 1
    
    # Run tracking
    success = run_tracking(mode, input_video, output_video, model_path)
    
    if success:
        print("\n" + "=" * 70)
        print(" " * 25 + "✅ ALL DONE!")
        print("=" * 70 + "\n")
    else:
        print("\n" + "=" * 70)
        print(" " * 20 + "❌ TRACKING FAILED")
        print("=" * 70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
