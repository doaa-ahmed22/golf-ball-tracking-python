"""
Trajectory Analysis and Export Utilities
Tools for analyzing and exporting ball trajectory data.
"""

import csv
import json
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


class TrajectoryAnalyzer:
    """
    Analyze and export golf ball trajectory data.
    """
    
    def __init__(self, trajectory: List[Tuple[int, int]]):
        """
        Initialize analyzer with trajectory data.
        
        Args:
            trajectory: List of (x, y) positions
        """
        self.trajectory = trajectory
        self.points = np.array(trajectory)
    
    def get_total_distance(self) -> float:
        """
        Calculate total distance traveled along trajectory.
        
        Returns:
            Total distance in pixels
        """
        if len(self.points) < 2:
            return 0.0
        
        distances = np.sqrt(np.sum(np.diff(self.points, axis=0)**2, axis=1))
        return float(np.sum(distances))
    
    def get_displacement(self) -> Tuple[float, float, float]:
        """
        Calculate straight-line displacement from start to end.
        
        Returns:
            Tuple of (dx, dy, magnitude)
        """
        if len(self.points) < 2:
            return (0.0, 0.0, 0.0)
        
        start = self.points[0]
        end = self.points[-1]
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        magnitude = float(np.sqrt(dx**2 + dy**2))
        
        return (dx, dy, magnitude)
    
    def get_velocity_profile(self) -> List[float]:
        """
        Calculate instantaneous velocity (speed) at each point.
        
        Returns:
            List of velocities in pixels per frame
        """
        if len(self.points) < 2:
            return []
        
        velocities = np.sqrt(np.sum(np.diff(self.points, axis=0)**2, axis=1))
        return velocities.tolist()
    
    def get_acceleration_profile(self) -> List[float]:
        """
        Calculate instantaneous acceleration at each point.
        
        Returns:
            List of accelerations in pixels per frame²
        """
        velocities = self.get_velocity_profile()
        if len(velocities) < 2:
            return []
        
        velocities_array = np.array(velocities)
        accelerations = np.diff(velocities_array)
        return accelerations.tolist()
    
    def get_average_speed(self) -> float:
        """
        Calculate average speed across trajectory.
        
        Returns:
            Average speed in pixels per frame
        """
        velocities = self.get_velocity_profile()
        if len(velocities) == 0:
            return 0.0
        return float(np.mean(velocities))
    
    def get_max_speed(self) -> float:
        """
        Get maximum speed reached.
        
        Returns:
            Maximum speed in pixels per frame
        """
        velocities = self.get_velocity_profile()
        if len(velocities) == 0:
            return 0.0
        return float(np.max(velocities))
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Get bounding box of trajectory.
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if len(self.points) == 0:
            return (0, 0, 0, 0)
        
        min_x = int(np.min(self.points[:, 0]))
        min_y = int(np.min(self.points[:, 1]))
        max_x = int(np.max(self.points[:, 0]))
        max_y = int(np.max(self.points[:, 1]))
        
        return (min_x, min_y, max_x, max_y)
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about trajectory.
        
        Returns:
            Dictionary of statistics
        """
        dx, dy, displacement = self.get_displacement()
        
        stats = {
            'num_points': len(self.trajectory),
            'duration_frames': len(self.trajectory) - 1,
            'total_distance_px': self.get_total_distance(),
            'displacement_px': displacement,
            'displacement_x_px': dx,
            'displacement_y_px': dy,
            'average_speed_px_per_frame': self.get_average_speed(),
            'max_speed_px_per_frame': self.get_max_speed(),
            'bounding_box': self.get_bounding_box()
        }
        
        return stats
    
    def export_to_csv(self, filename: str):
        """
        Export trajectory to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'X', 'Y'])
            
            for i, (x, y) in enumerate(self.trajectory):
                writer.writerow([i, x, y])
        
        print(f"✓ Trajectory exported to {filename}")
    
    def export_to_json(self, filename: str, include_analysis: bool = True):
        """
        Export trajectory to JSON file with optional analysis.
        
        Args:
            filename: Output JSON filename
            include_analysis: Include statistical analysis in export
        """
        data = {
            'trajectory': [{'frame': i, 'x': x, 'y': y} 
                          for i, (x, y) in enumerate(self.trajectory)]
        }
        
        if include_analysis:
            data['statistics'] = self.get_statistics()
            data['velocity_profile'] = self.get_velocity_profile()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Trajectory exported to {filename}")
    
    def plot_trajectory(self, filename: str = None):
        """
        Plot trajectory as a 2D path.
        
        Args:
            filename: If provided, save plot to file. Otherwise, display.
        """
        if len(self.points) == 0:
            print("No trajectory data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot trajectory path
        plt.subplot(2, 2, 1)
        plt.plot(self.points[:, 0], self.points[:, 1], 'g-', linewidth=2, label='Trajectory')
        plt.plot(self.points[0, 0], self.points[0, 1], 'go', markersize=10, label='Start')
        plt.plot(self.points[-1, 0], self.points[-1, 1], 'ro', markersize=10, label='End')
        plt.xlabel('X Position (px)')
        plt.ylabel('Y Position (px)')
        plt.title('Ball Trajectory (2D Path)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y axis (image coordinates)
        
        # Plot X position over time
        plt.subplot(2, 2, 2)
        plt.plot(range(len(self.points)), self.points[:, 0], 'b-', linewidth=2)
        plt.xlabel('Frame')
        plt.ylabel('X Position (px)')
        plt.title('Horizontal Position Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot Y position over time
        plt.subplot(2, 2, 3)
        plt.plot(range(len(self.points)), self.points[:, 1], 'r-', linewidth=2)
        plt.xlabel('Frame')
        plt.ylabel('Y Position (px)')
        plt.title('Vertical Position Over Time')
        plt.grid(True, alpha=0.3)
        
        # Plot velocity profile
        plt.subplot(2, 2, 4)
        velocities = self.get_velocity_profile()
        plt.plot(range(len(velocities)), velocities, 'purple', linewidth=2)
        plt.xlabel('Frame')
        plt.ylabel('Speed (px/frame)')
        plt.title('Speed Profile')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150)
            print(f"✓ Plot saved to {filename}")
        else:
            plt.show()
    
    def print_statistics(self):
        """
        Print trajectory statistics to console.
        """
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("Trajectory Statistics")
        print("=" * 60)
        print(f"Number of points:      {stats['num_points']}")
        print(f"Duration (frames):     {stats['duration_frames']}")
        print(f"Total distance:        {stats['total_distance_px']:.2f} px")
        print(f"Displacement:          {stats['displacement_px']:.2f} px")
        print(f"  - Horizontal (ΔX):   {stats['displacement_x_px']:.2f} px")
        print(f"  - Vertical (ΔY):     {stats['displacement_y_px']:.2f} px")
        print(f"Average speed:         {stats['average_speed_px_per_frame']:.2f} px/frame")
        print(f"Maximum speed:         {stats['max_speed_px_per_frame']:.2f} px/frame")
        print(f"Bounding box:          {stats['bounding_box']}")
        print("=" * 60)


def compare_trajectories(
    traj1: List[Tuple[int, int]],
    traj2: List[Tuple[int, int]],
    labels: Tuple[str, str] = ("Trajectory 1", "Trajectory 2")
):
    """
    Compare two trajectories and visualize differences.
    
    Args:
        traj1: First trajectory
        traj2: Second trajectory
        labels: Labels for the two trajectories
    """
    analyzer1 = TrajectoryAnalyzer(traj1)
    analyzer2 = TrajectoryAnalyzer(traj2)
    
    stats1 = analyzer1.get_statistics()
    stats2 = analyzer2.get_statistics()
    
    print("\n" + "=" * 60)
    print("Trajectory Comparison")
    print("=" * 60)
    
    metrics = [
        ('Number of points', 'num_points'),
        ('Total distance (px)', 'total_distance_px'),
        ('Displacement (px)', 'displacement_px'),
        ('Average speed (px/frame)', 'average_speed_px_per_frame'),
        ('Maximum speed (px/frame)', 'max_speed_px_per_frame')
    ]
    
    print(f"\n{'Metric':<30} {labels[0]:<20} {labels[1]:<20}")
    print("-" * 70)
    
    for name, key in metrics:
        val1 = stats1[key]
        val2 = stats2[key]
        
        if isinstance(val1, float):
            print(f"{name:<30} {val1:<20.2f} {val2:<20.2f}")
        else:
            print(f"{name:<30} {val1:<20} {val2:<20}")
    
    print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Create sample trajectory
    print("Trajectory Analysis Utilities Demo")
    print("=" * 60)
    
    # Simulate a parabolic trajectory (like a golf ball flight)
    trajectory = []
    for i in range(100):
        x = i * 10
        y = 200 + i * 5 - 0.5 * i**2 / 10  # Parabola
        trajectory.append((int(x), int(y)))
    
    # Analyze trajectory
    analyzer = TrajectoryAnalyzer(trajectory)
    
    # Print statistics
    analyzer.print_statistics()
    
    # Export to files
    analyzer.export_to_csv("trajectory_sample.csv")
    analyzer.export_to_json("trajectory_sample.json", include_analysis=True)
    
    # Plot trajectory (uncomment to display)
    # analyzer.plot_trajectory("trajectory_plot.png")
    
    print("\n✓ Demo complete!")
