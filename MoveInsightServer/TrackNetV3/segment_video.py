import os
import cv2
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import argparse
from typing import List, Tuple
import matplotlib.pyplot as plt


def load_trajectory_data(csv_path: str) -> pd.DataFrame:
    """Load shuttlecock trajectory data from CSV file."""
    df = pd.read_csv(csv_path)
    # Filter out invisible frames
    df = df[df['Visibility'] == 1].reset_index(drop=True)
    return df


def calculate_velocity_and_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate velocity and acceleration vectors from position data."""
    # Calculate velocity (change in position)
    df['vx'] = df['X'].diff()
    df['vy'] = df['Y'].diff()
    df['velocity_magnitude'] = np.sqrt(df['vx']**2 + df['vy']**2)
    
    # Calculate acceleration (change in velocity)
    df['ax'] = df['vx'].diff()
    df['ay'] = df['vy'].diff()
    df['acceleration_magnitude'] = np.sqrt(df['ax']**2 + df['ay']**2)
    
    # Calculate direction change (angle between consecutive velocity vectors)
    df['direction_change'] = np.nan
    for i in range(2, len(df)):
        v1 = np.array([df.loc[i-1, 'vx'], df.loc[i-1, 'vy']])
        v2 = np.array([df.loc[i, 'vx'], df.loc[i, 'vy']])
        
        # Skip if either velocity is zero (no movement)
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            continue
            
        # Calculate angle between vectors
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = np.clip(dot_product / norms, -1, 1)
        angle = np.arccos(cos_angle)
        df.loc[i, 'direction_change'] = np.degrees(angle)
    
    return df


def detect_hits(df: pd.DataFrame, min_direction_change: float = 45, 
                min_distance: int = 15, prominence: float = 20,
                min_acceleration: float = 8.0, min_velocity: float = 3.0) -> List[int]:
    """
    Detect hits based on sudden direction changes, acceleration, and velocity.
    
    Args:
        df: DataFrame with trajectory data
        min_direction_change: Minimum direction change in degrees to consider as potential hit
        min_distance: Minimum distance between detected hits (in frames)
        prominence: Minimum prominence of peaks in direction change
        min_acceleration: Minimum acceleration magnitude required for a valid hit
        min_velocity: Minimum velocity magnitude required for a valid hit
    
    Returns:
        List of frame indices where hits are detected
    """
    # Fill NaN values with 0 for peak detection
    direction_changes = df['direction_change'].fillna(0)
    
    # Find peaks in direction change
    peaks, properties = find_peaks(direction_changes, 
                                   height=min_direction_change,
                                   distance=min_distance,
                                   prominence=prominence)
    
    # Filter peaks based on acceleration and velocity criteria
    valid_peaks = []
    for peak in peaks:
        acceleration = df.iloc[peak]['acceleration_magnitude']
        velocity = df.iloc[peak]['velocity_magnitude']
        
        # Skip if acceleration or velocity data is missing
        if pd.isna(acceleration) or pd.isna(velocity):
            continue
            
        # Check if peak meets acceleration and velocity thresholds
        if acceleration >= min_acceleration and velocity >= min_velocity:
            valid_peaks.append(peak)
    
    valid_peaks = np.array(valid_peaks)
    
    # Convert to original frame indices
    hit_frames = df.iloc[valid_peaks]['Frame'].tolist() if len(valid_peaks) > 0 else []
    
    return hit_frames, valid_peaks


def segment_video(video_path: str, hit_frames: List[int], output_dir: str, 
                  fps: float = 30.0, segment_duration: float = 1.5):
    """
    Segment video around detected hits.
    
    Args:
        video_path: Path to input video
        hit_frames: List of frame indices where hits occur
        output_dir: Directory to save segmented videos
        fps: Video frame rate
        segment_duration: Total duration of each segment in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}' for segmentation")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frames before and after hit
    half_duration_frames = int(segment_duration * fps / 2)  # 0.75 seconds = 22.5 frames
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, hit_frame in enumerate(hit_frames):
        start_frame = max(0, hit_frame - half_duration_frames)
        end_frame = min(total_frames - 1, hit_frame + half_duration_frames)
        
        # Skip if segment is too short
        if end_frame - start_frame < fps:  # Less than 1 second
            continue
            
        output_path = os.path.join(output_dir, f'hit_{i+1:03d}_frame_{hit_frame}.mp4')
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                               int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        # Extract frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        print(f"Created segment: {output_path} (frames {start_frame}-{end_frame})")
    
    cap.release()


def visualize_trajectory_and_hits(df: pd.DataFrame, hit_indices: List[int], 
                                  output_path: str = None):
    """Visualize the trajectory and detected hits."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot hit points only
    axes[0, 0].scatter(df.iloc[hit_indices]['X'], df.iloc[hit_indices]['Y'], 
                       c='red', s=100, zorder=5, label='Detected Hits')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Y Position')
    axes[0, 0].set_title('Hit Locations')
    axes[0, 0].legend()
    axes[0, 0].invert_yaxis()  # Invert Y-axis to match image coordinates
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot direction changes
    axes[0, 1].plot(df['Frame'], df['direction_change'], 'g-', linewidth=1)
    axes[0, 1].scatter(df.iloc[hit_indices]['Frame'], 
                       df.iloc[hit_indices]['direction_change'], 
                       c='red', s=100, zorder=5)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Direction Change (degrees)')
    axes[0, 1].set_title('Direction Changes Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot velocity magnitude
    axes[1, 0].plot(df['Frame'], df['velocity_magnitude'], 'purple', linewidth=1)
    axes[1, 0].scatter(df.iloc[hit_indices]['Frame'], 
                       df.iloc[hit_indices]['velocity_magnitude'], 
                       c='red', s=100, zorder=5)
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Velocity Magnitude (pixels/frame)')
    axes[1, 0].set_title('Velocity Magnitude Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot acceleration magnitude
    axes[1, 1].plot(df['Frame'], df['acceleration_magnitude'], 'orange', linewidth=1)
    axes[1, 1].scatter(df.iloc[hit_indices]['Frame'], 
                       df.iloc[hit_indices]['acceleration_magnitude'], 
                       c='red', s=100, zorder=5)
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Acceleration Magnitude (pixels/frame²)')
    axes[1, 1].set_title('Acceleration Magnitude Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory visualization saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Segment badminton video based on hit detection')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to trajectory CSV file')
    parser.add_argument('--output_dir', type=str, default='segments', help='Output directory for segments')
    parser.add_argument('--min_direction_change', type=float, default=45, 
                        help='Minimum direction change in degrees to detect hit')
    parser.add_argument('--min_distance', type=int, default=15, 
                        help='Minimum distance between hits in frames')
    parser.add_argument('--prominence', type=float, default=20, 
                        help='Minimum prominence for peak detection')
    parser.add_argument('--min_acceleration', type=float, default=8.0,
                        help='Minimum acceleration magnitude required for hit detection')
    parser.add_argument('--min_velocity', type=float, default=3.0,
                        help='Minimum velocity magnitude required for hit detection')
    parser.add_argument('--segment_duration', type=float, default=1.5, 
                        help='Duration of each segment in seconds')
    parser.add_argument('--visualize', action='store_true', 
                        help='Show trajectory visualization')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found!")
        return
    
    # Load video properties
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video_path}'")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps == 0:
        print(f"Warning: Could not determine FPS from video, using default 30 FPS")
        fps = 30.0
    
    print(f"Video FPS: {fps}")
    
    # Load and analyze trajectory data
    print("Loading trajectory data...")
    df = load_trajectory_data(args.csv_path)
    print(f"Loaded {len(df)} frames with visible shuttlecock")
    
    # Calculate motion metrics
    print("Calculating motion metrics...")
    df = calculate_velocity_and_acceleration(df)
    
    # Detect hits
    print("Detecting hits...")
    hit_frames, hit_indices = detect_hits(df, 
                                          min_direction_change=args.min_direction_change,
                                          min_distance=args.min_distance,
                                          prominence=args.prominence,
                                          min_acceleration=args.min_acceleration,
                                          min_velocity=args.min_velocity)
    
    print(f"Detected {len(hit_frames)} hits at frames: {hit_frames}")
    
    # Visualize if requested
    if args.visualize:
        visualize_trajectory_and_hits(df, hit_indices, 
                                      output_path=os.path.join(args.output_dir, 'trajectory_analysis.png'))
    
    # Segment video
    if hit_frames:
        print("Segmenting video...")
        segment_video(args.video_path, hit_frames, args.output_dir, fps, args.segment_duration)
        print(f"Video segmentation complete. Segments saved to: {args.output_dir}")
    else:
        print("No hits detected. Try adjusting the detection parameters.")


if __name__ == '__main__':
    main()