# Badminton Video Segmentation System

This system automatically detects badminton hits by analyzing shuttlecock trajectory data and segments the video into clips around each hit.

## How It Works

1. **Trajectory Analysis**: Analyzes the shuttlecock's position data from the CSV file
2. **Hit Detection**: Identifies sudden direction changes in the trajectory that indicate hits
3. **Video Segmentation**: Creates 1.5-second video clips (0.75s before + 0.75s after each hit)

## Usage

```bash
python segment_video.py --video_path VIDEO_FILE --csv_path CSV_FILE [OPTIONS]
```

### Required Arguments
- `--video_path`: Path to the input video file
- `--csv_path`: Path to the shuttlecock trajectory CSV file

### Optional Arguments
- `--output_dir`: Output directory for segments (default: 'segments')
- `--min_direction_change`: Minimum direction change in degrees to detect hit (default: 45)
- `--min_distance`: Minimum distance between hits in frames (default: 15)
- `--prominence`: Minimum prominence for peak detection (default: 20)
- `--segment_duration`: Duration of each segment in seconds (default: 1.5)
- `--visualize`: Show trajectory visualization with detected hits

## Example

```bash
# Basic usage
python segment_video.py --video_path test.MP4 --csv_path prediction/test_ball.csv

# With custom parameters and visualization
python segment_video.py --video_path test.MP4 --csv_path prediction/test_ball.csv --output_dir my_segments --min_direction_change 30 --visualize
```

## Output

- **Video Segments**: Individual MP4 files for each detected hit
  - Format: `hit_XXX_frame_YYY.mp4` (XXX = hit number, YYY = frame number)
  - Duration: 1.5 seconds each
- **Trajectory Visualization**: `trajectory_analysis.png` (if `--visualize` is used)
  - Shows shuttlecock trajectory, direction changes, velocity, and acceleration
  - Red dots indicate detected hits

## Tuning Parameters

### For More Sensitive Detection (detect more hits):
- Reduce `--min_direction_change` (e.g., 30 degrees)
- Reduce `--prominence` (e.g., 15)
- Reduce `--min_distance` (e.g., 10 frames)

### For Less Sensitive Detection (detect fewer hits):
- Increase `--min_direction_change` (e.g., 60 degrees)
- Increase `--prominence` (e.g., 25)
- Increase `--min_distance` (e.g., 20 frames)

## Test Results

On the provided test video:
- **Video**: 5.37 seconds, 161 frames, 30 FPS
- **Detected Hits**: 7 hits at frames [14, 38, 55, 82, 104, 123, 142]
- **Output**: 7 video segments successfully created

The system successfully identified rally exchanges and direction changes in the shuttlecock trajectory.