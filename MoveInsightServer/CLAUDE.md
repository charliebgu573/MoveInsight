# MoveInsight Analysis Server - Claude Context

## Project Overview
MoveInsight is a badminton analysis system with:
- **analysis_server.py**: FastAPI server for video analysis endpoints
- **court-detection/**: C++ binary for badminton court detection  
- **TrackNetV3/**: AI model for shuttlecock trajectory tracking
- **swing_diagnose.py**: Human pose analysis and technique evaluation

## System Architecture

### Core Components
1. **FastAPI Server** (`analysis_server.py`): Main API server with CORS middleware and request logging
2. **Court Detection**: C++ binary at `/court-detection/build/bin/detect` that outputs 6 court key points
3. **TrackNetV3**: Deep learning model for shuttlecock tracking with ensemble prediction
4. **MediaPipe Pose**: Human pose estimation for technique analysis
5. **3D Pose Processing**: Alignment and interpolation of keypoints across video frames
6. **Court Transformation**: Perspective transformation for top-down court view and real-world distance calculation

### Key Dependencies
- **PyTorch**: TrackNetV3 models (TrackNet_best.pt, InpaintNet_best.pt)
- **OpenCV**: Video processing and manipulation
- **MediaPipe**: Real-time pose estimation
- **FastAPI/Uvicorn**: Web server framework

## API Endpoints (v1.4.0 - Refactored)

### RECOMMENDED: `/analyze/match_analysis/` (POST)
**Purpose**: Comprehensive badminton match analysis combining all tracking systems

**Complete Workflow**:
1. **Court Detection & Video Cropping**: Detects badminton court and crops video to optimal boundaries
2. **Multi-Person Pose Tracking**: Tracks multiple players with intelligent court filtering
3. **Human Blackout System**: Blacks out non-tracked humans while preserving tracked players for better shuttlecock detection
4. **Shuttlecock Tracking**: Uses TrackNetV3 on blacked video for accurate trajectory tracking
5. **Combined Video Generation**: Creates comprehensive overlay video with court lines + pose skeletons + shuttlecock tracking
6. **Top-Down Movement Analysis**: Generates bird's-eye view of player movements with distance calculations
7. **Comprehensive Data Export**: Exports all analysis data in multiple formats

**Parameters**:
- `file`: Video file to process
- `num_people`: Number of people to track (default: 2)
- `court_type`: 'doubles' or 'singles' (default: 'doubles')
- `track_trail_length`: Shuttlecock trail length (default: 10)
- `eval_mode`: TrackNet evaluation mode - 'weight', 'average', or 'nonoverlap' (default: 'weight')
- `batch_size`: Batch size for inference (default: 16)
- `dominant_side`: Dominant side for swing analysis (default: 'Right')

**Returns**: ZIP file containing:
- `*_match_analysis_combined.mp4`: **Main output** - Combined video with court lines + player poses + shuttlecock tracking
- `*_top_down_movement.mp4`: Bird's-eye view of player movements on court
- `*_blacked.mp4`: Video with non-tracked humans blacked out (for debugging)
- `*_shuttlecock_tracking.csv`: Frame-by-frame shuttlecock coordinates (Frame, X, Y, Visibility)
- `*_pose_data.json`: Complete pose data for all tracked players with 3D joint coordinates
- `*_movement_data.csv`: Player movement analysis with real-world distances
- `*_court_coordinates.txt`: Court key points adjusted for cropped video coordinates

**Advanced Features**:
- **Consistent Player Tracking**: Establishes player identities in first 10 frames to eliminate jitter and ID switching
- **Intelligent Blackout**: Only blacks out non-tracked humans, preserves tracked players with proper overlap detection
- **Professional Court Lines**: Comprehensive badminton court overlay including service lines, singles lines, and net markings
- **Multi-Player Pose Skeletons**: Different colors for each player (Blue/Red) with full skeleton connections
- **Shuttlecock Trail Effects**: Fading trail visualization with position markers
- **Advanced Movement Smoothing**: Uses PlayerTracker with median filtering, moving average, and spline interpolation
- **Real-World Distance Calculation**: Accurate movement distances in meters with court transformation
- **Optimized Output**: Only includes final combined video, removes intermediate files for clean delivery

### 1. `/analyze/video_crop/` (POST)
**Purpose**: Standalone video cropping (now integrated into match_analysis)

**Returns**: ZIP file containing:
- `*_cropped.mp4`: Cropped video with optimal boundaries
- `*_court_coordinates.txt`: 6 court key points (x;y format)

### 2. `/analyze/shuttlecock_tracking/` (POST)
**Purpose**: Shuttlecock trajectory tracking with visual overlay

**Process**:
1. Load TrackNetV3 model for ball detection
2. Process video through sliding window sequences
3. Generate trajectory predictions per frame
4. Create video overlay with tracking trail and position display
5. Export coordinates to CSV

**Returns**: ZIP file containing:
- `*_tracked.mp4`: Video with shuttlecock tracking overlay
- `*_shuttlecock_tracking.csv`: Frame-by-frame coordinates (Frame, X, Y, Visibility)

**Parameters**:
- `track_trail_length`: Number of recent positions in trail (default: 10)

### 2. `/analyze/pose_tracking/` (POST)
**Purpose**: Single-person pose analysis (simplified from multi-person)

**Process**:
1. Run court detection for player filtering
2. Detect and track the **largest person** in the video using MediaPipe Pose
3. Perform 3D pose analysis and swing evaluation

**Parameters**:
- `dominant_side`: "Right" or "Left" for swing analysis (default: "Right")

**Returns**: JSON with joint data and swing analysis results for largest detected player

**Note**: This endpoint has been simplified to track only one person. For multi-person analysis, use `/analyze/match_analysis/`

### 4. `/analyze/technique_comparison/` (POST)
**Purpose**: Compare user technique against reference model

### 5. `/analyze/court_movement/` (POST)
**Purpose**: Top-down court view with player movement tracking and distance calculation

**Process**:
1. Run court detection to get 6 key points for perspective transformation
2. Create perspective transformation matrix to top-down court view
3. Track multiple players throughout video using MediaPipe Pose
4. Transform player positions to top-down coordinates
5. Calculate real-world distances moved using official badminton court dimensions
6. Generate visualization video with player trails and statistics

**Parameters**:
- `num_people`: Number of people to track (default: 2)
- `court_type`: "doubles" or "singles" court type (default: "doubles")

**Returns**: ZIP file containing:
- `*_court_movement.mp4`: Top-down court view with player movement trails
- `*_movement_data.csv`: Frame-by-frame positions and distances for all players
- `*_transform_data.txt`: Court transformation matrix and summary statistics

### 6. `/analyze/video_tracking/` (POST)
**Purpose**: Full TrackNetV3 analysis with hit detection and video segmentation

## Technical Implementation Details

### Court Detection Integration
- **Binary Path**: `/mnt/c/Users/charlie/Dev/MoveInsight/moveinsightserver/court-detection/build/bin/detect`
- **Output Format**: Text file with 6 lines of `x;y` coordinates
- **Key Points Order**: Lower-left, Lower-right, Upper-left, Upper-right, Left-net, Right-net

### TrackNetV3 Integration
- **Model Loading**: Requires both TrackNet and InpaintNet checkpoints
- **Sequence Processing**: Uses sliding window approach with configurable sequence length
- **Data Types**: Ensure `.float()` conversion for CUDA compatibility
- **Parameter Order**: `Video_IterableDataset(video_path, seq_len, sliding_step, bg_mode)`

### Court Transformation Integration
- **Badminton Court Dimensions**: Official BWF dimensions (13.4m x 6.1m for doubles, 5.18m for singles)
- **Perspective Transformation**: Uses 4 corner points from court detection for perspective transformation matrix
- **Top-Down View**: Creates standardized court template with proper markings (service lines, center line, boundaries)
- **Real-World Distance**: Calculates actual distances in meters using court scale factors
- **Multi-Person Tracking**: Filters players based on court area overlap and selects largest bounding boxes

### Video Processing Pipeline
1. **Input Validation**: Check video format and accessibility
2. **Temporary Storage**: Use `tempfile.mkdtemp()` for isolated processing
3. **Model Inference**: GPU/CPU device detection and tensor management
4. **Output Generation**: ZIP file creation with multiple artifacts
5. **Cleanup**: Automatic temporary file management

## Common Issues and Solutions

### 1. TrackNetV3 Integration Errors

**Error**: `TypeError: '<' not supported between instances of 'int' and 'str'`
**Cause**: Model parameters loaded as strings instead of integers
**Solution**: Cast sequence length to int: `tracknet_seq_len = int(tracknet_ckpt['param_dict']['seq_len'])`

**Error**: `Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same`
**Cause**: Data type mismatch between input tensors and model weights
**Solution**: Convert input to float: `tracknet(data.float())`

**Error**: `Video_IterableDataset() takes 2 positional arguments but 4 were given`
**Cause**: Incorrect parameter order in dataset constructor
**Solution**: Use correct order: `Video_IterableDataset(video_path, seq_len, sliding_step, bg_mode)`

**Error**: `forward() missing 1 required positional argument: 'm'`
**Cause**: InpaintNet requires additional mask parameter for inference
**Solution**: Skip InpaintNet for simplified tracking: `c_pred = None`

### 2. Data Processing Issues

**Error**: Frame indexing mismatches in tracking overlay
**Cause**: Prediction dictionary contains mixed data types
**Solution**: Create lookup dictionary with proper type conversion:
```python
frame_data_lookup = {}
for i in range(len(pred_dict['Frame'])):
    frame_num = int(pred_dict['Frame'][i]) if str(pred_dict['Frame'][i]).isdigit() else i
    frame_data_lookup[frame_num] = {
        'x': float(pred_dict['X'][i]) if pred_dict['X'][i] != '' else 0,
        'y': float(pred_dict['Y'][i]) if pred_dict['Y'][i] != '' else 0,
        'visibility': float(pred_dict['Visibility'][i]) if pred_dict['Visibility'][i] != '' else 0
    }
```

### 3. Court Detection Requirements
- Requires actual badminton court perspective in video
- Synthetic test videos may fail court detection
- Court must be clearly visible with proper lighting
- Multiple courts in frame may cause detection conflicts

## Development Notes

### Model Files Required
- `/ckpts/TrackNet_best.pt`: Main shuttlecock detection model
- `/ckpts/InpaintNet_best.pt`: Trajectory completion model (optional)

### Testing Approach
1. Create synthetic test videos for basic functionality
2. Use real badminton court videos for full pipeline testing
3. Monitor server logs for detailed error tracking
4. Validate ZIP file contents and sizes

### Performance Considerations
- GPU acceleration available when CUDA is detected
- Batch processing for video sequences (batch_size=16)
- Temporary file management for large video processing
- Background median generation for court detection

### Server Management
```bash
# Start server
cd /mnt/c/Users/charlie/Dev/MoveInsight/moveinsightserver
nohup python analysis_server.py > server.log 2>&1 &

# Monitor logs
tail -f server.log

# Test endpoint
curl -X POST "http://localhost:8000/analyze/shuttlecock_tracking/" \
  -F "file=@test.mp4" \
  -F "track_trail_length=5" \
  -o result.zip
```

## Recent Implementation History

### Version 1.4.0 - Major System Refactoring (Latest)

**Comprehensive Match Analysis Endpoint**:
- **Created unified `/analyze/match_analysis/` endpoint** combining video_crop → court_movement → shuttlecock_tracking workflow
- **Eliminated redundant court detection**: Now runs only once and reuses results across all analysis steps
- **Implemented intelligent human blackout system**: 
  - Detects all people in each frame using HOG + MediaPipe
  - Identifies tracked vs non-tracked players based on court overlap
  - Blacks out non-tracked humans while preserving tracked player areas
  - Handles bounding box overlaps to prevent covering tracked players
  - Fixes flashing issues with consistent detection across frames

**Enhanced Video Analysis Pipeline**:
- **Combined overlay video creation**: Court lines + multi-player pose skeletons + shuttlecock tracking in single video
- **Court line overlay system**: Draws court boundaries and net line on every frame
- **Multi-player pose visualization**: Different colors for each player (Blue/Red) with skeleton connections
- **Shuttlecock tracking integration**: Uses TrackNetV3 on blacked video with trail effects
- **Comprehensive data export**: Pose JSON, shuttlecock CSV, movement data, top-down video, court coordinates

**Simplified Pose Tracking Endpoint**:
- **Refactored `/analyze/pose_tracking/`** to track only the largest person (single bounding box)
- Removed multi-person complexity for simpler use cases
- Maintained backward compatibility with existing API response format

**System Architecture Improvements**:
- **Optimized court detection workflow**: Single detection run with coordinate adjustment for cropped video space
- **Advanced blackout algorithm**: Proper overlap detection and mask-based blackout with tracked player preservation
- **Trajectory smoothing integration**: Applied to both pose tracking and court movement analysis
- **Real-world distance calculation**: Accurate meter-based measurements using court transformation

### Version 1.4.1 - Advanced Tracking Optimizations (Latest Update)

**Consistent Player Tracking System**:
- **Eliminated player tracking jitter**: Establishes consistent player identities in first 10 frames
- **Player ID persistence**: Maintains same player assignments throughout entire video
- **Distance-based tracking**: Uses spatial proximity to maintain player consistency across frames
- **Prevents player switching**: Advanced algorithms ensure Player 1 stays Player 1 throughout analysis

**Professional Court Line Drawing**:
- **Comprehensive court overlay**: Based on BadmintonCourtModel.cpp from court detection system
- **Complete court markings**: Main boundary, singles lines, service lines, center line, and net
- **Official court proportions**: Uses BWF standard court dimensions and line spacing
- **Enhanced visibility**: Color-coded lines with proper thickness and labeling

**Advanced Movement Smoothing**:
- **PlayerTracker integration**: Uses movement_optimization.py's advanced smoothing algorithms
- **Multi-stage filtering**: Median filter → moving average → spline interpolation
- **Real-world validation**: Speed limits and net-crossing prevention
- **Professional visualization**: Smooth trajectory lines with varying thickness and fading effects

**Optimized Output Delivery**:
- **Streamlined output**: Only final combined video + top-down movement + data files
- **Removed intermediate files**: No more blacked or cropped videos in final output
- **Professional presentation**: Clean, focused delivery with only essential analysis results
- **Improved performance**: Reduced file sizes and faster processing

### Previous Implementation History

### Smart Video Cropping Enhancement
- Added court-first cropping algorithm (court boundaries as minimum)
- Implemented player extension detection beyond court boundaries  
- Added configurable shuttlecock headroom for trajectory capture
- Enhanced endpoint to return both video and court coordinates in ZIP format

### Shuttlecock Tracking Endpoint
- Created simplified tracking endpoint using TrackNetV3
- Implemented visual overlay with trajectory trail effect
- Added frame-by-frame coordinate export to CSV
- Resolved multiple integration issues with model parameters and data types

### Top-Down Court Movement Analysis
- Created comprehensive court transformation system using official BWF dimensions
- Implemented perspective transformation from video view to top-down court view
- **Fixed player position tracking algorithm**: Now uses foot/ankle positions instead of torso center for accurate court positioning
- **Fixed multi-person detection**: Replaced flawed regional sampling with HOG person detection + MediaPipe pose for each person
- Added real-world distance calculation in meters using court scale factors  
- **Advanced Movement Optimization System**:
  - **Net crossing prevention**: Players cannot illegally cross the net during gameplay
  - **Movement speed validation**: Filters unrealistic position jumps (max 8 m/s professional speed)
  - **Player identity consistency**: Maintains player assignments across frames to prevent switching
  - **Advanced smoothing**: Median filtering + moving average + spline interpolation for smooth trajectories
  - **Outlier detection**: Statistical validation to remove tracking artifacts
- **Smooth trajectory visualization**: Replaced frame-by-frame dots with professional smooth movement lines
- Generated detailed CSV output with smoothed positions and accurate distance calculations
- **Added comprehensive debugging**: Court corner visualization, transformation matrix testing, and detailed logging

### Key Lessons Learned
1. **Parameter Type Safety**: Always cast model parameters to expected types
2. **Data Type Consistency**: Ensure tensor types match model expectations
3. **Error Handling**: Comprehensive logging for debugging complex AI pipelines
4. **Modular Design**: Separate concerns between detection, tracking, and visualization
5. **Testing Strategy**: Use synthetic data for development, real data for validation
6. **Court Transformation**: Accurate perspective transformation requires proper court corner detection
7. **Court Transformation Debugging**: Always include debug visualizations and transformation matrix testing to verify homography accuracy
8. **Player Position Tracking**: Use foot/ankle positions for court movement analysis, not torso center. Torso movements don't represent actual court positioning and lead to unrealistic trajectories. Always track ground contact points for sports analysis.
9. **Multi-Person Detection**: MediaPipe Pose only detects ONE person per frame. Regional sampling leads to duplicate detections. Use proper person detection (HOG, YOLO) first, then pose estimation on each detected person's bounding box.
10. **Sports Movement Validation**: Implement domain-specific constraints (net crossing prevention, realistic speed limits) to eliminate impossible movements in sports analysis.
11. **Advanced Trajectory Smoothing**: Combine multiple filtering techniques (median filter + moving average + spline interpolation) for professional-quality movement visualization.
12. **Human Blackout for AI Accuracy**: When using multiple AI models in sequence (pose detection → shuttlecock tracking), black out irrelevant humans to improve downstream model performance while preserving tracked subjects.
13. **Workflow Optimization**: Combine multiple endpoints into comprehensive analysis pipelines to reduce redundant processing and provide complete analysis in a single request.
14. **Overlap Detection in Computer Vision**: When applying masks or blackouts, always check for overlaps with important regions and preserve them using proper intersection calculations.

## System Migration Guide

### For Users Currently Using Multiple Endpoints
**Old Workflow** (multiple API calls):
1. POST `/analyze/video_crop/` → get cropped video + court coordinates
2. POST `/analyze/court_movement/` → get top-down movement video + movement data  
3. POST `/analyze/shuttlecock_tracking/` → get shuttlecock tracking video + CSV

**New Workflow** (single API call):
1. POST `/analyze/match_analysis/` → get **all outputs** from above steps **plus** comprehensive combined video with all overlays

### Benefits of Migration
- **Single API call** instead of 3 separate calls
- **No redundant court detection** (3x performance improvement)
- **Improved shuttlecock tracking accuracy** (blackout system removes interfering humans)
- **Professional combined video** with court lines + poses + shuttlecock in one output
- **Comprehensive data package** with all analysis formats

### Backward Compatibility
All existing endpoints remain functional, but the new `/analyze/match_analysis/` endpoint is recommended for all new integrations.