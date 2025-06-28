# MoveInsight Analysis Server

Badminton analysis system with video processing, pose tracking, court detection, and shuttlecock trajectory analysis.

## System Overview

- **FastAPI Server**: Video analysis endpoints with CORS middleware
- **Court Detection**: C++ binary for badminton court key point detection
- **TrackNetV3**: AI model for shuttlecock trajectory tracking
- **MediaPipe Pose**: Human pose estimation for technique analysis
- **3D Pose Processing**: Alignment and interpolation of keypoints
- **Court Transformation**: Perspective transformation for top-down court view

## Prerequisites

### System Dependencies (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libgtk2.0-dev \
    build-essential \
    cmake \
    git \
    ffmpeg
```

### Python Environment
```bash
conda create -n moveinsight python=3.9
conda activate moveinsight
```

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download TrackNetV3 Model
```bash
# Create ckpts directory
mkdir -p ckpts

# Download and extract TrackNet model files
wget -O TrackNetV3_ckpts.zip "https://drive.google.com/uc?export=download&id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
unzip TrackNetV3_ckpts.zip -d ckpts/

# Verify files exist
ls ckpts/
# Should show: TrackNet_best.pt  InpaintNet_best.pt
```

### 3. Build Court Detection
```bash
cd court-detection
mkdir build && cd build
cmake ..
make -j$(nproc)  # Use -j2 on systems with <8GB RAM
cd ../..
```

### 4. Start Server
```bash
python analysis_server.py
```

Server will start on `http://localhost:8000` with interactive API docs at `http://localhost:8000/docs`

## API Endpoints

### 1. Video Cropping - `/analyze/video_crop/`
Smart video cropping for badminton court isolation
- **Input**: Video file + parameters (sample_frames, margin, shuttlecock_headroom)
- **Output**: ZIP with cropped video + court coordinates

### 2. Shuttlecock Tracking - `/analyze/shuttlecock_tracking/`
Shuttlecock trajectory tracking with visual overlay
- **Input**: Video file + track_trail_length parameter
- **Output**: ZIP with tracked video + CSV coordinates

### 3. Pose Tracking - `/analyze/pose_tracking/`
Multi-person 3D human pose analysis with swing evaluation
- **Input**: Video file + num_people, dominant_side parameters
- **Output**: JSON with joint data and swing analysis

### 4. Court Movement - `/analyze/court_movement/`
Top-down court view with player movement tracking
- **Input**: Video file + num_people, court_type parameters
- **Output**: ZIP with court movement video + movement data CSV

### 5. Video Tracking - `/analyze/video_tracking/`
Full TrackNetV3 analysis with hit detection and segmentation

### 6. Technique Comparison - `/analyze/technique_comparison/`
Compare user technique against reference model

## Usage Examples

### Test Server
```bash
curl -X GET "http://localhost:8000/"
```

### Upload Video for Shuttlecock Tracking
```bash
curl -X POST "http://localhost:8000/analyze/shuttlecock_tracking/" \
  -F "file=@test_video.mp4" \
  -F "track_trail_length=10" \
  -o tracking_result.zip
```

### Upload Video for Court Movement Analysis
```bash
curl -X POST "http://localhost:8000/analyze/court_movement/" \
  -F "file=@badminton_game.mp4" \
  -F "num_people=2" \
  -F "court_type=doubles" \
  -o movement_analysis.zip
```

## Connecting iOS App

When the MoveInsight iOS app prompts for a server address:

1. Find your computer's IP address:
   - **Linux/Mac**: `ip addr show` or `ifconfig`
   - **Windows**: `ipconfig`
   - **Web**: Visit [whatismyipaddress.com](https://whatismyipaddress.com/)
2. Enter the IP address in the iOS app server settings (e.g., `192.168.1.100:8000`)

### Troubleshooting Connection Issues

- **Firewall**: Ensure port 8000 is open
- **Network**: Try mobile data instead of WiFi
- **Server logs**: Check `server.log` for detailed error information
- **CORS**: Server includes CORS middleware for cross-origin requests

## Server Management

### Run in Background
```bash
nohup python analysis_server.py > server.log 2>&1 &
```

### Monitor Logs
```bash
tail -f server.log
```

### Stop Server
```bash
pkill -f analysis_server.py
```

## System Requirements

- **OS**: Linux (Ubuntu 18.04+), Windows (WSL2), macOS
- **RAM**: 8GB minimum (16GB+ recommended for large videos)
- **GPU**: CUDA-compatible GPU recommended for TrackNetV3
- **Storage**: 5GB for models and temporary processing
- **Python**: 3.8-3.10 (3.9 recommended)

## Performance Notes

- **GPU Acceleration**: Automatically detected when CUDA available
- **Processing Speed**: ~60-100 seconds per 1080p video frame
- **Memory Usage**: Peaks at 2-4GB during model inference
- **Batch Processing**: Optimized with batch_size=16 for sequences

## Troubleshooting

### Common Issues

1. **Court Detection Fails**
   - Ensure video contains clear badminton court perspective
   - Check court-detection binary exists and is executable
   - Verify video format compatibility (MP4, AVI, MOV supported)

2. **TrackNet Model Not Found**
   - Verify `ckpts/TrackNet_best.pt` and `ckpts/InpaintNet_best.pt` exist
   - Re-download model files if corrupted

3. **Memory Errors**
   - Reduce batch size in model inference
   - Use smaller video resolution
   - Ensure sufficient RAM available

4. **Build Issues**
   - Install all system dependencies
   - Use `make -j2` on memory-constrained systems
   - Check GCC version compatibility

### Video Format Issues

Convert unsupported formats:
```bash
# Convert to supported format
ffmpeg -i input.mov -c:v libx264 -pix_fmt yuv420p output.mp4

# Handle 10-bit videos
ffmpeg -i input_10bit.mp4 -c:v libx264 -pix_fmt yuv420p -crf 23 output_8bit.mp4
```

## Development

### Project Structure
```
moveinsightserver/
├── analysis_server.py          # Main FastAPI server
├── court_detection.py          # Court detection utilities
├── court_transformation.py     # Perspective transformation
├── movement_optimization.py    # Player movement analysis
├── pose_analysis.py            # MediaPipe pose processing
├── shuttlecock_tracking.py     # TrackNetV3 integration
├── swing_diagnose.py           # Technique evaluation
├── court-detection/            # C++ court detection binary
├── TrackNetV3/                 # AI model for shuttlecock tracking
└── ckpts/                      # Model checkpoints (download required)
```

### Key Dependencies
- **PyTorch**: TrackNetV3 deep learning models
- **OpenCV**: Video processing and computer vision
- **MediaPipe**: Real-time pose estimation
- **FastAPI**: Web API framework
- **NumPy/SciPy**: Numerical computing and signal processing

## Acknowledgments

- **TrackNetV3**: Enhanced shuttlecock tracking model [[paper](https://dl.acm.org/doi/10.1145/3595916.3626370)]
- **Court Detection**: Based on Farin D. et al. robust camera calibration research
- **MediaPipe**: Google's real-time pose estimation framework