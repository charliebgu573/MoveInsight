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