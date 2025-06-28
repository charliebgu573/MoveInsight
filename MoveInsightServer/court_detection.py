# court_detection.py
import cv2
import mediapipe as mp
import tempfile
import os
import subprocess
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("court_detection")

# Court Detection Configuration
COURT_DETECTION_BINARY = os.path.join(os.path.dirname(__file__), "court-detection", "build", "bin", "detect")

def run_court_detection(video_path: str) -> Optional[List[Tuple[float, float]]]:
    """
    Run court detection on a video and return the 6 court key points.
    Returns list of (x, y) tuples in the order specified by the README:
    1. Lower base line + left side line intersection
    2. Lower base line + right side line intersection  
    3. Upper base line + left side line intersection
    4. Upper base line + right side line intersection
    5. Left netpole + net top intersection
    6. Right netpole + net top intersection
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_output:
            output_path = tmp_output.name

        # Run court detection
        result = subprocess.run(
            [COURT_DETECTION_BINARY, video_path, output_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Court detection failed: {result.stderr}")
            return None
            
        # Parse output file
        court_points = []
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ';' in line:
                    x, y = line.split(';')
                    court_points.append((float(x), float(y)))
        
        os.unlink(output_path)
        
        if len(court_points) != 6:
            logger.error(f"Expected 6 court points, got {len(court_points)}")
            return None
            
        logger.info(f"Court detection successful: {court_points}")
        return court_points
        
    except Exception as e:
        logger.error(f"Error in court detection: {str(e)}")
        return None

def get_court_bounding_box(court_points: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    """
    Calculate bounding box (x_min, y_min, x_max, y_max) from court key points.
    """
    if not court_points or len(court_points) != 6:
        raise ValueError("Invalid court points")
    
    x_coords = [p[0] for p in court_points]
    y_coords = [p[1] for p in court_points]
    
    x_min = int(min(x_coords))
    y_min = int(min(y_coords))
    x_max = int(max(x_coords))
    y_max = int(max(y_coords))
    
    return x_min, y_min, x_max, y_max

def point_in_court_region(x: float, y: float, court_points: List[Tuple[float, float]], margin: float = 50.0) -> bool:
    """
    Check if a point (x, y) is within the court region with optional margin.
    """
    if not court_points or len(court_points) != 6:
        return False
    
    x_min, y_min, x_max, y_max = get_court_bounding_box(court_points)
    
    return (x_min - margin <= x <= x_max + margin and 
            y_min - margin <= y <= y_max + margin)

def get_player_bounding_boxes(frame: np.ndarray, court_points: List[Tuple[float, float]]) -> List[Tuple[int, int, int, int]]:
    """
    Detect people in frame and return bounding boxes for those within court region.
    Returns list of (x_min, y_min, x_max, y_max) for each detected player.
    """
    # Use MediaPipe Pose to detect people
    mp_pose = mp.solutions.pose
    player_boxes = []
    
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Get all landmark coordinates
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                landmarks.append((x, y))
            
            # Check if any landmark is within court region
            if any(point_in_court_region(x, y, court_points) for x, y in landmarks):
                # Calculate bounding box of this person
                x_coords = [x for x, y in landmarks]
                y_coords = [y for x, y in landmarks]
                
                x_min = max(0, min(x_coords) - 20)
                y_min = max(0, min(y_coords) - 20)
                x_max = min(frame.shape[1], max(x_coords) + 20)
                y_max = min(frame.shape[0], max(y_coords) + 20)
                
                player_boxes.append((x_min, y_min, x_max, y_max))
    
    return player_boxes

def calculate_court_width_crop_boundaries(video_path: str, court_points: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    """
    Calculate simplified crop boundaries using only court width and preserving full video height.
    
    Args:
        video_path: Path to video file
        court_points: List of 6 court key points from court detection
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) for cropping
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Get court bounding box for width calculation
    court_x_min, court_y_min, court_x_max, court_y_max = get_court_bounding_box(court_points)
    
    # Use court width but preserve full video height
    crop_x_min = max(0, court_x_min)
    crop_y_min = 0  # Full height from top
    crop_x_max = min(width, court_x_max)
    crop_y_max = height  # Full height to bottom
    
    logger.info(f"Court bounding box: ({court_x_min}, {court_y_min}, {court_x_max}, {court_y_max})")
    logger.info(f"Simplified crop boundaries (court width + full height): ({crop_x_min}, {crop_y_min}, {crop_x_max}, {crop_y_max})")
    logger.info(f"Court width: {court_x_max - court_x_min}px, Full video height: {height}px")
    
    return crop_x_min, crop_y_min, crop_x_max, crop_y_max

def crop_video(input_path: str, output_path: str, x_min: int, y_min: int, x_max: int, y_max: int) -> bool:
    """
    Crop video to specified boundaries using OpenCV.
    """
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate crop dimensions
        crop_width = x_max - x_min
        crop_height = y_max - y_min
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))
        
        logger.info(f"Cropping video: {crop_width}x{crop_height} from ({x_min},{y_min}) to ({x_max},{y_max})")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            out.write(cropped_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        logger.info(f"Video cropping completed: {frame_count} frames processed")
        return True
        
    except Exception as e:
        logger.error(f"Error cropping video: {str(e)}")
        return False