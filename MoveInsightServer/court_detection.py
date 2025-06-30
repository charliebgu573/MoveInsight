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
    Returns list of (x, y) tuples in the ACTUAL order from C++ BadmintonCourtModel::writeToFile:
    1. Upper base line + left side line intersection (upper-left corner)
    2. Lower base line + left side line intersection (lower-left corner)
    3. Lower base line + right side line intersection (lower-right corner)
    4. Upper base line + right side line intersection (upper-right corner)
    5. Left net post position
    6. Right net post position
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
    Uses simple bounding box approach for backward compatibility.
    """
    if not court_points or len(court_points) != 6:
        return False
    
    x_min, y_min, x_max, y_max = get_court_bounding_box(court_points)
    
    return (x_min - margin <= x <= x_max + margin and 
            y_min - margin <= y <= y_max + margin)

def point_in_court_polygon(x: float, y: float, court_points: List[Tuple[float, float]], margin: float = 0.0) -> bool:
    """
    Check if a point (x, y) is within the actual court polygon using the 4 corner points.
    This is more accurate than the bounding box approach as it uses the actual court shape.
    
    Args:
        x, y: Point coordinates to test
        court_points: List of 6 court key points from court detection in order:
                     [0] Upper-left, [1] Lower-left, [2] Lower-right, [3] Upper-right, [4] Left-net, [5] Right-net
        margin: Optional margin to expand the court polygon (in pixels)
    
    Returns:
        True if point is inside the court polygon, False otherwise
    """
    if not court_points or len(court_points) != 6:
        return False
    
    # Extract the 4 corner points in correct order: Upper-left, Lower-left, Lower-right, Upper-right
    # Need to reorder them to form a proper polygon: Upper-left -> Upper-right -> Lower-right -> Lower-left
    upper_left = court_points[0]    # Point 0: Upper-left corner
    lower_left = court_points[1]    # Point 1: Lower-left corner  
    lower_right = court_points[2]   # Point 2: Lower-right corner
    upper_right = court_points[3]   # Point 3: Upper-right corner
    
    # Create polygon in clockwise order for proper polygon test
    corners = [upper_left, upper_right, lower_right, lower_left]
    
    # Apply margin by expanding the polygon outward
    if margin > 0:
        # Calculate centroid of the court
        cx = sum(p[0] for p in corners) / 4
        cy = sum(p[1] for p in corners) / 4
        
        # Expand each corner point away from centroid
        expanded_corners = []
        for px, py in corners:
            # Vector from centroid to corner
            dx = px - cx
            dy = py - cy
            # Normalize and scale by margin
            length = np.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx_norm = dx / length
                dy_norm = dy / length
                # Move corner outward by margin
                expanded_x = px + dx_norm * margin
                expanded_y = py + dy_norm * margin
                expanded_corners.append((expanded_x, expanded_y))
            else:
                expanded_corners.append((px, py))
        corners = expanded_corners
    
    # Use OpenCV's point in polygon test
    contour = np.array(corners, dtype=np.float32)
    result = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
    
    # pointPolygonTest returns:
    # +1 if point is inside
    # 0 if point is on the edge  
    # -1 if point is outside
    return result >= 0

def debug_visualize_court_polygon(frame: np.ndarray, court_points: List[Tuple[float, float]], output_path: str = None) -> np.ndarray:
    """
    Create a debug visualization showing the court polygon overlay on the frame.
    
    Args:
        frame: Input frame to overlay court polygon on
        court_points: List of 6 court key points
        output_path: Optional path to save the debug image
    
    Returns:
        Frame with court polygon overlay
    """
    if not court_points or len(court_points) != 6:
        return frame
    
    debug_frame = frame.copy()
    
    # Extract corners in proper polygon order
    upper_left = court_points[0]
    lower_left = court_points[1]
    lower_right = court_points[2]
    upper_right = court_points[3]
    left_net = court_points[4]
    right_net = court_points[5]
    
    # Create polygon corners in clockwise order
    corners = [upper_left, upper_right, lower_right, lower_left]
    
    # Draw court polygon boundary
    polygon_pts = np.array(corners, dtype=np.int32)
    cv2.polylines(debug_frame, [polygon_pts], True, (0, 255, 0), 3)  # Green polygon
    
    # Draw and label corner points
    corner_labels = ["UL", "UR", "LR", "LL"]
    for i, (corner, label) in enumerate(zip(corners, corner_labels)):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(debug_frame, (x, y), 8, (0, 0, 255), -1)  # Red circles
        cv2.putText(debug_frame, f"{label}({i})", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw net points
    for i, (net_point, label) in enumerate([(left_net, "LNet"), (right_net, "RNet")]):
        x, y = int(net_point[0]), int(net_point[1])
        cv2.circle(debug_frame, (x, y), 6, (255, 0, 0), -1)  # Blue circles
        cv2.putText(debug_frame, f"{label}({i+4})", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw net line
    net_x1, net_y1 = int(left_net[0]), int(left_net[1])
    net_x2, net_y2 = int(right_net[0]), int(right_net[1])
    cv2.line(debug_frame, (net_x1, net_y1), (net_x2, net_y2), (255, 255, 0), 2)  # Yellow net line
    
    # Add title
    cv2.putText(debug_frame, "Court Detection Debug", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(debug_frame, "Green=Court, Red=Corners, Blue=Net, Yellow=NetLine", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, debug_frame)
        logger.info(f"Court polygon debug visualization saved to: {output_path}")
    
    return debug_frame

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