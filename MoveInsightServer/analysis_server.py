from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
import time
import traceback
import warnings
import cv2
import tempfile
import os

# Suppress warnings and unnecessary output
warnings.filterwarnings("ignore")
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import sys
import shutil
import zipfile
from pathlib import Path
import subprocess
import json
from ultralytics import YOLO

# Add TrackNetV3 to path
tracknet_path = os.path.join(os.path.dirname(__file__), 'TrackNetV3')
if tracknet_path not in sys.path:
    sys.path.append(tracknet_path)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis_server")

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, using numpy-based smoothing")

from overhead_clear_diagnose import evaluate_swing_rules, align_keypoints_with_interpolation

try:
    import torch
    import tracknet_test
    from tracknet_test import get_ensemble_weight, generate_inpaint_mask, predict_location
    from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
    from utils.general import get_model, write_pred_csv
    from predict import predict
    
    # TrackNet constants
    WIDTH = 640
    HEIGHT = 360
    
    TRACKNET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TrackNetV3 dependencies not available: {e}")
    TRACKNET_AVAILABLE = False
    
    # Define constants even if TrackNet is not available
    WIDTH = 640
    HEIGHT = 360

from court_detection import (
    run_court_detection,
    calculate_court_width_crop_boundaries,
    crop_video,
    point_in_court_polygon,
    debug_visualize_court_polygon
)

from court_transformation import (
    get_court_corners_from_keypoints,
    create_top_down_court_template,
    get_perspective_transformation_matrix,
    transform_point_to_top_down,
    calculate_real_world_distance,
    create_player_trail_visualization,
    TOP_DOWN_COURT_HEIGHT
)

from shuttlecock_tracking import apply_noise_filtering
from segment_hits import detect_hits, segment_video

from pose_analysis import (
    JOINT_MAP, JOINT_NAME_REMAPPING, mp_pose,
    transform_pydantic_to_numpy, transform_numpy_to_pydantic,
    get_required_joints_for_swing_analysis, get_default_swing_rules_keys
)

from models import ComparisonResultModel, JointDataItem, FrameDataItem, VideoAnalysisResponseModel, TechniqueComparisonRequestDataModel, OverheadClearAnalysis, MatchAnalysisResponseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def smooth_player_trajectory(positions: List[Tuple[float, float]], 
                           window_size: int = 15, 
                           poly_order: int = 3) -> List[Tuple[float, float]]:
    """
    Smooth player trajectory to reduce pose tracking jitter while preserving movement.
    """
    if len(positions) < 3:
        return positions
    
    x_coords = np.array([pos[0] for pos in positions])
    y_coords = np.array([pos[1] for pos in positions])
    
    actual_window_size = min(window_size, len(positions))
    if actual_window_size % 2 == 0:
        actual_window_size -= 1
    actual_window_size = max(actual_window_size, 3)
    
    actual_poly_order = min(poly_order, actual_window_size - 2)
    actual_poly_order = max(actual_poly_order, 1)
    
    if SCIPY_AVAILABLE and len(positions) >= actual_window_size:
        try:
            x_smoothed = savgol_filter(x_coords, actual_window_size, actual_poly_order)
            y_smoothed = savgol_filter(y_coords, actual_window_size, actual_poly_order)
        except Exception as e:
            logger.warning(f"Savitzky-Golay smoothing failed, using moving average: {e}")
            x_smoothed = moving_average_smooth(x_coords, actual_window_size)
            y_smoothed = moving_average_smooth(y_coords, actual_window_size)
    else:
        x_smoothed = moving_average_smooth(x_coords, actual_window_size)
        y_smoothed = moving_average_smooth(y_coords, actual_window_size)
    
    smoothed_positions = [(float(x), float(y)) for x, y in zip(x_smoothed, y_smoothed)]
    
    return smoothed_positions

def moving_average_smooth(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average smoothing to 1D data.
    """
    if len(data) < window_size:
        return data
    
    kernel = np.ones(window_size) / window_size
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    smoothed = np.convolve(padded_data, kernel, mode='same')
    return smoothed[pad_size:pad_size + len(data)]

def point_in_poly(point, poly):
    """Check if point is inside polygon using OpenCV."""
    return cv2.pointPolygonTest(poly, point, False) >= 0

def point_on_side(point, p1, p2):
    """Determine which side of line the point is on."""
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    return (x2-x1)*(y-y1) - (y2-y1)*(x-x1)

def determine_player_court_side(player_foot_position, court_points, court_transformation_matrix=None):
    """
    Determine which side of the court a player is on using court transformation.
    
    Args:
        player_foot_position: (x, y) foot position in original video coordinates
        court_points: Court detection points
        court_transformation_matrix: Optional transformation matrix for accurate positioning
    
    Returns:
        str: 'near' for near side (closer to camera), 'far' for far side (further from camera)
    """
    try:
        if court_transformation_matrix is not None:
            # Transform player position to top-down court coordinates
            transformed_pos = transform_point_to_top_down(player_foot_position, court_transformation_matrix)
            
            # In top-down view, y=0 is far baseline, y=max is near baseline
            # Midline is at y = court_height / 2
            court_height = TOP_DOWN_COURT_HEIGHT  # From court_transformation.py
            midline_y = court_height / 2
            
            if transformed_pos[1] < midline_y:
                return 'far'  # Player is on far side (upper half in top-down view)
            else:
                return 'near'  # Player is on near side (lower half in top-down view)
        else:
            # Fallback to line-based side determination
            net_p1, net_p2 = get_court_split_line(court_points)
            side_value = point_on_side(player_foot_position, net_p1, net_p2)
            
            # Based on court_dection.py logic: side >= 0 is near, side < 0 is far
            return 'near' if side_value >= 0 else 'far'
            
    except Exception as e:
        logger.warning(f"Error in player side determination: {e}")
        # Ultimate fallback - use simple line method
        net_p1, net_p2 = get_court_split_line(court_points)
        side_value = point_on_side(player_foot_position, net_p1, net_p2)
        return 'near' if side_value >= 0 else 'far'

def get_court_split_line(court_points):
    """Get the net line (split line) from court points using proper court transformation."""
    # court_points: [upper_left, lower_left, lower_right, upper_right, left_net, right_net]
    if len(court_points) >= 6:
        left_net = court_points[4]
        right_net = court_points[5]
        return (int(left_net[0]), int(left_net[1])), (int(right_net[0]), int(right_net[1]))
    else:
        # Use court transformation to get accurate midline
        try:
            # Get court corners using court_transformation functions
            court_corners = get_court_corners_from_keypoints(court_points)
            
            # Calculate midline as the line connecting midpoints of opposite sides
            # court_corners order: [top-left, top-right, bottom-right, bottom-left]
            top_left, top_right, bottom_right, bottom_left = court_corners
            
            # Calculate midpoints of left and right sides
            left_mid = (
                int((top_left[0] + bottom_left[0]) / 2),
                int((top_left[1] + bottom_left[1]) / 2)
            )
            right_mid = (
                int((top_right[0] + bottom_right[0]) / 2),
                int((top_right[1] + bottom_right[1]) / 2)
            )
            
            logger.info(f"Calculated court midline using transformation: {left_mid} to {right_mid}")
            return left_mid, right_mid
            
        except Exception as e:
            logger.warning(f"Failed to use court transformation for midline calculation: {e}")
            # Fallback: use simple middle calculation
            upper_left, lower_left, lower_right, upper_right = court_points[:4]
            left_mid = (int((upper_left[0] + lower_left[0]) // 2), int((upper_left[1] + lower_left[1]) // 2))
            right_mid = (int((upper_right[0] + lower_right[0]) // 2), int((upper_right[1] + lower_right[1]) // 2))
            return left_mid, right_mid

def create_court_homography(court_points, court_w=244, court_h=536):
    """Create homography matrix for court transformation."""
    # Use first 4 points as corners
    src_pts = np.float32(court_points[:4])
    dst_pts = np.float32([
        [0, 0],                 # upper_left
        [0, court_h-1],         # lower_left
        [court_w-1, court_h-1], # lower_right
        [court_w-1, 0]          # upper_right
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H, court_w, court_h

def map_to_court(foot_xy, H, court_w, court_h):
    """Map foot position to court coordinates."""
    pts = np.array([[foot_xy]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, H)
    return int(np.clip(mapped[0][0][0], 0, court_w-1)), int(np.clip(mapped[0][0][1], 0, court_h-1))

def smooth_point(traj, new_point, alpha=0.4):
    """Apply alpha blending smoothing to point trajectory."""
    if not traj:
        return new_point
    last = traj[-1]
    return (int(last[0] * (1-alpha) + new_point[0] * alpha),
            int(last[1] * (1-alpha) + new_point[1] * alpha))

def draw_court_with_trajectories(court_w, court_h, traj_p1, traj_p2, dist_p1, dist_p2, px_to_meter):
    """Draw court with player trajectories using court_detection.py style with distance legend below."""
    
    # Calculate legend height  
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_height = 25
    legend_height = text_height * 2 + 20  # Space for 2 lines + padding
    
    # Create expanded canvas: court + legend below
    total_height = court_h + legend_height
    full_img = np.full((total_height, court_w, 3), 255, dtype=np.uint8)  # White background
    
    # Draw court on upper portion
    court_img = full_img[:court_h, :, :]  # Reference to court area
    lw = 2
    black = (0, 0, 0)  # Black lines
    
    # Draw court boundaries and lines (EXACT from court_detection.py)
    mx, my = court_w/6.1, court_h/13.4
    cv2.rectangle(court_img, (0,0), (court_w-1, court_h-1), black, lw)
    cv2.rectangle(court_img, (int(0.46*mx),0), (court_w-1-int(0.46*mx),court_h-1), black, lw)
    cv2.line(court_img, (0,0), (court_w-1,0), black, lw)
    cv2.line(court_img, (0,court_h-1), (court_w-1,court_h-1), black, lw)
    
    # Net line (horizontal)
    cv2.line(court_img, (0, int(court_h/2)), (court_w-1, int(court_h/2)), black, lw)
    
    # Service lines
    net_y = court_h/2
    front_srv = int(net_y - 1.98*my)
    back_srv = int(net_y + 1.98*my)
    cv2.line(court_img, (0, front_srv), (court_w-1, front_srv), black, lw)
    cv2.line(court_img, (0, back_srv), (court_w-1, back_srv), black, lw)
    cv2.line(court_img, (0, int(0.76*my)), (court_w-1, int(0.76*my)), black, lw)
    cv2.line(court_img, (0, court_h-1-int(0.76*my)), (court_w-1, court_h-1-int(0.76*my)), black, lw)
    
    # Center line - FIXED: stops at service lines, not crossing net
    center_x = court_w // 2
    cv2.line(court_img, (center_x, 0), (center_x, front_srv), black, lw)  # Top half
    cv2.line(court_img, (center_x, back_srv), (center_x, court_h-1), black, lw)  # Bottom half
    
    # Draw player trajectories (EXACT from court_detection.py)
    for i in range(1, len(traj_p1)):
        cv2.line(court_img, traj_p1[i-1], traj_p1[i], (0,255,0), 2)
    for pt in traj_p1:
        cv2.circle(court_img, pt, 8, (0,255,0), -1)
        
    for i in range(1, len(traj_p2)):
        cv2.line(court_img, traj_p2[i-1], traj_p2[i], (0,0,255), 2)
    for pt in traj_p2:
        cv2.circle(court_img, pt, 8, (0,0,255), -1)
    
    # Draw distance legend BELOW the court (no more overlay on court!)
    legend_area = full_img[court_h:, :, :]  # Reference to legend area
    
    p1_text = f"Player 1: {dist_p1 * px_to_meter:.2f}m"
    p2_text = f"Player 2: {dist_p2 * px_to_meter:.2f}m"
    
    text_color_p1 = (0, 150, 0)  # Green
    text_color_p2 = (0, 0, 150)  # Red
    
    # Center the text horizontally
    (t1_w, t1_h), _ = cv2.getTextSize(p1_text, font, font_scale, font_thickness)
    (t2_w, t2_h), _ = cv2.getTextSize(p2_text, font, font_scale, font_thickness)
    
    p1_x = (court_w - t1_w) // 2
    p2_x = (court_w - t2_w) // 2
    
    # Draw legend text below court
    cv2.putText(full_img, p1_text, (p1_x, court_h + 25), font, font_scale, text_color_p1, font_thickness)
    cv2.putText(full_img, p2_text, (p2_x, court_h + 50), font, font_scale, text_color_p2, font_thickness)
    
    return full_img

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Request started: {request.method} {request.url.path} - Client: {client_host}")
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request completed: {request.method} {request.url.path} - {response.status_code} in {process_time:.2f}s")
        return response
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path} - Error: {str(e)} - Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

async def _perform_overhead_clear_evaluation(video_path: str, dominant_side: str) -> VideoAnalysisResponseModel:
    logger.info(f"Performing overhead clear evaluation for video: {video_path}, dominant side: {dominant_side}")
    
    try:
        court_points = run_court_detection(video_path)
        if court_points:
            logger.info(f"Court detected with {len(court_points)} key points")
        else:
            logger.warning("Court detection failed - will track people without court filtering")

        single_person_joint_data_raw: Dict[str, List[List[float]]] = {}
        
        from pose_analysis import get_all_expected_joint_names, detect_multiple_people_in_frame
        all_expected_joint_names = get_all_expected_joint_names()
        
        for name in all_expected_joint_names:
            single_person_joint_data_raw[name] = []

        frame_count = 0
        
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, 
                         min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                raise HTTPException(status_code=400, detail="Could not open video file.")

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                people_in_frame = detect_multiple_people_in_frame(frame, pose_estimator, court_points, num_people=1)
                
                if people_in_frame:
                    person_joints = people_in_frame[0]
                    for joint_name, coords in person_joints.items():
                        if joint_name in single_person_joint_data_raw:
                            single_person_joint_data_raw[joint_name].append(coords)
                        else:
                            logger.warning(f"Unexpected joint name '{joint_name}' for largest person")
                
                frame_count += 1
            cap.release()
        
        logger.info(f"Processed {frame_count} frames from video for single person 3D data.")

        if frame_count == 0:
            logger.warning("No frames detected in the video.")
            return VideoAnalysisResponseModel(total_frames=0, joint_data_per_frame=[], swing_analysis=None, overall_score=0.0)

        logger.info("Processing detected person data...")
        
        person_keypoints_3d_aligned = align_keypoints_with_interpolation(single_person_joint_data_raw, frame_count)
        person_joint_data_pydantic = transform_numpy_to_pydantic(person_keypoints_3d_aligned, frame_count)
        
        person_swing_analysis = None
        overall_score = 0.0
        required_keys_for_swing = get_required_joints_for_swing_analysis()
        default_swing_rules_keys = get_default_swing_rules_keys()

        keypoints_2d_for_swing: Dict[str, np.ndarray] = {}
        all_required_present_and_valid_for_swing = True

        for k in required_keys_for_swing:
            if k in person_keypoints_3d_aligned and person_keypoints_3d_aligned[k].shape[0] == frame_count and not np.all(np.isnan(person_keypoints_3d_aligned[k][:, :2])):
                keypoints_2d_for_swing[k] = person_keypoints_3d_aligned[k][:, :2]
            else:
                logger.warning(f"Keypoint '{k}' missing, has insufficient frames, or is all NaNs for swing analysis.")
                all_required_present_and_valid_for_swing = False
                break
        
        if all_required_present_and_valid_for_swing:
            try:
                person_swing_analysis = evaluate_swing_rules(keypoints_2d_for_swing, dominant_side=dominant_side)
                num_criteria = len(default_swing_rules_keys)
                correct_criteria = sum(1 for rule_key in default_swing_rules_keys if person_swing_analysis.get(rule_key, False))
                overall_score = (correct_criteria / num_criteria) * 100.0 if num_criteria > 0 else 0.0
                logger.info(f"Single person swing analysis (2D) for dominant side {dominant_side}: {person_swing_analysis}, Score: {overall_score:.2f}")
            except Exception as e:
                logger.error(f"Error in evaluate_swing_rules: {str(e)}")
                person_swing_analysis = {rule_key: False for rule_key in default_swing_rules_keys}
        else:
            logger.warning("Missing required key points for swing evaluation. Defaulting to False.")
            person_swing_analysis = {rule_key: False for rule_key in default_swing_rules_keys}

        logger.info(f"Returning {len(person_joint_data_pydantic)} frames of 3D joint data for single person.")
        return VideoAnalysisResponseModel(
            total_frames=frame_count,
            joint_data_per_frame=person_joint_data_pydantic,
            swing_analysis=person_swing_analysis,
            overall_score=overall_score
        )
    except Exception as e:
        logger.error(f"Error in _perform_overhead_clear_evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Overhead clear evaluation failed: {str(e)}")

@app.post("/analyze/overhead_clear_eval/", response_model=VideoAnalysisResponseModel)
async def overhead_clear_eval_endpoint(
    file: UploadFile = File(...),
    dominant_side: str = Form("Right")
):
    logger.info(f"Received overhead clear evaluation request: {file.filename}, dominant side: {dominant_side}")
    temp_video_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            temp_video_path = tmp.name
        
        response = await _perform_overhead_clear_evaluation(temp_video_path, dominant_side)
        return response
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                logger.info(f"Temporary video file {temp_video_path} deleted.")
            except Exception as e:
                logger.error(f"Error deleting temporary file {temp_video_path}: {e}")

@app.post("/analyze/technique_comparison/", response_model=ComparisonResultModel)
async def analyze_technique_comparison(data: TechniqueComparisonRequestDataModel):
    logger.info(f"Received 3D technique comparison request for dominant side: {data.dominant_side}, technique: {data.technique_type}")

    user_joint_data_raw_lists = transform_pydantic_to_numpy(data.user_video_frames)
    model_joint_data_raw_lists = transform_pydantic_to_numpy(data.model_video_frames)
    
    user_frame_count = len(data.user_video_frames)
    model_frame_count = len(data.model_video_frames)
    
    logger.info(f"Processing {user_frame_count} user video frames and {model_frame_count} model video frames for 3D comparison.")
    
    user_keypoints_3d_aligned = align_keypoints_with_interpolation(user_joint_data_raw_lists, user_frame_count)
    model_keypoints_3d_aligned = align_keypoints_with_interpolation(model_joint_data_raw_lists, model_frame_count)
    
    required_keys_for_swing = get_required_joints_for_swing_analysis()
    default_swing_rules_keys = get_default_swing_rules_keys()

    def get_2d_slice_for_swing(kp_3d_aligned: Dict[str, np.ndarray], frame_count_val: int, req_keys: List[str]) -> Tuple[Dict[str, np.ndarray], bool]:
        kp_2d: Dict[str, np.ndarray] = {}
        all_valid = True
        for k in req_keys:
            if k in kp_3d_aligned and kp_3d_aligned[k].shape[0] == frame_count_val and not np.all(np.isnan(kp_3d_aligned[k][:,:2])):
                kp_2d[k] = kp_3d_aligned[k][:, :2]
            else:
                logger.warning(f"Keypoint '{k}' missing or invalid for 2D slicing in comparison. Frames: {kp_3d_aligned.get(k, np.array([])).shape[0]}/{frame_count_val}")
                all_valid = False
                break
        return kp_2d, all_valid

    user_swing_details = {key: False for key in default_swing_rules_keys}
    model_swing_details = {key: False for key in default_swing_rules_keys}

    if data.technique_type == "overhead_clear":
        user_keypoints_2d, user_swing_data_valid = get_2d_slice_for_swing(user_keypoints_3d_aligned, user_frame_count, required_keys_for_swing)
        model_keypoints_2d, model_swing_data_valid = get_2d_slice_for_swing(model_keypoints_3d_aligned, model_frame_count, required_keys_for_swing)

        user_swing_details = evaluate_swing_rules(user_keypoints_2d, data.dominant_side) if user_swing_data_valid else user_swing_details.copy()
        model_swing_details = evaluate_swing_rules(model_keypoints_2d, data.dominant_side) if model_swing_data_valid else model_swing_details.copy()
    else:
        logger.warning(f"Technique comparison for '{data.technique_type}' is not yet implemented. Returning default (False) for all rules.")
    
    num_criteria = len(default_swing_rules_keys)
    
    user_correct_criteria = sum(1 for rule_key in default_swing_rules_keys if user_swing_details.get(rule_key, False))
    user_score = (user_correct_criteria / num_criteria) * 100.0 if num_criteria > 0 else 0.0

    model_correct_criteria = sum(1 for rule_key in default_swing_rules_keys if model_swing_details.get(rule_key, False))
    reference_score = (model_correct_criteria / num_criteria) * 100.0 if num_criteria > 0 else 0.0

    similarity = {}
    for rule_name in default_swing_rules_keys:
        similarity[rule_name] = (user_swing_details.get(rule_name, False) == model_swing_details.get(rule_name, False))
    
    logger.info(f"Comparison complete (using 2D for swing rules): User Score {user_score}, Reference Score {reference_score}")

    return ComparisonResultModel(
        user_score=user_score,
        reference_score=reference_score,
        similarity=similarity,
        user_details=user_swing_details,
        reference_details=model_swing_details
    )

@app.post("/analyze/match_analysis/")
async def analyze_match_analysis(
    file: UploadFile = File(...),
    num_people: int = Form(2),
    court_type: str = Form("doubles"),
    techniques: str = Form("none")
):
    """
    Comprehensive badminton match analysis with integrated video processing.
    
    Parameters:
    - techniques: Comma-separated list of analyses to run on hit segments 
                 (e.g., "overhead_clear" or "overhead_clear,smash"). Default: "none"
    """
    logger.info(f"Received match analysis request: {file.filename}, num_people: {num_people}, court_type: {court_type}, techniques: {techniques}")
    
    # Parse techniques parameter
    requested_techniques = []
    if techniques and techniques.lower() != "none":
        requested_techniques = [t.strip().lower() for t in techniques.split(",") if t.strip()]
    logger.info(f"Requested techniques for hit analysis: {requested_techniques}")
    
    temp_dir = tempfile.mkdtemp(prefix="match_analysis_")
    temp_video_path = ""
    cropped_video_path = ""
    
    try:
        video_name = Path(file.filename).stem
        temp_video_path = os.path.join(temp_dir, f"input_{file.filename}")
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"Video saved to: {temp_video_path}")
        
        logger.info("Step 1: Running court detection...")
        court_points = run_court_detection(temp_video_path)
        if not court_points:
            raise HTTPException(status_code=400, detail="Court detection failed - no court found in video")
        
        logger.info(f"Court detected with {len(court_points)} key points")
        
        logger.info("Step 2: Cropping video based on court boundaries...")
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = calculate_court_width_crop_boundaries(
            temp_video_path, court_points
        )
        
        logger.info(f"Crop boundaries: ({crop_x_min}, {crop_y_min}, {crop_x_max}, {crop_y_max})")
        
        cropped_video_path = os.path.join(temp_dir, f"{video_name}_cropped.mp4")
        success = crop_video(temp_video_path, cropped_video_path, crop_x_min, crop_y_min, crop_x_max, crop_y_max)
        if not success:
            raise HTTPException(status_code=500, detail="Video cropping failed")
        
        logger.info(f"Video cropping completed: {cropped_video_path}")
        
        logger.info("Step 3: Adjusting court coordinates for cropped video space...")
        adjusted_court_points = []
        for x, y in court_points:
            adjusted_x = x - crop_x_min
            adjusted_y = y - crop_y_min
            adjusted_court_points.append((adjusted_x, adjusted_y))
        
        logger.info(f"Adjusted court points for cropped video: {adjusted_court_points}")
        
        court_coords_path = os.path.join(temp_dir, f"{video_name}_court_coordinates.txt")
        with open(court_coords_path, 'w') as f:
            f.write("# Court Key Points Coordinates for Cropped Video (x, y)\n")
            f.write("# Format: x;y (one point per line)\n")
            f.write("# Order: Upper-left, Lower-left, Lower-right, Upper-right, Left-net, Right-net\n")
            for i, (x, y) in enumerate(adjusted_court_points):
                f.write(f"{x};{y}\n")
        
        logger.info(f"Adjusted court coordinates saved to: {court_coords_path}")
        
        court_points = adjusted_court_points
        temp_video_path = cropped_video_path
        
        logger.info("Step 4: Creating perspective transformation and YOLO model...")
        
        # Create proper court transformation matrix using court_transformation.py functions
        court_transformation_matrix = None
        try:
            # Get court corners using the proper transformation functions
            if len(court_points) >= 4:
                court_corners = get_court_corners_from_keypoints(court_points)
                court_transformation_matrix = get_perspective_transformation_matrix(court_corners)
                logger.info("Created court transformation matrix for accurate player side determination")
        except Exception as e:
            logger.warning(f"Failed to create court transformation matrix: {e}")
        
        # Create court homography for top-down transformation (keep existing functionality)
        H, court_w, court_h = create_court_homography(court_points)
        p1, p2 = get_court_split_line(court_points)
        # Ensure p1 is the left-most point and p2 is the right-most point of the net line
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        
        # Convert court points to integer coordinates for OpenCV
        court_points_int = [(int(x), int(y)) for x, y in court_points[:4]]
        court_poly = np.array(court_points_int, dtype=np.int32)
        
        # Initialize YOLO model (silent mode to reduce output)
        yolo_model = YOLO('yolov8n.pt')
        yolo_model.verbose = False
        
        logger.info("Step 5: Tracking player movements with YOLO...")
        
        # Player tracking variables (simplified approach)
        traj_p1, traj_p2 = [], []
        dist_p1, dist_p2 = 0.0, 0.0
        move_thresh = 2  # pixels
        px_to_meter = 13.4 / court_h  # conversion factor
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open input video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        movement_video_path = os.path.join(temp_dir, f"{video_name}_court_movement.mp4")
        blacked_video_path = os.path.join(temp_dir, f"{video_name}_blacked.mp4")
        
        # Video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_blacked = cv2.VideoWriter(blacked_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        all_court_frames = []
        
        # Enhanced two-player tracking with consistent identities
        player_trackers = []  # Will store consistent player tracking data
        tracked_players_data = []  # Store all frame data for analysis
        debug_frames = []  # Store frames with all debug overlays
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = yolo_model(frame)
            debug_frame = frame.copy()
            
            # Draw court boundaries and net line for visualization
            cv2.polylines(debug_frame, [court_poly], isClosed=True, color=(0,0,255), thickness=2)
            cv2.line(debug_frame, p1, p2, (255,0,0), 2)
            
            # Player identification based on court side using improved court transformation
            near_side_candidates = []  # Player 1 candidates (near side of court)
            far_side_candidates = []  # Player 2 candidates (far side of court)
            all_people_boxes = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if yolo_model.names[cls_id] == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        foot = ((x1 + x2) // 2, y2)
                        all_people_boxes.append((x1, y1, x2, y2))

                        if point_in_poly(foot, court_poly):
                            # Use improved side determination with court transformation
                            player_side = determine_player_court_side(foot, court_points, court_transformation_matrix)
                            person_data = {'bbox': (x1, y1, x2, y2), 'foot': foot, 'y2': y2, 'side': player_side}
                            
                            if player_side == 'near':
                                near_side_candidates.append(person_data)
                            else:  # player_side == 'far'
                                far_side_candidates.append(person_data)

            # Select the most prominent player from each side (one with feet lowest in the frame)
            near_side_candidates.sort(key=lambda p: p['y2'], reverse=True)
            far_side_candidates.sort(key=lambda p: p['y2'], reverse=True)

            # Player 1 is on near side (closer to camera), Player 2 is on far side (further from camera)
            player1 = near_side_candidates[0] if near_side_candidates else None
            player2 = far_side_candidates[0] if far_side_candidates else None
            
            # Log side assignments for debugging
            if player1:
                logger.debug(f"Player 1 assigned to {player1['side']} side")
            if player2:
                logger.debug(f"Player 2 assigned to {player2['side']} side")
            
            # Player identity assignment and drawing
            current_players = {}
            tracked_boxes = []
            p1_foot = p2_foot = None
            
            if player1:
                current_players['player1'] = player1
                bbox = player1['bbox']
                foot = player1['foot']
                
                # Draw player 1
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                cv2.circle(debug_frame, foot, 6, (0,180,0), -1)
                cv2.putText(debug_frame, 'P1', (foot[0]-10, foot[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                p1_foot = foot
                tracked_boxes.append(bbox)
            
            if player2:
                current_players['player2'] = player2
                bbox = player2['bbox']
                foot = player2['foot']
                
                # Draw player 2  
                cv2.rectangle(debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
                cv2.circle(debug_frame, foot, 6, (0,0,255), -1)
                cv2.putText(debug_frame, 'P2', (foot[0]-10, foot[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                p2_foot = foot
                tracked_boxes.append(bbox)
            
            # Update trajectories using court_detection.py smoothing logic
            if p1_foot is not None:
                mapped = map_to_court(p1_foot, H, court_w, court_h)
                mapped = smooth_point(traj_p1, mapped, alpha=0.4)  # court_detection.py smoothing
                if traj_p1:
                    dx = mapped[0] - traj_p1[-1][0]
                    dy = mapped[1] - traj_p1[-1][1]
                    delta = (dx**2 + dy**2)**0.5
                    if delta > move_thresh:  # court_detection.py threshold
                        dist_p1 += delta
                traj_p1.append(mapped)
                if len(traj_p1) > 5:  # court_detection.py limit
                    traj_p1.pop(0)
            
            if p2_foot is not None:
                mapped = map_to_court(p2_foot, H, court_w, court_h)
                mapped = smooth_point(traj_p2, mapped, alpha=0.4)  # court_detection.py smoothing
                if traj_p2:
                    dx = mapped[0] - traj_p2[-1][0]
                    dy = mapped[1] - traj_p2[-1][1]
                    delta = (dx**2 + dy**2)**0.5
                    if delta > move_thresh:  # court_detection.py threshold
                        dist_p2 += delta
                traj_p2.append(mapped)
                if len(traj_p2) > 5:  # court_detection.py limit
                    traj_p2.pop(0)
            
            # Store frame data for later analysis
            tracked_players_data.append(current_players)
            
            # Create blackout frame - black out all non-tracked people
            blacked_frame = frame.copy()
            for (x1, y1, x2, y2) in all_people_boxes:
                is_tracked = False
                for (tx1, ty1, tx2, ty2) in tracked_boxes:
                    # Check overlap
                    overlap_x1 = max(x1, tx1)
                    overlap_y1 = max(y1, ty1)
                    overlap_x2 = min(x2, tx2)
                    overlap_y2 = min(y2, ty2)
                    
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        detected_area = (x2 - x1) * (y2 - y1)
                        overlap_ratio = overlap_area / detected_area if detected_area > 0 else 0
                        
                        if overlap_ratio > 0.3:
                            is_tracked = True
                            break
                
                # Black out non-tracked people
                if not is_tracked:
                    margin = 10
                    x1_m = max(0, x1 - margin)
                    y1_m = max(0, y1 - margin)
                    x2_m = min(width, x2 + margin)
                    y2_m = min(height, y2 + margin)
                    blacked_frame[y1_m:y2_m, x1_m:x2_m] = [0, 0, 0]
            
            # Create court visualization frame with real-time distance display
            court_canvas = draw_court_with_trajectories(court_w, court_h, traj_p1, traj_p2, dist_p1, dist_p2, px_to_meter)
            all_court_frames.append(court_canvas)
            
            # Store debug frame and blacked frame
            debug_frames.append(debug_frame)
            out_blacked.write(blacked_frame)
            frame_idx += 1
        
        cap.release()
        out_blacked.release()
        logger.info(f"Processed {frame_idx} frames with enhanced two-player YOLO tracking")
        
        # Create movement video with proper dimensions (court + legend)
        if all_court_frames:
            # Get dimensions from first frame (includes legend area)
            first_frame = all_court_frames[0]
            video_height, video_width = first_frame.shape[:2]
            
            out_movement = cv2.VideoWriter(movement_video_path, fourcc, fps, (video_width, video_height))
            for court_frame in all_court_frames:
                out_movement.write(court_frame)
            out_movement.release()
            logger.info(f"Movement video created: {movement_video_path}")
        
        # Skip the complex MediaPipe pose analysis - just create basic data for compatibility
        
        logger.info("Step 6: Running complete shuttlecock tracking and hit analysis...")
        shuttlecock_csv_path = ""
        overhead_clear_count = 0
        overhead_clear_analyses: List[OverheadClearAnalysis] = []
        shuttlecock_trajectory_data = []
        hit_segment_paths = []  # Store paths to hit segment videos

        if TRACKNET_AVAILABLE:
            try:
                tracknet_model_path = os.path.join(os.path.dirname(__file__), "TrackNetV3", "ckpts", "TrackNet_best.pt")
                
                if os.path.exists(tracknet_model_path):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    logger.info(f"Using device for shuttlecock tracking: {device}")
                    
                    cap_shuttle = cv2.VideoCapture(blacked_video_path)
                    w_shuttle, h_shuttle = (int(cap_shuttle.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_shuttle.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    cap_shuttle.release()
                    logger.info(f"Shuttlecock tracking - Video dimensions: {w_shuttle}x{h_shuttle}")
                    
                    # NOTE: Will determine actual model output dimensions after first prediction
                    # to fix coordinate scaling issue (shuttlecock appearing left of actual position)
                    img_scaler = None  # Will be set after we know actual model output size
                    
                    tracknet_ckpt = torch.load(tracknet_model_path, map_location=device)
                    tracknet_seq_len = int(tracknet_ckpt['param_dict']['seq_len'])
                    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
                    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
                    tracknet.load_state_dict(tracknet_ckpt['model'])
                    tracknet.eval()
                    
                    tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
                                         'Img_scaler': img_scaler, 'Img_shape': (w_shuttle, h_shuttle)}
                    
                    logger.info(f"Running TrackNet prediction with weight ensemble...")
                    seq_len = tracknet_seq_len
                    eval_mode = 'weight'
                    batch_size = 16
                    
                    dataset = Video_IterableDataset(blacked_video_path, seq_len, 1, bg_mode)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
                    video_len = dataset.video_len
                    logger.info(f"Shuttlecock tracking - Video length: {video_len} frames")
                    
                    num_sample, sample_count = video_len - seq_len + 1, 0
                    buffer_size = seq_len - 1
                    batch_i = torch.arange(seq_len)
                    frame_i = torch.arange(seq_len - 1, -1, -1)
                    weight = get_ensemble_weight(seq_len, eval_mode)
                    
                    # Initialize buffer with actual model output dimensions
                    y_pred_buffer = None
                    
                    # Process shuttlecock frames (silent mode)
                    # Calculate expected steps: (video_len - seq_len + 1) / batch_size
                    expected_steps = (video_len - seq_len + 1 + batch_size - 1) // batch_size  # Ceiling division
                    
                    for step, (i, x) in enumerate(dataloader):
                        if step % 10 == 0:
                            logger.info(f"Processing shuttlecock batch {step + 1}/{expected_steps}")
                        x = x.float().to(device)
                        b_size, seq_len_curr = i.shape[0], i.shape[1]
                        
                        with torch.no_grad():
                            y_pred = tracknet(x).detach().cpu()
                        
                        # Initialize buffer with correct dimensions from first prediction
                        if y_pred_buffer is None:
                            pred_height, pred_width = y_pred.shape[-2:]
                            y_pred_buffer = torch.zeros((buffer_size, seq_len, pred_height, pred_width), dtype=torch.float32)
                            
                            # FIXED: Calculate correct img_scaler using actual model output dimensions
                            w_scaler = w_shuttle / pred_width  # Use actual model width
                            h_scaler = h_shuttle / pred_height  # Use actual model height
                            img_scaler = (w_scaler, h_scaler)
                            
                            logger.info(f"Initialized prediction buffer with dimensions: {pred_height}x{pred_width}")
                            logger.info(f"FIXED scaling factors: w_scaler={w_scaler:.3f}, h_scaler={h_scaler:.3f}")
                            logger.info(f"Video {w_shuttle}x{h_shuttle} -> Model {pred_width}x{pred_height}")
                        
                        y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
                        ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                        
                        # Use actual prediction dimensions for ensemble
                        pred_height, pred_width = y_pred.shape[-2:]
                        ensemble_y_pred = torch.empty((0, 1, pred_height, pred_width), dtype=torch.float32)
                        
                        for b in range(b_size):
                            if sample_count < buffer_size:
                                y_pred_ens = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                            else:
                                y_pred_ens = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)
                            
                            ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                            ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_ens.reshape(1, 1, pred_height, pred_width)), dim=0)
                            sample_count += 1
                            
                            if sample_count == num_sample:
                                y_zero_pad = torch.zeros((buffer_size, seq_len, pred_height, pred_width), dtype=torch.float32)
                                y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                                
                                for f in range(1, seq_len):
                                    y_pred_ens = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                                    ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_ens.reshape(1, 1, pred_height, pred_width)), dim=0)
                        
                        tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
                        for key in tmp_pred.keys():
                            tracknet_pred_dict[key].extend(tmp_pred[key])
                        
                        y_pred_buffer = y_pred_buffer[-buffer_size:]
                    
                    logger.info("Applying noise filtering to shuttlecock tracking...")
                    final_pred_dict = apply_noise_filtering(tracknet_pred_dict, fps=fps, video_width=w_shuttle, video_height=h_shuttle)
                    
                    if final_pred_dict['Frame']:
                        shuttlecock_csv_path = os.path.join(temp_dir, f"{video_name}_shuttlecock_tracking.csv")
                        write_pred_csv(final_pred_dict, shuttlecock_csv_path)
                        logger.info(f"Shuttlecock trajectory saved to: {shuttlecock_csv_path}")
                        
                        # Store shuttlecock data for debug video overlay
                        for i in range(len(final_pred_dict['Frame'])):
                            if str(final_pred_dict['Frame'][i]).isdigit():
                                frame_num = int(final_pred_dict['Frame'][i])
                                x = float(final_pred_dict['X'][i]) if final_pred_dict['X'][i] != '' else 0
                                y = float(final_pred_dict['Y'][i]) if final_pred_dict['Y'][i] != '' else 0
                                visibility = float(final_pred_dict['Visibility'][i]) if final_pred_dict['Visibility'][i] != '' else 0
                                shuttlecock_trajectory_data.append({
                                    'frame': frame_num,
                                    'x': x,
                                    'y': y,
                                    'visibility': visibility
                                })
                        
                        # Hit detection and segmentation
                        hit_frames = detect_hits(shuttlecock_csv_path)
                        logger.info(f"Detected {len(hit_frames)} hit frames: {hit_frames}")

                        # Store segment video paths for output
                        hit_segment_paths = []

                        for i, hit_frame in enumerate(hit_frames):
                            segment_prefix = os.path.join(temp_dir, f"hit_segment_{i+1}")
                            segment_video(blacked_video_path, [hit_frame], output_prefix=segment_prefix)
                            segment_output_path = f"{segment_prefix}_1.mp4"
                            
                            if os.path.exists(segment_output_path):
                                # Rename for better output naming
                                final_segment_path = os.path.join(temp_dir, f"{video_name}_hit_{i+1}_frame_{hit_frame}.mp4")
                                shutil.move(segment_output_path, final_segment_path)
                                hit_segment_paths.append(final_segment_path)
                                
                                # Run requested technique analyses on segment
                                if "overhead_clear" in requested_techniques:
                                    try:
                                        eval_result = await _perform_overhead_clear_evaluation(final_segment_path, "Right")
                                        if eval_result.overall_score is not None and eval_result.overall_score >= 40:
                                            overhead_clear_count += 1
                                            overhead_clear_analyses.append(OverheadClearAnalysis(
                                                score=eval_result.overall_score,
                                                details=eval_result.swing_analysis if eval_result.swing_analysis else {}
                                            ))
                                            logger.info(f"Hit {i+1} overhead clear score: {eval_result.overall_score:.1f}")
                                    except Exception as e:
                                        logger.error(f"Error evaluating overhead clear for segment {i+1}: {e}")
                                
                                # Add support for other techniques here in the future
                                # if "smash" in requested_techniques:
                                #     # Run smash analysis
                                # if "drop_shot" in requested_techniques:
                                #     # Run drop shot analysis
                    else:
                        logger.warning("No valid shuttlecock tracking data after noise filtering")
                else:
                    logger.warning("TrackNet model not found, skipping shuttlecock tracking")
            except Exception as e:
                logger.error(f"Error in shuttlecock tracking or hit analysis: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("TrackNetV3 dependencies not available, skipping shuttlecock tracking")
        
        logger.info("Step 7: Creating combined debug video with all overlays...")
        
        # Create debug video with all tracking overlays
        debug_video_path = os.path.join(temp_dir, f"{video_name}_debug_tracking.mp4")
        
        if debug_frames:
            # Create shuttlecock lookup for fast access (if available)
            shuttlecock_lookup = {}
            if shuttlecock_trajectory_data:
                for data in shuttlecock_trajectory_data:
                    shuttlecock_lookup[data['frame']] = data
            
            out_debug = cv2.VideoWriter(debug_video_path, fourcc, fps, (width, height))
            
            for frame_idx, debug_frame in enumerate(debug_frames):
                enhanced_debug_frame = debug_frame.copy()
                
                # Add shuttlecock tracking overlay (if available)
                if shuttlecock_lookup and frame_idx in shuttlecock_lookup:
                    shuttle_data = shuttlecock_lookup[frame_idx]
                    if shuttle_data['visibility'] > 0.5:
                        x, y = int(shuttle_data['x']), int(shuttle_data['y'])
                        # Draw shuttlecock position
                        cv2.circle(enhanced_debug_frame, (x, y), 8, (255, 255, 0), -1)  # Yellow circle
                        cv2.circle(enhanced_debug_frame, (x, y), 12, (255, 255, 255), 2)  # White border
                        cv2.putText(enhanced_debug_frame, 'Shuttlecock', (x-30, y-15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add distance information overlay
                info_text = f"Frame: {frame_idx + 1} | P1 Distance: {dist_p1*px_to_meter:.1f}m | P2 Distance: {dist_p2*px_to_meter:.1f}m"
                cv2.putText(enhanced_debug_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out_debug.write(enhanced_debug_frame)
            
            out_debug.release()
            logger.info(f"Debug video created: {debug_video_path}")
        
        # Prepare comprehensive analysis results
        analysis_results = MatchAnalysisResponseModel(
            movement_video_path=f"{video_name}_top_down_movement.mp4",
            overhead_clear_count=overhead_clear_count,
            overhead_clear_analyses=overhead_clear_analyses
        )

        # Create comprehensive analysis JSON
        analysis_json_data = {
            "match_analysis_summary": {
                "total_frames_processed": frame_idx,
                "players_tracked": 2,
                "court_detection_successful": True,
                "shuttlecock_tracking_enabled": len(shuttlecock_trajectory_data) > 0,
                "total_hits_detected": len(hit_frames) if 'hit_frames' in locals() else 0,
                "techniques_requested": requested_techniques,
                "overhead_clear_count": overhead_clear_count,
                "player_distances": {
                    "player1_total_distance_meters": dist_p1 * px_to_meter,
                    "player2_total_distance_meters": dist_p2 * px_to_meter
                }
            },
            "shuttlecock_analysis": {
                "total_trajectory_points": len(shuttlecock_trajectory_data),
                "hit_frames": hit_frames if 'hit_frames' in locals() else [],
                "csv_file": f"{video_name}_shuttlecock_tracking.csv" if shuttlecock_csv_path else None
            },
            "technique_analyses": {
                "overhead_clear": [
                    {
                        "hit_number": i + 1,
                        "score": analysis.score,
                        "details": analysis.details
                    }
                    for i, analysis in enumerate(overhead_clear_analyses)
                ] if "overhead_clear" in requested_techniques else [],
                "techniques_run": requested_techniques
            },
            "output_files": {
                "top_down_movement_video": f"{video_name}_top_down_movement.mp4",
                "debug_tracking_video": f"{video_name}_debug_tracking.mp4",
                "shuttlecock_data": f"{video_name}_shuttlecock_tracking.csv" if shuttlecock_csv_path else None,
                "court_coordinates": f"{video_name}_court_coordinates.txt",
                "hit_segment_videos": [os.path.basename(path) for path in hit_segment_paths] if hit_segment_paths else []
            }
        }
        
        analysis_json_path = os.path.join(temp_dir, f"{video_name}_analysis_results.json")
        with open(analysis_json_path, "w") as f:
            json.dump(analysis_json_data, f, indent=4)

        # Create comprehensive output ZIP file
        zip_path = os.path.join(temp_dir, f"{video_name}_match_analysis.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add movement video
            if os.path.exists(movement_video_path):
                zipf.write(movement_video_path, f"{video_name}_top_down_movement.mp4")
            
            # Add debug video
            if os.path.exists(debug_video_path):
                zipf.write(debug_video_path, f"{video_name}_debug_tracking.mp4")
            
            # Add analysis JSON
            if os.path.exists(analysis_json_path):
                zipf.write(analysis_json_path, f"{video_name}_analysis_results.json")
            
            # Add shuttlecock data
            if shuttlecock_csv_path and os.path.exists(shuttlecock_csv_path):
                zipf.write(shuttlecock_csv_path, f"{video_name}_shuttlecock_tracking.csv")
            
            # Add court coordinates
            if os.path.exists(court_coords_path):
                zipf.write(court_coords_path, f"{video_name}_court_coordinates.txt")
            
            # Add hit segment videos
            if hit_segment_paths:
                # Create a hits folder in the ZIP
                for segment_path in hit_segment_paths:
                    if os.path.exists(segment_path):
                        segment_filename = os.path.basename(segment_path)
                        zipf.write(segment_path, f"hit_segments/{segment_filename}")
                logger.info(f"Added {len(hit_segment_paths)} hit segment videos to output")
        
        logger.info(f"Comprehensive YOLO-based match analysis complete!")
        logger.info(f"Output includes:")
        logger.info(f"  - Top-down movement video with real-time distance display")
        logger.info(f"  - Debug video with FIXED shuttlecock positioning + player tracking + court overlays")
        logger.info(f"  - Analysis JSON with hit detection and technique analysis results")
        logger.info(f"  - Shuttlecock trajectory CSV data")
        logger.info(f"  - Court coordinates for reference")
        if hit_segment_paths:
            logger.info(f"  - {len(hit_segment_paths)} hit segment videos in hit_segments/ folder")
        if requested_techniques:
            logger.info(f"  - Technique analyses run: {', '.join(requested_techniques)}")
        else:
            logger.info(f"  - No technique analyses requested (use techniques parameter)")
        
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{video_name}_match_analysis.zip",
            background=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in match analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Match analysis failed: {str(e)}")
    
    finally:
        pass

@app.get("/")
async def root():
    return {"status": "MoveInsight Analysis Server is running", "version": "2.0.0 - YOLO-based Match Analysis"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MoveInsight Analysis Server with YOLO-based match analysis...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
