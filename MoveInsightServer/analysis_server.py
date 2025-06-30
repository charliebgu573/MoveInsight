# MoveInsightServer/analysis_server.py
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import logging
import time
import traceback
import cv2
import mediapipe as mp
import tempfile
import os
import sys
import shutil
import zipfile
from pathlib import Path
import subprocess
import json

# Add TrackNetV3 to path
tracknet_path = os.path.join(os.path.dirname(__file__), 'TrackNetV3')
if tracknet_path not in sys.path:
    sys.path.append(tracknet_path)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis_server")

# Import smoothing functions
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, using numpy-based smoothing")

# Import functions from swing_diagnose.py
from swing_diagnose import evaluate_swing_rules, align_keypoints_with_interpolation

# Import TrackNetV3 functions
try:
    import torch
    
    # Import from TrackNetV3 directory directly
    import tracknet_test
    from tracknet_test import get_ensemble_weight, generate_inpaint_mask, predict_location
    from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
    from utils.general import *  # This includes get_model and write_pred_csv
    from predict import predict
    TRACKNET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TrackNetV3 dependencies not available: {e}")
    TRACKNET_AVAILABLE = False

# --- Import Court Detection Functions ---
from court_detection import (
    run_court_detection,
    calculate_court_width_crop_boundaries,
    crop_video,
    point_in_court_polygon,
    debug_visualize_court_polygon
)

# --- Import Court Transformation Functions ---
from court_transformation import (
    get_court_corners_from_keypoints,
    create_top_down_court_template,
    get_perspective_transformation_matrix,
    transform_point_to_top_down,
    calculate_real_world_distance,
    create_player_trail_visualization
)


# --- Import Shuttlecock Tracking Functions ---
from shuttlecock_tracking import apply_noise_filtering

# --- Import Pose Analysis Functions ---
from pose_analysis import (
    JOINT_MAP, JOINT_NAME_REMAPPING, mp_pose,
    transform_pydantic_to_numpy, transform_numpy_to_pydantic,
    get_required_joints_for_swing_analysis, get_default_swing_rules_keys
)

# --- Import Pydantic Models ---
from models import (
    JointDataItem, FrameDataItem, VideoAnalysisResponseModel,
    TechniqueComparisonRequestDataModel, ComparisonResultModel
)

# --- FastAPI App Setup ---
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
    
    Args:
        positions: List of (x, y) position tuples
        window_size: Window size for smoothing (must be odd and >= poly_order + 2)
        poly_order: Polynomial order for Savitzky-Golay filter
        
    Returns:
        List of smoothed (x, y) position tuples
    """
    if len(positions) < 3:
        return positions
    
    # Convert to numpy arrays for processing
    x_coords = np.array([pos[0] for pos in positions])
    y_coords = np.array([pos[1] for pos in positions])
    
    # Adjust window size if necessary
    actual_window_size = min(window_size, len(positions))
    if actual_window_size % 2 == 0:
        actual_window_size -= 1  # Must be odd
    actual_window_size = max(actual_window_size, 3)  # Minimum size
    
    # Adjust polynomial order if necessary
    actual_poly_order = min(poly_order, actual_window_size - 2)
    actual_poly_order = max(actual_poly_order, 1)  # Minimum order
    
    if SCIPY_AVAILABLE and len(positions) >= actual_window_size:
        try:
            # Use Savitzky-Golay filter for high-quality smoothing
            x_smoothed = savgol_filter(x_coords, actual_window_size, actual_poly_order)
            y_smoothed = savgol_filter(y_coords, actual_window_size, actual_poly_order)
        except Exception as e:
            logger.warning(f"Savitzky-Golay smoothing failed, using moving average: {e}")
            # Fallback to moving average
            x_smoothed = moving_average_smooth(x_coords, actual_window_size)
            y_smoothed = moving_average_smooth(y_coords, actual_window_size)
    else:
        # Use moving average as fallback
        x_smoothed = moving_average_smooth(x_coords, actual_window_size)
        y_smoothed = moving_average_smooth(y_coords, actual_window_size)
    
    # Convert back to list of tuples
    smoothed_positions = [(float(x), float(y)) for x, y in zip(x_smoothed, y_smoothed)]
    
    return smoothed_positions

def moving_average_smooth(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average smoothing to 1D data.
    
    Args:
        data: Input data array
        window_size: Size of moving average window
        
    Returns:
        Smoothed data array
    """
    if len(data) < window_size:
        return data
    
    # Use numpy convolution for moving average
    kernel = np.ones(window_size) / window_size
    
    # Pad the data to handle edges
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')
    
    # Apply convolution
    smoothed = np.convolve(padded_data, kernel, mode='same')
    
    # Remove padding
    return smoothed[pad_size:pad_size + len(data)]



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


# --- API Endpoints ---
# Note: video_crop endpoint removed - functionality integrated into court_movement endpoint

@app.post("/analyze/pose_tracking/", response_model=VideoAnalysisResponseModel)
async def analyze_pose_tracking(
    file: UploadFile = File(...),
    dominant_side: str = Form("Right")
):
    logger.info(f"Received pose tracking request: {file.filename}, dominant side: {dominant_side} - tracking largest person only")
    temp_video_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            temp_video_path = tmp.name
        logger.info(f"Video saved temporarily to: {temp_video_path}")

        # Step 1: Run court detection to get court area for filtering
        logger.info("Running court detection for player filtering...")
        court_points = run_court_detection(temp_video_path)
        if court_points:
            logger.info(f"Court detected with {len(court_points)} key points")
        else:
            logger.warning("Court detection failed - will track people without court filtering")

        # Store raw detected points for single person: {'joint_name': [[x,y,z], ...]}
        single_person_joint_data_raw: Dict[str, List[List[float]]] = {}
        
        # Initialize for single person tracking
        from pose_analysis import get_all_expected_joint_names, detect_multiple_people_in_frame
        all_expected_joint_names = get_all_expected_joint_names()
        
        for name in all_expected_joint_names:
            single_person_joint_data_raw[name] = []

        frame_count = 0
        
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {temp_video_path}")
                raise HTTPException(status_code=400, detail="Could not open video file.")

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                # Detect only the largest person in this frame
                people_in_frame = detect_multiple_people_in_frame(frame, pose_estimator, court_points, num_people=1)
                
                # Store joint data for the detected person
                if people_in_frame:
                    # Person detected in this frame (take the first and only person)
                    person_joints = people_in_frame[0]
                    for joint_name, coords in person_joints.items():
                        if joint_name in single_person_joint_data_raw:
                            single_person_joint_data_raw[joint_name].append(coords)
                        else:
                            logger.warning(f"Unexpected joint name '{joint_name}' for largest person")
                # If no person detected in this frame, we skip adding data (will be interpolated later)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames for single person tracking")
            cap.release()
        
        logger.info(f"Processed {frame_count} frames from video for single person 3D data.")

        if frame_count == 0:
            logger.warning("No frames detected in the video.")
            return VideoAnalysisResponseModel(total_frames=0, joint_data_per_frame=[], swing_analysis=None)

        # Process the single person's data
        logger.info("Processing detected person data...")
        
        # Align and interpolate 3D data for the single person
        person_keypoints_3d_aligned = align_keypoints_with_interpolation(single_person_joint_data_raw, frame_count)
        
        # Convert aligned 3D keypoints to pydantic models for response
        person_joint_data_pydantic = transform_numpy_to_pydantic(person_keypoints_3d_aligned, frame_count)
        
        # --- Swing Analysis (using 2D slice) ---
        person_swing_analysis = None
        required_keys_for_swing = get_required_joints_for_swing_analysis()
        default_swing_rules_keys = get_default_swing_rules_keys()

        keypoints_2d_for_swing: Dict[str, np.ndarray] = {}
        all_required_present_and_valid_for_swing = True

        for k in required_keys_for_swing:
            if k in person_keypoints_3d_aligned and person_keypoints_3d_aligned[k].shape[0] == frame_count and not np.all(np.isnan(person_keypoints_3d_aligned[k][:, :2])):
                keypoints_2d_for_swing[k] = person_keypoints_3d_aligned[k][:, :2] # Take x, y
            else:
                logger.warning(f"Keypoint '{k}' missing, has insufficient frames, or is all NaNs for swing analysis.")
                all_required_present_and_valid_for_swing = False
                break
        
        if all_required_present_and_valid_for_swing:
            try:
                person_swing_analysis = evaluate_swing_rules(keypoints_2d_for_swing, dominant_side=dominant_side)
                logger.info(f"Single person swing analysis (2D) for dominant side {dominant_side}: {person_swing_analysis}")
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
            swing_analysis=person_swing_analysis
        )
    finally:
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                logger.info(f"Temporary video file {temp_video_path} deleted.")
            except Exception as e:
                logger.error(f"Error deleting temporary file {temp_video_path}: {e}")


@app.post("/analyze/technique_comparison/", response_model=ComparisonResultModel)
async def analyze_technique_comparison(data: TechniqueComparisonRequestDataModel):
    logger.info(f"Received 3D technique comparison request for dominant side: {data.dominant_side}")

    user_joint_data_raw_lists = transform_pydantic_to_numpy(data.user_video_frames)
    model_joint_data_raw_lists = transform_pydantic_to_numpy(data.model_video_frames)
    
    user_frame_count = len(data.user_video_frames)
    model_frame_count = len(data.model_video_frames)
    
    logger.info(f"Processing {user_frame_count} user video frames and {model_frame_count} model video frames for 3D comparison.")
    
    user_keypoints_3d_aligned = align_keypoints_with_interpolation(user_joint_data_raw_lists, user_frame_count)
    model_keypoints_3d_aligned = align_keypoints_with_interpolation(model_joint_data_raw_lists, model_frame_count)
    
    # --- Swing Analysis (using 2D slice) ---
    required_keys_for_swing = get_required_joints_for_swing_analysis()
    default_swing_rules_keys = get_default_swing_rules_keys()


    def get_2d_slice_for_swing(kp_3d_aligned: Dict[str, np.ndarray], frame_count_val: int, req_keys: List[str]) -> Tuple[Dict[str, np.ndarray], bool]:
        kp_2d: Dict[str, np.ndarray] = {}
        all_valid = True
        for k in req_keys:
            if k in kp_3d_aligned and kp_3d_aligned[k].shape[0] == frame_count_val and not np.all(np.isnan(kp_3d_aligned[k][:,:2])):
                kp_2d[k] = kp_3d_aligned[k][:, :2] # Slice X, Y
            else:
                logger.warning(f"Keypoint '{k}' missing or invalid for 2D slicing in comparison. Frames: {kp_3d_aligned.get(k, np.array([])).shape[0]}/{frame_count_val}")
                all_valid = False
                break
        return kp_2d, all_valid

    user_keypoints_2d, user_swing_data_valid = get_2d_slice_for_swing(user_keypoints_3d_aligned, user_frame_count, required_keys_for_swing)
    model_keypoints_2d, model_swing_data_valid = get_2d_slice_for_swing(model_keypoints_3d_aligned, model_frame_count, required_keys_for_swing)

    default_swing_details = {key: False for key in default_swing_rules_keys}

    user_swing_details = evaluate_swing_rules(user_keypoints_2d, data.dominant_side) if user_swing_data_valid else default_swing_details.copy()
    model_swing_details = evaluate_swing_rules(model_keypoints_2d, data.dominant_side) if model_swing_data_valid else default_swing_details.copy()
    
    num_criteria = len(default_swing_rules_keys) # Use the definitive list of rules
    
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
        user_details=user_swing_details, # Ensure this dict contains all keys from default_swing_rules_keys
        reference_details=model_swing_details # Ensure this dict contains all keys from default_swing_rules_keys
    )


# Note: shuttlecock_tracking endpoint removed - functionality integrated into match_analysis endpoint

@app.post("/analyze/match_analysis/")
async def analyze_match_analysis(
    file: UploadFile = File(...),
    num_people: int = Form(2),
    court_type: str = Form("doubles")
):
    """
    Comprehensive badminton match analysis with integrated video processing.
    
    This enhanced endpoint:
    1. Detects badminton court and crops video to optimal boundaries
    2. Tracks multiple players throughout the cropped video with intelligent human blackout
    3. Creates top-down court view with player movement trails
    4. Creates blacked video with non-tracked people removed (preserving tracked players)
    5. Runs shuttlecock tracking on the blacked video for optimal detection
    6. Calculates total distance moved for each player
    
    Returns a zip file containing:
    - Blacked video (non-tracked people removed, tracked players preserved)
    - Top-down court movement visualization video
    - Shuttlecock tracking video with trail overlay
    - Shuttlecock tracking CSV data
    
    Args:
        file: Video file to process
        num_people: Number of people to track (default: 2)
        court_type: 'doubles' or 'singles' (default: 'doubles')
    """
    logger.info(f"Received match analysis request: {file.filename}, num_people: {num_people}, court_type: {court_type}")
    
    temp_dir = tempfile.mkdtemp(prefix="match_analysis_")
    temp_video_path = ""
    cropped_video_path = ""
    
    try:
        # Save uploaded video
        video_name = Path(file.filename).stem
        temp_video_path = os.path.join(temp_dir, f"input_{file.filename}")
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"Video saved to: {temp_video_path}")
        
        # Step 1: Run court detection
        logger.info("Step 1: Running court detection...")
        court_points = run_court_detection(temp_video_path)
        if not court_points:
            raise HTTPException(status_code=400, detail="Court detection failed - no court found in video")
        
        logger.info(f"Court detected with {len(court_points)} key points")
        
        # Step 2: Crop video based on court detection
        logger.info("Step 2: Cropping video based on court boundaries...")
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = calculate_court_width_crop_boundaries(
            temp_video_path, court_points
        )
        
        logger.info(f"Crop boundaries: ({crop_x_min}, {crop_y_min}, {crop_x_max}, {crop_y_max})")
        
        # Crop the video
        cropped_video_path = os.path.join(temp_dir, f"{video_name}_cropped.mp4")
        success = crop_video(temp_video_path, cropped_video_path, crop_x_min, crop_y_min, crop_x_max, crop_y_max)
        if not success:
            raise HTTPException(status_code=500, detail="Video cropping failed")
        
        logger.info(f"Video cropping completed: {cropped_video_path}")
        
        # Step 3: Adjust court coordinates for cropped video space
        logger.info("Step 3: Adjusting court coordinates for cropped video space...")
        adjusted_court_points = []
        for x, y in court_points:
            adjusted_x = x - crop_x_min
            adjusted_y = y - crop_y_min
            adjusted_court_points.append((adjusted_x, adjusted_y))
        
        logger.info(f"Adjusted court points for cropped video: {adjusted_court_points}")
        
        # Save adjusted court coordinates to txt file
        court_coords_path = os.path.join(temp_dir, f"{video_name}_court_coordinates.txt")
        with open(court_coords_path, 'w') as f:
            f.write("# Court Key Points Coordinates for Cropped Video (x, y)\n")
            f.write("# Format: x;y (one point per line)\n")
            f.write("# Order: Upper-left, Lower-left, Lower-right, Upper-right, Left-net, Right-net\n")
            for i, (x, y) in enumerate(adjusted_court_points):
                f.write(f"{x};{y}\n")
        
        logger.info(f"Adjusted court coordinates saved to: {court_coords_path}")
        
        # Use adjusted court points and cropped video for the rest of the analysis
        court_points = adjusted_court_points
        temp_video_path = cropped_video_path  # Switch to using cropped video
        
        # Step 4: Create perspective transformation
        logger.info("Step 4: Creating perspective transformation...")
        court_corners = get_court_corners_from_keypoints(court_points)
        transform_matrix = get_perspective_transformation_matrix(court_corners)
        
        # Create top-down court template
        court_template = create_top_down_court_template(court_type=court_type)
        
        # DEBUG: Save a debug image showing the detected court polygon
        debug_court_img_path = os.path.join(temp_dir, f"{video_name}_debug_court_polygon.jpg")
        cap_debug = cv2.VideoCapture(temp_video_path)
        ret_debug, debug_frame = cap_debug.read()
        cap_debug.release()
        
        if ret_debug:
            # Use the new debug visualization function for accurate court polygon
            debug_visualize_court_polygon(debug_frame, court_points, debug_court_img_path)
            logger.info(f"DEBUG: Court polygon visualization saved to {debug_court_img_path}")
        
        # DEBUG: Test transformation with a few known points
        logger.info("DEBUG: Testing transformation with center court point")
        # Calculate approximate court center from corners
        center_x = sum(corner[0] for corner in court_corners) / 4
        center_y = sum(corner[1] for corner in court_corners) / 4
        center_transformed = transform_point_to_top_down((center_x, center_y), transform_matrix)
        logger.info(f"Court center ({center_x:.1f}, {center_y:.1f}) → top-down ({center_transformed[0]:.1f}, {center_transformed[1]:.1f})")
        
        # Expected center should be around (template_width/2, template_height/2)
        expected_center = (court_template.shape[1]/2, court_template.shape[0]/2)
        logger.info(f"Expected center in top-down: {expected_center}")
        
        # Step 5: Track players throughout video with human blackout
        logger.info("Step 5: Tracking player movements with human blackout...")
        
        # Initialize storage for player positions in top-down view
        player_positions_top_down = [[] for _ in range(num_people)]
        player_distances = [0.0 for _ in range(num_people)]
        
        # Initialize storage for pose data per frame for overlay video
        all_frame_pose_data = []  # List of frames, each containing pose data for all detected players
        
        # Initialize player side tracking for net crossing prevention
        player_sides = [None for _ in range(num_people)]  # 'top' or 'bottom' relative to net
        
        # Video processing setup
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open input video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # We'll create the video after smoothing, so just prepare the path
        movement_video_path = os.path.join(temp_dir, f"{video_name}_court_movement.mp4")
        
        frame_idx = 0
        
        
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, 
                         min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect multiple people in this frame
                from pose_analysis import detect_multiple_people_in_frame
                people_in_frame = detect_multiple_people_in_frame(frame, pose_estimator, court_points, num_people)
                
                # Store pose data for this frame for overlay video
                all_frame_pose_data.append(people_in_frame)
                
                # DEBUG: Log detection results
                if frame_idx % 30 == 0:  # Log every 30 frames
                    logger.info(f"Frame {frame_idx}: Detected {len(people_in_frame)} people")
                
                # Process each detected person
                current_frame_positions = []
                
                for person_idx in range(num_people):
                    if person_idx < len(people_in_frame):
                        person_joints = people_in_frame[person_idx]
                        
                        # Use foot/ankle positions to determine actual court standing position
                        foot_joints = ['LeftAnkle', 'RightAnkle', 'LeftHeel', 'RightHeel', 'LeftToe', 'RightToe']
                        valid_foot_positions = []
                        
                        for joint_name in foot_joints:
                            if joint_name in person_joints:
                                x, y, z = person_joints[joint_name]
                                valid_foot_positions.append((x, y))
                        
                        if valid_foot_positions:
                            # Calculate player's actual court position (center between feet)
                            avg_x = sum(pos[0] for pos in valid_foot_positions) / len(valid_foot_positions)
                            lowest_y = max(pos[1] for pos in valid_foot_positions)
                            ground_contact_point = (avg_x, lowest_y)
                            
                            # IMPROVED: Check if foot position is within original court polygon first
                            is_in_court = point_in_court_polygon(ground_contact_point[0], ground_contact_point[1], court_points, margin=30.0)
                            if frame_idx % 100 == 0:  # Debug logging every 100 frames
                                logger.debug(f"Frame {frame_idx}: Player {person_idx} foot pos ({ground_contact_point[0]:.1f}, {ground_contact_point[1]:.1f}) - In court: {is_in_court}")
                            
                            if is_in_court:
                                # Transform to top-down view only if within court
                                top_down_x, top_down_y = transform_point_to_top_down(ground_contact_point, transform_matrix)
                                
                                # Additional validation: ensure top-down position is reasonable  
                                court_template_height, court_template_width = court_template.shape[:2]
                                if (0 <= top_down_x <= court_template_width and 0 <= top_down_y <= court_template_height):
                                    # Net crossing prevention only - no other movement validation
                                    should_add_position = True
                                    rejection_reason = ""
                                
                                    # Calculate net position (middle of court height - Y axis)
                                    net_y_position = court_template_height / 2
                                    current_side = 'top' if top_down_y < net_y_position else 'bottom'
                                    
                                    # Check for net crossing prevention only
                                    if player_sides[person_idx] is not None and player_sides[person_idx] != current_side:
                                        # Player is trying to cross net - prevent this
                                        should_add_position = False
                                        rejection_reason = "net_crossing_prevented"
                                    
                                    if should_add_position:
                                        # Initialize or update player side
                                        if player_sides[person_idx] is None:
                                            player_sides[person_idx] = current_side
                                            logger.info(f"Player {person_idx} initialized on {current_side} side of net")
                                        else:
                                            player_sides[person_idx] = current_side
                                        
                                        player_positions_top_down[person_idx].append((top_down_x, top_down_y))
                                        current_frame_positions.append((top_down_x, top_down_y))
                                        
                                        # Calculate distance moved since last frame
                                        if len(player_positions_top_down[person_idx]) > 1:
                                            prev_pos = player_positions_top_down[person_idx][-2]
                                            curr_pos = player_positions_top_down[person_idx][-1]
                                            distance = calculate_real_world_distance(prev_pos, curr_pos)
                                            player_distances[person_idx] += distance
                                    else:
                                        current_frame_positions.append(None)
                                        if frame_idx % 50 == 0:  # Log occasionally
                                            logger.debug(f"Frame {frame_idx}: Player {person_idx} position rejected - {rejection_reason}")
                                else:
                                    current_frame_positions.append(None)
                            else:
                                # Foot position not within court polygon - skip this detection
                                current_frame_positions.append(None)
                                if frame_idx % 50 == 0:  # Log occasionally
                                    logger.debug(f"Frame {frame_idx}: Player {person_idx} foot position outside court polygon - skipped")
                        else:
                            # Fallback: use hip position
                            fallback_joints = ['LeftHip', 'RightHip']
                            valid_hip_positions = []
                            
                            for joint_name in fallback_joints:
                                if joint_name in person_joints:
                                    x, y, z = person_joints[joint_name]
                                    valid_hip_positions.append((x, y))
                            
                            if valid_hip_positions:
                                avg_x = sum(pos[0] for pos in valid_hip_positions) / len(valid_hip_positions)
                                avg_y = sum(pos[1] for pos in valid_hip_positions) / len(valid_hip_positions)
                                
                                # IMPROVED: Check if hip position is within original court polygon first
                                is_hip_in_court = point_in_court_polygon(avg_x, avg_y, court_points, margin=50.0)
                                if frame_idx % 100 == 0:  # Debug logging every 100 frames
                                    logger.debug(f"Frame {frame_idx}: Player {person_idx} hip pos ({avg_x:.1f}, {avg_y:.1f}) - In court: {is_hip_in_court}")
                                
                                if is_hip_in_court:
                                    # Transform to top-down view only if within court
                                    top_down_x, top_down_y = transform_point_to_top_down((avg_x, avg_y), transform_matrix)
                                    
                                    # Additional validation: ensure top-down position is reasonable
                                    court_template_height, court_template_width = court_template.shape[:2]
                                    if (0 <= top_down_x <= court_template_width and 0 <= top_down_y <= court_template_height):
                                        # Net crossing prevention only - no other movement validation
                                        should_add_position = True
                                        rejection_reason = ""
                                        
                                        # Calculate net position (middle of court height - Y axis)
                                        net_y_position = court_template_height / 2
                                        current_side = 'top' if top_down_y < net_y_position else 'bottom'
                                        
                                        # Check for net crossing prevention only
                                        if player_sides[person_idx] is not None and player_sides[person_idx] != current_side:
                                            # Player is trying to cross net - prevent this
                                            should_add_position = False
                                            rejection_reason = "net_crossing_prevented_hip"
                                        
                                        if should_add_position:
                                            # Initialize or update player side
                                            if player_sides[person_idx] is None:
                                                player_sides[person_idx] = current_side
                                                logger.info(f"Player {person_idx} initialized on {current_side} side of net (hip)")
                                            else:
                                                player_sides[person_idx] = current_side
                                            
                                            player_positions_top_down[person_idx].append((top_down_x, top_down_y))
                                            current_frame_positions.append((top_down_x, top_down_y))
                                            
                                            # Calculate distance moved since last frame
                                            if len(player_positions_top_down[person_idx]) > 1:
                                                prev_pos = player_positions_top_down[person_idx][-2]
                                                curr_pos = player_positions_top_down[person_idx][-1]
                                                distance = calculate_real_world_distance(prev_pos, curr_pos)
                                                player_distances[person_idx] += distance
                                        else:
                                            current_frame_positions.append(None)
                                            if frame_idx % 50 == 0:  # Log occasionally
                                                logger.debug(f"Frame {frame_idx}: Player {person_idx} hip position rejected - {rejection_reason}")
                                    else:
                                        current_frame_positions.append(None)
                                else:
                                    # Hip position not within court polygon - skip this detection
                                    current_frame_positions.append(None)
                                    if frame_idx % 50 == 0:  # Log occasionally
                                        logger.debug(f"Frame {frame_idx}: Player {person_idx} hip position outside court polygon - skipped")
                            else:
                                current_frame_positions.append(None)
                    else:
                        current_frame_positions.append(None)
                
                # Skip video creation during tracking - we'll do it after smoothing
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames for court movement")
        
        cap.release()
        
        logger.info(f"Finished tracking {frame_idx} frames with {len(all_frame_pose_data)} pose data frames")
        
        # Step 5.5: Apply trajectory smoothing to reduce pose tracking jitter
        logger.info("Applying trajectory smoothing to reduce pose tracking jitter...")
        
        # Store original positions for comparison
        original_player_positions = [positions.copy() for positions in player_positions_top_down]
        
        # Apply smoothing to each player's trajectory
        for player_idx in range(num_people):
            if len(player_positions_top_down[player_idx]) > 2:
                # Apply smoothing
                smoothed_positions = smooth_player_trajectory(
                    player_positions_top_down[player_idx],
                    window_size=15,  # Adjustable smoothing window
                    poly_order=3     # Polynomial order for curve fitting
                )
                
                # Replace original positions with smoothed ones
                player_positions_top_down[player_idx] = smoothed_positions
                
                # Recalculate distances using smoothed positions
                player_distances[player_idx] = 0.0
                for i in range(1, len(smoothed_positions)):
                    prev_pos = smoothed_positions[i-1]
                    curr_pos = smoothed_positions[i]
                    distance = calculate_real_world_distance(prev_pos, curr_pos)
                    player_distances[player_idx] += distance
                
                logger.info(f"Player {player_idx}: smoothed {len(original_player_positions[player_idx])} → {len(smoothed_positions)} positions, distance: {player_distances[player_idx]:.2f}m")
            else:
                logger.info(f"Player {player_idx}: insufficient positions for smoothing ({len(player_positions_top_down[player_idx])} points)")
        
        logger.info("Trajectory smoothing completed")
        
        # Step 5.6: Create video visualization using smoothed trajectories
        logger.info("Creating movement video with smoothed trajectories...")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(movement_video_path, fourcc, fps, 
                            (court_template.shape[1], court_template.shape[0]))
        
        # Create visualization for each frame using smoothed positions
        max_frames = max(len(positions) for positions in player_positions_top_down) if player_positions_top_down else 0
        
        for frame_num in range(max_frames):
            # Create current frame positions for visualization
            current_frame_positions_for_viz = [[] for _ in range(num_people)]
            current_frame_distances = [0.0 for _ in range(num_people)]
            
            # Build position history up to current frame for each player
            for player_idx in range(num_people):
                player_positions_up_to_frame = []
                cumulative_distance = 0.0
                
                # Get positions up to current frame
                for i in range(min(frame_num + 1, len(player_positions_top_down[player_idx]))):
                    position = player_positions_top_down[player_idx][i]
                    player_positions_up_to_frame.append(position)
                    
                    # Calculate cumulative distance up to this point
                    if i > 0:
                        prev_pos = player_positions_up_to_frame[i-1] 
                        curr_pos = player_positions_up_to_frame[i]
                        distance = calculate_real_world_distance(prev_pos, curr_pos)
                        cumulative_distance += distance
                
                current_frame_positions_for_viz[player_idx] = player_positions_up_to_frame
                current_frame_distances[player_idx] = cumulative_distance
            
            # Create visualization frame using current trajectory state with dynamic distances
            viz_frame = create_player_trail_visualization(
                court_template, 
                current_frame_positions_for_viz,
                current_distances=current_frame_distances
            )
            
            out.write(viz_frame)
            
            if frame_num % 100 == 0:
                logger.info(f"Generated video frame {frame_num+1}/{max_frames}")
        
        out.release()
        logger.info(f"Smoothed movement video created: {movement_video_path}")
        
        # Step 5.7: Create cropped video with human blackout only (no overlays)
        logger.info("Creating cropped video with human blackout (preserving tracked players)...")
        blacked_video_path = os.path.join(temp_dir, f"{video_name}_blacked.mp4")
        
        # Reopen cropped video for pose overlay
        cap_original = cv2.VideoCapture(temp_video_path)  # Now using cropped video
        if not cap_original.isOpened():
            raise HTTPException(status_code=500, detail="Cannot reopen cropped video for pose overlay")
        
        width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer for blacked video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_blacked = cv2.VideoWriter(blacked_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        # Initialize HOG descriptor for human detection
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        while cap_original.isOpened():
            ret, frame = cap_original.read()
            if not ret:
                break
            
            # Step 5.7.1: Detect all people in current frame for blackout
            logger.debug(f"Frame {frame_idx}: Detecting all people for blackout...")
            all_people_boxes, weights = hog.detectMultiScale(frame, 
                                                            winStride=(4,4),
                                                            padding=(8,8), 
                                                            scale=1.05)
            
            # Step 5.7.2: Get tracked people data for this frame
            tracked_people_boxes = []
            people_data = []
            if frame_idx < len(all_frame_pose_data):
                people_data = all_frame_pose_data[frame_idx]
                
                # Create bounding boxes for tracked people based on their pose data
                for person_idx, person_joints in enumerate(people_data):
                    if person_idx < num_people and person_joints:  # Valid tracked person
                        # Calculate bounding box from joint positions
                        joint_x_coords = []
                        joint_y_coords = []
                        
                        for joint_name, (x, y, z) in person_joints.items():
                            if x > 0 and y > 0:  # Valid coordinates
                                joint_x_coords.append(x)
                                joint_y_coords.append(y)
                        
                        if joint_x_coords and joint_y_coords:
                            margin = 30
                            x_min = max(0, min(joint_x_coords) - margin)
                            y_min = max(0, min(joint_y_coords) - margin)
                            x_max = min(width, max(joint_x_coords) + margin)
                            y_max = min(height, max(joint_y_coords) + margin)
                            tracked_people_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
            
            # Step 5.7.3: Create optimized blackout with tracked player preservation
            final_frame = frame.copy()
            
            # Create blackout mask for non-tracked people
            for (x, y, w, h) in all_people_boxes:
                is_tracked = False
                
                # Check if this detected person overlaps with any tracked person
                for (tx, ty, tw, th) in tracked_people_boxes:
                    # Calculate overlap between detected person and tracked person
                    overlap_x1 = max(x, tx)
                    overlap_y1 = max(y, ty)
                    overlap_x2 = min(x + w, tx + tw)
                    overlap_y2 = min(y + h, ty + th)
                    
                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                        detected_area = w * h
                        overlap_ratio = overlap_area / detected_area if detected_area > 0 else 0
                        
                        if overlap_ratio > 0.3:  # 30% overlap threshold
                            is_tracked = True
                            break
                
                # If not tracked, black out this person's area
                if not is_tracked:
                    # Expand blackout area slightly to ensure full coverage
                    margin = 10
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(width, x + w + margin)
                    y2 = min(height, y + h + margin)
                    
                    # Set this area to black
                    final_frame[y1:y2, x1:x2] = [0, 0, 0]
                    
                    logger.debug(f"Frame {frame_idx}: Blacked out non-tracked person at ({x}, {y}, {w}, {h})")
            
            # Step 5.7.4: Preserve tracked player areas (override any blackout)
            # This ensures tracked players are always visible even if they overlap with detected non-tracked people
            for (tx, ty, tw, th) in tracked_people_boxes:
                # Restore original frame content for tracked player areas
                tx1 = max(0, int(tx))
                ty1 = max(0, int(ty))
                tx2 = min(width, int(tx + tw))
                ty2 = min(height, int(ty + th))
                
                # Copy original frame content back to preserve tracked player
                final_frame[ty1:ty2, tx1:tx2] = frame[ty1:ty2, tx1:tx2]
            
            out_blacked.write(final_frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Generated pose overlay frame {frame_idx}")
        
        cap_original.release()
        out_blacked.release()
        logger.info(f"Blacked video created: {blacked_video_path}")
        
        # Step 5.8: Run shuttlecock tracking on blacked video
        logger.info("Step 5.8: Running shuttlecock tracking on blacked video...")
        shuttlecock_tracked_video_path = ""
        shuttlecock_csv_path = ""
        
        if TRACKNET_AVAILABLE:
            try:
                # Setup model paths
                tracknet_model_path = os.path.join(os.path.dirname(__file__), "TrackNetV3", "ckpts", "TrackNet_best.pt")
                inpaintnet_model_path = os.path.join(os.path.dirname(__file__), "TrackNetV3", "ckpts", "InpaintNet_best.pt")
                
                if os.path.exists(tracknet_model_path):
                    # Setup device
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    logger.info(f"Using device for shuttlecock tracking: {device}")
                    
                    # Get video properties for image scaling
                    cap_shuttle = cv2.VideoCapture(blacked_video_path)
                    w_shuttle, h_shuttle = (int(cap_shuttle.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_shuttle.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                    cap_shuttle.release()
                    w_scaler, h_scaler = w_shuttle / WIDTH, h_shuttle / HEIGHT
                    img_scaler = (w_scaler, h_scaler)
                    logger.info(f"Shuttlecock tracking - Video dimensions: {w_shuttle}x{h_shuttle}, scaling: {img_scaler}")
                    
                    # Load TrackNet model
                    tracknet_ckpt = torch.load(tracknet_model_path, map_location=device)
                    tracknet_seq_len = int(tracknet_ckpt['param_dict']['seq_len'])
                    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
                    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
                    tracknet.load_state_dict(tracknet_ckpt['model'])
                    tracknet.eval()
                    
                    # Initialize prediction dictionary
                    tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
                                         'Img_scaler': img_scaler, 'Img_shape': (w_shuttle, h_shuttle)}
                    
                    # Run TrackNet prediction with temporal ensemble (weight mode)
                    logger.info(f"Running TrackNet prediction with weight ensemble...")
                    seq_len = tracknet_seq_len
                    eval_mode = 'weight'  # Use weight mode for best results
                    batch_size = 16
                    
                    # Temporal ensemble (weight mode)
                    dataset = Video_IterableDataset(blacked_video_path, seq_len, 1, bg_mode)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
                    video_len = dataset.video_len
                    logger.info(f"Shuttlecock tracking - Video length: {video_len} frames")
                    
                    # Initialize ensemble parameters
                    num_sample, sample_count = video_len - seq_len + 1, 0
                    buffer_size = seq_len - 1
                    batch_i = torch.arange(seq_len)
                    frame_i = torch.arange(seq_len - 1, -1, -1)
                    y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    weight = get_ensemble_weight(seq_len, eval_mode)
                    
                    from tqdm import tqdm
                    for step, (i, x) in enumerate(tqdm(dataloader, desc="Processing shuttlecock frames")):
                        x = x.float().to(device)
                        b_size, seq_len_curr = i.shape[0], i.shape[1]
                        
                        with torch.no_grad():
                            y_pred = tracknet(x).detach().cpu()
                        
                        y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
                        ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                        ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)
                        
                        for b in range(b_size):
                            if sample_count < buffer_size:
                                # Incomplete buffer
                                y_pred_ens = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                            else:
                                # General case with ensemble weights
                                y_pred_ens = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)
                            
                            ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                            ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_ens.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                            sample_count += 1
                            
                            if sample_count == num_sample:
                                # Handle last frames
                                y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                                y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                                
                                for f in range(1, seq_len):
                                    y_pred_ens = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                                    ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                                    ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred_ens.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                        
                        # Get predictions with proper image scaling
                        tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
                        for key in tmp_pred.keys():
                            tracknet_pred_dict[key].extend(tmp_pred[key])
                        
                        # Update buffer
                        y_pred_buffer = y_pred_buffer[-buffer_size:]
                    
                    # Apply noise filtering to shuttlecock tracking
                    logger.info("Applying noise filtering to shuttlecock tracking...")
                    final_pred_dict = apply_noise_filtering(tracknet_pred_dict, fps=fps, video_width=w_shuttle, video_height=h_shuttle)
                    
                    if final_pred_dict['Frame']:
                        # Save shuttlecock trajectory CSV
                        shuttlecock_csv_path = os.path.join(temp_dir, f"{video_name}_shuttlecock_tracking.csv")
                        write_pred_csv(final_pred_dict, shuttlecock_csv_path)
                        logger.info(f"Shuttlecock trajectory saved to: {shuttlecock_csv_path}")
                        
                        # Create shuttlecock tracking video
                        logger.info("Creating shuttlecock tracking video...")
                        shuttlecock_tracked_video_path = os.path.join(temp_dir, f"{video_name}_shuttlecock_tracked.mp4")
                        
                        # Read blacked video for shuttlecock overlay
                        cap_shuttle = cv2.VideoCapture(blacked_video_path)
                        if cap_shuttle.isOpened():
                            # Setup video writer
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out_shuttle = cv2.VideoWriter(shuttlecock_tracked_video_path, fourcc, fps, (w_shuttle, h_shuttle))
                            
                            # Process frames with shuttlecock tracking overlay
                            trail_positions = []  # Store recent positions for trail effect
                            frame_idx_shuttle = 0
                            track_trail_length = 10  # Default trail length
                            
                            # Create lookup dictionary for shuttlecock frame data
                            shuttle_frame_data_lookup = {}
                            for i in range(len(final_pred_dict['Frame'])):
                                frame_num = int(final_pred_dict['Frame'][i]) if str(final_pred_dict['Frame'][i]).isdigit() else i
                                shuttle_frame_data_lookup[frame_num] = {
                                    'x': float(final_pred_dict['X'][i]) if final_pred_dict['X'][i] != '' else 0,
                                    'y': float(final_pred_dict['Y'][i]) if final_pred_dict['Y'][i] != '' else 0,
                                    'visibility': float(final_pred_dict['Visibility'][i]) if final_pred_dict['Visibility'][i] != '' else 0
                                }
                            
                            while cap_shuttle.isOpened():
                                ret, frame = cap_shuttle.read()
                                if not ret:
                                    break
                                
                                # Get shuttlecock position for this frame
                                if frame_idx_shuttle in shuttle_frame_data_lookup:
                                    x = shuttle_frame_data_lookup[frame_idx_shuttle]['x']
                                    y = shuttle_frame_data_lookup[frame_idx_shuttle]['y']
                                    visibility = shuttle_frame_data_lookup[frame_idx_shuttle]['visibility']
                                    
                                    # Add to trail if visible
                                    if visibility > 0.5 and x > 0 and y > 0:
                                        trail_positions.append((int(x), int(y)))
                                        
                                        # Keep only recent positions for trail
                                        if len(trail_positions) > track_trail_length:
                                            trail_positions.pop(0)
                                        
                                        # Draw trail (fading effect)
                                        for i, (tx, ty) in enumerate(trail_positions):
                                            alpha = (i + 1) / len(trail_positions)  # Fade from 0 to 1
                                            color_intensity = int(255 * alpha)
                                            
                                            # Draw trail point
                                            cv2.circle(frame, (tx, ty), max(2, int(4 * alpha)), 
                                                     (0, color_intensity, 255), -1)
                                        
                                        # Draw current position (larger, brighter)
                                        cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), -1)
                                        cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), 2)
                                
                                out_shuttle.write(frame)
                                frame_idx_shuttle += 1
                            
                            cap_shuttle.release()
                            out_shuttle.release()
                            logger.info(f"Shuttlecock tracking video created: {shuttlecock_tracked_video_path}")
                        else:
                            logger.error("Cannot open blacked video for shuttlecock tracking")
                    else:
                        logger.warning("No valid shuttlecock tracking data after noise filtering")
                else:
                    logger.warning("TrackNet model not found, skipping shuttlecock tracking")
            except Exception as e:
                logger.error(f"Error in shuttlecock tracking: {str(e)}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("TrackNetV3 dependencies not available, skipping shuttlecock tracking")
        
        # Removed CSV and transformation data exports - focus on videos only
        
        # Create zip file with match analysis results
        zip_path = os.path.join(temp_dir, f"{video_name}_match_analysis.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add blacked video (non-tracked people removed, tracked players preserved)
            zipf.write(blacked_video_path, f"{video_name}_blacked.mp4")
            # Add top-down movement video
            zipf.write(movement_video_path, f"{video_name}_top_down_movement.mp4")
            # Add shuttlecock tracking results if available
            if shuttlecock_tracked_video_path and os.path.exists(shuttlecock_tracked_video_path):
                zipf.write(shuttlecock_tracked_video_path, f"{video_name}_shuttlecock_tracked.mp4")
            if shuttlecock_csv_path and os.path.exists(shuttlecock_csv_path):
                zipf.write(shuttlecock_csv_path, f"{video_name}_shuttlecock_tracking.csv")
        
        logger.info(f"Match analysis complete. Zip file created: {zip_path}")
        
        # Return zip file
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
        # Cleanup will happen after response is sent
        pass

@app.get("/")
async def root():
    return {"status": "MoveInsight Analysis Server is running", "version": "2.0.0 - Comprehensive Match Analysis with Integrated Shuttlecock Tracking"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MoveInsight Analysis Server with comprehensive match analysis (v2.0.0)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
