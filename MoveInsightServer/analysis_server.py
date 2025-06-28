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
    crop_video
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
@app.post("/analyze/video_crop/")
async def analyze_video_crop(
    file: UploadFile = File(...)
):
    """
    Simplified video crop using only detected court width while preserving full video height.
    This endpoint detects the badminton court and crops the video to court width 
    while maintaining the original video height.

    Returns a zip file containing:
    - Cropped video with court width and full height
    - Court coordinates txt file with detected key points

    Args:
        file: Video file to process
    """
    logger.info(f"Received video crop request: {file.filename}")
    
    temp_dir = tempfile.mkdtemp(prefix="video_crop_")
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
        
        # Step 2: Calculate simplified crop boundaries (court width + full height)
        logger.info("Step 2: Calculating simplified crop boundaries (court width + full height)...")
        crop_x_min, crop_y_min, crop_x_max, crop_y_max = calculate_court_width_crop_boundaries(
            temp_video_path, court_points
        )
        
        logger.info(f"Final crop boundaries: ({crop_x_min}, {crop_y_min}, {crop_x_max}, {crop_y_max})")
        
        # Step 3: Crop the video
        logger.info("Step 3: Cropping video...")
        cropped_video_path = os.path.join(temp_dir, f"{video_name}_cropped.mp4")
        
        success = crop_video(temp_video_path, cropped_video_path, crop_x_min, crop_y_min, crop_x_max, crop_y_max)
        if not success:
            raise HTTPException(status_code=500, detail="Video cropping failed")
        
        logger.info(f"Video cropping completed: {cropped_video_path}")
        
        # Save court coordinates to txt file
        court_coords_path = os.path.join(temp_dir, f"{video_name}_court_coordinates.txt")
        with open(court_coords_path, 'w') as f:
            f.write("# Court Key Points Coordinates (x, y)\n")
            f.write("# Format: x;y (one point per line)\n")
            f.write("# Order: Lower-left, Lower-right, Upper-left, Upper-right, Left-net, Right-net\n")
            for i, (x, y) in enumerate(court_points):
                f.write(f"{x};{y}\n")
        
        logger.info(f"Court coordinates saved to: {court_coords_path}")
        
        # Create zip file with both cropped video and court coordinates
        zip_path = os.path.join(temp_dir, f"{video_name}_cropped_analysis.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add cropped video
            zipf.write(cropped_video_path, f"{video_name}_cropped.mp4")
            # Add court coordinates
            zipf.write(court_coords_path, f"{video_name}_court_coordinates.txt")
        
        logger.info(f"Created zip file with cropped video and court coordinates: {zip_path}")
        
        # Return the zip file
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{video_name}_cropped_analysis.zip",
            background=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in video crop analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Video crop analysis failed: {str(e)}")
    
    finally:
        # Note: temp_dir cleanup will happen when response is sent
        pass

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


@app.post("/analyze/shuttlecock_tracking/")
async def analyze_shuttlecock_tracking(
    file: UploadFile = File(...),
    track_trail_length: int = Form(10),
    eval_mode: str = Form('weight'),
    batch_size: int = Form(16)
):
    """
    Track shuttlecock trajectory in video using TrackNetV3 and return tracking video with CSV.
    This endpoint replicates the behavior of predict.py with temporal ensemble.
    
    Args:
        file: Video file to process
        track_trail_length: Number of recent positions to show in trail (default: 10)
        eval_mode: Evaluation mode - 'weight', 'average', or 'nonoverlap' (default: 'weight')
        batch_size: Batch size for inference (default: 16)
    """
    if not TRACKNET_AVAILABLE:
        raise HTTPException(status_code=503, detail="TrackNetV3 dependencies not available")
    
    logger.info(f"Received shuttlecock tracking request: {file.filename}, eval_mode: {eval_mode}")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="shuttlecock_tracking_")
    temp_video_path = ""
    temp_csv_path = ""
    tracked_video_path = ""
    zip_path = ""
    
    try:
        # Save uploaded video
        video_name = Path(file.filename).stem
        temp_video_path = os.path.join(temp_dir, f"input_{file.filename}")
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"Video saved to: {temp_video_path}")
        
        # Setup model paths
        tracknet_model_path = os.path.join(os.path.dirname(__file__), "TrackNetV3", "ckpts", "TrackNet_best.pt")
        inpaintnet_model_path = os.path.join(os.path.dirname(__file__), "TrackNetV3", "ckpts", "InpaintNet_best.pt")
        
        if not os.path.exists(tracknet_model_path):
            raise HTTPException(status_code=500, detail="TrackNet model not found")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Get video properties for image scaling
        cap = cv2.VideoCapture(temp_video_path)
        w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        w_scaler, h_scaler = w / WIDTH, h / HEIGHT
        img_scaler = (w_scaler, h_scaler)
        logger.info(f"Video dimensions: {w}x{h}, scaling: {img_scaler}")
        
        # Load TrackNet model
        tracknet_ckpt = torch.load(tracknet_model_path, map_location=device)
        tracknet_seq_len = int(tracknet_ckpt['param_dict']['seq_len'])
        bg_mode = tracknet_ckpt['param_dict']['bg_mode']
        tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
        tracknet.load_state_dict(tracknet_ckpt['model'])
        tracknet.eval()
        
        # Load InpaintNet if available
        inpaintnet = None
        inpaintnet_seq_len = None
        if os.path.exists(inpaintnet_model_path):
            inpaintnet_ckpt = torch.load(inpaintnet_model_path, map_location=device)
            inpaintnet_seq_len = int(inpaintnet_ckpt['param_dict']['seq_len'])
            inpaintnet = get_model('InpaintNet').to(device)
            inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
            inpaintnet.eval()
            logger.info("InpaintNet loaded successfully")
        
        # Initialize prediction dictionary with image scaling info
        tracknet_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': [], 'Inpaint_Mask': [],
                             'Img_scaler': img_scaler, 'Img_shape': (w, h)}
        
        # Run TrackNet prediction with temporal ensemble (matching predict.py)
        logger.info(f"Running TrackNet prediction with {eval_mode} ensemble...")
        seq_len = tracknet_seq_len
        
        if eval_mode == 'nonoverlap':
            # Non-overlap sampling (original endpoint behavior)
            dataset = Video_IterableDataset(temp_video_path, seq_len, seq_len, bg_mode)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
            
            for batch_data in dataloader:
                indices, data = batch_data
                data = data.to(device, non_blocking=True)
                
                with torch.no_grad():
                    y_pred = tracknet(data.float()).detach().cpu()
                
                # Get predictions with proper image scaling
                batch_pred = predict(indices, y_pred=y_pred, img_scaler=img_scaler)
                
                # Accumulate results
                for key in batch_pred.keys():
                    tracknet_pred_dict[key].extend(batch_pred[key])
        else:
            # Temporal ensemble (weight or average mode)
            dataset = Video_IterableDataset(temp_video_path, seq_len, 1, bg_mode)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
            video_len = dataset.video_len
            logger.info(f"Video length: {video_len} frames")
            
            # Initialize ensemble parameters
            num_sample, sample_count = video_len - seq_len + 1, 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len)
            frame_i = torch.arange(seq_len - 1, -1, -1)
            y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
            weight = get_ensemble_weight(seq_len, eval_mode)
            
            from tqdm import tqdm
            for step, (i, x) in enumerate(tqdm(dataloader, desc="Processing frames")):
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
        
        # Run InpaintNet if available (matching predict.py)
        final_pred_dict = tracknet_pred_dict
        if inpaintnet is not None:
            logger.info("Running InpaintNet refinement...")
            from tracknet_test import generate_inpaint_mask
            tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h*0.05)
            inpaint_pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
            
            if eval_mode == 'nonoverlap':
                # Simple non-overlap InpaintNet processing
                from dataset import Shuttlecock_Trajectory_Dataset
                dataset = Shuttlecock_Trajectory_Dataset(
                    seq_len=inpaintnet_seq_len, sliding_step=inpaintnet_seq_len, 
                    data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True
                )
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                
                for step, (i, coor_pred, inpaint_mask) in enumerate(dataloader):
                    coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                    with torch.no_grad():
                        coor_inpaint = inpaintnet(coor_pred.to(device), inpaint_mask.to(device)).detach().cpu()
                        coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)
                    
                    # Thresholding
                    th_mask = ((coor_inpaint[:, :, 0] < COOR_TH) & (coor_inpaint[:, :, 1] < COOR_TH))
                    coor_inpaint[th_mask] = 0.
                    
                    # Get predictions
                    tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                    for key in tmp_pred.keys():
                        inpaint_pred_dict[key].extend(tmp_pred[key])
            
            final_pred_dict = inpaint_pred_dict if inpaint_pred_dict['Frame'] else tracknet_pred_dict
        
        # Get video properties for noise filtering
        cap_temp = cv2.VideoCapture(temp_video_path)
        fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
        cap_temp.release()
        
        # Apply enhanced noise filtering with track quality scoring to remove artifacts
        logger.info("Applying enhanced noise filtering to remove tracking artifacts...")
        final_pred_dict = apply_noise_filtering(final_pred_dict, fps=fps, video_width=w, video_height=h)
        
        if not final_pred_dict['Frame']:
            logger.warning("No valid shuttlecock tracking data remains after noise filtering")
            # Create empty CSV and video for consistency
            temp_csv_path = os.path.join(temp_dir, f"{video_name}_shuttlecock_tracking.csv")
            with open(temp_csv_path, 'w') as f:
                f.write("Frame,X,Y,Visibility\n")
            
            # Create zip with empty results
            zip_path = os.path.join(temp_dir, f"{video_name}_shuttlecock_tracking.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(temp_csv_path, f"{video_name}_shuttlecock_tracking.csv")
            
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename=f"{video_name}_shuttlecock_tracking.zip",
                background=None
            )
        
        # Save filtered trajectory CSV
        temp_csv_path = os.path.join(temp_dir, f"{video_name}_shuttlecock_tracking.csv")
        write_pred_csv(final_pred_dict, temp_csv_path)
        logger.info(f"Filtered shuttlecock trajectory saved to: {temp_csv_path}")
        
        # Create video with tracking overlay
        logger.info("Creating video with filtered shuttlecock tracking overlay...")
        tracked_video_path = os.path.join(temp_dir, f"{video_name}_tracked.mp4")
        
        # Read original video
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Cannot open input video")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tracked_video_path, fourcc, fps, (width, height))
        
        # Process frames with tracking overlay
        trail_positions = []  # Store recent positions for trail effect
        frame_idx = 0
        
        # Create a lookup dictionary for frame data
        frame_data_lookup = {}
        for i in range(len(final_pred_dict['Frame'])):
            frame_num = int(final_pred_dict['Frame'][i]) if str(final_pred_dict['Frame'][i]).isdigit() else i
            frame_data_lookup[frame_num] = {
                'x': float(final_pred_dict['X'][i]) if final_pred_dict['X'][i] != '' else 0,
                'y': float(final_pred_dict['Y'][i]) if final_pred_dict['Y'][i] != '' else 0,
                'visibility': float(final_pred_dict['Visibility'][i]) if final_pred_dict['Visibility'][i] != '' else 0
            }
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get shuttlecock position for this frame
            if frame_idx in frame_data_lookup:
                x = frame_data_lookup[frame_idx]['x']
                y = frame_data_lookup[frame_idx]['y']  
                visibility = frame_data_lookup[frame_idx]['visibility']
                
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
                    
                    # Draw frame number and coordinates
                    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Shuttlecock: ({int(x)}, {int(y)})", (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames with tracking overlay")
        
        cap.release()
        out.release()
        
        logger.info(f"Tracked video created: {tracked_video_path}")
        
        # Create zip file with both tracked video and CSV
        zip_path = os.path.join(temp_dir, f"{video_name}_shuttlecock_tracking.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add tracked video
            zipf.write(tracked_video_path, f"{video_name}_tracked.mp4")
            # Add trajectory CSV
            zipf.write(temp_csv_path, f"{video_name}_shuttlecock_tracking.csv")
        
        logger.info(f"Shuttlecock tracking analysis complete. Zip file created: {zip_path}")
        
        # Return zip file
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{video_name}_shuttlecock_tracking.zip",
            background=None
        )
        
    except Exception as e:
        logger.error(f"Error in shuttlecock tracking analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Shuttlecock tracking analysis failed: {str(e)}")
    
    finally:
        # Cleanup will happen after response is sent
        pass

@app.post("/analyze/court_movement/")
async def analyze_court_movement(
    file: UploadFile = File(...),
    num_people: int = Form(2),
    court_type: str = Form("doubles")
):
    """
    Analyze player movements on badminton court with top-down view visualization.
    
    This endpoint:
    1. Detects badminton court and creates perspective transformation
    2. Tracks multiple players throughout the video
    3. Creates top-down court view with player movement trails
    4. Creates original video with player pose overlays
    5. Calculates total distance moved for each player
    
    Returns a zip file containing:
    - Top-down court movement visualization video
    - Original video with player pose overlays
    - Player movement data CSV with positions and distances
    - Court transformation matrix data
    
    Args:
        file: Video file to process
        num_people: Number of people to track (default: 2)
        court_type: 'doubles' or 'singles' (default: 'doubles')
    """
    logger.info(f"Received court movement analysis request: {file.filename}, num_people: {num_people}, court_type: {court_type}")
    
    temp_dir = tempfile.mkdtemp(prefix="court_movement_")
    temp_video_path = ""
    
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
        
        # Step 2: Create perspective transformation
        logger.info("Step 2: Creating perspective transformation...")
        court_corners = get_court_corners_from_keypoints(court_points)
        transform_matrix = get_perspective_transformation_matrix(court_corners)
        
        # Create top-down court template
        court_template = create_top_down_court_template(court_type=court_type)
        
        # DEBUG: Save a debug image showing the detected court corners
        debug_court_img_path = os.path.join(temp_dir, f"{video_name}_debug_court_detection.jpg")
        cap_debug = cv2.VideoCapture(temp_video_path)
        ret_debug, debug_frame = cap_debug.read()
        cap_debug.release()
        
        if ret_debug:
            # Draw court corners on the image
            for i, (x, y) in enumerate(court_corners):
                cv2.circle(debug_frame, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red circles
                cv2.putText(debug_frame, f"C{i}", (int(x)+15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw lines connecting the corners
            pts = np.array(court_corners, np.int32)
            cv2.polylines(debug_frame, [pts], True, (0, 255, 0), 3)  # Green lines
            
            cv2.imwrite(debug_court_img_path, debug_frame)
            logger.info(f"DEBUG: Court detection visualization saved to {debug_court_img_path}")
        
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
        
        # Step 3: Track players throughout video
        logger.info("Step 3: Tracking player movements...")
        
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
                            
                            # Transform to top-down view
                            top_down_x, top_down_y = transform_point_to_top_down(ground_contact_point, transform_matrix)
                            
                            # Validate position is within court bounds
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
                                
                                # Transform to top-down view
                                top_down_x, top_down_y = transform_point_to_top_down((avg_x, avg_y), transform_matrix)
                                
                                # Validate position is within court bounds
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
                                current_frame_positions.append(None)
                    else:
                        current_frame_positions.append(None)
                
                # Skip video creation during tracking - we'll do it after smoothing
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames for court movement")
        
        cap.release()
        
        logger.info(f"Finished tracking {frame_idx} frames with {len(all_frame_pose_data)} pose data frames")
        
        # Step 3.5: Apply trajectory smoothing to reduce pose tracking jitter
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
        
        # Step 3.6: Create video visualization using smoothed trajectories
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
            
            # Build position history up to current frame for each player
            for player_idx in range(num_people):
                for i in range(min(frame_num + 1, len(player_positions_top_down[player_idx]))):
                    current_frame_positions_for_viz[player_idx].append(player_positions_top_down[player_idx][i])
            
            # Create visualization frame using current trajectory state
            viz_frame = create_player_trail_visualization(
                court_template, 
                current_frame_positions_for_viz
            )
            
            # Add frame information
            cv2.putText(viz_frame, f"Frame: {frame_num+1}/{max_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add distance information (final distances)
            for i, distance in enumerate(player_distances):
                cv2.putText(viz_frame, f"P{i+1} Distance: {distance:.2f}m", (10, 60 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add smoothing indicator
            cv2.putText(viz_frame, "Smoothed Trajectory", (10, court_template.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out.write(viz_frame)
            
            if frame_num % 100 == 0:
                logger.info(f"Generated video frame {frame_num+1}/{max_frames}")
        
        out.release()
        logger.info(f"Smoothed movement video created: {movement_video_path}")
        
        # Step 3.7: Create original video with pose overlays
        logger.info("Creating original video with player pose overlays...")
        pose_overlay_video_path = os.path.join(temp_dir, f"{video_name}_pose_overlay.mp4")
        
        # Reopen original video for pose overlay
        cap_original = cv2.VideoCapture(temp_video_path)
        if not cap_original.isOpened():
            raise HTTPException(status_code=500, detail="Cannot reopen original video for pose overlay")
        
        width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer for pose overlay
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_pose = cv2.VideoWriter(pose_overlay_video_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        
        while cap_original.isOpened():
            ret, frame = cap_original.read()
            if not ret:
                break
            
            # Get pose data for this frame if available
            if frame_idx < len(all_frame_pose_data):
                people_data = all_frame_pose_data[frame_idx]
                
                # Draw pose skeletons for each detected person
                for person_idx, person_joints in enumerate(people_data):
                    if person_idx < num_people and person_joints:  # Check for non-empty pose data
                        # Define player colors
                        player_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]  # Blue, Red, Green, Yellow
                        color = player_colors[person_idx % len(player_colors)]
                        
                        # Draw skeleton connections
                        pose_connections = [
                            # Face
                            ('LeftEye', 'RightEye'), ('LeftEye', 'Nose'), ('RightEye', 'Nose'),
                            ('LeftEar', 'LeftEye'), ('RightEar', 'RightEye'),
                            
                            # Upper body
                            ('LeftShoulder', 'RightShoulder'),
                            ('LeftShoulder', 'LeftElbow'), ('LeftElbow', 'LeftWrist'),
                            ('RightShoulder', 'RightElbow'), ('RightElbow', 'RightWrist'),
                            ('LeftShoulder', 'LeftHip'), ('RightShoulder', 'RightHip'),
                            ('LeftHip', 'RightHip'),
                            
                            # Lower body
                            ('LeftHip', 'LeftKnee'), ('LeftKnee', 'LeftAnkle'),
                            ('RightHip', 'RightKnee'), ('RightKnee', 'RightAnkle'),
                            
                            # Hands
                            ('LeftWrist', 'LeftThumb'), ('LeftWrist', 'LeftIndex'),
                            ('LeftWrist', 'LeftPinky'), ('LeftIndex', 'LeftPinky'),
                            ('RightWrist', 'RightThumb'), ('RightWrist', 'RightIndex'),
                            ('RightWrist', 'RightPinky'), ('RightIndex', 'RightPinky'),
                            
                            # Feet
                            ('LeftAnkle', 'LeftHeel'), ('LeftAnkle', 'LeftToe'),
                            ('RightAnkle', 'RightHeel'), ('RightAnkle', 'RightToe')
                        ]
                        
                        # Draw connections
                        for joint1, joint2 in pose_connections:
                            if joint1 in person_joints and joint2 in person_joints:
                                x1, y1, z1 = person_joints[joint1]
                                x2, y2, z2 = person_joints[joint2]
                                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:  # Valid coordinates
                                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw joint points
                        for joint_name, (x, y, z) in person_joints.items():
                            if x > 0 and y > 0:  # Valid coordinates
                                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                                cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 1)
                        
                        # Add player label
                        if 'LeftShoulder' in person_joints and 'RightShoulder' in person_joints:
                            left_shoulder = person_joints['LeftShoulder']
                            right_shoulder = person_joints['RightShoulder']
                            if left_shoulder[0] > 0 and left_shoulder[1] > 0 and right_shoulder[0] > 0 and right_shoulder[1] > 0:
                                label_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
                                label_y = int(min(left_shoulder[1], right_shoulder[1]) - 20)
                                cv2.putText(frame, f"Player {person_idx + 1}", (label_x - 40, label_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add frame information and tracking status
            cv2.putText(frame, f"Frame: {frame_idx + 1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add tracking status indicator
            if frame_idx < len(all_frame_pose_data):
                people_tracked = len([p for p in all_frame_pose_data[frame_idx] if p])
                tracking_status = f"Detected: {people_tracked}/{num_people}" if people_tracked > 0 else "No Detection"
                status_color = (0, 255, 0) if people_tracked > 0 else (0, 0, 255)
                cv2.putText(frame, tracking_status, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            out_pose.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Generated pose overlay frame {frame_idx}")
        
        cap_original.release()
        out_pose.release()
        logger.info(f"Pose overlay video created: {pose_overlay_video_path}")
        
        # Step 4: Save movement data to CSV
        movement_csv_path = os.path.join(temp_dir, f"{video_name}_movement_data.csv")
        with open(movement_csv_path, 'w') as f:
            f.write("Frame,Player,X_TopDown,Y_TopDown,Distance_Moved_m,Total_Distance_m\n")
            
            max_frames = max(len(positions) for positions in player_positions_top_down) if player_positions_top_down else 0
            
            for frame_num in range(max_frames):
                for player_idx in range(num_people):
                    if frame_num < len(player_positions_top_down[player_idx]):
                        pos = player_positions_top_down[player_idx][frame_num]
                        
                        # Calculate frame-to-frame distance
                        frame_distance = 0.0
                        if frame_num > 0 and frame_num-1 < len(player_positions_top_down[player_idx]):
                            prev_pos = player_positions_top_down[player_idx][frame_num-1]
                            frame_distance = calculate_real_world_distance(prev_pos, pos)
                        
                        # Calculate cumulative distance up to this frame
                        cumulative_distance = 0.0
                        for i in range(1, frame_num + 1):
                            if i < len(player_positions_top_down[player_idx]):
                                prev_pos = player_positions_top_down[player_idx][i-1]
                                curr_pos = player_positions_top_down[player_idx][i]
                                cumulative_distance += calculate_real_world_distance(prev_pos, curr_pos)
                        
                        f.write(f"{frame_num},{player_idx+1},{pos[0]:.2f},{pos[1]:.2f},{frame_distance:.4f},{cumulative_distance:.4f}\n")
        
        logger.info(f"Movement data saved to: {movement_csv_path}")
        
        # Step 5: Save transformation matrix
        transform_data_path = os.path.join(temp_dir, f"{video_name}_transform_data.txt")
        with open(transform_data_path, 'w') as f:
            f.write("# Court Movement Analysis Data\n")
            f.write(f"# Video: {file.filename}\n")
            f.write(f"# Players tracked: {num_people}\n")
            f.write(f"# Court type: {court_type}\n")
            f.write("\n# Total distances moved (meters):\n")
            for i, distance in enumerate(player_distances):
                f.write(f"Player_{i+1}_Total_Distance: {distance:.4f}\n")
            
            f.write("\n# Court key points (original video coordinates):\n")
            for i, (x, y) in enumerate(court_points):
                f.write(f"Court_Point_{i+1}: {x};{y}\n")
            
            f.write("\n# Perspective transformation matrix:\n")
            for i, row in enumerate(transform_matrix):
                f.write(f"Transform_Row_{i+1}: {','.join(map(str, row))}\n")
        
        logger.info(f"Transform data saved to: {transform_data_path}")
        
        # Step 6: Create summary statistics
        summary_stats = {
            'total_frames_processed': frame_idx,
            'players_tracked': num_people,
            'court_type': court_type,
            'player_distances': {f'player_{i+1}': round(dist, 4) for i, dist in enumerate(player_distances)},
            'video_duration_seconds': frame_idx / fps if fps > 0 else 0,
            'movement_validation': 'net_crossing_prevention_only',
            'trajectory_smoothing': 'enabled_savgol_filter' if SCIPY_AVAILABLE else 'enabled_moving_average'
        }
        
        logger.info(f"Movement analysis summary: {summary_stats}")
        
        # Create zip file with all results
        zip_path = os.path.join(temp_dir, f"{video_name}_court_movement_analysis.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add movement video
            zipf.write(movement_video_path, f"{video_name}_court_movement.mp4")
            # Add pose overlay video
            zipf.write(pose_overlay_video_path, f"{video_name}_pose_overlay.mp4")
            # Add movement data CSV
            zipf.write(movement_csv_path, f"{video_name}_movement_data.csv")  
            # Add transformation data
            zipf.write(transform_data_path, f"{video_name}_transform_data.txt")
            # Add debug court detection image
            if os.path.exists(debug_court_img_path):
                zipf.write(debug_court_img_path, f"{video_name}_debug_court_detection.jpg")
        
        logger.info(f"Court movement analysis complete. Zip file created: {zip_path}")
        
        # Return zip file
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=f"{video_name}_court_movement_analysis.zip",
            background=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in court movement analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Court movement analysis failed: {str(e)}")
    
    finally:
        # Cleanup will happen after response is sent
        pass

@app.get("/")
async def root():
    return {"status": "MoveInsight Analysis Server is running", "version": "1.4.0 - Refactored Match Analysis"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MoveInsight Analysis Server with refactored match analysis (v1.4.0)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
