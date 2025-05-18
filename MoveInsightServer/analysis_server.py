# MoveInsightServer/analysis_server.py
from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import numpy as np
import logging
import time
import traceback
import cv2
import mediapipe as mp
import tempfile
import os

# Import functions from swing_diagnose.py
from swing_diagnose import evaluate_swing_rules, align_keypoints_with_interpolation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis_server")

# --- FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

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

# --- MediaPipe Pose Estimation Setup ---
mp_pose = mp.solutions.pose
JOINT_MAP = {
    0: 'Nose', 11: 'LeftShoulder', 12: 'RightShoulder', 13: 'LeftElbow', 14: 'RightElbow',
    15: 'LeftWrist', 16: 'RightWrist', 23: 'LeftHip', 24: 'RightHip', 25: 'LeftKnee',
    26: 'RightKnee', 27: 'LeftAnkle', 28: 'RightAnkle', 29: 'LeftHeel', 30: 'RightHeel',
    31: 'LeftFootIndex', 32: 'RightFootIndex'
}
JOINT_NAME_REMAPPING = { 'LeftFootIndex': 'LeftToe', 'RightFootIndex': 'RightToe' }

# --- Pydantic Models ---
class JointDataItem(BaseModel):
    x: float
    y: float
    z: Optional[float] = None # Added Z coordinate
    confidence: Optional[float] = None

class FrameDataItem(BaseModel):
    joints: Dict[str, JointDataItem]

class VideoAnalysisResponseModel(BaseModel):
    total_frames: int
    joint_data_per_frame: List[FrameDataItem]
    swing_analysis: Optional[Dict[str, bool]] = None

class TechniqueComparisonRequestDataModel(BaseModel):
    user_video_frames: List[FrameDataItem] # Will now contain 3D data
    model_video_frames: List[FrameDataItem] # Will now contain 3D data
    dominant_side: str

class ComparisonResultModel(BaseModel):
    user_score: float
    reference_score: float
    similarity: Dict[str, bool]
    user_details: Dict[str, bool]
    reference_details: Dict[str, bool]

# --- Helper Functions ---
def transform_pydantic_to_numpy(joint_data_per_frame_pydantic: List[FrameDataItem]) -> Dict[str, List[List[float]]]:
    """
    Transform pydantic models to format needed for align_keypoints_with_interpolation.
    Output format: {'joint_name': [[x1,y1,z1], [x2,y2,z2], ...]} 
    The inner lists can have varying lengths if a joint is not detected in all frames.
    """
    joint_data_lists: Dict[str, List[List[float]]] = {}
    if not joint_data_per_frame_pydantic:
        return joint_data_lists
    
    # Initialize lists for all possible joint names that might appear
    all_possible_joint_names = set(JOINT_MAP.values())
    all_possible_joint_names.update(JOINT_NAME_REMAPPING.values())

    for name in all_possible_joint_names:
        joint_data_lists[name] = []
    
    for frame_data in joint_data_per_frame_pydantic:
        for joint_name_in_frame, joint_info in frame_data.joints.items():
            # Ensure z is not None, default to 0.0 if it is
            z_val = joint_info.z if joint_info.z is not None else 0.0
            # Append to the list for this joint_name
            # If joint_name_in_frame is not pre-initialized (e.g. unexpected name), it will error.
            # This assumes joint_name_in_frame will always be in all_possible_joint_names.
            if joint_name_in_frame in joint_data_lists:
                 joint_data_lists[joint_name_in_frame].append([joint_info.x, joint_info.y, z_val])
            else:
                logger.warning(f"Unexpected joint name '{joint_name_in_frame}' encountered in pydantic transform.")
    
    return joint_data_lists

def transform_numpy_to_pydantic(keypoints: Dict[str, np.ndarray], frame_count: int) -> List[FrameDataItem]:
    """
    Transform numpy arrays (shape (T,3)) back to pydantic models for response
    """
    result: List[FrameDataItem] = []
    for frame_idx in range(frame_count):
        frame_joints: Dict[str, JointDataItem] = {}
        for joint_name, points_array in keypoints.items():
            if frame_idx < points_array.shape[0] and not np.isnan(points_array[frame_idx]).any(): # Check if frame_idx is within bounds and not all NaN
                point = points_array[frame_idx]
                if points_array.shape[1] == 3: # Ensure it's 3D data
                    frame_joints[joint_name] = JointDataItem(
                        x=float(point[0]), 
                        y=float(point[1]),
                        z=float(point[2]), # Add Z
                        confidence=0.9  # Default confidence; consider passing actual confidence if available
                    )
                # No fallback for 2D, expect 3D from align_keypoints
        if frame_joints: # Only add frame if it has any valid joints
            result.append(FrameDataItem(joints=frame_joints))
    return result

# --- API Endpoints ---
@app.post("/analyze/video_upload/", response_model=VideoAnalysisResponseModel)
async def analyze_video_upload(
    file: UploadFile = File(...),
    dominant_side: str = Form("Right")
):
    logger.info(f"Received video upload for 3D analysis: {file.filename}, dominant side: {dominant_side}")
    temp_video_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            temp_video_path = tmp.name
        logger.info(f"Video saved temporarily to: {temp_video_path}")

        # Store raw detected points per joint: {'joint_name': [[x,y,z], [x,y,z], ...]}
        # The inner lists will only contain data for frames where the joint was detected.
        joint_data_raw_detections: Dict[str, List[List[float]]] = {}
        
        # Initialize with all possible joint names that MediaPipe might output via our mapping
        all_expected_client_joint_names = set(JOINT_MAP.values())
        all_expected_client_joint_names.update(JOINT_NAME_REMAPPING.values())
        for name in all_expected_client_joint_names:
            joint_data_raw_detections[name] = []

        frame_count = 0
        
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_estimator:
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {temp_video_path}")
                raise HTTPException(status_code=400, detail="Could not open video file.")

            while cap.isOpened():
                success, frame = cap.read()
                if not success: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose_estimator.process(frame_rgb)
                
                if results.pose_landmarks:
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        if idx in JOINT_MAP:
                            original_joint_name = JOINT_MAP[idx]
                            # Use the remapped name if it exists, otherwise the original name
                            joint_name_for_client = JOINT_NAME_REMAPPING.get(original_joint_name, original_joint_name)
                            
                            # lm.visibility is also available if needed for confidence
                            if joint_name_for_client in joint_data_raw_detections: # Ensure key exists
                                joint_data_raw_detections[joint_name_for_client].append([lm.x, lm.y, lm.z])
                            else:
                                # This case should ideally not happen if all_expected_client_joint_names is comprehensive
                                logger.warning(f"Detected joint '{joint_name_for_client}' not in pre-initialized keys. Adding it.")
                                joint_data_raw_detections[joint_name_for_client] = [[lm.x, lm.y, lm.z]]
                
                frame_count += 1
            cap.release()
        
        logger.info(f"Processed {frame_count} frames from video for 3D data.")

        if frame_count == 0:
            logger.warning("No frames detected in the video.")
            return VideoAnalysisResponseModel(total_frames=0, joint_data_per_frame=[], swing_analysis=None)

        # Align and interpolate, now expecting 3D data
        # align_keypoints_with_interpolation will fill missing frames with NaNs or interpolated values
        keypoints_3d_aligned = align_keypoints_with_interpolation(joint_data_raw_detections, frame_count)
        
        # Convert aligned 3D keypoints to pydantic models for response
        joint_data_per_frame_pydantic = transform_numpy_to_pydantic(keypoints_3d_aligned, frame_count)
        
        # --- Swing Analysis (using 2D slice) ---
        swing_analysis_results = None
        # Define keys required by evaluate_swing_rules
        required_keys_for_swing = [
            'RightShoulder', 'LeftShoulder', 'RightElbow', 'LeftElbow', 'RightWrist',
            'RightHip', 'LeftHip', 'RightHeel', 'RightToe', 'LeftHeel', 'LeftToe'
        ]
        default_swing_rules_keys = [ # Must match ComparisonResultModel.user_details keys
            'shoulder_abduction', 'elbow_flexion', 'elbow_lower', 
            'foot_direction_aligned', 'proximal_to_distal_sequence', 
            'hip_forward_shift', 'trunk_rotation_completed'
        ]

        keypoints_2d_for_swing: Dict[str, np.ndarray] = {}
        all_required_present_and_valid_for_swing = True

        for k in required_keys_for_swing:
            if k in keypoints_3d_aligned and keypoints_3d_aligned[k].shape[0] == frame_count and not np.all(np.isnan(keypoints_3d_aligned[k][:, :2])):
                keypoints_2d_for_swing[k] = keypoints_3d_aligned[k][:, :2] # Take x, y
            else:
                logger.warning(f"Keypoint '{k}' missing, has insufficient frames ({keypoints_3d_aligned.get(k, np.array([])).shape[0]}/{frame_count}), or is all NaNs for swing analysis.")
                all_required_present_and_valid_for_swing = False
                break
        
        if all_required_present_and_valid_for_swing:
            try:
                swing_analysis_results = evaluate_swing_rules(keypoints_2d_for_swing, dominant_side=dominant_side)
                logger.info(f"Swing analysis (2D) for dominant side {dominant_side}: {swing_analysis_results}")
            except Exception as e:
                logger.error(f"Error in evaluate_swing_rules: {str(e)} - Traceback: {traceback.format_exc()}")
                swing_analysis_results = {rule_key: False for rule_key in default_swing_rules_keys}
        else:
            logger.warning("Missing or incomplete required key points for swing evaluation. Defaulting swing analysis to False.")
            swing_analysis_results = {rule_key: False for rule_key in default_swing_rules_keys}


        logger.info(f"Returning {len(joint_data_per_frame_pydantic)} frames of 3D joint data and 2D swing analysis.")
        return VideoAnalysisResponseModel(
            total_frames=frame_count,
            joint_data_per_frame=joint_data_per_frame_pydantic,
            swing_analysis=swing_analysis_results
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
    required_keys_for_swing = [ # Must match evaluate_swing_rules expectations
            'RightShoulder', 'LeftShoulder', 'RightElbow', 'LeftElbow', 'RightWrist',
            'RightHip', 'LeftHip', 'RightHeel', 'RightToe', 'LeftHeel', 'LeftToe'
    ]
    default_swing_rules_keys = list(ComparisonResultModel.model_fields['user_details'].default.keys() if ComparisonResultModel.model_fields['user_details'].default else [
            'shoulder_abduction', 'elbow_flexion', 'elbow_lower', 
            'foot_direction_aligned', 'proximal_to_distal_sequence', 
            'hip_forward_shift', 'trunk_rotation_completed'
        ])


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

@app.get("/")
async def root():
    return {"status": "MoveInsight Analysis Server is running", "version": "1.3.1 - 3D Pose Update with Robustness"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MoveInsight Analysis Server with 3D pose capabilities (v1.3.1)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
