# pose_analysis.py
import mediapipe as mp
import numpy as np
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("pose_analysis")

# --- MediaPipe Pose Estimation Setup ---
mp_pose = mp.solutions.pose

# Joint mapping from MediaPipe indices to meaningful names
JOINT_MAP = {
    0: 'Nose', 11: 'LeftShoulder', 12: 'RightShoulder', 13: 'LeftElbow', 14: 'RightElbow',
    15: 'LeftWrist', 16: 'RightWrist', 23: 'LeftHip', 24: 'RightHip', 25: 'LeftKnee',
    26: 'RightKnee', 27: 'LeftAnkle', 28: 'RightAnkle', 29: 'LeftHeel', 30: 'RightHeel',
    31: 'LeftFootIndex', 32: 'RightFootIndex'
}

# Remapping for more intuitive joint names
JOINT_NAME_REMAPPING = { 
    'LeftFootIndex': 'LeftToe', 
    'RightFootIndex': 'RightToe' 
}

def get_all_expected_joint_names() -> Set[str]:
    """
    Get all possible joint names that might appear in pose analysis.
    
    Returns:
        Set of all expected joint names
    """
    all_expected_joint_names = set(JOINT_MAP.values())
    all_expected_joint_names.update(JOINT_NAME_REMAPPING.values())
    return all_expected_joint_names

def transform_pydantic_to_numpy(joint_data_per_frame_pydantic: List) -> Dict[str, List[List[float]]]:
    """
    Transform pydantic models to format needed for align_keypoints_with_interpolation.
    Output format: {'joint_name': [[x1,y1,z1], [x2,y2,z2], ...]} 
    The inner lists can have varying lengths if a joint is not detected in all frames.
    
    Args:
        joint_data_per_frame_pydantic: List of FrameDataItem objects
        
    Returns:
        Dictionary mapping joint names to lists of coordinate arrays
    """
    joint_data_lists: Dict[str, List[List[float]]] = {}
    if not joint_data_per_frame_pydantic:
        return joint_data_lists
    
    # Initialize lists for all possible joint names that might appear
    all_possible_joint_names = get_all_expected_joint_names()

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

def transform_numpy_to_pydantic(keypoints: Dict[str, np.ndarray], frame_count: int):
    """
    Transform numpy arrays (shape (T,3)) back to pydantic models for response.
    
    Args:
        keypoints: Dictionary mapping joint names to numpy arrays of shape (T, 3)
        frame_count: Total number of frames
        
    Returns:
        List of FrameDataItem objects
    """
    # Import here to avoid circular imports
    from models import FrameDataItem, JointDataItem
    
    result = []
    for frame_idx in range(frame_count):
        frame_joints = {}
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

def extract_pose_landmarks_from_frame(frame, pose_estimator, width: int, height: int) -> Dict[str, List[float]]:
    """
    Extract pose landmarks from a single frame using MediaPipe.
    
    Args:
        frame: Input frame (BGR format)
        pose_estimator: MediaPipe pose estimator instance
        width: Frame width
        height: Frame height
        
    Returns:
        Dictionary mapping joint names to [x, y, z] coordinates
    """
    import cv2
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(frame_rgb)
    
    joint_data = {}
    if results.pose_landmarks:
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in JOINT_MAP:
                original_joint_name = JOINT_MAP[idx]
                # Use the remapped name if it exists, otherwise the original name
                joint_name_for_client = JOINT_NAME_REMAPPING.get(original_joint_name, original_joint_name)
                
                # Convert normalized coordinates to pixel coordinates
                x = lm.x * width if hasattr(lm, 'x') else lm.x
                y = lm.y * height if hasattr(lm, 'y') else lm.y
                z = lm.z if hasattr(lm, 'z') else 0.0
                
                joint_data[joint_name_for_client] = [x, y, z]
    
    return joint_data

def get_required_joints_for_swing_analysis() -> List[str]:
    """
    Get the list of joints required for swing analysis.
    
    Returns:
        List of joint names required for swing evaluation
    """
    return [
        'RightShoulder', 'LeftShoulder', 'RightElbow', 'LeftElbow', 'RightWrist',
        'RightHip', 'LeftHip', 'RightHeel', 'RightToe', 'LeftHeel', 'LeftToe'
    ]

def get_default_swing_rules_keys() -> List[str]:
    """
    Get the default swing analysis rule keys.
    
    Returns:
        List of swing rule names
    """
    return [
        'shoulder_abduction', 'elbow_flexion', 'elbow_lower', 
        'foot_direction_aligned', 'proximal_to_distal_sequence', 
        'hip_forward_shift', 'trunk_rotation_completed'
    ]

def detect_multiple_people_in_frame(frame, pose_estimator, court_points: Optional[List[Tuple[float, float]]] = None, num_people: int = 2) -> List[Dict[str, List[float]]]:
    """
    FIXED: Proper multi-person detection for badminton court analysis.
    
    MediaPipe Pose only detects one person per frame, so we need a different approach.
    This function uses person detection first, then pose estimation on each person.
    
    Args:
        frame: Input frame (BGR format)
        pose_estimator: MediaPipe pose estimator instance  
        court_points: Optional list of 6 court key points for filtering
        num_people: Maximum number of people to track
        
    Returns:
        List of dictionaries, each containing joint data for one person
    """
    import cv2
    import numpy as np
    
    # Convert to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    
    detected_people = []
    
    # Since MediaPipe Pose only detects one person, we'll use a different strategy:
    # 1. Use OpenCV's person detection to find multiple people
    # 2. Then run MediaPipe on each detected person's region
    
    # Initialize HOG descriptor for person detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Detect people using HOG
    people_boxes, weights = hog.detectMultiScale(frame, 
                                                winStride=(4,4),
                                                padding=(8,8), 
                                                scale=1.05)
    
    # Process each detected person
    for i, (x, y, w, h) in enumerate(people_boxes):
        if len(detected_people) >= num_people:
            break
            
        # Expand bounding box slightly to ensure full person is captured
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(width, x + w + margin)
        y2 = min(height, y + h + margin)
        
        # Extract person region
        person_region = frame_rgb[y1:y2, x1:x2]
        
        if person_region.size == 0:
            continue
            
        # Run MediaPipe pose estimation on this person
        results = pose_estimator.process(person_region)
        
        if results.pose_landmarks:
            joint_data = {}
            landmarks_in_court = 0
            total_landmarks = 0
            
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if idx in JOINT_MAP:
                    original_joint_name = JOINT_MAP[idx]
                    joint_name = JOINT_NAME_REMAPPING.get(original_joint_name, original_joint_name)
                    
                    # Convert from person region coordinates to full frame coordinates
                    x_full = (lm.x * (x2 - x1)) + x1
                    y_full = (lm.y * (y2 - y1)) + y1
                    z = lm.z
                    
                    joint_data[joint_name] = [x_full, y_full, z]
                    total_landmarks += 1
                    
                    # Check if landmark is in court area using accurate polygon method
                    if court_points:
                        from court_detection import point_in_court_polygon
                        if point_in_court_polygon(x_full, y_full, court_points, margin=20.0):
                            landmarks_in_court += 1
            
            # Calculate court overlap ratio
            court_overlap_ratio = landmarks_in_court / total_landmarks if total_landmarks > 0 else 0
            
            person_data = {
                'joint_data': joint_data,
                'bbox_area': w * h,
                'court_overlap_ratio': court_overlap_ratio,
                'bbox': (x, y, w, h)
            }
            
            # Filter by court area if court_points provided
            if court_points is None or court_overlap_ratio > 0.3:  # At least 30% overlap with court
                detected_people.append(person_data)
    
    # If HOG didn't detect enough people, fallback to single person detection on full frame
    if len(detected_people) == 0:
        results = pose_estimator.process(frame_rgb)
        
        if results.pose_landmarks:
            joint_data = {}
            landmarks_in_court = 0
            total_landmarks = 0
            
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if idx in JOINT_MAP:
                    original_joint_name = JOINT_MAP[idx]
                    joint_name = JOINT_NAME_REMAPPING.get(original_joint_name, original_joint_name)
                    
                    x = lm.x * width
                    y = lm.y * height
                    z = lm.z
                    
                    joint_data[joint_name] = [x, y, z]
                    total_landmarks += 1
                    
                    # Check if landmark is in court area using accurate polygon method
                    if court_points:
                        from court_detection import point_in_court_polygon
                        if point_in_court_polygon(x, y, court_points, margin=20.0):
                            landmarks_in_court += 1
            
            # Calculate court overlap ratio for fallback detection
            court_overlap_ratio = landmarks_in_court / total_landmarks if total_landmarks > 0 else 0
            
            # Only add if person has sufficient court overlap (same 30% threshold)
            if court_points is None or court_overlap_ratio > 0.3:
                detected_people.append({
                    'joint_data': joint_data,
                    'bbox_area': width * height,
                    'court_overlap_ratio': court_overlap_ratio,
                    'bbox': (0, 0, width, height)
                })
    
    # Sort by bounding box area (largest first) and take top num_people
    detected_people.sort(key=lambda x: x['bbox_area'], reverse=True)
    detected_people = detected_people[:num_people]
    
    # Return just the joint data
    return [person['joint_data'] for person in detected_people]