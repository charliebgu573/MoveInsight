# MoveInsightServer/swing_diagnose.py
# Technical indicators to be judged:
#   1. **shoulder_abduction**: Avoid dropping the elbow.
#   2. **elbow_flexion**: Keep the elbow joint tight.
#   3. **elbow_lower**: The elbow of the hitting hand should be lower than the non-hitting hand's elbow.
#   4. **foot_direction_aligned**: Are the directions of the toes of both feet consistent?
#   5. **proximal_to_distal_sequence**: Is the sequence of upper limb exertion correct during the swing?
#   6. **hip_forward_shift**: Hip moves forward during the swing.
#   7. **trunk_rotation_completed**: Is the trunk rotation sufficient during the swing?

from typing import Dict, List
import numpy as np
# IMPORTANT: This script currently processes 2D keypoints (x, y).
# If 3D keypoints (x, y, z) are passed from the server, 
# the server should slice them to 2D (x,y) before calling evaluate_swing_rules.
# Future improvements could adapt these rules for 3D analysis.

def evaluate_swing_rules(keypoints: Dict[str, np.ndarray], dominant_side: str ='Right') -> Dict[str, bool]:
    """
    Evaluates swing rules based on 2D keypoint data.

    KEYPOINTS Input (EXPECTS 2D DATA):
    A dictionary where keys are joint names (e.g., 'RightShoulder') and
    values are NumPy arrays of shape (T, 2) representing (X, Y) coordinates
    for T frames.

    If 3D data (T,3) is available, the caller MUST pass data[:, :2].
    """
    result: Dict[str, bool] = {}
    default_rules_keys = [ # These keys define the structure of the output
        'shoulder_abduction', 'elbow_flexion', 'elbow_lower', 
        'foot_direction_aligned', 'proximal_to_distal_sequence', 
        'hip_forward_shift', 'trunk_rotation_completed'
    ]

    # Initialize all rules to False
    for rule_key in default_rules_keys:
        result[rule_key] = False

    # --- Initial checks for data validity ---
    if not keypoints:
        print("⚠️ evaluate_swing_rules: Empty keypoints dictionary. Returning all rules as False.")
        return result.copy()

    D = dominant_side
    ND = 'Left' if D == 'Right' else 'Right'
    
    required_joint_names = [
        f'{D}Shoulder', f'{D}Elbow', f'{D}Wrist', f'{D}Hip',
        f'{ND}Shoulder', f'{ND}Elbow', f'{ND}Hip', # Non-dominant side joints also needed
        'RightToe', 'RightHeel', 'LeftToe', 'LeftHeel'
    ]
    
    # Check for presence and shape of required joints
    min_frames = float('inf')
    for joint_name in required_joint_names:
        if joint_name not in keypoints or not isinstance(keypoints[joint_name], np.ndarray) or keypoints[joint_name].ndim != 2 or keypoints[joint_name].shape[1] != 2:
            print(f"⚠️ evaluate_swing_rules: Missing, not a NumPy array, or incorrect shape for required joint: {joint_name}. Expected (T,2). Returning all rules as False.")
            return result.copy()
        if keypoints[joint_name].shape[0] == 0: # No frames for this joint
             print(f"⚠️ evaluate_swing_rules: Zero frames for required joint: {joint_name}. Returning all rules as False.")
             return result.copy()
        min_frames = min(min_frames, keypoints[joint_name].shape[0])

    if min_frames == float('inf') or min_frames == 0:
        print("⚠️ evaluate_swing_rules: No valid frame data found across required joints. Returning all rules as False.")
        return result.copy()

    T = min_frames # Use the minimum number of frames available across all required joints

    # Ensure all keypoints are consistently sliced to T frames and handle potential NaNs from upstream
    processed_keypoints: Dict[str, np.ndarray] = {}
    for joint_name in required_joint_names: # Only process required joints
        kp_data = keypoints[joint_name][:T] # Slice to T frames
        # If align_keypoints_with_interpolation filled with NaNs, they will persist here.
        # Calculations below need to be NaN-aware.
        processed_keypoints[joint_name] = kp_data
    
    # Add optional hand keypoint if present and valid
    optional_hand_key = f'{D}Hand'
    if optional_hand_key in keypoints and \
       isinstance(keypoints[optional_hand_key], np.ndarray) and \
       keypoints[optional_hand_key].ndim == 2 and \
       keypoints[optional_hand_key].shape[1] == 2 and \
       keypoints[optional_hand_key].shape[0] > 0:
        processed_keypoints[optional_hand_key] = keypoints[optional_hand_key][:T]


    # --- Angle Calculation Helper Functions (NaN-aware) ---
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        if np.isnan(v1).any() or np.isnan(v2).any(): return np.nan
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 < 1e-6 or norm_v2 < 1e-6: return 180.0 # Default for zero vectors
        
        unit_v1 = v1 / norm_v1
        unit_v2 = v2 / norm_v2
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return np.degrees(np.arccos(dot_product))

    def get_joint_angle_at_frame(kp_dict: Dict[str, np.ndarray], joint_A_name: str, joint_B_name: str, joint_C_name: str, frame_idx: int) -> float:
        pt_A = kp_dict[joint_A_name][frame_idx]
        pt_B = kp_dict[joint_B_name][frame_idx]
        pt_C = kp_dict[joint_C_name][frame_idx]
        if np.isnan(pt_A).any() or np.isnan(pt_B).any() or np.isnan(pt_C).any():
            return np.nan
        return angle_between(pt_A - pt_B, pt_C - pt_B)

    # --- Phase Split Logic ---
    elbow_x_vel = np.zeros(T)
    if T > 1:
        # Ensure dominant elbow data is not all NaN before differencing
        dom_elbow_x_coords = processed_keypoints[f'{D}Elbow'][:, 0]
        if not np.all(np.isnan(dom_elbow_x_coords)):
            elbow_x_vel[1:] = np.diff(dom_elbow_x_coords) # NaNs will propagate if present
        else: # All NaN, velocity is effectively NaN or zero
            elbow_x_vel.fill(np.nan)


    # Smooth velocity, handling NaNs by ignoring them in convolution or using NaN-aware mean
    smoothed_vel = elbow_x_vel.copy() # Default to original if not enough points for smoothing
    window_size = min(5, T - 1 if T > 1 else 1)
    if window_size >= 2 and T > window_size: # window_size must be at least 1 for np.ones
        # Basic NaN-robust smoothing: convolve only valid parts or use rolling mean
        # For simplicity, let's use a simple moving average that handles NaNs
        # This is a placeholder for a more robust NaN-aware smoothing if needed.
        # A common approach is to interpolate NaNs before smoothing, or use pandas.Series.rolling(..).mean()
        # For now, if NaNs are present, smoothing quality might degrade.
        # Let's assume align_keypoints minimizes NaNs for critical joints.
        if not np.all(np.isnan(elbow_x_vel)):
            # Simple convolution, NaNs will propagate.
            conv_weights = np.ones(window_size) / window_size
            smoothed_vel_conv = np.convolve(elbow_x_vel, conv_weights, mode='valid')
            
            padding_len = T - len(smoothed_vel_conv)
            padding_start = padding_len // 2
            padding_end = padding_len - padding_start
            
            if len(smoothed_vel_conv) > 0: # Ensure conv result is not empty
                 smoothed_vel = np.pad(smoothed_vel_conv, (padding_start, padding_end), 'edge')
            # else smoothed_vel remains elbow_x_vel (e.g. if T is too small for 'valid' mode to produce output)

    swing_start_idx = 0
    if T > 1 and not np.all(np.isnan(smoothed_vel)):
        # Find where velocity changes from negative to positive (elbow starts moving forward)
        for t in range(1, T):
            if not np.isnan(smoothed_vel[t-1]) and not np.isnan(smoothed_vel[t]):
                if smoothed_vel[t-1] < 0 and smoothed_vel[t] >= 0:
                    swing_start_idx = t
                    break
        if swing_start_idx == 0: # Fallback: first significant positive velocity
            vel_abs_max = np.nanmax(np.abs(smoothed_vel))
            vel_threshold = vel_abs_max * 0.15 if vel_abs_max > 0 else 0.01
            for t in range(1,T):
                if not np.isnan(smoothed_vel[t]) and smoothed_vel[t] > vel_threshold:
                    swing_start_idx = t
                    break
    if swing_start_idx == 0: swing_start_idx = T // 2 # Default if still not found

    swing_end_idx = T - 1

    if swing_start_idx >= swing_end_idx and T > 0: # If swing phase is too short or invalid
        print(f"⚠️ Swing phase calculation resulted in start ({swing_start_idx}) >= end ({swing_end_idx}). Defaulting rules to False.")
        return result.copy() # Return dict with all False

    # --- Rule Evaluation (NaN-aware) ---

    # 1. Shoulder Abduction (引拍期 - Preparation Phase)
    prep_phase_angles_shoulder = [get_joint_angle_at_frame(processed_keypoints, f'{D}Hip', f'{D}Shoulder', f'{D}Elbow', t) for t in range(swing_start_idx)]
    prep_phase_angles_shoulder_np = np.array(prep_phase_angles_shoulder)
    valid_shoulder_angles = prep_phase_angles_shoulder_np[~np.isnan(prep_phase_angles_shoulder_np)]
    if valid_shoulder_angles.size > 0:
        result['shoulder_abduction'] = np.any((valid_shoulder_angles >= 60) & (valid_shoulder_angles <= 90))

    # 2. Elbow Flexion (引拍期)
    prep_phase_angles_elbow = [get_joint_angle_at_frame(processed_keypoints, f'{D}Shoulder', f'{D}Elbow', f'{D}Wrist', t) for t in range(swing_start_idx)]
    prep_phase_angles_elbow_np = np.array(prep_phase_angles_elbow)
    valid_elbow_angles = prep_phase_angles_elbow_np[~np.isnan(prep_phase_angles_elbow_np)]
    if valid_elbow_angles.size > 0:
        result['elbow_flexion'] = np.any(valid_elbow_angles < 90)

    # 3. Elbow Lower (引拍期)
    dom_elbow_y = processed_keypoints[f'{D}Elbow'][:swing_start_idx, 1]
    nondom_elbow_y = processed_keypoints[f'{ND}Elbow'][:swing_start_idx, 1]
    valid_mask = ~np.isnan(dom_elbow_y) & ~np.isnan(nondom_elbow_y)
    if np.any(valid_mask):
        result['elbow_lower'] = np.any(dom_elbow_y[valid_mask] > nondom_elbow_y[valid_mask]) # In image coords, larger Y is lower

    # 4. Foot Direction Aligned (引拍期)
    foot_angles_prep = []
    for t in range(swing_start_idx):
        v_r = processed_keypoints['RightToe'][t] - processed_keypoints['RightHeel'][t]
        v_l = processed_keypoints['LeftToe'][t] - processed_keypoints['LeftHeel'][t]
        foot_angles_prep.append(angle_between(v_r, v_l))
    foot_angles_prep_np = np.array(foot_angles_prep)
    valid_foot_angles = foot_angles_prep_np[~np.isnan(foot_angles_prep_np)]
    if valid_foot_angles.size > 0:
        result['foot_direction_aligned'] = np.any(valid_foot_angles < 30)

    # --- Swinging Phase Rules ---
    def get_full_angle_series(kp_dict, joint_A, joint_B, joint_C):
        return np.array([get_joint_angle_at_frame(kp_dict, joint_A, joint_B, joint_C, t) for t in range(T)])

    swing_shoulder_angle_seq = get_full_angle_series(processed_keypoints, f'{D}Hip', f'{D}Shoulder', f'{D}Elbow')
    swing_elbow_angle_seq = get_full_angle_series(processed_keypoints, f'{D}Shoulder', f'{D}Elbow', f'{D}Wrist')
    
    # Use optional hand key if available and valid, otherwise default to Wrist (making it a zero-length segment for angle calc)
    hand_key_for_wrist_angle = optional_hand_key if optional_hand_key in processed_keypoints else f'{D}Wrist'
    swing_wrist_angle_seq = get_full_angle_series(processed_keypoints, f'{D}Elbow', f'{D}Wrist', hand_key_for_wrist_angle)

    # 5. Proximal to Distal Sequence (挥拍期)
    # Calculate angular velocities (absolute change in angle per frame)
    # Prepend with NaN to keep array length T for easier indexing with swing_start_idx
    vel_shoulder_angle = np.abs(np.diff(swing_shoulder_angle_seq, prepend=np.nan))
    vel_elbow_angle = np.abs(np.diff(swing_elbow_angle_seq, prepend=np.nan))
    # vel_wrist_angle = np.abs(np.diff(swing_wrist_angle_seq, prepend=np.nan)) # Wrist part currently excluded

    swing_phase_slice = slice(swing_start_idx, swing_end_idx + 1)
    
    # Find time of peak angular velocity for shoulder and elbow in the swing phase
    # Ensure there are non-NaN values in the slice before calling nanargmax
    t_s_peak_vel = swing_start_idx # Default
    if vel_shoulder_angle[swing_phase_slice].size > 0 and not np.all(np.isnan(vel_shoulder_angle[swing_phase_slice])):
        t_s_peak_vel = np.nanargmax(vel_shoulder_angle[swing_phase_slice]) + swing_start_idx
        
    t_e_peak_vel = swing_start_idx # Default
    if vel_elbow_angle[swing_phase_slice].size > 0 and not np.all(np.isnan(vel_elbow_angle[swing_phase_slice])):
        t_e_peak_vel = np.nanargmax(vel_elbow_angle[swing_phase_slice]) + swing_start_idx
    
    result['proximal_to_distal_sequence'] = t_s_peak_vel <= t_e_peak_vel

    # 6. Hip Forward Shift (挥拍期)
    dom_hip_x = processed_keypoints[f'{D}Hip'][:, 0]
    nondom_hip_x = processed_keypoints[f'{ND}Hip'][:, 0]
    valid_hip_x_swing_mask = ~np.isnan(dom_hip_x[swing_phase_slice]) & ~np.isnan(nondom_hip_x[swing_phase_slice])
    
    if np.any(valid_hip_x_swing_mask):
        hip_center_x_swing = 0.5 * (dom_hip_x[swing_phase_slice][valid_hip_x_swing_mask] + nondom_hip_x[swing_phase_slice][valid_hip_x_swing_mask])
        if hip_center_x_swing.size > 0:
            hip_range = np.max(hip_center_x_swing) - np.min(hip_center_x_swing)
            # Assuming normalized coordinates (0-1). Threshold might need adjustment if coords are pixels.
            result['hip_forward_shift'] = hip_range >= 0.08 
            
    # 7. Trunk Rotation Completed (挥拍期)
    dom_shoulder_pts_swing = processed_keypoints[f'{D}Shoulder'][swing_phase_slice]
    nondom_shoulder_pts_swing = processed_keypoints[f'{ND}Shoulder'][swing_phase_slice]

    dist_shoulders_init_frame = processed_keypoints[f'{D}Shoulder'][swing_start_idx] - processed_keypoints[f'{ND}Shoulder'][swing_start_idx]
    if not np.isnan(dist_shoulders_init_frame).any():
        dist_init = np.linalg.norm(dist_shoulders_init_frame)
        
        min_dist_swing = float('inf')
        found_valid_dist_in_swing = False
        for t_swing_local in range(len(dom_shoulder_pts_swing)): # Iterate over local indices of swing phase slice
            d_s_pt = dom_shoulder_pts_swing[t_swing_local]
            nd_s_pt = nondom_shoulder_pts_swing[t_swing_local]
            if not np.isnan(d_s_pt).any() and not np.isnan(nd_s_pt).any():
                dist = np.linalg.norm(d_s_pt - nd_s_pt)
                min_dist_swing = min(min_dist_swing, dist)
                found_valid_dist_in_swing = True
        
        if found_valid_dist_in_swing and dist_init > 1e-6 : # Avoid division by zero
            result['trunk_rotation_completed'] = (dist_init - min_dist_swing) / dist_init >= 0.4
            
    return result


def align_keypoints_with_interpolation(
    joint_data_raw_lists: Dict[str, List[List[float]]], 
    frame_count: int
) -> Dict[str, np.ndarray]:
    """
    Aligns keypoints from lists of detected coordinates, using linear interpolation for missing frames.
    Handles 3D data (expects lists of [x,y,z]).
    
    Input: 
        joint_data_raw_lists: {'joint_name': [[x,y,z], [x,y,z], ...], ...}
                              where each inner list contains coordinates for frames where the joint was DETECTED.
                              The length of these inner lists can vary per joint.
    Output: 
        {'joint_name': np.ndarray of shape (frame_count, 3), ...}
        Array will contain NaNs for joints with no detections or frames that couldn't be interpolated.
    """
    keypoints_aligned: Dict[str, np.ndarray] = {}
    num_dimensions = 3  # X, Y, Z

    all_joint_names = list(joint_data_raw_lists.keys())

    for name in all_joint_names:
        points_for_joint = joint_data_raw_lists.get(name, [])
        
        # Convert list of detected points to a NumPy array.
        # If points_for_joint is empty, pts_detected_np will be empty.
        pts_detected_np = np.array(points_for_joint, dtype=float) # Shape (num_detections, 3)
        num_detections = pts_detected_np.shape[0]

        # Initialize the full_frames_data array for this joint with NaNs.
        full_frames_data = np.full((frame_count, num_dimensions), np.nan)

        if num_detections == 0:
            # No detections for this joint across all frames.
            print(f"INFO: No detections for joint '{name}'. It will be all NaNs for {frame_count} frames.")
            keypoints_aligned[name] = full_frames_data # Already all NaNs
            continue
        
        if num_detections == frame_count and not np.isnan(pts_detected_np).all():
            # All frames have detections for this joint (ideal case, or already interpolated).
            keypoints_aligned[name] = pts_detected_np
            continue

        # Interpolation logic:
        # We need to know at which of the `frame_count` frames these `num_detections` occurred.
        # The current input `joint_data_raw_lists` doesn't explicitly provide the original frame indices
        # for each detection. It's just a list of detected coordinates.
        # This implies an assumption: the `k`-th item in `points_for_joint` corresponds to the `k`-th
        # frame *in which this joint was visible*, not necessarily the `k`-th frame of the video.
        
        # To perform meaningful interpolation, we need a mapping from these sparse detections
        # to the global `frame_count`.
        #
        # Assumption made by the original `np.linspace` approach:
        # The `num_detections` are spread somewhat evenly across `frame_count`.
        # `detected_at_video_frames`: indices in the original video (0 to frame_count-1) where this joint was detected.
        
        # If `joint_data_raw_lists` actually means "for each frame, if joint detected, here's its coord, else it's missing from this frame's data"
        # then the server-side processing loop needs to change to build `joint_data_raw_lists` differently.
        # E.g. `joint_data_raw_lists[joint_name]` would be a list of length `frame_count`, with `None` or `[nan,nan,nan]` for missing.
        
        # Given the current structure of `joint_data_raw_lists` (list of detections, length != frame_count):
        # We'll use the `np.linspace` strategy to map these `num_detections` to `frame_count` indices.
        # This is a heuristic if original frame indices are lost.
        
        # Indices in the `full_frames_data` array where we have known (detected) data.
        # These are the "xp" for np.interp.
        known_data_indices = np.linspace(0, frame_count - 1, num=num_detections, dtype=int)
        
        # Remove duplicate indices from known_data_indices that might arise from linspace if num_detections is large
        # and map to the same integer frame index. We take the point corresponding to the first occurrence.
        unique_known_indices, first_occurrence_idx = np.unique(known_data_indices, return_index=True)
        
        # The actual detected points corresponding to these unique known_data_indices. These are "fp" for np.interp.
        points_at_unique_known_indices = pts_detected_np[first_occurrence_idx]

        for dim_idx in range(num_dimensions):
            # Values for the current dimension at the unique known indices
            dim_values_at_known = points_at_unique_known_indices[:, dim_idx]
            
            # Filter out any NaNs that might be in the detected data itself for this dimension
            valid_dim_mask = ~np.isnan(dim_values_at_known)
            
            if np.sum(valid_dim_mask) < 1: # No valid data points for this dimension for this joint
                # print(f"DEBUG: Joint '{name}', Dim {dim_idx}: No valid detected points. Stays NaN.")
                continue # full_frames_data for this dim remains NaN
            
            if np.sum(valid_dim_mask) == 1: # Only one valid data point
                # print(f"DEBUG: Joint '{name}', Dim {dim_idx}: Only one valid point. Filling all frames with it.")
                full_frames_data[:, dim_idx] = dim_values_at_known[valid_dim_mask][0]
                continue

            # At least two valid points exist for this dimension, proceed with interpolation
            interp_xp = unique_known_indices[valid_dim_mask]
            interp_fp = dim_values_at_known[valid_dim_mask]
            
            # Ensure interp_xp is sorted (should be by np.unique) and fp corresponds
            # np.interp requires xp to be increasing.
            sort_indices = np.argsort(interp_xp)
            interp_xp_sorted = interp_xp[sort_indices]
            interp_fp_sorted = interp_fp[sort_indices]
            
            # Final check for unique xp after sorting (in case of issues)
            final_unique_xp, final_unique_idx = np.unique(interp_xp_sorted, return_index=True)
            final_fp = interp_fp_sorted[final_unique_idx]

            if len(final_unique_xp) < 2: # Still not enough unique points after all filtering
                 if len(final_unique_xp) == 1:
                     full_frames_data[:, dim_idx] = final_fp[0]
                 # else: stays NaN
                 continue
            
            full_frames_data[:, dim_idx] = np.interp(
                np.arange(frame_count), # x-coordinates to evaluate interpolation at (all frames)
                final_unique_xp,        # xp: x-coordinates of known data points (must be increasing)
                final_fp                # fp: y-coordinates of known data points
            )
        keypoints_aligned[name] = full_frames_data
        
    return keypoints_aligned
