# shuttlecock_tracking.py
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger("shuttlecock_tracking")

@dataclass 
class TrackSegment:
    """Represents a continuous track segment with quality metrics"""
    indices: List[int]
    positions: List[Tuple[float, float]]
    confidences: List[float]
    frames: List[int]
    quality_score: float = 0.0
    
    def __len__(self):
        return len(self.indices)
    
    def duration_frames(self) -> int:
        return max(self.frames) - min(self.frames) + 1 if self.frames else 0

def build_track_segments(pred_dict: dict, max_gap_frames: int = 10, min_confidence: float = 0.6) -> List[TrackSegment]:
    """
    Build track segments from prediction dictionary with enhanced quality validation.
    
    Args:
        pred_dict: Prediction dictionary with Frame, X, Y, Visibility
        max_gap_frames: Maximum frame gap within a segment
        min_confidence: Minimum confidence threshold
    
    Returns:
        List of TrackSegment objects
    """
    if not pred_dict['Frame']:
        return []
    
    # Filter by confidence first
    valid_detections = []
    for i in range(len(pred_dict['Frame'])):
        visibility = float(pred_dict['Visibility'][i]) if pred_dict['Visibility'][i] != '' else 0
        x = float(pred_dict['X'][i]) if pred_dict['X'][i] != '' else 0
        y = float(pred_dict['Y'][i]) if pred_dict['Y'][i] != '' else 0
        frame = int(pred_dict['Frame'][i]) if str(pred_dict['Frame'][i]).isdigit() else i
        
        if visibility >= min_confidence and x > 0 and y > 0:
            valid_detections.append((i, frame, x, y, visibility))
    
    if not valid_detections:
        return []
    
    # Sort by frame number
    valid_detections.sort(key=lambda x: x[1])
    
    # Group into segments
    segments = []
    current_segment = []
    
    for detection in valid_detections:
        idx, frame, x, y, visibility = detection
        
        if not current_segment:
            current_segment = [detection]
        else:
            last_frame = current_segment[-1][1]
            if frame - last_frame <= max_gap_frames:
                current_segment.append(detection)
            else:
                # Start new segment
                if len(current_segment) >= 3:  # Minimum 3 detections per segment
                    segments.append(current_segment)
                current_segment = [detection]
    
    # Add final segment
    if len(current_segment) >= 3:
        segments.append(current_segment)
    
    # Convert to TrackSegment objects
    track_segments = []
    for segment in segments:
        indices = [det[0] for det in segment]
        positions = [(det[2], det[3]) for det in segment]
        confidences = [det[4] for det in segment]
        frames = [det[1] for det in segment]
        
        track_segments.append(TrackSegment(indices, positions, confidences, frames))
    
    logger.info(f"Built {len(track_segments)} track segments from {len(valid_detections)} valid detections")
    return track_segments

def calculate_track_quality_score(segment: TrackSegment, fps: float = 30.0, video_width: int = 1920, video_height: int = 1080) -> float:
    """
    Calculate comprehensive quality score for a track segment.
    
    Args:
        segment: TrackSegment to evaluate
        fps: Video frame rate
        video_width: Video width for pixel-to-meter conversion estimation
        video_height: Video height for pixel-to-meter conversion estimation
    
    Returns:
        Quality score (0-100, higher is better)
    """
    if len(segment) < 2:
        return 0.0
    
    scores = {}
    
    # 1. Persistence Score (0-25 points): Longer tracks are better
    duration_seconds = segment.duration_frames() / fps
    scores['persistence'] = min(25.0, duration_seconds * 10.0)  # Up to 2.5 seconds = full points
    
    # 2. Confidence Score (0-20 points): Higher average confidence is better  
    avg_confidence = np.mean(segment.confidences)
    scores['confidence'] = avg_confidence * 20.0
    
    # 3. Smoothness Score (0-25 points): Smooth trajectories are better
    velocities = []
    for i in range(1, len(segment.positions)):
        dx = segment.positions[i][0] - segment.positions[i-1][0]
        dy = segment.positions[i][1] - segment.positions[i-1][1]
        velocity = np.sqrt(dx*dx + dy*dy)
        velocities.append(velocity)
    
    if velocities:
        velocity_std = np.std(velocities)
        # Lower standard deviation = smoother = higher score
        # Normalize by typical video dimensions
        smoothness_normalized = 1.0 - min(1.0, velocity_std / (video_width * 0.05))
        scores['smoothness'] = smoothness_normalized * 25.0
    else:
        scores['smoothness'] = 0.0
    
    # 4. Physics Realism Score (0-20 points): Realistic velocities and accelerations
    physics_score = 20.0
    
    # Estimate pixel-to-meter conversion (rough approximation)
    # Assume badminton court is ~13.4m wide, occupies ~60% of video width
    pixels_per_meter = (video_width * 0.6) / 13.4
    
    for velocity in velocities:
        # Convert to m/s 
        velocity_ms = (velocity * fps) / pixels_per_meter
        
        # Realistic shuttlecock speeds: 0-50 m/s (professional can reach 100+ m/s)
        if velocity_ms > 60.0:  # Unrealistic speed
            physics_score -= 5.0
        elif velocity_ms > 40.0:  # High but possible
            physics_score -= 1.0
    
    # Check for impossible acceleration changes
    if len(velocities) >= 2:
        for i in range(1, len(velocities)):
            if velocities[i-1] > 0:
                accel_ratio = velocities[i] / velocities[i-1]
                if accel_ratio > 4.0 or accel_ratio < 0.25:  # 4x speed change per frame is unrealistic
                    physics_score -= 3.0
    
    scores['physics'] = max(0.0, physics_score)
    
    # 5. Temporal Consistency Score (0-10 points): Consistent frame progression
    frame_gaps = []
    for i in range(1, len(segment.frames)):
        frame_gaps.append(segment.frames[i] - segment.frames[i-1])
    
    if frame_gaps:
        gap_consistency = 1.0 - min(1.0, np.std(frame_gaps) / 5.0)  # Penalize irregular gaps
        scores['temporal'] = gap_consistency * 10.0
    else:
        scores['temporal'] = 10.0
    
    total_score = sum(scores.values())
    
    # Log detailed scoring for debugging
    logger.debug(f"Track segment quality: {scores}, total: {total_score:.1f}")
    
    return total_score

def resolve_overlapping_segments(segments: List[TrackSegment], max_overlap_frames: int = 5) -> List[TrackSegment]:
    """
    Resolve overlapping track segments by selecting the highest quality ones.
    
    Args:
        segments: List of TrackSegment objects
        max_overlap_frames: Maximum allowed frame overlap
    
    Returns:
        List of non-overlapping TrackSegment objects
    """
    if len(segments) <= 1:
        return segments
    
    # Sort by quality score (highest first)
    sorted_segments = sorted(segments, key=lambda x: x.quality_score, reverse=True)
    
    selected_segments = []
    
    for candidate in sorted_segments:
        # Check if this candidate overlaps significantly with any already selected segment
        overlaps = False
        
        for selected in selected_segments:
            # Check frame overlap
            candidate_frames = set(candidate.frames)
            selected_frames = set(selected.frames)
            overlap_frames = len(candidate_frames.intersection(selected_frames))
            
            if overlap_frames > max_overlap_frames:
                overlaps = True
                break
        
        if not overlaps:
            selected_segments.append(candidate)
    
    # Sort by frame order for final output
    selected_segments.sort(key=lambda x: min(x.frames))
    
    logger.info(f"Overlap resolution: {len(segments)} -> {len(selected_segments)} segments")
    return selected_segments

def filter_stationary_sequences(pred_dict: dict, min_movement_threshold: float = 5.0, max_stationary_frames: int = 15) -> dict:
    """
    Remove sequences where shuttlecock remains stationary for too long.
    
    Args:
        pred_dict: Prediction dictionary with Frame, X, Y, Visibility keys
        min_movement_threshold: Minimum pixel movement to consider as motion (default: 5.0)
        max_stationary_frames: Maximum consecutive frames without significant movement (default: 15)
    
    Returns:
        Filtered prediction dictionary
    """
    if not pred_dict['Frame']:
        return pred_dict
    
    filtered_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    stationary_count = 0
    last_x, last_y = None, None
    
    for i in range(len(pred_dict['Frame'])):
        x = float(pred_dict['X'][i]) if pred_dict['X'][i] != '' else 0
        y = float(pred_dict['Y'][i]) if pred_dict['Y'][i] != '' else 0
        visibility = float(pred_dict['Visibility'][i]) if pred_dict['Visibility'][i] != '' else 0
        
        # Skip if not visible
        if visibility < 0.5 or x <= 0 or y <= 0:
            stationary_count = 0
            last_x, last_y = None, None
            continue
        
        # Calculate movement from last position
        if last_x is not None and last_y is not None:
            movement = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            
            if movement < min_movement_threshold:
                stationary_count += 1
            else:
                stationary_count = 0
        else:
            stationary_count = 0
        
        # Only include if not stationary for too long
        if stationary_count <= max_stationary_frames:
            filtered_dict['Frame'].append(pred_dict['Frame'][i])
            filtered_dict['X'].append(pred_dict['X'][i])
            filtered_dict['Y'].append(pred_dict['Y'][i])
            filtered_dict['Visibility'].append(pred_dict['Visibility'][i])
        
        last_x, last_y = x, y
    
    logger.info(f"Stationary filtering: {len(pred_dict['Frame'])} -> {len(filtered_dict['Frame'])} frames")
    return filtered_dict

def group_rally_segments(pred_dict: dict, max_gap_frames: int = 30) -> List[List[int]]:
    """
    Group consecutive visible detections into rally segments.
    
    Args:
        pred_dict: Prediction dictionary with Frame, X, Y, Visibility keys
        max_gap_frames: Maximum frame gap to still consider same rally (default: 30)
    
    Returns:
        List of rally segments, each containing list of frame indices
    """
    if not pred_dict['Frame']:
        return []
    
    rally_segments = []
    current_segment = []
    last_frame = None
    
    for i in range(len(pred_dict['Frame'])):
        frame_num = int(pred_dict['Frame'][i]) if str(pred_dict['Frame'][i]).isdigit() else i
        visibility = float(pred_dict['Visibility'][i]) if pred_dict['Visibility'][i] != '' else 0
        x = float(pred_dict['X'][i]) if pred_dict['X'][i] != '' else 0
        y = float(pred_dict['Y'][i]) if pred_dict['Y'][i] != '' else 0
        
        # Skip if not visible or invalid position
        if visibility < 0.5 or x <= 0 or y <= 0:
            continue
        
        # Check if this continues the current segment
        if last_frame is None or (frame_num - last_frame) <= max_gap_frames:
            current_segment.append(i)
        else:
            # Start new segment
            if current_segment:
                rally_segments.append(current_segment)
            current_segment = [i]
        
        last_frame = frame_num
    
    # Add final segment
    if current_segment:
        rally_segments.append(current_segment)
    
    logger.info(f"Grouped into {len(rally_segments)} rally segments")
    return rally_segments

def filter_short_tracks(pred_dict: dict, rally_segments: List[List[int]], min_rally_duration_frames: int = 15) -> dict:
    """
    Remove rally segments that are too short to be meaningful.
    
    Args:
        pred_dict: Prediction dictionary
        rally_segments: List of rally segments from group_rally_segments()
        min_rally_duration_frames: Minimum frames for a valid rally (default: 15, ~0.5s at 30fps)
    
    Returns:
        Filtered prediction dictionary
    """
    if not rally_segments:
        return {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    # Keep only segments that meet minimum duration
    valid_segments = [seg for seg in rally_segments if len(seg) >= min_rally_duration_frames]
    
    # Flatten valid segment indices
    valid_indices = []
    for segment in valid_segments:
        valid_indices.extend(segment)
    
    # Build filtered dictionary
    filtered_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    for i in sorted(valid_indices):
        filtered_dict['Frame'].append(pred_dict['Frame'][i])
        filtered_dict['X'].append(pred_dict['X'][i])
        filtered_dict['Y'].append(pred_dict['Y'][i])
        filtered_dict['Visibility'].append(pred_dict['Visibility'][i])
    
    logger.info(f"Short track filtering: {len(rally_segments)} -> {len(valid_segments)} segments, {len(pred_dict['Frame'])} -> {len(filtered_dict['Frame'])} frames")
    return filtered_dict

def validate_trajectory_realism(pred_dict: dict, rally_segments: List[List[int]], max_speed_pixels_per_frame: float = 50.0, max_acceleration_ratio: float = 3.0) -> dict:
    """
    Remove rally segments with unrealistic physics (impossible speeds/accelerations).
    
    Args:
        pred_dict: Prediction dictionary
        rally_segments: List of rally segments
        max_speed_pixels_per_frame: Maximum realistic speed in pixels per frame (default: 50.0)
        max_acceleration_ratio: Maximum acceleration change ratio between consecutive frame pairs (default: 3.0)
    
    Returns:
        Filtered prediction dictionary with realistic trajectories only
    """
    if not rally_segments:
        return {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    valid_segments = []
    
    for segment in rally_segments:
        if len(segment) < 3:  # Need at least 3 points to check acceleration
            continue
        
        # Extract positions for this segment
        positions = []
        for i in segment:
            x = float(pred_dict['X'][i]) if pred_dict['X'][i] != '' else 0
            y = float(pred_dict['Y'][i]) if pred_dict['Y'][i] != '' else 0
            if x > 0 and y > 0:
                positions.append((x, y))
        
        if len(positions) < 3:
            continue
        
        # Check if trajectory is realistic
        is_realistic = True
        speeds = []
        
        # Calculate speeds between consecutive points
        for j in range(1, len(positions)):
            dx = positions[j][0] - positions[j-1][0]
            dy = positions[j][1] - positions[j-1][1]
            speed = np.sqrt(dx*dx + dy*dy)
            speeds.append(speed)
            
            # Check for impossible speeds
            if speed > max_speed_pixels_per_frame:
                is_realistic = False
                break
        
        # Check acceleration changes if speeds are reasonable
        if is_realistic and len(speeds) >= 2:
            for j in range(1, len(speeds)):
                if speeds[j-1] > 0:  # Avoid division by zero
                    accel_ratio = speeds[j] / speeds[j-1]
                    # Check for impossible acceleration changes
                    if accel_ratio > max_acceleration_ratio or accel_ratio < (1.0 / max_acceleration_ratio):
                        is_realistic = False
                        break
        
        if is_realistic:
            valid_segments.append(segment)
    
    # Build filtered dictionary from valid segments
    valid_indices = []
    for segment in valid_segments:
        valid_indices.extend(segment)
    
    filtered_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    for i in sorted(valid_indices):
        filtered_dict['Frame'].append(pred_dict['Frame'][i])
        filtered_dict['X'].append(pred_dict['X'][i])
        filtered_dict['Y'].append(pred_dict['Y'][i])
        filtered_dict['Visibility'].append(pred_dict['Visibility'][i])
    
    logger.info(f"Physics validation: {len(rally_segments)} -> {len(valid_segments)} segments, {len(filtered_dict['Frame'])} frames remain")
    return filtered_dict

def apply_noise_filtering(pred_dict: dict, fps: float = 30.0, video_width: int = 1920, video_height: int = 1080) -> dict:
    """
    Apply enhanced noise filtering to shuttlecock tracking data using track quality scoring.
    
    Args:
        pred_dict: Raw prediction dictionary from TrackNetV3
        fps: Video frame rate for time-based calculations (default: 30.0)
        video_width: Video width for physics calculations (default: 1920)
        video_height: Video height for physics calculations (default: 1080)
    
    Returns:
        Filtered prediction dictionary with noise removed using quality-based selection
    """
    logger.info(f"Starting enhanced noise filtering on {len(pred_dict['Frame'])} total detections")
    
    if not pred_dict['Frame']:
        return {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    # Step 1: Build track segments with confidence filtering
    track_segments = build_track_segments(
        pred_dict,
        max_gap_frames=max(3, int(fps * 0.2)),  # 0.2 seconds max gap within segment
        min_confidence=0.6  # Higher confidence threshold
    )
    
    if not track_segments:
        logger.warning("No valid track segments found")
        return {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    logger.info(f"Built {len(track_segments)} initial track segments")
    
    # Step 2: Calculate quality scores for each segment
    for segment in track_segments:
        segment.quality_score = calculate_track_quality_score(
            segment, fps, video_width, video_height
        )
    
    # Step 3: Filter segments by minimum quality threshold
    min_quality_threshold = 40.0  # Out of 100 points
    quality_filtered_segments = [s for s in track_segments if s.quality_score >= min_quality_threshold]
    
    logger.info(f"Quality filtering: {len(track_segments)} -> {len(quality_filtered_segments)} segments (threshold: {min_quality_threshold})")
    
    if not quality_filtered_segments:
        logger.warning("No segments meet minimum quality threshold")
        return {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    # Step 4: Resolve overlapping segments by selecting highest quality ones
    final_segments = resolve_overlapping_segments(
        quality_filtered_segments,
        max_overlap_frames=5  # Allow up to 5 frame overlap
    )
    
    # Step 5: Additional length filtering for final segments
    min_duration_frames = max(5, int(fps * 0.3))  # At least 0.3 seconds or 5 frames
    final_segments = [s for s in final_segments if len(s) >= min_duration_frames]
    
    logger.info(f"Final segment count: {len(final_segments)}")
    
    # Step 6: Reconstruct prediction dictionary from selected segments
    filtered_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    
    # Collect all indices from final segments and sort by frame order
    all_indices = []
    for segment in final_segments:
        all_indices.extend(segment.indices)
    
    all_indices.sort(key=lambda i: int(pred_dict['Frame'][i]) if str(pred_dict['Frame'][i]).isdigit() else i)
    
    # Build filtered dictionary
    for idx in all_indices:
        filtered_dict['Frame'].append(pred_dict['Frame'][idx])
        filtered_dict['X'].append(pred_dict['X'][idx])  
        filtered_dict['Y'].append(pred_dict['Y'][idx])
        filtered_dict['Visibility'].append(pred_dict['Visibility'][idx])
    
    # Log quality scores for debugging
    if final_segments:
        quality_scores = [s.quality_score for s in final_segments]
        logger.info(f"Selected segment quality scores: min={min(quality_scores):.1f}, max={max(quality_scores):.1f}, avg={np.mean(quality_scores):.1f}")
    
    logger.info(f"Enhanced noise filtering complete: {len(pred_dict['Frame'])} -> {len(filtered_dict['Frame'])} frames retained")
    return filtered_dict

def test_enhanced_filtering():
    """
    Test the enhanced noise filtering algorithm with synthetic data.
    """
    print("Testing enhanced shuttlecock noise filtering...")
    
    # Create synthetic test data with noise and good trajectory
    test_data = {
        'Frame': [],
        'X': [],
        'Y': [],
        'Visibility': []
    }
    
    # Add a good trajectory (smooth arc)
    for i in range(30):
        frame = i
        x = 100 + i * 10  # Moving right
        y = 200 + 50 * np.sin(i * 0.2)  # Smooth arc
        visibility = 0.8 + 0.1 * np.random.random()  # High confidence
        
        test_data['Frame'].append(frame)
        test_data['X'].append(x)
        test_data['Y'].append(y)
        test_data['Visibility'].append(visibility)
    
    # Add some noise (single frame detections with low confidence)
    for i in range(5):
        frame = 40 + i * 3
        x = np.random.randint(0, 1000)  # Random position
        y = np.random.randint(0, 800)
        visibility = 0.3  # Low confidence
        
        test_data['Frame'].append(frame)
        test_data['X'].append(x)
        test_data['Y'].append(y)
        test_data['Visibility'].append(visibility)
    
    # Add another good trajectory
    for i in range(20):
        frame = 60 + i
        x = 500 - i * 5  # Moving left
        y = 300 + 30 * np.cos(i * 0.3)  # Smooth trajectory
        visibility = 0.9
        
        test_data['Frame'].append(frame)
        test_data['X'].append(x)
        test_data['Y'].append(y)
        test_data['Visibility'].append(visibility)
    
    print(f"Test data: {len(test_data['Frame'])} total detections")
    
    # Apply enhanced filtering
    filtered_data = apply_noise_filtering(test_data, fps=30.0, video_width=1000, video_height=800)
    
    print(f"Filtered data: {len(filtered_data['Frame'])} detections retained")
    print("Enhanced filtering test completed successfully!")
    
    return len(filtered_data['Frame']) > 0  # Should retain some good data

if __name__ == "__main__":
    test_enhanced_filtering()