# court_transformation.py
import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("court_transformation")

# Official badminton court dimensions in meters
BADMINTON_COURT_LENGTH = 13.4  # meters
BADMINTON_COURT_WIDTH = 6.1    # meters (doubles)
BADMINTON_COURT_WIDTH_SINGLES = 5.18  # meters (singles)

# Court markings in meters from the net center
NET_TO_SHORT_SERVICE_LINE = 1.98
NET_TO_LONG_SERVICE_LINE_DOUBLES = 0.76  # from back boundary
SIDE_LINE_TO_CENTER_LINE = BADMINTON_COURT_WIDTH / 2

# Visualization parameters
TOP_DOWN_COURT_HEIGHT = 800  # pixels
TOP_DOWN_COURT_WIDTH = int(TOP_DOWN_COURT_HEIGHT * BADMINTON_COURT_WIDTH / BADMINTON_COURT_LENGTH)

def get_court_corners_from_keypoints(court_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Convert the 6 court key points to 4 corner points for perspective transformation.
    
    Court points order from detection:
    0: Lower-left (baseline + left sideline)
    1: Lower-right (baseline + right sideline)  
    2: Upper-left (baseline + left sideline)
    3: Upper-right (baseline + right sideline)
    4: Left-net intersection
    5: Right-net intersection
    
    Args:
        court_points: List of 6 court key points as (x, y) tuples
        
    Returns:
        List of 4 corner points: [bottom-left, bottom-right, top-right, top-left]
    """
    if len(court_points) != 6:
        raise ValueError(f"Expected 6 court points, got {len(court_points)}")
    
    # Court points order from detection (looking at the debug output):
    # The debug shows Y coordinates suggest:
    # Points 0,3: Y ~281-285 (smaller Y = higher on image = far baseline)
    # Points 1,2: Y ~506-511 (larger Y = lower on image = near baseline)
    
    # Let's identify corners based on actual Y coordinates (not assumptions)
    points_with_indices = [(i, court_points[i]) for i in range(4)]  # Only use first 4 points
    
    # Sort by Y coordinate to identify near vs far baselines
    points_by_y = sorted(points_with_indices, key=lambda x: x[1][1])
    
    # Far baseline (smaller Y): points_by_y[0] and points_by_y[1]
    # Near baseline (larger Y): points_by_y[2] and points_by_y[3]
    
    far_points = [points_by_y[0][1], points_by_y[1][1]]
    near_points = [points_by_y[2][1], points_by_y[3][1]]
    
    # Sort each baseline by X coordinate (left to right)
    far_points.sort(key=lambda p: p[0])  # [far_left, far_right]
    near_points.sort(key=lambda p: p[0])  # [near_left, near_right]
    
    # Assign corners for homography (perspective transformation)
    # In image coordinates: smaller Y = top, larger Y = bottom
    top_left = far_points[0]      # Far baseline, left side
    top_right = far_points[1]     # Far baseline, right side  
    bottom_left = near_points[0]  # Near baseline, left side
    bottom_right = near_points[1] # Near baseline, right side
    
    # DEBUG: Log the court corners for verification
    logger.info(f"Court corners extracted (corrected interpretation):")
    logger.info(f"  Top-left (far baseline, left): {top_left}")
    logger.info(f"  Top-right (far baseline, right): {top_right}")
    logger.info(f"  Bottom-left (near baseline, left): {bottom_left}")
    logger.info(f"  Bottom-right (near baseline, right): {bottom_right}")
    
    # Verify the corners make sense (bottom should have higher Y than top in image coordinates)
    if bottom_left[1] < top_left[1] or bottom_right[1] < top_right[1]:
        logger.warning("WARNING: Court corner Y coordinates still seem wrong after correction!")
        logger.warning(f"Bottom-left Y: {bottom_left[1]}, Top-left Y: {top_left[1]}")
        logger.warning(f"Bottom-right Y: {bottom_right[1]}, Top-right Y: {top_right[1]}")
    else:
        logger.info("✓ Court corner Y coordinates look correct: bottom > top")
    
    # Return in order expected by OpenCV getPerspectiveTransform: [top-left, top-right, bottom-right, bottom-left]
    corners = [top_left, top_right, bottom_right, bottom_left]
    logger.info(f"Final corner order for homography: {corners}")
    return corners

def create_top_down_court_template(width: int = TOP_DOWN_COURT_WIDTH, 
                                 height: int = TOP_DOWN_COURT_HEIGHT,
                                 court_type: str = 'doubles') -> np.ndarray:
    """
    Create a top-down badminton court template with proper markings.
    
    Args:
        width: Court template width in pixels
        height: Court template height in pixels  
        court_type: 'doubles' or 'singles'
        
    Returns:
        Court template image as numpy array
    """
    # Create blank court image (white background)
    court_img = np.zeros((height, width, 3), dtype=np.uint8)
    court_img[:] = (255, 255, 255)  # White background
    
    # Calculate court dimensions in pixels
    court_width_m = BADMINTON_COURT_WIDTH if court_type == 'doubles' else BADMINTON_COURT_WIDTH_SINGLES
    pixels_per_meter_x = width / court_width_m
    pixels_per_meter_y = height / BADMINTON_COURT_LENGTH
    
    # Line color (black)
    line_color = (0, 0, 0)
    line_thickness = 2
    
    # Court boundaries
    boundary_points = np.array([
        [0, 0],
        [width-1, 0], 
        [width-1, height-1],
        [0, height-1]
    ], dtype=np.int32)
    cv2.polylines(court_img, [boundary_points], True, line_color, line_thickness)
    
    # Net line (horizontal center)
    net_y = height // 2
    cv2.line(court_img, (0, net_y), (width-1, net_y), line_color, line_thickness)
    
    # Short service lines (1.98m from net on each side)
    short_service_offset_pixels = int(NET_TO_SHORT_SERVICE_LINE * pixels_per_meter_y)
    
    # Upper short service line
    upper_service_y = net_y - short_service_offset_pixels
    cv2.line(court_img, (0, upper_service_y), (width-1, upper_service_y), line_color, line_thickness)
    
    # Lower short service line  
    lower_service_y = net_y + short_service_offset_pixels
    cv2.line(court_img, (0, lower_service_y), (width-1, lower_service_y), line_color, line_thickness)
    
    # Long service lines for doubles (0.76m from back boundary)
    if court_type == 'doubles':
        long_service_offset_pixels = int(NET_TO_LONG_SERVICE_LINE_DOUBLES * pixels_per_meter_y)
        
        # Upper long service line
        upper_long_service_y = long_service_offset_pixels
        cv2.line(court_img, (0, upper_long_service_y), (width-1, upper_long_service_y), line_color, line_thickness)
        
        # Lower long service line
        lower_long_service_y = height - 1 - long_service_offset_pixels
        cv2.line(court_img, (0, lower_long_service_y), (width-1, lower_long_service_y), line_color, line_thickness)
    
    # Center line (vertical) - CORRECTED: should not cross the net
    center_x = width // 2
    
    # Draw center line in two segments: from baselines to service lines (not crossing net)
    # Upper segment: from top boundary to upper short service line
    cv2.line(court_img, (center_x, 0), (center_x, upper_service_y), line_color, line_thickness)
    
    # Lower segment: from lower short service line to bottom boundary  
    cv2.line(court_img, (center_x, lower_service_y), (center_x, height-1), line_color, line_thickness)
    
    # Service court divisions for doubles
    if court_type == 'doubles':
        # Side service lines (for singles within doubles court)
        singles_court_width_pixels = int(BADMINTON_COURT_WIDTH_SINGLES * pixels_per_meter_x)
        singles_offset = (width - singles_court_width_pixels) // 2
        
        # Left singles service line
        cv2.line(court_img, (singles_offset, 0), (singles_offset, height-1), line_color, line_thickness)
        
        # Right singles service line
        cv2.line(court_img, (width - singles_offset, 0), (width - singles_offset, height-1), line_color, line_thickness)
    
    return court_img

def get_perspective_transformation_matrix(court_corners: List[Tuple[float, float]], 
                                        output_width: int = TOP_DOWN_COURT_WIDTH,
                                        output_height: int = TOP_DOWN_COURT_HEIGHT) -> np.ndarray:
    """
    Calculate perspective transformation matrix from detected court corners to top-down view.
    
    Args:
        court_corners: 4 corner points [bottom-left, bottom-right, top-right, top-left]
        output_width: Target top-down view width in pixels
        output_height: Target top-down view height in pixels
        
    Returns:
        Perspective transformation matrix (3x3)
    """
    # Source points (detected court corners)
    src_points = np.float32(court_corners)
    
    # Destination points (top-down view corners)
    # Source corners are in order: [top-left, top-right, bottom-right, bottom-left]
    # Map to a rectangle where:
    # - (0, 0) is top-left of the court in top-down view
    # - (width, height) is bottom-right of the court in top-down view
    dst_points = np.float32([
        [0, 0],                       # top-left → top-left of top-down
        [output_width, 0],            # top-right → top-right of top-down  
        [output_width, output_height], # bottom-right → bottom-right of top-down
        [0, output_height]            # bottom-left → bottom-left of top-down
    ])
    
    logger.info(f"Homography transformation setup:")
    logger.info(f"Source points (image coordinates): {court_corners}")
    logger.info(f"Destination points (top-down): {dst_points.tolist()}")
    logger.info(f"Output dimensions: {output_width}x{output_height}")
    
    # Calculate perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    logger.info(f"Transformation matrix:")
    for i, row in enumerate(transform_matrix):
        logger.info(f"  Row {i}: [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    
    # Test the transformation with the source corners
    logger.info("Testing transformation with source corners:")
    for i, src_point in enumerate(court_corners):
        transformed = transform_point_to_top_down(src_point, transform_matrix)
        expected = dst_points[i]
        logger.info(f"  {src_point} → {transformed} (expected: {expected})")
        
        # Check if transformation is working correctly
        error = np.sqrt((transformed[0] - expected[0])**2 + (transformed[1] - expected[1])**2)
        if error > 1.0:  # More than 1 pixel error
            logger.warning(f"    ERROR: {error:.2f} pixel error in transformation!")
    
    return transform_matrix

def transform_point_to_top_down(point: Tuple[float, float], 
                               transform_matrix: np.ndarray) -> Tuple[float, float]:
    """
    Transform a 2D point from the original video perspective to top-down court view.
    
    Args:
        point: (x, y) coordinates in original video
        transform_matrix: Perspective transformation matrix
        
    Returns:
        Transformed (x, y) coordinates in top-down view
    """
    # Convert point to homogeneous coordinates
    src_point = np.array([[point[0], point[1]]], dtype=np.float32)
    src_point = src_point.reshape(-1, 1, 2)
    
    # Apply perspective transformation
    dst_point = cv2.perspectiveTransform(src_point, transform_matrix)
    
    # Extract transformed coordinates
    x, y = dst_point[0][0]
    return float(x), float(y)

def transform_video_frame_to_top_down(frame: np.ndarray, 
                                    transform_matrix: np.ndarray,
                                    output_width: int = TOP_DOWN_COURT_WIDTH,
                                    output_height: int = TOP_DOWN_COURT_HEIGHT) -> np.ndarray:
    """
    Transform a video frame from perspective view to top-down court view.
    
    Args:
        frame: Input video frame
        transform_matrix: Perspective transformation matrix
        output_width: Target width in pixels
        output_height: Target height in pixels
        
    Returns:
        Transformed top-down view frame
    """
    return cv2.warpPerspective(frame, transform_matrix, (output_width, output_height))

def calculate_real_world_distance(point1: Tuple[float, float], 
                                point2: Tuple[float, float],
                                court_width_pixels: int = TOP_DOWN_COURT_WIDTH,
                                court_height_pixels: int = TOP_DOWN_COURT_HEIGHT) -> float:
    """
    Calculate real-world distance in meters between two points in top-down court view.
    
    Args:
        point1: First point (x, y) in top-down court pixels
        point2: Second point (x, y) in top-down court pixels  
        court_width_pixels: Court width in pixels
        court_height_pixels: Court height in pixels
        
    Returns:
        Distance in meters
    """
    # Calculate pixel distance
    pixel_distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    # Convert to real-world distance using court dimensions
    meters_per_pixel_x = BADMINTON_COURT_WIDTH / court_width_pixels
    meters_per_pixel_y = BADMINTON_COURT_LENGTH / court_height_pixels
    
    # Use average scale factor for distance calculation
    avg_meters_per_pixel = (meters_per_pixel_x + meters_per_pixel_y) / 2
    
    real_distance = pixel_distance * avg_meters_per_pixel
    
    return real_distance

def create_player_trail_visualization(court_template: np.ndarray,
                                    player_positions: List[List[Tuple[float, float]]],
                                    colors: Optional[List[Tuple[int, int, int]]] = None,
                                    current_distances: Optional[List[float]] = None) -> np.ndarray:
    """
    Create visualization of player movements on the court template with smooth trails.
    
    Args:
        court_template: Base court template image
        player_positions: List of position lists for each player [(x1,y1), (x2,y2), ...]
        colors: Optional list of colors for each player
        current_distances: Optional list of current distances moved for each player
        
    Returns:
        Court image with player trails overlaid
    """
    result_img = court_template.copy()
    
    # Default colors for players (darker colors for white background, BGR format)
    if colors is None:
        colors = [
            (0, 0, 180),    # Red for player 1
            (180, 0, 0),    # Blue for player 2
            (0, 120, 0),    # Green for player 3
            (0, 120, 180),  # Orange for player 4
        ]
    
    for player_idx, positions in enumerate(player_positions):
        if not positions or len(positions) < 2:
            continue
            
        color = colors[player_idx % len(colors)]
        
        # Create smooth trail using cubic interpolation for better smoothness
        if len(positions) >= 4:
            # Convert positions to numpy array for processing
            pos_array = np.array(positions)
            
            # Create smooth path using spline interpolation
            from scipy import interpolate
            try:
                # Create parameter array for interpolation
                t = np.linspace(0, 1, len(positions))
                t_smooth = np.linspace(0, 1, len(positions) * 3)  # 3x more points for smoothness
                
                # Interpolate x and y coordinates separately
                fx = interpolate.interp1d(t, pos_array[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
                fy = interpolate.interp1d(t, pos_array[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
                
                smooth_x = fx(t_smooth)
                smooth_y = fy(t_smooth)
                
                # Draw smooth trail with varying thickness
                for i in range(1, len(smooth_x)):
                    progress = i / len(smooth_x)
                    thickness = max(1, int(3 * progress))  # Gradually increase thickness
                    alpha = 0.4 + 0.6 * progress  # Gradually increase opacity
                    
                    pt1 = (int(smooth_x[i-1]), int(smooth_y[i-1]))
                    pt2 = (int(smooth_x[i]), int(smooth_y[i]))
                    
                    # Create overlay for transparency effect
                    overlay = result_img.copy()
                    cv2.line(overlay, pt1, pt2, color, thickness)
                    cv2.addWeighted(result_img, 1 - alpha * 0.3, overlay, alpha * 0.3, 0, result_img)
                    
            except ImportError:
                # Fallback to simple line drawing if scipy not available
                for i in range(1, len(positions)):
                    pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                    pt2 = (int(positions[i][0]), int(positions[i][1]))
                    cv2.line(result_img, pt1, pt2, color, 3)
        else:
            # Simple line drawing for few points
            for i in range(1, len(positions)):
                pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                pt2 = (int(positions[i][0]), int(positions[i][1]))
                cv2.line(result_img, pt1, pt2, color, 3)
        
        # Draw current position with label
        if positions:
            current_pos = positions[-1]
            cv2.circle(result_img, (int(current_pos[0]), int(current_pos[1])), 8, color, -1)
            cv2.circle(result_img, (int(current_pos[0]), int(current_pos[1])), 10, (0, 0, 0), 2)
            
            # Add player label with distance if provided
            if current_distances and player_idx < len(current_distances):
                distance = current_distances[player_idx]
                label = f"P{player_idx + 1}: {distance:.1f}m"
            else:
                label = f"P{player_idx + 1}"
            
            cv2.putText(result_img, label, (int(current_pos[0]) + 15, int(current_pos[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return result_img