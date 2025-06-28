# movement_optimization.py
import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Dict
from scipy import interpolate
from scipy.ndimage import median_filter

logger = logging.getLogger("movement_optimization")

class PlayerTracker:
    """Advanced player movement tracking with smoothing and validation."""
    
    def __init__(self, court_width: int, court_height: int, max_speed_mps: float = 8.0, fps: float = 30.0):
        """
        Initialize player tracker.
        
        Args:
            court_width: Court width in pixels (top-down view)
            court_height: Court height in pixels (top-down view) 
            max_speed_mps: Maximum realistic player speed in meters/second
            fps: Video frame rate
        """
        self.court_width = court_width
        self.court_height = court_height
        self.max_speed_mps = max_speed_mps
        self.fps = fps
        
        # Net line is at the center of the court (vertical line)
        self.net_x = court_width / 2
        
        # Maximum distance per frame (in pixels, approximate)
        # Convert from m/s to pixels/frame using court dimensions
        court_length_m = 13.4  # meters
        court_width_m = 6.1   # meters
        pixels_per_meter = court_height / court_length_m
        self.max_distance_per_frame = (max_speed_mps / fps) * pixels_per_meter
        
        # Player tracking state
        self.player_sides = {}  # {player_id: 'left' or 'right'}
        self.player_last_valid_positions = {}  # {player_id: (x, y)}
        self.player_position_history = {}  # {player_id: [(x, y), ...]}
        
        logger.info(f"PlayerTracker initialized: court {court_width}x{court_height}, max_speed {max_speed_mps}m/s")
        logger.info(f"Net line at x={self.net_x:.1f}, max_distance_per_frame={self.max_distance_per_frame:.1f}px")
    
    def determine_player_side(self, player_id: int, position: Tuple[float, float]) -> str:
        """Determine which side of the court a player is on."""
        x, y = position
        return 'left' if x < self.net_x else 'right'
    
    def is_net_crossing(self, player_id: int, new_position: Tuple[float, float]) -> bool:
        """Check if a position would cause illegal net crossing."""
        if player_id not in self.player_sides:
            return False
            
        current_side = self.player_sides[player_id]
        new_side = self.determine_player_side(player_id, new_position)
        
        return current_side != new_side
    
    def is_realistic_movement(self, player_id: int, new_position: Tuple[float, float]) -> bool:
        """Check if movement speed is realistic."""
        if player_id not in self.player_last_valid_positions:
            return True
            
        last_pos = self.player_last_valid_positions[player_id]
        distance = np.sqrt((new_position[0] - last_pos[0])**2 + (new_position[1] - last_pos[1])**2)
        
        return distance <= self.max_distance_per_frame
    
    def is_position_valid(self, position: Tuple[float, float]) -> bool:
        """Check if position is within court bounds."""
        x, y = position
        return 0 <= x <= self.court_width and 0 <= y <= self.court_height
    
    def validate_and_filter_position(self, player_id: int, new_position: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Validate a new position and return filtered/corrected position if valid.
        
        Returns:
            Valid position tuple or None if position should be rejected
        """
        # Check basic bounds
        if not self.is_position_valid(new_position):
            logger.debug(f"Player {player_id}: Position {new_position} out of bounds")
            return None
        
        # Check realistic movement speed
        if not self.is_realistic_movement(player_id, new_position):
            logger.debug(f"Player {player_id}: Unrealistic movement to {new_position}")
            return None
        
        # Check net crossing (only after player side is established)
        if self.is_net_crossing(player_id, new_position):
            logger.debug(f"Player {player_id}: Illegal net crossing prevented at {new_position}")
            return None
        
        # Position is valid
        return new_position
    
    def update_player_position(self, player_id: int, position: Tuple[float, float]) -> bool:
        """
        Update player position with validation.
        
        Returns:
            True if position was accepted, False if rejected
        """
        validated_position = self.validate_and_filter_position(player_id, position)
        
        if validated_position is None:
            return False
        
        # Initialize player tracking if first valid position
        if player_id not in self.player_sides:
            self.player_sides[player_id] = self.determine_player_side(player_id, validated_position)
            self.player_position_history[player_id] = []
            logger.info(f"Player {player_id} initialized on {self.player_sides[player_id]} side")
        
        # Update tracking state
        self.player_last_valid_positions[player_id] = validated_position
        self.player_position_history[player_id].append(validated_position)
        
        return True
    
    def get_smoothed_trajectory(self, player_id: int, smoothing_window: int = 15) -> List[Tuple[float, float]]:
        """
        Get smoothed trajectory for a player using advanced filtering.
        
        Args:
            player_id: Player identifier
            smoothing_window: Window size for smoothing filter
            
        Returns:
            List of smoothed position tuples
        """
        if player_id not in self.player_position_history:
            return []
        
        positions = self.player_position_history[player_id]
        if len(positions) < 3:
            return positions
        
        # Convert to numpy arrays
        x_coords = np.array([pos[0] for pos in positions])
        y_coords = np.array([pos[1] for pos in positions])
        
        # Apply median filter to remove outliers
        if len(positions) >= 5:
            x_coords = median_filter(x_coords, size=min(5, len(positions)))
            y_coords = median_filter(y_coords, size=min(5, len(positions)))
        
        # Apply moving average for smoothing
        if len(positions) >= smoothing_window:
            # Use convolution for efficient moving average
            kernel = np.ones(smoothing_window) / smoothing_window
            x_coords = np.convolve(x_coords, kernel, mode='same')
            y_coords = np.convolve(y_coords, kernel, mode='same')
        
        # Recombine into position tuples
        smoothed_positions = [(float(x), float(y)) for x, y in zip(x_coords, y_coords)]
        
        return smoothed_positions
    
    def get_trajectory_spline(self, player_id: int, num_points: int = 50) -> List[Tuple[float, float]]:
        """
        Get spline-interpolated smooth trajectory for visualization.
        
        Args:
            player_id: Player identifier  
            num_points: Number of points in final smooth trajectory
            
        Returns:
            List of smooth trajectory points
        """
        smoothed_positions = self.get_smoothed_trajectory(player_id)
        
        if len(smoothed_positions) < 4:
            return smoothed_positions
        
        # Extract coordinates
        x_coords = [pos[0] for pos in smoothed_positions]
        y_coords = [pos[1] for pos in smoothed_positions]
        
        # Create parameter array (0 to 1)
        t = np.linspace(0, 1, len(smoothed_positions))
        
        # Create spline interpolations
        try:
            spline_x = interpolate.UnivariateSpline(t, x_coords, s=len(smoothed_positions)*0.1)
            spline_y = interpolate.UnivariateSpline(t, y_coords, s=len(smoothed_positions)*0.1)
            
            # Generate smooth trajectory points
            t_smooth = np.linspace(0, 1, num_points)
            smooth_x = spline_x(t_smooth)
            smooth_y = spline_y(t_smooth)
            
            # Ensure points stay within court bounds
            smooth_x = np.clip(smooth_x, 0, self.court_width)
            smooth_y = np.clip(smooth_y, 0, self.court_height)
            
            return [(float(x), float(y)) for x, y in zip(smooth_x, smooth_y)]
            
        except Exception as e:
            logger.warning(f"Spline interpolation failed for player {player_id}: {e}")
            return smoothed_positions
    
    def calculate_total_distance_meters(self, player_id: int, court_length_m: float = 13.4) -> float:
        """
        Calculate total distance moved by player in meters.
        
        Args:
            player_id: Player identifier
            court_length_m: Real court length in meters
            
        Returns:
            Total distance in meters
        """
        smoothed_positions = self.get_smoothed_trajectory(player_id)
        
        if len(smoothed_positions) < 2:
            return 0.0
        
        # Calculate pixel distance
        total_pixels = 0.0
        for i in range(1, len(smoothed_positions)):
            prev_pos = smoothed_positions[i-1]
            curr_pos = smoothed_positions[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_pixels += distance
        
        # Convert to meters using court scale
        pixels_per_meter = self.court_height / court_length_m
        total_meters = total_pixels / pixels_per_meter
        
        return total_meters


def create_smooth_trajectory_visualization(court_template: np.ndarray,
                                        player_trackers: List[PlayerTracker],
                                        colors: Optional[List[Tuple[int, int, int]]] = None) -> np.ndarray:
    """
    Create visualization with smooth trajectory lines instead of frame-by-frame dots.
    
    Args:
        court_template: Base court template image
        player_trackers: List of PlayerTracker objects with trajectory data
        colors: Optional list of colors for each player
        
    Returns:
        Court image with smooth trajectory visualization
    """
    result_img = court_template.copy()
    
    # Default colors for players
    if colors is None:
        colors = [
            (255, 0, 0),    # Red for player 1
            (0, 0, 255),    # Blue for player 2
            (0, 255, 0),    # Green for player 3
            (255, 255, 0),  # Yellow for player 4
        ]
    
    for player_id, tracker in enumerate(player_trackers):
        if player_id not in tracker.player_position_history:
            continue
            
        color = colors[player_id % len(colors)]
        
        # Get smooth trajectory
        smooth_trajectory = tracker.get_trajectory_spline(player_id, num_points=100)
        
        if len(smooth_trajectory) < 2:
            continue
        
        # Draw smooth trajectory line
        trajectory_points = np.array([(int(pos[0]), int(pos[1])) for pos in smooth_trajectory])
        
        # Draw the trajectory as a series of connected lines with varying thickness
        for i in range(1, len(trajectory_points)):
            # Vary line thickness along trajectory (thicker = more recent)
            alpha = i / len(trajectory_points)
            thickness = max(1, int(4 * alpha))
            
            cv2.line(result_img, 
                    tuple(trajectory_points[i-1]), 
                    tuple(trajectory_points[i]), 
                    color, thickness)
        
        # Draw start position
        if smooth_trajectory:
            start_pos = smooth_trajectory[0]
            cv2.circle(result_img, (int(start_pos[0]), int(start_pos[1])), 8, color, -1)
            cv2.circle(result_img, (int(start_pos[0]), int(start_pos[1])), 10, (255, 255, 255), 2)
            cv2.putText(result_img, f"P{player_id+1} Start", 
                       (int(start_pos[0]) + 15, int(start_pos[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw end position  
        if smooth_trajectory:
            end_pos = smooth_trajectory[-1]
            cv2.circle(result_img, (int(end_pos[0]), int(end_pos[1])), 8, color, -1)
            cv2.circle(result_img, (int(end_pos[0]), int(end_pos[1])), 10, (255, 255, 255), 2)
            cv2.putText(result_img, f"P{player_id+1} End", 
                       (int(end_pos[0]) + 15, int(end_pos[1]) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw net line for reference
    net_x = court_template.shape[1] // 2
    cv2.line(result_img, (net_x, 0), (net_x, court_template.shape[0]), (255, 255, 255), 3)
    cv2.putText(result_img, "NET", (net_x + 10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result_img