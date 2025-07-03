# models.py
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class JointDataItem(BaseModel):
    """Represents a single joint's position and confidence data."""
    x: float
    y: float
    z: Optional[float] = None  # Added Z coordinate for 3D analysis
    confidence: Optional[float] = None

class FrameDataItem(BaseModel):
    """Represents all joint data for a single frame."""
    joints: Dict[str, JointDataItem]

class VideoAnalysisResponseModel(BaseModel):
    """Response model for video analysis endpoint."""
    total_frames: int
    joint_data_per_frame: List[FrameDataItem]
    swing_analysis: Optional[Dict[str, bool]] = None
    overall_score: Optional[float] = None

class TechniqueComparisonRequestDataModel(BaseModel):
    """Request model for technique comparison endpoint."""
    user_video_frames: List[FrameDataItem]  # Contains 3D data
    model_video_frames: List[FrameDataItem]  # Contains 3D data
    dominant_side: str
    technique_type: str = "overhead_clear" # New field for technique type

class ComparisonResultModel(BaseModel):
    """Response model for technique comparison results."""
    user_score: float
    reference_score: float
    similarity: Dict[str, bool]
    user_details: Dict[str, bool]
    reference_details: Dict[str, bool]

class OverheadClearAnalysis(BaseModel):
    score: float
    details: Dict[str, bool]

class MatchAnalysisResponseModel(BaseModel):
    movement_video_path: str
    overhead_clear_count: int
    overhead_clear_analyses: List[OverheadClearAnalysis]