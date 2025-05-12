# analysis_server.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import logging
import time
import traceback

# Import the evaluation functions from overhead_diagnose.py
from overhead_diagnose import evaluate_swing_rules, align_keypoints_with_interpolation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("analysis_server")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add middleware to log request timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Get client IP and request details
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Request started: {request.method} {request.url.path} - Client: {client_host}")
    
    # Process the request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request completed: {request.method} {request.url.path} - {response.status_code} in {process_time:.2f}s")
        return response
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url.path}")
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# ------------------------- FASTAPI SERVER CODE -------------------------

class JointData(BaseModel):
    x: float
    y: float
    confidence: Optional[float] = 1.0

class FrameJoints(BaseModel):
    joints: Dict[str, List[JointData]]
    dominant_side: str = "Right"

@app.post("/analyze/technique")
async def analyze_technique(data: FrameJoints):
    try:
        logger.info(f"Received technique analysis request with {len(data.joints)} joints")
        # Convert to the format expected by evaluate_swing_rules
        keypoints = {}
        
        # Check if we have enough joints to proceed
        required_joints = [
            f"{data.dominant_side}Shoulder", f"{data.dominant_side}Elbow", 
            f"{data.dominant_side}Wrist", f"{data.dominant_side}Hip",
            'RightHeel', 'RightToe', 'LeftHeel', 'LeftToe'
        ]
        non_dominant = "Left" if data.dominant_side == "Right" else "Right"
        required_joints.extend([f"{non_dominant}Shoulder", f"{non_dominant}Elbow", f"{non_dominant}Hip"])
        
        missing_joints = [joint for joint in required_joints if joint not in data.joints]
        if missing_joints:
            logger.warning(f"Missing required joints: {missing_joints}")
            return {
                "overall_score": 0,
                "details": {
                    "shoulder_abduction": False,
                    "elbow_flexion": False,
                    "elbow_lower": False,
                    "foot_direction_aligned": False,
                    "proximal_to_distal_sequence": False,
                    "hip_forward_shift": False,
                    "trunk_rotation_completed": False
                },
                "error": f"Missing required joints: {', '.join(missing_joints)}"
            }
        
        # Convert from frontend format to NumPy arrays
        for joint_name, joint_data in data.joints.items():
            # Create a numpy array of shape (T, 2) where T is the number of frames
            points = np.array([[point.x, point.y] for point in joint_data])
            keypoints[joint_name] = points
        
        frame_count = len(next(iter(data.joints.values())))
        logger.info(f"Processing {frame_count} frames of motion data")
        
        keypoints = align_keypoints_with_interpolation(keypoints, frame_count)
        
        # Run the evaluation
        results = evaluate_swing_rules(keypoints, dominant_side=data.dominant_side)
        
        # Convert NumPy types to standard Python types
        results = convert_numpy_types(results)
        
        # Format results for display
        formatted_results = {
            "overall_score": sum(1 for v in results.values() if v) / len(results) * 100,
            "details": results
        }
        
        logger.info(f"Technique analysis complete. Overall score: {formatted_results['overall_score']:.1f}%")
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error during technique analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/comparison")
async def analyze_comparison(data: Dict[str, FrameJoints]):
    """
    Compare user technique against model or another video
    Expected keys: 'user', 'reference'
    """
    try:
        logger.info("Received comparison analysis request")
        
        if "user" not in data or "reference" not in data:
            logger.error("Missing required 'user' or 'reference' data in comparison request")
            raise HTTPException(status_code=400, detail="Both 'user' and 'reference' data are required")
        
        user_results = await analyze_technique(data["user"])
        reference_results = await analyze_technique(data["reference"])
        
        # Calculate similarity scores between user and reference
        similarity = {}
        for rule in user_results["details"]:
            similarity[rule] = user_results["details"][rule] == reference_results["details"][rule]
        
        response = {
            "user_score": user_results["overall_score"],
            "reference_score": reference_results["overall_score"],
            "similarity": similarity,
            "user_details": user_results["details"],
            "reference_details": reference_results["details"]
        }
        
        logger.info(f"Comparison complete. User score: {response['user_score']:.1f}%, Reference score: {response['reference_score']:.1f}%")
        return response
    
    except Exception as e:
        logger.error(f"Error during comparison analysis: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.get("/")
async def root():
    return {"status": "Server is running", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting MoveInsight Analysis Server...")
    
    # Configure Uvicorn with appropriate settings
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",  # CRITICAL CHANGE: Listen on all interfaces, not just localhost
        port=8000,
        log_level="info",
        timeout_keep_alive=65,  # Keep connections alive longer
        limit_concurrency=10    # Limit concurrent connections to prevent overload
    )
    
    server = uvicorn.Server(config)
    server.run()