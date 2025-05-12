import SwiftUI
import Vision
import simd

// MARK: - Body Connection Structure
// Defines a connection between two body joints for drawing the skeleton
struct BodyConnection: Identifiable {
    let id = UUID()
    let from: VNHumanBodyPoseObservation.JointName
    let to: VNHumanBodyPoseObservation.JointName
}

// MARK: - 3D Pose Point
// Represents a 3D pose point with position in world space
struct Pose3DPoint {
    let jointName: VNHumanBodyPoseObservation.JointName
    let position: SIMD3<Float> // x, y, z coordinates
    let confidence: Float
    
    init(jointName: VNHumanBodyPoseObservation.JointName,
         position: SIMD3<Float>,
         confidence: Float = 1.0) {
        self.jointName = jointName
        self.position = position
        self.confidence = confidence
    }
}

// MARK: - 3D Pose Body
// Represents a complete 3D human pose
struct Pose3DBody: Identifiable {
    let id = UUID()
    let joints: [VNHumanBodyPoseObservation.JointName: Pose3DPoint]
    let videoSource: VideoSource // Identifies which video this pose came from
    
    enum VideoSource {
        case primary
        case secondary
    }
    
    // Fixed initializer with proper closure parameters
    init(joints: [VNHumanBodyPoseObservation.JointName: SIMD3<Float>],
         videoSource: VideoSource) {
        self.videoSource = videoSource
        
        // Create a dictionary with jointName -> Pose3DPoint mapping
        var posePoints: [VNHumanBodyPoseObservation.JointName: Pose3DPoint] = [:]
        for (jointName, position) in joints {
            posePoints[jointName] = Pose3DPoint(jointName: jointName, position: position)
        }
        self.joints = posePoints
    }
}
