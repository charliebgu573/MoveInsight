// MoveInsight/BodyPoseTypes.swift
import SwiftUI
import simd // For SIMD3<Float>

// MARK: - 2D Body Connection Structure (for 2D Overlays)
// Defines a connection between two body joints for drawing the 2D skeleton overlay.
// Uses String joint names that correspond to server/MediaPipe output.
struct StringBodyConnection: Identifiable {
    let id = UUID()
    let from: String // Joint name (e.g., "LeftShoulder")
    let to: String   // Joint name (e.g., "LeftElbow")
}

// MARK: - 3D Body Connection Structure (for SceneKit Rendering)
// Defines a connection between two body joints for drawing the 3D skeleton.
// Uses String joint names.
struct BodyConnection3D: Identifiable {
    let id = UUID()
    let from: String // Joint name (e.g., "LeftShoulder")
    let to: String   // Joint name (e.g., "LeftElbow")
}

// Standard set of connections for rendering a 3D skeleton.
let HumanBodyJoints: [BodyConnection3D] = [
    // Torso
    BodyConnection3D(from: "LeftShoulder", to: "RightShoulder"),
    BodyConnection3D(from: "LeftHip", to: "RightHip"),
    BodyConnection3D(from: "LeftShoulder", to: "LeftHip"),
    BodyConnection3D(from: "RightShoulder", to: "RightHip"),
    
    // Optional: Connect nose to a central point like a "Neck" if available,
    // or to mid-point of shoulders for a basic head representation.
    // BodyConnection3D(from: "Nose", to: "LeftShoulder"), // Example, adjust as needed
    // BodyConnection3D(from: "Nose", to: "RightShoulder"), // Example, adjust as needed

    // Left Arm
    BodyConnection3D(from: "LeftShoulder", to: "LeftElbow"),
    BodyConnection3D(from: "LeftElbow", to: "LeftWrist"),
    
    // Right Arm
    BodyConnection3D(from: "RightShoulder", to: "RightElbow"),
    BodyConnection3D(from: "RightElbow", to: "RightWrist"),

    // Left Leg
    BodyConnection3D(from: "LeftHip", to: "LeftKnee"),
    BodyConnection3D(from: "LeftKnee", to: "LeftAnkle"),
    // Optional finer details for feet:
    // BodyConnection3D(from: "LeftAnkle", to: "LeftHeel"),
    // BodyConnection3D(from: "LeftHeel", to: "LeftToe"), // "LeftToe" is often "LeftFootIndex" from MediaPipe

    // Right Leg
    BodyConnection3D(from: "RightHip", to: "RightKnee"),
    BodyConnection3D(from: "RightKnee", to: "RightAnkle"),
    // Optional finer details for feet:
    // BodyConnection3D(from: "RightAnkle", to: "RightHeel"),
    // BodyConnection3D(from: "RightHeel", to: "RightToe") // "RightToe" is often "RightFootIndex"
]


// MARK: - 3D Pose Point (String-based joint name)
// Represents a single 3D point for a joint.
struct Pose3DPoint: Identifiable { // Identifiable for use in SwiftUI lists if needed.
    let id = UUID()
    let jointName: String        // The name of the joint (e.g., "Nose", "LeftElbow")
    let position: SIMD3<Float>   // The 3D coordinates (x, y, z) of the joint

    init(jointName: String, position: SIMD3<Float>) {
        self.jointName = jointName
        self.position = position
    }
}

// MARK: - 3D Pose Body (String-based joints)
// Represents a complete 3D human pose for a single person/video source at a specific frame.
// This structure holds all detected 3D joint positions for one skeleton.
struct Pose3DBody: Identifiable {
    let id = UUID()
    // A dictionary mapping joint names (Strings) to their 3D positions (SIMD3<Float>).
    let joints: [String: SIMD3<Float>]
    // Indicates the source of this pose data (e.g., user's primary video or model/secondary video).
    let videoSource: VideoSource
    
    // Enum to differentiate between video sources, useful for coloring or offsetting skeletons.
    enum VideoSource {
        case primary   // Typically the user's video
        case secondary // Typically the model or comparison video
    }
    
    init(joints: [String: SIMD3<Float>], videoSource: VideoSource) {
        self.joints = joints
        self.videoSource = videoSource
    }
}
